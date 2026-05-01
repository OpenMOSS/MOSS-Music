import logging
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from sglang.srt.configs.moss_audio import MossAudioConfig, MossAudioEncoderConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.num_positions = num_positions
        max_timescale = 10000.0
        log_timescale_increment = math.log(max_timescale) / (embedding_dim // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(embedding_dim // 2).float()
        )
        self.register_buffer("inv_timescales", inv_timescales)

    def forward(self, seq_len: int, device: torch.device):
        inv_timescales = self._buffers["inv_timescales"]
        assert isinstance(inv_timescales, torch.Tensor)
        time = torch.arange(seq_len, device=device, dtype=inv_timescales.dtype)
        scaled_time = time[:, None] * inv_timescales[None, :]
        sin_emb = torch.sin(scaled_time)
        cos_emb = torch.cos(scaled_time)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=1)
        return pos_emb.unsqueeze(0)


class MossAudioEncoder(nn.Module):
    def __init__(self, config: MossAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.gelu = nn.GELU()
        whisper_config = WhisperConfig(
            d_model=config.d_model,
            encoder_attention_heads=config.encoder_attention_heads,
            encoder_ffn_dim=config.encoder_ffn_dim,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            activation_function=config.activation_function,
            layer_norm_eps=config.layer_norm_eps,
        )
        whisper_config._attn_implementation = config._attn_implementation

        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        self.stem_proj = nn.Linear(config.downsample_hidden_size * 16, config.d_model)
        self.embed_positions = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(whisper_config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_proj = (
            nn.Linear(config.d_model, config.output_dim, bias=False)
            if config.output_dim != config.d_model
            else nn.Identity()
        )

        self.deepstack_encoder_layer_indexes = list(
            getattr(config, "deepstack_encoder_layer_indexes", []) or []
        )
        self._deepstack_capture_map = {
            layer_idx: capture_idx
            for capture_idx, layer_idx in enumerate(
                self.deepstack_encoder_layer_indexes
            )
        }

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        def conv_out_len(x: torch.Tensor) -> torch.Tensor:
            return (x - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(lengths)))

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        output_deepstack_hidden_states: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        if input_features.dim() != 3:
            raise ValueError(
                f"Expected [n_mels, T] or [B, n_mels, T], got {tuple(input_features.shape)}."
            )

        if feature_lens is None:
            feature_lens = torch.full(
                (input_features.size(0),),
                input_features.size(-1),
                device=input_features.device,
                dtype=torch.long,
            )
        else:
            feature_lens = feature_lens.to(
                device=input_features.device, dtype=torch.long
            )

        downsampled_lengths = self._compute_downsampled_length(feature_lens)

        x = input_features.unsqueeze(1)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))

        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.stem_proj(x)

        max_len = int(downsampled_lengths.max().item())
        if x.size(1) > max_len:
            x = x[:, :max_len, :]

        positions = self.embed_positions(x.shape[1], x.device)
        x = x + positions.to(x.dtype)

        padding_mask = (
            torch.arange(x.size(1), device=x.device)[None, :]
            >= downsampled_lengths[:, None]
        )
        attention_mask = (1.0 - (~padding_mask).to(dtype=x.dtype)) * torch.finfo(
            x.dtype
        ).min
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        deepstack_states: List[Optional[torch.Tensor]] = [None] * len(
            self.deepstack_encoder_layer_indexes
        )
        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )[0]
            capture_idx = self._deepstack_capture_map.get(layer_idx)
            if output_deepstack_hidden_states and capture_idx is not None:
                deepstack_states[capture_idx] = x

        x = self.layer_norm(x)
        x = self.out_proj(x)

        ordered_deepstack_states = [
            hidden_state
            for hidden_state in deepstack_states
            if hidden_state is not None
        ]
        if not isinstance(self.out_proj, nn.Identity):
            ordered_deepstack_states = [
                self.out_proj(hidden_state) for hidden_state in ordered_deepstack_states
            ]
        return x, ordered_deepstack_states


class GatedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MossAudioQwen3Model(Qwen3Model):
    def __init__(
        self,
        config,
        num_deepstack_embeddings: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        if not self.pp_group.is_first_rank:
            assert (
                self.start_layer >= num_deepstack_embeddings
            ), "Deepstack injection layers must stay on the first pipeline stage."

        self.hidden_size = config.hidden_size
        self.deepstack_embed_to_decoder_layer = range(num_deepstack_embeddings)

    def get_deepstack_embeds(
        self, layer_idx: int, input_deepstack_embeds: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if (
            input_deepstack_embeds is None
            or layer_idx not in self.deepstack_embed_to_decoder_layer
        ):
            return None

        start = self.hidden_size * layer_idx
        return input_deepstack_embeds[:, start : start + self.hidden_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            assert isinstance(hidden_states, torch.Tensor)
            assert residual is None or isinstance(residual, torch.Tensor)

        layers = self.layers
        assert isinstance(layers, nn.ModuleList)
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
            deepstack_embeds = self.get_deepstack_embeds(
                layer_idx, input_deepstack_embeds
            )
            if deepstack_embeds is not None:
                hidden_states = hidden_states + deepstack_embeds

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class MossAudioModel(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: MossAudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        audio_config = self.config.audio_config
        if isinstance(audio_config, dict):
            audio_config = MossAudioEncoderConfig.from_dict(audio_config)
        elif audio_config is None:
            audio_config = MossAudioEncoderConfig()

        deepstack_visual_indexes = list(
            getattr(audio_config, "deepstack_encoder_layer_indexes", []) or []
        )
        deepstack_num_inject_layers = self.config.deepstack_num_inject_layers
        if deepstack_num_inject_layers is not None:
            deepstack_visual_indexes = deepstack_visual_indexes[
                : int(deepstack_num_inject_layers)
            ]
        audio_config.deepstack_encoder_layer_indexes = deepstack_visual_indexes

        self.audio_encoder = MossAudioEncoder(audio_config)
        self.n_window = int(getattr(audio_config, "n_window", 200))
        self.chunk_frames = int(self.n_window * 2)
        self.conv_chunksize = int(getattr(audio_config, "conv_chunksize", 64))

        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
        self.use_deepstack = (
            {Modality.AUDIO: True} if self.num_deepstack_embeddings > 0 else {}
        )

        language_config = self.config.language_config
        self.language_model = MossAudioQwen3Model(
            config=language_config,
            num_deepstack_embeddings=self.num_deepstack_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.language_model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    language_config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                embed_token_weight = self.language_model.embed_tokens.weight
                assert isinstance(embed_token_weight, torch.Tensor)
                self.pp_group.send(embed_token_weight, dst=self.pp_group.last_rank)
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=torch.Size([config.vocab_size, language_config.hidden_size]),
                    dtype=next(self.language_model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                lm_head_weight = self.lm_head.weight
                assert isinstance(lm_head_weight, torch.Tensor)
                lm_head_weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        adapter_hidden_size = int(
            getattr(
                self.config,
                "adapter_hidden_size",
                max(language_config.hidden_size * 4, audio_config.output_dim),
            )
        )
        self.audio_adapter = GatedMLP(
            input_size=audio_config.output_dim,
            hidden_size=adapter_hidden_size,
            output_size=language_config.hidden_size,
        )
        self.deepstack_audio_merger_list = nn.ModuleList(
            [
                GatedMLP(
                    input_size=audio_config.output_dim,
                    hidden_size=adapter_hidden_size,
                    output_size=language_config.hidden_size,
                )
                for _ in range(self.num_deepstack_embeddings)
            ]
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        return MossAudioEncoder._compute_downsampled_length(lengths)

    @staticmethod
    def _concat_audio_batch_from_list(
        mels: List[torch.Tensor],
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if len(mels) == 0:
            raise ValueError("Empty audio list.")

        n_mels = int(mels[0].shape[0])
        concat_mels = []
        for mel, length in zip(mels, lengths):
            if mel.dim() != 2 or int(mel.shape[0]) != n_mels:
                raise ValueError(f"Expected mel [n_mels, T], got {tuple(mel.shape)}.")
            concat_mels.append(
                mel[:, : int(length.item())].to(dtype=dtype, device=device)
            )
        return torch.cat(concat_mels, dim=1)

    def _encode_audio_chunks(
        self, input_features: torch.Tensor, feature_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feature_lens = feature_lens.to(
            device=input_features.device,
            dtype=torch.long,
        )

        chunk_num = torch.ceil(
            feature_lens.to(torch.float32) / float(self.chunk_frames)
        ).long()
        total_chunks = int(chunk_num.sum().item())
        if total_chunks == 0:
            empty_audio = torch.empty(
                (0, self.audio_encoder.config.output_dim),
                device=input_features.device,
                dtype=input_features.dtype,
            )
            empty_deepstack = [
                torch.empty(
                    (0, self.audio_encoder.config.output_dim),
                    device=input_features.device,
                    dtype=input_features.dtype,
                )
                for _ in range(self.num_deepstack_embeddings)
            ]
            return empty_audio, empty_deepstack

        chunk_lengths = torch.full(
            (total_chunks,),
            self.chunk_frames,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % self.chunk_frames
        chunk_lengths[chunk_lengths == 0] = self.chunk_frames

        chunk_list = list(
            input_features.transpose(0, 1).split(chunk_lengths.tolist(), dim=0)
        )
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        feature_lens_after_cnn = self.audio_encoder._compute_downsampled_length(
            chunk_lengths
        )
        t_down_max = int(feature_lens_after_cnn.max().item())
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(
                    int(length.item()),
                    dtype=torch.bool,
                    device=padded_feature.device,
                )
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        if padded_mask_after_cnn.shape[1] < t_down_max:
            padded_mask_after_cnn = F.pad(
                padded_mask_after_cnn,
                (0, t_down_max - padded_mask_after_cnn.shape[1]),
                value=False,
            )

        padded_embeds: List[torch.Tensor] = []
        deepstack_padded_embeds: List[List[torch.Tensor]] = [
            [] for _ in range(self.num_deepstack_embeddings)
        ]
        for feat_chunk, len_chunk in zip(
            padded_feature.split(self.conv_chunksize, dim=0),
            chunk_lengths.split(self.conv_chunksize, dim=0),
        ):
            out, deepstack_outs = self.audio_encoder(
                feat_chunk,
                len_chunk,
                output_deepstack_hidden_states=self.num_deepstack_embeddings > 0,
            )
            if out.shape[1] < t_down_max:
                out = F.pad(out, (0, 0, 0, t_down_max - out.shape[1]))
            padded_embeds.append(out)

            if len(deepstack_outs) != self.num_deepstack_embeddings:
                raise RuntimeError(
                    "Deepstack output count does not match configured layer indexes."
                )
            for capture_idx, deepstack_out in enumerate(deepstack_outs):
                if deepstack_out.shape[1] < t_down_max:
                    deepstack_out = F.pad(
                        deepstack_out, (0, 0, 0, t_down_max - deepstack_out.shape[1])
                    )
                deepstack_padded_embeds[capture_idx].append(deepstack_out)

        padded_embed = torch.cat(padded_embeds, dim=0)
        audio_embeds = padded_embed[padded_mask_after_cnn]

        deepstack_embeds: List[torch.Tensor] = []
        for deepstack_chunks in deepstack_padded_embeds:
            deepstack_embed = torch.cat(deepstack_chunks, dim=0)
            deepstack_embeds.append(deepstack_embed[padded_mask_after_cnn])

        return audio_embeds, deepstack_embeds

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def separate_deepstack_embeds(
        self, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_size = self.language_model.hidden_size
        expected_hidden_size = hidden_size * (1 + self.num_deepstack_embeddings)
        if embedding.shape[-1] != expected_hidden_size:
            raise ValueError(
                "Unexpected audio embedding width for deepstack: "
                f"expected {expected_hidden_size}, got {embedding.shape[-1]}."
            )
        return embedding[:, :hidden_size], embedding[:, hidden_size:]

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        device = next(self.audio_encoder.parameters()).device
        dtype = self.audio_encoder.dtype

        mel_list: List[torch.Tensor] = []
        raw_len_list: List[int] = []
        expected_out_lens: List[int] = []

        for item in items:
            feature = item.feature
            if not isinstance(feature, torch.Tensor):
                feature = torch.as_tensor(feature)
            feature = feature.to(device=device, dtype=dtype)

            raw_lens = None
            if hasattr(item, "audio_data_seqlens"):
                raw_lens = (
                    torch.as_tensor(item.audio_data_seqlens, device=device)
                    .flatten()
                    .long()
                )
            elif hasattr(item, "feature_attention_mask"):
                raw_lens = (
                    torch.as_tensor(item.feature_attention_mask, device=device)
                    .sum(dim=-1)
                    .flatten()
                    .long()
                )

            exp_lens = None
            if hasattr(item, "audio_feature_lens"):
                exp_lens = (
                    torch.as_tensor(item.audio_feature_lens, device=device)
                    .flatten()
                    .long()
                )

            if feature.dim() == 2:
                mel_list.append(feature)
                raw_len = (
                    int(raw_lens[0].item())
                    if raw_lens is not None
                    else int(feature.shape[-1])
                )
                raw_len_list.append(raw_len)
                if exp_lens is not None and exp_lens.numel() > 0:
                    expected_out_lens.append(int(exp_lens[0].item()))
                else:
                    expected_out_lens.append(
                        int(
                            self._compute_downsampled_length(torch.tensor([raw_len]))[
                                0
                            ].item()
                        )
                    )
            elif feature.dim() == 3:
                batch_size = int(feature.shape[0])
                if raw_lens is None:
                    raw_lens = torch.full(
                        (batch_size,),
                        int(feature.shape[-1]),
                        device=device,
                        dtype=torch.long,
                    )
                if exp_lens is None:
                    exp_lens = self._compute_downsampled_length(raw_lens)

                for idx in range(batch_size):
                    mel_list.append(feature[idx])
                    raw_len_list.append(int(raw_lens[idx].item()))
                    expected_out_lens.append(int(exp_lens[idx].item()))
            else:
                raise ValueError(
                    f"Unsupported audio feature shape: {tuple(feature.shape)}"
                )

        raw_lens_tensor = torch.as_tensor(raw_len_list, device=device, dtype=torch.long)
        concat_feature = self._concat_audio_batch_from_list(
            mel_list,
            raw_lens_tensor,
            dtype=dtype,
            device=device,
        )
        audio_embeds, deepstack_audio_embeds = self._encode_audio_chunks(
            concat_feature,
            raw_lens_tensor,
        )

        audio_embeds = self.audio_adapter(audio_embeds)
        expected_total = int(sum(expected_out_lens))
        if expected_total > 0 and audio_embeds.shape[0] > expected_total:
            audio_embeds = audio_embeds[:expected_total]

        if len(deepstack_audio_embeds) != self.num_deepstack_embeddings:
            raise RuntimeError(
                "Deepstack feature count does not match configured merger count."
            )
        if self.num_deepstack_embeddings > 0:
            merged_deepstack_embeds = []
            for merger, deepstack_embed in zip(
                self.deepstack_audio_merger_list, deepstack_audio_embeds
            ):
                merged_deepstack = merger(deepstack_embed)
                if expected_total > 0 and merged_deepstack.shape[0] > expected_total:
                    merged_deepstack = merged_deepstack[:expected_total]
                merged_deepstack_embeds.append(merged_deepstack)
            audio_embeds = torch.cat([audio_embeds] + merged_deepstack_embeds, dim=-1)

        return audio_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
            use_deepstack=self.use_deepstack,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                )
            return self.pooler(hidden_states, forward_batch)

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers_dict = dict(self.named_buffers(remove_duplicate=False))

        for name, loaded_weight in weights:
            if (
                getattr(self.config, "tie_word_embeddings", False)
                and name == "lm_head.weight"
            ):
                continue

            matched_stacked_param = False
            if name.startswith("language_model.layers."):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    stacked_name = name.replace(weight_name, param_name)
                    if (
                        stacked_name.endswith(".bias")
                        and stacked_name not in params_dict
                    ):
                        continue
                    if stacked_name not in params_dict:
                        continue

                    param = params_dict[stacked_name]
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight, shard_id)
                    matched_stacked_param = True
                    break

            if matched_stacked_param:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif name in buffers_dict:
                buffer = buffers_dict[name]
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
            elif not (name.endswith(".bias") and name not in params_dict):
                logger.warning(
                    "Parameter %s not found in current model.",
                    name,
                )


EntryClass = MossAudioModel
