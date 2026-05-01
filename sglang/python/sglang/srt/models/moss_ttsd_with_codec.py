import logging
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import BatchFeature

from sglang.srt.configs.moss_ttsd_with_codec import MossTTSDWithCodecConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.mm_utils import (
    MultimodalInputs,
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.mm_utils_modified import embed_mm_inputs
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.moss_ttsd import MossTTSDForCausalLM
from sglang.srt.utils.common import add_prefix
from sglang.third_party.moss_ttsd.modeling_xy_tokenizer import XYTokenizer

logger = logging.getLogger(__name__)


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    audio_embedding_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    token_embedding_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    if not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank:
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            mm_input_indices = [
                i
                for i, mm_input in enumerate(forward_batch.mm_inputs)
                if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            input_embeds, other_info = embed_mm_inputs(
                forward_batch=forward_batch,
                mm_inputs_list=mm_inputs_list,
                mm_input_indices=mm_input_indices,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                multimodal_model=multimodal_model,
                input_embedding=token_embedding_func,
                data_embedding_func_mapping={Modality.AUDIO: audio_embedding_func},
            )

            if mm_inputs_list:
                for mm_input_obj in mm_inputs_list:
                    if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
                        for mm_item in mm_input_obj.mm_items:
                            feature = getattr(mm_item, "feature", None)
                            if isinstance(feature, torch.Tensor) and feature.is_cuda:
                                mm_item.feature = feature.to("cpu", non_blocking=True)
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
        else:
            input_embeds = token_embedding_func(input_ids)
        # Copy to pre-allocated buffer if available (for CUDA graph address stability)
        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
    else:
        input_embeds = None

    return input_embeds


class MossTTSDWithCodec(MossTTSDForCausalLM):
    def __init__(
        self,
        config: MossTTSDWithCodecConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "model",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.codec_cfg = config.get_codec_config()
        self.speech_token_range = config.speech_token_range
        self.channels = config.channels
        self.pad_token_id = config.pad_token
        self.speech_pad_token = config.speech_pad_token
        if self.pp_group.is_last_rank:
            # Initialize codec_model with float32 to preserve checkpoint dtype
            # instead of converting to target dtype
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.codec_model = XYTokenizer(self.codec_cfg)
            finally:
                torch.set_default_dtype(old_dtype)
        self.input_sample_rate = self.codec_cfg.input_sample_rate
        self.output_sample_rate = self.codec_cfg.output_sample_rate
        self.encoder_downsample_rate = self.codec_cfg.encoder_downsample_rate
        self.tokens_per_second = float(self.input_sample_rate) / float(
            self.encoder_downsample_rate
        )
        self.tokens_per_chunk = int(round(10.0 * self.tokens_per_second))
        self.samples_per_token = int(
            round(self.output_sample_rate / self.tokens_per_second)
        )
        self.stride_tokens = 85
        self.keep_tokens = 85
        self.left_ctx_tokens = 20
        self.crossfade_seconds = 0.1
        self.crossfade_samples = int(
            round(self.crossfade_seconds * self.output_sample_rate)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    @staticmethod
    def _shift_inputs(
        input_ids: torch.Tensor,
        pad_token_id: int,
        speech_pad_token_id: int,
        max_channels: int,
    ) -> torch.Tensor:
        """
        Convert (T, C) grid to time-shifted multi-channel layout (preserving original implementation logic).

        Creates a shifted layout where new_len = T + C - 1, with column j shifted backwards by j positions.
        This enables the model to process multiple codebook channels with temporal alignment.

        Args:
            input_ids: Input grid with shape (T, C).
            pad_token_id: Padding token ID for text tokens.
            max_channels: Maximum number of channels.

        Returns:
            Shifted array with shape (T + max_channels - 1, max_channels).
        """
        T, _ = input_ids.shape
        new_len = T + max_channels - 1
        shifted = torch.full(
            (new_len, max_channels),
            fill_value=speech_pad_token_id,
            dtype=torch.int64,
            device=input_ids.device,
        )
        shifted[:, 0] = torch.full(
            (new_len,),
            pad_token_id,
            dtype=torch.int64,
            device=input_ids.device,
        )
        for j in range(max_channels):
            shifted[j : (T + j), j] = input_ids[:, j]
        return shifted

    @staticmethod
    def shifting_outputs(
        output_ids: torch.Tensor,
        speech_token_range: list[int],
        max_channels: int = 8,
    ) -> "torch.Tensor":
        """
        Restore time-shifted layout to per-timestep C-channel arrangement and reverse-offset first codebook.

        Converts the time-shifted multi-channel output back to standard (batch, time, channels) format
        and maps the first codebook tokens back to their original space by subtracting the speech token offset.

        Args:
            output_ids: Time-shifted output tensor.
            speech_token_range: Speech token range for reverse mapping.
            max_channels: Number of codebook channels.

        Returns:
            Restored tensor with shape (batch, seq_len, max_channels).
        """
        seq_len = output_ids.shape[1] - max_channels + 1
        speech_ids = torch.full(
            (output_ids.shape[0], seq_len, max_channels),
            0,
            dtype=output_ids.dtype,
            device=output_ids.device,
        )
        for j in range(max_channels):
            speech_ids[..., j] = output_ids[:, j : seq_len + j, j]
            if j == 0:
                speech_ids[..., j] = speech_ids[..., j] - speech_token_range[0]
        return speech_ids

    @staticmethod
    def _find_max_valid_positions(data: "torch.Tensor", invalid_value: int = 1024):
        """
        Locate continuous valid audio segment intervals in each sequence (all non-text channels valid simultaneously).

        Identifies contiguous spans where all audio channels (columns 1+) contain valid tokens
        (not the invalid_value padding token).

        Args:
            data: Input tensor with shape (batch, time, channels).
            invalid_value: Token ID considered as invalid/padding.

        Returns:
            List of lists containing valid audio segments for each sequence in the batch.
        """
        mask = torch.all(data[:, :, 1:] != invalid_value, dim=2)
        valid_indices = torch.where(mask)
        result = [[] for _ in range(len(data))]
        if valid_indices[0].numel() == 0:
            return result
        grouped = []
        group_ids = []
        for i, seq_no in enumerate(valid_indices[0]):
            pos = valid_indices[1][i]
            if not group_ids or seq_no > group_ids[-1]:
                group_ids.append(seq_no)
                grouped.append([[pos, pos + 1]])
            elif pos == grouped[-1][-1][-1]:
                grouped[-1][-1][-1] += 1
            else:
                grouped[-1].append([pos, pos + 1])
        for gid, spans in zip(group_ids, grouped):
            for s, e in spans:
                result[gid].append(data[gid, s:e, :])
        return result

    def get_audio_embeddings(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,
        indices: List[int],
    ) -> torch.Tensor:
        audio_embeddings = []
        # Ensure audio token indices land on the same device as embedding weights.
        for i, item in enumerate(items):
            data = {
                "input_features": item.feature,
                "attention_mask": item.model_specific_data["audio_attention_mask"],
            }
            ref_audio = {
                "input_features": item.model_specific_data["ref_audio_features"],
                "attention_mask": item.model_specific_data["ref_audio_attention_mask"],
            }
            feat = BatchFeature(data=data, tensor_type="pt")
            ref_feat = BatchFeature(data=ref_audio, tensor_type="pt")
            with torch.no_grad():
                enc = self.codec_model.encode(feat)
                # (time, codebooks)
                audio_codes = enc["audio_codes"][
                    :, 0, : enc["codes_lengths"][0].item()
                ].permute(1, 0)
                ref_enc = self.codec_model.encode(ref_feat)
                ref_audio_codes = ref_enc["audio_codes"][
                    :, 0, : ref_enc["codes_lengths"][0].item()
                ]
            # Map first codebook to speech token range
            audio_codes[:, 0] = audio_codes[:, 0] + self.speech_token_range[0]
            audio_codes = self._shift_inputs(
                audio_codes,
                pad_token_id=self.pad_token_id[0],
                speech_pad_token_id=self.speech_pad_token,
                max_channels=self.channels,
            )
            truncated_input_ids = audio_codes[-(self.channels - 1) :, :]
            audio_codes = audio_codes[: -(self.channels - 1), :]
            audio_embed = self._prepare_multi_modal_inputs(audio_codes)
            audio_embeddings.append(audio_embed)
            forward_batch.truncated_input_ids[indices[i]] = truncated_input_ids
            forward_batch.ref_audio_codes[indices[i]] = ref_audio_codes
        return torch.cat(audio_embeddings, dim=0)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def crossfade_concat(
        self, segments: list, sample_rate: int, crossfade_seconds: float = 0.1
    ) -> torch.Tensor:
        """Concatenate segments with linear crossfade.

        Args:
            segments: list of (1, T) tensors
            sample_rate: sampling rate
            crossfade_seconds: overlap time for crossfade
        Returns:
            (1, T_total) tensor
        """
        if len(segments) == 0:
            return torch.zeros(1, 0)
        if len(segments) == 1:
            return segments[0]
        out = segments[0]
        cf_len_target = int(round(crossfade_seconds * sample_rate))
        for k in range(1, len(segments)):
            nxt = segments[k]
            if cf_len_target <= 0:
                out = torch.cat([out, nxt], dim=-1)
                continue
            cf_len = min(cf_len_target, out.shape[-1], nxt.shape[-1])
            if cf_len <= 0:
                out = torch.cat([out, nxt], dim=-1)
                continue
            fade_out = torch.linspace(
                1.0, 0.0, steps=cf_len, dtype=out.dtype, device=out.device
            )
            fade_in = torch.linspace(
                0.0, 1.0, steps=cf_len, dtype=nxt.dtype, device=nxt.device
            )
            overlap = out[0, -cf_len:] * fade_out + nxt[0, :cf_len] * fade_in
            out = torch.cat(
                [out[:, :-cf_len], overlap.unsqueeze(0), nxt[:, cf_len:]], dim=-1
            )
        return out

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_embedding:
            raise NotImplementedError(
                "get_embedding is not implemented for MOSSTTSDWithCodec"
            )

        input_embeds = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self,
            audio_embedding_func=self.get_audio_embeddings,
            token_embedding_func=self._prepare_multi_modal_inputs,
        )
        return super().forward(
            None,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def decode(
        self,
        audio_code: torch.Tensor,
        forward_batch: ForwardBatch,
        i: int,
    ) -> list[torch.Tensor]:
        if (
            audio_code is None
            or audio_code.ndim != 2
            or audio_code.shape[1] != self.channels
            or forward_batch.forward_mode.is_audio_decode() is False
        ):
            return None
        elif (
            audio_code is None
            or audio_code.ndim != 2
            or audio_code.shape[1] != self.channels
            or audio_code.shape[0] < self.channels
            or forward_batch.forward_mode.is_audio_decode() is False
        ):
            return torch.zeros(1)

        normal = self.shifting_outputs(
            audio_code.unsqueeze(0), self.speech_token_range, self.channels
        )
        audio_frags = self._find_max_valid_positions(normal, self.speech_pad_token)[0]
        if len(audio_frags):
            frag = torch.cat([f.permute(1, 0).unsqueeze(1) for f in audio_frags], dim=1)
            # ref_audio_codes is guaranteed to be a (channels, T_ref) tensor (e.g., [8, 250]).
            ref_codes = forward_batch.ref_audio_codes[i]
            if ref_codes is None:
                audio = self.codec_model.decode(frag, overlap_seconds=10)[
                    "audio_values"
                ][0]
            else:
                codec_device = self.codec_model.device
                ref_codes = ref_codes.to(device=codec_device, dtype=torch.long)
                # frag has shape (nq, B, T). We decode a single generated track (B=1)
                # with reference context by slicing on the token axis (last dim).
                gen_codes = frag[:, 0, :]
                total_tokens = gen_codes.shape[-1]
                kept_segments = []
                chunk_idx = 0
                ref_samples = int(round(20.0 * self.output_sample_rate))
                while True:
                    st_tok = chunk_idx * self.stride_tokens
                    if st_tok >= total_tokens:
                        break
                    ed_tok = min(st_tok + self.tokens_per_chunk, total_tokens)
                    gen_chunk = gen_codes[:, st_tok:ed_tok]
                    if gen_chunk.shape[-1] == 0:
                        break

                    # Concatenate reference tokens with current window tokens
                    combined_codes = torch.cat(
                        [
                            ref_codes,
                            gen_chunk.to(device=codec_device, dtype=torch.long),
                        ],
                        dim=1,
                    )  # (nq, T_ref + T_chunk)
                    codes_lengths = torch.tensor(
                        [combined_codes.shape[-1]],
                        dtype=torch.long,
                        device=codec_device,
                    )
                    combined_codes_batched = combined_codes.unsqueeze(1)  # (nq, 1, T)
                    with torch.no_grad():
                        audio_values = self.codec_model._decode(
                            combined_codes_batched,
                            codes_lengths,
                        )["audio_values"][0, 0]

                    # Remove 20s reference portion (in samples)
                    if audio_values.shape[-1] <= ref_samples:
                        chunk_idx += 1
                        continue
                    chunk_audio_values = audio_values[ref_samples:]

                    # Determine kept region within current window
                    window_len = gen_chunk.shape[-1]
                    remains = total_tokens - st_tok
                    is_first = chunk_idx == 0
                    is_last = ed_tok >= total_tokens
                    if is_first:
                        keep_start_tok = 0
                        keep_end_tok = min(
                            self.keep_tokens + self.left_ctx_tokens, window_len
                        )
                    elif is_last and remains < 105:
                        keep_start_tok = (
                            0 if is_first else min(self.left_ctx_tokens, window_len)
                        )
                        keep_end_tok = window_len
                    else:
                        keep_start_tok = min(self.left_ctx_tokens, window_len)
                        keep_end_tok = min(
                            self.left_ctx_tokens + self.keep_tokens, window_len
                        )
                    keep_start_smps = keep_start_tok * self.samples_per_token
                    keep_end_smps = keep_end_tok * self.samples_per_token
                    left_margin = 0
                    right_margin = self.crossfade_samples if not is_last else 0
                    seg_start = max(0, keep_start_smps - left_margin)
                    seg_end = min(
                        chunk_audio_values.shape[-1], keep_end_smps + right_margin
                    )
                    if seg_end > seg_start:
                        kept_segments.append(
                            chunk_audio_values[seg_start:seg_end].unsqueeze(0)
                        )

                    chunk_idx += 1

                # Concatenate with crossfade; if empty, return tiny silence
                if len(kept_segments) == 0:
                    audio = torch.zeros(1, int(0.01 * self.output_sample_rate))
                else:
                    audio = self.crossfade_concat(
                        kept_segments,
                        self.output_sample_rate,
                        crossfade_seconds=self.crossfade_seconds,
                    )
        else:
            audio = None
        return audio.detach().cpu()

    def batch_decode(
        self,
        audio_codes: list[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> list[tuple[torch.Tensor, int]]:
        if audio_codes is None or len(audio_codes) == 0:
            return None
        audio_wavs = []
        for i, audio_code in enumerate(audio_codes):
            audio_wav = self.decode(audio_code, forward_batch, i)
            audio_wavs.append(
                (audio_wav, self.codec_model.output_sample_rate)
                if audio_wav is not None
                else None
            )
        return audio_wavs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        for name, loaded_weight in weights:
            name = self.remove_prefix(name, "model.")
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.language_model, "start_layer")
                and (
                    layer_id < self.language_model.start_layer
                    or layer_id >= self.language_model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(
                            lambda x: x[0] == "language_model.embed_tokens.weight",
                            weights,
                        )
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if (
                name.startswith("language_model.vision_tower")
                and name not in params_dict
            ):
                continue

            if name.startswith("codec_model."):
                # Backward-compat: older XYTokenizer checkpoints stored sinusoidal embeddings
                # as a direct buffer `...positional_embedding`. The current implementation wraps
                # it in a module and stores the tensor at `...positional_embedding.weight`.
                if (
                    name.endswith(".positional_embedding")
                    and name not in params_dict
                    and (name + ".weight") in params_dict
                ):
                    name = name + ".weight"
                if name in params_dict.keys():
                    param = params_dict[name]
                    # Bypass dtype conversion: directly replace param data with
                    # checkpoint dtype to preserve original precision
                    weight_to_load = loaded_weight.to(device=param.device)
                    # Validate shape to catch mismatches early with a clear error message
                    assert param.size() == weight_to_load.size(), (
                        f"Attempted to load weight ({weight_to_load.size()}) "
                        f"into parameter ({param.size()}) for {name}"
                    )
                    param.data = weight_to_load
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = MossTTSDWithCodec
