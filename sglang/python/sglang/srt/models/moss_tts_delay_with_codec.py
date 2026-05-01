import logging
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.configs.moss_tts_delay_with_codec import MossTTSDelayWithCodecConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.mm_utils_modified import embed_mm_inputs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.moss_tts_delay import MossTTSDelayModel
from sglang.third_party.moss_tts.modeling_moss_audio_tokenizer import (
    MossAudioTokenizerModel,
)

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


class MossTTSDelayWithCodec(MossTTSDelayModel):
    def __init__(
        self,
        config: MossTTSDelayWithCodecConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.config = config
        self.codec_cfg = config.get_codec_config()
        self.audio_start_token_id = config.audio_start_token_id
        self.audio_end_token_id = config.audio_end_token_id
        self.audio_pad_code = config.audio_pad_code
        self.audio_user_slot_token_id = config.audio_user_slot_token_id
        self.audio_assistant_gen_slot_token_id = (
            config.audio_assistant_gen_slot_token_id
        )
        self.audio_assistant_delay_slot_token_id = (
            config.audio_assistant_delay_slot_token_id
        )
        self.channels = config.channels
        self.n_vq = config.n_vq
        if self.pp_group.is_first_rank or self.pp_group.is_last_rank:
            # Build codec weights in fp32 before checkpoint load.
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.codec_model = MossAudioTokenizerModel(self.codec_cfg)
            finally:
                torch.set_default_dtype(old_dtype)
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    @staticmethod
    def apply_delay_pattern(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        delayed_tokens = torch.full(
            (tokens.shape[0] + tokens.shape[1] - 1, tokens.shape[1]),
            pad_id,
            device=tokens.device,
        )
        delayed_tokens[:, 0] = torch.cat(
            [
                tokens[:, 0],
                torch.full((tokens.shape[1] - 1,), pad_id, device=tokens.device),
            ]
        )
        for i in range(1, tokens.shape[1]):
            delayed_tokens[i : i + tokens.shape[0], i] = tokens[:, i]
        return delayed_tokens

    @torch.inference_mode()
    def get_audio_embeddings(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,
        indices: List[int],
    ) -> torch.Tensor:
        audio_embeddings = []
        for idx, item in enumerate(items):
            features = (
                item.feature if item.feature.dim() == 2 else item.feature.unsqueeze(0)
            )
            feature_attention_mask = item.model_specific_data.get(
                "feature_attention_mask",
                torch.full(
                    (features.shape[0],),
                    features.shape[1],
                    dtype=torch.long,
                    device=features.device,
                ),
            )
            continuation_feature = item.model_specific_data.get(
                "continuation_feature",
                None,
            )
            for i in range(features.shape[0]):
                valid_len = int(feature_attention_mask[i].sum().item())
                feature = features[i][:valid_len]
                enc = self.codec_model.encode(
                    feature.unsqueeze(0).unsqueeze(0),
                    return_dict=True,
                    chunk_duration=10,
                )
                # (time, codebooks)
                audio_codes = enc.audio_codes.squeeze(1).transpose(0, 1)[
                    : enc.audio_codes_lengths, : self.n_vq
                ]
                audio_codes = self.apply_delay_pattern(
                    audio_codes, pad_id=self.audio_pad_code
                )
                audio_codes = F.pad(
                    audio_codes, (1, 0), value=self.audio_user_slot_token_id
                )
                audio_embed = self._prepare_multi_modal_inputs(audio_codes)
                audio_embeddings.append(audio_embed)
            if continuation_feature is not None:
                continuation_feature = continuation_feature.to(self.codec_model.device)
                cont_enc = self.codec_model.encode(
                    continuation_feature.unsqueeze(0).unsqueeze(0),
                    return_dict=True,
                    chunk_duration=10,
                )
                cont_codes = cont_enc.audio_codes.squeeze(1).transpose(0, 1)[
                    : cont_enc.audio_codes_lengths, : self.n_vq
                ]  # (time, codebooks)
                forward_batch.ref_audio_codes[indices[idx]] = cont_codes
                cont_codes = self.apply_delay_pattern(
                    cont_codes, pad_id=self.audio_pad_code
                )[: -(self.n_vq - 1)]
                cont_codes = F.pad(
                    cont_codes, (1, 0), value=self.audio_assistant_gen_slot_token_id
                )  # No delay pattern for continuation feature
                cont_embed = self._prepare_multi_modal_inputs(cont_codes)
                audio_embeddings.append(cont_embed)
        return torch.cat(audio_embeddings, dim=0)

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
                "get_embedding is not implemented for MossTTSDelayWithCodec"
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

    @staticmethod
    def apply_de_delay_pattern(delayed_tokens: torch.Tensor) -> torch.Tensor:
        tokens = delayed_tokens.new_full(
            (
                delayed_tokens.shape[0] - delayed_tokens.shape[1] + 1,
                delayed_tokens.shape[1],
            ),
            0,
        )
        for i in range(delayed_tokens.shape[1]):
            tokens[:, i] = delayed_tokens[i : i + tokens.shape[0], i]
        return tokens

    def _resolve_audio_payload_bounds(
        self, text_token: torch.Tensor
    ) -> tuple[int, int] | None:
        bos_pos = (text_token == self.audio_start_token_id).nonzero(as_tuple=False)
        if bos_pos.numel() == 0:
            assistant_gen_slot = (
                text_token == self.audio_assistant_gen_slot_token_id
            ).nonzero(as_tuple=False)
            if assistant_gen_slot.numel() == 0:
                return None
            fallback_start_idx = int(assistant_gen_slot[0].item())
            if fallback_start_idx < 0:
                return None
            start = fallback_start_idx
        else:
            start = int(bos_pos[0].item()) + 1

        eos_pos = (text_token[start:] == self.audio_end_token_id).nonzero(
            as_tuple=False
        )
        if eos_pos.numel() == 0:
            end_candidates: list[int] = []

            assistant_gen_slot = (
                text_token[start:] == self.audio_assistant_gen_slot_token_id
            ).nonzero(as_tuple=False)
            if assistant_gen_slot.numel() > 0:
                end_candidates.append(start + int(assistant_gen_slot[-1].item()))

            assistant_delay_slot = (
                text_token[start:] == self.audio_assistant_delay_slot_token_id
            ).nonzero(as_tuple=False)
            if assistant_delay_slot.numel() > 0:
                end_candidates.append(start + int(assistant_delay_slot[-1].item()))

            if not end_candidates:
                return None

            # Keep the final slot row inside payload when EOS is absent.
            end = max(end_candidates) + 1
        else:
            end = start + int(eos_pos[0].item())

        if (
            end <= start + self.n_vq
        ):  # Need at least one row of payload after delay pattern
            return None
        return start, end

    def decode(
        self,
        audio_code: torch.Tensor,
        forward_batch: ForwardBatch,
        idx: int,
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

        # Channel 0 carries BOS/EOS markers; codec codebooks are in [1:].
        text_token = audio_code[:, 0]

        payload_bounds = self._resolve_audio_payload_bounds(text_token)
        if payload_bounds is None:
            return torch.zeros(1)

        start, end = payload_bounds

        payload = audio_code[start:end, 1:]
        if payload.numel() == 0 or payload.shape[0] < payload.shape[1]:
            return torch.zeros(1)
        normal = self.apply_de_delay_pattern(payload).to(self.codec_model.device)
        ref_audio_length = None
        if (
            forward_batch.ref_audio_codes is not None
            and forward_batch.ref_audio_codes[idx] is not None
        ):
            ref_audio_length = (
                forward_batch.ref_audio_codes[idx].shape[0]
                * self.codec_cfg.downsample_rate
            )
            normal = torch.concat([forward_batch.ref_audio_codes[idx], normal], dim=0)
        audio = self.codec_model.decode(
            normal.to(self.codec_model.device).transpose(0, 1),
            return_dict=True,
            chunk_duration=8,
        ).audio.squeeze(0)
        if ref_audio_length is not None:
            audio = audio[:, ref_audio_length:]
        return audio.detach().cpu()

    def batch_decode(
        self,
        audio_codes: list[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> list[tuple[torch.Tensor, int]]:
        if audio_codes is None or len(audio_codes) == 0:
            return None
        audio_wavs = []
        for idx, audio_code in enumerate(audio_codes):
            audio_wav = self.decode(audio_code, forward_batch, idx)
            audio_wavs.append(
                (audio_wav, self.codec_model.sampling_rate)
                if audio_wav is not None
                else None
            )
        return audio_wavs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            # Map external audio embeddings onto embedding_list[1:].
            if name.startswith("emb_ext.") and name.endswith(".weight"):
                try:
                    idx = int(name.split(".")[1]) + 1  # shift by +1 for audio ch start
                except Exception:
                    idx = None
                if idx is not None:
                    mapped = f"embedding_list.{idx}.weight"
                    if mapped in params_dict:
                        param = params_dict[mapped]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        continue

            if name.startswith("lm_heads.") and name.endswith(".weight"):
                try:
                    idx = int(name.split(".")[1])
                except Exception:
                    idx = None
                if idx is not None and name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    continue

            # Mirror text embeddings into channel 0 of embedding_list.
            if name == "model.embed_tokens.weight":
                mapped = "embedding_list.0.weight"
                if mapped in params_dict:
                    param = params_dict[mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            if name.startswith("codec_model."):
                if not (self.pp_group.is_first_rank or self.pp_group.is_last_rank):
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_to_load = loaded_weight.to(device=param.device)
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
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
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


class MossTTSDdelayWithCodec(MossTTSDelayWithCodec):
    def __init__(
        self,
        config: MossTTSDelayWithCodecConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)


EntryClass = [MossTTSDelayWithCodec, MossTTSDdelayWithCodec]
