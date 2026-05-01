from typing import List, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import get_global_server_args


class AdditionalSampleProcessMixin:
    def __init__(self):
        self._postprocess_dispatcher = {
            "MossTTSDWithCodec": self.moss_ttsd_postprocess_tokens,
            "MossTTSDForCausalLM": self.moss_ttsd_postprocess_tokens,
            "MossTTSDelayWithCodec": self.moss_tts_delay_postprocess_tokens,
            "MossTTSDdelayWithCodec": self.moss_tts_delay_postprocess_tokens,
        }
        self._preprocess_dispatcher = {
            "MossTTSDWithCodec": self.moss_ttsd_preprocess_logits,
            "MossTTSDForCausalLM": self.moss_ttsd_preprocess_logits,
            "MossTTSDelayWithCodec": self.moss_tts_delay_preprocess_logits,
            "MossTTSDdelayWithCodec": self.moss_tts_delay_preprocess_logits,
        }

    def preprocess_logits(
        self,
        logits_output: Union[LogitsProcessorOutput, List[LogitsProcessorOutput]],
        sampling_info: SamplingBatchInfo,
        current_generation_step: torch.Tensor,
        is_audio_stage: torch.Tensor,
    ):
        if get_global_server_args().delay_pattern:
            model_arch = self.model_config.hf_config.architectures[0]
            preprocess_fn = self._preprocess_dispatcher.get(model_arch)
            if preprocess_fn is not None:
                return preprocess_fn(
                    logits_output,
                    sampling_info,
                    current_generation_step,
                    is_audio_stage,
                )

    def _postprocess_tokens(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if get_global_server_args().delay_pattern:
            model_arch = self.model_config.hf_config.architectures[0]
            postprocess_fn = self._postprocess_dispatcher.get(model_arch)
            if postprocess_fn is not None:
                (
                    forward_batch.current_generation_step,
                    forward_batch.needs_additional_steps,
                    forward_batch.unfinished_sequences,
                    forward_batch.is_audio_stage,
                ) = postprocess_fn(
                    next_token_ids,
                    forward_batch.current_generation_step,
                    forward_batch.truncated_input_ids,
                    forward_batch.needs_additional_steps,
                    forward_batch.unfinished_sequences,
                    forward_batch.is_audio_stage,
                )

    def is_speech_token(self, token_id: torch.Tensor) -> torch.Tensor:
        return (token_id <= self.model_config.hf_text_config.speech_token_range[1]) & (
            token_id >= self.model_config.hf_text_config.speech_token_range[0]
        )

    def moss_ttsd_preprocess_logits(
        self,
        logits_output: Union[LogitsProcessorOutput, List[LogitsProcessorOutput]],
        sampling_info: SamplingBatchInfo,
        current_generation_step: torch.Tensor,
        is_audio_stage: torch.Tensor,
    ):
        steps = current_generation_step
        if steps.device != logits_output[0].next_token_logits.device:
            steps = steps.to(device=logits_output[0].next_token_logits.device)
        if (
            steps.dim() != 1
            or steps.shape[0] != logits_output[0].next_token_logits.shape[0]
        ):
            raise ValueError(
                "current_generation_step must be a 1D tensor of shape [batch]. "
                f"Got {tuple(steps.shape)} with batch={logits_output[0].next_token_logits.shape[0]}"
            )

        for i, logits in enumerate(logits_output):
            pad_id = self.model_config.hf_config.pad_token[i]
            if i != 0:
                row_mask = (steps + 1) > i
                if torch.any(row_mask):
                    logits.next_token_logits[row_mask, pad_id] = -torch.inf
            else:
                row_mask = steps <= (self.model_config.channels - 1)
                if torch.any(row_mask):
                    logits.next_token_logits[row_mask, pad_id] = -torch.inf
        return logits_output

    def moss_ttsd_postprocess_tokens(
        self,
        next_token_ids: torch.Tensor,
        current_generation_step: torch.Tensor,
        truncated_input_ids: torch.Tensor,
        needs_additional_steps: torch.Tensor,
        unfinished_sequences: torch.Tensor,
        is_audio_stage: torch.Tensor,
    ):
        channels = self.model_config.channels
        indices = (~self.is_speech_token(next_token_ids[:, 0])) & (
            needs_additional_steps < 0
        )
        needs_additional_steps[indices] = channels - 1

        # current_generation_step is a per-sample tensor of shape [batch].
        # truncated_input_ids is expected to be [batch, channels - 1, channels].
        # For each sample b with step s < channels - 1, we fill positions (s+1 .. end)
        # from truncated_input_ids[b, s, :].
        steps = current_generation_step
        if steps.device != next_token_ids.device:
            steps = steps.to(device=next_token_ids.device)
        if steps.dim() != 1 or steps.shape[0] != next_token_ids.shape[0]:
            raise ValueError(
                "current_generation_step must be a 1D tensor of shape [batch]. "
                f"Got {tuple(steps.shape)} with batch={next_token_ids.shape[0]}"
            )

        valid = steps < (channels - 1)
        # Additional guard: if this sample's truncated_input_ids are all zeros,
        # treat it as invalid for delay-pattern filling.
        if (
            isinstance(truncated_input_ids, torch.Tensor)
            and truncated_input_ids.dim() >= 2
            and truncated_input_ids.shape[0] == next_token_ids.shape[0]
        ):
            all_zero = (truncated_input_ids == 0).flatten(start_dim=1).all(dim=1)
            if all_zero.device != valid.device:
                all_zero = all_zero.to(device=valid.device)
            valid = valid & (~all_zero)
        if torch.any(valid):
            truncated = truncated_input_ids
            if truncated.device != next_token_ids.device:
                truncated = truncated.to(device=next_token_ids.device)
            if (
                truncated.dim() == 3
                and truncated.shape[0] == next_token_ids.shape[0]
                and truncated.shape[1] == channels - 1
                and truncated.shape[2] == channels
            ):
                steps_idx = steps.to(torch.long).clamp(min=0, max=channels - 2)
                batch_idx = torch.arange(
                    next_token_ids.shape[0], device=next_token_ids.device
                )
                gathered = truncated[batch_idx, steps_idx, :].to(
                    dtype=next_token_ids.dtype
                )
                pos = torch.arange(channels, device=next_token_ids.device).unsqueeze(0)
                fill_mask = valid.unsqueeze(1) & (pos > steps.unsqueeze(1))
                next_token_ids[fill_mask] = gathered[fill_mask]
            else:
                # Fallback: use pad tokens (per-channel), only for positions beyond step.
                pos = torch.arange(channels, device=next_token_ids.device).unsqueeze(0)
                fill_mask = valid.unsqueeze(1) & (pos > steps.unsqueeze(1))
                pad_row = torch.tensor(
                    self.model_config.hf_config.pad_token,
                    dtype=next_token_ids.dtype,
                    device=next_token_ids.device,
                ).unsqueeze(0)
                next_token_ids[fill_mask] = pad_row.expand_as(next_token_ids)[fill_mask]

        # Handle tensor boolean operations properly
        mask = (needs_additional_steps > 0) & (needs_additional_steps < channels - 1)
        if torch.any(mask):
            next_token_ids[mask, 0] = next(iter(self.model_config.hf_eos_token_id))
            for i in range(1, channels):
                channel_mask = mask & (needs_additional_steps < channels - i)
                if torch.any(channel_mask):
                    next_token_ids[channel_mask, i] = (
                        self.model_config.hf_config.pad_token[i]
                    )

        for i in range(channels):
            pddp = (
                next(iter(self.model_config.hf_eos_token_id))
                if i == 0
                else self.model_config.hf_config.pad_token[i]
            )
            next_token_ids[:, i] = next_token_ids[
                :, i
            ] * unfinished_sequences + pddp * (1 - unfinished_sequences)

        needs_additional_steps = torch.where(
            needs_additional_steps > 0,
            needs_additional_steps - 1,
            needs_additional_steps,
        )

        stopping = (
            next_token_ids[:, 0] == self.model_config.hf_config.pad_token[0]
        ) | (needs_additional_steps == 0)
        unfinished_sequences = unfinished_sequences & ~stopping
        unfinished_sequences = unfinished_sequences | (needs_additional_steps > 0)
        current_generation_step += 1
        return (
            current_generation_step,
            needs_additional_steps,
            unfinished_sequences,
            is_audio_stage,
        )

    def moss_tts_delay_preprocess_logits(
        self,
        logits_output: Union[LogitsProcessorOutput, List[LogitsProcessorOutput]],
        sampling_info: SamplingBatchInfo,
        current_generation_step: torch.Tensor,
        is_audio_stage: torch.Tensor,
    ):
        pre_exclude_mask0 = torch.tensor(
            [
                self.model_config.hf_config.text_pad_id,
                self.model_config.hf_config.audio_assistant_gen_slot_token_id,
                self.model_config.hf_config.audio_assistant_delay_slot_token_id,
                self.model_config.hf_config.audio_end_token_id,
            ],
            device=logits_output[0].next_token_logits.device,
        )
        pre_exclude_mask1 = torch.ones(
            self.model_config.vocab_size,
            device=logits_output[0].next_token_logits.device,
        ).bool()
        pre_exclude_mask1[
            [
                self.model_config.hf_config.audio_assistant_gen_slot_token_id,
                self.model_config.hf_config.audio_assistant_delay_slot_token_id,
            ]
        ] = False
        audio_stage_mask = is_audio_stage.bool()
        if torch.any(~audio_stage_mask):
            # Use proper broadcasting: row_indices [N, 1] x col_indices [1, M] -> [N, M]
            row_indices = (~audio_stage_mask).nonzero(as_tuple=False).squeeze(-1)
            logits_output[0].next_token_logits[
                row_indices.unsqueeze(1), pre_exclude_mask0.unsqueeze(0)
            ] = -torch.inf
        if torch.any(audio_stage_mask):
            # pre_exclude_mask1 is a boolean mask, get the column indices where it's True
            row_indices = audio_stage_mask.nonzero(as_tuple=False).squeeze(-1)
            col_indices = pre_exclude_mask1.nonzero(as_tuple=False).squeeze(-1)
            logits_output[0].next_token_logits[
                row_indices.unsqueeze(1), col_indices.unsqueeze(0)
            ] = -torch.inf

        steps = current_generation_step
        if steps.device != logits_output[0].next_token_logits.device:
            steps = steps.to(device=logits_output[0].next_token_logits.device)
        if (
            steps.dim() != 1
            or steps.shape[0] != logits_output[0].next_token_logits.shape[0]
        ):
            raise ValueError(
                "current_generation_step must be a 1D tensor of shape [batch]. "
                f"Got {tuple(steps.shape)} with batch={logits_output[0].next_token_logits.shape[0]}"
            )
        indices = steps <= self.model_config.hf_config.n_vq
        if torch.any(indices):
            logits_output[0].next_token_logits[
                indices, self.model_config.hf_config.eos_token_id
            ] = -torch.inf
        for logits in logits_output[1:]:
            logits.next_token_logits[:, self.model_config.hf_config.audio_pad_code] = (
                -torch.inf
            )

        return logits_output

    def moss_tts_delay_postprocess_tokens(
        self,
        next_token_ids: torch.Tensor,
        current_generation_step: torch.Tensor,
        truncated_input_ids: torch.Tensor,
        needs_additional_steps: torch.Tensor,
        unfinished_sequences: torch.Tensor,
        is_audio_stage: torch.Tensor,
    ):
        # Text channel tokens handling
        audio_start_token_id = self.model_config.hf_config.audio_start_token_id
        audio_end_token_id = self.model_config.hf_config.audio_end_token_id
        audio_pad_code = self.model_config.hf_config.audio_pad_code
        channels = self.model_config.hf_config.n_vq
        audio_assistant_delay_slot_token_id = (
            self.model_config.hf_config.audio_assistant_delay_slot_token_id
        )
        audio_assistant_gen_slot_token_id = (
            self.model_config.hf_config.audio_assistant_gen_slot_token_id
        )

        # Vectorized update for column-0 tokens.
        # - needs_additional_steps <  channels => force gen_delay
        # - needs_additional_steps == channels => force audio_eos
        # - needs_additional_steps >  channels => keep sampled token
        col0 = next_token_ids[:, 0]
        col0 = torch.where(
            needs_additional_steps < channels,
            col0.new_full((), audio_assistant_delay_slot_token_id),
            col0,
        )
        col0 = torch.where(
            needs_additional_steps == channels,
            col0.new_full((), audio_end_token_id),
            col0,
        )
        next_token_ids[:, 0] = col0

        # Vectorized update for audio stage flags.
        # - When needs_additional_steps == channels, audio stage ends.
        # - When we emit audio_bos, audio stage begins.
        is_audio_stage = torch.where(
            needs_additional_steps == channels,
            torch.zeros_like(is_audio_stage),
            is_audio_stage,
        )
        is_audio_stage = torch.where(
            next_token_ids[:, 0] == audio_start_token_id,
            torch.ones_like(is_audio_stage),
            is_audio_stage,
        )

        # If EOS is generated, the sequence is finished.
        eos_mask = next_token_ids[:, 0] == self.model_config.hf_config.eos_token_id
        unfinished_sequences = torch.where(
            eos_mask,
            torch.zeros_like(unfinished_sequences),
            unfinished_sequences,
        )

        # Audio channel tokens handling
        pre_audio_mask = current_generation_step.unsqueeze(1) > torch.arange(
            channels, dtype=int, device=next_token_ids.device
        ).expand(current_generation_step.size(0), channels)
        post_audio_mask = (
            torch.arange(channels, dtype=int, device=next_token_ids.device).expand(
                needs_additional_steps.size(0), channels
            )
            > needs_additional_steps.unsqueeze(1) - 1
        )
        post_audio_mask[needs_additional_steps == torch.iinfo(torch.int64).max] = True
        sampling_audio_mask = pre_audio_mask & post_audio_mask
        next_token_ids[:, 1:][~sampling_audio_mask] = audio_pad_code

        current_generation_step[
            (next_token_ids[:, 0] == audio_start_token_id)
            | (next_token_ids[:, 0] == audio_assistant_gen_slot_token_id)
            | (next_token_ids[:, 0] == audio_assistant_delay_slot_token_id)
        ] += 1
        current_generation_step[next_token_ids[:, 0] == audio_end_token_id] = 0
        needs_additional_steps[
            (needs_additional_steps == torch.iinfo(torch.int64).max)
            & (next_token_ids[:, 0] == audio_assistant_delay_slot_token_id)
        ] = 0
        needs_additional_steps[
            needs_additional_steps != torch.iinfo(torch.int64).max
        ] += 1
        needs_additional_steps[needs_additional_steps > channels] = torch.iinfo(
            torch.int64
        ).max

        return (
            current_generation_step,
            needs_additional_steps,
            unfinished_sequences,
            is_audio_stage,
        )
