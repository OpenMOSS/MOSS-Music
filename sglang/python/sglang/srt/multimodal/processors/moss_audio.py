import re
from typing import List, Optional

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.moss_audio import MossAudioModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MossAudioMultimodalProcessor(BaseMultimodalProcessor):
    models = [MossAudioModel]
    always_process_mm_data = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )

        tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        self._time_marker_digit_ids = {
            str(digit): tokenizer.convert_tokens_to_ids(str(digit))
            for digit in range(10)
        }
        self.time_marker_every_seconds = 2
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update(
            {
                "feature_attention_mask": Modality.AUDIO,
                "audio_data": Modality.AUDIO,
                "audio_data_seqlens": Modality.AUDIO,
            }
        )
        if "audio_data" not in self.FEATURE_NAMES:
            self.FEATURE_NAMES.append("audio_data")

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        def conv_out_len(x):
            return (x - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(lengths)))

    @staticmethod
    def _get_raw_feature_lengths(
        audio_item, ret: Optional[dict[str, object]]
    ) -> torch.Tensor:
        if hasattr(audio_item, "audio_data_seqlens"):
            return (
                torch.as_tensor(audio_item.audio_data_seqlens)
                .to(dtype=torch.long)
                .flatten()
            )
        if hasattr(audio_item, "feature_attention_mask"):
            return (
                torch.as_tensor(audio_item.feature_attention_mask)
                .sum(dim=-1)
                .to(dtype=torch.long)
                .flatten()
            )
        if ret is not None and "audio_data_seqlens" in ret:
            return (
                torch.as_tensor(ret["audio_data_seqlens"])
                .to(dtype=torch.long)
                .flatten()
            )
        if ret is not None and "feature_attention_mask" in ret:
            return (
                torch.as_tensor(ret["feature_attention_mask"])
                .sum(dim=-1)
                .to(dtype=torch.long)
                .flatten()
            )
        feature = torch.as_tensor(audio_item.feature)
        if feature.dim() == 3:
            return torch.full(
                (feature.shape[0],), int(feature.shape[-1]), dtype=torch.long
            )
        return torch.tensor([int(feature.shape[-1])], dtype=torch.long)

    def _get_time_marker_token_ids(self, second: int) -> List[int]:
        return [self._time_marker_digit_ids[digit] for digit in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> List[int]:
        total_duration_seconds = audio_seq_len / self.audio_tokens_per_second
        num_full_seconds = int(total_duration_seconds)

        tokens: List[int] = []
        audio_tokens_consumed = 0
        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            audio_segment_len = marker_pos - audio_tokens_consumed
            if audio_segment_len > 0:
                tokens.extend([self.audio_token_id] * audio_segment_len)
                audio_tokens_consumed += audio_segment_len
            tokens.extend(self._get_time_marker_token_ids(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            tokens.extend([self.audio_token_id] * remaining)
        return tokens

    def _apply_time_markers_to_input_ids(
        self, input_ids: torch.Tensor, audio_feature_lens: torch.Tensor
    ) -> torch.Tensor:
        audio_spans = self.get_mm_items_offset_by_pair(
            input_ids=input_ids,
            mm_start_id=self.audio_start_id,
            mm_end_id=self.audio_end_id,
        )
        if not audio_spans:
            return input_ids

        audio_feature_len_list = (
            torch.as_tensor(audio_feature_lens).flatten().long().tolist()
        )
        if len(audio_feature_len_list) != len(audio_spans):
            raise ValueError(
                "Mismatch between audio spans and feature lengths: "
                f"{len(audio_spans)} spans vs {len(audio_feature_len_list)} lengths."
            )

        rewritten_input_ids: List[int] = []
        cursor = 0
        for (audio_start, audio_end), audio_seq_len in zip(
            audio_spans, audio_feature_len_list
        ):
            bos_idx = audio_start - 1
            eos_idx = audio_end + 1

            rewritten_input_ids.extend(input_ids[cursor : bos_idx + 1].tolist())
            rewritten_input_ids.extend(
                self._build_audio_tokens_with_time_markers(int(audio_seq_len))
            )
            rewritten_input_ids.append(self.audio_end_id)
            cursor = eos_idx + 1

        rewritten_input_ids.extend(input_ids[cursor:].tolist())
        return torch.tensor(
            rewritten_input_ids, dtype=input_ids.dtype, device=input_ids.device
        )

    def _build_prompt_for_mm(self, input_text: str, audio_data) -> str:
        prompt_text = self.AUDIO_TOKEN_REGEX.sub("", input_text or "").strip()

        if audio_data:
            audio_prefix = "\n".join([self.AUDIO_TOKEN] * len(audio_data))
            user_content = (
                f"{audio_prefix}\n{prompt_text}" if prompt_text else audio_prefix
            )
        else:
            user_content = prompt_text

        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        _ = image_data, request_obj
        prompt_for_mm = self._build_prompt_for_mm(input_text, audio_data)

        base_output = self.load_mm_data(
            prompt=prompt_for_mm,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output,
            self.mm_tokens,
        )
        audio_item = next((item for item in mm_items if item.is_audio()), None)
        if audio_item is not None:
            raw_lengths = self._get_raw_feature_lengths(audio_item, ret)
            audio_feature_lens = self._compute_downsampled_length(raw_lengths)
            setattr(audio_item, "audio_feature_lens", audio_feature_lens)
            input_ids = self._apply_time_markers_to_input_ids(
                input_ids, audio_feature_lens
            )
            audio_item.offsets = self.get_mm_items_offset(
                input_ids, self.audio_token_id
            )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
