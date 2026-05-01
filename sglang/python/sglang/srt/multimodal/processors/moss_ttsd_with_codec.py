import re

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.moss_ttsd_with_codec import MossTTSDWithCodec
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MossTTSDWithCodecMultimodalProcessor(BaseMultimodalProcessor):
    models = [MossTTSDWithCodec]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.AUDIO_TOKEN = "<|audio_bos|><|image_pad|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|image_pad\|>)+<\|audio_eos\|>"
        )
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"ref_audio_features": Modality.AUDIO})
        self.ATTR_NAME_TO_MODALITY.update({"ref_audio_attention_mask": Modality.AUDIO})

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=self.AUDIO_TOKEN * len(audio_data) if audio_data else input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        base_output.input_text = input_text

        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_token_id": self.audio_token_id,
        }
