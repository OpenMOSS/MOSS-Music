import re

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.moss_tts_delay_with_codec import (
    MossTTSDdelayWithCodec,
    MossTTSDelayWithCodec,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MossTTSDelayWithCodecMultimodalProcessor(BaseMultimodalProcessor):
    models = [MossTTSDelayWithCodec]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.AUDIO_TOKEN = "<|Audio|>"
        self.AUDIO_TOKEN_REGEX = re.compile(r"<\|Audio\|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_user_slot|>")
        self.template = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""
        self.TOKEN_REGEX = re.compile(r"\$\{token:(\d+)\}")
        self.INSTRUCTION_REGEX = re.compile(r"\$\{instruction:(.*?)\}")
        self.AMBIENT_SOUND_REGEX = re.compile(r"\$\{ambient_sound:(.*?)\}")
        self.AUDIO_SAMPLE_RATE = 24000
        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update(
            {
                "feature_attention_mask": Modality.AUDIO,
                "continuation_feature": Modality.AUDIO,
            }
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize multi-speaker script.

        1. Don't preserve line breaks.
        2. Preserve bracketed segments like [] () <> even when they are not speaker tags.
        3. Remove decorative symbols: 【】《》（）『』「」～~-_.
        4. Internal punctuation ；：、 → ，；keep ？！?.
        5. Multiple 。 keep only the last one, others → ，。
        6. Replace consecutive "哈" (>=2) with "(笑)".
        7. Auto-recognize [S1] / [S2] … tags; if missing, treat as whole segment.
        8. Merge adjacent identical speaker tags.
        """
        # Replace [1], [2] etc. format with [S1], [S2] etc. format
        text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

        # Remove decorative characters
        remove_chars = "【】《》（）『』「」" '"-_“”～~'

        # Use positive lookahead to split text by speaker tags (tags themselves are still preserved)
        segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
        processed_parts = []

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # Extract tags
            m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
            tag, content = m.groups() if m else ("", seg)

            # Remove irrelevant symbols
            content = re.sub(f"[{re.escape(remove_chars)}]", "", content)

            # Handle consecutive "哈" characters: replace 2 or more with "(笑)"
            content = re.sub(r"哈{2,}", "[笑]", content)

            # Handle English laughter (e.g., "haha", "ha ha")
            content = re.sub(
                r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE
            )

            # First handle multi-character punctuation marks
            content = content.replace("——", "，")
            content = content.replace("……", "，")

            # Handle single-character internal punctuation marks
            internal_punct_map = str.maketrans(
                {"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}
            )
            content = content.translate(internal_punct_map)
            content = content.strip()

            # Keep only the final period
            if len(content) > 1:
                last_ch = (
                    "。"
                    if content[-1] == "，"
                    else ("." if content[-1] == "," else content[-1])
                )
                body = content[:-1].replace("。", "，")
                content = body + last_ch

            processed_parts.append({"tag": tag, "content": content})

        if not processed_parts:
            return ""

        # Merge consecutive same speakers
        merged_lines = []
        current_tag = processed_parts[0]["tag"]
        current_content = [processed_parts[0]["content"]]

        for part in processed_parts[1:]:
            if part["tag"] == current_tag and current_tag:
                current_content.append(part["content"])
            else:
                merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
                current_tag = part["tag"]
                current_content = [part["content"]]

        merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

        return "".join(merged_lines).replace("‘", "'").replace("’", "'")

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        # Extract tokens from input_text using TOKEN_REGEX
        tokens = None
        instruction = None
        ambient_sound = None
        if input_text:
            token_match = self.TOKEN_REGEX.match(input_text)
            if token_match:
                tokens = int(float(token_match.group(1)))
                input_text = self.TOKEN_REGEX.sub("", input_text, count=1)

            instruction_match = self.INSTRUCTION_REGEX.match(input_text)
            if instruction_match:
                instruction = instruction_match.group(1)
                input_text = self.INSTRUCTION_REGEX.sub("", input_text, count=1)

            ambient_sound_match = self.AMBIENT_SOUND_REGEX.match(input_text)
            if ambient_sound_match:
                ambient_sound = ambient_sound_match.group(1)
                input_text = self.AMBIENT_SOUND_REGEX.sub("", input_text, count=1)

        input_text = "" if input_text is None else str(self.normalize_text(input_text))

        text = (
            self.template.replace("{instruction}", str(instruction))
            .replace("{tokens}", str(tokens))
            .replace("{volume}", str(None))
            .replace("{quality}", str(None))
            .replace("{sound_event}", str(None))
            .replace("{ambient_sound}", str(ambient_sound))
            .replace("{language}", str(None))
            .replace("{text}", str(input_text))
        )
        if audio_data is not None:
            reference_list = []
            for i, _ in enumerate(audio_data):
                content = f"[S{i+1}]:\n{self.AUDIO_TOKEN}"
                reference_list.append(content)
            references = "\n".join(reference_list)
            text = text.replace("{reference}", references)
        else:
            text = text.replace("{reference}", str(None))

        base_output = self.load_mm_data(
            prompt=text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.AUDIO_SAMPLE_RATE,
        )

        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, use_forced_processor=True
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_token_id": self.audio_token_id,
        }


class MossTTSDdelayWithCodecMultimodalProcessor(
    MossTTSDelayWithCodecMultimodalProcessor
):
    models = [MossTTSDdelayWithCodec]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.template = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
None
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Scene:
{scene}
- Text:
{text}
</user_inst>"""
        self.NORMALIZE_TEXT_REGEX = re.compile(r"\$\{normalize_text\}")

    @staticmethod
    def normalize_text(text: str) -> str:
        text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

        remove_chars = "【】《》（）『』「」" '"-_“”～~‘’'

        segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
        processed_parts = []

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
            tag, content = m.groups() if m else ("", seg)

            content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
            content = re.sub(r"哈{2,}", "[笑]", content)
            content = re.sub(
                r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE
            )

            content = content.replace("——", "，")
            content = content.replace("……", "，")
            content = content.replace("...", "，")
            content = content.replace("⸺", "，")
            content = content.replace("―", "，")
            content = content.replace("—", "，")
            content = content.replace("…", "，")

            internal_punct_map = str.maketrans(
                {"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}
            )
            content = content.translate(internal_punct_map)
            content = content.strip()

            content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)

            if len(content) > 1:
                last_ch = (
                    "。"
                    if content[-1] == "，"
                    else ("." if content[-1] == "," else content[-1])
                )
                body = content[:-1].replace("。", "，")
                content = body + last_ch

            processed_parts.append({"tag": tag, "content": content})

        if not processed_parts:
            return ""

        merged_lines = []
        current_tag = processed_parts[0]["tag"]
        current_content = [processed_parts[0]["content"]]

        for part in processed_parts[1:]:
            if part["tag"] == current_tag and current_tag:
                current_content.append(part["content"])
            else:
                merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
                current_tag = part["tag"]
                current_content = [part["content"]]

        merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

        return "".join(merged_lines).replace("‘", "'").replace("’", "'")

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        # Extract tokens from input_text using TOKEN_REGEX
        instruction = None
        ambient_sound = None
        normalize_text_flag = False
        if input_text:
            normalize_text_match = self.NORMALIZE_TEXT_REGEX.match(input_text)
            if normalize_text_match:
                normalize_text_flag = True
                input_text = self.NORMALIZE_TEXT_REGEX.sub("", input_text, count=1)
            instruction_match = self.INSTRUCTION_REGEX.match(input_text)
            if instruction_match:
                instruction = instruction_match.group(1)
                input_text = self.INSTRUCTION_REGEX.sub("", input_text, count=1)

            ambient_sound_match = self.AMBIENT_SOUND_REGEX.match(input_text)
            if ambient_sound_match:
                ambient_sound = ambient_sound_match.group(1)
                input_text = self.AMBIENT_SOUND_REGEX.sub("", input_text, count=1)

        input_text = (
            ""
            if input_text is None
            else str(
                self.normalize_text(input_text) if normalize_text_flag else input_text
            )
        )

        text = (
            self.template.replace("{instruction}", str(instruction))
            .replace("{tokens}", str(None))
            .replace("{volume}", str(None))
            .replace("{quality}", str(None))
            .replace("{sound_event}", str(None))
            .replace("{ambient_sound}", str(ambient_sound))
            .replace("{language}", str(None))
            .replace("{scene}", "None")
            .replace("{text}", str(input_text))
        )

        if audio_data is not None:
            reference_list = []
            for i, _ in enumerate(audio_data):
                content = f"[S{i+1}]:\n{self.AUDIO_TOKEN}"
                reference_list.append(content)
            references = "\n".join(reference_list)
            text = text.replace("{reference}", references)
        else:
            reference_list = []
            speaker_indices = sorted(
                {
                    int(match.group(1)) - 1
                    for match in re.finditer(r"\[S(\d+)\]", input_text)
                    if int(match.group(1)) > 0
                }
            )
            for speaker_idx in speaker_indices:
                reference_list.append(f"[S{speaker_idx + 1}]: None")

            references = "\n".join(reference_list)
            text = text.replace("{reference}", references)

        base_output = self.load_mm_data(
            prompt=text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.AUDIO_SAMPLE_RATE,
        )

        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, use_forced_processor=True
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_token_id": self.audio_token_id,
        }
