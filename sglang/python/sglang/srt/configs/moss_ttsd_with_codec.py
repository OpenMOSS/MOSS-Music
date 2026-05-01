from typing import Any, Dict, Optional

from transformers import PretrainedConfig

from sglang.third_party.moss_ttsd.configuration_xy_tokenizer import XYTokenizerConfig


class MossTTSDWithCodecConfig(PretrainedConfig):
    """Configuration class for Moss-TTSD-With-Codec model.

    This config uses XY-Tokenizer codec config together with top-level
    Moss TTSD generation/runtime fields.
    """

    model_type = "moss_ttsd_with_codec"

    def __init__(
        self,
        codec_config: Optional[Any] = None,
        channels: int = 8,
        eos_token_id: int = 151643,
        hidden_size: int = 2048,
        initializer_range: float = 0.02,
        pad_token_id: int = 151643,
        pad_token=None,
        speech_eos_token: int = 152694,
        speech_pad_token: int = 1024,
        speech_token_range=None,
        speech_vocab_size: int = 1025,
        vocab_size: int = 152697,
        vocab_size_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Codec config (XYTokenizer)
        if codec_config is None:
            self.codec_config = XYTokenizerConfig()
        elif isinstance(codec_config, XYTokenizerConfig):
            self.codec_config = codec_config
        elif isinstance(codec_config, dict):
            self.codec_config = XYTokenizerConfig(**codec_config)
        else:
            self.codec_config = XYTokenizerConfig()

        self.channels = int(channels)
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.speech_eos_token = speech_eos_token
        self.speech_pad_token = speech_pad_token
        self.speech_vocab_size = speech_vocab_size
        self.vocab_size = int(vocab_size)

        self.vocab_size_list = self._init_vocab_size_list(vocab_size_list)
        self.pad_token = self._init_pad_token_list(pad_token)
        self.speech_token_range = self._init_speech_token_range(speech_token_range)

        if len(self.vocab_size_list) != self.channels:
            raise ValueError(
                f"MossTTSDWithCodecConfig expects vocab_size_list length {self.channels}, "
                f"got {len(self.vocab_size_list)}"
            )
        if len(self.pad_token) != self.channels:
            raise ValueError(
                f"MossTTSDWithCodecConfig expects pad_token length {self.channels}, "
                f"got {len(self.pad_token)}"
            )

    def get_codec_config(self) -> XYTokenizerConfig:
        """Return the codec config as an XYTokenizerConfig object."""
        cfg = getattr(self, "codec_config", None)
        if isinstance(cfg, XYTokenizerConfig):
            return cfg
        if isinstance(cfg, dict):
            return XYTokenizerConfig(**cfg)
        return XYTokenizerConfig()

    def _init_vocab_size_list(self, vocab_size_list):
        if vocab_size_list:
            if len(vocab_size_list) != self.channels:
                raise ValueError(
                    f"Expected vocab_size_list length {self.channels}, "
                    f"got {len(vocab_size_list)}"
                )
            return list(map(int, vocab_size_list))

        text_vocab = int(self.vocab_size)
        speech_vocab = int(self.speech_vocab_size)
        if self.channels <= 1:
            return [text_vocab]
        return [text_vocab] + [speech_vocab] * (self.channels - 1)

    def _init_pad_token_list(self, pad_token):
        if pad_token:
            if len(pad_token) != self.channels:
                raise ValueError(
                    f"Expected pad_token length {self.channels}, got {len(pad_token)}"
                )
            return list(map(int, pad_token))

        text_pad = int(getattr(self, "speech_eos_token", self.pad_token_id))
        if self.channels <= 1:
            return [text_pad]
        return [text_pad] + [int(self.speech_pad_token)] * (self.channels - 1)

    def _init_speech_token_range(self, speech_token_range):
        if speech_token_range:
            # Accept either a flat [start, end] or per-channel ranges.
            if isinstance(speech_token_range[0], (list, tuple)):
                per_channel = [list(map(int, rng)) for rng in speech_token_range]
                expected = max(self.channels - 1, 0)
                if per_channel and len(per_channel) != expected:
                    raise ValueError(
                        "speech_token_range should only cover audio channels "
                        f"(expected {expected}, got {len(per_channel)})"
                    )
                if per_channel:
                    start = min(rng[0] for rng in per_channel)
                    end = max(rng[1] for rng in per_channel)
                    return [start, end]
                return []

            if len(speech_token_range) != 2:
                raise ValueError(
                    "speech_token_range must be [start, end] when provided as a flat list."
                )
            rng = list(map(int, speech_token_range))
            return rng
        return []

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["codec_config"] = self.get_codec_config().to_dict()

        # Keep nested codec_config dtype in sync with the top-level dtype.
        top_level_dtype = output.get("dtype", None)
        if isinstance(top_level_dtype, str) and isinstance(
            output.get("codec_config"), dict
        ):
            output["codec_config"]["dtype"] = top_level_dtype

        return output
