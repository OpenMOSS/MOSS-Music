from typing import Any, Dict, Optional

from transformers import PretrainedConfig, Qwen3Config

from sglang.third_party.moss_tts.configuration_moss_audio_tokenizer import (
    MossAudioTokenizerConfig,
)


class MossTTSDelayWithCodecConfig(PretrainedConfig):
    """Configuration class for MossTTSDelayWithCodec model.

    This config combines the MossTTS language model config with the
    MossAudioTokenizer codec config.
    """

    model_type = "moss_tts_delay_with_codec"

    def __init__(
        self,
        language_config: Optional[Any] = None,
        codec_config: Optional[Any] = None,
        audio_start_token_id: int = 151652,
        audio_end_token_id: int = 151653,
        audio_pad_code: int = 1024,
        audio_user_slot_token_id: int = 151654,
        audio_vocab_size: int = 1024,
        eos_token_id: int = 151645,
        audio_assistant_delay_slot_token_id: int = 151662,
        audio_assistant_gen_slot_token_id: int = 151656,
        hidden_size: int = 4096,
        initializer_range: float = 0.02,
        n_vq: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Language model config (Qwen3)
        if language_config is None:
            self.language_config = Qwen3Config()
        elif isinstance(language_config, Qwen3Config):
            self.language_config = language_config
        elif isinstance(language_config, dict):
            self.language_config = Qwen3Config(**language_config)
        else:
            self.language_config = Qwen3Config()

        # Codec config (MossAudioTokenizer)
        if codec_config is None:
            self.codec_config = MossAudioTokenizerConfig()
        elif isinstance(codec_config, MossAudioTokenizerConfig):
            self.codec_config = codec_config
        elif isinstance(codec_config, dict):
            self.codec_config = MossAudioTokenizerConfig(**codec_config)
        else:
            self.codec_config = MossAudioTokenizerConfig()

        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.audio_pad_code = audio_pad_code
        self.audio_user_slot_token_id = audio_user_slot_token_id
        self.audio_vocab_size = audio_vocab_size
        self.eos_token_id = eos_token_id
        self.audio_assistant_delay_slot_token_id = audio_assistant_delay_slot_token_id
        self.audio_assistant_gen_slot_token_id = audio_assistant_gen_slot_token_id
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.n_vq = n_vq

        self.vocab_size = self.language_config.vocab_size
        self.text_pad_id = self.language_config.pad_token_id

        # Align core transformer parameters with the underlying language config if provided.
        if isinstance(self.language_config, Qwen3Config):
            skip_keys = {"model_type", "architectures"}
            for key, val in self.language_config.to_dict().items():
                if key in skip_keys:
                    continue
                if hasattr(self, key):
                    setattr(self, key, val)

        # Infer channel count from `n_vq` (number of audio codebooks) if it is not provided.
        n_vq = getattr(self, "n_vq", None)
        if n_vq is not None:
            self.channels = int(n_vq) + 1  # 1 text channel + n_vq audio channels
        else:
            self.channels = 1

        self.vocab_size_list = self._init_vocab_size_list(None)
        self.pad_token = self._init_pad_token_list(None)

        if len(self.vocab_size_list) != self.channels:
            raise ValueError(
                f"AsteroidConfig expects vocab_size_list length {self.channels}, "
                f"got {len(self.vocab_size_list)}"
            )
        if len(self.pad_token) != self.channels:
            raise ValueError(
                f"AsteroidConfig expects pad_token length {self.channels}, "
                f"got {len(self.pad_token)}"
            )

        # Propagate multi-channel metadata back into the underlying language config
        if isinstance(self.language_config, Qwen3Config):
            setattr(self.language_config, "channels", self.channels)
            setattr(self.language_config, "vocab_size_list", self.vocab_size_list)
            setattr(self.language_config, "pad_token", self.pad_token)

    def get_language_config(self) -> Qwen3Config:
        """Return the language config as a Qwen3Config object."""
        cfg = getattr(self, "language_config", None)
        if isinstance(cfg, Qwen3Config):
            return cfg
        if isinstance(cfg, dict):
            return Qwen3Config(**cfg)
        return Qwen3Config()

    def get_codec_config(self) -> MossAudioTokenizerConfig:
        """Return the codec config as a MossAudioTokenizerConfig object."""
        cfg = getattr(self, "codec_config", None)
        if isinstance(cfg, MossAudioTokenizerConfig):
            return cfg
        if isinstance(cfg, dict):
            return MossAudioTokenizerConfig(**cfg)
        return MossAudioTokenizerConfig()

    def _init_vocab_size_list(self, vocab_size_list):
        if vocab_size_list:
            if len(vocab_size_list) != self.channels:
                raise ValueError(
                    f"Expected vocab_size_list length {self.channels}, "
                    f"got {len(vocab_size_list)}"
                )
            return list(map(int, vocab_size_list))

        text_vocab = int(self.vocab_size)
        audio_vocab = getattr(self, "audio_vocab_size", None)
        if audio_vocab is None:
            # Fallback for bare instantiations (e.g. default constructor in HF utils).
            return [text_vocab] * self.channels

        return [text_vocab] + [int(audio_vocab) + 1] * (self.channels - 1)

    def _init_pad_token_list(self, pad_token):
        if pad_token:
            if len(pad_token) != self.channels:
                raise ValueError(
                    f"Expected pad_token length {self.channels}, got {len(pad_token)}"
                )
            return list(map(int, pad_token))

        text_pad = getattr(self, "text_pad_id", 0) or 0
        audio_vocab = getattr(self, "audio_vocab_size", None)
        audio_pad = int(audio_vocab) if audio_vocab is not None else 0

        return [int(text_pad)] + [audio_pad] * (self.channels - 1)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["language_config"] = self.get_language_config().to_dict()
        output["codec_config"] = self.get_codec_config().to_dict()

        # Keep nested language_config dtype in sync with the top-level dtype.
        # Transformers may auto-infer and write top-level `dtype` based on model
        # weights when saving, but nested configs won't be updated automatically.
        top_level_dtype = output.get("dtype", None)
        if isinstance(top_level_dtype, str) and isinstance(
            output.get("language_config"), dict
        ):
            output["language_config"]["dtype"] = top_level_dtype

        return output
