from dataclasses import dataclass, field
from typing import Any, List, cast

import torch
from transformers import PretrainedConfig, Qwen3Config


@dataclass
class MossAudioEncoderConfig:
    d_model: int = 1280
    output_dim: int = 1280
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    downsample_rate: int = 8
    downsample_hidden_size: int = 480
    encoder_attention_window_size: int = 100
    max_source_positions: int = 1500
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5
    _attn_implementation: str = "eager"
    pretrained_path: str = ""
    n_window: int = 200
    conv_chunksize: int = 64
    deepstack_encoder_layer_indexes: List[int] = field(
        default_factory=lambda: [8, 16, 24]
    )

    @classmethod
    def from_dict(cls, config_dict):
        if config_dict is None:
            return cls()
        allowed_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in allowed_keys}
        return cls(**filtered)

    def to_dict(self):
        return {
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "num_mel_bins": self.num_mel_bins,
            "encoder_layers": self.encoder_layers,
            "encoder_attention_heads": self.encoder_attention_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "downsample_rate": self.downsample_rate,
            "downsample_hidden_size": self.downsample_hidden_size,
            "encoder_attention_window_size": self.encoder_attention_window_size,
            "max_source_positions": self.max_source_positions,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "activation_function": self.activation_function,
            "layer_norm_eps": self.layer_norm_eps,
            "_attn_implementation": self._attn_implementation,
            "pretrained_path": self.pretrained_path,
            "n_window": self.n_window,
            "conv_chunksize": self.conv_chunksize,
            "deepstack_encoder_layer_indexes": list(
                self.deepstack_encoder_layer_indexes or []
            ),
        }


class MossAudioConfig(PretrainedConfig):
    model_type = "moss_audio"
    is_composition = True

    @staticmethod
    def _normalize_dtype_value(dtype: Any) -> str | None:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return str(dtype).removeprefix("torch.")
        if isinstance(dtype, str):
            return dtype.removeprefix("torch.")
        return str(dtype)

    def __init__(
        self,
        adapter_hidden_size=8192,
        audio_config=None,
        deepstack_num_inject_layers=None,
        dtype="bfloat16",
        ignore_index=-100,
        language_config=None,
        **kwargs,
    ):
        requested_dtype = self._normalize_dtype_value(dtype)

        if isinstance(language_config, dict):
            language_config_dict = dict(language_config)
            if requested_dtype is not None and "dtype" not in language_config_dict:
                language_config_dict["dtype"] = requested_dtype
            language_config = Qwen3Config(**language_config_dict)
        elif language_config is None:
            language_config = Qwen3Config(dtype=requested_dtype)
        elif (
            requested_dtype is not None
            and getattr(language_config, "dtype", None) is None
        ):
            language_config.dtype = requested_dtype

        if isinstance(audio_config, dict):
            audio_config = MossAudioEncoderConfig.from_dict(audio_config)
        elif audio_config is None:
            audio_config = MossAudioEncoderConfig()

        self.adapter_hidden_size = adapter_hidden_size
        self.audio_config = audio_config
        self.deepstack_num_inject_layers = deepstack_num_inject_layers
        self.dtype = requested_dtype
        self.ignore_index = ignore_index
        self.language_config = language_config

        for key in ("num_hidden_layers", "eos_token_id", "bos_token_id", "vocab_size"):
            kwargs.setdefault(key, getattr(language_config, key, None))

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output["audio_config"] = (
            self.audio_config.to_dict()
            if hasattr(self.audio_config, "to_dict")
            else self.audio_config
        )
        output["language_config"] = (
            self.language_config.to_dict()
            if hasattr(self.language_config, "to_dict")
            else self.language_config
        )
        output["adapter_hidden_size"] = self.adapter_hidden_size
        output["deepstack_num_inject_layers"] = self.deepstack_num_inject_layers
        output["ignore_index"] = self.ignore_index
        output["dtype"] = self.dtype

        top_level_dtype = output.get("dtype", None)
        language_output = cast(
            dict[str, Any] | None, output.get("language_config", None)
        )
        if isinstance(top_level_dtype, str) and isinstance(language_output, dict):
            language_output["dtype"] = top_level_dtype

        return output
