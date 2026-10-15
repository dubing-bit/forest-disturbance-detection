"""Position-encoded Transformer and safe, portable checkpoint handling.

The state-dict names deliberately match the original ``savemodel/PET_*.pth``
files. A sequence is always ``(batch, time, feature)``; attention never mixes
different pixels in a batch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Architecture saved with new checkpoints.

    ``positional_dropout=0.1`` preserves the original positional encoder,
    whose dropout was independent of the Transformer dropout.
    """

    window_size: int = 9
    input_dim: int = 1
    num_classes: int = 2
    d_model: int = 64
    n_head: int = 1
    n_layers: int = 3
    d_inner: int = 128
    dropout: float = 0.2
    activation: str = "relu"
    positional_dropout: float = 0.1

    def __post_init__(self) -> None:
        for field in (
            "window_size", "input_dim", "num_classes", "d_model", "n_head",
            "n_layers", "d_inner",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        for field in ("dropout", "positional_dropout"):
            if not 0 <= getattr(self, field) < 1:
                raise ValueError(f"{field} must lie in [0, 1)")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding with the buffer shape used in legacy PET files."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        position = torch.arange(config.window_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model)
        )
        pe = torch.zeros(config.window_size, 1, config.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[: config.d_model // 2])
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(config.positional_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[: x.shape[1]].transpose(0, 1))


class PETransformerModel(nn.Module):
    """Binary window classifier returning log probabilities.

    Pass a :class:`ModelConfig`, or individual config keywords. ``max_len`` is
    accepted as an alias for ``window_size`` for migration from the original
    constructor.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if "max_len" in kwargs:
            if "window_size" in kwargs:
                raise ValueError("Specify only one of max_len and window_size")
            kwargs["window_size"] = kwargs.pop("max_len")
        self.config = replace(config or ModelConfig(), **kwargs)
        self.metadata: dict[str, Any] = {}
        cfg = self.config
        self.inlinear = nn.Linear(cfg.input_dim, cfg.d_model)
        self.relu = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_head, cfg.d_inner, cfg.dropout,
            activation=cfg.activation, batch_first=True,
        )
        self.transformerencoder = nn.TransformerEncoder(
            encoder_layer, cfg.n_layers, nn.LayerNorm(cfg.d_model),
            enable_nested_tensor=False,
        )
        self.outlinear = nn.Linear(cfg.d_model, cfg.num_classes)
        self.pe = PositionalEncoding(cfg)

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (cfg.window_size, cfg.input_dim):
            raise ValueError(
                f"Expected (batch, {cfg.window_size}, {cfg.input_dim}), "
                f"received {tuple(x.shape)}"
            )
        x = self.pe(self.inlinear(x))
        x = self.transformerencoder(x)
        return F.log_softmax(self.outlinear(self.relu(x.amax(dim=1))), dim=-1)


def save_checkpoint(
    model: PETransformerModel,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save weights and primitive metadata without pickling a model object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(model.config),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": dict(model.metadata if metadata is None else metadata),
        },
        path,
    )


def _legacy_config(state: Mapping[str, Tensor]) -> ModelConfig:
    """Infer tensor dimensions; PET attention head count was one."""
    try:
        layer_ids = {
            int(match.group(1)) for key in state
            if (match := re.match(r"transformerencoder\.layers\.(\d+)\.", key))
        }
        if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
            raise ValueError("Transformer layer keys must be contiguous")
        return ModelConfig(
            window_size=state["pe.pe"].shape[0],
            input_dim=state["inlinear.weight"].shape[1],
            d_model=state["inlinear.weight"].shape[0],
            num_classes=state["outlinear.weight"].shape[0],
            n_layers=len(layer_ids),
            d_inner=state["transformerencoder.layers.0.linear1.weight"].shape[0],
        )
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError("Not a supported PET Transformer state dictionary") from exc


def load_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    window_size: int | None = None,
) -> PETransformerModel:
    """Load a new checkpoint or an original PET state dict in evaluation mode.

    Loading always uses ``weights_only=True``. Serialized full Python model
    objects (such as ``finish_model.pkl``) are deliberately unsupported. Raw
    PET state dictionaries use one attention head, as in the supplied training
    program; custom architectures should use metadata checkpoints instead.
    """
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Cannot safely load checkpoint {checkpoint!s}. Use a PET state "
            "dictionary or a metadata checkpoint; full pickled models are unsupported."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary")

    if "state_dict" in payload:
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format_version")
        state = payload["state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint state_dict must be a mapping")
        config_data = payload.get("model_config")
        if not isinstance(config_data, Mapping):
            raise ValueError("Checkpoint model_config must be a mapping")
        try:
            config = ModelConfig(**config_data)
        except TypeError as exc:
            raise ValueError("Unsupported checkpoint model configuration") from exc
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Checkpoint metadata must be a mapping")
    else:
        state = payload
        if not all(
            isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()
        ):
            raise ValueError("State dictionary must map parameter names to tensors")
        config = _legacy_config(state)
        metadata = {"legacy_state_dict": True}

    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()):
        raise ValueError("State dictionary must map parameter names to tensors")
    if window_size is not None and window_size != config.window_size:
        raise ValueError(
            f"Checkpoint expects window_size={config.window_size}, received {window_size}"
        )
    model = PETransformerModel(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint tensors do not match the PET model configuration") from exc
    model.metadata = dict(metadata)
    return model.to(device).eval()
