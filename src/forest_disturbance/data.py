"""Validated CSV windows and deterministic stratified dataset splitting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowData:
    """Single-index samples, shaped (sample, time, 1), and binary labels."""

    features: np.ndarray
    labels: np.ndarray
    feature_columns: tuple[str, ...]


def load_window_csv(
    path: str | Path,
    *,
    feature_columns: Sequence[str] | None = None,
    label_column: str = "label",
    scale_factor: float = 1.0,
) -> WindowData:
    """Read pre-extracted windows; values are divided by ``scale_factor``.

    Unless feature columns are explicitly provided, every column except the
    label is a temporal feature, in CSV header order. There is no implicit
    rescaling or index-column removal.
    """
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers):
            raise ValueError("CSV must have a header with unique column names")
        if label_column not in headers:
            raise ValueError(f"CSV is missing label column {label_column!r}")
        columns = tuple(
            feature_columns if feature_columns is not None
            else (name for name in headers if name != label_column)
        )
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Specify at least one unique temporal feature column")
        if label_column in columns:
            raise ValueError("Label column cannot be a feature")
        missing = set(columns) - set(headers)
        if missing:
            raise ValueError(f"CSV is missing feature columns: {sorted(missing)}")
        features: list[list[float]] = []
        labels: list[int] = []
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in columns]
                label = float(row[label_column])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV row {line} contains missing or nonnumeric values") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"CSV row {line} contains nonfinite features")
            if label not in (0.0, 1.0):
                raise ValueError(f"CSV row {line}: labels must be 0 or 1")
            features.append(values)
            labels.append(int(label))
    if not features:
        raise ValueError("CSV contains no samples")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        array = (np.asarray(features, dtype=np.float64) / scale_factor).astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Scaled CSV features exceed float32 range")
    return WindowData(array[..., None], np.asarray(labels, dtype=np.int64), columns)


class WindowDataset(Dataset):
    """CPU tensors; batches move to the chosen device in the training loop."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def stratified_split(
    labels: np.ndarray, validation_fraction: float = 0.2, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each class once; validation is never oversampled or shuffled later."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    train_parts, validation_parts = [], []
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Training requires both binary classes 0 and 1")
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("Each class needs at least two samples for train/validation splitting")
        rng.shuffle(indices)
        validation_count = max(1, min(len(indices) - 1, round(len(indices) * validation_fraction)))
        validation_parts.append(indices[:validation_count])
        train_parts.append(indices[validation_count:])
    train_indices = np.concatenate(train_parts)
    rng.shuffle(train_indices)
    return train_indices, np.sort(np.concatenate(validation_parts))
