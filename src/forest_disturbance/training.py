"""Reproducible training with an untouched validation split and early stopping."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import WindowDataset, load_window_csv, stratified_split
from .model import ModelConfig, PETransformerModel, save_checkpoint


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    best_epoch: int
    best_validation_loss: float
    history: tuple[EpochMetrics, ...]
    train_samples: int
    validation_samples: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU runs are reproducible; accelerator reproducibility also depends on
    # the selected PyTorch version, hardware, and available deterministic kernels.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _run_epoch(
    model: PETransformerModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    criterion = nn.NLLLoss()
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(inputs)
            loss = criterion(log_probabilities, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; check input values and scale_factor")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_count = labels.numel()
            total_loss += float(loss.detach()) * batch_count
            correct += int((log_probabilities.argmax(dim=1) == labels).sum())
            count += batch_count
    return total_loss / count, correct / count


def train_csv(
    csv_path: str | Path,
    checkpoint_path: str | Path,
    *,
    config: ModelConfig | None = None,
    feature_columns: Sequence[str] | None = None,
    label_column: str = "label",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    patience: int = 20,
    seed: int = 42,
    device: str | torch.device = "cpu",
    scale_factor: float = 1.0,
    balance_classes: bool = True,
) -> TrainingResult:
    """Train from numeric temporal columns and a ``label`` column of 0/1.

    The best validation-loss checkpoint is always written, including on the
    first epoch. Class balancing applies only to the training loader. The
    caller should supply an independent spatial/temporal test set for final
    scientific evaluation; this random split is for early stopping only.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("patience", patience)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if Path(csv_path).resolve() == Path(checkpoint_path).resolve():
        raise ValueError("Checkpoint path must differ from the training CSV")
    data = load_window_csv(
        csv_path, feature_columns=feature_columns, label_column=label_column,
        scale_factor=scale_factor,
    )
    cfg = config or ModelConfig(window_size=data.features.shape[1])
    if cfg.input_dim != 1 or cfg.num_classes != 2:
        raise ValueError("CSV training requires input_dim=1 and num_classes=2")
    if data.features.shape[1] != cfg.window_size:
        raise ValueError(f"CSV has {data.features.shape[1]} features; config expects {cfg.window_size}")
    train_idx, validation_idx = stratified_split(data.labels, validation_fraction, seed)
    _seed_everything(seed)
    device = torch.device(device)
    model = PETransformerModel(cfg).to(device)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balance_classes:
        train_labels = data.labels[train_idx]
        counts = np.bincount(train_labels, minlength=2)
        weights = torch.as_tensor(1.0 / counts[train_labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    train_loader = DataLoader(
        WindowDataset(data.features[train_idx], data.labels[train_idx]),
        batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
        generator=generator,
    )
    validation_loader = DataLoader(
        WindowDataset(data.features[validation_idx], data.labels[validation_idx]),
        batch_size=batch_size, shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = Path(checkpoint_path)
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    history: list[EpochMetrics] = []
    metadata = {
        "scale_factor": float(scale_factor),
        "feature_columns": list(data.feature_columns),
        "label_column": label_column,
        "seed": seed,
        "train_samples": len(train_idx),
        "validation_samples": len(validation_idx),
        "validation_fraction": validation_fraction,
        "balance_classes": balance_classes,
    }
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, device, optimizer)
        validation_loss, validation_accuracy = _run_epoch(model, validation_loader, device)
        metrics = EpochMetrics(epoch, train_loss, train_accuracy, validation_loss, validation_accuracy)
        history.append(metrics)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            save_checkpoint(
                model, checkpoint_path,
                metadata={**metadata, "best_epoch": epoch, "metrics": asdict(metrics)},
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return TrainingResult(
        checkpoint_path, best_epoch, best_loss, tuple(history), len(train_idx), len(validation_idx),
    )
