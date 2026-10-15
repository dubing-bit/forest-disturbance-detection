import csv

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from forest_disturbance.data import load_window_csv, stratified_split
from forest_disturbance.model import ModelConfig, load_model
from forest_disturbance.training import _run_epoch, train_csv


def write_csv(path, samples=12):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t0", "t1", "t2", "label"])
        for index in range(samples):
            label = index % 2
            writer.writerow([100 + label * 50, 100 + label * 50, 100 - label * 50, label])


def test_csv_loading_scales_in_explicit_temporal_order(tmp_path):
    path = tmp_path / "train.csv"
    write_csv(path)
    data = load_window_csv(path, feature_columns=["t2", "t0"], scale_factor=100)
    assert data.features.shape == (12, 2, 1)
    assert data.feature_columns == ("t2", "t0")
    np.testing.assert_array_equal(data.features[1, :, 0], [0.5, 1.5])
    assert data.labels.dtype == np.int64


@pytest.mark.parametrize("row", ["1,,3,1", "1,nan,3,0", "1,2,3,1988"])
def test_csv_rejects_invalid_training_values(tmp_path, row):
    path = tmp_path / "invalid.csv"
    path.write_text(f"t0,t1,t2,label\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV row"):
        load_window_csv(path)


def test_split_is_disjoint_stratified_and_reproducible():
    labels = np.asarray([0] * 30 + [1] * 10)
    train, validation = stratified_split(labels, 0.2, 42)
    again_train, again_validation = stratified_split(labels, 0.2, 42)
    np.testing.assert_array_equal(train, again_train)
    np.testing.assert_array_equal(validation, again_validation)
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(len(labels)))
    np.testing.assert_array_equal(np.bincount(labels[validation]), [6, 2])


def test_validation_uses_unrounded_probabilities_and_sample_weighted_loss():
    class FixedModel(torch.nn.Module):
        def forward(self, values):
            return values

    log_probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]).log()
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(log_probabilities, labels), batch_size=2)
    loss, accuracy = _run_epoch(FixedModel(), loader, torch.device("cpu"))
    expected = torch.nn.functional.nll_loss(log_probabilities, labels).item()
    assert loss == pytest.approx(expected)
    assert accuracy == pytest.approx(2 / 3)


def test_training_saves_best_checkpoint_and_is_reproducible(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path)
    config = ModelConfig(window_size=3, d_model=4, n_layers=1, d_inner=8)
    kwargs = dict(config=config, epochs=2, batch_size=4, seed=10, scale_factor=100)
    first = train_csv(csv_path, tmp_path / "first.pth", **kwargs)
    second = train_csv(csv_path, tmp_path / "second.pth", **kwargs)
    assert first.history == second.history
    assert len(first.history) == 2
    assert first.train_samples + first.validation_samples == 12
    assert first.best_validation_loss == min(item.validation_loss for item in first.history)
    model = load_model(first.checkpoint)
    assert model.metadata["scale_factor"] == 100
    assert model.metadata["best_epoch"] == first.best_epoch
    assert model.metadata["validation_samples"] == first.validation_samples
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, load_model(second.checkpoint).state_dict()[key])


def test_early_stopping_tracks_validation_loss(tmp_path, monkeypatch):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path)
    validation_losses = iter([0.5, 0.6, 0.7, 0.1])

    def fake_epoch(model, loader, device, optimizer=None):
        return (0.8, 0.5) if optimizer is not None else (next(validation_losses), 0.5)

    monkeypatch.setattr("forest_disturbance.training._run_epoch", fake_epoch)
    result = train_csv(
        csv_path, tmp_path / "best.pth", config=ModelConfig(window_size=3),
        epochs=8, patience=2,
    )
    assert len(result.history) == 3
    assert result.best_epoch == 1
    assert result.best_validation_loss == 0.5
    assert load_model(result.checkpoint).metadata["best_epoch"] == 1


def test_training_cannot_overwrite_its_csv(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path)
    with pytest.raises(ValueError, match="must differ"):
        train_csv(csv_path, csv_path)
    assert csv_path.read_text(encoding="utf-8").startswith("t0,t1,t2,label")
