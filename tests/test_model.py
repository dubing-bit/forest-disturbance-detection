from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from forest_disturbance.model import ModelConfig, PETransformerModel, load_model, save_checkpoint


def small_config(**kwargs):
    return replace(ModelConfig(window_size=7, d_model=8, n_layers=1, d_inner=12), **kwargs)


def test_eval_predictions_are_independent_of_batch_partition():
    torch.manual_seed(4)
    model = PETransformerModel(small_config()).eval()
    inputs = torch.randn(5, 7, 1)
    with torch.inference_mode():
        whole = model(inputs)
        separate = torch.cat([model(sample[None]) for sample in inputs])
    torch.testing.assert_close(whole, separate, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(whole.exp().sum(dim=1), torch.ones(5))


def test_legacy_sequence_first_forward_is_preserved():
    """Compare against the operations of the original PET implementation."""
    torch.manual_seed(3)
    cfg = small_config()
    model = PETransformerModel(cfg).eval()
    layer = nn.TransformerEncoderLayer(
        cfg.d_model, cfg.n_head, cfg.d_inner, cfg.dropout, activation=cfg.activation,
    )
    reference_encoder = nn.TransformerEncoder(
        layer, cfg.n_layers, nn.LayerNorm(cfg.d_model), enable_nested_tensor=False,
    ).eval()
    reference_encoder.load_state_dict(model.transformerencoder.state_dict())
    inputs = torch.randn(4, cfg.window_size, cfg.input_dim)
    with torch.inference_mode():
        expected = model.inlinear(inputs).transpose(0, 1)
        expected = expected + model.pe.pe
        expected = reference_encoder(expected).transpose(0, 1)
        expected = model.outlinear(expected.max(dim=1).values.relu()).log_softmax(dim=-1)
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("metadata_checkpoint", [False, True])
def test_checkpoint_round_trip(tmp_path, metadata_checkpoint):
    torch.manual_seed(8)
    original = PETransformerModel(small_config()).eval()
    path = tmp_path / "model.pth"
    if metadata_checkpoint:
        save_checkpoint(original, path, metadata={"scale_factor": 1000.0})
    else:
        torch.save(original.state_dict(), path)
    loaded = load_model(path)
    assert not loaded.training
    assert loaded.config == original.config
    inputs = torch.randn(3, 7, 1)
    with torch.inference_mode():
        torch.testing.assert_close(original(inputs), loaded(inputs))
    if metadata_checkpoint:
        assert loaded.metadata["scale_factor"] == 1000.0
    else:
        assert loaded.metadata["legacy_state_dict"]
    with pytest.raises(ValueError, match="window_size"):
        load_model(path, window_size=9)


def test_metadata_preserves_nonlegacy_architecture(tmp_path):
    config = small_config(n_head=2, activation="gelu", positional_dropout=0.2)
    path = tmp_path / "metadata.pth"
    save_checkpoint(PETransformerModel(config), path)
    assert load_model(path).config == config


def test_full_pickled_model_is_rejected(tmp_path):
    path = tmp_path / "unsafe.pkl"
    torch.save(PETransformerModel(small_config()), path)
    with pytest.raises(ValueError, match="safely load"):
        load_model(path)


def test_malformed_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "invalid.pth"
    torch.save({1: torch.zeros(3)}, path)
    with pytest.raises(ValueError, match="parameter names"):
        load_model(path)


def test_real_legacy_pet_checkpoints_are_loadable():
    """Use supplied release weights when present in a source checkout."""
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "models").glob("PET_*.pth"))
    if not paths:
        paths = list(root.glob("实验程序/forest_disturbance_detect/forest_disturbance/savemodel/PET_*.pth"))
    if not paths:
        pytest.skip("Release weights are optional")
    for path in paths:
        model = load_model(path)
        assert model.config.input_dim == 1
        assert model.config.num_classes == 2
        cfg = model.config
        reference_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_head, cfg.d_inner, cfg.dropout,
        )
        reference_encoder = nn.TransformerEncoder(
            reference_layer, cfg.n_layers, nn.LayerNorm(cfg.d_model),
            enable_nested_tensor=False,
        ).eval()
        reference_encoder.load_state_dict(model.transformerencoder.state_dict())
        inputs = torch.linspace(-0.2, 0.8, 2 * cfg.window_size).reshape(2, cfg.window_size, 1)
        with torch.inference_mode():
            output = model(inputs)
            # Exactly the sequence-first computation in the original code,
            # with dropout disabled by eval().
            expected = model.inlinear(inputs).transpose(0, 1) + model.pe.pe
            expected = reference_encoder(expected).transpose(0, 1)
            expected = model.outlinear(expected.max(dim=1).values.relu()).log_softmax(dim=-1)
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bad_shape", [(7, 1), (2, 9, 1), (2, 7, 2)])
def test_forward_rejects_wrong_axes(bad_shape):
    with pytest.raises(ValueError, match="Expected"):
        PETransformerModel(small_config())(torch.zeros(bad_shape))
