"""Independent scalar parity and batching checks for temporal inference."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from forest_disturbance.inference import NODATA, PredictConfig, pad_series, predict_series


class DropClassifier(torch.nn.Module):
    """A deterministic, sample-independent classifier with observable calls."""

    def __init__(self, *, always_positive: bool = False):
        super().__init__()
        self.always_positive = always_positive
        self.batch_sizes: list[int] = []

    def forward(self, inputs):
        if self.training or not torch.is_inference_mode_enabled():
            raise AssertionError("prediction must use eval mode and inference mode")
        self.batch_sizes.append(len(inputs))
        if self.always_positive:
            score = torch.ones(len(inputs), device=inputs.device)
        else:
            score = inputs[:, :3, 0].mean(1) - inputs[:, -3:, 0].mean(1) - 0.08
        return torch.stack((torch.zeros_like(score), score), dim=1)


def original_padding(series: np.ndarray, window_size: int) -> np.ndarray:
    """Use the original repeated np.insert operations, including its right edge."""
    output = series.copy()
    for _ in range(window_size // 2):
        output = np.insert(output, 0, output[:, 0] - 0.001, axis=1)
        output = np.insert(output, -1, output[:, -1] + 0.001, axis=1)
    return output


def scalar_rule(window: np.ndarray, config: PredictConfig) -> int:
    candidates = sorted(
        ((p, window[p] - window[p - 1]) for p in range(2, len(window) - 2)),
        key=lambda pair: pair[1],
    )
    for position, drop in candidates:
        if drop < config.min_drop:
            slope, intercept = np.polyfit(
                [0, 1, 3, 4],
                window[[position - 2, position - 1, position + 1, position + 2]],
                1,
            )
            if slope < config.min_slope and intercept > config.min_intercept:
                return position
    return -1


def scalar_predict(series: np.ndarray, config: PredictConfig) -> np.ndarray:
    """Predict pixel by pixel using original padding and independent polyfit."""
    series = np.asarray(series, dtype=np.float32)
    result = np.full(len(series), NODATA, dtype=np.int16)
    half = config.window_size // 2
    for pixel, raw in enumerate(series):
        if not np.isfinite(raw).all():
            continue
        scaled = raw / config.scale_factor
        if config.padding == "legacy":
            padded = original_padding(scaled[None], config.window_size)[0]
        elif config.padding == "edge":
            padded = np.pad(scaled, (half, half), mode="edge")
        else:
            padded = np.array(
                [scaled[0] - i * 0.001 for i in range(half, 0, -1)]
                + list(scaled)
                + [scaled[-1] + i * 0.001 for i in range(1, half + 1)],
                dtype=np.float32,
            )
        accepted = []
        for start in range(0, len(raw), config.step):
            window = padded[start : start + config.window_size]
            if window[:3].mean() - window[-3:].mean() - 0.08 <= 0:
                continue
            local = scalar_rule(window, config)
            position = start + local - half
            if local >= 0 and 0 <= position < len(raw):
                accepted.append(position)
        result[pixel] = config.start_year + min(accepted) if accepted else 0
    return result


class SeriesInferenceTests(unittest.TestCase):
    def test_batch_predictions_match_independent_scalar_reference(self):
        rng = np.random.default_rng(311)
        series = rng.uniform(-0.2, 0.95, (35, 14)).astype(np.float32) * 10000
        series[0] = 7000
        series[1, 5] = np.nan
        series[2, 0] = np.inf
        for mode in ("legacy", "edge", "linear"):
            for step in (1, 3):
                config = PredictConfig(padding=mode, step=step, batch_size=13, start_year=1987)
                with self.subTest(padding=mode, step=step):
                    np.testing.assert_array_equal(
                        predict_series(DropClassifier(), series, config),
                        scalar_predict(series, config),
                    )

    def test_batch_size_invariance_and_bounded_model_calls(self):
        rng = np.random.default_rng(799)
        series = rng.uniform(0, 9000, (11, 12)).astype(np.float32)
        series[4, 3] = np.nan
        config = PredictConfig(step=2)
        expected = scalar_predict(series, config)
        total_windows = 10 * 6
        candidate_count = 0
        for raw in series:
            if not np.isfinite(raw).all():
                continue
            padded = original_padding((raw / config.scale_factor)[None], config.window_size)[0]
            for start in range(0, len(raw), config.step):
                local = scalar_rule(padded[start : start + config.window_size], config)
                location = start + local - config.window_size // 2
                candidate_count += int(local >= 0 and 0 <= location < len(raw))
        self.assertGreater(candidate_count, 0)
        self.assertLess(candidate_count, total_windows)
        for batch_size in (1, 7, 4096):
            model = DropClassifier()
            actual = predict_series(model, series, replace(config, batch_size=batch_size))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(sum(model.batch_sizes), candidate_count)
            self.assertTrue(all(size <= batch_size for size in model.batch_sizes))
            self.assertLessEqual(
                len(model.batch_sizes), (total_windows + batch_size - 1) // batch_size
            )

    def test_legacy_padding_matches_original_repeated_insert(self):
        rng = np.random.default_rng(8)
        for years in (1, 3, 12):
            for width in (5, 9, 11):
                data = rng.uniform(-1, 1, (4, years)).astype(np.float32)
                np.testing.assert_allclose(
                    pad_series(data, width, "legacy"),
                    original_padding(data, width),
                    rtol=0,
                    atol=2e-7,
                )

    def test_padding_modes_have_documented_boundary_values(self):
        data = np.array([[0.7, 0.2, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(
            pad_series(data, 5, "legacy"),
            [[0.698, 0.699, 0.7, 0.2, 0.401, 0.401, 0.4]],
        )
        np.testing.assert_allclose(
            pad_series(data, 5, "edge"),
            [[0.7, 0.7, 0.7, 0.2, 0.4, 0.4, 0.4]],
        )
        np.testing.assert_allclose(
            pad_series(data, 5, "linear"),
            [[0.698, 0.699, 0.7, 0.2, 0.4, 0.401, 0.402]],
        )

    def test_first_year_is_preserved_and_events_outside_original_extent_are_rejected(self):
        def fixed_position(position):
            return lambda windows, **kwargs: np.full(len(windows), position, dtype=np.int64)

        config = PredictConfig(window_size=9, start_year=1990, step=20)
        for years, local, expected in ((9, 4, 1990), (3, 2, 0), (1, 6, 0)):
            with self.subTest(years=years, local=local):
                with patch("forest_disturbance.inference.detect_change", side_effect=fixed_position(local)):
                    np.testing.assert_array_equal(
                        predict_series(DropClassifier(always_positive=True), np.ones((1, years)), config),
                        [expected],
                    )

    def test_stable_pixels_and_incomplete_pixels_have_distinct_codes(self):
        series = np.array([[7000] * 9, [np.nan] + [7000] * 8, [np.inf] * 9])
        np.testing.assert_array_equal(
            predict_series(DropClassifier(), series, PredictConfig()), [0, NODATA, NODATA]
        )

    def test_no_model_call_for_empty_or_entirely_invalid_input(self):
        model = DropClassifier()
        np.testing.assert_array_equal(
            predict_series(model, np.full((3, 9), np.nan), PredictConfig()), [NODATA] * 3
        )
        empty = predict_series(model, np.empty((0, 9)), PredictConfig())
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, np.dtype(np.int16))
        self.assertEqual(model.batch_sizes, [])

    def test_stable_windows_skip_model_execution(self):
        # Even a classifier that would declare every window disturbed is never
        # called when the exact final date rule cannot accept any candidate.
        model = DropClassifier(always_positive=True)
        result = predict_series(
            model, np.full((8, 15), 7000, dtype=np.float32),
            PredictConfig(batch_size=7),
        )
        np.testing.assert_array_equal(result, np.zeros(8, dtype=np.int16))
        self.assertEqual(model.batch_sizes, [])

    def test_input_checkpoint_and_model_output_validation(self):
        config = PredictConfig()
        for series in (np.ones(9), np.empty((2, 0)), np.ones((2, 3, 9))):
            with self.assertRaises(ValueError):
                predict_series(DropClassifier(), series, config)
        with self.assertRaises(ValueError):
            predict_series(DropClassifier(), np.ones((2, 2)), replace(config, start_year=32767))
        model = DropClassifier()
        model.config = SimpleNamespace(window_size=11)
        with self.assertRaises(ValueError):
            predict_series(model, np.ones((2, 9)), config)
        # Both pixels pass the date rule in the only temporal window, ensuring
        # the prefilter does not skip model-output validation.
        disturbed = np.array([[7000, 7000] + [1000] * 7] * 2, dtype=np.float32)
        for output, error in (
            (torch.zeros(2, 3), "class scores"),
            (torch.full((2, 2), float("nan")), "non-finite"),
        ):
            model = unittest.mock.Mock(wraps=DropClassifier())
            model.config = None
            model.to.return_value = model
            model.eval.return_value = model
            model.return_value = output
            with self.assertRaisesRegex(ValueError, error):
                predict_series(model, disturbed, replace(config, batch_size=2, step=20))

    def test_config_rejects_invalid_values(self):
        for kwargs in (
            {"window_size": 4}, {"window_size": 8}, {"step": 0}, {"batch_size": 0},
            {"block_size": -1}, {"scale_factor": 0}, {"scale_factor": np.nan},
            {"min_slope": np.inf}, {"min_intercept": np.nan}, {"min_drop": -np.inf},
            {"padding": "unknown"}, {"start_year": 0}, {"start_year": 32768},
            {"batch_size": 1.5}, {"step": True}, {"start_year": 2000.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PredictConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
