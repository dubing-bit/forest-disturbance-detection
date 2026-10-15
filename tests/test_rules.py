"""Parity and boundary checks for the original four-point disturbance rule."""

from __future__ import annotations

import unittest

import numpy as np

from forest_disturbance.rules import detect_change


def scalar_reference(
    window: np.ndarray,
    min_slope: float = -0.05,
    min_intercept: float = 0.2,
    min_drop: float = -0.1,
) -> int:
    """Independently reproduce the original sort-and-polyfit implementation."""
    if not np.isfinite(window).all():
        return -1
    candidates = [
        (position, window[position] - window[position - 1])
        for position in range(2, len(window) - 2)
    ]
    for position, drop in sorted(candidates, key=lambda candidate: candidate[1]):
        if drop < min_drop:
            observations = [
                window[position - 2],
                window[position - 1],
                window[position + 1],
                window[position + 2],
            ]
            slope, intercept = np.polyfit([0, 1, 3, 4], observations, 1)
            if slope < min_slope and intercept > min_intercept:
                return position
    return -1


class DetectChangeTests(unittest.TestCase):
    def test_matches_scalar_polyfit_reference(self):
        rng = np.random.default_rng(20261015)
        for window_size in (5, 7, 9, 11, 21):
            windows = rng.uniform(-0.3, 1.0, size=(300, window_size))
            for thresholds in ((-0.05, 0.2, -0.1), (-0.08, 0.4, -0.2)):
                expected = np.array(
                    [scalar_reference(window, *thresholds) for window in windows]
                )
                np.testing.assert_array_equal(
                    detect_change(windows, *thresholds), expected
                )

    def test_equal_drops_choose_earliest_passing_position(self):
        windows = [[1, 1, 0.5, 0.5, 0.5, 0, 0, 0, 0]]
        np.testing.assert_array_equal(detect_change(windows), [2])

    def test_rejects_largest_drop_then_tries_next_candidate(self):
        windows = [[1, 1, 0.5, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6]]
        np.testing.assert_array_equal(detect_change(windows), [5])

    def test_drops_near_window_edges_are_excluded(self):
        windows = [[1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]
        np.testing.assert_array_equal(detect_change(windows), [-1, -1])

    def test_strict_threshold_comparisons(self):
        # The four fitted observations lie exactly on y = 0.5 - 0.125t.
        windows = [[0.5, 0.375, 0, 0.125, 0]]
        np.testing.assert_array_equal(detect_change(windows), [2])
        np.testing.assert_array_equal(
            detect_change(windows, min_slope=-0.125), [-1]
        )
        np.testing.assert_array_equal(
            detect_change(windows, min_intercept=0.5), [-1]
        )
        np.testing.assert_array_equal(
            detect_change(windows, min_drop=-0.375), [-1]
        )
        np.testing.assert_array_equal(
            detect_change(windows, min_slope=np.nextafter(-0.125, 0)), [2]
        )
        np.testing.assert_array_equal(
            detect_change(windows, min_intercept=np.nextafter(0.5, 0)), [2]
        )
        np.testing.assert_array_equal(
            detect_change(windows, min_drop=np.nextafter(-0.375, 0)), [2]
        )

    def test_nonfinite_windows_are_invalid(self):
        windows = np.array([[1, 1, 0.5, 0, 0]] * 3, dtype=float)
        windows[0, 0] = np.nan
        windows[1, 2] = np.inf
        windows[2, 4] = -np.inf
        np.testing.assert_array_equal(detect_change(windows), [-1, -1, -1])

    def test_no_event_and_empty_batch(self):
        np.testing.assert_array_equal(detect_change(np.ones((4, 9))), [-1] * 4)
        result = detect_change(np.empty((0, 9)))
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.dtype(np.int64))

    def test_integer_input_is_converted_before_subtracting(self):
        windows = np.array([[100, 100, 40, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(detect_change(windows), [2])

    def test_invalid_inputs(self):
        for windows in (np.ones(9), np.ones((2, 3, 4)), np.ones((2, 4))):
            with self.assertRaises(ValueError):
                detect_change(windows)
        for windows in (np.ones((2, 9), dtype=complex), [["1"] * 9]):
            with self.assertRaises(TypeError):
                detect_change(windows)
        for name in ("min_slope", "min_intercept", "min_drop"):
            for value in (np.nan, np.inf, -np.inf, "0"):
                with self.assertRaises(ValueError):
                    detect_change(np.ones((2, 9)), **{name: value})


if __name__ == "__main__":
    unittest.main()
