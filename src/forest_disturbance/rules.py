"""Vectorized disturbance-date rules from the original PETransformer pipeline."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def detect_change(
    windows: ArrayLike,
    min_slope: float = -0.05,
    min_intercept: float = 0.2,
    min_drop: float = -0.1,
) -> NDArray[np.int64]:
    """Return one local, zero-based disturbance position for every window.

    ``windows`` has shape ``(samples, window_size)`` and must contain at least
    five observations per window. Candidate positions are ``2..window_size-3``.
    For candidate ``p``, the immediate drop is ``x[p] - x[p-1]``. A line is fit
    through ``x[p-2], x[p-1], x[p+1], x[p+2]`` at times ``0, 1, 3, 4``. The
    candidate passes when the drop and fitted slope are strictly below their
    thresholds and the fitted intercept is strictly above its threshold.

    The most negative *passing* drop wins; ties select the earlier position.
    No passing candidate, or any nonfinite observation in a window, produces
    ``-1``. The four-point least-squares fit uses an exact closed-form formula
    evaluated in float64, without per-sample loops, sorting, or ``np.polyfit``.

    Finite-input behavior follows ``classification/Change_trendv2.py`` from
    the original code. Values exactly on a threshold can differ from the old
    numerical solver's rounding; comparisons here remain strictly ``<``/``>``.
    """
    values = np.asarray(windows)
    if values.ndim != 2:
        raise ValueError("windows must have shape (samples, window_size)")
    if values.shape[1] < 5:
        raise ValueError("window_size must be at least 5")
    if values.dtype.kind not in "biuf":
        raise TypeError("windows must contain real numeric values")
    for name, threshold in (
        ("min_slope", min_slope),
        ("min_intercept", min_intercept),
        ("min_drop", min_drop),
    ):
        if not isinstance(threshold, Real) or not np.isfinite(threshold):
            raise ValueError(f"{name} must be a finite real number")

    values = values.astype(np.float64, copy=False)
    finite_windows = np.isfinite(values).all(axis=1)

    # These slices align all W-4 candidates, p = 2, ..., W-3.
    before_two = values[:, :-4]
    before_one = values[:, 1:-3]
    after_one = values[:, 3:-1]
    after_two = values[:, 4:]
    with np.errstate(invalid="ignore", over="ignore"):
        drops = values[:, 2:-2] - before_one
        # mean(t)=2 and sum((t-2)**2)=10 for t=(0, 1, 3, 4).
        slopes = (
            -2.0 * before_two - before_one + after_one + 2.0 * after_two
        ) / 10.0
        intercepts = (
            before_two + before_one + after_one + after_two
        ) / 4.0 - 2.0 * slopes

    passing = (
        finite_windows[:, None]
        & np.isfinite(drops)
        & np.isfinite(slopes)
        & np.isfinite(intercepts)
        & (drops < min_drop)
        & (slopes < min_slope)
        & (intercepts > min_intercept)
    )
    candidates = np.where(passing, drops, np.inf)
    # argmin is stable for equal minima, preserving the original sorted order.
    positions = np.argmin(candidates, axis=1).astype(np.int64) + 2
    return np.where(passing.any(axis=1), positions, -1)
