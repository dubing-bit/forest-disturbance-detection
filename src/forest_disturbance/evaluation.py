"""Year-aware validation with explicit treatment of incorrectly dated events."""

import csv
from pathlib import Path

import numpy as np


def evaluate_years(reference, prediction, tolerance: int = 0) -> dict:
    """0 means stable forest; positive integers are calendar years.

    A disturbed pixel dated outside the tolerance counts as both an omission
    and a commission. Negative/invalid values must be removed explicitly by
    the caller instead of silently changing the evaluated population.
    """
    ref, pred = np.asarray(reference), np.asarray(prediction)
    if ref.ndim != 1 or ref.shape != pred.shape or ref.size == 0:
        raise ValueError("reference and prediction must be nonempty equal-length vectors")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative integer")
    for array in (ref, pred):
        if not np.isfinite(array).all() or (array < 0).any() or (array != np.floor(array)).any():
            raise ValueError("year labels must be nonnegative finite integers")
    disturbed_ref, disturbed_pred = ref > 0, pred > 0
    correct_event = disturbed_ref & disturbed_pred & (np.abs(ref - pred) <= tolerance)
    correct_stable = ~disturbed_ref & ~disturbed_pred
    tp = int(correct_event.sum())
    tn = int(correct_stable.sum())
    fn = int((disturbed_ref & ~correct_event).sum())
    fp = int((disturbed_pred & ~correct_event).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "samples": int(ref.size), "year_tolerance": tolerance,
        "correct_disturbance": tp, "correct_stable": tn,
        "omissions": fn, "commissions": fp,
        "year_accuracy": (tp + tn) / ref.size,
        "binary_accuracy": float((disturbed_ref == disturbed_pred).mean()),
        "disturbance_precision": ratio(tp, tp + fp),
        "disturbance_recall": ratio(tp, tp + fn),
        "disturbance_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "omission_rate": ratio(fn, tp + fn),
        "commission_rate": ratio(fp, tp + fp),
    }


def evaluate_csv(path: str | Path, *, reference_column="reference", prediction_column="prediction",
                 tolerance=0) -> dict:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    try:
        reference = np.array([float(row[reference_column]) for row in rows])
        prediction = np.array([float(row[prediction_column]) for row in rows])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("evaluation CSV must contain numeric reference and prediction columns") from exc
    return evaluate_years(reference, prediction, tolerance)
