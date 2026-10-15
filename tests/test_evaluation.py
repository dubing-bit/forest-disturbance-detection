import numpy as np
import pytest

from forest_disturbance.evaluation import evaluate_years


def test_misdated_event_is_both_omission_and_commission():
    result = evaluate_years([0, 2001, 2005, 0, 2010], [0, 2001, 2007, 2011, 0])
    assert result["omissions"] == 2
    assert result["commissions"] == 2
    assert result["year_accuracy"] == 0.4
    assert result["binary_accuracy"] == 0.6
    assert result["disturbance_precision"] == pytest.approx(1 / 3)


def test_year_tolerance_and_undefined_rates():
    assert evaluate_years([2005], [2006], 1)["year_accuracy"] == 1
    assert evaluate_years([0], [0])["disturbance_precision"] is None
    with pytest.raises(ValueError):
        evaluate_years([np.nan], [0])
    with pytest.raises(ValueError):
        evaluate_years([-32768], [0])
