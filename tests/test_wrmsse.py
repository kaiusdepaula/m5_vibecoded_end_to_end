import math

import pytest

from m5_forecast.evaluation.wrmsse import level_wrmsse, rmsse, wrmsse


def test_single_series_rmsse_matches_hand_computed_fixture():
    # train diffs: 2, -4, 6, -8 -> squared: 4, 16, 36, 64 -> mean = 30 (scale)
    # forecast error: (10-15)^2=25, (20-15)^2=25 -> mean = 25
    # rmsse = sqrt(25 / 30) = 0.912870...
    result = rmsse(actual=[10, 20], forecast=[15, 15], train_history=[10, 12, 8, 14, 6])
    assert result == pytest.approx(0.9129, abs=1e-4)


def test_level_wrmsse_weights_series_by_dollar_share():
    # hand-computed: 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    result = level_wrmsse(
        rmsse_by_series={"A": 1.0, "B": 0.5},
        dollar_weight_by_series={"A": 0.6, "B": 0.4},
    )
    assert result == pytest.approx(0.8, abs=1e-9)


def test_zero_variance_history_returns_nan_and_is_excluded_from_level_aggregation():
    constant_history_score = rmsse(actual=[7, 7], forecast=[8, 8], train_history=[7, 7, 7, 7])
    assert math.isnan(constant_history_score)

    # B's NaN score must be excluded and A's weight renormalized to 1.0, not
    # silently propagate NaN through the whole level.
    result = level_wrmsse(
        rmsse_by_series={"A": 1.0, "B": constant_history_score},
        dollar_weight_by_series={"A": 0.6, "B": 0.4},
    )
    assert result == pytest.approx(1.0, abs=1e-9)


def test_mismatched_actual_forecast_lengths_raises():
    with pytest.raises(ValueError):
        rmsse(actual=[1, 2, 3], forecast=[1, 2], train_history=[1, 2, 3, 4])


def test_wrmsse_averages_level_scores_with_equal_weight():
    result = wrmsse({"item": 1.0, "total": 3.0})
    assert result == pytest.approx(2.0, abs=1e-9)
