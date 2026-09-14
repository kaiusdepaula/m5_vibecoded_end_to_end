import math

import pandas as pd
import pytest

from m5_forecast.evaluation.backtest import BacktestWindow
from m5_forecast.evaluation.scoring import _rmsse_by_series
from m5_forecast.evaluation.wrmsse import rmsse


def test_vectorized_rmsse_matches_scalar_rmsse_per_series():
    # Series A: same numbers as the hand-verified fixture in test_wrmsse.py.
    # Series B: a constant history -> the zero-variance NaN guard.
    window = BacktestWindow(
        train_end=pd.Timestamp("2016-01-05"),
        test_start=pd.Timestamp("2016-01-06"),
        test_end=pd.Timestamp("2016-01-07"),
    )
    train_dates = pd.date_range("2016-01-01", periods=5)
    test_dates = pd.date_range("2016-01-06", periods=2)

    actual = pd.DataFrame(
        {
            "unique_id": (["A"] * 5) + (["A"] * 2) + (["B"] * 5) + (["B"] * 2),
            "date": list(train_dates) + list(test_dates) + list(train_dates) + list(test_dates),
            "sales": [10, 12, 8, 14, 6, 10, 20] + [7, 7, 7, 7, 7, 7, 7],
        }
    )
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B"],
            "date": list(test_dates) * 2,
            "value": [15, 15, 8, 8],
        }
    )

    result = _rmsse_by_series(actual, forecast, window)

    expected_a = rmsse(actual=[10, 20], forecast=[15, 15], train_history=[10, 12, 8, 14, 6])
    assert result["A"] == pytest.approx(expected_a, abs=1e-9)
    assert math.isnan(result["B"])
