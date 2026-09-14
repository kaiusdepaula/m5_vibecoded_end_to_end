import pandas as pd
import pytest

from m5_forecast.evaluation.backtest import (
    BacktestWindow,
    LeakageError,
    generate_windows,
    leakage_check,
)


def test_generates_non_overlapping_rolling_windows_that_pass_leakage_check():
    dates = pd.date_range("2016-01-01", periods=120, freq="D")

    windows = generate_windows(dates, horizon=28, window_count=3)

    assert len(windows) == 3
    leakage_check(windows)  # must not raise


def test_deliberately_overlapping_window_is_flagged_by_leakage_check():
    overlapping = BacktestWindow(
        train_end=pd.Timestamp("2016-02-01"),
        test_start=pd.Timestamp("2016-01-25"),  # before train_end -- deliberate overlap
        test_end=pd.Timestamp("2016-02-22"),
    )

    with pytest.raises(LeakageError):
        leakage_check([overlapping])


def test_requesting_more_windows_than_history_supports_fails_loudly():
    dates = pd.date_range("2016-01-01", periods=40, freq="D")

    with pytest.raises(ValueError):
        generate_windows(dates, horizon=28, window_count=3)
