from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestWindow:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class LeakageError(Exception):
    """Raised when a backtest window would let training see data from its own test period."""


def generate_windows(dates, horizon: int, window_count: int) -> list[BacktestWindow]:
    if window_count < 1:
        raise ValueError("window_count must be >= 1")

    unique_dates = pd.Series(sorted(pd.to_datetime(pd.Series(dates)).unique()))
    required = horizon * window_count
    if len(unique_dates) <= required:
        raise ValueError(
            f"not enough history for {window_count} window(s) of horizon {horizon}: "
            f"have {len(unique_dates)} day(s), need more than {required}"
        )

    last_date = unique_dates.iloc[-1]
    windows = []
    for i in range(window_count):
        test_end = last_date - pd.Timedelta(days=horizon * i)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        windows.append(BacktestWindow(train_end=train_end, test_start=test_start, test_end=test_end))
    return list(reversed(windows))


def leakage_check(windows: list[BacktestWindow]) -> None:
    for window in windows:
        if window.train_end >= window.test_start:
            raise LeakageError(
                f"window leaks future data into training: train_end={window.train_end} "
                f">= test_start={window.test_start}"
            )
