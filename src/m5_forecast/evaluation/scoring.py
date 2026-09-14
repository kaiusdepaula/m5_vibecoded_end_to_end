from typing import Callable

import numpy as np
import pandas as pd

from ..data.build_features import HIERARCHY_LEVELS, dollar_weights
from ..models.reconcile import reconcile_bottom_up
from .backtest import BacktestWindow
from .wrmsse import level_wrmsse, wrmsse

ForecastFn = Callable[[pd.DataFrame, BacktestWindow], pd.DataFrame]


def _rmsse_by_series(actual_df: pd.DataFrame, forecast_df: pd.DataFrame, window: BacktestWindow) -> dict[str, float]:
    """Vectorized per-series RMSSE for one window: one groupby-diff pass for the
    naive-forecast scale and one merge+groupby pass for the forecast error,
    instead of a Python loop and an O(series) forecast lookup per series."""
    train = actual_df[actual_df["date"] <= window.train_end].sort_values(["unique_id", "date"])
    naive_diffs = train.groupby("unique_id")["sales"].diff()
    denominator = (naive_diffs**2).groupby(train["unique_id"]).mean()

    test = actual_df[(actual_df["date"] >= window.test_start) & (actual_df["date"] <= window.test_end)]
    merged = test.merge(forecast_df, on=["unique_id", "date"], how="left")
    squared_error = (merged["sales"] - merged["value"]) ** 2
    numerator = squared_error.groupby(merged["unique_id"]).mean()

    combined = numerator.to_frame("numerator").join(denominator.to_frame("denominator"), how="outer")
    result = np.sqrt(combined["numerator"] / combined["denominator"])
    result = result.where(combined["denominator"].fillna(0) != 0, np.nan)
    return result.to_dict()


def score_model(
    hierarchy: dict[str, pd.DataFrame],
    id_map: pd.DataFrame,
    item_store_forecast_fn: ForecastFn,
    windows: list[BacktestWindow],
    weight_lookback_days: int,
) -> dict[str, float]:
    """Scores a model that forecasts only at item_store level, reconciling
    bottom-up to store/state/total (FR-11) before computing per-level WRMSSE
    against each window, then averaging across windows. Takes the already
    computed hierarchy (see aggregate_hierarchy) rather than the raw modelling
    table, since the item_store level is nearly as large as the full table and
    recomputing it for every model scored would multiply peak memory use."""
    scores_by_level: dict[str, list[float]] = {level: [] for level in HIERARCHY_LEVELS}

    for window in windows:
        item_store_actual = hierarchy["item_store"]
        train = item_store_actual[item_store_actual["date"] <= window.train_end]
        forecast = item_store_forecast_fn(train, window).rename(columns={"yhat": "value"})

        forecast_by_level = {
            "item_store": forecast,
            "store": reconcile_bottom_up(forecast, id_map, ["store_id"]),
            "state": reconcile_bottom_up(forecast, id_map, ["state_id"]),
            "total": reconcile_bottom_up(forecast, id_map, []),
        }

        for level in HIERARCHY_LEVELS:
            actual_level_df = hierarchy[level]
            weights = dollar_weights(actual_level_df, window.train_end, weight_lookback_days)
            rmsse_by_series = _rmsse_by_series(actual_level_df, forecast_by_level[level], window)
            scores_by_level[level].append(level_wrmsse(rmsse_by_series, weights))

    level_scores = {level: float(pd.Series(scores).mean()) for level, scores in scores_by_level.items()}
    # "overall" is the equal-weighted mean across the 4 levels, distinct from
    # "total", which is the WRMSSE of the single aggregate TOTAL series.
    level_scores["overall"] = wrmsse(level_scores)
    return level_scores
