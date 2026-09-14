import hashlib
import json
import logging
import resource
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from .config import Settings, load_settings
from .data.build_features import aggregate_hierarchy, build_modelling_table
from .data.load import load_calendar, load_sales, load_sell_prices
from .evaluation.backtest import BacktestWindow, generate_windows, leakage_check
from .logging import configure_logging
from .models.baseline import make_seasonal_naive_forecaster, train_and_log_baseline
from .models.candidate import make_lightgbm_forecaster, train_and_log_candidate
from .models.forecast_model import ForecastLookupModel
from .models.reconcile import build_id_map

logger = logging.getLogger("m5_forecast.train")

COMPARISON_LEVEL = "total"


def _log_stage(stage: str) -> None:
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    logger.info(f"stage={stage} peak_rss_mb={peak_mb:.1f}")


def compute_data_hash(*frames: pd.DataFrame) -> str:
    # Deliberately not pandas.util.hash_pandas_object: its default categorize=True
    # path has been observed to raise IndexError on categorical columns (codes
    # referencing more categories than exist) -- modelling_table is categorical
    # in several columns by design, so this bypasses that path entirely.
    hasher = hashlib.sha256()
    for frame in frames:
        for column_name in frame.columns:
            column = frame[column_name]
            hasher.update(str(column_name).encode("utf-8"))
            if isinstance(column.dtype, pd.CategoricalDtype):
                hasher.update(column.cat.codes.to_numpy().tobytes())
                hasher.update("|".join(map(str, column.cat.categories)).encode("utf-8"))
            elif pd.api.types.is_datetime64_any_dtype(column):
                hasher.update(column.to_numpy().astype("int64").tobytes())
            elif pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
                hasher.update(np.ascontiguousarray(column.to_numpy()).tobytes())
            else:
                # Any other (string-like/object) dtype: numpy's .tobytes() on
                # an object array serializes Python-object pointers, not their
                # content, and is non-deterministic across calls -- hash the
                # string values themselves instead.
                hasher.update("\x1f".join(column.fillna("\x00").astype(str)).encode("utf-8"))
    return hasher.hexdigest()


def final_fit_forecast(forecast_fn, item_store_actual: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Refits on the full history to produce the forecast that actually gets
    served, distinct from the backtest windows used to score/select the model."""
    last_date = item_store_actual["date"].max()
    final_window = BacktestWindow(
        train_end=last_date,
        test_start=last_date + pd.Timedelta(days=1),
        test_end=last_date + pd.Timedelta(days=horizon),
    )
    forecast = forecast_fn(item_store_actual, final_window)
    return forecast.rename(columns={"yhat": "value"})


def run_training(modelling_table: pd.DataFrame, settings: Settings) -> dict:
    _log_stage("run_training:start")
    id_map = build_id_map(modelling_table)
    windows = generate_windows(
        modelling_table["date"], horizon=settings.horizon, window_count=settings.backtest.window_count
    )
    leakage_check(windows)

    data_hash = compute_data_hash(modelling_table)
    _log_stage("data_hash_computed")

    # Computed once and reused for both models' scoring and the final fit --
    # item_store is nearly as large as modelling_table itself, so recomputing
    # it per model would needlessly multiply peak memory.
    hierarchy = aggregate_hierarchy(modelling_table)
    del modelling_table
    _log_stage("hierarchy_aggregated")

    baseline_run_id, baseline_scores = train_and_log_baseline(
        hierarchy, id_map, windows, data_hash, settings.hierarchy_weight_lookback_days
    )
    _log_stage("baseline_trained")
    candidate_run_id, candidate_scores = train_and_log_candidate(
        hierarchy, id_map, windows, data_hash, settings.hierarchy_weight_lookback_days
    )
    _log_stage("candidate_trained")

    if baseline_scores[COMPARISON_LEVEL] <= candidate_scores[COMPARISON_LEVEL]:
        winner_type, winner_run_id, forecast_fn = "baseline", baseline_run_id, make_seasonal_naive_forecaster()
    else:
        winner_type, winner_run_id, forecast_fn = "candidate", candidate_run_id, make_lightgbm_forecaster()

    final_forecast = final_fit_forecast(forecast_fn, hierarchy["item_store"], settings.horizon)
    _log_stage("final_fit_done")

    with mlflow.start_run(run_id=winner_run_id):
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ForecastLookupModel(final_forecast, id_map),
            registered_model_name=settings.registered_model_name,
        )
    _log_stage("model_registered")

    client = MlflowClient()
    client.set_registered_model_alias(
        settings.registered_model_name, settings.model_alias, model_info.registered_model_version
    )

    return {
        "winner": winner_type,
        "winner_run_id": winner_run_id,
        "version": model_info.registered_model_version,
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
    }


def main() -> None:
    configure_logging()
    settings = load_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("m5_hierarchical")

    data_dir = Path("/app/data")
    calendar = load_calendar(data_dir / "calendar.csv")
    _log_stage("calendar_loaded")
    sales = load_sales(
        data_dir / "sales_train_validation.csv",
        sample_series=settings.sample_series,
        random_state=settings.random_seed,
    )
    _log_stage("sales_loaded")
    sell_prices = load_sell_prices(data_dir / "sell_prices.csv")
    _log_stage("sell_prices_loaded")

    if settings.sample_series is not None:
        # No need to carry price rows for series we didn't sample into sales.
        sell_prices = sell_prices[
            sell_prices["item_id"].isin(sales["item_id"].unique())
            & sell_prices["store_id"].isin(sales["store_id"].unique())
        ]
        _log_stage("sell_prices_filtered_to_sample")

    modelling_table = build_modelling_table(sales, calendar, sell_prices)
    del sales, calendar, sell_prices
    _log_stage("modelling_table_built")

    result = run_training(modelling_table, settings)
    summary = {k: v for k, v in result.items() if not k.endswith("_scores")}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
