import mlflow

from m5_forecast.data.build_features import aggregate_hierarchy, build_modelling_table
from m5_forecast.data.load import melt_sales
from m5_forecast.data.schema import CALENDAR_SCHEMA, SALES_LONG_SCHEMA, validate
from m5_forecast.evaluation.backtest import generate_windows
from m5_forecast.models.baseline import train_and_log_baseline
from m5_forecast.models.reconcile import build_id_map

from synthetic_data import make_synthetic_m5_extract

EXPECTED_LEVELS = {"item_store", "store", "state", "total", "overall"}


def test_train_and_log_baseline_creates_one_run_with_expected_metrics_and_tags(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-baseline")

    sales_wide, calendar_raw, sell_prices = make_synthetic_m5_extract(num_days=40)
    sales = validate(melt_sales(sales_wide), SALES_LONG_SCHEMA, "sales")
    calendar = validate(calendar_raw, CALENDAR_SCHEMA, "calendar")
    modelling_table = build_modelling_table(sales, calendar, sell_prices)
    id_map = build_id_map(modelling_table)
    windows = generate_windows(modelling_table["date"], horizon=7, window_count=1)
    hierarchy = aggregate_hierarchy(modelling_table)

    run_id, level_scores = train_and_log_baseline(
        hierarchy, id_map, windows, data_hash="testhash", weight_lookback_days=28
    )

    assert set(level_scores) == EXPECTED_LEVELS

    runs = mlflow.search_runs(experiment_names=["test-baseline"])
    assert len(runs) == 1

    run = mlflow.get_run(run_id)
    assert run.data.tags["model_type"] == "baseline"
    assert run.data.tags["data_hash"] == "testhash"
    assert "backtest_windows" in run.data.tags
    for level in EXPECTED_LEVELS:
        assert f"wrmsse_{level}" in run.data.metrics
