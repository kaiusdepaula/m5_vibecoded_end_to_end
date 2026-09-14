import mlflow
import pytest
from mlflow import MlflowClient

import m5_forecast.train as train_module
from m5_forecast.config import load_settings
from m5_forecast.data.build_features import build_modelling_table
from m5_forecast.data.load import melt_sales
from m5_forecast.data.schema import CALENDAR_SCHEMA, SALES_LONG_SCHEMA, validate

from synthetic_data import make_synthetic_m5_extract


def _fake_trainer(model_type: str, total_score: float):
    def trainer(modelling_table, id_map, windows, data_hash, weight_lookback_days):
        with mlflow.start_run() as run:
            mlflow.set_tag("model_type", model_type)
        scores = {"item_store": total_score, "store": total_score, "state": total_score, "total": total_score}
        scores["overall"] = total_score
        return run.info.run_id, scores

    return trainer


@pytest.mark.parametrize(
    "baseline_score,candidate_score,expected_winner",
    [(0.5, 1.0, "baseline"), (1.0, 0.5, "candidate")],
)
def test_selection_promotes_lower_wrmsse_model_in_either_direction(
    tmp_path, monkeypatch, baseline_score, candidate_score, expected_winner
):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment_name = "test-selection"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=f"file://{tmp_path}/artifacts")
    mlflow.set_experiment(experiment_name)

    monkeypatch.setattr(train_module, "train_and_log_baseline", _fake_trainer("baseline", baseline_score))
    monkeypatch.setattr(train_module, "train_and_log_candidate", _fake_trainer("candidate", candidate_score))

    settings = load_settings()
    settings.registered_model_name = f"m5_test_{expected_winner}_{str(baseline_score).replace('.', '')}"
    settings.horizon = 7
    settings.backtest.window_count = 1

    sales_wide, calendar_raw, sell_prices = make_synthetic_m5_extract(num_days=100)
    sales = validate(melt_sales(sales_wide), SALES_LONG_SCHEMA, "sales")
    calendar = validate(calendar_raw, CALENDAR_SCHEMA, "calendar")
    modelling_table = build_modelling_table(sales, calendar, sell_prices)

    result = train_module.run_training(modelling_table, settings)

    assert result["winner"] == expected_winner

    client = MlflowClient()
    champion_version = client.get_model_version_by_alias(settings.registered_model_name, settings.model_alias)
    assert int(champion_version.version) == result["version"]
