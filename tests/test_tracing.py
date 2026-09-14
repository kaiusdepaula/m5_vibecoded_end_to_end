import json

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from m5_forecast.api.main import app
from m5_forecast.api.schemas import ModelRef
from m5_forecast.config import Settings
from m5_forecast.logging import configure_logging


class FakeModel:
    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        row = model_input.iloc[0]
        if row["level"] == "item_store" and row["key"] != "ITEM_1_CA_1":
            return pd.DataFrame(columns=["date", "value"])
        return pd.DataFrame({"date": pd.date_range("2016-04-25", periods=2, freq="D"), "value": [1.4, 1.1]})


class FakeRun:
    class _Info:
        run_id = "fake-run-id"

    class _Data:
        tags = {"backtest_windows": "[]", "feature_list": "[]"}
        metrics = {}

    info = _Info()
    data = _Data()


@pytest.fixture
def client(tmp_path):
    experiment_name = "test-tracing"
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.create_experiment(experiment_name, artifact_location=f"file://{tmp_path}/artifacts")
    mlflow.set_experiment(experiment_name)

    app.state.settings = Settings(horizon=28)
    app.state.model = FakeModel()
    app.state.model_ref = ModelRef(name="m5_hierarchical", version=3)
    app.state.run = FakeRun()
    app.state.examples = {"item_store": "ITEM_1_CA_1", "store": "CA_1", "state": "CA"}

    return TestClient(app)


def test_forecast_response_request_id_matches_trace_tag(client):
    response = client.post("/forecast", json={"level": "item_store", "series_id": "ITEM_1_CA_1"})
    assert response.status_code == 200
    request_id = response.json()["request_id"]

    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    matching = [t for t in traces if t.info.tags.get("request_id") == request_id]

    assert len(matching) == 1
    trace = matching[0]
    assert trace.info.tags["endpoint"] == "/forecast"
    assert trace.info.tags["level"] == "item_store"
    assert trace.info.tags["status"] == "ok"
    assert "duration_ms" in trace.info.tags


def test_rejected_payload_is_traced_with_reason(client):
    response = client.post("/forecast", json={"level": "item_store", "series_id": "DOES_NOT_EXIST"})
    assert response.status_code == 404
    request_id = response.json()["request_id"]

    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    matching = [t for t in traces if t.info.tags.get("request_id") == request_id]

    assert len(matching) == 1
    assert matching[0].info.tags["status"] == "rejected"
    assert matching[0].info.tags["reason"] == "unknown_series"


def test_response_request_id_matches_json_log_line(client, capsys):
    configure_logging()

    response = client.post("/forecast", json={"level": "item_store", "series_id": "ITEM_1_CA_1"})
    request_id = response.json()["request_id"]

    log_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("{")]
    matching = [json.loads(line) for line in log_lines if json.loads(line).get("request_id") == request_id]

    assert len(matching) >= 1


def test_malformed_payload_is_traced_with_validation_reason(client):
    response = client.post("/forecast", json={"level": "item_store"})
    assert response.status_code == 400
    request_id = response.json()["request_id"]

    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(return_type="list")
    matching = [t for t in traces if t.info.tags.get("request_id") == request_id]

    assert len(matching) == 1
    assert matching[0].info.tags["status"] == "rejected"
