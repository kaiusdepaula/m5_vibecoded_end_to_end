import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from m5_forecast.api.main import app
from m5_forecast.api.schemas import ModelRef
from m5_forecast.config import Settings


class FakeModel:
    """Mimics ForecastLookupModel.predict without needing a real MLflow model."""

    KNOWN_SERIES = {"ITEM_1_CA_1"}

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        row = model_input.iloc[0]
        if row["level"] == "item_store" and row["key"] not in self.KNOWN_SERIES:
            return pd.DataFrame(columns=["date", "value"])
        dates = pd.date_range("2016-04-25", periods=3, freq="D")
        return pd.DataFrame({"date": dates, "value": [1.4, 1.1, 1.0]})


class FakeRun:
    class _Info:
        run_id = "fake-run-id"

    class _Data:
        tags = {"backtest_windows": "[]", "feature_list": "[]"}
        metrics = {"wrmsse_item_store": 0.9, "wrmsse_total": 0.8}

    info = _Info()
    data = _Data()


@pytest.fixture(autouse=True)
def stub_app_state(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    app.state.settings = Settings(horizon=28)
    app.state.model = FakeModel()
    app.state.model_ref = ModelRef(name="m5_hierarchical", version=3)
    app.state.run = FakeRun()
    app.state.examples = {"item_store": "ITEM_1_CA_1", "store": "CA_1", "state": "CA"}
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_forecast_by_series_returns_28_style_dated_values_and_model_ref(client):
    response = client.post("/forecast", json={"level": "item_store", "series_id": "ITEM_1_CA_1"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == {"name": "m5_hierarchical", "version": 3}
    assert len(body["forecast"]) == 3
    assert body["forecast"][0] == {"date": "2016-04-25", "value": 1.4}
    assert "request_id" in body


def test_forecast_by_state_level_returns_aggregate_values(client):
    response = client.post("/forecast", json={"level": "state", "key": "CA"})

    assert response.status_code == 200
    assert response.json()["level"] == "state"


def test_get_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_model_info_returns_documented_fields(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "m5_hierarchical"
    assert body["version"] == 3
    assert body["run_id"] == "fake-run-id"
    assert "wrmsse" in body
    assert body["examples"]["item_store"] == "ITEM_1_CA_1"


def test_malformed_payload_returns_400_with_actionable_message_not_5xx(client):
    response = client.post("/forecast", json={"level": "item_store"})  # missing series_id

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_payload"
    assert "series_id" in body["message"]
    assert "request_id" in body


def test_unknown_series_returns_handled_error_and_health_still_ok(client):
    response = client.post("/forecast", json={"level": "item_store", "series_id": "DOES_NOT_EXIST"})

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "unknown_series"
    assert "DOES_NOT_EXIST" in body["message"]

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
