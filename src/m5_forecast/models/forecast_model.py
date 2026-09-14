import mlflow.pyfunc
import pandas as pd

from .reconcile import reconcile_bottom_up


class ForecastLookupModel(mlflow.pyfunc.PythonModel):
    """Serves pre-computed item_store forecasts (produced by a final fit over
    the full history at training time) and reconciles bottom-up to store/state/
    total on demand, rather than re-running the model per request."""

    def __init__(self, item_store_forecast: pd.DataFrame, id_map: pd.DataFrame):
        self.item_store_forecast = item_store_forecast  # unique_id, date, value
        self.id_map = id_map  # unique_id, store_id, state_id

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        row = model_input.iloc[0]
        level, key = row["level"], row["key"]

        if level == "item_store":
            series = self.item_store_forecast[self.item_store_forecast["unique_id"] == key]
        elif level == "store":
            reconciled = reconcile_bottom_up(self.item_store_forecast, self.id_map, ["store_id"])
            series = reconciled[reconciled["unique_id"] == key]
        elif level == "state":
            reconciled = reconcile_bottom_up(self.item_store_forecast, self.id_map, ["state_id"])
            series = reconciled[reconciled["unique_id"] == key]
        elif level == "total":
            series = reconcile_bottom_up(self.item_store_forecast, self.id_map, [])
        else:
            raise ValueError(f"unknown level: {level!r}")

        return series[["date", "value"]].reset_index(drop=True)
