import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from ..evaluation.backtest import BacktestWindow
from ..evaluation.scoring import score_model
from ..mlflow_utils import log_training_run

LAGS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)


def feature_list(lags=LAGS, rolling_windows=ROLLING_WINDOWS) -> list[str]:
    return [f"lag_{lag}" for lag in lags] + [f"rolling_mean_{w}" for w in rolling_windows] + ["dayofweek", "month"]


def make_lightgbm_forecaster(lags=LAGS, rolling_windows=ROLLING_WINDOWS):
    def forecast_fn(train_df: pd.DataFrame, window: BacktestWindow) -> pd.DataFrame:
        horizon = (window.test_end - window.test_start).days + 1
        sf_input = train_df.rename(columns={"date": "ds", "sales": "y"})[["unique_id", "ds", "y"]]
        sf_input["unique_id"] = sf_input["unique_id"].astype(str)

        model = MLForecast(
            models={"lightgbm": lgb.LGBMRegressor(objective="tweedie", verbosity=-1)},
            freq="D",
            lags=list(lags),
            lag_transforms={w: [RollingMean(window_size=w)] for w in rolling_windows},
            date_features=["dayofweek", "month"],
        )
        model.fit(sf_input, static_features=[])
        forecast = model.predict(h=horizon)
        return forecast.rename(columns={"ds": "date", "lightgbm": "yhat"})[["unique_id", "date", "yhat"]]

    return forecast_fn


def train_and_log_candidate(
    hierarchy: dict[str, pd.DataFrame],
    id_map: pd.DataFrame,
    windows: list[BacktestWindow],
    data_hash: str,
    weight_lookback_days: int,
    lags=LAGS,
    rolling_windows=ROLLING_WINDOWS,
) -> tuple[str, dict[str, float]]:
    level_scores = score_model(
        hierarchy,
        id_map,
        make_lightgbm_forecaster(lags, rolling_windows),
        windows,
        weight_lookback_days,
    )
    run_id = log_training_run(
        model_type="candidate",
        params={"model": "lightgbm_tweedie", "lags": list(lags), "rolling_windows": list(rolling_windows)},
        level_scores=level_scores,
        data_hash=data_hash,
        feature_list=feature_list(lags, rolling_windows),
        windows=windows,
    )
    return run_id, level_scores
