import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from ..evaluation.backtest import BacktestWindow
from ..evaluation.scoring import score_model
from ..mlflow_utils import log_training_run

SEASON_LENGTH = 7


def make_seasonal_naive_forecaster(season_length: int = SEASON_LENGTH):
    def forecast_fn(train_df: pd.DataFrame, window: BacktestWindow) -> pd.DataFrame:
        horizon = (window.test_end - window.test_start).days + 1
        sf_input = train_df.rename(columns={"date": "ds", "sales": "y"})[["unique_id", "ds", "y"]]
        sf_input["unique_id"] = sf_input["unique_id"].astype(str)
        sf = StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq="D")
        forecast = sf.forecast(df=sf_input, h=horizon)
        return forecast.rename(columns={"ds": "date", "SeasonalNaive": "yhat"})[["unique_id", "date", "yhat"]]

    return forecast_fn


def train_and_log_baseline(
    hierarchy: dict[str, pd.DataFrame],
    id_map: pd.DataFrame,
    windows: list[BacktestWindow],
    data_hash: str,
    weight_lookback_days: int,
    season_length: int = SEASON_LENGTH,
) -> tuple[str, dict[str, float]]:
    level_scores = score_model(
        hierarchy,
        id_map,
        make_seasonal_naive_forecaster(season_length),
        windows,
        weight_lookback_days,
    )
    run_id = log_training_run(
        model_type="baseline",
        params={"model": "seasonal_naive", "season_length": season_length},
        level_scores=level_scores,
        data_hash=data_hash,
        feature_list=[],
        windows=windows,
    )
    return run_id, level_scores
