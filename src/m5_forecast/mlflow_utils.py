import json

import mlflow

from .evaluation.backtest import BacktestWindow


def log_training_run(
    model_type: str,
    params: dict,
    level_scores: dict[str, float],
    data_hash: str,
    feature_list: list[str],
    windows: list[BacktestWindow],
) -> str:
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("feature_list", json.dumps(feature_list))
        mlflow.set_tag(
            "backtest_windows",
            json.dumps(
                [
                    {
                        "train_end": str(w.train_end.date()),
                        "test_start": str(w.test_start.date()),
                        "test_end": str(w.test_end.date()),
                    }
                    for w in windows
                ]
            ),
        )
        mlflow.log_params(params)
        mlflow.log_metrics({f"wrmsse_{level}": score for level, score in level_scores.items()})
        return run.info.run_id
