import numpy as np


def rmsse(actual, forecast, train_history) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    train_history = np.asarray(train_history, dtype=float)

    if actual.shape != forecast.shape:
        raise ValueError(
            f"actual and forecast must have the same shape, got {actual.shape} vs {forecast.shape}"
        )
    if train_history.size < 2:
        raise ValueError("train_history needs at least 2 observations to scale against")

    numerator = np.mean((actual - forecast) ** 2)
    denominator = np.mean(np.diff(train_history) ** 2)

    if denominator == 0:
        # Constant training history: the one-step-naive baseline has zero error,
        # so RMSSE is undefined here. Documented rule: return NaN and exclude the
        # series from weighted aggregation in level_wrmsse, rather than divide by zero.
        return float("nan")

    return float(np.sqrt(numerator / denominator))


def level_wrmsse(rmsse_by_series: dict[str, float], dollar_weight_by_series: dict[str, float]) -> float:
    valid = {
        series: weight
        for series, weight in dollar_weight_by_series.items()
        if not np.isnan(rmsse_by_series.get(series, float("nan")))
    }
    total_weight = sum(valid.values())
    if total_weight == 0:
        return float("nan")
    return sum((weight / total_weight) * rmsse_by_series[series] for series, weight in valid.items())


def wrmsse(level_scores: dict[str, float]) -> float:
    scores = [score for score in level_scores.values() if not np.isnan(score)]
    if not scores:
        return float("nan")
    return float(np.mean(scores))
