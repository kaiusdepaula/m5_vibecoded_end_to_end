# M5 Hierarchical Forecasting POC

28-day hierarchical demand forecasting on the M5 dataset extract, tracked and served through MLflow, with a REST API and request-level tracing. See `m5_challenge_spec.md` for the full requirements and `docs/plans/2026-09-10-001-feat-m5-hierarchical-forecasting-poc-plan.md` for the implementation plan.

## Running it

1. Place the M5 extract in `data/` (see `data/README.md`).
2. `docker compose up --build`

Three services come up: `mlflow` (tracking server + registry, UI at `http://localhost:5000`), `train` (one-shot: loads data, backtests baseline and candidate, registers the winner as `m5_hierarchical@champion`), and `app` (FastAPI at `http://localhost:8000`, waits for `train` to finish before loading the champion model).

## API

- `POST /forecast` — `{"level": "item_store", "series_id": "..."}` or `{"level": "store"|"state", "key": "..."}` or `{"level": "total"}` → 28 dated values, naming the model and registry version that produced them.
- `GET /health` — `{"status": "ok"}`.
- `GET /model-info` — registry name/version, source run id, backtest windows, features, per-level WRMSSE.

Every response carries a `request_id` that also appears in the JSON logs and in the corresponding MLflow trace (`http://localhost:5000`, Traces tab).

## Architecture decisions and trade-offs

**Bottom-up reconciliation, not MinT.** Only the item_store level is modeled directly; store/state/total forecasts are produced by summing item_store forecasts (`src/m5_forecast/models/reconcile.py`). This is coherent by construction and needs no covariance estimation, unlike MinT reconciliation. MinT would likely improve accuracy further but is out of scope for a 2-day POC — noted here as future work, not built.

**SQLite backend store, not Postgres/MinIO.** MLflow's Model Registry requires a database-backed tracking store; a plain file store won't support it. SQLite satisfies that with no extra moving parts, which fits both the timebox and the spec's exclusion of external databases. The `train`-then-`app` service ordering in `docker-compose.yml` makes this a single-writer, then read-only-reader pattern, so SQLite's concurrency limits are never actually exercised.

**LightGBM (via `mlforecast`) as the candidate, not a classical statistical model.** A single pooled global LightGBM model with lag/rolling/calendar features matches the approach that dominated the actual M5 leaderboard, and `mlforecast` automates the feature generation, making it the lowest-effort path to a real accuracy edge within a 1-day budget.

**Seasonal-naive (lag-7), not Croston/SBA, as the baseline.** M5 demand is calendar-seasonal (day-of-week, holidays), not the sparse-erratic pattern Croston-family methods are designed for; seasonal-naive is also the standard M5 competition baseline.

**Forecasts are pre-computed at training time, not re-inferred per request.** The registered model artifact (`ForecastLookupModel`) bundles a 28-day item_store forecast produced by a final fit over the full history, plus store/state/total reconciliation on demand. This avoids carrying the full feature-engineering pipeline into the serving container and matches "serves 28-day forecasts" rather than requiring live re-inference per call.

**Trains on a random subsample of series (`sample_series: 3000` in `config/settings.yaml`), not the full ~30,490.** The provided extract turned out to be the complete M5 dataset; melting and training on all of it repeatedly exhausted available memory during this POC's timebox, even after fixing several real inefficiencies (categorical dtypes for high-cardinality columns, avoiding redundant hierarchy recomputation, a pandas categorical-hashing bug worked around in `compute_data_hash`). Sampling is a deliberate, documented scope reduction consistent with NFR-09 ("model tuning limited to what fits the timebox"), not a silent one. `load_sales(..., sample_series=..., random_state=...)` samples at the wide-series level (before melting), so the full history for each sampled series is kept intact, and `sell_prices` is filtered to match. Set `sample_series: null` to use the full extract on a machine with enough memory.

## Documented data rules

- **Zero-sale days are retained, not dropped** (FR-04). `melt_sales` keeps every `d_*` column regardless of value; a series's own history for WRMSSE scaling starts wherever the caller's train/test windows say it does, not at some zero-filtered point.
- **Missing sell-price weeks keep their sales row** (FR-05). `build_modelling_table` left-joins prices onto sales; a week with no matching price row gets `has_price=False` and `sell_price=NaN`, and its dollar contribution to level weighting is treated as 0 (`dollar_weights`) rather than imputed.
- **Temporal validation**: rolling-origin backtest windows with a fixed step equal to the horizon (28 days by default), no shuffling (`src/m5_forecast/evaluation/backtest.py`). The window count and step are both externalized in `config/settings.yaml` rather than hardcoded, so the walkthrough can be re-run with a shorter horizon for a faster demo without touching code.
- **Reproducibility tolerance** (NFR-01): declared as `1e-4` in `config/settings.yaml` (`reproducibility_tolerance`). Two clean-container training runs should match per-level WRMSSE within this tolerance, given the fixed `random_seed` also in config.

## WRMSSE, hand-derived

`src/m5_forecast/evaluation/wrmsse.py` implements RMSSE and its weighted, multi-level aggregation from scratch (no forecasting-metrics library), per FR-09. The test fixture in `tests/test_wrmsse.py` is hand-derivable:

- Training history `[10, 12, 8, 14, 6]` has day-over-day diffs `2, -4, 6, -8`, squared `4, 16, 36, 64`, mean `30` — this is the naive-forecast scale (the denominator).
- Test actuals `[10, 20]` against a forecast of `[15, 15]` give squared errors `25, 25`, mean `25` — the numerator.
- `RMSSE = sqrt(25 / 30) ≈ 0.9129`.

A second fixture in the same file hand-verifies the dollar-weighted level aggregation independently of the RMSSE computation itself (two series with contrived RMSSE values `1.0` and `0.5` and weights `0.6`/`0.4` combine to `0.6*1.0 + 0.4*0.5 = 0.8`), and a third exercises the divide-by-zero guard: a constant training history has zero scale, so `rmsse` returns `NaN` and `level_wrmsse` excludes that series and renormalizes the remaining weights, rather than propagating `NaN` through the whole level.

## Leakage check

`src/m5_forecast/evaluation/backtest.py`'s `leakage_check` asserts `train_end < test_start` for every backtest window. `tests/test_leakage_check.py` includes a deliberately-overlapping window and asserts the check raises on it — a check that can't fire proves nothing (per the spec's own AC-12).

## What's not built

MinT reconciliation, prediction intervals, retraining/drift alerting, and a broader automated test suite are all out of scope — see `m5_challenge_spec.md` section 5.
