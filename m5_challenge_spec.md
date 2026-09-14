# ML/MLOps Challenge: Hierarchical Forecasting (M5)

**Project:** 28 day hierarchical demand forecasting with a served model
**Type:** Proof of Concept
**Dataset:** M5 Forecasting Accuracy, provided extract. Horizon fixed at 28 days.
**Stack:** MLflow for experiment tracking, model versioning and request tracing. Everything else is the implementer's choice: language, libraries, web framework and orchestration are all open, and the rest of this document is stated at the interface level.
**Deployment constraint:** The whole flow comes up with one command on a clean machine, with the application and the observability stack starting together

---

## 1. Overview

The POC takes the raw M5 extract, cleans and validates it, trains and backtests several forecasting models, picks the best one by a WRMSSE metric implemented inside the project, and serves 28 day forecasts over a local REST API. The API reports its own health and the model it is running. Every request emits a trace to the MLflow tracking server and a structured log line carrying the same request id, so any single call can be reconstructed afterwards.

MLflow carries the reproducibility story as well. Each training run is an MLflow run holding its parameters, its per-level WRMSSE, the data hash, and the model and preprocessing pipeline as artifacts. The winner is promoted into the Model Registry, and the service serves a named registry version rather than a file someone copied into place.

The goal is to prove the full path from raw files to a served, observable, reproducible model. It is not to win the Kaggle leaderboard. Success is a pipeline that a reviewer can run, inspect, and trust, with an honest comparison against a statistical baseline.

### Reference architecture

```
  one command, clean machine
  ──────────────────────────

  raw extract (sales, calendar, sell_prices)
        │
        ▼
  schema contract (types, ranges, allowed nulls)
        │            enforced at training and at inference
        ▼
  feature build  ──►  sliding-window backtest  ──►  WRMSSE
                            (no future data)       (in-project)
        │
        ▼
  MLflow run, one per training run
    params · WRMSSE per level as metrics · data hash, features and
    backtest windows as tags · model and preprocessing as artifacts
        │
        ▼
  MLflow Model Registry
    the selected model, as a named version
        │
        ▼
  HTTP service, loads the registered version at startup
    POST /forecast     GET /health     GET /model-info
        │
        ├──►  one trace per request, spanning the pipeline stages
        │           │
        │           ▼
        │     MLflow tracking server  (traces, runs, registry, UI)
        │
        └──►  JSON log line carrying the same request id
```

Everything above comes up from that one command, the tracking server included. No manual steps between raw files and a live endpoint.

---

## 2. Functional Requirements

### Data

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Load `sales`, `calendar` and `sell_prices` from the provided extract, clean them, and join them into a modelling table. | Must |
| FR-02 | The schema contract is code, not prose: column types, value ranges, and which nulls are allowed. | Must |
| FR-03 | The same contract runs at training and at inference. A violation fails loudly with a message naming the column and the rule broken. | Must |
| FR-04 | Intermittent series and zero-sale days follow a documented rule. Silent dropping is not acceptable. | Must |
| FR-05 | Periods with no price are handled explicitly and the rule is documented. | Must |
| FR-06 | The temporal validation strategy is written down and justified. Why these windows, why this cutoff. | Must |

### Modeling

| ID | Requirement | Priority |
|---|---|---|
| FR-07 | Train and evaluate at least two models, one of which is a statistical baseline such as seasonal naive or Croston. | Must |
| FR-08 | Backtest with sliding windows. No training window may see data from after its own cutoff. | Must |
| FR-09 | WRMSSE is implemented inside the project, not imported from a library. Its correctness is evidenced per FR-26. | Must |
| FR-10 | The best model is chosen by WRMSSE, reported per hierarchy level, and the choice is traceable. | Must |
| FR-11 | Forecasts are coherent across item, store, state and total. Lower levels aggregate to the higher ones within a declared tolerance. | Must |

### Artifacts and reproducibility

| ID | Requirement | Priority |
|---|---|---|
| FR-12 | Every training run is logged to an MLflow tracking server: parameters, WRMSSE per hierarchy level as metrics, and the fitted model together with its preprocessing pipeline as artifacts of that run. | Must |
| FR-13 | The hash of the input data, the feature list and the backtest windows are recorded on the run as tags or artifacts. The run is the model card. Nothing is maintained by hand alongside it. | Must |
| FR-14 | The selected model is registered in the MLflow Model Registry under a stable name. The served build is identified by registry name and version. | Must |

### API

| ID | Requirement | Priority |
|---|---|---|
| FR-15 | `POST /forecast` accepts a series identifier or a hierarchy level and returns 28 dated values. | Must |
| FR-16 | `GET /health` returns service status. | Must |
| FR-17 | `GET /model-info` returns the registry name and version being served, the source run id, the training window, the features, per-level WRMSSE, and the reference forecast distribution from training. | Must |
| FR-18 | Every forecast response names the model that produced it, by registry name and version. | Must |
| FR-19 | An invalid payload returns 4xx with a message the caller can act on. Never 5xx. | Must |
| FR-20 | An unknown series or level is answered cleanly. The service does not crash or degrade. | Must |

### Observability

| ID | Requirement | Priority |
|---|---|---|
| FR-21 | Each request produces one trace on the MLflow tracking server, spanning the pipeline stages, carrying the request id, endpoint, status, duration and the hierarchy level served. | Must |
| FR-22 | Rejected payloads are traced as well, with the rejection reason, so refused traffic is visible rather than silently dropped. | Must |
| FR-23 | Logs are structured JSON. Each carries a request id that also appears in the trace and in the response. | Must |
| FR-24 | The MLflow tracking server starts together with the application from the same single command, and its UI is reachable at a documented port. | Must |
| FR-25 | The live forecast distribution is compared against the training reference distribution recorded with the run, and the comparison is exposed at `GET /model-info`. | Should |

Aggregate views, requests by status, latency spread and volume per level, come from querying the logged traces. No separate metrics backend is required.

### Correctness evidence and configuration

| ID | Requirement | Priority |
|---|---|---|
| FR-26 | The WRMSSE implementation is shown to be correct on a small worked example whose expected value was computed by hand. | Must |
| FR-27 | The validation setup is shown to be free of temporal leakage, either by an automated check or by a documented walkthrough of the windows used. | Must |
| FR-28 | Configuration is externalised. No magic numbers buried in code. | Must |
| FR-29 | README covers architecture decisions and the trade-offs behind them. | Must |

A broader automated test suite is optional. See Out of Scope.

### Execution

| ID | Requirement | Priority |
|---|---|---|
| FR-30 | One command takes a clean machine from raw data to a running API with the tracking server attached. No manual steps in between. | Must |

### 2.1 API contract

`POST /forecast`, request by series:

```json
{
  "level": "item_store",
  "series_id": "HOBBIES_1_001_CA_1"
}
```

Request by hierarchy level:

```json
{
  "level": "state",
  "key": "CA"
}
```

Success (`200`):

```json
{
  "level": "item_store",
  "series_id": "HOBBIES_1_001_CA_1",
  "model": {"name": "m5_hierarchical", "version": 3},
  "horizon": 28,
  "forecast": [
    {"date": "2016-04-25", "value": 1.4},
    {"date": "2016-04-26", "value": 1.1}
  ],
  "request_id": "9f2c..."
}
```

Rejected input (`400` / `404`):

```json
{
  "error": "unknown_series",
  "message": "series_id 'FOO_1_001' not found. Call GET /model-info for the available levels and an example id.",
  "request_id": "9f2c..."
}
```

`GET /health` returns `{"status": "ok"}`.

Field names are a starting point. The requirement is that a caller can get 28 days, know which model produced them, and correlate the call to a log line and a trace.

---

## 3. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 Reproducibility | Two runs in a clean container produce identical WRMSSE, or values within a tolerance the team declares in the README. Both appear as MLflow runs with their metrics and tags. |
| NFR-02 Reliability | Bad client input never produces a 5xx. The service stays up through the walkthrough. |
| NFR-03 Correctness | The WRMSSE implementation is validated against an example small enough to check by hand. |
| NFR-04 No leakage | Absence of temporal leakage is demonstrated explicitly, not asserted in passing. |
| NFR-05 Observability | A single request can be followed from the response, through the JSON log, to its trace in MLflow, using one request id. |
| NFR-06 Performance | `/forecast` answers fast enough to feel interactive. Training completes inside the timebox on the provided extract. |
| NFR-07 Portability | One documented command on any machine with a container runtime. No host side dependencies, no manual data download step if the extract is vendored. |
| NFR-08 Maintainability | Clear separation between data, features, training, evaluation and serving. Config externalised. README present. |
| NFR-09 Scope control | The provided extract only. Point forecasts only. Model tuning limited to what fits the timebox. |

---

## 4. Acceptance Criteria

The POC is successful when all of these are demonstrable.

- **AC-01** Given a clean machine with only a container runtime installed, when the reviewer runs the documented single command, then the pipeline runs and both the API and the MLflow tracking server come up with no manual steps.
- **AC-02** Given a deliberately corrupted input column, when the pipeline runs, then it fails with a message naming the column and the rule violated, not an opaque stack trace from inside a data library.
- **AC-03** Given a valid item and store identifier, when `POST /forecast` is called, then 28 dated values are returned, naming the registry model and version that produced them.
- **AC-04** Given `level: state` and a state key, when `POST /forecast` is called, then 28 values for that aggregate level are returned.
- **AC-05** Given forecasts at item level, when they are summed to store, state and total, then they match the forecasts served for those levels within the declared tolerance.
- **AC-06** Given a malformed payload, when it is posted, then the response is 4xx with an actionable message and the logs show no 5xx.
- **AC-07** Given an identifier that does not exist, when it is posted, then the response is a handled error and `/health` still reports ok.
- **AC-08** Given a handful of forecast requests and one rejected payload, when the reviewer opens the MLflow UI, then a trace exists for each, showing status, duration and the level served, and the rejected one shows its reason.
- **AC-09** Given the stack is running, when `GET /model-info` is called, then it returns the registry name and version, the source run id, data hash, WRMSSE per level, features, backtest windows, and the reference forecast distribution.
- **AC-10** Given one forecast request, when the reviewer takes the request id from the response and looks it up in the JSON logs and in the MLflow trace, then all three agree.
- **AC-11** Given the worked WRMSSE example, when the project's implementation is run on it, then the result matches the hand-computed value.
- **AC-12** Given a validation window deliberately made to overlap the training data, when the leakage check runs, then it flags the overlap. A check that never fires proves nothing.
- **AC-13** Given two runs in clean containers, when the two MLflow runs are compared, then WRMSSE per level is identical or inside the declared tolerance, and the data hash tags match.
- **AC-14** Given the runs in MLflow, when baseline and candidate are compared, then per-level WRMSSE for both is visible and the version in the registry is the better of the two.

---

## 5. Out of Scope

- Authentication, multi-tenancy, rate limiting.
- Cloud deployment, CI/CD, orchestration platforms, autoscaling, high availability. Local single-command run only.
- The full M5 dataset or other competition tracks. The provided extract only.
- The M5 Uncertainty track. Point forecasts only, no prediction intervals or quantiles.
- Feature stores, data warehouses, external databases.
- A user interface. The MLflow UI is the only front end.
- Metrics backends, scrape based monitoring and custom dashboards. MLflow runs and traces are the whole observability surface.
- Automated retraining, drift alerting, or acting on the distribution comparison. The comparison is exposed, nothing reacts to it.
- Large scale hyperparameter search, AutoML, or ensembling beyond picking the better model.
- A broad automated test suite or any coverage target. This is a single scoped task, not a codebase to maintain. The only correctness evidence required is the worked WRMSSE example and the leakage demonstration.
- Competitive accuracy. A correct pipeline with an honest baseline comparison beats an unexplainable better number.

---

## 6. Assumptions and Constraints

- The M5 extract is available at the start and small enough to train on inside the timebox.
- The horizon is fixed at 28 days.
- The reproducibility tolerance is declared by the team in the README rather than fixed by this document.
- Timebox is assumed to be 1 days unless stated otherwise. Must requirements come first, Should items only if time allows.
- Judgement is a live walkthrough plus a test run, not a leaderboard position.

---

## 7. Definition of Done

- Every Must requirement implemented and demonstrated.
- AC-01 to AC-14 pass in a live run.
- The whole flow comes up with one documented command on a clean machine.
- WRMSSE correctness shown on the worked example, and the leakage check shown to fire when leakage is introduced.
- Two separate clean runs visible in MLflow, with matching metrics inside the declared tolerance, and the selected model registered as a version.
- README explains the architecture and the trade-offs taken.
- A short demo script is ready.
