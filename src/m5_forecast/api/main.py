import logging
import uuid
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow import MlflowClient

from ..config import load_settings
from ..logging import configure_logging, request_id_var
from .schemas import ErrorResponse, ForecastPoint, ForecastRequest, ForecastResponse, ModelRef
from .tracing import traced_request

logger = logging.getLogger("m5_forecast.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = load_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("m5_hierarchical")

    client = MlflowClient()
    version = client.get_model_version_by_alias(settings.registered_model_name, settings.model_alias)
    model = mlflow.pyfunc.load_model(f"models:/{settings.registered_model_name}@{settings.model_alias}")
    run = client.get_run(version.run_id)

    # Exposed via /model-info so a caller (or the demo script) can discover a
    # real id rather than guessing -- particularly important now that training
    # runs on a random series sample, so no specific id is guaranteed to exist.
    forecast_impl = model.unwrap_python_model()
    example_series_id = str(forecast_impl.item_store_forecast["unique_id"].iloc[0])
    example_store = str(forecast_impl.id_map["store_id"].iloc[0])
    example_state = str(forecast_impl.id_map["state_id"].iloc[0])

    app.state.settings = settings
    app.state.model = model
    app.state.model_ref = ModelRef(name=settings.registered_model_name, version=int(version.version))
    app.state.run = run
    app.state.examples = {"item_store": example_series_id, "store": example_store, "state": example_state}

    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = request_id_var.get() or str(uuid.uuid4())
    errors = exc.errors()
    message = errors[0]["msg"] if errors else "invalid payload"
    body = exc.body if isinstance(exc.body, dict) else {}
    with traced_request(request.url.path, request_id, inputs=body) as outcome:
        outcome["status"] = "rejected"
        outcome["reason"] = message
        outcome["output"] = {"error": "invalid_payload", "message": message}
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="invalid_payload", message=message, request_id=request_id).model_dump(),
    )


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        logger.info(f"{request.method} {request.url.path} -> {response.status_code}")
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    settings = request.app.state.settings
    run = request.app.state.run
    return {
        "name": settings.registered_model_name,
        "version": request.app.state.model_ref.version,
        "run_id": run.info.run_id,
        "backtest_windows": run.data.tags.get("backtest_windows"),
        "features": run.data.tags.get("feature_list"),
        "wrmsse": {k: v for k, v in run.data.metrics.items() if k.startswith("wrmsse_")},
        "examples": request.app.state.examples,
    }


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def forecast(payload: ForecastRequest, request: Request):
    request_id = request_id_var.get() or str(uuid.uuid4())
    model = request.app.state.model
    settings = request.app.state.settings

    with traced_request("/forecast", request_id, level=payload.level, inputs=payload.model_dump()) as outcome:
        result = model.predict(pd.DataFrame([{"level": payload.level, "key": payload.lookup_key}]))
        if result.empty:
            identifier_name = "series_id" if payload.level == "item_store" else "key"
            message = (
                f"{identifier_name} '{payload.lookup_key}' not found. "
                "Call GET /model-info for the available levels and an example id."
            )
            outcome["status"] = "rejected"
            outcome["reason"] = "unknown_series"
            outcome["output"] = {"error": "unknown_series", "message": message}
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(error="unknown_series", message=message, request_id=request_id).model_dump(),
            )

        forecast_points = [
            ForecastPoint(date=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"), value=float(row["value"]))
            for _, row in result.iterrows()
        ]
        response = ForecastResponse(
            level=payload.level,
            series_id=payload.series_id,
            key=payload.key,
            model=request.app.state.model_ref,
            horizon=settings.horizon,
            forecast=forecast_points,
            request_id=request_id,
        )
        outcome["output"] = response.model_dump()
        return response
