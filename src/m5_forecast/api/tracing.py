import logging
import time
from contextlib import contextmanager

import mlflow

logger = logging.getLogger(__name__)


@contextmanager
def traced_request(endpoint: str, request_id: str, level: str | None = None, inputs: dict | None = None):
    """One MLflow trace per request, tagged with request_id/endpoint/level/status/
    duration -- including rejected requests (set outcome['status']='rejected' and
    outcome['reason']). The caller can set outcome['output'] before the block ends
    to record it as the span's output. Trace-write failures are caught and logged,
    never raised, so an observability hiccup can't turn into a 5xx."""
    start = time.monotonic()
    outcome = {"status": "ok", "reason": None, "output": None}

    with mlflow.start_span(name=endpoint, span_type="AGENT") as span:
        try:
            tags = {"request_id": request_id, "endpoint": endpoint}
            if level:
                tags["level"] = level
            mlflow.update_current_trace(tags=tags, client_request_id=request_id)
            if inputs is not None:
                span.set_inputs(inputs)
        except Exception:
            logger.exception("failed to tag trace at start; continuing without it")

        try:
            yield outcome
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            final_tags = {"status": outcome["status"], "duration_ms": f"{duration_ms:.2f}"}
            if outcome["reason"]:
                final_tags["reason"] = outcome["reason"]
            try:
                mlflow.update_current_trace(tags=final_tags, client_request_id=request_id)
                if outcome["output"] is not None:
                    span.set_outputs(outcome["output"])
            except Exception:
                logger.exception("failed to record trace status/duration; continuing")
