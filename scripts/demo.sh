#!/usr/bin/env bash
set -euo pipefail

wait_for_train() {
  echo "Waiting for the train service to complete..."
  while true; do
    # -a/--all is required here: once train exits, a plain `ps` (running
    # containers only) returns nothing and this loop would never terminate.
    status=$(docker compose ps -a -q train | xargs -r docker inspect -f '{{.State.Status}}' 2>/dev/null || echo "")
    [ "$status" = "exited" ] && break
    sleep 2
  done
  docker compose logs train --tail 30

  exit_code=$(docker compose ps -a -q train | xargs -r docker inspect -f '{{.State.ExitCode}}')
  if [ "$exit_code" != "0" ]; then
    echo "train exited with code $exit_code -- aborting rather than continuing with no model." >&2
    exit 1
  fi
}

wait_for_app() {
  # train exiting successfully only means Compose will start app; app's own
  # FastAPI startup (loading the model, calling unwrap_python_model()) still
  # takes a few seconds, and curl-ing before that finishes fails silently
  # under -s. Poll /health instead of assuming readiness.
  echo "Waiting for the app service to become healthy..."
  for _ in $(seq 1 60); do
    if curl -sf -o /dev/null http://localhost:8000/health 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "app did not become reachable at http://localhost:8000/health -- check 'docker compose logs app'." >&2
  exit 1
}

echo "== Clean run 1 =="
docker compose down -v
docker compose up --build -d
wait_for_train
wait_for_app

echo
echo "== model-info (run 1) =="
curl -s http://localhost:8000/model-info | tee /tmp/m5_demo_model_info_1.json
echo

# Training now runs on a random series sample (config/settings.yaml:
# sample_series), so no specific series id is guaranteed to exist -- pull a
# real one from model-info's "examples" field rather than hardcoding one.
example_series_id=$(python3 -c "import json; print(json.load(open('/tmp/m5_demo_model_info_1.json'))['examples']['item_store'])")
example_state=$(python3 -c "import json; print(json.load(open('/tmp/m5_demo_model_info_1.json'))['examples']['state'])")

echo
echo "== A few forecast calls (using example ids: series=$example_series_id, state=$example_state) =="
curl -s -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d "{\"level\": \"item_store\", \"series_id\": \"$example_series_id\"}" | tee /tmp/m5_demo_forecast_1.json
echo

curl -s -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d "{\"level\": \"state\", \"key\": \"$example_state\"}"
echo

echo
echo "== One rejected payload (missing series_id) =="
curl -s -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"level": "item_store"}'
echo

echo
echo "== Clean run 2 (reproducibility check, AC-13) =="
docker compose down -v
docker compose up --build -d
wait_for_train
wait_for_app

curl -s http://localhost:8000/model-info | tee /tmp/m5_demo_model_info_2.json
echo

echo
echo "Compare /tmp/m5_demo_model_info_1.json and /tmp/m5_demo_model_info_2.json:"
echo "per-level WRMSSE should match within the tolerance declared in README.md (NFR-01)."
echo
echo "MLflow UI: http://localhost:5000"
echo "Take the request_id from any forecast response above, grep it in 'docker compose logs app',"
echo "and look it up in the MLflow UI Traces tab -- all three should agree (AC-10)."
