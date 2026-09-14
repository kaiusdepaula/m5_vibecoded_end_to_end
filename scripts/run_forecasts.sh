#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-http://localhost:8000}"

info=$(curl -sf "$HOST/model-info")
series_id=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['examples']['item_store'])" "$info")
store=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['examples']['store'])" "$info")
state=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['examples']['state'])" "$info")

forecast() {
  local payload="$1"
  echo "-> $payload"
  curl -sf -X POST "$HOST/forecast" -H "Content-Type: application/json" -d "$payload" | python3 -m json.tool
  echo
}

echo "== item_store: $series_id =="
forecast "{\"level\": \"item_store\", \"series_id\": \"$series_id\"}"

echo "== store: $store =="
forecast "{\"level\": \"store\", \"key\": \"$store\"}"

echo "== state: $state =="
forecast "{\"level\": \"state\", \"key\": \"$state\"}"

echo "== total =="
forecast '{"level": "total"}'
