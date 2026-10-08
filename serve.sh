#!/usr/bin/env bash
# Serve RED-SNOW 2.49bpw on ONE DGX Spark with an OpenAI-compatible /v1 (chat, streaming, tool calls).
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-$HOME/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-2.49bpw}"
IMAGE="${IMAGE:-redsnow-exl3:7c1636f}"
PORT="${PORT:-8899}"
CTX="${CTX:-262144}"
GPU_SPLIT_GB="${GPU_SPLIT_GB:-112}"      # explicit split: the auto-split undercounts free unified memory on GB10
NAME="${NAME:-redsnow}"
HERE="$(cd "$(dirname "$0")" && pwd)"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# The Spark's page cache holds the ~105 GB of weights after reading them; the loader does not count it as free.
# Drop it before and during the load (needs passwordless sudo for tee; otherwise run the two lines yourself).
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "warn: could not drop caches"
( for i in $(seq 1 300); do sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; sleep 2; done ) &
DROPPER=$!
trap 'kill $DROPPER 2>/dev/null || true' EXIT

docker run -d --name "$NAME" --gpus all --network host --ipc host --ulimit memlock=-1 \
  -v "$MODEL_DIR":/model:ro -v "$HERE/server":/srv:ro -w /srv "$IMAGE" \
  python serve_glm.py -m /model -gs "$GPU_SPLIT_GB" \
    --cache_size "$CTX" --cache_quant 4 --mtp --num_draft_tokens 2 \
    --max-model-len "$CTX" --host 0.0.0.0 --port "$PORT" \
    --served-model-name RED-SNOW-5.3-FLASH-EXL3-2.49 --request-timeout 7200 >/dev/null

for i in $(seq 1 20); do
  sleep 30
  if docker logs "$NAME" 2>&1 | grep -q "Native API listening"; then echo "UP on :$PORT"; exit 0; fi
  docker ps -q -f name="$NAME" | grep -q . || { echo "container died:"; docker logs --tail 30 "$NAME"; exit 1; }
done
echo "not up after 10 min"; docker logs --tail 30 "$NAME"; exit 1
