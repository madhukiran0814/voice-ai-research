#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Project root + optional shared ML packages from /data/venv
PYTHONPATH_PARTS=("$ROOT")
if [[ -d /data/venv/lib/python3.13/site-packages ]]; then
  PYTHONPATH_PARTS+=("/data/venv/lib/python3.13/site-packages")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "$ROOT/.venv/bin/uvicorn" ]]; then
  UVICORN="$ROOT/.venv/bin/uvicorn"
elif [[ -x /data/venv/bin/uvicorn ]]; then
  UVICORN=/data/venv/bin/uvicorn
else
  UVICORN=uvicorn
fi

if [[ -z "${HF_HOME:-}" && -d /data/huggingface ]]; then
  export HF_HOME=/data/huggingface
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

EXTRA_ARGS=()
if [[ "${RELOAD:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--reload)
else
  EXTRA_ARGS+=(--workers "$WORKERS")
fi

exec "$UVICORN" app.main:app --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
