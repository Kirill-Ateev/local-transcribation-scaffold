#!/usr/bin/env bash
# Единственная точка запуска сервиса: ./run.sh
# HOST/PORT и остальная конфигурация берутся из server/.env.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f server/.env ]]; then
    echo "Нет server/.env — выполните: cp server/.env.example server/.env" >&2
    exit 1
fi
set -a; source server/.env; set +a

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PY=.venv/bin/python
    else
        PY=python3
    fi
fi
if ! "$PY" -c "import uvicorn" 2>/dev/null; then
    echo "uvicorn не найден в $PY. Выполните:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt" >&2
    exit 1
fi

exec "$PY" -m uvicorn app.main:app \
    --app-dir server \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8337}"
