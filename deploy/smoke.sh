#!/usr/bin/env bash
# Smoke-check развёрнутого сервиса: health + контрольная транскрибация.
# Использование:
#   ./smoke.sh http://<spark-ip>:8337 <token> [control.wav]
set -euo pipefail

BASE="${1:?укажите базовый URL, например http://192.168.1.50:8337}"
TOKEN="${2:?укажите токен}"
CONTROL="${3:-}"

echo "== GET /health =="
curl -fsS "$BASE/health" | python3 -m json.tool

echo "== POST /v1/audio/transcriptions (auth reject check) =="
if curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/audio/transcriptions" \
    -F "file=@$0" | grep -q 401; then
    echo "OK: без токена сервис отвечает 401"
else
    echo "ОШИБКА: запрос без токена не отклонён" >&2
    exit 1
fi

if [[ -n "$CONTROL" ]]; then
    echo "== контрольная транскрибация ($CONTROL) =="
    curl -fsS -X POST "$BASE/v1/audio/transcriptions" \
        -H "Authorization: Bearer $TOKEN" \
        -F "file=@$CONTROL" \
        -F "language=ru"
    echo
fi

echo "Smoke-check завершён."
