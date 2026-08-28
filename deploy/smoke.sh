#!/usr/bin/env bash
# Smoke-check развёрнутого сервиса: health + контрольная транскрибация.
# Использование:
#   ./smoke.sh http://<spark-ip>:8337 <token> [control.wav] [language]
# language опционален: без него сервер применит DEFAULT_LANGUAGE (auto — авто-детект).
set -euo pipefail

BASE="${1:?укажите базовый URL, например http://192.168.1.50:8337}"
TOKEN="${2:?укажите токен}"
CONTROL="${3:-}"
LANG_OPT="${4:-}"

echo "== GET /health =="
curl -fsS "$BASE/health" | python3 -m json.tool

echo "== POST /v1/audio/transcriptions (auth check) =="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/audio/transcriptions" \
    -F "file=@$0")
if [[ "$code" == "401" ]]; then
    echo "OK: без токена сервис отвечает 401 (авторизация включена)"
elif [[ "$code" == "400" ]]; then
    echo "ВНИМАНИЕ: авторизация отключена (AUTH_REQUIRED=0) — анонимные запросы принимаются"
else
    echo "ОШИБКА: неожиданный код $code на запрос без токена" >&2
    exit 1
fi

if [[ -n "$CONTROL" ]]; then
    echo "== контрольная транскрибация ($CONTROL) =="
    args=( -X POST "$BASE/v1/audio/transcriptions"
           -H "Authorization: Bearer $TOKEN"
           -F "file=@$CONTROL" )
    [[ -n "$LANG_OPT" ]] && args+=( -F "language=$LANG_OPT" )
    curl -fsS "${args[@]}"
    echo
fi

echo "Smoke-check завершён."
