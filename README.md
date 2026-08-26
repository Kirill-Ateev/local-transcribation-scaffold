# local-transcribation-stack

Локальная диктовка уровня Wispr Flow: DGX Spark держит STT-сервис
(faster-whisper, единственная резидентная модель **Whisper large-v3**),
MacBook — клиент OpenWhispr с push-to-talk. Только батч, только локальная сеть.

```
MacBook (OpenWhispr)  ──HTTP──▶  DGX Spark (FastAPI + faster-whisper)
  хоткей → запись                  VAD (Silero) → large-v3 → текст
  ◀── вставка в курсор ──────────  JSON {"text", "language", "duration"}
```

## Быстрый старт (сервер)

```bash
python3 -m venv .venv
.venv/bin/pip install -r server/requirements.txt

cp server/.env.example server/.env
# впишите токен: openssl rand -hex 32 → TRANSCRIBE_TOKEN=...

chmod +x run.sh && ./run.sh          # HOST/PORT берутся из server/.env
```

Первый старт скачивает веса (~3 ГБ для large-v3); `/health` до готовности
отвечает `"status":"loading"`. Для пробы без GPU достаточно дефолтов — сервис
сам уйдёт в CPU: временно поставьте в `server/.env` `WHISPER_MODEL=base`.

Проверка:

```bash
curl -s http://127.0.0.1:8337/health
curl -s -X POST http://127.0.0.1:8337/v1/audio/transcriptions \
     -H "Authorization: Bearer $TOKEN" -F file=@фраза.wav -F language=ru
```

Эндпоинты: `POST /v1/audio/transcriptions` (multipart; wav/mp3/flac/m4a/ogg/
opus/webm или сырой `.pcm` s16le mono 16 kHz), `GET /v1/models`,
`GET /health` (без авторизации). Ошибки: 401 / 400 / 413 / 503+Retry-After.

## Тесты

```bash
.venv/bin/python -m pytest        # конфиг pytest.ini: сервер на tiny/CPU, ~1 мин
```

Длинное аудио (локальный прокси-тест часовой записи):
`.venv/bin/python server/scripts/test_hour_long.py`

## Оценка качества (eval)

Записи фраз из `eval/phrases.yaml` кладутся в `eval/audio/<id>.wav`, затем:

```bash
.venv/bin/python eval/score.py --url http://<spark>:8337 --token $TOKEN --label baseline
.venv/bin/python eval/score.py ... --prompt-file glossary.txt --label with-prompt
```

## Развёртывание и клиент

- DGX Spark: `deploy/README.md` — вариант A (основной): Docker-рецепт
  [`faster-whisper-dgx-spark`](https://github.com/paruparu/faster-whisper-dgx-spark)
  + наш сервис в его контейнере (`deploy/docker-compose.spark.yml`);
  вариант B (альтернатива): venv + systemd; LAN-bind, smoke-check
- MacBook: `client/README.md` (OpenWhispr как custom provider, замер латентности)
- Спайк рантайма GB10: `docs/spike-runtime.md`

Развёртывание на целевых машинах и приёмочные проверки выполняются владельцем
отдельно; репозиторий поставляет код сервиса, eval-харнесс и инструкции.

## Планирование

Артефакты изменений — в `openspec/changes/local-voice-transcription-stack`
(proposal, specs, design, tasks).
