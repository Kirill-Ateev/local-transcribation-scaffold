# local-transcribation-stack

Локальная диктовка уровня Wispr Flow: DGX Spark держит STT-сервис
(faster-whisper, единственная резидентная модель **Breeze-ASR-25** — fine-tune
Whisper large-v2 от MediaTek: традиционный китайский + английский, включая их
смешение; меняется переменной `WHISPER_MODEL`), MacBook — клиент OpenWhispr с
push-to-talk. Только батч, только локальная сеть.

```
MacBook (OpenWhispr)  ──HTTP──▶  DGX Spark (FastAPI + faster-whisper)
  хоткей → запись                  VAD (Silero) → breeze-asr-25 → текст
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

Первый старт скачивает веса (~3.1 ГБ для Breeze-ASR-25 fp16; заранее —
`.venv/bin/python server/scripts/download_model.py`); `/health` до готовности
отвечает `"status":"loading"`. Для пробы без GPU достаточно дефолтов — сервис
сам уйдёт в CPU: временно поставьте в `server/.env` `WHISPER_MODEL=base`.

Проверка:

```bash
curl -s http://127.0.0.1:8337/health
curl -s -X POST http://127.0.0.1:8337/v1/audio/transcriptions \
     -H "Authorization: Bearer $TOKEN" -F file=@фраза.wav -F language=zh
```

Эндпоинты: `POST /v1/audio/transcriptions` (multipart; wav/mp3/flac/m4a/ogg/
opus/webm или сырой `.pcm` s16le mono 16 kHz; `language`: `auto`/пусто —
авто-детект по умолчанию, иначе ISO-код), `GET /v1/models`, `GET /health`
(без авторизации). Ошибки: 401 / 400 / 413 / 503+Retry-After.

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

Набор фраз русскоязычный — это удобный харнесс для регрессий и сравнения
режимов (prompt), но учтите: Breeze-ASR-25 оптимизирована под zh/en, поэтому
абсолютные WER на русском будут хуже, чем у large-v3.

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
