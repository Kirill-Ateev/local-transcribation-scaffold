# Развёртывание STT-сервиса на DGX Spark

Два способа. **Вариант A — целевой**: рантайм из рецепта
[`paruparu/faster-whisper-dgx-spark`](https://github.com/paruparu/faster-whisper-dgx-spark)
(Docker; CTranslate2 собирается из исходников под CUDA 12.4 + cuDNN 9, что
обходит несовместимость pip-CT2 с GB10 при CUDA 13 на хосте). **Вариант B —
альтернатива** «на всякий случай»: собственное окружение venv + systemd.

В обоих вариантах внутри работает один и тот же код `server/` —
OpenAI-совместимый API с токеном, VAD и поддержкой длинных аудио.

## Вариант A (основной): рецепт faster-whisper-dgx-spark

Предпосылки: NVIDIA Container Toolkit установлен, драйвер NVIDIA работает;
хостовой CUDA 13 менять не нужно.

### A.0. Смоук рантайма рецепта (штатный server.py рецепта)

```bash
git clone https://github.com/paruparu/faster-whisper-dgx-spark external/faster-whisper-dgx-spark
cd external/faster-whisper-dgx-spark
docker compose build && docker compose up -d
curl http://localhost:8002/health        # ожидаем device=cuda
curl -X POST http://localhost:8002/transcribe -F "file=@test/003.wav" -F "language=ru"
```

Если `/health` показывает не `cuda` — дальше не идти, зафиксировать вывод в
`docs/spike-runtime.md` и разбираться с рецептом/драйвером.

### A.1. Наш сервис внутри контейнера рецепта

```bash
cd /path/to/local-transcribation-scaffold
cp server/.env.example server/.env       # вписать TRANSCRIBE_TOKEN (openssl rand -hex 32)

TRANSCRIBE_TOKEN=$(grep -oP '(?<=^TRANSCRIBE_TOKEN=).*' server/.env) \
  docker compose -f deploy/docker-compose.spark.yml up -d --build

curl http://localhost:8337/health        # status:ready, device:cuda — готово
```

Compose берёт Dockerfile и собранный CTranslate2 из рецепта, монтирует наш
`server/` внутрь и поднимает uvicorn вместо штатного `server.py` рецепта.
Веса кэшируются в volume `hf-cache`. Конфиг — `server/.env`; `HOST/PORT`
внутри контейнера фиксированы (`0.0.0.0:8337`). Останов: `docker compose -f
deploy/docker-compose.spark.yml down`.

## Вариант B (альтернатива): собственное окружение venv + systemd

Годится для отладки на CPU, для запуска без Docker или если вариант А
недоступен. На GPU вне контейнера рецепта pip-CT2 может не получить CUDA —
проверяйте `backend.device` в `/health`.

```bash
sudo mkdir -p /opt/local-transcribation && sudo chown $USER /opt/local-transcribation
git clone <repo-url> /opt/local-transcribation && cd /opt/local-transcribation
python3 -m venv .venv
.venv/bin/pip install -r server/requirements.txt

cp server/.env.example server/.env
TOKEN=$(openssl rand -hex 32)
sed -i "s/^TRANSCRIBE_TOKEN=$/TRANSCRIBE_TOKEN=$TOKEN/" server/.env
echo "Токен для клиента MacBook: $TOKEN"
```

В `server/.env` задайте `HOST=<IP Spark в домашней LAN>` (`hostname -I`),
при GPU-сборке — `DEVICE=cuda`, `COMPUTE_TYPE=float16`.

```bash
sudo cp deploy/transcribe.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now transcribe.service
journalctl -u transcribe.service -f      # строка "model ready" = прогрев завершён
```

Юнит даёт автозапуск после перезагрузки и перезапуск после падения.

## Сеть (общее для A и B)

- Сервис доступен только из домашней LAN по `http://<spark-ip>:8337`.
- Проброс портов на роутере наружу НЕ настраивать; проверка: обращение через
  внешний адрес роутера отклоняется.
- Для варианта B дополнительно bind к конкретному интерфейсу через `HOST`.

## Smoke-check (с MacBook, общее)

```bash
cd deploy && ./smoke.sh http://<spark-ip>:8337 <token> control.wav
```

Ожидаемо: `/health` со `"status":"ready"`, `"device":"cuda"`; JSON с текстом и
пунктуацией; отказ 401 без токена.

## Клиент MacBook

`client/README.md` (OpenWhispr → custom provider → `http://<spark>:8337/v1`).

## Траблшутинг

- **`device=cpu` вместо `cuda`**: в варианте B — ожидаемое ограничение pip-CT2
  на GB10, используйте вариант A; в варианте A — сверить сборку CT2 и логи
  entrypoint (`LD_LIBRARY_PATH`, cublas/cudnn), зафиксировать в
  `docs/spike-runtime.md`.
- **503 Retry-After сразу после старта** — веса large-v3 (~3 ГБ) грузятся,
  это штатно.
- **413 при загрузке** — увеличьте `MAX_UPLOAD_MB` в `server/.env`.
- **Латентность выше ожиданий по Wi-Fi** — замерьте по Ethernet (протокол в
  `client/README.md`).
