"""Предзагрузка весов резидентной модели в кэш HuggingFace.

Чтобы старт сервиса не зависел от сети, веса можно скачать заранее: скрипт
работает и на хосте, и внутри контейнера (compose-сервис `download-model`),
складывает файлы в общий кэш (HF_HOME), откуда faster-whisper их подхватит.

Использование:
    python3 server/scripts/download_model.py                 # модель из WHISPER_MODEL
    python3 server/scripts/download_model.py --model <hf-id>
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def main() -> int:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from app.config import DEFAULT_MODEL_ID

    parser = argparse.ArgumentParser(description="Предзагрузка весов модели в кэш HF")
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", DEFAULT_MODEL_ID),
        help=f"HF id или локальный путь (по умолчанию {DEFAULT_MODEL_ID})",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub не установлен: pip install -r server/requirements.txt",
            file=sys.stderr,
        )
        return 1

    cache = os.environ.get("HF_HOME", "~/.cache/huggingface")
    print(f"загрузка {args.model} в кэш HuggingFace ({cache})...")
    path = snapshot_download(args.model)

    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    print(f"готово: {path} ({total / 1e9:.2f} ГБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
