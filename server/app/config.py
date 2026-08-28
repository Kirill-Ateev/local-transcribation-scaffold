"""Конфигурация сервиса из переменных окружения и server/.env.

Порядок приоритета: реальные переменные окружения > server/.env > дефолты.
Файл .env подхватывается автоматически, чтобы сервис запускался одной
командой после `cp .env.example .env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _load_env_file(path: Path) -> None:
    """Мини-парсер KEY=VALUE (комментарии, export, кавычки). Не переопределяет env."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(_ENV_FILE)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _require_token() -> str:
    token = os.environ.get("TRANSCRIBE_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TRANSCRIBE_TOKEN не задан. Скопируйте server/.env.example в server/.env "
            "и заполните токен (openssl rand -hex 32), либо экспортируйте переменную."
        )
    return token


# Единственная резидентная модель: CT2-конверсия MediaTek Breeze-ASR-25
# (fine-tune Whisper large-v2, оптимизация под традиционный китайский + английский).
DEFAULT_MODEL_ID = "SoybeanMilk/faster-whisper-Breeze-ASR-25"
# Язык по умолчанию: "auto" — автораспознавание на каждый запрос.
DEFAULT_LANG = "auto"


@dataclass
class Settings:
    token: str = field(default_factory=_require_token)
    whisper_model: str = field(
        default_factory=lambda: os.environ.get("WHISPER_MODEL", DEFAULT_MODEL_ID)
    )
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "auto"))
    compute_type: str = field(
        default_factory=lambda: os.environ.get("COMPUTE_TYPE", "auto")
    )
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int_env("PORT", 8337))
    max_upload_mb: int = field(
        default_factory=lambda: _int_env("MAX_UPLOAD_MB", 512)
    )
    default_language: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_LANGUAGE", DEFAULT_LANG)
    )
    initial_prompt: str = field(
        default_factory=lambda: os.environ.get("INITIAL_PROMPT", "")
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


VERSION = "0.1.0"
