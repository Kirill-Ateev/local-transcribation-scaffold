"""Фикстуры: окружение до импорта приложения, генераторы аудио."""

from __future__ import annotations

import io
import os
import time
import wave

import numpy as np
import pytest

os.environ.setdefault("TRANSCRIBE_TOKEN", "test-token")
os.environ.setdefault("WHISPER_MODEL", "tiny")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("COMPUTE_TYPE", "int8")
os.environ.setdefault("MAX_UPLOAD_MB", "8")
os.environ.setdefault("LOG_LEVEL", "WARNING")

SR = 16_000


def wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def tone_bursts(seconds: float, freq: int = 300, sr: int = SR) -> np.ndarray:
    """Громкие тональные всплески с паузами — имитация речевой активности."""
    total = int(seconds * sr)
    out = np.zeros(total, dtype=np.float32)
    t = np.arange(total) / sr
    envelope = ((t * 2.2) % 1.0) < 0.6
    out[:] = 0.6 * np.sin(2 * np.pi * freq * t) * envelope
    return out


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.config import Settings

    app = create_app(Settings())
    with TestClient(app) as c:
        deadline = time.monotonic() + 300
        while c.get("/health").json()["status"] != "ready":
            if time.monotonic() > deadline:
                raise RuntimeError("модель не загрузилась за 300с")
            time.sleep(0.5)
        yield c


AUTH = {"Authorization": "Bearer test-token"}
