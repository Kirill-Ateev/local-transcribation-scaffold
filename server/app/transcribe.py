"""Резидентная модель, сериализация инференса, конвейер транскрипции."""

from __future__ import annotations

import io
import logging
import threading
import time

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from .config import Settings

log = logging.getLogger("transcribe.service")

# Допустимые входные форматы: контейнеры, декодируемые PyAV, плюс сырой PCM.
CONTAINER_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".webm",
}
RAW_PCM_EXTENSIONS = {".pcm"}
AUDIO_CONTENT_PREFIX = "audio/"


class AudioError(ValueError):
    """Некорректный или неподдерживаемый аудиовход (HTTP 400)."""


class ModelHolder:
    """Владеет единственной моделью; статусы для /health; замок очереди."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = "loading"
        self.error: str | None = None
        self.device = settings.device
        self.compute_type = settings.compute_type
        self.load_count = 0
        self.inference_lock = threading.Lock()
        self._model: WhisperModel | None = None

    def start_loading(self) -> None:
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            started = time.monotonic()
            log.info(
                "loading model=%s device=%s compute_type=%s",
                self.settings.whisper_model, self.settings.device,
                self.settings.compute_type,
            )
            self._model = WhisperModel(
                self.settings.whisper_model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
            try:
                self.device = self._model.model.device
                self.compute_type = self._model.model.compute_type
            except AttributeError:
                pass
            self.load_count += 1
            self.status = "ready"
            log.info("model ready in %.1fs", time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 - состояние должно попасть в /health
            self.status = "error"
            self.error = str(exc)
            log.exception("model loading failed")

    def transcribe(self, audio: np.ndarray, language: str | None,
                   initial_prompt: str | None) -> tuple[str, str, float]:
        """Синхронная транскрипция под замком: запросы обрабатываются по одному."""
        assert self._model is not None
        with self.inference_lock:
            started = time.monotonic()
            segments, info = self._model.transcribe(
                audio,
                language=language,
                task="transcribe",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                    "max_speech_duration_s": 28,
                },
                initial_prompt=initial_prompt or None,
                no_speech_threshold=0.6,
            )
            parts = [segment.text for segment in segments]
            text = " ".join(part.strip() for part in parts if part.strip())
            elapsed = time.monotonic() - started
            log.info(
                "transcribed duration=%.1fs took=%.2fs lang=%s chars=%d",
                info.duration, elapsed, info.language, len(text),
            )
            return text, info.language, float(info.duration)


def validate_upload(filename: str, content_type: str) -> bool:
    """Проверяет, что тип загрузки заявлен как аудио (содержимое проверит декодер)."""
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    dot = name.rfind(".")
    ext = name[dot:] if dot != -1 else ""
    if ext in RAW_PCM_EXTENSIONS:
        return True
    if ext in CONTAINER_EXTENSIONS:
        return True
    return ctype.startswith(AUDIO_CONTENT_PREFIX)


def load_audio(data: bytes, filename: str, content_type: str) -> np.ndarray:
    """Декодирует загрузку в float32 mono 16 kHz. Сырой PCM — s16le mono 16 kHz."""
    if not data:
        raise AudioError("пустой файл")
    name = (filename or "").lower()
    ext = name[name.rfind("."):] if "." in name else ""
    if ext in RAW_PCM_EXTENSIONS or content_type == "audio/pcm":
        try:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError as exc:
            raise AudioError("битый сырой PCM: ожидается s16le mono 16 kHz") from exc
        if samples.size == 0:
            raise AudioError("пустой файл")
        return samples
    try:
        audio = decode_audio(io.BytesIO(data), sampling_rate=16000)
    except Exception as exc:  # noqa: BLE001 - любая ошибка декодирования это 400
        raise AudioError(f"не удалось декодировать аудио: {exc}") from exc
    if audio.size == 0:
        raise AudioError("в файле нет аудиоданных")
    return audio
