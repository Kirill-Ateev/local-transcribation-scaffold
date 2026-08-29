"""Резидентная модель, сериализация инференса, конвейер транскрипции."""

from __future__ import annotations

import io
import logging
import threading
import time

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from .config import Settings
from .postprocess import apply_capglue

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
        self.warmup_seconds = 0.0
        self.inference_lock = threading.Lock()
        self._model: WhisperModel | None = None

    def start_loading(self) -> None:
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            started = time.monotonic()
            cuda_count = ctranslate2.get_cuda_device_count()
            log.info(
                "loading model=%s device=%s compute_type=%s ct2=%s cuda_devices=%d",
                self.settings.whisper_model, self.settings.device,
                self.settings.compute_type, ctranslate2.__version__, cuda_count,
            )
            if self.settings.device == "cuda" and cuda_count == 0:
                raise RuntimeError(
                    "DEVICE=cuda, но CTranslate2 не видит CUDA-устройств — "
                    "проверьте проброс GPU (nvidia-smi внутри контейнера)"
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
            self._warmup()
            self.status = "ready"
            log.info("model ready in %.1fs", time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 - состояние должно попасть в /health
            self.status = "error"
            self.error = str(exc)
            log.exception("model loading failed")

    def _warmup(self) -> None:
        """Прогрев до статуса ready: CUDA-контекст, JIT-компиляция ядер под GB10,
        cuBLAS/cuDNN и инициализация Silero VAD. Без него всё это оплачивает
        первый реальный запрос (~20 с). Ошибка прогрева не валит старт."""
        assert self._model is not None
        try:
            probe = (0.1 * np.sin(2 * np.pi * 440.0 * np.arange(16000) / 16000)).astype(np.float32)
            warm_started = time.monotonic()
            with self.inference_lock:
                # vad=False: полный проход энкодер+декодер (с VAD чистый тон
                # мог бы быть отрезан как «не речь», и декодер не прогрелся бы);
                # vad=True: инициализация модели VAD.
                for vad in (False, True):
                    list(self._model.transcribe(
                        probe, language="ru", beam_size=1, temperature=0.0,
                        without_timestamps=True, vad_filter=vad,
                    ))
            self.warmup_seconds = time.monotonic() - warm_started
            log.info("warmup done in %.1fs", self.warmup_seconds)
        except Exception:  # noqa: BLE001
            log.warning("warmup failed (не критично)", exc_info=True)

    def transcribe(self, audio: np.ndarray, language: str | None,
                   initial_prompt: str | None) -> tuple[str, str, float]:
        """Синхронная транскрипция под замком: запросы обрабатываются по одному.

        Параметры декодера под латентность диктовки на DGX Spark:
        beam_size=1 — жадный поиск (дефолтный beam 5 делал декодер впятеро дороже);
        temperature=0.0 — один проход без температурных ретраев, каждый из которых
        перекодирует 30-с окно заново; для диктовки безопасно — анти-петли гасятся
        condition_on_previous_text=False + VAD + capglue. Если качество просядет —
        удалите строку temperature=0.0 (вернутся ретраи) или поднимите beam_size до 3.
        without_timestamps=True — декодер не предсказывает токены-таймстампы
        (минус 10–15% шагов декодирования), сегменты крупнее; для диктовки это неважно.
        """
        assert self._model is not None
        with self.inference_lock:
            started = time.monotonic()
            segments, info = self._model.transcribe(
                audio,
                language=language,
                task="transcribe",
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                    "max_speech_duration_s": 180,
                },
                initial_prompt=initial_prompt or None,
                no_speech_threshold=0.6,
            )
            parts: list[str] = []
            max_temp = 0.0
            speech_s = 0.0
            seg_count = 0
            for segment in segments:
                parts.append(segment.text)
                seg_count += 1
                max_temp = max(max_temp, float(segment.temperature))
                # Оценка речи после VAD: в без-таймстампном режиме сегменты —
                # это окна вокруг речевых кусков, сумма их длительностей ~ речь.
                speech_s += segment.end - segment.start
            text = " ".join(part.strip() for part in parts if part.strip())
            if self.settings.capglue:
                glued = apply_capglue(text)
                if glued != text:
                    log.info(
                        "capglue: починены стыки предложений (%d -> %d символов)",
                        len(text), len(glued),
                    )
                text = glued
            elapsed = time.monotonic() - started
            log.info(
                "transcribed duration=%.1fs speech~%.1fs took=%.2fs lang=%s "
                "seg=%d chars=%d max_temp=%.2f",
                info.duration, speech_s, elapsed, info.language,
                seg_count, len(text), max_temp,
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