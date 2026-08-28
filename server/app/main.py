"""STT-сервис на faster-whisper: батчевый OpenAI-совместимый эндпоинт."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import require_token
from .config import VERSION, Settings
from .transcribe import (
    AudioError,
    ModelHolder,
    load_audio,
    validate_upload,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("transcribe.api")


def _error(message: str, status_code: int, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message}},
        headers=headers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    holder = ModelHolder(settings)
    app.state.holder = holder
    holder.start_loading()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="local-transcribation-service", version=VERSION,
                  lifespan=lifespan)
    app.state.settings = settings or Settings()
    if not app.state.settings.token:
        log.warning("TRANSCRIBE_TOKEN пуст: все запросы будут отклоняться с 401")

    @app.exception_handler(RequestValidationError)
    async def malformed_request(_: Request, __: RequestValidationError):
        return _error("битый multipart-запрос", 400)

    @app.exception_handler(HTTPException)
    async def http_exception(_: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else (
            exc.detail or {}
        )
        message = detail.get("message") if isinstance(detail, dict) else detail
        return _error(message or "ошибка запроса", exc.status_code)

    @app.get("/health")
    async def health(request: Request):
        holder: ModelHolder = request.app.state.holder
        body = {
            "status": holder.status,
            "model": holder.settings.whisper_model,
            "version": VERSION,
            "auth_required": holder.settings.auth_required,
            "backend": {
                "runtime": "faster-whisper",
                "device": holder.device,
                "compute_type": holder.compute_type,
            },
        }
        if holder.error:
            body["error"] = holder.error
        return JSONResponse(content=body)

    # Канонический путь OpenAI Audio API плюс алиас без /v1: контракт Self-Hosted
    # провайдера OpenWhispr — POST {Server URL}/audio/transcriptions (клиент сам
    # не добавляет /v1), поэтому сервер принимает оба варианта пути.
    @app.post("/v1/audio/transcriptions", dependencies=[Depends(require_token)])
    @app.post("/audio/transcriptions", dependencies=[Depends(require_token)])
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(""),
        language: str = Form(""),
        prompt: str = Form(""),
    ):
        settings: Settings = request.app.state.settings
        holder: ModelHolder = request.app.state.holder

        if holder.status != "ready":
            headers = {"Retry-After": "5"} if holder.status == "loading" else None
            message = {
                "loading": "модель ещё загружается, повторите позже",
                "error": f"модель не загрузилась: {holder.error}",
            }[holder.status]
            return _error(message, 503, headers)

        requested = (language or "").strip().lower()
        if requested and (len(requested) < 2 or len(requested) > 8):
            return _error("некорректный параметр language", 400)
        # "" и "auto" означают автораспознавание языка; иначе — дефолт из конфига.
        effective_language = requested or settings.default_language.strip().lower()
        decode_language = None if effective_language in ("auto", "") else effective_language

        length_header = request.headers.get("content-length")
        if length_header and length_header.isdigit():
            if int(length_header) > settings.max_upload_bytes + 2048:
                return _error(
                    f"файл больше лимита {settings.max_upload_mb} МБ", 413
                )

        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            return _error(f"файл больше лимита {settings.max_upload_mb} МБ", 413)

        if not validate_upload(file.filename or "", file.content_type or ""):
            return _error(
                "неподдерживаемый формат: отправьте аудиофайл "
                "(wav/mp3/flac/m4a/aac/ogg/opus/webm) или сырой PCM (.pcm)", 400
            )

        try:
            audio = load_audio(data, file.filename or "", file.content_type or "")
        except AudioError as exc:
            return _error(str(exc), 400)

        effective_prompt = prompt if prompt else settings.initial_prompt
        text, detected, duration = holder.transcribe(
            audio, decode_language, effective_prompt
        )
        return {
            "text": text,
            "language": detected,
            "duration": round(duration, 3),
        }

    @app.get("/v1/models", dependencies=[Depends(require_token)])
    @app.get("/models", dependencies=[Depends(require_token)])
    async def list_models(request: Request):
        """Совместимость с клиентами протокола OpenAI: единственная резидентная модель.

        Алиас /models — для клиентов, которые запрашивают список без префикса /v1.
        """
        settings: Settings = request.app.state.settings
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.whisper_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    return app


app = create_app()
