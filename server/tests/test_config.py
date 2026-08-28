"""Проверки конфигурации: дефолты резидентной модели и языка, env-переопределения."""

from __future__ import annotations

from app.config import DEFAULT_LANG, DEFAULT_MODEL_ID, Settings


def test_default_model_is_breeze(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert Settings().whisper_model == DEFAULT_MODEL_ID
    assert DEFAULT_MODEL_ID == "SoybeanMilk/faster-whisper-Breeze-ASR-25"


def test_default_language_is_auto(monkeypatch):
    monkeypatch.delenv("DEFAULT_LANGUAGE", raising=False)
    s = Settings()
    assert s.default_language == DEFAULT_LANG == "auto"


def test_env_overrides_model_and_language(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "en")
    s = Settings()
    assert s.whisper_model == "tiny"
    assert s.default_language == "en"
