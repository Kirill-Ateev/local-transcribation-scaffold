"""Проверки конфигурации: дефолты резидентной модели и языка, env-переопределения."""

from __future__ import annotations

from app.config import (
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_LANG,
    DEFAULT_MODEL_ID,
    Settings,
)


def test_default_model_is_breeze(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert Settings().whisper_model == DEFAULT_MODEL_ID
    assert DEFAULT_MODEL_ID == "SoybeanMilk/faster-whisper-Breeze-ASR-25"


def test_default_language_is_auto(monkeypatch):
    monkeypatch.delenv("DEFAULT_LANGUAGE", raising=False)
    s = Settings()
    assert s.default_language == DEFAULT_LANG == "auto"


def test_default_prompt_is_promptv3(monkeypatch):
    monkeypatch.delenv("INITIAL_PROMPT", raising=False)
    s = Settings()
    assert s.initial_prompt == DEFAULT_INITIAL_PROMPT
    assert "Preserve English in Latin" in s.initial_prompt


def test_default_capglue_enabled(monkeypatch):
    monkeypatch.delenv("CAPGLUE", raising=False)
    assert Settings().capglue is True


def test_auth_required_default_true(monkeypatch):
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    assert Settings().auth_required is True


def test_env_can_disable_auth(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "0")
    assert Settings().auth_required is False


def test_env_overrides_model_and_language(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "en")
    s = Settings()
    assert s.whisper_model == "tiny"
    assert s.default_language == "en"


def test_env_can_disable_capglue_and_prompt(monkeypatch):
    monkeypatch.setenv("CAPGLUE", "0")
    monkeypatch.setenv("INITIAL_PROMPT", "")
    s = Settings()
    assert s.capglue is False
    assert s.initial_prompt == ""
