"""Интеграционные проверки API против локального прогона (модель tiny, CPU)."""

from __future__ import annotations

import threading
import time

from .conftest import AUTH, silence, tone_bursts, wav_bytes


def test_health_ready_after_startup(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["version"]
    assert body["backend"]["runtime"] == "faster-whisper"


def test_health_shows_loading_when_not_ready(client, monkeypatch):
    holder = client.app.state.holder
    monkeypatch.setattr(holder, "status", "loading")
    body = client.get("/health").json()
    assert body["status"] == "loading"


def test_transcribe_503_while_loading(client, monkeypatch):
    holder = client.app.state.holder
    monkeypatch.setattr(holder, "status", "loading")
    files = {"file": ("a.wav", wav_bytes(silence(0.5)), "audio/wav")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 503
    assert "Retry-After" in r.headers
    assert "error" in r.json()


def test_auth_missing_and_wrong_token(client):
    files = {"file": ("a.wav", wav_bytes(silence(0.5)), "audio/wav")}
    r1 = client.post("/v1/audio/transcriptions", files=files)
    assert r1.status_code == 401
    r2 = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer wrong"},
        files=files,
    )
    assert r2.status_code == 401
    assert "error" in r2.json()


def test_auth_disabled_accepts_anonymous(client, monkeypatch):
    """AUTH_REQUIRED=0: запросы без заголовка принимаются (доверенная LAN)."""
    settings = client.app.state.settings
    monkeypatch.setattr(settings, "auth_required", False)
    files = {"file": ("s.wav", wav_bytes(tone_bursts(1.0)), "audio/wav")}
    r = client.post("/audio/transcriptions", files=files)
    assert r.status_code == 200
    assert "text" in r.json()
    # алиас /models тоже открыт
    assert client.get("/models").status_code == 200
    # а /health по-прежнему открыт и отражает режим
    assert client.get("/health").json()["auth_required"] is False


def test_transcribe_happy_path_defaults_ru(client):
    files = {"file": ("speech.wav", wav_bytes(tone_bursts(2.0)), "audio/wav")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"text", "language", "duration"}
    assert body["language"] == "ru"
    assert isinstance(body["text"], str)


def test_model_param_is_accepted_but_ignored(client):
    files = {"file": ("s.wav", wav_bytes(tone_bursts(1.5)), "audio/wav")}
    data = {"model": "whisper-unknown-999", "language": ""}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files, data=data)
    assert r.status_code == 200
    holder = client.app.state.holder
    assert holder.load_count == 1, "переключение модели не должно грузить веса"


def test_alias_route_without_v1_prefix(client):
    """Контракт OpenWhispr Self-Hosted: POST {Server URL}/audio/transcriptions."""
    files = {"file": ("s.wav", wav_bytes(tone_bursts(1.0)), "audio/wav")}
    r = client.post("/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 200
    assert "text" in r.json()


def test_alias_route_requires_token(client):
    files = {"file": ("a.wav", wav_bytes(silence(0.5)), "audio/wav")}
    r = client.post("/audio/transcriptions", files=files)
    assert r.status_code == 401


def test_models_alias_without_v1_prefix(client):
    r = client.get("/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["object"] == "list"
    r_noauth = client.get("/models")
    assert r_noauth.status_code == 401


def test_language_override_en(client):
    files = {"file": ("s.wav", wav_bytes(tone_bursts(1.5)), "audio/wav")}
    data = {"language": "en"}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files, data=data)
    assert r.status_code == 200
    assert r.json()["language"] == "en"


def test_silent_audio_returns_empty_text(client):
    files = {"file": ("silence.wav", wav_bytes(silence(3.0)), "audio/wav")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 200
    assert r.json()["text"] == ""


def test_unsupported_format_txt_rejected(client):
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 400
    assert "формат" in r.json()["error"]["message"]


def test_empty_file_rejected(client):
    files = {"file": ("empty.wav", b"", "audio/wav")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 400


def test_corrupted_wav_rejected_as_decode_error(client):
    files = {"file": ("bad.wav", b"RIFFnotreallyawavfile" * 10, "audio/wav")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 400
    assert "декодир" in r.json()["error"]["message"]


def test_malformed_multipart_returns_400(client):
    r = client.post(
        "/v1/audio/transcriptions",
        headers={**AUTH, "Content-Type": "multipart/form-data; boundary=zzz"},
        content=b"--zzz\r\nbroken\r\n",
    )
    assert r.status_code == 400


def test_oversize_upload_413(client, monkeypatch):
    settings = client.app.state.settings
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    big = wav_bytes(silence(3.0)) + b"\0" * (2 * 1024 * 1024)
    files = {"file": ("big.wav", big, "audio/wav")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 413
    assert "лимит" in r.json()["error"]["message"]


def test_raw_pcm_s16le_mono_16k_accepted(client):
    pcm = (silence(1.0) * 32767).astype("<i2").tobytes()
    files = {"file": ("raw.pcm", pcm, "application/octet-stream")}
    r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
    assert r.status_code == 200
    assert r.json()["text"] == ""


def test_concurrent_requests_are_serialized(client, monkeypatch):
    from app.transcribe import ModelHolder

    events: list[tuple[str, float]] = []
    real = ModelHolder.transcribe

    def slow(self, audio, language, initial_prompt):
        events.append(("enter", time.monotonic()))
        time.sleep(0.4)
        result = real(self, audio, language, initial_prompt)
        events.append(("exit", time.monotonic()))
        return result

    monkeypatch.setattr(ModelHolder, "transcribe", slow)

    results: list[int] = []

    def post():
        files = {"file": ("c.wav", wav_bytes(tone_bursts(0.7)), "audio/wav")}
        r = client.post("/v1/audio/transcriptions", headers=AUTH, files=files)
        results.append(r.status_code)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [200, 200]
    enters = [ts for kind, ts in events if kind == "enter"]
    exits = [ts for kind, ts in events if kind == "exit"]
    assert len(enters) == 2 and len(exits) == 2
    first_exit = min(exits)
    second_enter = max(enters)
    assert second_enter >= first_exit - 0.01, "инференс не должен пересекаться"
