"""Локальный прокси-тест задачи 3.2: часовое аудио одним синхронным запросом.

Генерирует запись ~61 минуты (тишина с редкими тональными всплесками),
кодирует в MP3 (при наличии кодека), отправляет одним POST на поднятый
сервис и проверяет: полный ответ, длительность, стабильность памяти процесса
(VmRSS/VmHWM из /proc до и после запроса).

Запуск: .venv/bin/python server/scripts/test_hour_long.py [--seconds 3660]
"""

from __future__ import annotations

import argparse
import io
import os
import signal
import subprocess
import sys
import time
import urllib.request
import json as jsonlib

import httpx
import numpy as np

SR = 16_000
TOKEN = "hour-test-token"
PORT = 8399


def synth(seconds: float) -> np.ndarray:
    total = int(seconds * SR)
    out = np.zeros(total, dtype=np.float32)
    rng = np.random.default_rng(7)
    burst = np.zeros(int(0.8 * SR))
    t = np.arange(burst.size) / SR
    burst[:] = 0.7 * np.sin(2 * np.pi * 220 * t)
    ramp = np.linspace(0, 1, SR // 100)
    burst[: SR // 100] *= ramp
    burst[-SR // 100:] *= ramp[::-1]
    pos = 0
    while pos + burst.size < total:
        out[pos : pos + burst.size] = burst
        out[pos : pos + burst.size] += (
            0.05 * rng.standard_normal(burst.size).astype(np.float32)
        )
        pos += 300 * SR
    return out


def to_mp3(samples: np.ndarray) -> tuple[bytes, str]:
    import av

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp3")
    stream = container.add_stream("mp3", rate=SR)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
    frame.sample_rate = SR
    for chunk in range(0, pcm.size, SR):
        f = av.AudioFrame.from_ndarray(
            pcm[None, chunk : chunk + SR], format="s16", layout="mono"
        )
        f.sample_rate = SR
        f.pts = None
        for packet in stream.encode(f):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue(), "mp3"


def read_status(pid: int) -> dict:
    fields = {}
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, _unit = line.split()
                fields[name.rstrip(":")] = int(value)
    return fields


def wait_ready(base: str, deadline_s: int = 300) -> dict:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                body = jsonlib.load(r)
                if body["status"] == "ready":
                    return body
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("сервис не стал ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3660)
    args = ap.parse_args()

    print(f"[1/4] генерация {args.seconds / 60:.0f} мин аудио...")
    samples = synth(args.seconds)
    payload, codec = to_mp3(samples)
    print(f"      формат={codec} размер={len(payload) / 1e6:.1f} МБ")

    env = dict(os.environ,
               TRANSCRIBE_TOKEN=TOKEN, WHISPER_MODEL="tiny",
               DEVICE="cpu", COMPUTE_TYPE="int8",
               MAX_UPLOAD_MB="200", LOG_LEVEL="WARNING", HOST="127.0.0.1",
               PORT=str(PORT))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "warning"],
        cwd="server", env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{PORT}"
    try:
        health = wait_ready(base)
        print(f"[2/4] сервис готов: {health['backend']}")

        status_before = read_status(proc.pid)
        started = time.monotonic()
        r = httpx.post(
            f"{base}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"file": ("long.mp3", payload, "audio/mpeg")},
            data={"language": "ru"},
            timeout=900,
        )
        elapsed = time.monotonic() - started
        status_after = read_status(proc.pid)

        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
        body = r.json()
        dur = body["duration"]
        assert abs(dur - args.seconds) < 10, f"длительность потеряна: {dur}"

        rss_delta_mb = (status_after["VmRSS"] - status_before["VmRSS"]) / 1024
        hwm_mb = status_after["VmHWM"] / 1024
        print("[3/4] результат:")
        print(f"      HTTP 200 за {elapsed:.1f}s, duration ответа = {dur:.0f}s")
        print(f"      RSS delta за запрос: {rss_delta_mb:+.0f} МБ, VmHWM: {hwm_mb:.0f} МБ")
        assert hwm_mb < 3072, f"HWM слишком велик: {hwm_mb:.0f} МБ"
        assert abs(rss_delta_mb) < 1500, f"память выросла на запрос: {rss_delta_mb:+.0f} МБ"

        print("[4/4] OK: часовое аудио обработано одним запросом, память ограничена")
        return 0
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
