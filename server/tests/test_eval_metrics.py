"""Юнит-проверки метрик eval/score.py."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from score import cer, find_audio, wer  # noqa: E402


def test_wer_identical():
    assert wer("Открой деплой в Cursor", "открой деплой  в cursor!") == 0.0
    assert wer("Ёлка, ёж!", "елка еж") == 0.0


def test_wer_partial():
    # 5 слов, одно лишнее и одно неверное -> расстояние 2 из 5
    ref = "а б в г д"
    hyp = "а б x г"
    assert wer(ref, hyp) == 0.4


def test_wer_empty_hypothesis():
    assert wer("раз два три", "") == 1.0


def test_cer_counts_chars():
    assert cer("тест", "текст") > 0
    assert cer("тест", "тест") == 0.0


def test_find_audio_prefers_existing(tmp_path):
    (tmp_path / "ru_01.mp3").write_bytes(b"x")
    found = find_audio(tmp_path, "ru_01")
    assert found is not None and found.suffix == ".mp3"
    assert find_audio(tmp_path, "nope") is None
