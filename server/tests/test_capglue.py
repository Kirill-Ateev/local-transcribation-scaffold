"""Юнит-тесты capglue: починка склеек Breeze без повреждения IT-лексики."""

from __future__ import annotations

from app.postprocess import apply_capglue


def test_inserts_period_between_glued_cyrillic_words():
    assert apply_capglue("кулинариейПочему") == "кулинарией. Почему"


def test_glue_after_latin_term():
    assert apply_capglue("задеплоил featureСегодня") == "задеплоил feature. Сегодня"


def test_glue_into_latin_camelcase_sentence_start():
    assert apply_capglue("закончилиCode review") == "закончили. Code review"


def test_inserts_space_after_period():
    assert apply_capglue("сделал backup.Недавно начал заново") == \
        "сделал backup. Недавно начал заново"


def test_restores_capital_after_period():
    assert apply_capglue("…кнопку. дальше я…") == "…кнопку. Дальше я…"


def test_restores_capital_after_question_mark():
    assert apply_capglue("ты идёшь? потом скажу") == "ты идёшь? Потом скажу"


def test_camelcase_identifiers_untouched():
    text = "используй useState и getUserData, но не parseFloat"
    assert apply_capglue(text) == text


def test_domains_extensions_numbers_untouched():
    text = "залей на example.com в app.py, версия 3.14"
    assert apply_capglue(text) == text


def test_ellipsis_not_treated_as_sentence_end():
    assert apply_capglue("так... вот так") == "так... вот так"


def test_abbreviations_still_get_capital():
    # известный компромисс: после «т.д.» регистр поднимается
    assert apply_capglue("и т.д. потом") == "и т.д. Потом"


def test_chinese_punctuation_untouched():
    text = "今天天氣很好。我們去爬山，好嗎？"
    assert apply_capglue(text) == text


def test_empty_string_noop():
    assert apply_capglue("") == ""


def test_idempotent():
    once = apply_capglue("конец.Начало следующего")
    assert apply_capglue(once) == once
