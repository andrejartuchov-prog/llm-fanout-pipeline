"""RED spec for JSON enforcement (maps to client screening Q2: enforcing JSON
from a model with no native response_format)."""
import pytest

from llm_fanout import extract_json, JSONExtractError


def test_plain_json():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_json_with_markdown_fences():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nthanks!'
    assert extract_json(text) == {"a": 1}


def test_json_with_leading_prose():
    text = 'Sure — the result is {"score": 9, "ok": true} as requested.'
    assert extract_json(text) == {"score": 9, "ok": True}


def test_assistant_prefill_brace():
    # model was prefilled with '{' so it continues without the opening brace
    text = '"verdict": "pass", "n": 3}'
    assert extract_json(text) == {"verdict": "pass", "n": 3}


def test_nested_object():
    text = '{"outer": {"inner": [1, 2, 3]}, "k": null}'
    assert extract_json(text) == {"outer": {"inner": [1, 2, 3]}, "k": None}


def test_invalid_raises():
    with pytest.raises(JSONExtractError):
        extract_json("totally not json, no braces at all")


def test_empty_raises():
    with pytest.raises(JSONExtractError):
        extract_json("")
