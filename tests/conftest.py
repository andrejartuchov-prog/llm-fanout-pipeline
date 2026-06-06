"""Test fixtures. StubProvider lets us drive the engine deterministically
without network — it returns a scripted sequence of raw completions, so we can
assert retry/repair/fallback behaviour exactly.

This file is test-support and is part of the immutable spec (do not edit to make
production code pass).
"""
import pytest

from llm_fanout.providers import Provider


class StubProvider(Provider):
    """Returns scripted raw completions in order; records every call.

    `responses` is a list of raw strings the model "returns" on successive
    calls. If calls exceed the list, the last entry repeats. `calls` captures
    (system, user, prefill) tuples for assertions (e.g. stricter retry text).
    """

    def __init__(self, name, responses):
        self.name = name
        self._responses = list(responses)
        self.calls = []

    def call(self, system: str, user: str, prefill: str = "{") -> str:
        self.calls.append((system, user, prefill))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


@pytest.fixture
def valid_json_text():
    return '{"verdict": "ok", "score": 7}'
