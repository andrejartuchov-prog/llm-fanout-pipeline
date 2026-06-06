"""LLM provider abstraction — pluggable adapters so the same engine drives
OpenAI / Anthropic / Google / Perplexity (or any HTTP LLM) identically.

Only the Anthropic adapter is implemented as a real call here; the other three
are intentionally left as thin pluggable stubs (honest: this repo proves the
orchestration engine, not four vendor integrations). The engine treats every
provider through the `Provider` interface below.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class Provider(ABC):
    """A single LLM endpoint the engine can call.

    `name` identifies the model in aggregated output. `call` returns the raw
    text completion (the engine, not the provider, enforces JSON).
    """

    name: str

    @abstractmethod
    def call(self, system: str, user: str, prefill: str = "{") -> str:
        """Return raw text completion.

        `prefill` is the assistant-turn prefill used to constrain the model to
        emit JSON (for Anthropic: a trailing {role:'assistant', content:'{'}).
        """
        raise NotImplementedError


class AnthropicProvider(Provider):
    """Real Claude adapter (Messages API).

    POST https://api.anthropic.com/v1/messages with headers
    x-api-key / anthropic-version / content-type. JSON is constrained via the
    system instruction plus the assistant `prefill` brace.

    NOTE: implementation stubbed (RED) — driven to GREEN against the test suite.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-4-8", max_tokens: int = 2048):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def call(self, system: str, user: str, prefill: str = "{") -> str:
        raise NotImplementedError
