"""Strict JSON extraction from LLM text output.

Claude (and most chat LLMs) have no native `response_format` JSON mode, so the
model output may contain prose, markdown ```json fences, or a leading assistant
prefill brace. `extract_json` recovers the intended JSON object defensively.

NOTE: implementation intentionally stubbed (RED). Driven to GREEN against the
immutable test suite in tests/test_json_enforce.py.
"""


class JSONExtractError(ValueError):
    """Raised when no valid JSON object can be recovered from model text."""


def extract_json(text: str) -> dict:
    """Recover a JSON object from raw model text.

    Must handle: plain JSON, ```json fenced blocks, leading/trailing prose,
    and an assistant-prefill leading '{'. Raises JSONExtractError if nothing
    parseable is found.
    """
    raise NotImplementedError
