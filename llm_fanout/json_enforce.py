"""Strict JSON extraction from LLM text output.

Claude (and most chat LLMs) have no native `response_format` JSON mode, so the
model output may contain prose, markdown ```json fences, or a leading assistant
prefill brace. `extract_json` recovers the intended JSON object defensively.
"""
from __future__ import annotations

import json
import re


class JSONExtractError(ValueError):
    """Raised when no valid JSON object can be recovered from model text."""


def extract_json(text: str) -> dict:
    """Recover a JSON object from raw model text.

    Must handle: plain JSON, ```json fenced blocks, leading/trailing prose,
    and an assistant-prefill leading '{'. Raises JSONExtractError if nothing
    parseable is found.
    """
    stripped = text.strip()
    if not stripped:
        raise JSONExtractError("cannot extract JSON from empty text")

    # Ordered candidates, cheapest first; the first that parses to a dict wins.
    # A bare object parses directly; a fenced or prose-wrapped blob needs
    # slicing; a prefilled completion has lost its opening brace.
    candidates = [stripped]

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    first, last = stripped.find("{"), stripped.rfind("}")
    if 0 <= first < last:
        candidates.append(stripped[first : last + 1])

    if not stripped.startswith("{"):
        # Assistant turn was prefilled with '{', so the completion dropped it.
        candidates.append("{" + stripped)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise JSONExtractError(f"no valid JSON object found in: {text!r}")
