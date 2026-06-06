"""llm-fanout-pipeline: parallel multi-model LLM fan-out with strict JSON
enforcement, retry-repair, and fallback aggregation.

Public API:
    from llm_fanout import extract_json, call_with_repair, fanout
    from llm_fanout.providers import Provider, AnthropicProvider
"""
from .json_enforce import extract_json, JSONExtractError
from .engine import call_with_repair, fanout, FALLBACK_STATUS

__all__ = [
    "extract_json",
    "JSONExtractError",
    "call_with_repair",
    "fanout",
    "FALLBACK_STATUS",
]
