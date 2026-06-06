"""Orchestration engine: retry-repair around a single provider, and parallel
fan-out + aggregation across many providers.

Maps 1:1 to the Make.com scenario this proves out:
  - call_with_repair  ↔  HTTP module + Parse JSON + two-retry error handler + fallback
  - fanout            ↔  Router (N routes) + Array Aggregator waiting for all routes

NOTE: implementation intentionally stubbed (RED). Driven to GREEN against the
immutable test suite in tests/test_engine.py.
"""
from __future__ import annotations
from typing import Sequence

from .providers import Provider

FALLBACK_STATUS = "fallback"


def call_with_repair(
    provider: Provider,
    system: str,
    user: str,
    *,
    max_retries: int = 2,
) -> dict:
    """Call `provider`, parse JSON, and self-repair on invalid output.

    On invalid JSON, re-call up to `max_retries` times, each time appending a
    stricter instruction ("Your previous output was not valid JSON. Return ONLY
    the JSON object."). If still invalid after the retries are exhausted, return
    a fallback object {"status": FALLBACK_STATUS, "model": provider.name,
    "error": "invalid_json"} instead of raising — so a single bad model never
    fails the whole report.
    """
    raise NotImplementedError


def fanout(
    providers: Sequence[Provider],
    system: str,
    user: str,
    *,
    max_retries: int = 2,
) -> dict:
    """Run `call_with_repair` across all providers concurrently and aggregate.

    Returns {provider.name: result_dict}. A provider that errors or never
    yields valid JSON contributes its fallback object; fan-out never raises.
    """
    raise NotImplementedError
