"""Orchestration engine: retry-repair around a single provider, and parallel
fan-out + aggregation across many providers.

Maps 1:1 to the Make.com scenario this proves out:
  - call_with_repair  ↔  HTTP module + Parse JSON + two-retry error handler + fallback
  - fanout            ↔  Router (N routes) + Array Aggregator waiting for all routes
"""
from __future__ import annotations

import concurrent.futures
from typing import Sequence

from .json_enforce import JSONExtractError, extract_json
from .providers import Provider

FALLBACK_STATUS = "fallback"


def _fallback(name: str) -> dict:
    """Aggregation-safe placeholder for a provider that never returned valid JSON."""
    return {"status": FALLBACK_STATUS, "model": name, "error": "invalid_json"}


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
    repair_user = user
    for _ in range(max_retries + 1):
        raw = provider.call(system, repair_user)
        try:
            return extract_json(raw)
        except JSONExtractError:
            # Escalate: quote the rejected output so the model can correct it.
            repair_user = (
                f"{user}\n\nYour previous output was not valid JSON:\n{raw}\n"
                "Return ONLY a single valid JSON object and nothing else."
            )
    return _fallback(provider.name)


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
    if not providers:
        return {}

    results: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as executor:
        future_to_provider = {
            executor.submit(call_with_repair, p, system, user, max_retries=max_retries): p
            for p in providers
        }
        for future in concurrent.futures.as_completed(future_to_provider):
            provider = future_to_provider[future]
            try:
                results[provider.name] = future.result()
            except Exception:
                # A crashing provider must not sink the whole report.
                results[provider.name] = _fallback(provider.name)
    return results
