"""RED spec for the orchestration engine: retry-repair (client Q3) and parallel
fan-out + aggregation (client Q1)."""
import time

from llm_fanout import call_with_repair, fanout, FALLBACK_STATUS
from conftest import StubProvider


# ---- call_with_repair: retry → repair → fallback (Q3) ----

def test_succeeds_first_try():
    p = StubProvider("m", ['{"score": 5}'])
    assert call_with_repair(p, "sys", "user") == {"score": 5}
    assert len(p.calls) == 1


def test_retries_then_succeeds():
    p = StubProvider("m", ["not json", "still broken", '{"ok": true}'])
    result = call_with_repair(p, "sys", "user", max_retries=2)
    assert result == {"ok": True}
    assert len(p.calls) == 3  # initial + 2 retries
    # retry must add a stricter instruction referencing the previous output
    retry_text = (p.calls[1][0] + p.calls[1][1]).lower()
    assert "previous" in retry_text


def test_exhausts_to_fallback():
    p = StubProvider("gemini", ["nope", "nope", "nope"])
    result = call_with_repair(p, "sys", "user", max_retries=2)
    assert result == {"status": FALLBACK_STATUS, "model": "gemini", "error": "invalid_json"}
    assert len(p.calls) == 3  # initial + 2 retries, no more


def test_respects_max_retries_zero():
    p = StubProvider("m", ["bad", '{"ok": 1}'])
    result = call_with_repair(p, "sys", "user", max_retries=0)
    assert result["status"] == FALLBACK_STATUS
    assert len(p.calls) == 1


# ---- fanout: aggregate across providers, isolate failures (Q1) ----

def test_fanout_aggregates_all():
    ps = [
        StubProvider("openai", ['{"v": 1}']),
        StubProvider("anthropic", ['{"v": 2}']),
        StubProvider("perplexity", ['{"v": 3}']),
    ]
    out = fanout(ps, "sys", "user")
    assert out == {"openai": {"v": 1}, "anthropic": {"v": 2}, "perplexity": {"v": 3}}


def test_fanout_isolates_one_failure():
    ps = [
        StubProvider("openai", ['{"v": 1}']),
        StubProvider("bad", ["x", "x", "x"]),
        StubProvider("perplexity", ['{"v": 3}']),
    ]
    out = fanout(ps, "sys", "user", max_retries=2)
    assert set(out) == {"openai", "bad", "perplexity"}
    assert out["openai"] == {"v": 1}
    assert out["perplexity"] == {"v": 3}
    assert out["bad"]["status"] == FALLBACK_STATUS  # one bad model doesn't sink the report


def test_fanout_runs_concurrently():
    class SlowStub(StubProvider):
        def call(self, system, user, prefill="{"):
            time.sleep(0.2)
            return super().call(system, user, prefill)

    ps = [SlowStub(f"m{i}", ['{"v": 1}']) for i in range(4)]
    t0 = time.monotonic()
    out = fanout(ps, "sys", "user")
    elapsed = time.monotonic() - t0
    assert len(out) == 4
    # 4 × 0.2s sequential = 0.8s; concurrent should be well under half that
    assert elapsed < 0.45, f"fan-out not concurrent (took {elapsed:.2f}s)"
