# llm-fanout-pipeline

Parallel **multi-model LLM fan-out** with strict **JSON enforcement**,
**retry-repair**, and **fallback aggregation** — the reliability core of an
"submission → four models → one structured report" pipeline.

It answers the three things that actually break these pipelines in production,
regardless of whether the orchestration lives in code or in a no-code tool like
Make.com:

| Concern | Where | How this repo handles it |
|---|---|---|
| **Parallel fan-out across N models, wait for all** | `engine.fanout` | Calls every provider concurrently, aggregates into one `{model: result}` object, returns when all are done. ↔ Make Router (N routes) + Array Aggregator. |
| **Strict JSON from a model with no `response_format`** | `json_enforce.extract_json` | Strips ```json fences / prose, recovers from an assistant-prefill leading `{`, parses defensively. ↔ Parse JSON module. |
| **Don't let one bad model sink the report** | `engine.call_with_repair` | On invalid JSON, retries up to N times with a *stricter* instruction, then injects a fallback object instead of raising. ↔ two-retry error handler + fallback. |

## Why these three

Most "AI pipeline" failures aren't the happy path — they're a model returning
prose instead of JSON, or one of four parallel calls timing out and breaking the
aggregation. This library isolates those failure modes and tests them
explicitly, so the pipeline degrades gracefully instead of erroring out.

## Design

```
fanout(providers, system, user)            # parallel across providers
  └─ call_with_repair(provider, ...)       # per-provider: call → parse → retry → fallback
        ├─ provider.call(system, user, prefill="{")   # raw completion
        └─ extract_json(text)              # defensive JSON recovery
```

- **`Provider`** is a thin interface (`name`, `call`). `AnthropicProvider` is a
  real Claude (Messages API) adapter; OpenAI / Google / Perplexity plug in the
  same way. Anthropic JSON is constrained via a system instruction **plus an
  assistant-turn prefill `{`**.
- **`call_with_repair`** never raises on bad output — it returns
  `{"status": "fallback", "model": <name>, "error": "invalid_json"}` after the
  retries are exhausted.
- **`fanout`** runs providers concurrently and aggregates; a single failing
  provider contributes its fallback and never sinks the whole report.

## Usage

```python
from llm_fanout import fanout
from llm_fanout.providers import AnthropicProvider

providers = [AnthropicProvider(api_key="...")]  # + OpenAI/Gemini/Perplexity adapters
report = fanout(providers, system="Return ONLY JSON ...", user="<submission>")
# -> {"anthropic": {...}, ...}
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite is the spec — it pins the retry/repair/fallback and concurrent
fan-out behaviour exactly (`tests/test_json_enforce.py`, `tests/test_engine.py`).

## License

MIT
