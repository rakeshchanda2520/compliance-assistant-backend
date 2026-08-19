"""
Optional Langfuse tracing.

Enabled only when both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
(`config.TRACING_ENABLED`) — the same shape as `llm.py`'s provider selection:
real behaviour when configured, zero dependency or behaviour cost when not.
Nothing else in this codebase imports `langfuse` directly; every call goes
through `trace()` / `step()` here, so the rest of the system never has to
know whether tracing is on.

One request becomes one root trace with three nested observations —
`retrieve` (as_type="retriever"), `generate` (as_type="generation"),
`verify_citations` (as_type="evaluator") — deliberately mirroring the three
SSE stages `app.py` already streams and the three fields `audit.py` already
writes to `logs/audit.jsonl`. A trace in the Langfuse UI, the browser's
network tab, and a line in the local audit log all describe the same request
the same way.

Verified resilient to a misconfigured or unreachable Langfuse endpoint: a
failed span export is logged as a warning by the SDK and the request still
completes normally. Tracing is an assurance layer, never a dependency of the
actual answer — that must hold even if this module is wrong or Langfuse is
down.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager

from . import config

log = logging.getLogger(__name__)

_client = None


def _get_client():
    """Imported lazily so a process that never configures tracing never pays
    for importing `langfuse` (and its opentelemetry dependency chain) —
    mirrors `llm.py`'s lazy `import anthropic` inside `ClaudeProvider`."""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    return _client


def check() -> str | None:
    """Return an error string if tracing is configured but unreachable or
    misauthenticated, else None. Mirrors `llm.check()`'s contract so
    `/api/health` can report both the same way."""
    if not config.TRACING_ENABLED:
        return None
    try:
        if not _get_client().auth_check():
            return "Langfuse credentials are set but authentication failed"
    except Exception as exc:                   # noqa: BLE001
        return f"Langfuse unreachable: {type(exc).__name__}"
    return None


@contextmanager
def _observation(name: str, as_type: str, **fields):
    """Shared implementation for `trace()` and `step()`.

    Exception scoping here is deliberate and easy to get wrong: only a
    failure to CREATE the Langfuse observation degrades to a no-op. Once the
    caller's block is running, any exception it raises must propagate
    untouched — a naive `try: ... yield ... except Exception:` around the
    whole thing would also catch exceptions raised by the CALLER's own code
    (a `@contextmanager` receives the caller's exception at the `yield`
    point), silently swallowing a real application bug and misreporting it
    as "tracing unavailable". That would be worse than having no tracing at
    all: an error a developer needed to see would just vanish.
    """
    if not config.TRACING_ENABLED:
        yield None
        return

    try:
        manager = _get_client().start_as_current_observation(
            name=name, as_type=as_type, **fields)
        observation = manager.__enter__()
    except Exception:                            # noqa: BLE001
        log.warning("tracing unavailable for this request", exc_info=True)
        yield None
        return

    try:
        yield observation
    except BaseException:
        # Let Langfuse record the failure on the span, then re-raise the
        # caller's real exception unchanged — never swallowed here.
        manager.__exit__(*sys.exc_info())
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:                        # noqa: BLE001
            log.warning("tracing span close failed", exc_info=True)


@contextmanager
def trace(name: str, *, user_id: str = "", session_id: str = "", **fields):
    """Root span for one request. A no-op (yields None) when tracing is off,
    so callers unconditionally write `with observability.trace(...) as t:`
    and guard direct use of `t` behind `if t:` — never behind a separate
    `if config.TRACING_ENABLED:` scattered through the caller.

    `user_id`/`session_id` are Langfuse's own first-class trace attributes —
    setting them is what turns the Langfuse UI's Users/Sessions views on,
    and is what lets "every trace for this user" or "every trace in this
    sign-in session" be filtered without grepping metadata. Applied via
    `propagate_attributes()` so every child `step()` opened inside this
    `with` block inherits them automatically, rather than every call site
    threading the same two values through by hand.
    """
    with _observation(name, "span", **fields) as root:
        if not config.TRACING_ENABLED or root is None or not (user_id or session_id):
            yield root
            return
        from langfuse import propagate_attributes
        with propagate_attributes(user_id=user_id or None, session_id=session_id or None):
            yield root


def step(name: str, as_type: str = "span", **fields):
    """A child observation nested under whichever trace/step is currently
    open. Also a no-op when tracing is off, and equally tolerant of a
    tracing failure mid-request — an answer must never fail because Langfuse
    did.

    as_type: "span" (default), "generation", "retriever", "evaluator",
    "embedding", "agent", "tool", "chain", "guardrail". The specific type is
    not cosmetic — Langfuse renders each differently, and "generation" is
    what enables token-usage/cost tracking if usage details are ever passed
    to `.update()`.
    """
    return _observation(name, as_type, **fields)


def flush() -> None:
    """Langfuse batches events and sends them asynchronously. A short-lived
    process would lose buffered events on exit; a long-lived server like
    this one doesn't strictly need this per request, but calling it keeps
    trace latency predictable rather than deferred, at negligible cost."""
    if config.TRACING_ENABLED and _client is not None:
        try:
            _client.flush()
        except Exception:                       # noqa: BLE001
            log.warning("tracing flush failed", exc_info=True)
