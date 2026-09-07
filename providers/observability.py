"""
Optional Langfuse tracing.

On only when both keys are set. Nothing else imports `langfuse`, so a
deployment that never traces never pays for the dependency chain.

**One trace per request, one span per STAGE — every stage, not only the ones
that emit an event.** An earlier system instrumented only retrieve, generate
and verify, which produced two misleading shapes: embedding and routing ran
untracked inside the trace window (a second of unexplained dead time), and a
template answer — no model, by design, correctly — rendered as a trace
containing `retrieve` and nothing else, indistinguishable from a request that
had died.

A tracing failure must NEVER take down an answer.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager

import config

log = logging.getLogger(__name__)

_client = None


def _get():
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(public_key=config.LANGFUSE_PUBLIC_KEY,
                           secret_key=config.LANGFUSE_SECRET_KEY,
                           host=config.LANGFUSE_HOST)
    return _client


def check() -> str | None:
    if not config.TRACING:
        return None
    try:
        if not _get().auth_check():
            return "Langfuse credentials are set but authentication failed"
    except Exception as exc:                               # noqa: BLE001
        return f"Langfuse unreachable: {type(exc).__name__}"
    return None


@contextmanager
def _observation(name: str, as_type: str, **fields):
    """Only a failure to CREATE the observation degrades to a no-op.

    Once the caller's block is running, its exceptions propagate untouched. A
    naive try/except around the whole thing would also swallow the CALLER's
    exceptions — a @contextmanager receives them at the yield — and misreport
    a real application bug as "tracing unavailable", which is worse than
    having no tracing at all.
    """
    if not config.TRACING:
        yield None
        return
    try:
        manager = _get().start_as_current_observation(
            name=name, as_type=as_type, **fields)
        observation = manager.__enter__()
    except Exception:                                      # noqa: BLE001
        log.warning("tracing unavailable for this request", exc_info=True)
        yield None
        return
    try:
        yield observation
    except BaseException:
        manager.__exit__(*sys.exc_info())
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:                                  # noqa: BLE001
            log.warning("tracing span close failed", exc_info=True)


@contextmanager
def trace(name: str, *, user_id: str = "", session_id: str = "",
          tags: list[str] | None = None, metadata: dict | None = None,
          **fields):
    """The root span.

    `session_id` carries the CONVERSATION thread, not the sign-in session:
    Langfuse's Sessions view is built to show a multi-turn conversation as one
    object, and that is the grouping anyone reading traces actually wants. The
    sign-in session is coarser — one sign-in spans many conversations — and
    rides in metadata instead.
    """
    with _observation(name, "span", **fields) as root:
        if not config.TRACING or root is None:
            yield root
            return
        try:
            from langfuse import propagate_attributes
            with propagate_attributes(user_id=user_id or None,
                                      session_id=session_id or None,
                                      tags=tags or None,
                                      metadata=metadata or None):
                yield root
        except Exception:                                  # noqa: BLE001
            yield root


def step(name: str, as_type: str = "span", **fields):
    """A child observation. as_type: span | generation | retriever | evaluator
    | embedding | chain | tool | guardrail."""
    return _observation(name, as_type, **fields)


def flush() -> None:
    if config.TRACING and _client is not None:
        try:
            _client.flush()
        except Exception:                                  # noqa: BLE001
            log.warning("tracing flush failed")
