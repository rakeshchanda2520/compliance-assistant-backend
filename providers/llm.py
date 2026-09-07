"""
Language model providers behind one interface.

Three providers, no SDK where the API is plain HTTP. OpenRouter and Ollama are
both OpenAI-compatible chat completions, so `urllib` is enough and adds no
dependency; Anthropic gets its official SDK, imported lazily so a deployment
that never uses it never pays for the import.

Nothing outside this module talks to a model. That is what makes the engine
testable — `Deps.synthesize` is a callable, and the tests pass a generator.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Iterator

import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A model call failed in a way the caller should surface, not retry."""


class Provider(ABC):
    name: str

    @abstractmethod
    def check(self) -> str | None:
        """Return an error string if unusable, else None. Never raises."""

    @abstractmethod
    def stream(self, prompt: str, system: str, temperature: float,
               model: str) -> Iterator[str]: ...

    def stream_structured(self, prompt: str, system: str, temperature: float,
                          model: str, schema: dict) -> Iterator[str]:
        """Default: ignore the schema and stream prose.

        A provider that cannot constrain output must still answer — the
        verification layer degrades to the regex citation path rather than
        losing an answer the model did produce.
        """
        return self.stream(prompt, system, temperature, model)


def _post(url: str, payload: dict, headers: dict, timeout: int):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read()[:400].decode("utf-8", "replace")
        # 429 is named explicitly: on a free tier it is the single most common
        # failure and reads as "the model is broken" unless it is spelled out.
        if exc.code == 429:
            raise LLMError("rate limit reached at the model provider — the "
                           "free tier caps requests; wait or configure a paid "
                           "key") from exc
        raise LLMError(f"model provider returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"could not reach the model provider: {exc.reason}") from exc


def _sse_deltas(response) -> Iterator[str]:
    """Yield content deltas from an OpenAI-compatible SSE stream."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            return
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices", ()):
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                yield piece


class OpenRouterProvider(Provider):
    name = "openrouter"

    def check(self) -> str | None:
        if not config.OPENROUTER_API_KEY:
            return "OPENROUTER_API_KEY is not set"
        if not config.MODEL:
            return ("DPDP_MODEL must name a vendor-prefixed model, e.g. "
                    "openai/gpt-4o-mini — OpenRouter fronts many vendors and "
                    "has no sane default")
        return None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}

    def stream(self, prompt: str, system: str, temperature: float,
               model: str) -> Iterator[str]:
        payload = {"model": model, "stream": True, "temperature": temperature,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}]}
        yield from _sse_deltas(_post(f"{config.OPENROUTER_HOST}/chat/completions",
                                     payload, self._headers(), config.LLM_TIMEOUT))

    def stream_structured(self, prompt: str, system: str, temperature: float,
                          model: str, schema: dict) -> Iterator[str]:
        payload = {"model": model, "stream": True, "temperature": temperature,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}],
                   "response_format": {"type": "json_schema",
                                       "json_schema": schema}}
        try:
            yield from _sse_deltas(_post(f"{config.OPENROUTER_HOST}/chat/completions",
                                         payload, self._headers(), config.LLM_TIMEOUT))
        except LLMError as exc:
            # Not every model on OpenRouter supports json_schema, and the
            # failure is a 400 rather than a capability flag. Degrading to
            # prose is strictly better than losing the answer.
            log.warning("structured output rejected (%s); retrying as prose", exc)
            yield from self.stream(prompt, system, temperature, model)


class OllamaProvider(Provider):
    name = "ollama"

    def check(self) -> str | None:
        try:
            with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/tags",
                                        timeout=3) as response:
                json.load(response)
            return None
        except Exception:                                  # noqa: BLE001
            return (f"Ollama is not reachable at {config.OLLAMA_HOST} — "
                    f"start it with `ollama serve`")

    def stream(self, prompt: str, system: str, temperature: float,
               model: str) -> Iterator[str]:
        payload = {"model": model, "stream": True,
                   "options": {"temperature": temperature,
                               "num_ctx": 16384},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": prompt}]}
        response = _post(f"{config.OLLAMA_HOST}/api/chat", payload, {},
                         config.LLM_TIMEOUT)
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = (chunk.get("message") or {}).get("content")
            if piece:
                yield piece
            if chunk.get("done"):
                return


class ClaudeProvider(Provider):
    name = "claude"

    def check(self) -> str | None:
        if not config.ANTHROPIC_API_KEY:
            return "ANTHROPIC_API_KEY is not set"
        try:
            import anthropic                               # noqa: F401
        except ImportError:
            return "the `anthropic` package is not installed"
        return None

    def stream(self, prompt: str, system: str, temperature: float,
               model: str) -> Iterator[str]:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY,
                                     timeout=config.LLM_TIMEOUT)
        try:
            with client.messages.stream(
                    model=model, max_tokens=2048, temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]) as stream:
                yield from stream.text_stream
        except Exception as exc:                           # noqa: BLE001
            raise LLMError(f"Anthropic call failed: {exc}") from exc


_PROVIDERS = {"openrouter": OpenRouterProvider, "ollama": OllamaProvider,
              "claude": ClaudeProvider}


def provider(name: str = "") -> Provider:
    cls = _PROVIDERS.get((name or config.PROVIDER).lower())
    if cls is None:
        raise LLMError(f"unknown provider {name or config.PROVIDER!r}; "
                       f"choose one of {sorted(_PROVIDERS)}")
    return cls()


def check() -> str | None:
    try:
        return provider().check()
    except LLMError as exc:
        return str(exc)


# --------------------------------------------------------------------------- #
# model routing
# --------------------------------------------------------------------------- #

# Questions whose answer requires holding two provisions in mind at once. A
# small model has already misread this corpus twice; paying for a larger one
# on every question is waste, so complexity decides.
_COMPOUND = ("and what", "as well as", "also", "in addition", "both",
             "compare", "difference between", "versus", " vs ")


def needs_large_model(question: str, provision_count: int, intent) -> bool:
    low = question.lower()
    if any(marker in low for marker in _COMPOUND):
        return True
    # Many provisions in context means the answer has to reconcile them.
    return provision_count >= 10


def model_for(large: bool) -> str:
    return config.LARGE_MODEL if large else config.MODEL


def synthesizer(schema: dict | None = None):
    """Build the callable the engine takes as `Deps.synthesize`.

    Returned as a closure rather than exposed as a class so the engine keeps
    knowing nothing about providers, models or schemas — it calls one function
    and receives strings.
    """
    from answering.prompt import SYSTEM_PROMPT, STRUCTURED_INSTRUCTION

    def synthesize(*, question: str, context: str, allowed_ids: str, plan):
        large = needs_large_model(question, allowed_ids.count("\n") + 1, plan.intent)
        model = model_for(large)
        system = SYSTEM_PROMPT
        prompt = (f"{context}\n\nProvision ids you may cite:\n{allowed_ids}"
                  f"\n\nQuestion: {question}")

        p = provider()
        if schema and config.STRUCTURED_OUTPUT:
            system = f"{SYSTEM_PROMPT}\n{STRUCTURED_INSTRUCTION}"
            yield from p.stream_structured(prompt, system, 0.1, model, schema)
        else:
            yield from p.stream(prompt, system, 0.1, model)

    return synthesize
