"""
Language-model access behind one interface.

Three providers, identical surface, selected by DPDP_PROVIDER:

    ollama       local, stdlib HTTP. Default, so the system runs with no key.
    claude       hosted, official SDK. Materially better at reading statute.
    openrouter   hosted, stdlib HTTP (OpenAI-compatible chat completions).
                 One key, any of OpenRouter's vendors — DPDP_MODEL picks the
                 model, e.g. "openai/gpt-4o-mini" or "anthropic/claude-sonnet-5".

Callers never import a provider directly. Swapping models is an environment
variable, not a code change.

Error handling is deliberate: provider responses can echo prompt fragments or
internal hostnames, so `LLMError` carries a short operator-facing message and
the raw detail goes to the log, never to the caller's caller.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator

from . import config

log = logging.getLogger(__name__)

# Small models are fine for indexing but misread statute — they have produced
# both a wrong penalty figure and a wrong reading of §14 in this corpus.
SMALL_TAGS = ("1b", "1.5b", "2b", "3b", "4b")


class LLMError(RuntimeError):
    """Safe to show a user. Detail is logged, not carried in the message."""


class Provider(ABC):
    @abstractmethod
    def check(self) -> str | None: ...

    @abstractmethod
    def complete(self, prompt: str, system: str, temperature: float) -> str: ...

    @abstractmethod
    def stream(self, prompt: str, system: str,
               temperature: float) -> Iterator[str]: ...

    def stream_structured(self, prompt: str, system: str, temperature: float,
                          schema: dict) -> Iterator[str]:
        """Stream while constrained to `schema`.

        The default is an honest no-op: stream normally and let the prompt do
        the work. A provider that cannot truly constrain decoding should NOT
        pretend to — `streaming.AnswerStream` already handles a model that
        ignores the schema, so an unconstrained provider degrades to the
        free-text path instead of failing. Overriding this is an optimisation,
        not a correctness requirement.
        """
        yield from self.stream(prompt, system, temperature)


class OllamaProvider(Provider):
    def _post(self, path: str, payload: dict, *, stream: bool):
        request = urllib.request.Request(
            config.OLLAMA_HOST + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT)

    def _payload(self, prompt: str, system: str, temperature: float,
                 stream: bool, schema: dict | None = None) -> dict:
        payload = {
            "model": config.MODEL,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
            "stream": stream,
            "options": {"num_ctx": config.NUM_CTX, "temperature": temperature},
        }
        if schema:
            # Ollama takes the JSON schema directly in `format` (a bare
            # "json" string also works on older builds). This is real
            # constrained decoding via llama.cpp's grammar support, which
            # matters here: the local model is the 3B one, and it is the
            # least likely of the three to follow a schema from prose alone.
            payload["format"] = schema
        return payload

    def check(self) -> str | None:
        try:
            with urllib.request.urlopen(
                    config.OLLAMA_HOST + "/api/tags", timeout=10) as response:
                installed = {m["name"] for m in json.loads(response.read()).get("models", [])}
        except OSError:
            return "local model server is not reachable"
        if config.MODEL not in installed and f"{config.MODEL}:latest" not in installed:
            return f"model '{config.MODEL}' is not installed"
        return None

    def complete(self, prompt: str, system: str, temperature: float) -> str:
        try:
            with self._post("/api/chat",
                            self._payload(prompt, system, temperature, False),
                            stream=False) as response:
                return json.loads(response.read())["message"]["content"]
        except (OSError, KeyError, ValueError) as exc:
            log.exception("ollama completion failed")
            raise LLMError("the language model did not return an answer") from exc

    def stream_structured(self, prompt: str, system: str, temperature: float,
                          schema: dict) -> Iterator[str]:
        yield from self._stream(prompt, system, temperature, schema)

    def stream(self, prompt: str, system: str, temperature: float) -> Iterator[str]:
        yield from self._stream(prompt, system, temperature)

    def _stream(self, prompt: str, system: str, temperature: float,
                schema: dict | None = None) -> Iterator[str]:
        try:
            with self._post("/api/chat",
                            self._payload(prompt, system, temperature, True, schema),
                            stream=True) as response:
                for raw in response:
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw)
                    if fragment := chunk.get("message", {}).get("content"):
                        yield fragment
                    if chunk.get("done"):
                        return
        except (OSError, ValueError) as exc:
            log.exception("ollama stream failed")
            raise LLMError("the language model stopped responding") from exc


class ClaudeProvider(Provider):
    """`anthropic` is imported lazily so a deployment that never selects this
    provider does not need the package installed."""

    MAX_TOKENS = 4096

    def __init__(self) -> None:
        self._client = None

    def _get(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        return self._client

    def check(self) -> str | None:
        import os
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return "ANTHROPIC_API_KEY is not set"
        return None

    def complete(self, prompt: str, system: str, temperature: float) -> str:
        import anthropic
        try:
            response = self._get().with_options(
                timeout=float(config.LLM_TIMEOUT)).messages.create(
                model=config.MODEL, max_tokens=self.MAX_TOKENS,
                temperature=temperature, system=system,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in response.content if b.type == "text")
        except anthropic.APIError as exc:
            log.exception("claude completion failed")
            raise LLMError("the language model did not return an answer") from exc

    def stream(self, prompt: str, system: str, temperature: float) -> Iterator[str]:
        import anthropic
        try:
            with self._get().with_options(
                    timeout=float(config.LLM_TIMEOUT)).messages.stream(
                    model=config.MODEL, max_tokens=self.MAX_TOKENS,
                    temperature=temperature, system=system,
                    messages=[{"role": "user", "content": prompt}]) as stream:
                yield from stream.text_stream
        except anthropic.APIError as exc:
            log.exception("claude stream failed")
            raise LLMError("the language model stopped responding") from exc


class OpenRouterProvider(Provider):
    """OpenRouter's API is OpenAI-compatible chat completions over plain HTTP,
    so this needs no SDK — the same stdlib `urllib` approach as
    `OllamaProvider`, just with an `Authorization` header and a different
    response shape (`choices[0].message`/`.delta`, not `message`)."""

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            # Optional per OpenRouter's docs, used only for their own request
            # attribution/leaderboards — never sent anywhere else.
            "X-Title": "DPDP Compliance Assistant",
        }

    def _payload(self, prompt: str, system: str, temperature: float,
                 stream: bool, schema: dict | None = None,
                 model: str | None = None) -> dict:
        payload = {
            "model": model or config.MODEL,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": stream,
        }
        if schema:
            # OpenRouter passes json_schema through to vendors that support
            # constrained decoding and degrades gracefully on those that
            # don't — which is exactly the contract AnswerStream expects.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "compliance_answer",
                                "strict": True, "schema": schema},
            }
        return payload

    def check(self) -> str | None:
        if not config.OPENROUTER_API_KEY:
            return "OPENROUTER_API_KEY is not set"
        if not config.MODEL:
            return "DPDP_MODEL is not set — OpenRouter has no default model"
        return None

    def complete(self, prompt: str, system: str, temperature: float) -> str:
        request = urllib.request.Request(
            config.OPENROUTER_HOST + "/chat/completions",
            data=json.dumps(self._payload(prompt, system, temperature, False)).encode("utf-8"),
            headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            log.exception("openrouter completion failed: %s", exc.read()[:500])
            raise LLMError("the language model did not return an answer") from exc
        except (OSError, KeyError, IndexError, ValueError) as exc:
            log.exception("openrouter completion failed")
            raise LLMError("the language model did not return an answer") from exc

    def stream_structured(self, prompt: str, system: str, temperature: float,
                          schema: dict) -> Iterator[str]:
        yield from self._stream(prompt, system, temperature, schema=schema)

    def stream(self, prompt: str, system: str, temperature: float) -> Iterator[str]:
        yield from self._stream(prompt, system, temperature)

    def _stream(self, prompt: str, system: str, temperature: float,
                schema: dict | None = None,
                model: str | None = None) -> Iterator[str]:
        request = urllib.request.Request(
            config.OPENROUTER_HOST + "/chat/completions",
            data=json.dumps(self._payload(prompt, system, temperature, True,
                                          schema, model)).encode("utf-8"),
            headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    # Server-sent events: "data: {...}" per chunk, "data: [DONE]"
                    # to close the stream, blank lines as keep-alive padding.
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if fragment := delta.get("content"):
                        yield fragment
        except urllib.error.HTTPError as exc:
            log.exception("openrouter stream failed: %s", exc.read()[:500])
            raise LLMError("the language model stopped responding") from exc
        except (OSError, ValueError) as exc:
            log.exception("openrouter stream failed")
            raise LLMError("the language model stopped responding") from exc


_PROVIDERS = {"ollama": OllamaProvider, "claude": ClaudeProvider,
             "openrouter": OpenRouterProvider}
_instance: Provider | None = None


def provider() -> Provider:
    global _instance
    if _instance is None:
        try:
            _instance = _PROVIDERS[config.PROVIDER]()
        except KeyError:
            raise RuntimeError(
                f"unknown DPDP_PROVIDER {config.PROVIDER!r}; "
                f"expected one of {sorted(_PROVIDERS)}")
    return _instance


def check() -> str | None:
    return provider().check()


def complete(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    return provider().complete(prompt, system, temperature)


def stream(prompt: str, system: str = "", temperature: float = 0.1) -> Iterator[str]:
    yield from provider().stream(prompt, system, temperature)


def stream_structured(prompt: str, system: str, schema: dict,
                      temperature: float = 0.1,
                      large: bool = False) -> Iterator[str]:
    """Stream constrained to `schema`, optionally on the larger model."""
    engine = large_provider() if large else provider()
    yield from engine.stream_structured(prompt, system, temperature, schema)


def is_small_model() -> bool:
    return config.PROVIDER == "ollama" and any(
        tag in config.MODEL.lower() for tag in SMALL_TAGS)


# --------------------------------------------------------------------------- #
# complexity routing
# --------------------------------------------------------------------------- #

_large_instance: Provider | None = None


class _LargeOpenRouter(OpenRouterProvider):
    """The large model, reached through the same OpenRouter transport.

    A subclass rather than a config flip because both models must be usable
    in the same process: a simple question and a compound one can be in
    flight simultaneously, and mutating config.MODEL between them would make
    the model that answered a given request unknowable.
    """

    def _payload(self, prompt, system, temperature, stream, schema=None, model=None):
        return super()._payload(prompt, system, temperature, stream, schema,
                                model or config.LARGE_MODEL)

    def check(self) -> str | None:
        if not config.OPENROUTER_API_KEY:
            return "OPENROUTER_API_KEY is not set (needed for the large model)"
        if not config.LARGE_MODEL:
            return "DPDP_LARGE_MODEL is not set"
        return None


def large_provider() -> Provider:
    """The model used for genuinely complex questions.

    Falls back to the default provider when the large one is unconfigured —
    a missing large model should degrade answer quality, never fail a request
    that the default model can serve.
    """
    global _large_instance
    if _large_instance is None:
        if config.LARGE_PROVIDER == "openrouter":
            _large_instance = _LargeOpenRouter()
        elif config.LARGE_PROVIDER in _PROVIDERS:
            _large_instance = _PROVIDERS[config.LARGE_PROVIDER]()
        else:
            log.warning("unknown DPDP_LARGE_PROVIDER %r; using the default "
                        "provider for complex questions", config.LARGE_PROVIDER)
            _large_instance = provider()
        if problem := _large_instance.check():
            log.warning("large model unavailable (%s); complex questions will "
                        "use %s/%s", problem, config.PROVIDER, config.MODEL)
            _large_instance = provider()
    return _large_instance


def needs_large_model(question: str, provision_count: int, intent: str) -> bool:
    """Route by complexity, not by preference.

    The 3B default produced BOTH errors on record, but making every question
    pay for a large model is the wrong correction — templates need no model
    at all, and a single-provision lookup does not need a frontier one. The
    signals below are the ones that actually correlate with the failures:
    reasoning across several provisions, and explicit comparison.
    """
    if config.FORCE_LARGE:
        return True
    if intent == "compound":
        return True
    if provision_count >= 3:
        return True
    low = question.lower()
    return any(word in low for word in
               ("compare", "difference", "versus", " vs ", "both", "whereas"))


def model_name(large: bool) -> str:
    """Which model actually answered — recorded per request, so a bad answer
    can be attributed rather than guessed at."""
    if not large:
        return config.MODEL
    return (config.LARGE_MODEL if isinstance(large_provider(), _LargeOpenRouter)
            else config.MODEL)
