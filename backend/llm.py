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


class OllamaProvider(Provider):
    def _post(self, path: str, payload: dict, *, stream: bool):
        request = urllib.request.Request(
            config.OLLAMA_HOST + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT)

    def _payload(self, prompt: str, system: str, temperature: float,
                 stream: bool) -> dict:
        return {
            "model": config.MODEL,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
            "stream": stream,
            "options": {"num_ctx": config.NUM_CTX, "temperature": temperature},
        }

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

    def stream(self, prompt: str, system: str, temperature: float) -> Iterator[str]:
        try:
            with self._post("/api/chat",
                            self._payload(prompt, system, temperature, True),
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
                 stream: bool) -> dict:
        return {
            "model": config.MODEL,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": stream,
        }

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

    def stream(self, prompt: str, system: str, temperature: float) -> Iterator[str]:
        request = urllib.request.Request(
            config.OPENROUTER_HOST + "/chat/completions",
            data=json.dumps(self._payload(prompt, system, temperature, True)).encode("utf-8"),
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


def is_small_model() -> bool:
    return config.PROVIDER == "ollama" and any(
        tag in config.MODEL.lower() for tag in SMALL_TAGS)
