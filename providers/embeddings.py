"""
Embeddings: hosted or self-hosted, one interface.

**Build time and query time must use the SAME model.** Cosine between vectors
from two different models is meaningless and nothing downstream detects it —
retrieval just quietly returns worse results forever. `retrieval.dense` carries
a canary for this; the dimension check here is the second line of defence.

The self-hosted path uses fastembed (ONNX), never sentence-transformers:
`import torch` alone costs ~168MB of RSS, which does not fit a small instance
and buys nothing for a 237-chunk corpus.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import config

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Embedding failed. ALWAYS recoverable — retrieval falls back to lexical."""


class Embedder(ABC):
    dims: int = 0

    @abstractmethod
    def check(self) -> str | None: ...

    @abstractmethod
    def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]: ...

    def embed_one(self, text: str, *, is_query: bool = True) -> list[float]:
        vectors = self.embed([text], is_query=is_query)
        if not vectors:
            raise EmbeddingError("the embedding provider returned nothing")
        return vectors[0]


def _l2(vector: list[float]) -> list[float]:
    """Normalise once, here, so every consumer can use a dot product as the
    cosine and nothing has to remember to divide."""
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else vector


class OpenRouterEmbedder(Embedder):
    """Hosted. No RAM cost, no model download — but a quota.

    When the free tier's daily cap trips, this raises and retrieval degrades to
    lexical-only. That degradation is a CORRECTNESS risk, not a speed one: the
    answer still streams, it is just quieter and worse, with no symptom unless
    the trace is read. Hence the explicit 429 message.
    """
    name = "openrouter"

    def check(self) -> str | None:
        if not config.OPENROUTER_API_KEY:
            return "OPENROUTER_API_KEY is not set"
        return None

    def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        prefix = config.EMBED_QUERY_PREFIX if is_query else ""
        payload = {"model": config.EMBED_MODEL,
                   "input": [f"{prefix}{t}"[:config.EMBED_MAX_CHARS] for t in texts]}
        request = urllib.request.Request(
            f"{config.OPENROUTER_HOST}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {config.OPENROUTER_API_KEY}"})
        try:
            with urllib.request.urlopen(request, timeout=config.EMBED_TIMEOUT) as r:
                body = json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise EmbeddingError(
                    "embedding rate limit reached — the free tier caps daily "
                    "requests. Retrieval is running lexical-only until it "
                    "resets; set DPDP_EMBED_PROVIDER=local to remove the cap."
                ) from exc
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise EmbeddingError(f"embeddings HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"embeddings unreachable: {exc.reason}") from exc

        vectors = [_l2(item["embedding"]) for item in body.get("data", ())]
        if vectors:
            self.dims = len(vectors[0])
        return vectors


class LocalEmbedder(Embedder):
    """fastembed/ONNX. No quota, no network, ~40-60MB RSS.

    The model is chosen by DIMENSION as much as quality: the shipped index is
    1024-dimensional, and bge-small-en-v1.5 is 384. Switching to it is a full
    paired rebuild of chunks and vectors, not a config flip — startup will
    refuse the mismatch rather than serve on incomparable vectors.
    """
    name = "local"

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbeddingError(
                    "fastembed is not installed — `pip install fastembed`, or "
                    "set DPDP_EMBED_PROVIDER=openrouter") from exc
            self._model = TextEmbedding(model_name=config.EMBED_MODEL)
        return self._model

    def check(self) -> str | None:
        try:
            self._load()
            return None
        except EmbeddingError as exc:
            return str(exc)

    def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        prefix = config.EMBED_QUERY_PREFIX if is_query else ""
        prepared = [f"{prefix}{t}"[:config.EMBED_MAX_CHARS] for t in texts]
        vectors = [_l2(list(map(float, v))) for v in self._load().embed(prepared)]
        if vectors:
            self.dims = len(vectors[0])
        return vectors


_EMBEDDERS = {"openrouter": OpenRouterEmbedder, "local": LocalEmbedder}
_cached: Embedder | None = None


def embedder() -> Embedder:
    global _cached
    if _cached is None:
        cls = _EMBEDDERS.get(config.EMBED_PROVIDER)
        if cls is None:
            raise EmbeddingError(
                f"unknown embedding provider {config.EMBED_PROVIDER!r}; "
                f"choose one of {sorted(_EMBEDDERS)}")
        _cached = cls()
    return _cached


def check() -> str | None:
    try:
        return embedder().check()
    except EmbeddingError as exc:
        return str(exc)


def query_embedder(expect_dims: int = 0):
    """The callable the engine takes as `Deps.embed_query`.

    Returns None — disabling dense retrieval — rather than raising, when the
    provider is unusable at startup. A missing embedder is a quality
    degradation, never a reason to refuse questions.
    """
    if not config.HYBRID:
        return None

    problem = check()
    if problem:
        log.warning("dense retrieval disabled: %s", problem)
        return None

    def embed_query(question: str):
        vector = embedder().embed_one(question, is_query=True)
        if expect_dims and len(vector) != expect_dims:
            raise EmbeddingError(
                f"the provider returned a {len(vector)}-dimensional vector but "
                f"the index is {expect_dims}-dimensional — the embedding model "
                f"changed behind its name. Rebuild the index.")
        return vector

    return embed_query
