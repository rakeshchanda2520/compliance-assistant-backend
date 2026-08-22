"""
Embedding access behind one interface. Deliberately the same shape as llm.py.

Two providers, identical surface, selected by DPDP_EMBED_PROVIDER:

    openrouter  hosted, stdlib HTTP (OpenAI-compatible /embeddings). No RAM
                cost and no model download, which is what makes a 512MB free
                tier viable at all. Measured: ~1065ms median per query, and a
                free-tier daily request cap.
    local       fastembed (ONNX Runtime). ~5-15ms, no quota, no network.
                NEVER sentence-transformers: `import torch` costs +168MB RSS
                on import alone (measured), before any weights load.

Used at BUILD time (kg_build/embed.py, embedding 237 passages once) and at
QUERY time (one query vector per request). Both go through here.

THE ONE RULE THIS MODULE EXISTS TO ENFORCE
------------------------------------------
Build and query MUST use the same provider and model. Cosine similarity
between vectors from two different models is not "slightly worse" — it is
meaningless, and nothing downstream detects it. There is no exception raised,
no failed round-trip, no wrong-looking output: retrieval simply returns
plausible nonsense.

That is precisely the silent-corruption class this project's round-trip gate
and build_id exist to prevent everywhere else, so embeddings get the same
treatment: `fingerprint()` stamps the artifact, and `verify_fingerprint()`
refuses to start on drift. A hosted model can change behind a stable name at
any time; this is what turns that into a loud startup failure.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from . import config

log = logging.getLogger(__name__)

# Embedded on every build and re-embedded at every startup. If the provider
# has changed the model behind its name, this vector moves and startup fails
# rather than retrieval quietly degrading.
CANARY = "reasonable security safeguards"
# Cosine below this between the stored and freshly computed canary means the
# model is not the one that built the index. Generous on purpose: hosted
# providers are not bit-reproducible, but a genuine model swap moves a vector
# far more than floating-point noise ever does.
CANARY_MIN_COSINE = 0.999


class EmbeddingError(RuntimeError):
    """Safe to show an operator. Detail is logged, never carried here."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def check(self) -> str | None: ...

    @abstractmethod
    def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """Embed a batch. `is_query` selects the asymmetric prefix some models
        (the bge family) require on queries but never on passages — getting
        that backwards measurably degrades recall without failing anything."""


class OpenRouterEmbedder(EmbeddingProvider):
    """OpenAI-compatible /embeddings over plain stdlib HTTP, same reasoning as
    llm.OpenRouterProvider: no SDK dependency for one POST.

    Note the model id is NOT in OpenRouter's /models catalogue (that lists
    chat models only) — the endpoint is real regardless. Verified live.
    """

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "X-Title": "DPDP Compliance Assistant",
        }

    def check(self) -> str | None:
        if not config.OPENROUTER_API_KEY:
            return "OPENROUTER_API_KEY is not set (needed for embeddings)"
        if not config.EMBED_MODEL:
            return "DPDP_EMBED_MODEL is not set"
        return None

    def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        prefix = config.EMBED_QUERY_PREFIX if is_query else ""
        payload = {"model": config.EMBED_MODEL,
                   "input": [prefix + t for t in texts]}
        request = urllib.request.Request(
            config.OPENROUTER_HOST + "/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers())
        try:
            with urllib.request.urlopen(
                    request, timeout=config.EMBED_TIMEOUT) as response:
                body = json.loads(response.read())
            # Order is not guaranteed by the spec; sort by index rather than
            # trusting arrival order, or passage N gets passage M's vector —
            # a corruption that would pass every other check in this system.
            rows = sorted(body["data"], key=lambda d: d.get("index", 0))
            return [r["embedding"] for r in rows]
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500]
            log.error("openrouter embeddings failed: %s", detail)
            if exc.code == 429:
                raise EmbeddingError(
                    "embedding rate limit reached — the free tier caps daily "
                    "requests; set DPDP_EMBED_PROVIDER=local or wait") from exc
            raise EmbeddingError("the embedding provider returned an error") from exc
        except (OSError, KeyError, IndexError, ValueError) as exc:
            log.exception("openrouter embeddings failed")
            raise EmbeddingError("the embedding provider is unreachable") from exc


class LocalEmbedder(EmbeddingProvider):
    """fastembed / ONNX Runtime. Imported lazily so a deployment using the
    hosted provider never pays for the dependency — the same pattern
    llm.ClaudeProvider uses for the anthropic SDK."""

    _model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbeddingError(
                    "DPDP_EMBED_PROVIDER=local needs fastembed: "
                    "pip install fastembed") from exc
            self._model = TextEmbedding(model_name=config.EMBED_MODEL)
        return self._model

    def check(self) -> str | None:
        try:
            self._load()
        except EmbeddingError as exc:
            return str(exc)
        except Exception as exc:                      # noqa: BLE001
            return f"local embedding model unavailable: {type(exc).__name__}"
        return None

    def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if not texts:
            return []
        prefix = config.EMBED_QUERY_PREFIX if is_query else ""
        model = self._load()
        return [list(map(float, v))
                for v in model.embed([prefix + t for t in texts])]


_PROVIDERS = {"openrouter": OpenRouterEmbedder, "local": LocalEmbedder}
_instance: EmbeddingProvider | None = None


def provider() -> EmbeddingProvider:
    global _instance
    if _instance is None:
        try:
            _instance = _PROVIDERS[config.EMBED_PROVIDER]()
        except KeyError:
            raise RuntimeError(
                f"unknown DPDP_EMBED_PROVIDER {config.EMBED_PROVIDER!r}; "
                f"expected one of {sorted(_PROVIDERS)}")
    return _instance


def check() -> str | None:
    """Mirrors llm.check()'s contract so /api/health reports both the same
    way. None when hybrid retrieval is off — an unconfigured embedder is not
    a fault if nothing uses it."""
    if not config.HYBRID:
        return None
    return provider().check()


def embed(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    return provider().embed(texts, is_query=is_query)


def embed_one(text: str, *, is_query: bool = True) -> list[float]:
    vectors = embed([text], is_query=is_query)
    if not vectors:
        raise EmbeddingError("embedding provider returned nothing")
    return vectors[0]


# --------------------------------------------------------------------------- #
# the symmetry guard
# --------------------------------------------------------------------------- #

def fingerprint() -> dict:
    """Identity of the embedding setup that produced an index.

    Written into embeddings.npz at build time and checked at startup. The
    canary vector is the part that catches a *silent* model change: provider
    and model names stay identical when a vendor updates the weights behind
    them, and only the vector moves.
    """
    return {
        "provider": config.EMBED_PROVIDER,
        "model": config.EMBED_MODEL,
        "query_prefix": config.EMBED_QUERY_PREFIX,
        "canary_text": CANARY,
        "canary": embed_one(CANARY, is_query=False),
    }


def verify_fingerprint(stored: dict) -> str | None:
    """None if this process can safely query an index built with `stored`,
    else an operator-facing reason. Called at startup; a non-None result must
    abort rather than warn — serving on mismatched vectors returns confident
    nonsense with no other symptom."""
    if stored.get("provider") != config.EMBED_PROVIDER:
        return (f"index was built with embedding provider "
                f"{stored.get('provider')!r} but this process is configured "
                f"for {config.EMBED_PROVIDER!r}. Rebuild the index or change "
                f"DPDP_EMBED_PROVIDER back.")
    if stored.get("model") != config.EMBED_MODEL:
        return (f"index was built with embedding model {stored.get('model')!r} "
                f"but this process is configured for {config.EMBED_MODEL!r}. "
                f"Vectors from two models are not comparable — rebuild.")
    if stored.get("query_prefix") != config.EMBED_QUERY_PREFIX:
        return ("DPDP_EMBED_QUERY_PREFIX differs from the one used at build "
                "time; queries and passages would be embedded asymmetrically.")

    stored_canary = stored.get("canary")
    if not stored_canary:
        log.warning("index carries no canary vector — cannot detect a silent "
                    "model change behind a stable name")
        return None
    try:
        fresh = embed_one(stored.get("canary_text") or CANARY, is_query=False)
    except EmbeddingError as exc:
        # Reachability is llm-style check()'s job and is reported separately;
        # failing to verify is not the same as failing verification.
        log.warning("could not re-embed canary: %s", exc)
        return None

    similarity = _cosine(stored_canary, fresh)
    if similarity < CANARY_MIN_COSINE:
        return (f"the embedding model has CHANGED behind the name "
                f"{config.EMBED_MODEL!r} (canary cosine {similarity:.4f} < "
                f"{CANARY_MIN_COSINE}). Stored vectors are no longer "
                f"comparable to new queries. Rebuild the index.")
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
