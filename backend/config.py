"""
Configuration, validated once at import.

Every setting is read here and nowhere else. Two rules this enforces:

1. **Fail at startup, not at first request.** A missing NEO4J_PASSWORD should
   stop the process immediately with a clear message, not surface as a 500 to
   whoever asks the first question.
2. **Secrets never leave this module.** `public_settings()` is the only thing
   any route may return, and it is a hand-written allow-list. A blanket
   `dict(os.environ)` or a `__repr__` of a settings object is how credentials
   end up in a health endpoint or a log line.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
# No FRONTEND_DIR: the frontend is a separate service (frontend/server.py)
# and this backend never reads frontend/ at all.
# Outside data/: data/ is rebuilt by kg_build, and an audit trail that a
# rebuild can delete is not an audit trail.
LOG_DIR = BASE_DIR / "logs"


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default if default is not None else "")
    if required and not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {os.environ[name]!r}")


def load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Minimal .env reader — no dependency for something this small.

    Deliberately does NOT overwrite variables already in the environment:
    a value injected by the container or CI must win over a stale file left
    on a developer's disk.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

# --- Neo4j: the graph is the source of truth at runtime -------------------- #
NEO4J_URI = _env("NEO4J_URI", required=True)
NEO4J_USER = _env("NEO4J_USER") or _env("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", required=True)
NEO4J_DATABASE = _env("NEO4J_DATABASE", "neo4j")

# --- Language model -------------------------------------------------------- #
PROVIDER = _env("DPDP_PROVIDER", "ollama").lower()
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
# OpenRouter fronts many vendors under one API, so there is no sane default
# model to fall back to — DPDP_MODEL must name one explicitly, e.g.
# "openai/gpt-4o-mini" or "anthropic/claude-sonnet-5". OpenRouterProvider.check()
# reports a clear error rather than silently starting with an empty model string.
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_HOST = _env("OPENROUTER_HOST", "https://openrouter.ai/api/v1").rstrip("/")
_DEFAULT_MODEL = {"ollama": "qwen2.5:3b-instruct", "claude": "claude-sonnet-5"}
MODEL = _env("DPDP_MODEL") or _DEFAULT_MODEL.get(PROVIDER, _DEFAULT_MODEL["ollama"])
NUM_CTX = _int("DPDP_NUM_CTX", 16384)
LLM_TIMEOUT = _int("DPDP_LLM_TIMEOUT", 600)

# --- Model routing (V2 phase 3) -------------------------------------------- #
# The 3B default produced BOTH errors this project has on record (a misread
# Schedule figure, a wrong reading of s.14). Rather than make every question
# pay for a large model, complexity decides: templates need no model at all,
# simple synthesis uses the local/default one, and genuinely compound
# questions go to the larger one. See llm.route_for().
LARGE_PROVIDER = _env("DPDP_LARGE_PROVIDER", "openrouter").lower()
LARGE_MODEL = _env("DPDP_LARGE_MODEL", "openai/gpt-4o-mini")
FORCE_LARGE = _flag("DPDP_FORCE_LARGE", False)

# --- Embeddings (V2 phase 1) ----------------------------------------------- #
# Dense retrieval is fused with BM25, never replacing it: BM25 is what keeps
# "250 crore" from blurring into "200 crore", which embeddings measurably do.
#
# Two providers, same interface as llm.py, chosen by DPDP_EMBED_PROVIDER:
#   openrouter  hosted. No RAM cost, no model download. ~1s per query and a
#               free-tier daily cap — measured, see V2_PLAN.md 2.4.
#   local       fastembed/ONNX. ~5-15ms, no quota, ~40-60MB RSS. NEVER
#               sentence-transformers: `import torch` alone costs +168MB RSS,
#               measured, which does not fit a 512MB free tier.
#
# CRITICAL: build time and query time must use the SAME provider and model.
# Cosine similarity between vectors from different models is meaningless, and
# nothing detects it — retrieval just quietly degrades. embeddings.npy carries
# a canary (see kg_build/embed.py) and startup refuses a mismatch.
EMBED_PROVIDER = _env("DPDP_EMBED_PROVIDER", "openrouter").lower()
_DEFAULT_EMBED_MODEL = {
    "openrouter": "liquid/lfm-2.5-embedding-350m:free",
    "local": "BAAI/bge-small-en-v1.5",
}
EMBED_MODEL = _env("DPDP_EMBED_MODEL") or _DEFAULT_EMBED_MODEL.get(
    EMBED_PROVIDER, _DEFAULT_EMBED_MODEL["openrouter"])
EMBED_TIMEOUT = _int("DPDP_EMBED_TIMEOUT", 60)
# Embedding models cap their input. LFM2.5-Embedding-350M rejects anything
# over 512 tokens outright (measured: a 5,992-char chunk is ~1,350 tokens, so
# roughly 4.4 chars/token on this corpus's legal prose). 1800 chars leaves
# real headroom, since punctuation-dense statutory text tokenises worse than
# the average. See Chunk.embedding_text for what gets kept when it does not fit.
EMBED_MAX_CHARS = _int("DPDP_EMBED_MAX_CHARS", 1800)
# bge-family models want this prefix on QUERIES (never on passages) or recall
# measurably drops. Empty for models that do not use one.
EMBED_QUERY_PREFIX = _env(
    "DPDP_EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: "
    if EMBED_PROVIDER == "local" else "")

# --- Retrieval ------------------------------------------------------------- #
MAX_CONTEXT_CHARS = _int("DPDP_MAX_CONTEXT_CHARS", 10000)

# V2 feature flags. Each phase is independently reversible: flag off restores
# exactly the V1 path, so a regression can be bisected to one phase.
HYBRID = _flag("DPDP_HYBRID", True)              # phase 1: dense + RRF
INTENT_ROUTER = _flag("DPDP_INTENT_ROUTER", True)  # phase 2: gate/router/templates
STRUCTURED_OUTPUT = _flag("DPDP_STRUCTURED_OUTPUT", True)  # phase 3
NUMERIC_CHECK = _flag("DPDP_NUMERIC_CHECK", True)   # phase 4
CONVERSATIONS = _flag("DPDP_CONVERSATIONS", True)   # phase 5
VERIFY_CLAIMS = _flag("DPDP_VERIFY_CLAIMS", False)  # phase 9, off: CPU-heavy

# RRF constant from the original paper. Rank-based, so BM25's unbounded
# scores and cosine's [-1,1] never have to be normalised against each other.
RRF_K = _int("DPDP_RRF_K", 60)
# vocab.yaml demoted from query expansion to a score boost (V2_PLAN.md 1.3).
# Boost rather than expansion so the ON/OFF ablation is interpretable —
# expansion changes what BM25 scores, which confounds the measurement.
VOCAB_BOOST = float(_env("DPDP_VOCAB_BOOST", "1.5"))

# --- Rate limiting (V2 phase 6) --------------------------------------------- #
# In-memory, per-process. Does NOT survive a restart and does NOT coordinate
# across instances — deliberate: Redis is paid infrastructure this does not
# need yet. Closes the "sign-in makes abuse attributable, not impossible" gap.
RATE_LIMIT = _int("DPDP_RATE_LIMIT", 30)          # requests per user per hour
RATE_WINDOW = _int("DPDP_RATE_WINDOW", 3600)

# --- Temporal validity (V2 phase 7) ----------------------------------------- #
# Staged commencement is real: rules 1, 2 and 17-21 are in force on
# publication, rule 4 after a year, the rest after eighteen months. Without
# this, "is rule 4 in force?" is a reading-comprehension question put to a
# language model instead of a date comparison.
AS_OF_DEFAULT = _env("DPDP_AS_OF_DEFAULT", "today")
# Below this BM25 score the corpus contains nothing resembling an answer.
# Calibrated against out-of-scope probes: clearly unrelated questions score
# 7-12, genuine ones 22-85. It catches the obvious end only — questions from
# adjacent legal domains (GDPR, HIPAA) share enough vocabulary to score high,
# so this is a first-line filter, not a domain classifier.
ABSTAIN_THRESHOLD = float(_env("DPDP_ABSTAIN_THRESHOLD", "15.0"))

# --- Authentication (Supabase) ---------------------------------------------- #
# Required: every answer must be attributable to a signed-in user, and an open
# endpoint spends real money on LLM calls. `/api/health` stays public so infra
# monitoring does not need a token.
#
# Three keys, three very different exposure levels — do not mix them up:
#   ANON_KEY          public by design. Ships to the browser, identifies the
#                     project, grants only what RLS policies allow.
#   SERVICE_ROLE_KEY  server-side only. BYPASSES row-level security entirely.
#                     Never send it to the browser, never log it.
#   JWT_SECRET        only needed on projects still using legacy HS256 signing.
SUPABASE_URL = _env("SUPABASE_URL", required=True).rstrip("/")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY", required=True)
SUPABASE_SERVICE_ROLE_KEY = _env("SUPABASE_SERVICE_ROLE_KEY", required=True)

# Supabase is mid-migration from symmetric HS256 (one shared secret) to
# asymmetric ES256/RS256 (JWKS, public keys only). Their docs are explicit that
# the JWKS endpoint "does not return any keys if you are not using asymmetric
# JWT signing keys", so a JWKS-only client fails 100% of auth on a project
# still on the legacy secret. `auth.py` supports both and picks by the token's
# own `alg` header; this secret is only consulted on the HS256 path, so it is
# optional and stays unset on a project already using signing keys.
SUPABASE_JWT_SECRET = _env("SUPABASE_JWT_SECRET")
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
# Supabase stamps this issuer on every access token it signs.
SUPABASE_ISSUER = f"{SUPABASE_URL}/auth/v1"
# The audience Supabase gives a signed-in (non-anonymous) user.
SUPABASE_AUDIENCE = "authenticated"

# --- Q&A history and usage metrics (MongoDB) -------------------------------- #
# Required, same reasoning as Neo4j: this is where every answered question
# ends up, and a system that silently stopped recording that would be worse
# than one that refuses to start. Supabase Postgres stays scoped to identity
# and login history only (auth.users, profiles, login_events) — no question,
# answer, or citation content lives there. See supabase_setup.sql and
# mongo.py for the split.
MONGODB_URI = _env("MONGODB_URI", required=True)
MONGODB_DB = _env("MONGODB_DB", "compliance_assistant")

# --- Tracing (optional) ----------------------------------------------------- #
# Off unless BOTH keys are set — no partial/broken state where a public key
# exists but tracing silently never authenticates. Never required: unlike
# NEO4J_*, an install with nothing set here just doesn't trace.
LANGFUSE_PUBLIC_KEY = _env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _env("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = _env("LANGFUSE_HOST", "https://cloud.langfuse.com")
TRACING_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# --- HTTP ------------------------------------------------------------------ #
# Not optional in practice: the frontend is a separate service
# (frontend/server.py) on its own origin, so this must name it or the
# browser blocks every request before it reaches this app at all — no CORS
# error surfaces in this app's own logs, only in the browser's console.
# Comma-separated for more than one origin. "*" is refused below rather than
# silently honoured — this API answers from a private corpus and writes an
# audit log.
_origins = _env("DPDP_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]
if "*" in CORS_ORIGINS:
    raise RuntimeError(
        "DPDP_CORS_ORIGINS=* is refused. This API answers from a private "
        "corpus and writes an audit log; list real origins explicitly.")

# Surfaces internal error detail in HTTP responses. Off in production so a
# stack trace or a connection string can never reach a browser.
DEBUG = _flag("DPDP_DEBUG", False)

# Interactive API docs (/docs) and the OpenAPI schema. Deliberately its OWN
# flag rather than riding on DEBUG, which it used to: DEBUG leaks internal
# error detail and must stay off in production, but docs are useful there and
# leak far less than they appear to. Every endpoint this schema describes is
# already named in the frontend's own JavaScript, which is public by
# definition — hiding /docs does not hide the API surface, it only makes it
# inconvenient to read. Everything sensitive is gated on a verified token,
# not on the endpoint being unguessable.
DOCS_ENABLED = _flag("DPDP_DOCS", True)


def public_settings() -> dict:
    """The ONLY settings any HTTP response may include.

    An allow-list, not a filter: adding a config value above must not
    silently make it public. Nothing here is a credential, a host, or a path.
    """
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "tracing_enabled": TRACING_ENABLED,
        # Which V2 paths are live. Named capabilities, not credentials — the
        # frontend renders differently when routing is on, and a bug report
        # that says "hybrid was off" is worth far more than one that doesn't.
        "features": {
            "hybrid": HYBRID,
            "intent_router": INTENT_ROUTER,
            "structured_output": STRUCTURED_OUTPUT,
            "numeric_check": NUMERIC_CHECK,
            "conversations": CONVERSATIONS,
            "verify_claims": VERIFY_CLAIMS,
        },
        # The model id only. EMBED_PROVIDER's key never appears anywhere.
        "embed_model": EMBED_MODEL if HYBRID else "",
    }


def frontend_config() -> dict:
    """The ONLY values embedded into the served HTML.

    Deliberately a second allow-list rather than an addition to
    `public_settings()`: these two are *published to the browser*, which is a
    stronger claim than "safe in a health response", and keeping the lists
    apart means neither can be widened by accident. Both values here are
    public by Supabase's own design — the anon key identifies the project and
    grants only what row-level security allows.

    SUPABASE_SERVICE_ROLE_KEY must never appear here. It bypasses RLS.
    """
    return {
        "supabaseUrl": SUPABASE_URL,
        "supabaseAnonKey": SUPABASE_ANON_KEY,
    }
