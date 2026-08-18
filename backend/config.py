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

# --- Retrieval ------------------------------------------------------------- #
MAX_CONTEXT_CHARS = _int("DPDP_MAX_CONTEXT_CHARS", 10000)
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
