"""
Configuration, validated once at import.

Two rules this enforces:

1. **Fail at startup, not at first request.** A missing NEO4J_PASSWORD should
   stop the process immediately, not surface as a 500 to whoever asks first.
2. **Secrets never leave this module.** `public_settings()` is a hand-written
   allow-list. A blanket `dict(os.environ)` or a settings `__repr__` is how
   credentials end up in a health endpoint.

Every retrieval-affecting constant carries the reasoning for its value, because
the last system lost a week to a threshold whose comment described a different
corpus than the one it was running against.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Outside data/: data/ is rebuilt by the corpus build, and an audit trail a
# rebuild can delete is not an audit trail.
LOG_DIR = BASE_DIR / "logs"


def _env(name: str, default: str = "", *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env.")
    return value


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {os.environ[name]!r}")


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, got {os.environ[name]!r}")


def load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Minimal reader — no dependency for something this small. Deliberately
    does not overwrite variables already in the environment: a value injected
    by the container must win over a stale file on a developer's disk."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

# --- corpus ---------------------------------------------------------------- #
# The graph is the runtime source of truth; chunks.json is the search index.
# Both are produced by the same build and MUST travel together — startup
# refuses to serve if they disagree, because mismatched vectors return
# confident nonsense with no other symptom.
NEO4J_URI = _env("NEO4J_URI")
NEO4J_USER = _env("NEO4J_USER") or _env("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD")
NEO4J_DATABASE = _env("NEO4J_DATABASE", "neo4j")
# When Neo4j is unreachable the graph can be reconstructed from chunks.json
# well enough to serve retrieval and citation checks. Structural edges
# (PENALISED_BY, REFERENCES) are NOT in chunks.json, so template paths that
# need them degrade — see providers.graphdb.
ALLOW_OFFLINE_GRAPH = _flag("DPDP_ALLOW_OFFLINE_GRAPH", True)

# --- language model -------------------------------------------------------- #
PROVIDER = _env("DPDP_PROVIDER", "openrouter").lower()
MODEL = _env("DPDP_MODEL", "openai/gpt-4o-mini")
# A separate, cheaper model for the auxiliary calls on the critical path
# (rewriting, grey-zone sufficiency). NEW_FRAMEWORK §7.1: these must not be
# charged at synthesis latency, or the p95 budget cannot hold.
FAST_MODEL = _env("DPDP_FAST_MODEL", "openai/gpt-4o-mini")
LARGE_MODEL = _env("DPDP_LARGE_MODEL", "openai/gpt-4o")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_HOST = _env("OPENROUTER_HOST", "https://openrouter.ai/api/v1").rstrip("/")
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
LLM_TIMEOUT = _int("DPDP_LLM_TIMEOUT", 120)

# --- embeddings ------------------------------------------------------------ #
# CRITICAL: build time and query time must use the SAME provider and model.
# Cosine between vectors from different models is meaningless and nothing
# detects it — retrieval just quietly degrades. embeddings.npz carries a
# canary and startup refuses a mismatch.
#
# The shipped index is 1024-dimensional. bge-small-en-v1.5 is 384-dim, so
# switching to it is a full paired rebuild, not a config flip.
EMBED_PROVIDER = _env("DPDP_EMBED_PROVIDER", "openrouter").lower()
EMBED_MODEL = _env("DPDP_EMBED_MODEL", "liquid/lfm-2.5-embedding-350m:free")
EMBED_TIMEOUT = _int("DPDP_EMBED_TIMEOUT", 30)
# Hosted embedders cap input hard: the current one rejects >512 tokens, and
# statutory prose tokenises worse than average (~4.4 chars/token measured).
EMBED_MAX_CHARS = _int("DPDP_EMBED_MAX_CHARS", 1800)
EMBED_QUERY_PREFIX = _env("DPDP_EMBED_QUERY_PREFIX", "")

# --- retrieval ------------------------------------------------------------- #
# Three-way fusion. Raw BM25 is kept ALONGSIDE contextual BM25, never replaced
# by it: prepending generated context changes term frequencies and dilutes the
# exact-token signal BM25 exists for. "250 crore" must not blur into "200
# crore". A bad context can then only add a ranker, never remove one.
HYBRID = _flag("DPDP_HYBRID", True)
CONTEXTUAL = _flag("DPDP_CONTEXTUAL", True)
RRF_K = _int("DPDP_RRF_K", 60)          # from the original RRF paper
CANDIDATES = _int("DPDP_CANDIDATES", 30)  # fused pool handed to the reranker
TOP_K = _int("DPDP_TOP_K", 8)             # survivors after reranking

# vocab.yaml is a post-hoc BOOST, never a pre-BM25 expansion. Expansion changes
# what BM25 scores, which confounds any measurement of what vocab contributes —
# tried and reverted once already.
VOCAB_BOOST = _float("DPDP_VOCAB_BOOST", 1.5)
# A vocabulary target matching more than this share of the corpus carries no
# discriminative signal. "customer" maps to "personal data", which appears in
# most of the statute — boosting on it promoted the shortest chunks (two
# Definitions) above every Penalty row for a penalty question.
VOCAB_MAX_SHARE = _float("DPDP_VOCAB_MAX_SHARE", 0.20)
# An EXACT phrase match on a chunk's own defined term or headnote. Weighted
# well above the vocabulary boost because it is a far stronger signal: BM25
# scores "Data Fiduciary" as two common words, so the phrase itself carries no
# extra weight without this. It is also the signal most robust to phrasing —
# however the sentence is built around it, naming the term is naming the term.
PHRASE_BOOST = _float("DPDP_PHRASE_BOOST", 3.0)
# The largest class of provision that may be injected wholesale. The Act's
# Schedule is seven rows and ranking them against each other was measured
# unreliable, so completeness wins there. Thirty-two definitions is not a
# class worth injecting — it buries the answer it was meant to surface.
MAX_INJECT = _int("DPDP_MAX_INJECT", 10)

# Graph expansion. MENTIONS is barred from the second hop: those edges all
# point into definitions, so following them twice floods the context with
# vocabulary instead of law.
MAX_HOPS = _int("DPDP_MAX_HOPS", 2)
MAX_EXPANDED = _int("DPDP_MAX_EXPANDED", 10)
HOP_DECAY = _float("DPDP_HOP_DECAY", 0.6)
INTENT_BOOST = _float("DPDP_INTENT_BOOST", 1.6)

MAX_CONTEXT_CHARS = _int("DPDP_MAX_CONTEXT_CHARS", 12000)

# --- reranking ------------------------------------------------------------- #
# Off by default until measured on the deploy host. A 568M-parameter
# cross-encoder scoring 30 pairs on a shared CPU is seconds, not milliseconds;
# the budget is 500ms and the model must be chosen to fit it, quantised.
RERANK = _flag("DPDP_RERANK", False)
RERANK_MODEL = _env("DPDP_RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_TIMEOUT_MS = _int("DPDP_RERANK_TIMEOUT_MS", 800)

# --- understanding --------------------------------------------------------- #
# Tier 1 regex is instant, free and confidence 1.00 — it stays tier 1. A
# learned tier 0 would pay latency and quota for questions regex already
# answers perfectly.
#
# The two cosine floors are deliberately unequal: routing to the wrong
# TEMPLATE produces a confidently wrong answer, while a wrong jurisdiction call
# only over- or under-refuses. Intent is held to the higher bar.
INTENT_MIN_COSINE = _float("DPDP_INTENT_MIN_COSINE", 0.62)
JURISDICTION_MIN_COSINE = _float("DPDP_JURISDICTION_MIN_COSINE", 0.58)
# Foreign must beat domestic by this much, or a DPDP question using shared
# vocabulary flips foreign on a hairline.
JURISDICTION_MARGIN = _float("DPDP_JURISDICTION_MARGIN", 0.04)

# Query rewriting is skipped when tier-1 regex already matched — the common
# case for the highest-traffic question types, and how the latency budget holds.
REWRITE = _flag("DPDP_REWRITE", False)
DECOMPOSE = _flag("DPDP_DECOMPOSE", True)   # deterministic split; no model

# --- sufficiency ----------------------------------------------------------- #
# Replaces a single BM25 threshold, which was MEASURED not to separate. The
# populations overlap, so no threshold admits every genuine question and
# refuses every unrelated one; a single number was always going to be wrong,
# and the only choice is which way it fails.
#
# RE-MEASURED ON THIS INDEX (20 genuine compliance questions, 15 unrelated).
# The inherited 7.3–24.6 / 6.5–15.6 numbers came from a different index and do
# NOT hold here — this one indexes label and headnote, which moves every score:
#
#     genuine    5.02 ──────────────► 11.80
#     unrelated  0.00 ──────► 7.25
#                     ^^^^^^^^^^ overlap
#
#     floor  genuine refused   unrelated refused
#      5.0        0/20               10/15
#      5.5        1/20               11/15
#      6.0        7/20               11/15     <- the inherited value
#      7.0       10/20               14/15
#
# 6.0 refused SEVEN of twenty genuine questions — 35%, including "how long can
# we keep customer records?" whose retrieval had correctly returned the Third
# Schedule's retention rules. That is the failure this gate exists to avoid,
# not the one it exists to cause. 5.0 costs four more unrelated questions a
# model call each and refuses none of the twenty.
#
# Re-measure after any change to what the index covers. Do not carry a floor
# across a corpus or an indexing change — it is not portable.
# How close a query term must be to a corpus term before it is treated as a
# misspelling of it. MEASURED against 12 real misspellings and 21 words from
# out-of-scope questions:
#
#     cutoff   typos repaired   out-of-scope words touched
#      0.90        7/12                 0/21
#      0.85       10/12                 1/21
#      0.82       11/12                 2/21
#      0.75       12/12                 4/21
#
# 0.75, because the two extra words it touches are harmless in a way the
# ratio alone does not show: "mountain"->"contain" and "weather"->"whether"
# resolve to near-stopwords that carry almost no IDF and match no statutory
# phrase, so they cannot flip an out-of-scope question into an answered one
# (verified — every out-of-scope question in eval/measure_floor.py is still
# refused). What 0.82 could NOT repair was "fudiciary" -> "fiduciary", a
# transposition whose ratio is 0.778 — and that is the exact spelling this
# was reported with.
FUZZY_CUTOFF = _float("DPDP_FUZZY_CUTOFF", 0.75)

SUFFICIENCY_FLOOR = _float("DPDP_SUFFICIENCY_FLOOR", 5.0)   # below: refuse
# Above the genuine set's own maximum (11.80), nothing was ever SUFFICIENT on
# the lexical tier and every question fell into the grey band. 9.0 sits above
# every unrelated score measured (7.25) and below the top quartile of genuine
# ones, so a strong match is recognised as strong.
SUFFICIENCY_CEILING = _float("DPDP_SUFFICIENCY_CEILING", 9.0)   # above: answer
SUFFICIENCY_LLM = _flag("DPDP_SUFFICIENCY_LLM", False)      # grey zone only

# --- verification ---------------------------------------------------------- #
STRUCTURED_OUTPUT = _flag("DPDP_STRUCTURED_OUTPUT", True)
NUMERIC_CHECK = _flag("DPDP_NUMERIC_CHECK", True)
QUOTE_CHECK = _flag("DPDP_QUOTE_CHECK", True)   # free: substring test
ENTAILMENT = _flag("DPDP_ENTAILMENT", False)    # post-hoc, never blocking
ENTAILMENT_MODEL = _env("DPDP_ENTAILMENT_MODEL", "cross-encoder/nli-deberta-v3-base")

AS_OF_DEFAULT = _env("DPDP_AS_OF_DEFAULT", "today")

# --- conversations --------------------------------------------------------- #
# Prior turns contribute PROVISIONS, never prior answers. Carrying an answer
# forward is how a system defends a hallucination three turns later.
CONVERSATIONS = _flag("DPDP_CONVERSATIONS", True)
CONVERSATION_TURNS = _int("DPDP_CONVERSATION_TURNS", 3)
CONVERSATION_SEEDS = _int("DPDP_CONVERSATION_SEEDS", 6)

# --- auth, storage, limits ------------------------------------------------- #
SUPABASE_URL = _env("SUPABASE_URL").rstrip("/")
SUPABASE_ANON_KEY = _env("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = _env("SUPABASE_JWT_SECRET")
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json" if SUPABASE_URL else ""
SUPABASE_ISSUER = f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else ""
SUPABASE_AUDIENCE = "authenticated"
# Small clock skew between the signing host and this one is normal; without
# leeway it surfaces as a login that silently did not take.
JWT_LEEWAY = _int("DPDP_JWT_LEEWAY", 30)
# Set false ONLY for local development against a corpus with no auth provider.
REQUIRE_AUTH = _flag("DPDP_REQUIRE_AUTH", True)

MONGODB_URI = _env("MONGODB_URI")
MONGODB_DB = _env("MONGODB_DB", "dpdp_assistant")

# In-memory, per process: does not survive a restart and does not coordinate
# across instances. A 56-question eval baseline CANNOT be captured under 30/h,
# so the eval identity gets its own limit rather than a test-only auth bypass —
# an auth hole is exactly the kind of thing that survives into production.
RATE_LIMIT = _int("DPDP_RATE_LIMIT", 30)
RATE_WINDOW = _int("DPDP_RATE_WINDOW", 3600)
EVAL_USER_IDS = tuple(u for u in _env("DPDP_EVAL_USER_IDS", "").split(",") if u)
EVAL_RATE_LIMIT = _int("DPDP_EVAL_RATE_LIMIT", 500)

# --- observability --------------------------------------------------------- #
LANGFUSE_PUBLIC_KEY = _env("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _env("LANGFUSE_SECRET_KEY")
# Both spellings: the host variable has been renamed once already, and reading
# only one silently sends every trace to the wrong place with no error.
LANGFUSE_HOST = (_env("LANGFUSE_HOST") or _env("LANGFUSE_BASE_URL")
                 or "https://cloud.langfuse.com").rstrip("/")
TRACING = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# --- http ------------------------------------------------------------------ #
_origins = _env("DPDP_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]
if "*" in CORS_ORIGINS:
    raise RuntimeError("DPDP_CORS_ORIGINS=* is refused: this API answers from "
                       "a private corpus and writes an audit log.")
DEBUG = _flag("DPDP_DEBUG", False)
DOCS = _flag("DPDP_DOCS", True)


def public_settings() -> dict:
    """The ONLY settings any HTTP response may include. An allow-list, not a
    filter — adding a value above must not silently make it public."""
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "embed_model": EMBED_MODEL if HYBRID else "",
        "features": {
            "hybrid": HYBRID, "contextual": CONTEXTUAL, "rerank": RERANK,
            "rewrite": REWRITE, "decompose": DECOMPOSE,
            "structured_output": STRUCTURED_OUTPUT,
            "numeric_check": NUMERIC_CHECK, "quote_check": QUOTE_CHECK,
            "entailment": ENTAILMENT, "conversations": CONVERSATIONS,
        },
        "top_k": TOP_K,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "tracing_enabled": TRACING,
    }


def frontend_config() -> dict:
    """The ONLY values embedded in the served page. A SECOND allow-list, kept
    apart from `public_settings` on purpose: these are published to a browser,
    which is a stronger claim than "safe in a health response". Widening one
    must not widen the other. The service-role key must never appear here."""
    return {"supabaseUrl": SUPABASE_URL, "supabaseAnonKey": SUPABASE_ANON_KEY}
