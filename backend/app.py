"""
HTTP layer.

`POST /api/chat` streams over SSE in stages so the wait is legible instead of
blank:

    retrieval   which provisions were found, before the model is called
    token       answer fragments as they are produced
    citations   every citation, checked against the graph
    done        timings and the build id
    abstain     sent instead of an answer when the corpus does not cover it
    error       a safe, human-readable failure

Security posture of this module:
  * Answering requires a verified Google sign-in (`auth.require_user`). Only
    `/api/health` and `/api/config` are public, so infrastructure and the
    frontend's own startup can reach them without a credential.
  * No internal detail in responses. Exceptions are logged with a request id;
    the client receives that id and a generic message unless DEBUG is on.
  * Request bodies are bounded by Pydantic before any work is done.
  * This service is API-only — it serves no HTML. The frontend is a separate
    service (`frontend/server.py`); `DPDP_CORS_ORIGINS` must name its origin
    or the browser blocks every cross-origin request.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import yaml
from fastapi import (Depends, FastAPI, HTTPException, Query, Request,
                     Response)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from . import (audit, auth, citations, config, dense, embeddings, graph_store,
               llm, mongo, numeric, observability, ratelimit, retrieval,
               schema, streaming, templates, temporal, understanding)
from .indexing import load_chunks
from .prompt import SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("compliance")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything once, and fail loudly here rather than on request one."""
    graph = graph_store.load_graph()
    chunks = load_chunks(config.DATA_DIR / "chunks.json")
    vocab = yaml.safe_load((config.DATA_DIR / "vocab.yaml").read_text(encoding="utf-8"))

    # Raises when DPDP_HYBRID=1 but the index is missing or was built by a
    # different embedding model — see dense.load / embeddings.verify_fingerprint.
    # Serving on mismatched vectors returns confident nonsense with no other
    # symptom, so this is one of the few things worth refusing to start over.
    dense_index = dense.load(config.DATA_DIR / "embeddings.npz")

    # Corpus integrity. `backend/data/chunks.json` is a MANUAL copy of the
    # kg_build output (the backend deploys as its own repo, with no access to
    # kg_build and no shared filesystem), so the one failure this deployment
    # genuinely invites is a stale copy: a graph rebuilt in Neo4j while the
    # shipped chunks still describe the previous corpus. Retrieval would then
    # score text that no longer matches the provisions being cited, and
    # nothing else in the system would notice.
    orphans = {c.node_id for c in chunks} - set(graph.provisions)
    if orphans:
        raise RuntimeError(
            f"chunks.json is out of step with Neo4j: {len(orphans)} chunk(s) "
            f"reference provisions that no longer exist "
            f"(e.g. {sorted(orphans)[:5]}). The graph was rebuilt without "
            f"copying the new chunks. Run `python -m kg_build --neo4j --embed` "
            f"then `cp data/chunks.json data/embeddings.npz backend/data/`.")

    if dense_index is not None:
        drift = set(dense_index.node_ids) ^ {c.node_id for c in chunks}
        if drift:
            raise RuntimeError(
                f"embeddings.npz does not match chunks.json: {len(drift)} "
                f"node(s) differ (e.g. {sorted(drift)[:5]}). Both are written "
                f"by the same build — copy them together, and rebuild with "
                f"`python -m kg_build --embed` if either changed.")

    STATE["graph"] = graph
    STATE["retriever"] = retrieval.Retriever(chunks, graph, vocab, dense_index)
    STATE["audit"] = audit.AuditLog(config.LOG_DIR)
    STATE["chunk_count"] = len(chunks)
    STATE["limiter"] = ratelimit.RateLimiter()
    STATE["commencement"] = temporal.load(config.DATA_DIR / "commencement.yaml")

    classifier = understanding.Classifier()
    # Uses the exemplar vectors cached in the dense index, so a restart costs
    # no embedding requests. Degrades to the regex tier on failure.
    classifier.warm(dense_index)
    STATE["classifier"] = classifier

    log.info("ready: %d provisions, %d chunks, build %s, model %s/%s",
             len(graph.provisions), len(chunks), graph.build_id,
             config.PROVIDER, config.MODEL)
    log.info("v2 features: hybrid=%s router=%s structured=%s numeric=%s "
             "conversations=%s", config.HYBRID, config.INTENT_ROUTER,
             config.STRUCTURED_OUTPUT, config.NUMERIC_CHECK, config.CONVERSATIONS)
    if llm.is_small_model():
        log.warning("%s is a small model; it has misread statute in this "
                    "corpus. Prefer a larger model for real use.", config.MODEL)
    yield
    STATE.clear()


app = FastAPI(
    title="DPDP Compliance Assistant API",
    version="2.0",
    lifespan=lifespan,
    summary="Answers questions about India's DPDP Act, 2023 and Rules, 2025, "
            "quoting the statute verbatim with every citation verified against "
            "a knowledge graph.",
    description=(
        "All endpoints are JSON except `POST /api/chat`, which streams "
        "Server-Sent Events.\n\n"
        "**Authentication** — `/api/chat`, `/api/history` and "
        "`/api/provision/{node_id}` require a Supabase-issued Google sign-in "
        "token as `Authorization: Bearer <jwt>`. `/`, `/api/live`, "
        "`/api/health` and `/api/config` are public so infrastructure probes "
        "and the frontend's own bootstrap work without a credential.\n\n"
        "**Answers are informational, not legal advice.** Verify every "
        "citation against the Act or Rules before relying on it."
    ),
    # Docs are on by default and controlled by their own flag — see
    # config.DOCS_ENABLED for why this no longer rides on DEBUG.
    docs_url="/docs" if config.DOCS_ENABLED else None,
    redoc_url="/redoc" if config.DOCS_ENABLED else None,
    openapi_url="/openapi.json" if config.DOCS_ENABLED else None,
    openapi_tags=[
        {"name": "meta", "description": "Service identity, health and public "
                                        "configuration. No authentication."},
        {"name": "corpus", "description": "The Act and Rules themselves. "
                                          "Requires sign-in."},
    ],
)

if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        # "Authorization" matters now in a way it didn't when the frontend
        # was served same-origin: a cross-origin request carrying a custom
        # header triggers a CORS preflight, and a preflight the server
        # doesn't explicitly allow this header on is rejected before the
        # request handler — or auth.require_user — ever runs. Without this,
        # every authenticated route (/api/chat, /api/history,
        # /api/provision) would fail purely from the browser's own
        # preflight, while /api/health kept working, which is a confusing
        # way to discover it.
        allow_headers=["Content-Type", "Authorization"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Headers that cost nothing and close real classes of attack, plus the
    request id that ties a user's report to the logs.

    This service serves no HTML — the frontend is a separate service
    (`frontend/server.py`), which carries its own CSP scoped to what a
    *page* needs (Supabase's client bundle, Google's consent screen, and so
    on). None of that protects anything here: a JSON API has no inline
    script to restrict and nothing for a Content-Security-Policy to narrow,
    so this only sets the headers that still mean something for an API
    response — never rendered, never framed, and any error page a
    downstream proxy generates from this response shouldn't leak a referrer.

    The request id is minted here, once, and put on `request.state` so
    `/api/chat` records the SAME id in the audit log and in MongoDB that the
    caller saw in `X-Request-ID`. Before this it minted its own separately,
    so a user reporting "request abc123 gave a wrong answer" was quoting an
    id that appeared in no record anywhere.
    """
    request.state.request_id = audit.AuditLog.new_request_id()
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", request.state.request_id)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Nothing internal reaches the client. The id ties the user's report to
    the stack trace in the log."""
    # getattr, not request.state.request_id: an exception raised *before* the
    # middleware sets it (or in a path that bypassed it) must still produce a
    # usable error response rather than an AttributeError inside the handler
    # that is itself supposed to be the last line of defence.
    request_id = getattr(request.state, "request_id", None) \
        or audit.AuditLog.new_request_id()
    log.exception("unhandled error [%s] on %s", request_id, request.url.path)
    detail = str(exc) if config.DEBUG else "internal error"
    return JSONResponse(status_code=500,
                        content={"error": detail, "request_id": request_id},
                        headers={"X-Request-ID": request_id})


class Question(BaseModel):
    # Bounded before any work happens: an unbounded question becomes an
    # unbounded prompt, which is both a cost and a context-overflow problem.
    question: str = Field(min_length=2, max_length=800,
                          description="A plain-English question about the Act "
                                      "or the Rules.")
    k: int = Field(default=6, ge=1, le=15,
                   description="How many provisions to retrieve before "
                               "answering.")
    conversation_id: str | None = Field(
        default=None, max_length=64,
        description="Groups turns of one conversation. Prior turns contribute "
                    "their questions and provision ids as extra retrieval "
                    "seeds — never their answers.")
    as_of: str | None = Field(
        default=None, pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO date. Provisions are flagged in force or not as of "
                    "this date. Defaults to today.")

    model_config = {"json_schema_extra": {"examples": [
        {"question": "what is the fine if customer data leaks?", "k": 6}]}}

    @field_validator("conversation_id")
    @classmethod
    def clean_conversation_id(cls, value: str | None) -> str | None:
        """Opaque to us, but it reaches a Mongo query — so constrain it to
        characters that cannot carry structure. Reads are scoped to the
        caller's own user id as well (see mongo.recent_turns), so this is
        defence in depth rather than the only guard."""
        if value is None:
            return None
        cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "-_")
        return cleaned or None

    @field_validator("question")
    @classmethod
    def clean(cls, value: str) -> str:
        # Control characters can corrupt the SSE framing and the audit log.
        text = "".join(ch for ch in value if ch == "\n" or ch >= " ").strip()
        if not text:
            raise ValueError("question is empty")
        return text


class Liveness(BaseModel):
    status: str = Field(description='Always "alive" if the process answered.')
    version: str


class Health(BaseModel):
    ok: bool = Field(description="False if the corpus or the model provider "
                                 "is unavailable. The HTTP status mirrors "
                                 "this: 200 when true, 503 when false.")
    detail: str
    provisions: int
    chunks: int
    build_id: str = Field(description="Content hash of the corpus that will "
                                      "answer. Changes when the Act or Rules "
                                      "are rebuilt.")
    tracing_detail: str
    provider: str
    model: str
    abstain_threshold: float
    max_context_chars: int
    tracing_enabled: bool
    features: dict = Field(description="Which V2 paths are live.")
    embed_model: str = ""
    embedding_detail: str = Field(
        default="", description="Empty when hybrid retrieval is off.")
    dense_vectors: int = 0
    rate_limit_scope: str = Field(
        default="", description='"process" — the limiter is in-memory and '
                                'does not coordinate across instances.')


class FrontendConfig(BaseModel):
    supabaseUrl: str
    supabaseAnonKey: str = Field(description="Public by Supabase's own "
                                             "design; grants only what "
                                             "row-level security allows.")


class Provision(BaseModel):
    id: str
    label: str
    kind: str
    headnote: str
    text: str = Field(description="The instrument's exact words, verbatim.")
    penalty: str
    page: int


class ServiceInfo(BaseModel):
    service: str
    version: str
    docs: str | None = Field(description="Null when docs are disabled "
                                         "(DPDP_DOCS=0).")


@app.get("/", tags=["meta"], summary="What this service is")
def root() -> ServiceInfo:
    """Service identity. Public, cheap, and safe as a platform health-check
    path — several hosts probe `/` by default, and a 404 there reads as a
    broken deployment even when the API underneath is fine."""
    return ServiceInfo(service="dpdp-compliance-assistant-api", version="2.0",
                       docs="/docs" if config.DOCS_ENABLED else None)


@app.get("/api/live", tags=["meta"], summary="Liveness probe")
def live() -> Liveness:
    """Is the process up? Nothing else.

    Deliberately separate from `/api/health`: this touches no dependency, so
    it is safe to poll every few seconds and it answers the only question a
    *liveness* probe should ask — "should this container be restarted?" A
    liveness probe that fails because Neo4j is briefly unreachable would
    restart a perfectly healthy process and fix nothing.
    """
    return Liveness(status="alive", version="2.0")

# @app.get("/health")
# def health():
#     return {"status": "ok"}

@app.get("/api/health", tags=["meta"], summary="Readiness probe",
         responses={503: {"model": Health,
                          "description": "Corpus or model provider "
                                         "unavailable — do not send traffic."}})
def health(response: Response) -> Health:
    """Is this instance ready to answer? Checks the corpus and the provider.

    Returns **503 when not ready**, not a 200 carrying `"ok": false`. Load
    balancers and platform health checks read the status code and ignore the
    body, so the old always-200 version reported a backend that had lost
    Neo4j as perfectly healthy and kept it in rotation.

    Use `/api/live` for liveness — this one makes a real call to the model
    provider and is the wrong thing to poll on a tight loop.
    """
    graph = STATE.get("graph")
    llm_error = llm.check()
    # Hybrid is degradable, not fatal: retrieval.  _seed falls back to BM25
    # when the embedder is unreachable, so an embedding outage lowers answer
    # quality but must not take the instance out of rotation.
    embed_error = embeddings.check() if config.HYBRID else None
    index = getattr(STATE.get("retriever"), "dense_index", None)

    ok = llm_error is None and graph is not None
    if not ok:
        response.status_code = 503
    return Health(
        ok=ok,
        detail=llm_error or ("ready" if graph else "corpus not loaded"),
        provisions=len(graph.provisions) if graph else 0,
        chunks=STATE.get("chunk_count", 0),
        build_id=graph.build_id if graph else "",
        tracing_detail=observability.check() or (
            "ready" if config.TRACING_ENABLED else "not configured"),
        embedding_detail=embed_error or (
            "ready" if config.HYBRID else "not configured"),
        dense_vectors=len(index) if index else 0,
        rate_limit_scope="process" if config.RATE_LIMIT > 0 else "disabled",
        **config.public_settings(),
    )


@app.get("/api/config", tags=["meta"], summary="Frontend bootstrap config")
def frontend_config() -> FrontendConfig:
    """The frontend's own startup config — public, no sign-in required.

    Replaces the marker-substitution `render_page()` used to do when this
    service also served the page directly. Now the frontend is a separate
    service with no Supabase credentials of its own; it fetches this once,
    before it can offer a sign-in button at all. `config.frontend_config()`
    is the same hand-written allow-list it always was — nothing becomes
    public here by being added to `config.py`, it has to be added to that
    allow-list explicitly.
    """
    return FrontendConfig(**config.frontend_config())


@app.get("/api/provision/{node_id}", tags=["corpus"],
         summary="One provision, verbatim",
         responses={404: {"description": "No such provision."}})
def provision(node_id: str,
              user: auth.Identity = Depends(auth.require_user)) -> Provision:
    """Verbatim text of one provision — what a citation click opens.

    Behind the same sign-in gate as `/api/chat`: without it the whole corpus
    is enumerable one provision at a time, which would make the gate on
    answering cosmetic.

    `node_id` is used only as a dictionary key against provisions loaded from
    Neo4j, so an unknown or hostile value can only ever miss and 404.
    """
    graph: graph_store.Graph = STATE["graph"]
    found = graph.provisions.get(node_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such provision")
    return Provision(id=found.id, label=found.label, kind=found.kind,
                     headnote=found.headnote, text=found.text,
                     penalty=found.penalty, page=found.page)


@app.get("/api/history", tags=["corpus"], summary="Your own past answers",
         responses={503: {"description": "History store unreachable."}})
def history_list(limit: int = Query(default=30, ge=1, le=50),
                 before: str | None = Query(
                     default=None,
                     description="ISO-8601 timestamp; returns answers older "
                                 "than this. Use the oldest `created_at` from "
                                 "the previous page."),
                 user: auth.Identity = Depends(auth.require_user)) -> list[dict]:
    """A signed-in user's own past answered questions, for the history panel.

    `user.sub` comes from the verified token, exactly as everywhere else in
    this file — never from `limit`/`before`, and there is no user-id
    parameter to tamper with. A caller cannot ask for anyone else's history;
    no endpoint accepts a client-supplied user id.
    """
    try:
        return mongo.list_for_user(user.sub, limit, before)
    except Exception:
        log.exception("history read failed")
        raise HTTPException(status_code=503, detail="could not load history")


def _sse(event: str, payload: dict) -> dict:
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


async def _stream_template(rendered, graph, results, plan, commencement,
                           as_of, elapsed_ms, finish, router_payload, root,
                           retrieval_payload):
    """Emit a template answer over the SAME event sequence as a model answer.

    It streams rather than arriving whole, and that is a deliberate cost: the
    text is already in memory, so this is pure UX. A UI that is instant for
    penalty questions and progressive for everything else reads as broken
    rather than fast, and the frontend would need a second rendering path to
    handle it.
    """
    for piece in rendered.chunks():
        yield _sse("token", {"t": piece})

    checked = citations.check_template(rendered.citations, graph)
    penalties = citations.penalty_facts(results, graph)
    citations_payload = {"citations": [c.to_dict() for c in checked],
                         "penalties": penalties}
    yield _sse("citations", citations_payload)

    # No numeric check here, and not an omission: every figure came out of
    # Provision.penalty or a verbatim text field, so it is sourced by
    # construction. Checking would only manufacture false positives from
    # formatting differences.
    claims_payload: list = []

    stale = commencement.not_yet_in_force(rendered.citations, as_of)
    if stale:
        claims_payload.append({
            "claim": "commencement",
            "verdict": "not_yet_in_force",
            "note": f"as of {as_of.isoformat()}, "
                    + ", ".join(citations.label_for(n) for n in stale)
                    + " has not commenced"})
    if plan.caveat:
        claims_payload.append({"claim": "scope", "verdict": "caveat",
                               "note": plan.caveat})
    if claims_payload:
        yield _sse("claims", claims_payload)

    done_payload = {
        "elapsed_ms": elapsed_ms(),
        "model": None,                 # explicitly: no model was involved
        "provider": "template",
        "path": "template",
        "intent": plan.intent,
        "build_id": graph.build_id,
        "context_chars": 0,
        "as_of": as_of.isoformat(),
    }
    yield _sse("done", done_payload)

    if root:
        root.update(output=rendered.text,
                    metadata={"path": "template", "intent": plan.intent})
    observability.flush()

    finish("answered", answer=rendered.text, intent=plan.intent,
           model=None, provider="template", elapsed_ms=elapsed_ms(),
           context_chars=0, router=router_payload, claims=claims_payload,
           citations=[c.to_dict() for c in checked],
           retrieval_payload=retrieval_payload,
           citations_payload=citations_payload, done_payload=done_payload)


@app.post("/api/chat", tags=["corpus"], summary="Ask a question (SSE stream)",
          response_class=EventSourceResponse,
          responses={200: {"content": {"text/event-stream": {}},
                           "description":
                               "A Server-Sent Events stream. Events arrive in "
                               "order: `retrieval` (which provisions were "
                               "found), then either `abstain` (out of scope, "
                               "no model call spent) or `token`×N (the answer, "
                               "as produced) followed by `citations` (each one "
                               "verified against the graph) and `done` "
                               "(timings, model, build id). `error` replaces "
                               "the remainder on failure."}})
async def chat(q: Question, request: Request,
               user: auth.Identity = Depends(auth.require_user)) -> EventSourceResponse:
    graph: graph_store.Graph = STATE["graph"]
    retriever: retrieval.Retriever = STATE["retriever"]
    trail: audit.AuditLog = STATE["audit"]
    limiter: ratelimit.RateLimiter = STATE["limiter"]
    commencement: temporal.Commencement = STATE["commencement"]
    classifier: understanding.Classifier = STATE["classifier"]

    # Before any work, and before the stream opens: a 429 delivered as an SSE
    # error event would be invisible to anything that reads status codes, and
    # a rate limiter that costs a retrieval pass to enforce is a poor one.
    allowed, remaining, retry_after = limiter.allow(user.sub)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit reached ({config.RATE_LIMIT} questions per "
                   f"hour). Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after),
                     "RateLimit-Limit": str(config.RATE_LIMIT),
                     "RateLimit-Remaining": "0"})

    as_of = temporal.resolve_as_of(q.as_of, config.AS_OF_DEFAULT)
    # The id the caller already saw in this response's X-Request-ID header,
    # so their bug report, the audit log and the Mongo record all name one id.
    request_id = getattr(request.state, "request_id", None) \
        or trail.new_request_id()
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    async def stream():
        base = {"request_id": request_id, "question": q.question,
                "build_id": graph.build_id,
                # Identity comes from a verified token, never from the request
                # body — the caller cannot claim to be someone else.
                # session_id is Supabase's own auth.sessions.id (see
                # auth.Identity) — the same value Langfuse groups traces by
                # below, so a conversation can be followed across all three
                # stores on one shared UUID.
                "user_id": user.sub, "email": user.email, "name": user.name,
                "session_id": user.session_id,
                "conversation_id": q.conversation_id or "",
                "as_of": as_of.isoformat()}

        def finish(outcome: str, *, retrieval_payload: dict | None = None,
                   citations_payload: dict | None = None,
                   done_payload: dict | None = None, **extra) -> None:
            """Close the request in both records.

            One helper rather than a `trail.write` and a `mongo.record` at
            each of the five terminal paths: the two stores answer different
            questions and must not drift apart, and a new outcome added later
            should be unable to land in one and miss the other.

            `retrieval_payload`/`citations_payload`/`done_payload` are the
            exact SSE event bodies already yielded to the browser — stored
            verbatim so `/api/history` can hand them back unchanged, and the
            frontend's replay feeds the same `renderTrace`/`renderSources`
            /`renderStatements`/`renderPenalties` functions the live stream
            already uses, with no second rendering path to maintain.
            """
            trail.write({**base, "outcome": outcome, **extra})
            mongo.record({
                "user_id": user.sub,
                "session_id": user.session_id,
                "conversation_id": q.conversation_id or "",
                "email": user.email,
                "name": user.name,
                "question": q.question,
                "outcome": outcome,
                "answer": extra.get("answer"),
                "retrieval": retrieval_payload,
                "citations": citations_payload,
                "done": done_payload,
                "router": extra.get("router"),
                "claims": extra.get("claims"),
                "elapsed_ms": elapsed_ms(),
                "build_id": graph.build_id,
            })
            # The conversation turn carries the question and the provisions it
            # reached — never the answer. See mongo.record_turn for why.
            if config.CONVERSATIONS and q.conversation_id:
                mongo.record_turn(
                    q.conversation_id, user.sub, q.question,
                    [p["id"] for p in (retrieval_payload or {}).get("provisions", [])],
                    extra.get("intent", ""))

        # One Langfuse trace per request, with a child observation per stage
        # — the same three stages as the SSE events above and the fields
        # audit.py writes, so a trace in the Langfuse UI, the network tab,
        # and a line in the local audit log describe one request the same
        # way. A no-op everywhere tracing isn't configured (observability.py)
        # — this function's control flow is identical whether it's on or off.
        with observability.trace(
                "compliance.answer", input=q.question,
                user_id=user.sub, session_id=user.session_id,
                metadata={"request_id": request_id, "k": q.k}) as root:

            # 0. Understand the question before retrieving anything.
            #    The query vector is computed ONCE here and reused for both
            #    routing and dense retrieval — with a hosted embedder that is
            #    a ~1s call, and paying it twice per question would be the
            #    single most expensive mistake available in this file.
            query_vector = None
            if config.INTENT_ROUTER or config.HYBRID:
                try:
                    query_vector = await asyncio.to_thread(
                        embeddings.embed_one, q.question, True)
                except Exception as exc:               # noqa: BLE001
                    log.warning("query embedding unavailable: %s", exc)

            plan = await asyncio.to_thread(
                understanding.understand, q.question, classifier, query_vector)

            router_payload = {
                "path": "template" if plan.uses_template else "llm",
                **plan.to_dict(),
            }

            # 0a. Foreign jurisdiction. Refused BEFORE retrieval — this is the
            #     documented V1 gap: a GDPR question scores as high on BM25 as
            #     a genuine one, so no score threshold can catch it. It is a
            #     scope judgement, not a confidence one.
            if plan.should_abstain:
                router_payload["path"] = "abstain"
                yield _sse("router", router_payload)
                finish("out_of_jurisdiction", intent=plan.intent,
                       router=router_payload,
                       reason=f"markers: {plan.jurisdiction_markers}")
                if root:
                    root.update(output="out_of_jurisdiction", level="WARNING")
                observability.flush()
                yield _sse("abstain", {
                    "message": "That question is about a different privacy "
                               "regime. This assistant only covers India's "
                               "Digital Personal Data Protection Act, 2023 and "
                               "its Rules, 2025.",
                    "reason": "out of jurisdiction: "
                              + ", ".join(plan.jurisdiction_markers or ["foreign law"])})
                return

            yield _sse("router", router_payload)

            # 0b. Prior turns contribute provisions, never prose.
            prior_ids: list[str] = []
            if config.CONVERSATIONS and q.conversation_id:
                turns = await asyncio.to_thread(
                    mongo.recent_turns, q.conversation_id, user.sub, 3)
                # Only when the question cannot stand alone. A self-contained
                # follow-up should retrieve on its own merits, or turn 3 keeps
                # dragging turn 1's provisions into an unrelated answer.
                if plan.has_anaphora:
                    prior_ids = [pid for t in turns for pid in t["provision_ids"]][:6]

            # 1. Retrieve. Sent immediately: it is fast, and showing the
            #    evidence before the argument is the whole trust model.
            with observability.step("retrieve", as_type="retriever",
                                    input=q.question) as retr_span:
                results, trace = await asyncio.to_thread(
                    retriever.retrieve, q.question, q.k, prior_ids)
                if retr_span:
                    retr_span.update(
                        output=[r.chunk.node_id for r in results],
                        metadata={"vocab_hits": trace.vocab_hits,
                                 "intents": trace.intents,
                                 "fused": trace.fused,
                                 "intent": plan.intent})

            # A direct lookup names its own provision, so an empty BM25 result
            # is irrelevant — the graph still has the node.
            if not results and not plan.provision_id:
                finish("no_results", intent=plan.intent, router=router_payload)
                if root:
                    root.update(output="no_results", level="WARNING")
                observability.flush()
                yield _sse("abstain", {
                    "message": "Nothing in this Act matches that question.",
                    "reason": "no provision scored above zero"})
                return

            retrieval_payload = {
                "elapsed_ms": elapsed_ms(),
                "build_id": graph.build_id,
                "vocabulary": trace.vocab_hits,
                "intents": trace.intents,
                "as_of": as_of.isoformat(),
                # Dense scores are surfaced, not hidden. V1's case for
                # vocab.yaml over embeddings was auditability; the honest
                # answer to "dense is a black box" is to show its working,
                # so a wrong dense hit is as inspectable as a wrong vocab hit.
                "fused": trace.fused,
                "dense_error": trace.dense_error,
                "provisions": [{
                    "id": r.chunk.node_id, "label": r.chunk.label, "kind": r.chunk.kind,
                    "headnote": r.chunk.headnote, "hop": r.hop,
                    "score": r.score, "via": r.via,
                    "bm25_rank": trace.bm25_ranks.get(r.chunk.node_id),
                    "dense_rank": trace.dense_ranks.get(r.chunk.node_id),
                    "dense_score": trace.dense_scores.get(r.chunk.node_id),
                    **commencement.annotate(r.chunk.node_id, as_of),
                } for r in results],
            }
            yield _sse("retrieval", retrieval_payload)

            # 1a. TEMPLATE PATH — the highest-value part of V2. Both errors
            #     this project has on record happened on questions whose
            #     answers were already structured data in the graph. Rendering
            #     them removes the model, and with it the failure mode.
            if plan.uses_template:
                rendered = await asyncio.to_thread(
                    templates.render, plan.intent, results, graph,
                    plan.provision_id)
                if rendered is not None:
                    async for event in _stream_template(
                            rendered, graph, results, plan, commencement,
                            as_of, elapsed_ms, finish, router_payload, root,
                            retrieval_payload):
                        yield event
                    return
                # The graph could not actually support the intent; fall
                # through to synthesis rather than render a guess.
                log.info("template %r declined; using synthesis", plan.intent)
                router_payload["path"] = "llm"
                router_payload["template_declined"] = True

            # 2. Abstain before spending a generation call on an out-of-scope
            #    question — deterministic, not left to the model's judgement.
            if reason := retrieval.should_abstain(results, config.ABSTAIN_THRESHOLD):
                finish("abstained", reason=reason, retrieval_payload=retrieval_payload)
                if root:
                    root.update(output=f"abstained: {reason}", level="WARNING")
                observability.flush()
                yield _sse("abstain", {
                    "message": "This doesn't look like something the Digital Personal "
                               "Data Protection Act, 2023 covers. The closest match was "
                               "too weak to answer from — try rephrasing, or this may "
                               "be outside the Act's scope.",
                    "reason": reason})
                return

            if error := llm.check():
                finish("llm_unavailable", error=error, retrieval_payload=retrieval_payload)
                if root:
                    root.update(output=f"llm_unavailable: {error}", level="ERROR")
                observability.flush()
                yield _sse("error", {"message": error})
                return

            # 3. Generate. The blocking provider call runs on a worker thread
            #    and feeds this coroutine through a queue, so the event loop
            #    keeps serving other requests while one answer streams.
            context = retrieval.build_context(results, config.MAX_CONTEXT_CHARS)
            # Node ids are given to the model explicitly so structured output
            # can cite by id. Without this the model has labels ("section
            # 8(5)") but not the ids the verifier checks membership against.
            id_list = "\n".join(
                f"  {r.chunk.node_id}  = {r.chunk.label}" for r in results)
            prompt = (f"{context}\n\nProvision ids you may cite:\n{id_list}"
                      f"\n\nQuestion: {q.question}")

            use_large = llm.needs_large_model(q.question, len(results), plan.intent)
            active_model = llm.model_name(use_large)
            router_payload.update({"model": active_model,
                                   "large": use_large,
                                   "structured": config.STRUCTURED_OUTPUT})

            system = SYSTEM_PROMPT
            if config.STRUCTURED_OUTPUT:
                system = f"{SYSTEM_PROMPT}\n{schema.STRUCTURED_INSTRUCTION}"

            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def produce() -> None:
                try:
                    if config.STRUCTURED_OUTPUT:
                        source = llm.stream_structured(
                            prompt, system, schema.json_schema(), large=use_large)
                    elif use_large:
                        source = llm.large_provider().stream(prompt, system, 0.1)
                    else:
                        source = llm.stream(prompt, system)
                    for fragment in source:
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", fragment))
                except llm.LLMError as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                except Exception:                    # noqa: BLE001
                    log.exception("generation failed [%s]", request_id)
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("error", "the answer could not be generated"))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

            with observability.step("generate", as_type="generation",
                                    model=active_model, input=prompt) as gen_span:
                loop.run_in_executor(None, produce)

                # Extracts the `answer` field out of the JSON prefix as it
                # arrives, so structured output still streams token by token.
                # See streaming.py — this is what keeps V2 from regressing the
                # one thing V1's UX got right.
                extractor = streaming.AnswerStream()
                parts: list[str] = []

                while True:
                    kind, payload = await queue.get()
                    if kind == "eof":
                        break
                    if kind == "error":
                        finish("generation_error", error=payload,
                              intent=plan.intent, router=router_payload,
                              retrieval_payload=retrieval_payload)
                        if gen_span:
                            gen_span.update(level="ERROR", status_message=payload)
                        if root:
                            root.update(output=f"generation_error: {payload}",
                                       level="ERROR")
                        observability.flush()
                        yield _sse("error", {"message": payload})
                        return

                    if config.STRUCTURED_OUTPUT:
                        for piece in extractor.feed(payload):
                            parts.append(piece)
                            yield _sse("token", {"t": piece})
                    else:
                        parts.append(payload)
                        yield _sse("token", {"t": payload})

                if config.STRUCTURED_OUTPUT:
                    parsed = extractor.finish()
                    answer = parsed["answer"]
                    claimed_citations = parsed["citations"]
                    structured_ok = parsed["structured"]
                    if not structured_ok:
                        # The model ignored the schema. Degrade to the V1
                        # regex path rather than lose an answer it did produce.
                        router_payload["structured_failed"] = True
                        log.warning("structured output failed [%s]; using "
                                    "regex citation path", request_id)
                        # Anything not already streamed still has to reach the
                        # client, or the user sees a truncated answer.
                        remainder = answer[len("".join(parts)):] \
                            if answer.startswith("".join(parts)) else answer
                        if remainder:
                            yield _sse("token", {"t": remainder})
                else:
                    answer = "".join(parts)
                    claimed_citations = []
                    structured_ok = False

                if gen_span:
                    gen_span.update(output=answer)

            # 4. Check what it cited, and render amounts from the graph.
            with observability.step("verify_citations", as_type="evaluator",
                                    input=answer) as verify_span:
                retrieved_ids = {r.chunk.node_id for r in results}
                # Set membership when the model emitted ids, regex otherwise.
                # The structured path cannot be defeated by citation phrasing;
                # the regex path is kept as a live fallback, not as legacy.
                checked = (citations.check_structured(
                               claimed_citations, retrieved_ids, graph)
                           if structured_ok and claimed_citations
                           else citations.check(answer, retrieved_ids, graph))
                penalties = citations.penalty_facts(results, graph)
                if verify_span:
                    verify_span.update(
                        output=[{"id": c.id, "status": c.status} for c in checked])

            citations_payload = {
                "citations": [c.to_dict() for c in checked],
                "penalties": penalties}
            yield _sse("citations", citations_payload)

            # 5. Every number in the answer must exist in the evidence.
            #    Deterministic, and what actually catches the Schedule-figure
            #    class of error — no NLI model required.
            claims_payload: list = []
            if config.NUMERIC_CHECK:
                evidence = [r.chunk.verbatim for r in results]
                amounts = [graph.provisions[r.chunk.node_id].penalty
                           for r in results
                           if r.chunk.node_id in graph.provisions
                           and graph.provisions[r.chunk.node_id].penalty]
                numeric_claims = await asyncio.to_thread(
                    numeric.check, answer, evidence, amounts, q.question)
                claims_payload.extend(c.to_dict() for c in numeric_claims)
                if numeric.has_contradiction(numeric_claims):
                    log.warning("unsupported figure in answer [%s]: %s",
                                request_id,
                                [c.surface for c in numeric_claims
                                 if c.verdict == "unsupported"])

            stale = commencement.not_yet_in_force(
                [c.id for c in checked], as_of)
            if stale:
                claims_payload.append({
                    "claim": "commencement", "verdict": "not_yet_in_force",
                    "note": f"as of {as_of.isoformat()}, "
                            + ", ".join(citations.label_for(n) for n in stale)
                            + " has not commenced"})
            if plan.caveat:
                claims_payload.append({"claim": "scope", "verdict": "caveat",
                                       "note": plan.caveat})
            if claims_payload:
                yield _sse("claims", claims_payload)

            done_payload = {
                "elapsed_ms": elapsed_ms(),
                "model": active_model,
                "provider": config.LARGE_PROVIDER if use_large else config.PROVIDER,
                "path": "llm",
                "intent": plan.intent,
                "structured": structured_ok,
                "build_id": graph.build_id,
                "as_of": as_of.isoformat(),
                "context_chars": len(prompt)}
            yield _sse("done", done_payload)

            if root:
                root.update(output=answer, metadata={
                    "citation_statuses": [c.status for c in checked],
                    "path": "llm", "intent": plan.intent,
                    "model": active_model, "structured": structured_ok,
                    "elapsed_ms": elapsed_ms()})
            observability.flush()

            finish(
                "answered",
                model=active_model,
                provider=config.LARGE_PROVIDER if use_large else config.PROVIDER,
                intent=plan.intent,
                router=router_payload,
                claims=claims_payload,
                elapsed_ms=elapsed_ms(),
                context_chars=len(prompt),
                retrieved=[{"id": r.chunk.node_id, "hop": r.hop, "score": r.score}
                           for r in results],
                answer=answer,
                citations=[{"id": c.id, "status": c.status} for c in checked],
                retrieval_payload=retrieval_payload,
                citations_payload=citations_payload,
                done_payload=done_payload,
            )

    return EventSourceResponse(stream())
