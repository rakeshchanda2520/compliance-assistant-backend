"""
The HTTP layer. API only — this service serves no HTML.

Thin by design. All the reasoning lives in `answering.engine`, which yields
typed events and knows nothing about HTTP; this module turns those into SSE
frames, applies auth and rate limiting, and records the outcome. The previous
system interleaved control flow, framing, audit writes and tracing in one
1,100-line handler where the order of operations could only be established by
reading all of it.

Security posture:
  * Answering requires a verified sign-in. Only `/`, `/api/live`,
    `/api/health` and `/api/config` are public — the last by necessity, since
    the frontend has no credential with which to fetch its own configuration.
  * No internal detail in responses. Exceptions are logged with a request id;
    the client gets that id and a generic message unless DEBUG is on.
  * Request bodies are bounded by Pydantic before any work is done.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

import config
from answering import schema
from answering.engine import Deps, Engine
from api import auth
from api.ratelimit import RateLimiter
from core.corpus import load_chunks, verify_against_graph, verify_against_index
from providers import embeddings, graphdb, llm, observability, store
from retrieval import dense, rerank
from retrieval.pipeline import Retriever
from understanding.router import Router
from verification import temporal

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("dpdp")

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything once, and fail loudly HERE rather than on request one."""
    corpus = load_chunks(config.DATA_DIR / "chunks.json")
    vocab = yaml.safe_load((config.DATA_DIR / "vocab.yaml").read_text(encoding="utf-8"))
    graph = graphdb.load(config, corpus)

    if not graph.degraded:
        verify_against_graph(corpus, graph.provisions)

    index = dense.load(config.DATA_DIR / "embeddings.npz",
                       expect_model=config.EMBED_MODEL) if config.HYBRID else None
    if index is not None:
        verify_against_index(corpus, index.node_ids)

    STATE.update(
        corpus=corpus,
        graph=graph,
        commencement=temporal.load(config.DATA_DIR / "commencement.yaml"),
        limiter=RateLimiter(),
        retriever=Retriever(corpus, edges=graph.edges, dense_index=index,
                            vocab=vocab, settings=config),
        router=Router(exemplars=index.exemplars if index else None,
                      intent_floor=config.INTENT_MIN_COSINE,
                      jurisdiction_floor=config.JURISDICTION_MIN_COSINE,
                      jurisdiction_margin=config.JURISDICTION_MARGIN),
        reranker=rerank.build(config),
        embed_query=embeddings.query_embedder(index.dims if index else 0),
        store_ready=store.connect(),
    )

    log.info("ready: %d chunks, %d provisions, build %s%s",
             len(corpus), len(graph.provisions), corpus.build_id,
             " (GRAPH DEGRADED)" if graph.degraded else "")
    if not config.REQUIRE_AUTH:
        log.warning("DPDP_REQUIRE_AUTH is off — every request is treated as a "
                    "local development user. Never do this in production.")
    yield
    STATE.clear()


app = FastAPI(
    title="DPDP Compliance Assistant",
    version="1.0",
    lifespan=lifespan,
    summary="Answers questions on India's DPDP Act, 2023 and Rules, 2025, "
            "quoting the statute verbatim with every citation verified "
            "against a knowledge graph.",
    docs_url="/docs" if config.DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if config.DOCS else None)

if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        # "Authorization" is required, not optional: a cross-origin request
        # carrying a custom header triggers a preflight, and a preflight the
        # server does not allow it on is rejected before the handler — or
        # require_user — ever runs. Every authenticated route would fail while
        # /api/health kept working, which is a confusing way to discover it.
        allow_headers=["Content-Type", "Authorization"])


@app.middleware("http")
async def request_id_and_headers(request: Request, call_next):
    """One id per request, minted here and reused everywhere.

    The client sees it in `X-Request-ID`, and the same id goes into the store
    and the trace — so "request abc123 gave a wrong answer" names something
    findable rather than an id that appears nowhere.
    """
    request.state.request_id = uuid.uuid4().hex[:16]
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", request.state.request_id)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "") or uuid.uuid4().hex[:16]
    log.exception("unhandled error [%s] on %s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc) if config.DEBUG else "internal error",
                 "request_id": request_id},
        headers={"X-Request-ID": request_id})


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

class Question(BaseModel):
    # Bounded before any work happens: an unbounded question becomes an
    # unbounded prompt, which is both a cost and a context-overflow problem.
    question: str = Field(min_length=2, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=64)
    as_of: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("question")
    @classmethod
    def clean(cls, value: str) -> str:
        # Control characters can corrupt SSE framing and the audit log.
        text = "".join(c for c in value if c == "\n" or c >= " ").strip()
        if not text:
            raise ValueError("question is empty")
        return text

    @field_validator("conversation_id")
    @classmethod
    def clean_conversation(cls, value: str | None) -> str | None:
        """Opaque to us, but it reaches a database query — constrain it to
        characters that cannot carry structure. Reads are scoped to the
        caller's own user id as well, so this is defence in depth."""
        if value is None:
            return None
        return "".join(c for c in value if c.isalnum() or c in "-_") or None


# --------------------------------------------------------------------------- #
# public routes
# --------------------------------------------------------------------------- #

@app.get("/", tags=["meta"])
def root() -> dict:
    """Service identity. Several platforms probe `/` by default and a 404
    there reads as a broken deployment even when the API is fine."""
    return {"service": "dpdp-compliance-assistant", "version": "1.0",
            "docs": "/docs" if config.DOCS else None}


@app.get("/api/live", tags=["meta"])
def live() -> dict:
    """Is the process up? Nothing else.

    Separate from `/api/health` on purpose: this touches no dependency, so it
    is safe to poll every few seconds and answers the only question a LIVENESS
    probe should ask. A liveness probe that fails because Neo4j blinked would
    restart a healthy process and fix nothing.
    """
    return {"status": "alive", "version": "1.0"}


@app.get("/api/health", tags=["meta"])
def health(response: Response) -> dict:
    """Readiness. Returns 503 when not ready, NOT a 200 carrying ok:false —
    load balancers read the status code and ignore the body."""
    corpus = STATE.get("corpus")
    graph = STATE.get("graph")
    llm_error = llm.check()
    ready = corpus is not None and graph is not None and llm_error is None
    if not ready:
        response.status_code = 503
    return {
        "ok": ready,
        "detail": llm_error or ("ready" if corpus else "corpus not loaded"),
        "chunks": len(corpus) if corpus else 0,
        "provisions": len(graph.provisions) if graph else 0,
        "build_id": corpus.build_id if corpus else "",
        "graph_degraded": bool(graph and graph.degraded),
        "graph_detail": graph.detail if graph else "",
        "embedding_detail": embeddings.check() or "ready",
        "tracing_detail": observability.check() or (
            "ready" if config.TRACING else "not configured"),
        "store": "ready" if STATE.get("store_ready") else "unavailable",
        "auth_required": config.REQUIRE_AUTH,
        "reranker": "on" if STATE.get("reranker") else "off",
        **config.public_settings(),
    }


@app.get("/api/config", tags=["meta"])
def frontend_config() -> dict:
    """The frontend's own bootstrap config. Public by necessity — the frontend
    holds no Supabase credential of its own with which to fetch it."""
    return config.frontend_config()


# --------------------------------------------------------------------------- #
# corpus routes
# --------------------------------------------------------------------------- #

@app.get("/api/provision/{node_id}", tags=["corpus"])
def provision(node_id: str,
              user: auth.Identity = Depends(auth.require_user)) -> dict:
    """One provision, verbatim — what a citation click opens.

    Behind the same gate as /api/chat: without it the whole corpus is
    enumerable one provision at a time, which makes the gate on answering
    cosmetic. `node_id` is only ever a dictionary key, so a hostile value can
    only miss and 404.
    """
    graph = STATE["graph"]
    found = graph.provisions.get(node_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such provision")
    from verification.citations import display_text
    return {"id": found.id, "label": found.label, "kind": found.kind,
            "headnote": found.headnote, "text": display_text(node_id, graph),
            "penalty": found.penalty, "page": found.page}


@app.get("/api/history", tags=["corpus"])
def history(limit: int = Query(default=30, ge=1, le=50),
            before: str | None = Query(default=None),
            user: auth.Identity = Depends(auth.require_user)) -> list[dict]:
    """A signed-in user's own past answers.

    `user.sub` comes from the verified token, exactly as everywhere else —
    there is no user-id parameter to tamper with, so a caller cannot request
    anyone else's history.
    """
    try:
        return store.history(user.sub, limit, before)
    except Exception:
        log.exception("history read failed")
        raise HTTPException(status_code=503, detail="could not load history")


# --------------------------------------------------------------------------- #
# the answer stream
# --------------------------------------------------------------------------- #

def _sse(name: str, payload: dict) -> dict:
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}


@app.post("/api/chat", tags=["corpus"], response_class=EventSourceResponse)
async def chat(q: Question, request: Request,
               user: auth.Identity = Depends(auth.require_user)):
    """Ask a question. Streams Server-Sent Events.

    Events arrive in order: `router` (how it was routed — sent again if the
    route changes), `retrieval`, then either `abstain` or `token`xN followed
    by `citations`, `claims` and `done`. `error` replaces the remainder on
    failure.
    """
    limiter: RateLimiter = STATE["limiter"]
    allowed, remaining, retry_after = limiter.allow(user.sub)
    if not allowed:
        # Before the stream opens, so it is a real status code. A 429
        # delivered as an SSE event is invisible to anything reading statuses.
        raise HTTPException(
            status_code=429,
            detail=f"rate limit reached ({limiter.limit_for(user.sub)} "
                   f"questions per hour). Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after),
                     "RateLimit-Limit": str(limiter.limit_for(user.sub)),
                     "RateLimit-Remaining": "0"})

    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
    as_of = temporal.resolve_as_of(q.as_of, config.AS_OF_DEFAULT)
    conversation = q.conversation_id or ""
    started = time.perf_counter()

    corpus = STATE["corpus"]
    graph = STATE["graph"]

    deps = Deps(corpus=corpus, graph=graph, retriever=STATE["retriever"],
                router=STATE["router"], settings=config)
    deps.embed_query = STATE.get("embed_query")
    deps.reranker = STATE.get("reranker")
    deps.synthesize = llm.synthesizer(schema.json_schema())
    if config.CONVERSATIONS and conversation:
        deps.prior_provisions = lambda cid: store.prior_provisions(
            cid, user.sub, config.CONVERSATION_TURNS)

    engine = Engine(deps)

    async def stream():
        # `session_id` is the CONVERSATION thread, not the sign-in — that is
        # the grouping Langfuse's Sessions view is built for.
        with observability.trace(
                "compliance.answer", input=q.question,
                user_id=user.sub,
                session_id=conversation or user.session_id or request_id,
                tags=[f"provider:{config.PROVIDER}", f"build:{corpus.build_id}"],
                metadata={"request_id": request_id,
                          "auth_session_id": user.session_id,
                          "conversation_id": conversation,
                          "as_of": as_of.isoformat()}) as root:

            record: dict = {"outcome": "incomplete", "answer": "", "path": ""}
            answer_parts: list[str] = []
            provisions: list[str] = []

            try:
                # The engine is synchronous and CPU-bound in places; running it
                # on a worker thread keeps the event loop serving other
                # requests while one answer streams.
                for event in await asyncio.to_thread(
                        lambda: list(engine.answer(
                            q.question, conversation_id=conversation,
                            as_of=as_of))):
                    if event.name == "token":
                        answer_parts.append(event.data.get("t", ""))
                    elif event.name == "retrieval":
                        provisions = [p["id"] for p in event.data.get("provisions", ())]
                    elif event.name in ("done", "abstain", "error"):
                        record["outcome"] = (
                            "answered" if event.name == "done" else event.name)
                        record["path"] = event.data.get("path", "")
                    record[event.name] = event.data
                    yield _sse(event.name, event.data)

            except Exception:                              # noqa: BLE001
                log.exception("answer failed [%s]", request_id)
                yield _sse("error", {"message": "the answer could not be "
                                                "generated", "request_id": request_id})
                record["outcome"] = "error"

            answer = "".join(answer_parts)
            if root:
                root.update(output=answer or record["outcome"],
                            metadata={"path": record.get("path")})
            observability.flush()

            # Both stores written from ONE place, so a newly added outcome
            # cannot land in one and miss the other.
            store.record({
                "request_id": request_id, "user_id": user.sub,
                "session_id": user.session_id, "conversation_id": conversation,
                "email": user.email, "name": user.name,
                "question": q.question, "outcome": record["outcome"],
                "path": record.get("path"), "answer": answer,
                # The LAST router event, not the first: a template that
                # declined and a sufficiency abstain both re-announce the
                # route, and only the last one is true. Stored because
                # /api/history replays it, and "answered from the graph, with
                # no model involved" is the strongest trust signal an answer
                # carries — a replay that drops it is less trustworthy than
                # the live view of the same answer.
                "router": record.get("router"),
                "retrieval": record.get("retrieval"),
                "citations": record.get("citations"),
                "claims": record.get("claims"), "done": record.get("done"),
                "build_id": corpus.build_id, "as_of": as_of.isoformat(),
                "elapsed_ms": int((time.perf_counter() - started) * 1000)})

            if config.CONVERSATIONS and conversation:
                store.record_turn(conversation, user.sub, q.question,
                                  provisions,
                                  (record.get("router") or {}).get("intent", ""))

    return EventSourceResponse(stream())
