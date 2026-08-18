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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from . import (audit, auth, citations, config, graph_store, llm, mongo,
               observability, retrieval)
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

    STATE["graph"] = graph
    STATE["retriever"] = retrieval.Retriever(chunks, graph, vocab)
    STATE["audit"] = audit.AuditLog(config.LOG_DIR)
    STATE["chunk_count"] = len(chunks)

    log.info("ready: %d provisions, %d chunks, build %s, model %s/%s",
             len(graph.provisions), len(chunks), graph.build_id,
             config.PROVIDER, config.MODEL)
    if llm.is_small_model():
        log.warning("%s is a small model; it has misread statute in this "
                    "corpus. Prefer a larger model for real use.", config.MODEL)
    yield
    STATE.clear()


app = FastAPI(title="DPDP Compliance Assistant", version="2.0",
              lifespan=lifespan,
              # No interactive docs in production: they enumerate the API
              # surface for anyone who finds the port.
              docs_url="/docs" if config.DEBUG else None,
              redoc_url=None,
              openapi_url="/openapi.json" if config.DEBUG else None)

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
    """Headers that cost nothing and close real classes of attack.

    This service serves no HTML — the frontend is a separate service
    (`frontend/server.py`), which carries its own CSP scoped to what a
    *page* needs (Supabase's client bundle, Google's consent screen, and so
    on). None of that protects anything here: a JSON API has no inline
    script to restrict and nothing for a Content-Security-Policy to narrow,
    so this only sets the headers that still mean something for an API
    response — never rendered, never framed, and any error page a
    downstream proxy generates from this response shouldn't leak a referrer.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Nothing internal reaches the client. The id ties the user's report to
    the stack trace in the log."""
    request_id = audit.AuditLog.new_request_id()
    log.exception("unhandled error [%s] on %s", request_id, request.url.path)
    detail = str(exc) if config.DEBUG else "internal error"
    return JSONResponse(status_code=500,
                        content={"error": detail, "request_id": request_id})


class Question(BaseModel):
    # Bounded before any work happens: an unbounded question becomes an
    # unbounded prompt, which is both a cost and a context-overflow problem.
    question: str = Field(min_length=2, max_length=800)
    k: int = Field(default=6, ge=1, le=15)

    @field_validator("question")
    @classmethod
    def clean(cls, value: str) -> str:
        # Control characters can corrupt the SSE framing and the audit log.
        text = "".join(ch for ch in value if ch == "\n" or ch >= " ").strip()
        if not text:
            raise ValueError("question is empty")
        return text


@app.get("/api/health")
def health() -> dict:
    graph = STATE.get("graph")
    llm_error = llm.check()
    return {
        "ok": llm_error is None and graph is not None,
        "detail": llm_error or "ready",
        "provisions": len(graph.provisions) if graph else 0,
        "chunks": STATE.get("chunk_count", 0),
        "build_id": graph.build_id if graph else "",
        "tracing_detail": observability.check() or (
            "ready" if config.TRACING_ENABLED else "not configured"),
        **config.public_settings(),
    }


@app.get("/api/config")
def frontend_config() -> dict:
    """The frontend's own startup config — public, no sign-in required.

    Replaces the marker-substitution `render_page()` used to do when this
    service also served the page directly. Now the frontend is a separate
    service with no Supabase credentials of its own; it fetches this once,
    before it can offer a sign-in button at all. `config.frontend_config()`
    is the same hand-written allow-list it always was — nothing becomes
    public here by being added to `config.py`, it has to be added to that
    allow-list explicitly.
    """
    return config.frontend_config()


@app.get("/api/provision/{node_id}")
def provision(node_id: str,
              user: auth.Identity = Depends(auth.require_user)) -> dict:
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
    return {"id": found.id, "label": found.label, "kind": found.kind,
            "headnote": found.headnote, "text": found.text,
            "penalty": found.penalty, "page": found.page}


@app.get("/api/history")
def history_list(limit: int = 30, before: str | None = None,
                 user: auth.Identity = Depends(auth.require_user)) -> list[dict]:
    """A signed-in user's own past answered questions, for the history panel.

    `user.sub` comes from the verified token, exactly as everywhere else in
    this file — never from `limit`/`before`, and there is no user-id
    parameter to tamper with. A caller cannot ask for anyone else's history;
    no endpoint accepts a client-supplied user id.
    """
    try:
        return mongo.list_for_user(user.sub, min(limit, 50), before)
    except Exception:
        log.exception("history read failed")
        raise HTTPException(status_code=503, detail="could not load history")


def _sse(event: str, payload: dict) -> dict:
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


@app.post("/api/chat")
async def chat(q: Question,
               user: auth.Identity = Depends(auth.require_user)) -> EventSourceResponse:
    graph: graph_store.Graph = STATE["graph"]
    retriever: retrieval.Retriever = STATE["retriever"]
    trail: audit.AuditLog = STATE["audit"]
    request_id = trail.new_request_id()
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    async def stream():
        base = {"request_id": request_id, "question": q.question,
                "build_id": graph.build_id,
                # Identity comes from a verified token, never from the request
                # body — the caller cannot claim to be someone else.
                "user_id": user.sub, "email": user.email, "name": user.name}

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
                "email": user.email,
                "name": user.name,
                "question": q.question,
                "outcome": outcome,
                "answer": extra.get("answer"),
                "retrieval": retrieval_payload,
                "citations": citations_payload,
                "done": done_payload,
                "elapsed_ms": elapsed_ms(),
                "build_id": graph.build_id,
            })

        # One Langfuse trace per request, with a child observation per stage
        # — the same three stages as the SSE events above and the fields
        # audit.py writes, so a trace in the Langfuse UI, the network tab,
        # and a line in the local audit log describe one request the same
        # way. A no-op everywhere tracing isn't configured (observability.py)
        # — this function's control flow is identical whether it's on or off.
        with observability.trace(
                "compliance.answer", input=q.question,
                metadata={"request_id": request_id, "k": q.k}) as root:

            # 1. Retrieve. Sent immediately: it is fast, and showing the
            #    evidence before the argument is the whole trust model.
            with observability.step("retrieve", as_type="retriever",
                                    input=q.question) as retr_span:
                results, trace = await asyncio.to_thread(
                    retriever.retrieve, q.question, q.k)
                if retr_span:
                    retr_span.update(
                        output=[r.chunk.node_id for r in results],
                        metadata={"vocab_hits": trace.vocab_hits,
                                 "intents": trace.intents})

            if not results:
                finish("no_results")
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
                "provisions": [{
                    "id": r.chunk.node_id, "label": r.chunk.label, "kind": r.chunk.kind,
                    "headnote": r.chunk.headnote, "hop": r.hop,
                    "score": r.score, "via": r.via,
                } for r in results],
            }
            yield _sse("retrieval", retrieval_payload)

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
            prompt = f"{context}\n\nQuestion: {q.question}"
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def produce() -> None:
                try:
                    for fragment in llm.stream(prompt, SYSTEM_PROMPT):
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
                                    model=config.MODEL, input=prompt) as gen_span:
                loop.run_in_executor(None, produce)

                parts: list[str] = []
                while True:
                    kind, payload = await queue.get()
                    if kind == "eof":
                        break
                    if kind == "error":
                        finish("generation_error", error=payload,
                              retrieval_payload=retrieval_payload)
                        if gen_span:
                            gen_span.update(level="ERROR", status_message=payload)
                        if root:
                            root.update(output=f"generation_error: {payload}",
                                       level="ERROR")
                        observability.flush()
                        yield _sse("error", {"message": payload})
                        return
                    parts.append(payload)
                    yield _sse("token", {"t": payload})

                answer = "".join(parts)
                if gen_span:
                    gen_span.update(output=answer)

            # 4. Check what it cited, and render amounts from the graph.
            with observability.step("verify_citations", as_type="evaluator",
                                    input=answer) as verify_span:
                retrieved_ids = {r.chunk.node_id for r in results}
                checked = citations.check(answer, retrieved_ids, graph)
                penalties = citations.penalty_facts(results, graph)
                if verify_span:
                    verify_span.update(
                        output=[{"id": c.id, "status": c.status} for c in checked])

            citations_payload = {
                "citations": [c.to_dict() for c in checked],
                "penalties": penalties}
            yield _sse("citations", citations_payload)

            done_payload = {
                "elapsed_ms": elapsed_ms(),
                "model": config.MODEL,
                "provider": config.PROVIDER,
                "build_id": graph.build_id,
                "context_chars": len(prompt)}
            yield _sse("done", done_payload)

            if root:
                root.update(output=answer, metadata={
                    "citation_statuses": [c.status for c in checked],
                    "elapsed_ms": elapsed_ms()})
            observability.flush()

            finish(
                "answered",
                model=config.MODEL,
                provider=config.PROVIDER,
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
