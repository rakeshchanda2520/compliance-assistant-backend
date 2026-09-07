# Backend — how to run it

The API. Answers questions about the DPDP Act, 2023 and the Rules, 2025 over
Server-Sent Events, with every citation checked against a knowledge graph.

See `FLOW.md` for what happens inside a request.

---

## 1. Before you start

You need **Python 3.11+**.

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill it in — see §2
```

The corpus (`data/chunks.json`, `data/embeddings.npz`) is **already here**. It
is built offline by `KG_creation/` and committed, not generated at boot. You
do not need to run the KG build to start the backend.

---

## 2. Configuration

Only two settings are genuinely required to answer a question:

```bash
DPDP_PROVIDER=openrouter
DPDP_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=sk-...
```

Everything else has a working default or degrades gracefully.

| Variable | Needed for | If missing |
|---|---|---|
| `OPENROUTER_API_KEY` | generated answers | template answers still work; others report no provider |
| `DPDP_EMBED_MODEL` | dense search | **must match what built `embeddings.npz`** — see the warning below |
| `NEO4J_URI` / `_USER` / `_PASSWORD` | the full graph | falls back to a graph rebuilt from `chunks.json`; logs a loud warning |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` | sign-in | set `DPDP_REQUIRE_AUTH=0` to run open, for local work only |
| `MONGODB_URI` | history | answers still stream; nothing is recorded |
| `DPDP_CORS_ORIGINS` | **the frontend** | see the warning below |
| `LANGFUSE_*` | tracing | skipped |

### Two settings that fail confusingly if wrong

> **`DPDP_CORS_ORIGINS` is required, not optional.**
> The frontend is a separate service, so it is never same-origin. If this is
> empty the CORS middleware is not installed at all, and the browser blocks
> every call — with **nothing in the backend's log**, because the request
> never arrives. Set it to the frontend's origin:
> `DPDP_CORS_ORIGINS=http://localhost:3100`

> **`DPDP_EMBED_MODEL` must match the model that built `embeddings.npz`.**
> Cosine similarity between vectors from two different models is a number; it
> is just a meaningless one, and nothing downstream can tell. The file stores
> its own fingerprint and the backend refuses to start on a mismatch — that
> refusal is the feature.

---

## 3. Run it

```bash
uvicorn api.app:app --port 8100
```

Run from **inside `backend/`**. Healthy startup looks like:

```
INFO  dpdp: ready: 237 chunks, 237 provisions, build 760b8bd2796d
INFO  Uvicorn running on http://127.0.0.1:8100
```

With `--reload` for development. In production, add workers:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8100 --workers 4
```

### Startup lines worth reading

| Line | Meaning |
|---|---|
| `dense index: 237 vectors, 1024 dims, 45 exemplars` | dense search is on |
| `Neo4j unavailable ... falling back to a graph rebuilt from chunks.json` | **degraded.** Search and citations work; penalty and cross-reference answers are incomplete |
| `no MONGODB_URI` | nothing will be recorded and history will be empty |
| `contextual retrieval is enabled but no chunk carries a context field` | expected — the optional context layer is not built yet; raw BM25 is used |

---

## 3b. Run it in Docker

```bash
docker build -t regrock-backend .
docker run -d --name regrock \
  -p 8100:8100 \
  --env-file .env \
  regrock-backend
```

Verified: image **447 MB**, two workers, Docker reports `healthy`, answers
stream, and `docker stop` completes in **2 seconds** (a clean SIGTERM, not a
10-second SIGKILL).

### What the image does and does not contain

```
   IN                                OUT (see .dockerignore)
   ──                                ───
   the code                          .env          ← credentials
   data/chunks.json                  .git          ← history, and any secret
   data/embeddings.npz                                ever committed to it
   data/vocab.yaml                   tests/ eval/ scripts/
   data/commencement.yaml            *.md
   the virtualenv                    logs/
```

> **The corpus is baked in, not mounted and not generated at build time.**
> Both alternatives were tried and both fail on any host without a shared
> filesystem — `RuntimeError: search index missing at chunks.json` on every
> boot. Updating the law means rebuilding the image, which is correct: the
> corpus *is* a version of this service, and `build_id` makes that visible in
> every answer.

> **`.dockerignore` is the only thing keeping `.env` out of a layer.** Deleting
> a line in it is a security change, not a cleanup. Layers are additive — a
> credential copied in stays readable to anyone who can pull the image even if
> a later layer deletes it. Configuration is injected at run time.

### Three deliberate choices

| Choice | Why |
|---|---|
| `HEALTHCHECK` hits **`/api/live`**, not `/api/health` | `/api/health` returns 503 when Neo4j is unreachable. That is right for a load balancer (stop sending traffic) and wrong for Docker (restart the container). Restarting because Neo4j blinked fixes nothing and discards a warm corpus. |
| `CMD` uses `sh -c "exec uvicorn ..."` | the shell is needed to expand `${WEB_CONCURRENCY}`; `exec` makes uvicorn **PID 1**, so it receives SIGTERM. Without it `/bin/sh` sits at PID 1, forwards nothing, and every deploy SIGKILLs the workers — cutting SSE responses off mid-answer. |
| runs as uid **10001**, never root | a container escape should not start from a process that already owns the filesystem |

### Configuration

Everything comes in at run time. Nothing is baked:

```bash
docker run -d -p 8100:8100 \
  -e DPDP_PROVIDER=openrouter \
  -e DPDP_MODEL=openai/gpt-4o-mini \
  -e OPENROUTER_API_KEY=sk-... \
  -e DPDP_CORS_ORIGINS=https://your-frontend-origin \
  -e NEO4J_URI=... -e NEO4J_PASSWORD=... \
  -e WEB_CONCURRENCY=4 \
  regrock-backend
```

`WEB_CONCURRENCY` defaults to **2**. Each worker holds its own copy of the
corpus and the dense matrix (~30 MB resident), so raising it trades memory for
concurrency — set it against the container's actual memory limit.

### Rebuilds are cheap, on purpose

`requirements.txt` is copied and installed **before** the source. Docker caches
that layer on the file's hash, so editing a Python file does not reinstall
numpy. That is the difference between a 4-second rebuild and a 3-minute one.

---

## 4. Check it is working

```bash
curl http://localhost:8100/api/live      # is the process up?
curl http://localhost:8100/api/health    # are its dependencies up?
```

`/api/health` returns **503**, not a 200 saying `ok: false`, when something is
wrong. Load balancers read the status code and ignore the body — a 200 would
keep a broken instance in rotation.

`/api/live` touches no dependency. Restarting a container because Neo4j
blinked fixes nothing, so liveness and readiness are separate probes.

Ask a question (with auth off):

```bash
curl -N -X POST http://localhost:8100/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"what is the fine if customer data leaks?"}'
```

Interactive docs are at `http://localhost:8100/docs`.

---

## 5. Tests

```bash
python tests/run_all.py
```

**116 checks. No network, no database, no API key, about 8 seconds.**

The answer engine is transport-agnostic — it yields typed events and knows
nothing about HTTP — so the entire path runs offline with a stubbed model.
That is what makes it cheap enough to run on every change.

Individually:

```bash
python tests/test_understanding.py   # 49 — normalisation, scope, intent
python tests/test_pipeline.py        # 67 — retrieval → answer → verification
```

Under pytest, use the runner (`run_all.py`), not `pytest tests/` — the test
files call `sys.exit()` at module level, which pytest reports as a collection
error even on a clean pass.

---

## 6. Calibration tools

```bash
python eval/measure_floor.py
```

Re-measures the sufficiency floor against this corpus.

**Never carry a floor across an index.** A floor is a property of an index,
not of a domain. The value inherited from the previous system (6.0) refused
**7 of 20 genuine compliance questions** on this one — including *"how long
can we keep customer records?"*, whose retrieval had correctly found the Third
Schedule's retention rules and whose answer the gate then threw away.

Run this after any change to what the index covers.

---

## 7. The Supabase schema

Run **once** against the project, in the Supabase SQL editor:

```
supabase_setup.sql
```

It creates `profiles` and `login_events`, and the trigger that fills them.

> **The trigger runs inside the sign-up transaction. If it raises, nobody can
> create an account.** This shipped broken once and produced
> `error_description=Database+error+saving+new+user` in production, with the
> confusing signature that *every new sign-up failed while every existing user
> kept signing in normally*. Three things in that file prevent it — a
> `coalesce` on the timestamp, an exception handler around the whole body, and
> a guard for the trigger firing twice per sign-up. The comments explain each.
> Do not remove them.

---

## 8. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Frontend shows "Offline", nothing in the backend log | CORS — the request never arrived | set `DPDP_CORS_ORIGINS` to the frontend's origin |
| Startup refuses: embeddings mismatch | `DPDP_EMBED_MODEL` changed | set it back, or rebuild the index in `KG_creation/` |
| `search index missing at chunks.json` | corpus not present | it is committed; check you are running from `backend/` |
| Every answer says "no model provider is configured" | no API key | set `OPENROUTER_API_KEY` |
| 429 from the provider | free-tier cap | wait, or use a paid key |
| Penalty answers are incomplete | Neo4j unreachable — degraded mode | check the startup warning; structural edges are not in `chunks.json` |
| 401 on every request | expired token, or auth misconfigured | check `SUPABASE_URL`; for local work set `DPDP_REQUIRE_AUTH=0` |
| 503 with `Retry-After` on auth | the JWKS endpoint is unreachable | transient. This is deliberately **not** a 401 — see `FLOW.md` §10 |

---

## 9. Routes

| Route | Auth | Purpose |
|---|---|---|
| `GET /` | no | identifies the service so a root probe is not a 404 |
| `GET /api/live` | no | liveness — touches no dependency |
| `GET /api/health` | no | readiness — **503** when not ready |
| `GET /api/config` | no | the frontend's bootstrap (Supabase URL + anon key) |
| `POST /api/chat` | **yes** | ask a question; streams SSE |
| `GET /api/history` | **yes** | this user's past answers |
| `GET /api/provision/{id}` | **yes** | one provision, verbatim |

Every response carries `X-Request-ID`, minted once in middleware and reused by
`/api/chat`, so the header, the MongoDB record and the trace all name the same
request.

---

## 10. Updating the corpus

The corpus is built by `KG_creation/`, not here:

```bash
cd ../KG_creation
python -m kg_build --embed --neo4j
cp data/chunks.json data/embeddings.npz ../backend/data/
cd ../backend
# restart — the corpus is loaded once at startup
```

**Copy both files or neither.** They are one artifact in two parts, and the
backend refuses to start when they disagree.

There is no version number to bump. `build_id` is a content hash of the
corpus, so it changes automatically the moment the law does — and every logged
answer names the exact version of the text it was generated against.
