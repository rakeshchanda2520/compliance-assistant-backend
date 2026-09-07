# DPDP Compliance Assistant — rebuild

A ground-up rebuild of Regrock's serving layer, implementing `NEW_FRAMEWORK.md`
from the previous solution. Same business need, same infrastructure, rebuilt
around the failures the old system was measured to have.

**Status: end to end — the pipeline, the HTTP layer and the page all run
together and have been verified doing so.** The eval baseline and Contextual
Retrieval are still missing; §7 says exactly what exists and what does not.

```bash
pip install -r requirements.txt
python tests/run_all.py          # 116 checks, no network, no database
```

```bash
uvicorn api.app:app --port 8100     # from this directory
```

The page is a SEPARATE component, a sibling of this folder — see
`../frontend/run.md`. This service serves no HTML at all. `run.md` here has
the full setup; `FLOW.md` has what happens inside a request.

---

## 1. What is different from the old system

Each row is a failure that was **measured** on the old system, not a
speculative improvement.

| Old failure | Cause | What this does instead |
|---|---|---|
| Every model-path question abstained | The gate compared an **RRF** score (~0.05) against a **BM25** threshold (15.0) | Scores are never collapsed into one field. `Scored` names each scale separately, and the gate reads `trace.top_bm25` explicitly |
| Unrelated questions got confident "verified" answers | Templates fired on the intent label alone | A template must **agree with retrieval** before it renders (§4) |
| 6 penalty questions → byte-identical answers | The template dumped all 7 rows with no ordering | All 7 still render — completeness is right — but ordered, with the best-supported one named in the summary |
| Rules citations were unclickable | The citation regex was written twice, kept in step by a test | **One spec** in `core/ids.py`; the frontend copy is generated from it |
| Definitions ranked below near-misses | BM25 is bag-of-words, so "Data Fiduciary" is two common terms | An exact **phrase** signal, plus structural metadata in the index |
| A country name refused a real question | "Singapore" was treated as a foreign regime | Only **named regimes** are foreign markers (§3) |
| Free-tier embedding exhaustion silently degraded retrieval | No signal when it happened | Degradation is logged, traced and surfaced in the trace |

---

## 2. Layout

Dependencies point one way: `core` knows nothing about anything else.

```
core/           domain types, the citation spec, corpus loading   (no I/O)
understanding/  normalisation + the three-tier router
retrieval/      lexical · dense · fusion · graph expansion · pipeline
answering/      sufficiency · templates · the engine
verification/   citations · claims (quote, numeric)
providers/      graph, and later: llm, embeddings, store          (all I/O)
tests/          two suites, no network, no database
data/           chunks.json · vocab.yaml · commencement.yaml · embeddings.npz
```

`answering/engine.py` is the whole pipeline in one readable function. It is
**transport-agnostic** — it yields typed events and knows nothing about SSE or
HTTP — which is what lets the test suite run the identical path offline in
under a second.

---

## 3. The question will not match your data

The governing assumption of `understanding/normalize.py`. People do not write
*"processing of personal data of children"*. They write:

```
"CHILDREN DATA???"
"Hi, can you please tell me wat r our obligatons if PII of a child leaks? thanks"
"kya hoga agar bachche ka data leak ho jaye"
"what is a data fiduciory"
```

All four are handled: greeting and sign-off stripping (looped, because they
stack), shouting folded, mixed punctuation runs collapsed, chat shorthand and
abbreviations expanded, a closed table of typos in defined terms, and romanised
Hindi **appended** rather than substituted so a mistranslation can never delete
a term the user typed.

**Regex is a fast path, not the primary one.** Tier 1 is free and instant when
it hits and misses far more than it catches on real traffic. The safety
property that makes that acceptable:

> a routing **miss** costs a model call.
> a routing **error** costs a confidently wrong answer.

So every tier abstains from deciding rather than guessing. An unrecognised
question becomes `GENERAL` and takes the model path, which is always safe.

---

## 4. Why templates are safe

A template renders from the graph, returns **before** the sufficiency gate, and
its citations are verified **by construction**. Nothing downstream can catch it
answering the wrong question — so it must not be able to.

Two guards, both tested:

- `_definition` requires the question to **name the defined term** — every word
  of it, order-free. "who is a Data Principal?" matches; "what is the tallest
  mountain" cannot.
- `_retention` requires the Third Schedule or rule 8 to be **among the
  retrieved results**, rather than reading them from the graph regardless.

A false negative is cheap: the request falls through to synthesis, which
answers from the same provisions. A false positive is a confident wrong answer
wearing a verified citation.

---

## 5. Retrieval

```
                 ┌─ raw BM25        (verbatim + label + header)
question ────────┼─ contextual BM25 (+ generated context)          ─┐
                 └─ dense           (embeddings)                    │
                                                                    ▼
                            RRF (k=60) ─► phrase/vocab boost ─► rerank
                                                                    │
                                          two-hop graph walk ◄──────┘
```

**Raw BM25 is kept as a separate ranker, never replaced by the contextual
one.** Prepending generated context changes term frequencies and dilutes
exact-token matching — and exact tokens are the whole reason BM25 is here,
because "250 crore" and "200 crore" are semantically near-identical and legally
opposite. Fused as separate rankers, a bad generated context can only **add** a
signal, never remove one.

**Boost and inject are different tools.** A multiplicative boost re-ranks what
BM25 already found; it cannot lift a chunk that scored zero. So an intent
naming a whole *class* of provision (penalty → the Schedule) **injects** those
rows directly — but only when the class is small enough to enumerate
(`MAX_INJECT`), because injecting 32 definitions buries the one that was asked
about.

**Both `PENALISED_BY` and `REFERENCES` are walked backwards.** The edge is
stored `r-6 —REFERENCES→ s-8-5`, so a walk seeded on section 8(5) could never
reach rule 6 going forwards — and "the Act states the duty, the Rules state
what discharges it" is the most valuable join in this corpus.

`ExpansionTrace.why_missing(node_id)` answers *"why is X not in the context?"*
in one line, distinguishing a build problem from a ranking problem from a
context-budget problem. Diagnosing that previously meant guessing.

---

## 6. Sufficiency, not a threshold

The old BM25 threshold was measured not to separate on this corpus:

```
genuine     7.3 ─────────────────────────► 24.6
unrelated   6.5 ───────────► 15.6
                 ^^^^^^^^^^^^ overlap
```

No single number works, so the only question is which way it fails. **It fails
toward answering**: refusing a real compliance question breaks the product,
while attempting an unrelated one costs one call and yields an honest "these
provisions do not settle this".

Deterministic tiers first — reranker score, margin, candidate count — with a
model call only inside the grey band. A question that **names its own
provision** is always sufficient: *"what does section 8 say"* scores 7.3, below
any workable floor, and is perfectly answerable.

### A BM25 score is not comparable across query lengths

This is the same class of mistake as comparing an RRF score to a BM25 floor,
and it shipped here too. BM25 **sums** over query terms, so a floor calibrated
on six-to-ten word questions is unreachable for a two-word one however exact it
is. `"what is a data fiduciary"` scored **2.73** with `def-data-fiduciary`
ranked FIRST and was refused as outside the scope of an Act that defines the
term. Reported as *"Daata Fudiciary"* returning "out of scope".

Three signals now decide it, and each answers a question magnitude cannot:

| Signal | What it establishes | Why magnitude cannot |
|---|---|---|
| **Distinctive phrase** (`trace.phrase_hits`) | the question names a term this instrument defines | length-independent; a two-word query names it as fully as a ten-word one |
| **Short query, all terms statutory** (`query_terms <= 3` and every one scorable) | there is nothing to judge magnitude *with* | measured: short in-scope queries score 3.4-11.4, and 8 of 10 short off-topic ones score **exactly 0.00** |
| **Fuzzy repair** (`retrieval/pipeline.py::_repair`) | a word absent from the index is a misspelling of one that is present | a hand-written typo table only covers misspellings someone thought of |

Two constraints hold these together, and undoing either reopens the hole:

1. **A repair may create RANKING signal; it must never create GATE signal.**
   `trace.top_bm25` is scored on the question AS ASKED. Feeding repairs into it
   pushed nine of fifteen off-topic questions past the floor, because
   `"mountain"` -> `"contain"` and `"weather"` -> `"whether"` add score without
   adding meaning.
2. **Only a DISTINCTIVE phrase settles scope.** The Act defines `"data"`,
   `"person"`, `"state"`, `"gain"`, `"loss"` and `"she"`; if a bare one of
   those settled scope, almost every sentence in the language would be in
   scope. Two words, or one of nine-plus characters. They stay usable as a
   ranking boost, which is the weaker claim.

Measured after: **0 of 20** genuine and **0 of 20** short in-scope questions
refused; **1 of 10** short off-topic and 5 of 15 long off-topic ones still
reach the model, which is the documented cost of failing toward answering.
`tests/test_pipeline.py` pins all of it.

---

## 7. What is built, and what is not

| Component | State |
|---|---|
| `core/` — types, id spec, corpus | **done**, tested |
| `understanding/` — normalise, route | **done**, 49 checks |
| `retrieval/` — lexical, fusion, expand, pipeline | **done**, exercised end to end |
| `answering/` — sufficiency, templates, engine | **done**, 39 checks |
| `verification/` — citations, quote, numeric | **done** |
| `providers/graphdb.py` | **done**, with an offline fallback |
| `providers/llm.py`, `embeddings.py`, `store.py`, `observability.py` | **done** — three providers behind one interface |
| `retrieval/rerank.py` | **done**, fails soft, off by default until measured on the deploy host |
| `api/` — FastAPI, SSE, auth, rate limit | **done**, 10 routes, verified booting |
| `../frontend/` — page, citation badges, evidence rail | **done**, verified end to end against the API |
| `../frontend/` history drawer — read-only replay of stored answers | **done**, through the same renderers the live stream uses |
| `supabase_setup.sql` — profiles, login_events, the sign-up trigger | **done**, carried over unchanged; run it once against the project |
| `eval/measure_floor.py` — sufficiency calibration | **done**, and it already found a real defect |
| `verification/entailment.py` | **not built** — the flag exists |
| `scripts/build_context.py` — Contextual Retrieval | **not built**; the pipeline logs that it is falling back to raw BM25 |
| `eval/` — golden set, baseline | **not built**, and it is the next thing |

### The floor was re-measured, and the inherited one was wrong

`eval/measure_floor.py` exists because a sufficiency floor is a property of an
INDEX, not of a domain. The value carried over from the old system (6.0, from a
`7.3–24.6` genuine population) refused **7 of 20 genuine compliance questions**
on this index — including *"how long can we keep customer records?"*, whose
retrieval had correctly returned the Third Schedule's retention rules and whose
answer the gate then threw away. This index scores differently because it
indexes label and headnote; re-measured, genuine questions run `5.02–11.80` and
unrelated ones `0.00–7.25`, so the floor is **5.0** (refuses none of the twenty,
still refuses 10 of 15 unrelated). Rerun it after any change to what the index
covers, and never carry a floor across one.

**Deliberately reused, not rewritten:** `data/chunks.json` and the PDF
extraction that produced it. `NEW_FRAMEWORK` §1 lists the round-trip validation
under KEEP — it has already caught a Gazette printing error and an id
collision, and rewriting 1,477 lines of PDF column geometry would be a
downgrade, not a rebuild.

### The next thing is the baseline, not a feature

`NEW_FRAMEWORK` §5 is explicit and it is the rule this rebuild is most at risk
of breaking: **capture the baseline before changing retrieval again.** The old
system shipped 56 golden questions and never captured one, which is why every
quality claim about it was an argument rather than a measurement.

That work has a prerequisite the old system also hit: the eval cannot run under
a 30-request/hour limit. `config.EVAL_USER_IDS` exists for a dedicated eval
identity — **not** a test-only auth bypass, which is exactly the kind of hole
that survives into production.

---

## 8. Running it

```bash
python tests/run_all.py            # everything, offline
python tests/test_understanding.py # normalisation + routing
python tests/test_pipeline.py      # the whole answer path, real corpus
python -m pytest tests -q          # same checks, under pytest
python eval/measure_floor.py       # re-calibrate the sufficiency gate
```

Two services, not one — the frontend holds no credential of its own and fetches
Supabase's public config from the API at load:

```bash
cp .env.example .env               # API: Neo4j, provider key, Supabase, Mongo
python -m uvicorn api.app:app --port 8100

cd frontend
echo DPDP_API_BASE=http://localhost:8100 > .env
python -m uvicorn server:app --port 3100
```

`DPDP_CORS_ORIGINS` must name the frontend's **browser-reachable** origin or
every call is blocked with nothing in the API's own logs. `frontend/assets/`
is committed on purpose — the page deploys as static files with no build step,
so `citations.js` (generated from `core/ids.py`) and the vendored supabase-js
UMD build have to be checked in. The CSP is `script-src 'self'`: a CDN is not
reachable from the page, by design.

Both suites run with no network, no Neo4j and no model provider. When Neo4j is
unreachable the graph is rebuilt from `chunks.json` and marked
`degraded=True` — enough for retrieval, citations and direct lookups, but
**not** for the structural edges, so penalty answers lose the join that makes
them worth trusting. That is surfaced, never silent.
