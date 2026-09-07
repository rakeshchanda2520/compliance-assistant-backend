# Backend — how a question becomes a verified answer

One question comes in. A stream of events goes out. This document follows that
path end to end, in order, and says what each stage decides and why.

`KG_creation/FLOW.md` covers how the corpus was built. This one starts from
the moment a corpus already exists.

---

## 1. The idea in one picture

Most question-answering systems ask a model a question and show you what it
says. This one does not trust the model with anything it does not have to.

```
   ┌──────────────────────────────────────────────────────────┐
   │  THE MODEL IS ALLOWED TO:                                 │
   │    • write sentences that explain retrieved text          │
   │                                                           │
   │  THE MODEL IS NOT ALLOWED TO:                             │
   │    • supply a penalty amount    (read from the graph)      │
   │    • supply a definition        (read from the graph)      │
   │    • decide a citation is valid (checked after the fact)   │
   │    • decide the question is in scope (decided before it)   │
   └──────────────────────────────────────────────────────────┘
```

For some questions the model is not called **at all**. A penalty question is
answered by reading the Schedule out of the graph and formatting it. Zero
hallucination risk, because there is nothing to hallucinate with.

---

## 2. The whole request, in order

```
  POST /api/chat   { question, conversation_id }
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 0  AUTH          api/auth.py                  │
  │    Verify the token. Identity comes from the  │
  │    token, never the request body.             │
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 1  NORMALISE     understanding/normalize.py   │
  │    Strip greetings. Fix known typos. Expand   │
  │    shorthand. Keep the original for display.  │
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 2  EMBED ONCE    providers/embeddings.py      │
  │    One vector, reused by routing AND search.  │
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 3  ROUTE         understanding/router.py      │
  │    Which law? What kind of question?          │  ──► foreign law?
  └───────────────────────────────────────────────┘      REFUSE, stop here
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 4  RETRIEVE      retrieval/pipeline.py        │
  │    3 searches → fuse → boost → graph hops     │
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 5  TEMPLATE?     answering/templates.py       │
  │    Can the graph answer this outright?        │  ──► YES: render it,
  └───────────────────────────────────────────────┘      NO MODEL CALL
        │ no
        ▼
  ┌───────────────────────────────────────────────┐
  │ 6  ENOUGH?       answering/sufficiency.py     │
  │    Did we actually find anything?             │  ──► no: say so,
  └───────────────────────────────────────────────┘      NO MODEL CALL
        │ yes
        ▼
  ┌───────────────────────────────────────────────┐
  │ 7  GENERATE      providers/llm.py             │
  │    Stream an answer from the retrieved text.  │
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 8  VERIFY        verification/*.py            │
  │    Check every citation and every figure      │
  │    AFTER the fact. Never blocks the stream.   │
  └───────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │ 9  RECORD        providers/store.py           │
  │    Write to MongoDB. Never fails the answer.  │
  └───────────────────────────────────────────────┘
```

Three of those steps can end the request **without calling a model at all**:
step 3 (wrong law), step 5 (the graph knows), step 6 (nothing found).

---

## 3. Step 1 — Normalise: people do not type like a statute

A real question looks like this:

```
   "hi, quick question — wat r our obligatons if PII of a child leaks pls??!!"
```

Nothing in the Act contains the words "wat", "obligatons", or "pls". Normalise
cleans it up **for searching only**. The original is kept for display and for
the record.

```
   as typed  ──►  strip greeting     "wat r our obligatons if PII of a child leaks"
             ──►  fix known typos    "what r our obligations if PII of a child leaks"
             ──►  expand shorthand   "what are our obligations if personal data
                                       of a child leaks"
             ──►  collapse "??!!"    → "?!"
```

Two rules that matter:

- **Hinglish and synonyms are APPENDED, never substituted.** If someone writes
  "bacche ka data", the statutory term is added alongside — the original words
  keep whatever weight they earned. Replacing them can only lose signal.
- **A pleasantries-only message never normalises to empty.** "hi there" must
  still be a question the system can respond to, not a crash.

---

## 4. Step 3 — Route: two independent questions

### 4a. Is this even our law?

This is a **scope** decision, not a confidence one, and it reads the
**original** text — because normalisation could smuggle a foreign regime into
domestic vocabulary.

```
   "What does GDPR say about consent?"     ──►  REFUSE. Named foreign regime.
   "What does HIPAA require?"              ──►  REFUSE. Named foreign regime.

   "Can we send data to Singapore?"        ──►  ANSWER. A country name is not
                                                a foreign legal regime — this
                                                is a cross-border transfer
                                                question, squarely in scope.

   "How does DPDP compare to GDPR?"        ──►  ANSWER, with a caveat noting
                                                the comparison is out of scope.
```

The distinction in the middle row is deliberate and was a real bug: putting
bare country names in the refusal list made the system refuse genuine
cross-border questions, which are exactly what section 16 is about.

### 4b. What kind of question is it?

```
   penalty        "what's the fine for..."       ──► graph can answer outright
   definition     "what is a Data Fiduciary"     ──► graph can answer outright
   retention      "how long can we keep..."      ──► graph can answer outright
   direct lookup  "what does section 8 say"      ──► graph can answer outright
   obligation     "what must we do about..."     ──► needs a model
   general        anything else                  ──► needs a model
```

Three tiers decide it, cheapest first: a **regex** fast path, then **vector
similarity** against exemplar phrases, then a **fallthrough to `general`**.

A miss is safe — it just means the model answers instead. A wrong label is
not, which is why templates check their own work in step 5.

---

## 5. Step 4 — Retrieve: three searches, then the graph

### Three rankers, fused

```
   the question
        │
        ├──► BM25 on the raw text ──────┐
        │    exact words. "250 crore"   │
        │    must not blur into         │
        │    "200 crore".               │
        │                               │
        ├──► BM25 on contextual text ───┼──► fuse by RANK,
        │    same index, plus a         │    not by score
        │    situating sentence         │         │
        │                               │         ▼
        └──► dense vector search ───────┘    top candidates
             meaning, not words
```

Why fuse by **rank** and not by score: the three produce numbers on completely
different scales. Adding them means whichever happens to have the largest
range silently wins. Rank fusion only asks "how near the top did each ranker
put this?", which is comparable by construction.

Why keep raw BM25 **alongside** the contextual one instead of replacing it:
adding context changes term frequencies and dilutes exactly the exact-token
signal BM25 exists for. Keeping both means a bad context can only add a
ranker, never remove one.

### Then boosts

```
   phrase boost   the question names a defined term, exactly
                  "Data Fiduciary" → boost def-data-fiduciary
                  Longest phrase wins, so "Significant Data Fiduciary"
                  beats "Data Fiduciary".

   vocab boost    layperson word → statutory term, from vocab.yaml
                  "leak" → "personal data breach"
                  Capped: a term appearing in >20% of the corpus carries
                  no discriminating signal and is ignored.
```

That cap exists because of a real failure. "customer" maps to "personal data",
which appears in most of the statute — boosting on it promoted the two
shortest Definitions above every Penalty row on a penalty question.

### Then two graph hops

This is the part a pure search cannot do:

```
                  ┌──────────────┐
   search found → │ Section 8(5) │
                  └──────────────┘
                    │          │
        PENALISED_BY│          │REFERENCES
                    ▼          ▼
          ┌──────────────┐  ┌──────────┐     hop 1  (weight × 0.6)
          │ Schedule 1   │  │ Rule 6   │
          │ ₹250 crore   │  └──────────┘
          └──────────────┘       │
                                 │ REFERENCES
                                 ▼
                          ┌──────────────┐   hop 2  (weight × 0.36)
                          │  Rule 6(1)   │
                          └──────────────┘
```

Each hop is weighted down, so distant provisions stay reachable without
flooding the context. `MENTIONS` edges are **barred from the second hop** —
they all point into definitions, so following them twice fills the answer with
vocabulary instead of law.

---

## 6. Step 5 — Templates: answers with no model at all

For four kinds of question, the graph already holds the answer exactly. Asking
a model to restate it can only introduce error.

```
   ┌──────────────────────────────────────────────────────────────┐
   │  "what is the fine if customer data leaks?"                   │
   │                                                               │
   │   graph  ──► all 7 Schedule rows, amounts read from the       │
   │              Penalty field, formatted, most relevant first    │
   │                                                               │
   │   model calls: 0        citations: verified by construction   │
   └──────────────────────────────────────────────────────────────┘
```

### The guard that makes this safe

A template renders from the graph and returns **before** the sufficiency
check, and its citations are verified by construction. So nothing downstream
can catch it answering the wrong question. With only the intent label
deciding, this actually happened:

```
   "what is the tallest mountain?"
        intent regex said: definition
        template rendered: the DPDP definition of "automated"
        marked:            VERIFIED
```

A confident, wrong answer wearing a verified citation — the worst output this
system can produce. So every template must now **agree with retrieval** before
it fires:

| Template | Must be true before it renders |
|---|---|
| definition | the question **names** the defined term — every word, any order |
| retention | the Third Schedule or rule 8 is actually in the search results |
| penalty | at least one Penalty row was retrieved |
| direct lookup | the question named a provision id that resolves |

A false negative is cheap: it falls through to the model. A false positive is
a confidently wrong answer. So the guards are deliberately strict.

**When a template declines, the route is re-announced.** Otherwise every
consumer still believes a model-written answer came from the graph with zero
hallucination risk — the exact opposite of the truth.

---

## 7. Step 6 — Sufficiency: did we actually find anything?

The honest problem: **there is no threshold that separates in-scope from
out-of-scope questions.** Measured on this corpus:

```
   genuine questions    5.0 ──────────────────► 11.8
   unrelated questions  0.0 ─────────► 7.3
                             ^^^^^^^^^^^ they overlap
```

GDPR and HIPAA questions score as high as real ones, because they share real
legal vocabulary. No single number works.

So the only question is **which way it fails**, and the answer is: it fails
toward answering. Refusing a real compliance question breaks the product.
Attempting an unrelated one costs one call and produces an honest "these
provisions do not settle this".

### A score is not comparable across question lengths

This is subtle and it shipped broken. BM25 **sums** over the words in your
question. A ten-word question can accumulate a score a two-word question
never can — however exact the two-word one is.

```
   "what is a data fiduciary"    score 2.73    ← REFUSED as out of scope
        │                                         by a floor of 5.0
        └── and the top result was def-data-fiduciary, ranked FIRST.
```

An Act that defines "Data Fiduciary" told the user that Data Fiduciaries were
outside its scope. Three signals now decide it, each answering something a
raw magnitude cannot:

| Signal | What it establishes |
|---|---|
| **the question names a defined term** | length-independent — two words name it as fully as ten |
| **short question, every word statutory** | measured: short in-scope questions score 3.4–11.4, and 8 of 10 short off-topic ones score **exactly 0.00** |
| **fuzzy repair** | a word absent from the whole corpus is probably a misspelling of one that is present |

Two constraints hold these together:

1. **A repair may improve RANKING; it must never create the SCOPE signal.**
   The abstention score is computed on the question *as asked*. Feeding
   repairs into it pushed 9 of 15 off-topic questions past the floor, because
   "mountain"→"contain" and "weather"→"whether" add score without adding
   meaning.
2. **Only a distinctive phrase settles scope.** The Act defines "data",
   "person", "state", "gain", "loss" and "she". If a bare one of those settled
   scope, nearly every sentence in the language would be in scope. Two words,
   or one of nine-plus characters.

---

## 8. Step 8 — Verification: checking after, not before

The model has already streamed its answer to the user. Now every claim in it
is checked against the graph. **This never blocks the stream** — the user is
reading while this runs.

### Citations get one of three labels

```
   ┌──────────────┬──────────────────────────────────────────────┐
   │ verified     │ the provision exists AND was retrieved        │
   │              │ → the model read it                          │
   ├──────────────┼──────────────────────────────────────────────┤
   │out_of_context│ it exists but was NOT retrieved              │
   │              │ → recalled from training, not read. Suspect. │
   ├──────────────┼──────────────────────────────────────────────┤
   │ unresolved   │ no such provision exists at all              │
   │              │ → invented                                   │
   └──────────────┴──────────────────────────────────────────────┘
```

The middle one earns its keep: a model can cite a real section it was never
shown. That is a citation recalled rather than read, and it looks identical to
a good one until you check.

### Figures are checked separately

Every rupee amount and every number in the answer is compared against the
retrieved text. A figure the evidence does not support is flagged
**unsupported** — a defect in the answer.

### Three states in the UI, not two

```
   GREEN   supported            the evidence backs this
   AMBER   not yet in force     the answer is RIGHT; the law is not
                                enforceable yet
   AMBER   caveat               scope note
   RED     unsupported          the answer states something the evidence
                                does not support
```

Red is reserved for the last one. A provision that has not commenced is the
**opposite** of an error — the answer is correct and the rule simply is not in
force yet. Painting that red tells a reader their correct answer is wrong,
which is the more expensive of the two mistakes.

---

## 9. Two databases, on purpose

```
   ┌────────────────────────┐        ┌────────────────────────┐
   │  Supabase Postgres     │        │  MongoDB               │
   │  ───────────────       │        │  ───────              │
   │  WHO signed in         │        │  WHAT was asked        │
   │                        │        │                        │
   │  auth.users            │        │  interactions          │
   │  profiles              │        │    question + full     │
   │  login_events          │        │    answer + citations  │
   │                        │        │                        │
   │  NO question or answer │        │  conversations         │
   │  text, ever            │        │    question + which    │
   │                        │        │    provisions, and     │
   │  Written by a database │        │    NOT the answer      │
   │  trigger, never by     │        │                        │
   │  this service          │        │                        │
   └────────────────────────┘        └────────────────────────┘
              │                                  │
              └──────────── same key ────────────┘
                        auth.users.id
```

They correlate on one shared key, taken from the verified token in both write
paths. There is no live join across two engines and none is needed — there is
only one place a user id is ever minted.

### Why `conversations` stores the question but not the answer

A follow-up question is seeded with the **provisions** earlier turns reached,
never their prose.

```
   turn 1   "what must we do about security?"
            → reached s-8-5, r-6, r-6-1
            → answer: "... you must encrypt ..."

   turn 2   "and what's the fine for that?"
            seeded with:  s-8-5, r-6, r-6-1        ✅ subject carried
            NOT seeded with: the turn-1 answer      ❌ error not carried
```

Carrying a prior answer forward is how a system defends a hallucination three
turns later: the model reads its own earlier mistake as established context
and elaborates on it. Provision ids give continuity of **subject** without
continuity of **error**, so every turn re-derives its answer from the statute.

### Writes and reads fail differently, on purpose

```
   record()    NEVER raises   an unreachable database must not fail an
                              answer the user has already received

   history()   ALWAYS raises  a broken read must never render as
                              "you have no history" — that is a silent lie
```

---

## 10. Degrading instead of dying

Every external dependency has a defined failure behaviour. None of them takes
the service down.

| If this is unavailable | What happens |
|---|---|
| **Neo4j** | the graph is rebuilt from `chunks.json`. Search and citations still work. Structural edges are gone, so penalty and cross-reference answers are incomplete — and this is **logged loudly**, not hidden. |
| **the embedder** | dense search is dropped. BM25 and the graph carry the query. |
| **MongoDB** | answers still stream. Nothing is recorded, and a warning says so. |
| **Langfuse** | tracing is skipped. Never affects an answer. |
| **the reranker** | fails soft; the fused order is used. Off by default until measured on the deploy host. |
| **the JWKS endpoint** | **503 + Retry-After**, never 401. A 401 tells the user to sign in again, which cannot help — they would loop forever. |

---

## 11. What the browser actually receives

`POST /api/chat` streams Server-Sent Events, in this order:

```
   event: router      which law, what kind of question, which path
   event: retrieval   what was found, and how — before generation starts
   event: token       ◄─┐
   event: token         │  the answer, word by word
   event: token       ◄─┘
   event: citations   every citation with its verified/out_of_context/unresolved
   event: claims      anything flagged during verification
   event: done        timings, path taken, build_id
```

Two variations:

- **A template answer streams too**, although the text is already in memory.
  That is deliberate: an interface that is instant for some questions and
  progressive for others reads as broken rather than fast, and the client
  would need a second rendering path.
- **`abstain` replaces the tokens** when the question is out of scope or
  nothing was found.

The **last** `router` event is always the truth. A declined template and a
sufficiency abstain both re-announce the route, because the first announcement
described the plan *before* retrieval ran.

---

## 12. Layout

```
backend/
├── FLOW.md          this file
├── run.md           how to run it
├── config.py        every constant, with the reasoning for its value
│
├── core/            types shared by everything
│   ├── ids.py          THE citation spec — parsing and labels, one place
│   ├── models.py       Chunk, Provision, Scored, Citation, Claim
│   └── corpus.py       loading chunks.json, and verifying it
│
├── understanding/   what is being asked
│   ├── normalize.py    typos, shorthand, greetings, Hinglish
│   └── router.py       jurisdiction and intent
│
├── retrieval/       finding the law
│   ├── lexical.py      BM25, written out rather than imported
│   ├── dense.py        vectors from embeddings.npz
│   ├── fusion.py       rank fusion
│   ├── expand.py       the two graph hops
│   ├── rerank.py       optional cross-encoder, off by default
│   └── pipeline.py     the stages, in order
│
├── answering/       producing the answer
│   ├── templates.py    graph-only answers, and their guards
│   ├── sufficiency.py  did we find enough?
│   ├── prompt.py       what the model is told
│   ├── schema.py       structured output parsing
│   └── engine.py       the whole pipeline as one readable function
│
├── verification/    checking the answer
│   ├── citations.py    verified / out_of_context / unresolved
│   ├── claims.py       figures and quotes
│   └── temporal.py     which provisions are actually in force
│
├── providers/       the outside world, each behind one interface
│   ├── llm.py          ollama · claude · openrouter
│   ├── embeddings.py   build-time and query-time must match
│   ├── graphdb.py      Neo4j, with the offline fallback
│   ├── store.py        MongoDB
│   └── observability.py Langfuse
│
├── api/             HTTP
│   ├── app.py          routes, SSE framing, recording
│   ├── auth.py         token verification
│   └── ratelimit.py
│
├── data/            the corpus, produced by KG_creation
└── tests/           116 checks, no network, no database
```

The engine is **transport-agnostic** — it yields typed events and knows
nothing about SSE, FastAPI or a browser. That is what lets the whole answer
path run offline in a test in under a second, which is what makes it
affordable to check on every change.
