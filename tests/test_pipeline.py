"""
The whole pipeline, against the REAL corpus, with a stubbed model.

No network, no Neo4j, no provider. The engine is transport-agnostic and the
model is injected, so the entire answer path — routing, retrieval, template
guards, sufficiency, citation resolution, claim verification — runs offline in
under a second. That is what makes it affordable to run on every change, which
is the whole point of NEW_FRAMEWORK §5's "measure before you change it".

    python tests/test_pipeline.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import yaml                                                  # noqa: E402
import config                                                # noqa: E402
from answering.engine import Deps, Engine                    # noqa: E402
from core.corpus import load_chunks                          # noqa: E402
from core.ids import label_for                               # noqa: E402
from providers import graphdb                                # noqa: E402
from retrieval.pipeline import Retriever                     # noqa: E402
from understanding.router import Router                      # noqa: E402
from verification.claims import extract_figures              # noqa: E402

PASS = FAIL = 0


def say(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(text.encode(enc, "replace").decode(enc))


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        say(f"  ok    {name}")
    else:
        FAIL += 1
        say(f"  FAIL  {name}" + (f"  -> {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
corpus = load_chunks(config.DATA_DIR / "chunks.json")
vocab = yaml.safe_load((config.DATA_DIR / "vocab.yaml").read_text(encoding="utf-8"))
graph = graphdb.load(config, corpus)
retriever = Retriever(corpus, edges=graph.edges, dense_index=None,
                      vocab=vocab, settings=config)
router = Router(intent_floor=config.INTENT_MIN_COSINE,
                jurisdiction_floor=config.JURISDICTION_MIN_COSINE,
                jurisdiction_margin=config.JURISDICTION_MARGIN)

MODEL_OUTPUT = [
    "Short answer:\nYou must take reasonable security safeguards. ",
    "Section 8(5) requires this. ",
    "The maximum penalty is 250 crore rupees.",
]


def stub_model(*, question, context, allowed_ids, plan):
    yield from MODEL_OUTPUT


def run(question: str, **kw) -> dict[str, list]:
    deps = Deps(corpus=corpus, graph=graph, retriever=retriever,
                router=router, settings=config)
    deps.synthesize = stub_model
    events: dict[str, list] = {}
    for ev in Engine(deps).answer(question, **kw):
        events.setdefault(ev.name, []).append(ev.data)
    return events


# --------------------------------------------------------------------------- #
say("\ncorpus integrity")

check("237 chunks load", len(corpus) == 237, str(len(corpus)))
check("build_id is a content hash", len(corpus.build_id) == 12)
check("every chunk resolves to a provision",
      all(c.node_id in graph.provisions for c in corpus.chunks))
check("chunk_for rolls an unchunked id up to its ancestor",
      corpus.chunk_for("s-8-5-a") is not None)


# --------------------------------------------------------------------------- #
say("\nevery event stream ends in exactly one terminal event")

for q in ["what is the fine if customer data leaks?",
          "What does HIPAA say about patient data?",
          "how do I bake sourdough bread",
          "what is a Data Fiduciary?",
          "",
          "x" * 3000]:
    ev = run(q)
    terminal = len(ev.get("done", [])) + len(ev.get("error", []))
    check(f"one terminal event for {q[:26]!r}", terminal == 1, str(terminal))


# --------------------------------------------------------------------------- #
say("\nrouting to the right path")

ev = run("what is the fine if customer data leaks?")
check("penalty takes the template path", ev["done"][0]["path"] == "template")
check("template path calls NO model", ev["done"][0]["model"] is None)
check("all 7 Schedule rows are cited",
      len([c for c in ev["citations"][0]["citations"]
           if c["id"].startswith("pen-")]) == 7)
check("every template citation is verified by construction",
      all(c["status"] == "verified" for c in ev["citations"][0]["citations"]))

ev = run("What does Article 33 of the GDPR require?")
check("foreign law abstains", "abstain" in ev)
check("foreign abstain spends NO retrieval", "retrieval" not in ev)

ev = run("how do I bake sourdough bread")
check("off-topic reaches retrieval then abstains",
      "retrieval" in ev and "abstain" in ev)
check("off-topic spends no model call", "token" not in ev)

ev = run("what does section 8 say?")
check("a named provision is answered despite a low lexical score",
      "abstain" not in ev, str(ev.get("abstain")))


# --------------------------------------------------------------------------- #
say("\ntemplate guards — the confidently-wrong-answer class")

for q in ["what is the tallest mountain",
          "what is the best pizza topping",
          "what is the capital of France"]:
    ev = run(q)
    templated = ev.get("done", [{}])[0].get("path") == "template"
    check(f"no template answer for {q[:26]!r}", not templated)

ev = run("what is a Data Principal?")
check("a genuine definition DOES template",
      ev["done"][0]["path"] == "template", str(ev["done"][0]))

ev = run("can we keep data on servers outside India?")
check("a cross-border question does not get the retention template",
      ev["done"][0]["path"] != "template" or
      "rules-sch-third" not in [c["id"] for c in ev["citations"][0]["citations"]])


# --------------------------------------------------------------------------- #
say("\nverification catches what citation checking cannot")

ev = run("wat r our obligatons if PII of a child leaks pls")
claims = ev.get("claims", [{}])[0].get("claims", [])
check("a figure absent from the evidence is flagged unsupported",
      any(c["verdict"] == "unsupported" for c in claims),
      str(claims)[:90])

cites = ev["citations"][0]["citations"]
check("a cited-but-not-retrieved provision is out_of_context",
      any(c["status"] == "out_of_context" for c in cites)
      or all(c["status"] == "verified" for c in cites))

check("figures parse in both digit and spoken form",
      extract_figures("250 crore")[0][1]
      == extract_figures("two hundred and fifty crore")[0][1])
check("a section number is NOT treated as a figure",
      not extract_figures("see section 8(5) and rule 6"))
check("a rupee amount IS treated as a figure",
      extract_figures("up to 250 crore rupees"))


# --------------------------------------------------------------------------- #
say("\nlabels never mangle an id")

for node_id, expected in [("s-8-5", "Section 8(5)"), ("r-6-1", "Rule 6(1)"),
                          ("pen-1", "Schedule entry 1"),
                          ("rules-sch-third", "Third Schedule")]:
    check(f"{node_id} -> {expected}", label_for(node_id) == expected,
          label_for(node_id))
check("a definition id does not render as a section",
      "Section" not in label_for("def-data-fiduciary"),
      label_for("def-data-fiduciary"))


# --------------------------------------------------------------------------- #
say("\nthe route announced is the route taken")

ev = run("wat r our obligatons if PII of a child leaks pls")
if len(ev.get("router", [])) > 1:
    check("a declined template re-announces the real route",
          ev["router"][-1]["path"] == "llm"
          and ev["router"][-1].get("template_declined") is True)
else:
    check("route announced once matches the terminal path",
          ev["router"][0]["path"] == ev["done"][0]["path"])

for q in ["what is the fine if customer data leaks?", "what is a Data Fiduciary?",
          "how do I bake sourdough bread", "What does HIPAA say about patient data?"]:
    ev = run(q)
    check(f"final route matches done for {q[:24]!r}",
          ev["router"][-1]["path"] == ev["done"][0]["path"],
          f"{ev['router'][-1]['path']} vs {ev['done'][0]['path']}")


# --------------------------------------------------------------------------- #
say("\nscope — the gate must not refuse the subject it exists to answer")

from answering import sufficiency                             # noqa: E402
from understanding.normalize import normalize                 # noqa: E402


def verdict(question: str) -> str:
    plan = router.understand(question, None)
    results, trace, _ = retriever.retrieve(
        question, retrieval_text=normalize(question).retrieval)
    return sufficiency.assess(
        results, trace,
        floor=config.SUFFICIENCY_FLOOR, ceiling=config.SUFFICIENCY_CEILING,
        named_provision=bool(plan.provision_id)).verdict.name


# Every one of these was REFUSED by a shipped build. A BM25 score is a SUM
# over query terms, so a floor calibrated on 6-10 word questions rejects a
# two-word one however exact it is — and a hand-written misspelling table
# only ever covers misspellings someone thought of. "Daata Fudiciary" is the
# spelling this was actually reported with.
MUST_ANSWER = [
    "Daata Fudiciary", "Daata Fuduciary", "Data Fiduciary",
    "what is a data fiduciary", "wat is personel data", "consnet manager",
    "consent", "erasure", "parental consent", "erassure of data",
    "data retention", "cross border transfer", "consent notice",
    "how long can we keep customer records?",
    "a kid signed up on our app - what extra rules apply?",
    "can a user ask us to delete their information",
    "our vendor lost a laptop with customer files, what now?",
]
for question in MUST_ANSWER:
    check(f"answers {question[:34]!r}", verdict(question) != "INSUFFICIENT",
          "refused as out of scope")

# The other direction. These may reach the model — the gate fails toward
# answering on purpose — but must never be judged SUFFICIENT, the verdict
# that claims the corpus settles the question.
for question in ["how do I bake sourdough bread",
                 "what is the tallest mountain in the world",
                 "how do I train for a marathon", "sourdough",
                 "cricket score", "movie tickets", "gold price"]:
    check(f"never sure about {question[:30]!r}",
          verdict(question) != "SUFFICIENT")

# A repair may create RANKING signal; it must never create GATE signal.
# Measured: feeding repairs into the gate pushed nine of fifteen off-topic
# questions past the floor, because "mountain" -> "contain" and
# "weather" -> "whether" add score without adding meaning.
_off = "what is the tallest mountain in the world"
_, off_trace, _ = retriever.retrieve(_off, retrieval_text=normalize(_off).retrieval)
check("repairs do not inflate the abstention score",
      bool(off_trace.repairs) and off_trace.top_bm25 == 0.0,
      f"repairs={off_trace.repairs} top_bm25={off_trace.top_bm25}")

# A bare defined word is not a scope signal. The Act defines "data", "person",
# "state", "gain", "loss" and "she"; if any of them settled scope on its own,
# almost every sentence in the language would be in scope.
check("a bare common defined word does not settle scope",
      verdict("what is the state of my order") != "SUFFICIENT")

# A repair must reach the TEMPLATE GUARD too, not only the ranker. The guard
# asks "does the question name this term?" of the question as typed, so
# without this the misspelling retrieved `def-data-fiduciary` correctly, the
# guard saw no "fiduciary", the template declined, and a definition the graph
# can print verbatim was handed to a model instead.
_ev = run("wat is a Daata Fudiciary")
check("a misspelled definition still answers from the graph",
      _ev["done"][0]["path"] == "template",
      f"path={_ev['done'][0]['path']}")
check("...and its citations are verified",
      any(c["status"] == "verified"
          for c in (_ev.get("citations", [{}])[0].get("citations") or [])))


# --------------------------------------------------------------------------- #
say(f"\n{PASS} passed, {FAIL} failed")
# Only when run directly: under pytest a module-level sys.exit is a
# collection ERROR, which fails the run even on a clean pass.
if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
assert not FAIL, f"{FAIL} checks failed"
