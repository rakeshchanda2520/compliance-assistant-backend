"""
Re-measure SUFFICIENCY_FLOOR / SUFFICIENCY_CEILING against the live index.

    python eval/measure_floor.py

A floor is not portable. It is a property of THIS index — change what the
index covers (add the label and headnote to the lexical field, rebuild the
corpus, swap the tokeniser) and every score moves under it. The inherited
7.3-24.6 / 6.5-15.6 numbers were carried across exactly such a change and the
result was a gate that refused seven of twenty genuine compliance questions,
including one whose retrieval had returned the correct provisions.

So this prints the two populations and the cost of every candidate floor,
rather than a pass/fail. There is no threshold that separates them — the
populations overlap by construction, because an unrelated question written in
business English shares real vocabulary with a statute about business. The
only decision available is WHICH WAY the gate fails, and the answer is toward
answering: a refused compliance question breaks the product, while an
attempted unrelated one costs one model call and yields "the provisions
supplied do not settle this".

Read the table, pick the highest floor that refuses ZERO genuine questions,
and record the measurement next to the constant in config.py.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml                                                # noqa: E402

import config                                              # noqa: E402
from core.corpus import load_chunks                        # noqa: E402
from providers import graphdb                              # noqa: E402
from retrieval.pipeline import Retriever                   # noqa: E402
from understanding.normalize import normalize              # noqa: E402

# Written the way users write, not the way the Act does — the whole point is
# to measure the gate against real phrasing. Several deliberately share no
# statutory vocabulary at all ("our vendor lost a laptop with customer files").
GENUINE = [
    "a kid signed up on our app - what extra rules apply?",
    "how long can we keep customer records?",
    "what security safeguards must we implement?",
    "can we keep data on servers outside India?",
    "what is the fine if customer data leaks?",
    "do we need consent before sending marketing texts?",
    "our vendor lost a laptop with customer files, what now?",
    "who is a significant data fiduciary",
    "can a user ask us to delete their information",
    "what must a consent notice contain",
    "how do we verify a parent's consent",
    "what happens if we ignore a Board order",
    "is there an exemption for small startups",
    "do we have to appoint someone in India",
    "what rights does a customer have over their data",
    "how quickly must we report a breach",
    "can we use employee data without consent",
    "what does a consent manager do",
    "who hears an appeal against the Board",
    "are government agencies exempt",
]

# Out of scope, but NOT nonsense — nonsense scores zero and proves nothing.
# These are ordinary questions in ordinary business English, which is what
# actually lands near the genuine population.
UNRELATED = [
    "how do I bake sourdough bread",
    "what is the tallest mountain in the world",
    "write me a python function to sort a list",
    "who won the cricket world cup",
    "what is the capital of France",
    "how do I fix a leaking tap",
    "recommend a good laptop under 50000",
    "what is the weather tomorrow",
    "translate hello into spanish",
    "how many calories in a banana",
    "what time does the bank open",
    "explain quantum entanglement",
    "best route from delhi to jaipur",
    "how do I train for a marathon",
    "what stocks should I buy",
]

CANDIDATES = [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]


def main() -> None:
    logging.disable(logging.WARNING)
    corpus = load_chunks(config.DATA_DIR / "chunks.json")
    vocab = yaml.safe_load(
        (config.DATA_DIR / "vocab.yaml").read_text(encoding="utf-8"))
    graph = graphdb.load(config, corpus)
    # No dense index and no reranker: this measures the LEXICAL tier, which is
    # the one the floor is compared against. Mixing in a signal the gate never
    # reads would produce a number that cannot be acted on.
    retriever = Retriever(corpus, edges=graph.edges, dense_index=None,
                          vocab=vocab, settings=config)

    def score(question: str) -> float:
        _, trace, _ = retriever.retrieve(
            question, retrieval_text=normalize(question).retrieval)
        return trace.top_bm25

    genuine = sorted((score(q), q) for q in GENUINE)
    unrelated = sorted((score(q), q) for q in UNRELATED)

    print(f"\ncorpus: {len(corpus)} chunks, build {corpus.build_id}"
          f"{'  (GRAPH DEGRADED)' if graph.degraded else ''}")
    print(f"\ngenuine    {genuine[0][0]:6.2f} -> {genuine[-1][0]:6.2f}"
          f"   ({len(genuine)} questions)")
    for value, question in genuine[:5]:
        print(f"    {value:6.2f}  {question}")
    print(f"\nunrelated  {unrelated[0][0]:6.2f} -> {unrelated[-1][0]:6.2f}"
          f"   ({len(unrelated)} questions)")
    for value, question in unrelated[-5:]:
        print(f"    {value:6.2f}  {question}")

    print("\n floor   genuine refused   unrelated refused")
    best = None
    for floor in CANDIDATES:
        missed = sum(1 for v, _ in genuine if v < floor)
        caught = sum(1 for v, _ in unrelated if v < floor)
        flag = ""
        if missed == 0:
            best = floor
            flag = "  <- admits every genuine question"
        print(f"  {floor:4.1f}   {missed:2d}/{len(genuine)}"
              f"             {caught:2d}/{len(unrelated)}{flag}")

    print(f"\nconfigured floor {config.SUFFICIENCY_FLOOR}, "
          f"ceiling {config.SUFFICIENCY_CEILING}")
    if best is not None:
        print(f"highest floor refusing no genuine question: {best}")
    live = sum(1 for v, _ in genuine if v < config.SUFFICIENCY_FLOOR)
    if live:
        print(f"\n!! the configured floor refuses {live}/{len(genuine)} "
              f"genuine questions — that is the failure this gate exists to "
              f"avoid, not to cause")
    ceiling_hits = sum(1 for v, _ in genuine if v >= config.SUFFICIENCY_CEILING)
    if not ceiling_hits:
        print(f"!! no genuine question reaches the ceiling "
              f"({config.SUFFICIENCY_CEILING}) — nothing is ever SUFFICIENT "
              f"on the lexical tier, so the ceiling is doing no work")


if __name__ == "__main__":
    main()
