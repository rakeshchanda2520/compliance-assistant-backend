"""
Is the retrieved evidence enough to answer from?

This replaces a single BM25 threshold, which was MEASURED not to separate on
this corpus (20 genuine compliance questions, 15 unrelated — rerun with
`python eval/measure_floor.py`):

    genuine     5.02 ──────────────► 11.80
    unrelated   0.00 ──────► 7.25
                      ^^^^^^^^^^ overlap

No threshold exists that admits every genuine question and refuses every
unrelated one. So a single number was always going to be wrong; the question
is only which way it fails.

**It fails toward answering.** Refusing a real compliance question breaks the
product — the user gets nothing and cannot tell whether the law is silent or
the tool is broken. Attempting an unrelated one costs a single model call and
produces "the provisions supplied do not settle this", which is honest and
harmless.

Deterministic tiers first. A model call happens only inside a narrow band
where the cheap signals genuinely disagree, because the latency budget cannot
absorb another unconditional call on the critical path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from core.models import RetrievalTrace, Scored

log = logging.getLogger(__name__)


class Sufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNCERTAIN = "uncertain"      # the grey band; only this may cost a call


@dataclass(frozen=True)
class Assessment:
    verdict: Sufficiency
    reason: str
    signal: str          # WHICH signal decided, named with its scale
    value: float

    def to_dict(self) -> dict:
        return {"verdict": self.verdict.value, "reason": self.reason,
                "signal": self.signal, "value": round(self.value, 4)}


def assess(results: list[Scored], trace: RetrievalTrace, *,
           floor: float, ceiling: float, named_provision: bool = False
           ) -> Assessment:
    """Decide from the cheapest signals available.

    The signal is always reported WITH ITS SCALE. A previous system compared a
    Reciprocal Rank Fusion score (which lives in roughly [0.016, 0.05] because
    RRF is rank-based) against a threshold calibrated on BM25's unbounded
    scale. `top < threshold` was true for every question ever asked, so every
    request that reached the gate abstained — while template questions, which
    return earlier, kept working. The system looked healthy and refused five
    of the six example questions on its own home screen.
    """
    # A question naming its own provision cannot be out of scope: it already
    # told us where the answer lives. "what does section 8 say" scores 7.3 on
    # BM25 — below any workable floor — and is perfectly answerable.
    if named_provision:
        return Assessment(Sufficiency.SUFFICIENT,
                          "the question names a provision outright",
                          "named_provision", 1.0)

    if not results:
        return Assessment(Sufficiency.INSUFFICIENT,
                          "nothing in this corpus matched the question at all",
                          "candidates", 0.0)

    # A BM25 score is a SUM over query terms, so it is not comparable across
    # query lengths: the floor was calibrated on natural-language questions of
    # 6-10 words, and a two-word one cannot reach it no matter how exact.
    # "what is a data fiduciary" scored 2.73 with `def-data-fiduciary` ranked
    # FIRST, and was refused as outside the scope of an Act that defines the
    # term. Naming a defined term or a headnote of this instrument settles
    # scope on its own — it is the one signal that is length-independent, and
    # nothing out of scope produces it.
    #
    # Only a DISTINCTIVE phrase counts. The Act defines several bare English
    # words — "data", "person", "state", "gain", "loss", "she" — and any of
    # them appearing in a sentence says nothing about scope; treating "data"
    # as a scope signal would make almost every question in the language
    # answerable. Two words, or one long enough to be a name rather than a
    # word. (They remain fine as a RANKING boost, which is a weaker claim.)
    distinctive = next((p for p in trace.phrase_hits
                        if " " in p or len(p) >= 9), "")
    if distinctive:
        return Assessment(Sufficiency.SUFFICIENT,
                          f"the question names {distinctive!r}, a term this "
                          f"instrument defines",
                          "phrase", 1.0)

    # Tier 1: the reranker, when present. Its scores are calibrated and
    # comparable across queries in a way BM25's are not.
    if trace.top_rerank is not None:
        if trace.top_rerank < floor:
            return Assessment(Sufficiency.INSUFFICIENT,
                              f"best match scored {trace.top_rerank:.3f} after "
                              f"reranking, below the {floor:.2f} floor",
                              "rerank", trace.top_rerank)
        if trace.top_rerank >= ceiling:
            return Assessment(Sufficiency.SUFFICIENT, "strong reranked match",
                              "rerank", trace.top_rerank)
        return Assessment(Sufficiency.UNCERTAIN,
                          "reranked match is in the grey band",
                          "rerank", trace.top_rerank)

    # Tier 2: raw BM25, named explicitly so it can never be confused with a
    # fused score again.
    top = trace.top_bm25
    if top <= 0.0:
        return Assessment(Sufficiency.INSUFFICIENT,
                          "no provision shares any term with the question",
                          "bm25", top)
    # A short query cannot be judged against this floor, and forcing it to be
    # is how "consent" and "data retention" were refused by a system whose
    # entire subject is consent and data retention. The floor was calibrated
    # on questions of six to ten words; BM25 sums over terms, so three terms
    # have a structurally lower ceiling than ten no matter how exact they are.
    #
    # MEASURED on short queries (<=3 terms): in-scope ones score 3.4-11.4 and
    # eight of ten out-of-scope ones score EXACTLY 0.00 — for a short query
    # the signal is whether the corpus contains those words at all, which the
    # `top <= 0` branch above already reads.
    #
    # EVERY term must be statutory, not merely three of them. Counting only
    # the scorable ones conflates two opposite cases: "data retention" (2 of
    # 2 — short and in scope) and "what is the tallest mountain in the world"
    # (1 of 3 after stopwords — long and off topic). The first version of this
    # rule used the scorable count alone and let nine of fifteen off-topic
    # questions through; requiring the whole query to be statutory took that
    # back to five while still admitting every short in-scope one.
    #
    # UNCERTAIN, never SUFFICIENT: the answer is attempted with its weak
    # support on show, not asserted.
    if (trace.query_terms and trace.query_terms <= 3
            and trace.scorable_terms == trace.query_terms):
        return Assessment(Sufficiency.UNCERTAIN,
                          f"{trace.query_terms} term(s), all of them in the "
                          f"corpus — too short to judge by lexical magnitude",
                          "short_query", top)

    if top < floor:
        return Assessment(Sufficiency.INSUFFICIENT,
                          f"closest match scored {top:.1f}, below the "
                          f"{floor:.0f} floor for an in-scope question",
                          "bm25", top)
    if top >= ceiling:
        return Assessment(Sufficiency.SUFFICIENT, "strong lexical match",
                          "bm25", top)
    return Assessment(Sufficiency.UNCERTAIN,
                      f"lexical match {top:.1f} is between the {floor:.0f} "
                      f"floor and the {ceiling:.0f} ceiling",
                      "bm25", top)


def resolve_uncertain(assessment: Assessment, *, allow_model: bool) -> Assessment:
    """What an UNCERTAIN verdict becomes when no model call is available.

    It becomes SUFFICIENT. That is the asymmetry stated as code: in the band
    where the cheap signals cannot decide, answering and being honest about
    weak support beats refusing a question the corpus may well cover.
    """
    if assessment.verdict is not Sufficiency.UNCERTAIN:
        return assessment
    if allow_model:
        return assessment          # caller will spend one fast-model call
    return Assessment(Sufficiency.SUFFICIENT,
                      assessment.reason + " — answered rather than refused, "
                      "with support shown so the reader can judge",
                      assessment.signal, assessment.value)
