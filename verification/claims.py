"""
Claim-level verification: does the cited text actually SUPPORT the sentence?

Citation checking answers "does this provision exist and was it retrieved?"
It does not answer "does it say what the answer claims it says". Both errors
this project has on record live in that gap — each cited a real, retrieved
provision and then asserted something the provision did not support, and the
citation checker said `verified` for both.

Three checks, cheapest first:

    1. QUOTE      is the quoted string actually IN the cited provision?
                  Free. A substring test. Catches a fabricated quote outright.
    2. NUMERIC    does every figure appear in the evidence or a graph field?
                  Free. Deterministic. Catches the Schedule-figure error class.
    3. ENTAILMENT does the cited text entail the claim?
                  A model. Flagged, optional, and POST-HOC.

**Nothing here blocks the stream.** Answer tokens are already on their way to
the reader by the time verification runs; you cannot un-send them. Failures
surface as flagged claims after the fact, which is what the three-state
severity UI exists for. Buffering the answer until verification passed would
destroy streaming and worsen the very latency budget this design is trying to
protect.
"""
from __future__ import annotations

import logging
import re

from core.models import Claim, Scored, Verdict

log = logging.getLogger(__name__)

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    """Compare on letters and digits only.

    Quotation marks, curly apostrophes, line breaks and double spaces all
    differ between a model's rendering and the Gazette's, and none of those
    differences mean the quote is wrong.
    """
    return _NORM.sub(" ", text.lower()).strip()


def check_quote(claim: Claim, results: list[Scored], graph) -> Claim:
    """Is the claim's quote really in one of the provisions it cites?

    The cheapest possible grounding check and the one most people skip. A
    model that invents a plausible-sounding quote from a real provision passes
    every citation check ever written; this catches it with a substring test.
    """
    if not claim.quote.strip():
        return claim

    needle = _norm(claim.quote)
    if len(needle) < 12:
        return claim              # too short to be evidence either way

    for node_id in claim.cites:
        provision = graph.provisions.get(node_id)
        haystacks = []
        if provision is not None:
            haystacks.append(provision.text)
            haystacks.extend(graph.provisions[c].text
                             for c in graph.descendants_of(node_id)
                             if c in graph.provisions)
        haystacks.extend(r.chunk.verbatim for r in results
                         if r.node_id == node_id)
        if any(needle in _norm(h) for h in haystacks if h):
            return claim

    claim.verdict = Verdict.UNSUPPORTED
    claim.note = ("the quoted words do not appear in the provision this claim "
                  "cites — treat the quotation as unverified")
    return claim


# --------------------------------------------------------------------------- #
# numeric
# --------------------------------------------------------------------------- #

_MULTIPLIER = {"crore": 10_000_000, "lakh": 100_000,
               "thousand": 1_000, "hundred": 100, "million": 1_000_000}
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
          "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
          "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_FIGURE = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mult>crore|lakh|million|thousand)?"
    r"|(?P<words>(?:(?:" + "|".join(_UNITS) + r"|hundred|and)\s+){1,8})"
    r"(?P<wmult>crore|lakh|thousand)",
    re.I)

# Numbers that are citations, not quantities. "section 8(5)" and "rule 6" must
# never be fact-checked as figures — they are addresses.
_CITATION_NUMBER = re.compile(
    r"(?:§|\bsections?|\brules?|\bschedule\s+entry|\bclause|\bsub-?section)\s*\d",
    re.I)


def _spoken_to_number(text: str) -> int | None:
    total, current = 0, 0
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _UNITS:
            current += _UNITS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in _MULTIPLIER:
            total += (current or 1) * _MULTIPLIER[word]
            current = 0
        elif word != "and":
            return None
    return total + current or None


def extract_figures(text: str) -> list[tuple[str, float]]:
    """(surface form, canonical value) for every quantity in `text`."""
    out: list[tuple[str, float]] = []
    for m in _FIGURE.finditer(text):
        start = max(0, m.start() - 24)
        if _CITATION_NUMBER.search(text[start:m.end()]):
            continue
        if m.group("num"):
            try:
                value = float(m.group("num").replace(",", ""))
            except ValueError:
                continue
            if m.group("mult"):
                value *= _MULTIPLIER[m.group("mult").lower()]
        else:
            spoken = _spoken_to_number(f"{m.group('words')} {m.group('wmult')}")
            if spoken is None:
                continue
            value = float(spoken)
        out.append((m.group(0).strip(), value))
    return out


def check_numbers(claim: Claim, results: list[Scored], graph) -> Claim:
    """Every figure in the claim must appear in the evidence.

    Deterministic, no second model. This is what catches the Schedule-figure
    class of error — the correct amount was in `Provision.penalty` the whole
    time; nothing had ever compared the model's number against it.
    """
    figures = extract_figures(claim.text)
    if not figures:
        return claim

    evidence: list[str] = [r.chunk.verbatim for r in results]
    for node_id in claim.cites:
        provision = graph.provisions.get(node_id)
        if provision:
            evidence.append(provision.text)
            if provision.penalty:
                evidence.append(provision.penalty)

    supported: set[float] = set()
    for text in evidence:
        for _, value in extract_figures(text or ""):
            supported.add(value)

    unsupported = [surface for surface, value in figures if value not in supported]
    if unsupported:
        claim.verdict = Verdict.UNSUPPORTED
        claim.note = (f"{', '.join(unsupported)} does not appear in the cited "
                      f"text — check it against the Act before relying on it")
    return claim


def verify(claims: list[Claim], results: list[Scored], graph, *,
           quote_check: bool, numeric_check: bool) -> list[Claim]:
    """Run the free checks. Entailment, when enabled, runs separately."""
    for claim in claims:
        if claim.verdict is Verdict.UNSUPPORTED:
            continue                       # already failed a cheaper check
        if quote_check:
            claim = check_quote(claim, results, graph)
        if numeric_check and claim.verdict is not Verdict.UNSUPPORTED:
            claim = check_numbers(claim, results, graph)
    return claims
