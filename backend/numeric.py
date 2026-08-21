"""
Every number in the answer must exist in the evidence. Deterministic.

This is the check that catches V1's recorded Schedule-figure error, and it
catches it by arithmetic rather than by judgement. The blueprint proposed an
NLI verifier for the same job; that is a ~184M-parameter model, several
seconds of CPU per answer, and a probabilistic verdict. For a wrong rupee
figure it is entirely unnecessary — the correct amount is sitting in the
graph, and the wrong one simply is not.

The rule: a number in the answer is supported if it appears in a retrieved
provision's verbatim text, in a graph penalty field, or in the question
itself. Anything else is flagged. That is a strong invariant for this corpus
because the answer is supposed to be grounded in quoted statute — a figure
with no source in the evidence was, definitionally, produced by the model.

Two things it deliberately does NOT do:

  * It does not flag on absence. A correct answer that omits a figure is not
    wrong, so recall of evidence numbers is not measured here.
  * It does not run on template answers. Those are assembled FROM the graph,
    so every number in them is sourced by construction, and checking would
    only manufacture false positives from formatting differences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Indian numbering is the point: the Act writes "two hundred and fifty crore
# rupees" and a user asks about "250 crore". Both forms must reduce to the
# same value or the check is trivially defeated by phrasing.
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_SCALES = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000,
           "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000,
           "billion": 1_000_000_000}

_WORD_ALT = "|".join(sorted(_WORDS, key=len, reverse=True))
_SCALE_ALT = "|".join(sorted(_SCALES, key=len, reverse=True))

RE_WORD_AMOUNT = re.compile(
    rf"\b((?:(?:{_WORD_ALT}|hundred|and)\s+)+)({_SCALE_ALT})\b", re.I)
RE_DIGIT_AMOUNT = re.compile(
    rf"(?:₹|rs\.?\s*|inr\s*)?(\d[\d,]*(?:\.\d+)?)\s*({_SCALE_ALT})?\b", re.I)
# Durations carry the same risk as money — "seventy-two hours" vs "24 hours"
# is a compliance answer, and the Rules state real periods.
RE_DURATION = re.compile(
    r"\b(\d+|" + _WORD_ALT + r")[-\s]+(day|days|month|months|year|years|"
    r"hour|hours|week|weeks)\b", re.I)

# Section and rule numbers are citations, verified by citations.py against the
# graph. Re-checking them here would double-report the same problem in two
# different vocabularies.
RE_CITATION_NUMBER = re.compile(
    r"(?:§|\bsections?\b|\brules?\b|\bschedule\s+entry\b|\bclause\b|"
    r"\bsub-?(?:section|rule)\b)\s*\(?\d", re.I)


@dataclass
class NumericClaim:
    surface: str          # as written in the answer
    value: float          # canonical
    unit: str             # "rupees" | "days" | ...
    verdict: str          # supported | unsupported
    note: str = ""

    def to_dict(self) -> dict:
        return {"claim": self.surface, "value": self.value, "unit": self.unit,
                "verdict": self.verdict, "note": self.note}


def _word_value(phrase: str) -> int:
    total = current = 0
    for token in phrase.lower().split():
        if token in ("and",):
            continue
        if token == "hundred":
            current = (current or 1) * 100
        elif token in _WORDS:
            current += _WORDS[token]
    return total + current


def extract(text: str) -> list[tuple[str, float, str]]:
    """Every (surface, canonical value, unit) in `text`.

    Spans already consumed are masked out rather than skipped, so
    "250 crore rupees" is read once as 2.5e9 and not also as a bare 250.
    """
    found: list[tuple[str, float, str]] = []
    masked = list(text)

    def consume(start: int, end: int) -> None:
        for i in range(start, end):
            masked[i] = "\0"

    for match in RE_WORD_AMOUNT.finditer(text):
        value = _word_value(match.group(1)) * _SCALES[match.group(2).lower()]
        found.append((match.group(0).strip(), float(value), "rupees"))
        consume(*match.span())

    current = "".join(masked)

    for match in RE_DURATION.finditer(current):
        raw = match.group(1).lower()
        amount = float(raw) if raw.isdigit() else float(_WORDS.get(raw, 0))
        if not amount:
            continue
        unit = match.group(2).lower().rstrip("s")
        found.append((match.group(0).strip(), amount, unit))
        consume(*match.span())

    current = "".join(masked)

    for match in RE_DIGIT_AMOUNT.finditer(current):
        # Skip anything that is part of a citation — citations.py owns those.
        window = current[max(0, match.start() - 24):match.start()]
        if RE_CITATION_NUMBER.search(window + match.group(0)):
            continue
        try:
            base = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        scale = _SCALES.get((match.group(2) or "").lower(), 1)
        # A bare small integer is almost always a list marker or a year, not
        # a claim. Amounts that matter here are scaled or large.
        if scale == 1 and base < 1000:
            continue
        found.append((match.group(0).strip(), base * scale, "rupees"))

    return found


def check(answer: str, evidence_texts: list[str], graph_amounts: list[str],
          question: str = "") -> list[NumericClaim]:
    """Flag every number in `answer` with no source in the evidence."""
    supported: set[float] = set()
    for source in list(evidence_texts) + list(graph_amounts) + [question]:
        for _surface, value, _unit in extract(source or ""):
            supported.add(value)

    claims: list[NumericClaim] = []
    for surface, value, unit in extract(answer):
        if value in supported:
            claims.append(NumericClaim(surface, value, unit, "supported"))
            continue
        # Tolerate unit-scale restatement: "2.5 billion" for 250 crore is the
        # same amount, and flagging it would be a false positive.
        near = any(abs(value - candidate) < 1e-6 for candidate in supported)
        claims.append(NumericClaim(
            surface, value, unit,
            "supported" if near else "unsupported",
            "" if near else
            "this figure does not appear in any retrieved provision or in the "
            "graph's own amounts"))
    return claims


def has_contradiction(claims: list[NumericClaim]) -> bool:
    return any(c.verdict == "unsupported" for c in claims)
