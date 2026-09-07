"""
Routing: what kind of question is this, and is it even ours?

Three tiers, and the proportions matter. Tier 1 regex is free and instant, so
it runs first — but on real traffic it is a **fast path, not the primary
one**. People do not phrase questions the way a pattern expects. Tier 2
(embedding similarity against cached exemplars) is the workhorse, and tier 3
guarantees that a routing miss is never an answer failure: it falls through to
synthesis, which answers from the same retrieved provisions.

The safety property that makes this design work:

    a routing MISS costs a model call.
    a routing ERROR costs a confidently wrong answer.

So every tier is tuned to abstain from deciding rather than to guess. An
unrecognised question becomes `GENERAL` and takes the model path, which is
always safe. Only a high-confidence match takes a template.
"""
from __future__ import annotations

import logging
import re

from core.models import Intent, Jurisdiction, Tier, Understanding
from . import normalize as norm

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# jurisdiction
# --------------------------------------------------------------------------- #

# Named regimes this corpus does not cover. A GDPR question scores as high on
# BM25 as a genuine one because it shares real legal vocabulary — no score
# threshold can separate them. It is a question of SCOPE, not confidence, so it
# gets its own gate, before retrieval.
# NAMED REGIMES AND INSTITUTIONS ONLY — never bare country names.
# "our processor in Singapore lost records, what do we owe?" is a genuine DPDP
# cross-border question (section 16 territorial scope). Listing "singapore"
# here refused it as foreign law, which is a FALSE REFUSAL — the most
# expensive error this system can make, because a real compliance question
# comes back unanswered. Naming a country is not asking about its statute.
FOREIGN_MARKERS = frozenset({
    "gdpr", "hipaa", "ccpa", "cpra", "lgpd", "pipeda", "pdpa", "appi", "popia",
    "uk gdpr", "eu ai act", "schrems", "privacy shield", "safe harbor",
    "european union", "european commission", "supervisory authority",
    "article 33", "article 17", "recital",
})

# Names only this corpus uses. Their presence pulls a question back domestic.
DOMESTIC_MARKERS = frozenset({
    "dpdp", "dpdpa", "digital personal data protection",
    "data fiduciary", "data principal", "consent manager",
    "significant data fiduciary", "data protection board", "dpb",
    "crore", "lakh", "₹", "rupee", "rupees", "gazette", "india", "indian",
    "meity", "central government", "adjudicating officer",
})

# Present in both regimes, decisive in neither. These do NOT trigger a caveat
# on their own: "breach", "consent" and "personal data" are the bread-and-butter
# vocabulary of THIS statute, and stamping a scope warning on every obligation
# question is noise that trains users to ignore the warning that matters. A
# caveat is raised only when a foreign regime is actually NAMED alongside.
AMBIGUOUS_MARKERS = frozenset({
    "data controller", "data processor", "consent", "breach", "personal data",
    "processing", "erasure", "rectification", "data subject", "privacy",
    "controller", "processor", "dpo", "data protection officer",
})

SCOPE_CAVEAT = ("This answer is from India's DPDP Act, 2023 and its Rules, "
                "2025. The terms in your question are used by other privacy "
                "regimes too — check you meant the Indian law.")


def _markers_in(text: str, markers: frozenset[str]) -> tuple[str, ...]:
    low = f" {text.lower()} "
    return tuple(sorted(m for m in markers
                        if re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", low)))


# --------------------------------------------------------------------------- #
# intent — tier 1
# --------------------------------------------------------------------------- #

# HIGH-CONFIDENCE phrasings only. Anything needing a "probably" belongs in the
# exemplar file for tier 2, not here — a regex that fires on a maybe is how a
# question gets a confidently wrong template.
#
# Order matters: direct_lookup is tested first because "what does section 8
# say" would also satisfy the looser definition pattern below.
INTENT_PATTERNS: tuple[tuple[Intent, re.Pattern], ...] = (
    (Intent.DIRECT_LOOKUP, re.compile(
        r"\b(?:what\s+does|show|explain|read|quote|text\s+of|give\s+me|"
        r"tell\s+me\s+about)\b.{0,24}\b(?:section|§|rule|schedule)\b\s*\d", re.I)),
    (Intent.PENALTY, re.compile(
        r"\b(?:fine|fined|penalt\w*|punish\w*|liable|liability|sanction\w*)\b"
        r"|₹\s*\d|\bcrore\b|\blakh\b", re.I)),
    (Intent.RETENTION, re.compile(
        r"\bhow\s+long\b|\bretain\w*\b|\bretention\b|\bstorage\s+period\b"
        r"|\bdelete\b|\berase\w*\b|\bkeep\s+(?:the\s+)?(?:data|records?)\b", re.I)),
    (Intent.DEFINITION, re.compile(
        r"\bwhat\s+(?:is|are)\s+(?:an?\s+|the\s+)?\w+"
        r"|^\s*define\b|\bdefinition\s+of\b|\bmeaning\s+of\b"
        r"|\bwho\s+(?:counts\s+as|is)\s+an?\b|\bwhat\s+counts\s+as\b", re.I)),
    (Intent.TEMPORAL, re.compile(
        r"\bin\s+force\b|\bcommence\w*\b|\beffective\s+date\b"
        r"|\bwhen\s+does\b.{0,30}\b(?:apply|start|come\s+into)\b", re.I)),
    # The modal need not sit adjacent to "what": "what security safeguards
    # MUST WE implement" is the same question as "what must we implement",
    # and requiring adjacency dropped it to fallthrough. Matching the modal
    # plus subject anywhere in the sentence is both looser and still precise —
    # "must we", "should I", "do we have to" are not ambiguous phrases.
    (Intent.OBLIGATION, re.compile(
        r"\b(?:must|should)\s+(?:we|i|my\s+company|a\s+data\s+fiduciary)\b"
        r"|\bdo\s+(?:we|i)\s+(?:have\s+to|need\s+to)\b"
        r"|\bdut(?:y|ies)\b|\bobligat\w*\b|\brequire\w*\s+to\b"
        r"|\bwhat\s+do\s+we\s+owe\b|\b(?:are|am)\s+(?:we|i)\s+allowed\b"
        r"|\bwhat\s+(?:do|are)\s+(?:we|i|my\s+company)\b", re.I)),
)

# A provision named outright: "section 8", "rule 6(1)", "§8(5)".
PROVISION_REF = re.compile(
    r"\b(?:section|§)\s*(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\brules?\s*(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)", re.I)
_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")

# A question that cannot stand alone — it inherits its subject from the turn
# before. Detected so a follow-up can borrow the prior turn's PROVISIONS.
ANAPHORA = re.compile(
    r"\b(?:its|it|that|this|these|those|them|the\s+same|there(?:of|to))\b"
    r"|^\s*(?:and|what\s+about|how\s+about|why|ok(?:ay)?\s+(?:and|but))\b", re.I)


def provision_reference(text: str) -> str:
    """The provision id a question names outright, or ""."""
    m = PROVISION_REF.search(text)
    if not m:
        return ""
    if m.group(1):
        return "-".join(["s", m.group(1)] + _PART.findall(m.group(2) or ""))
    return "-".join(["r", m.group(3)] + _PART.findall(m.group(4) or ""))


# --------------------------------------------------------------------------- #
# the router
# --------------------------------------------------------------------------- #

class Router:
    """Stateless apart from the cached exemplar vectors.

    `exemplars` maps a group ("intent" / "jurisdiction") to a list of
    (label, vector). They are computed at BUILD time and cached in the index,
    so a restart costs zero embedding calls — an earlier system re-embedded
    ~45 exemplars on every boot and exhausted a daily free-tier quota in a
    single afternoon of restarts, silently dropping itself to regex-only.
    """

    def __init__(self, exemplars: dict[str, list[tuple[str, list[float]]]] | None = None,
                 *, intent_floor: float, jurisdiction_floor: float,
                 jurisdiction_margin: float) -> None:
        self.exemplars = exemplars or {}
        self.intent_floor = intent_floor
        self.jurisdiction_floor = jurisdiction_floor
        self.jurisdiction_margin = jurisdiction_margin

    # -- tier 2 helpers ----------------------------------------------------- #

    def _best(self, group: str, vector: list[float]) -> tuple[str, float]:
        pool = self.exemplars.get(group) or []
        if not pool or not vector:
            return "", 0.0
        best_label, best_score = "", -1.0
        for label, ex in pool:
            # Vectors are L2-normalised at build time, so a dot product IS the
            # cosine. Not re-normalising here keeps this loop cheap.
            score = sum(a * b for a, b in zip(vector, ex))
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score

    def _best_per_label(self, group: str, vector: list[float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for label, ex in self.exemplars.get(group) or []:
            if not vector:
                break
            score = sum(a * b for a, b in zip(vector, ex))
            if score > out.get(label, -1.0):
                out[label] = score
        return out

    # -- jurisdiction ------------------------------------------------------- #

    def jurisdiction(self, text: str, vector: list[float] | None
                     ) -> tuple[Jurisdiction, tuple[str, ...], str]:
        """Decided on the ORIGINAL question text.

        Never on a rewritten one: a rewrite into statutory vocabulary can
        launder the foreign markers out of "Under GDPR, what is the fine?" and
        this gate is the one thing no score threshold can replace.
        """
        foreign = _markers_in(text, FOREIGN_MARKERS)
        domestic = _markers_in(text, DOMESTIC_MARKERS)

        # An explicit foreign regime with no Indian marker is refused outright,
        # regardless of what any model thinks.
        if foreign and not domestic:
            return Jurisdiction.FOREIGN, foreign, ""
        if foreign and domestic:
            # Both named — likely a comparison. Answer the Indian half and say so.
            return (Jurisdiction.DOMESTIC, foreign + domestic,
                    "Your question also names another privacy regime. Only "
                    "India's DPDP Act, 2023 and Rules, 2025 are covered here.")
        if domestic:
            return Jurisdiction.DOMESTIC, domestic, ""

        # Tier 2: no decisive marker either way.
        if vector and self.exemplars.get("jurisdiction"):
            scores = self._best_per_label("jurisdiction", vector)
            f = scores.get("foreign", 0.0)
            d = scores.get("domestic", 0.0)
            # Foreign must WIN BY A MARGIN. Without one, a DPDP question using
            # shared vocabulary flips foreign on a hairline difference.
            if f >= self.jurisdiction_floor and f - d >= self.jurisdiction_margin:
                return Jurisdiction.FOREIGN, ("similarity",), ""

        # Default domestic. This IS the DPDP assistant: a question arriving
        # here is about the DPDP Act unless it says otherwise. Shared
        # vocabulary alone is not evidence to the contrary, and biasing toward
        # answering is the whole asymmetry — refusing a real compliance
        # question breaks the product, while answering an out-of-scope one
        # costs a call and yields an honest "these provisions do not settle
        # this".
        ambiguous = _markers_in(text, AMBIGUOUS_MARKERS)
        return Jurisdiction.DOMESTIC, ambiguous, ""

    # -- intent ------------------------------------------------------------- #

    def intent(self, text: str, vector: list[float] | None
               ) -> tuple[Intent, Tier, float]:
        for intent, pattern in INTENT_PATTERNS:
            if pattern.search(text):
                return intent, Tier.REGEX, 1.0

        if vector and self.exemplars.get("intent"):
            label, score = self._best("intent", vector)
            # A HIGHER bar than jurisdiction, deliberately: routing to the
            # wrong TEMPLATE yields a confidently wrong answer, while a wrong
            # jurisdiction call only over- or under-refuses.
            if label and score >= self.intent_floor:
                try:
                    return Intent(label), Tier.EMBEDDING, score
                except ValueError:
                    log.warning("exemplar names unknown intent %r", label)

        # Fallthrough is SAFE by construction: GENERAL takes the model path,
        # which answers from the same retrieved provisions.
        return Intent.GENERAL, Tier.FALLTHROUGH, 0.0

    # -- the whole decision ------------------------------------------------- #

    def understand(self, question: str, vector: list[float] | None = None
                   ) -> Understanding:
        n = norm.normalize(question)

        # Jurisdiction reads the ORIGINAL text plus the normalised form, so a
        # marker cannot be lost to typo-correction or shorthand expansion.
        juris, markers, caveat = self.jurisdiction(
            f"{question} {n.clean}", vector)

        intent, tier, confidence = self.intent(n.classify, vector)
        provision_id = provision_reference(n.clean)

        # A named provision with no other signal IS a direct lookup — the
        # question already states its own answer's location.
        if provision_id and intent is Intent.GENERAL:
            intent, tier, confidence = Intent.DIRECT_LOOKUP, Tier.REGEX, 1.0

        # DIRECT_LOOKUP without a resolvable provision id is not a lookup at
        # all; it is a general question that happened to match the phrasing.
        if intent is Intent.DIRECT_LOOKUP and not provision_id:
            intent, tier, confidence = Intent.GENERAL, Tier.FALLTHROUGH, 0.0

        return Understanding(
            question=question, intent=intent, tier=tier, jurisdiction=juris,
            confidence=confidence, provision_id=provision_id, markers=markers,
            caveat=caveat,
            has_anaphora=bool(ANAPHORA.search(n.clean)),
        )
