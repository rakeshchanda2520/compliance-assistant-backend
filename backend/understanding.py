"""
What is this question, and is it even ours? Decided before retrieval runs.

Two decisions, both made here, both before a single chunk is scored:

    jurisdiction   is this question about THIS corpus at all?
    intent         which of the answer paths should serve it?

Each uses the same three-tier cascade, and the ordering is the whole design:

    1. REGEX / TOKEN SET   explicit, deterministic, confidence 1.0
    2. EMBEDDING           cosine against labelled exemplars.yaml
    3. FALLTHROUGH         "general" — synthesis, the V1 path

There is deliberately NO language-model tier. That is not caution about
models generally — it is that this specific job wants a decision that is
auditable and identical on every repeat. An embedding classifier gives both:
a misroute logs as `matched "what is the fine for this" @ 0.83`, and the fix
is one line in exemplars.yaml. An LLM's misclassification traces to nothing,
costs 300ms-2s, and can differ between two runs of the same question.

The embedding tier is close to free here: the query vector is already being
computed for dense retrieval, so classification is a dot product against a
few dozen cached exemplar vectors.

WHY THE JURISDICTION GATE EXISTS
--------------------------------
V1's abstention gate is a BM25 score threshold, and its documented blind spot
is adjacent legal domains: GDPR and HIPAA questions score as high as genuine
DPDP ones because they share real legal vocabulary. No threshold can separate
them — "what must a data controller do about a breach" is a perfectly
well-formed question that this corpus simply does not answer. That is a scope
judgement, and it needs its own gate.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config, embeddings

log = logging.getLogger(__name__)

EXEMPLARS = Path(__file__).resolve().parent / "exemplars.yaml"

# Tier 1, jurisdiction. Unambiguous names of other instruments. A question
# carrying one of these is about another country's law regardless of how it
# is phrased.
FOREIGN_MARKERS = {
    "gdpr", "hipaa", "ccpa", "cpra", "lgpd", "pipeda", "pdpa", "appi",
    "eu ai act", "uk gdpr", "article 33", "article 17", "schrems",
    "supervisory authority", "data protection authority", "dpa",
    "european union", "european commission", "safe harbor", "privacy shield",
}
# Names only this corpus uses. Their presence pulls a question back domestic.
DPDP_MARKERS = {
    "dpdp", "digital personal data protection", "data fiduciary",
    "data principal", "consent manager", "significant data fiduciary",
    "data protection board", "crore", "lakh", "₹", "rupee", "gazette",
    "indian", "india",
}
# Shared vocabulary. Present in both regimes, decisive in neither — a question
# built only from these is answered, but stamped with a scope caveat.
AMBIGUOUS_MARKERS = {
    "data controller", "data processor", "consent", "breach", "personal data",
    "processing", "erasure", "rectification", "data subject",
}

# Tier 1, intent. Only high-confidence phrasings belong here; anything
# needing a "probably" belongs in exemplars.yaml instead.
INTENT_PATTERNS = {
    # Ordered: direct_lookup first, because "what does section 8 say" also
    # matches nothing else but would be caught by a looser pattern below.
    "direct_lookup": re.compile(
        r"\b(?:what\s+does|show|explain|read|quote|text\s+of|give\s+me)\b"
        r".{0,20}\b(?:section|§|rule|schedule)\b", re.I),
    "penalty": re.compile(
        r"\b(?:fine|fined|penalt\w*|punish\w*|liable|liability|sanction\w*)\b"
        r"|₹|\bcrore\b|\blakh\b", re.I),
    "retention": re.compile(
        r"\bhow\s+long\b|\bretain\w*\b|\bretention\b|\bstorage\s+period\b"
        r"|\bdelete\b|\berase\w*\b|\bkeep\s+(?:the\s+)?(?:data|records?)\b", re.I),
    "definition": re.compile(
        r"\bwhat\s+is\s+(?:an?\s+)?(?:the\s+)?\w+|^\s*define\b|\bdefinition\s+of\b"
        r"|\bmeaning\s+of\b|\bwho\s+(?:counts\s+as|is)\s+an?\b"
        r"|\bwhat\s+counts\s+as\b", re.I),
    "temporal": re.compile(
        r"\bin\s+force\b|\bcommence\w*\b|\beffective\s+date\b"
        r"|\bwhen\s+does\b.{0,30}\b(?:apply|start|come\s+into)\b", re.I),
    "obligation": re.compile(
        r"\bwhat\s+(?:must|should|do)\s+we\b|\bdut(?:y|ies)\b|\bobligat\w*\b"
        r"|\brequire\w*\s+to\b|\bwhat\s+do\s+we\s+owe\b|\bare\s+we\s+allowed\b", re.I),
}

# A provision named outright: "section 8", "rule 6(1)", "§8(5)".
RE_PROVISION_REF = re.compile(
    r"\b(?:section|§)\s*(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\brule\s*(\d{1,2})((?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)", re.I)
RE_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")

# Anaphora — a question that cannot stand alone (phase 5).
RE_ANAPHORA = re.compile(
    r"\b(?:its|it|that|this|these|those|them|the\s+same|there(?:of|to))\b"
    r"|^\s*(?:and|what\s+about|how\s+about|why)\b", re.I)

# Cosine floors for the embedding tier. Deliberately not equal: routing to the
# wrong TEMPLATE produces a confidently wrong answer, while a wrong
# jurisdiction call only over- or under-refuses, so intent is held to a
# higher bar.
INTENT_MIN_COSINE = 0.62
JURISDICTION_MIN_COSINE = 0.58
# Foreign must beat domestic by this much. Without a margin, a DPDP question
# using shared vocabulary ("what must a data processor do") flips foreign on
# a hairline difference.
JURISDICTION_MARGIN = 0.04

TEMPLATE_INTENTS = ("penalty", "definition", "retention", "direct_lookup")


@dataclass
class Understanding:
    """The routing decision, with every input that produced it.

    Verbose on purpose: this object is streamed to the client as the `router`
    SSE event and written to the audit log, so a wrong route is diagnosable
    from the record alone rather than by re-running the question.
    """
    intent: str = "general"
    intent_via: str = "fallthrough"          # regex | embedding | fallthrough
    intent_confidence: float = 0.0
    intent_exemplar: str = ""

    jurisdiction: str = "domestic"           # domestic | foreign | ambiguous
    jurisdiction_via: str = "default"
    jurisdiction_confidence: float = 0.0
    jurisdiction_markers: list[str] = field(default_factory=list)

    provision_id: str = ""                   # set when a provision is named
    has_anaphora: bool = False
    caveat: str = ""

    @property
    def should_abstain(self) -> bool:
        return self.jurisdiction == "foreign"

    @property
    def uses_template(self) -> bool:
        return self.intent in TEMPLATE_INTENTS

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_via": self.intent_via,
            "intent_confidence": round(self.intent_confidence, 4),
            "intent_exemplar": self.intent_exemplar,
            "jurisdiction": self.jurisdiction,
            "jurisdiction_via": self.jurisdiction_via,
            "jurisdiction_confidence": round(self.jurisdiction_confidence, 4),
            "jurisdiction_markers": self.jurisdiction_markers,
            "provision_id": self.provision_id,
            "caveat": self.caveat,
        }


class Classifier:
    """Holds the exemplar vectors. Built once at startup; embedding a few
    dozen short strings is a one-off cost, and doing it per request would put
    a network round trip on the hot path for a decision that never changes."""

    def __init__(self) -> None:
        raw = yaml.safe_load(EXEMPLARS.read_text(encoding="utf-8")) or {}
        self.intent_exemplars: dict[str, list[str]] = raw.get("intents", {}) or {}
        self.jurisdiction_exemplars: dict[str, list[str]] = \
            raw.get("jurisdiction", {}) or {}
        self._vectors: dict[str, list[tuple[str, str, list[float]]]] = {}
        self.ready = False
        self.error = ""

    def warm(self, dense_index=None) -> None:
        """Make exemplar vectors available.

        Prefers the ones cached in the dense index at build time. That is not
        just an optimisation: embedding ~50 exemplars on every restart costs
        real requests against a provider whose free tier caps them daily, and
        a restart was measured exhausting the quota and silently dropping the
        router to its regex tier. Static work belongs at build time.

        Falls back to embedding them live for an index built before this
        existed. Failure is never fatal — the regex tier still works, and a
        degraded router beats a backend that will not start.
        """
        if not config.INTENT_ROUTER:
            return

        cached = getattr(dense_index, "exemplars", None)
        if cached:
            for group, label, text, vector in cached:
                self._vectors.setdefault(group, []).append((label, text, vector))
            self.ready = True
            log.info("router: %d intent + %d jurisdiction exemplars from the "
                     "build index (no embedding calls)",
                     len(self._vectors.get("intent", [])),
                     len(self._vectors.get("jurisdiction", [])))
            return

        log.info("router: no cached exemplars in the dense index; embedding "
                 "them now. Rebuild with `python -m kg_build --embed` to "
                 "avoid this on every restart.")
        try:
            for group, mapping in (("intent", self.intent_exemplars),
                                   ("jurisdiction", self.jurisdiction_exemplars)):
                pairs = [(label, text)
                         for label, texts in mapping.items() for text in texts]
                if not pairs:
                    continue
                vectors = embeddings.embed([t for _, t in pairs], is_query=True)
                self._vectors[group] = [
                    (label, text, vec)
                    for (label, text), vec in zip(pairs, vectors)]
            self.ready = True
            log.info("router: %d intent + %d jurisdiction exemplars embedded",
                     len(self._vectors.get("intent", [])),
                     len(self._vectors.get("jurisdiction", [])))
        except Exception as exc:                      # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("router: exemplars not embedded, regex tier only (%s)",
                        self.error)

    def nearest(self, group: str, vector: list[float]) -> tuple[str, str, float]:
        """(label, exemplar, cosine) of the closest exemplar in `group`."""
        best = ("", "", -1.0)
        for label, text, exemplar_vec in self._vectors.get(group, []):
            score = embeddings._cosine(vector, exemplar_vec)
            if score > best[2]:
                best = (label, text, score)
        return best

    def best_per_label(self, group: str,
                       vector: list[float]) -> dict[str, float]:
        """Best cosine per label — needed for the foreign-vs-domestic margin,
        which a single nearest-neighbour lookup cannot express."""
        out: dict[str, float] = {}
        for label, _text, exemplar_vec in self._vectors.get(group, []):
            score = embeddings._cosine(vector, exemplar_vec)
            if score > out.get(label, -1.0):
                out[label] = score
        return out


def provision_reference(question: str) -> str:
    """Node id of a provision named outright, or "".

    This is what sends "What does section 8 say?" to a direct graph lookup
    instead of through BM25 — a question that names its own answer should
    never be a search problem.
    """
    match = RE_PROVISION_REF.search(question)
    if not match:
        return ""
    if match.group(1):
        return "-".join(["s", match.group(1)] + RE_PART.findall(match.group(2) or ""))
    if match.group(3):
        return "-".join(["r", match.group(3)] + RE_PART.findall(match.group(4) or ""))
    return ""


def _markers(low: str, markers: set[str]) -> list[str]:
    return sorted(m for m in markers if m in low)


def understand(question: str, classifier: Classifier | None,
               query_vector: list[float] | None = None) -> Understanding:
    """Route one question. Never raises — a classifier failure degrades to
    the V1 path rather than failing the request."""
    result = Understanding()
    if not config.INTENT_ROUTER:
        return result

    low = question.lower()
    result.has_anaphora = bool(RE_ANAPHORA.search(question))

    # ---------------- jurisdiction, tier 1: explicit markers -------------- #
    foreign_hits = _markers(low, FOREIGN_MARKERS)
    domestic_hits = _markers(low, DPDP_MARKERS)

    if foreign_hits and not domestic_hits:
        result.jurisdiction = "foreign"
        result.jurisdiction_via = "marker"
        result.jurisdiction_confidence = 1.0
        result.jurisdiction_markers = foreign_hits
        return result                        # abstain; nothing else matters

    if domestic_hits:
        result.jurisdiction = "domestic"
        result.jurisdiction_via = "marker"
        result.jurisdiction_confidence = 1.0
        result.jurisdiction_markers = domestic_hits

    # ---------------- jurisdiction, tier 2: exemplars --------------------- #
    elif classifier and classifier.ready and query_vector:
        scores = classifier.best_per_label("jurisdiction", query_vector)
        foreign = scores.get("foreign", -1.0)
        domestic = scores.get("domestic", -1.0)
        if foreign >= JURISDICTION_MIN_COSINE and foreign - domestic >= JURISDICTION_MARGIN:
            result.jurisdiction = "foreign"
            result.jurisdiction_via = "embedding"
            result.jurisdiction_confidence = foreign
            return result
        result.jurisdiction_confidence = max(foreign, domestic)
        result.jurisdiction_via = "embedding"

    # Only shared vocabulary and nothing decisive: answer, but say so.
    if result.jurisdiction == "domestic" and not domestic_hits:
        if ambiguous := _markers(low, AMBIGUOUS_MARKERS):
            result.jurisdiction = "ambiguous"
            result.jurisdiction_markers = ambiguous
            result.caveat = (
                "This answer is based only on India's DPDP Act, 2023 and its "
                "Rules, 2025. The terms used here also appear in other "
                "privacy regimes, which this system does not cover.")

    # ---------------- intent, tier 0: a provision is named ---------------- #
    if provision_id := provision_reference(question):
        result.provision_id = provision_id
        # "What's the penalty under section 8?" names a provision but wants the
        # penalty path — only route to a bare lookup when nothing else fired.
        if not INTENT_PATTERNS["penalty"].search(question) and \
           not INTENT_PATTERNS["retention"].search(question):
            result.intent = "direct_lookup"
            result.intent_via = "regex"
            result.intent_confidence = 1.0
            return result

    # ---------------- intent, tier 1: regex ------------------------------- #
    for name, pattern in INTENT_PATTERNS.items():
        if pattern.search(question):
            result.intent = name
            result.intent_via = "regex"
            result.intent_confidence = 1.0
            return result

    # ---------------- intent, tier 2: exemplars --------------------------- #
    if classifier and classifier.ready and query_vector:
        label, exemplar, score = classifier.nearest("intent", query_vector)
        if label and score >= INTENT_MIN_COSINE:
            result.intent = label
            result.intent_via = "embedding"
            result.intent_confidence = score
            result.intent_exemplar = exemplar
            return result
        # Logged as a rule gap: these are the questions exemplars.yaml should
        # grow to cover, and they are invisible unless recorded.
        log.info("router gap: no intent matched (best %s @ %.3f) for %r",
                 label or "-", score, question[:80])
        result.intent_confidence = max(score, 0.0)
        result.intent_exemplar = exemplar

    # ---------------- tier 3: general synthesis --------------------------- #
    return result
