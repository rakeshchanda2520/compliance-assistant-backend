"""
The domain types. No I/O, no framework, no provider — importable from a test
with nothing running.

Everything downstream (retrieval, answering, verification, the API) speaks in
these. Keeping them free of infrastructure is what lets the whole answer
pipeline be exercised offline against a fixture corpus, which is in turn what
makes the eval in §5 of NEW_FRAMEWORK affordable to run often.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Provision:
    """A node in the graph. `text` is the instrument's exact words.

    A Section in this corpus is frequently a CONTAINER: sections 9, 10 and 32
    carry a headnote and no text of their own, because their substance lives
    in sub-sections. `display_text()` in verification.citations is what turns
    that into something a reader can see; storing it here would mean
    duplicating statute text across nodes.
    """
    id: str
    label: str
    kind: str
    text: str = ""
    headnote: str = ""
    penalty: str = ""
    chapter: str = ""
    page: int = 0
    authority: float = 0.0


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. Not the same id space as Provision — see core.ids."""
    id: str
    node_id: str
    kind: str
    label: str
    verbatim: str
    header: str = ""
    headnote: str = ""
    chapter: str = ""
    page: int = 0
    # Generated, indexed, NEVER quoted or cited. Two separate layers:
    #   plain_english / questions  a plain-language paraphrase (Act only)
    #   context                    NEW_FRAMEWORK P1's situating context
    # Both exist to help RETRIEVAL find the right verbatim text. Neither may
    # reach an answer, and neither enters the round-trip check.
    plain_english: str = ""
    questions: tuple[str, ...] = ()
    context: str = ""

    def lexical_text(self, *, contextual: bool) -> str:
        """What a BM25 index sees.

        Two indexes are built from the same chunks and fused as SEPARATE
        rankers — see retrieval.fusion for why they are not merged.

        The raw index includes `header` and `label` alongside `verbatim`.
        Those are EXTRACTED from the Gazette, not generated, so they cannot
        dilute the exact-token guarantee the raw ranker exists to protect —
        and they carry the strongest signal a chunk has. Omitting them made
        "what is a Data Fiduciary?" fail to surface `def-data-fiduciary` at
        all: the query tokens reduce to ["data", "fiduciary"], both near-
        universal in this corpus, while the one string that names the chunk
        exactly ("Definition of 'Data Fiduciary'") was not indexed.

        The contextual index adds the GENERATED layers on top. Those stay out
        of the raw index precisely so a bad generated context can never remove
        a signal the raw ranker would have found.
        """
        structural = "\n".join(p for p in (self.label, self.header, self.verbatim) if p)
        if not contextual:
            return structural
        generated = "\n".join(p for p in (self.context, self.plain_english,
                                          " ".join(self.questions)) if p)
        return f"{generated}\n{structural}" if generated else structural

    def embedding_text(self, max_chars: int) -> str:
        """What the embedder sees, truncated to the model's input cap.

        Leads with the semantic layer and fills the remainder with verbatim
        text: hosted embedding models cap input hard (512 tokens on the
        current one, which a 6,000-character chunk blows through), and the
        first tokens are the ones that survive.
        """
        lead = [p for p in (self.context, self.headnote, self.plain_english,
                            " ".join(self.questions)) if p]
        head = "\n".join(lead)[:max_chars]
        remaining = max_chars - len(head) - 1
        if remaining <= 0:
            return head
        tail = self.verbatim[:remaining]
        if len(tail) == remaining and " " in tail:
            tail = tail[:tail.rfind(" ")]      # never split a word
        return f"{head}\n{tail}".strip()


# --------------------------------------------------------------------------- #
# understanding
# --------------------------------------------------------------------------- #

class Jurisdiction(str, Enum):
    DOMESTIC = "domestic"
    FOREIGN = "foreign"
    AMBIGUOUS = "ambiguous"


class Intent(str, Enum):
    PENALTY = "penalty"
    DEFINITION = "definition"
    RETENTION = "retention"
    DIRECT_LOOKUP = "direct_lookup"
    OBLIGATION = "obligation"
    TEMPORAL = "temporal"
    GENERAL = "general"

    @property
    def has_template(self) -> bool:
        """Intents the graph can answer with no model at all."""
        return self in _TEMPLATE_INTENTS


_TEMPLATE_INTENTS = frozenset({
    Intent.PENALTY, Intent.DEFINITION, Intent.RETENTION, Intent.DIRECT_LOOKUP})


class Tier(str, Enum):
    """Which stage of the routing cascade decided. Recorded so a misroute is
    attributable to a rule rather than to "the classifier"."""
    REGEX = "regex"
    EMBEDDING = "embedding"
    FALLTHROUGH = "fallthrough"


@dataclass(frozen=True)
class Understanding:
    """The routing decision, and how it was reached."""
    question: str
    intent: Intent
    tier: Tier
    jurisdiction: Jurisdiction
    confidence: float = 0.0
    provision_id: str = ""          # set when the question names one outright
    markers: tuple[str, ...] = ()
    caveat: str = ""
    has_anaphora: bool = False
    rewritten: str = ""             # retrieval-only; never quoted, never cited
    sub_questions: tuple[str, ...] = ()

    @property
    def should_abstain(self) -> bool:
        return self.jurisdiction is Jurisdiction.FOREIGN

    @property
    def uses_template(self) -> bool:
        return self.intent.has_template

    def to_dict(self) -> dict:
        return {"intent": self.intent.value, "tier": self.tier.value,
                "jurisdiction": self.jurisdiction.value,
                "confidence": round(self.confidence, 3),
                "provision_id": self.provision_id,
                "markers": list(self.markers), "caveat": self.caveat,
                "has_anaphora": self.has_anaphora,
                "rewritten": self.rewritten,
                "sub_questions": list(self.sub_questions)}


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #

@dataclass
class Scored:
    """One retrieved chunk and every score that touched it.

    Scores are kept SEPARATE and named rather than collapsed into one number.
    The previous system stored a single `score` that meant BM25 in one
    configuration and Reciprocal Rank Fusion in another; an abstention gate
    calibrated on the first was silently fed the second, and refused every
    question that reached it. Never again — a consumer must name the scale it
    wants.
    """
    chunk: Chunk
    bm25: float = 0.0
    bm25_contextual: float = 0.0
    dense: float = 0.0
    fused: float = 0.0
    rerank: float | None = None
    hop: int = 0
    via: str = ""

    @property
    def node_id(self) -> str:
        return self.chunk.node_id

    def to_dict(self) -> dict:
        return {"id": self.node_id, "label": self.chunk.label,
                "kind": self.chunk.kind, "headnote": self.chunk.headnote,
                "hop": self.hop, "via": self.via,
                "bm25": round(self.bm25, 3),
                "bm25_contextual": round(self.bm25_contextual, 3),
                "dense": round(self.dense, 4),
                "fused": round(self.fused, 6),
                "rerank": None if self.rerank is None else round(self.rerank, 4)}


@dataclass
class RetrievalTrace:
    """What retrieval did, so an answer is auditable without re-running it."""
    query: str
    rewritten: str = ""
    sub_questions: tuple[str, ...] = ()
    vocab_hits: tuple[str, ...] = ()
    # Exact statutory phrases the question named, kept apart from vocab_hits
    # (which merges them with layperson synonyms for display). Naming a defined
    # term of THIS instrument is a different, much stronger claim than using a
    # word that maps to one, and only the gate can act on the difference.
    phrase_hits: tuple[str, ...] = ()
    # (as typed, as scored) for terms the index could not score at all. An
    # answer built on a guessed spelling must be auditable as such.
    repairs: tuple[tuple[str, str], ...] = ()
    # How many terms of the question AS ASKED the index can actually score.
    # BM25 is a SUM over these, so the number is what makes a score
    # comparable — or not — to a floor calibrated on longer questions.
    scorable_terms: int = 0
    # Total scoreable-position terms in the question as asked, stopwords
    # already removed. Paired with scorable_terms it says whether a query is
    # SHORT (few terms, all of them statutory) or merely OFF-TOPIC (many
    # terms, few of them statutory) — two cases a single count conflates.
    query_terms: int = 0
    # Named explicitly, with its scale, so no consumer has to guess. This is
    # the field an abstention gate reads.
    top_bm25: float = 0.0
    top_rerank: float | None = None
    rerank_margin: float | None = None
    candidates: int = 0
    fused: bool = False
    reranked: bool = False
    dense_error: str = ""
    seeds_from_prior_turn: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"query": self.query, "rewritten": self.rewritten,
                "sub_questions": list(self.sub_questions),
                "vocabulary": list(self.vocab_hits),
                "phrases": list(self.phrase_hits),
                "repairs": [list(r) for r in self.repairs],
                "scorable_terms": self.scorable_terms,
                "query_terms": self.query_terms,
                "top_bm25": round(self.top_bm25, 3),
                "top_rerank": None if self.top_rerank is None else round(self.top_rerank, 4),
                "rerank_margin": None if self.rerank_margin is None else round(self.rerank_margin, 4),
                "candidates": self.candidates, "fused": self.fused,
                "reranked": self.reranked, "dense_error": self.dense_error,
                "prior_seeds": list(self.seeds_from_prior_turn)}


# --------------------------------------------------------------------------- #
# answering and verification
# --------------------------------------------------------------------------- #

class CitationStatus(str, Enum):
    VERIFIED = "verified"              # exists AND was retrieved
    OUT_OF_CONTEXT = "out_of_context"  # exists, was NOT retrieved — recalled
    UNRESOLVED = "unresolved"          # no such provision


_STATUS_ORDER = {CitationStatus.UNRESOLVED: 0,
                 CitationStatus.OUT_OF_CONTEXT: 1,
                 CitationStatus.VERIFIED: 2}


@dataclass
class Citation:
    id: str
    label: str
    status: CitationStatus
    text: str = ""
    headnote: str = ""
    note: str = ""

    @property
    def sort_key(self) -> tuple[int, str]:
        """Worst first — a reader should meet the problem before the
        reassurance, not scroll past a list of ticks to find it."""
        return (_STATUS_ORDER[self.status], self.id)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "status": self.status.value,
                "text": self.text, "headnote": self.headnote, "note": self.note}


class Verdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"        # RED: a defect in the answer
    NOT_YET_IN_FORCE = "not_yet_in_force"   # AMBER: answer correct, law pending
    CAVEAT = "caveat"                       # AMBER: scope note

    @property
    def is_defect(self) -> bool:
        """Only UNSUPPORTED means the answer is wrong.

        Red is reserved for that. Painting a not-yet-commenced provision red
        told a reader their correct answer was wrong, which is the more
        expensive of the two mistakes.
        """
        return self is Verdict.UNSUPPORTED


@dataclass
class Claim:
    """One assertion in an answer, with the evidence it rests on.

    Claim-level rather than answer-level is what makes verification meaningful:
    a four-sentence answer carrying one citation cannot say which sentence that
    citation supports.
    """
    text: str
    cites: tuple[str, ...] = ()
    quote: str = ""
    verdict: Verdict = Verdict.SUPPORTED
    note: str = ""
    entailment: float | None = None

    def to_dict(self) -> dict:
        return {"claim": self.text, "cites": list(self.cites),
                "quote": self.quote, "verdict": self.verdict.value,
                "note": self.note,
                "entailment": None if self.entailment is None else round(self.entailment, 3)}


class Path(str, Enum):
    TEMPLATE = "template"      # rendered from the graph, zero model calls
    MODEL = "llm"
    ABSTAIN = "abstain"


@dataclass
class Rendered:
    """A template answer plus the provisions it was built from.

    `cites` is not parsed back out of `text` — it is the set of nodes the
    renderer actually read. Verification for this path is therefore exact by
    construction rather than a regex approximation of itself.
    """
    text: str
    cites: tuple[str, ...] = ()
    intent: Intent = Intent.GENERAL
    highlight: str = ""          # the provision that answers the question

    def stream_chunks(self, size: int = 24):
        """Split on whitespace so a word never straddles two SSE frames."""
        buffer = ""
        for word in self.text.split(" "):
            candidate = f"{buffer} {word}" if buffer else word
            if len(candidate) >= size:
                yield candidate + " "
                buffer = ""
            else:
                buffer = candidate
        if buffer:
            yield buffer


@dataclass
class Outcome:
    """How one request ended. Every terminal path produces exactly one."""
    kind: str
    path: Path
    reason: str = ""
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    model: str | None = None
