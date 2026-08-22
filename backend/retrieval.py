"""
Retrieval: vocabulary bridge -> BM25 -> graph expansion.

BM25 over 142 chunks, not a vector store. In law the exact token is the
answer: "§8(5)", "250 crore", "Significant Data Fiduciary". Embeddings blur
precisely the distinctions that decide a question, and at this corpus size
they buy little. The gap they would close — a layperson's words sharing no
root with the statute's — is closed instead by two cheaper, auditable
mechanisms: `vocab.yaml` (a reviewed synonym map) and a per-chunk
plain-language layer built at index time.

Graph expansion is the part a flat index cannot do. "What's the penalty for
failing to notify a breach?" needs §8(6), which contains no rupee figure,
joined to the Schedule row, which says nothing about notification. That join
is an edge, not a similarity.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from . import config, embeddings
from .graph_store import Graph
from .indexing import Chunk, tokenize

log = logging.getLogger(__name__)

# A cited provision is more load-bearing than a merely mentioned term, and
# MENTIONS is exhaustive by construction — unranked, it buries the cited
# sections under definitions. Lower number wins.
EXPAND_PRIORITY = {"PENALISED_BY": 0, "PENALISES": 0, "REFERENCES": 1,
                   "CITED_BY": 1, "DEFINES": 2, "HAS_ENTRY": 3, "MENTIONS": 4}

# Some edges must be walked backwards: a penalty question lands on the
# Schedule, and the useful next hop is *up* to the duty that carries it —
# the opposite direction from how the edge is stored.
REVERSIBLE = {"PENALISED_BY": "PENALISES", "REFERENCES": "CITED_BY"}

MAX_HOPS = 2
MAX_EXPANDED = 10
HOP_DECAY = 0.6      # hop-2 is reachable but ranked below hop-1
INTENT_BOOST = 1.6


@dataclass
class Result:
    chunk: Chunk
    score: float
    hop: int
    weight: float = 1.0
    via: str = ""

    @property
    def id(self) -> str:
        return self.chunk.id


@dataclass
class Trace:
    """What retrieval did, so an answer can be audited without re-running it.

    V2 adds the per-retriever ranks. That is not decoration: the V1 argument
    for vocab.yaml over embeddings was auditability, and the honest answer to
    "dense retrieval is a black box" is to surface its scores rather than to
    avoid using it. A wrong dense hit is now as inspectable as a wrong vocab
    expansion.
    """
    expanded_query: str
    vocab_hits: list[str]
    intents: list[str]
    bm25_ranks: dict[str, int] = field(default_factory=dict)
    dense_ranks: dict[str, int] = field(default_factory=dict)
    dense_scores: dict[str, float] = field(default_factory=dict)
    fused: bool = False
    dense_error: str = ""


class Retriever:
    """Built once at startup. BM25 is constructed here, not per request —
    the corpus does not change between builds, and rebuilding it on every
    question was measurable latency for no gain."""

    def __init__(self, chunks: list[Chunk], graph: Graph, vocab: dict,
                 dense_index=None) -> None:
        self.chunks = chunks
        self.graph = graph
        self.vocab = vocab
        self.dense_index = dense_index
        self._bm25 = BM25Okapi([tokenize(c.document) for c in chunks])
        self._by_node = {c.node_id: c for c in chunks}
        self._by_chunk_id = {c.id: c for c in chunks}
        self._adjacency = self._build_adjacency()

    # -- query rewriting --------------------------------------------------- #

    def expand_query(self, query: str) -> tuple[str, list[str], list[str]]:
        """Rewrite into statutory vocabulary. Every expansion is a line in
        vocab.yaml, so a wrong hit traces to a rule and is fixed without
        touching code."""
        low = query.lower()
        added: list[str] = []
        hits: list[str] = []
        terms = self.vocab.get("terms", {})
        for phrase, statutory in sorted(terms.items(), key=lambda kv: -len(kv[0])):
            # Tolerate regular inflections so vocab.yaml stays singular.
            # Irregular forms ("stole") still need their own entry.
            if re.search(rf"\b{re.escape(phrase)}(?:s|es|ed|ing)?\b", low):
                hits.append(phrase)
                added.extend(statutory)
        intents = [
            name for name, cfg in (self.vocab.get("intents") or {}).items()
            if any(trigger in low for trigger in cfg.get("triggers", []))
        ]
        return f"{query} {' '.join(added)}", sorted(set(hits)), intents

    # -- graph adjacency ---------------------------------------------------- #

    def _chunk_for(self, node_id: str) -> Chunk | None:
        """Walk up to the nearest node that is itself a chunk.

        Load-bearing: a short section is one whole-section chunk, but its
        REFERENCES/MENTIONS edges are recorded on child nodes that are never
        a chunk themselves. Without this rollup those sections could never
        expand at all — only long, sub-chunked ones would.
        """
        seen: set[str] = set()
        current = node_id
        while current and current not in seen:
            if current in self._by_node:
                return self._by_node[current]
            seen.add(current)
            current = self.graph.parent_of.get(current)
        return None

    def _build_adjacency(self) -> dict[str, list[tuple[str, str]]]:
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for src, dst, etype in self.graph.edges:
            if etype in EXPAND_PRIORITY:
                if chunk := self._chunk_for(src):
                    adjacency[chunk.node_id].append((dst, etype))
            if etype in REVERSIBLE:
                if chunk := self._chunk_for(dst):
                    adjacency[chunk.node_id].append((src, REVERSIBLE[etype]))
        return adjacency

    # -- retrieval ---------------------------------------------------------- #

    def retrieve(self, query: str, k: int = 6,
                 seed_ids: list[str] | None = None) -> tuple[list[Result], Trace]:
        """Seed with BM25 (fused with dense when enabled), then walk the graph.

        `seed_ids` injects provisions from a prior conversation turn as
        additional hop-0 seeds (phase 5). They join the seed set rather than
        replacing it, so a follow-up can still reach something the earlier
        turn never mentioned.
        """
        expanded, hits, intents = self.expand_query(query)
        # V2: vocab is a post-hoc BOOST, not a pre-BM25 expansion, so the
        # ON/OFF ablation is interpretable. Expansion changes what BM25
        # scores, which confounds any measurement of what vocab contributes.
        bm25_query = query if config.HYBRID else expanded
        scores = self._bm25.get_scores(tokenize(bm25_query))

        boost_kinds: set[str] = set()
        boost_chapters: set[str] = set()
        for name in intents:
            cfg = (self.vocab.get("intents") or {}).get(name) or {}
            boost_kinds.update(cfg.get("boost_kinds", []))
            if chapter := cfg.get("boost_chapter"):
                boost_chapters.add(chapter)
        if boost_kinds or boost_chapters:
            for i, chunk in enumerate(self.chunks):
                if chunk.kind in boost_kinds or chunk.chapter in boost_chapters:
                    scores[i] *= INTENT_BOOST

        trace = Trace(expanded, hits, intents)
        seeds = self._seed(query, expanded, scores, hits, k, trace)

        if seed_ids:
            chosen = {s.id for s in seeds}
            for node_id in seed_ids:
                chunk = self._chunk_for(node_id)
                if chunk is not None and chunk.id not in chosen:
                    seeds.append(Result(chunk, 0.0, hop=0, via="prior turn"))
                    chosen.add(chunk.id)

        # The Schedule is seven rows. For a penalty question, ranking them
        # against each other is the wrong problem — and it actively fails:
        # entry 5 (the Data Principal's own duties) is the only row that never
        # says "Data Principal" or "personal data", so expanding "customer"
        # into those terms pushes the one row about the customer to last.
        # Completeness beats a cleverer score at seven rows.
        if "penalty_lookup" in intents:
            chosen = {s.id for s in seeds}
            for i, chunk in enumerate(self.chunks):
                if chunk.kind == "Penalty" and chunk.id not in chosen:
                    seeds.append(Result(chunk, round(float(scores[i]), 4), hop=0))

        return self._expand(seeds), trace

    # -- seeding ------------------------------------------------------------ #

    def _seed(self, query: str, expanded: str, bm25_scores, vocab_hits: list[str],
              k: int, trace: Trace) -> list[Result]:
        """Top-k hop-0 chunks: BM25 alone, or BM25 fused with dense via RRF."""
        order = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        trace.bm25_ranks = {self.chunks[i].node_id: rank
                            for rank, i in enumerate(order[:20])}

        if not (config.HYBRID and self.dense_index):
            return [Result(self.chunks[i], round(float(bm25_scores[i]), 4), hop=0)
                    for i in order[:k] if bm25_scores[i] > 0]

        try:
            vector = embeddings.embed_one(query, is_query=True)
            dense_hits = self.dense_index.search(vector, 20)
        except Exception as exc:                       # noqa: BLE001
            # Dense is additive. If the provider is down or rate-limited, BM25
            # alone is a degraded but CORRECT answer — far better than failing
            # a question outright. The trace records that it happened so a
            # quality drop is explainable rather than mysterious.
            log.warning("dense retrieval unavailable, using BM25 only: %s", exc)
            trace.dense_error = f"{type(exc).__name__}: {exc}"
            return [Result(self.chunks[i], round(float(bm25_scores[i]), 4), hop=0)
                    for i in order[:k] if bm25_scores[i] > 0]

        trace.fused = True
        trace.dense_ranks = {nid: rank for rank, (nid, _) in enumerate(dense_hits)}
        trace.dense_scores = {nid: round(score, 4) for nid, score in dense_hits}

        # Reciprocal Rank Fusion. Rank-based on purpose: BM25 scores are
        # unbounded and cosine sits in [-1,1], so any score-level combination
        # needs a normalisation that would itself need calibrating per corpus.
        # Ranks need none.
        fused: dict[str, float] = {}
        for rank, i in enumerate(order):
            if bm25_scores[i] > 0:
                fused[self.chunks[i].node_id] = 1.0 / (config.RRF_K + rank)
        for rank, (node_id, _) in enumerate(dense_hits):
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (config.RRF_K + rank)

        # vocab.yaml, demoted to a boost. It survives dense retrieval for the
        # cases dense genuinely cannot do: abbreviations ("SDF"), statute-coined
        # jargon that embeds near a generic near-synonym, and Rules vocabulary
        # where there is no plain-language layer to help.
        if vocab_hits:
            boosted = self._vocab_targets(vocab_hits)
            for node_id in boosted:
                if node_id in fused:
                    fused[node_id] *= config.VOCAB_BOOST

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Result(self._by_node[nid], round(score, 6), hop=0)
                for nid, score in ranked if nid in self._by_node]

    def _vocab_targets(self, phrases: list[str]) -> set[str]:
        """Chunks whose text carries a statutory term one of the matched
        vocab phrases maps to. Computed per query rather than cached: the
        matched-phrase set is small and this keeps vocab.yaml hot-editable."""
        terms = self.vocab.get("terms", {})
        wanted = {t.lower() for phrase in phrases for t in terms.get(phrase, [])}
        if not wanted:
            return set()
        return {c.node_id for c in self.chunks
                if any(term in c.document.lower() for term in wanted)}

    def _expand(self, seeds: list[Result]) -> list[Result]:
        picked: dict[str, Result] = {s.id: s for s in seeds}
        frontier = seeds

        for hop in range(1, MAX_HOPS + 1):
            candidates = []
            for rank, node in enumerate(frontier):
                for target, etype in self._adjacency.get(node.chunk.node_id, []):
                    # MENTIONS is exhaustive; at hop 2 it would drag in a
                    # large slice of the Act through terms mentioned two
                    # edges away. That is noise, not a compound question.
                    if hop == MAX_HOPS and etype == "MENTIONS":
                        continue
                    neighbour = self._chunk_for(target)
                    if neighbour is None or neighbour.id in picked:
                        continue
                    authority = self.graph.provisions[neighbour.node_id].authority \
                        if neighbour.node_id in self.graph.provisions else 0.0
                    candidates.append(
                        (EXPAND_PRIORITY[etype], rank, -authority, neighbour, node, etype))

            added: list[Result] = []
            # Sort key is (edge priority, seed rank, -authority): authority is
            # only ever a tie-break *within* a priority tier, so a cited
            # provision always beats a merely-mentioned one regardless of how
            # central the mentioned one is.
            for _prio, _rank, _auth, neighbour, node, etype in sorted(
                    candidates, key=lambda t: (t[0], t[1], t[2])):
                if len(picked) - len(seeds) >= MAX_EXPANDED:
                    break
                result = Result(neighbour, 0.0, hop=hop,
                                weight=node.weight * HOP_DECAY,
                                via=f"{node.chunk.label} —{etype}→")
                picked[neighbour.id] = result
                added.append(result)

            frontier = added
            if not frontier or len(picked) - len(seeds) >= MAX_EXPANDED:
                break

        return sorted(picked.values(), key=lambda r: (r.hop, -r.weight, -r.score))


def build_context(results: list[Result], max_chars: int) -> str:
    """Assemble the prompt context, stopping at the character budget rather
    than truncating mid-provision — half a section is worse than one fewer."""
    parts: list[str] = []
    total = 0
    for r in results:
        block = (f"=== {r.chunk.label} ===\n{r.chunk.header}\n---\n"
                 f"{r.chunk.verbatim}\n")
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def should_abstain(results: list[Result], threshold: float) -> str | None:
    """Refuse before spending a generation call on a question the corpus
    plainly does not cover.

    Deterministic and auditable, unlike asking the model to judge its own
    competence — which is exactly the judgement a small model is worst at.
    """
    top = max((r.score for r in results if r.hop == 0), default=0.0)
    if top < threshold:
        return (f"closest match scored {top:.1f}, below the {threshold:.0f} "
                f"threshold for an in-scope question")
    return None
