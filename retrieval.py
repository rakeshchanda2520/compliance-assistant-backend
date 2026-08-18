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

import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .graph_store import Graph
from .indexing import Chunk, tokenize

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
    """What retrieval did, so an answer can be audited without re-running it."""
    expanded_query: str
    vocab_hits: list[str]
    intents: list[str]


class Retriever:
    """Built once at startup. BM25 is constructed here, not per request —
    the corpus does not change between builds, and rebuilding it on every
    question was measurable latency for no gain."""

    def __init__(self, chunks: list[Chunk], graph: Graph, vocab: dict) -> None:
        self.chunks = chunks
        self.graph = graph
        self.vocab = vocab
        self._bm25 = BM25Okapi([tokenize(c.document) for c in chunks])
        self._by_node = {c.node_id: c for c in chunks}
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

    def retrieve(self, query: str, k: int = 6) -> tuple[list[Result], Trace]:
        expanded, hits, intents = self.expand_query(query)
        scores = self._bm25.get_scores(tokenize(expanded))

        boost_kinds: set[str] = set()
        boost_chapters: set[str] = set()
        for name in intents:
            cfg = self.vocab["intents"][name]
            boost_kinds.update(cfg.get("boost_kinds", []))
            if chapter := cfg.get("boost_chapter"):
                boost_chapters.add(chapter)
        if boost_kinds or boost_chapters:
            for i, chunk in enumerate(self.chunks):
                if chunk.kind in boost_kinds or chunk.chapter in boost_chapters:
                    scores[i] *= INTENT_BOOST

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        seeds = [
            Result(self.chunks[i], round(float(scores[i]), 4), hop=0)
            for i in ranked[:k] if scores[i] > 0
        ]

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

        return self._expand(seeds), Trace(expanded, hits, intents)

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
