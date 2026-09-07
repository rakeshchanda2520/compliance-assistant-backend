"""
Two-hop graph expansion — the "Graph" in GraphRAG.

Seeds are what MATCHED THE WORDS. Expansion is what those seeds DEPEND ON: the
definition of a term they use, the penalty they attract, the rule that makes
them concrete. No amount of text similarity finds those, because the texts
share almost no vocabulary — which is the entire reason a graph exists here.

Every expansion is instrumented. When a provision that should have been
reachable is missing from an answer, `ExpansionTrace` says which stage lost it:
no edge in the graph, an edge that would not roll up to a chunk, a candidate
cut by the fan-out cap, or a survivor cut by the context budget. Diagnosing
that from logs alone previously meant guessing between six possibilities.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from core.corpus import Corpus
from core.models import Scored

log = logging.getLogger(__name__)

# Lower number wins. A cited provision is more load-bearing than a merely
# mentioned term, and MENTIONS is exhaustive by construction — unranked, it
# buries the cited sections under definitions.
PRIORITY = {
    "PENALISED_BY": 0, "PENALISES": 0,
    "REFERENCES": 1, "CITED_BY": 1,
    "DEFINES": 2, "HAS_ENTRY": 3,
    "MENTIONS": 4,
}

# Edges that must be walkable BACKWARDS, and why each one matters:
#
#   PENALISED_BY  a penalty question lands on the Schedule row; the useful
#                 next hop is UP to the duty that carries it.
#   REFERENCES    stored as `r-6 --REFERENCES--> s-8-5`. A walk seeded on
#                 section 8(5) could never reach rule 6 going forwards, and
#                 "the Act states the duty, the Rules state what discharges
#                 it" is the single most valuable join in this corpus. Without
#                 this reversal the cross-document case silently does nothing.
REVERSIBLE = {"PENALISED_BY": "PENALISES", "REFERENCES": "CITED_BY"}


@dataclass
class ExpansionTrace:
    """Where each candidate provision ended up. Read this before adding a model."""
    edges_considered: int = 0
    # Edge existed but neither endpoint rolled up to a chunk — a BUILD issue.
    unrollable: list[str] = field(default_factory=list)
    # Reached, then cut by the fan-out cap — a RANKING issue.
    dropped_by_cap: list[str] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"edges": self.edges_considered,
                "unrollable": self.unrollable[:8],
                "dropped_by_cap": self.dropped_by_cap[:8],
                "expanded": self.expanded}

    def why_missing(self, node_id: str) -> str:
        """A one-line answer to 'why is X not in the context?'"""
        if node_id in self.expanded:
            return "expanded — if absent from the answer it was cut by the context budget"
        if node_id in self.dropped_by_cap:
            return "reached, then cut by MAX_EXPANDED — a ranking problem, not retrieval"
        if node_id in self.unrollable:
            return "an edge exists but rolls up to no chunk — a build/chunking problem"
        return "never reached: no edge from any seed within MAX_HOPS"


class Expander:
    """Adjacency built once at startup, in CHUNK space."""

    def __init__(self, corpus: Corpus, edges: list[tuple[str, str, str]],
                 *, max_hops: int, max_expanded: int, hop_decay: float) -> None:
        self.corpus = corpus
        self.max_hops = max_hops
        self.max_expanded = max_expanded
        self.hop_decay = hop_decay
        self.adjacency, self.unrollable = self._build(edges)

    def _build(self, edges: list[tuple[str, str, str]]):
        """Roll every edge up to the nearest chunked ancestor on BOTH ends.

        This rollup is essential and easy to omit: a short section's
        REFERENCES edge frequently hangs off a sub-section that was never
        chunked on its own. Without rolling up, expansion could never leave
        such a section — and those are exactly the provisions the
        cross-document join depends on.
        """
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        unrollable: list[str] = []

        for src, dst, etype in edges:
            if etype not in PRIORITY and etype not in REVERSIBLE:
                continue
            src_chunk = self.corpus.chunk_for(src)
            dst_chunk = self.corpus.chunk_for(dst)

            if etype in PRIORITY:
                if src_chunk:
                    adjacency[src_chunk.node_id].append((dst, etype))
                elif dst_chunk:
                    unrollable.append(f"{src} -{etype}-> {dst}")

            if etype in REVERSIBLE:
                if dst_chunk:
                    adjacency[dst_chunk.node_id].append((src, REVERSIBLE[etype]))
                elif src_chunk:
                    unrollable.append(f"{dst} <-{etype}- {src}")

        if unrollable:
            log.info("%d edges roll up to no chunk (e.g. %s)",
                     len(unrollable), unrollable[:3])
        return adjacency, unrollable

    def expand(self, seeds: list[Scored]) -> tuple[list[Scored], ExpansionTrace]:
        trace = ExpansionTrace()
        picked: dict[str, Scored] = {s.node_id: s for s in seeds}
        frontier = list(seeds)

        for hop in range(1, self.max_hops + 1):
            candidates: list[tuple[tuple, str, str, str]] = []

            for result in frontier:
                for neighbour, etype in self.adjacency.get(result.node_id, ()):
                    # MENTIONS is barred from the SECOND hop. Those edges all
                    # point into definitions, so following them twice floods
                    # the context with vocabulary instead of law.
                    if hop == self.max_hops and etype == "MENTIONS":
                        continue
                    chunk = self.corpus.chunk_for(neighbour)
                    trace.edges_considered += 1
                    if chunk is None:
                        trace.unrollable.append(neighbour)
                        continue
                    if chunk.node_id in picked:
                        continue
                    # Sorted by (edge priority, seed rank) so the strongest
                    # relationship from the best seed wins a contested slot.
                    candidates.append(
                        ((PRIORITY.get(etype, 9), -result.fused),
                         chunk.node_id, result.node_id, etype))

            if not candidates:
                break

            candidates.sort(key=lambda c: c[0])
            room = self.max_expanded - (len(picked) - len(seeds))
            for rank, (_, node_id, via_node, etype) in enumerate(candidates):
                if node_id in picked:
                    continue
                if rank >= room:
                    trace.dropped_by_cap.append(node_id)
                    continue
                chunk = self.corpus.get(node_id)
                if chunk is None:
                    continue
                source = picked[via_node]
                picked[node_id] = Scored(
                    chunk=chunk,
                    fused=source.fused * (self.hop_decay ** hop),
                    hop=hop,
                    via=f"{etype} from {via_node}",
                )
                trace.expanded.append(node_id)

            frontier = [r for r in picked.values() if r.hop == hop]

        ordered = sorted(picked.values(), key=lambda r: (r.hop, -r.fused))
        return ordered, trace
