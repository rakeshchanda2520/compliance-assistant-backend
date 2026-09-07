"""
The graph: provisions and the edges between them.

Neo4j is the runtime source of truth. When it is unreachable the graph is
reconstructed from `chunks.json`, which is enough to keep retrieval, citation
resolution and direct lookups working — but NOT the structural edges, because
those exist only in the graph.

That distinction is stated loudly rather than papered over. A degraded graph
silently loses `PENALISED_BY`, which is the single join this whole design
exists for: section 8(5) states a duty and never says what breaching it costs;
Schedule entry 1 states an amount and never says which duty it penalises. A
penalty answer built without that edge is not slightly worse, it is missing
the connection that made the answer worth trusting.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.corpus import Corpus
from core.models import Provision

log = logging.getLogger(__name__)

# Cypher cannot parameterise labels or relationship types, so anything
# interpolated into a query is validated against this first.
SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Graph:
    provisions: dict[str, Provision] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    degraded: bool = False
    detail: str = ""

    # -- derived views, computed once ---------------------------------------- #

    def __post_init__(self) -> None:
        self._children: dict[str, list[str]] = {}
        self._penalised_by: dict[str, list[str]] = {}
        self._penalty_for: dict[str, str] = {}
        self._mentions: dict[str, list[str]] = {}
        self._index()

    def _index(self) -> None:
        for src, dst, etype in self.edges:
            if etype in ("CONTAINS", "HAS_SUBSECTION", "HAS_ENTRY"):
                self._children.setdefault(src, []).append(dst)
            elif etype == "PENALISED_BY":
                self._penalised_by.setdefault(dst, []).append(src)
                self._penalty_for[src] = dst
            elif etype == "MENTIONS":
                self._mentions.setdefault(dst, []).append(src)

        # Containment is also derivable from the id shape (`s-8` contains
        # `s-8-5`), which is what keeps direct lookups working when the graph
        # is degraded and no CONTAINS edge was loaded.
        if not self._children:
            for node_id in self.provisions:
                parts = node_id.split("-")
                if len(parts) > 2:
                    parent = "-".join(parts[:-1])
                    if parent in self.provisions:
                        self._children.setdefault(parent, []).append(node_id)

    def children_of(self, node_id: str) -> list[str]:
        return sorted(self._children.get(node_id, []), key=_marker_key)

    def descendants_of(self, node_id: str) -> list[str]:
        out: list[str] = []
        for child in self.children_of(node_id):
            out.append(child)
            out.extend(self.descendants_of(child))
        return out

    def penalised_by(self) -> dict[str, list[str]]:
        """Schedule row -> the duties it penalises."""
        return self._penalised_by

    def penalty_for(self) -> dict[str, str]:
        """Duty -> the Schedule row that penalises it."""
        return self._penalty_for

    def mentions_of(self, node_id: str) -> list[str]:
        return self._mentions.get(node_id, [])


def _marker_key(node_id: str):
    """Document order, not string order.

    `s-8-10` must sort after `s-8-9`, not between `s-8-1` and `s-8-2` — which
    is what plain string comparison does, and section 8 runs to eleven
    sub-sections.
    """
    out = []
    for part in node_id.split("-"):
        out.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return out


# --------------------------------------------------------------------------- #

def load(settings, corpus: Corpus) -> Graph:
    """Neo4j if reachable, else a degraded graph rebuilt from the corpus."""
    if settings.NEO4J_URI and settings.NEO4J_PASSWORD:
        try:
            return _from_neo4j(settings)
        except Exception as exc:                          # noqa: BLE001
            if not settings.ALLOW_OFFLINE_GRAPH:
                raise
            log.warning("Neo4j unavailable (%s) — falling back to a graph "
                        "rebuilt from chunks.json. Structural edges are NOT "
                        "available; penalty and cross-reference answers will "
                        "be incomplete.", exc)
            return _from_corpus(corpus, detail=f"Neo4j unreachable: {exc}")

    log.warning("no Neo4j configured — running on a corpus-derived graph")
    return _from_corpus(corpus, detail="no Neo4j configured")


def _from_neo4j(settings) -> Graph:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            provisions: dict[str, Provision] = {}
            for rec in session.run(
                    "MATCH (n:Provision) RETURN n.id AS id, n.label AS label, "
                    "n.kind AS kind, n.text AS text, n.headnote AS headnote, "
                    "n.penalty AS penalty, n.chapter AS chapter, "
                    "n.page AS page, n.authority AS authority"):
                provisions[rec["id"]] = Provision(
                    id=rec["id"], label=rec["label"] or rec["id"],
                    kind=rec["kind"] or "", text=rec["text"] or "",
                    headnote=rec["headnote"] or "", penalty=rec["penalty"] or "",
                    chapter=rec["chapter"] or "", page=int(rec["page"] or 0),
                    authority=float(rec["authority"] or 0.0))

            edges = [(r["src"], r["dst"], r["type"]) for r in session.run(
                "MATCH (a:Provision)-[e]->(b:Provision) "
                "RETURN a.id AS src, b.id AS dst, type(e) AS type")]

        log.info("graph: %d provisions, %d edges", len(provisions), len(edges))
        return Graph(provisions, edges)
    finally:
        driver.close()


def _from_corpus(corpus: Corpus, *, detail: str) -> Graph:
    """A graph with no structural edges. Retrieval and citations still work.

    Explicitly `degraded=True` so `/api/health` can report it and the template
    paths can decline rather than render an answer missing its central join.
    """
    provisions = {
        c.node_id: Provision(
            id=c.node_id, label=c.label, kind=c.kind, text=c.verbatim,
            headnote=c.headnote, chapter=c.chapter, page=c.page)
        for c in corpus.chunks
    }
    return Graph(provisions, [], degraded=True, detail=detail)
