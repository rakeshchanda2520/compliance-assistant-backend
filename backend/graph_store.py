"""
Neo4j access. The graph is the source of truth at runtime — there is no
networkx here and no graph JSON in the serving path.

Loaded once into memory at startup rather than queried per request. That is a
deliberate trade, and it is the right one for this corpus: 404 nodes and 1088
edges is a few hundred kilobytes, retrieval touches nearly all of it on every
question (BM25 scores the whole corpus, expansion walks arbitrary edges), and
a per-request round trip to a hosted Aura instance would add latency to every
answer for no benefit. A corpus that outgrew memory would want the traversal
pushed into Cypher instead.

Every query is parameterised. Node ids arriving from a URL or from a model's
citation text are never interpolated into Cypher.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from neo4j import GraphDatabase

from . import config

log = logging.getLogger(__name__)

# Edge types the retriever may traverse. Anything not listed is structural or
# noise and is never followed at query time.
TRAVERSABLE = ("REFERENCES", "PENALISED_BY", "DEFINES", "MENTIONS", "HAS_ENTRY")


@dataclass(frozen=True)
class Provision:
    """One node of the corpus. `text` is the instrument's own words, verbatim.

    `doc` says which instrument it came from ("act" or "rules") so an answer
    can keep the duty and the detail apart instead of presenting a rule as if
    it were a section.
    """
    id: str
    kind: str
    label: str
    text: str = ""
    headnote: str = ""
    chapter: str = ""
    penalty: str = ""
    page: int = 0
    authority: float = 0.0
    doc: str = "act"


@dataclass
class Graph:
    provisions: dict[str, Provision]
    # (source, target, type) — direction as stored.
    edges: list[tuple[str, str, str]]
    build_id: str
    parent_of: dict[str, str] = field(default_factory=dict)

    def penalised_by(self) -> dict[str, list[str]]:
        """Schedule entry -> the duties it penalises. This join is the reason
        the graph exists: §8(5) carries no rupee figure and the Schedule row
        carries no description of the duty."""
        out: dict[str, list[str]] = {}
        for src, dst, etype in self.edges:
            if etype == "PENALISED_BY":
                out.setdefault(dst, []).append(src)
        return out

    def penalty_for(self) -> dict[str, str]:
        """Duty -> the Schedule entry that penalises it. The forward direction
        of `penalised_by`, which the penalty template needs: it starts from a
        retrieved duty and has to reach the amount."""
        return {src: dst for src, dst, etype in self.edges
                if etype == "PENALISED_BY"}

    def children_of(self, node_id: str) -> list[str]:
        """Direct children, in document order.

        Order matters and is not incidental: `_marker_key` sorts s-8-10 after
        s-8-9 rather than after s-8-1, which plain string sorting gets wrong
        the moment a section passes nine sub-sections — and §8 has eleven.
        """
        kids = [child for child, parent in self.parent_of.items()
                if parent == node_id]
        return sorted(kids, key=_marker_key)

    def descendants_of(self, node_id: str) -> list[str]:
        """Every node beneath this one, depth-first in document order."""
        out: list[str] = []
        for child in self.children_of(node_id):
            out.append(child)
            out.extend(self.descendants_of(child))
        return out


def _marker_key(node_id: str) -> tuple:
    """Sort key that orders document markers the way the document does.

    Numeric parts compare numerically, alphabetic parts alphabetically, so
    s-8-2 precedes s-8-10 and r-6-1-a precedes r-6-1-b.
    """
    key: list = []
    for part in node_id.split("-"):
        if part.isdigit():
            key.append((0, int(part), ""))
        else:
            key.append((1, 0, part))
    return tuple(key)


def _driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        # Fail fast on a dead instance instead of hanging a request thread.
        connection_acquisition_timeout=15,
    )


def load_graph() -> Graph:
    """Read every provision and traversable edge once, at startup."""
    with _driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=config.NEO4J_DATABASE) as session:
            rows = session.run(
                "MATCH (n:Provision) RETURN n.id AS id, n.kind AS kind, "
                "n.label AS label, n.text AS text, n.headnote AS headnote, "
                "n.chapter AS chapter, n.penalty AS penalty, n.page AS page, "
                "n.authority AS authority, n.doc AS doc"
            ).data()

            edge_rows = session.run(
                "MATCH (a:Provision)-[r]->(b:Provision) "
                "RETURN a.id AS src, b.id AS dst, type(r) AS type"
            ).data()

    provisions = {
        r["id"]: Provision(
            id=r["id"],
            kind=r["kind"] or "Node",
            label=r["label"] or r["id"],
            text=r["text"] or "",
            headnote=r["headnote"] or "",
            chapter=r["chapter"] or "",
            penalty=r["penalty"] or "",
            page=int(r["page"] or 0),
            # Absent on graphs pushed before authority was computed, and on
            # any push that dropped it — default rather than fail, since it
            # is only ever a tie-breaker.
            authority=float(r["authority"] or 0.0),
            doc=r.get("doc") or "act",
        )
        for r in rows if r.get("id")
    }

    edges, parent_of = [], {}
    for r in edge_rows:
        src, dst, etype = r["src"], r["dst"], r["type"]
        if etype.startswith("HAS_"):
            # Containment: §8 HAS_SUBSECTION §8(5). Kept separately because
            # it is how a clause resolves to the chunk that contains it, not
            # something retrieval should ever traverse as a citation.
            parent_of[dst] = src
        if etype in TRAVERSABLE:
            edges.append((src, dst, etype))

    build_id = _build_id(provisions, edges)
    log.info("graph loaded: %d provisions, %d edges, build %s",
             len(provisions), len(edges), build_id)
    return Graph(provisions=provisions, edges=edges, build_id=build_id,
                 parent_of=parent_of)


def _build_id(provisions: dict[str, Provision], edges: list) -> str:
    """Content hash of the graph, not a number someone must remember to bump.

    Stamped on every answer and audit record, so "which answers predate the
    amendment?" is answerable the moment the corpus is rebuilt.
    """
    payload = json.dumps(
        {"nodes": sorted((p.id, p.text) for p in provisions.values()),
         "edges": sorted(edges)},
        ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def check() -> str | None:
    """Return an error string if Neo4j is unreachable, else None.

    The message is for operators (it reaches logs and, in DEBUG only, the
    health endpoint), so it deliberately names no credentials.
    """
    try:
        with _driver() as driver:
            driver.verify_connectivity()
        return None
    except Exception as exc:               # noqa: BLE001 — surfaced as text
        return f"Neo4j unreachable: {type(exc).__name__}"
