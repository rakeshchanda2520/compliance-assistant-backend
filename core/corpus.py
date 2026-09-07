"""
The corpus: chunks, provisions, and the invariants that bind them.

Loaded once at startup and treated as immutable. Two integrity checks run here
rather than at first request, because both failure modes are silent: a stale
`chunks.json` scores text that no longer matches the provisions being cited,
and a mismatched `embeddings.npz` returns confident nonsense. Neither raises on
its own.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .models import Chunk, Provision

log = logging.getLogger(__name__)


class CorpusError(RuntimeError):
    """The corpus cannot be trusted. Never degrade past this — refuse to start."""


@dataclass(frozen=True)
class Corpus:
    chunks: tuple[Chunk, ...]
    by_node: dict[str, Chunk]
    build_id: str

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, node_id: str) -> Chunk | None:
        return self.by_node.get(node_id)

    def chunk_for(self, node_id: str) -> Chunk | None:
        """The nearest CHUNKED ancestor of a provision id.

        Structural edges frequently live on un-chunked children: a short
        section's REFERENCES edge hangs off a sub-section that was never
        chunked on its own. Without this rollup, graph expansion could never
        leave such a section, and the cross-document join — the entire reason
        the graph exists — would silently do nothing for exactly the
        provisions that need it most.
        """
        if node_id in self.by_node:
            return self.by_node[node_id]
        parts = node_id.split("-")
        for i in range(len(parts) - 1, 1, -1):
            candidate = "-".join(parts[:i])
            if candidate in self.by_node:
                return self.by_node[candidate]
        return None


def load_chunks(path: Path) -> Corpus:
    if not path.is_file():
        raise CorpusError(
            f"search index missing at {path.name}. The corpus is built "
            f"offline and shipped with the service — it is not generated at "
            f"boot. See scripts/build_corpus.py.")

    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw["chunks"] if isinstance(raw, dict) else raw

    chunks: list[Chunk] = []
    for r in records:
        chunks.append(Chunk(
            id=r["id"], node_id=r["node_id"], kind=r["kind"], label=r["label"],
            verbatim=r.get("verbatim", ""), header=r.get("header", ""),
            headnote=r.get("headnote", ""), chapter=r.get("chapter", "") or "",
            page=int(r.get("page") or 0),
            plain_english=r.get("plain_english", "") or "",
            questions=tuple(r.get("questions") or ()),
            # Absent until the contextual build runs. Its absence is not an
            # error: the contextual rankers simply fall back to the raw text,
            # which is why raw BM25 is kept as a separate ranker.
            context=r.get("context", "") or "",
        ))

    by_node: dict[str, Chunk] = {}
    duplicates = []
    for c in chunks:
        if c.node_id in by_node:
            duplicates.append(c.node_id)
        by_node[c.node_id] = c
    if duplicates:
        raise CorpusError(
            f"duplicate node_ids in {path.name}: {duplicates[:5]}. Two chunks "
            f"claiming one provision means citations resolve to whichever "
            f"loaded last.")

    return Corpus(tuple(chunks), by_node, build_id=_build_id(chunks))


def _build_id(chunks: list[Chunk]) -> str:
    """A content hash, not a version number anyone must remember to bump.

    It changes automatically the moment the law is amended and rebuilt, so a
    logged or displayed answer names exactly which text produced it. Covers
    the generated context too — otherwise re-contextualising the corpus would
    silently change retrieval while claiming to be the same build.
    """
    import hashlib
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda x: x.node_id):
        h.update(c.node_id.encode())
        h.update(c.verbatim.encode())
        h.update(c.context.encode())
    return h.hexdigest()[:12]


def verify_against_graph(corpus: Corpus, provisions: dict[str, Provision]) -> None:
    """Every chunk must name a provision that exists.

    The one failure this deployment genuinely invites is a stale copy: the
    graph rebuilt while the shipped chunks still describe the previous corpus.
    Retrieval would then score text that no longer matches the provisions being
    cited, and nothing else in the system would notice.
    """
    orphans = sorted({c.node_id for c in corpus.chunks} - set(provisions))
    if orphans:
        raise CorpusError(
            f"chunks.json is out of step with the graph: {len(orphans)} "
            f"chunk(s) reference provisions that no longer exist "
            f"(e.g. {orphans[:5]}). Rebuild and copy chunks.json and "
            f"embeddings.npz together.")


def verify_against_index(corpus: Corpus, indexed_ids: list[str]) -> None:
    """The dense index and the chunks must describe the same corpus."""
    drift = sorted(set(indexed_ids) ^ {c.node_id for c in corpus.chunks})
    if drift:
        raise CorpusError(
            f"embeddings.npz does not match chunks.json: {len(drift)} node(s) "
            f"differ (e.g. {drift[:5]}). Both are written by the same build — "
            f"copy them together.")
