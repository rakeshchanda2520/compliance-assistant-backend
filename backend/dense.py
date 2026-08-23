"""
The dense half of hybrid retrieval. A numpy array, not a vector database.

237 chunks x 1024 dims x 4 bytes is under a megabyte. It loads at startup and
a query is one matrix-vector product over 237 rows — microseconds. Qdrant,
LanceDB and Pinecone all solve a problem this corpus does not have, and each
would add an service to operate, a client to configure, and a second place
for the index to drift out of sync with chunks.json.

Passage vectors are built once by `kg_build/embed.py` and shipped in
`embeddings.npz` beside `chunks.json`. Only the QUERY is embedded at request
time.

This module never decides ranking on its own — `retrieval.py` fuses these
ranks with BM25's via RRF. Dense retrieval alone regresses badly on exactly
the tokens this domain turns on ("250 crore" vs "200 crore", "s.8(5)" vs
"s.8(6)"), which is why V1 measured its way to BM25 and why V2 adds dense
rather than substituting it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config, embeddings

log = logging.getLogger(__name__)


class DenseIndex:
    """Passage vectors plus the node ids they belong to, in matching order."""

    def __init__(self, node_ids: list[str], matrix: np.ndarray,
                 fingerprint: dict, exemplars: list | None = None) -> None:
        self.node_ids = node_ids
        # L2-normalised once at load, so a query is a dot product rather than
        # a cosine with per-row norms recomputed on every request.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (matrix / norms).astype(np.float32)
        self.fingerprint = fingerprint
        # (group, label, text, vector) for the router. Built once at build
        # time and carried here so startup costs no embedding requests.
        self.exemplars: list[tuple[str, str, str, list[float]]] = exemplars or []

    def __len__(self) -> int:
        return len(self.node_ids)

    @property
    def dims(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        """Top-k (node_id, cosine), best first."""
        if not len(self):
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != self.matrix.shape[1]:
            # Belt and braces: verify_fingerprint should have caught this at
            # startup, but a dimension mismatch here would otherwise raise a
            # numpy error mid-request rather than say what is actually wrong.
            log.error("query vector has %d dims, index has %d — embedding "
                      "model mismatch", q.shape[0], self.matrix.shape[1])
            return []
        norm = float(np.linalg.norm(q)) or 1.0
        scores = self.matrix @ (q / norm)
        top = np.argsort(-scores)[:k]
        return [(self.node_ids[i], float(scores[i])) for i in top]


def load(path: Path) -> DenseIndex | None:
    """Load the passage index, or None when hybrid retrieval is off.

    Raises when hybrid is ON but the index is unusable — a backend that
    silently fell back to BM25-only would report itself healthy while serving
    measurably different results from the ones its configuration promises.
    """
    if not config.HYBRID:
        return None

    if not path.is_file():
        raise RuntimeError(
            f"dense index missing at {path.name} but DPDP_HYBRID=1. "
            f"Run: python -m kg_build   (or set DPDP_HYBRID=0)")

    with np.load(path, allow_pickle=True) as data:
        node_ids = [str(n) for n in data["node_ids"]]
        matrix = data["vectors"]
        fingerprint = dict(data["fingerprint"].item())

        exemplars: list[tuple[str, str, str, list[float]]] = []
        # Absent in an index built before exemplar caching existed. Not an
        # error: the router falls back to embedding them at startup, which is
        # what it used to do anyway.
        if "exemplar_vectors" in data.files and len(data["exemplar_vectors"]):
            exemplars = [
                (str(g), str(lbl), str(txt), [float(x) for x in vec])
                for g, lbl, txt, vec in zip(
                    data["exemplar_groups"], data["exemplar_labels"],
                    data["exemplar_texts"], data["exemplar_vectors"])]

    if len(node_ids) != matrix.shape[0]:
        raise RuntimeError(
            f"dense index is corrupt: {len(node_ids)} ids but "
            f"{matrix.shape[0]} vectors")

    # The symmetry guard. Abort, never warn — see embeddings.verify_fingerprint.
    if problem := embeddings.verify_fingerprint(fingerprint):
        raise RuntimeError(f"embedding index mismatch: {problem}")

    index = DenseIndex(node_ids, matrix, fingerprint, exemplars)
    index.chunk_ids = set(node_ids)
    log.info("dense index loaded: %d vectors, %d dims, %d cached exemplars, "
             "model %s/%s", len(index), index.dims, len(exemplars),
             fingerprint.get("provider"), fingerprint.get("model"))
    return index
