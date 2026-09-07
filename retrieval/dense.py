"""
Dense retrieval over a numpy matrix. No vector database.

237 chunks x 1024 dimensions is a 970KB array. Pinecone or Weaviate would add
a vendor, a cost, a network hop and an entire class of sync bug for a
brute-force cosine that takes microseconds.

The index carries a FINGERPRINT of the embedding model that built it. Vectors
from two different models are not comparable, and nothing downstream detects
it — retrieval just quietly returns worse results forever. So a mismatch
refuses to start rather than degrading silently.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class DenseIndex:
    """Passage vectors plus the router's exemplars, in one file.

    The exemplars live here rather than being embedded at boot because a
    previous system re-embedded ~45 of them on every restart and exhausted a
    daily free-tier quota in one afternoon of deploys — after which the router
    silently ran regex-only with no error anywhere.
    """

    __slots__ = ("node_ids", "matrix", "model", "dims", "exemplars", "canary")

    def __init__(self, node_ids: list[str], matrix, model: str,
                 exemplars: dict[str, list[tuple[str, list[float]]]] | None = None,
                 canary: list[float] | None = None) -> None:
        self.node_ids = node_ids
        self.matrix = matrix
        self.model = model
        self.dims = int(matrix.shape[1]) if matrix is not None and len(matrix) else 0
        self.exemplars = exemplars or {}
        self.canary = canary or []

    def __len__(self) -> int:
        return len(self.node_ids)

    def search(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        """Cosine similarity, best first.

        Vectors are L2-normalised at build AND query time, so a dot product IS
        the cosine — no per-query normalisation, no division.
        """
        import numpy as np
        if not len(self.node_ids) or not vector:
            return []
        q = np.asarray(vector, dtype="float32")
        if q.shape[0] != self.dims:
            # Caught here rather than producing a meaningless number: a
            # dimension mismatch means the query model is not the index model.
            raise ValueError(
                f"query vector is {q.shape[0]}-dimensional but the index is "
                f"{self.dims}-dimensional — the embedding model changed. "
                f"Rebuild the index; do not serve on mismatched vectors.")
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm
        sims = self.matrix @ q
        top = sims.argsort()[::-1][:limit]
        return [(self.node_ids[i], float(sims[i])) for i in top]

    def rank(self, vector: list[float], limit: int):
        from .fusion import Ranking
        hits = self.search(vector, limit)
        return Ranking(name="dense",
                       order=tuple(nid for nid, _ in hits),
                       scores={nid: s for nid, s in hits})


def load(path: Path, *, expect_model: str) -> DenseIndex | None:
    """Load the index, or None when dense retrieval is simply not configured.

    Returns None for "no index present" — a legitimate configuration in which
    the system runs lexical-only. RAISES for "index present but wrong", which
    is never legitimate.
    """
    if not path.is_file():
        log.info("no dense index at %s — running lexical-only", path.name)
        return None

    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover
        log.warning("numpy not installed — running lexical-only")
        return None

    data = np.load(path, allow_pickle=True)
    node_ids = [str(x) for x in data["node_ids"]]
    matrix = data["vectors"].astype("float32")
    model = str(data["model"]) if "model" in data else ""

    if expect_model and model and model != expect_model:
        raise RuntimeError(
            f"embeddings.npz was built by {model!r} but this service is "
            f"configured for {expect_model!r}. Cosine between vectors from "
            f"two models is meaningless and nothing downstream would notice. "
            f"Rebuild the index or fix DPDP_EMBED_MODEL.")

    exemplars: dict[str, list[tuple[str, list[float]]]] = {}
    if "exemplar_groups" in data:
        groups = [str(g) for g in data["exemplar_groups"]]
        labels = [str(l) for l in data["exemplar_labels"]]
        vectors = data["exemplar_vectors"].astype("float32")
        for group, label, vec in zip(groups, labels, vectors):
            exemplars.setdefault(group, []).append((label, vec.tolist()))

    canary = data["canary"].astype("float32").tolist() if "canary" in data else []

    index = DenseIndex(node_ids, matrix, model, exemplars, canary)
    log.info("dense index: %d vectors, %d dims, %d exemplars, model %s",
             len(index), index.dims,
             sum(len(v) for v in exemplars.values()), model or "unknown")
    return index


def verify_canary(index: DenseIndex, embed_one, *, tolerance: float = 0.02) -> str:
    """Re-embed a known string and compare against the stored vector.

    Guards the case a model-name check cannot: the provider silently changed
    the model behind a stable name. Returns "" when fine, else a description.

    Deliberately returns rather than raises — the caller decides whether a
    drifting embedder is fatal or merely worth logging, and at startup it is
    fatal while mid-request it must not be.
    """
    if not index.canary:
        return ""
    try:
        fresh = embed_one(CANARY_TEXT)
    except Exception as exc:                              # noqa: BLE001
        return f"canary could not be re-embedded: {exc}"
    if len(fresh) != len(index.canary):
        return (f"canary dimension changed: index {len(index.canary)}, "
                f"provider {len(fresh)}")
    dot = sum(a * b for a, b in zip(fresh, index.canary))
    if dot < 1.0 - tolerance:
        return (f"embedding model drifted: canary similarity {dot:.4f}, "
                f"expected ~1.0. Vectors in the index are no longer "
                f"comparable to fresh ones.")
    return ""


# A sentence using this corpus's own vocabulary, so drift shows up on the kind
# of text that actually matters here rather than on generic English.
CANARY_TEXT = ("A Data Fiduciary shall protect personal data by taking "
               "reasonable security safeguards to prevent a personal data breach.")
