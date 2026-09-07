"""
Reciprocal Rank Fusion, over an arbitrary number of rankers.

RRF combines RANKS, never scores. That is the whole point: BM25 produces
unbounded numbers, cosine sits in [-1, 1], and any score-level combination
needs a normalisation that itself needs calibrating per corpus. Ranks need
none.

    score(d) = Σ over rankers  1 / (k + rank_r(d))

Three rankers here rather than two, and the third is the important one:

    raw BM25          the exact-token signal, on verbatim text ONLY
    contextual BM25   the same algorithm over context-prepended text
    dense             cosine over embeddings

Raw BM25 is kept as a SEPARATE ranker rather than being replaced by the
contextual one. Prepending 50-100 generated tokens to every chunk changes term
frequencies and dilutes exact-token matching — and exact tokens are precisely
what BM25 is here for, because "250 crore" and "200 crore" are semantically
near-identical and legally opposite. Fusing them as two rankers means a bad
generated context can only ADD a signal, never remove one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ranking:
    """One ranker's opinion: node ids best-first, plus the raw scores it used.

    The scores are carried for AUDIT and for downstream gates, never for
    fusion — fusion reads only the position in `order`.
    """
    name: str
    order: tuple[str, ...]
    scores: dict[str, float]

    def rank_of(self, node_id: str) -> int | None:
        try:
            return self.order.index(node_id)
        except ValueError:
            return None


def fuse(rankings: list[Ranking], *, k: int = 60,
         boost: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Fused ids, best first.

    `boost` multiplies a node's fused score AFTER fusion — used by the
    vocabulary layer. Applied post-hoc on purpose: a boost applied before
    ranking would change what BM25 scores, which confounds any measurement of
    what the vocabulary is contributing. That was tried once and reverted.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, node_id in enumerate(ranking.order):
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (k + position)

    if boost:
        for node_id, factor in boost.items():
            if node_id in fused:
                fused[node_id] *= factor

    return sorted(fused.items(), key=lambda kv: -kv[1])


def rrf_ceiling(n_rankers: int, k: int = 60) -> float:
    """The maximum a fused score can reach.

    Recorded because it is the reason a fused score can NEVER be an abstention
    signal: the top item scores about `n_rankers / k` whether or not anything
    relevant was found. Rank says which result is best; it says nothing about
    whether the best one is any good. A previous system compared this value
    against a threshold calibrated on BM25's unbounded scale and refused every
    question that reached the gate.
    """
    return n_rankers / k
