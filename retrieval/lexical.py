"""
BM25, twice: once over verbatim text, once over context-prepended text.

Two indexes over the same chunks, fused downstream as separate rankers. See
`retrieval.fusion` for why they are not merged into one.

The tokenizer is shared with build time. That is not a convenience — an index
built with one tokenizer and queried with another silently retrieves nothing
useful, and nothing raises.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from core.models import Chunk
from .fusion import Ranking

log = logging.getLogger(__name__)

# Statutory text is full of "8(5)", "₹250", "r-6-1". A tokenizer that splits on
# every non-letter destroys exactly the tokens that distinguish one provision
# from another, so digits and the section sign survive.
_TOKEN = re.compile(r"[a-z0-9§]+")

# Words carrying no discriminative signal in a corpus where EVERY document is
# about data protection. Deliberately short: an aggressive stop list removes
# "data", which is half the vocabulary of the statute but also the word that
# distinguishes "personal data breach" from "breach of contract".
_STOP = frozenset({
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is", "are",
    "be", "by", "as", "at", "it", "its", "this", "that", "which", "shall",
    "any", "such", "with", "from", "under", "may", "we", "our", "i", "my",
    "do", "does", "what", "how", "can", "if", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Shared by build and query. Changing this invalidates every index."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 with no external dependency.

    Written out rather than pulled from `rank_bm25` for two reasons: the
    corpus is 237 documents so performance is irrelevant, and having the
    scoring visible is what makes the abstention discussion possible at all —
    a black-box score cannot be reasoned about or calibrated.
    """

    __slots__ = ("k1", "b", "node_ids", "_len", "_avg", "_tf", "_idf", "_n")

    def __init__(self, documents: list[tuple[str, str]],
                 *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.node_ids = [node_id for node_id, _ in documents]
        tokenized = [tokenize(text) for _, text in documents]

        self._n = len(tokenized)
        self._len = [len(t) for t in tokenized]
        self._avg = (sum(self._len) / self._n) if self._n else 0.0
        self._tf = [Counter(t) for t in tokenized]

        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        # The +0.5/+0.5 smoothing keeps IDF positive for a term appearing in
        # every document, which in a single-subject corpus is common.
        self._idf = {term: math.log(1 + (self._n - n + 0.5) / (n + 0.5))
                     for term, n in df.items()}

    @property
    def vocabulary(self) -> frozenset[str]:
        """Every term the index can score. A query term absent from this set
        contributes exactly nothing — which is what makes a misspelling
        invisible rather than merely weak."""
        return frozenset(self._idf)

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * self._n
        if not terms or not self._avg:
            return out
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self._tf):
                freq = tf.get(term)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self._len[i] / self._avg)
                out[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return out

    def rank(self, query: str, name: str, limit: int) -> Ranking:
        scores = self.scores(query)
        order = sorted(range(self._n), key=lambda i: -scores[i])
        # Zero-scoring documents are EXCLUDED, not ranked last: including them
        # would let a document with no query term in it occupy a fusion slot
        # purely by existing.
        kept = [i for i in order[:limit] if scores[i] > 0]
        return Ranking(
            name=name,
            order=tuple(self.node_ids[i] for i in kept),
            scores={self.node_ids[i]: scores[i] for i in kept},
        )


def build(chunks: tuple[Chunk, ...], *, contextual: bool) -> BM25:
    return BM25([(c.node_id, c.lexical_text(contextual=contextual)) for c in chunks])
