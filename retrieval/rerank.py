"""
Cross-encoder reranking — off by default until measured on the deploy host.

A cross-encoder reads the query and the document TOGETHER, so it can tell that
"encryption, obfuscation or masking" answers "what safeguards must we take"
even though the two share almost no vocabulary. That is what a bi-encoder and
BM25 both miss.

**The default is off, and the model is not chosen by parameter count.**
`bge-reranker-v2-m3` is ~568M parameters; scoring 30 pairs on a shared CPU is
seconds, not the 200ms a plan might assume. The budget is 500ms, quantised, and
the only acceptable way to pick a model is to measure it on the machine that
will run it — `scripts/measure_rerank.py` does exactly that.

A reranker that misses its budget is worse than no reranker: it converts a
quality improvement into a latency regression on every single question.
"""
from __future__ import annotations

import logging
import time

from core.models import Scored

log = logging.getLogger(__name__)


class Reranker:
    """Wraps an ONNX cross-encoder. Fails SOFT, always.

    Any failure — model missing, timeout, bad output — returns the input order
    untouched. Reranking is a refinement of a list that is already usable, so
    it must never be able to fail a question.
    """

    def __init__(self, model_name: str, *, timeout_ms: int) -> None:
        self.model_name = model_name
        self.timeout_ms = timeout_ms
        self._model = None
        self._broken = False

    def _load(self):
        if self._model is None and not self._broken:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._model = TextCrossEncoder(model_name=self.model_name)
                log.info("reranker loaded: %s", self.model_name)
            except Exception as exc:                       # noqa: BLE001
                self._broken = True
                log.warning("reranker unavailable (%s) — retrieval will use "
                            "fusion order. Install fastembed, or set "
                            "DPDP_RERANK=0 to stop trying.", exc)
        return self._model

    def rerank(self, query: str, candidates: list[Scored]) -> list[Scored]:
        model = self._load()
        if model is None or len(candidates) < 2:
            return candidates

        started = time.perf_counter()
        try:
            # The chunk's own label and headnote are included: they name what
            # the provision IS, which is often the deciding signal for a
            # question phrased around a defined term.
            documents = [f"{c.chunk.label}. {c.chunk.headnote}\n{c.chunk.verbatim}"[:2000]
                         for c in candidates]
            scores = list(model.rerank(query, documents))
        except Exception as exc:                           # noqa: BLE001
            log.warning("reranking failed (%s) — keeping fusion order", exc)
            return candidates

        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms > self.timeout_ms:
            # Logged loudly rather than silently absorbed: a reranker over
            # budget is a latency regression on EVERY question, and the fix is
            # a smaller or quantised model, not a bigger timeout.
            log.warning("reranking took %.0fms, over the %dms budget — "
                        "quantise the model or choose a smaller one",
                        elapsed_ms, self.timeout_ms)

        for candidate, score in zip(candidates, scores):
            candidate.rerank = float(score)
        return sorted(candidates, key=lambda c: -(c.rerank or 0.0))


def build(settings) -> Reranker | None:
    if not settings.RERANK:
        return None
    return Reranker(settings.RERANK_MODEL,
                    timeout_ms=settings.RERANK_TIMEOUT_MS)
