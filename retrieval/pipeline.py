"""
The retrieval pipeline: rank, fuse, boost, rerank, expand.

Ordering is deliberate and each stage is separately observable, because the
question "why did this provision not reach the answer?" must be answerable
from the trace rather than by re-running with print statements.
"""
from __future__ import annotations

import difflib
import logging
import re

from core.corpus import Corpus
from core.models import Chunk, RetrievalTrace, Scored
from . import lexical
from .dense import DenseIndex
from .expand import Expander, ExpansionTrace
from .fusion import Ranking, fuse

log = logging.getLogger(__name__)

# The Gazette sets defined terms in curly quotes: Definition of “Data
# Fiduciary”. Straight quotes are accepted too, because the extraction is not
# guaranteed to preserve the typographic pair on every page.
QUOTED_TERM = re.compile(
    "[" + "“‘\"'" + "]"
    "([^" + "”’\"'" + "]{3,60})"
    "[" + "”’\"'" + "]")


class Retriever:
    """Holds the indexes. Constructed once; `retrieve` is pure after that."""

    def __init__(self, corpus: Corpus, *, edges: list[tuple[str, str, str]],
                 dense_index: DenseIndex | None, vocab: dict,
                 settings) -> None:
        self.corpus = corpus
        self.dense_index = dense_index
        self.vocab = vocab or {}
        self.s = settings

        self.bm25_raw = lexical.build(corpus.chunks, contextual=False)
        # Built only when at least one chunk actually carries a context: an
        # index identical to the raw one would contribute a duplicate ranking
        # and silently double raw BM25's weight in the fusion.
        self.has_context = any(c.context for c in corpus.chunks)
        self.bm25_ctx = (lexical.build(corpus.chunks, contextual=True)
                         if self.has_context and settings.CONTEXTUAL else None)
        if settings.CONTEXTUAL and not self.has_context:
            log.info("contextual retrieval is enabled but no chunk carries a "
                     "context field — run scripts/build_context.py. Falling "
                     "back to raw BM25 only.")

        self.expander = Expander(
            corpus, edges,
            max_hops=settings.MAX_HOPS, max_expanded=settings.MAX_EXPANDED,
            hop_decay=settings.HOP_DECAY)

        self._vocab_index = self._index_vocab()
        self._intent_index = self._index_intents()
        self._phrases = self._index_phrases()
        self._corpus_terms = self.bm25_raw.vocabulary

    # -- vocabulary --------------------------------------------------------- #

    def _index_vocab(self) -> dict[str, tuple[str, ...]]:
        """phrase -> statutory terms it should pull in.

        Reads `terms:` — the key the shipped vocab.yaml actually uses. An
        earlier draft read `synonyms:`, found nothing, and silently ran with
        NO vocabulary layer at all: every lookup returned {} and no boost was
        ever applied. A missing key is not an error in YAML, which is exactly
        how a whole retrieval stage can be absent without a single warning.

        A post-hoc BOOST, never a pre-BM25 expansion: expansion changes what
        BM25 scores, making the vocabulary's contribution unmeasurable.
        """
        out: dict[str, tuple[str, ...]] = {}
        for phrase, targets in (self.vocab.get("terms") or {}).items():
            out[str(phrase).lower()] = (tuple(targets) if isinstance(targets, list)
                                        else (str(targets),))
        if not out:
            log.warning("vocab.yaml has no `terms:` map — the vocabulary "
                        "boost is inactive")
        return out

    def _index_intents(self) -> list[tuple[tuple[str, ...], tuple[str, ...], str]]:
        """(triggers, boost_kinds, boost_chapter) from vocab.yaml's `intents:`.

        Kept as data rather than hard-coded so a new question shape is a YAML
        edit, not a code change.
        """
        out = []
        for name, cfg in (self.vocab.get("intents") or {}).items():
            cfg = cfg or {}
            out.append((
                tuple(str(t).lower() for t in (cfg.get("triggers") or ())),
                tuple(cfg.get("boost_kinds") or ()),
                str(cfg.get("boost_chapter") or ""),
            ))
        return out

    def _index_phrases(self) -> list[tuple[str, str]]:
        """(exact phrase, node_id) for every chunk that names itself.

        BM25 is bag-of-words, so "Data Fiduciary" scores as two independent
        common terms and `def-significant-data-fiduciary` outranked
        `def-data-fiduciary` for the query "what is a Data Fiduciary?" — it
        contains both tokens too. An exact phrase is a far stronger signal
        than its words, and it is the one signal that survives ANY phrasing:
        however the user writes the sentence around it, naming the term is
        naming the term.

        Longest phrase first, so "Significant Data Fiduciary" is tested before
        "Data Fiduciary" and the more specific one wins.
        """
        out: list[tuple[str, str]] = []
        for chunk in self.corpus.chunks:
            quoted = QUOTED_TERM.findall(chunk.label)
            for term in quoted:
                out.append((term.lower().strip(), chunk.node_id))
            if chunk.headnote and len(chunk.headnote) > 8:
                out.append((chunk.headnote.lower().strip(" ."), chunk.node_id))
        out.sort(key=lambda kv: -len(kv[0]))
        return out

    def _repair(self, text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
        """Map query terms the index cannot score onto the nearest term it can.

        `understanding.normalize` carries a hand-written misspelling table, and
        a hand-written table can only ever cover misspellings someone thought
        of. "Daata Fuduciary" is not in it, scores 0.00 against every one of
        237 chunks, and is refused as out of scope — the user is told the DPDP
        Act does not cover Data Fiduciaries.

        A term absent from the index contributes NOTHING, so replacing it can
        only add signal; there is no ranking to damage. The guards keep the
        replacement honest rather than creative:

          - only terms of 5+ characters. "gdpr" is 4 and would fuzz to "gdp"-
            adjacent statutory noise, and short tokens have too few characters
            for a ratio to mean anything.
          - a 0.82 similarity cutoff, so "sourdough" finds no home and stays
            absent — an out-of-scope question must still score zero.
          - the ORIGINAL token is kept alongside the repair, never replaced.
            If the user really did mean a word this corpus lacks, nothing has
            been taken away from them.

        Returns the augmented text and the repairs, which the trace records so
        an answer built on a guessed spelling is auditable as such.
        """
        repairs: list[tuple[str, str]] = []
        for term in dict.fromkeys(lexical.tokenize(text)):
            if len(term) < 5 or term in self._corpus_terms:
                continue
            near = difflib.get_close_matches(term, self._corpus_terms, n=1,
                                             cutoff=self.s.FUZZY_CUTOFF)
            if near:
                repairs.append((term, near[0]))
        if not repairs:
            return text, ()
        # Appended, not substituted — the same rule the Hinglish layer follows.
        return f"{text} " + " ".join(good for _, good in repairs), tuple(repairs)

    def _phrase_boost(self, query: str) -> tuple[dict[str, float], tuple[str, ...]]:
        low = f" {query.lower()} "
        boost: dict[str, float] = {}
        matched: list[str] = []
        for phrase, node_id in self._phrases:
            if f" {phrase} " in low or f" {phrase}?" in low or f" {phrase}," in low:
                boost[node_id] = self.s.PHRASE_BOOST
                matched.append(phrase)
                # One winner per query: the longest phrase already sorted
                # first is the most specific reading.
                break
        return boost, tuple(matched)

    def _vocab_hits(self, query: str) -> tuple[tuple[str, ...], dict[str, float], set[str]]:
        """Phrases that matched, the multipliers they imply, and ids to INJECT.

        Two mechanisms, because a multiplier alone cannot do the job:

        BOOST re-ranks what BM25 already found. It is capped by SELECTIVITY —
        a target term appearing in more than a fifth of the corpus carries no
        discriminative signal, and boosting on it promotes whatever chunk is
        shortest. "customer" maps to "personal data", which appears nearly
        everywhere; boosting on it put two Definition chunks above every
        Penalty row for "what is the fine if customer data leaks?".

        INJECT adds chunks BM25 never ranked at all. A multiplicative boost
        cannot lift a zero, so an intent that names a whole class of provision
        ("penalty" -> the Schedule) has to place them in the pool directly.
        Ranking seven near-identical Schedule rows against each other was
        measured unreliable; completeness beats a cleverer score at seven rows.
        """
        low = f" {query.lower()} "
        hits: list[str] = []
        boost: dict[str, float] = {}
        inject: set[str] = set()

        for phrase, targets in self._vocab_index.items():
            if f" {phrase} " not in low:
                continue
            hits.append(phrase)
            for target in targets:
                matched = [c.node_id for c in self.corpus.chunks
                           if target.lower() in c.verbatim.lower()]
                # Too common to discriminate — skip rather than flatten the
                # ranking with a boost that applies to almost everything.
                if len(matched) > len(self.corpus.chunks) * self.s.VOCAB_MAX_SHARE:
                    continue
                for node_id in matched:
                    boost[node_id] = max(boost.get(node_id, 1.0), self.s.VOCAB_BOOST)

        # Chapter-level boosts from vocab.yaml's `intents:` still apply —
        # they only re-rank what BM25 found. KIND-level INJECTION does not:
        # that is driven by the ROUTER's resolved intent, passed in as
        # `inject_kinds`, not re-derived here from keyword triggers.
        #
        # Deriving it here fired every rule whose trigger happened to appear:
        # "what is the fine if customer data leaks?" contains "what is", so
        # `definition_lookup` injected all 32 Definition chunks alongside the
        # 7 Penalty rows and retrieval returned 52 provisions for a question
        # with one answer. The router already resolves intent through three
        # tiers; retrieval must not guess at it a second time.
        for triggers, kinds, chapter in self._intent_index:
            if chapter and any(t in low for t in triggers):
                for chunk in self.corpus.chunks:
                    if chunk.chapter == chapter:
                        boost[chunk.node_id] = (boost.get(chunk.node_id, 1.0)
                                                * self.s.INTENT_BOOST)

        return tuple(sorted(hits)), boost, inject

    # -- the pipeline ------------------------------------------------------- #

    def retrieve(self, query: str, *, query_vector: list[float] | None = None,
                 retrieval_text: str = "", prior_ids: tuple[str, ...] = (),
                 intent_kinds: tuple[str, ...] = (),
                 reranker=None) -> tuple[list[Scored], RetrievalTrace, ExpansionTrace]:
        text = retrieval_text or query
        trace = RetrievalTrace(query=query, seeds_from_prior_turn=prior_ids)

        # Before anything is ranked: a term the index cannot score is a term
        # that is not in the question at all, as far as retrieval is concerned.
        as_asked = text
        text, repairs = self._repair(text)
        trace.repairs = repairs
        asked_terms = lexical.tokenize(as_asked)
        repaired = {bad for bad, _ in repairs}
        trace.query_terms = len(asked_terms)
        # A repaired term COUNTS here, and this is not the inflation the
        # `top_bm25` comment above refuses. These are two different questions.
        # There, a guess would have added SCORE, inventing evidence of how
        # well the corpus matched. Here it answers only "is this word a word
        # of this instrument?" — and "parental" being one character from
        # "parent" is real evidence about the word, whatever the score.
        # Measured both ways: counting repairs stopped "parental consent" and
        # "erassure of data" being refused, and additionally refused "python
        # list", because a repair proves nothing about the terms AROUND it.
        trace.scorable_terms = sum(1 for t in asked_terms
                                   if t in self._corpus_terms or t in repaired)

        rankings: list[Ranking] = []
        ctx: Ranking | None = None

        raw = self.bm25_raw.rank(text, "bm25_raw", self.s.CANDIDATES)
        rankings.append(raw)
        # The abstention signal, named with its scale so no consumer has to
        # guess. A fused score is NOT usable here — see fusion.rrf_ceiling.
        #
        # Scored on the question AS ASKED, never on the repaired text. Every
        # repair is a guess, and a guess must be allowed to improve RANKING
        # without manufacturing the evidence that the question was in scope:
        # measured, feeding repairs into this number pushed nine of fifteen
        # out-of-scope questions past the floor ("mountain" -> "contain",
        # "weather" -> "whether"), because near-stopwords that add no meaning
        # still add score. A genuine misspelling is carried into an answer by
        # the PHRASE signal instead, which cannot be produced by accident.
        trace.top_bm25 = (max(raw.scores.values(), default=0.0) if not repairs
                          else max(self.bm25_raw.scores(as_asked), default=0.0))

        if self.bm25_ctx is not None:
            ctx = self.bm25_ctx.rank(text, "bm25_ctx", self.s.CANDIDATES)
            rankings.append(ctx)

        if self.dense_index is not None and query_vector:
            try:
                rankings.append(self.dense_index.rank(query_vector, self.s.CANDIDATES))
                trace.fused = True
            except Exception as exc:                        # noqa: BLE001
                # Dense is additive. A provider outage or a quota exhaustion
                # degrades quality; it must never fail a question.
                log.warning("dense retrieval unavailable: %s", exc)
                trace.dense_error = f"{type(exc).__name__}: {exc}"

        vocab_hits, boost, inject = self._vocab_hits(text)
        phrase_boost, phrases = self._phrase_boost(text)
        for node_id, factor in phrase_boost.items():
            boost[node_id] = boost.get(node_id, 1.0) * factor
        trace.vocab_hits = tuple(sorted(set(vocab_hits) | set(phrases)))
        # Kept SEPARATE from vocab_hits, which merges the two for display. An
        # exact statutory phrase and a layperson synonym are different claims:
        # the first says the question named a provision of this instrument, and
        # the sufficiency gate must be able to tell them apart.
        trace.phrase_hits = phrases

        # A whole CLASS of provision, injected because completeness beats
        # ranking for it. Bounded: injecting a large class buries the answer,
        # so only a genuinely small, enumerable set qualifies (the Schedule is
        # seven rows; the definitions are thirty-two and do not).
        for kind in intent_kinds:
            members = [c.node_id for c in self.corpus.chunks if c.kind == kind]
            if len(members) <= self.s.MAX_INJECT:
                inject.update(members)
            else:
                log.debug("not injecting %d %r chunks — above MAX_INJECT",
                          len(members), kind)

        fused = fuse(rankings, k=self.s.RRF_K, boost=boost)
        trace.candidates = len(fused)

        # Injected ids that fusion never saw are appended below the ranked
        # ones: present for completeness, explicitly NOT claiming to have
        # ranked. `fused` order is preserved for everything BM25 did find.
        ranked_ids = {nid for nid, _ in fused[:self.s.CANDIDATES]}
        for node_id in sorted(inject - ranked_ids):
            fused.append((node_id, 0.0))

        candidates: list[Scored] = []
        for node_id, score in fused[:self.s.CANDIDATES + len(inject)]:
            chunk = self.corpus.get(node_id)
            if chunk is None:
                continue
            candidates.append(Scored(
                chunk=chunk, fused=score,
                bm25=raw.scores.get(node_id, 0.0),
                bm25_contextual=ctx.scores.get(node_id, 0.0) if ctx else 0.0,
                dense=next((r.scores.get(node_id, 0.0) for r in rankings
                            if r.name == "dense"), 0.0),
            ))

        # Prior-turn provisions join the candidate set rather than replacing
        # it, so a follow-up can still reach something the earlier turn never
        # mentioned. Never inherited answers — only provisions.
        for node_id in prior_ids:
            chunk = self.corpus.chunk_for(node_id)
            if chunk and not any(c.node_id == chunk.node_id for c in candidates):
                candidates.append(Scored(chunk=chunk, fused=0.0, via="prior turn"))

        if reranker is not None and candidates:
            candidates = reranker.rerank(query, candidates)
            trace.reranked = True
            scores = [c.rerank for c in candidates if c.rerank is not None]
            if scores:
                trace.top_rerank = scores[0]
                trace.rerank_margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]

        # Injected class members are added ALONGSIDE the ranked top-K, not in
        # competition with it. Appending them to the candidate list and then
        # slicing to TOP_K discarded every one of them — an injection that
        # cannot reach the seed set does nothing at all. Completeness beats a
        # cleverer score at seven Schedule rows.
        seeds = candidates[:self.s.TOP_K]
        chosen = {c.node_id for c in seeds}
        for cand in candidates[self.s.TOP_K:]:
            if cand.node_id in inject and cand.node_id not in chosen:
                seeds.append(cand)
                chosen.add(cand.node_id)
        expanded, expansion = self.expander.expand(seeds)
        return expanded, trace, expansion


def build_context(results: list[Scored], max_chars: int) -> str:
    """The provisions, verbatim, as the model will see them.

    Budgeted by RELEVANCE rather than by document order: truncating the tail
    of an ordered list silently drops the graph-expanded provisions, which are
    exactly the ones the model could not have found on its own.
    """
    parts: list[str] = []
    used = 0
    for r in sorted(results, key=lambda x: (x.hop, -x.fused)):
        block = (f"[{r.chunk.label}]  id={r.node_id}\n"
                 f"{r.chunk.header}\n{r.chunk.verbatim}".strip())
        if used + len(block) > max_chars:
            continue          # skip, do not stop — a later item may still fit
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)
