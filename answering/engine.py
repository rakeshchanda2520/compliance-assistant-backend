"""
The answer engine: one question in, a sequence of events out.

This is the whole pipeline in one readable function, deliberately. The
previous system spread it across a 1,100-line HTTP handler where the control
flow, the SSE framing, the audit writes and the tracing were interleaved, and
the order of operations could only be established by reading all of it.

Here the engine is transport-agnostic — it yields typed events and knows
nothing about SSE, FastAPI or a browser. That is what lets the eval harness
run the identical path offline with no server, which is the only way the
"measure before you change it" rule in NEW_FRAMEWORK §5 is affordable.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from answering import sufficiency, templates
from core.models import (Claim, Intent, Path, Scored, Understanding, Verdict)
from retrieval.pipeline import build_context
from verification import citations as cite
from verification import claims as claimcheck

log = logging.getLogger(__name__)

# Classes of provision worth injecting wholesale for a given intent. Only
# PENALTY qualifies: the Schedule is seven rows and ranking them against each
# other was measured unreliable, so completeness wins. Thirty-two definitions
# is not such a class — injecting them buries the one that was asked about.
INJECT_KINDS: dict[Intent, tuple[str, ...]] = {Intent.PENALTY: ("Penalty",)}


@dataclass
class Event:
    """One thing the caller should know about, in order."""
    name: str
    data: dict = field(default_factory=dict)


@dataclass
class Deps:
    """Everything the engine needs, injected so it can be faked in a test."""
    corpus: object
    graph: object
    retriever: object
    router: object
    settings: object
    embed_query = None          # callable(str) -> list[float] | None
    synthesize = None           # callable(...) -> Iterator[str]
    reranker = None
    prior_provisions = None     # callable(conversation_id) -> tuple[str, ...]


class Engine:
    def __init__(self, deps: Deps) -> None:
        self.d = deps
        self.s = deps.settings

    # -- the pipeline -------------------------------------------------------- #

    def answer(self, question: str, *, conversation_id: str = "",
               as_of=None) -> Iterator[Event]:
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        # ---- 0. embed once, reused by routing AND dense retrieval ----------
        # With a hosted embedder this is a ~1s call; paying it twice per
        # question would be the single most expensive mistake available here.
        vector = None
        if self.d.embed_query is not None:
            try:
                vector = self.d.embed_query(question)
            except Exception as exc:                      # noqa: BLE001
                # Degradation, not failure: retrieval falls back to lexical and
                # the router to its regex tier. Quota exhaustion is the common
                # cause and it must never fail a question.
                log.warning("query embedding unavailable: %s", exc)

        # ---- 1. route (jurisdiction gate reads the ORIGINAL text) ----------
        plan: Understanding = self.d.router.understand(question, vector)
        yield Event("router", {"path": self._path_of(plan).value, **plan.to_dict()})

        if plan.should_abstain:
            yield Event("abstain", {
                "message": "That question is about a different privacy regime. "
                           "This assistant covers only India's Digital Personal "
                           "Data Protection Act, 2023 and its Rules, 2025.",
                "reason": "out of jurisdiction: "
                          + ", ".join(plan.markers or ("foreign law",))})
            yield Event("done", self._done(plan, Path.ABSTAIN, elapsed(), None))
            return

        # ---- 2. prior turns contribute PROVISIONS, never prose -------------
        prior: tuple[str, ...] = ()
        if (self.s.CONVERSATIONS and conversation_id and plan.has_anaphora
                and self.d.prior_provisions is not None):
            prior = self.d.prior_provisions(conversation_id)[:self.s.CONVERSATION_SEEDS]

        # ---- 3. retrieve ---------------------------------------------------
        from understanding.normalize import normalize
        n = normalize(question)
        results, trace, expansion = self.d.retriever.retrieve(
            question, query_vector=vector, retrieval_text=n.retrieval,
            prior_ids=prior, intent_kinds=INJECT_KINDS.get(plan.intent, ()),
            reranker=self.d.reranker)

        yield Event("retrieval", {
            "elapsed_ms": elapsed(),
            "provisions": [r.to_dict() for r in results],
            **trace.to_dict(), "expansion": expansion.to_dict()})

        # ---- 4. template path — must AGREE with retrieval ------------------
        if plan.uses_template:
            # The template's subject guard asks "does the question NAME this
            # term?", and it must ask that of the repaired spelling. Otherwise
            # retrieval finds `def-data-fiduciary` for "wat is a Daata
            # Fudiciary", the guard sees no "fiduciary" in the question, the
            # template declines, and a definition the graph could have printed
            # verbatim is handed to a model instead.
            #
            # SUBSTITUTED here, not appended as retrieval does: this is a
            # word-presence test, not a score, and "Daata" still failing a
            # word-presence check is the whole problem.
            rendered = templates.render(plan.intent, _apply(trace.repairs, n.clean),
                                        results, self.d.graph, plan.provision_id)
            if rendered is not None:
                yield from self._stream_template(rendered, results, plan,
                                                 elapsed)
                return
            log.info("template %r declined; using synthesis", plan.intent)
            # Re-announce the route. Without this correction every consumer
            # still believes a model-generated answer came from the graph with
            # zero hallucination risk — the exact opposite of the truth.
            yield Event("router", {"path": Path.MODEL.value,
                                   "template_declined": True,
                                   **plan.to_dict()})

        # ---- 5. sufficiency (deterministic first) --------------------------
        verdict = sufficiency.assess(
            results, trace,
            floor=self.s.SUFFICIENCY_FLOOR, ceiling=self.s.SUFFICIENCY_CEILING,
            named_provision=bool(plan.provision_id))
        verdict = sufficiency.resolve_uncertain(
            verdict, allow_model=self.s.SUFFICIENCY_LLM)

        if verdict.verdict is sufficiency.Sufficiency.INSUFFICIENT:
            # Re-announce, for the same reason the template decline does: the
            # first `router` event said "llm" because that was the plan before
            # retrieval ran. A consumer that trusted it would report a model
            # answer for a question no model ever saw. Every route change is
            # announced; the LAST router event is always the truth.
            yield Event("router", {"path": Path.ABSTAIN.value,
                                   "abstained_on": verdict.signal,
                                   **plan.to_dict()})
            yield Event("abstain", {
                "message": "This doesn't look like something the DPDP Act, "
                           "2023 or its Rules, 2025 cover. Try rephrasing, or "
                           "it may be outside their scope.",
                "reason": verdict.reason, "signal": verdict.to_dict()})
            yield Event("done", self._done(plan, Path.ABSTAIN, elapsed(), None))
            return

        # ---- 6. synthesise -------------------------------------------------
        if self.d.synthesize is None:
            yield Event("error", {"message": "no model provider is configured"})
            return

        context = build_context(results, self.s.MAX_CONTEXT_CHARS)
        allowed = "\n".join(f"  {r.node_id}  = {r.chunk.label}" for r in results)

        parts: list[str] = []
        try:
            for piece in self.d.synthesize(question=question, context=context,
                                           allowed_ids=allowed, plan=plan):
                parts.append(piece)
                yield Event("token", {"t": piece})
        except Exception as exc:                          # noqa: BLE001
            log.exception("generation failed")
            # An LLMError's message is OURS — written for a reader, naming the
            # cause and what to do ("rate limit reached ... wait or configure a
            # paid key"). Flattening it to "could not be generated" hides the
            # one thing that distinguishes a transient cap from a real outage.
            # Anything else stays generic: it may carry a provider's internals.
            from providers.llm import LLMError
            yield Event("error", {
                "message": str(exc) if isinstance(exc, LLMError)
                           else "the answer could not be generated"})
            return

        answer = "".join(parts)

        # ---- 7. verify — POST-HOC, never blocking --------------------------
        claims = self._claims_from(answer, plan)
        checked = cite.check_text(answer, results, self.d.graph)
        claims = claimcheck.verify(claims, results, self.d.graph,
                                   quote_check=self.s.QUOTE_CHECK,
                                   numeric_check=self.s.NUMERIC_CHECK)

        yield Event("citations", {
            "citations": [c.to_dict() for c in checked],
            "penalties": cite.penalty_facts(results, self.d.graph)})

        flags = [c for c in claims if c.verdict is not Verdict.SUPPORTED]
        if plan.caveat:
            flags.append(Claim(text="scope", verdict=Verdict.CAVEAT,
                               note=plan.caveat))
        if flags:
            yield Event("claims", {"claims": [c.to_dict() for c in flags]})

        yield Event("done", self._done(plan, Path.MODEL, elapsed(),
                                       getattr(self.s, "MODEL", None),
                                       context_chars=len(context)))

    # -- helpers ------------------------------------------------------------- #

    def _path_of(self, plan: Understanding) -> Path:
        if plan.should_abstain:
            return Path.ABSTAIN
        return Path.TEMPLATE if plan.uses_template else Path.MODEL

    def _claims_from(self, answer: str, plan: Understanding) -> list[Claim]:
        """Split a free-text answer into claims for verification.

        Sentence-level is a coarse approximation of the structured-output
        path, used when the model returned prose. It is deliberately kept —
        a degraded verification is far better than none, and it is the only
        thing standing between a schema-ignoring model and an unchecked answer.
        """
        from core import ids
        out: list[Claim] = []
        for sentence in _sentences(answer):
            cited = tuple(node_id for node_id, _ in ids.find_all(sentence))
            if cited or _has_figure(sentence):
                out.append(Claim(text=sentence, cites=cited))
        return out

    def _stream_template(self, rendered, results, plan, elapsed) -> Iterator[Event]:
        """A template answer, over the SAME event sequence as a model answer.

        It streams although the text is already in memory. That is pure UX and
        deliberate: an interface that is instant for some questions and
        progressive for others reads as broken rather than fast, and the
        client would need a second rendering path.
        """
        for piece in rendered.stream_chunks():
            yield Event("token", {"t": piece})

        checked = cite.check_rendered(rendered.cites, self.d.graph)
        yield Event("citations", {
            "citations": [c.to_dict() for c in checked],
            "penalties": cite.penalty_facts(results, self.d.graph),
            "highlight": rendered.highlight})

        # No numeric check here, and that is not an omission: every figure came
        # from Provision.penalty or a verbatim text field, so it is sourced by
        # construction. Checking would only manufacture false positives from
        # formatting differences.
        flags: list[Claim] = []
        if plan.caveat:
            flags.append(Claim(text="scope", verdict=Verdict.CAVEAT,
                               note=plan.caveat))
        if flags:
            yield Event("claims", {"claims": [c.to_dict() for c in flags]})

        yield Event("done", self._done(plan, Path.TEMPLATE, elapsed(), None))

    def _done(self, plan, path: Path, ms: int, model, context_chars: int = 0) -> dict:
        return {"elapsed_ms": ms, "path": path.value, "intent": plan.intent.value,
                "model": model, "context_chars": context_chars,
                "build_id": getattr(self.d.corpus, "build_id", ""),
                "graph_degraded": getattr(self.d.graph, "degraded", False)}


def _apply(repairs: tuple[tuple[str, str], ...], text: str) -> str:
    """Rewrite a question in the spellings retrieval actually scored.

    Retrieval APPENDS its repairs, because there it is feeding a scorer and
    the original term must keep whatever weight it earned. A template guard
    is not a scorer — it asks "does this question name the term?" — so here
    the misspelling is REPLACED. Appending would leave "Daata" in the text
    and the guard's every-word test would still fail on it.
    """
    if not repairs:
        return text
    for bad, good in repairs:
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text, flags=re.I)
    return text


_SENTENCE = None


def _sentences(text: str) -> list[str]:
    global _SENTENCE
    if _SENTENCE is None:
        import re
        # Not a general sentence splitter: it must NOT split on "section 8(5)."
        # or "rule 6(1)." which end in a period mid-sentence.
        _SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
    return [s.strip() for s in _SENTENCE.split(text) if len(s.strip()) > 12]


def _has_figure(text: str) -> bool:
    return bool(claimcheck.extract_figures(text))
