"""
Answers assembled from the graph. Zero model calls, zero hallucination surface.

The highest-value path in the system, and the reasoning is blunt: both errors
this project has on record — a misread Schedule figure and a wrong reading of
section 14 — happened on questions whose answers were already sitting in the
graph as structured data. A model was asked to restate facts that did not need
restating, and it restated one of them wrongly.

Every string these functions emit is either a fixed connective written here or
text read straight from a Provision. Nothing is paraphrased, so nothing can be
paraphrased wrongly.

**A template must AGREE with retrieval before it fires.** This is the guard
that makes the path safe, and it is not optional. A template renders from the
graph and returns BEFORE the sufficiency gate, and its citations are verified
by construction — so nothing downstream can catch it answering the wrong
question. Without the guards below, the intent LABEL alone decided, and that
label comes from a regex:

    "what is the tallest mountain"   -> the DPDP definition of "automated",
                                        marked verified
    "what is the best pizza topping" -> the definition of "Data Fiduciary"
    "can we keep data outside India?"-> the Third Schedule's retention periods,
                                        while retrieval had correctly found s-16

A false negative here is cheap: the request falls through to synthesis, which
answers from the same provisions. A false positive is a confident wrong answer
wearing a verified citation.
"""
from __future__ import annotations

import logging
import re

from core.ids import label_for
from core.models import Chunk, Intent, Rendered, Scored

log = logging.getLogger(__name__)

_WORDS = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower()))


def term_of(label: str) -> str:
    """The defined term out of a Definition's label.

    Labels read: Definition of “Data Fiduciary” — curly quotes, from the
    Gazette. Falls back to stripping the prefix when the quotes are absent.
    """
    quoted = re.search("[" + "“‘\"'" + "]([^" + "”’\"'" + "]+)", label)
    if quoted:
        return quoted.group(1)
    return re.sub(r"^\s*definitions?\s+of\s+", "", label, flags=re.I).strip()


def question_names(term: str, question: str) -> bool:
    """Does the question actually name this term?

    Every word of the term must appear, order-free — so "who is a Data
    Principal?" matches "Data Principal", while "what is the tallest mountain"
    matches nothing in the corpus and correctly declines.
    """
    term_words = _words(term)
    return bool(term_words) and term_words <= _words(question)


def render(intent: Intent, question: str, results: list[Scored], graph,
           provision_id: str = "") -> Rendered | None:
    """Dispatch. None means "the graph cannot answer this" — fall through."""
    try:
        if intent is Intent.PENALTY:
            return _penalty(results, graph)
        if intent is Intent.DEFINITION:
            return _definition(results, graph, question)
        if intent is Intent.RETENTION:
            return _retention(results, graph)
        if intent is Intent.DIRECT_LOOKUP:
            return _direct_lookup(provision_id, graph)
    except Exception:                                     # noqa: BLE001
        # A template must never take down a request. Falling through to
        # synthesis is a worse answer, not a failed one.
        log.exception("template %r failed; falling through to synthesis", intent)
    return None


# --------------------------------------------------------------------------- #

def _penalty(results: list[Scored], graph) -> Rendered | None:
    """Schedule rows, their amounts, and the duty each one penalises.

    ALL rows are rendered, ordered by relevance — never filtered to one.
    Ranking seven near-identical rows against each other was measured
    unreliable, and "fine if customer data leaks" genuinely maps to more than
    one (security safeguards AND breach notification). The reranker may ORDER
    and HIGHLIGHT; it may never delete a row.

    Amounts come from `Provision.penalty`, never from prose.
    """
    rows = [r for r in results if r.chunk.kind == "Penalty"]
    if not rows:
        return None

    duty_of = graph.penalised_by()
    # Best-supported row first, so the summary line can name it — but every
    # row still appears below.
    rows.sort(key=lambda r: -(r.rerank if r.rerank is not None else r.fused))
    highlight = rows[0].node_id

    lines: list[str] = []
    cited: list[str] = []
    for row in rows:
        provision = graph.provisions.get(row.node_id)
        if provision is None:
            continue
        cited.append(provision.id)
        duties = sorted(duty_of.get(provision.id, ()))
        cited.extend(duties)

        where = f" ({', '.join(label_for(d) for d in duties)})" if duties else ""
        breach = (provision.text or "").strip().rstrip(".")
        amount = (provision.penalty or "").strip()
        # Omitted entirely rather than printed empty. `penalty` is unset when
        # the graph is the corpus-derived fallback (no Neo4j), and a label
        # with nothing after it reads as a missing figure rather than an
        # unavailable one — the amount is still in `breach` above.
        maximum = f"\nMaximum penalty: {amount}" if amount else ""
        lines.append(f"**{provision.label}**{where}\n"
                     f"What it covers: {breach}.{maximum}")

    if not lines:
        return None

    top = graph.provisions.get(highlight)
    # The summary amount is composed from the GRAPH field, never from
    # generated text — the same rule the rest of the system applies to every
    # rupee figure, extended to this line.
    headline = (f"Short answer:\nThe most relevant entry is **{top.label}** — "
                f"up to {(top.penalty or '').strip()}. "
                f"The Schedule's other entries are listed below; all amounts "
                f"are read directly from the Act, not restated from memory.\n\n"
                if top and top.penalty else
                "Short answer:\nThe Act's Schedule sets the following maximum "
                "penalties, read directly from the Act.\n\n")

    body = "\n\n".join(lines)
    footer = ("\n\nWhy:\nThese are ceilings, not fixed fines. A penalty is "
              "imposed by the Data Protection Board of India only after an "
              "inquiry under section 27, and the Board sets the actual amount "
              "within the maximum shown.")
    return Rendered(f"{headline}The law says:\n\n{body}{footer}",
                    cites=tuple(dict.fromkeys(cited)),
                    intent=Intent.PENALTY, highlight=highlight)


def _definition(results: list[Scored], graph, question: str) -> Rendered | None:
    """The verbatim definition, plus where the term is actually used.

    Guarded: the question must NAME the term. Retrieval always returns
    something, so ranking alone cannot distinguish "the definition they asked
    for" from "the nearest definition to a question about pizza".
    """
    hits = [r for r in results if r.chunk.kind == "Definition"]
    if not hits:
        return None

    top = next((h for h in hits
                if question_names(term_of(h.chunk.label), question)), None)
    if top is None:
        log.info("definition template declined: the question names no defined "
                 "term (best candidate was %r)", hits[0].chunk.label)
        return None

    provision = graph.provisions.get(top.node_id)
    if provision is None or not provision.text.strip():
        return None

    cited = [provision.id]
    used_in = sorted(graph.mentions_of(provision.id))
    used_in.sort(key=lambda n: -(graph.provisions[n].authority
                                 if n in graph.provisions else 0.0))
    shown = used_in[:3]
    cited.extend(shown)

    text = ("Short answer:\nThe law defines this term itself. Here it is in "
            f"full, in the Act's own words.\n\nThe law says:\n"
            f"**{provision.label}** — {provision.text.strip()}")

    if shown:
        where = ", ".join(label_for(n) for n in shown)
        more = (f", and in {len(used_in) - len(shown)} other provisions"
                if len(used_in) > len(shown) else "")
        text += (f"\n\nWhy it matters:\nThis term carries the meaning above "
                 f"everywhere it appears. It is used in {where}{more}, so the "
                 f"definition decides how each of those applies to you.")

    return Rendered(text, cites=tuple(dict.fromkeys(cited)),
                    intent=Intent.DEFINITION, highlight=provision.id)


def _retention(results: list[Scored], graph) -> Rendered | None:
    """The Third Schedule's retention periods, with the rule that invokes them.

    Guarded: retrieval must AGREE this is about retention. This template reads
    `rules-sch-third` straight out of the graph and never looks at `results`,
    so on the intent label alone it answered "can we keep data on servers
    outside India?" — a section 16 cross-border question retrieval had got
    RIGHT — with the Third Schedule's retention periods.
    """
    schedule = graph.provisions.get("rules-sch-third")
    if schedule is None:
        return None

    if not any(r.node_id == "rules-sch-third"
               or r.node_id.startswith(("rules-sch-third-", "r-8"))
               for r in results):
        log.info("retention template declined: retrieval found %s, not the "
                 "Third Schedule",
                 results[0].chunk.label if results else "nothing")
        return None

    cited = ["rules-sch-third"]
    text = ("Short answer:\nHow long you may keep personal data is set by "
            "rule 8 of the DPDP Rules, 2025, read with the Third Schedule. "
            "The Schedule gives the period; rule 8 says what happens when it "
            "runs out.\n\nThe law says:\n")

    for rule_id in ("r-8-1", "r-8"):
        rule = graph.provisions.get(rule_id)
        if rule and rule.text.strip():
            text += f"{rule.text.strip()}\n\n"
            cited.append(rule_id)
            break

    rows = [graph.provisions[n] for n in graph.descendants_of("rules-sch-third")
            if n in graph.provisions and "row" in n]
    if rows:
        text += "**Third Schedule — retention periods**\n\n"
        for row in rows:
            text += f"- {row.text.strip()}\n"
            cited.append(row.id)
    else:
        text += schedule.text.strip()

    text += ("\n\nWhat to do:\nOnce the period has passed, the purpose you "
             "collected the data for is treated as served and you must delete "
             "it, along with any copies your processors hold. The one "
             "exception is data another law requires you to keep — record "
             "which law that is.")
    return Rendered(text, cites=tuple(dict.fromkeys(cited)),
                    intent=Intent.RETENTION, highlight="rules-sch-third")


def _direct_lookup(provision_id: str, graph) -> Rendered | None:
    """A named provision, verbatim, with its children in document order.

    "What does section 8 say?" is not a search problem — the question names
    its own answer. Routing it through a ranker asks an algorithm to
    rediscover a fact the caller already stated.
    """
    provision = graph.provisions.get(provision_id)
    if provision is None:
        return None

    cited = [provision.id]
    header = provision.headnote.strip() or provision.label
    text = (f"The law says:\n**{provision.label} — {header}**, in full, in the "
            f"instrument's own words:\n\n")

    if provision.text.strip():
        text += provision.text.strip() + "\n"

    for child in graph.descendants_of(provision_id):
        node = graph.provisions.get(child)
        if node is None or not node.text.strip():
            continue
        depth = child.count("-") - provision_id.count("-")
        text += f"\n{'  ' * max(depth - 1, 0)}{node.text.strip()}"
        cited.append(child)

    entry = graph.penalty_for().get(provision_id)
    if entry:
        row = graph.provisions.get(entry)
        if row:
            text += (f"\n\nPenalty:\nBreaching this provision is penalised "
                     f"under {row.label}, up to {row.penalty}.")
            cited.append(entry)

    return Rendered(text.rstrip(), cites=tuple(dict.fromkeys(cited)),
                    intent=Intent.DIRECT_LOOKUP, highlight=provision_id)
