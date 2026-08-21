"""
Answers assembled from the graph. Zero model calls, zero hallucination surface.

This is the highest-value part of V2, and the reasoning is blunt: BOTH errors
this project has on record — a misread Schedule figure and a wrong reading of
s.14 — happened on questions whose answers were already sitting in the graph
as structured data. A model was asked to restate facts that did not need
restating, and it restated one of them wrongly.

Four intents are fully answerable without generation:

    penalty        Schedule row + its amount + the duty it penalises
    definition     the verbatim definition + where the term is used
    retention      the Third Schedule's rows + the rule that points at them
    direct_lookup  a named provision and its children, verbatim

Every string these functions emit is either a fixed connective written here or
text read straight from a Provision. Nothing is paraphrased, so nothing can be
paraphrased wrongly.

They still STREAM (phase 4 of V2_PLAN.md's override 4): the rendered text is
chunked out over SSE like any other answer, because a UI that is instant for
some questions and progressive for others reads as broken rather than fast.

`render()` returns None when the graph cannot actually support the intent —
a definition question about an undefined term, say. The caller falls through
to synthesis. A template that guessed would be worse than no template.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .citations import label_for
from .graph_store import Graph
from .retrieval import Result

log = logging.getLogger(__name__)

# Roughly a word or two per event: enough for the answer to visibly build,
# few enough that a 400-word rendering is not thousands of SSE frames.
STREAM_CHUNK_CHARS = 24


@dataclass
class Rendered:
    """A template answer plus the provisions it is built from.

    `citations` is not parsed back out of the text — it is the set of nodes
    the renderer actually read. Citation verification for a template path is
    therefore exact by construction, not a regex approximation of itself.
    """
    text: str
    citations: list[str] = field(default_factory=list)
    intent: str = ""

    def chunks(self, size: int = STREAM_CHUNK_CHARS):
        """Split on whitespace boundaries so a word never straddles two SSE
        frames — the frontend concatenates fragments verbatim."""
        buffer = ""
        for word in self.text.split(" "):
            candidate = f"{buffer} {word}" if buffer else word
            if len(candidate) >= size:
                yield candidate + " "
                buffer = ""
            else:
                buffer = candidate
        if buffer:
            yield buffer


def render(intent: str, results: list[Result], graph: Graph,
           provision_id: str = "") -> Rendered | None:
    """Dispatch. None means "the graph cannot answer this" — fall through."""
    try:
        if intent == "penalty":
            return _penalty(results, graph)
        if intent == "definition":
            return _definition(results, graph)
        if intent == "retention":
            return _retention(results, graph)
        if intent == "direct_lookup":
            return _direct_lookup(provision_id, graph)
    except Exception:                                  # noqa: BLE001
        # A template must never take down a request. Falling through to
        # synthesis is a worse answer, not a failed one.
        log.exception("template %r failed; falling through to synthesis", intent)
    return None


# --------------------------------------------------------------------------- #

def _penalty(results: list[Result], graph: Graph) -> Rendered | None:
    """Schedule rows, their amounts, and the duty each one penalises.

    Amounts come from `Provision.penalty`, never from prose. `penalised_by()`
    supplies the duty, which is the cross-document join the graph exists for:
    the row states an amount and describes no duty; the duty states no amount.
    """
    rows = [r for r in results if r.chunk.kind == "Penalty"]
    if not rows:
        return None

    duty_of = graph.penalised_by()
    lines: list[str] = []
    cited: list[str] = []

    rows.sort(key=lambda r: r.chunk.node_id)
    for row in rows:
        provision = graph.provisions.get(row.chunk.node_id)
        if provision is None:
            continue
        cited.append(provision.id)

        duties = sorted(duty_of.get(provision.id, []))
        cited.extend(duties)

        breach = (provision.text or "").strip().rstrip(".")
        amount = (provision.penalty or "").strip()
        where = f" ({', '.join(label_for(d) for d in duties)})" if duties else ""

        lines.append(
            f"**{provision.label}**{where}\n"
            f"{breach}.\n"
            f"Penalty: {amount}")

    if not lines:
        return None

    body = "\n\n".join(lines)
    preamble = ("The Schedule to the Act sets these penalties. Amounts are read "
                "directly from the Act's Schedule:"
                if len(lines) > 1 else
                "The Act's Schedule sets this penalty:")
    footer = ("\n\nPenalties are imposed by the Data Protection Board after an "
              "inquiry under section 27; the amounts above are the maximum the "
              "Schedule permits.")
    return Rendered(f"{preamble}\n\n{body}{footer}",
                    citations=_dedupe(cited), intent="penalty")


def _definition(results: list[Result], graph: Graph) -> Rendered | None:
    """The verbatim definition, plus where the term is actually used.

    The usage sites come from MENTIONS edges — exhaustive by construction at
    build time, so this answers "where does this matter?" with a complete list
    rather than whatever retrieval happened to surface.
    """
    hits = [r for r in results if r.chunk.kind == "Definition"]
    if not hits:
        return None

    top = hits[0]
    provision = graph.provisions.get(top.chunk.node_id)
    if provision is None or not provision.text.strip():
        return None

    cited = [provision.id]
    used_in = sorted({src for src, dst, etype in graph.edges
                      if etype == "MENTIONS" and dst == provision.id})

    # Rank usage sites by authority so the three shown are the load-bearing
    # ones rather than the first three alphabetically.
    used_in.sort(key=lambda n: -graph.provisions[n].authority
                 if n in graph.provisions else 0.0)
    shown = used_in[:3]
    cited.extend(shown)

    text = f"**{provision.label}** is defined as follows:\n\n{provision.text.strip()}"

    if shown:
        where = ", ".join(label_for(n) for n in shown)
        more = f", and {len(used_in) - len(shown)} other provisions" \
            if len(used_in) > len(shown) else ""
        text += f"\n\nThis term is used in {where}{more}."

    others = [r for r in hits[1:3]]
    if others:
        text += ("\n\nRelated definitions: "
                 + ", ".join(r.chunk.label for r in others) + ".")
        cited.extend(r.chunk.node_id for r in others)

    return Rendered(text, citations=_dedupe(cited), intent="definition")


def _retention(results: list[Result], graph: Graph) -> Rendered | None:
    """The Third Schedule's retention periods, with the rule that invokes them.

    The table rows are separate nodes (recovered column-wise at build time
    precisely because pdfplumber reads them scrambled), so this reads them
    from the graph rather than from the schedule's own running text.
    """
    schedule = graph.provisions.get("rules-sch-third")
    if schedule is None:
        return None

    rows = [graph.provisions[n] for n in graph.children_of("rules-sch-third")
            if n in graph.provisions and n.endswith(tuple("0123456789"))
            and "row" in n]
    if not rows:
        rows = [graph.provisions[n] for n in graph.descendants_of("rules-sch-third")
                if n in graph.provisions and "row" in n]

    cited = ["rules-sch-third"]
    text = ("Retention is governed by rule 8 of the DPDP Rules, 2025, read "
            "with the Third Schedule.\n\n")

    if rule8 := graph.provisions.get("r-8-1"):
        text += f"{rule8.text.strip()}\n\n"
        cited.append("r-8-1")
    elif rule8 := graph.provisions.get("r-8"):
        text += f"{rule8.text.strip()}\n\n"
        cited.append("r-8")

    if rows:
        text += "**Third Schedule — retention periods**\n\n"
        for row in rows:
            text += f"- {row.text.strip()}\n"
            cited.append(row.id)
    else:
        text += schedule.text.strip()

    text += ("\n\nAfter the applicable period, the specified purpose is deemed "
             "no longer served and the personal data must be erased unless "
             "retention is required by law.")
    return Rendered(text, citations=_dedupe(cited), intent="retention")


def _direct_lookup(provision_id: str, graph: Graph) -> Rendered | None:
    """A named provision, verbatim, with its children in document order.

    "What does section 8 say?" is not a search problem — the question names
    its own answer. Routing it through BM25 was V1 asking a ranking algorithm
    to rediscover a fact the caller already stated.
    """
    provision = graph.provisions.get(provision_id)
    if provision is None:
        return None

    cited = [provision.id]
    header = provision.headnote.strip() or provision.label
    text = f"**{provision.label} — {header}**\n\n"

    if provision.text.strip():
        text += provision.text.strip() + "\n"

    for child in graph.descendants_of(provision_id):
        node = graph.provisions.get(child)
        if node is None or not node.text.strip():
            continue
        # Indent by depth so a clause reads as belonging to its sub-section.
        depth = child.count("-") - provision_id.count("-")
        text += f"\n{'  ' * max(depth - 1, 0)}{node.text.strip()}"
        cited.append(child)

    if penalty_entry := graph.penalty_for().get(provision_id):
        entry = graph.provisions.get(penalty_entry)
        if entry:
            text += (f"\n\nBreach of this provision is penalised by "
                     f"{entry.label}: {entry.penalty}")
            cited.append(penalty_entry)

    return Rendered(text.rstrip(), citations=_dedupe(cited),
                    intent="direct_lookup")


def _dedupe(ids: list[str]) -> list[str]:
    """Order-preserving, so citations appear in the order the answer uses
    them rather than in whatever order set iteration produces."""
    seen: set[str] = set()
    out: list[str] = []
    for node_id in ids:
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out
