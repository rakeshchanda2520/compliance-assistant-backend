"""
Citation verification — the point of the whole system.

A model can write "section 8(5)" whether or not section 8(5) says what it
claims. Every citation is resolved against the graph and labelled:

    verified        exists AND was in the retrieved context
    out_of_context  exists but was NOT retrieved — recalled from training
                    rather than read. Treat with suspicion.
    unresolved      no such provision. The model invented it.

`out_of_context` earns its keep even on the structured path: a model can emit
a node id it was never shown, and that is precisely a citation recalled rather
than read.
"""
from __future__ import annotations

from core import ids
from core.models import Citation, CitationStatus, Scored


def display_text(node_id: str, graph) -> str:
    """What a source card shows — NOT simply `provision.text`.

    A Section here is frequently a CONTAINER: sections 9, 10 and 32 carry a
    headnote and no text of their own because their substance lives in
    sub-sections. All three are cited by the penalty template as the duty a
    Schedule row penalises, so a card for "Section 32" rendered as a bare
    headnote with nothing under it — indistinguishable, to a reader, from a
    broken citation.

    Falling back to the children is safe and is not a paraphrase: they are the
    section, verbatim and in document order.
    """
    provision = graph.provisions.get(node_id)
    if provision is None:
        return ""

    text = (provision.text or "").strip()
    if node_id.startswith("pen-"):
        # A Schedule row is only meaningful with its amount attached.
        return f"{text}  —  {provision.penalty}".strip(" —")
    if text:
        return text

    parts = [graph.provisions[c].text.strip()
             for c in graph.descendants_of(node_id)
             if c in graph.provisions and graph.provisions[c].text.strip()]
    return "\n".join(parts)


def display_label(node_id: str, graph) -> str:
    """How a citation is captioned — the Gazette's own words where we have them.

    `ids.label_for` reconstructs a label from a node id, and a node id is
    lowercased, so the true casing of a defined term is simply not in it. It
    rendered `def-data-fiduciary` as `the definition of “data fiduciary”`,
    which is not how the Act writes it. Title-casing cannot fix that either:
    the Act writes "Data Fiduciary" but also "child" and "automated", and only
    the source knows which is which.

    Sections and rules keep the computed label. There `label_for` is not
    guessing — "Section 8(5)" is derived, canonical, and covered by tests.
    """
    if node_id.startswith(("def-", "rules-def-")):
        provision = graph.provisions.get(node_id)
        if provision is not None and provision.label:
            return provision.label
    return ids.label_for(node_id)


def _retrieved_ids(results: list[Scored]) -> set[str]:
    return {r.node_id for r in results}


def _in_context(node_id: str, retrieved: set[str]) -> bool:
    return any(ids.covers(node_id, r) for r in retrieved)


def _resolve(node_id: str, retrieved: set[str], graph) -> Citation:
    provision = graph.provisions.get(node_id)
    if provision is None:
        parent = ids.parent_of(node_id)
        note = ("no such provision in this corpus"
                if not parent or parent not in graph.provisions
                else f"no such provision; the nearest real one is "
                     f"{ids.label_for(parent)}")
        return Citation(node_id, ids.label_for(node_id),
                        CitationStatus.UNRESOLVED, note=note)

    verified = _in_context(node_id, retrieved)
    return Citation(
        id=node_id,
        label=display_label(node_id, graph),
        status=CitationStatus.VERIFIED if verified else CitationStatus.OUT_OF_CONTEXT,
        text=display_text(node_id, graph),
        headnote=provision.headnote,
        note="" if verified else
             "this provision exists but was not retrieved for this question",
    )


def check_text(answer: str, results: list[Scored], graph) -> list[Citation]:
    """Regex path: every citation the answer's PROSE makes.

    The fallback for a model that ignored the output schema. Kept live rather
    than as legacy, because a degraded answer is better than none.
    """
    retrieved = _retrieved_ids(results)
    found = [_resolve(node_id, retrieved, graph)
             for node_id, _ in ids.find_all(answer)]
    return sorted(found, key=lambda c: c.sort_key)


def check_claimed(claimed: list[str], results: list[Scored], graph) -> list[Citation]:
    """Structured path: the model emitted node ids directly.

    Set membership, not pattern matching. `check_text` has to anticipate every
    way a model might spell a citation — miss one format and a fully sourced
    answer looks unsourced. Here there is no format to miss.
    """
    retrieved = _retrieved_ids(results)
    seen: dict[str, Citation] = {}
    for node_id in claimed:
        node_id = (node_id or "").strip()
        if node_id and node_id not in seen:
            seen[node_id] = _resolve(node_id, retrieved, graph)
    return sorted(seen.values(), key=lambda c: c.sort_key)


def check_rendered(node_ids: tuple[str, ...], graph) -> list[Citation]:
    """Template path: verified by CONSTRUCTION.

    Not a shortcut. These are the nodes the renderer actually read out of the
    graph, so the provision both exists and was in context by definition.
    There is no model claim here to doubt.
    """
    out = []
    for node_id in node_ids:
        if node_id in graph.provisions:
            out.append(Citation(
                id=node_id, label=display_label(node_id, graph),
                status=CitationStatus.VERIFIED,
                text=display_text(node_id, graph),
                headnote=graph.provisions[node_id].headnote))
    return out


def penalty_facts(results: list[Scored], graph) -> list[dict]:
    """Amounts read from the graph, never from the model.

    A small model has already misread a Schedule figure in this corpus, and an
    amount is structured data the build already resolved. There is no reason
    to let a model restate it.
    """
    duty_of = graph.penalised_by()
    facts = []
    for r in results:
        if r.chunk.kind != "Penalty":
            continue
        provision = graph.provisions.get(r.node_id)
        if provision is None:
            continue
        facts.append({
            "entry": r.chunk.label,
            "amount": provision.penalty,
            "applies_to": [ids.label_for(d)
                           for d in sorted(duty_of.get(r.node_id, ()))],
        })
    return facts
