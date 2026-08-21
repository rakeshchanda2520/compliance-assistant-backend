"""
Citation verification — the point of the whole system.

A language model can write "§8(5)" whether or not §8(5) says what it claims.
Every citation in an answer is therefore resolved against the graph and
labelled:

    verified        the provision exists AND was in the retrieved context
    out_of_context  it exists but was NOT retrieved — recalled from training
                    rather than read. Treat with suspicion.
    unresolved      no such provision. The model invented it.

Penalty amounts bypass the model entirely: they are read from the graph and
rendered directly. A small model has already misread a Schedule figure in
this corpus, and an amount is structured data the build already resolved —
there is no reason to let a model restate it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .graph_store import Graph

# Three citation vocabularies, because the corpus now holds two documents:
#   Act    "§8(5)", "§ 8 (5)(a)", "section 8(5)"      -> s-8-5
#   Rules  "rule 6(1)", "rule 6(1)(a)"                -> r-6-1
#   both   "Schedule entry 2" (Act's penalty table)   -> pen-2
#          "First Schedule", "Part B of First Schedule" (Rules)
# The Act's Schedule is a numbered penalty table; the Rules' schedules are
# named by ordinal, so the two never collide.
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
RE_CITATION = re.compile(
    r"(?:§\s*|\bsections?\s+)(?P<sec>\d{1,2})(?P<sec_parts>(?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\brules?\s+(?P<rule>\d{1,2})(?P<rule_parts>(?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\bSchedule\s+(?:entry\s+)?(?P<entry>\d)\b"
    r"|\b(?:Part\s+(?P<part>[A-Z])\s+of\s+)?(?P<ord>" + "|".join(ORDINALS) + r")\s+Schedule\b",
    re.IGNORECASE)
RE_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")

STATUS_ORDER = {"unresolved": 0, "out_of_context": 1, "verified": 2}


@dataclass
class Citation:
    id: str
    label: str
    status: str
    text: str = ""
    headnote: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "status": self.status,
                "text": self.text, "headnote": self.headnote, "note": self.note}


def node_id_for(prefix: str, unit: str, parts: str) -> str:
    return "-".join([prefix, unit] + RE_PART.findall(parts or ""))


def node_id_from_match(m: re.Match) -> str | None:
    """The node id one citation points at, or None if it matched nothing."""
    if m.group("sec"):
        return node_id_for("s", m.group("sec"), m.group("sec_parts"))
    if m.group("rule"):
        return node_id_for("r", m.group("rule"), m.group("rule_parts"))
    if m.group("entry"):
        return f"pen-{m.group('entry')}"
    if m.group("ord"):
        node_id = f"rules-sch-{m.group('ord').lower()}"
        return f"{node_id}-part-{m.group('part').lower()}" if m.group("part") else node_id
    return None


def label_for(node_id: str) -> str:
    if node_id.startswith("pen-"):
        return f"Schedule entry {node_id[4:]}"
    if node_id.startswith("rules-sch-"):
        bits = node_id[len("rules-sch-"):].split("-part-")
        label = f"{bits[0].capitalize()} Schedule"
        return f"Part {bits[1].upper()} of {label}" if len(bits) > 1 else label
    bits = node_id.split("-")
    if len(bits) < 2:
        return node_id
    head = "rule " if bits[0] == "r" else "§"
    return f"{head}{bits[1]}" + "".join(f"({b})" for b in bits[2:])


def _in_context(node_id: str, retrieved: set[str]) -> bool:
    """A citation counts as retrieved if the exact node was retrieved, or if
    retrieval covered a parent or child of it — quoting §8(5) from a chunk
    that held all of §8 is not an out-of-context citation."""
    return node_id in retrieved or any(
        r.startswith(node_id + "-") or node_id.startswith(r + "-")
        for r in retrieved)


def check(answer: str, retrieved_node_ids: set[str], graph: Graph) -> list[Citation]:
    """Resolve every citation the answer makes. Ordered worst-first, so a
    reader sees what needs attention before what is fine."""
    seen: dict[str, Citation] = {}

    for match in RE_CITATION.finditer(answer):
        node_id = node_id_from_match(match)
        if node_id is None or node_id in seen:
            continue

        provision = graph.provisions.get(node_id)
        if provision is None:
            # A model writing §8(5)(z) is still pointing at §8(5); naming the
            # nearest real provision is more useful than "invented".
            parts_ = node_id.split("-")
            parent = "-".join(parts_[:-1])
            note = ("no such provision in this corpus"
                    if len(parts_) <= 2 or parent not in graph.provisions
                    else f"no such provision; nearest is {label_for(parent)}")
            seen[node_id] = Citation(node_id, label_for(node_id), "unresolved", note=note)
            continue

        text = provision.text
        if node_id.startswith("pen-"):
            # A Schedule row is only meaningful with its amount attached.
            text = f"{provision.text}  —  {provision.penalty}".strip(" —")

        verified = _in_context(node_id, retrieved_node_ids)
        seen[node_id] = Citation(
            id=node_id,
            label=label_for(node_id),
            status="verified" if verified else "out_of_context",
            text=text.strip(),
            headnote=provision.headnote,
            note="" if verified else
                 "this provision exists but was not retrieved for this question",
        )

    return sorted(seen.values(), key=lambda c: (STATUS_ORDER[c.status], c.id))


def check_structured(claimed: list[dict], retrieved_node_ids: set[str],
                     graph: Graph) -> list[Citation]:
    """Verification when the model emitted node ids directly (phase 3).

    The whole point of structured output: this is set membership, not pattern
    matching. `check()` above has to anticipate every way a model might spell
    a citation — miss one format and the answer looks unsourced when it was
    fully sourced. Here there is no format to miss.

    The statuses are unchanged, and `out_of_context` still earns its keep: a
    model can emit an id it was never shown, and that is precisely a citation
    recalled from training rather than read from the context.
    """
    seen: dict[str, Citation] = {}

    for item in claimed:
        node_id = (item.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue

        provision = graph.provisions.get(node_id)
        if provision is None:
            parts_ = node_id.split("-")
            parent = "-".join(parts_[:-1])
            note = ("no such provision in this corpus"
                    if len(parts_) <= 2 or parent not in graph.provisions
                    else f"no such provision; nearest is {label_for(parent)}")
            seen[node_id] = Citation(node_id, label_for(node_id),
                                     "unresolved", note=note)
            continue

        text = provision.text
        if node_id.startswith("pen-"):
            text = f"{provision.text}  —  {provision.penalty}".strip(" —")

        verified = _in_context(node_id, retrieved_node_ids)
        seen[node_id] = Citation(
            id=node_id,
            label=label_for(node_id),
            status="verified" if verified else "out_of_context",
            text=text.strip(),
            headnote=provision.headnote,
            note="" if verified else
                 "this provision exists but was not retrieved for this question",
        )

    return sorted(seen.values(), key=lambda c: (STATUS_ORDER[c.status], c.id))


def check_template(node_ids: list[str], graph: Graph) -> list[Citation]:
    """Citations for a template answer.

    Always `verified`, and that is not a shortcut: a template's citations are
    the nodes the renderer actually read out of the graph, so the provision
    both exists and was in context by construction. There is no model claim
    here to doubt.
    """
    out: list[Citation] = []
    for node_id in node_ids:
        provision = graph.provisions.get(node_id)
        if provision is None:
            continue
        text = provision.text
        if node_id.startswith("pen-"):
            text = f"{provision.text}  —  {provision.penalty}".strip(" —")
        out.append(Citation(id=node_id, label=label_for(node_id),
                            status="verified", text=text.strip(),
                            headnote=provision.headnote))
    return out


def penalty_facts(results, graph: Graph) -> list[dict]:
    """Penalty amounts read from the graph, never from the model."""
    duty_of = graph.penalised_by()
    facts = []
    for r in results:
        if r.chunk.kind != "Penalty":
            continue
        provision = graph.provisions.get(r.chunk.node_id)
        if provision is None:
            continue
        facts.append({
            "entry": r.chunk.label,
            "amount": provision.penalty,
            "applies_to": [label_for(d) for d in sorted(duty_of.get(r.chunk.node_id, []))],
        })
    return facts
