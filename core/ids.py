"""
Provision identifiers — the ONE place citation syntax is defined.

Every other component derives from this module: the verifier, the prompt's
citation instruction, the frontend's clickable-reference regex, and the tests.
The previous system wrote the same regex twice (once in Python, once in
JavaScript) and kept the two in step with a parity test. That test caught real
drift — a UI that knew `§8(5)` but not `rule 6(1)`, so every Rules citation
rendered with a source card and nothing in the prose pointing at it — but a
test that detects divergence is a workaround. Not diverging is the design.

`spec()` emits the whole grammar as data, and `scripts/build_frontend_spec.py`
compiles that into the JavaScript the page ships. One edit, both sides.

Two id spaces, and they must never be conflated:

    provision ids   what a CITATION may point at   (~686 nodes)
    chunk ids       what RETRIEVAL can return      (237 chunks)

`rules-sch-first` is a legitimate citation target but is not a chunk —
retrieval only ever sees `rules-sch-first-part-a` and `-part-b`. A checker
that tested both against one set would either reject a valid citation or
demand something retrieval can never produce.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# The Rules' schedules are named by ordinal; the Act's Schedule is a numbered
# penalty table. The two never collide, which is why one grammar covers both.
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")

# Written once, here. Named groups are load-bearing — `node_id_from_match`
# and the generated JavaScript both address them by name.
PATTERN = (
    r"(?:§\s*|\bsections?\s+)(?P<sec>\d{1,2})(?P<sec_parts>(?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\brules?\s+(?P<rule>\d{1,2})(?P<rule_parts>(?:\s*\(\s*[0-9a-zA-Z]{1,3}\s*\))*)"
    r"|\bSchedule\s+(?:entry\s+)?(?P<entry>\d)\b"
    r"|\b(?:Part\s+(?P<part>[A-Z])\s+of\s+)?(?P<ord>" + "|".join(ORDINALS) + r")\s+Schedule\b"
)

CITATION = re.compile(PATTERN, re.IGNORECASE)
_PART = re.compile(r"\(\s*([0-9a-zA-Z]{1,3})\s*\)")


def node_id_from_match(match: re.Match) -> str | None:
    """The provision id one matched citation points at, or None."""
    groups = match.groupdict()
    if groups.get("sec"):
        return "-".join(["s", groups["sec"]] + _PART.findall(groups.get("sec_parts") or ""))
    if groups.get("rule"):
        return "-".join(["r", groups["rule"]] + _PART.findall(groups.get("rule_parts") or ""))
    if groups.get("entry"):
        return f"pen-{groups['entry']}"
    if groups.get("ord"):
        base = f"rules-sch-{groups['ord'].lower()}"
        return f"{base}-part-{groups['part'].lower()}" if groups.get("part") else base
    return None


def find_all(text: str) -> list[tuple[str, re.Match]]:
    """Every citation in `text`, in order, de-duplicated by provision id."""
    seen: set[str] = set()
    out: list[tuple[str, re.Match]] = []
    for match in CITATION.finditer(text):
        node_id = node_id_from_match(match)
        if node_id and node_id not in seen:
            seen.add(node_id)
            out.append((node_id, match))
    return out


def label_for(node_id: str) -> str:
    """How a provision id is written for a human.

    "Section 8(5)", never "§8(5)". The section sign is legal-publishing
    convention and near-meaningless outside it — a reader who does not
    recognise the glyph cannot say the citation aloud or look it up, and this
    is written for compliance staff, engineers and product managers.

    Display only. `CITATION` above still PARSES `§`, because a model may well
    emit it and refusing a citation over how it was typed is exactly the
    failure structured output exists to avoid.
    """
    if node_id.startswith("pen-"):
        return f"Schedule entry {node_id[4:]}"
    if node_id.startswith("rules-sch-"):
        bits = node_id[len("rules-sch-"):].split("-part-")
        label = f"{bits[0].capitalize()} Schedule"
        return f"Part {bits[1].upper()} of {label}" if len(bits) > 1 else label
    # Definitions, from both instruments. Without this they fell through to the
    # section branch below and `def-data-fiduciary` rendered as
    # "Section data(fiduciary)" — the generic path reads bits[1] as a section
    # number and every later segment as a sub-section marker.
    for prefix in ("rules-def-", "def-"):
        if node_id.startswith(prefix):
            term = node_id[len(prefix):].replace("-", " ")
            return f"the definition of “{term}”"
    bits = node_id.split("-")
    if len(bits) < 2:
        return node_id
    # Anything whose second segment is not a number is not a section or rule
    # reference; returning it unchanged is more honest than inventing one.
    if not bits[1].isdigit():
        return node_id
    head = "Rule " if bits[0] == "r" else "Section "
    return f"{head}{bits[1]}" + "".join(f"({b})" for b in bits[2:])


def parent_of(node_id: str) -> str | None:
    """The enclosing provision, or None at the top.

    A model writing `s-8-5-z` is still pointing at `s-8-5`; naming the nearest
    real provision is more useful to a reader than "invented".
    """
    bits = node_id.split("-")
    if node_id.startswith("rules-sch-") or node_id.startswith("pen-") or len(bits) <= 2:
        return None
    return "-".join(bits[:-1])


def covers(node_id: str, other: str) -> bool:
    """True when one provision contains the other, either direction.

    Quoting section 8(5) out of a chunk that held all of section 8 is not an
    out-of-context citation, so containment counts as retrieval in both
    directions.
    """
    return (node_id == other
            or other.startswith(node_id + "-")
            or node_id.startswith(other + "-"))


@dataclass(frozen=True)
class CitationSpec:
    """The grammar, as data, for generating the frontend's copy."""
    pattern: str
    ordinals: tuple[str, ...]
    part_pattern: str


def spec() -> CitationSpec:
    return CitationSpec(pattern=PATTERN, ordinals=ORDINALS,
                        part_pattern=_PART.pattern)
