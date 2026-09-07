"""
Staged commencement — is a cited provision actually in force?

The Rules commence in stages: rules 1, 2 and 17-21 on publication, rule 4 after
a year, the rest after eighteen months. "Is rule 4 in force?" is a DATE
COMPARISON, not a reading-comprehension question to put to a language model —
and putting it to one is precisely the class of question that produced this
project's recorded errors.

The flag this produces is AMBER, never red. Red means "the answer states
something the evidence does not support" — a defect in the answer. A provision
that has not commenced is the opposite: the answer is correct, the law simply
is not enforceable yet. Painting that red tells a reader their right answer is
wrong, which is the more expensive of the two mistakes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Commencement:
    dates: dict[str, date] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    published_on: date | None = None

    def _date_for(self, node_id: str) -> date | None:
        """Walks UP the id hierarchy: `r-6-1-a` inherits `r-6`'s date.

        The Rules commence whole rules, not individual clauses, and listing
        every descendant in the YAML would be a maintenance trap that silently
        goes stale the next time a clause is added.
        """
        parts = node_id.split("-")
        for i in range(len(parts), 0, -1):
            key = "-".join(parts[:i])
            if key in self.dates:
                return self.dates[key]
        return None

    def reason_for(self, node_id: str) -> str:
        parts = node_id.split("-")
        for i in range(len(parts), 0, -1):
            key = "-".join(parts[:i])
            if key in self.reasons:
                return self.reasons[key]
        return ""

    def in_force_on(self, node_id: str, when: date) -> bool:
        """In force unless a commencement date says otherwise."""
        start = self._date_for(node_id)
        return start is None or when >= start

    def not_yet_in_force(self, node_ids, when: date) -> list[str]:
        return [n for n in node_ids if not self.in_force_on(n, when)]

    def annotate(self, node_id: str, when: date) -> dict:
        start = self._date_for(node_id)
        live = start is None or when >= start
        return {"in_force": live,
                "in_force_from": start.isoformat() if start else None}

    def pending_note(self, node_ids: list[str], when: date, label_for) -> str:
        """Why a provision is flagged, in words a reader can act on.

        The old note said only "<X> has not commenced", which states the
        conclusion and withholds everything needed to use it: what
        "commenced" means, when it changes, and whether the answer above is
        therefore wrong. It is not wrong — the provision IS the law on the
        point, it simply is not enforceable yet — and saying so is the entire
        value of the flag.
        """
        parts = []
        for node_id in node_ids:
            start = self._date_for(node_id)
            when_txt = (f"takes effect on {start.strftime('%d %B %Y')}"
                        if start else "has no commencement date on record")
            parts.append(f"{label_for(node_id)} {when_txt}")

        # The reason is attached only for a SINGLE provision: rule 4 commences
        # at one year and rule 15 at eighteen months, so one reason cannot
        # describe a list of several without being wrong about some of them.
        reason = self.reason_for(node_ids[0]) if len(node_ids) == 1 else ""
        detail = f" ({reason})" if reason else ""
        one = len(node_ids) == 1
        subj, obj, verb = ("it", "it", "is") if one else ("they", "them", "are")
        return (f"Quoted because {subj} {verb} the law on this point, but not "
                f"enforceable yet: {'; '.join(parts)}{detail}. As of "
                f"{when.strftime('%d %B %Y')} you cannot be penalised under "
                f"{obj} — though {subj} {verb} what you will be held to once "
                f"in force.")


def load(path: Path) -> Commencement:
    """Missing or malformed data means "treat everything as in force".

    Deliberate: a temporal flag is an enhancement. Failing to start because a
    supplementary YAML is absent would trade a missing caveat for a dead
    service.
    """
    if not path.is_file():
        log.warning("no commencement data at %s — every provision will be "
                    "treated as in force", path.name)
        return Commencement()

    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                                      # noqa: BLE001
        log.exception("%s is malformed; treating all provisions as in force",
                      path.name)
        return Commencement()

    dates: dict[str, date] = {}
    reasons: dict[str, str] = {}
    for group in (raw.get("rules") or {}).values():
        group = group or {}
        start = _as_date(group.get("from"))
        if start is None:
            continue
        for unit in group.get("units") or ():
            dates[str(unit)] = start
            if group.get("reason"):
                reasons[str(unit)] = str(group["reason"])

    return Commencement(dates, reasons, _as_date(raw.get("published_on")))


def _as_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def resolve_as_of(requested: str | None, default: str) -> date:
    """The date an answer is stated as of. Defaults to today."""
    for candidate in (requested, default):
        if candidate and candidate != "today":
            parsed = _as_date(candidate)
            if parsed:
                return parsed
    return date.today()
