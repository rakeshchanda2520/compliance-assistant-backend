"""
Is this provision in force? A date comparison, not a reading-comprehension task.

The Rules commence in three stages (rule 1(2)): some on publication, rule 4
after a year, the rest after eighteen months. V1 stored rule 1's text stating
this and computed nothing from it — so "is rule 4 in force today?" was a
question about prose, answered by a language model, from a provision it had to
locate first. That is three chances to be wrong about a fact that is a date.

`data/commencement.yaml` holds those dates, human-reviewed against the
Gazette. This module joins them to provisions and answers the question
arithmetically.

Deliberately additive: a provision with no entry is treated as in force, which
is the V1 behaviour. A missing or malformed file degrades to "everything is in
force" with a warning rather than failing startup — a temporal annotation is
useful, but it is not worth refusing to answer anything at all over.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


class Commencement:
    """provision id -> the date it comes into force."""

    def __init__(self, dates: dict[str, date], reasons: dict[str, str],
                 published_on: date | None) -> None:
        self.dates = dates
        self.reasons = reasons
        self.published_on = published_on

    def in_force_on(self, node_id: str, when: date) -> bool:
        """A provision is in force unless a commencement date says otherwise.

        Walks up the id hierarchy: r-6-1-a inherits r-6's date, because the
        Rules commence whole rules, not individual clauses, and listing every
        descendant in the YAML would be a maintenance trap.
        """
        start = self._date_for(node_id)
        return start is None or when >= start

    def date_for(self, node_id: str) -> date | None:
        return self._date_for(node_id)

    def reason_for(self, node_id: str) -> str:
        parts = node_id.split("-")
        for i in range(len(parts), 0, -1):
            key = "-".join(parts[:i])
            if key in self.reasons:
                return self.reasons[key]
        return ""

    def _date_for(self, node_id: str) -> date | None:
        parts = node_id.split("-")
        for i in range(len(parts), 0, -1):
            key = "-".join(parts[:i])
            if key in self.dates:
                return self.dates[key]
        return None

    def annotate(self, node_id: str, when: date) -> dict:
        """The shape the SSE `retrieval` event carries per provision."""
        start = self._date_for(node_id)
        live = start is None or when >= start
        return {
            "in_force": live,
            "in_force_from": start.isoformat() if start else None,
            "commencement_note": "" if live else (
                f"not yet in force — commences {start.isoformat()}"
                f"{'; ' + self.reason_for(node_id) if self.reason_for(node_id) else ''}"),
        }

    def pending_note(self, node_ids, when: date, label_for) -> str:
        """A plain-English explanation of why a cited provision is flagged.

        The old note said only "<X> has not commenced", which states the
        conclusion and withholds everything a reader needs to act on it: what
        "commenced" means, when it changes, and whether the answer above is
        therefore wrong. It is not wrong — the provision IS the law on the
        point, it simply is not enforceable yet — and saying so is the whole
        value of the flag.

        `label_for` is passed in rather than imported: citations.py already
        imports graph_store, and importing it back here would be a cycle.
        """
        parts = []
        for node_id in node_ids:
            start = self._date_for(node_id)
            when_txt = (f"takes effect on {start.strftime('%d %B %Y')}"
                        if start else "has no commencement date on record")
            parts.append(f"{label_for(node_id)} {when_txt}")

        # The reason is only appended for a SINGLE provision. Different
        # provisions commence on different schedules (rule 4 at one year,
        # rule 15 at eighteen months), so attaching the first one's reason to
        # a list of several states it of all of them, which is wrong.
        reason = self.reason_for(node_ids[0]) if len(node_ids) == 1 else ""
        detail = f" ({reason})" if reason else ""
        # Subject and object pronouns are separate words. Reusing one for
        # both produced "them are the law on this point".
        one = len(node_ids) == 1
        subj, obj, verb = ("it", "it", "is") if one else ("they", "them", "are")
        return (
            f"Quoted because {subj} {verb} the law on this point, but not "
            f"enforceable yet: {'; '.join(parts)}{detail}. "
            f"As of {when.strftime('%d %B %Y')} you cannot be penalised under "
            f"{obj} — though {subj} {verb} what you will be held to once in "
            "force.")

    def not_yet_in_force(self, node_ids, when: date) -> list[str]:
        return [n for n in node_ids if not self.in_force_on(n, when)]


def load(path: Path) -> Commencement:
    if not path.is_file():
        log.warning("no commencement data at %s — every provision will be "
                    "treated as in force", path.name)
        return Commencement({}, {}, None)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        log.exception("commencement.yaml is malformed; treating all provisions "
                      "as in force")
        return Commencement({}, {}, None)

    dates: dict[str, date] = {}
    reasons: dict[str, str] = {}

    for _group, entry in (raw.get("rules") or {}).items():
        start = _as_date(entry.get("from"))
        if start is None:
            continue
        for unit in entry.get("units") or []:
            dates[unit] = start
            if reason := entry.get("reason"):
                reasons[unit] = reason

    published = _as_date((raw.get("published_on")))
    log.info("commencement: %d provisions carry a date", len(dates))
    return Commencement(dates, reasons, published)


def resolve_as_of(value: str | None, default: str = "today") -> date:
    """Parse the `as_of` query parameter. Falls back to today on anything
    unparseable — an answer as of today is a sensible default, and rejecting
    the request over a malformed optional parameter is not."""
    for candidate in (value, default):
        if not candidate or candidate == "today":
            continue
        try:
            return datetime.strptime(candidate, "%Y-%m-%d").date()
        except ValueError:
            log.warning("unparseable as_of %r; using today", candidate)
            break
    return date.today()


def _as_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None
