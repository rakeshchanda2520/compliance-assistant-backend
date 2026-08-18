"""
The searchable chunk, and the tokenizer both sides must share.

Chunks follow the Act's own boundaries — section, sub-section, definition,
Schedule row — not a fixed token window. The drafters already divided this
document into citable units; splitting on a window instead would cut clause
lists in half and discard the structure the parser just recovered.

`kg_build` writes `chunks.json`; the backend loads it. It is a derived search
artifact, not graph data: the graph lives in Neo4j, and a BM25 corpus with a
generated plain-language layer is not something a graph database should hold.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("english")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit.

    `verbatim` is the Act's exact words and is the only field an answer may
    quote. `plain_english` and `questions` are generated at build time to
    bridge the vocabulary gap and are indexed but never quoted or cited.
    """
    id: str
    node_id: str
    kind: str
    label: str
    verbatim: str
    header: str = ""
    headnote: str = ""
    chapter: str = ""
    page: int = 0
    plain_english: str = ""
    questions: list[str] = field(default_factory=list)

    @property
    def document(self) -> str:
        """What BM25 actually scores: the Act's words plus the generated
        layer, so a user's phrasing can match even when it shares nothing
        with the statute."""
        return " ".join(filter(None, [
            self.label, self.headnote, self.header, self.verbatim,
            self.plain_english, " ".join(self.questions),
        ]))

    def to_dict(self) -> dict:
        return {
            "id": self.id, "node_id": self.node_id, "kind": self.kind,
            "label": self.label, "verbatim": self.verbatim, "header": self.header,
            "headnote": self.headnote, "chapter": self.chapter, "page": self.page,
            "plain_english": self.plain_english, "questions": list(self.questions),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Chunk":
        return cls(
            id=raw["id"], node_id=raw.get("node_id", raw["id"]),
            kind=raw.get("kind", ""), label=raw.get("label", raw["id"]),
            verbatim=raw.get("verbatim", ""), header=raw.get("header", ""),
            headnote=raw.get("headnote", ""), chapter=raw.get("chapter", ""),
            page=int(raw.get("page", 0) or 0),
            plain_english=raw.get("plain_english", ""),
            questions=list(raw.get("questions", [])),
        )


def stem(token: str) -> str | None:
    """Snowball stem, or None if the token should be left alone.

    The isalpha guard is the important part: "8(5)", "250" and "2023" are
    exactly the tokens a legal question turns on, and they must survive
    untouched. Hand-rolled suffix stripping got this wrong — any rule
    conservative enough to protect "data" also refused to relate "died" and
    "dies"; Snowball maps both to "die".
    """
    if not token.isalpha() or len(token) <= 3:
        return None
    stemmed = _STEMMER.stemWord(token)
    return stemmed if stemmed != token else None


def tokenize(text: str) -> list[str]:
    """Index every word under both its surface form and its stem.

    Keeping both means an exact match still scores highest, while "someone
    died" can still reach text written as "in the event of death". Section
    numbers and rupee figures pass through intact.
    """
    words = re.findall(r"[a-z0-9]+(?:\([a-z0-9]+\))?", text.lower())
    return words + [s for w in words if (s := stem(w))]


def load_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise RuntimeError(
            f"search index missing at {path.name}. Run: python -m kg_build")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.from_dict(c) for c in raw]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8")
