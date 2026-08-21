"""
Streaming a JSON answer field. The one real tension in structured output.

`{"answer": "...", "citations": [...]}` is not valid JSON until the closing
brace, so the obvious implementation cannot stream at all — the user watches a
spinner for thirty seconds and then the whole answer appears. That is a
straight downgrade from V1, which streamed tokens as the model produced them.

The fix does not need a streaming JSON parser. `answer` is the FIRST field in
the schema (schema.py enforces this), so the prose is simply the text between
`"answer":"` and the next unescaped quote. A small state machine finds the
opening, emits everything after it as it arrives, and stops at the close.
Citations and confidence arrive afterwards and are parsed normally from the
complete buffer.

Two things this has to get right, both of which are silent when wrong:

  * JSON escapes. `\\"` inside the answer must not terminate it, and `\\n`
    must reach the client as a newline, not as two characters.
  * Non-compliance. A 3B model will sometimes emit prose with no JSON at all.
    That is not an error to fail on — `finish()` falls back to treating the
    whole buffer as the answer, and the request degrades to the V1 path
    instead of losing an answer the model actually produced.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# Tolerates whitespace and either quote style around the key, and a ```json
# fence before it, which several models add despite being told not to.
RE_ANSWER_START = re.compile(r'"answer"\s*:\s*"')

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}


class AnswerStream:
    """Feed raw model fragments in, get answer text out.

    Usage:
        stream = AnswerStream()
        for fragment in llm.stream(...):
            for text in stream.feed(fragment):
                yield sse("token", {"t": text})
        parsed = stream.finish()
    """

    def __init__(self) -> None:
        self.raw: list[str] = []          # everything the model sent
        self._pending = ""                # not yet scanned for the opener
        self._started = False             # inside the answer string
        self._closed = False              # answer string ended
        self._escape = False              # previous char was a backslash
        self._unicode = ""                # collecting a \uXXXX escape
        self._answer: list[str] = []

    # -- streaming ---------------------------------------------------------- #

    def feed(self, fragment: str) -> list[str]:
        """Consume one model fragment; return answer text to emit (possibly
        empty). Never raises — malformed input is handled at finish()."""
        self.raw.append(fragment)

        if self._closed:
            return []

        if not self._started:
            self._pending += fragment
            match = RE_ANSWER_START.search(self._pending)
            if not match:
                # Keep only enough tail to match an opener split across
                # fragments. 32 chars comfortably exceeds '"answer" : "'.
                if len(self._pending) > 512:
                    self._pending = self._pending[-32:]
                return []
            self._started = True
            body = self._pending[match.end():]
            self._pending = ""
            return self._consume(body)

        return self._consume(fragment)

    def _consume(self, text: str) -> list[str]:
        """Walk characters, honouring JSON string escapes, until the closing
        quote. Returns the decoded pieces to emit."""
        out: list[str] = []
        buffer: list[str] = []

        for char in text:
            if self._unicode:
                self._unicode += char
                if len(self._unicode) == 5:          # 'u' + 4 hex digits
                    try:
                        buffer.append(chr(int(self._unicode[1:], 16)))
                    except ValueError:
                        buffer.append(self._unicode)
                    self._unicode = ""
                continue

            if self._escape:
                self._escape = False
                if char == "u":
                    self._unicode = "u"
                else:
                    buffer.append(_ESCAPES.get(char, char))
                continue

            if char == "\\":
                self._escape = True
                continue

            if char == '"':
                self._closed = True
                break

            buffer.append(char)

        if buffer:
            piece = "".join(buffer)
            self._answer.append(piece)
            out.append(piece)
        return out

    # -- completion --------------------------------------------------------- #

    @property
    def answer(self) -> str:
        return "".join(self._answer)

    def finish(self) -> dict:
        """Parse the complete buffer.

        Returns {"answer", "citations", "confidence", "structured"}.
        `structured` is False when the model did not comply — the caller then
        treats the output as free text and runs the V1 regex citation path,
        which is a degraded answer rather than no answer.
        """
        raw = "".join(self.raw)
        payload = _extract_json(raw)

        if payload is None:
            if self._started:
                # The answer field opened but the object never closed — a
                # truncated stream. The prose we captured is still good.
                log.warning("structured output truncated; using captured answer")
                return {"answer": self.answer, "citations": [],
                        "confidence": "low", "structured": False}
            log.warning("model did not emit JSON; falling back to free text")
            return {"answer": raw.strip(), "citations": [],
                    "confidence": "low", "structured": False}

        citations = []
        for item in payload.get("citations") or []:
            if isinstance(item, dict) and item.get("node_id"):
                citations.append({"node_id": str(item["node_id"]),
                                  "quote": str(item.get("quote", ""))})
            elif isinstance(item, str):
                # Some models flatten the objects to bare ids.
                citations.append({"node_id": item, "quote": ""})

        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = self.answer or raw.strip()

        return {
            "answer": answer,
            "citations": citations,
            "confidence": payload.get("confidence") or "medium",
            "structured": True,
        }


def _extract_json(text: str) -> dict | None:
    """The outermost JSON object in `text`, or None.

    Brace-counting rather than a regex because the answer field routinely
    contains braces, and string-aware because it routinely contains quotes.
    Handles a ```json fence and any prose a model prepends.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
