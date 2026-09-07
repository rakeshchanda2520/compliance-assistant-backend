"""
The structured-output schema, and the streaming extractor that keeps it fluid.

Structured output replaces regex citation scanning with SET MEMBERSHIP: the
model emits node ids directly, and a claimed id either was or was not in the
retrieved context. `check_text` has to anticipate every way a model might spell
a citation — miss one format and a fully sourced answer looks unsourced. Here
there is no format to miss.

The cost is that the model now returns JSON, and JSON does not stream as prose.
`AnswerStream` solves that: `answer` is the FIRST field in the schema, so the
extractor can lift it out of the JSON prefix character by character and emit it
token by token exactly as before. Streaming is not sacrificed for verification.
"""
from __future__ import annotations

import json
from typing import Any

NAME = "compliance_answer"


def json_schema() -> dict:
    """OpenAI strict-mode compliant.

    `additionalProperties: false` must be present on EVERY nested object and
    every property must appear in `required` — a schema that sets it only at
    the root is rejected with a 400 that names neither the offending object
    nor the rule. Written out here rather than generated from Pydantic for
    exactly that reason: the transform is invisible and easy to get wrong.
    """
    return {
        "name": NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            # `answer` FIRST so it streams before anything else arrives.
            "required": ["answer", "claims", "confidence"],
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The full prose answer, formatted with the "
                                   "headings from the system prompt.",
                },
                "claims": {
                    "type": "array",
                    "description": "One entry per assertion about the law.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "cites", "quote"],
                        "properties": {
                            "text": {"type": "string"},
                            "cites": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Provision ids as supplied — "
                                               "s-8-5, r-6-1, pen-1.",
                            },
                            "quote": {
                                "type": "string",
                                "description": "Exact words from the cited "
                                               "provision, or empty.",
                            },
                        },
                    },
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
        },
    }


class AnswerStream:
    """Extracts `answer` from a JSON prefix as it arrives.

    A hand-written scanner rather than incremental JSON parsing, because only
    one field needs to stream and the rest can be parsed once at the end. It
    must survive every escape a legal answer contains — quotation marks around
    statutory text, newlines between headings, backslashes, and the rupee sign.
    """

    __slots__ = ("_buffer", "_emitted", "_in_answer", "_done", "_escape")

    def __init__(self) -> None:
        self._buffer = ""
        self._emitted = ""
        self._in_answer = False
        self._done = False
        self._escape = False

    def feed(self, fragment: str) -> list[str]:
        """Return the pieces of `answer` newly available."""
        self._buffer += fragment
        if self._done:
            return []

        if not self._in_answer:
            marker = self._buffer.find('"answer"')
            if marker < 0:
                return []
            colon = self._buffer.find(":", marker)
            quote = self._buffer.find('"', colon + 1) if colon >= 0 else -1
            if quote < 0:
                return []
            self._in_answer = True
            self._buffer = self._buffer[quote + 1:]

        out: list[str] = []
        consumed = 0
        for i, ch in enumerate(self._buffer):
            if self._escape:
                self._escape = False
                consumed = i + 1
                continue
            if ch == "\\":
                self._escape = True
                continue
            if ch == '"':
                # End of the answer string. Decode what we have and stop.
                raw = self._buffer[:i]
                consumed = i + 1
                self._done = True
                break
            consumed = i + 1
        else:
            raw = self._buffer[:consumed]

        if raw:
            try:
                decoded = json.loads(f'"{raw}"')
            except json.JSONDecodeError:
                # A trailing partial escape — wait for the rest rather than
                # emitting a broken character.
                return out
            new = decoded[len(self._emitted):]
            if new:
                self._emitted = decoded
                out.append(new)
        if self._done:
            self._buffer = self._buffer[consumed:]
        return out

    @property
    def answer(self) -> str:
        return self._emitted

    def finish(self, whole: str) -> dict[str, Any]:
        """Parse the complete response once the stream has ended.

        `structured` false means the model ignored the schema. The caller then
        degrades to the regex citation path rather than losing an answer that
        was produced — a worse answer, not a failed one.
        """
        parsed = _extract_json(whole)
        if not isinstance(parsed, dict) or "answer" not in parsed:
            return {"answer": self._emitted or whole, "claims": [],
                    "confidence": "", "structured": False}
        claims = parsed.get("claims")
        return {
            "answer": parsed.get("answer") or self._emitted or whole,
            "claims": claims if isinstance(claims, list) else [],
            "confidence": parsed.get("confidence", ""),
            "structured": True,
        }


def _extract_json(text: str) -> Any:
    """The first JSON object in `text`, tolerating a code fence or preamble.

    Models wrap JSON in ```json fences even when told not to, and a leading
    sentence before the object is common. Neither is worth failing over.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
