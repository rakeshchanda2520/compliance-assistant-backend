"""
The shape a model must answer in, when structured output is on.

Why this replaces regex citation scanning: a model writes "§8(5)", "section
8(5)", "Section 8, sub-section (5)", "s. 8(5)" and "the fifth sub-section of
section 8" for the same provision, and `citations.RE_CITATION` has to
anticipate every one. Miss a format and verification silently reports zero
citations — the answer looks unsourced when it was fully sourced. Ask for the
node id instead and verification becomes set membership, which cannot be
fooled by phrasing.

FIELD ORDER IS LOad-BEARING. `answer` is declared first because the streaming
extractor (streaming.py) reads it out of the JSON prefix as it arrives. Move
it and the answer stops streaming — it would arrive in one lump at the end.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CitationRef(BaseModel):
    """One provision the answer relies on.

    `node_id` must come from the retrieved context. The prompt says so, and
    `citations.check_structured` verifies it against what was actually
    retrieved rather than trusting the claim — a model can still emit an id it
    was never shown, and that case is exactly what `out_of_context` labels.
    """
    node_id: str = Field(
        description="Exact id from the provided context, e.g. 's-8-5', "
                    "'r-6-1', 'pen-1'. Never invent one.")
    quote: str = Field(
        default="",
        description="A short exact substring of that provision's text.")


class ComplianceAnswer(BaseModel):
    # FIRST — see the module docstring. streaming.py depends on this.
    answer: str = Field(
        description="The answer in plain English, citing provisions by their "
                    "human-readable label (section 8(5), rule 6, Schedule "
                    "entry 1).")
    citations: list[CitationRef] = Field(
        default_factory=list,
        description="Every provision relied on, by node id.")
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="low when the retrieved context does not fully answer "
                    "the question.")


def json_schema() -> dict:
    """The schema as strict-mode providers actually require it.

    Pydantic's `model_json_schema()` is valid JSON Schema but is REJECTED by
    OpenAI-compatible strict mode, which imposes two extra rules:

      1. `additionalProperties: false` on EVERY object, not just the root —
         including each `$defs` entry (here, CitationRef).
      2. Every property listed in `required`. Strict mode has no concept of
         an optional field, so a Pydantic default does not make one.

    Getting this wrong fails loudly rather than silently, which is the one
    mercy: OpenRouter returned

        Invalid schema for response_format 'compliance_answer':
        In context=(), 'additionalProperties' is required to be supplied
        and to be false.

    Defaults still matter on the Python side — `streaming.finish()` fills in
    anything a non-compliant model omits — so requiring every field here
    costs nothing and buys constrained decoding.
    """
    return _strict(ComplianceAnswer.model_json_schema())


def _strict(node):
    """Recursively apply strict-mode rules to every object in the schema."""
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {key: _strict(value) for key, value in node.items()}
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out


# Appended to the system prompt when structured output is on. Deliberately
# short: the schema itself carries the field descriptions, and restating them
# in prose invites the model to treat them as advisory.
STRUCTURED_INSTRUCTION = """
Reply with a single JSON object matching the provided schema, and nothing else.

- "answer" comes first and contains your full prose answer.
- "citations" lists the node_id of every provision you relied on. Use ONLY ids
  that appear in the context you were given, exactly as written there. If you
  cannot support a statement with a provided provision, do not make it.
- Set "confidence" to "low" when the context does not fully answer the question.
"""
