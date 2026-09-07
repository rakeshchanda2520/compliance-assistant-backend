"""
Where questions and answers go.

Identity and content live in different databases, deliberately. Supabase
Postgres holds who signed in and when — nothing else, written by a database
trigger this service never calls. Every question and answer goes to MongoDB.

They correlate on ONE shared key: `auth.users.id`, taken from the verified
token's `sub` claim and written identically to both. There is no live join
across two engines and there does not need to be, because there is only one
place a user id is ever minted.

Writes and reads have DIFFERENT failure contracts, and the difference matters:

    record()        never raises. An unreachable cluster must not fail an
                    answer the user already received.
    history()       raises. A broken history read must never render as
                    "you have no history" — that is a silent data-loss lie.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

_client = None
_interactions = None
_conversations = None


def connect() -> bool:
    """True when a store is available. Called once at startup."""
    global _client, _interactions, _conversations
    if not config.MONGODB_URI:
        log.warning("no MONGODB_URI — questions and answers will not be "
                    "recorded, and history will be empty")
        return False
    try:
        from pymongo import MongoClient
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=4000)
        _client.admin.command("ping")
        db = _client[config.MONGODB_DB]
        _interactions = db["interactions"]
        _conversations = db["conversations"]
        _interactions.create_index([("user_id", 1), ("created_at", -1)])
        _conversations.create_index([("conversation_id", 1), ("created_at", 1)])
        _conversations.create_index([("user_id", 1), ("created_at", -1)])
        log.info("store ready: %s", config.MONGODB_DB)
        return True
    except Exception as exc:                               # noqa: BLE001
        log.warning("store unavailable (%s) — answers will still be served, "
                    "but nothing will be recorded", exc)
        _client = _interactions = _conversations = None
        return False


def available() -> bool:
    return _interactions is not None


def record(event: dict) -> None:
    """One interaction. NEVER raises."""
    if _interactions is None:
        return
    try:
        _interactions.insert_one({**event,
                                  "created_at": datetime.now(timezone.utc)})
    except Exception as exc:                               # noqa: BLE001
        log.warning("interaction not recorded: %s", type(exc).__name__)


def record_turn(conversation_id: str, user_id: str, question: str,
                provision_ids: list[str], intent: str) -> None:
    """One conversation turn. NEVER raises.

    WHAT IS STORED IS THE POINT: the question, the provisions it reached, and
    the intent — and deliberately NOT the answer.

    Carrying a prior answer forward is how a system defends a hallucination
    three turns later: the model reads its own earlier mistake as established
    context and elaborates on it. Questions and provision ids give continuity
    of SUBJECT without continuity of ERROR, so every follow-up re-derives its
    answer from the statute.
    """
    if _conversations is None or not conversation_id:
        return
    try:
        _conversations.insert_one({
            "conversation_id": conversation_id, "user_id": user_id,
            "question": question, "provision_ids": list(provision_ids),
            "intent": intent, "created_at": datetime.now(timezone.utc)})
    except Exception as exc:                               # noqa: BLE001
        log.warning("conversation turn not recorded: %s", type(exc).__name__)


def prior_provisions(conversation_id: str, user_id: str, limit: int) -> tuple[str, ...]:
    """Provision ids from recent turns, oldest first. NEVER raises.

    Scoped to `user_id` as well as `conversation_id`: a conversation id is a
    client-supplied opaque string, so without that scope a caller could read
    another user's turns by guessing one. The id alone is not a credential.
    """
    if _conversations is None or not conversation_id:
        return ()
    try:
        docs = (_conversations
                .find({"conversation_id": conversation_id, "user_id": user_id})
                .sort("created_at", -1).limit(limit))
        out: list[str] = []
        for doc in reversed(list(docs)):
            out.extend(doc.get("provision_ids") or ())
        return tuple(dict.fromkeys(out))
    except Exception as exc:                               # noqa: BLE001
        log.warning("prior turns unavailable: %s", type(exc).__name__)
        return ()


def history(user_id: str, limit: int, before: str | None = None) -> list[dict]:
    """A user's own past answers. RAISES on failure — see the module docstring."""
    if _interactions is None:
        raise RuntimeError("no history store is configured")
    query: dict = {"user_id": user_id, "outcome": "answered"}
    if before:
        query["created_at"] = {"$lt": datetime.fromisoformat(before)}
    docs = _interactions.find(query).sort("created_at", -1).limit(limit)
    return [_public(d) for d in docs]


def _public(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    created = doc.get("created_at")
    doc["created_at"] = created.isoformat() if created else ""
    return doc
