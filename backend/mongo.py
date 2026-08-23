"""
Every question asked, and its full answer — durable across deployments.

`audit.py` already records every request in full to local disk, and remains
the forensic record. This module exists for the same reason `usage.py` used
to: local disk is the wrong place for "who is using this" on a hosted
platform, since the container's filesystem is ephemeral and per-instance.

It replaces `usage.py` rather than sitting alongside it, and stores strictly
more: the full answer, citations and retrieval trace, not just that a
question was asked — that is what lets a signed-in user reopen a past
answer later (`/api/history`).

This content deliberately lives outside Supabase. Supabase Postgres stays
scoped to identity and login history (`auth.users`, `profiles`,
`login_events` — see `supabase_setup.sql`); no question or answer text is
stored there. The two stores correlate on two shared keys, both minted by
Supabase and never by this service: `user_id` is always `auth.users.id`,
and `session_id` is always `auth.sessions.id` (the `session_id` claim in the
verified JWT — see `auth.Identity`), the same one sign-in that produced a
`login_events` row. Langfuse traces (`observability.trace()`) carry the
identical pair, so a user or a single sign-in session can be followed across
Supabase, MongoDB and Langfuse without any of them talking to each other.

`pymongo` (sync driver), not `motor`. `finish()` in `app.py` already calls
this from a plain sync function inside an async generator — matching
`usage.record()`'s precedent exactly, a 5-second cap is the mitigation for a
slow cluster, not full non-blocking I/O, and that trade-off is unchanged
here.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from . import config

log = logging.getLogger(__name__)

TIMEOUT_MS = 5000     # analytics must never hold a request open

_client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=TIMEOUT_MS,
                      connectTimeoutMS=TIMEOUT_MS)
_collection = _client[config.MONGODB_DB]["interactions"]
# Conversation turns (V2 phase 5). A separate collection rather than a field
# on `interactions`: a turn is written on every outcome including abstains,
# and `interactions` is the record of ANSWERS. Conflating them would make
# `/api/history` show abstained turns as reopenable answers.
_conversations = _client[config.MONGODB_DB]["conversations"]

try:
    # Idempotent — safe to run on every startup, not just the first one. But
    # NOT best-effort by accident: `MongoClient(...)` above never actually
    # connects (the driver is lazy), so an unreachable cluster surfaces for
    # the first time right here, at module import — inside `lifespan()`,
    # which would take the whole app down on startup if this were allowed to
    # raise. That is exactly the failure mode `record()` below is designed to
    # never have; index creation must honour the same contract.
    _collection.create_index([("user_id", 1), ("created_at", -1)])
    _collection.create_index([("session_id", 1), ("created_at", -1)])
    _collection.create_index([("created_at", -1)])
    _conversations.create_index([("conversation_id", 1), ("created_at", 1)])
    _conversations.create_index([("user_id", 1), ("created_at", -1)])
except PyMongoError as exc:
    # The app still starts and still answers questions — it just runs
    # without these indexes until the next restart finds Mongo reachable.
    # `record()`/`list_for_user()` work either way; the indexes only affect
    # how fast `list_for_user()` is once there is real volume to page through.
    log.warning("could not create indexes on startup: %s", type(exc).__name__)


def record(event: dict) -> None:
    """Insert one interaction. Never raises.

    Analytics must never be able to fail an answer the user is waiting on —
    if Mongo is unreachable, the request still completes and the gap is a
    logged warning, exactly like the old `usage.record()`.
    """
    try:
        _collection.insert_one({**event, "created_at": datetime.now(timezone.utc)})
    except PyMongoError as exc:
        log.warning("interaction not recorded: %s", type(exc).__name__)
    except Exception as exc:                      # noqa: BLE001
        log.warning("interaction not recorded: %s", type(exc).__name__)


def list_for_user(user_id: str, limit: int, before: str | None) -> list[dict]:
    """A signed-in user's own past answered questions, newest first.

    Unlike `record()`, this RAISES on failure rather than swallowing it — the
    caller is `/api/history`, a request a real user is waiting on. A broken
    read must surface as a real error, never render as "you have no
    history," which would be actively misleading.

    Filtered to `outcome == "answered"`: an abstained or failed request has
    no answer to replay, so there is nothing to show in a history panel for
    it (it is still recorded, just not listable here).
    """
    query: dict = {"user_id": user_id, "outcome": "answered"}
    if before:
        query["created_at"] = {"$lt": datetime.fromisoformat(before)}
    docs = (_collection.find(query)
            .sort("created_at", -1)
            .limit(limit))
    return [_public(d) for d in docs]


def record_turn(conversation_id: str, user_id: str, question: str,
                provision_ids: list[str], intent: str) -> None:
    """One conversation turn. Never raises, same contract as `record()`.

    WHAT IS STORED IS THE POINT. The question, the provisions it reached, and
    the intent — and deliberately NOT the model's answer.

    Carrying a prior answer forward is how a system defends a hallucination
    three turns later: the model reads its own earlier mistake as established
    context and elaborates on it. Questions and provision ids give continuity
    of SUBJECT without continuity of ERROR. A follow-up re-derives its answer
    from the statute every time.
    """
    if not conversation_id:
        return
    try:
        _conversations.insert_one({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "question": question,
            "provision_ids": list(provision_ids),
            "intent": intent,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as exc:                          # noqa: BLE001
        log.warning("conversation turn not recorded: %s", type(exc).__name__)


def recent_turns(conversation_id: str, user_id: str,
                 limit: int = 3) -> list[dict]:
    """The last few turns of one conversation, oldest first. Never raises.

    Scoped to `user_id` as well as `conversation_id`: a conversation id is a
    client-supplied opaque string, so without this scope a caller could read
    another user's turns by guessing one. The id alone is not a credential.

    Returns [] on failure rather than raising — losing conversational context
    degrades a follow-up into a standalone question, which is the V1
    behaviour and perfectly serviceable. Failing the request would not be.
    """
    if not conversation_id:
        return []
    try:
        docs = (_conversations
                .find({"conversation_id": conversation_id, "user_id": user_id})
                .sort("created_at", -1)
                .limit(limit))
        return [{"question": d.get("question", ""),
                 "provision_ids": list(d.get("provision_ids") or []),
                 "intent": d.get("intent", "")}
                for d in reversed(list(docs))]
    except Exception as exc:                          # noqa: BLE001
        log.warning("conversation turns unavailable: %s", type(exc).__name__)
        return []


def _public(doc: dict) -> dict:
    """Mongo's `_id` (an ObjectId) is not JSON-serialisable and is not a
    stable id worth exposing anyway — the frontend only ever needs
    `created_at` as its pagination cursor."""
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    doc["created_at"] = doc["created_at"].isoformat()
    return doc
