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
stored there. The two stores correlate on one shared key: `user_id` is
always `auth.users.id`, the same UUID from the verified JWT's `sub` claim in
both places — there is only one place a user id is ever minted, so nothing
has to be done to keep them in sync.

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

try:
    # Idempotent — safe to run on every startup, not just the first one. But
    # NOT best-effort by accident: `MongoClient(...)` above never actually
    # connects (the driver is lazy), so an unreachable cluster surfaces for
    # the first time right here, at module import — inside `lifespan()`,
    # which would take the whole app down on startup if this were allowed to
    # raise. That is exactly the failure mode `record()` below is designed to
    # never have; index creation must honour the same contract.
    _collection.create_index([("user_id", 1), ("created_at", -1)])
    _collection.create_index([("created_at", -1)])
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


def _public(doc: dict) -> dict:
    """Mongo's `_id` (an ObjectId) is not JSON-serialisable and is not a
    stable id worth exposing anyway — the frontend only ever needs
    `created_at` as its pagination cursor."""
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    doc["created_at"] = doc["created_at"].isoformat()
    return doc
