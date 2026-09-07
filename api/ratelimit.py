"""
Per-user rate limiting. In memory, per process.

Does NOT survive a restart and does NOT coordinate across instances — so N
replicas mean N times the limit. Deliberate: Redis is paid infrastructure this
does not need yet, and the gap it closes ("sign-in makes abuse attributable,
not impossible") is closed either way for a single instance.

**The eval identity gets its own limit.** A 56-question baseline cannot be
captured under 30 requests an hour — that was hit for real, and the run came
back full of 429s. The alternative, a test-only auth bypass, is exactly the
kind of hole that survives into production, so it is a raised LIMIT for a
named user, not a way around the gate.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import config


class RateLimiter:
    def __init__(self, *, limit: int = 0, window: int = 0,
                 eval_ids: tuple[str, ...] = (), eval_limit: int = 0) -> None:
        self.limit = limit or config.RATE_LIMIT
        self.window = window or config.RATE_WINDOW
        self.eval_ids = eval_ids or config.EVAL_USER_IDS
        self.eval_limit = eval_limit or config.EVAL_RATE_LIMIT
        self._hits: dict[str, deque] = defaultdict(deque)

    def limit_for(self, user_id: str) -> int:
        return self.eval_limit if user_id in self.eval_ids else self.limit

    def allow(self, user_id: str) -> tuple[bool, int, int]:
        """(allowed, remaining, retry_after_seconds)."""
        cap = self.limit_for(user_id)
        if cap <= 0:
            return True, -1, 0

        now = time.monotonic()
        hits = self._hits[user_id]
        while hits and now - hits[0] > self.window:
            hits.popleft()

        if len(hits) >= cap:
            retry = int(self.window - (now - hits[0])) + 1
            return False, 0, max(retry, 1)

        hits.append(now)
        return True, cap - len(hits), 0

    def sweep(self) -> None:
        """Drop users with no hits left in the window, so memory is bounded
        by ACTIVE users rather than by every user ever seen."""
        now = time.monotonic()
        for user_id in list(self._hits):
            hits = self._hits[user_id]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if not hits:
                del self._hits[user_id]

    @property
    def tracked_users(self) -> int:
        return len(self._hits)
