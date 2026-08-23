"""
Per-user request cap. In-memory, per-process, and honest about both.

Closes the gap PROJECT.md states plainly: "sign-in makes abuse attributable,
not impossible". Every /api/chat call costs a real model invocation, and a
signed-in user could previously make them without limit.

Deliberately NOT Redis. Redis is paid infrastructure for this deployment, and
a counter that resets on restart still stops the failure mode that actually
matters — one user looping a script through a day's quota. What it does not
survive is a restart or a second instance, and pretending otherwise would be
worse than the limitation itself, so `/api/health` reports which it is.

Keyed on `user.sub` from the verified JWT, never on an IP or a header: those
are attacker-controlled, and the whole point of requiring sign-in is that
this key is not.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

from . import config


class RateLimiter:
    """Sliding window. Not a token bucket: a legal question is not bursty
    traffic, and a window is what the limit is actually described in
    ("30 an hour") — matching the two avoids surprising anyone reading the
    number in the config."""

    def __init__(self, limit: int | None = None, window: int | None = None):
        self.limit = config.RATE_LIMIT if limit is None else limit
        self.window = config.RATE_WINDOW if window is None else window
        self._hits: dict[str, list[float]] = defaultdict(list)
        # FastAPI serves requests on a thread pool; two requests from the
        # same user can land concurrently and both read a stale count.
        self._lock = threading.Lock()

    def allow(self, user_id: str) -> tuple[bool, int, int]:
        """(allowed, remaining, retry_after_seconds).

        Returns the numbers rather than just a bool so the caller can set
        the RateLimit headers a client needs to back off sensibly, instead of
        making it guess from a bare 429.
        """
        if self.limit <= 0:                       # 0 disables the limiter
            return True, -1, 0

        now = time.time()
        with self._lock:
            hits = self._hits[user_id]
            cutoff = now - self.window
            # Prune in place: rebinding the list would orphan the dict entry.
            hits[:] = [t for t in hits if t > cutoff]

            if len(hits) >= self.limit:
                retry = int(hits[0] + self.window - now) + 1
                return False, 0, max(retry, 1)

            hits.append(now)
            return True, self.limit - len(hits), 0

    def sweep(self) -> int:
        """Drop users with no recent requests.

        Without this the dict grows one entry per user forever — small, but
        it is a leak, and a leak in the component whose job is to bound
        resource use would be a poor joke.
        """
        cutoff = time.time() - self.window
        with self._lock:
            stale = [uid for uid, hits in self._hits.items()
                     if not hits or hits[-1] <= cutoff]
            for uid in stale:
                del self._hits[uid]
            return len(stale)

    @property
    def tracked_users(self) -> int:
        with self._lock:
            return len(self._hits)
