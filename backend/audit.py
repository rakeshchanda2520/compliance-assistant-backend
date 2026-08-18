"""
Append-only audit trail.

One line per request: what was asked, what was retrieved, what was answered,
how each citation resolved, and which build of the corpus produced it. This
is the record a "your tool told me X last week — was that right?" complaint
gets investigated against, and it is why `build_id` is stamped on every entry.

Append-only by construction: opened in "a" mode, never read-modify-written.
A log that can be silently edited after the fact is not an audit trail.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

# Writes come from an async request handler; serialise them so two concurrent
# answers cannot interleave halves of a line.
_lock = threading.Lock()


class AuditLog:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "audit.jsonl"
        # The log contains full questions and answers. On POSIX keep it
        # owner-only; a compliance question can itself be sensitive.
        if not self.path.exists():
            self.path.touch(mode=0o600, exist_ok=True)
        elif os.name == "posix":
            try:
                self.path.chmod(0o600)
            except OSError:
                log.warning("could not restrict permissions on %s", self.path.name)

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex[:12]

    def write(self, record: dict) -> None:
        """Never raises. Losing an answer because logging failed would be a
        worse outcome than a gap in the log, which the error line records."""
        record.setdefault("ts", time.time())
        try:
            line = json.dumps(record, ensure_ascii=False)
            with _lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:                      # noqa: BLE001
            log.exception("audit write failed for request %s",
                          record.get("request_id", "?"))
