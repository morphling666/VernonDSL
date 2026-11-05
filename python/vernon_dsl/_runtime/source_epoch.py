"""Process-wide source epoch (specs/performance.md section 3.1).

A call compares one integer instead of checking files. A watcher thread polls the watched sources and bumps the epoch
when one changes; `VERNON_DSL_WATCH_SOURCES=0` disables it, as in a release, where the epoch only moves on `bump`.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from pathlib import Path

_POLL_SECONDS = 0.1


class _SourceEpoch:
    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()
        self._stamps: dict[Path, tuple[int, int] | None] = {}
        self._thread: threading.Thread | None = None
        self._enabled = os.environ.get("VERNON_DSL_WATCH_SOURCES", "1") != "0"

    @staticmethod
    def _stamp(path: Path) -> tuple[int, int] | None:
        try:
            status = path.stat()
        except OSError:
            return None
        return (status.st_mtime_ns, status.st_size)

    def bump(self) -> None:
        with self._lock:
            self.value += 1

    def watch(self, paths: Iterable[Path]) -> int:
        """Starts watching `paths` and returns the epoch a caller validated them at."""
        with self._lock:
            for path in paths:
                if path not in self._stamps:
                    self._stamps[path] = self._stamp(path)
            if self._enabled and self._thread is None:
                self._thread = threading.Thread(target=self._poll, name="vernon-source-epoch", daemon=True)
                self._thread.start()
            return self.value

    def _poll(self) -> None:
        event = threading.Event()
        while not event.wait(_POLL_SECONDS):
            with self._lock:
                paths = tuple(self._stamps)
            changed = {path: stamp for path in paths if (stamp := self._stamp(path)) != self._stamps.get(path)}
            if changed:
                with self._lock:
                    self._stamps.update(changed)
                    self.value += 1


source_epoch = _SourceEpoch()

__all__ = ["source_epoch"]
