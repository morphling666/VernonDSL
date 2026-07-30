from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from ..source_identity import source_file_digest
from .request import FrontendCompileRequest, FrontendCompileResult


@dataclass(frozen=True)
class _CacheEntry:
    result: FrontendCompileResult
    dependencies: tuple[tuple[Path, str], ...]


class FrontendCache:
    def __init__(self) -> None:
        self._entries: dict[FrontendCompileRequest, _CacheEntry] = {}
        self._lock = RLock()

    def get(self, request: FrontendCompileRequest) -> FrontendCompileResult | None:
        with self._lock:
            entry = self._entries.get(request)
        if entry is None:
            return None
        if all(source_file_digest(path) == digest for path, digest in entry.dependencies):
            return entry.result
        with self._lock:
            if self._entries.get(request) is entry:
                self._entries.pop(request)
        return None

    def put(
        self,
        request: FrontendCompileRequest,
        result: FrontendCompileResult,
        dependencies: tuple[tuple[Path, str], ...],
    ) -> None:
        with self._lock:
            self._entries[request] = _CacheEntry(result, dependencies)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


frontend_cache = FrontendCache()
