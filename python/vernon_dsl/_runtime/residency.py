from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Residency:
    session_identity: int
    handle: Any


class _SingleResidency:
    """One session-tagged native residency, replaceable by Phase 6 partitioning."""

    def __init__(self) -> None:
        self._current: _Residency | None = None

    @property
    def current(self) -> _Residency | None:
        return self._current

    def handle_for(self, session_identity: int) -> Any | None:
        current = self._current
        if current is None or current.session_identity != session_identity:
            return None
        return current.handle

    def replace(self, session_identity: int, handle: Any) -> Any:
        self._current = _Residency(session_identity, handle)
        return handle

    def restore(self, residency: _Residency | None) -> None:
        self._current = residency

    def clear(self) -> None:
        self._current = None


class _ResourceAccessDomain:
    """Owner-local claim domain preceding every native materialization."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.claims: list[tuple[object, object, str]] = []
        self.session_identity: int | None = None

    def validate_session(self, session_identity: int) -> None:
        if self.session_identity is not None and self.session_identity != session_identity:
            raise RuntimeError("resource is already borrowed by another RuntimeSession")

    def add(self, token: object, resource: object, access: str, session_identity: int) -> None:
        self.validate_session(session_identity)
        self.session_identity = session_identity
        self.claims.append((token, resource, access))

    def release(self, token: object, *, writes_only: bool = False) -> None:
        self.claims[:] = [
            claim for claim in self.claims if claim[0] is not token or (writes_only and claim[2] == "read")
        ]
        if not self.claims:
            self.session_identity = None

    def require_host_mutation(self, resource_name: str) -> None:
        with self.lock:
            if self.claims:
                raise RuntimeError(f"host mutation is forbidden while a device dispatch borrows {resource_name}")

    def require_host_read(self, resource_name: str) -> None:
        with self.lock:
            if any(access != "read" for _, _, access in self.claims):
                raise RuntimeError(f"host reads are forbidden while a device dispatch writes {resource_name}")
