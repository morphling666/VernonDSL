from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from itertools import count
from typing import Any


@dataclass
class _Residency:
    session_identity: int
    handle: Any
    version: int = 0


class _ResidencySet:
    """Session-partitioned native residencies that retain their native owners."""

    def __init__(self) -> None:
        self._entries: dict[int, _Residency] = {}

    def get(self, session: Any) -> _Residency | None:
        return self._entries.get(session.identity)

    def replace(self, session: Any, handle: Any, *, version: int = 0) -> _Residency:
        residency = _Residency(session.identity, handle, version)
        self._entries[session.identity] = residency
        return residency

    def latest(self) -> _Residency | None:
        return max(self._entries.values(), key=lambda residency: residency.version, default=None)


@dataclass(frozen=True)
class _BufferRegion:
    ranges: tuple[tuple[int, int], ...]

    def overlaps(self, other: _BufferRegion) -> bool:
        left = 0
        right = 0
        while left < len(self.ranges) and right < len(other.ranges):
            left_begin, left_end = self.ranges[left]
            right_begin, right_end = other.ranges[right]
            if left_end <= right_begin:
                left += 1
            elif right_end <= left_begin:
                right += 1
            else:
                return True
        return False


@dataclass(frozen=True)
class _ImageRegion:
    subresources: frozenset[tuple[str, int, int]]

    def overlaps(self, other: _ImageRegion) -> bool:
        return not self.subresources.isdisjoint(other.subresources)


_ResourceRegion = _BufferRegion | _ImageRegion
_OWNER_IDS = count(1)
_IO_TICKETS = count(1)


@dataclass(frozen=True)
class _AccessClaim:
    token: object
    region: _ResourceRegion
    access: str
    session_identity: int | None


@dataclass(frozen=True)
class _IoReservation:
    ticket: int
    region: _ResourceRegion
    access: str


class _ResourceControlBlock:
    """Owner-local admission, authority, version, and residency state."""

    def __init__(self, coherence: Any) -> None:
        self.owner_id = next(_OWNER_IDS)
        self.lock = threading.RLock()
        self.io_condition = threading.Condition(self.lock)
        self._io_reservations: list[_IoReservation] = []
        self._claims: list[_AccessClaim] = []
        self.version = 1
        self.host_version = 1
        self.residencies = _ResidencySet()
        self.coherence = coherence
        self.poisoned = False

    def reserve_io(self, region: _ResourceRegion, access: str) -> _IoReservation:
        while any(
            region.overlaps(reservation.region) and not (access == reservation.access == "read")
            for reservation in self._io_reservations
        ):
            self.io_condition.wait()
        reservation = _IoReservation(next(_IO_TICKETS), region, access)
        self._io_reservations.append(reservation)
        return reservation

    def release_io(self, reservation: _IoReservation) -> None:
        self._io_reservations.remove(reservation)
        self.io_condition.notify_all()

    def validate_device_claim(
        self,
        region: _ResourceRegion,
        access: str,
        session_identity: int,
    ) -> None:
        if self.poisoned:
            raise RuntimeError("resource contents are unknown after a failed device write")
        for claim in self._claims:
            if claim.session_identity is not None and claim.session_identity != session_identity:
                raise RuntimeError("resource is already borrowed by another RuntimeSession")
            if not region.overlaps(claim.region) or access == claim.access == "read":
                continue
            raise RuntimeError("resource access conflicts with an outstanding borrow")

    def add_device_claim(
        self,
        token: object,
        region: _ResourceRegion,
        access: str,
        session_identity: int,
    ) -> None:
        self._claims.append(_AccessClaim(token, region, access, session_identity))

    def release(self, token: object, *, writes_only: bool = False) -> None:
        self._claims[:] = [
            claim for claim in self._claims if claim.token is not token or (writes_only and claim.access == "read")
        ]

    @contextmanager
    def host_access(
        self,
        region: _ResourceRegion,
        access: str,
        resource_name: str,
    ) -> Iterator[None]:
        token = object()
        with self.lock:
            if access == "read" and self.poisoned:
                raise RuntimeError(f"host reads require recovery of poisoned {resource_name}")
            for claim in self._claims:
                if not region.overlaps(claim.region):
                    continue
                if access == claim.access == "read":
                    continue
                operation = "reads" if access == "read" else "mutation"
                raise RuntimeError(f"host {operation} are forbidden while a device dispatch borrows {resource_name}")
            self._claims.append(_AccessClaim(token, region, access, None))
        try:
            yield
        finally:
            with self.lock:
                self.release(token)

    def publish_host_write(self, *, host_complete: bool, recovers_unknown: bool = False) -> int:
        self.version += 1
        if host_complete:
            self.host_version = self.version
        if recovers_unknown:
            self.poisoned = False
        return self.version

    def publish_device_write(self, session: Any) -> int:
        residency = self.residencies.get(session)
        if residency is None:
            raise RuntimeError("device write completed without a prepared residency")
        self.version += 1
        residency.version = self.version
        return self.version

    def poison(self) -> None:
        self.poisoned = True


class _ResourceTransaction:
    """Atomic multi-owner claims and execution publication."""

    def __init__(
        self,
        requests: list[tuple[Any, Any, _ResourceRegion, str]],
        context: Any,
    ):
        self._context = context
        self._owners = sorted({owner for _, owner, _, _ in requests}, key=lambda owner: owner._control.owner_id)
        self._token: object | None = object()
        self._resolved = False
        for owner in self._owners:
            owner._control.lock.acquire()
        try:
            requests = [
                (
                    identifier,
                    owner,
                    owner._device_claim_region(context, region, access)
                    if hasattr(owner, "_device_claim_region")
                    else region,
                    access,
                )
                for identifier, owner, region, access in requests
            ]
            self._write_requests = tuple(
                (identifier, owner, region) for identifier, owner, region, access in requests if access != "read"
            )
            for name, owner, region, access in requests:
                try:
                    owner._control.validate_device_claim(
                        region,
                        access,
                        context.identity,
                    )
                except RuntimeError as error:
                    if "another RuntimeSession" in str(error):
                        raise RuntimeError(
                            f"dispatch argument '{name}' is already borrowed by another RuntimeSession"
                        ) from error
                    raise RuntimeError(
                        f"dispatch argument '{name}' conflicts with an outstanding device borrow"
                    ) from error
            for _, owner, region, access in requests:
                owner._control.add_device_claim(self._token, region, access, context.identity)
        finally:
            for owner in reversed(self._owners):
                owner._control.lock.release()

    def resolve(self, outcome: Any) -> None:
        mutations: dict[Any, int] = {}
        for kind, slot, state in outcome.mutations:
            identifier: Any = int(slot) if int(kind) == 0 else ("render_pass", int(slot))
            mutations[identifier] = int(state)
        self.resolve_mutations(mutations)

    def resolve_mutations(self, mutations: dict[Any, int]) -> None:
        if self._token is None:
            raise RuntimeError("cannot resolve a released resource transaction")
        if self._resolved:
            raise RuntimeError("resource transaction outcome was already resolved")
        committed: dict[Any, list[_ResourceRegion]] = {}
        indeterminate: set[Any] = set()
        for identifier, owner, region in self._write_requests:
            state = mutations.get(identifier, 0)
            if state in {1, 2}:
                committed.setdefault(owner, []).append(region)
            elif state == 3:
                indeterminate.add(owner)
        for owner in self._owners:
            owner._control.lock.acquire()
        try:
            for owner, regions in committed.items():
                owner._publish_device_write(self._context, tuple(regions))
            for owner in indeterminate:
                owner._control.poison()
            self._resolved = True
            for owner in self._owners:
                owner._control.release(self._token, writes_only=True)
        finally:
            for owner in reversed(self._owners):
                owner._control.lock.release()

    def release(self) -> None:
        token = getattr(self, "_token", None)
        if token is None:
            return
        self._token = None
        owners = getattr(self, "_owners", ())
        for owner in owners:
            owner._control.lock.acquire()
        try:
            for owner in owners:
                owner._control.release(token)
        finally:
            for owner in reversed(owners):
                owner._control.lock.release()

    def __enter__(self) -> _ResourceTransaction:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


@contextmanager
def _resource_transaction_scope(
    requests: list[tuple[Any, Any, _ResourceRegion, str]],
    context: Any,
) -> Iterator[_ResourceTransaction]:
    transaction = _ResourceTransaction(requests, context)
    try:
        yield transaction
    finally:
        transaction.release()


def _scoped_host_access(access: str):
    if access not in {"read", "write"}:
        raise ValueError("host access must be read or write")

    def decorate(operation: Any) -> Any:
        @wraps(operation)
        def invoke(resource: Any, *args: Any, **kwargs: Any) -> Any:
            owner = getattr(resource, "owner", resource)
            region = (
                resource._host_access_region(access) if hasattr(resource, "_host_access_region") else resource._region
            )
            with owner._control.host_access(region, access, type(owner).__name__):
                return operation(resource, *args, **kwargs)

        return invoke

    return decorate
