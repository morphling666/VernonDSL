"""Frontend-owned interception for host program capture."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator, Protocol


class KernelCapture(Protocol):
    def capture_kernel(
        self,
        kernel: Any,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int] | None,
        features: tuple[str, ...],
    ) -> bool:
        """Record a call and return whether normal kernel dispatch is consumed."""
        ...

    def capture_allocation(self, value: Any, initializer: str) -> None:
        """Record a graph-owned allocation created while exporting."""
        ...


_ACTIVE_CAPTURE: contextvars.ContextVar[KernelCapture | None] = contextvars.ContextVar(
    "vernon_frontend_capture", default=None
)


@contextmanager
def capture_scope(capture: KernelCapture) -> Iterator[None]:
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(token)


def capture_kernel_call(
    kernel: Any,
    arguments: tuple[Any, ...],
    grid: tuple[int, int, int] | None,
    features: tuple[str, ...],
) -> bool:
    active = _ACTIVE_CAPTURE.get()
    return False if active is None else active.capture_kernel(kernel, arguments, grid, features)


def current_capture() -> KernelCapture | None:
    return _ACTIVE_CAPTURE.get()


def capture_allocation(value: Any, initializer: str) -> None:
    active = _ACTIVE_CAPTURE.get()
    callback = getattr(active, "capture_allocation", None)
    if callback is not None:
        callback(value, initializer)


__all__ = ["KernelCapture", "capture_allocation", "capture_kernel_call", "capture_scope", "current_capture"]
