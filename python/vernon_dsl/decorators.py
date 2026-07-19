from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


_T = TypeVar("_T")


def _mark(value: _T, kind: str, **options: Any) -> _T:
    setattr(value, "__vernon_dsl__", (kind, options))
    return value


def vertex(function: _T) -> _T:
    return _mark(function, "vertex")


def fragment(function: _T) -> _T:
    return _mark(function, "fragment")


def compute(
    function: _T | None = None, *, workgroup_size: tuple[int, int, int] = (1, 1, 1)
) -> _T | Callable[[_T], _T]:
    def decorate(value: _T) -> _T:
        return _mark(value, "compute", workgroup_size=workgroup_size)

    return decorate(function) if function is not None else decorate


def struct(value: _T) -> _T:
    return _mark(value, "struct")
