from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from typing import Any, TypeVar

_T = TypeVar("_T")


def _mark(value: _T, kind: str, **options: Any) -> _T:
    setattr(value, "__vernon_dsl__", (kind, options))
    return value


class ShaderFunction:
    """A declarative shader function that cannot execute as host Python."""

    def __init__(self,
                 function: Callable[..., Any],
                 kind: str,
                 *,
                 shared: bool = False):
        self.function = function
        self.__vernon_dsl__ = (kind, {"shared": shared})
        self.shared = shared
        update_wrapper(self, function)

    @property
    def kind(self) -> str:
        return self.__vernon_dsl__[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.shared:
            return self.function(*args, **kwargs)
        del args, kwargs
        raise TypeError(
            f"@{self.kind} function '{self.__name__}' is shader-only and "
            "cannot be called from host Python")

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        if instance is None:
            return self
        if not self.shared:
            return self
        return self.function.__get__(instance, owner)


def vertex(function: _T) -> _T:
    return ShaderFunction(function,
                          "vertex")  # type: ignore[arg-type,return-value]


def fragment(function: _T) -> _T:
    return ShaderFunction(function,
                          "fragment")  # type: ignore[arg-type,return-value]


def func(function: _T | None = None,
         *,
         shared: bool = False) -> _T | Callable[[_T], _T]:
    """Declare a reusable stage-polymorphic shader helper."""

    def decorate(value: _T) -> _T:
        return ShaderFunction(
            value, "func",
            shared=shared)  # type: ignore[arg-type,return-value]

    return decorate(function) if function is not None else decorate


def compute(
    function: _T | None = None,
    *,
    workgroup_size: tuple[int, int, int] = (1, 1, 1)
) -> _T | Callable[[_T], _T]:

    def decorate(value: _T) -> _T:
        from .runtime import Kernel

        return Kernel(
            value, workgroup_size=workgroup_size)  # type: ignore[return-value]

    return decorate(function) if function is not None else decorate


def kernel(
    function: _T | None = None,
    *,
    workgroup_size: tuple[int, int, int] = (1, 1, 1)
) -> _T | Callable[[_T], _T]:
    """Mark a compute kernel; ``compute`` remains the compatibility spelling."""
    return compute(function, workgroup_size=workgroup_size)


def struct(value: _T | None = None,
           *,
           shared: bool = False) -> _T | Callable[[_T], _T]:

    def decorate(cls: _T) -> _T:
        from .host_values import forbid_struct_construction, make_shared_struct

        _mark(cls, "struct", shared=shared)
        transformed = (make_shared_struct(cls)
                       if shared else forbid_struct_construction(cls))
        return transformed

    return decorate(value) if value is not None else decorate
