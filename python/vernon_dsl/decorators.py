from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from ._markers import dsl_kind, mark

if TYPE_CHECKING:
    from .runtime import Kernel

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")


class ShaderFunction:
    """A declarative shader function that cannot execute as host Python."""

    def __init__(self, function: Callable[..., Any], kind: str, *, shared: bool = False):
        self.function = function
        mark(self, kind, shared=shared)
        self.shared = shared
        update_wrapper(self, function)

    @property
    def kind(self) -> str:
        return dsl_kind(self)  # type: ignore[return-value]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.shared:
            return self.function(*args, **kwargs)
        del args, kwargs
        raise TypeError(f"@{self.kind} function '{self.__name__}' is shader-only and cannot be called from host Python")

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        if instance is None:
            return self
        if not self.shared:
            return self
        return self.function.__get__(instance, owner)


def vertex(function: _T) -> _T:
    return ShaderFunction(function, "vertex")  # type: ignore[arg-type,return-value]


def fragment(function: _T) -> _T:
    return ShaderFunction(function, "fragment")  # type: ignore[arg-type,return-value]


@overload
def func(function: Callable[_P, _R], *, shared: bool = False) -> Callable[_P, _R]: ...


@overload
def func(function: None = None, *, shared: bool = False) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def func(function: Callable[..., Any] | None = None, *, shared: bool = False) -> Any:
    """Declare a reusable stage-polymorphic shader helper."""

    def decorate(value: _T) -> _T:
        return ShaderFunction(value, "func", shared=shared)  # type: ignore[arg-type,return-value]

    return decorate(function) if function is not None else decorate


@overload
def kernel(function: Callable[..., Any], *, workgroup_size: tuple[int, int, int] = (1, 1, 1)) -> Kernel: ...


@overload
def kernel(
    function: None = None, *, workgroup_size: tuple[int, int, int] = (1, 1, 1)
) -> Callable[[Callable[..., Any]], Kernel]: ...


def kernel(function: Callable[..., Any] | None = None, *, workgroup_size: tuple[int, int, int] = (1, 1, 1)) -> Any:
    """Declare a compute entry point."""

    def decorate(value: _T) -> _T:
        from .runtime import Kernel

        return Kernel(value, workgroup_size=workgroup_size)  # type: ignore[return-value]

    return decorate(function) if function is not None else decorate


def struct(value: _T | None = None, *, shared: bool = False) -> _T | Callable[[_T], _T]:

    def decorate(cls: _T) -> _T:
        from .host_values import forbid_struct_construction, make_shared_struct

        mark(cls, "struct", shared=shared)
        transformed = make_shared_struct(cls) if shared else forbid_struct_construction(cls)
        return transformed

    return decorate(value) if value is not None else decorate


def binding_group(cls: _T) -> _T:
    """Declare a named group of entry interface items; on the host it is a keyword-only container."""

    from .binding_groups import make_binding_group

    mark(cls, "binding_group")
    return make_binding_group(cls)  # type: ignore[arg-type,return-value]
