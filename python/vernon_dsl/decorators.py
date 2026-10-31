from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from ._markers import dsl_kind, mark, record_declaration_source

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


def host(function: Any = None, *, effects: str = "io", meta: Mapping[str, Any] | None = None) -> Any:
    """Declare Host code: CPU code that may call `@extern` C functions. In development it runs as the Python it is
    written in; packaged builds compile it ahead of time to a native or wasm32 object.

    On a function, `effects` bounds what its extern calls may do: pure, read, write, or io.

    On a class, the annotated fields are mutable state with defaults, and methods take `self` by reference; a
    method is `io` unless it carries its own `@host(effects=...)`. Instances keep their state in the C layout the
    compiled code uses; compiled, `<Class>__init` initializes memory the application owns. `meta=vd.meta(...)`
    attaches metadata the compiler records in the reflection table without interpreting it; fields take theirs as
    `Annotated[T, vd.meta(...)]`."""

    def decorate(value: Any) -> Any:
        if isinstance(value, type):
            from .hostlang.native import make_host_class

            mark(value, "host", meta=dict(meta or {}))
            record_declaration_source(value)
            return make_host_class(value)
        if meta is not None:
            raise TypeError("meta= applies to @host classes; annotate function results instead")
        from .hostlang.function import HostFunction

        return HostFunction(value, effects=effects)

    return decorate(function) if function is not None else decorate


def meta(**entries: Any) -> Mapping[str, Any]:
    """Metadata of a `@host` class or field, recorded in the reflection table as JSON values and never interpreted
    by the compiler: an inspector's ranges or tooltips, or an application's roles for its classes."""

    return dict(entries)


def extern(
    function: Callable[..., Any] | None = None,
    *,
    symbol: str | None = None,
    effects: str = "io",
    abi: str = "direct",
    schema: str | None = None,
) -> Any:
    """Declare a C function Host code calls; the body is `...`. `abi="direct"` returns the result, which must be a
    scalar or handle; `abi="status"` returns an int32 status, zero on success, and writes the result through a
    trailing out pointer. Handles and scalars pass by value, aggregates by const pointer, and str as a
    {const char *data; uint64_t size} pair by value. `schema` identifies the C API revision the declaration was
    generated from; a binding registered under another schema is rejected.

    A parameter annotated `type[T]` makes the extern generic: Host code passes a `@struct` type there, as in
    `read(entity, Health)`, and the other parameters and the result may use `T`. Each call site's Struct specializes
    the extern, and the C function receives that Struct's `const VhType *` where the type was passed, so one C
    function can store or serialize any Struct by its reflected layout."""

    def decorate(value: Callable[..., Any]) -> Any:
        from .hostlang.function import ExternFunction

        return ExternFunction(value, symbol=symbol or value.__name__, effects=effects, abi=abi, schema=schema)

    return decorate(function) if function is not None else decorate


def handle(value: _T | None = None, *, c_type: str | None = None) -> Any:
    """Declare an opaque 64-bit Host handle type, such as an engine World or Entity. Host code compares handles and
    constructs only the null handle, `Handle()`; externs make the others. `c_type` names the C type generated headers
    use for it, a struct holding one uint64_t. In Python a handle is its bits as an int, so `Handle()` is 0."""

    def decorate(cls: _T) -> _T:
        mark(cls, "handle", c_type=c_type)
        record_declaration_source(cls)
        cls.__new__ = staticmethod(lambda _cls, bits=0: int(bits))  # type: ignore[attr-defined]
        return cls

    return decorate(value) if value is not None else decorate


def struct(
    value: _T | None = None,
    *,
    shared: bool = False,
    c_type: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> _T | Callable[[_T], _T]:
    """Declare a Value Struct. In Host code, `meta=vd.meta(...)` and fields annotated `Annotated[T, vd.meta(...)]`
    record metadata in the reflection table, as on `@host` classes."""

    def decorate(cls: _T) -> _T:
        from .host_values import forbid_struct_construction, make_shared_struct

        options: dict[str, Any] = {"c_type": c_type} if c_type is not None else {}
        if meta is not None:
            options["meta"] = dict(meta)
        mark(cls, "struct", shared=shared, **options)
        record_declaration_source(cls)
        transformed = make_shared_struct(cls) if shared else forbid_struct_construction(cls)
        return transformed

    return decorate(value) if value is not None else decorate


def params(cls: _T) -> _T:
    """Declare a bundle of entry parameters with no physical meaning; on the host it is a keyword-only container."""

    from ._params import make_params

    mark(cls, "params")
    record_declaration_source(cls)
    return make_params(cls)  # type: ignore[arg-type,return-value]
