from __future__ import annotations

import contextlib
import inspect
import sys
import threading
from collections.abc import Callable, Mapping
from functools import update_wrapper
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

from .._markers import mark
from .externs import ExternRegistry, default_registry
from .model import HostProgram

if TYPE_CHECKING:
    from ..decorators import Library
    from .module import HostModule
    from .native import Entry, HostRuntime


class HostFunction:
    """A `@host` function. Calling it runs its own body as CPython code. The first call from Python type-checks the
    program it reaches, so code the compiler would reject fails here too, and binds every extern it may call.

    Called from other Host code it is a plain call, running with that caller's registry; called from Python it runs
    with `registry`, or the default registry, and converts its arguments and result to the declared types."""

    def __init__(self, function: Callable[..., Any], *, effects: str = "io"):
        self.function = function
        self.effects = effects
        mark(self, "host", effects=effects)
        update_wrapper(self, function)
        source = inspect.getsourcefile(function)
        if source is None:
            raise TypeError(f"@host function '{function.__name__}' must be defined in a source file")
        self.source_path = Path(source).resolve()
        self._lock = threading.Lock()
        self._program: HostProgram | None = None
        self._entry: Entry | None = None
        self._runtimes: dict[int, tuple[ExternRegistry, HostRuntime]] = {}

    def program(self) -> HostProgram:
        """The typed program, built the first time it is needed."""

        from .frontend import load_host_program

        with self._lock:
            if self._program is None:
                self._program = load_host_program(self.source_path, self.__name__)
            return self._program

    def runtime(self, registry: ExternRegistry | None = None) -> HostRuntime:
        from .native import Entry, HostRuntime

        registry = registry or default_registry()
        program = self.program()
        with self._lock:
            cached = self._runtimes.get(id(registry))
            if cached is None or cached[0] is not registry:
                if self._entry is None:
                    self._entry = Entry.of(self.function)
                entry = self._entry

                def entries(symbol: str) -> Entry:
                    if symbol != program.entry:
                        raise KeyError(f"'{self.__name__}' exports no '{symbol}'")
                    return entry

                cached = self._runtimes[id(registry)] = (registry, HostRuntime([program], registry, entries))
            return cached[1]

    def call(self, *args: Any, registry: ExternRegistry | None = None) -> Any:
        from .native import active_runtime

        if registry is None and active_runtime() is not None:
            return self.function(*args)
        if registry is None:
            attached = self._module()
            if attached is not None:
                # The runtime the module's classes run with (class_runtime), so an instance the function makes and
                # moves into C carries the module's table, whose methods run this source, as the AOT object's would.
                return attached.runtime.invoke(attached.entry(self.__name__), args)
        runtime = self.runtime(registry)
        assert self._entry is not None
        return runtime.invoke(self._entry, args)

    def __call__(self, *args: Any) -> Any:
        return self.call(*args)

    def _module(self) -> HostModule | None:
        """The HostModule attached to the imported module defining the function; None for a function whose module
        is not the source file, such as one a test defines inside another function's scope."""

        from .module import HostModule

        module = sys.modules.get(self.function.__module__)
        source = getattr(module, "__file__", None)
        if module is None or source is None or Path(source).resolve() != self.source_path:
            return None
        if self.function.__qualname__ != self.function.__name__:
            return None
        return HostModule.attach(module, eager=False)


class ExternFunction:
    """An `@extern` declaration. It names a C function; Host code calling it reaches the binding of the registry the
    outermost Host call runs with."""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        symbol: str,
        effects: str,
        abi: str,
        meta: Mapping[str, Any] | None = None,
        library: Library | None = None,
        closed_by: ExternFunction | None = None,
    ):
        self.function = function
        self.closed_by = closed_by
        # The `@handle` class whose body declares the extern as a method, whose first parameter, `self`, it is.
        self.receiver: type[Any] | None = None
        self.symbol = symbol
        self.effects = effects
        self.abi = abi
        self.library = library
        self.meta = dict(meta) if meta is not None else {}
        mark(self, "extern", symbol=symbol, effects=effects, abi=abi, meta=self.meta)
        update_wrapper(self, function)

    @property
    def schema(self) -> str | None:
        """The identity of the extern's `vd.library`, which a binding must have been generated from; none
        outside a library."""

        return None if self.library is None else self.library.identity

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> Any:
        return self if instance is None else MethodType(self, instance)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .native import active_runtime

        runtime = active_runtime()
        if runtime is None or kwargs:
            raise TypeError(f"@extern '{self.__name__}' is a C function; call it from @host code")
        result = runtime.call_extern(self, args)
        if self.closed_by is not None:
            return ExternScope(result, self.closed_by, args[0])
        return result


class ExternScope:
    """A scoped extern's call in Python mode: `with` binds its result and calls the closer on the handle it opened
    on however the block exits. A closer failing while an error is already leaving the block is dropped, so that
    error stands, as in compiled code."""

    def __init__(self, result: Any, closer: ExternFunction, handle: Any):
        self.result = result
        self.closer = closer
        self.handle = handle

    def __enter__(self) -> Any:
        return self.result

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: Any) -> bool:
        if kind is None:
            self.closer(self.handle)
        else:
            with contextlib.suppress(Exception):
                self.closer(self.handle)
        return False
