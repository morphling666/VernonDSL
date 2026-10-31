from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from functools import update_wrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._markers import mark
from .externs import ExternRegistry, default_registry
from .model import HostProgram

if TYPE_CHECKING:
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
        runtime = self.runtime(registry)
        assert self._entry is not None
        return runtime.invoke(self._entry, args)

    def __call__(self, *args: Any) -> Any:
        return self.call(*args)


class ExternFunction:
    """An `@extern` declaration. It names a C function; Host code calling it reaches the binding of the registry the
    outermost Host call runs with."""

    def __init__(self, function: Callable[..., Any], *, symbol: str, effects: str, abi: str, schema: str | None):
        self.function = function
        self.symbol = symbol
        self.effects = effects
        self.abi = abi
        self.schema = schema
        mark(self, "extern", symbol=symbol, effects=effects, abi=abi, schema=schema)
        update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .native import active_runtime

        runtime = active_runtime()
        if runtime is None or kwargs:
            raise TypeError(f"@extern '{self.__name__}' is a C function; call it from @host code")
        return runtime.call_extern(self, args)
