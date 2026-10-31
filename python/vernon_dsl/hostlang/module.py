"""A Host module loaded in-process: the development counterpart of linking the AOT object.

The source file runs as an ordinary Python module, and its `@host` classes keep their instances' state in C memory
with the compiled layout. The module's reflection table points at C trampolines into the Python code, so a C++
application reaches development code exactly as it reaches the packaged object.
"""

from __future__ import annotations

import ctypes
import importlib
import itertools
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..source_identity import source_file_digest
from .aot import HostSources, load_programs, source_paths
from .externs import ExternRegistry, default_registry
from .model import HostProgram
from .native import Entry, HostRuntime, address_of, check_layout, initializer
from .reflect import CLASS, DICT, LIST, STORAGE, STR, VhType

_MODULE_NAMES = itertools.count()
_ATTACHED: dict[str, HostModule] = {}


def _execute(path: Path) -> types.ModuleType:
    """Runs `path` as a new module. It is registered in `sys.modules` under a unique name, because type annotations
    resolve through the module of the class that declares them; its directory is importable while it runs, as it is
    for a script."""

    name = f"_vernon_host_{path.stem}_{next(_MODULE_NAMES)}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    directory = str(path.parent)
    added = directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    try:
        # Compiled from the text, not through the bytecode cache, which keys on whole-second modification times.
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)  # noqa: S102
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        if added:
            sys.path.remove(directory)
    return module


def _execute_all(paths: list[Path]) -> list[types.ModuleType]:
    """Runs the sources of one Host module in order. While a later source runs, an import of an earlier one by its
    plain name finds the module run here, so a class one source imports from another is the class this Host module
    binds to its runtime, not a second copy."""

    executed: list[types.ModuleType] = []
    shadowed: dict[str, types.ModuleType | None] = {}
    try:
        for path in paths:
            executed.append(_execute(path))
            if len(paths) > 1 and path.stem not in shadowed:
                shadowed[path.stem] = sys.modules.get(path.stem)
                sys.modules[path.stem] = executed[-1]
    finally:
        for name, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return executed


@dataclass(frozen=True)
class _Source:
    path: Path
    program: HostProgram
    namespace: types.ModuleType

    def defined_classes(self) -> list[str]:
        exported = {symbol for _, symbol in self.program.exports}
        return [name for name, host_class in self.program.classes.items() if host_class.init in exported]


def _reload_dependencies(programs: list[HostProgram]) -> None:
    """Re-imports the project modules the programs import whose source changed, so a reload sees their edits."""

    own = {program.source_path for program in programs}
    changed = {
        path.resolve()
        for program in programs
        for path, digest in program.dependencies
        if path.resolve() not in own and source_file_digest(path) != digest
    }
    if not changed:
        return
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve() in changed:
            importlib.reload(module)


class HostModule:
    """Every `@host` function and class of one or more source files, with the externs `registry` binds, behind one
    reflection table as one AOT object holds them. The constructor type-checks the files, then runs each as a new
    Python module; `attach` adopts a module Python already imported."""

    def __init__(
        self,
        sources: HostSources,
        registry: ExternRegistry | None = None,
        *,
        namespace: types.ModuleType | None = None,
        eager: bool = True,
        prefix: str = "vernon_host_",
        name: str | None = None,
    ):
        paths = source_paths(sources)
        if namespace is not None and len(paths) != 1:
            raise ValueError("a namespace is adopted for a single source file")
        self.asset_id: str | None = None
        self.name = name
        self.registry = registry or default_registry()
        programs = load_programs(paths)
        namespaces = [namespace] if namespace is not None else _execute_all(paths)
        self.sources = [
            _Source(path, program, module) for path, program, module in zip(paths, programs, namespaces, strict=True)
        ]
        self.runtime = HostRuntime(programs, self.registry, self._entry, eager=eager, prefix=prefix, module_name=name)
        for source in self.sources:
            for class_name in source.defined_classes():
                cls = getattr(source.namespace, class_name)
                check_layout(cls, source.program)
                cls.__vernon_runtime__ = self.runtime

    def _single(self) -> _Source:
        if len(self.sources) != 1:
            raise AttributeError("a Host module of several source files has no single program or namespace")
        return self.sources[0]

    @property
    def source_path(self) -> Path:
        return self._single().path

    @property
    def program(self) -> HostProgram:
        return self._single().program

    @property
    def namespace(self) -> types.ModuleType:
        return self._single().namespace

    @classmethod
    def load(
        cls, sources: HostSources, registry: ExternRegistry | None = None, *, prefix: str = "vernon_host_"
    ) -> HostModule:
        """`prefix` names the table's symbols as an AOT build with the same `--prefix` exports them."""

        return cls(sources, registry, prefix=prefix)

    @classmethod
    def from_asset(cls, asset: Any, registry: ExternRegistry | None = None) -> HostModule:
        """Python mode of a Host module Program Asset (`vd.program_asset(program=<module or modules>)`): its sources
        loaded with the asset's prefix and named by its id, so the table matches the cooked object's. An application
        registers `table` under `asset_id` (`vernonRuntimeRegisterStaticHostModule`) to shadow the linked object."""

        if not getattr(asset, "is_host", False):
            raise TypeError("from_asset takes a Host module Program Asset declaration")
        loaded = cls(asset.host_sources, registry, prefix=asset.prefix, name=asset.id)
        loaded.asset_id = asset.id
        return loaded

    @classmethod
    def attach(
        cls, module: types.ModuleType, registry: ExternRegistry | None = None, *, eager: bool = True
    ) -> HostModule:
        """The HostModule of an imported Python module, created once per module; `eager` as for `HostRuntime`."""

        attached = _ATTACHED.get(module.__name__)
        if attached is None or attached.namespace is not module:
            attached = cls(module.__file__, registry, namespace=module, eager=eager)  # type: ignore[arg-type]
            _ATTACHED[module.__name__] = attached
        return attached

    @property
    def table(self) -> Any:
        """The `VhModule` reflection table (`vernon_host/reflect.h`)."""

        return self.runtime.table

    @property
    def errors(self) -> list[BaseException]:
        """Exceptions raised under the table's C entry points, which cannot unwind into their C callers."""

        return self.runtime.errors

    def take_error(self) -> BaseException | None:
        """The exception behind the most recent nonzero status a C caller received, clearing the kept ones, so an
        application reporting that status can chain the Python traceback onto its own error."""

        errors = self.runtime.errors
        error = errors[-1] if errors else None
        errors.clear()
        return error

    @property
    def stale(self) -> bool:
        """Whether a source file the module was built from changed since."""

        return any(
            source_file_digest(path) != digest
            for source in self.sources
            for path, digest in source.program.dependencies
        )

    def reload(self) -> HostModule:
        """The module built and run again from its sources; this one stays usable while its instances are alive."""

        _reload_dependencies([source.program for source in self.sources])
        reloaded = HostModule(
            [source.path for source in self.sources], self.registry, prefix=self.runtime.prefix, name=self.name
        )
        reloaded.asset_id = self.asset_id
        return reloaded

    def call(self, name: str, *args: Any) -> Any:
        """Calls the module's `@host` function `name` from Python."""

        for source in self.sources:
            if name in dict(source.program.exports) and name in source.program.functions:
                return self.runtime.invoke(self._entry(name), args)
        raise KeyError(f"the module exports no @host function '{name}'")

    def type(self, name: str) -> VhType:
        table = self.table
        for index in range(table.type_count):
            descriptor = table.types[index].contents
            if descriptor.name.decode() == name:
                return descriptor
        raise KeyError(f"the module has no type '{name}'")

    def new(self, class_name: str) -> Any:
        """A new instance of the class, its fields at their defaults."""

        for source in self.sources:
            if class_name in source.defined_classes():
                return getattr(source.namespace, class_name)()
        raise KeyError(f"the module defines no Host class '{class_name}'")

    def to_json(self, instance: Any) -> dict[str, Any]:
        """The fields of a Host class instance in their JSON form (`value_json`)."""

        from .native import class_runtime
        from .value_json import to_json

        return to_json(class_runtime(type(instance)).descriptor(type(instance).__name__), address_of(instance))

    def from_json(self, instance: Any, value: dict[str, Any]) -> None:
        """Writes the fields `value` names into a Host class instance, checking all of them before writing any."""

        from .native import class_runtime
        from .value_json import from_json

        name = type(instance).__name__
        runtime = class_runtime(type(instance))
        from_json(runtime.descriptor(name), value, address_of(instance), path=name, allocator=runtime.allocator.value)

    def migrate(self, instance: Any) -> Any:
        """A new instance of this module's class of the same name, carrying over the fields of `instance` that kept
        their name and layout, as `vh_migrate` does. Containers and storages move, leaving those of `instance`
        empty."""

        from .native import class_runtime

        name = type(instance).__name__
        migrated = self.new(name)
        _migrate(
            class_runtime(type(instance)).descriptor(name),
            address_of(instance),
            self.runtime.descriptor(name),
            address_of(migrated),
        )
        return migrated

    def _entry(self, symbol: str) -> Entry:
        for source in self.sources:
            for class_name in source.defined_classes():
                host_class = source.program.classes[class_name]
                cls = getattr(source.namespace, class_name)
                if symbol == host_class.init:
                    return initializer(cls)
                for method in host_class.methods:
                    if method.symbol == symbol:
                        return Entry.of(getattr(cls, method.name).__vernon_function__, cls)
        # A source's own exports first: another source may import a function of the same name.
        owners = sorted(self.sources, key=lambda source: symbol not in dict(source.program.exports))
        for source in owners:
            function = getattr(source.namespace, symbol, None)
            if function is not None and hasattr(function, "function"):
                return Entry.of(function.function)
        raise KeyError(f"the module has no @host function '{symbol}'")


_FINI = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _forget(value_type: VhType, address: int) -> None:
    """Zeroes the container records of a value whose memory moved elsewhere, as `vh_forget` does."""

    if value_type.kind in {LIST, STR, DICT, STORAGE}:
        ctypes.memset(address, 0, value_type.size)
    elif value_type.kind == CLASS and value_type.fini:
        for index in range(value_type.field_count):
            field = value_type.fields[index]
            _forget(field.type.contents, address + field.offset)


def _migrate(from_type: VhType, source: int, to_type: VhType, target: int) -> None:
    """`vh_migrate`: containers move, releasing the target's defaults and zeroing the source's records."""

    if from_type.layout_hash == to_type.layout_hash:
        if to_type.fini:
            _FINI(to_type.fini)(target)
        ctypes.memmove(target, source, to_type.size)
        _forget(from_type, source)
        return
    sources = {from_type.fields[index].name: from_type.fields[index] for index in range(from_type.field_count)}
    for index in range(to_type.field_count):
        field = to_type.fields[index]
        previous = sources.get(field.name)
        if previous is None:
            continue
        old, new = previous.type.contents, field.type.contents
        if old.layout_hash == new.layout_hash or (old.kind == CLASS and new.kind == CLASS):
            _migrate(old, source + previous.offset, new, target + field.offset)


__all__ = ["HostModule", "address_of"]
