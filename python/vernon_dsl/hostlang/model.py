from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from ..frontend.model import ConcreteType, TypedExpression, TypedFunctionInstance, TypedStatement


class HostEffect(IntEnum):
    """What a Host function may do through its extern calls, ordered so a caller's effect bounds its callees'."""

    PURE = 0
    READ = 1
    WRITE = 2
    IO = 3

    @classmethod
    def parse(cls, value: str) -> "HostEffect":
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"unknown Host effect '{value}'; expected pure, read, write, or io") from None

    @property
    def spelling(self) -> str:
        return self.name.lower()


class HostStatus(IntEnum):
    """Statuses a Host function returns itself. Extern statuses are passed through unchanged, so these are negative
    to stay apart from the positive failures C APIs report."""

    OK = 0
    DIVISION_BY_ZERO = -1
    INDEX_OUT_OF_RANGE = -2
    INVALID_RANGE_STEP = -3
    # The allocator could not give a list, str or dict the memory it grows into.
    OUT_OF_MEMORY = -4
    # A dict has no entry for the key read, popped or deleted (Python's KeyError).
    KEY_NOT_FOUND = -5
    # A dict changed size while a loop iterated it (Python's RuntimeError).
    CONTAINER_CHANGED = -6
    # A TensorStorage was created with a negative extent, or copied from a Tensor of another shape (numpy's
    # ValueError).
    INVALID_SHAPE = -7


class ExternAbi:
    DIRECT = "direct"
    STATUS = "status"
    ALL = frozenset({DIRECT, STATUS})


@dataclass(frozen=True)
class HostStruct:
    name: str
    fields: tuple[tuple[str, ConcreteType], ...]
    c_type: str | None = None
    # A Host class's state is a Struct type to inference, but its values are places: they live in memory the
    # application owns and are reached only through `self`, never copied.
    place: bool = False
    doc: str | None = None
    metadata: "Metadata" = ()
    # Per-field docs and metadata, in field order; empty when the declaration has none to record.
    field_info: tuple["HostField", ...] = ()


# A metadata value as JSON text, so the reflection table carries it without a type system of its own.
Metadata = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HostField:
    name: str
    type: ConcreteType
    doc: str | None = None
    metadata: Metadata = ()


@dataclass(frozen=True)
class HostMethod:
    name: str
    symbol: str
    # Leading-underscore methods are callable only from the class's own code and are not exported.
    exported: bool


@dataclass(frozen=True)
class HostClass:
    """A `@host` class: mutable state with methods taking `self` by reference. `init` writes the field defaults
    over zeroed memory."""

    name: str
    fields: tuple[HostField, ...]
    methods: tuple[HostMethod, ...]
    init: str
    doc: str | None = None
    metadata: Metadata = ()


@dataclass(frozen=True)
class HostHandle:
    name: str
    c_type: str | None = None
    metadata: Metadata = ()
    # Method name -> extern name, for the externs the handle's class body declares taking it first.
    methods: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class HostEnum:
    """A `@vd.enum` IntEnum, held as an int32_t. Members keep declaration order; two may share a value, as
    aliases do in Python and C."""

    name: str
    members: tuple[tuple[str, int], ...]
    c_type: str | None = None
    doc: str | None = None
    metadata: Metadata = ()

    def value(self, member: str) -> int:
        return dict(self.members)[member]


@dataclass(frozen=True)
class HostExtern:
    """An `@extern` declaration. `name` is its name in the combined module, which prefixes imported names; `symbol`
    is the C symbol the interpreter binds and the object imports."""

    name: str
    symbol: str
    parameters: tuple[tuple[str, ConcreteType], ...]
    result: ConcreteType | None
    effect: HostEffect
    abi: str
    schema: str | None
    source: ast.FunctionDef
    # A generic extern's specialization: the C parameter index receiving the bound Struct's `const VhType *`,
    # which `parameters` omits, and the Struct's name.
    type_argument: tuple[int, str] | None = None
    metadata: Metadata = ()
    # Whether a generic extern's calls pass the bound Struct as a `type: type[T]` argument. If not, the value in
    # `parameters[type_argument[0]]` binds it, and the descriptor goes just before that value.
    type_passed: bool = True
    # Whether the type variable is bound to `vd.HostClass`: the value binding it is a Host class instance the call
    # moves into the callee.
    type_moves: bool = False
    # The `binding=` module of the declaring `vd.library`, which Python mode binds the symbol from when nothing did.
    binding: str | None = None
    # The extern a scoped extern's `with` calls on its first argument however the block exits.
    closed_by: str | None = None


@dataclass
class HostFunctionInfo:
    """A typed function reachable from the entry: a `@host` function or a pure `@func` helper. The indexes key
    typing facts by AST node identity, as the device emitter does."""

    typed: TypedFunctionInstance
    kind: str
    declared_effect: HostEffect
    effect: HostEffect = HostEffect.PURE
    # Names of the externs the function can call, directly or through its callees, in name order.
    reached_externs: tuple[str, ...] = ()
    expressions: dict[int, TypedExpression] = field(default_factory=dict)
    statements: dict[int, TypedStatement] = field(default_factory=dict)
    local_types: dict[str, ConcreteType] = field(default_factory=dict)
    # Class-typed locals: "owned" for an instance the function holds and releases, "borrowed" for a name of a field
    # reached from a parameter, "element" for a loop's element.
    class_locals: dict[str, str] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.typed.symbol

    @property
    def source(self) -> ast.FunctionDef:
        return self.typed.source

    @property
    def parameters(self) -> tuple[tuple[str, ConcreteType], ...]:
        return tuple((parameter.name, parameter.type) for parameter in self.typed.parameters)

    @property
    def result(self) -> ConcreteType | None:
        return self.typed.result_type

    def expression_type(self, node: ast.expr) -> ConcreteType:
        typed = self.expressions.get(id(node))
        if typed is None:
            raise KeyError(f"expression at line {getattr(node, 'lineno', '?')} is missing from the typed Host model")
        return typed.type


@dataclass(frozen=True)
class HostProgram:
    """Host code typed once and shared by Python mode's checks, reflection, and AOT: one `@host` entry with
    everything reachable from it, or a whole module's exported functions and classes."""

    source_path: Path
    entry: str
    structs: dict[str, HostStruct]
    handles: dict[str, HostHandle]
    externs: dict[str, HostExtern]
    functions: dict[str, HostFunctionInfo]
    # Every source file the program was built from, with its digest at build time.
    dependencies: tuple[tuple[Path, str], ...]
    # (exported name without prefix, function symbol); a single-entry program exports only its entry.
    exports: tuple[tuple[str, str], ...] = ()
    classes: dict[str, HostClass] = field(default_factory=dict)
    # The module name the reflection table records; empty for single-entry programs.
    module_name: str = ""
    enums: dict[str, HostEnum] = field(default_factory=dict)

    @property
    def entry_function(self) -> HostFunctionInfo:
        return self.functions[self.entry]

    def exported(self) -> tuple[tuple[str, HostFunctionInfo], ...]:
        return tuple((name, self.functions[symbol]) for name, symbol in self.exports)

    def is_place(self, value_type: ConcreteType) -> bool:
        return value_type.kind == "struct" and self.structs[value_type.name].place

    def struct_fields(self, name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return self.structs[name].fields

    def public_symbols(self) -> frozenset[str]:
        """Function symbols callers reach by name (exports, class initializers and methods); the emitter keeps
        these names unqualified, while it qualifies every other function by its program."""

        symbols = {symbol for _, symbol in self.exports}
        for host_class in self.classes.values():
            symbols.add(host_class.init)
            symbols.update(method.symbol for method in host_class.methods)
        return frozenset(symbols)

    def reachable_externs(self) -> tuple[HostExtern, ...]:
        """Externs the entry can call, in name order; binding all of them before execution makes a missing symbol
        fail before any effect happens."""

        called: set[str] = set()
        for function in self.functions.values():
            for node in ast.walk(function.source):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.externs:
                    called.add(node.func.id)
                    # The `with` a scoped extern opens calls its closer.
                    closer = self.externs[node.func.id].closed_by
                    if closer is not None:
                        called.add(closer)
        return tuple(self.externs[name] for name in sorted(called))


def check_combinable(programs: list[HostProgram]) -> None:
    """Rejects Host modules that cannot share one object and table: the same exported name or class from two source
    files, whose C symbols would collide under the shared prefix, or one Struct name with two layouts, which the
    table names once."""

    exporters: dict[str, Path] = {}
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]] = {}
    for program in programs:
        # `classes` also holds the classes a module imports; those whose initializer it exports are its own.
        exported = {symbol for _, symbol in program.exports}
        defined = [name for name, host_class in program.classes.items() if host_class.init in exported]
        for name in [name for name, _ in program.exports] + defined:
            previous = exporters.setdefault(name, program.source_path)
            if previous != program.source_path:
                raise ValueError(
                    f"'{name}' is defined by both {previous.name} and {program.source_path.name}; the Host modules "
                    "of one object share a symbol prefix, so their exported functions and classes need distinct names"
                )
        for name, host_struct in program.structs.items():
            known = structs.setdefault(name, host_struct.fields)
            if known != host_struct.fields:
                raise ValueError(f"Host modules disagree on the fields of Struct '{name}'")
