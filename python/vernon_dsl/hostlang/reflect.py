"""The reflection table every Host object exports as `<prefix>module`; its C layout is `vernon_host/reflect.h`.

The table is LLVM constants of structs mirroring the header field for field, so LLVM lays them out with the
target's C rules, pointer width included. Layout hashes are FNV-1a 64 of a canonical spelling of the layout, with
field names, so a renamed or retyped field changes its class's hash.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..frontend.model import ConcreteType
from .layout import layout, members
from .llvm_emitter import type_descriptor
from .model import HostClass, HostExtern, HostFunctionInfo, HostProgram, Metadata
from .values import element_of, key_of

if TYPE_CHECKING:
    from .llvm_emitter import ModuleEmitter

REFLECT_VERSION = 3
KIND_CODES = {"bool": 1, "i32": 2, "u32": 3, "i64": 4, "u64": 5, "f32": 6, "f64": 7}
HANDLE, ARRAY, STRUCT, TUPLE, CLASS, LIST, STR, DICT, STORAGE, ENUM = 8, 9, 10, 11, 12, 13, 14, 15, 16, 17

# Records starting with `i32` begin with their `struct_size`.
_TYPES = {
    "meta": "{ ptr, ptr }",
    "field": "{ i32, ptr, ptr, i64, ptr, ptr, i32 }",
    "param": "{ ptr, ptr }",
    "call": "{ i32, i32, ptr, ptr, ptr, i32 }",
    "function": "{ i32, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, i32 }",
    "enumerator": "{ ptr, i64 }",
    "type": (
        "{ i32, i32, i32, i64, ptr, i64, ptr, ptr, i32, i32, ptr, ptr, i32, ptr, ptr, ptr, i32, ptr, ptr, ptr, i32 }"
    ),
    "module": "{ i32, i32, i32, ptr, i64, ptr, ptr, i32, ptr }",
}


def _size_of(record: str) -> str:
    """The target's size of a reflection record, as an LLVM constant."""

    return f"i32 ptrtoint (ptr getelementptr (%vh.{record}, ptr null, i32 1) to i32)"


def include_dir() -> Path:
    """The directory holding `vernon_host/reflect.h`, for applications' include paths."""

    return Path(str(resources.files("vernon_dsl.hostlang") / "include"))


def fnv1a(text: str) -> int:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


class _Fields:
    def __init__(self, programs: list[HostProgram]):
        self.structs = {name: value for program in programs for name, value in program.structs.items()}
        self.enums = {name: value for program in programs for name, value in program.enums.items()}

    def __call__(self, name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return self.structs[name].fields


def canonical(value_type: ConcreteType, fields: _Fields, spelling: frozenset[str] = frozenset()) -> str:
    """The layout spelling layout hashes are taken of. A class met again inside its own spelling, through a list or
    dict of itself, is spelled by name, so a recursive type has a finite spelling."""

    if value_type.kind == "scalar":
        return value_type.name
    if value_type.kind == "handle":
        return f"handle {value_type.name}"
    if value_type.kind == "enum":
        enumerators = ",".join(f"{name}={value}" for name, value in fields.enums[value_type.name].members)
        return f"enum {value_type.name}{{{enumerators}}}i32"
    if value_type.kind == "tensor":
        shape = "x".join(str(extent) for extent in value_type.arguments[1:])
        return f"[{shape}]{canonical(element_of(value_type), fields, spelling)}"
    if value_type.kind == "str":
        return "str"
    if value_type.kind == "list":
        return f"list[{canonical(element_of(value_type), fields, spelling)}]"
    if value_type.kind == "dict":
        key, value = (
            canonical(key_of(value_type), fields, spelling),
            canonical(element_of(value_type), fields, spelling),
        )
        return f"dict[{key},{value}]"
    if value_type.kind == "storage":
        shape = "x".join(str(extent) for extent in value_type.arguments[1])  # type: ignore[union-attr]
        return f"storage[{shape}]{canonical(element_of(value_type), fields, spelling)}"
    if value_type.kind == "struct" and value_type.name in spelling:
        return f"class {value_type.name}"
    inner = spelling | {value_type.name} if value_type.kind == "struct" else spelling
    value_layout = layout(value_type, fields)
    if value_type.kind == "tuple":
        names = [f"item{index}" for index in range(len(value_layout.offsets))]
        title = "tuple"
    else:
        names = [name for name, _ in fields(value_type.name)]
        title = ("class " if fields.structs[value_type.name].place else "struct ") + value_type.name
    body = ",".join(
        f"{name}@{offset}:{canonical(member, fields, inner)}"
        for name, offset, member in zip(names, value_layout.offsets, members(value_type, fields), strict=True)
    )
    return f"{title}{{{body}}}{value_layout.size}/{value_layout.alignment}"


@dataclass(frozen=True)
class FieldRecord:
    name: str
    type: ConcreteType
    offset: int
    doc: str | None
    metadata: Metadata


@dataclass(frozen=True)
class CallRecord:
    symbol: str
    effect: int
    # The Struct bound to a generic extern's type parameter.
    type: ConcreteType | None
    metadata: Metadata = ()


@dataclass(frozen=True)
class FunctionRecord:
    name: str
    # The internal function the table's address points at, and the exported symbol, None for private methods.
    symbol: str
    exported: str | None
    parameters: tuple[tuple[str, ConcreteType], ...]
    effect: int
    result: ConcreteType | None
    calls: tuple[CallRecord, ...] = ()


@dataclass
class TypeRecord:
    value_type: ConcreteType
    kind: int
    alignment: int
    size: int
    name: str
    layout_hash: int
    element: ConcreteType | None = None
    shape: tuple[int, ...] = ()
    fields: list[FieldRecord] = field(default_factory=list)
    methods: list[FunctionRecord] = field(default_factory=list)
    init: str | None = None
    doc: str | None = None
    metadata: Metadata = ()
    # Whether values own memory their type's fini releases: a list, str, dict or storage, or a class holding one.
    owns: bool = False
    # A dict's key type.
    key: ConcreteType | None = None
    # An enum's members, in declaration order.
    enumerators: tuple[tuple[str, int], ...] = ()


@dataclass
class ModuleRecord:
    """The reflection table of a set of programs, independent of whether LLVM constants or ctypes hold it."""

    name: str
    hash: int
    # Every described type in the order its descriptor is completed: a type follows the types it refers to.
    types: list[TypeRecord]
    listed: list[ConcreteType]
    functions: list[FunctionRecord]


class _Describer:
    def __init__(self, programs: list[HostProgram], prefix: str):
        self.programs = programs
        self.fields = _Fields(programs)
        self.classes: dict[str, HostClass] = {
            name: value for program in programs for name, value in program.classes.items()
        }
        self.functions: dict[str, HostFunctionInfo] = {
            symbol: info for program in programs for symbol, info in program.functions.items()
        }
        self.externs: dict[str, HostExtern] = {
            name: extern for program in programs for name, extern in program.externs.items()
        }
        self.exported_names = {symbol: prefix + name for program in programs for name, symbol in program.exports}
        self.seen: set[ConcreteType] = set()
        self.ordered: list[TypeRecord] = []
        self.records: dict[ConcreteType, TypeRecord] = {}
        self.handles = {name: value for program in programs for name, value in program.handles.items()}

    def function(self, name: str, info: HostFunctionInfo, method: bool) -> FunctionRecord:
        parameters = info.parameters[1:] if method else info.parameters
        for _, value_type in parameters:
            self.type(value_type)
        if info.result is not None:
            self.type(info.result)
        calls = set()
        for extern_name in info.reached_externs:
            extern = self.externs[extern_name]
            bound = None
            if extern.type_argument is not None:
                bound = ConcreteType("struct", extern.type_argument[1])
                self.type(bound)
            calls.add(CallRecord(extern.symbol, int(extern.effect), bound, extern.metadata))
        return FunctionRecord(
            name,
            info.symbol,
            self.exported_names.get(info.symbol),
            parameters,
            int(info.declared_effect),
            info.result,
            tuple(sorted(calls, key=lambda call: (call.symbol, call.type.name if call.type else ""))),
        )

    def type(self, value_type: ConcreteType) -> None:
        if value_type in self.seen:
            return
        self.seen.add(value_type)
        value_layout = layout(value_type, self.fields)
        record = TypeRecord(
            value_type, 0, value_layout.alignment, value_layout.size, "", fnv1a(canonical(value_type, self.fields))
        )
        if value_type.kind == "scalar":
            record.kind, record.name = KIND_CODES[value_type.name], value_type.name
        elif value_type.kind == "handle":
            record.kind, record.name = HANDLE, value_type.name
            declared_handle = self.handles.get(value_type.name)
            record.metadata = declared_handle.metadata if declared_handle is not None else ()
        elif value_type.kind == "enum":
            declared_enum = self.fields.enums[value_type.name]
            record.kind, record.name = ENUM, value_type.name
            record.element = ConcreteType("scalar", "i32")
            self.type(record.element)
            record.doc, record.metadata = declared_enum.doc, declared_enum.metadata
            record.enumerators = declared_enum.members
        elif value_type.kind == "tensor":
            record.kind, record.name = ARRAY, value_type.spelling
            record.element = element_of(value_type)
            self.type(record.element)
            record.shape = tuple(int(extent) for extent in value_type.arguments[1:])  # type: ignore[arg-type]
        elif value_type.kind == "list":
            record.kind, record.name, record.owns = LIST, value_type.spelling, True
            record.element = element_of(value_type)
            self.type(record.element)
        elif value_type.kind == "str":
            record.kind, record.name, record.owns = STR, "str", True
        elif value_type.kind == "dict":
            record.kind, record.name, record.owns = DICT, value_type.spelling, True
            record.key, record.element = key_of(value_type), element_of(value_type)
            self.type(record.key)
            self.type(record.element)
        elif value_type.kind == "storage":
            record.kind, record.name, record.owns = STORAGE, value_type.spelling, True
            record.element = element_of(value_type)
            self.type(record.element)
            # A dyn extent is 0, which no static extent is.
            record.shape = tuple(extent if isinstance(extent, int) else 0 for extent in value_type.arguments[1])  # type: ignore[union-attr]
        else:
            member_types = members(value_type, self.fields)
            host_class = self.classes.get(value_type.name) if value_type.kind == "struct" else None
            if value_type.kind == "tuple":
                record.kind, record.name = TUPLE, value_type.spelling
                entries: list[tuple[str, str | None, Metadata]] = [
                    (f"item{index}", None, ()) for index in range(len(member_types))
                ]
            else:
                record.kind = CLASS if host_class is not None else STRUCT
                record.name = value_type.name
                declared = self.fields.structs[value_type.name]
                if host_class is not None:
                    entries = [(item.name, item.doc, item.metadata) for item in host_class.fields]
                    record.doc, record.metadata = host_class.doc, host_class.metadata
                else:
                    if declared.field_info:
                        entries = [(item.name, item.doc, item.metadata) for item in declared.field_info]
                    else:
                        entries = [(name, None, ()) for name, _ in self.fields(value_type.name)]
                    record.doc, record.metadata = declared.doc, declared.metadata
            for (name, doc, metadata), member, offset in zip(entries, member_types, value_layout.offsets, strict=True):
                self.type(member)
                record.fields.append(FieldRecord(name, member, offset, doc, metadata))
                record.owns |= self.records[member].owns
            if host_class is not None:
                record.methods = [
                    self.function(method.name, self.functions[method.symbol], True)
                    for method in host_class.methods
                    if method.symbol in self.functions
                ]
                record.init = host_class.init if host_class.init in self.functions else None
        self.ordered.append(record)
        self.records[value_type] = record

    def describe(self) -> ModuleRecord:
        for program in self.programs:
            for name in program.structs:
                self.type(ConcreteType("struct", name))
            for name in program.handles:
                self.type(ConcreteType("handle", name))
            for name in program.enums:
                self.type(ConcreteType("enum", name))
        class_symbols = {
            symbol
            for host_class in self.classes.values()
            for symbol in (host_class.init, *(method.symbol for method in host_class.methods))
        }
        functions: list[FunctionRecord] = []
        seen: set[str] = set()
        for program in self.programs:
            for name, info in program.exported():
                if info.symbol in class_symbols or info.symbol in seen:
                    continue
                seen.add(info.symbol)
                functions.append(self.function(name, info, False))
        listed = [
            record.value_type
            for record in self.ordered
            if record.value_type.kind in {"struct", "tuple", "handle", "enum"}
        ]
        signature = "|".join(canonical(value, self.fields) for value in listed)
        signature += "|" + "|".join(
            f"{info.symbol}({','.join(canonical(value, self.fields) for _, value in info.parameters)})"
            f"->{canonical(info.result, self.fields) if info.result is not None else 'none'}"
            for program in self.programs
            for _, info in program.exported()
        )
        module_name = next((program.module_name for program in self.programs if program.module_name), "")
        return ModuleRecord(module_name, fnv1a(signature), self.ordered, listed, functions)


def describe(programs: list[HostProgram], prefix: str, module_name: str | None = None) -> ModuleRecord:
    """The table of `programs`; `module_name` overrides the name recorded, which is otherwise the first program's
    module name (a Host module asset records its id)."""

    record = _Describer(programs, prefix).describe()
    return record if module_name is None else replace(record, name=module_name)


class _LlvmTable:
    def __init__(self, module: "ModuleEmitter"):
        self.module = module
        self.descriptors: dict[ConcreteType, str] = {}
        self.lines: list[str] = []
        self.counter = 0

    def name(self, hint: str) -> str:
        self.counter += 1
        return f'@"vh.reflect.{hint}.{self.counter}"'

    def string(self, value: str | None) -> str:
        return "null" if value is None else self.module.string(value)[0]

    def array(self, hint: str, element: str, items: list[str]) -> str:
        if not items:
            return "null"
        name = self.name(hint)
        self.lines.append(
            f"{name} = private unnamed_addr constant [{len(items)} x {element}] [{', '.join(items)}], align 8"
        )
        return name

    def metadata(self, entries: Metadata) -> tuple[str, int]:
        items = [f"%vh.meta {{ ptr {self.string(key)}, ptr {self.string(value)} }}" for key, value in entries]
        return self.array("meta", "%vh.meta", items), len(items)

    def function(self, record: FunctionRecord) -> str:
        items = [
            f"%vh.param {{ ptr {self.string(name)}, ptr {self.descriptors[value]} }}"
            for name, value in record.parameters
        ]
        params = self.array("params", "%vh.param", items)
        result = "null" if record.result is None else self.descriptors[record.result]
        call_items = []
        for call in record.calls:
            meta, meta_count = self.metadata(call.metadata)
            call_items.append(
                f"%vh.call {{ {_size_of('call')}, i32 {call.effect}, ptr {self.string(call.symbol)}, "
                f"ptr {'null' if call.type is None else self.descriptors[call.type]}, ptr {meta}, i32 {meta_count} }}"
            )
        calls = self.array("calls", "%vh.call", call_items)
        return (
            f"%vh.function {{ {_size_of('function')}, ptr {self.string(record.name)}, "
            f'ptr {self.string(record.exported)}, ptr @"vh.{record.symbol}", ptr {params}, i32 {len(items)}, '
            f"i32 {record.effect}, ptr {result}, ptr {calls}, i32 {len(call_items)} }}"
        )

    def emit(self, table: ModuleRecord) -> None:
        for record in table.types:
            named = record.value_type.kind in {"struct", "handle"}
            self.descriptors[record.value_type] = type_descriptor(record.value_type) if named else self.name("type")
        for record in table.types:
            element = "null" if record.element is None else self.descriptors[record.element]
            shape = self.array("shape", "i32", [f"i32 {extent}" for extent in record.shape])
            items = []
            for item in record.fields:
                meta_array, meta_count = self.metadata(item.metadata)
                items.append(
                    f"%vh.field {{ {_size_of('field')}, ptr {self.string(item.name)}, "
                    f"ptr {self.descriptors[item.type]}, i64 {item.offset}, ptr {self.string(item.doc)}, "
                    f"ptr {meta_array}, i32 {meta_count} }}"
                )
            fields = self.array("fields", "%vh.field", items)
            methods = self.array("methods", "%vh.function", [self.function(method) for method in record.methods])
            init = "null" if record.init is None else f'@"vh.{record.init}"'
            fini = self.module.fini_function(record.value_type) if record.owns else "null"
            key = "null" if record.key is None else self.descriptors[record.key]
            meta, meta_count = self.metadata(record.metadata)
            enumerators = self.array(
                "enumerators",
                "%vh.enumerator",
                [f"%vh.enumerator {{ ptr {self.string(name)}, i64 {value} }}" for name, value in record.enumerators],
            )
            self.lines.append(
                f"{self.descriptors[record.value_type]} = private unnamed_addr constant %vh.type {{ "
                f"{_size_of('type')}, i32 {record.kind}, i32 {record.alignment}, i64 {record.size}, "
                f"ptr {self.string(record.name)}, i64 {_signed(record.layout_hash)}, ptr {element}, ptr {shape}, "
                f"i32 {len(record.shape)}, i32 {len(items)}, ptr {fields}, ptr {methods}, i32 {len(record.methods)}, "
                f"ptr {init}, ptr {self.string(record.doc)}, ptr {meta}, i32 {meta_count}, ptr {fini}, "
                f"ptr {key}, ptr {enumerators}, i32 {len(record.enumerators)} }}, align 8"
            )
        functions = self.array("functions", "%vh.function", [self.function(item) for item in table.functions])
        types = self.array("types", "ptr", [f"ptr {self.descriptors[value]}" for value in table.listed])
        self.lines.append(
            f'@"{self.module.prefix}module" = constant %vh.module {{ {_size_of("module")}, i32 {REFLECT_VERSION}, '
            f"i32 {len(table.listed)}, ptr {self.string(table.name)}, i64 {_signed(table.hash)}, ptr {types}, "
            f'ptr {functions}, i32 {len(table.functions)}, ptr @"vh.allocator" }}, align 8'
        )


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def emit_reflection(module: "ModuleEmitter") -> None:
    if any(line.startswith("%vh.module") for line in module.type_lines.values()):
        return
    for name, body in _TYPES.items():
        module.type_lines[f"vh.{name}"] = f"%vh.{name} = type {body}"
    table = _LlvmTable(module)
    table.emit(describe(module.programs, module.prefix, module.module_name))
    module.global_lines.extend(table.lines)
    if module.asset_id is not None:
        _emit_registration(module, module.asset_id)


def _emit_registration(module: "ModuleEmitter", asset_id: str) -> None:
    """Static constructor and destructor that register the table with VernonRuntime under the asset id
    (`vernonRuntimeRegisterStaticHostModule`); the record is passed by pointer, which every C ABI agrees on."""

    size = "i32" if module.abi.kind == "wasm32" else "i64"
    name, length = module.string(asset_id)
    module.type_lines["vh.static_module"] = f"%vh.static_module = type {{ ptr, {size}, ptr }}"
    module.global_lines.append(
        f'@"vh.registration" = private constant %vh.static_module {{ ptr {name}, {size} {length}, '
        f'ptr @"{module.prefix}module" }}, align 8'
    )
    for function, runtime in (
        ("vh.register", "vernonRuntimeRegisterStaticHostModule"),
        ("vh.unregister", "vernonRuntimeUnregisterStaticHostModule"),
    ):
        module.declare(runtime, f"declare i32 @{runtime}(ptr)")
        module.function_text[function] = (
            f'define internal void @"{function}"() {{\n'
            f'entry:\n  %status = call i32 @{runtime}(ptr @"vh.registration")\n  ret void\n}}\n'
        )
    for table, function in (("llvm.global_ctors", "vh.register"), ("llvm.global_dtors", "vh.unregister")):
        module.global_lines.append(
            f"@{table} = appending global [1 x {{ i32, ptr, ptr }}] "
            f'[{{ i32, ptr, ptr }} {{ i32 65535, ptr @"{function}", ptr null }}]'
        )


def layout_hash(value_type: ConcreteType, programs: list[HostProgram]) -> int:
    return fnv1a(canonical(value_type, _Fields(programs)))


# The ctypes mirror of reflect.h, for reading a table loaded into this process.


class VhMeta(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_char_p)]


class VhType(ctypes.Structure):
    pass


class VhParam(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("type", ctypes.POINTER(VhType))]


class VhField(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("type", ctypes.POINTER(VhType)),
        ("offset", ctypes.c_uint64),
        ("doc", ctypes.c_char_p),
        ("meta", ctypes.POINTER(VhMeta)),
        ("meta_count", ctypes.c_uint32),
    ]


class VhEnumerator(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("value", ctypes.c_int64)]


class VhCall(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("effect", ctypes.c_uint32),
        ("symbol", ctypes.c_char_p),
        ("type", ctypes.POINTER(VhType)),
        ("meta", ctypes.POINTER(VhMeta)),
        ("meta_count", ctypes.c_uint32),
    ]


class VhFunction(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
        ("params", ctypes.POINTER(VhParam)),
        ("param_count", ctypes.c_uint32),
        ("effect", ctypes.c_uint32),
        ("result", ctypes.POINTER(VhType)),
        ("calls", ctypes.POINTER(VhCall)),
        ("call_count", ctypes.c_uint32),
    ]


VhType._fields_ = [
    ("struct_size", ctypes.c_uint32),
    ("kind", ctypes.c_uint32),
    ("alignment", ctypes.c_uint32),
    ("size", ctypes.c_uint64),
    ("name", ctypes.c_char_p),
    ("layout_hash", ctypes.c_uint64),
    ("element", ctypes.POINTER(VhType)),
    ("shape", ctypes.POINTER(ctypes.c_uint32)),
    ("rank", ctypes.c_uint32),
    ("field_count", ctypes.c_uint32),
    ("fields", ctypes.POINTER(VhField)),
    ("methods", ctypes.POINTER(VhFunction)),
    ("method_count", ctypes.c_uint32),
    ("init", ctypes.c_void_p),
    ("doc", ctypes.c_char_p),
    ("meta", ctypes.POINTER(VhMeta)),
    ("meta_count", ctypes.c_uint32),
    ("fini", ctypes.c_void_p),
    ("key", ctypes.POINTER(VhType)),
    ("enumerators", ctypes.POINTER(VhEnumerator)),
    ("enumerator_count", ctypes.c_uint32),
]


class VhModule(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("type_count", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("hash", ctypes.c_uint64),
        ("types", ctypes.POINTER(ctypes.POINTER(VhType))),
        ("functions", ctypes.POINTER(VhFunction)),
        ("function_count", ctypes.c_uint32),
        ("allocator", ctypes.POINTER(ctypes.c_void_p)),
    ]


def read_module(address: int) -> Any:
    """The `VhModule` at `address`, as a ctypes structure."""

    return VhModule.from_address(address)


class BuiltTable:
    """A reflection table built in this process's memory. Every ctypes object it points into is held here, so
    the table stays valid exactly as long as this object lives."""

    def __init__(
        self,
        table: ModuleRecord,
        address_of: Callable[[str], int],
        fini_of: Callable[[ConcreteType], int] | None = None,
        allocator: ctypes.c_void_p | None = None,
    ):
        """`address_of` gives a function symbol's address; `fini_of` that of a type's fini, for types that own
        memory; `allocator` is the slot the table's `allocator` points at."""

        self._alive: list[Any] = []
        self.allocator = ctypes.c_void_p() if allocator is None else allocator
        self.descriptors: dict[ConcreteType, VhType] = {record.value_type: VhType() for record in table.types}
        self._alive.extend(self.descriptors.values())
        for record in table.types:
            descriptor = self.descriptors[record.value_type]
            descriptor.struct_size = ctypes.sizeof(VhType)
            descriptor.kind, descriptor.alignment, descriptor.size = record.kind, record.alignment, record.size
            descriptor.name = self._text(record.name)
            descriptor.layout_hash = record.layout_hash
            if record.element is not None:
                descriptor.element = ctypes.pointer(self.descriptors[record.element])
            if record.key is not None:
                descriptor.key = ctypes.pointer(self.descriptors[record.key])
            if record.shape:
                descriptor.shape = self._array(ctypes.c_uint32, list(record.shape))
                descriptor.rank = len(record.shape)
            fields = []
            for item in record.fields:
                entry = VhField()
                entry.struct_size = ctypes.sizeof(VhField)
                entry.name, entry.doc = self._text(item.name), self._text(item.doc)
                entry.type = ctypes.pointer(self.descriptors[item.type])
                entry.offset = item.offset
                entry.meta, entry.meta_count = self._metadata(item.metadata)
                fields.append(entry)
            if fields:
                descriptor.fields = self._array(VhField, fields)
                descriptor.field_count = len(fields)
            if record.methods:
                descriptor.methods = self._array(
                    VhFunction, [self._function(item, address_of) for item in record.methods]
                )
                descriptor.method_count = len(record.methods)
            if record.init is not None:
                descriptor.init = address_of(record.init)
            if record.owns and fini_of is not None:
                descriptor.fini = fini_of(record.value_type)
            descriptor.doc = self._text(record.doc)
            descriptor.meta, descriptor.meta_count = self._metadata(record.metadata)
            if record.enumerators:
                descriptor.enumerators = self._array(
                    VhEnumerator, [VhEnumerator(self._text(name), value) for name, value in record.enumerators]
                )
                descriptor.enumerator_count = len(record.enumerators)
        self.module = VhModule()
        self.module.struct_size = ctypes.sizeof(VhModule)
        self.module.version, self.module.type_count = REFLECT_VERSION, len(table.listed)
        self.module.name, self.module.hash = self._text(table.name), table.hash
        if table.listed:
            pointers = [ctypes.pointer(self.descriptors[value]) for value in table.listed]
            self.module.types = self._array(ctypes.POINTER(VhType), pointers)
        if table.functions:
            self.module.functions = self._array(
                VhFunction, [self._function(item, address_of) for item in table.functions]
            )
            self.module.function_count = len(table.functions)
        self.module.allocator = ctypes.pointer(self.allocator)

    @property
    def address(self) -> int:
        return ctypes.addressof(self.module)

    def descriptor(self, value_type: ConcreteType) -> VhType:
        return self.descriptors[value_type]

    def _text(self, value: str | None) -> bytes | None:
        if value is None:
            return None
        data = value.encode("utf-8")
        self._alive.append(data)
        return data

    def _array(self, element: Any, items: list[Any]) -> Any:
        array = (element * len(items))(*items)
        self._alive.append(array)
        return ctypes.cast(array, ctypes.POINTER(element))

    def _metadata(self, entries: Metadata) -> tuple[Any, int]:
        if not entries:
            return None, 0
        items = []
        for key, value in entries:
            entry = VhMeta()
            entry.key, entry.value = self._text(key), self._text(value)
            items.append(entry)
        return self._array(VhMeta, items), len(items)

    def _function(self, record: FunctionRecord, address_of: Callable[[str], int]) -> VhFunction:
        entry = VhFunction()
        entry.struct_size = ctypes.sizeof(VhFunction)
        entry.name, entry.symbol = self._text(record.name), self._text(record.exported)
        entry.address = address_of(record.symbol)
        if record.parameters:
            params = []
            for name, value in record.parameters:
                param = VhParam()
                param.name, param.type = self._text(name), ctypes.pointer(self.descriptors[value])
                params.append(param)
            entry.params = self._array(VhParam, params)
            entry.param_count = len(params)
        entry.effect = record.effect
        if record.result is not None:
            entry.result = ctypes.pointer(self.descriptors[record.result])
        if record.calls:
            calls = []
            for call in record.calls:
                item = VhCall()
                item.struct_size, item.effect = ctypes.sizeof(VhCall), call.effect
                item.symbol = self._text(call.symbol)
                if call.type is not None:
                    item.type = ctypes.pointer(self.descriptors[call.type])
                item.meta, item.meta_count = self._metadata(call.metadata)
                calls.append(item)
            entry.calls = self._array(VhCall, calls)
            entry.call_count = len(calls)
        return entry
