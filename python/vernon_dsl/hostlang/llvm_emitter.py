"""Lowers typed Host programs to textual LLVM IR.

Every Host function, exported or internal, has one convention: it returns an `i32` status, zero on success, takes
scalars and handles by value and aggregates by pointer, and writes its result through a trailing pointer. Each
operand is converted to the type its operation was typed at. Locals live in entry-block allocas that `mem2reg`
promotes.

Scalar rules, the same for every AOT target:

- integers wrap at their width (`add`, `sub`, `mul` without nsw/nuw);
- integer `/` is true division in floating point, `//` and `%` floor for signed and truncate for unsigned
  operands, a zero divisor fails with DIVISION_BY_ZERO, and MIN // -1 wraps to MIN;
- float to integer conversion truncates and saturates, NaN converting to zero (`fptosi.sat`, `fptoui.sat`), and
  float to bool is `fcmp une x, 0`, so NaN is true;
- float arithmetic rounds in the operand width and is never contracted into fused operations;
- float `%` takes the sign of the divisor as CPython does, and every comparison involving NaN is false, `!=`
  included (`fcmp one`), as in device code;
- `min`/`max` select by `olt`/`ogt`, so a NaN first operand yields the second;
- index and range-step failures return INDEX_OUT_OF_RANGE and INVALID_RANGE_STEP.

Transcendental functions call the platform libm and are only as portable as it is.

The C ABI of externs is lowered here because LLVM leaves aggregate argument passing to the frontend. The only
by-value aggregate is a str argument, `{const char *, uint64_t}`; handles are single-field structs that every
supported ABI passes as their uint64_t.

A list or str class field is a `%vh.list` record, `{data, size, capacity}`, and a dict field a `%vh.dict` record,
which only VernonRuntime's dict functions change, given the type's `vh.dict_layout` constant. A TensorStorage field
or local is a `{data, [rank x i64] shape}` record of dense row-major elements. Growing and creating call
VernonRuntime with the allocator in the module's `vh.allocator` slot, and a type owning memory gets a generated
`vh.fini.<type>`; a module using no container references neither.
"""

from __future__ import annotations

import ast
import itertools
import struct
from dataclasses import dataclass

from ..frontend.model import ConcreteType
from ..frontend.type_solver import common_type
from ..language.ast_utils import dotted_name, tensor_update_parts
from ..language.slicing import AxisIndex, AxisRange, AxisSelection, is_slicing, plan_selection
from .format_spec import FormatSpec, parse_format_spec, spec_text, value_kind
from .layout import (
    OWNED_KINDS,
    TRANSFERRED_KINDS,
    StructFields,
    dict_entry,
    layout,
    members,
    storage_rank,
    tensor_count,
    transfer_record,
)
from .model import ExternAbi, HostExtern, HostFunctionInfo, HostProgram, HostStatus
from .values import element_of, key_of, literal_value

_BINARY = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.LShift: "shl",
    ast.RShift: "shr",
}
_COMPARE = {ast.Eq: "eq", ast.NotEq: "noteq", ast.Lt: "lt", ast.LtE: "lte", ast.Gt: "gt", ast.GtE: "gte"}
_FLOAT_PREDICATES = {"eq": "oeq", "noteq": "one", "lt": "olt", "lte": "ole", "gt": "ogt", "gte": "oge"}
_SIGNED_PREDICATES = {"eq": "eq", "noteq": "ne", "lt": "slt", "lte": "sle", "gt": "sgt", "gte": "sge"}
_UNSIGNED_PREDICATES = {"eq": "eq", "noteq": "ne", "lt": "ult", "lte": "ule", "gt": "ugt", "gte": "uge"}
_SWIZZLE = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2, "a": 3}
_CAST_NAMES = frozenset({"bool", "int", "float", "i32", "u32", "i64", "u64", "f32", "f64"})
_LLVM_SCALARS = {"bool": "i1", "i32": "i32", "u32": "i32", "i64": "i64", "u64": "i64", "f32": "float", "f64": "double"}
_BITS = {"i32": 32, "u32": 32, "i64": 64, "u64": 64}
# Locals holding a record the function owns, zeroed at entry and released by every exit.
_OWNED_LOCAL_KINDS = frozenset({"str", "storage", "list", "dict"})
_UNARY_INTRINSICS = {"sqrt": "sqrt", "floor": "floor", "sin": "sin", "cos": "cos", "exp": "exp", "log": "log"}
_LIBM = {"acos": ("acosf", "acos"), "atan2": ("atan2f", "atan2")}


@dataclass(frozen=True)
class TargetAbi:
    """The C calling convention facts the emitter needs, derived from a normalized target triple."""

    triple: str
    kind: str

    @classmethod
    def from_triple(cls, triple: str) -> "TargetAbi":
        arch = triple.split("-", 1)[0]
        if arch in {"x86_64", "amd64"}:
            return cls(triple, "win64" if "windows" in triple or "win32" in triple else "sysv")
        if arch in {"aarch64", "arm64", "arm64e"}:
            return cls(triple, "aarch64")
        if arch == "wasm32":
            return cls(triple, "wasm32")
        raise ValueError(f"Host AOT does not support target '{triple}'")


def _is_float(name: str) -> bool:
    return name in {"f32", "f64"}


def _float_literal(value: float, name: str) -> str:
    # LLVM spells float constants as the hex of the equivalent double; f32 values are exactly representable there.
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _int_literal(value: int, name: str) -> str:
    bits = _BITS[name]
    value &= (1 << bits) - 1
    return str(value - (1 << bits) if value >= 1 << (bits - 1) else value)


def type_descriptor(value_type: ConcreteType) -> str:
    """The reflection descriptor symbol of a named type, which code references before the table is emitted."""

    return f'@"vh.type.{value_type.kind}.{value_type.name}"'


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return (dotted_name(node.func) or "").split(".")[-1]


def _copied(node: ast.expr) -> ast.expr | None:
    """What a copy, `x.copy()` or `deepcopy(x)`, copies; None when `node` is no copy. Typing has resolved method
    calls on classes to helpers, so an attribute call `copy` is a container's."""

    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute) and node.func.attr == "copy" and not node.args:
        return node.func.value
    if dotted_name(node.func) in {"deepcopy", "copy.deepcopy"} and len(node.args) == 1:
        return node.args[0]
    return None


class ModuleEmitter:
    def __init__(self, abi: TargetAbi, prefix: str, asset_id: str | None = None, module_name: str | None = None):
        self.abi = abi
        self.prefix = prefix
        self.asset_id = asset_id
        self.module_name = module_name
        self.type_lines: dict[str, str] = {
            "vh.list": "%vh.list = type { i64, i64, i64 }",
            "vh.dict": "%vh.dict = type { i64, i64, i64, i64, i64, i64 }",
        }
        # The allocator slot the table's `allocator` points at: containers take memory from what it holds.
        self.global_lines: list[str] = ['@"vh.allocator" = internal global ptr null, align 8']
        self.declarations: dict[str, str] = {}
        self.function_text: dict[str, str] = {}
        self.exports: dict[str, str] = {}
        self.strings: dict[bytes, str] = {}
        self.fields_by_struct: dict[str, tuple[tuple[str, ConcreteType], ...]] = {}
        self.programs: list[HostProgram] = []
        # By program identity: the index that qualifies its internal function names, and its public symbols.
        self.program_index: dict[int, tuple[int, frozenset[str]]] = {}

    def fields(self, name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return self.fields_by_struct[name]

    # Types.

    def memory_type(self, value_type: ConcreteType) -> str:
        kind = value_type.kind
        if kind == "scalar":
            return "i8" if value_type.name == "bool" else _LLVM_SCALARS[value_type.name]
        if kind == "handle":
            return "i64"
        if kind == "enum":
            return "i32"
        if kind == "tensor":
            return f"[{tensor_count(value_type)} x {self.memory_type(element_of(value_type))}]"
        if kind == "struct":
            return f'%"struct.{value_type.name}"'
        if kind == "tuple":
            return "{ " + ", ".join(self.memory_type(item) for item in members(value_type, self.fields)) + " }"
        if kind == "dict":
            return "%vh.dict"
        if kind == "storage":
            return f"{{ i64, [{storage_rank(value_type)} x i64] }}"
        if kind in OWNED_KINDS:
            return "%vh.list"
        raise ValueError(f"{value_type.spelling} has no Host representation")

    # Containers.

    def owns_memory(self, value_type: ConcreteType) -> bool:
        if value_type.kind in OWNED_KINDS:
            return True
        return value_type.kind == "struct" and any(self.owns_memory(field) for _, field in self.fields(value_type.name))

    def element_layout(self, value_type: ConcreteType) -> tuple[int, int]:
        """The size and alignment of one element of a list, or of a str's bytes."""

        if value_type.kind == "str":
            return 1, 1
        element = layout(element_of(value_type), self.fields)
        return element.size, element.alignment

    def container_runtime(self, function: str) -> str:
        if function == "reserve":
            self.declare(
                "vernonRuntimeHostListReserve", "declare i32 @vernonRuntimeHostListReserve(ptr, i64, i64, i64, ptr)"
            )
            return "@vernonRuntimeHostListReserve"
        self.declare("vernonRuntimeHostListRelease", "declare void @vernonRuntimeHostListRelease(ptr, i64, i64, ptr)")
        return "@vernonRuntimeHostListRelease"

    def storage_runtime(self, function: str) -> str:
        """VernonRuntime's `vernonRuntimeHostStorage<function>`, declared on first use."""

        result, parameters = {
            "Allocate": ("i32", "ptr, i32, ptr, i64, i64, ptr"),
            "Release": ("void", "ptr, i32, i64, i64, ptr"),
            "Fill": ("void", "ptr, i32, ptr, i64"),
        }[function]
        name = f"vernonRuntimeHostStorage{function}"
        self.declare(name, f"declare {result} @{name}({parameters})")
        return f"@{name}"

    def storage_element_size(self, value_type: ConcreteType) -> int:
        """The size of a storage's element, which is also its alignment."""

        return layout(element_of(value_type), self.fields).size

    def dict_runtime(self, function: str) -> str:
        """VernonRuntime's `vernonRuntimeHostDict<function>`, declared on first use."""

        result, parameters = {
            "Find": ("i64", "ptr, ptr, ptr"),
            "Insert": ("i32", "ptr, ptr, ptr, ptr, ptr"),
            "Remove": ("i32", "ptr, ptr, ptr, ptr"),
            "Clear": ("void", "ptr, ptr, ptr"),
            "Release": ("void", "ptr, ptr, ptr"),
        }[function]
        name = f"vernonRuntimeHostDict{function}"
        self.declare(name, f"declare {result} @{name}({parameters})")
        return f"@{name}"

    def dict_layout(self, value_type: ConcreteType) -> str:
        """The `VernonHostDictLayout` constant of a dict type, which every runtime call on its values passes."""

        name = f'@"vh.dict_layout.{value_type.spelling}"'
        if not any(line.startswith(name + " ") for line in self.global_lines):
            entry = layout(dict_entry(value_type), self.fields)
            key = key_of(value_type)
            values = (
                entry.size,
                entry.alignment,
                entry.offsets[1],
                layout(key, self.fields).size,
                entry.offsets[2],
                1 if key.kind == "str" else 0,
            )
            self.global_lines.append(
                f"{name} = private unnamed_addr constant [6 x i64] [{', '.join(f'i64 {item}' for item in values)}]"
                ", align 8"
            )
        return name

    def str_length_function(self) -> str:
        """`vh.str_len(ptr, i64) -> i64`: the code points of UTF-8 bytes, which Python's len counts: every byte
        but continuation bytes, 0b10xxxxxx."""

        name = '@"vh.str_len"'
        if name not in self.function_text:
            self.function_text[name] = (
                f"define internal i64 {name}(ptr %data, i64 %size) {{\n"
                "entry:\n"
                "  br label %loop\n"
                "loop:\n"
                "  %i = phi i64 [ 0, %entry ], [ %next, %body ]\n"
                "  %count = phi i64 [ 0, %entry ], [ %counted, %body ]\n"
                "  %done = icmp eq i64 %i, %size\n"
                "  br i1 %done, label %exit, label %body\n"
                "body:\n"
                "  %p = getelementptr i8, ptr %data, i64 %i\n"
                "  %byte = load i8, ptr %p\n"
                "  %top = and i8 %byte, -64\n"
                "  %starts = icmp ne i8 %top, -128\n"
                "  %step = zext i1 %starts to i64\n"
                "  %counted = add i64 %count, %step\n"
                "  %next = add i64 %i, 1\n"
                "  br label %loop\n"
                "exit:\n"
                "  ret i64 %count\n"
                "}\n"
            )
        return name

    def intrinsic_declaration(self, name: str, line: str) -> str:
        self.declare(name, line)
        return f"@{name}"

    def str_runtime(self, kind: str) -> str:
        """VernonRuntime's `vernonRuntimeHostStrAppend<kind>`, which f-strings and `+` build strs with."""

        parameters = {
            "": "ptr, ptr, i64, ptr, ptr",
            "Integer": "ptr, i64, i32, ptr, ptr",
            "Float": "ptr, double, ptr, ptr",
        }
        name = f"vernonRuntimeHostStrAppend{kind}"
        self.declare(name, f"declare i32 @{name}({parameters[kind]})")
        return f"@{name}"

    def format_spec(self, spec: FormatSpec) -> str:
        """A `VernonHostFormatSpec` constant for `spec`, or null for the empty spec."""

        if spec.empty:
            return "null"
        values = (
            ord(spec.fill) if spec.fill else 0,
            ord(spec.align) if spec.align else 0,
            ord(spec.sign) if spec.sign else 0,
            ord(spec.type) if spec.type else 0,
            0,
            spec.width,
            spec.precision,
        )
        name = f'@"vh.format.{"_".join(str(value) for value in values)}"'
        if not any(line.startswith(name + " ") for line in self.global_lines):
            self.global_lines.append(
                f"{name} = private unnamed_addr constant {{ i32, i8, i8, i8, i8, i32, i32 }} "
                f"{{ i32 {values[0]}, i8 {values[1]}, i8 {values[2]}, i8 {values[3]}, i8 0, i32 {values[5]}, "
                f"i32 {values[6]} }}, align 4"
            )
        return name

    def str_equal_function(self) -> str:
        """`vh.str_eq(ptr, i64, ptr, i64) -> i1`: byte equality, as a loop rather than memcmp so a module needs no
        C library."""

        name = '@"vh.str_eq"'
        if name not in self.function_text:
            self.function_text[name] = (
                f"define internal i1 {name}(ptr %a, i64 %an, ptr %b, i64 %bn) {{\n"
                "entry:\n"
                "  %same = icmp eq i64 %an, %bn\n"
                "  br i1 %same, label %loop, label %differ\n"
                "loop:\n"
                "  %i = phi i64 [ 0, %entry ], [ %next, %body ]\n"
                "  %done = icmp eq i64 %i, %an\n"
                "  br i1 %done, label %equal, label %body\n"
                "body:\n"
                "  %pa = getelementptr i8, ptr %a, i64 %i\n"
                "  %pb = getelementptr i8, ptr %b, i64 %i\n"
                "  %ca = load i8, ptr %pa\n"
                "  %cb = load i8, ptr %pb\n"
                "  %next = add i64 %i, 1\n"
                "  %match = icmp eq i8 %ca, %cb\n"
                "  br i1 %match, label %loop, label %differ\n"
                "equal:\n"
                "  ret i1 true\n"
                "differ:\n"
                "  ret i1 false\n"
                "}\n"
            )
        return name

    def fini_function(self, value_type: ConcreteType) -> str:
        """The fini of a type that owns memory: it releases a list, str, dict or storage, or every such field of a
        class through its class fields, and leaves the records zeroed."""

        name = f'@"vh.fini.{value_type.spelling}"'
        if name in self.function_text:
            return name
        self.function_text[name] = ""
        lines = ["entry:"]
        if value_type.kind in {"list", "dict"} and self.owns_memory(element_of(value_type)):
            lines.append(f"  call void {self.release_elements_function(value_type)}(ptr %self)")
        if value_type.kind == "dict":
            lines.append('  %allocator = load ptr, ptr @"vh.allocator"')
            lines.append(
                f"  call void {self.dict_runtime('Release')}(ptr %self, ptr {self.dict_layout(value_type)}, "
                "ptr %allocator)"
            )
        elif value_type.kind == "storage":
            size = self.storage_element_size(value_type)
            lines.append('  %allocator = load ptr, ptr @"vh.allocator"')
            lines.append(
                f"  call void {self.storage_runtime('Release')}(ptr %self, i32 {storage_rank(value_type)}, "
                f"i64 {size}, i64 {size}, ptr %allocator)"
            )
        elif value_type.kind in OWNED_KINDS:
            size, alignment = self.element_layout(value_type)
            release = self.container_runtime("release")
            lines.append('  %allocator = load ptr, ptr @"vh.allocator"')
            lines.append(f"  call void {release}(ptr %self, i64 {size}, i64 {alignment}, ptr %allocator)")
        else:
            for index, (_, field) in enumerate(self.fields(value_type.name)):
                if self.owns_memory(field):
                    lines.append(
                        f"  %f{index} = getelementptr {self.memory_type(value_type)}, ptr %self, i32 0, i32 {index}"
                    )
                    lines.append(f"  call void {self.fini_function(field)}(ptr %f{index})")
        lines.append("  ret void")
        self.function_text[name] = f"define internal void {name}(ptr %self) {{\n" + "\n".join(lines) + "\n}\n"
        return name

    def release_elements_function(self, value_type: ConcreteType) -> str:
        """Releases every element of a list, or every live value of a dict, of instances that own memory, leaving
        the container's records for the caller to empty or release."""

        name = f'@"vh.release_elements.{value_type.spelling}"'
        if name in self.function_text:
            return name
        self.function_text[name] = ""
        element = element_of(value_type)
        fini = self.fini_function(element)
        if value_type.kind == "list":
            memory, count = self.memory_type(element), "%size"
            lines = [
                "entry:",
                "  %size_pointer = getelementptr %vh.list, ptr %self, i32 0, i32 1",
                "  %size = load i64, ptr %size_pointer",
                "  %address = load i64, ptr %self",
                "  %data = inttoptr i64 %address to ptr",
                "  br label %loop",
                "loop:",
                "  %index = phi i64 [ 0, %entry ], [ %next, %step ]",
                f"  %done = icmp eq i64 %index, {count}",
                "  br i1 %done, label %exit, label %body",
                "body:",
                f"  %element = getelementptr {memory}, ptr %data, i64 %index",
                f"  call void {fini}(ptr %element)",
                "  br label %step",
            ]
        else:
            memory = self.memory_type(dict_entry(value_type))
            lines = [
                "entry:",
                "  %used_pointer = getelementptr %vh.dict, ptr %self, i32 0, i32 2",
                "  %used = load i64, ptr %used_pointer",
                "  %address = load i64, ptr %self",
                "  %data = inttoptr i64 %address to ptr",
                "  br label %loop",
                "loop:",
                "  %index = phi i64 [ 0, %entry ], [ %next, %step ]",
                "  %done = icmp eq i64 %index, %used",
                "  br i1 %done, label %exit, label %live",
                "live:",
                f"  %entry_pointer = getelementptr {memory}, ptr %data, i64 %index",
                "  %hash = load i64, ptr %entry_pointer",
                "  %removed = icmp eq i64 %hash, -1",
                "  br i1 %removed, label %step, label %body",
                "body:",
                f"  %element = getelementptr {memory}, ptr %entry_pointer, i32 0, i32 2",
                f"  call void {fini}(ptr %element)",
                "  br label %step",
            ]
        lines += ["step:", "  %next = add i64 %index, 1", "  br label %loop", "exit:", "  ret void"]
        self.function_text[name] = f"define internal void {name}(ptr %self) {{\n" + "\n".join(lines) + "\n}\n"
        return name

    def copy_function(self, value_type: ConcreteType) -> str:
        """`vh.copy.<type>(ptr %target, ptr %source) -> i32`, the deep copy of a type that owns memory into zeroed
        memory. It returns a status; a failed copy leaves the target holding what it copied so far, which the
        target's fini releases, so the target is zeroed before each part is copied into it."""

        name = f'@"vh.copy.{value_type.spelling}"'
        if name in self.function_text:
            return name
        self.function_text[name] = ""
        self.declare("llvm.memcpy.p0.p0.i64", "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)")
        self.declare("llvm.memset.p0.i64", "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)")
        memory = self.memory_type(value_type)
        if value_type.kind == "storage":
            lines = self._copy_storage(value_type, memory)
        elif value_type.kind == "dict":
            lines = self._copy_dict(value_type)
        elif value_type.kind in OWNED_KINDS:
            lines = self._copy_list(value_type)
        else:
            lines = [
                "entry:",
                f"  call void @llvm.memcpy.p0.p0.i64(ptr %target, ptr %source, i64 "
                f"{layout(value_type, self.fields).size}, i1 false)",
            ]
            owning = [
                (index, field)
                for index, (_, field) in enumerate(self.fields(value_type.name))
                if self.owns_memory(field)
            ]
            # Every owning field is emptied before any is copied: until then it holds the source's memory.
            for index, field in owning:
                lines.append(f"  %t{index} = getelementptr {memory}, ptr %target, i32 0, i32 {index}")
                lines.append(
                    f"  call void @llvm.memset.p0.i64(ptr %t{index}, i8 0, i64 {layout(field, self.fields).size}, "
                    "i1 false)"
                )
            for index, field in owning:
                lines += [
                    f"  %s{index} = getelementptr {memory}, ptr %source, i32 0, i32 {index}",
                    f"  %status{index} = call i32 {self.copy_function(field)}(ptr %t{index}, ptr %s{index})",
                    f"  %failed{index} = icmp ne i32 %status{index}, 0",
                    f"  br i1 %failed{index}, label %fail{index}, label %copied{index}",
                    f"fail{index}:",
                    f"  ret i32 %status{index}",
                    f"copied{index}:",
                ]
            lines.append("  ret i32 0")
        self.function_text[name] = (
            f"define internal i32 {name}(ptr %target, ptr %source) {{\n" + "\n".join(lines) + "\n}\n"
        )
        return name

    def _copy_storage(self, value_type: ConcreteType, memory: str) -> list[str]:
        rank = storage_rank(value_type)
        size = self.storage_element_size(value_type)
        lines = [
            "entry:",
            f"  %shape = getelementptr {memory}, ptr %source, i32 0, i32 1",
            '  %allocator = load ptr, ptr @"vh.allocator"',
            f"  %status = call i32 {self.storage_runtime('Allocate')}(ptr %target, i32 {rank}, ptr %shape, "
            f"i64 {size}, i64 {size}, ptr %allocator)",
            "  %failed = icmp ne i32 %status, 0",
            "  br i1 %failed, label %fail, label %copy",
            "fail:",
            "  ret i32 %status",
            "copy:",
        ]
        count = f"{size}"
        for axis in range(rank):
            lines.append(f"  %extent{axis}.pointer = getelementptr [{rank} x i64], ptr %shape, i32 0, i32 {axis}")
            lines.append(f"  %extent{axis} = load i64, ptr %extent{axis}.pointer")
            lines.append(f"  %bytes{axis} = mul i64 {count}, %extent{axis}")
            count = f"%bytes{axis}"
        lines += [
            "  %source.address = load i64, ptr %source",
            "  %source.data = inttoptr i64 %source.address to ptr",
            "  %target.address = load i64, ptr %target",
            "  %target.data = inttoptr i64 %target.address to ptr",
            f"  call void @llvm.memcpy.p0.p0.i64(ptr %target.data, ptr %source.data, i64 {count}, i1 false)",
            "  ret i32 0",
        ]
        return lines

    def _copy_list(self, value_type: ConcreteType) -> list[str]:
        """A list or str: reserved at the source's size, then its elements copied. Elements that own memory are
        copied one at a time, each counted in the target's size before it is copied into."""

        size, alignment = self.element_layout(value_type)
        lines = [
            "entry:",
            "  %source.size.pointer = getelementptr %vh.list, ptr %source, i32 0, i32 1",
            "  %size = load i64, ptr %source.size.pointer",
            '  %allocator = load ptr, ptr @"vh.allocator"',
            f"  %status = call i32 {self.container_runtime('reserve')}(ptr %target, i64 %size, i64 {size}, "
            f"i64 {alignment}, ptr %allocator)",
            "  %failed = icmp ne i32 %status, 0",
            "  br i1 %failed, label %fail, label %reserved",
            "fail:",
            "  ret i32 %status",
            "reserved:",
            "  %source.address = load i64, ptr %source",
            "  %source.data = inttoptr i64 %source.address to ptr",
            "  %target.address = load i64, ptr %target",
            "  %target.data = inttoptr i64 %target.address to ptr",
            "  %target.size.pointer = getelementptr %vh.list, ptr %target, i32 0, i32 1",
        ]
        element = None if value_type.kind == "str" else element_of(value_type)
        if element is None or not self.owns_memory(element):
            lines += [
                f"  %bytes = mul i64 %size, {size}",
                "  call void @llvm.memcpy.p0.p0.i64(ptr %target.data, ptr %source.data, i64 %bytes, i1 false)",
                "  store i64 %size, ptr %target.size.pointer",
                "  ret i32 0",
            ]
            return lines
        memory = self.memory_type(element)
        return lines + [
            "  br label %loop",
            "loop:",
            "  %index = phi i64 [ 0, %reserved ], [ %next, %body ]",
            "  %done = icmp eq i64 %index, %size",
            "  br i1 %done, label %exit, label %body",
            "body:",
            f"  %target.element = getelementptr {memory}, ptr %target.data, i64 %index",
            f"  %source.element = getelementptr {memory}, ptr %source.data, i64 %index",
            f"  call void @llvm.memset.p0.i64(ptr %target.element, i8 0, i64 {size}, i1 false)",
            "  %next = add i64 %index, 1",
            "  store i64 %next, ptr %target.size.pointer",
            f"  %element.status = call i32 {self.copy_function(element)}(ptr %target.element, ptr %source.element)",
            "  %element.failed = icmp ne i32 %element.status, 0",
            "  br i1 %element.failed, label %element.fail, label %loop",
            "element.fail:",
            "  ret i32 %element.status",
            "exit:",
            "  ret i32 0",
        ]

    def _copy_dict(self, value_type: ConcreteType) -> list[str]:
        """Inserts each live entry's key in insertion order, then copies its value into the entry Insert zeroed.
        A str key's VhList record starts with the `{data, size}` view Insert takes."""

        entry = self.memory_type(dict_entry(value_type))
        element = element_of(value_type)
        lines = [
            "entry:",
            "  %out = alloca i64",
            "  %used.pointer = getelementptr %vh.dict, ptr %source, i32 0, i32 2",
            "  %used = load i64, ptr %used.pointer",
            "  %source.address = load i64, ptr %source",
            "  %source.data = inttoptr i64 %source.address to ptr",
            '  %allocator = load ptr, ptr @"vh.allocator"',
            "  br label %loop",
            "loop:",
            "  %index = phi i64 [ 0, %entry ], [ %next, %step ]",
            "  %done = icmp eq i64 %index, %used",
            "  br i1 %done, label %exit, label %live",
            "live:",
            f"  %source.entry = getelementptr {entry}, ptr %source.data, i64 %index",
            "  %hash = load i64, ptr %source.entry",
            "  %removed = icmp eq i64 %hash, -1",
            "  br i1 %removed, label %step, label %body",
            "body:",
            f"  %key = getelementptr {entry}, ptr %source.entry, i32 0, i32 1",
            f"  %status = call i32 {self.dict_runtime('Insert')}(ptr %target, ptr %key, "
            f"ptr {self.dict_layout(value_type)}, ptr %allocator, ptr %out)",
            "  %failed = icmp ne i32 %status, 0",
            "  br i1 %failed, label %fail, label %inserted",
            "fail:",
            "  ret i32 %status",
            "inserted:",
            "  %target.index = load i64, ptr %out",
            "  %target.address = load i64, ptr %target",
            "  %target.data = inttoptr i64 %target.address to ptr",
            f"  %target.entry = getelementptr {entry}, ptr %target.data, i64 %target.index",
            f"  %target.value = getelementptr {entry}, ptr %target.entry, i32 0, i32 2",
            f"  %source.value = getelementptr {entry}, ptr %source.entry, i32 0, i32 2",
        ]
        if self.owns_memory(element):
            lines += [
                f"  %value.status = call i32 {self.copy_function(element)}(ptr %target.value, ptr %source.value)",
                "  %value.failed = icmp ne i32 %value.status, 0",
                "  br i1 %value.failed, label %value.fail, label %step",
                "value.fail:",
                "  ret i32 %value.status",
            ]
        else:
            size = layout(element, self.fields).size
            lines += [
                f"  call void @llvm.memcpy.p0.p0.i64(ptr %target.value, ptr %source.value, i64 {size}, i1 false)",
                "  br label %step",
            ]
        return lines + ["step:", "  %next = add i64 %index, 1", "  br label %loop", "exit:", "  ret i32 0"]

    def register_type(self, value_type: ConcreteType) -> str:
        if value_type.kind == "scalar" and value_type.name == "bool":
            return "i1"
        return self.memory_type(value_type)

    def add_program(self, program: HostProgram) -> None:
        for name, host_struct in program.structs.items():
            existing = self.fields_by_struct.get(name)
            if existing is not None and existing != host_struct.fields:
                raise ValueError(f"Host programs disagree on the fields of Struct '{name}'")
            self.fields_by_struct[name] = host_struct.fields
        for name, host_struct in program.structs.items():
            body = ", ".join(self.memory_type(field) for _, field in host_struct.fields) or "i8"
            self.type_lines[name] = f'%"struct.{name}" = type {{ {body} }}'
        self.program_index[id(program)] = (len(self.programs), program.public_symbols())
        for symbol, info in program.functions.items():
            name = self.function_name(program, symbol)
            if name not in self.function_text:
                self.function_text[name] = _FunctionEmitter(self, program, info).emit()
        for name, info in program.exported():
            exported = self.prefix + name
            if exported not in self.exports:
                self.exports[exported] = self._export(info, exported)
        self.programs.append(program)

    def function_name(self, program: HostProgram, symbol: str) -> str:
        """The LLVM name of a function of `program`. Public symbols keep their name, which reflection and exports
        refer to; every other function is qualified by its program, because each source file names its own
        helpers and the frontend gives an imported one the same `__vernon_shared__` name in every program."""

        index, public = self.program_index[id(program)]
        return f'@"vh.{symbol}"' if symbol in public else f'@"vh.{index}.{symbol}"'

    def _export(self, info: HostFunctionInfo, exported: str) -> str:
        parameters = self.parameter_types(info.parameters, info.result)
        declared = ", ".join(f"{kind} %a{index}" for index, kind in enumerate(parameters))
        forwarded = ", ".join(f"{kind.split(' ')[0]} %a{index}" for index, kind in enumerate(parameters))
        return (
            f'define i32 @"{exported}"({declared}) {{\n'
            f'entry:\n  %status = call i32 @"vh.{info.symbol}"({forwarded})\n  ret i32 %status\n}}\n'
        )

    def parameter_types(
        self, parameters: tuple[tuple[str, ConcreteType], ...], result: ConcreteType | None
    ) -> list[str]:
        lowered = []
        for _, value_type in parameters:
            if value_type.kind == "scalar" and value_type.name == "bool":
                lowered.append("i1 zeroext")
            elif value_type.kind in {"scalar", "handle", "enum"}:
                lowered.append(self.register_type(value_type))
            else:
                lowered.append("ptr")
        if result is not None:
            lowered.append("ptr")
        return lowered

    def declare(self, name: str, line: str) -> None:
        self.declarations.setdefault(name, line)

    def intrinsic(self, base: str, value_type: str, arity: int = 1) -> str:
        suffix = {"float": "f32", "double": "f64", "i32": "i32", "i64": "i64"}[value_type]
        name = f"llvm.{base}.{suffix}"
        operands = ", ".join([value_type] * arity)
        self.declare(name, f"declare {value_type} @{name}({operands})")
        return f"@{name}"

    def string(self, value: str) -> tuple[str, int]:
        data = value.encode("utf-8")
        name = self.strings.get(data)
        if name is None:
            name = f"@.vh.str.{len(self.strings)}"
            self.strings[data] = name
            escaped = "".join(
                chr(byte) if 0x20 <= byte < 0x7F and byte not in {0x22, 0x5C} else f"\\{byte:02X}" for byte in data
            )
            self.global_lines.append(
                f'{name} = private unnamed_addr constant [{len(data) + 1} x i8] c"{escaped}\\00", align 1'
            )
        return name, len(data)

    def extern_parameters(self, extern: HostExtern) -> list[tuple[str, str]]:
        """(lowering, LLVM parameter text) per extern parameter, in C ABI order, plus the status out pointer."""

        lowered: list[tuple[str, str]] = []
        free_integer_registers = 6
        parameters: list[ConcreteType | None] = [value_type for _, value_type in extern.parameters]
        if extern.type_argument is not None:
            parameters.insert(extern.type_argument[0], None)
        for value_type in parameters:
            if value_type is None:
                lowered.append(("type", "ptr"))
                free_integer_registers = max(free_integer_registers - 1, 0)
                continue
            if value_type.kind in TRANSFERRED_KINDS:
                # A str and a VhSpan are both `{pointer, uint64_t}`, passed alike.
                if self.abi.kind == "sysv":
                    # An aggregate of two INTEGER eightbytes goes in registers only if both fit; otherwise on the stack.
                    if free_integer_registers >= 2:
                        lowered.append(("str_pair", "ptr, i64"))
                        free_integer_registers -= 2
                    else:
                        lowered.append(("str_byval", "ptr byval({ ptr, i64 }) align 8"))
                elif self.abi.kind == "aarch64":
                    lowered.append(("str_array", "[2 x i64]"))
                elif self.abi.kind == "wasm32":
                    lowered.append(("str_byval", "ptr byval({ ptr, i64 }) align 8"))
                else:
                    lowered.append(("str_indirect", "ptr"))
                continue
            if value_type.kind == "scalar" and value_type.name == "bool":
                lowered.append(("value", "i1 zeroext"))
            elif value_type.kind in {"scalar", "handle", "enum"}:
                lowered.append(("value", self.register_type(value_type)))
            else:
                lowered.append(("pointer", "ptr"))
            if not (value_type.kind == "scalar" and _is_float(value_type.name)):
                free_integer_registers = max(free_integer_registers - 1, 0)
        if extern.abi == ExternAbi.STATUS and extern.result is not None:
            lowered.append(("out", "ptr"))
            if extern.result.kind in TRANSFERRED_KINDS:
                lowered.append(("allocator", "ptr"))
        return lowered

    def declare_extern(self, extern: HostExtern) -> None:
        parameters = ", ".join(text for _, text in self.extern_parameters(extern))
        if extern.abi == ExternAbi.STATUS:
            result = "i32"
        elif extern.result is None:
            result = "void"
        elif extern.result.kind == "scalar" and extern.result.name == "bool":
            result = "zeroext i1"
        else:
            result = self.register_type(extern.result)
        self.declare(extern.symbol, f'declare {result} @"{extern.symbol}"({parameters})')

    def text(self) -> str:
        from .reflect import emit_reflection

        emit_reflection(self)
        parts = [*self.type_lines.values(), "", *self.global_lines, "", *self.declarations.values(), ""]
        parts.extend(self.function_text.values())
        parts.extend(self.exports.values())
        return "\n".join(parts) + "\n"


class _FunctionEmitter:
    def __init__(self, module: ModuleEmitter, program: HostProgram, info: HostFunctionInfo):
        self.module = module
        self.program = program
        self.info = info
        self.lines: list[str] = []
        self.allocas: list[str] = []
        self.counter = 0
        self.slots: dict[str, str] = {}
        # Class-typed parameters: the instance pointer itself, never copied into a slot.
        self.places: dict[str, str] = {}
        # Each loop's continue and break labels, with the depth of `scopes` where its body starts.
        self.loops: list[tuple[str, str, int]] = []
        # The open `with` scopes, innermost last: each closer and the slot holding the handle it closes.
        self.scopes: list[tuple[HostExtern, str]] = []
        self.terminated = False
        # Expressions already evaluated, by node identity: an augmented assignment's indices, which Python
        # evaluates once although the target is both read and written.
        self.evaluated: dict[int, str] = {}
        # The records the function owns, with their types: its str and storage locals and what its f-strings and +
        # build, which every exit releases. A function owning any returns through `vh.exit`, with its status in
        # `status_slot`.
        self.owned: list[tuple[str, ConcreteType]] = []
        self.cleanup = False
        self.status_slot = ""

    # Blocks and instructions.

    def temporary(self) -> str:
        self.counter += 1
        return f"%t{self.counter}"

    def label(self, hint: str) -> str:
        self.counter += 1
        return f"{hint}{self.counter}"

    def start(self, label: str) -> None:
        self.lines.append(f"{label}:")
        self.terminated = False

    def instruction(self, text: str) -> None:
        if self.terminated:
            # Code after break, continue, or return still has to sit in a block; it is unreachable.
            self.start(self.label("dead"))
        self.lines.append(f"  {text}")

    def value(self, text: str) -> str:
        result = self.temporary()
        self.instruction(f"{result} = {text}")
        return result

    def terminate(self, text: str) -> None:
        self.instruction(text)
        self.terminated = True

    def alloca(self, llvm_type: str) -> str:
        name = self.temporary()
        self.allocas.append(f"  {name} = alloca {llvm_type}")
        return name

    def branch_on(self, condition: str, true_label: str, false_label: str) -> None:
        self.terminate(f"br i1 {condition}, label %{true_label}, label %{false_label}")

    def jump(self, label: str) -> None:
        self.terminate(f"br label %{label}")

    def fail_if(self, condition: str, status: int) -> None:
        failed, passed = self.label("fail"), self.label("ok")
        self.branch_on(condition, failed, passed)
        self.start(failed)
        self.leave(status)
        self.start(passed)

    def propagate(self, status: str) -> None:
        failed = self.value(f"icmp ne i32 {status}, 0")
        bad, passed = self.label("fail"), self.label("ok")
        self.branch_on(failed, bad, passed)
        self.start(bad)
        self.leave(status)
        self.start(passed)

    def leave(self, status: str | int, scopes: list[tuple[HostExtern, str]] | None = None) -> None:
        """Returns `status`, through the exit block that releases the function's records when it owns any. The open
        `scopes` close first, innermost first: on a failure each closes and the failure stands; on success a closer's
        failure becomes the result, and the scopes outside it close as on any failure."""

        scopes = self.scopes if scopes is None else scopes
        if status == 0:
            self.close_scopes(scopes, 0)
        else:
            for closer, slot in reversed(scopes):
                self.call_closer(closer, slot)
        if not self.cleanup:
            self.terminate(f"ret i32 {status}")
            return
        self.instruction(f"store i32 {status}, ptr {self.status_slot}")
        self.jump("vh.exit")

    def close_scopes(self, scopes: list[tuple[HostExtern, str]], depth: int) -> None:
        """Closes `scopes[depth:]` innermost first on a successful path; a closer's failure leaves the function with
        its status, closing the scopes outside it."""

        for index in reversed(range(depth, len(scopes))):
            status = self.call_closer(*scopes[index])
            if status is not None:
                failed = self.value(f"icmp ne i32 {status}, 0")
                bad, passed = self.label("fail"), self.label("ok")
                self.branch_on(failed, bad, passed)
                self.start(bad)
                self.leave(status, scopes[:index])
                self.start(passed)

    def call_closer(self, closer: HostExtern, slot: str) -> str | None:
        """Calls `closer` on the handle in `slot`; its status, when its abi returns one."""

        module = self.module
        module.declare_extern(closer)
        handle_type = closer.parameters[0][1]
        handle = self.load(handle_type, slot)
        lowering, text = next(item for item in module.extern_parameters(closer) if item[0] != "type")
        argument = f"{text} {handle}" if lowering == "value" else f"ptr {self.spill(handle, handle_type)}"
        if closer.abi == ExternAbi.STATUS:
            return self.value(f'call i32 @"{closer.symbol}"({argument})')
        self.instruction(f'call void @"{closer.symbol}"({argument})')
        return None

    def owned_record(self, value_type: ConcreteType) -> str:
        """A str or storage record the function owns: empty from entry, so every exit can release it."""

        memory = self.module.memory_type(value_type)
        record = self.alloca(memory)
        self.allocas.append(f"  store {memory} zeroinitializer, ptr {record}")
        self.owned.append((record, value_type))
        return record

    def owned_str(self) -> str:
        return self.owned_record(ConcreteType("str", "str"))

    # Class instances: a function owns the instances its owned locals hold and the temporaries that new instances
    # are built in. Each starts zeroed and is released at every exit; a move zeroes its source, so releasing it again
    # does nothing.

    def instance_slot(self, value_type: ConcreteType) -> str:
        if self.module.owns_memory(value_type):
            return self.owned_record(value_type)
        memory = self.module.memory_type(value_type)
        slot = self.alloca(memory)
        self.allocas.append(f"  store {memory} zeroinitializer, ptr {slot}")
        return slot

    def size_of(self, value_type: ConcreteType) -> int:
        return layout(value_type, self.module.fields).size

    def release(self, address: str, value_type: ConcreteType) -> None:
        if self.module.owns_memory(value_type):
            self.instruction(f"call void {self.module.fini_function(value_type)}(ptr {address})")

    def move(self, target: str, source: str, value_type: ConcreteType) -> None:
        """Moves an instance into memory holding none, leaving the source zeroed: empty, with nothing to release."""

        size = self.size_of(value_type)
        self.memcpy(target, source, size)
        self.module.declare("llvm.memset.p0.i64", "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)")
        self.instruction(f"call void @llvm.memset.p0.i64(ptr {source}, i8 0, i64 {size}, i1 false)")

    def copy_into(self, target: str, source: str, value_type: ConcreteType) -> None:
        """Deep-copies `source` into zeroed memory at `target`; a failure leaves a partial copy that releasing the
        target frees."""

        if self.module.owns_memory(value_type):
            self.propagate(self.value(f"call i32 {self.module.copy_function(value_type)}(ptr {target}, ptr {source})"))
        else:
            self.memcpy(target, source, self.size_of(value_type))

    def assign_copy(self, target: ast.expr, value: ast.expr) -> None:
        """Copies a list, dict or storage into a record of its own, then releases the target's and moves the copy
        in, so the copy may read the container it replaces and a failed one leaves the target as it was."""

        value_type = self.local_or_expression_type(value)
        copied = _copied(value)
        assert copied is not None
        source = self.storage_record(copied) if value_type.kind == "storage" else self.container(copied)
        built = self.owned_record(value_type)
        self.copy_into(built, source, value_type)
        record = self.storage_record(target) if value_type.kind == "storage" else self.container(target)
        self.release(record, value_type)
        self.move(record, built, value_type)

    def creates(self, node: ast.expr) -> bool:
        """Whether `node` evaluates to a new instance: a constructor, a call returning one, a pop or a deepcopy."""

        return isinstance(node, ast.Call) and self.program.is_place(self.info.expression_type(node))

    def instance(self, node: ast.expr) -> str:
        """The address of the new instance `node` evaluates to, which the caller moves or releases: an owned local
        moved out, or a temporary built by a constructor, a call or a pop."""

        if isinstance(node, ast.Name):
            return self.slots[node.id]
        assert isinstance(node, ast.Call)
        value_type = self.info.expression_type(node)
        temporary = self.instance_slot(value_type)
        if (copied := _copied(node)) is not None:
            source = self.instance(copied) if self.creates(copied) else self.place_address(copied)
            assert source is not None
            self.copy_into(temporary, source, value_type)
            return temporary
        if isinstance(node.func, ast.Attribute):
            container = node.func.value
            container_type = self.info.expression_type(container)
            if container_type.kind == "list":
                record = self.container(container)
                size = self.list_size(record)
                self.fail_if(self.value(f"icmp eq i64 {size}, 0"), HostStatus.INDEX_OUT_OF_RANGE)
                last = self.value(f"sub i64 {size}, 1")
                self.memcpy(temporary, self.list_element(record, container_type, last), self.size_of(value_type))
                self.instruction(f"store i64 {last}, ptr {self.list_size_pointer(record)}")
                return temporary
            key = self.dict_key(node.args[0], container_type)
            index = self.dict_call("Find", container, key)
            self.fail_if(self.value(f"icmp slt i64 {index}, 0"), HostStatus.KEY_NOT_FOUND)
            self.memcpy(temporary, self.dict_member(container, index, 2), self.size_of(value_type))
            self.dict_call("Remove", container, key, self.allocator())
            return temporary
        name = _call_name(node)
        if name in self.program.functions:
            self.function_call(self.program.functions[name], node, temporary)
            return temporary
        init = self.module.function_name(self.program, self.program.classes[name].init)
        self.propagate(self.value(f"call i32 {init}(ptr {temporary})"))
        self.construct_fields(temporary, value_type, node)
        return temporary

    def construct_fields(self, address: str, class_type: ConcreteType, node: ast.Call) -> None:
        """Assigns the `field=value` keywords of a constructor, in order, into the new instance at `address`, which
        nothing else can reach yet. A failing value leaves the instance for the caller's release."""

        fields = self.module.fields(class_type.name)
        names = [name for name, _ in fields]
        memory = self.module.memory_type(class_type)
        for position, keyword in enumerate(node.keywords):
            assert keyword.arg is not None
            field_type = dict(fields)[keyword.arg]
            field = self.value(f"getelementptr {memory}, ptr {address}, i32 0, i32 {names.index(keyword.arg)}")
            if field_type.kind == "str":
                later = [item.value for item in node.keywords[position + 1 :]]
                data, size = self.stable_str_operand(keyword.value, later)
                self.copy_str(field, data, size)
            elif self.program.is_place(field_type):
                source = self.instance(keyword.value)
                self.release(field, field_type)
                self.move(field, source, field_type)
            else:
                self.store(self.typed(keyword.value, field_type), field_type, field)

    def assign_instance(self, target: ast.expr, value: ast.expr) -> None:
        """Binds a borrowed name, or moves a new instance into a place, releasing what the place held. The instance
        is built first, as Python evaluates the value before the target."""

        if isinstance(target, ast.Name) and self.info.class_locals.get(target.id) != "owned":
            address = self.place_address(value)
            assert address is not None
            self.instruction(f"store ptr {address}, ptr {self.slots[target.id]}")
            return
        value_type = self.info.expression_type(value)
        source = self.instance(value)
        address = self.slots[target.id] if isinstance(target, ast.Name) else self.place_address(target, True)
        assert address is not None
        self.release(address, value_type)
        self.move(address, source, value_type)

    # Memory and aggregates.

    def to_memory(self, value: str, value_type: ConcreteType) -> str:
        if value_type.kind == "scalar" and value_type.name == "bool":
            return self.value(f"zext i1 {value} to i8")
        return value

    def from_memory(self, value: str, value_type: ConcreteType) -> str:
        if value_type.kind == "scalar" and value_type.name == "bool":
            return self.value(f"icmp ne i8 {value}, 0")
        return value

    def store(self, value: str, value_type: ConcreteType, pointer: str) -> None:
        memory = self.to_memory(value, value_type)
        self.instruction(f"store {self.module.memory_type(value_type)} {memory}, ptr {pointer}")

    def load(self, value_type: ConcreteType, pointer: str) -> str:
        loaded = self.value(f"load {self.module.memory_type(value_type)}, ptr {pointer}")
        return self.from_memory(loaded, value_type)

    def spill(self, value: str, value_type: ConcreteType) -> str:
        pointer = self.alloca(self.module.memory_type(value_type))
        self.store(value, value_type, pointer)
        return pointer

    def member_types(self, value_type: ConcreteType) -> tuple[ConcreteType, ...]:
        if value_type.kind == "tensor":
            return (element_of(value_type),) * tensor_count(value_type)
        return members(value_type, self.module.fields)

    def extract(self, aggregate: str, value_type: ConcreteType, index: int) -> str:
        member = self.member_types(value_type)[index]
        item = self.value(f"extractvalue {self.module.memory_type(value_type)} {aggregate}, {index}")
        return self.from_memory(item, member)

    def build(self, value_type: ConcreteType, items: list[str]) -> str:
        memory = self.module.memory_type(value_type)
        member_types = self.member_types(value_type)
        if not member_types:
            return "zeroinitializer"
        aggregate = "poison"
        for index, (item, member) in enumerate(zip(items, member_types, strict=True)):
            stored = self.to_memory(item, member)
            aggregate = self.value(
                f"insertvalue {memory} {aggregate}, {self.module.memory_type(member)} {stored}, {index}"
            )
        return aggregate

    # Conversions.

    def convert(self, value: str, source: ConcreteType, target: ConcreteType) -> str:
        if source == target:
            return value
        if target.kind == "tensor":
            target_element = element_of(target)
            if source.kind == "scalar":
                item = self.convert_scalar(value, source.name, target_element.name)
                return self.build(target, [item] * tensor_count(target))
            source_element = element_of(source)
            return self.build(
                target,
                [
                    self.convert_scalar(self.extract(value, source, index), source_element.name, target_element.name)
                    for index in range(tensor_count(target))
                ],
            )
        if target.kind == "scalar" and source.kind == "scalar":
            return self.convert_scalar(value, source.name, target.name)
        if target.kind == "scalar" and source.kind == "enum":
            return self.convert_scalar(value, "i32", target.name)
        if target.kind == "tuple" and source.kind == "tuple":
            sources = self.member_types(source)
            return self.build(
                target,
                [
                    self.convert(self.extract(value, source, index), item_source, item_target)
                    for index, (item_source, item_target) in enumerate(
                        zip(sources, self.member_types(target), strict=True)
                    )
                ],
            )
        raise TypeError(f"cannot convert {source.spelling} to {target.spelling}")

    def convert_scalar(self, value: str, source: str, target: str) -> str:
        if source == target:
            return value
        source_type, target_type = _LLVM_SCALARS[source], _LLVM_SCALARS[target]
        if target == "bool":
            if _is_float(source):
                return self.value(f"fcmp une {source_type} {value}, 0.0")
            return self.value(f"icmp ne {source_type} {value}, 0")
        if _is_float(target):
            if source == "bool":
                return self.value(f"uitofp i1 {value} to {target_type}")
            if _is_float(source):
                operation = "fpext" if target == "f64" else "fptrunc"
                return self.value(f"{operation} {source_type} {value} to {target_type}")
            operation = "uitofp" if source.startswith("u") else "sitofp"
            return self.value(f"{operation} {source_type} {value} to {target_type}")
        if _is_float(source):
            base = "fptoui.sat" if target.startswith("u") else "fptosi.sat"
            suffix = "f32" if source == "f32" else "f64"
            name = f"llvm.{base}.{target_type}.{suffix}"
            self.module.declare(name, f"declare {target_type} @{name}({source_type})")
            return self.value(f"call {target_type} @{name}({source_type} {value})")
        if source == "bool":
            return self.value(f"zext i1 {value} to {target_type}")
        if source_type == target_type:
            return value
        if _BITS[target] > _BITS[source]:
            operation = "zext" if source.startswith("u") else "sext"
            return self.value(f"{operation} {source_type} {value} to {target_type}")
        return self.value(f"trunc {source_type} {value} to {target_type}")

    # Functions and statements.

    def emit(self) -> str:
        info = self.info
        module = self.module
        parameters = module.parameter_types(info.parameters, info.result)
        declared = ", ".join(f"{kind} %a{index}" for index, kind in enumerate(parameters))
        self.start("entry")
        self.cleanup = _owns_records(info, self.program)
        if self.cleanup:
            self.status_slot = self.alloca("i32")
        for index, (name, value_type) in enumerate(info.parameters):
            if self.program.is_place(value_type):
                self.places[name] = f"%a{index}"
        for name, value_type in info.local_types.items():
            if name in self.places:
                continue
            class_local = info.class_locals.get(name)
            if class_local == "owned":
                self.slots[name] = self.instance_slot(value_type)
            elif class_local is not None:
                # A borrowed name or a loop's element holds the address of the instance it refers to.
                self.slots[name] = self.alloca("ptr")
            elif value_type.kind in _OWNED_LOCAL_KINDS:
                self.slots[name] = self.owned_record(value_type)
            else:
                self.slots[name] = self.alloca(module.memory_type(value_type))
        if any(host_class.init == info.symbol for host_class in self.program.classes.values()):
            # Fields without a default, and a class field's padding, start zeroed.
            self_type = info.parameters[0][1]
            self.instruction(f"store {module.memory_type(self_type)} zeroinitializer, ptr {self.places['self']}")
            self.allocate_static_storages(self_type)
        for index, (name, value_type) in enumerate(info.parameters):
            if name in self.places:
                continue
            if value_type.kind in {"scalar", "handle", "enum"}:
                self.store(f"%a{index}", value_type, self.slots[name])
            elif value_type.kind == "str":
                # The caller's view may read a field this function assigns, so the parameter is a copy.
                data = self.value(f"load ptr, ptr %a{index}")
                size_pointer = self.value(f"getelementptr inbounds {{ ptr, i64 }}, ptr %a{index}, i32 0, i32 1")
                self.copy_str(self.slots[name], data, self.value(f"load i64, ptr {size_pointer}"))
            else:
                memory = module.memory_type(value_type)
                copied = self.value(f"load {memory}, ptr %a{index}")
                self.instruction(f"store {memory} {copied}, ptr {self.slots[name]}")
        self.block(info.source.body)
        if not self.terminated:
            self.leave(0)
        if self.cleanup:
            self.start("vh.exit")
            for record, value_type in self.owned:
                self.instruction(f"call void {module.fini_function(value_type)}(ptr {record})")
            self.terminate(f"ret i32 {self.value(f'load i32, ptr {self.status_slot}')}")
        body = [self.lines[0], *self.allocas, *self.lines[1:]]
        name = self.module.function_name(self.program, info.symbol)
        return f"define internal i32 {name}({declared}) {{\n" + "\n".join(body) + "\n}\n"

    @property
    def out_pointer(self) -> str:
        return f"%a{len(self.info.parameters)}"

    def block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, statement: ast.stmt) -> None:
        info = self.info
        # `[]` and `{}` empty a container field, or (re)declare a container local, which Python makes a new empty one.
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(statement.value, ast.List):
            self.list_clear(statement.targets[0] if isinstance(statement, ast.Assign) else statement.target)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(statement.value, ast.Dict):
            self.dict_clear(statement.targets[0] if isinstance(statement, ast.Assign) else statement.target)
        elif (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and statement.value is not None
            and self.program.is_place(info.expression_type(statement.value))
        ):
            self.assign_instance(
                statement.targets[0] if isinstance(statement, ast.Assign) else statement.target, statement.value
            )
        elif (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and (extern := self.transferring_extern(statement.value)) is not None
        ):
            assert isinstance(statement.value, ast.Call)
            self.assign_extern_result(
                statement.targets[0] if isinstance(statement, ast.Assign) else statement.target,
                extern,
                statement.value,
            )
        elif isinstance(statement, ast.Expr) and self.creates(statement.value):
            self.release(self.instance(statement.value), info.expression_type(statement.value))
        elif isinstance(statement, ast.Return) and info.result is not None and self.program.is_place(info.result):
            assert statement.value is not None
            self.move(self.out_pointer, self.instance(statement.value), info.result)
            self.leave(0)
        elif isinstance(statement, ast.Assign) and info.expression_type(statement.value).kind == "str":
            self.assign_str(statement.targets[0], statement.value)
        elif (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and statement.value is not None
            and info.expression_type(statement.value).kind in {"list", "dict", "storage"}
            and _copied(statement.value) is not None
        ):
            self.assign_copy(
                statement.targets[0] if isinstance(statement, ast.Assign) else statement.target, statement.value
            )
        elif (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and statement.value is not None
            and info.expression_type(statement.value).kind == "storage"
        ):
            target = statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
            assert isinstance(statement.value, ast.Call)
            self.assign_storage(target, statement.value)
        elif isinstance(statement, ast.Delete):
            for target in statement.targets:
                assert isinstance(target, ast.Subscript)
                dict_type = info.expression_type(target.value)
                key = self.dict_key(target.slice, dict_type)
                if self.module.owns_memory(element_of(dict_type)):
                    index = self.dict_call("Find", target.value, key)
                    self.fail_if(self.value(f"icmp slt i64 {index}, 0"), HostStatus.KEY_NOT_FOUND)
                    self.release(self.dict_member(target.value, index, 2), element_of(dict_type))
                removed = self.dict_call("Remove", target.value, key, self.value('load ptr, ptr @"vh.allocator"'))
                self.fail_if(self.value(f"icmp eq i32 {removed}, 0"), HostStatus.KEY_NOT_FOUND)
        elif isinstance(statement, ast.Assign):
            value = self.expression(statement.value)
            self.assign(statement.targets[0], value, info.expression_type(statement.value))
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None and info.expression_type(statement.value).kind == "str":
                self.assign_str(statement.target, statement.value)
            elif statement.value is not None:
                value = self.expression(statement.value)
                self.assign(statement.target, value, info.expression_type(statement.value))
        elif isinstance(statement, ast.AugAssign) and info.expression_type(statement.value).kind == "str":
            # Python reads the target before evaluating the operand, which may change it.
            joined = self.owned_str()
            self.instruction(f"store i64 0, ptr {self.list_size_pointer(joined)}")
            self.append_str(joined, statement.target)
            self.append_str(joined, statement.value)
            self.copy_str(self.str_record(statement.target), self.list_data(joined), self.list_size(joined))
        elif isinstance(statement, ast.AugAssign):
            target_type = info.expression_type(statement.target)
            if isinstance(statement.target, ast.Subscript) and (plan := self.selection_plan(statement.target)):
                for index in self.selection_index_nodes(plan):
                    self.evaluated[id(index)] = self.expression(index)
            elif isinstance(statement.target, ast.Subscript):
                indices = statement.target.slice
                for index in indices.elts if isinstance(indices, ast.Tuple) else [indices]:
                    if info.expression_type(index).kind != "str":
                        self.evaluated[id(index)] = self.expression(index)
            current = self.expression(statement.target)
            value = self.expression(statement.value)
            result = self.arithmetic(
                _BINARY[type(statement.op)],
                current,
                target_type,
                value,
                info.expression_type(statement.value),
                target_type,
            )
            self.assign(statement.target, result, target_type)
        elif isinstance(statement, ast.Expr):
            self.expression(statement.value)
        elif isinstance(statement, ast.Return):
            if statement.value is not None:
                assert info.result is not None
                value = self.typed(statement.value, info.result)
                self.store(value, info.result, self.out_pointer)
            self.leave(0)
        elif isinstance(statement, ast.If):
            then_label, else_label, done = self.label("then"), self.label("else"), self.label("endif")
            self.branch_on(self.condition(statement.test), then_label, else_label)
            self.start(then_label)
            self.block(statement.body)
            self.jump(done)
            self.start(else_label)
            self.block(statement.orelse)
            self.jump(done)
            self.start(done)
        elif isinstance(statement, ast.While):
            header, body, otherwise, done = (self.label(hint) for hint in ("while", "body", "whileelse", "endwhile"))
            self.jump(header)
            self.start(header)
            self.branch_on(self.condition(statement.test), body, otherwise)
            self.start(body)
            self.loops.append((header, done, len(self.scopes)))
            self.block(statement.body)
            self.loops.pop()
            self.jump(header)
            self.start(otherwise)
            self.block(statement.orelse)
            self.jump(done)
            self.start(done)
        elif isinstance(statement, ast.For):
            self.for_loop(statement)
        elif isinstance(statement, (ast.Break, ast.Continue)):
            latch, done, depth = self.loops[-1]
            self.close_scopes(self.scopes, depth)
            self.jump(done if isinstance(statement, ast.Break) else latch)
        elif isinstance(statement, ast.With):
            self.scoped(statement.items, statement.body)
        elif isinstance(statement, ast.Pass):
            return
        else:
            raise TypeError(f"unsupported Host statement {type(statement).__name__}")

    def scoped(self, items: list[ast.withitem], body: list[ast.stmt]) -> None:
        """Runs `body` inside the scopes `items` open, the later ones nested in the earlier, as Python nests them.
        The opener's handle is evaluated once and kept, since its closer runs on every exit."""

        if not items:
            self.block(body)
            return
        item = items[0]
        call = item.context_expr
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        opener = self.program.externs[call.func.id]
        assert opener.closed_by is not None
        handle_type = opener.parameters[0][1]
        slot = self.alloca(self.module.memory_type(handle_type))
        handle = self.typed(call.args[0], handle_type)
        self.store(handle, handle_type, slot)
        self.evaluated[id(call.args[0])] = handle
        result = self.extern_call(opener, call)
        if item.optional_vars is not None:
            assert opener.result is not None
            self.assign(item.optional_vars, result, opener.result)
        self.scopes.append((self.program.externs[opener.closed_by], slot))
        self.scoped(items[1:], body)
        scope = self.scopes.pop()
        if not self.terminated:
            status = self.call_closer(*scope)
            if status is not None:
                self.propagate(status)

    def for_loop(self, statement: ast.For) -> None:
        counted = isinstance(statement.iter, ast.Call) and dotted_name(statement.iter.func) == "range"
        if not counted and self.info.expression_type(statement.iter).kind == "dict":
            self.dict_loop(statement)
            return
        if not isinstance(statement.iter, ast.Call):
            self.list_loop(statement)
            return
        assert isinstance(statement.target, ast.Name)
        i32 = ConcreteType("scalar", "i32")
        bounds = [self.typed(argument, i32) for argument in statement.iter.args]
        if len(bounds) == 1:
            start, stop, step = "0", bounds[0], "1"
        elif len(bounds) == 2:
            start, stop, step = bounds[0], bounds[1], "1"
        else:
            start, stop, step = bounds
        self.fail_if(self.value(f"icmp eq i32 {step}, 0"), HostStatus.INVALID_RANGE_STEP)
        induction = self.alloca("i32")
        self.instruction(f"store i32 {start}, ptr {induction}")
        header, body, latch, otherwise, done = (
            self.label(hint) for hint in ("for", "forbody", "forlatch", "forelse", "endfor")
        )
        self.jump(header)
        self.start(header)
        index = self.value(f"load i32, ptr {induction}")
        upward = self.value(f"icmp sgt i32 {step}, 0")
        below = self.value(f"icmp slt i32 {index}, {stop}")
        above = self.value(f"icmp sgt i32 {index}, {stop}")
        self.branch_on(self.value(f"select i1 {upward}, i1 {below}, i1 {above}"), body, otherwise)
        self.start(body)
        self.instruction(f"store i32 {index}, ptr {self.slots[statement.target.id]}")
        self.loops.append((latch, done, len(self.scopes)))
        self.block(statement.body)
        self.loops.pop()
        self.jump(latch)
        self.start(latch)
        current = self.value(f"load i32, ptr {induction}")
        self.instruction(f"store i32 {self.value(f'add i32 {current}, {step}')}, ptr {induction}")
        self.jump(header)
        self.start(otherwise)
        self.block(statement.orelse)
        self.jump(done)
        self.start(done)

    # Containers: a list or str is a `%vh.list` record in a class field, {data, size, capacity}.

    def container(self, node: ast.expr) -> str:
        address = self.place_address(node)
        assert address is not None, "the Host validator admits lists and dicts only as fields and locals"
        return address

    def list_size_pointer(self, record: str) -> str:
        return self.value(f"getelementptr %vh.list, ptr {record}, i32 0, i32 1")

    def list_size(self, record: str) -> str:
        return self.value(f"load i64, ptr {self.list_size_pointer(record)}")

    def list_data(self, record: str) -> str:
        return self.value(f"inttoptr i64 {self.value(f'load i64, ptr {record}')} to ptr")

    def list_element(self, record: str, list_type: ConcreteType, index: str) -> str:
        element = self.module.memory_type(element_of(list_type))
        return self.value(f"getelementptr {element}, ptr {self.list_data(record)}, i64 {index}")

    def reserve(self, record: str, capacity: str, container_type: ConcreteType) -> None:
        size, alignment = self.module.element_layout(container_type)
        function = self.module.container_runtime("reserve")
        allocator = self.value('load ptr, ptr @"vh.allocator"')
        status = self.value(
            f"call i32 {function}(ptr {record}, i64 {capacity}, i64 {size}, i64 {alignment}, ptr {allocator})"
        )
        self.propagate(status)

    def list_index(self, node: ast.Subscript, record: str, list_type: ConcreteType) -> str:
        # The index is evaluated before the size and data are read, since evaluating it may grow the list.
        index_type = self.info.expression_type(node.slice)
        index = self.convert_scalar(self.expression(node.slice), index_type.name, "i64")
        size = self.list_size(record)
        if not index_type.name.startswith("u"):
            negative = self.value(f"icmp slt i64 {index}, 0")
            wrapped = self.value(f"add i64 {index}, {size}")
            index = self.value(f"select i1 {negative}, i64 {wrapped}, i64 {index}")
        self.fail_if(self.value(f"icmp uge i64 {index}, {size}"), HostStatus.INDEX_OUT_OF_RANGE)
        return self.list_element(record, list_type, index)

    def list_method(self, node: ast.Call) -> str:
        assert isinstance(node.func, ast.Attribute)
        list_type = self.info.expression_type(node.func.value)
        element = element_of(list_type)
        if node.func.attr == "append":
            # Python evaluates the argument before appending, and evaluating it may change the list.
            instance = self.program.is_place(element)
            value = self.instance(node.args[0]) if instance else self.typed(node.args[0], element)
            record = self.container(node.func.value)
            size = self.list_size(record)
            grown = self.value(f"add i64 {size}, 1")
            self.reserve(record, grown, list_type)
            if instance:
                self.move(self.list_element(record, list_type, size), value, element)
            else:
                self.store(value, element, self.list_element(record, list_type, size))
            self.instruction(f"store i64 {grown}, ptr {self.list_size_pointer(record)}")
            return "poison"
        if node.func.attr == "clear":
            self.list_clear(node.func.value)
            return "poison"
        record = self.container(node.func.value)
        size = self.list_size(record)
        self.fail_if(self.value(f"icmp eq i64 {size}, 0"), HostStatus.INDEX_OUT_OF_RANGE)
        last = self.value(f"sub i64 {size}, 1")
        value = self.load(element, self.list_element(record, list_type, last))
        self.instruction(f"store i64 {last}, ptr {self.list_size_pointer(record)}")
        return value

    def list_loop(self, statement: ast.For) -> None:
        # Like Python's list iterator, each step reads the current length and data, so the body may change the list.
        assert isinstance(statement.target, ast.Name)
        list_type = self.info.expression_type(statement.iter)
        record = self.container(statement.iter)
        position = self.alloca("i64")
        self.instruction(f"store i64 0, ptr {position}")
        header, body, latch, otherwise, done = (
            self.label(hint) for hint in ("for", "forbody", "forlatch", "forelse", "endfor")
        )
        self.jump(header)
        self.start(header)
        index = self.value(f"load i64, ptr {position}")
        self.branch_on(self.value(f"icmp ult i64 {index}, {self.list_size(record)}"), body, otherwise)
        self.start(body)
        element = element_of(list_type)
        address = self.list_element(record, list_type, index)
        if self.program.is_place(element):
            self.instruction(f"store ptr {address}, ptr {self.slots[statement.target.id]}")
        else:
            self.store(self.load(element, address), element, self.slots[statement.target.id])
        self.loops.append((latch, done, len(self.scopes)))
        self.block(statement.body)
        self.loops.pop()
        self.jump(latch)
        self.start(latch)
        current = self.value(f"load i64, ptr {position}")
        self.instruction(f"store i64 {self.value(f'add i64 {current}, 1')}, ptr {position}")
        self.jump(header)
        self.start(otherwise)
        self.block(statement.orelse)
        self.jump(done)
        self.start(done)

    # strs: a `%vh.list` of UTF-8 bytes. A str local or field is read in place. An f-string or + builds into a record
    # the function owns, one per expression, which reuses its memory each time the expression runs.

    def str_operand(self, node: ast.expr) -> tuple[str, str]:
        """The (data pointer, byte size) of a str expression."""

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            data, size = self.module.string(node.value)
            return data, str(size)
        if isinstance(node, ast.IfExp):
            data_slot, size_slot = self.alloca("ptr"), self.alloca("i64")
            then_label, else_label, done = self.label("ifexp"), self.label("ifelse"), self.label("ifend")
            self.branch_on(self.condition(node.test), then_label, else_label)
            for label, branch in ((then_label, node.body), (else_label, node.orelse)):
                self.start(label)
                data, size = self.str_operand(branch)
                self.instruction(f"store ptr {data}, ptr {data_slot}")
                self.instruction(f"store i64 {size}, ptr {size_slot}")
                self.jump(done)
            self.start(done)
            return self.value(f"load ptr, ptr {data_slot}"), self.value(f"load i64, ptr {size_slot}")
        if isinstance(node, (ast.JoinedStr, ast.BinOp)):
            record = self.owned_str()
            self.instruction(f"store i64 0, ptr {self.list_size_pointer(record)}")
            self.append_str(record, node)
        else:
            record = self.str_record(node)
        return self.list_data(record), self.list_size(record)

    def stable_str_operand(self, node: ast.expr, later: list[ast.expr]) -> tuple[str, str]:
        """A str operand that stays valid while `later` operands are evaluated: a call among them may assign the
        str field `node` reads in place, so the field is copied first."""

        if isinstance(node, ast.Attribute) and any(
            isinstance(item, ast.Call) for other in later for item in ast.walk(other)
        ):
            record = self.owned_str()
            self.instruction(f"store i64 0, ptr {self.list_size_pointer(record)}")
            self.append_str(record, node)
            return self.list_data(record), self.list_size(record)
        return self.str_operand(node)

    def str_record(self, node: ast.expr) -> str:
        """The record of a str local or str field."""

        if isinstance(node, ast.Name) and node.id in self.slots:
            return self.slots[node.id]
        return self.container(node)

    def append_str(self, record: str, node: ast.expr, spec: FormatSpec | None = None) -> None:
        """Appends the str expression `node` to `record`, which nothing in `node` reads, padded under `spec`."""

        if spec is None or spec.empty:
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.FormattedValue):
                        self.append_formatted(record, part)
                    else:
                        self.append_bytes(record, *self.str_operand(part))
                return
            if isinstance(node, ast.BinOp):
                self.append_str(record, node.left)
                self.append_str(record, node.right)
                return
        self.append_bytes(record, *self.str_operand(node), spec)

    def append_bytes(self, record: str, data: str, size: str, spec: FormatSpec | None = None) -> None:
        function = self.module.str_runtime("")
        format_spec = self.module.format_spec(spec or FormatSpec())
        status = self.value(
            f"call i32 {function}(ptr {record}, ptr {data}, i64 {size}, ptr {format_spec}, ptr {self.allocator()})"
        )
        self.propagate(status)

    def append_formatted(self, record: str, part: ast.FormattedValue) -> None:
        value_type = self.info.expression_type(part.value)
        kind = value_kind(value_type)
        assert kind is not None, "the Host validator admits only formattable replacement fields"
        spec = parse_format_spec(spec_text(part.format_spec) or "", kind)
        if kind == "str":
            self.append_str(record, part.value, spec)
            return
        value = self.typed(part.value, value_type)
        if kind == "bool":
            true_data, _ = self.module.string("True")
            false_data, _ = self.module.string("False")
            data = self.value(f"select i1 {value}, ptr {true_data}, ptr {false_data}")
            self.append_bytes(record, data, self.value(f"select i1 {value}, i64 4, i64 5"))
            return
        format_spec = self.module.format_spec(spec)
        if kind == "integer":
            name = "u64" if value_type.kind == "handle" else value_type.name
            signed = 0 if name.startswith("u") else 1
            bits = self.convert_scalar(value, name, "i64" if signed else "u64")
            text = f"ptr {record}, i64 {bits}, i32 {signed}, ptr {format_spec}, ptr {self.allocator()}"
            status = self.value(f"call i32 {self.module.str_runtime('Integer')}({text})")
        else:
            wide = self.convert_scalar(value, value_type.name, "f64")
            text = f"ptr {record}, double {wide}, ptr {format_spec}, ptr {self.allocator()}"
            status = self.value(f"call i32 {self.module.str_runtime('Float')}({text})")
        self.propagate(status)

    def copy_str(self, record: str, data: str, size: str) -> None:
        # Growing never happens when the source is the target itself, whose size is within its capacity, so the
        # source bytes stay valid; memmove covers the overlap.
        self.reserve(record, size, ConcreteType("str", "str"))
        memmove = self.module.intrinsic_declaration(
            "llvm.memmove.p0.p0.i64", "declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)"
        )
        self.instruction(f"call void {memmove}(ptr {self.list_data(record)}, ptr {data}, i64 {size}, i1 false)")
        self.instruction(f"store i64 {size}, ptr {self.list_size_pointer(record)}")

    def assign_str(self, target: ast.expr, value: ast.expr) -> None:
        data, size = self.str_operand(value)
        self.copy_str(self.str_record(target), data, size)

    def str_equal(self, left: ast.expr, right: ast.expr) -> str:
        left_data, left_size = self.stable_str_operand(left, [right])
        right_data, right_size = self.str_operand(right)
        equal = self.module.str_equal_function()
        return self.value(f"call i1 {equal}(ptr {left_data}, i64 {left_size}, ptr {right_data}, i64 {right_size})")

    def str_length(self, node: ast.expr) -> str:
        data, size = self.str_operand(node)
        return self.value(f"call i64 {self.module.str_length_function()}(ptr {data}, i64 {size})")

    # Dicts: a `%vh.dict` record in a class field, changed only through VernonRuntime; entries are
    # `{ i64 hash, K key, V value }`, and a removed entry's hash is all ones.

    def dict_key(self, node: ast.expr, dict_type: ConcreteType) -> str:
        """A pointer to the key the runtime takes: the key's bytes, or a str key's `{ data, size }`."""

        key = key_of(dict_type)
        if key.kind != "str":
            return self.spill(self.typed(node, key), key)
        data, size = self.str_operand(node)
        view = self.alloca("{ i64, i64 }")
        self.instruction(f"store i64 {self.value(f'ptrtoint ptr {data} to i64')}, ptr {view}")
        size_slot = self.value(f"getelementptr {{ i64, i64 }}, ptr {view}, i32 0, i32 1")
        self.instruction(f"store i64 {size}, ptr {size_slot}")
        return view

    def dict_call(self, function: str, container: ast.expr, key: str | None, *rest: str) -> str:
        """Calls VernonRuntime's `vernonRuntimeHostDict<function>` on the dict field `container`, passing the
        key when the function takes one, the type's layout, then `rest`."""

        # A local's declaration is a store target, which typing records only as the local's type.
        if isinstance(container, ast.Name) and container.id in self.info.local_types:
            dict_type = self.info.local_types[container.id]
        else:
            dict_type = self.info.expression_type(container)
        arguments = [self.container(container), *([key] if key is not None else []), self.module.dict_layout(dict_type)]
        text = f"{self.module.dict_runtime(function)}({', '.join(f'ptr {item}' for item in (*arguments, *rest))})"
        result = {"Find": "i64", "Insert": "i32", "Remove": "i32"}.get(function)
        if result is None:
            self.instruction(f"call void {text}")
            return "poison"
        return self.value(f"call {result} {text}")

    def allocator(self) -> str:
        return self.value('load ptr, ptr @"vh.allocator"')

    def transfer_allocator(self) -> str:
        """The module's allocator, or VernonRuntime's default as an object: an extern filling an owned result
        never sees NULL, so a library linking another C runtime allocates from the heap that frees it."""

        installed = self.allocator()
        self.module.declare("vernonRuntimeHostDefaultAllocator", "declare ptr @vernonRuntimeHostDefaultAllocator()")
        default = self.value("call ptr @vernonRuntimeHostDefaultAllocator()")
        return self.value(f"select i1 {self.value(f'icmp eq ptr {installed}, null')}, ptr {default}, ptr {installed}")

    def dict_member(self, container: ast.expr, index: str, member: int) -> str:
        """The key (1) or value (2) of entry `index`, read after any call that may have moved the entries."""

        dict_type = self.info.expression_type(container)
        entries = self.value(f"inttoptr i64 {self.value(f'load i64, ptr {self.container(container)}')} to ptr")
        entry_type = self.module.memory_type(dict_entry(dict_type))
        entry = self.value(f"getelementptr {entry_type}, ptr {entries}, i64 {index}")
        return self.value(f"getelementptr {entry_type}, ptr {entry}, i32 0, i32 {member}")

    def dict_span(self, container: ast.expr) -> tuple[str, str]:
        """The data and size of a span of the dict's `{K key; V value;}` entries in insertion order, laid out in a
        list the function owns."""

        dict_type = self.local_or_expression_type(container)
        span_type = transfer_record(dict_type)
        pair = self.module.memory_type(element_of(span_type))
        key, value = key_of(dict_type), element_of(dict_type)
        span = self.owned_record(span_type)
        size_pointer = self.list_size_pointer(span)
        self.instruction(f"store i64 0, ptr {size_pointer}")
        record = self.container(container)
        self.reserve(
            span,
            self.value(f"load i64, ptr {self.value(f'getelementptr %vh.dict, ptr {record}, i32 0, i32 1')}"),
            span_type,
        )
        used = self.value(f"load i64, ptr {self.value(f'getelementptr %vh.dict, ptr {record}, i32 0, i32 2')}")
        position = self.alloca("i64")
        self.instruction(f"store i64 0, ptr {position}")
        header, live, copy, latch, done = (
            self.label(hint) for hint in ("span", "spanentry", "spancopy", "spannext", "spanend")
        )
        self.jump(header)
        self.start(header)
        index = self.value(f"load i64, ptr {position}")
        self.branch_on(self.value(f"icmp ult i64 {index}, {used}"), live, done)
        self.start(live)
        removed = self.value(f"icmp eq i64 {self.value(f'load i64, ptr {self.dict_member(container, index, 0)}')}, -1")
        self.branch_on(removed, latch, copy)
        self.start(copy)
        filled = self.value(f"load i64, ptr {size_pointer}")
        slot = self.value(f"getelementptr {pair}, ptr {self.list_data(span)}, i64 {filled}")
        self.memcpy(
            self.value(f"getelementptr {pair}, ptr {slot}, i32 0, i32 0"),
            self.dict_member(container, index, 1),
            self.size_of(key),
        )
        self.memcpy(
            self.value(f"getelementptr {pair}, ptr {slot}, i32 0, i32 1"),
            self.dict_member(container, index, 2),
            self.size_of(value),
        )
        self.instruction(f"store i64 {self.value(f'add i64 {filled}, 1')}, ptr {size_pointer}")
        self.jump(latch)
        self.start(latch)
        self.instruction(f"store i64 {self.value(f'add i64 {index}, 1')}, ptr {position}")
        self.jump(header)
        self.start(done)
        return self.list_data(span), self.value(f"load i64, ptr {size_pointer}")

    def transferring_extern(self, node: ast.expr | None) -> HostExtern | None:
        """The extern `node` calls when it returns a str, list or dict."""

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        extern = self.program.externs.get(node.func.id)
        if extern is None or extern.result is None or extern.result.kind not in TRANSFERRED_KINDS:
            return None
        return extern

    def assign_extern_result(self, target: ast.expr, extern: HostExtern, node: ast.Call) -> None:
        """Receives an extern's str, list or dict like a copy: into a record of its own, then releases the target's
        and moves the result in, so a failed call leaves the target as it was."""

        result = extern.result
        assert result is not None
        received = self.owned_record(transfer_record(result))
        self.extern_call(extern, node, received)
        built = received
        if result.kind == "dict":
            built = self.owned_record(result)
            self.insert_entries(built, received, result)
            # The entries record is reused by the next call, which finds it empty.
            self.release(received, transfer_record(result))
        record = self.str_record(target) if result.kind == "str" else self.container(target)
        self.release(record, result)
        self.move(record, built, result)

    def insert_entries(self, built: str, entries: str, dict_type: ConcreteType) -> None:
        """Inserts a list of `{K key; V value;}` entries into the empty dict `built` in order, so a repeated key keeps
        its last value."""

        entries_type = transfer_record(dict_type)
        pair = self.module.memory_type(element_of(entries_type))
        value = element_of(dict_type)
        entry_type = self.module.memory_type(dict_entry(dict_type))
        layout_name = self.module.dict_layout(dict_type)
        size = self.list_size(entries)
        position = self.alloca("i64")
        self.instruction(f"store i64 0, ptr {position}")
        header, body, done = (self.label(hint) for hint in ("insert", "insertentry", "insertend"))
        self.jump(header)
        self.start(header)
        index = self.value(f"load i64, ptr {position}")
        self.branch_on(self.value(f"icmp ult i64 {index}, {size}"), body, done)
        self.start(body)
        slot = self.value(f"getelementptr {pair}, ptr {self.list_data(entries)}, i64 {index}")
        out = self.alloca("i64")
        insert = self.module.dict_runtime("Insert")
        key_pointer = self.value(f"getelementptr {pair}, ptr {slot}, i32 0, i32 0")
        allocator = self.allocator()
        self.propagate(
            self.value(
                f"call i32 {insert}(ptr {built}, ptr {key_pointer}, ptr {layout_name}, ptr {allocator}, ptr {out})"
            )
        )
        entries_pointer = self.value(f"inttoptr i64 {self.value(f'load i64, ptr {built}')} to ptr")
        entry = self.value(
            f"getelementptr {entry_type}, ptr {entries_pointer}, i64 {self.value(f'load i64, ptr {out}')}"
        )
        self.memcpy(
            self.value(f"getelementptr {entry_type}, ptr {entry}, i32 0, i32 2"),
            self.value(f"getelementptr {pair}, ptr {slot}, i32 0, i32 1"),
            self.size_of(value),
        )
        self.instruction(f"store i64 {self.value(f'add i64 {index}, 1')}, ptr {position}")
        self.jump(header)
        self.start(done)

    def dict_value(self, node: ast.Subscript, store: bool) -> str:
        """The value of `node`'s key: an entry it is stored into is inserted if missing, and one read must exist."""

        key = self.dict_key(node.slice, self.info.expression_type(node.value))
        if store:
            out = self.alloca("i64")
            self.propagate(self.dict_call("Insert", node.value, key, self.allocator(), out))
            index = self.value(f"load i64, ptr {out}")
        else:
            index = self.dict_call("Find", node.value, key)
            self.fail_if(self.value(f"icmp slt i64 {index}, 0"), HostStatus.KEY_NOT_FOUND)
        return self.dict_member(node.value, index, 2)

    def release_elements(self, container: ast.expr) -> None:
        container_type = self.local_or_expression_type(container)
        if self.module.owns_memory(element_of(container_type)):
            function = self.module.release_elements_function(container_type)
            self.instruction(f"call void {function}(ptr {self.container(container)})")

    def local_or_expression_type(self, node: ast.expr) -> ConcreteType:
        # A local's declaration is a store target, which typing records only as the local's type.
        if isinstance(node, ast.Name) and node.id in self.info.local_types:
            return self.info.local_types[node.id]
        return self.info.expression_type(node)

    def list_clear(self, container: ast.expr) -> None:
        self.release_elements(container)
        self.instruction(f"store i64 0, ptr {self.list_size_pointer(self.container(container))}")

    def dict_clear(self, container: ast.expr) -> None:
        self.release_elements(container)
        self.dict_call("Clear", container, None, self.allocator())

    def dict_method(self, node: ast.Call) -> str:
        assert isinstance(node.func, ast.Attribute)
        container, method = node.func.value, node.func.attr
        if method == "clear":
            self.dict_clear(container)
            return "poison"
        value_type = element_of(self.info.expression_type(container))
        key = self.dict_key(node.args[0], self.info.expression_type(container))
        default = self.typed(node.args[1], value_type) if len(node.args) == 2 else None
        index = self.dict_call("Find", container, key)
        if default is None:
            self.fail_if(self.value(f"icmp slt i64 {index}, 0"), HostStatus.KEY_NOT_FOUND)
            value = self.load(value_type, self.dict_member(container, index, 2))
            self.dict_call("Remove", container, key, self.allocator())
            return value
        slot = self.alloca(self.module.memory_type(value_type))
        hit, miss, done = self.label("dicthit"), self.label("dictmiss"), self.label("dictend")
        self.branch_on(self.value(f"icmp sge i64 {index}, 0"), hit, miss)
        self.start(hit)
        self.store(self.load(value_type, self.dict_member(container, index, 2)), value_type, slot)
        if method == "pop":
            self.dict_call("Remove", container, key, self.allocator())
        self.jump(done)
        self.start(miss)
        self.store(default, value_type, slot)
        self.jump(done)
        self.start(done)
        return self.load(value_type, slot)

    def dict_contains(self, node: ast.Compare) -> str:
        container = node.comparators[0]
        index = self.dict_call("Find", container, self.dict_key(node.left, self.info.expression_type(container)))
        return self.value(f"icmp {'sge' if isinstance(node.ops[0], ast.In) else 'slt'} i64 {index}, 0")

    def dict_loop(self, statement: ast.For) -> None:
        # Like Python's dict iterators, every step first fails if the dict's size changed since the loop began, and
        # reads the entries afresh, so a body that only assigns existing keys runs on.
        iterated = statement.iter
        view = iterated.func.attr if isinstance(iterated, ast.Call) else "keys"  # type: ignore[attr-defined]
        container = iterated.func.value if isinstance(iterated, ast.Call) else iterated  # type: ignore[attr-defined]
        dict_type = self.info.expression_type(container)
        record = self.container(container)
        size_pointer = self.value(f"getelementptr %vh.dict, ptr {record}, i32 0, i32 1")
        used_pointer = self.value(f"getelementptr %vh.dict, ptr {record}, i32 0, i32 2")
        size = self.value(f"load i64, ptr {size_pointer}")
        position = self.alloca("i64")
        self.instruction(f"store i64 0, ptr {position}")
        header, live, body, latch, otherwise, done = (
            self.label(hint) for hint in ("for", "forentry", "forbody", "forlatch", "forelse", "endfor")
        )
        self.jump(header)
        self.start(header)
        current = self.value(f"load i64, ptr {size_pointer}")
        self.fail_if(self.value(f"icmp ne i64 {current}, {size}"), HostStatus.CONTAINER_CHANGED)
        index = self.value(f"load i64, ptr {position}")
        used = self.value(f"load i64, ptr {used_pointer}")
        self.branch_on(self.value(f"icmp ult i64 {index}, {used}"), live, otherwise)
        self.start(live)
        hash_pointer = self.dict_member(container, index, 0)
        removed = self.value(f"icmp eq i64 {self.value(f'load i64, ptr {hash_pointer}')}, -1")
        self.branch_on(removed, latch, body)
        self.start(body)
        if view == "items":
            assert isinstance(statement.target, ast.Tuple)
            names = [item.id for item in statement.target.elts]  # type: ignore[attr-defined]
            bound = [(names[0], 1, key_of(dict_type)), (names[1], 2, element_of(dict_type))]
        else:
            assert isinstance(statement.target, ast.Name)
            member = 1 if view == "keys" else 2
            bound = [(statement.target.id, member, key_of(dict_type) if member == 1 else element_of(dict_type))]
        for name, member, member_type in bound:
            pointer = self.dict_member(container, index, member)
            if member_type.kind == "str":
                # The key is copied, as Python binds the key object: the body may remove it from the dict.
                self.copy_str(self.slots[name], self.list_data(pointer), self.list_size(pointer))
                continue
            if self.program.is_place(member_type):
                self.instruction(f"store ptr {pointer}, ptr {self.slots[name]}")
                continue
            self.store(self.load(member_type, pointer), member_type, self.slots[name])
        self.loops.append((latch, done, len(self.scopes)))
        self.block(statement.body)
        self.loops.pop()
        self.jump(latch)
        self.start(latch)
        current = self.value(f"load i64, ptr {position}")
        self.instruction(f"store i64 {self.value(f'add i64 {current}, 1')}, ptr {position}")
        self.jump(header)
        self.start(otherwise)
        self.block(statement.orelse)
        self.jump(done)
        self.start(done)

    # Storages: a `{data, [rank x i64] shape}` record in a class field or a local the function owns. Elements are
    # dense and row-major, and data is null when a storage has no elements.

    def storage_record(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name) and node.id in self.slots:
            return self.slots[node.id]
        address = self.place_address(node)
        assert address is not None, "the Host validator admits storages only as fields and locals"
        return address

    def storage_extent(self, record: str, storage_type: ConcreteType, axis: int) -> str:
        memory = self.module.memory_type(storage_type)
        pointer = self.value(f"getelementptr {memory}, ptr {record}, i32 0, i32 1, i32 {axis}")
        return self.value(f"load i64, ptr {pointer}")

    def storage_data(self, record: str) -> str:
        return self.value(f"inttoptr i64 {self.value(f'load i64, ptr {record}')} to ptr")

    def storage_element(self, node: ast.Subscript) -> str:
        """The address of `s[i, j, ...]`: an index below zero counts from the end of its axis, as in numpy."""

        storage_type = self.info.expression_type(node.value)
        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        # Indices are evaluated before the record is read, since evaluating them may replace the storage.
        values = []
        for index_node in indices:
            index_type = self.info.expression_type(index_node)
            values.append((self.convert_scalar(self.expression(index_node), index_type.name, "i64"), index_type))
        record = self.storage_record(node.value)
        offset = "0"
        for axis, (index, index_type) in enumerate(values):
            extent = self.storage_extent(record, storage_type, axis)
            if not index_type.name.startswith("u"):
                negative = self.value(f"icmp slt i64 {index}, 0")
                wrapped = self.value(f"add i64 {index}, {extent}")
                index = self.value(f"select i1 {negative}, i64 {wrapped}, i64 {index}")
            self.fail_if(self.value(f"icmp uge i64 {index}, {extent}"), HostStatus.INDEX_OUT_OF_RANGE)
            scaled = self.value(f"mul i64 {offset}, {extent}")
            offset = self.value(f"add i64 {scaled}, {index}")
        element = self.module.memory_type(element_of(storage_type))
        return self.value(f"getelementptr {element}, ptr {self.storage_data(record)}, i64 {offset}")

    def storage_shape(self, node: ast.expr) -> str:
        """`s.shape`, a Tuple of i32 extents."""

        storage_type = self.info.expression_type(node)
        record = self.storage_record(node)
        extents = [
            self.value(f"trunc i64 {self.storage_extent(record, storage_type, axis)} to i32")
            for axis in range(storage_rank(storage_type))
        ]
        i32 = ConcreteType("scalar", "i32")
        return self.build(ConcreteType("tuple", "Tuple", (i32,) * len(extents)), extents)

    def assign_storage(self, target: ast.expr, creation: ast.Call) -> None:
        """Creates a storage into a record of its own, then releases the target's and moves the new one in, so the
        creation may read the storage it replaces and a failed one leaves the target as it was."""

        info = self.info
        storage_type = info.expression_type(creation)
        rank = storage_rank(storage_type)
        size = self.module.storage_element_size(storage_type)
        values = None
        if _call_name(creation).endswith("_like"):
            source = self.storage_record(creation.args[0])
            memory = self.module.memory_type(info.expression_type(creation.args[0]))
            shape = self.value(f"getelementptr {memory}, ptr {source}, i32 0, i32 1")
        elif _call_name(creation) == "from_values":
            values_type = info.expression_type(creation.args[0])
            values = (self.spill(self.expression(creation.args[0]), values_type), tensor_count(values_type) * size)
            shape = self.alloca(f"[{rank} x i64]")
            extents = ", ".join(f"i64 {extent}" for extent in values_type.arguments[1:])
            self.instruction(f"store [{rank} x i64] [{extents}], ptr {shape}")
        else:
            shape_node = next(keyword.value for keyword in creation.keywords if keyword.arg == "shape")
            if isinstance(shape_node, ast.Tuple):
                extents = [self.typed_i64(item) for item in shape_node.elts]
            else:
                shape_type = info.expression_type(shape_node)
                aggregate = self.expression(shape_node)
                extents = [
                    self.convert_scalar(self.extract(aggregate, shape_type, axis), item.name, "i64")
                    for axis, item in enumerate(self.member_types(shape_type))
                ]
            shape = self.alloca(f"[{rank} x i64]")
            for axis, extent in enumerate(extents):
                pointer = self.value(f"getelementptr [{rank} x i64], ptr {shape}, i32 0, i32 {axis}")
                self.instruction(f"store i64 {extent}, ptr {pointer}")
        created = self.alloca(self.module.memory_type(storage_type))
        status = self.value(
            f"call i32 {self.module.storage_runtime('Allocate')}(ptr {created}, i32 {rank}, ptr {shape}, "
            f"i64 {size}, i64 {size}, ptr {self.allocator()})"
        )
        self.propagate(status)
        if values is not None:
            self.memcpy(self.storage_data(created), *values)
        record = self.storage_record(target)
        self.instruction(f"call void {self.module.fini_function(storage_type)}(ptr {record})")
        memory = self.module.memory_type(storage_type)
        self.instruction(f"store {memory} {self.value(f'load {memory}, ptr {created}')}, ptr {record}")

    def typed_i64(self, node: ast.expr) -> str:
        return self.convert_scalar(self.expression(node), self.info.expression_type(node).name, "i64")

    def allocate_static_storages(self, self_type: ConcreteType) -> None:
        """An initializer gives each storage field of static extents its zeroed elements, unless the field's
        default creates them."""

        defaulted = {
            target.attr
            for statement in self.info.source.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Attribute)
        }
        for index, (name, field) in enumerate(self.module.fields(self_type.name)):
            if field.kind != "storage" or name in defaulted or "?" in field.arguments[1]:  # type: ignore[operator]
                continue
            record = self.value(
                f"getelementptr {self.module.memory_type(self_type)}, ptr {self.places['self']}, i32 0, i32 {index}"
            )
            rank = storage_rank(field)
            shape = self.alloca(f"[{rank} x i64]")
            extents = ", ".join(f"i64 {extent}" for extent in field.arguments[1])  # type: ignore[union-attr]
            self.instruction(f"store [{rank} x i64] [{extents}], ptr {shape}")
            size = self.module.storage_element_size(field)
            status = self.value(
                f"call i32 {self.module.storage_runtime('Allocate')}(ptr {record}, i32 {rank}, ptr {shape}, "
                f"i64 {size}, i64 {size}, ptr {self.allocator()})"
            )
            self.propagate(status)

    def storage_method(self, node: ast.Call) -> str:
        assert isinstance(node.func, ast.Attribute)
        storage_type = self.info.expression_type(node.func.value)
        element = element_of(storage_type)
        size = self.module.storage_element_size(storage_type)
        rank = storage_rank(storage_type)
        method = node.func.attr
        if method == "fill":
            value = self.spill(self.typed(node.args[0], element), element)
            record = self.storage_record(node.func.value)
            fill = self.module.storage_runtime("Fill")
            self.instruction(f"call void {fill}(ptr {record}, i32 {rank}, ptr {value}, i64 {size})")
            return "poison"
        if method == "copy_from_values":
            tensor_type = self.info.expression_type(node.args[0])
            source = self.spill(self.expression(node.args[0]), tensor_type)
            record = self.storage_record(node.func.value)
            for axis, extent in enumerate(tensor_type.arguments[1:]):
                differs = self.value(f"icmp ne i64 {self.storage_extent(record, storage_type, axis)}, {extent}")
                self.fail_if(differs, HostStatus.INVALID_SHAPE)
            self.memcpy(self.storage_data(record), source, tensor_count(tensor_type) * size)
            return "poison"
        result_type = self.info.expression_type(node)
        record = self.storage_record(node.func.value)
        copied = self.alloca(self.module.memory_type(result_type))
        self.memcpy(copied, self.storage_data(record), tensor_count(result_type) * size)
        return self.load(result_type, copied)

    def memcpy(self, target: str, source: str, size: int) -> None:
        self.module.declare("llvm.memcpy.p0.p0.i64", "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)")
        self.instruction(f"call void @llvm.memcpy.p0.p0.i64(ptr {target}, ptr {source}, i64 {size}, i1 false)")

    def place_address(self, node: ast.expr, store: bool = False) -> str | None:
        """The address of a field or element reached from a class-typed parameter; None for value expressions,
        which have no address of their own. With `store`, `node` is assigned, which inserts a missing dict key."""

        if isinstance(node, ast.Subscript) and self.info.expression_type(node.value).kind == "storage":
            return self.storage_element(node)
        if isinstance(node, ast.Name):
            local = self.info.local_types.get(node.id)
            if local is not None and local.kind in {"list", "dict"}:
                return self.slots[node.id]
            class_local = self.info.class_locals.get(node.id)
            if class_local == "owned":
                return self.slots[node.id]
            if class_local is not None:
                return self.value(f"load ptr, ptr {self.slots[node.id]}")
            return self.places.get(node.id)
        if isinstance(node, ast.Attribute):
            base_type = self.info.expression_type(node.value)
            if base_type.kind != "struct":
                return None
            base = self.place_address(node.value)
            if base is None:
                return None
            index = [name for name, _ in self.module.fields(base_type.name)].index(node.attr)
            return self.value(f"getelementptr {self.module.memory_type(base_type)}, ptr {base}, i32 0, i32 {index}")
        if isinstance(node, ast.Subscript):
            base_type = self.info.expression_type(node.value)
            if base_type.kind == "list":
                return self.list_index(node, self.container(node.value), base_type)
            if base_type.kind == "dict":
                return self.dict_value(node, store)
            if base_type.kind != "tensor":
                return None
            base = self.place_address(node.value)
            if base is None:
                return None
            offset = self.flat_index(node, base_type)
            return self.value(f"getelementptr {self.module.memory_type(base_type)}, ptr {base}, i64 0, i64 {offset}")
        return None

    def place_type(self, node: ast.Attribute | ast.Subscript) -> ConcreteType:
        """The type of a store target, which typing records only for its base."""

        base_type = self.info.expression_type(node.value)
        if isinstance(node, ast.Attribute):
            return dict(self.module.fields(base_type.name))[node.attr]
        return element_of(base_type)

    def assign(self, target: ast.expr, value: str, value_type: ConcreteType) -> None:
        if isinstance(target, ast.Name):
            local_type = self.info.local_types[target.id]
            self.store(self.convert(value, value_type, local_type), local_type, self.slots[target.id])
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, (item, item_type) in enumerate(zip(target.elts, self.member_types(value_type), strict=True)):
                self.assign(item, self.extract(value, value_type, index), item_type)
        elif isinstance(target, ast.Subscript) and (plan := self.selection_plan(target)) is not None:
            storage_type = self.info.expression_type(target.value)
            self.scatter(self.storage_selection(target, plan), element_of(storage_type), value, value_type)
        elif isinstance(target, (ast.Attribute, ast.Subscript)) and (address := self.place_address(target, True)):
            target_type = self.place_type(target)
            self.store(self.convert(value, value_type, target_type), target_type, address)
        else:
            raise TypeError(f"unsupported Host assignment target {type(target).__name__}")

    # Expressions.

    def typed(self, node: ast.expr, target: ConcreteType) -> str:
        return self.convert(self.expression(node), self.info.expression_type(node), target)

    def condition(self, node: ast.expr) -> str:
        return self.typed(node, ConcreteType("scalar", "bool"))

    def expression(self, node: ast.expr) -> str:
        if id(node) in self.evaluated:
            return self.evaluated[id(node)]
        info = self.info
        result_type = info.expression_type(node)
        if isinstance(node, ast.Name):
            local_type = info.local_types[node.id]
            return self.convert(self.load(local_type, self.slots[node.id]), local_type, result_type)
        if isinstance(node, ast.Constant):
            value = literal_value(node.value, result_type.name)
            if result_type.name == "bool":
                return "true" if value else "false"
            if _is_float(result_type.name):
                return _float_literal(value, result_type.name)
            return _int_literal(value, result_type.name)
        if isinstance(node, ast.BinOp):
            left = self.expression(node.left)
            right = self.expression(node.right)
            return self.arithmetic(
                _BINARY[type(node.op)],
                left,
                info.expression_type(node.left),
                right,
                info.expression_type(node.right),
                result_type,
            )
        if isinstance(node, ast.UnaryOp):
            operand = self.typed(node.operand, result_type)
            if isinstance(node.op, ast.Not):
                return self.value(f"xor i1 {operand}, true")
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Invert):
                return self.elementwise(
                    result_type, [operand], lambda item, name: self.value(f"xor {_LLVM_SCALARS[name]} {item}, -1")
                )
            return self.elementwise(result_type, [operand], lambda item, name: self.negate(item, name))
        if isinstance(node, ast.BoolOp):
            slot = self.alloca("i1")
            done = self.label("boolend")
            short_circuit = isinstance(node.op, ast.Or)
            for value in node.values:
                condition = self.condition(value)
                self.instruction(f"store i1 {condition}, ptr {slot}")
                following = self.label("boolnext")
                if short_circuit:
                    self.branch_on(condition, done, following)
                else:
                    self.branch_on(condition, following, done)
                self.start(following)
            self.jump(done)
            self.start(done)
            return self.value(f"load i1, ptr {slot}")
        if isinstance(node, ast.IfExp):
            memory = self.module.memory_type(result_type)
            slot = self.alloca(memory)
            then_label, else_label, done = self.label("ifexp"), self.label("ifelse"), self.label("ifend")
            self.branch_on(self.condition(node.test), then_label, else_label)
            for label, branch in ((then_label, node.body), (else_label, node.orelse)):
                self.start(label)
                self.store(self.typed(branch, result_type), result_type, slot)
                self.jump(done)
            self.start(done)
            return self.load(result_type, slot)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.Tuple):
            return self.build(
                result_type,
                [
                    self.typed(item, item_type)
                    for item, item_type in zip(node.elts, self.member_types(result_type), strict=True)
                ],
            )
        if isinstance(node, ast.Subscript) and (plan := self.selection_plan(node)) is not None:
            if info.expression_type(node.value).kind == "storage":
                return self.gathered(result_type, element_of(result_type), self.storage_selection(node, plan))
            return self.tensor_selection(node, plan)
        if isinstance(node, ast.Attribute) and result_type.kind == "enum" and id(node.value) not in info.expressions:
            return _int_literal(self.program.enums[result_type.name].value(node.attr), "i32")
        if isinstance(node, (ast.Attribute, ast.Subscript)) and (address := self.place_address(node)):
            return self.load(result_type, address)
        if isinstance(node, ast.Subscript):
            value_type = info.expression_type(node.value)
            if value_type.kind == "tuple":
                assert isinstance(node.slice, ast.Constant)
                return self.extract(self.expression(node.value), value_type, int(node.slice.value))
            value = self.expression(node.value)
            offset = self.flat_index(node, value_type)
            pointer = self.spill(value, value_type)
            element = self.value(
                f"getelementptr {self.module.memory_type(value_type)}, ptr {pointer}, i64 0, i64 {offset}"
            )
            return self.load(element_of(value_type), element)
        if isinstance(node, ast.Attribute):
            value_type = info.expression_type(node.value)
            if value_type.kind == "storage":
                return self.storage_shape(node.value)
            value = self.expression(node.value)
            if value_type.kind == "struct":
                names = [name for name, _ in self.module.fields(value_type.name)]
                return self.extract(value, value_type, names.index(node.attr))
            picked = [self.extract(value, value_type, _SWIZZLE[letter]) for letter in node.attr]
            return picked[0] if len(picked) == 1 else self.build(result_type, picked)
        if isinstance(node, ast.Call):
            return self.call(node, result_type)
        raise TypeError(f"unsupported Host expression {type(node).__name__}")

    def flat_index(self, node: ast.Subscript, tensor_type: ConcreteType) -> str:
        """The row-major element offset of `v[i, j, ...]` into a Tensor: an index below zero counts from the end of
        its axis, as in numpy, and one still outside the axis fails."""

        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        extents = [int(extent) for extent in tensor_type.arguments[1:]]  # type: ignore[arg-type]
        offset = "0"
        for index_node, extent in zip(indices, extents, strict=True):
            index_type = self.info.expression_type(index_node)
            index = self.convert_scalar(self.expression(index_node), index_type.name, "i64")
            if not index_type.name.startswith("u"):
                negative = self.value(f"icmp slt i64 {index}, 0")
                wrapped = self.value(f"add i64 {index}, {extent}")
                index = self.value(f"select i1 {negative}, i64 {wrapped}, i64 {index}")
            self.fail_if(self.value(f"icmp uge i64 {index}, {extent}"), HostStatus.INDEX_OUT_OF_RANGE)
            scaled = self.value(f"mul i64 {offset}, {extent}")
            offset = self.value(f"add i64 {scaled}, {index}")
        return offset

    def elementwise(self, result_type: ConcreteType, operands: list[str], operation) -> str:
        scalar = element_of(result_type).name
        if result_type.kind != "tensor":
            return operation(*operands, scalar)
        items = []
        for index in range(tensor_count(result_type)):
            parts = [self.extract(operand, result_type, index) for operand in operands]
            items.append(operation(*parts, scalar))
        return self.build(result_type, items)

    def negate(self, value: str, name: str) -> str:
        llvm_type = _LLVM_SCALARS[name]
        if _is_float(name):
            return self.value(f"fsub {llvm_type} 0.0, {value}")
        return self.value(f"sub {llvm_type} 0, {value}")

    def arithmetic(
        self,
        operation: str,
        left: str,
        left_type: ConcreteType,
        right: str,
        right_type: ConcreteType,
        result_type: ConcreteType,
    ) -> str:
        left = self.convert(left, left_type, result_type)
        right = self.convert(right, right_type, result_type)
        return self.elementwise(
            result_type, [left, right], lambda a, b, name: self.binary_scalar(operation, a, b, name)
        )

    def binary_scalar(self, operation: str, left: str, right: str, name: str) -> str:
        llvm_type = _LLVM_SCALARS[name]
        if _is_float(name):
            if operation in {"add", "sub", "mul", "div"}:
                instruction = {"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv"}[operation]
                return self.value(f"{instruction} {llvm_type} {left}, {right}")
            if operation == "pow":
                function = self.module.intrinsic("pow", llvm_type, 2)
                return self.value(f"call {llvm_type} {function}({llvm_type} {left}, {llvm_type} {right})")
            if operation == "mod":
                return self.float_remainder(left, right, llvm_type)
            if operation == "floordiv":
                return self.float_floor_quotient(left, right, llvm_type)
            raise ValueError(f"unsupported float operation {operation}")
        if operation in {"add", "sub", "mul", "and", "or", "xor"}:
            return self.value(f"{operation} {llvm_type} {left}, {right}")
        if operation in {"shl", "shr"}:
            return self.shift(operation, left, right, name)
        if operation not in {"floordiv", "mod"}:
            raise ValueError(f"unsupported integer operation {operation}")
        self.fail_if(self.value(f"icmp eq {llvm_type} {right}, 0"), HostStatus.DIVISION_BY_ZERO)
        if name.startswith("u"):
            return self.value(f"{'udiv' if operation == 'floordiv' else 'urem'} {llvm_type} {left}, {right}")
        # sdiv and srem are undefined for MIN / -1; dividing by 1 instead and negating gives the wrapped quotient.
        minus_one = self.value(f"icmp eq {llvm_type} {right}, -1")
        divisor = self.value(f"select i1 {minus_one}, {llvm_type} 1, {llvm_type} {right}")
        remainder = self.value(f"srem {llvm_type} {left}, {divisor}")
        inexact = self.value(f"icmp ne {llvm_type} {remainder}, 0")
        signs = self.value(f"xor {llvm_type} {remainder}, {divisor}")
        opposite = self.value(f"icmp slt {llvm_type} {signs}, 0")
        adjust = self.value(f"and i1 {inexact}, {opposite}")
        if operation == "mod":
            adjusted = self.value(f"add {llvm_type} {remainder}, {divisor}")
            return self.value(f"select i1 {adjust}, {llvm_type} {adjusted}, {llvm_type} {remainder}")
        quotient = self.value(f"sdiv {llvm_type} {left}, {divisor}")
        floored = self.value(f"sub {llvm_type} {quotient}, 1")
        quotient = self.value(f"select i1 {adjust}, {llvm_type} {floored}, {llvm_type} {quotient}")
        negated = self.value(f"sub {llvm_type} 0, {left}")
        return self.value(f"select i1 {minus_one}, {llvm_type} {negated}, {llvm_type} {quotient}")

    def shift(self, operation: str, left: str, right: str, name: str) -> str:
        # LLVM shifts are poison from the bit width on; numpy defines them, and so does Host code: a left or
        # unsigned right shift by the width or more, or by a negative amount, is 0, and a signed right shift fills
        # with the sign. The unsigned compare also catches negative amounts.
        llvm_type = _LLVM_SCALARS[name]
        width = int(llvm_type[1:])
        oversized = self.value(f"icmp uge {llvm_type} {right}, {width}")
        if operation == "shr" and not name.startswith("u"):
            amount = self.value(f"select i1 {oversized}, {llvm_type} {width - 1}, {llvm_type} {right}")
            return self.value(f"ashr {llvm_type} {left}, {amount}")
        shifted = self.value(f"{'shl' if operation == 'shl' else 'lshr'} {llvm_type} {left}, {right}")
        return self.value(f"select i1 {oversized}, {llvm_type} 0, {llvm_type} {shifted}")

    def float_remainder(self, left: str, right: str, llvm_type: str) -> str:
        remainder = self.value(f"frem {llvm_type} {left}, {right}")
        nonzero = self.value(f"fcmp une {llvm_type} {remainder}, 0.0")
        remainder_negative = self.value(f"fcmp olt {llvm_type} {remainder}, 0.0")
        divisor_negative = self.value(f"fcmp olt {llvm_type} {right}, 0.0")
        differ = self.value(f"xor i1 {remainder_negative}, {divisor_negative}")
        adjusted = self.value(f"fadd {llvm_type} {remainder}, {right}")
        signed = self.value(f"select i1 {differ}, {llvm_type} {adjusted}, {llvm_type} {remainder}")
        copysign = self.module.intrinsic("copysign", llvm_type, 2)
        zero = self.value(f"call {llvm_type} {copysign}({llvm_type} 0.0, {llvm_type} {right})")
        return self.value(f"select i1 {nonzero}, {llvm_type} {signed}, {llvm_type} {zero}")

    def float_floor_quotient(self, left: str, right: str, llvm_type: str) -> str:
        # numpy npy_divmod; see _floored_float_quotient in the device frontend. A zero divisor gives left / right.
        quotient = self.value(f"fdiv {llvm_type} {left}, {right}")
        remainder = self.value(f"frem {llvm_type} {left}, {right}")
        difference = self.value(f"fsub {llvm_type} {left}, {remainder}")
        exact = self.value(f"fdiv {llvm_type} {difference}, {right}")
        nonzero = self.value(f"fcmp une {llvm_type} {remainder}, 0.0")
        remainder_negative = self.value(f"fcmp olt {llvm_type} {remainder}, 0.0")
        divisor_negative = self.value(f"fcmp olt {llvm_type} {right}, 0.0")
        differ = self.value(f"xor i1 {remainder_negative}, {divisor_negative}")
        shift = self.value(f"and i1 {nonzero}, {differ}")
        lowered = self.value(f"fsub {llvm_type} {exact}, 1.0")
        exact = self.value(f"select i1 {shift}, {llvm_type} {lowered}, {llvm_type} {exact}")
        floor = self.module.intrinsic("floor", llvm_type)
        floored = self.value(f"call {llvm_type} {floor}({llvm_type} {exact})")
        fraction = self.value(f"fsub {llvm_type} {exact}, {floored}")
        rounds_up = self.value(f"fcmp ogt {llvm_type} {fraction}, 0.5")
        raised = self.value(f"fadd {llvm_type} {floored}, 1.0")
        floored = self.value(f"select i1 {rounds_up}, {llvm_type} {raised}, {llvm_type} {floored}")
        copysign = self.module.intrinsic("copysign", llvm_type, 2)
        zero = self.value(f"call {llvm_type} {copysign}({llvm_type} 0.0, {llvm_type} {quotient})")
        nonzero_quotient = self.value(f"fcmp une {llvm_type} {exact}, 0.0")
        result = self.value(f"select i1 {nonzero_quotient}, {llvm_type} {floored}, {llvm_type} {zero}")
        divisor_zero = self.value(f"fcmp oeq {llvm_type} {right}, 0.0")
        return self.value(f"select i1 {divisor_zero}, {llvm_type} {quotient}, {llvm_type} {result}")

    def compare_scalar(self, operation: str, left: str, right: str, name: str) -> str:
        llvm_type = _LLVM_SCALARS[name]
        if _is_float(name):
            return self.value(f"fcmp {_FLOAT_PREDICATES[operation]} {llvm_type} {left}, {right}")
        predicates = _UNSIGNED_PREDICATES if name.startswith("u") or name == "bool" else _SIGNED_PREDICATES
        return self.value(f"icmp {predicates[operation]} {llvm_type} {left}, {right}")

    def compare(self, node: ast.Compare) -> str:
        if isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return self.dict_contains(node)
        operation = _COMPARE[type(node.ops[0])]
        if self.info.expression_type(node.left).kind == "str":
            equal = self.str_equal(node.left, node.comparators[0])
            return equal if operation == "eq" else self.value(f"xor i1 {equal}, true")
        typed = self.info.expressions[id(node)]
        common = typed.operand_types[0] if typed.operand_types else self.info.expression_type(node.left)
        left = self.typed(node.left, common)
        right = self.typed(node.comparators[0], common)
        if common.kind in {"handle", "enum"}:
            width = self.module.memory_type(common)
            return self.value(f"icmp {'eq' if operation == 'eq' else 'ne'} {width} {left}, {right}")
        if common.kind == "tensor":
            scalar = element_of(common).name
            items = [
                self.compare_scalar(
                    operation, self.extract(left, common, index), self.extract(right, common, index), scalar
                )
                for index in range(tensor_count(common))
            ]
            return self.build(self.info.expression_type(node), items)
        return self.compare_scalar(operation, left, right, common.name)

    # Calls.

    # Selections: numpy-style indices with slices into a Tensor value or a TensorStorage (see language/slicing.py).
    # Each axis evaluates to a checked position, constant positions (a `range`), or a window `(start, length)`;
    # a window start is clamped into a Tensor value and checked against a storage.

    def selection_plan(self, node: ast.Subscript) -> list[AxisSelection] | None:
        """The plan of a subscript that selects more than one element, or None for an element or a non-tensor."""

        container = self.info.expression_type(node.value)
        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        extents = self.selection_extents(container)
        if extents is None or not (is_slicing(indices, len(extents)) or len(indices) > len(extents)):
            return None
        return plan_selection(node, indices, extents)

    @staticmethod
    def selection_extents(container: ConcreteType) -> list[int | None] | None:
        if container.kind == "tensor":
            return [int(extent) for extent in container.arguments[1:]]  # type: ignore[arg-type]
        if container.kind == "storage":
            return [extent if isinstance(extent, int) else None for extent in container.arguments[1]]  # type: ignore[union-attr]
        return None

    @staticmethod
    def selection_index_nodes(plan: list[AxisSelection]) -> list[ast.expr]:
        return [
            axis.node if isinstance(axis, AxisIndex) else axis.start for axis in plan if not isinstance(axis, AxisRange)
        ]

    def selection_axes(
        self, plan: list[AxisSelection], extents: list[str], clamp: bool, raw: list[tuple[str, bool]]
    ) -> list[str | range | tuple[str, int]]:
        raw_values = iter(raw)
        axes: list[str | range | tuple[str, int]] = []
        for axis, extent in zip(plan, extents, strict=True):
            if isinstance(axis, AxisRange):
                axes.append(range(axis.start, axis.start + axis.step * axis.length, axis.step))
                continue
            index, signed = next(raw_values)
            if signed:
                negative = self.value(f"icmp slt i64 {index}, 0")
                wrapped = self.value(f"add i64 {index}, {extent}")
                index = self.value(f"select i1 {negative}, i64 {wrapped}, i64 {index}")
            if isinstance(axis, AxisIndex):
                self.fail_if(self.value(f"icmp uge i64 {index}, {extent}"), HostStatus.INDEX_OUT_OF_RANGE)
                axes.append(index)
            elif clamp:
                last = self.value(f"sub i64 {extent}, {axis.length}")
                below = self.value(f"icmp slt i64 {index}, 0")
                index = self.value(f"select i1 {below}, i64 0, i64 {index}")
                above = self.value(f"icmp sgt i64 {index}, {last}")
                axes.append((self.value(f"select i1 {above}, i64 {last}, i64 {index}"), axis.length))
            else:
                end = self.value(f"add i64 {index}, {axis.length}")
                before = self.value(f"icmp slt i64 {index}, 0")
                after = self.value(f"icmp sgt i64 {end}, {extent}")
                self.fail_if(self.value(f"or i1 {before}, {after}"), HostStatus.INDEX_OUT_OF_RANGE)
                axes.append((index, axis.length))
        return axes

    def selection_raw(self, plan: list[AxisSelection]) -> list[tuple[str, bool]]:
        """The index and window start values, in source order, as i64 with their signedness."""

        raw = []
        for node in self.selection_index_nodes(plan):
            index_type = self.info.expression_type(node)
            raw.append(
                (
                    self.convert_scalar(self.expression(node), index_type.name, "i64"),
                    not index_type.name.startswith("u"),
                )
            )
        return raw

    def selection_offsets(self, axes: list[str | range | tuple[str, int]], extents: list[str]) -> list[str]:
        """The row-major element offsets of the selection, in the order of the selected Tensor's elements."""

        options: list[list[str]] = []
        for axis in axes:
            if isinstance(axis, range):
                options.append([str(position) for position in axis])
            elif isinstance(axis, tuple):
                start, length = axis
                options.append([start, *(self.value(f"add i64 {start}, {offset}") for offset in range(1, length))])
            else:
                options.append([axis])
        offsets = []
        for coordinate in itertools.product(*options):
            offset = "0"
            for position, extent in zip(coordinate, extents, strict=True):
                offset = self.value(f"add i64 {self.value(f'mul i64 {offset}, {extent}')}, {position}")
            offsets.append(offset)
        return offsets

    def gathered(self, result_type: ConcreteType, element: ConcreteType, pointers: list[str]) -> str:
        items = [self.load(element, pointer) for pointer in pointers]
        return self.build(result_type, items) if result_type.kind == "tensor" else items[0]

    def tensor_selection(self, node: ast.Subscript, plan: list[AxisSelection]) -> str:
        """`v[...]` with slices: a Tensor of the selected elements, or one element."""

        tensor_type = self.info.expression_type(node.value)
        memory = self.module.memory_type(tensor_type)
        pointer = self.spill(self.expression(node.value), tensor_type)
        extents = [str(extent) for extent in tensor_type.arguments[1:]]
        axes = self.selection_axes(plan, extents, True, self.selection_raw(plan))
        pointers = [
            self.value(f"getelementptr {memory}, ptr {pointer}, i64 0, i64 {offset}")
            for offset in self.selection_offsets(axes, extents)
        ]
        return self.gathered(self.info.expression_type(node), element_of(tensor_type), pointers)

    def storage_selection(self, node: ast.Subscript, plan: list[AxisSelection]) -> list[str]:
        """The element addresses `s[...]` with slices selects. Indices are evaluated before the record is read, as
        for an element."""

        storage_type = self.info.expression_type(node.value)
        raw = self.selection_raw(plan)
        record = self.storage_record(node.value)
        extents = [self.storage_extent(record, storage_type, axis) for axis in range(storage_rank(storage_type))]
        axes = self.selection_axes(plan, extents, False, raw)
        element = self.module.memory_type(element_of(storage_type))
        data = self.storage_data(record)
        return [
            self.value(f"getelementptr {element}, ptr {data}, i64 {offset}")
            for offset in self.selection_offsets(axes, extents)
        ]

    def scatter(
        self, pointers: list[str], element: ConcreteType, value: str, value_type: ConcreteType, add: bool = False
    ) -> None:
        """Stores a scalar into every selected element, or a Tensor of the selection's shape element by element."""

        if value_type.kind == "tensor":
            block_type = ConcreteType("tensor", "Tensor", (element, *value_type.arguments[1:]))
            block = self.spill(self.convert(value, value_type, block_type), block_type)
            memory = self.module.memory_type(block_type)
            sources = [
                self.load(element, self.value(f"getelementptr {memory}, ptr {block}, i64 0, i64 {index}"))
                for index in range(len(pointers))
            ]
        else:
            sources = [self.convert(value, value_type, element)] * len(pointers)
        for pointer, source in zip(pointers, sources, strict=True):
            if add:
                old = self.load(element, pointer)
                source = self.arithmetic(_BINARY[ast.Add], old, element, source, element, element)
            self.store(source, element, pointer)

    def tensor_update(self, node: ast.Call, tensor_node: ast.expr, indices: list[ast.expr], method: str) -> str:
        """`v.at[...].set(x)` or `.add(x)`: a copy of `v` with the selected elements replaced. The receiver, the
        indices and the value are evaluated in that order, as Python evaluates them."""

        tensor_type = self.info.expression_type(tensor_node)
        element = element_of(tensor_type)
        memory = self.module.memory_type(tensor_type)
        copy = self.spill(self.expression(tensor_node), tensor_type)
        assert isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Subscript)
        plan = plan_selection(node.func.value, indices, self.selection_extents(tensor_type) or [])
        extents = [str(extent) for extent in tensor_type.arguments[1:]]
        axes = self.selection_axes(plan, extents, True, self.selection_raw(plan))
        pointers = [
            self.value(f"getelementptr {memory}, ptr {copy}, i64 0, i64 {offset}")
            for offset in self.selection_offsets(axes, extents)
        ]
        value = self.expression(node.args[0])
        self.scatter(pointers, element, value, self.info.expression_type(node.args[0]), add=method == "add")
        return self.load(tensor_type, copy)

    def struct_replace(self, node: ast.Call, struct_type: ConcreteType) -> str:
        """`replace(p, f=x, ...)`: `p` with the named fields changed, evaluated in source order."""

        struct = self.expression(node.args[0])
        fields = self.program.structs[struct_type.name].fields
        changes = {keyword.arg: self.typed(keyword.value, dict(fields)[keyword.arg]) for keyword in node.keywords}
        return self.build(
            struct_type,
            [
                changes[name] if name in changes else self.extract(struct, struct_type, index)
                for index, (name, _) in enumerate(fields)
            ],
        )

    def call(self, node: ast.Call, result_type: ConcreteType) -> str:
        name = _call_name(node)
        program = self.program
        if (update := tensor_update_parts(node)) is not None:
            return self.tensor_update(node, *update)
        if name == "replace" and name not in program.functions and result_type.kind == "struct":
            return self.struct_replace(node, result_type)
        if isinstance(node.func, ast.Attribute) and id(node.func.value) in self.info.expressions:
            receiver = self.info.expression_type(node.func.value).kind
            if receiver == "list":
                return self.list_method(node)
            if receiver == "dict":
                return self.dict_method(node)
            if receiver == "storage":
                return self.storage_method(node)
        if name == "len" and isinstance(node.func, ast.Name):
            measured = self.info.expression_type(node.args[0])
            if measured.kind == "str":
                size = self.str_length(node.args[0])
            elif measured.kind == "storage":
                size = self.storage_extent(self.storage_record(node.args[0]), measured, 0)
            else:
                # A list's and a dict's size are both their record's second i64.
                size = self.list_size(self.container(node.args[0]))
            return self.value(f"trunc i64 {size} to i32")
        if name in program.externs:
            return self.extern_call(program.externs[name], node)
        if name in program.functions:
            return self.function_call(program.functions[name], node)
        if name in program.structs:
            fields = program.structs[name].fields
            # Positional values, then keywords, evaluate in source order, as Python does.
            values = {
                field_name: self.typed(argument, field)
                for argument, (field_name, field) in zip(node.args, fields, strict=False)
            }
            types = dict(fields)
            for keyword in node.keywords:
                assert keyword.arg is not None
                values[keyword.arg] = self.typed(keyword.value, types[keyword.arg])
            return self.build(result_type, [values[field_name] for field_name, _ in fields])
        if name in program.handles:
            return "0"
        if name in _CAST_NAMES:
            return self.typed(node.args[0], result_type)
        if name in {"Vector", "Matrix", "Tensor"}:
            return self.aggregate(node, result_type)
        if name == "Tuple":
            return self.build(
                result_type,
                [
                    self.typed(item, item_type)
                    for item, item_type in zip(node.args, self.member_types(result_type), strict=True)
                ],
            )
        return self.intrinsic(name, node, result_type)

    def function_call(self, callee: HostFunctionInfo, node: ast.Call, out: str | None = None) -> str:
        """Calls a Host function. A class result is built in `out`, which holds no instance; a new instance passed
        as an argument lives in a temporary, released after the call."""

        arguments = []
        temporaries = []
        for position, (argument, (_, parameter)) in enumerate(zip(node.args, callee.parameters, strict=True)):
            if parameter.kind == "str":
                data, size = self.stable_str_operand(argument, node.args[position + 1 :])
                view = self.alloca("{ ptr, i64 }")
                self.instruction(f"store ptr {data}, ptr {view}")
                size_pointer = self.value(f"getelementptr inbounds {{ ptr, i64 }}, ptr {view}, i32 0, i32 1")
                self.instruction(f"store i64 {size}, ptr {size_pointer}")
                arguments.append(f"ptr {view}")
                continue
            if self.program.is_place(parameter):
                if self.creates(argument):
                    address = self.instance(argument)
                    temporaries.append((address, parameter))
                else:
                    address = self.place_address(argument)
                assert address is not None, "the Host validator admits only places and new instances as instances"
                arguments.append(f"ptr {address}")
                continue
            value = self.typed(argument, parameter)
            if parameter.kind in {"scalar", "handle", "enum"}:
                arguments.append(f"{self.module.register_type(parameter)} {value}")
            else:
                arguments.append(f"ptr {self.spill(value, parameter)}")
        instance_result = callee.result is not None and self.program.is_place(callee.result)
        assert (out is not None) == instance_result, "a class result is built where the caller says"
        if callee.result is not None:
            if out is None:
                out = self.alloca(self.module.memory_type(callee.result))
            arguments.append(f"ptr {out}")
        callee_name = self.module.function_name(self.program, callee.symbol)
        status = self.value(f"call i32 {callee_name}({', '.join(arguments)})")
        self.propagate(status)
        for address, parameter in temporaries:
            self.release(address, parameter)
        if out is None or callee.result is None or instance_result:
            return "poison"
        return self.load(callee.result, out)

    def extern_call(self, extern: HostExtern, node: ast.Call, received: str | None = None) -> str:
        """Calls `extern`; one returning a str, list or dict fills `received`, an empty record of its
        `transfer_record` type."""

        module = self.module
        module.declare_extern(extern)
        lowerings = [item for item in module.extern_parameters(extern) if item[0] != "type"]
        arguments: list[str] = []
        # Instances the call moves into the callee, which takes their bytes whatever its status.
        moved: list[tuple[str, ConcreteType]] = []
        for position, (argument, (_, parameter), (lowering, text)) in enumerate(
            zip(node.args, extern.parameters, lowerings, strict=False)
        ):
            if self.program.is_place(parameter):
                address = self.instance(argument)
                moved.append((address, parameter))
                arguments.append(f"ptr {address}")
                continue
            if parameter.kind in TRANSFERRED_KINDS:
                if parameter.kind == "str":
                    data, size = self.stable_str_operand(argument, node.args[position + 1 :])
                elif parameter.kind == "dict":
                    data, size = self.dict_span(argument)
                else:
                    record = self.container(argument)
                    data, size = self.list_data(record), self.list_size(record)
                if lowering == "str_pair":
                    arguments.append(f"ptr {data}, i64 {size}")
                elif lowering == "str_array":
                    address = self.value(f"ptrtoint ptr {data} to i64")
                    first = self.value(f"insertvalue [2 x i64] poison, i64 {address}, 0")
                    arguments.append(f"[2 x i64] {self.value(f'insertvalue [2 x i64] {first}, i64 {size}, 1')}")
                else:
                    slot = self.alloca("{ ptr, i64 }")
                    pair = self.value(f"insertvalue {{ ptr, i64 }} poison, ptr {data}, 0")
                    pair = self.value(f"insertvalue {{ ptr, i64 }} {pair}, i64 {size}, 1")
                    self.instruction(f"store {{ ptr, i64 }} {pair}, ptr {slot}")
                    arguments.append(f"{text} {slot}")
                continue
            value = self.typed(argument, parameter)
            if lowering == "value":
                arguments.append(f"{text} {value}")
            else:
                arguments.append(f"ptr {self.spill(value, parameter)}")
        if extern.type_argument is not None:
            index, bound = extern.type_argument
            arguments.insert(index, f"ptr {type_descriptor(ConcreteType('struct', bound))}")
        if extern.abi == ExternAbi.STATUS:
            out = None
            if extern.result is not None and extern.result.kind in TRANSFERRED_KINDS:
                assert received is not None, "the Host validator admits these results only as assigned values"
                out = received
                arguments.append(f"ptr {out}, ptr {self.transfer_allocator()}")
            elif extern.result is not None:
                out = self.alloca(module.memory_type(extern.result))
                arguments.append(f"ptr {out}")
            status = self.value(f'call i32 @"{extern.symbol}"({", ".join(arguments)})')
            self.forget(moved)
            # On failure `received` is the function's own record, which its exit releases.
            self.propagate(status)
            if out is None or extern.result is None or out is received:
                return "poison"
            return self.load(extern.result, out)
        if extern.result is None:
            self.instruction(f'call void @"{extern.symbol}"({", ".join(arguments)})')
            self.forget(moved)
            return "poison"
        result_type = module.register_type(extern.result)
        attribute = "zeroext " if result_type == "i1" else ""
        result = self.value(f'call {attribute}{result_type} @"{extern.symbol}"({", ".join(arguments)})')
        self.forget(moved)
        return result

    def forget(self, moved: list[tuple[str, ConcreteType]]) -> None:
        """Zeroes instances a callee took, so the exits releasing their memory release nothing."""

        if not moved:
            return
        self.module.declare("llvm.memset.p0.i64", "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)")
        for address, value_type in moved:
            size = self.size_of(value_type)
            self.instruction(f"call void @llvm.memset.p0.i64(ptr {address}, i8 0, i64 {size}, i1 false)")

    def aggregate(self, node: ast.Call, result_type: ConcreteType) -> str:
        element = element_of(result_type)
        items: list[str] = []

        def flatten(literal: ast.expr) -> None:
            if isinstance(literal, (ast.List, ast.Tuple)):
                for item in literal.elts:
                    flatten(item)
                return
            item_type = self.info.expression_type(literal)
            if item_type.kind == "tensor":
                target = ConcreteType("tensor", "Tensor", (element, *item_type.arguments[1:]))
                value = self.typed(literal, target)
                items.extend(self.extract(value, target, index) for index in range(tensor_count(target)))
            else:
                items.append(self.typed(literal, element))

        flatten(node.args[0])
        if len(items) == 1 and tensor_count(result_type) != 1:
            items = items * tensor_count(result_type)
        return self.build(result_type, items)

    def intrinsic(self, name: str, node: ast.Call, result_type: ConcreteType) -> str:
        info = self.info
        arguments = list(node.args)
        if isinstance(node.func, ast.Attribute) and name in {"norm", "normalize"} and not arguments:
            arguments = [node.func.value]
        if name in {"any", "all"}:
            value_type = info.expression_type(arguments[0])
            value = self.typed(arguments[0], value_type)
            if value_type.kind != "tensor":
                return value
            combined = self.extract(value, value_type, 0)
            for index in range(1, tensor_count(value_type)):
                item = self.extract(value, value_type, index)
                combined = self.value(f"{'or' if name == 'any' else 'and'} i1 {combined}, {item}")
            return combined
        if name == "select":
            condition_type = info.expression_type(arguments[0])
            condition = self.typed(arguments[0], condition_type)
            true_value, false_value = (self.typed(argument, result_type) for argument in arguments[1:])
            llvm_type = _LLVM_SCALARS[element_of(result_type).name]
            if result_type.kind != "tensor":
                return self.value(f"select i1 {condition}, {llvm_type} {true_value}, {llvm_type} {false_value}")
            items = []
            for index in range(tensor_count(result_type)):
                picked = (
                    self.extract(condition, condition_type, index) if condition_type.kind == "tensor" else condition
                )
                when_true = self.extract(true_value, result_type, index)
                when_false = self.extract(false_value, result_type, index)
                items.append(self.value(f"select i1 {picked}, {llvm_type} {when_true}, {llvm_type} {when_false}"))
            return self.build(result_type, items)
        if name in {"dot", "norm", "normalize", "cross"}:
            vector_type = info.expression_type(arguments[0])
            if len(arguments) == 2:
                vector_type = _common_tensor(vector_type, info.expression_type(arguments[1]))
            vectors = [self.typed(argument, vector_type) for argument in arguments]
            scalar = element_of(vector_type).name
            count = tensor_count(vector_type)
            components = [[self.extract(vector, vector_type, index) for index in range(count)] for vector in vectors]

            def multiply(a: str, b: str) -> str:
                return self.binary_scalar("mul", a, b, scalar)

            def subtract(a: str, b: str) -> str:
                return self.binary_scalar("sub", a, b, scalar)

            if name == "cross":
                (a0, a1, a2), (b0, b1, b2) = components
                return self.build(
                    vector_type,
                    [
                        subtract(multiply(a1, b2), multiply(a2, b1)),
                        subtract(multiply(a2, b0), multiply(a0, b2)),
                        subtract(multiply(a0, b1), multiply(a1, b0)),
                    ],
                )
            left = components[0]
            right = components[1] if name == "dot" else components[0]
            total = multiply(left[0], right[0])
            for a, b in zip(left[1:], right[1:], strict=True):
                total = self.binary_scalar("add", total, multiply(a, b), scalar)
            if name == "dot":
                return total
            length = self.unary_scalar("sqrt", total, scalar)
            if name == "norm":
                return length
            return self.build(vector_type, [self.binary_scalar("div", item, length, scalar) for item in left])
        values = [self.typed(argument, result_type) for argument in arguments]
        if name in {"abs", "sqrt", "floor", "sin", "cos", "exp", "log", "acos"}:
            return self.elementwise(result_type, values, lambda value, scalar: self.unary_scalar(name, value, scalar))
        if name in {"min", "max", "pow", "atan2"}:
            return self.elementwise(
                result_type, values, lambda left, right, scalar: self.binary_intrinsic(name, left, right, scalar)
            )
        if name == "clamp":
            return self.elementwise(
                result_type,
                values,
                lambda value, low, high, scalar: self.binary_intrinsic(
                    "min", self.binary_intrinsic("max", value, low, scalar), high, scalar
                ),
            )
        raise TypeError(f"Host code cannot call '{name}'")

    def unary_scalar(self, name: str, value: str, scalar: str) -> str:
        llvm_type = _LLVM_SCALARS[scalar]
        if not _is_float(scalar):
            if name != "abs":
                raise ValueError(f"{name} requires floating-point operands")
            # `i1 false`: abs(MIN) wraps to MIN instead of being poison.
            self.module.declare(f"llvm.abs.{llvm_type}", f"declare {llvm_type} @llvm.abs.{llvm_type}({llvm_type}, i1)")
            return self.value(f"call {llvm_type} @llvm.abs.{llvm_type}({llvm_type} {value}, i1 false)")
        if name == "abs":
            function = self.module.intrinsic("fabs", llvm_type)
        elif name in _LIBM:
            function = self.libm(name, llvm_type, 1)
        else:
            function = self.module.intrinsic(_UNARY_INTRINSICS[name], llvm_type)
        return self.value(f"call {llvm_type} {function}({llvm_type} {value})")

    def libm(self, name: str, llvm_type: str, arity: int) -> str:
        symbol = _LIBM[name][0 if llvm_type == "float" else 1]
        operands = ", ".join([llvm_type] * arity)
        self.module.declare(symbol, f"declare {llvm_type} @{symbol}({operands})")
        return f"@{symbol}"

    def binary_intrinsic(self, name: str, left: str, right: str, scalar: str) -> str:
        llvm_type = _LLVM_SCALARS[scalar]
        if name in {"min", "max"}:
            picked = self.compare_scalar("lt" if name == "min" else "gt", left, right, scalar)
            return self.value(f"select i1 {picked}, {llvm_type} {left}, {llvm_type} {right}")
        if name == "pow":
            function = self.module.intrinsic("pow", llvm_type, 2)
        else:
            function = self.libm(name, llvm_type, 2)
        return self.value(f"call {llvm_type} {function}({llvm_type} {left}, {llvm_type} {right})")


def _owns_records(info: HostFunctionInfo, program: HostProgram) -> bool:
    """Whether a function needs records of its own: a str, storage, list or dict local, an instance an owned local
    or a temporary holds, or a str expression other than a literal, which an f-string or + builds into, or which a
    field is copied into before a call could change it."""

    if any(value_type.kind in _OWNED_LOCAL_KINDS for value_type in info.local_types.values()):
        return True
    if "owned" in info.class_locals.values():
        return True

    def owns(node: ast.AST) -> bool:
        # The span of a dict passed to an extern is laid out in a list of the function's own.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in program.externs:
            if any(parameter.kind == "dict" for _, parameter in program.externs[node.func.id].parameters):
                return True
        if id(node) not in info.expressions:
            return False
        value_type = info.expression_type(node)  # type: ignore[arg-type]
        if value_type.kind == "str":
            return not isinstance(node, ast.Constant)
        # A container copy and an extern's container result are built in records of the function's own.
        if isinstance(node, ast.Call) and value_type.kind in {"list", "dict"}:
            return True
        return isinstance(node, ast.Call) and program.is_place(value_type)

    return any(owns(node) for statement in info.source.body for node in ast.walk(statement))


def _common_tensor(left: ConcreteType, right: ConcreteType) -> ConcreteType:
    if left == right:
        return left
    common = common_type(left, right)
    if not isinstance(common, ConcreteType):
        raise TypeError(f"no common type for {left.spelling} and {right.spelling}")
    return common


def emit_llvm(
    programs: list[HostProgram],
    abi: TargetAbi,
    prefix: str,
    asset_id: str | None = None,
    module_name: str | None = None,
) -> str:
    module = ModuleEmitter(abi, prefix, asset_id, module_name)
    for program in programs:
        module.add_program(program)
    return module.text()


__all__ = ["ModuleEmitter", "TargetAbi", "emit_llvm", "StructFields"]
