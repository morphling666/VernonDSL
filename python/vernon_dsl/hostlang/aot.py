"""Ahead-of-time compilation of Host programs to native or wasm32 objects, the code packaged builds link.

An artifact is the object, a C header declaring its entries and the types they exchange, and reflection JSON. Every
entry has the internal Host convention: `int32_t <prefix><entry>(params..., Result *out)`, returning zero or a
nonzero status.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..frontend.model import ConcreteType
from .layout import OWNED_KINDS, layout, members, storage_rank, tensor_count
from .llvm_emitter import TargetAbi, emit_llvm
from .model import HostProgram, HostStatus, check_combinable
from .reflect import layout_hash
from .values import element_of, key_of

REFLECTION_VERSION = 1
_C_SCALARS = {
    "bool": "bool",
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
    "f32": "float",
    "f64": "double",
}


def _native() -> Any:
    from .. import _native

    return _native


def host_triple() -> str:
    """The normalized triple of the machine running Python, the default object target."""

    return _native()._default_host_triple()


HostSources = str | Path | Sequence[str | Path]


def source_paths(sources: HostSources) -> list[Path]:
    """One source file or several, resolved, each once, in the given order."""

    items = [sources] if isinstance(sources, (str, Path)) else list(sources)
    if not items:
        raise ValueError("a Host module needs at least one source file")
    return list(dict.fromkeys(Path(item).resolve() for item in items))


def load_programs(sources: HostSources, entries: Iterable[str] | None = None) -> list[HostProgram]:
    """The programs of `entries` of one source file, or of every listed source file's whole module when `entries`
    is None; several modules share one object and table, so they must pass `check_combinable`."""

    from .frontend import load_host_module, load_host_program

    paths = source_paths(sources)
    if entries is None:
        programs = [load_host_module(path) for path in paths]
        check_combinable(programs)
        return programs
    if len(paths) != 1:
        raise ValueError("entries select from one source file")
    return [load_host_program(paths[0], entry) for entry in entries]


@dataclass(frozen=True)
class HostArtifacts:
    object: bytes
    object_suffix: str
    header: str
    reflection: dict[str, Any]
    ir: str
    optimized_ir: str
    triple: str
    programs: tuple[HostProgram, ...] = field(repr=False, default=())

    def write(self, directory: str | Path, name: str) -> dict[str, Path]:
        """Writes `<name><object suffix>`, `<name>.h`, and `<name>.json`; returns their paths."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "object": output / f"{name}{self.object_suffix}",
            "header": output / f"{name}.h",
            "reflection": output / f"{name}.json",
        }
        paths["object"].write_bytes(self.object)
        paths["header"].write_text(self.header, encoding="utf-8", newline="\n")
        paths["reflection"].write_text(json.dumps(self.reflection, indent=2) + "\n", encoding="utf-8", newline="\n")
        return paths


def _object_suffix(triple: str) -> str:
    if triple.startswith("wasm"):
        return ".wasm.o"
    return ".obj" if "windows" in triple and "gnu" not in triple else ".o"


def compile_host(
    sources: HostSources,
    entries: Iterable[str] | None = None,
    *,
    target: str | None = None,
    prefix: str = "vernon_host_",
    type_prefix: str | None = None,
    types_header: str | None = None,
    cpu: str = "",
    features: str = "",
    asset_id: str | None = None,
) -> HostArtifacts:
    """Compiles the `@host` `entries` of one source file and everything they reach into one object; without
    `entries`, every `@host` function and class each listed source file defines, all in one object and table.

    `types_header` names a C header that already declares every `c_type` Struct and handle, such as an engine's C
    API header; the generated header includes it and statically checks those layouts instead of redefining them.

    `asset_id` makes the object register its table with VernonRuntime under that id from a static constructor, as a
    cooked Program asset does, and names the table by it; such an object links against VernonRuntime.
    """

    programs = load_programs(sources, entries)
    return compile_programs(
        programs,
        target=target,
        prefix=prefix,
        type_prefix=type_prefix,
        types_header=types_header,
        cpu=cpu,
        features=features,
        asset_id=asset_id,
    )


def compile_programs(
    programs: list[HostProgram],
    *,
    target: str | None = None,
    prefix: str = "vernon_host_",
    type_prefix: str | None = None,
    types_header: str | None = None,
    cpu: str = "",
    features: str = "",
    asset_id: str | None = None,
) -> HostArtifacts:
    if not prefix.isidentifier():
        raise ValueError(f"Host symbol prefix '{prefix}' is not a C identifier prefix")
    triple = target or host_triple()
    abi = TargetAbi.from_triple(triple)
    ir = emit_llvm(programs, abi, prefix, asset_id, module_name=asset_id)
    compiled = _native()._compile_host(ir, triple, cpu, features)
    names = _CNames(programs, type_prefix or _camel(prefix))
    return HostArtifacts(
        object=bytes(compiled["object"]),
        object_suffix=_object_suffix(compiled["triple"]),
        header=_header(programs, names, prefix, types_header),
        reflection=_reflection(programs, names, prefix, compiled["triple"]),
        ir=ir,
        optimized_ir=compiled["optimized_ir"],
        triple=compiled["triple"],
        programs=tuple(programs),
    )


def _camel(prefix: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in prefix.split("_") if part)


class _CNames:
    """C spellings of Host types: `c_type` names where declared, prefixed generated names otherwise."""

    def __init__(self, programs: list[HostProgram], type_prefix: str):
        self.type_prefix = type_prefix
        self.structs: dict[str, Any] = {}
        self.handles: dict[str, Any] = {}
        self.tuples: dict[ConcreteType, str] = {}
        for program in programs:
            self.structs.update(program.structs)
            self.handles.update(program.handles)

    def fields(self, name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return self.structs[name].fields

    def name(self, value_type: ConcreteType) -> str:
        if value_type.kind == "scalar":
            return _C_SCALARS[value_type.name]
        if value_type.kind == "handle":
            return self.handles[value_type.name].c_type or f"{self.type_prefix}{value_type.name}"
        if value_type.kind == "struct":
            return self.structs[value_type.name].c_type or f"{self.type_prefix}{value_type.name}"
        if value_type.kind == "tuple":
            if value_type not in self.tuples:
                self.tuples[value_type] = f"{self.type_prefix}Tuple{len(self.tuples)}"
            return self.tuples[value_type]
        if value_type.kind == "tensor":
            return self.name(element_of(value_type))
        # reflect.h spells these as the VhList, VhDict and storage records in C, and containers.hpp as the typed vh::
        # classes.
        if value_type.kind == "str":
            return "VH_STR"
        if value_type.kind == "list":
            return f"VH_LIST({self.type_id(element_of(value_type))})"
        if value_type.kind == "dict":
            return f"VH_DICT({self.type_id(key_of(value_type))}, {self.type_id(element_of(value_type))})"
        if value_type.kind == "storage":
            return f"VH_STORAGE({self.name(element_of(value_type))}, {storage_rank(value_type)})"
        raise ValueError(f"{value_type.spelling} has no C spelling")

    def type_id(self, value_type: ConcreteType) -> str:
        """The type as a template argument spells it: a Tensor is its flat C array."""

        if value_type.kind == "tensor":
            return f"{self.name(value_type)}[{tensor_count(value_type)}]"
        return self.name(value_type)

    def declarator(self, value_type: ConcreteType, name: str) -> str:
        if value_type.kind == "tensor":
            return f"{self.name(value_type)} {name}[{tensor_count(value_type)}]"
        return f"{self.name(value_type)} {name}"

    def external(self, value_type: ConcreteType) -> bool:
        if value_type.kind == "handle":
            return self.handles[value_type.name].c_type is not None
        if value_type.kind == "struct":
            return self.structs[value_type.name].c_type is not None
        return False


def _program_struct_fields(programs: list[HostProgram]) -> Any:
    def fields(name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return next(program.structs[name].fields for program in programs if name in program.structs)

    return fields


def _boundary_types(programs: list[HostProgram]) -> list[ConcreteType]:
    """Struct, handle, and Tuple types entries exchange, dependencies first."""

    ordered: list[ConcreteType] = []

    def visit(value_type: ConcreteType) -> None:
        if value_type.kind == "tensor":
            visit(element_of(value_type))
            return
        if value_type.kind in {"list", "dict"}:
            # C++ spells a container field with its key and element types, which must be declared before it.
            for argument in value_type.arguments:
                if isinstance(argument, ConcreteType):
                    visit(argument)
            return
        if value_type.kind not in {"struct", "handle", "tuple"} or value_type in ordered:
            return
        if value_type.kind != "handle":
            fields = _program_struct_fields(programs)
            for member in members(value_type, fields):
                visit(member)
        ordered.append(value_type)

    for program in programs:
        for host_struct in program.structs.values():
            visit(ConcreteType("struct", host_struct.name))
        for handle in program.handles.values():
            visit(ConcreteType("handle", handle.name))
        for _, info in program.exported():
            for _, parameter in info.parameters:
                visit(parameter)
            if info.result is not None:
                visit(info.result)
    return ordered


def _header(programs: list[HostProgram], names: _CNames, prefix: str, types_header: str | None) -> str:
    guard = re.sub(r"[^A-Za-z0-9]", "_", prefix).upper() + "H"
    status = prefix.upper()
    lines = [
        "/* Generated by vernon_dsl.hostlang; do not edit. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        '#include "vernon_host/reflect.h"',
    ]
    if types_header:
        lines.append(f'#include "{types_header}"')
    boundary = _boundary_types(programs)
    if any(
        member.kind in OWNED_KINDS
        for value_type in boundary
        if value_type.kind != "handle"
        for member in members(value_type, names.fields)
    ):
        lines += ["#ifdef __cplusplus", '#include "vernon_host/containers.hpp"', "#endif"]
    lines += [
        "",
        "#ifdef __cplusplus",
        "#define VERNON_HOST_STATIC_ASSERT(condition, message) static_assert(condition, message)",
        'extern "C" {',
        "#else",
        "#define VERNON_HOST_STATIC_ASSERT(condition, message) _Static_assert(condition, message)",
        "#endif",
        "",
        f"#define {status}OK 0",
    ]
    for value in HostStatus:
        if value is not HostStatus.OK:
            lines.append(f"#define {status}{value.name} ({int(value)})")
    lines.append("")
    for value_type in boundary:
        c_name = names.name(value_type)
        external = types_header is not None and names.external(value_type)
        if value_type.kind == "handle":
            if not external:
                lines.append(f"typedef struct {c_name} {{ uint64_t bits; }} {c_name};")
            lines.append(f'VERNON_HOST_STATIC_ASSERT(sizeof({c_name}) == 8, "{c_name} is one uint64_t");')
            continue
        value_layout = layout(value_type, names.fields)
        if value_type.kind == "struct":
            field_names = [name for name, _ in names.fields(value_type.name)]
        else:
            field_names = [f"item{index}" for index in range(len(value_layout.offsets))]
        member_types = members(value_type, names.fields)
        if not external:
            body = " ".join(
                f"{names.declarator(member, name)};" for name, member in zip(field_names, member_types, strict=True)
            )
            lines.append(f"typedef struct {c_name} {{ {body or 'uint8_t unused;'} }} {c_name};")
        lines.append(f'VERNON_HOST_STATIC_ASSERT(sizeof({c_name}) == {value_layout.size}, "{c_name} size");')
        for name, offset in zip(field_names, value_layout.offsets, strict=True):
            lines.append(
                f'VERNON_HOST_STATIC_ASSERT(offsetof({c_name}, {name}) == {offset}, "{c_name}.{name} offset");'
            )
    lines.append("")
    lines.append("/* The reflection table of every type and function below. */")
    lines.append(f"extern const VhModule {prefix}module;")
    lines.append("")
    declared: set[str] = set()
    for program in programs:
        for exported, info in program.exported():
            if exported in declared:
                continue
            declared.add(exported)
            parameters = []
            for name, parameter in info.parameters:
                if program.is_place(parameter):
                    parameters.append(f"{names.name(parameter)} *{name}")
                elif parameter.kind in {"scalar", "handle"}:
                    parameters.append(f"{names.name(parameter)} {name}")
                else:
                    parameters.append(f"const {names.name(parameter)} *{name}")
            if info.result is not None:
                parameters.append(f"{names.name(info.result)} *out")
            lines.append(f"/* effects: {info.declared_effect.spelling} */")
            lines.append(f"int32_t {prefix}{exported}({', '.join(parameters) or 'void'});")
    lines += ["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif /* {guard} */", ""]
    return "\n".join(lines)


def _type_record(value_type: ConcreteType) -> str:
    return value_type.spelling


def _reflection(programs: list[HostProgram], names: _CNames, prefix: str, triple: str) -> dict[str, Any]:
    structs = {}
    handles = {}
    externs = {}
    classes: dict[str, Any] = {}
    entries = []
    sources: dict[str, str] = {}
    for program in programs:
        for host_struct in program.structs.values():
            value_type = ConcreteType("struct", host_struct.name)
            value_layout = layout(value_type, names.fields)
            structs[host_struct.name] = {
                "c_type": names.name(value_type),
                "size": value_layout.size,
                "alignment": value_layout.alignment,
                "fields": [
                    {"name": name, "type": _type_record(member), "offset": offset}
                    for (name, member), offset in zip(host_struct.fields, value_layout.offsets, strict=True)
                ],
            }
        for handle in program.handles.values():
            handles[handle.name] = {"c_type": names.name(ConcreteType("handle", handle.name))}
        for extern in program.reachable_externs():
            externs[extern.symbol] = {
                "abi": extern.abi,
                "effect": extern.effect.spelling,
                "schema": extern.schema,
                "parameters": [{"name": name, "type": _type_record(value)} for name, value in extern.parameters],
                "result": None if extern.result is None else _type_record(extern.result),
            }
        for exported, info in program.exported():
            entries.append(
                {
                    "name": exported,
                    "symbol": prefix + exported,
                    "effect": info.effect.spelling,
                    "declared_effect": info.declared_effect.spelling,
                    "parameters": [{"name": name, "type": _type_record(value)} for name, value in info.parameters],
                    "result": None if info.result is None else _type_record(info.result),
                }
            )
        for host_class in program.classes.values():
            classes[host_class.name] = {
                "doc": host_class.doc,
                "meta": {key: json.loads(value) for key, value in host_class.metadata},
                "init": prefix + f"{host_class.name}__init",
                "fields": [
                    {
                        "name": field.name,
                        "type": _type_record(field.type),
                        "doc": field.doc,
                        "meta": {key: json.loads(value) for key, value in field.metadata},
                    }
                    for field in host_class.fields
                ],
                "methods": [
                    {"name": method.name, "symbol": prefix + f"{host_class.name}_{method.name}"}
                    for method in host_class.methods
                    if method.exported
                ],
                "layout_hash": f"{layout_hash(ConcreteType('struct', host_class.name), programs):016x}",
            }
        for path, digest in program.dependencies:
            sources[str(path)] = digest
    return {
        "version": REFLECTION_VERSION,
        "triple": triple,
        "prefix": prefix,
        "table": prefix + "module",
        "modules": list(dict.fromkeys(program.module_name for program in programs)),
        "entries": entries,
        "externs": externs,
        "structs": structs,
        "classes": classes,
        "handles": handles,
        "sources": sources,
    }


def read_reflection(path: str | Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
