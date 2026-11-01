"""Level 1 bindings: the C API, C++ glue and nanobind module of a `vd.library` module (specs/host/bindgen.md).

The Host module is the schema. Its externs become C functions `<name>_v<version>_<function>` whose thunks look up
handles, convert arguments and call the implementation in the library's `cpp_namespace`; the nanobind module wraps
the same C functions for Python and publishes their addresses, so Python-mode Host code and linked AOT objects call
one function. Outputs are emitted already in the project's style: no formatter runs, so the drift check does not
depend on tool versions. The C++ side targets C++17.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..frontend.model import ConcreteType
from ..hostlang.aot import CNames, boundary_types, camel, type_declarations
from ..hostlang.frontend import load_generic_externs, load_host_library
from ..hostlang.layout import tensor_count
from ..hostlang.library import library_identity
from ..hostlang.model import ExternAbi, HostEffect, HostExtern, HostProgram
from ..hostlang.reflect import STRUCT, describe
from ..hostlang.values import element_of, key_of


class BindgenError(ValueError):
    pass


_PYTHON_SCALARS = {"bool": "bool", "f32": "float", "f64": "float"}
_OWNED_RESULTS = {"str", "list", "dict"}
_ITEM_KINDS = {"scalar", "enum", "handle", "struct", "tensor"}


@dataclass(frozen=True)
class _Handle:
    """A handle's C++ mapping. With `cpp` alone the objects live in a handle table the thunks own; with `convert`
    too, the bits are a C++ value the library converts both ways, and nothing owns it. `depends_on` names the handle
    whose destruction first destroys every object of this one created with it."""

    name: str
    c_name: str
    cpp: str | None
    header: str | None
    convert: str | None = None
    depends_on: str | None = None

    @property
    def table(self) -> str:
        return f"{_snake(self.name)}_table"

    @property
    def tabled(self) -> bool:
        return self.cpp is not None and self.convert is None


@dataclass(frozen=True)
class _Struct:
    """A Struct's C++ mapping. With `cpp` alone the C++ type has the C layout and is cast; with `convert` too, the
    library implements `convert` both ways between the two."""

    name: str
    c_name: str
    cpp: str | None
    header: str | None
    convert: str | None


@dataclass(frozen=True)
class Library:
    """A `vd.library` module as the generator reads it."""

    path: Path
    name: str
    version: int
    cpp_namespace: str
    identity: str
    program: HostProgram
    names: CNames
    handles: dict[str, _Handle]
    structs: dict[str, _Struct]
    # Every extern in source order; a generic one is typed as written (`load_generic_externs`).
    externs: tuple[HostExtern, ...]
    # Extern docstrings by name, from the source: the frontend keeps extern bodies as `...`.
    docs: dict[str, str]
    # The unqualified C++ function in `cpp_namespace` each hooked thunk asks before calling the implementation.
    effect_check: str | None = None

    def hooked(self, extern: HostExtern) -> bool:
        """Whether the extern's thunk calls the effect check: a direct extern cannot fail, and a pure one has no
        effect to allow."""

        return self.effect_check is not None and extern.abi == ExternAbi.STATUS and extern.effect != HostEffect.PURE

    @property
    def prefix(self) -> str:
        return f"{self.name}_v{self.version}_"

    @property
    def macro(self) -> str:
        return self.name.upper()

    def handle(self, value_type: ConcreteType) -> _Handle | None:
        """The handle `value_type` is when it has a handle table."""

        if value_type.kind != "handle":
            return None
        handle = self.handles[value_type.name]
        return handle if handle.tabled else None

    def value_handle(self, value_type: ConcreteType) -> _Handle | None:
        """The handle `value_type` is when its bits convert to a C++ value."""

        if value_type.kind != "handle":
            return None
        handle = self.handles[value_type.name]
        return handle if handle.convert is not None else None

    def dependents(self, parent: _Handle) -> list[_Handle]:
        return [handle for handle in self.handles.values() if handle.depends_on == parent.name]

    def destroyer(self, handle: _Handle) -> HostExtern | None:
        """The extern destroying `handle`'s objects, if one does."""

        return next(
            (
                extern
                for extern in self.externs
                if (destroys := _destroys(extern)) is not None
                and dict(extern.parameters)[destroys] == ConcreteType("handle", handle.name)
            ),
            None,
        )

    def struct(self, value_type: ConcreteType) -> _Struct | None:
        """The Struct `value_type` is when it has a C++ type of its own."""

        if value_type.kind != "struct" or value_type.name not in self.structs:
            return None
        struct = self.structs[value_type.name]
        return struct if struct.cpp is not None else None


def _snake(name: str) -> str:
    return "".join(f"_{char.lower()}" if char.isupper() and index else char.lower() for index, char in enumerate(name))


def _metadata(entries: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return {key: json.loads(value) for key, value in entries}


def _declared_library(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_vernon_bindgen_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise BindgenError(f"{path} is not a Python module")
    module = importlib.util.module_from_spec(spec)
    # Declarations resolve their annotations through the module, by name.
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    declared = getattr(module, "__vernon_library__", None)
    if declared is None:
        raise BindgenError(f"{path.name} declares no vd.library")
    return declared


def read_library(path: str | Path) -> Library:
    path = Path(path).resolve()
    declared = _declared_library(path)
    cpp_namespace = declared.meta.get("cpp_namespace")
    if not isinstance(cpp_namespace, str) or not cpp_namespace:
        raise BindgenError(f"vd.library '{declared.name}' needs meta=vd.meta(cpp_namespace=...) for C++ bindings")
    effect_check = declared.meta.get("effect_check")
    if effect_check is not None and (not isinstance(effect_check, str) or not effect_check.isidentifier()):
        raise BindgenError(f"vd.library '{declared.name}': effect_check names an unqualified C++ function")
    program, _ = load_host_library(path)
    names = CNames([program], camel(declared.name))
    handles = {}
    for name, handle in program.handles.items():
        meta = _metadata(handle.metadata)
        cpp, header, convert, depends_on = (meta.get(key) for key in ("cpp", "header", "convert", "depends_on"))
        if cpp is not None and not isinstance(cpp, str):
            raise BindgenError(f"handle {name}: cpp names a C++ type")
        if convert is not None and (cpp is None or not isinstance(convert, str) or not convert.isidentifier()):
            raise BindgenError(f"handle {name}: convert names an unqualified C++ function, and needs cpp=")
        handles[name] = _Handle(name, names.name(ConcreteType("handle", name)), cpp, header, convert, depends_on)
    for name, handle in handles.items():
        if handle.depends_on is None:
            continue
        parent = handles.get(handle.depends_on)
        if not handle.tabled or parent is None or not parent.tabled:
            raise BindgenError(f"handle {name}: depends_on names a handle, and both need cpp= without convert=")
        seen = {name}
        while parent is not None:
            if parent.name in seen:
                raise BindgenError(f"handle {name}: depends_on forms a cycle through {parent.name}")
            seen.add(parent.name)
            parent = handles.get(parent.depends_on) if parent.depends_on is not None else None
    structs = {}
    for name, host_struct in program.structs.items():
        meta = _metadata(host_struct.metadata)
        cpp, header, convert = meta.get("cpp"), meta.get("header"), meta.get("convert")
        if cpp is not None and not isinstance(cpp, str):
            raise BindgenError(f"struct {name}: cpp names a C++ type")
        if convert is not None and (cpp is None or not isinstance(convert, str) or not convert.isidentifier()):
            raise BindgenError(f"struct {name}: convert names an unqualified C++ function, and needs cpp=")
        structs[name] = _Struct(name, names.name(ConcreteType("struct", name)), cpp, header, convert)
    externs = tuple(
        sorted((*program.externs.values(), *load_generic_externs(path)), key=lambda extern: extern.source.lineno)
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    docs = {
        node.name: doc
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and (doc := ast.get_docstring(node)) is not None
    }
    library = Library(
        path,
        declared.name,
        declared.version,
        cpp_namespace,
        library_identity(path, declared.name, declared.version),
        program,
        names,
        handles,
        structs,
        externs,
        docs,
        effect_check,
    )
    for extern in externs:
        _check(library, extern)
    for handle in handles.values():
        if handle.depends_on is not None:
            _check_dependent(library, handle)
    return library


def _parent_parameter(library: Library, extern: HostExtern, handle: _Handle) -> str:
    """The parameter of the extern creating a `handle` object that holds the parent it depends on."""

    parent = ConcreteType("handle", handle.depends_on or "")
    names = [name for name, value_type in extern.parameters if value_type == parent]
    if len(names) != 1:
        raise BindgenError(
            f"extern {extern.name} creates a {handle.name}, which depends on {handle.depends_on}, so it takes one "
            f"{handle.depends_on}"
        )
    return names[0]


def _check_dependent(library: Library, handle: _Handle) -> None:
    for extern in library.externs:
        if extern.result == ConcreteType("handle", handle.name):
            _parent_parameter(library, extern, handle)
    destroyer = library.destroyer(handle)
    if destroyer is not None and (len(destroyer.parameters) != 1 or destroyer.result is not None):
        raise BindgenError(
            f"extern {destroyer.name}: destroying {handle.depends_on} destroys its {handle.name} objects through it, "
            f"so it takes only the {handle.name} and returns nothing"
        )


# Generic externs: the type variable stands as a memberless Struct of its name, and the C function receives the
# bound Struct's descriptor at `type_argument[0]`, with `T` values as `descriptor->size` bytes behind a pointer.


def _is_variable(extern: HostExtern, value_type: ConcreteType) -> bool:
    return (
        extern.type_argument is not None and value_type.kind == "struct" and value_type.name == extern.type_argument[1]
    )


def _type_parameter(extern: HostExtern) -> tuple[int, str] | None:
    """The position and name of a generic extern's descriptor parameter: its `type[T]` parameter, or `type` just
    before the value that binds `T`."""

    if extern.type_argument is None:
        return None
    index = extern.type_argument[0]
    return index, extern.source.args.args[index].arg if extern.type_passed else "type"


def _items(value_type: ConcreteType) -> list[ConcreteType]:
    """What a list or dict holds: its element, or its key and value."""

    return [element_of(value_type)] if value_type.kind == "list" else [key_of(value_type), element_of(value_type)]


def _check(library: Library, extern: HostExtern) -> None:
    where = f"extern {extern.name}"
    status = extern.abi == ExternAbi.STATUS
    for name, parameter in extern.parameters:
        if _is_variable(extern, parameter):
            continue
        if parameter.kind in {"list", "dict"}:
            _check_items(library, f"{where}: parameter {name}", parameter)
        elif parameter.kind not in {*_ITEM_KINDS, "str"}:
            raise BindgenError(f"{where}: parameter {name} is a {parameter.spelling}, which bindings do not pass yet")
        if library.handle(parameter) is not None and not status:
            raise BindgenError(f"{where} takes a {parameter.name} handle, so it is abi='status': a stale handle fails")
    result = extern.result
    if result is not None and not _is_variable(extern, result):
        allowed = {*_ITEM_KINDS, *_OWNED_RESULTS} if status else {"scalar", "enum", "handle"}
        if result.kind not in allowed:
            raise BindgenError(f"{where}: a {result.spelling} result is not generated for abi='{extern.abi}'")
        if result.kind in {"list", "dict"}:
            _check_items(library, f"{where}: its result", result)
        if library.handle(result) is not None and not status:
            raise BindgenError(f"{where} creates a {result.name} handle, so it is abi='status'")
    destroys = _metadata(extern.metadata).get("destroys")
    if destroys is not None:
        parameter = dict(extern.parameters).get(destroys)
        if parameter is None or library.handle(parameter) is None:
            raise BindgenError(f"{where}: destroys='{destroys}' names no parameter holding a handle with cpp=")


def _check_items(library: Library, where: str, value_type: ConcreteType) -> None:
    for item in _items(value_type):
        if item.kind not in _ITEM_KINDS:
            raise BindgenError(f"{where} holds {item.spelling}, which bindings do not pass yet")
        if library.handle(item) is not None:
            raise BindgenError(f"{where} holds {item.name} handles, each of which would need a table lookup")


def _destroys(extern: HostExtern) -> str | None:
    return _metadata(extern.metadata).get("destroys")


def _converter(library: Library, value_type: ConcreteType) -> str | None:
    """The library function converting a value of `value_type` between C and C++, when it has one."""

    struct = library.struct(value_type)
    if struct is not None and struct.convert is not None:
        return struct.convert
    handle = library.value_handle(value_type)
    return handle.convert if handle is not None else None


# The C API.


def _c_parameter(library: Library, extern: HostExtern, name: str, value_type: ConcreteType) -> str:
    if _is_variable(extern, value_type):
        return f"const void *{name}"
    if value_type.kind in {"list", "dict"}:
        return f"VhSpan {name}"
    c_name = library.names.name(value_type)
    if value_type.kind in {"scalar", "enum", "handle"}:
        return f"{c_name} {name}"
    if value_type.kind == "str":
        return f"VhStrArg {name}"
    return f"const {c_name} *{name}"


def _c_prototype(library: Library, extern: HostExtern) -> str:
    parameters = [_c_parameter(library, extern, name, value) for name, value in extern.parameters]
    typed = _type_parameter(extern)
    if typed is not None:
        parameters.insert(typed[0], f"const VhType *{typed[1]}")
    result = extern.result
    if extern.abi == ExternAbi.STATUS:
        if result is not None and _is_variable(extern, result):
            parameters.append("void *out")
        elif result is not None and result.kind in _OWNED_RESULTS:
            parameters += ["VhList *out", "const VhAllocator *allocator"]
        elif result is not None:
            parameters.append(f"{library.names.name(result)} *out")
        returned = "int32_t"
    else:
        returned = "void" if result is None else library.names.name(result)
    return f"{returned} {extern.symbol}({', '.join(parameters) or 'void'})"


def _api_header(library: Library) -> str:
    guard = f"{library.macro}_API_H"
    roots = [value for extern in library.externs for _, value in extern.parameters if not _is_variable(extern, value)]
    roots += [
        extern.result
        for extern in library.externs
        if extern.result is not None and not _is_variable(extern, extern.result)
    ]
    boundary, forward = boundary_types([library.program], roots)
    lines = [
        f"/* Generated by vernon_dsl.bindgen from {library.path.name}; do not edit. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        '#include "vernon_host/reflect.h"',
        "",
        "#ifdef __cplusplus",
        "#define VERNON_HOST_STATIC_ASSERT(condition, message) static_assert(condition, message)",
        'extern "C" {',
        "#else",
        "#define VERNON_HOST_STATIC_ASSERT(condition, message) _Static_assert(condition, message)",
        "#endif",
        "",
        "/* Marks the functions below, for example as dllexport when they are built into a shared library. */",
        f"#ifndef {library.macro}_API",
        f"#define {library.macro}_API",
        "#endif",
        "",
        "/* The schema identity of the Host module these declarations were generated from. */",
        f'#define {library.prefix.upper()}SCHEMA "{library.identity}"',
        "",
    ]
    lines += type_declarations(boundary, forward, library.names, external_declared=False)
    lines.append("")
    for extern in library.externs:
        doc = library.docs.get(extern.name)
        if doc:
            text = doc.splitlines()
            lines.append(f"/* {text[0]}" if len(text) > 1 else f"/* {doc} */")
            if len(text) > 1:
                lines += [f" * {line}".rstrip() for line in text[1:]]
                lines.append(" */")
        lines.append(f"/* effects: {extern.effect.spelling} */")
        lines.append(f"{library.macro}_API {_c_prototype(library, extern)};")
    lines += [
        "",
        "/* Copies the message of the calling thread's last failed call into `buffer`, cut to `capacity` bytes with a",
        " * NUL, and returns the message's full length. */",
        f"{library.macro}_API uint64_t {library.prefix}last_error(char *buffer, uint64_t capacity);",
        f"/* {library.prefix.upper()}SCHEMA, as the linked library was generated. */",
        f"{library.macro}_API const char *{library.prefix}schema(void);",
        "/* The descriptors of the library's Structs, handles and enums, `*count` of them, each Struct after the",
        " * types its fields hold. A Struct's layout hash is the one Host code passes to a generic extern. */",
        f"{library.macro}_API const VhType *const *{library.prefix}types(uint32_t *count);",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


# The C++ side.


def _c_item(library: Library, value_type: ConcreteType) -> str:
    """The C++ spelling of a value with its C layout, as spans and owned results hold it."""

    if value_type.kind == "tensor":
        return f"std::array<{library.names.name(value_type)}, {tensor_count(value_type)}>"
    return library.names.name(value_type)


def _cpp_item(library: Library, value_type: ConcreteType) -> str:
    """The C++ type an implementation sees an item of a span or owned result as: a cast Struct's C++ type has its
    layout, so the items are the same bytes, and a converted Struct or handle is converted item by item."""

    struct = library.struct(value_type)
    if struct is not None:
        return struct.cpp  # type: ignore[return-value]
    handle = library.value_handle(value_type)
    return handle.cpp if handle is not None else _c_item(library, value_type)  # type: ignore[return-value]


def _converts(library: Library, value_type: ConcreteType) -> bool:
    """Whether the items of a list or dict convert one by one between C and C++."""

    return any(_converter(library, item) is not None for item in _items(value_type))


def _item_from_c(library: Library, value_type: ConcreteType, expression: str) -> str:
    converter = _converter(library, value_type)
    if converter is not None:
        return f"{library.cpp_namespace}::{converter}({expression})"
    struct = library.struct(value_type)
    if struct is not None:
        return f"reinterpret_cast<const {struct.cpp} &>({expression})"
    return expression


def _item_to_c(library: Library, value_type: ConcreteType, expression: str) -> str:
    converter = _converter(library, value_type)
    if converter is not None:
        return f"{library.cpp_namespace}::{converter}({expression})"
    struct = library.struct(value_type)
    if struct is not None:
        return f"reinterpret_cast<const {library.names.name(value_type)} &>({expression})"
    return expression


def _convert_items(library: Library, value_type: ConcreteType, item: str, convert: Any) -> str:
    """One item of a list or dict `item`, converted with `convert(library, type, expression)`."""

    if value_type.kind == "list":
        return convert(library, element_of(value_type), item)
    key, value = _items(value_type)
    return f"{{{convert(library, key, f'{item}.key')}, {convert(library, value, f'{item}.value')}}}"


def _items_type(value_type: ConcreteType, item: Any) -> str:
    """The C++ type of one item of a list or dict, spelled with `item`."""

    if value_type.kind == "list":
        return item(element_of(value_type))
    return f"vh::Entry<{item(key_of(value_type))}, {item(element_of(value_type))}>"


def _cpp_value(library: Library, value_type: ConcreteType) -> str:
    """The C++ type an implementation receives or returns a value of `value_type` as."""

    handle = library.handle(value_type)
    if handle is not None:
        return f"std::unique_ptr<{handle.cpp}>"
    if value_type.kind == "str":
        return "std::string"
    if value_type.kind in {"list", "dict"}:
        return f"std::vector<{_items_type(value_type, lambda item: _cpp_item(library, item))}>"
    return _cpp_item(library, value_type)


def _cpp_parameter(library: Library, extern: HostExtern, name: str, value_type: ConcreteType) -> str:
    if _is_variable(extern, value_type):
        return f"const void *{name}"
    handle = library.handle(value_type)
    if handle is not None:
        if _destroys(extern) == name:
            return f"std::unique_ptr<{handle.cpp}> {name}"
        constant = "const " if extern.effect in {HostEffect.PURE, HostEffect.READ} else ""
        return f"{constant}{handle.cpp} &{name}"
    value_handle = library.value_handle(value_type)
    if value_handle is not None:
        return f"{value_handle.cpp} {name}"
    if value_type.kind == "str":
        return f"std::string_view {name}"
    if value_type.kind in {"list", "dict"}:
        return f"vh::Span<{_items_type(value_type, lambda item: _cpp_item(library, item))}> {name}"
    if value_type.kind == "struct":
        return f"const {_cpp_item(library, value_type)} &{name}"
    c_name = library.names.name(value_type)
    if value_type.kind == "tensor":
        return f"const {c_name} (&{name})[{tensor_count(value_type)}]"
    return f"{c_name} {name}"


def _cpp_result(library: Library, extern: HostExtern) -> str:
    if extern.abi != ExternAbi.STATUS:
        if extern.result is None:
            return "void"
        value_handle = library.value_handle(extern.result)
        return value_handle.cpp if value_handle is not None else library.names.name(extern.result)  # type: ignore[return-value]
    if extern.result is None or _is_variable(extern, extern.result):
        return "vh::Status"
    return f"vh::Result<{_cpp_value(library, extern.result)}>"


def _cpp_declaration(library: Library, extern: HostExtern) -> str:
    parameters = [_cpp_parameter(library, extern, name, value) for name, value in extern.parameters]
    typed = _type_parameter(extern)
    if typed is not None:
        parameters.insert(typed[0], f"const VhType &{typed[1]}")
    if extern.result is not None and _is_variable(extern, extern.result):
        parameters.append("void *out")
    return f"{_cpp_result(library, extern)} {extern.name}({', '.join(parameters)})"


def _impl_header(library: Library) -> str:
    headers = sorted(
        {handle.header for handle in library.handles.values() if handle.header}
        | {struct.header for struct in library.structs.values() if struct.header}
    )
    lines = [
        f"// Generated by vernon_dsl.bindgen from {library.path.name}; do not edit.",
        f"// The functions the {library.name} library implements for its C API, {library.name}_api.h. A status",
        "// function reports failure with a positive status and a message; a direct one cannot fail and must not",
        "// throw.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <memory>",
        "#include <string>",
        "#include <string_view>",
        "#include <vector>",
        "",
        f'#include "{library.name}_api.h"',
        '#include "vernon_host/bindgen.hpp"',
    ]
    lines += [f'#include "{header}"' for header in headers]
    lines += ["", f"namespace {library.cpp_namespace} {{", ""]
    converted = [struct for struct in library.structs.values() if struct.convert is not None]
    for struct in converted:
        lines += [
            f"// Converts {struct.name} between its C layout and {struct.cpp}.",
            f"{struct.cpp} {struct.convert}(const {struct.c_name} &value);",
            f"{struct.c_name} {struct.convert}(const {struct.cpp} &value);",
        ]
    converted_handles = [handle for handle in library.handles.values() if handle.convert is not None]
    for handle in converted_handles:
        lines += [
            f"// Converts {handle.name} handle bits to and from {handle.cpp}.",
            f"{handle.cpp} {handle.convert}({handle.c_name} value);",
            f"{handle.c_name} {handle.convert}(const {handle.cpp} &value);",
        ]
    if converted or converted_handles:
        lines.append("")
    subjects = _effect_subjects(library)
    if subjects:
        lines += [
            "// The effect check: every status function but a pure one asks it once its handles resolve and before",
            "// calling its implementation, which runs only when it succeeds. `call` describes the function; a",
            "// function taking a handle with an object passes the first such object.",
        ]
        for subject in subjects:
            parameter = "" if subject is None else f", const {subject.cpp} &object"
            lines.append(f"vh::Status {library.effect_check}(const VhCall &call{parameter});")
        lines.append("")
    for extern in library.externs:
        doc = library.docs.get(extern.name)
        if doc:
            lines += [f"// {line}".rstrip() for line in doc.splitlines()]
        destroys = _destroys(extern)
        if destroys is not None:
            lines.append(f"// Receives the object behind `{destroys}`, whose handle is already gone.")
        typed = _type_parameter(extern)
        if typed is not None:
            lines.append(
                f"// Generic over Structs: `{typed[1]}` describes the bound Struct, and its values are "
                f"`{typed[1]}.size` bytes."
            )
        lines.append(f"{_cpp_declaration(library, extern)};")
    tabled = [handle for handle in library.handles.values() if handle.tabled]
    if tabled:
        lines += [
            "",
            "// The objects behind the handles the thunks own, for the library's own code, such as other bindings.",
            "namespace handles {",
            "",
            "// Destroys every object of the handle, each as destroying its handle would: for a teardown hook.",
            "template <typename Handle> vh::Status destroy_all();",
            "",
        ]
        for handle in tabled:
            lines += [
                f"// The object behind a live {handle.name}; null for a stale handle.",
                f"{handle.cpp} *find({handle.c_name} handle);",
                f"template <> vh::Status destroy_all<{handle.c_name}>();",
            ]
        lines += ["", "} // namespace handles"]
    lines += ["", f"}} // namespace {library.cpp_namespace}", ""]
    return "\n".join(lines)


def _thunk(library: Library, extern: HostExtern) -> list[str]:
    namespace = library.cpp_namespace
    destroys = _destroys(extern)
    body: list[str] = []
    arguments: list[str] = []
    # A destroyed handle's dependents go first, after its handle is gone, so no new one is created with it.
    cascade: str | None = None
    hooked = library.hooked(extern)
    # The object the effect check sees, and what runs after it: a destroyer takes its object only once the check
    # allows the call, so a refused destroy leaves the handle live.
    subject: str | None = None
    after_check: list[str] = []
    for name, value_type in extern.parameters:
        handle = library.handle(value_type)
        struct = library.struct(value_type)
        converter = _converter(library, value_type)
        if _is_variable(extern, value_type):
            arguments.append(name)
        elif handle is not None:
            missing = f'"{extern.symbol}: {name} is not a live {handle.name}"'
            found = f"{name}_object"
            if destroys == name:
                take = f"std::unique_ptr<{handle.cpp}> {name}_object = {handle.table}().take({name}.bits);"
                if hooked:
                    found = f"{name}_found"
                    body.append(f"{handle.cpp} *{found} = {handle.table}().find({name}.bits);")
                    after_check.append(take)
                else:
                    body.append(take)
                arguments.append(f"std::move({name}_object)")
                if library.dependents(handle):
                    cascade = f"destroy_{_snake(handle.name)}_dependents({name}.bits)"
            else:
                body.append(f"{handle.cpp} *{name}_object = {handle.table}().find({name}.bits);")
                arguments.append(f"*{name}_object")
            body += [
                f"if ({found} == nullptr) {{",
                f"    return fail(vh::kInvalidHandle, {missing});",
                "}",
            ]
            if subject is None:
                subject = f"*{found}"
        elif value_type.kind == "str":
            body.append(f"std::string_view {name}_view({name}.data, static_cast<size_t>({name}.size));")
            arguments.append(f"{name}_view")
        elif value_type.kind in {"list", "dict"} and _converts(library, value_type):
            c_item = _items_type(value_type, lambda item: _c_item(library, item))
            cpp_item = _items_type(value_type, lambda item: _cpp_item(library, item))
            body += [
                f"std::vector<{cpp_item}> {name}_values;",
                f"{name}_values.reserve(static_cast<size_t>({name}.size));",
                f"for (const {c_item} &item : vh::Span<{c_item}>({name})) {{",
                f"    {name}_values.push_back({_convert_items(library, value_type, 'item', _item_from_c)});",
                "}",
                f"vh::Span<{cpp_item}> {name}_items({name}_values.data(), {name}_values.size());",
            ]
            arguments.append(f"{name}_items")
        elif value_type.kind in {"list", "dict"}:
            item = _items_type(value_type, lambda item: _cpp_item(library, item))
            body.append(f"vh::Span<{item}> {name}_items({name});")
            arguments.append(f"{name}_items")
        elif converter is not None and value_type.kind == "handle":
            arguments.append(f"{namespace}::{converter}({name})")
        elif struct is not None and struct.convert is not None:
            body.append(f"{struct.cpp} {name}_value = {namespace}::{struct.convert}(*{name});")
            arguments.append(f"{name}_value")
        elif struct is not None:
            arguments.append(f"*reinterpret_cast<const {struct.cpp} *>({name})")
        elif value_type.kind == "struct":
            arguments.append(f"*{name}")
        elif value_type.kind == "tensor":
            count = tensor_count(value_type)
            element = library.names.name(value_type)
            arguments.append(f"*reinterpret_cast<const {element}(*)[{count}]>({name})")
        else:
            arguments.append(name)
    typed = _type_parameter(extern)
    if typed is not None:
        arguments.insert(typed[0], f"*{typed[1]}")
        body = [
            f"if ({typed[1]} == nullptr) {{",
            f'    return fail(vh::kFailed, "{extern.symbol}: {typed[1]} is a null VhType");',
            "}",
            *body,
        ]
    if hooked:
        record = f"call_{extern.name}"
        if typed is not None:
            body += [f"VhCall call = {record};", f"call.type = {typed[1]};"]
            record = "call"
        check = f"{namespace}::{library.effect_check}({record}{'' if subject is None else f', {subject}'})"
        body += [
            f"if (vh::Status allowed = {check}; !allowed.ok()) {{",
            "    return fail(std::move(allowed.error()));",
            "}",
            *after_check,
        ]
    result = extern.result
    if result is not None and _is_variable(extern, result):
        arguments.append("out")
    call = f"{namespace}::{extern.name}({', '.join(arguments)})"
    lines = [f'extern "C" {_c_prototype(library, extern)} {{']
    if extern.abi != ExternAbi.STATUS:
        lines += [f"    {line}" for line in body]
        result_converter = None if result is None else _converter(library, result)
        if result is None:
            lines.append(f"    {call};")
        elif result_converter is not None:
            lines.append(f"    return {namespace}::{result_converter}({call});")
        else:
            lines.append(f"    return {call};")
        lines.append("}")
        return lines
    if cascade is not None:
        body += [f"vh::Status dependents = {cascade};", f"vh::Status destroyed = {call};"]
        body.append("return finish(dependents.ok() ? std::move(destroyed) : std::move(dependents));")
    elif result is None or _is_variable(extern, result):
        body.append(f"return finish({call});")
    else:
        body.append(f"auto returned = {call};")
        body += ["if (!returned.ok()) {", "    return fail(std::move(returned.error()));", "}"]
        handle = library.handle(result)
        struct = library.struct(result)
        converter = _converter(library, result)
        if handle is not None:
            parent = "" if handle.depends_on is None else f", {_parent_parameter(library, extern, handle)}.bits"
            body += [
                "if (returned.value() == nullptr) {",
                f'    return fail(vh::kFailed, "{extern.symbol}: the implementation returned no {handle.name}");',
                "}",
                f"out->bits = {handle.table}().insert(std::move(returned.value()){parent});",
            ]
        elif result.kind in {"list", "dict"} and _converts(library, result):
            c_item = _items_type(result, lambda item: _c_item(library, item))
            body += [
                f"std::vector<{c_item}> converted;",
                "converted.reserve(returned.value().size());",
                "for (const auto &item : returned.value()) {",
                f"    converted.push_back({_convert_items(library, result, 'item', _item_to_c)});",
                "}",
                "if (vh::assign_result(out, allocator, converted) != 0) {",
                f'    last_error = "{extern.symbol}: the allocator could not hold the result";',
                "    return VH_STATUS_OUT_OF_MEMORY;",
                "}",
            ]
        elif converter is not None and result.kind == "handle":
            body.append(f"*out = {namespace}::{converter}(returned.value());")
        elif result.kind in _OWNED_RESULTS:
            body += [
                "if (vh::assign_result(out, allocator, returned.value()) != 0) {",
                f'    last_error = "{extern.symbol}: the allocator could not hold the result";',
                "    return VH_STATUS_OUT_OF_MEMORY;",
                "}",
            ]
        elif struct is not None and struct.convert is not None:
            body.append(f"*out = {namespace}::{struct.convert}(returned.value());")
        elif result.kind == "tensor":
            body.append("std::memcpy(out, returned.value().data(), sizeof(returned.value()));")
        elif struct is not None:
            body.append("std::memcpy(out, &returned.value(), sizeof(*out));")
        else:
            body.append("*out = returned.value();")
        body.append("return 0;")
    lines.append(f'    return guard("{extern.symbol}", [&]() -> int32_t {{')
    lines += [f"        {line}" for line in body]
    lines += ["    });", "}"]
    return lines


def _c_string(text: str | None) -> str:
    """A C++ string literal of `text`'s UTF-8 bytes, or nullptr."""

    if text is None:
        return "nullptr"
    escaped = []
    for byte in text.encode("utf-8"):
        character = chr(byte)
        if character in '"\\':
            escaped.append("\\" + character)
        elif character == "\n":
            escaped.append("\\n")
        elif 0x20 <= byte < 0x7F and character != "?":
            escaped.append(character)
        else:
            # Three octal digits end the escape, whatever follows; `?` is escaped so no trigraph forms.
            escaped.append(f"\\{byte:03o}")
    return '"' + "".join(escaped) + '"'


def _descriptors(library: Library) -> list[str]:
    """The VhType descriptors of the library's types, as `describe` records them for Host objects, so a Struct's
    layout hash matches the one Host code passes to a generic extern. They describe layout only: no fini, init or
    methods."""

    table = describe([library.program], library.prefix)
    names = {record.value_type: f"type_{index}" for index, record in enumerate(table.types)}
    lines = ["", "// The library's type descriptors, each after the types it refers to."]

    def metadata(name: str, entries: tuple[tuple[str, str], ...]) -> str:
        if not entries:
            return "nullptr, 0"
        items = ", ".join(f"{{{_c_string(key)}, {_c_string(value)}}}" for key, value in entries)
        lines.append(f"const VhMeta {name}[] = {{{items}}};")
        return f"{name}, {len(entries)}"

    for record in table.types:
        name = names[record.value_type]
        shape = "nullptr"
        if record.shape:
            lines.append(f"const uint32_t {name}_shape[] = {{{', '.join(str(extent) for extent in record.shape)}}};")
            shape = f"{name}_shape"
        fields = "nullptr"
        if record.fields:
            entries = []
            for position, item in enumerate(record.fields):
                meta = metadata(f"{name}_field_{position}_meta", item.metadata)
                entries.append(
                    f"    {{sizeof(VhField), {_c_string(item.name)}, &{names[item.type]}, {item.offset}, "
                    f"{_c_string(item.doc)}, {meta}}},"
                )
            lines += [f"const VhField {name}_fields[] = {{", *entries, "};"]
            fields = f"{name}_fields"
        enumerators = "nullptr"
        if record.enumerators:
            items = ", ".join(f"{{{_c_string(member)}, {value}}}" for member, value in record.enumerators)
            lines.append(f"const VhEnumerator {name}_enumerators[] = {{{items}}};")
            enumerators = f"{name}_enumerators"
        meta = metadata(f"{name}_meta", record.metadata)
        element = "nullptr" if record.element is None else f"&{names[record.element]}"
        key = "nullptr" if record.key is None else f"&{names[record.key]}"
        lines += [
            f"const VhType {name} = {{sizeof(VhType), {record.kind}, {record.alignment}, {record.size}, "
            f"{_c_string(record.name)}, UINT64_C(0x{record.layout_hash:016x}),",
            f"                       {element}, {shape}, {len(record.shape)}, {len(record.fields)}, {fields}, "
            f"nullptr, 0, nullptr, {_c_string(record.doc)}, {meta}, nullptr, {key}, {enumerators}, "
            f"{len(record.enumerators)}}};",
        ]
    listed = ", ".join(f"&{names[value_type]}" for value_type in table.listed)
    lines.append(f"const VhType *const listed_types[] = {{{listed or 'nullptr'}}};")
    lines.append(f"constexpr uint32_t listed_type_count = {len(table.listed)};")
    return lines


def _call_records(library: Library) -> list[str]:
    """The VhCall each hooked thunk passes its effect check: the extern's effect, symbol and `meta=` entries. A
    generic extern's thunk passes a copy whose `type` is the call's bound Struct."""

    hooked = [extern for extern in library.externs if library.hooked(extern)]
    if not hooked:
        return []
    lines = ["", "// What each thunk's effect check is asked about."]
    for extern in hooked:
        meta = "nullptr, 0"
        if extern.metadata:
            items = ", ".join(f"{{{_c_string(key)}, {_c_string(value)}}}" for key, value in extern.metadata)
            lines.append(f"const VhMeta call_{extern.name}_meta[] = {{{items}}};")
            meta = f"call_{extern.name}_meta, {len(extern.metadata)}"
        lines.append(
            f"const VhCall call_{extern.name} = {{sizeof(VhCall), {int(extern.effect)}, "
            f'"{extern.symbol}", nullptr, {meta}}};'
        )
    return lines


def _effect_subjects(library: Library) -> list[_Handle | None]:
    """The objects the effect check is asked about, in first use: a hooked thunk passes the object of its first
    table handle parameter, or none."""

    subjects: list[_Handle | None] = []
    for extern in library.externs:
        if not library.hooked(extern):
            continue
        handles = [handle for _, value in extern.parameters if (handle := library.handle(value)) is not None]
        subject = handles[0] if handles else None
        if subject not in subjects:
            subjects.append(subject)
    return subjects


def _tabled_children_first(library: Library) -> list[_Handle]:
    ordered: list[_Handle] = []

    def visit(handle: _Handle) -> None:
        if handle in ordered:
            return
        for dependent in library.dependents(handle):
            visit(dependent)
        ordered.append(handle)

    for handle in library.handles.values():
        if handle.tabled:
            visit(handle)
    return ordered


def _cascades(library: Library) -> list[str]:
    """For each table handle, `release_<handle>(bits, object)`, which destroys an object already taken from its
    table: its dependents first, then the object through its destroying extern's implementation, if one destroys it.
    `destroy_<parent>_dependents(bits)` takes and releases the objects created with a parent. A failure does not
    stop the others; the first one is the result."""

    lines: list[str] = []
    for handle in _tabled_children_first(library):
        snake = _snake(handle.name)
        dependents = library.dependents(handle)
        if dependents:
            lines += ["", f"vh::Status destroy_{snake}_dependents(uint64_t bits) {{", "    vh::Status status;"]
            for dependent in dependents:
                lines += [
                    f"    for (auto &child : {dependent.table}().take_children(bits)) {{",
                    f"        vh::Status released = release_{_snake(dependent.name)}(child.first, "
                    "std::move(child.second));",
                    "        if (status.ok() && !released.ok()) {",
                    "            status = std::move(released);",
                    "        }",
                    "    }",
                ]
            lines += ["    return status;", "}"]
        bits = "bits" if dependents else "/*bits*/"
        lines += [
            "",
            f"[[maybe_unused]] vh::Status release_{snake}(uint64_t {bits}, std::unique_ptr<{handle.cpp}> object) {{",
            f"    vh::Status status = destroy_{snake}_dependents(bits);" if dependents else "    vh::Status status;",
        ]
        destroyer = library.destroyer(handle)
        if destroyer is not None and len(destroyer.parameters) == 1 and destroyer.result is None:
            lines += [
                f"    vh::Status destroyed = {library.cpp_namespace}::{destroyer.name}(std::move(object));",
                "    return status.ok() ? std::move(destroyed) : std::move(status);",
            ]
        else:
            lines += ["    object.reset();", "    return status;"]
        lines.append("}")
    return lines


def _handle_access(library: Library) -> list[str]:
    """`<cpp_namespace>::handles`: the library's own code finds the object behind a handle, and destroys every
    object of a handle, as a teardown hook needs."""

    tabled = [handle for handle in library.handles.values() if handle.tabled]
    if not tabled:
        return []
    lines = ["", f"namespace {library.cpp_namespace}::handles {{", ""]
    for handle in tabled:
        lines += [
            f"{handle.cpp} *find({handle.c_name} handle) {{ return {handle.table}().find(handle.bits); }}",
            "",
            f"template <> vh::Status destroy_all<{handle.c_name}>() {{",
            "    vh::Status status;",
            f"    for (auto &taken : {handle.table}().take_all()) {{",
            f"        vh::Status released = release_{_snake(handle.name)}(taken.first, std::move(taken.second));",
            "        if (status.ok() && !released.ok()) {",
            "            status = std::move(released);",
            "        }",
            "    }",
            "    return status;",
            "}",
            "",
        ]
    lines += [f"}} // namespace {library.cpp_namespace}::handles", ""]
    return lines


def _thunks(library: Library) -> str:
    lines = [
        f"// Generated by vernon_dsl.bindgen from {library.path.name}; do not edit.",
        "// The C functions of the library: handle lookup, a call to the implementation, status and last error.",
        f'#include "{library.name}_api_impl.h"',
        "",
        "#include <cstring>",
        "#include <exception>",
        "#include <new>",
        "#include <string>",
        "#include <type_traits>",
        "#include <utility>",
        "",
    ]
    cast = [struct for struct in library.structs.values() if struct.cpp is not None and struct.convert is None]
    for struct in cast:
        lines += [
            f"static_assert(sizeof({struct.cpp}) == sizeof({struct.c_name}) && "
            f"alignof({struct.cpp}) == alignof({struct.c_name}),",
            f'              "{struct.cpp} has the C layout of {struct.name}, or needs convert=");',
            f"static_assert(std::is_trivially_copyable<{struct.cpp}>::value, "
            f'"{struct.cpp} is passed as the bytes of {struct.name}");',
        ]
    if cast:
        lines.append("")
    lines += [
        "namespace {",
        "",
        "thread_local std::string last_error;",
        "",
        "[[maybe_unused]] int32_t fail(int32_t status, std::string message) {",
        "    last_error = std::move(message);",
        "    // Zero is success and negative statuses are Host code's own.",
        "    return status > 0 ? status : vh::kFailed;",
        "}",
        "",
        "[[maybe_unused]] int32_t fail(vh::Error error) { return fail(error.status, std::move(error.message)); }",
        "",
        "[[maybe_unused]] int32_t finish(vh::Status status) {",
        "    return status.ok() ? 0 : fail(std::move(status.error()));",
        "}",
        "",
        "// C callers cannot unwind a C++ exception, so a throwing implementation fails the call instead.",
        "template <typename Body> int32_t guard(const char *symbol, Body &&body) noexcept {",
        "    try {",
        "        return body();",
        "    } catch (const std::bad_alloc &) {",
        '        return fail(vh::kOutOfMemory, std::string(symbol) + ": out of memory");',
        "    } catch (const std::exception &error) {",
        '        return fail(vh::kFailed, std::string(symbol) + ": " + error.what());',
        "    } catch (...) {",
        '        return fail(vh::kFailed, std::string(symbol) + ": an unknown C++ exception");',
        "    }",
        "}",
    ]
    for handle in library.handles.values():
        if not handle.tabled:
            continue
        lines += [
            "",
            f"vh::HandleTable<{handle.cpp}> &{handle.table}() {{",
            f"    static vh::HandleTable<{handle.cpp}> table;",
            "    return table;",
            "}",
        ]
    lines += _cascades(library)
    lines += _descriptors(library)
    lines += _call_records(library)
    lines += ["", "} // namespace"]
    lines += _handle_access(library)
    lines.append("")
    for extern in library.externs:
        lines += _thunk(library, extern)
        lines.append("")
    lines += [
        f'extern "C" uint64_t {library.prefix}last_error(char *buffer, uint64_t capacity) {{',
        "    return vh::copy_message(last_error, buffer, capacity);",
        "}",
        "",
        f'extern "C" const char *{library.prefix}schema(void) {{ return {library.prefix.upper()}SCHEMA; }}',
        "",
        f'extern "C" const VhType *const *{library.prefix}types(uint32_t *count) {{',
        "    if (count != nullptr) {",
        "        *count = listed_type_count;",
        "    }",
        "    return listed_types;",
        "}",
        "",
    ]
    return "\n".join(lines)


# The nanobind module.


def _python_type(library: Library, extern: HostExtern | None, value_type: ConcreteType) -> str:
    if extern is not None and _is_variable(extern, value_type):
        return "T"
    if value_type.kind == "scalar":
        return _PYTHON_SCALARS.get(value_type.name, "int")
    if value_type.kind in {"handle", "enum"}:
        return value_type.name
    if value_type.kind == "str":
        return "str"
    if value_type.kind == "tensor":
        return f"list[{_python_type(library, None, element_of(value_type))}]"
    if value_type.kind == "list":
        return f"list[{_python_type(library, None, element_of(value_type))}]"
    if value_type.kind == "dict":
        key, value = (_python_type(library, None, item) for item in _items(value_type))
        return f"dict[{key}, {value}]"
    return value_type.name


def _from_python(library: Library, value_type: ConcreteType, expression: str) -> str:
    """A C value made of the Python object `expression`, for a dict entry."""

    return f"nb::cast<{_c_item(library, value_type)}>({expression})"


def _to_python(value_type: ConcreteType, expression: str) -> str:
    """A Python object made of the C value `expression`, copied out of memory the call releases."""

    return f"nb::cast({expression}, nb::rv_policy::copy)"


def _owns(library: Library, handle: _Handle) -> bool:
    """Whether Python objects of the handle's created objects destroy them: a `<Handle>.Owned` does."""

    destroyer = library.destroyer(handle)
    return handle.tabled and destroyer is not None and len(destroyer.parameters) == 1 and destroyer.result is None


def _nanobind_parameter(library: Library, extern: HostExtern, name: str, value_type: ConcreteType) -> tuple[str, str]:
    """The lambda parameter a Python argument arrives as, and the C argument made of it."""

    if _is_variable(extern, value_type):
        return f"nb::handle {name}", f"nb::inst_ptr<void>({name})"
    if value_type.kind == "list":
        item = _c_item(library, element_of(value_type))
        return f"const std::vector<{item}> &{name}", f"VhSpan{{{name}.data(), {name}.size()}}"
    if value_type.kind == "dict":
        return f"const nb::dict &{name}", f"VhSpan{{{name}_entries.data(), {name}_entries.size()}}"
    c_name = library.names.name(value_type)
    if value_type.kind == "handle":
        # A destroyed handle's object forgets its bits, so an Owned one does not destroy it again.
        constant = "" if _destroys(extern) == name else "const "
        return f"{constant}{c_name} &{name}", name
    if value_type.kind == "str":
        return f"std::string_view {name}", f"VhStrArg{{{name}.data(), {name}.size()}}"
    if value_type.kind == "struct":
        return f"const {c_name} &{name}", f"&{name}"
    if value_type.kind == "tensor":
        return f"const {_c_item(library, value_type)} &{name}", f"{name}.data()"
    return f"{c_name} {name}", name


def _nanobind_function(library: Library, extern: HostExtern) -> list[str]:
    parameters = [_nanobind_parameter(library, extern, name, value) for name, value in extern.parameters]
    names = [name for name, _ in extern.parameters]
    typed = _type_parameter(extern)
    body: list[str] = []
    if typed is not None and extern.type_passed:
        index, type_name = typed
        parameters.insert(index, (f"nb::handle {type_name}", f"{type_name}_type"))
        names.insert(index, type_name)
        message = f"{extern.name}: {type_name} is not a Struct class of this module"
        body.append(f'const VhType *{type_name}_type = descriptor_of({type_name}, "{message}");')
    elif typed is not None:
        index, type_name = typed
        value = names[index]
        parameters.insert(index, ("", f"{type_name}_type"))
        message = f"{extern.name}: {value} is not a Struct of this module"
        body.append(f'const VhType *{type_name}_type = descriptor_of({value}.type(), "{message}");')
    for name, value_type in extern.parameters:
        if _is_variable(extern, value_type) and not extern.type_passed:
            continue
        if _is_variable(extern, value_type):
            assert typed is not None
            body += [
                f"if (PyObject_IsInstance({name}.ptr(), {typed[1]}.ptr()) != 1) {{",
                f'    throw nb::type_error("{extern.name}: {name} is not an instance of {typed[1]}");',
                "}",
            ]
        elif value_type.kind == "dict":
            key, value = _items(value_type)
            entry = f"vh::Entry<{_c_item(library, key)}, {_c_item(library, value)}>"
            body += [
                f"std::vector<{entry}> {name}_entries;",
                f"for (auto [key, value] : {name}) {{",
                f"    {name}_entries.push_back({{{_from_python(library, key, 'key')}, "
                f"{_from_python(library, value, 'value')}}});",
                "}",
            ]
    arguments = [argument for _, argument in parameters]
    result = extern.result
    destroys = _destroys(extern)
    policies = ""
    if extern.abi != ExternAbi.STATUS:
        call = f"{extern.symbol}({', '.join(arguments)})"
        if result is None:
            body.append(f"{call};")
        else:
            body.append(f"return {call};")
    elif destroys is not None and result is None:
        body += [
            f"int32_t status = {extern.symbol}({', '.join(arguments)});",
            "if (status == 0) {",
            f"    {destroys}.bits = 0;",
            "}",
            "check(status);",
        ]
    elif result is not None and _is_variable(extern, result):
        assert typed is not None
        body.append(f"nb::object out = {typed[1]}();")
        body.append(f"check({extern.symbol}({', '.join([*arguments, 'nb::inst_ptr<void>(out)'])}));")
        body.append("return out;")
    elif result is not None and result.kind in _OWNED_RESULTS:
        body.append("Received out;")
        body.append(f"check({extern.symbol}({', '.join([*arguments, '&out.list', '&kAllocator'])}));")
        if result.kind == "str":
            body.append("return out.text();")
        else:
            item = _items_type(result, lambda item: _c_item(library, item))
            body += [
                f"const auto *items = static_cast<const {item} *>(vh_list_data(&out.list));",
                f"nb::{result.kind} result;",
                "for (uint64_t index = 0; index < out.list.size; ++index) {",
            ]
            if result.kind == "list":
                body.append(f"    result.append({_to_python(element_of(result), 'items[index]')});")
            else:
                key, value = _items(result)
                body.append(
                    f"    result[{_to_python(key, 'items[index].key')}] = {_to_python(value, 'items[index].value')};"
                )
            body += ["}", "return result;"]
    else:
        if result is not None:
            if result.kind == "tensor":
                body.append(f"{_c_item(library, result)} out{{}};")
                arguments.append("out.data()")
            else:
                body.append(f"{library.names.name(result)} out{{}};")
                arguments.append("&out")
        body.append(f"check({extern.symbol}({', '.join(arguments)}));")
        handle = None if result is None else library.handle(result)
        if handle is not None and _owns(library, handle):
            body.append(f"return Owned{handle.name}(out.bits);")
            if handle.depends_on is not None:
                # The created object keeps the Python object of its parent alive, which destroying would destroy it.
                parent = names.index(_parent_parameter(library, extern, handle))
                policies = f", nb::keep_alive<0, {parent + 1}>()"
        elif result is not None:
            body.append("return out;")
    keywords = "".join(f', "{name}"_a' for name in names)
    lambda_parameters = ", ".join(parameter for parameter, _ in parameters if parameter)
    lines = [f'    m.def("{extern.name}", []({lambda_parameters}) {{']
    lines += [f"        {line}" for line in body]
    lines.append(f"    }}{keywords}{policies});")
    return lines


def _structs(library: Library) -> list[str]:
    """The library's Struct names, each after the Structs its fields hold."""

    ordered, _ = boundary_types([library.program])
    return [value_type.name for value_type in ordered if value_type.kind == "struct"]


def _owned_handles(library: Library) -> list[str]:
    """`Owned<Handle>`: the C handle of an object Python created, destroying it with the Python object unless an
    explicit destroy zeroed its bits first."""

    lines: list[str] = []
    for handle in library.handles.values():
        if not _owns(library, handle):
            continue
        destroyer = library.destroyer(handle)
        assert destroyer is not None
        owned = f"Owned{handle.name}"
        lines += [
            "",
            f"struct {owned} : {handle.c_name} {{",
            f"    explicit {owned}(uint64_t value) : {handle.c_name}{{value}} {{}}",
            f"    {owned}({owned} &&other) noexcept : {handle.c_name}{{other.bits}} {{ other.bits = 0; }}",
            f"    {owned}(const {owned} &) = delete;",
            f"    {owned} &operator=(const {owned} &) = delete;",
            f"    {owned} &operator=({owned} &&) = delete;",
            f"    ~{owned}() {{",
            "        if (bits != 0) {",
            "            // A destructor cannot raise; a failure leaves its message as the last error.",
            f"            {destroyer.symbol}({handle.c_name}{{bits}});",
            "        }",
            "    }",
            "};",
        ]
    return lines


def _nanobind_handles(library: Library) -> list[str]:
    lines: list[str] = []
    for handle in library.handles.values():
        c_name = handle.c_name
        variable = f"{_snake(handle.name)}_class"
        lines += [
            f'    auto {variable} = nb::class_<{c_name}>(m, "{handle.name}")',
            "        .def(nb::init<>())",
            f'        .def("__init__", []({c_name} *self, uint64_t bits) {{ new (self) {c_name}{{bits}}; }}, "bits"_a)',
            f'        .def_ro("bits", &{c_name}::bits)',
            f'        .def("__int__", []({_const(c_name)} self) {{ return self.bits; }})',
            f'        .def("__index__", []({_const(c_name)} self) {{ return self.bits; }})',
            f'        .def("__bool__", []({_const(c_name)} self) {{ return self.bits != 0; }})',
            f'        .def("__hash__", []({_const(c_name)} self) {{ return std::hash<uint64_t>()(self.bits); }})',
            f'        .def("__eq__", []({_const(c_name)} self, nb::handle other) {{',
            f"            return nb::isinstance<{c_name}>(other) &&",
            f"                   nb::cast<{_const(c_name)}>(other).bits == self.bits;",
            '        }, "other"_a.none())',
            f'        .def("__repr__", []({_const(c_name)} self) {{',
            f'            return "{handle.name}(" + std::to_string(self.bits) + ")";',
            "        });",
        ]
        if _owns(library, handle):
            lines.append(f'    nb::class_<Owned{handle.name}, {c_name}>({variable}, "Owned");')
        else:
            lines.append(f"    (void){variable};")
    return lines


def _const(c_name: str) -> str:
    return f"const {c_name} &"


def _nanobind_types(library: Library) -> list[str]:
    lines = _nanobind_handles(library)
    for host_enum in library.program.enums.values():
        members = f"{_snake(host_enum.name)}_members"
        lines.append(f"    nb::dict {members};")
        lines += [f'    {members}["{name}"] = {value};' for name, value in host_enum.members]
        lines.append(f'    m.attr("{host_enum.name}") = int_enum("{host_enum.name}", {members});')
    for struct_name in _structs(library):
        host_struct = library.program.structs[struct_name]
        c_name = library.names.name(ConcreteType("struct", host_struct.name))
        lines.append(f'    nb::class_<{c_name}>(m, "{host_struct.name}")')
        lines.append("        .def(nb::init<>())")
        parameters = []
        stores = []
        for field, field_type in host_struct.fields:
            if field_type.kind == "tensor":
                parameters.append(f"const {_c_item(library, field_type)} &{field}")
                stores.append(f"std::memcpy(self->{field}, {field}.data(), sizeof(self->{field}));")
            else:
                parameters.append(f"{_c_item(library, field_type)} {field}")
                stores.append(f"self->{field} = {field};")
        keywords = "".join(f', "{field}"_a' for field, _ in host_struct.fields)
        if host_struct.fields:
            lines += [
                f'        .def("__init__", []({c_name} *self, {", ".join(parameters)}) {{',
                f"            new (self) {c_name}{{}};",
                *(f"            {store}" for store in stores),
                f"        }}{keywords})",
            ]
        for field, field_type in host_struct.fields:
            if field_type.kind == "tensor":
                value = _c_item(library, field_type)
                lines += [
                    f'        .def_prop_rw("{field}",',
                    f"            [](const {c_name} &self) {{",
                    f"                {value} value;",
                    f"                std::memcpy(value.data(), self.{field}, sizeof(value));",
                    "                return value;",
                    "            },",
                    f"            []({c_name} &self, const {value} &value) {{",
                    f"                std::memcpy(self.{field}, value.data(), sizeof(value));",
                    "            })",
                ]
            else:
                lines.append(f'        .def_rw("{field}", &{c_name}::{field})')
        lines[-1] += ";"
    lines += [
        "    // A Struct class's descriptor, which generic functions pass for it.",
        "    uint32_t type_count = 0;",
        f"    const VhType *const *types = {library.prefix}types(&type_count);",
        "    for (uint32_t index = 0; index < type_count; ++index) {",
        f"        if (types[index]->kind == {STRUCT} && nb::hasattr(m, types[index]->name)) {{",
        '            m.attr(types[index]->name).attr("vh_type") = reinterpret_cast<uintptr_t>(types[index]);',
        "        }",
        "    }",
    ]
    return lines


def _nanobind(library: Library) -> str:
    module = f"{library.name}_api"
    lines = [
        f"// Generated by vernon_dsl.bindgen from {library.path.name}; do not edit.",
        f"// The {module} Python module: Python functions over the C functions, and the symbol table",
        "// ExternRegistry.bind_module binds Python-mode Host code to.",
        "#include <nanobind/nanobind.h>",
        "#include <nanobind/stl/array.h>",
        "#include <nanobind/stl/string.h>",
        "#include <nanobind/stl/string_view.h>",
        "#include <nanobind/stl/vector.h>",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <functional>",
        "#include <string>",
        "#include <vector>",
        "",
        f'#include "{module}.h"',
        '#include "vernon_host/bindgen.hpp"',
        "",
        "namespace nb = nanobind;",
        "using namespace nb::literals;",
        "",
        "namespace {",
        "",
        "std::string last_error() {",
        f"    std::string message(static_cast<size_t>({library.prefix}last_error(nullptr, 0)), '\\0');",
        f"    {library.prefix}last_error(message.data(), message.size() + 1);",
        "    return message;",
        "}",
        "",
        "// Raises HostExecutionError, as a failing extern does in Python-mode Host code. The message is read first:",
        "// the import can collect an Owned handle, whose destroy would replace it.",
        "void check(int32_t status) {",
        "    if (status == 0) {",
        "        return;",
        "    }",
        "    std::string message = last_error();",
        '    nb::object error_type = nb::module_::import_("vernon_dsl.hostlang").attr("HostExecutionError");',
        "    nb::object error = error_type(status, message);",
        "    PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(error.ptr())), error.ptr());",
        "    throw nb::python_error();",
        "}",
        "",
        "// The descriptor of a Struct class of this module, which a generic function's `type` argument names or",
        "// its value is an instance of; `message` is the error otherwise.",
        "[[maybe_unused]] const VhType *descriptor_of(nb::handle type, const char *message) {",
        '    if (!PyType_Check(type.ptr()) || !nb::hasattr(type, "vh_type")) {',
        "        throw nb::type_error(message);",
        "    }",
        '    return reinterpret_cast<const VhType *>(nb::cast<uintptr_t>(type.attr("vh_type")));',
        "}",
        *_owned_handles(library),
        "",
        "[[maybe_unused]] nb::object int_enum(const char *name, const nb::dict &members) {",
        '    return nb::module_::import_("enum").attr("IntEnum")(name, members);',
        "}",
        "",
        "// Owned results these functions receive, which the thunks, linked into this module, fill with its own",
        "// malloc; a Received frees them once read.",
        "[[maybe_unused]] void *allocate(void * /*user*/, uint64_t size, uint64_t /*alignment*/) {",
        "    return std::malloc(static_cast<size_t>(size));",
        "}",
        "",
        "[[maybe_unused]] void release(void * /*user*/, void *pointer, uint64_t /*size*/, uint64_t /*alignment*/) {",
        "    std::free(pointer);",
        "}",
        "",
        "[[maybe_unused]] const VhAllocator kAllocator{sizeof(VhAllocator), nullptr, allocate, release};",
        "",
        "struct [[maybe_unused]] Received {",
        "    VhList list{};",
        "    Received() = default;",
        "    Received(const Received &) = delete;",
        "    Received &operator=(const Received &) = delete;",
        "    ~Received() { std::free(vh_list_data(&list)); }",
        "    std::string text() const {",
        "        return list.size == 0 ? std::string()",
        "                              : std::string(static_cast<const char *>(vh_list_data(&list)),",
        "                                            static_cast<size_t>(list.size));",
        "    }",
        "};",
        "",
        "} // namespace",
        "",
        f"// Defines the module's contents in `m`. Defining {library.macro}_NANOBIND_HOSTED leaves out the module",
        "// itself, so an existing extension can hold these as a submodule, sharing the library's handle tables.",
        f"void {library.prefix}nanobind(nb::module_ &m) {{",
        f'    m.attr("vh_schema") = {library.prefix.upper()}SCHEMA;',
        '    m.def("vh_symbols", []() {',
        "        nb::dict symbols;",
    ]
    for extern in library.externs:
        lines.append(f'        symbols["{extern.symbol}"] = reinterpret_cast<uintptr_t>(&{extern.symbol});')
    lines += [
        "        return symbols;",
        "    });",
        '    m.def("vh_last_error", &last_error);',
        "    // What a failed status raises, so callers catch it from this module.",
        '    m.attr("HostExecutionError") = nb::module_::import_("vernon_dsl.hostlang").attr("HostExecutionError");',
    ]
    lines += _nanobind_types(library)
    for extern in library.externs:
        lines += _nanobind_function(library, extern)
    lines += [
        "}",
        "",
        f"#ifndef {library.macro}_NANOBIND_HOSTED",
        f"NB_MODULE({module}, m) {{ {library.prefix}nanobind(m); }}",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def _stubs(library: Library) -> str:
    lines = [
        f'"""Generated by vernon_dsl.bindgen from {library.path.name}; do not edit."""',
        "",
        "from enum import IntEnum",
        "from typing import TypeVar, overload",
        "",
        "from vernon_dsl.hostlang import HostExecutionError as HostExecutionError",
        "",
        'T = TypeVar("T")',
        "",
        "vh_schema: str",
        "",
        "def vh_symbols() -> dict[str, int]: ...",
        "def vh_last_error() -> str: ...",
    ]
    for handle in library.handles.values():
        lines += [
            "",
            f"class {handle.name}:",
            "    bits: int",
            "    def __init__(self, bits: int = 0) -> None: ...",
            "    def __int__(self) -> int: ...",
            "    def __index__(self) -> int: ...",
        ]
    for host_enum in library.program.enums.values():
        lines += ["", f"class {host_enum.name}(IntEnum):"]
        lines += [f"    {name} = {value}" for name, value in host_enum.members]
    for struct_name in _structs(library):
        host_struct = library.program.structs[struct_name]
        lines += ["", f"class {host_struct.name}:"]
        fields = [f"{field}: {_python_type(library, None, value)}" for field, value in host_struct.fields]
        lines += [f"    {field}" for field in fields]
        lines.append("    vh_type: int")
        if fields:
            lines += [
                "    @overload",
                "    def __init__(self) -> None: ...",
                "    @overload",
                f"    def __init__(self, {', '.join(fields)}) -> None: ...",
            ]
        else:
            lines.append("    def __init__(self) -> None: ...")
    lines.append("")
    for extern in library.externs:
        parameters = [f"{name}: {_python_type(library, extern, value)}" for name, value in extern.parameters]
        typed = _type_parameter(extern)
        if typed is not None and extern.type_passed:
            parameters.insert(typed[0], f"{typed[1]}: type[T]")
        result = "None" if extern.result is None else _python_type(library, extern, extern.result)
        lines.append(f"def {extern.name}({', '.join(parameters)}) -> {result}: ...")
    lines.append("")
    return "\n".join(lines)


def generate(path: str | Path) -> dict[str, str]:
    """The level 1 outputs of the `vd.library` module at `path`, by file name."""

    library = read_library(path)
    stem = f"{library.name}_api"
    return {
        f"{stem}.h": _api_header(library),
        f"{stem}_impl.h": _impl_header(library),
        f"{stem}_thunks.cpp": _thunks(library),
        f"{stem}_nanobind.cpp": _nanobind(library),
        f"{stem}.pyi": _stubs(library),
    }


def _outputs(path: str | Path, directory: str | Path, pyi: str | Path | None) -> dict[Path, str]:
    """The outputs by where they go: all in `directory`, except the stub at `pyi` when given, as where an editor
    finds it, beside the Python module that re-exports the extension's names."""

    outputs = generate(path)
    stub = next(name for name in outputs if name.endswith(".pyi"))
    return {
        Path(pyi) if pyi is not None and name == stub else Path(directory) / name: text
        for name, text in outputs.items()
    }


def write(path: str | Path, directory: str | Path, pyi: str | Path | None = None) -> list[Path]:
    written = []
    for target, text in _outputs(path, directory, pyi).items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        written.append(target)
    return written


def check(path: str | Path, directory: str | Path, pyi: str | Path | None = None) -> list[Path]:
    """The outputs that are missing or differ from what the module generates."""

    return [
        target
        for target, text in _outputs(path, directory, pyi).items()
        if not target.exists() or target.read_text(encoding="utf-8") != text
    ]
