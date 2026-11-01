"""Links Host AOT objects into a shared library this process loads, so tests run the code packaged builds ship.

The object imports its extern symbols. A generated C shim defines each as a call through a function-pointer slot,
filled from an ExternRegistry once the library is loaded, the way an engine's C API would provide them.
"""

from __future__ import annotations

import ctypes
import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from vernon_dsl import _native
from vernon_dsl.frontend.model import ConcreteType
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, default_registry
from vernon_dsl.hostlang.aot import compile_programs
from vernon_dsl.hostlang.externs import HostStr
from vernon_dsl.hostlang.layout import layout, pack, unpack
from vernon_dsl.hostlang.model import ExternAbi, HostExtern, HostFunctionInfo, HostProgram
from vernon_dsl.hostlang.reflect import read_module
from vernon_dsl.hostlang.values import from_python, to_python

_C_SCALARS = {
    "bool": "bool",
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
    "f32": "float",
    "f64": "double",
}
_CTYPES = {
    "bool": ctypes.c_bool,
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}


def _msvc_environment() -> dict[str, str] | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    root = subprocess.run(
        [str(vswhere), "-latest", "-property", "installationPath"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    ).stdout.strip()
    vcvars = Path(root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        return None
    output = subprocess.run(
        f'cmd /s /c ""{vcvars}" >nul && set"', capture_output=True, text=True, errors="replace", check=False, shell=True
    ).stdout
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


@functools.cache
def toolchain() -> tuple[str, str, dict[str, str]] | None:
    """(kind, compiler, environment) of the C toolchain objects link with: MSVC on Windows, `cc` elsewhere."""

    if sys.platform == "win32":
        environment = _msvc_environment()
        compiler = None if environment is None else shutil.which("cl", path=environment.get("Path", ""))
        return None if compiler is None else ("msvc", compiler, environment)  # type: ignore[return-value]
    compiler = shutil.which("cc")
    return None if compiler is None else ("cc", compiler, dict(os.environ))


def _c_type(value_type: ConcreteType) -> str:
    if value_type.kind == "scalar":
        return _C_SCALARS[value_type.name]
    if value_type.kind == "handle":
        return "uint64_t"
    if value_type.kind == "enum":
        return "int32_t"
    if value_type.kind in {"str", "list", "dict"}:
        # A str and a VhSpan share the `{pointer, uint64_t}` shape.
        return "VhShimStr"
    return "const void *"


_RUNTIME_FUNCTIONS = (
    ("int32_t", "vernonRuntimeHostListReserve", ("void *", "uint64_t", "uint64_t", "uint64_t", "const void *")),
    ("void", "vernonRuntimeHostListRelease", ("void *", "uint64_t", "uint64_t", "const void *")),
    ("const void *", "vernonRuntimeHostDefaultAllocator", ()),
    ("int64_t", "vernonRuntimeHostDictFind", ("const void *", "const void *", "const void *")),
    ("int32_t", "vernonRuntimeHostDictInsert", ("void *", "const void *", "const void *", "const void *", "void *")),
    ("int32_t", "vernonRuntimeHostDictRemove", ("void *", "const void *", "const void *", "const void *")),
    ("void", "vernonRuntimeHostDictClear", ("void *", "const void *", "const void *")),
    ("void", "vernonRuntimeHostDictRelease", ("void *", "const void *", "const void *")),
    ("int32_t", "vernonRuntimeHostStrAppend", ("void *", "const void *", "uint64_t", "const void *", "const void *")),
    (
        "int32_t",
        "vernonRuntimeHostStrAppendInteger",
        ("void *", "uint64_t", "int32_t", "const void *", "const void *"),
    ),
    ("int32_t", "vernonRuntimeHostStrAppendFloat", ("void *", "double", "const void *", "const void *")),
    (
        "int32_t",
        "vernonRuntimeHostStorageAllocate",
        ("void *", "uint32_t", "const void *", "uint64_t", "uint64_t", "const void *"),
    ),
    ("void", "vernonRuntimeHostStorageRelease", ("void *", "uint32_t", "uint64_t", "uint64_t", "const void *")),
    ("void", "vernonRuntimeHostStorageFill", ("void *", "uint32_t", "const void *", "uint64_t")),
)


def _shim(externs: list[HostExtern]) -> str:
    lines = [
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "#ifdef _WIN32",
        "#define VH_EXPORT __declspec(dllexport)",
        "#else",
        '#define VH_EXPORT __attribute__((visibility("default")))',
        "#endif",
        "typedef struct VhShimStr { const char *data; uint64_t size; } VhShimStr;",
    ]
    # Containers call into VernonRuntime; the slots take the runtime linked into vernon_dsl._native.
    for result, name, parameters in _RUNTIME_FUNCTIONS:
        declared = ", ".join(f"{kind} a{index}" for index, kind in enumerate(parameters)) or "void"
        forwarded = ", ".join(f"a{index}" for index in range(len(parameters)))
        pointer = f"({result} (*)({', '.join(parameters) or 'void'}))vh_slot_{name}"
        lines.append(f"VH_EXPORT void *vh_slot_{name};")
        lines.append(
            f"{result} {name}({declared}) {{ {'' if result == 'void' else 'return '}({pointer})({forwarded}); }}"
        )
    for extern in externs:
        parameters = [_c_type(value) for _, value in extern.parameters]
        if extern.type_argument is not None:
            parameters.insert(extern.type_argument[0], "const void *")
        if extern.abi == ExternAbi.STATUS:
            result = "int32_t"
            if extern.result is not None:
                parameters.append("void *")
                if extern.result.kind in {"str", "list", "dict"}:
                    parameters.append("const void *")
        else:
            result = "void" if extern.result is None else _c_type(extern.result)
        declared = ", ".join(f"{kind} a{index}" for index, kind in enumerate(parameters)) or "void"
        forwarded = ", ".join(f"a{index}" for index in range(len(parameters)))
        pointer = f"({result} (*)({', '.join(parameters) or 'void'}))vh_slot_{extern.symbol}"
        call = f"({pointer})({forwarded})"
        lines.append(f"VH_EXPORT void *vh_slot_{extern.symbol};")
        lines.append(f"{result} {extern.symbol}({declared}) {{ {'' if result == 'void' else 'return '}{call}; }}")
    return "\n".join(lines) + "\n"


HOST_INCLUDE = Path(__file__).resolve().parents[1] / "vernon_dsl" / "hostlang" / "include"


def build_c_library(source: Path, directory: Path, name: str) -> Path:
    """Builds `source` into a shared library exporting its `VH_EXPORT` functions, with `vernon_host/` headers on
    the include path, as a library a registry binds through `bind_library`."""

    found = toolchain()
    if found is None:
        raise RuntimeError("no C toolchain builds libraries on this machine")
    kind, compiler, environment = found
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "msvc":
        library = directory / f"{name}.dll"
        command = [
            compiler,
            "/nologo",
            "/LD",
            "/DVH_EXPORT=__declspec(dllexport)",
            f"/I{HOST_INCLUDE}",
            str(source),
            f"/Fe{library}",
            f"/Fo{directory}{os.sep}",
        ]
    else:
        library = directory / f"lib{name}.so"
        command = [
            compiler,
            "-shared",
            "-fPIC",
            '-DVH_EXPORT=__attribute__((visibility("default")))',
            f"-I{HOST_INCLUDE}",
            str(source),
            "-o",
            str(library),
        ]
    built = subprocess.run(
        command, capture_output=True, text=True, errors="replace", env=environment, cwd=directory, check=False
    )
    if built.returncode != 0:
        raise RuntimeError(f"building {source.name} failed:\n{built.stdout}{built.stderr}")
    return library


class LinkedHost:
    """Compiled Host programs in a shared library loaded into this process. Extern symbols resolve through
    `registry` once, at load: native addresses are called directly and Python bindings through ctypes callbacks."""

    def __init__(
        self,
        programs: list[HostProgram],
        directory: Path,
        registry: ExternRegistry | None = None,
        prefix: str = "vernon_host_",
    ):
        found = toolchain()
        if found is None:
            raise RuntimeError("no C toolchain links Host objects on this machine")
        kind, compiler, environment = found
        self.prefix = prefix
        self.programs = {name: program for program in programs for name, _ in program.exports}
        self.errors: list[BaseException] = []
        self._keep_alive: list[Any] = []
        self._entries: dict[str, Any] = {}
        registry = registry or default_registry()
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = compile_programs(programs, prefix=prefix)
        paths = artifacts.write(directory, "host")
        externs: dict[str, tuple[HostExtern, HostProgram]] = {}
        for program in programs:
            for extern in program.reachable_externs():
                externs.setdefault(extern.symbol, (extern, program))
        shim = directory / "shim.c"
        shim.write_text(_shim([extern for extern, _ in externs.values()]), encoding="utf-8")
        exported = sorted({prefix + name for program in programs for name, _ in program.exports})
        if kind == "msvc":
            library = directory / "host.dll"
            command = [
                compiler,
                "/nologo",
                "/LD",
                str(shim),
                str(paths["object"]),
                f"/Fe{library}",
                f"/Fo{directory}{os.sep}",
                "/link",
                *(f"/EXPORT:{symbol}" for symbol in exported),
                f"/EXPORT:{prefix}module,DATA",
            ]
        else:
            library = directory / "libhost.so"
            command = [compiler, "-shared", "-fPIC", str(shim), str(paths["object"]), "-lm", "-o", str(library)]
        built = subprocess.run(
            command, capture_output=True, text=True, errors="replace", env=environment, cwd=directory, check=False
        )
        if built.returncode != 0:
            raise RuntimeError(f"linking the Host object failed:\n{built.stdout}{built.stderr}")
        self.library = ctypes.CDLL(str(library))
        for name, address in _native._host_runtime_symbols().items():
            ctypes.c_void_p.in_dll(self.library, f"vh_slot_{name}").value = address
        for symbol, (extern, program) in externs.items():
            address, owner = registry.native_address(extern, program.struct_fields, self.errors)
            ctypes.c_void_p.in_dll(self.library, f"vh_slot_{symbol}").value = address
            self._keep_alive.append(owner)

    @property
    def table(self) -> Any:
        """The loaded `VhModule` reflection table (`vernon_host/reflect.h`)."""

        return read_module(ctypes.addressof(ctypes.c_char.in_dll(self.library, self.prefix + "module")))

    def _info(self, name: str) -> HostFunctionInfo:
        program = self.programs.get(name)
        if program is None:
            raise KeyError(f"the library exports no '{name}'")
        return program.functions[dict(program.exports)[name]]

    def _entry(self, name: str) -> Any:
        function = self._entries.get(name)
        if function is None:
            info = self._info(name)
            arguments: list[Any] = []
            for _, parameter in info.parameters:
                if parameter.kind == "scalar":
                    arguments.append(_CTYPES[parameter.name])
                elif parameter.kind == "handle":
                    arguments.append(ctypes.c_uint64)
                elif parameter.kind == "enum":
                    arguments.append(ctypes.c_int32)
                else:
                    arguments.append(ctypes.c_void_p)
            if info.result is not None:
                arguments.append(ctypes.c_void_p)
            prototype = ctypes.CFUNCTYPE(ctypes.c_int32, *arguments)
            function = self._entries[name] = prototype((self.prefix + name, self.library))
        return function

    def call(self, name: str, *args: Any) -> Any:
        """Calls export `name` with Python values; results come back as boundary values (Structs as dicts)."""

        info = self._info(name)
        program = self.programs[name]
        fields = program.struct_fields
        if len(args) != len(info.parameters):
            raise TypeError(f"'{name}' takes {len(info.parameters)} arguments, got {len(args)}")
        native: list[Any] = []
        buffers: list[Any] = []
        for value, (_, parameter) in zip(args, info.parameters, strict=True):
            if program.is_place(parameter):
                native.append(ctypes.c_void_p(int(value)))
                continue
            if parameter.kind == "str":
                encoded = str(value).encode("utf-8")
                view = HostStr(encoded, len(encoded))
                buffers.append(view)
                native.append(ctypes.cast(ctypes.pointer(view), ctypes.c_void_p))
                continue
            converted = from_python(value, parameter, fields)
            if parameter.kind in {"scalar", "handle", "enum"}:
                native.append(converted)
            else:
                buffer = ctypes.create_string_buffer(pack(converted, parameter, fields))
                buffers.append(buffer)
                native.append(ctypes.cast(buffer, ctypes.c_void_p))
        out = None
        if info.result is not None:
            out = ctypes.create_string_buffer(max(layout(info.result, fields).size, 1))
            native.append(ctypes.cast(out, ctypes.c_void_p))
        self.errors.clear()
        status = self._entry(name)(*native)
        if self.errors:
            error = self.errors[0]
            self.errors.clear()
            raise error
        if status != 0:
            raise HostExecutionError(status, f"Host function '{name}' failed")
        if out is None or info.result is None:
            return None
        return to_python(unpack(out.raw, info.result, fields), info.result, fields)
