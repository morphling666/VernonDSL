"""IntEnums in Host code: Python mode and AOT agree, and the header, reflection table and JSON name members."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import textwrap
from enum import IntEnum
from pathlib import Path
from typing import Any

import pytest
import vernon_dsl as vd
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostModule, load_host_program

PROGRAMS = Path(__file__).parent / "host_programs"
ENUMS = PROGRAMS / "enums.py"
sys.path.insert(0, str(PROGRAMS))

import enums  # noqa: E402
from enums import Light, Mode  # noqa: E402

_FUNCTIONS = ["body_speed", "code", "faster", "from_code", "lights", "walk"]


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_compile_host")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")


def _registry(calls: list[Any]) -> ExternRegistry:
    def mode_speed(mode: Any) -> float:
        calls.append(mode)
        return float(int(mode)) + 0.5

    registry = ExternRegistry()
    registry.bind("demo_mode_speed", mode_speed)
    registry.bind("demo_next_mode", lambda mode: Mode.RUN if mode == Mode.WALK else Mode.IDLE)
    return registry


_CASES = [
    ("faster", (Mode.IDLE,), Mode.WALK),
    ("faster", (Mode.WALK,), Mode.RUN),
    ("faster", (Mode.RUN,), Mode.RUN),
    # A value no member has crosses the boundary unchanged.
    ("faster", (7,), Mode.RUN),
    ("code", (Mode.BACK,), -20),
    ("from_code", (4,), True),
    ("body_speed", ({"mode": Mode.BACK, "speed": 3.0},), -3.0),
    ("body_speed", ({"mode": Mode.RUN, "speed": 3.0},), 3.0),
    ("lights", (Light.GREEN,), (Light.GREEN, Mode.BACK)),
]


@pytest.fixture(scope="module")
def linked(tmp_path_factory: pytest.TempPathFactory) -> LinkedHost:
    from vernon_dsl.hostlang.aot import load_programs

    if toolchain() is None:
        pytest.skip("no C toolchain links Host objects")
    return LinkedHost(load_programs(ENUMS, _FUNCTIONS), tmp_path_factory.mktemp("enums"), _registry([]), "demo_")


@pytest.mark.parametrize(("name", "arguments", "expected"), _CASES)
def test_python_mode_and_aot_agree_on_enum_values(
    linked: LinkedHost, name: str, arguments: tuple[Any, ...], expected: Any
) -> None:
    python = getattr(enums, name).call(*arguments, registry=_registry([]))
    assert python == expected
    if isinstance(expected, IntEnum):
        assert python is expected
    assert linked.call(name, *arguments) == expected


def test_enum_struct_fields_and_extern_arguments() -> None:
    calls: list[Any] = []
    body = enums.walk.call(2.0, registry=_registry(calls))
    # JOG aliases WALK, so the Struct holds the WALK member.
    assert (body.mode, body.speed) == (Mode.WALK, 3.5)
    assert calls == [Mode.WALK] and type(calls[0]) is int
    assert enums.Body(5, 1.0).mode == 5
    with pytest.raises(TypeError, match="requires a Mode or an i32"):
        enums.Body(1.5, 1.0)


def test_enum_keyed_dicts_and_class_fields_read_and_write_as_json() -> None:
    from vernon_dsl.hostlang.value_json import ValueJsonError

    module = HostModule.load(ENUMS, _registry([]))
    tally = module.new("Tally")
    assert tally.last == Mode.RUN and type(tally.last).__name__ == "Mode"
    assert [tally.add(Mode.RUN), tally.add(Mode.WALK), tally.add(Mode.RUN)] == [1, 1, 2]
    assert module.to_json(tally) == {"counts": {"RUN": 2, "WALK": 1}, "last": "RUN"}

    module.from_json(tally, {"last": "BACK", "counts": {"IDLE": 3, "5": 1}})
    assert tally.last == Mode.BACK
    assert module.to_json(tally) == {"counts": {"IDLE": 3, "5": 1}, "last": "BACK"}
    module.from_json(tally, {"last": 7})
    assert tally.last == 7 and module.to_json(tally)["last"] == 7
    for value, message in [
        ({"last": "SPRINT"}, "Mode has no member 'SPRINT'; its members are IDLE, WALK, RUN, BACK, JOG"),
        ({"last": 1.5}, "Mode expects an integer"),
        ({"last": 1 << 40}, "out of range for Mode"),
    ]:
        with pytest.raises(ValueJsonError, match=message):
            module.from_json(tally, value)


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_enum_keyed_dicts_run_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    script = [Mode.RUN, Mode.WALK, Mode.RUN, Mode.JOG, 7, Mode.BACK, 7]
    tally = HostModule(ENUMS, _registry([]), eager=False).namespace.Tally()
    python = [tally.add(mode) for mode in script]
    assert python == [1, 1, 2, 2, 1, 1, 2]

    library = LinkedHost(load_programs(ENUMS), tmp_path, _registry([]), "demo_")
    descriptor = next(
        library.table.types[index].contents
        for index in range(library.table.type_count)
        if library.table.types[index].contents.name == b"Tally"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Tally__init", address)
    assert [library.call("Tally_add", address, mode) for mode in script] == python
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)


def test_reflection_describes_enum_members() -> None:
    from vernon_dsl.hostlang.reflect import ENUM

    module = HostModule.load(ENUMS, _registry([]))
    mode = module.type("Mode")
    assert (mode.kind, mode.size, mode.alignment, ENUM) == (17, 4, 4, 17)
    assert mode.doc == b"How a body moves."
    members = [(mode.enumerators[index].name, mode.enumerators[index].value) for index in range(mode.enumerator_count)]
    assert members == [(b"IDLE", 0), (b"WALK", 1), (b"RUN", 4), (b"BACK", -2), (b"JOG", 1)]
    assert mode.meta_count == 1 and mode.meta[0].key == b"cpp"


def test_header_and_json_reflection_spell_enums() -> None:
    from vernon_dsl.hostlang.aot import compile_programs, load_programs

    artifacts = compile_programs(load_programs(ENUMS, _FUNCTIONS), prefix="demo_")
    header = artifacts.header
    assert "typedef int32_t DemoMode;" in header
    assert "enum { DemoMode_IDLE = 0, DemoMode_WALK = 1, DemoMode_RUN = 4, DemoMode_BACK = -2, DemoMode_JOG = 1 };" in (
        header
    )
    assert "typedef int32_t demo_Light;" not in header and "typedef int32_t DemoLight;" in header
    assert "typedef struct DemoBody { DemoMode mode; float speed; } DemoBody;" in header
    assert "int32_t demo_faster(DemoMode mode, DemoMode *out);" in header
    reflected = artifacts.reflection["enums"]["Mode"]
    assert reflected["c_type"] == "DemoMode" and reflected["meta"] == {"cpp": "demo::Mode"}
    assert reflected["members"][3] == {"name": "BACK", "value": -2}
    assert artifacts.reflection["enums"]["Light"]["c_type"] == "DemoLight"


def test_c_application_reads_enums_through_the_header_and_table(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import compile_programs, load_programs
    from vernon_dsl.hostlang.reflect import include_dir

    found = toolchain()
    if found is None:
        pytest.skip("no C toolchain")
    paths = compile_programs(load_programs(ENUMS, _FUNCTIONS), prefix="demo_").write(tmp_path, "enums")
    executable = tmp_path / "consumer.exe"
    kind, compiler, environment = found
    sources = [str(PROGRAMS / "enums_consumer.c"), str(paths["object"])]
    if kind == "msvc":
        flags = ["/nologo", "/W4", "/WX", "/std:c11", f"/I{tmp_path}", f"/I{include_dir()}"]
        command = [compiler, *flags, *sources, f"/Fe{executable}", f"/Fo{tmp_path}{os.sep}"]
    else:
        flags = ["-std=c11", "-Wall", "-Werror", f"-I{tmp_path}", f"-I{include_dir()}"]
        command = [compiler, *flags, *sources, "-lm", "-o", str(executable)]
    built = subprocess.run(command, capture_output=True, text=True, env=environment, cwd=tmp_path, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert run.stdout.splitlines() == [
        'mode_kind=17 size=4 count=5 doc=How a body moves. cpp="demo::Mode"',
        "names=WALK,BACK,null",
        "faster=4 code=-20 from_code=1",
        'body_json={"mode":"WALK","speed":3.5}',
        'unknown_json={"mode":7,"speed":3.5}',
        "body_speed=-3.5",
    ]


def test_library_identity_covers_enum_values(tmp_path: Path) -> None:
    from vernon_dsl.hostlang import library_identity

    source = textwrap.dedent(
        """
        from enum import IntEnum

        import vernon_dsl as vd

        vd.library("motion")


        @vd.enum(meta=vd.meta(cpp="motion::Mode"))
        class Mode(IntEnum):
            IDLE = 0
            RUN = 1


        @vd.extern(effects="write")
        def set_mode(mode: Mode) -> None: ...
        """
    )

    def identity(directory: str, text: str) -> str:
        path = tmp_path / directory / "binding.py"
        path.parent.mkdir()
        path.write_text(text)
        return library_identity(path, "motion", 0)

    base = identity("base", source)
    assert identity("meta", source.replace("motion::Mode", "x::Mode")) == base
    assert identity("value", source.replace("RUN = 1", "RUN = 2")) != base
    assert identity("added", source.replace("RUN = 1", "RUN = 1\n    JUMP = 2")) != base


def test_enum_decorator_requires_an_int_enum() -> None:
    with pytest.raises(TypeError, match="IntEnum"):

        @vd.enum
        class NotAnEnum:
            A = 0


_HEADER = """
from enum import Enum, IntEnum

import vernon_dsl as vd
from vernon_dsl import f32, i32


@vd.enum
class Mode(IntEnum):
    IDLE = 0
    RUN = 1
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            @vd.host(effects="pure")
            def run(mode: Mode) -> bool:
                return mode < Mode.RUN
            """,
            "compare only with == and !=",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(mode: Mode) -> i32:
                return mode + 1
            """,
            "no safe common type for Mode and integer literal",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(mode: Mode) -> bool:
                return mode == 1
            """,
            "no safe common type for Mode and integer literal",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(value: i32) -> Mode:
                return Mode(value)
            """,
            "not constructed from an integer; name a member",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> Mode:
                return Mode.SPRINT
            """,
            "no member 'SPRINT'",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(value: i32) -> Mode:
                return value
            """,
            "unsafe implicit conversion from i32 to Mode",
        ),
        (
            """
            @vd.enum
            class Shade(Enum):
                DARK = 0

            @vd.host(effects="pure")
            def run(shade: Shade) -> i32:
                return 0
            """,
            "IntEnum",
        ),
        (
            """
            @vd.enum
            class Shade(IntEnum):
                DARK = 0.5

            @vd.host(effects="pure")
            def run(shade: Shade) -> i32:
                return 0
            """,
            "integer literal",
        ),
        (
            """
            @vd.enum
            class Shade(IntEnum):
                DARK = 3000000000

            @vd.host(effects="pure")
            def run(shade: Shade) -> i32:
                return 0
            """,
            "does not fit in an int32_t",
        ),
        (
            """
            @vd.enum
            class Shade(IntEnum):
                DARK = 0
                DARK = 1

            @vd.host(effects="pure")
            def run(shade: Shade) -> i32:
                return 0
            """,
            "DARK",
        ),
    ],
)
def test_enum_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    source = tmp_path / "invalid.py"
    source.write_text(_HEADER + textwrap.dedent(body))
    with pytest.raises(CompileError, match=message):
        load_host_program(source, "run")
