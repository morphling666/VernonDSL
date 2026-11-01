"""list and dict locals behave alike in Python mode and AOT, and compiled code releases them on every exit."""

from __future__ import annotations

import ctypes
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from test_host_strs import _CountingAllocator
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, load_host_module
from vernon_dsl.hostlang.model import HostStatus

PROGRAMS = Path(__file__).parent / "host_programs"
LOCALS = PROGRAMS / "container_locals.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")

_RUN, _WALK = 4, 1
_SCRIPT: list[tuple[str, tuple[Any, ...], Any]] = [
    ("tally", (_RUN, _RUN), 21),
    ("tally", (_RUN, _WALK), 12),
    ("evens", (0,), 0),
    ("evens", (3,), 6),
    # Squares 0, 4, 16, 36 and 64 sum to 120; popping 64 adds 64000 and leaves 4 items.
    ("evens", (9,), 64124),
    # green twice, blue1 twice, blue0 once; "red" was never seen, so the last text is deleted.
    ("words", (0,), 7762),
    ("words", (1,), 7672),
    ("batches", (4,), 1222),
    # The last critical hit, index 5, deals 7.5.
    ("strongest", (6,), 7),
    ("strongest", (1,), 0),
    ("early", (3,), 4),
    ("early", (20,), -1),
    ("missing", (1,), 10),
    ("missing", (2,), ("status", HostStatus.KEY_NOT_FOUND)),
    ("scored", (7,), 70),
    ("scored", (-1,), ("status", 9)),
]


def _registry() -> ExternRegistry:
    def score(value: int) -> int:
        if value < 0:
            raise HostExecutionError(9, "negative score")
        return value * 10

    return ExternRegistry().bind("locals_score", score)


def _python(arena: Any, name: str, arguments: tuple[Any, ...]) -> Any:
    try:
        return int(getattr(arena, name)(*arguments))
    except HostExecutionError as error:
        return ("status", error.status)
    except KeyError:
        # A dict local is a Python dict in Python mode, so a missing key is Python's own error.
        return ("status", HostStatus.KEY_NOT_FOUND)


def test_python_mode_runs_container_locals_as_python_containers() -> None:
    module = HostModule(LOCALS, _registry(), eager=False)
    assert [module.call("tally", _RUN, _RUN), module.call("tally", _RUN, _WALK)] == [21, 12]
    arena = module.namespace.Arena()
    assert [_python(arena, name, arguments) for name, arguments, _ in _SCRIPT] == [
        expected for _, _, expected in _SCRIPT
    ]
    arena.hits = [{"damage": 2.0, "critical": True}, {"damage": 5.0, "critical": False}, (1.5, True)]
    assert arena.critical_total() == 3.5 and list(arena.names) == [2]


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_container_locals_run_alike_and_are_released(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(LOCALS), tmp_path, _registry())
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Arena"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Arena__init", address)
    outcomes = []
    for name, arguments, _ in _SCRIPT:
        try:
            outcomes.append(int(library.call(f"Arena_{name}", address, *arguments)))
        except HostExecutionError as error:
            outcomes.append(("status", error.status))
    library.errors.clear()
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)

    outcomes += [library.call("tally", _RUN, _RUN), library.call("tally", _RUN, _WALK)]
    table.allocator[0] = None

    assert outcomes == [expected for _, _, expected in _SCRIPT] + [21, 12]
    assert counting.allocations > 10
    assert counting.live == {}, "every container local is released, on early returns and failures too"
    assert counting.mismatched == []


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "bad_locals.py"
    source.write_text(
        textwrap.dedent(
            """
            import vernon_dsl as vd
            from vernon_dsl import f32, i32
            """
        )
        + textwrap.dedent(body)
    )
    with pytest.raises(CompileError) as error:
        load_host_module(source)
    return str(error.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                values = []
                return 0
            """,
            "a list local is declared with its type, as in values: list[T] = []",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                values: list[i32] = [1, 2]
                return 0
            """,
            "a list is not passed as a value",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                counts: dict[f32, i32] = {}
                return 0
            """,
            "dict keys are integers, bools, handles, enums or str; f32 is not one",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                values: list[i32] = []
                copy = values
                return 0
            """,
            "'values' is a list local; index it, iterate it",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> list[i32]:
                values: list[i32] = []
                return values
            """,
            "lives in a class field or a local, which owns its memory; it is not passed",
        ),
        (
            """
            @vd.host(effects="pure")
            def total(values: list[i32]) -> i32:
                return 0
            """,
            "lives in a class field or a local, which owns its memory; it is not passed",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                counts: dict[i32, i32] = {}
                for key in counts:
                    counts = {}
                return 0
            """,
            "the loop iterates 'counts', so its body cannot rebind it; call counts.clear()",
        ),
        (
            """
            @vd.func(shared=True)
            def helper(x: i32) -> i32:
                values: list[i32] = []
                return x

            @vd.host(effects="pure")
            def run() -> i32:
                return helper(1)
            """,
            "containers are Host-only",
        ),
    ],
    ids=["unannotated", "display", "float-keys", "aliased", "returned", "parameter", "rebound-in-loop", "helper"],
)
def test_container_local_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)
