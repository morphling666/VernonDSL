"""Host dicts: class fields owning a hash table, in Python mode and in linked AOT objects alike."""

from __future__ import annotations

import ctypes
import gc
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, load_host_module
from vernon_dsl.hostlang.model import HostStatus

PROGRAMS = Path(__file__).parent / "host_programs"
SCOREBOARD = PROGRAMS / "scoreboard.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")

_KEY_NOT_FOUND = ("status", HostStatus.KEY_NOT_FOUND)
# (method, arguments, expected) run in order on one fresh instance.
_SCRIPT: list[tuple[str, tuple[Any, ...], Any]] = [
    ("score", (3, 10), 1),
    ("score", (1, 5), 2),
    ("score", (3, 4), 2),
    ("score", (7, 1), 3),
    ("points", (3,), 14),
    ("points", (9,), _KEY_NOT_FOUND),
    ("points_or", (9, -2), -2),
    ("has", (9,), True),
    ("total", (), 20),
    ("weighted", (), 3 * 1 + 1 * 2 + 7 * 3),
    ("best", (), 3),
    ("forget", (1,), 2),
    ("forget", (1,), _KEY_NOT_FOUND),
    ("take", (7,), 1),
    ("take", (7,), _KEY_NOT_FOUND),
    ("take_or", (7, 9), 9),
    # A removed key inserted again goes last.
    ("score", (1, 2), 2),
    ("weighted", (), 3 * 1 + 1 * 2),
    ("double_all", (), 32),
    ("grow_while_iterating", (), ("status", HostStatus.CONTAINER_CHANGED)),
    ("total", (), 32),
    ("record", (2.0,), 3.5),
    # len counts code points: "héight" is 6, in 7 bytes.
    ("retitle", (), 63),
    ("place", (7, 1.5), 3.0),
    ("flag", (True,), 1),
    ("flag", (False,), 1),
    ("flag", (True,), 2),
]


def _outcome(function: Any, *args: Any) -> Any:
    try:
        result = function(*args)
    except HostExecutionError as error:
        return ("status", error.status)
    return result if isinstance(result, bool) else float(result)


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_dicts_run_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    board = HostModule(SCOREBOARD, ExternRegistry(), eager=False).namespace.Scoreboard()
    python = [_outcome(getattr(board, name), *args) for name, args, _ in _SCRIPT]
    assert python == [expected for _, _, expected in _SCRIPT]

    library = LinkedHost(load_programs(SCOREBOARD), tmp_path, ExternRegistry())
    table = library.table
    descriptor = next(
        table.types[index].contents
        for index in range(table.type_count)
        if table.types[index].contents.name == b"Scoreboard"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Scoreboard__init", address)
    aot = [_outcome(library.call, f"Scoreboard_{name}", address, *args) for name, args, _ in _SCRIPT]
    assert aot == python
    assert _outcome(library.call, "Scoreboard_reset", address) == 0.0
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    assert bytes(instance) == bytes(descriptor.size), "fini releases every record and zeroes it"


def test_python_mode_fields_read_as_dicts() -> None:
    module = HostModule(SCOREBOARD, ExternRegistry(), eager=False)
    board = module.namespace.Scoreboard()
    assert board.scores == {} and len(board.stats) == 0
    board.score(4, 2)
    board.scores[9] = 5
    assert dict(board.scores.items()) == {4: 2, 9: 5} and list(board.scores) == [4, 9]
    board.stats = {"a": 1, "b": 2.5}
    assert board.stats == {"a": 1.0, "b": 2.5} and "b" in board.stats
    del board.stats["a"]
    assert repr(board.stats) == "{'b': 2.5}"
    with pytest.raises(HostExecutionError) as error:
        board.stats["a"]
    assert error.value.status == HostStatus.KEY_NOT_FOUND
    many = {key: key * 3 for key in range(1000)}
    board.scores = many
    for key in range(0, 1000, 2):
        del board.scores[key]
    assert dict(board.scores.items()) == {key: value for key, value in many.items() if key % 2}


def test_instances_release_their_dicts_when_collected() -> None:
    board = HostModule(SCOREBOARD, ExternRegistry(), eager=False).namespace.Scoreboard()
    board.score(1, 1)
    board.record(1.0)
    state = board._vh_state
    assert any(bytes(state))
    del board
    gc.collect()
    assert not any(bytes(state)), "the finalizer releases and zeroes every record"


def test_reflection_json_and_migration_move_dicts(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.reflect import DICT, STR

    module = HostModule(SCOREBOARD, ExternRegistry(), eager=False)
    descriptor = module.runtime.descriptor("Scoreboard")
    fields = {
        descriptor.fields[i].name.decode(): descriptor.fields[i].type.contents for i in range(descriptor.field_count)
    }
    stats = fields["stats"]
    assert (stats.kind, stats.name, stats.key.contents.kind, stats.element.contents.name) == (
        DICT,
        b"dict[str, f32]",
        STR,
        b"f32",
    )
    assert stats.size == 48 and stats.fini

    board = module.namespace.Scoreboard()
    board.score(3, 1)
    board.score(-2, 4)
    board.record(1.0)
    board.place(7, 0.5)
    board.flag(False)
    expected = {
        "scores": {"3": 1, "-2": 4},
        "stats": {"speed": 2.0, "": -1.0},
        "seen": {"7": [0.5, 1.0]},
        "flags": {"false": 1},
        "title": "speed",
    }
    assert module.to_json(board) == expected
    module.from_json(board, {"scores": {"5": 6}, "stats": {"x": 1, "y": 2}, "flags": {"true": 3}})
    assert module.to_json(board)["scores"] == {"5": 6} and board.points(5) == 6
    assert module.to_json(board)["stats"] == {"x": 1.0, "y": 2.0} and board.flag(True) == 4
    with pytest.raises(ValueError, match=r"Scoreboard.scores\['seven'\]"):
        module.from_json(board, {"title": "lost", "scores": {"seven": 1}})
    assert module.to_json(board)["title"] == "speed", "a rejected edit writes nothing"

    reloaded = tmp_path / "scoreboard.py"
    source = SCOREBOARD.read_text(encoding="utf-8")
    reloaded.write_text(
        source.replace("    flags: dict[bool, i32]\n", "    flags: dict[bool, i32]\n    level: i32 = 7\n"),
        encoding="utf-8",
    )
    second = HostModule(reloaded, ExternRegistry(), eager=False)
    moved = second.migrate(board)
    assert second.to_json(moved)["scores"] == {"5": 6} and second.to_json(moved)["level"] == 7
    assert moved.points(5) == 6
    assert module.to_json(board)["scores"] == {} and module.to_json(board)["stats"] == {}


def test_header_spells_dicts_as_records() -> None:
    from vernon_dsl.hostlang.aot import compile_host

    artifacts = compile_host(SCOREBOARD, prefix="sb_")
    assert "VH_DICT(int32_t, int32_t) scores;" in artifacts.header and "VH_STR title;" in artifacts.header
    assert "VH_DICT(SbPlayer, float[2]) seen;" in artifacts.header
    assert artifacts.header.index("} SbPlayer;") < artifacts.header.index("} SbScoreboard;")
    assert "vernonRuntimeHostDictInsert" in artifacts.optimized_ir


_TYPED_CONSUMER = """
#include "scoreboard.h"

float touch(SbScoreboard *board, const VhModule *module) {
    const vh::Allocator allocator(module);
    float total = 0.0f;
    for (auto [player, points] : board->scores) {
        total += static_cast<float>(player + points);
    }
    for (auto [name, weight] : board->stats) {
        total += weight + static_cast<float>(name.size());
    }
    for (auto [who, place] : board->seen) {
        total += place[1] + static_cast<float>(who.bits);
    }
    if (const int32_t *flag = board->flags.find(true)) {
        total += static_cast<float>(*flag);
    }
    const float origin[2] = {0.0f, 0.0f};
    if (board->title != "speed" || !board->stats.assign(allocator, "jump", 1.5f) ||
        !board->seen.assign(allocator, SbPlayer{7}, origin) || !board->title.assign(allocator, "mile")) {
        return -1.0f;
    }
    return board->scores.erase(allocator, 3) ? total : -total;
}
"""


def test_generated_header_compiles_as_c_records_and_typed_cpp(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import compile_host
    from vernon_dsl.hostlang.reflect import include_dir

    found = toolchain()
    if found is None:
        pytest.skip("no C toolchain")
    kind, compiler, environment = found
    (tmp_path / "scoreboard.h").write_text(compile_host(SCOREBOARD, prefix="sb_").header, encoding="utf-8")
    (tmp_path / "records.c").write_text(
        '#include "scoreboard.h"\nuint64_t count(const SbScoreboard *board) { return board->scores.size; }\n',
        encoding="utf-8",
    )
    (tmp_path / "typed.cpp").write_text(_TYPED_CONSUMER, encoding="utf-8")
    runtime_include = Path(__file__).resolve().parents[2] / "source" / "include"
    includes = [str(tmp_path), str(include_dir()), str(runtime_include)]
    for source, standard in (("records.c", "c11"), ("typed.cpp", "c++17")):
        if kind == "msvc":
            command = [compiler, "/nologo", "/W4", "/WX", f"/std:{standard}", "/EHsc", "/c"]
            command += [f"/I{path}" for path in includes] + [str(tmp_path / source), f"/Fo{tmp_path}{os.sep}"]
        else:
            driver = compiler if standard == "c11" else (shutil.which("c++") or compiler)
            command = [driver, f"-std={standard}", "-Wall", "-Werror", "-c"]
            command += [f"-I{path}" for path in includes] + [str(tmp_path / source), "-o", str(tmp_path / "out.o")]
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=environment, check=False)
        assert result.returncode == 0, f"{source}:\n{result.stdout}{result.stderr}"


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "broken.py"
    source.write_text(
        textwrap.dedent(
            """
            import vernon_dsl as vd
            from vernon_dsl import f32, i32


            @vd.host
            class Inner:
                ticks: i32 = 0
            """
        )
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    with pytest.raises(CompileError) as error:
        load_host_module(source)
    return str(error.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            @vd.host
            class Bag:
                values: dict[f32, i32]
            """,
            "dict keys are integers, bools, handles or str",
        ),
        (
            """
            @vd.host
            class Bag:
                values: dict[i32, Inner]
            """,
            "dict values are Host Values without classes",
        ),
        (
            """
            @vd.host
            class Bag:
                values: dict[i32, f32]

                def lookup(self) -> f32:
                    return self.values["one"]
            """,
            "has i32 keys, not str",
        ),
        (
            """
            @vd.host
            class Bag:
                values: dict[i32, f32]

                def fill(self) -> None:
                    self.values = {1: 2.0}
            """,
            "{} only empties a dict field",
        ),
        (
            """
            @vd.host
            class Bag:
                values: dict[i32, f32]

                def keys(self) -> i32:
                    names = self.values.keys()
                    return 0
            """,
            "is iterated by a for loop",
        ),
        (
            """
            @vd.host
            class Bag:
                values: dict[i32, f32]

                def find(self) -> f32:
                    return self.values.get(1)
            """,
            "dict.get takes 2 arguments",
        ),
        (
            """
            @vd.host
            class Bag:
                values: dict[i32, f32]

                def copy(self) -> i32:
                    pair = (self.values, 1)
                    return 0
            """,
            "is a dict field",
        ),
        (
            """
            @vd.host
            class Bag:
                count: i32

                def drop(self) -> None:
                    del self.count
            """,
            "del removes a dict entry",
        ),
    ],
    ids=[
        "float-keys",
        "class-values",
        "wrong-key",
        "dict-display",
        "keys-as-value",
        "get-without-default",
        "dict-copied",
        "del-field",
    ],
)
def test_dict_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)
