"""Host TensorStorage fields and locals: Python mode runs over real `vd.TensorStorage` values, and compiled code
indexes, creates and copies alike."""

from __future__ import annotations

import ctypes
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import vernon_dsl as vd
from host_aot import LinkedHost, toolchain
from test_host_strs import _CountingAllocator
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, HostStatus, load_host_module

PROGRAMS = Path(__file__).parent / "host_programs"
STORAGES = PROGRAMS / "storages.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")

_TILE = np.arange(6, dtype=np.int32).reshape(2, 3)
_SCRIPT: list[tuple[str, tuple[Any, ...]]] = [
    ("corner", ()),
    ("resize", (3, 4)),
    ("raise_cell", (1, 2, 2.5)),
    ("raise_cell", (1, 2, 1.0)),
    ("raise_cell", (-1, -1, 4.0)),
    ("raise_cell", (0, -4, 0.25)),
    ("raise_cell", (3, 0, 1.0)),
    ("raise_cell", (0, -5, 1.0)),
    ("corner", ()),
    ("shape_sum", ()),
    ("load_tile", (_TILE,)),
    ("read_tile", ()),
    ("blurred_total", ()),
    ("weighted", ()),
    ("load_heights", (np.ones((2, 2), np.float32),)),
    ("out_of_range", (3,)),
    ("out_of_range", (-3,)),
    ("resize", (2, 2)),
    ("load_heights", (np.full((2, 2), 3.0, np.float32),)),
    ("blurred_total", ()),
    ("resize", (-1, 2)),
    ("shape_sum", ()),
    ("histogram_total", ()),
    ("adopt_heights", (np.arange(6, dtype=np.float32).reshape(3, 2),)),
    ("corner", ()),
    ("slab", (1,)),
    ("slab", (-3,)),
    ("slab", (2,)),
    ("tile_block", (1,)),
    ("tile_block", (2,)),
    ("read_tile", ()),
]

# What Python raises where compiled code returns a status.
_PYTHON_FAILURES = {
    IndexError: HostStatus.INDEX_OUT_OF_RANGE,
    ValueError: HostStatus.INVALID_SHAPE,
}


def _registry(logged: list[int]) -> ExternRegistry:
    return ExternRegistry().bind("storages_log", lambda value: logged.append(int(value)))


def _outcome(call: Any) -> Any:
    try:
        result = call()
    except HostExecutionError as error:
        return ("status", error.status)
    except (IndexError, ValueError) as error:
        return ("status", _PYTHON_FAILURES[type(error)])
    if result is None:
        return None
    array = np.asarray(result)
    return array.tolist() if array.ndim else array.item()


def _python_mode() -> tuple[list[Any], list[int], Any]:
    logged: list[int] = []
    module = HostModule(STORAGES, _registry(logged), eager=False)
    grid = module.namespace.Grid()
    return [_outcome(lambda name=name, args=args: getattr(grid, name)(*args)) for name, args in _SCRIPT], logged, grid


def test_python_mode_fields_are_tensor_storages_over_the_instance() -> None:
    results, logged, grid = _python_mode()

    assert results[:10] == [
        ("status", HostStatus.INDEX_OUT_OF_RANGE),
        3,
        2.5,
        3.5,
        4.0,
        0.25,
        ("status", HostStatus.INDEX_OUT_OF_RANGE),
        ("status", HostStatus.INDEX_OUT_OF_RANGE),
        4.0,
        308,
    ]
    assert results[10:12] == [15, [[7, 1, 2], [3, 4, 5]]]
    assert results[12:15] == [3.875, 3.75, ("status", HostStatus.INVALID_SHAPE)]
    assert results[15:17] == [("status", HostStatus.INDEX_OUT_OF_RANGE), 0.25]
    assert results[18:20] == [2, 6.0]
    assert results[20] == ("status", HostStatus.INVALID_SHAPE), "a negative extent fails, keeping the old storage"
    assert results[21:25] == [204, 0, 37.0, 5.0]
    assert results[25:28] == [39.0, 34.5, ("status", HostStatus.INDEX_OUT_OF_RANGE)], "a window is checked, not clamped"
    assert results[28:] == [-98, ("status", HostStatus.INDEX_OUT_OF_RANGE), [[7, 14, 2], [-1, -1, -1]]]
    assert logged == [3, 2]

    heights = grid.heights
    assert isinstance(heights, vd.TensorStorage) and heights is grid.heights
    heights[2, 1] = 9.0
    assert grid.corner() == 9.0 and grid.heights[2, 1] == 9.0, "Host code and Python share the field's memory"
    grid.resize(1, 1)
    assert heights.shape == (3, 2) and heights[2, 1] == 9.0, "a replaced storage keeps its elements"
    assert grid.tile.to_values().tolist() == [[7, 14, 2], [-1, -1, -1]]


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_storages_run_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    python, python_logged, _ = _python_mode()
    logged: list[int] = []
    library = LinkedHost(load_programs(STORAGES), tmp_path, _registry(logged))
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Grid"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Grid__init", address)
    aot = [
        _outcome(lambda name=name, args=args: library.call(f"Grid_{name}", address, *args)) for name, args in _SCRIPT
    ]
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert library.errors == []
    assert logged == python_logged
    assert aot == python
    # The init's tile and weights, two per successful resize, the three locals and from_values; a failed creation
    # takes none.
    assert counting.allocations == 10
    assert counting.live == {}, "every storage field and local is released, on failures too"
    assert counting.mismatched == []


def test_storages_read_and_write_as_nested_lists_and_migrate(tmp_path: Path) -> None:
    module = HostModule(STORAGES, _registry([]), eager=False)
    grid = module.namespace.Grid()
    grid.resize(2, 3)
    grid.raise_cell(1, 2, 1.5)
    value = module.to_json(grid)
    assert value["heights"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]
    assert value["tile"] == [[0, 0, 0], [0, 0, 0]] and value["histogram"] == [0, 0, 1]

    held = grid.heights
    module.from_json(grid, {"heights": [[1.0], [2.0], [3.0], [4.0]], "histogram": []})
    assert grid.heights.shape == (4, 1) and grid.corner() == 4.0 and len(grid.histogram) == 0
    assert held.to_values().tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], "a replaced storage keeps its elements"
    with pytest.raises(ValueError, match="axis 1"):
        module.from_json(grid, {"cells": 9, "tile": [[1, 2], [3, 4]]})
    with pytest.raises(ValueError, match="items"):
        module.from_json(grid, {"heights": [[1.0, 2.0], [3.0]]})
    assert module.to_json(grid)["cells"] == 6, "a rejected edit writes nothing"
    module.from_json(grid, {"heights": [[]] * 2})
    assert module.to_json(grid)["heights"] == [[], []] and grid.heights.shape == (2, 0)

    reloaded = tmp_path / "storages.py"
    reloaded.write_text(
        STORAGES.read_text(encoding="utf-8").replace(
            "    cells: i32 = 0\n", "    cells: i32 = 0\n    level: i32 = 7\n"
        ),
        encoding="utf-8",
    )
    module.from_json(grid, {"heights": [[5.0, 6.0]]})
    second = HostModule(reloaded, _registry([]), eager=False)
    moved = second.migrate(grid)
    assert second.to_json(moved)["heights"] == [[5.0, 6.0]] and moved.corner() == 6.0
    assert second.to_json(moved)["level"] == 7
    assert module.to_json(grid)["heights"] == [] and grid.heights.shape == (0, 0), "the source's storage moved"


def test_header_spells_storages_as_records() -> None:
    from vernon_dsl.hostlang.aot import compile_host

    artifacts = compile_host(STORAGES, prefix="st_")
    assert "VH_STORAGE(float, 2) heights;" in artifacts.header
    assert "VH_STORAGE(int32_t, 2) tile;" in artifacts.header
    assert "VH_STORAGE(uint32_t, 1) histogram;" in artifacts.header
    assert "vernon_host/containers.hpp" in artifacts.header


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "bad_storages.py"
    source.write_text(
        "import vernon_dsl as vd\n"
        "from vernon_dsl import Tensor, TensorStorage, dyn, f16, f32, i32, i64\n\n\n"
        "@vd.host\n"
        "class Bad:\n"
        "    a: TensorStorage[f32, (dyn, 3)]\n"
        "    b: TensorStorage[f32, (2, 3)]\n\n" + textwrap.indent(textwrap.dedent(body), "    "),
        encoding="utf-8",
    )
    with pytest.raises(CompileError) as raised:
        load_host_module(source)
    return str(raised.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("def f(self) -> None:\n    self.a = self.b\n", "assign a storage only from a creation"),
        ("def f(self, n: i32) -> None:\n    s = vd.zeros(dtype=f32, shape=(n,))\n    t = s\n", "is a TensorStorage"),
        ("def f(self) -> i32:\n    return len(vd.zeros(dtype=f32, shape=(2,)))\n", "creates a TensorStorage only"),
        ("def f(self) -> f32:\n    return self.a[:, 0][0]\n", "a dyn axis is selected by an index or a window"),
        ("def f(self, n: i32) -> f32:\n    return self.a[0:n, 0][0]\n", "or the slice is a window i:i+n"),
        ("def f(self) -> f32:\n    return self.b[0, 3:1][0]\n", "selects nothing"),
        ("def f(self) -> f32:\n    return self.b[0, 1:5][0]\n", "does not fit an axis of extent 3"),
        ("def f(self) -> None:\n    self.b[0, 0:2] = self.b[0, 0:3]\n", "into a selection of Tensor[f32, (2,)]"),
        ("def f(self) -> None:\n    x = self.a.size\n", "no attribute 'size'"),
        ("def f(self) -> None:\n    x = self.a.to_values()\n", "dyn extents"),
        ("def f(self, t: Tensor[i32, 2, 3]) -> None:\n    self.b.copy_from_values(t)\n", "element type and shape"),
        ("def f(self, n: i32) -> None:\n    self.b = vd.zeros(dtype=f32, shape=(n, 3))\n", "cannot store"),
        ("def f(self) -> None:\n    s = vd.zeros(dtype=f32, shape=(-1,))\n", "cannot be negative"),
        ("def f(self) -> None:\n    s = vd.zeros(shape=(2,))\n", "dtype= and shape="),
        ("def f(self) -> None:\n    s = vd.zeros(dtype=i64, shape=(2,))\n", "bool, i32, u32, f32 or f64"),
        ("c: TensorStorage[f16, (2,)]\n", "bool, i32, u32, f32 or f64"),
        ("c: TensorStorage[f32, ()]\n", "element type and a shape"),
        ("def f(self) -> None:\n    for x in self.a:\n        pass\n", "for loops"),
        ("def f(self) -> None:\n    del self.a[0, 0]\n", "del removes"),
        ("def f(self) -> None:\n    self.a += self.a\n", "not supported"),
        ("def f(self, s: TensorStorage[f32, (2,)]) -> None:\n    pass\n", "it is not passed"),
    ],
)
def test_storage_misuse_is_reported(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)


def test_helpers_and_structs_hold_no_storages(tmp_path: Path) -> None:
    source = tmp_path / "bad_helpers.py"
    source.write_text(
        "import vernon_dsl as vd\nfrom vernon_dsl import TensorStorage, f32, i32\n\n\n"
        "@vd.func(shared=True)\ndef helper(n: i32) -> i32:\n    s = vd.zeros(dtype=f32, shape=(2,))\n    return n\n\n\n"
        "@vd.host\ndef run(n: i32) -> i32:\n    return helper(n)\n",
        encoding="utf-8",
    )
    with pytest.raises(CompileError, match="cannot use TensorStorage"):
        load_host_module(source)
    source.write_text(
        "import vernon_dsl as vd\nfrom vernon_dsl import TensorStorage, f32\n\n\n"
        "@vd.struct(shared=True)\nclass Pair:\n    s: TensorStorage[f32, (2,)]\n",
        encoding="utf-8",
    )
    with pytest.raises(CompileError, match="only class instances own memory"):
        load_host_module(source)
