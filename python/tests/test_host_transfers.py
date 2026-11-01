"""Externs taking spans and returning owned strs, lists and dicts: Python bindings and C implementations, called
from Python mode and from compiled code, agree, and compiled code releases every result."""

from __future__ import annotations

import ctypes
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, build_c_library, toolchain
from test_host_strs import _CountingAllocator
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, load_host_module

PROGRAMS = Path(__file__).parent / "host_programs"
TRANSFERS = PROGRAMS / "transfers.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")
needs_toolchain = pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")

_SCRIPT: list[tuple[str, tuple[Any, ...]]] = [
    ("fill", (5,)),
    ("weigh", ()),
    ("measured", ()),
    ("local_spans", (4,)),
    ("rename", (12,)),
    ("rename_failing", (3,)),
    ("label_length", ()),
    ("rename", (5,)),
    ("tabulate", (5,)),
    ("tabulate", (0,)),
    ("fill", (0,)),
    ("weigh", ()),
    ("repeated", (6,)),
]


def _python_registry(calls: list[Any]) -> ExternRegistry:
    """Bindings computing what transfers_impl.c computes, recording what each receives."""

    def total(values: list[Any], weights: dict[Any, Any]) -> float:
        calls.append(("total", list(values), dict(weights)))
        return float(sum(values)) + sum(value * (index + 1) for index, value in enumerate(weights.values()))

    def samples(items: list[Any]) -> int:
        calls.append(("samples", [dict(item) for item in items]))
        return sum(item["count"] * (index + 1) for index, item in enumerate(items))

    def name(id: int) -> str:
        if id < 0:
            raise HostExecutionError(7, "no name")
        return f"item-{id}"

    def make_range(count: int) -> list[int]:
        return [index * 3 for index in range(count)]

    def table(count: int) -> dict[int, dict[str, Any]]:
        return {index % 3: {"weight": float(index), "count": index + 1} for index in range(count)}

    return (
        ExternRegistry()
        .bind("transfers_total", total)
        .bind("transfers_samples", samples)
        .bind("transfers_name", name)
        .bind("transfers_range", make_range)
        .bind("transfers_table", table)
    )


def _c_registry(directory: Path) -> ExternRegistry:
    library = build_c_library(PROGRAMS / "transfers_impl.c", directory, "transfers_impl")
    return ExternRegistry().bind_library(library)


def _outcome(call: Any) -> Any:
    try:
        result = call()
    except HostExecutionError as error:
        return ("failed", error.status)
    return None if result is None else round(float(result), 4)


def _python_mode(registry: ExternRegistry) -> list[Any]:
    module = HostModule(TRANSFERS, registry, eager=False)
    inventory = module.namespace.Inventory()
    return [_outcome(lambda name=name, args=args: getattr(inventory, name)(*args)) for name, args in _SCRIPT]


def _aot(registry: ExternRegistry, directory: Path) -> tuple[list[Any], _CountingAllocator]:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(TRANSFERS), directory, registry)
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents
        for index in range(table.type_count)
        if table.types[index].contents.name == b"Inventory"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Inventory__init", address)
    results = [
        _outcome(lambda name=name, args=args: library.call(f"Inventory_{name}", address, *args))
        for name, args in _SCRIPT
    ]
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None
    assert library.errors == []
    return results, counting


def test_python_bindings_receive_spans_and_return_owned_results() -> None:
    calls: list[Any] = []
    results = _python_mode(_python_registry(calls))

    # fill(5) holds ids 0, 3, 6, 9, 12 and their halves, without the first id's.
    assert calls[0] == ("total", [0, 3, 6, 9, 12], {3: 1.5, 6: 3.0, 9: 4.5, 12: 6.0})
    assert results[:3] == [5.0, 30.0 + 1.5 + 6.0 + 13.5 + 24.0, 75.0 + 5.0]
    assert calls[2] == ("total", [0, 3, 6, 9, 100], {7: 2.0})
    assert results[4:8] == [7.0, ("failed", 7), 7.0, 6.0]
    # tabulate(5): keys 0, 1, 2 keep entries 3, 4, 2, in first-insertion order.
    assert calls[3] == (
        "samples",
        [{"weight": 3.0, "count": 4}, {"weight": 4.0, "count": 5}, {"weight": 2.0, "count": 3}],
    )
    # fill(0) empties the ids but keeps the weights fill(5) set.
    assert results[8:] == [4 + 10 + 9 + 3, 0.0, 0.0, 45.0, sum(len(f"item-{n}") + n for n in range(6))]


@needs_toolchain
def test_python_mode_calls_c_implementations(tmp_path: Path) -> None:
    assert _python_mode(_c_registry(tmp_path)) == _python_mode(_python_registry([]))


@needs_toolchain
@pytest.mark.parametrize("implementation", ["python", "c"])
def test_compiled_code_agrees_and_releases_every_result(tmp_path: Path, implementation: str) -> None:
    registry = _python_registry([]) if implementation == "python" else _c_registry(tmp_path / "impl")
    results, counting = _aot(registry, tmp_path / "host")

    assert results == _python_mode(_python_registry([]))
    assert counting.allocations > 10
    assert counting.live == {}, "every owned result, span and failed call's record is released"
    assert counting.mismatched == []


@needs_toolchain
def test_compiled_code_without_an_allocator_uses_the_runtime_default(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(TRANSFERS), tmp_path / "host", _c_registry(tmp_path / "impl"))
    descriptor = next(
        library.table.types[index].contents
        for index in range(library.table.type_count)
        if library.table.types[index].contents.name == b"Inventory"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Inventory__init", address)
    assert library.call("Inventory_fill", address, 5) == 5
    assert library.call("Inventory_rename", address, 12) == 7
    assert library.call("Inventory_tabulate", address, 5) == 26
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "bad_transfers.py"
    source.write_text(
        "import vernon_dsl as vd\nfrom vernon_dsl import f32, i32\n\n\n"
        "@vd.extern(symbol='bad_sum', effects='read')\n"
        "def sum_of(values: list[i32], scale: i32) -> i32: ...\n\n\n"
        "@vd.extern(symbol='bad_name', effects='read', abi='status')\n"
        "def name(id: i32) -> str: ...\n\n\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    with pytest.raises(CompileError) as caught:
        load_host_module(source)
    return str(caught.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            @vd.extern(symbol='bad_direct')
            def direct() -> list[i32]: ...
            """,
            "only with abi='status'",
        ),
        (
            """
            @vd.extern(symbol='bad_names')
            def names(values: list[str]) -> None: ...
            """,
            "plain values",
        ),
        (
            """
            @vd.extern(symbol='bad_nested', abi='status')
            def nested() -> dict[i32, list[i32]]: ...
            """,
            "plain values",
        ),
        (
            """
            @vd.host
            class Box:
                values: list[i32]

                def bump(self) -> i32:
                    self.values.append(1)
                    return 1

                def run(self) -> i32:
                    return sum_of(self.values, self.bump())
            """,
            "no later argument may call",
        ),
        (
            """
            @vd.host(effects="read")
            def run() -> i32:
                return len(name(1))
            """,
            "received only as the value of an assignment",
        ),
    ],
)
def test_transfer_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)
