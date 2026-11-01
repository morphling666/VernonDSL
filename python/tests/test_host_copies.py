"""Explicit copies: `.copy()` and `copy.deepcopy` run alike in Python mode and AOT, leave the copy independent of its
source, release everything they allocate, and are rejected where Python would share or copy nothing."""

from __future__ import annotations

import copy
import ctypes
import gc
import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from test_host_class_values import _native_available
from test_host_strs import _CountingAllocator
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, load_host_module

COPIES = Path(__file__).parent / "host_programs" / "copies.py"

# fill(3) makes parts of weights 1..3 tagged [0], [10], [20]; each later step copies, changes the copy and reads both.
_SCRIPT: list[tuple[str, tuple[int, ...], float]] = [
    ("fill", (3,), 3),
    ("copy_values", (), 4332),
    ("copy_parts", (), 2161),
    ("copy_spare", (), 426),
    ("copy_indexed", (), 341),
    ("copy_heights", (), 16.5),
    ("recopy", (), 10),
    ("clone_first", (), 571),
]


def test_python_mode_copies_independently_and_releases_every_copy() -> None:
    module = HostModule(COPIES, ExternRegistry(), eager=False)
    counting = _CountingAllocator()
    module.runtime.allocator.value = ctypes.addressof(counting.record)
    shelf = module.namespace.Shelf()
    outcomes = [getattr(shelf, name)(*arguments) for name, arguments, _ in _SCRIPT]
    twin = copy.deepcopy(shelf)
    twin.parts[0].tags.append(1)
    twin.names["z"] = 26
    twin.heights[0] = 1.0
    assert len(shelf.parts[0].tags) == 2 and "z" not in shelf.names and shelf.heights[0] == 10.0
    assert len(twin.parts[0].tags) == 3 and twin.indexed[4].weight == 51 and twin.spare.name == "copied"
    del shelf, twin
    gc.collect()
    module.runtime.allocator.value = None

    assert outcomes == [expected for _, _, expected in _SCRIPT]
    assert counting.live == {}, "every copy is released"
    assert counting.mismatched == []


@pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")
@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_copies_run_alike_and_release_every_copy(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(COPIES), tmp_path, ExternRegistry())
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Shelf"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Shelf__init", address)
    outcomes = [library.call(f"Shelf_{name}", address, *arguments) for name, arguments, _ in _SCRIPT]
    live_before_fini = len(counting.live)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert outcomes == [expected for _, _, expected in _SCRIPT]
    assert live_before_fini > 8, "the parts, their tags and names, the dicts and the storage own memory"
    assert counting.live == {}, "fini releases every copy"
    assert counting.mismatched == []


class _FailingAllocator(_CountingAllocator):
    """Refuses every allocation once `limit` have been made."""

    def __init__(self) -> None:
        super().__init__()
        self.limit: int | None = None

    def _allocate(self, user: Any, size: int, alignment: int) -> int:
        if self.limit is not None and self.allocations >= self.limit:
            return 0
        return super()._allocate(user, size, alignment)


# recopy copies the parts, counts and spare back unchanged, so a failure at any allocation leaves the shelf reading
# as before, whichever assignment it hit.
_FAILURE_POINTS = range(1, 24)


def _shelf_contents(shelf: Any) -> Any:
    parts = [(part.weight, list(part.tags), part.name) for part in shelf.parts]
    return parts, list(shelf.counts), list(shelf.spare.tags)


def test_python_mode_failed_copies_leave_fields_as_they_were() -> None:
    module = HostModule(COPIES, ExternRegistry(), eager=False)
    failing = _FailingAllocator()
    module.runtime.allocator.value = ctypes.addressof(failing.record)
    shelf = module.namespace.Shelf()
    for name, arguments, _ in _SCRIPT[:-2]:
        getattr(shelf, name)(*arguments)
    before = _shelf_contents(shelf)
    failures = 0
    for point in _FAILURE_POINTS:
        failing.limit = failing.allocations + point
        try:
            shelf.recopy()
        except HostExecutionError:
            failures += 1
        failing.limit = None
        gc.collect()
        assert _shelf_contents(shelf) == before
    outcomes = [shelf.recopy(), shelf.clone_first()]
    del shelf
    gc.collect()
    module.runtime.allocator.value = None

    assert failures > 10 and outcomes == [10, 571]
    assert failing.live == {} and failing.mismatched == []


@pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")
@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_failed_copies_release_everything_in_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(COPIES), tmp_path, ExternRegistry())
    table = library.table
    failing = _FailingAllocator()
    table.allocator[0] = ctypes.addressof(failing.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Shelf"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Shelf__init", address)
    for name, arguments, _ in _SCRIPT[:-2]:
        library.call(f"Shelf_{name}", address, *arguments)
    failures = 0
    for point in _FAILURE_POINTS:
        failing.limit = failing.allocations + point
        try:
            library.call("Shelf_recopy", address)
        except HostExecutionError:
            failures += 1
        failing.limit = None
    outcomes = [library.call("Shelf_recopy", address), library.call("Shelf_clone_first", address)]
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert failures > 10 and outcomes == [10, 571], "a failed copy leaves every field as it was"
    assert failing.live == {} and failing.mismatched == []


_PRELUDE = """
from __future__ import annotations

import copy

import vernon_dsl as vd
from vernon_dsl import i32


@vd.host
class Part:
    tags: list[i32]


@vd.host
class Shelf:
    parts: list[Part]
    counts: list[i32]
    weight: i32 = 0

"""


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (
            """
            def f(self) -> None:
                parts = self.parts.copy()
            """,
            "self.parts.copy() would share its instances in Python; copy.deepcopy(self.parts) copies them",
        ),
        (
            """
            def f(self) -> i32:
                weight = copy.deepcopy(self.weight)
                return weight
            """,
            "is a value, which needs no copy",
        ),
        (
            """
            def f(self) -> i32:
                return len(self.counts.copy())
            """,
            "copies a container only as the value of a list or dict local or field",
        ),
        (
            """
            def f(self) -> None:
                counts = self.counts
            """,
            "so copy it explicitly",
        ),
    ],
    ids=["shallow-copy-of-classes", "copy-of-a-value", "copy-outside-an-assignment", "shared-container"],
)
def test_copy_diagnostics(tmp_path: Path, method: str, message: str) -> None:
    source = tmp_path / "broken.py"
    source.write_text(_PRELUDE + textwrap.indent(textwrap.dedent(method), "    "), encoding="utf-8")
    with pytest.raises(CompileError) as error:
        load_host_module(source)
    assert message in str(error.value)
