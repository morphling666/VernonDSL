"""Host classes holding Host classes: nested owned memory runs alike in Python mode and AOT and is released by fini."""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest
from host_aot import LinkedHost, toolchain
from test_host_strs import _CountingAllocator
from vernon_dsl.hostlang import ExternRegistry, HostModule

COMPOSITION = Path(__file__).parent / "host_programs" / "composition.py"

# fill(4): lefts 0..3, rights 0..30, shelf 100..103. Growing the grid on the third add zeroes it,
# so cell 2 holds the third value and the first two cells read 0.
_SCRIPT: list[tuple[str, tuple[int, ...], float]] = [
    ("fill", (4,), 472),
    ("direct", (), 77),
    ("text", (), 707),
    ("cells", (), 28.0),
    ("reset", (), 4),
]


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


def test_python_mode_composes_classes() -> None:
    owner = HostModule(COMPOSITION, ExternRegistry(), eager=False).namespace.Owner()
    assert [getattr(owner, name)(*arguments) for name, arguments, _ in _SCRIPT] == [
        expected for _, _, expected in _SCRIPT
    ]
    assert owner.shelf.bag.label == "bag++++" and list(owner.shelf.bag.items) == [100, 101, 102, 103]


@pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")
@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_nested_owned_memory_runs_alike_and_fini_releases_it(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    library = LinkedHost(load_programs(COMPOSITION), tmp_path, ExternRegistry())
    table = library.table
    counting = _CountingAllocator()
    table.allocator[0] = ctypes.addressof(counting.record)
    descriptor = next(
        table.types[index].contents for index in range(table.type_count) if table.types[index].contents.name == b"Owner"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Owner__init", address)
    outcomes = [library.call(f"Owner_{name}", address, *arguments) for name, arguments, _ in _SCRIPT]
    live_before_fini = len(counting.live)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    table.allocator[0] = None

    assert outcomes == [expected for _, _, expected in _SCRIPT]
    assert live_before_fini > 6, "each nested Bag holds records of its own"
    assert counting.live == {}, "fini releases owned memory through every nested class"
    assert counting.mismatched == []
