"""Host lists and strs: class fields owning memory, in Python mode and in linked AOT objects alike."""

from __future__ import annotations

import ctypes
import gc
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostExecutionError, HostModule, address_of, load_host_module
from vernon_dsl.hostlang.model import HostStatus
from vernon_dsl.hostlang.native import class_layout

PROGRAMS = Path(__file__).parent / "host_programs"
INVENTORY = PROGRAMS / "inventory.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_host_runtime_symbols")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")

# (method, arguments) run in order on one fresh instance; the last items expect a Host status.
_SCRIPT: list[tuple[str, tuple[Any, ...]]] = [
    ("add", (1.5, 2)),
    ("add", (4.0, 1)),
    ("add", (0.5, 3)),
    ("total_weight", ()),
    ("heaviest", ()),
    ("weight_at", (-1,)),
    ("count_at", (-3,)),
    ("flagged", ()),
    ("double_weights", ()),
    ("weight_at", (1,)),
    ("drop_last", ()),
    ("total_weight", ()),
    ("walk", (2.0,)),
    ("rename", (True,)),
    ("same_owner", (True,)),
    ("rename", (False,)),
    ("same_owner", (True,)),
    ("same_owner", (False,)),
    ("empty", ()),
    ("add", (3.0, 1)),
    ("grow_while_iterating", ()),
    ("weight_at", (1,)),
    ("weight_at", (5,)),
    ("empty", ()),
    ("drop_last", ()),
]


def _registry(logged: list[tuple[str, float]]) -> ExternRegistry:
    return ExternRegistry().bind("inventory_log", lambda message, value: logged.append((str(message), float(value))))


def _outcome(function: Any, *args: Any) -> Any:
    try:
        result = function(*args)
    except HostExecutionError as error:
        return ("status", error.status)
    return None if result is None else float(result) if not isinstance(result, bool) else result


def _python_mode() -> tuple[list[Any], list[tuple[str, float]]]:
    logged: list[tuple[str, float]] = []
    module = HostModule(INVENTORY, _registry(logged), eager=False)
    inventory = module.namespace.Inventory()
    return [_outcome(getattr(inventory, name), *args) for name, args in _SCRIPT], logged


@pytest.mark.skipif(toolchain() is None, reason="no C toolchain links Host objects")
def test_containers_run_alike_in_python_mode_and_aot(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    python, python_logged = _python_mode()
    logged: list[tuple[str, float]] = []
    library = LinkedHost(load_programs(INVENTORY), tmp_path, _registry(logged))
    table = library.table
    descriptor = next(
        table.types[index].contents
        for index in range(table.type_count)
        if table.types[index].contents.name == b"Inventory"
    )
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    library.call("Inventory__init", address)
    aot = [_outcome(library.call, f"Inventory_{name}", address, *args) for name, args in _SCRIPT]
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)

    assert aot == python
    assert logged == python_logged == [("a traveller with a long name", 2.0), ("ann", 2.0)]
    assert python[:8] == [1.0, 2.0, 3.0, 8.5, 4.0, 0.5, 2.0, 2.0]
    # Appending while iterating visits the appended item: 3.0, then -3.0.
    assert python[20:23] == [2.0, -3.0, ("status", HostStatus.INDEX_OUT_OF_RANGE)]
    assert python[-1] == ("status", HostStatus.INDEX_OUT_OF_RANGE)
    assert bytes(instance) == bytes(descriptor.size), "fini releases every record and zeroes it"


def test_python_mode_fields_read_as_lists_and_strs() -> None:
    module = HostModule(INVENTORY, _registry([]), eager=False)
    inventory = module.namespace.Inventory()
    assert inventory.owner == "nobody"
    assert len(inventory.items) == 0 and inventory.weights == []
    inventory.add(2.0, 2)
    item = inventory.items[0]
    assert (float(item.weight), int(item.count)) == (2.0, 2)
    inventory.weights = [1, 2, 3]
    inventory.owner = "zoë"
    assert inventory.weights == [1.0, 2.0, 3.0] and inventory.owner == "zoë"
    inventory.weights[-1] = 7
    assert float(inventory.heaviest()) == 7.0
    with pytest.raises(HostExecutionError) as error:
        inventory.weights[3]
    assert error.value.status == HostStatus.INDEX_OUT_OF_RANGE


def test_instances_release_their_containers_when_collected() -> None:
    module = HostModule(INVENTORY, _registry([]), eager=False)
    inventory = module.namespace.Inventory()
    inventory.add(1.0, 1)
    inventory.rename(True)
    state = inventory._vh_state
    assert any(bytes(state))
    del inventory
    gc.collect()
    assert not any(bytes(state)), "the finalizer releases and zeroes every record"


def test_reflection_json_and_migration_move_containers(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.reflect import LIST, STR

    module = HostModule(INVENTORY, _registry([]), eager=False)
    descriptor = module.runtime.descriptor("Inventory")
    fields = {
        descriptor.fields[i].name.decode(): descriptor.fields[i].type.contents for i in range(descriptor.field_count)
    }
    assert (fields["owner"].kind, fields["owner"].name) == (STR, b"str")
    assert (fields["items"].kind, fields["items"].name, fields["items"].element.contents.name) == (
        LIST,
        b"list[Item]",
        b"Item",
    )
    assert fields["items"].fini and descriptor.fini and module.table.allocator

    inventory = module.namespace.Inventory()
    inventory.add(1.5, 2)
    assert module.to_json(inventory) == {
        "owner": "nobody",
        "items": [{"weight": 1.5, "count": 2}],
        "weights": [1.5],
        "flags": [True],
        "path": [],
    }
    module.from_json(inventory, {"owner": "kit", "weights": [3, 4], "path": [[1, 2, 3]]})
    assert module.to_json(inventory)["weights"] == [3.0, 4.0]
    with pytest.raises(ValueError, match=r"Inventory.weights\[1\]"):
        module.from_json(inventory, {"owner": "lost", "weights": [1, "heavy"]})
    assert module.to_json(inventory)["owner"] == "kit", "a rejected edit writes nothing"

    reloaded = tmp_path / "inventory.py"
    source = INVENTORY.read_text(encoding="utf-8")
    reloaded.write_text(
        source.replace("    flags: list[bool]\n", "    level: i32 = 7\n    flags: list[bool]\n"), encoding="utf-8"
    )
    second = HostModule(reloaded, _registry([]), eager=False)
    moved = second.migrate(inventory)
    assert second.to_json(moved) == {
        "owner": "kit",
        "items": [{"weight": 1.5, "count": 2}],
        "weights": [3.0, 4.0],
        "level": 7,
        "flags": [True],
        "path": [[1.0, 2.0, 3.0]],
    }
    assert module.to_json(inventory) == {"owner": "", "items": [], "weights": [], "flags": [], "path": []}


def test_header_spells_containers_as_records_and_plain_modules_need_no_runtime() -> None:
    from vernon_dsl.hostlang.aot import compile_host

    artifacts = compile_host(INVENTORY, prefix="inv_")
    assert "VH_STR owner;" in artifacts.header and "VH_LIST(InvItem) items;" in artifacts.header
    assert "VH_LIST(float[3]) path;" in artifacts.header and "vernon_host/containers.hpp" in artifacts.header
    assert "containers.hpp" not in compile_host(PROGRAMS / "semantics.py", prefix="sem_").header
    assert "vernonRuntimeHostListReserve" in artifacts.optimized_ir
    assert artifacts.reflection["classes"]["Inventory"]["fields"][1]["type"] == "list[Item]"
    plain = compile_host(PROGRAMS / "semantics.py", prefix="sem_")
    assert "vernonRuntimeHostList" not in plain.optimized_ir


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "broken.py"
    source.write_text(
        textwrap.dedent(
            """
            import vernon_dsl as vd
            from vernon_dsl import f32, i32


            @vd.struct(shared=True)
            class Point:
                x: f32


            @vd.host
            class Inner:
                ticks: i32 = 0


            @vd.extern(symbol="take")
            def take(value: f32) -> None: ...
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
            @vd.struct(shared=True)
            class Named:
                name: str
            """,
            "only class instances own memory",
        ),
        (
            """
            @vd.host
            def run(values: list[f32]) -> i32:
                return 0
            """,
            "lives only in a class field",
        ),
        (
            """
            @vd.host
            class Bag:
                values: list[f32]

                def copy(self) -> i32:
                    pair = (self.values, 1)
                    return 0
            """,
            "is a list field",
        ),
        (
            """
            @vd.host
            class Bag:
                inner: list[Inner]
            """,
            "list elements are Host Values without classes",
        ),
        (
            """
            @vd.host
            class Bag:
                nested: list[list[f32]]
            """,
            "list elements are Host Values without classes",
        ),
        (
            """
            @vd.host
            class Bag:
                name: str

                def check(self) -> bool:
                    return self.name < "b"
            """,
            "compare only with == and !=",
        ),
        (
            """
            @vd.host
            class Bag:
                values: list[f32]

                def fill(self) -> None:
                    self.values = [1.0]
            """,
            "[] only empties a list field",
        ),
        (
            """
            @vd.host
            class Bag:
                values: list[f32]

                def sort(self) -> None:
                    self.values.sort()
            """,
            "list has no method 'sort'",
        ),
        (
            """
            @vd.host
            class Bag:
                values: list[f32]

                def add(self) -> None:
                    self.values.append(Point(1.0))
            """,
            "cannot append Point to list[f32]",
        ),
        (
            """
            @vd.host
            def run() -> i32:
                take("text")
                return 0
            """,
            "cannot pass str as f32",
        ),
    ],
    ids=[
        "str-in-struct",
        "list-parameter",
        "list-copied-into-local",
        "class-elements",
        "nested-lists",
        "str-ordering",
        "list-literal",
        "unknown-method",
        "wrong-element",
        "literal-to-scalar-extern",
    ],
)
def test_container_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)


def test_c_callers_free_python_mode_instances_through_the_table() -> None:
    module = HostModule(INVENTORY, _registry([]), eager=False)
    descriptor = module.runtime.descriptor("Inventory")
    instance = ctypes.create_string_buffer(descriptor.size)
    address = ctypes.addressof(instance)
    ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)(descriptor.init)(address)
    built = class_layout(module.namespace.Inventory)
    view = built.view(built.ctype.from_address(address))
    view.add(2.0, 1)
    assert address_of(view) == address and view.owner == "nobody"
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(address)
    assert not any(bytes(instance)) and module.take_error() is None
