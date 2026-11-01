"""Host classes, reflection, generic externs, and state migration, in Python mode and in AOT objects."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import vernon_dsl as vd
from host_aot import toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, address_of, load_host_module

PROGRAMS = Path(__file__).parent / "host_programs"
SYSTEMS = PROGRAMS / "systems.py"
sys.path.insert(0, str(PROGRAMS))


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_compile_host")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")


class Store:
    """Python bindings of the demo externs: one transform, and a generic Struct store keyed by type name."""

    def __init__(self) -> None:
        self.transform: dict[str, Any] = {"position": (0.0, 0.0, 0.0), "scale": 1.0}
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        self.calls: list[tuple[str, int, str]] = []

    def registry(self) -> ExternRegistry:
        registry = ExternRegistry()
        registry.bind("demo_get_transform", lambda entity: dict(self.transform))
        registry.bind("demo_set_transform", lambda entity, value: self.transform.update(value))
        registry.bind("demo_read", self.read)
        registry.bind("demo_write", self.write)
        return registry

    def read(self, entity: int, type_: Any) -> dict[str, Any]:
        self.calls.append(("read", entity, type_.name.decode()))
        return self.rows[(type_.name.decode(), entity)]

    def write(self, entity: int, type_: Any, value: dict[str, Any]) -> None:
        self.calls.append(("write", entity, type_.name.decode()))
        self.rows[(type_.name.decode(), entity)] = value


@pytest.fixture()
def store() -> Store:
    return Store()


@pytest.fixture()
def systems(store: Store) -> Any:
    from vernon_dsl.hostlang import HostModule

    return HostModule.load(SYSTEMS, store.registry())


def _fields(descriptor: Any) -> dict[str, Any]:
    return {descriptor.fields[index].name.decode(): descriptor.fields[index] for index in range(descriptor.field_count)}


def _meta(entry: Any) -> dict[str, Any]:
    return {entry.meta[index].key.decode(): json.loads(entry.meta[index].value) for index in range(entry.meta_count)}


def test_instances_start_from_field_defaults(systems: Any) -> None:
    mover = systems.new("Mover")

    assert mover.speed == 2.0 and type(mover.speed) is np.float32
    assert (tuple(mover.rest.position), mover.rest.scale) == ((0.0, 1.0, 0.0), 1.0)
    assert tuple(mover.history) == (0.0, 0.0, 0.0, 0.0)
    assert mover.updates == 0
    assert mover.target == type(mover.target)() and mover.target.bits == 0
    assert mover.timer.period == 0.5
    assert mover.timer.fired == 0


def test_methods_mutate_the_instance_in_place(systems: Any, store: Store) -> None:
    mover = systems.new("Mover")
    mover.target = 5
    mover.speed = 3.0

    results = [mover.update(0.3) for _ in range(4)]

    assert results == [1, 2, 3, 4]
    # The timer fires at 0.6 and 1.2 of accumulated time; `_record` writes the slot of the count after firing.
    assert mover.timer.fired == 2
    assert tuple(mover.history) == (0.0, 3.0, 3.0, 0.0)
    assert store.transform["position"][1] == pytest.approx(3.6, rel=1e-6)
    assert mover.height_after(2.0) == pytest.approx(7.0)


def test_tensor_and_struct_fields_read_as_values(systems: Any) -> None:
    mover = systems.new("Mover")
    history, rest = mover.history, mover.rest
    with pytest.raises(ValueError, match="read-only"):
        mover.history[2] = 4.0
    mover.history = mover.history.at[2].set(4.0)
    mover.rest = vd.replace(rest, position=rest.position * 2.0, scale=3.0)

    assert tuple(mover.history) == (0.0, 0.0, 4.0, 0.0)
    assert tuple(history) == (0.0, 0.0, 0.0, 0.0)
    assert (mover.rest.position.y, mover.rest.scale) == (2.0, 3.0)
    assert rest.scale == 1.0


def test_imported_classes_run_as_plain_python() -> None:
    import systems as imported

    timer = imported.Timer()
    assert [timer.tick(0.3) for _ in range(4)] == [False, True, False, True]
    assert (timer.fired, timer.elapsed) == (2, pytest.approx(0.2))


def test_nested_class_fields_are_views_of_the_parent(systems: Any) -> None:
    mover = systems.new("Mover")
    mover.timer.period = 0.1
    mover.update(0.25)

    assert mover.timer.fired == 1
    assert mover.timer.elapsed == pytest.approx(0.15)
    with pytest.raises(TypeError, match="takes a new Timer instance"):
        mover.timer = None


def test_free_functions_export_beside_classes(systems: Any) -> None:
    assert systems.call("scaled", 1.5, 3) == 4.5


def test_reflection_describes_classes(systems: Any) -> None:
    from vernon_dsl.hostlang.reflect import CLASS, STRUCT, layout_hash

    table = systems.table
    assert table.version == 3
    assert table.name == b"systems"
    assert [table.types[index].contents.name.decode() for index in range(table.type_count)] == [
        "Transform",
        "Timer",
        "Entity",
        "Mover",
        "Health",
        "Regen",
    ]
    mover = systems.type("Mover")
    assert mover.kind == CLASS
    assert (mover.size, mover.alignment) == (80, 8)
    assert mover.doc.decode().startswith("Raises its target")
    assert _meta(mover) == {"role": "system", "order": 10, "tags": ["movement", "demo"]}
    fields = _fields(mover)
    assert list(fields) == ["target", "speed", "rest", "history", "timer", "updates"]
    assert [fields[name].offset for name in fields] == [0, 8, 12, 28, 48, 72]
    assert fields["speed"].doc == b"How fast the target rises, in meters per second."
    assert _meta(fields["speed"]) == {"min": 0.0, "max": 10.0, "step": -0.5}
    assert fields["history"].type.contents.shape[0] == 4
    assert fields["rest"].type.contents.kind == STRUCT
    assert fields["timer"].type.contents.name == b"Timer"
    assert mover.layout_hash == layout_hash(_struct("Mover"), [systems.program])
    methods = {mover.methods[index].name.decode(): mover.methods[index] for index in range(mover.method_count)}
    assert list(methods) == ["update", "_record", "height_after"]
    assert methods["update"].symbol == b"vernon_host_Mover_update"
    assert methods["_record"].symbol is None
    assert (methods["update"].effect, methods["height_after"].effect) == (2, 0)
    assert methods["update"].param_count == 1
    assert methods["update"].params[0].name == b"dt"
    assert methods["update"].result.contents.name == b"i32"
    assert mover.init is not None
    assert table.function_count == 1
    assert table.functions[0].name == b"scaled"
    assert mover.fini is None


def _calls(function: Any) -> list[tuple[str, int, str | None]]:
    return [
        (
            function.calls[index].symbol.decode(),
            function.calls[index].effect,
            function.calls[index].type.contents.name.decode() if function.calls[index].type else None,
        )
        for index in range(function.call_count)
    ]


def test_reflection_lists_reachable_extern_calls(systems: Any) -> None:
    from vernon_dsl.hostlang.reflect import VhCall, VhField, VhFunction, VhModule, VhType

    regen = systems.type("Regen")
    assert _calls(regen.methods[0]) == [
        ("demo_read", 1, "Health"),
        ("demo_read", 1, "Transform"),
        ("demo_write", 2, "Health"),
    ]
    mover = systems.type("Mover")
    methods = {mover.methods[index].name.decode(): mover.methods[index] for index in range(mover.method_count)}
    # `update` reaches its externs directly; `_record` and `height_after` call none.
    assert _calls(methods["update"]) == [("demo_get_transform", 1, None), ("demo_set_transform", 2, None)]
    assert _calls(methods["height_after"]) == []
    table = systems.table
    assert table.struct_size == ctypes.sizeof(VhModule)
    assert mover.struct_size == ctypes.sizeof(VhType)
    assert mover.fields[0].struct_size == ctypes.sizeof(VhField)
    assert methods["update"].struct_size == ctypes.sizeof(VhFunction)
    assert regen.methods[0].calls[0].struct_size == ctypes.sizeof(VhCall)


def test_reflection_records_extern_and_handle_metadata(systems: Any) -> None:
    mover = systems.type("Mover")
    methods = {mover.methods[index].name.decode(): mover.methods[index] for index in range(mover.method_count)}
    get_call, set_call = (methods["update"].calls[index] for index in range(2))
    assert _meta(get_call) == {"cpp": "demo::getTransform", "owns": []}
    assert _meta(set_call) == {}
    assert _meta(systems.type("Entity")) == {"cpp": "demo::Entity", "header": "demo/entity.h"}


def test_instances_read_and_write_as_json(systems: Any) -> None:
    from vernon_dsl.hostlang.value_json import ValueJsonError

    mover = systems.new("Mover")
    assert systems.to_json(mover) == {
        "target": 0,
        "speed": 2.0,
        "rest": {"position": [0.0, 1.0, 0.0], "scale": 1.0},
        "history": [0.0, 0.0, 0.0, 0.0],
        "timer": {"period": 0.5, "elapsed": 0.0, "fired": 0},
        "updates": 0,
    }
    # An edit names only what it changes, including fields of a nested class.
    systems.from_json(mover, {"speed": 0.1, "history": [1, 2, 3, 4], "timer": {"period": "inf"}})
    assert systems.to_json(mover)["speed"] == 0.1
    assert mover.history.tolist() == [1, 2, 3, 4]
    assert mover.timer.period == float("inf")
    assert systems.to_json(mover)["timer"]["period"] == "inf"

    before = systems.to_json(mover)
    for value, message in [
        ({"speed": 1.0, "speedy": 1.0}, "Mover has no field 'speedy'"),
        ({"speed": 3.0, "updates": 1.5}, r"Mover.updates: i32 expects an integer"),
        ({"updates": 1 << 31}, "out of range for i32"),
        ({"history": [1, 2]}, r"Mover.history: expects a list of 4 items"),
        ({"rest": 1}, "Transform is an object of its fields"),
    ]:
        with pytest.raises(ValueJsonError, match=message):
            systems.from_json(mover, value)
    assert systems.to_json(mover) == before


def test_reflection_describes_struct_metadata(systems: Any) -> None:
    health = systems.type("Health")
    assert health.doc == b"Hit points."
    assert _meta(health) == {"role": "component", "name": "demo:Health"}
    fields = _fields(health)
    assert fields["current"].doc == b"What remains."
    assert _meta(fields["current"]) == {"min": 0.0}
    assert fields["maximum"].doc is None


def test_generic_externs_receive_each_struct_descriptor(systems: Any, store: Store) -> None:
    store.rows[("Health", 3)] = {"current": 4.0, "maximum": 5.0}
    store.rows[("Transform", 3)] = {"position": (0.0, 0.0, 0.0), "scale": 2.0}
    regen = systems.new("Regen")

    assert regen.update(3, 0.5) == 6.5
    assert regen.update(3, 0.75) == 7.0

    assert store.rows[("Health", 3)] == {"current": 5.0, "maximum": 5.0}
    assert store.calls[:3] == [("read", 3, "Health"), ("write", 3, "Health"), ("read", 3, "Transform")]


def _same_file(filename: str, path: Path) -> bool:
    return Path(filename).resolve() == path.resolve()


def test_python_mode_runs_at_source_lines_with_readable_locals(store: Store) -> None:
    from vernon_dsl.hostlang import HostModule

    mover = HostModule.load(SYSTEMS, store.registry()).new("Mover")
    lines: list[int] = []
    snapshots: dict[int, dict[str, str]] = {}

    def trace(frame: Any, event: str, argument: Any) -> Any:
        if not _same_file(frame.f_code.co_filename, SYSTEMS):
            return None
        if event == "line":
            lines.append(frame.f_lineno)
            snapshots[frame.f_lineno] = {name: repr(value) for name, value in frame.f_locals.items()}
        return trace

    sys.settrace(trace)
    try:
        mover.update(0.25)
    finally:
        sys.settrace(None)

    # Breakpoints bind to these: Mover.update, then Timer.tick it calls, each statement at its own line.
    assert lines[:1] == [58]
    assert {36, 37, 41} <= set(lines)
    assert {60, 61, 62, 63, 64} <= set(lines)
    assert 59 not in lines
    at_set = snapshots[62]
    assert at_set["dt"] == "0.25"
    assert at_set["transform"] == "Transform(position=Vector([0.0, 0.0, 0.0]), scale=1.0)"
    assert at_set["rise"] == "Vector([0.0, 0.5, 0.0])"
    assert at_set["self"].startswith(
        "Mover(target=Entity(0), speed=2.0, rest=Transform(position=Vector([0.0, 1.0, 0.0]), scale=1.0)"
    )
    assert "timer=Timer(period=0.5, elapsed=0.25, fired=0)" in at_set["self"]


def test_pdb_breakpoints_stop_in_host_methods(store: Store) -> None:
    import bdb

    from vernon_dsl.hostlang import HostModule

    mover = HostModule.load(SYSTEMS, store.registry()).new("Mover")
    hits: list[tuple[int, str]] = []

    class Debugger(bdb.Bdb):
        def user_line(self, frame: Any) -> None:
            if self.break_here(frame):
                hits.append((frame.f_lineno, repr(frame.f_locals.get("value"))))
            self.set_continue()

    debugger = Debugger()
    assert debugger.set_break(str(SYSTEMS), 68) is None
    debugger.runcall(mover.update, 0.5)
    debugger.runcall(mover.update, 0.25)
    assert hits == [(68, "2.0")]


def test_python_mode_tracebacks_reach_the_source_line(store: Store) -> None:
    import traceback

    from vernon_dsl.hostlang import HostModule

    def unavailable(entity: int) -> dict[str, Any]:
        raise RuntimeError("engine unavailable")

    registry = store.registry().bind("demo_get_transform", unavailable)
    mover = HostModule.load(SYSTEMS, registry).new("Mover")
    with pytest.raises(RuntimeError, match="engine unavailable") as error:
        mover.update(0.25)
    frames = traceback.extract_tb(error.value.__traceback__)
    host = [frame for frame in frames if _same_file(frame.filename, SYSTEMS)]
    assert [(frame.name, frame.lineno, frame.line) for frame in host] == [
        ("update", 60, "transform = get_transform(self.target)")
    ]


def test_c_callers_reach_python_mode_methods_through_the_table(store: Store) -> None:
    import ctypes

    from vernon_dsl.hostlang import HostModule

    systems = HostModule.load(SYSTEMS, store.registry())
    mover = systems.new("Mover")
    descriptor = systems.type("Mover")
    methods = {descriptor.methods[index].name: descriptor.methods[index] for index in range(descriptor.method_count)}
    update = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p)(
        methods[b"update"].address
    )
    out = ctypes.c_int32()
    assert update(address_of(mover), 0.5, ctypes.byref(out)) == 0
    assert (out.value, mover.updates, store.transform["position"][1]) == (1, 1, 1.0)

    store.transform = {}
    assert update(address_of(mover), 0.5, ctypes.byref(out)) == 1
    error = systems.take_error()
    assert isinstance(error, KeyError)
    # The traceback reaches the Python line that raised, under the C call.
    assert error.__traceback__ is not None
    assert systems.errors == [] and systems.take_error() is None


def _struct(name: str) -> Any:
    from vernon_dsl.frontend.model import ConcreteType

    return ConcreteType("struct", name)


_COUNTER = """
import vernon_dsl as vd
from vernon_dsl import f32, f64, i32


@vd.host
class Inner:
    ticks: i32 = 0
{inner}

@vd.host
class Counter:
{fields}

    @vd.host(effects="pure")
    def bump(self) -> i32:
        self.count += 1
        self.inner.ticks += 2
        return self.count
"""


def _write_counter(path: Path, fields: str, inner: str = "") -> None:
    path.write_text(_COUNTER.format(fields=textwrap.indent(fields, "    "), inner=inner), encoding="utf-8")


def test_reload_migrates_state_by_field_name(tmp_path: Path) -> None:
    from vernon_dsl.hostlang import HostModule

    source = tmp_path / "counter.py"
    _write_counter(source, "count: i32 = 0\nscale: f32 = 1.0\nlabel: i32 = 7\ninner: Inner\n")
    first = HostModule.load(source)
    counter = first.new("Counter")
    counter.bump()
    counter.bump()
    counter.label = 11
    assert not first.stale

    _write_counter(
        source,
        "inner: Inner\nbonus: i32 = 3\nscale: f64 = 2.0\ncount: i32 = 0\nlabel: i32 = 7\n",
        inner="    extra: f32 = 0.5\n",
    )
    assert first.stale
    second = first.reload()
    migrated = second.migrate(counter)

    assert migrated.count == 2
    assert migrated.label == 11
    # A retyped field keeps the new default; a new field takes its default.
    assert migrated.scale == 2.0
    assert migrated.bonus == 3
    # A class field whose layout changed migrates field by field.
    assert migrated.inner.ticks == 4
    assert migrated.inner.extra == 0.5
    assert migrated.bump() == 3


def test_migration_onto_an_unchanged_layout_copies_everything(tmp_path: Path) -> None:
    from vernon_dsl.hostlang import HostModule

    source = tmp_path / "counter.py"
    _write_counter(source, "count: i32 = 0\ninner: Inner\n")
    module = HostModule.load(source)
    counter = module.new("Counter")
    counter.bump()
    copy = module.reload().migrate(counter)
    assert (copy.count, copy.inner.ticks) == (1, 2)


_ADOPTER = """
from typing import TypeVar

import vernon_dsl as vd
from vernon_dsl import i32

S = TypeVar("S", bound=vd.HostClass)


@vd.extern(symbol="demo_adopt", effects="io", abi="status")
def adopt(order: i32, system: S) -> None: ...


@vd.host
class Ticker:
    ticks: i32 = 2
    history: list[i32]

    def tick(self) -> None:
        self.ticks += 1
        self.history.append(self.ticks)


@vd.host
def start() -> i32:
    ticker = Ticker()
    ticker.ticks = 5
    ticker.history.append(1)
    adopt(3, ticker)
    adopt(4, Ticker())
    ticker = Ticker()
    return ticker.ticks
"""


class _Adopted:
    """What an engine does with an instance an extern takes: moves its bytes into memory of its own, calls a method
    through the class's descriptor, and finalizes it once done."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int, list[int]]] = []

    def __call__(self, order: int, descriptor: Any, address: int) -> None:
        from vernon_dsl.hostlang.native import VhListRecord
        from vernon_dsl.hostlang.reflect import CLASS

        assert (descriptor.name, descriptor.kind) == (b"Ticker", CLASS)
        methods = [descriptor.methods[index] for index in range(descriptor.method_count)]
        tick = next(method for method in methods if method.name == b"tick")
        owned = ctypes.create_string_buffer(descriptor.size)
        ctypes.memmove(owned, address, descriptor.size)
        method = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)
        assert method(tick.address)(ctypes.addressof(owned)) == 0
        fields = _fields(descriptor)
        history = VhListRecord.from_buffer(owned, fields["history"].offset)
        values = [ctypes.c_int32.from_address(history.data + 4 * index).value for index in range(history.size)]
        self.seen.append((order, ctypes.c_int32.from_buffer(owned, fields["ticks"].offset).value, values))
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.fini)(ctypes.addressof(owned))


def test_externs_take_host_class_instances_by_move(tmp_path: Path) -> None:
    from host_aot import LinkedHost
    from vernon_dsl.hostlang import HostModule
    from vernon_dsl.hostlang.aot import load_programs

    source = tmp_path / "adopter.py"
    source.write_text(_ADOPTER, encoding="utf-8")
    adopted = _Adopted()
    registry = ExternRegistry().bind("demo_adopt", adopted)
    assert HostModule.load(source, registry).call("start") == 2
    assert adopted.seen == [(3, 6, [1, 6]), (4, 3, [3])]
    if toolchain() is None:
        pytest.skip("no C toolchain links Host objects")
    assert LinkedHost(load_programs(source), tmp_path / "linked", registry).call("start") == 2
    assert adopted.seen[2:] == [(3, 6, [1, 6]), (4, 3, [3])]


def _compile_error(tmp_path: Path, body: str) -> str:
    source = tmp_path / "broken.py"
    source.write_text(
        textwrap.dedent(
            """
            from typing import Annotated, TypeVar
            import vernon_dsl as vd
            from vernon_dsl import f32, f64, i32

            T = TypeVar("T")


            @vd.struct(shared=True)
            class Point:
                x: f32


            @vd.host
            class Inner:
                ticks: i32 = 0


            @vd.extern(symbol="demo_read", abi="status")
            def read(type: type[T]) -> T: ...
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
            class Holder:
                inner: Inner

                def copy(self) -> None:
                    local = self.inner
                    local = self.inner
            """,
            "a borrowed name is bound once",
        ),
        (
            """
            @vd.extern(symbol="demo_store", abi="status")
            def store(value: T) -> None: ...


            @vd.host(effects="pure")
            def keep(scale: f32) -> None:
                store(scale)
            """,
            "'store' takes a @struct value, not f32",
        ),
        (
            """
            @vd.host(effects="pure")
            def edit(point: Point) -> f32:
                point.x = 1.0
                return point.x
            """,
            "only fields of class instances are writable",
        ),
        (
            """
            @vd.host
            class Speed:
                value: f32 = 1.0

                def accelerate(self, dt: f64) -> None:
                    self.value += dt
            """,
            "convert the operand explicitly",
        ),
        (
            """
            @vd.struct(shared=True)
            class Bad:
                inner: Inner
            """,
            "cannot be held in a Struct",
        ),
        (
            """
            S = TypeVar("S", bound=vd.HostClass)


            @vd.extern(symbol="demo_swap", abi="status")
            def swap(system: S) -> S: ...
            """,
            "'swap' takes a Host class instance as S, which annotates nothing else",
        ),
        (
            """
            S = TypeVar("S", bound=vd.HostClass)


            @vd.extern(symbol="demo_adopt", abi="status")
            def adopt(system: S) -> None: ...


            @vd.host
            def twice() -> None:
                inner = Inner()
                adopt(inner)
                adopt(inner)
            """,
            "'inner' was moved",
        ),
        (
            """
            S = TypeVar("S", bound=vd.HostClass)


            @vd.extern(symbol="demo_adopt", abi="status")
            def adopt(system: S) -> None: ...


            @vd.host
            class Holder:
                inner: Inner

                def give(self) -> None:
                    adopt(self.inner)
            """,
            "names an existing instance",
        ),
        (
            """
            @vd.func(shared=True)
            def poke(inner: Inner) -> f32:
                return 1.0
            """,
            "a Host class is not passed to one",
        ),
        (
            """
            @vd.host
            class Built:
                value: f32 = 1.0

                def __init__(self) -> None:
                    pass
            """,
            "define no '__init__'",
        ),
        (
            """
            @vd.host
            class Loop:
                again: Loop
            """,
            "contains itself",
        ),
        (
            """
            @vd.host
            class Holder:
                inner: Inner

                def replace(self) -> None:
                    self.inner.ticks = 1
                    self.inner = self.inner
            """,
            "copy once stored in self.inner",
        ),
        (
            """
            @vd.host
            def wrong() -> f32:
                return read(f32)
            """,
            "is a @struct type",
        ),
        (
            """
            @vd.host
            class Labeled:
                value: Annotated[f32, vd.meta(range=print)] = 1.0
            """,
            "metadata values are",
        ),
        (
            """
            @vd.host
            class Placed:
                point: Point

                def move(self) -> None:
                    self.point.x = 1.0
            """,
            "'self.point' is a Struct value",
        ),
    ],
    ids=[
        "copy-into-local",
        "value-generic-scalar",
        "struct-field-store",
        "widening-update",
        "class-in-struct",
        "instance-variable-returned",
        "instance-used-after-move",
        "instance-moved-from-a-place",
        "class-parameter",
        "dunder-method",
        "containment-cycle",
        "whole-instance-store",
        "generic-scalar",
        "metadata-value",
        "struct-member-through-self",
    ],
)
def test_class_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    assert message in _compile_error(tmp_path, body)


def test_aot_objects_hold_classes_for_every_target() -> None:
    from vernon_dsl.hostlang.aot import compile_host

    for triple in (
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
        "aarch64-apple-ios",
        "wasm32-unknown-emscripten",
    ):
        artifacts = compile_host(SYSTEMS, target=triple, prefix="demo_")
        assert artifacts.object
        assert "demo_module" in artifacts.optimized_ir
        assert "demo_Mover_update" in artifacts.optimized_ir


def test_header_and_json_reflection_describe_classes() -> None:
    from vernon_dsl.hostlang.aot import compile_host

    artifacts = compile_host(SYSTEMS, prefix="demo_")
    header = artifacts.header
    assert '#include "vernon_host/reflect.h"' in header
    assert "extern const VhModule demo_module;" in header
    assert "int32_t demo_Mover_update(DemoMover *self, double dt, int32_t *out);" in header
    assert "int32_t demo_Mover__init(DemoMover *self);" in header
    assert "_record" not in header
    classes = artifacts.reflection["classes"]
    assert classes["Mover"]["meta"] == {"role": "system", "order": 10, "tags": ["movement", "demo"]}
    assert [method["name"] for method in classes["Mover"]["methods"]] == ["update", "height_after"]
    assert classes["Mover"]["fields"][1]["doc"] == "How fast the target rises, in meters per second."
    assert artifacts.reflection["table"] == "demo_module"


def test_host_modules_declare_as_program_assets() -> None:
    import systems
    import systems_asset
    import vernon_dsl as vd

    asset = systems_asset.asset
    assert asset.is_host and asset.prefix == "demo_" and asset.variants == ((),)
    assert vd.program_asset(id="game/systems-v2", program=systems).prefix == "game_systems_v2_"
    assert vd.program_asset(id="2d/systems", program=systems).prefix == "_2d_systems_"
    feature = vd.feature("FAST")
    with pytest.raises(ValueError, match="do not take variants"):
        vd.program_asset(id="demo/systems", program=systems, variants=({feature: True},))
    with pytest.raises(ValueError, match="not a C identifier prefix"):
        vd.program_asset(id="demo/systems", program=systems, prefix="demo-")


@pytest.mark.parametrize("triple", ["", "wasm32-unknown-emscripten"])
def test_host_module_assets_cook_a_registering_object_header_and_reflection(tmp_path: Path, triple: str) -> None:
    import vernon_dsl as vd
    from vernon_dsl.bundle import make_target_options
    from vernon_dsl.program_assets import cook_program_asset

    output = tmp_path / "host_systems"
    target = make_target_options("cpu", {"triple": triple} if triple else {})
    manifest_path = cook_program_asset(
        program_asset=f"{PROGRAMS / 'systems_asset.py'}:asset", output=output, target=target
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["type"] == "host_module" and manifest["id"] == "demo/systems"
    host = manifest["host_module"]
    assert host["table"] == "demo_module" and host["prefix"] == "demo_"
    blob = manifest["blobs"][host["object"]]
    data = (output / blob["location"]["uri"]).read_bytes()
    assert len(data) == blob["byte_length"]
    assert b"vernonRuntimeRegisterStaticHostModule" in data and b"demo/systems" in data
    assert "extern const VhModule demo_module;" in (output / host["header"]).read_text(encoding="utf-8")
    assert json.loads((output / host["reflection"]).read_text(encoding="utf-8"))["table"] == "demo_module"
    if triple:
        assert manifest["target"]["options"]["triple"] == triple
        assert blob["location"]["uri"].endswith(".wasm.o")
    with pytest.raises(ValueError, match="Host module asset"):
        vd.load_program(manifest_path)


def test_host_module_assets_combine_several_modules(tmp_path: Path) -> None:
    import abilities
    import systems
    import systems_asset
    import vernon_dsl as vd
    from vernon_dsl.program_assets import cook_program_asset

    asset = systems_asset.asset
    assert [Path(source).name for source in asset.host_sources] == ["systems.py", "abilities.py"]
    with pytest.raises(TypeError, match="tuple of only modules"):
        vd.program_asset(id="demo/mixed", program=(systems, systems.scaled))
    with pytest.raises(ValueError, match="lists a module twice"):
        vd.program_asset(id="demo/twice", program=(systems, abilities, systems))

    output = tmp_path / "host_systems"
    manifest_path = cook_program_asset(program_asset=f"{PROGRAMS / 'systems_asset.py'}:asset", output=output)
    host = json.loads(manifest_path.read_text(encoding="utf-8"))["host_module"]
    reflection = json.loads((output / host["reflection"]).read_text(encoding="utf-8"))
    assert reflection["modules"] == ["systems", "abilities"]
    # Timer, which abilities imports from systems, is one class of the combined module.
    assert sorted(reflection["classes"]) == ["Mover", "Poison", "Regen", "Timer", "Trail"]
    header = (output / host["header"]).read_text(encoding="utf-8")
    assert header.count("typedef struct DemoTimer ") == 1
    assert "int32_t demo_Poison_update(DemoPoison *self, DemoEntity target, double dt, float *out);" in header


def test_combined_modules_qualify_private_functions_by_module() -> None:
    from vernon_dsl.hostlang.aot import compile_host

    ir = compile_host([SYSTEMS, PROGRAMS / "abilities.py"], prefix="demo_").ir
    # systems exports `scaled`; abilities has a private helper of that name with another body.
    assert 'define internal i32 @"vh.scaled"(' in ir
    assert 'define internal i32 @"vh.1.scaled"(' in ir
    assert ir.count('define internal i32 @"vh.Timer__tick"(') == 1


def test_python_mode_runs_a_module_of_several_sources(store: Store) -> None:
    import systems_asset
    from vernon_dsl.hostlang import HostModule

    module = HostModule.from_asset(systems_asset.asset, store.registry())
    assert module.table.name == b"demo/systems"
    first, second = module.sources
    assert second.namespace.Timer is first.namespace.Timer, "abilities imports the Timer this module runs"
    with pytest.raises(AttributeError, match="several source files"):
        _ = module.program

    store.rows[("Health", 3)] = {"current": 4.5, "maximum": 5.0}
    poison = module.new("Poison")
    assert poison.update(3, 0.25) == 0.0
    assert poison.update(3, 0.25) == 0.25
    assert poison.ticks == 1 and store.rows[("Health", 3)]["current"] == 4.25
    assert module.call("scaled", 1.5, 3) == 4.5
    assert module.reload().table.name == b"demo/systems"


def test_combined_modules_reject_a_name_both_define(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import compile_host

    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(
            textwrap.dedent(
                """
                import vernon_dsl as vd
                from vernon_dsl import i32


                @vd.host
                class Counter:
                    count: i32 = 0
                """
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="'Counter__init' is defined by both first.py and second.py"):
        compile_host([tmp_path / "first.py", tmp_path / "second.py"])


def test_host_module_assets_cook_only_for_the_cpu_target(tmp_path: Path) -> None:
    from vernon_dsl.program_assets import ProgramCompileError, cook_program_asset

    with pytest.raises(ProgramCompileError, match="only for the cpu target"):
        cook_program_asset(program_asset=f"{PROGRAMS / 'systems_asset.py'}:asset", output=tmp_path, target="vulkan")


def test_c_application_drives_the_object_through_its_reflection_table(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import compile_host
    from vernon_dsl.hostlang.reflect import include_dir

    artifacts = compile_host(SYSTEMS, prefix="demo_")
    assert artifacts.reflection["externs"]["demo_get_transform"]["meta"] == {"cpp": "demo::getTransform", "owns": []}
    assert artifacts.reflection["handles"]["Entity"]["meta"]["header"] == "demo/entity.h"
    paths = artifacts.write(tmp_path, "systems")
    consumer = PROGRAMS / "systems_consumer.c"
    executable = tmp_path / "consumer.exe"
    found = toolchain()
    if found is None:
        pytest.skip("no C toolchain")
    kind, compiler, environment = found
    if kind == "msvc":
        command = [
            compiler,
            "/nologo",
            "/W4",
            "/WX",
            "/std:c11",
            f"/I{tmp_path}",
            f"/I{include_dir()}",
            str(consumer),
            str(paths["object"]),
            f"/Fe{executable}",
            f"/Fo{tmp_path}{os.sep}",
        ]
    else:
        command = [
            compiler,
            "-std=c11",
            "-Wall",
            "-Werror",
            f"-I{tmp_path}",
            f"-I{include_dir()}",
            str(consumer),
            str(paths["object"]),
            "-lm",
            "-o",
            str(executable),
        ]
    built = subprocess.run(command, capture_output=True, text=True, env=environment, cwd=tmp_path, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    lines = run.stdout.splitlines()
    (mover_json,) = [line for line in lines if line.startswith("mover_json=")]
    lines.remove(mover_json)
    mover = json.loads(mover_json.removeprefix("mover_json="))
    assert list(mover) == ["target", "speed", "rest", "history", "timer", "updates"]
    assert (mover["speed"], mover["updates"], mover["history"]) == (4, 3, [0, 4, 0, 0])
    assert mover["rest"] == {"position": [0, 1, 0], "scale": 1}
    assert mover["timer"] == {"period": 0.5, "elapsed": 0.25, "fired": 1}

    assert lines == [
        "version=3",
        "name=systems",
        "types=6",
        "functions=1",
        "mover_kind=12 size=80 align=8",
        "mover_doc=Raises its target every update, recording its speed each time the timer fires.",
        'mover_role="system"',
        "speed_offset=8 doc=How fast the target rises, in meters per second. max=10.0",
        "history_kind=9 count=4 element=f32",
        "init_status=0",
        "speed_default=2",
        "update_symbol=demo_Mover_update effect=2 params=1",
        "updates=3 height=3",
        "private_symbol=null",
        "height_after=5",
        "regen_status=0 healed=5.5 stored=4.5",
        'health_meta="demo:Health"',
        "migrated_speed=4",
        "scaled=4.5 function=scaled",
        "regen_call=demo_read:1:Health",
        "regen_call=demo_read:1:Transform",
        "regen_call=demo_write:2:Health",
        'mover_call=demo_get_transform:1:-:"demo::getTransform"',
        "mover_call=demo_set_transform:2:-:-",
        'entity_meta="demo/entity.h"',
        'health_json={"current":4.5,"maximum":5} needed=27',
        'truncated={"curre',
        "sizes=1 fini=null",
    ]
