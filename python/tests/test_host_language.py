"""Host functions: Python mode, which runs the source as written, and the AOT object, linked and loaded."""

from __future__ import annotations

import ctypes
import importlib.util
import math
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import vernon_dsl.types as vd_types
from host_aot import LinkedHost, toolchain
from vernon_dsl.diagnostics import CompileError
from vernon_dsl.hostlang import ExternRegistry, HostBindingError, HostExecutionError, HostStatus, load_host_program

PROGRAMS = Path(__file__).parent / "host_programs"
sys.path.insert(0, str(PROGRAMS))

import gameplay  # noqa: E402
import parity_cases  # noqa: E402
import semantics  # noqa: E402


def _native_available() -> bool:
    try:
        from vernon_dsl import _native

        return hasattr(_native, "_compile_host")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="vernon_dsl._native was built without Host AOT")


class World:
    def __init__(self) -> None:
        self.position = (1.0, 2.0, 3.0)
        self.scale = 1.0
        self.log: list[tuple[str, Any]] = []

    def registry(self) -> ExternRegistry:
        registry = ExternRegistry()
        registry.bind("demo_find", lambda name: 7 if name == "player" else 0)
        registry.bind("demo_get_transform", self.get_transform)
        registry.bind("demo_set_transform", self.set_transform)
        registry.bind("demo_log", lambda message, value: self.log.append((message, value)))
        return registry

    def get_transform(self, entity: int) -> dict[str, Any]:
        self.log.append(("get", entity))
        return {"position": self.position, "scale": self.scale}

    def set_transform(self, entity: int, value: dict[str, Any]) -> None:
        self.log.append(("set", entity))
        self.position = value["position"]
        self.scale = value["scale"]


def _count_registry(calls: list[int]) -> ExternRegistry:
    registry = ExternRegistry()

    def count(value: int) -> bool:
        calls.append(value)
        return value > 0

    registry.bind("semantics_count", count)
    return registry


_SEMANTICS_ENTRIES = sorted({name for name, _, _ in parity_cases.EXACT_CASES} | {"transcendental"})


@pytest.fixture(scope="module")
def compiled(tmp_path_factory: pytest.TempPathFactory) -> tuple[LinkedHost, list[int]]:
    from vernon_dsl.hostlang.aot import load_programs

    if toolchain() is None:
        pytest.skip("no C toolchain links Host objects")
    calls: list[int] = []
    programs = load_programs(semantics.__file__, _SEMANTICS_ENTRIES)
    return LinkedHost(programs, tmp_path_factory.mktemp("semantics"), _count_registry(calls)), calls


def _plain(value: Any) -> Any:
    """A Python-mode result in the boundary form compiled results come back in: Tensors as tuples, Structs as dicts."""

    if isinstance(value, np.ndarray):
        return tuple(_plain(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    fields = getattr(type(value), "__vernon_fields__", None)
    if fields is not None:
        return {name: _plain(getattr(value, name)) for name in fields}
    return value


def _flatten(value: Any) -> list[Any]:
    return [item for part in value for item in _flatten(part)] if isinstance(value, tuple) else [value]


def _python_outcome(name: str, arguments: tuple[Any, ...], calls: list[int]) -> tuple[Any, ...]:
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            result = getattr(semantics, name).call(*arguments, registry=_count_registry(calls))
        except HostExecutionError as error:
            return ("status", error.status, tuple(calls))
        except Exception as error:  # noqa: BLE001 - Python mode fails where compiled code returns a status.
            return ("raises", type(error).__name__, tuple(calls))
    return ("ok", repr(_plain(result)), tuple(calls))


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gameplay_script_runs_against_python_externs() -> None:
    world = World()
    result = gameplay.raise_entity.call(np.float32(0.5), 3, registry=world.registry())

    assert result == np.float32(3.5)
    assert world.position == (1.0, 3.5, 3.0)
    assert world.scale == 2.0
    assert world.log == [("get", 7), ("set", 7)]
    assert type(result) is np.float32


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    parity_cases.EXACT_CASES,
    ids=[f"{name}{index}" for index, (name, _, _) in enumerate(parity_cases.EXACT_CASES)],
)
def test_compiled_scalar_rules_are_exact(
    compiled: tuple[LinkedHost, list[int]], name: str, arguments: tuple[Any, ...], expected: tuple[Any, ...]
) -> None:
    library, calls = compiled
    calls.clear()
    try:
        outcome: tuple[Any, ...] = ("ok", repr(library.call(name, *arguments)), tuple(calls))
    except HostExecutionError as error:
        outcome = ("status", error.status, tuple(calls))
    assert outcome == expected


def test_compiled_transcendental_functions_follow_libm(compiled: tuple[LinkedHost, list[int]]) -> None:
    library, _ = compiled
    for arguments, expected in parity_cases.TRANSCENDENTAL_CASES:
        np.testing.assert_allclose(np.asarray(library.call("transcendental", *arguments)), expected, rtol=2e-6)


def test_python_mode_differs_from_compiled_code_only_where_known() -> None:
    calls: list[int] = []
    differences = {}
    for index, (name, arguments, expected) in enumerate(parity_cases.EXACT_CASES):
        outcome = _python_outcome(name, arguments, calls)
        if outcome != expected:
            differences[f"{name}{index}"] = outcome
    assert set(differences) - parity_cases.PYTHON_PLATFORM_DEPENDENT == set(parity_cases.PYTHON_DIFFERENCES)


def test_python_mode_agrees_with_compiled_code_on_finite_float_arithmetic(
    compiled: tuple[LinkedHost, list[int]],
) -> None:
    library, _ = compiled
    generator = np.random.default_rng(7)
    values = (generator.standard_normal(400) * generator.choice([1e-3, 1.0, 37.0, 1e4], 400)).astype(np.float32)
    for a, b in zip(values[::2].tolist(), values[1::2].tolist(), strict=True):
        for name, arguments in (
            ("rounding", (a, b)),
            ("vectors", ((a, b, a * 0.5), (b, 1.0, -a))),
            ("transcendental", (a,)),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                python = getattr(semantics, name)(*arguments)
            expected = library.call(name, *arguments)
            flat = [np.asarray(_flatten(part), dtype=np.float64) for part in (_plain(python), expected)]
            np.testing.assert_allclose(*flat, rtol=2e-6, atol=1e-6, equal_nan=True, err_msg=f"{name}{arguments}")


def test_results_keep_their_host_types() -> None:
    assert math.copysign(1.0, semantics.float_remainder(0.0, -1.0)) == -1.0
    assert type(semantics.true_division(1, 3)) is np.float32
    pair = semantics.structs(3, 5.0)
    assert type(pair) is semantics.Pair
    assert (pair.low, pair.high, pair.flag) == (6, 2.5, False)
    assert (type(pair.low), type(pair.high)) == (np.int32, np.float64)
    vectors = semantics.vectors((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert type(vectors[1]) is vd_types.Vector and vectors[1].dtype == np.float32
    assert vectors[2].y == pytest.approx(0.5345225)


def test_compiled_runtime_failures_return_statuses(compiled: tuple[LinkedHost, list[int]]) -> None:
    library, _ = compiled
    with pytest.raises(HostExecutionError) as division:
        library.call("integer_division", 1, 0)
    assert division.value.status == HostStatus.DIVISION_BY_ZERO
    with pytest.raises(HostExecutionError) as index:
        library.call("matrix_index", ((1, 2, 3), (4, 5, 6)), 0, 3)
    assert index.value.status == HostStatus.INDEX_OUT_OF_RANGE
    with pytest.raises(HostExecutionError) as step:
        library.call("loops", 3, 0)
    assert step.value.status == HostStatus.INVALID_RANGE_STEP


def test_short_circuit_evaluates_only_selected_extern_calls() -> None:
    calls: list[int] = []
    assert tuple(semantics.short_circuit.call(0, 1, registry=_count_registry(calls))) == (0, 1)
    assert calls == [0, 0, 1]


def test_missing_binding_fails_before_any_extern_runs() -> None:
    registry = ExternRegistry().bind("demo_find", lambda name: pytest.fail("ran before binding check"))
    with pytest.raises(HostBindingError, match="demo_get_transform"):
        gameplay.raise_entity.call(np.float32(1.0), 1, registry=registry)


def test_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "schema_script.py"
    source.write_text(
        textwrap.dedent(
            """
            import vernon_dsl as vd
            from vernon_dsl import i32

            @vd.extern(symbol="engine_tick", effects="read", schema="abc")
            def tick() -> i32: ...

            @vd.host(effects="read")
            def run() -> i32:
                return tick()
            """
        )
    )
    module = _load_module(source, "schema_script")
    with pytest.raises(HostBindingError, match="schema"):
        module.run.call(registry=ExternRegistry().bind("engine_tick", lambda: 1, schema="def"))
    assert module.run.call(registry=ExternRegistry().bind("engine_tick", lambda: 4, schema="abc")) == 4


def test_native_extern_address_binds_through_ctypes() -> None:
    @ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32)
    def native_count(value: int) -> bool:
        seen.append(value)
        return value > 0

    seen: list[int] = []
    address = ctypes.cast(native_count, ctypes.c_void_p).value
    registry = ExternRegistry().bind_address("semantics_count", address)
    assert tuple(semantics.short_circuit.call(1, 1, registry=registry)) == (1, 1)
    assert seen == [1, 1, 1]


def test_module_reload_runs_the_edited_source(tmp_path: Path) -> None:
    from vernon_dsl.hostlang import HostModule

    source = tmp_path / "reloaded.py"
    template = textwrap.dedent(
        """
        import vernon_dsl as vd
        from vernon_dsl import i32

        @vd.host(effects="pure")
        def value(x: i32) -> i32:
            return x * FACTOR
        """
    )
    source.write_text(template.replace("FACTOR", "2"))
    module = HostModule.load(source)
    assert module.call("value", 5) == 10
    source.write_text(template.replace("FACTOR", "3"))
    assert module.stale
    assert module.call("value", 5) == 10
    assert module.reload().call("value", 5) == 15


def test_rebinding_an_extern_reloads_the_compiled_module() -> None:
    registry = ExternRegistry().bind("semantics_count", lambda value: True)
    assert tuple(semantics.short_circuit.call(0, 0, registry=registry)) == (1, 1)
    registry.bind("semantics_count", lambda value: False)
    assert tuple(semantics.short_circuit.call(0, 0, registry=registry)) == (0, 0)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            @vd.func(shared=True)
            def helper() -> i32:
                return tick()

            @vd.host(effects="read")
            def run() -> i32:
                return helper()
            """,
            "cannot call",
        ),
        (
            """
            @vd.func
            def helper() -> i32:
                return 1

            @vd.host(effects="pure")
            def run() -> i32:
                return helper()
            """,
            r"declare it @func\(shared=True\)",
        ),
        (
            """
            @vd.struct
            class Point:
                x: f32

            @vd.host(effects="pure")
            def run() -> f32:
                return Point(x=1.0).x
            """,
            r"declare it @struct\(shared=True\)",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 3]) -> f32:
                values[0] = 2.0
                return values[0]
            """,
            "values is a Tensor value, which is never written into",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 3]) -> Vector[f32, 3]:
                return values.at[0].multiply(2.0)
            """,
            r"\.at\[\.\.\.\] has set\(x\) and add\(x\)",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(count: i32) -> i32:
                return count.at[0].set(1)
            """,
            "only a Tensor value has",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 3]) -> Vector[f32, 3]:
                return values.at[3].set(1.0)
            """,
            "outside an axis of extent 3",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 4], i: i32) -> Vector[f32, 2]:
                return values[i : i * 2]
            """,
            r"a dynamic start is a window i:i\+n",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 4], i: i32) -> Vector[f32, 2]:
                return values[abs(i) : abs(i) + 2]
            """,
            "a window start has no calls",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 4]) -> Vector[f32, 2]:
                return values[3:5]
            """,
            "does not fit an axis of extent 4",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 4]) -> Vector[f32, 4]:
                return values.at[1:3].set(values[0:3])
            """,
            r"into a selection of Tensor\[f32, \(2,\)\]",
        ),
        (
            """
            @vd.struct(shared=True)
            class Point:
                x: f32
                y: f32

            @vd.host(effects="pure")
            def run(p: Point) -> Point:
                return vd.replace(p, z=1.0)
            """,
            "Point has no field 'z'",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                return tick()
            """,
            "declares effects='pure'",
        ),
        (
            """
            @vd.host(effects="read")
            def run() -> i32:
                x = 1
                x = 2.5
                return x
            """,
            "unsafe implicit conversion",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(a: f32) -> f32:
                return a // 2.0
            """,
            "floor division requires integer operands",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(a) -> i32:
                return 0
            """,
            "requires a type annotation",
        ),
        (
            """
            @vd.extern(symbol="bad")
            def bad() -> i32:
                return 1

            @vd.host(effects="io")
            def run() -> i32:
                return bad()
            """,
            "body must be '...'",
        ),
        (
            """
            @vd.host(effects="pure")
            def run() -> i32:
                return 3000000000
            """,
            "does not fit in i32",
        ),
        (
            """
            @vd.host(effects="pure")
            def run(values: Vector[f32, 3]) -> f32:
                return values[0] if 1 < 2 < 3 else 0.0
            """,
            "chained comparisons",
        ),
    ],
)
def test_host_diagnostics(tmp_path: Path, body: str, message: str) -> None:
    source = tmp_path / "invalid.py"
    header = textwrap.dedent(
        """
        import vernon_dsl as vd
        from vernon_dsl import Vector, f32, i32

        @vd.extern(symbol="engine_tick", effects="read")
        def tick() -> i32: ...
        """
    )
    source.write_text(header + textwrap.dedent(body))
    with pytest.raises(CompileError, match=message):
        load_host_program(source, "run")


def test_device_function_cannot_call_host_code(tmp_path: Path) -> None:
    source = tmp_path / "device.py"
    source.write_text(
        textwrap.dedent(
            """
            import vernon_dsl as vd
            from vernon_dsl import i32

            @vd.extern(symbol="engine_tick", effects="read")
            def tick() -> i32: ...

            @vd.host(effects="read")
            def run() -> i32:
                return tick()

            @vd.func
            def helper() -> i32:
                return run()

            @vd.host(effects="read")
            def entry() -> i32:
                return helper()
            """
        )
    )
    with pytest.raises(CompileError, match="cannot call"):
        load_host_program(source, "entry")


def test_python_exceptions_in_externs_reach_the_caller(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import load_programs

    def broken(entity: int) -> dict[str, Any]:
        raise RuntimeError("engine unavailable")

    registry = World().registry().bind("demo_get_transform", broken)
    with pytest.raises(RuntimeError, match="engine unavailable"):
        gameplay.raise_entity.call(1.0, 1, registry=registry)
    if toolchain() is None:
        pytest.skip("no C toolchain links Host objects")
    library = LinkedHost(load_programs(gameplay.__file__, ["raise_entity"]), tmp_path, registry)
    with pytest.raises(RuntimeError, match="engine unavailable"):
        library.call("raise_entity", 1.0, 1)


@pytest.mark.parametrize(
    "target",
    ["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu", "aarch64-apple-ios", "wasm32-unknown-emscripten"],
)
def test_objects_compile_for_each_target(target: str) -> None:
    from vernon_dsl.hostlang.aot import compile_host

    artifacts = compile_host(gameplay.__file__, ["raise_entity", "integer_rules"], target=target, prefix="demo_")
    assert artifacts.object
    assert "define i32 @demo_raise_entity(" in artifacts.optimized_ir
    assert "@demo_find(" in artifacts.optimized_ir
    if target.startswith("aarch64"):
        assert "[2 x i64]" in artifacts.ir
    if target.startswith("wasm32"):
        assert artifacts.object.startswith(b"\0asm")


def test_header_and_reflection_describe_the_boundary(tmp_path: Path) -> None:
    from vernon_dsl.hostlang.aot import compile_host

    artifacts = compile_host(gameplay.__file__, ["raise_entity", "wide"], prefix="demo_")
    header = artifacts.header
    assert "typedef struct DemoTransform { float position[3]; float scale; } DemoTransform;" in header
    assert "int32_t demo_raise_entity(float height, int32_t steps, float *out);" in header
    assert "int32_t demo_wide(int64_t a, uint64_t b, int64_t *out);" in header
    external = compile_host(gameplay.__file__, ["raise_entity"], prefix="demo_", types_header="engine.h").header
    assert '#include "engine.h"' in external
    assert "typedef struct DemoTransform" not in external
    assert "offsetof(DemoTransform, scale) == 12" in external
    reflection = artifacts.reflection
    assert [entry["symbol"] for entry in reflection["entries"]] == ["demo_raise_entity", "demo_wide"]
    assert reflection["entries"][0]["effect"] == "write"
    assert reflection["externs"]["demo_get_transform"]["abi"] == "status"
    assert reflection["structs"]["Transform"]["fields"][1] == {"name": "scale", "type": "f32", "offset": 12}
    paths = artifacts.write(tmp_path, "gameplay")
    assert all(path.exists() for path in paths.values())
