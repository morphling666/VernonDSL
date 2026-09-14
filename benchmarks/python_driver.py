from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import vernon_dsl as vd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "tests"))

from backend_test_matrix import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    BACKEND_TEST_MATRIX,
    BackendRequirements,
    CapabilityUnavailable,
    probe_backend,
    require_available,
)
from workloads import (  # noqa: E402
    ControlDraw,
    IncrementChain,
    PersistentlyBoundAdd,
    RepeatedDraw,
    Square,
    elementwise_kernel,
    empty_kernel,
    increment,
    reduction_kernel,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    return parser.parse_args()


def _measurement(name: str, unit: str, samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "name": name,
        "unit": unit,
        "sample_count": len(samples),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _backend(name: str):
    normalized = name.replace("-", "").replace("_", "").lower()
    for row in BACKEND_TEST_MATRIX:
        if row.runtime_backend.replace("_", "").lower() == normalized:
            return row
    raise ValueError(f"unknown backend {name!r}")


def _requirements(fixture: str) -> BackendRequirements:
    if fixture in {"binding.control", "graphics.draw"}:
        return BackendRequirements(gpu=True, graphics=True, storage_buffers=True)
    if fixture == "backend.empty_kernel":
        return BackendRequirements(compute=True)
    return BackendRequirements(compute=True, storage_buffers=True)


def _samples(warmup: int, iterations: int, invoke: Callable[[int], Any]) -> tuple[list[float], Any]:
    result: Any = None
    for index in range(warmup):
        result = invoke(index)
    samples: list[float] = []
    for index in range(iterations):
        begin = time.perf_counter_ns()
        result = invoke(warmup + index)
        samples.append(float(time.perf_counter_ns() - begin))
    return samples, result


def _binding_tensor(parameters: dict[str, Any], warmup: int, iterations: int) -> dict[str, Any]:
    changed = parameters["update"] == "changed"
    count = 1024
    groups = np.uint32((count + 63) // 64)
    source = vd.storage.from_numpy(np.arange(count, dtype=np.float32))
    output = vd.storage.zeros(dtype=vd.f32, shape=(count,))
    module = PersistentlyBoundAdd()

    def invoke(index: int):
        amount = np.float32(2.0 if not changed or index % 2 == 0 else 3.0)
        return module(output, source, amount, groups)

    samples, _ = _samples(warmup, iterations, invoke)
    expected = np.arange(count, dtype=np.float32) + (2.0 if not changed or (warmup + iterations - 1) % 2 == 0 else 3.0)
    np.testing.assert_array_equal(output.to_numpy(), expected)
    session = vd.current_session()
    if session is None:
        raise RuntimeError("benchmark invocation did not create a runtime session")
    specialization = next(iter(module._program_cache._partition(session).snapshot.values()))
    telemetry = {name: int(value) for name, value in specialization.native_program.binding_telemetry.items()}
    return {
        "status": "completed",
        "measurements": [_measurement("call_ns", "ns", samples)],
        "counters": telemetry,
    }


def _binding_dynamic(parameters: dict[str, Any], warmup: int, iterations: int) -> dict[str, Any]:
    changed = parameters["update"] == "changed"
    sources = {count: vd.storage.from_numpy(np.arange(count, dtype=np.float32)) for count in (1024, 2048)}
    outputs = {count: vd.storage.zeros(dtype=vd.f32, shape=(count,)) for count in (1024, 2048)}

    def invoke(index: int):
        count = 2048 if changed and index % 2 else 1024
        increment(
            outputs[count].view(access="write"),
            sources[count].view(access="read"),
            grid=((count + 63) // 64, 1, 1),
        )
        return outputs[count]

    samples, result = _samples(warmup, iterations, invoke)
    expected_count = 2048 if changed and (warmup + iterations - 1) % 2 else 1024
    np.testing.assert_array_equal(result.to_numpy(), np.arange(expected_count, dtype=np.float32) + 1.0)
    return {
        "status": "completed",
        "measurements": [_measurement("call_ns", "ns", samples)],
    }


def _binding_control(parameters: dict[str, Any], warmup: int, iterations: int) -> dict[str, Any]:
    changed = parameters["update"] == "changed"
    module = ControlDraw()
    vertices = vd.storage.from_numpy(np.array([[-0.75, -0.75], [0.75, -0.75], [0.0, 0.75]], dtype=np.float32))
    texture = vd.Texture.zeros(shape=(16, 16))
    render_pass = vd.render_pass(
        vd.RenderTarget.from_attachments(colors={0: texture}),
        color=vd.clear((0.0, 0.0, 0.0, 1.0)),
    )
    draw = vd.draw(vertex_count=3)

    def invoke(index: int):
        width = 15 if changed and index % 2 else 16
        return module(vertices, render_pass, draw, vd.dynamic_state(viewport=(0, 0, width, 16)))

    samples, _ = _samples(warmup, iterations, invoke)
    if not np.any(texture.to_numpy()[..., :3]):
        raise RuntimeError("graphics control benchmark produced no fragments")
    return {
        "status": "completed",
        "measurements": [_measurement("call_ns", "ns", samples)],
    }


def _program(fixture: str, parameters: dict[str, Any], warmup: int, iterations: int) -> dict[str, Any]:
    node_count = int(parameters["node_count"])
    module = Square() if fixture == "program.square" else IncrementChain(node_count)
    count = 1024 if fixture == "program.chain" else 1
    source = vd.storage.from_numpy(np.ones(count, dtype=np.float32))
    groups = np.uint32((count + 63) // 64)

    def invoke(_: int):
        return module(source) if fixture == "program.square" else module(source, groups)

    samples, result = _samples(warmup, iterations, invoke)
    expected = np.ones(count, dtype=np.float32) if fixture == "program.square" else np.full(count, 1 + node_count)
    np.testing.assert_array_equal(result.to_numpy(), expected)
    return {
        "status": "completed",
        "measurements": [_measurement("call_ns", "ns", samples)],
        "counters": {"node_count": node_count},
    }


def _backend_benchmark(fixture: str, warmup: int, iterations: int) -> dict[str, Any]:
    if fixture == "backend.empty_kernel":

        def invoke(_: int):
            return empty_kernel(grid=(1, 1, 1))

        def validate() -> None:
            return None

    elif fixture == "backend.elementwise":
        source = vd.storage.from_numpy(np.ones(1048576, dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1048576,))

        def invoke(_: int):
            return elementwise_kernel(
                output.view(access="write"),
                source.view(access="read"),
                grid=(16384, 1, 1),
            )

        def validate() -> None:
            np.testing.assert_array_equal(output.to_numpy(), np.full(1048576, 3.0, dtype=np.float32))

    else:
        source = vd.storage.from_numpy(np.ones(1048576, dtype=np.float32))
        output = vd.storage.zeros(dtype=vd.f32, shape=(1,))

        def invoke(_: int):
            return reduction_kernel(
                output.view(access="write"),
                source.view(access="read"),
                grid=(1, 1, 1),
            )

        def validate() -> None:
            np.testing.assert_array_equal(output.to_numpy(), np.array([1048576.0], dtype=np.float32))

    samples, _ = _samples(warmup, iterations, invoke)
    validate()
    return {
        "status": "completed",
        "measurements": [_measurement("call_ns", "ns", samples)],
    }


def _graphics(parameters: dict[str, Any], warmup: int, iterations: int) -> dict[str, Any]:
    draw_count = int(parameters["draw_count"])
    change_state = bool(parameters["state_changes"])
    module = RepeatedDraw(draw_count)
    vertices = vd.storage.from_numpy(np.array([[-0.75, -0.75], [0.75, -0.75], [0.0, 0.75]], dtype=np.float32))
    texture = vd.Texture.zeros(shape=(64, 64))
    render_pass = vd.render_pass(
        vd.RenderTarget.from_attachments(colors={0: texture}),
        color=vd.clear((0.0, 0.0, 0.0, 1.0)),
    )
    draw = vd.draw(vertex_count=3)

    def invoke(index: int):
        width = 63 if change_state and index % 2 else 64
        return module(vertices, render_pass, draw, vd.dynamic_state(viewport=(0, 0, width, 64)))

    samples, _ = _samples(warmup, iterations, invoke)
    if not np.any(texture.to_numpy()[..., :3]):
        raise RuntimeError("graphics benchmark produced no fragments")
    return {
        "status": "completed",
        "measurements": [_measurement("call_ns", "ns", samples)],
        "counters": {"draw_count": draw_count},
    }


def main() -> int:
    arguments = _arguments()
    parameters = json.loads(arguments.parameters_json)
    try:
        backend = _backend(arguments.backend)
        require_available(probe_backend(backend, _requirements(arguments.fixture)))
        if arguments.fixture == "binding.tensor":
            result = _binding_tensor(parameters, arguments.warmup, arguments.iterations)
        elif arguments.fixture == "binding.dynamic_shape":
            result = _binding_dynamic(parameters, arguments.warmup, arguments.iterations)
        elif arguments.fixture == "binding.control":
            result = _binding_control(parameters, arguments.warmup, arguments.iterations)
        elif arguments.fixture in {"program.square", "program.chain"}:
            result = _program(arguments.fixture, parameters, arguments.warmup, arguments.iterations)
        elif arguments.fixture.startswith("backend."):
            result = _backend_benchmark(arguments.fixture, arguments.warmup, arguments.iterations)
        elif arguments.fixture == "graphics.draw":
            result = _graphics(parameters, arguments.warmup, arguments.iterations)
        else:
            raise ValueError(f"unsupported Python benchmark fixture {arguments.fixture!r}")
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return 0
    except CapabilityUnavailable as error:
        print(
            json.dumps(
                {"status": "skipped", "skip_reason": str(error), "measurements": []},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
