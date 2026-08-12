from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np  # type: ignore[import-not-found]
import vernon_dsl as vd  # type: ignore[import-not-found]
from vernon_dsl.compiler import compile_file  # type: ignore[import-not-found]
from vernon_dsl.pipeline_assets import cook_pipeline_asset  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_FIXTURE = ROOT / "python" / "tests" / "pipeline_asset_fixture.py"
NATIVE_FIXTURE = ROOT / "python" / "tests" / "advanced_pipeline_shader.py"


def _load_scale_kernel() -> Any:
    spec = importlib.util.spec_from_file_location("vernon_benchmark_fixture", FRONTEND_FIXTURE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark fixture {FRONTEND_FIXTURE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.scale


BASELINE_SCALE = _load_scale_kernel()


def _milliseconds(operation: Callable[[], None]) -> float:
    started = time.perf_counter_ns()
    operation()
    return (time.perf_counter_ns() - started) / 1_000_000.0


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    percentile_index = max(0, min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1))
    return {
        "count": len(samples),
        "minimum_ms": round(ordered[0], 4),
        "median_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[percentile_index], 4),
        "maximum_ms": round(ordered[-1], 4),
    }


def _architecture(name: str) -> object:
    return {
        "cpu": vd.cpu,
        "cuda": vd.cuda,
        "vulkan": vd.vulkan,
        "opengl": vd.opengl,
        "opengles": vd.opengles,
        "directx": vd.directx,
    }[name]


def benchmark_frontend(iterations: int) -> dict[str, object]:
    samples = [_milliseconds(lambda: compile_file(FRONTEND_FIXTURE, entry="scale")) for _ in range(iterations)]
    return {
        "scenario": "frontend",
        "source": FRONTEND_FIXTURE.relative_to(ROOT).as_posix(),
        "entry": "scale",
        "first_ms": round(samples[0], 4),
        "subsequent": _summary(samples[1:] or samples),
    }


def benchmark_native(iterations: int, target: str) -> dict[str, object]:
    from vernon_dsl import _native  # type: ignore[attr-defined,import-not-found]

    mlir = compile_file(NATIVE_FIXTURE, entry="advanced_fragment")
    native_target = {
        "cpu": _native.Target.CPU,
        "cuda": _native.Target.CUDA,
        "vulkan": _native.Target.VULKAN,
        "opengl": _native.Target.OPENGL,
        "opengles": _native.Target.OPENGL_ES,
        "directx": _native.Target.DIRECTX,
        "metal": _native.Target.METAL,
    }[target]
    compiler = _native.Compiler()

    def compile_once() -> None:
        program = compiler.compile_program_result(mlir, native_target)
        if not program.ok:
            raise RuntimeError(program.diagnostics)

    samples = [_milliseconds(compile_once) for _ in range(iterations)]
    return {
        "scenario": "native",
        "source": NATIVE_FIXTURE.relative_to(ROOT).as_posix(),
        "entry": "advanced_fragment",
        "target": target,
        "first_ms": round(samples[0], 4),
        "subsequent": _summary(samples[1:] or samples),
    }


def benchmark_cook(iterations: int, target: str) -> dict[str, object]:
    samples: list[float] = []
    with tempfile.TemporaryDirectory(prefix="vernon-cook-baseline-") as directory:
        root = Path(directory)
        for iteration in range(iterations):
            output = root / str(iteration)
            samples.append(
                _milliseconds(
                    lambda output=output: cook_pipeline_asset(
                        pipeline_asset=f"{FRONTEND_FIXTURE}:scale_asset",
                        output=output,
                        target=target,
                    )
                )
            )
    return {
        "scenario": "cook",
        "source": FRONTEND_FIXTURE.relative_to(ROOT).as_posix(),
        "asset": "scale_asset",
        "target": target,
        "first_ms": round(samples[0], 4),
        "subsequent": _summary(samples[1:] or samples),
    }


def benchmark_kernel(iterations: int, warmup_iterations: int, architecture: str, elements: int) -> dict[str, object]:
    api_version = (4, 3) if architecture == "opengl" else (3, 1) if architecture == "opengles" else None
    vd.init(arch=_architecture(architecture), api_version=api_version)
    type(BASELINE_SCALE).clear_cache()
    BASELINE_SCALE.compile_count = 0
    values = vd.storage.from_numpy(np.ones(elements, dtype=np.float32))
    grid = (elements, 1, 1)

    cold_ms = _milliseconds(lambda: BASELINE_SCALE(values, np.float32(1.0001), grid=grid))
    for _ in range(warmup_iterations):
        BASELINE_SCALE(values, np.float32(1.0001), grid=grid)
    warm_samples = [
        _milliseconds(lambda: BASELINE_SCALE(values, np.float32(1.0001), grid=grid)) for _ in range(iterations)
    ]

    host_values = np.ones(elements, dtype=np.float32)
    upload_samples = [_milliseconds(lambda: values.copy_from_numpy(host_values)) for _ in range(iterations)]
    readback_samples: list[float] = []
    for _ in range(iterations):
        BASELINE_SCALE(values, np.float32(1.0001), grid=grid)
        readback_samples.append(_milliseconds(lambda: values.to_numpy()))

    def transfer_round_trip() -> None:
        values.copy_from_numpy(host_values)
        BASELINE_SCALE(values, np.float32(1.0001), grid=grid)
        values.to_numpy()

    transfer_samples = [_milliseconds(transfer_round_trip) for _ in range(iterations)]
    return {
        "scenario": "kernel",
        "architecture": architecture,
        "elements": elements,
        "warmup_iterations": warmup_iterations,
        "cold_compile_and_dispatch_ms": round(cold_ms, 4),
        "warm_dispatch": _summary(warm_samples),
        "host_upload": _summary(upload_samples),
        "host_readback_after_dispatch": _summary(readback_samples),
        "upload_dispatch_readback": _summary(transfer_samples),
        "compile_count": BASELINE_SCALE.compile_count,
    }


def benchmark_showcase(architecture: str, showcase: str, frames: int, size: int) -> dict[str, object]:
    scripts = {
        "terrain": ROOT / "examples" / "terrain_showcase.py",
        "mandelbulb": ROOT / "examples" / "mandelbulb_showcase.py",
    }
    script = scripts[showcase]
    with tempfile.TemporaryDirectory(prefix="vernon-showcase-baseline-") as directory:
        output = Path(directory) / f"{showcase}.png"
        summary = Path(directory) / f"{showcase}.json"
        command = [
            sys.executable,
            str(script),
            "--arch",
            architecture,
            "--frames",
            str(frames),
            "--size",
            str(size),
            "--headless",
            "--output",
            str(output),
            "--result-json",
            str(summary),
        ]
        environment = os.environ.copy()
        python_paths = [str(ROOT / "python"), str(ROOT / "examples")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        started = time.perf_counter_ns()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "showcase benchmark failed")
        program_summary = json.loads(summary.read_text(encoding="utf-8"))
    return {
        "scenario": "showcase",
        "showcase": showcase,
        "architecture": architecture,
        "frames": frames,
        "size": size,
        "total_ms": round(elapsed_ms, 4),
        "average_ms_per_frame_including_startup": round(elapsed_ms / frames, 4),
        "program_summary": program_summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible VernonDSL baseline benchmarks.")
    parser.add_argument("scenario", choices=("frontend", "native", "cook", "kernel", "showcase"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan", "opengl", "opengles", "directx"), default="cpu")
    parser.add_argument(
        "--target",
        choices=("cpu", "cuda", "vulkan", "opengl", "opengles", "directx", "metal"),
        default="cpu",
    )
    parser.add_argument("--elements", type=int, default=262_144)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--showcase", choices=("terrain", "mandelbulb"), default="terrain")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    positive = (arguments.iterations, arguments.elements, arguments.frames, arguments.grid, arguments.size)
    if any(value <= 0 for value in positive):
        raise ValueError("iterations, elements, frames, grid, and size must be positive")
    if arguments.warmup_iterations < 0:
        raise ValueError("warmup iterations must be non-negative")
    if arguments.scenario == "frontend":
        result = benchmark_frontend(arguments.iterations)
    elif arguments.scenario == "native":
        result = benchmark_native(arguments.iterations, arguments.target)
    elif arguments.scenario == "cook":
        result = benchmark_cook(arguments.iterations, arguments.target)
    elif arguments.scenario == "kernel":
        result = benchmark_kernel(
            arguments.iterations,
            arguments.warmup_iterations,
            arguments.arch,
            arguments.elements,
        )
    else:
        if arguments.arch not in {"vulkan", "opengl", "directx"}:
            raise ValueError("the visual showcases support vulkan, opengl, and directx")
        result = benchmark_showcase(arguments.arch, arguments.showcase, arguments.frames, arguments.size)
    document = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "result": result,
    }
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
