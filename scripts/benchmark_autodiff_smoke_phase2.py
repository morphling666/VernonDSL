from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "python"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np  # noqa: E402
import vernon_dsl as vd  # noqa: E402

from examples.autodiff_smoke_mpc import SmokeFluidSimulation, v_target  # noqa: E402

DEFAULT_GRIDS = (32, 64, 128, 256)
BENCHMARK_TAPE_CONTEXT_LIMIT = 256 * 1024 * 1024
BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL = 469.8
BASELINE_COMPILER_TAPE_HINTS = {
    "advect": 208,
    "initialize": 88,
    "jacobi": 88,
    "project": 88,
    "transport": 160,
}
GRADIENT_PARITY_TEST = (
    "python.tests.test_smoke_fluid_graph.SmokeFluidGraphTests."
    "test_single_step_velocity_gradient_matches_finite_difference"
)


def _rss_bytes() -> int:
    try:
        kibibytes = int(subprocess.check_output(("ps", "-o", "rss=", "-p", str(os.getpid())), text=True).strip())
        return kibibytes * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return maximum if sys.platform == "darwin" else maximum * 1024


def _cotangents(grid: int) -> dict[str, Any]:
    return {
        "density": np.zeros((grid, grid), dtype=np.float32),
        "velocity": vd.storage.tangent_zeros(dtype=vd.Vector[vd.f32, 2], shape=(grid, grid)),
        "loss": np.ones((1,), dtype=np.float32),
    }


def _run_gradient_parity_gate() -> None:
    environment = os.environ.copy()
    python_path = os.pathsep.join((str(REPOSITORY_ROOT / "python"), str(REPOSITORY_ROOT)))
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    completed = subprocess.run(
        (sys.executable, "-m", "unittest", GRADIENT_PARITY_TEST),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"finite-difference gradient parity gate failed:\n{detail}")


def _run_grid(grid: int) -> dict[str, Any]:
    vd.init(arch=vd.cpu)
    simulation = SmokeFluidSimulation(grid=grid, pressure_iterations=1, differentiable=True)
    target = v_target(grid)

    warmup = simulation.step_vjp(target)
    warmup(_cotangents(grid))
    del warmup
    simulation.reset()
    gc.collect()

    rss_before = _rss_bytes()
    forward_started = time.perf_counter()
    pullback = simulation.step_vjp(target)
    forward_seconds = time.perf_counter() - forward_started
    logical_bytes = pullback.logical_residual_bytes
    resident_bytes = pullback.resident_tape_bytes
    allocated_bytes = pullback.allocated_tape_bytes
    checkpoint_bytes = pullback.checkpoint_bytes
    peak_runtime_managed_bytes = pullback.peak_runtime_managed_bytes
    tape_context_limit_bytes = pullback.tape_context_limit_bytes
    recomputation_factor = pullback.recomputation_factor
    pass_telemetry = pullback.pass_telemetry
    backward_started = time.perf_counter()
    pullback(_cotangents(grid))
    backward_seconds = time.perf_counter() - backward_started
    python_reverse_callback_count = pullback.reverse_python_callback_count
    elapsed = forward_seconds + backward_seconds
    rss_after = _rss_bytes()

    if (
        logical_bytes <= 0
        or resident_bytes < logical_bytes
        or allocated_bytes < resident_bytes
        or peak_runtime_managed_bytes < allocated_bytes + checkpoint_bytes
        or tape_context_limit_bytes != BENCHMARK_TAPE_CONTEXT_LIMIT
        or peak_runtime_managed_bytes > tape_context_limit_bytes
        or forward_seconds <= 0
        or backward_seconds <= 0
        or python_reverse_callback_count != 9
        or sum(int(item["logical_residual_bytes"]) for item in pass_telemetry) != logical_bytes
        or sum(int(item["resident_tape_bytes"]) for item in pass_telemetry) != resident_bytes
        or sum(int(item["allocated_tape_bytes"]) for item in pass_telemetry) != allocated_bytes
        or sum(int(item["checkpoint_bytes"]) for item in pass_telemetry) != checkpoint_bytes
        or any(
            item["residual_source_kind"].split("+")[:1] not in [["none"], ["static_capture"], ["dynamic_capture"]]
            or any(source != "pure_rematerialization" for source in item["residual_source_kind"].split("+")[1:])
            or item["control_history_kind"] not in {"none", "dynamic_capture"}
            for item in pass_telemetry
        )
    ):
        raise RuntimeError("smoke VJP reported inconsistent tape memory telemetry")
    return {
        "grid": grid,
        "logical_residual_bytes": logical_bytes,
        "resident_bytes": resident_bytes,
        "allocated_bytes": allocated_bytes,
        "native_checkpoint_bytes": checkpoint_bytes,
        "peak_runtime_managed_bytes": peak_runtime_managed_bytes,
        "resident_to_logical_ratio": resident_bytes / logical_bytes,
        "rss_bytes": rss_after,
        "rss_delta_bytes": max(rss_after - rss_before, 0),
        "elapsed_seconds": elapsed,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "recomputation_factor": recomputation_factor,
        "python_reverse_callback_count": python_reverse_callback_count,
        "tape_context_limit_bytes": tape_context_limit_bytes,
        "logical_bytes_per_active_cell": logical_bytes / (grid * grid),
        "pass_telemetry": pass_telemetry,
    }


def _markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2 smoke autodiff benchmark",
        "",
        "CPU, one warmed forward/backward smoke step per grid, pressure_iterations=1, "
        f"tape context limit={BENCHMARK_TAPE_CONTEXT_LIMIT} bytes.",
        "",
        f"Frozen baseline: {BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL:.1f} logical bytes per active cell; "
        f"compiler Tape hints per lane={BASELINE_COMPILER_TAPE_HINTS}.",
        "",
        "| Grid | Logical residual | Logical/cell | Resident | Allocated | Checkpoint | Peak managed | "
        "Resident/logical | RSS | "
        "RSS delta | Forward | Backward | Recompute | Python callbacks |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {grid} | {logical_residual_bytes} | {logical_bytes_per_active_cell:.3f} | "
            "{resident_bytes} | {allocated_bytes} | "
            "{native_checkpoint_bytes} | {peak_runtime_managed_bytes} | {resident_to_logical_ratio:.3f} | "
            "{rss_bytes} | {rss_delta_bytes} | {forward_seconds:.6f}s | {backward_seconds:.6f}s | "
            "{recomputation_factor:.3f} | {python_reverse_callback_count} |".format(**result)
        )
    lines.extend(
        (
            "",
            "Logical residual bytes are payload bytes retained by reusable pullbacks. Sealed static batches retain "
            "only payload; their construction descriptors and lane state are released. Resident and allocated bytes "
            "also include retained dynamic payload-chunk capacity and compact page metadata. RSS is sampled from the "
            "worker process after backward; each grid runs in a fresh process.",
            "",
            "Recomputation factor is 1 + compiler-estimated rematerialization cost / active operation count, plus "
            "any graph-checkpoint replay factor.",
            "",
            "Paging remains required: this phase freezes the immutable in-memory layout and measures its overhead; "
            "it does not make whole-dispatch storage bounded.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 static-tape smoke grids")
    parser.add_argument("--grids", type=int, nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--worker-grid", type=int)
    parser.add_argument(
        "--skip-gradient-parity",
        action="store_true",
        help="skip the finite-difference correctness gate (intended only for profiling iterations)",
    )
    arguments = parser.parse_args()
    if arguments.worker_grid is not None:
        print(json.dumps(_run_grid(arguments.worker_grid), sort_keys=True))
        return
    if any(grid <= 0 for grid in arguments.grids):
        raise ValueError("grid sizes must be positive")
    if not arguments.skip_gradient_parity:
        _run_gradient_parity_gate()

    results: list[dict[str, Any]] = []
    for grid in arguments.grids:
        completed = subprocess.run(
            (sys.executable, str(Path(__file__).resolve()), "--worker-grid", str(grid)),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"grid {grid} benchmark failed:\n{completed.stderr.strip()}")
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    report = {
        "schema_version": 2,
        "baseline": {
            "logical_bytes_per_active_cell": BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL,
            "compiler_tape_hints_per_lane": BASELINE_COMPILER_TAPE_HINTS,
            "gradient_parity_test": GRADIENT_PARITY_TEST,
        },
        "results": results,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(_markdown(results))


if __name__ == "__main__":
    main()
