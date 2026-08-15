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

DEFAULT_GRIDS = (32, 64, 128, 256, 512, 1024)
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


def _run_grid(grid: int, pressure_iterations: int, steps: int) -> dict[str, Any]:
    vd.init(arch=vd.cpu)
    simulation = SmokeFluidSimulation(
        grid=grid,
        pressure_iterations=pressure_iterations,
        differentiable=True,
    )
    target = v_target(grid)

    warmup = simulation.step_vjp(target)
    warmup(_cotangents(grid))
    del warmup
    simulation.reset()
    gc.collect()

    rss_before = _rss_bytes()
    forward_seconds = 0.0
    backward_seconds = 0.0
    rss_samples = [rss_before]
    tape_samples: list[tuple[int, int, int, int, int, int]] = []
    pass_telemetry: tuple[dict[str, Any], ...] = ()
    tape_context_limit_bytes = 0
    peak_temporary_tape_bytes = 0
    recomputation_factor = 0.0
    python_reverse_callback_count = 0
    for _ in range(steps):
        forward_started = time.perf_counter()
        pullback = simulation.step_vjp(target)
        forward_seconds += time.perf_counter() - forward_started
        tape_samples.append(
            (
                pullback.logical_residual_bytes,
                pullback.resident_tape_bytes,
                pullback.allocated_tape_bytes,
                pullback.retained_allocation_bytes,
                pullback.checkpoint_bytes,
                pullback.peak_runtime_managed_bytes,
            )
        )
        backward_started = time.perf_counter()
        pullback(_cotangents(grid))
        backward_seconds += time.perf_counter() - backward_started
        tape_samples[-1] = (*tape_samples[-1][:5], pullback.peak_runtime_managed_bytes)
        tape_context_limit_bytes = pullback.tape_context_limit_bytes
        recomputation_factor = pullback.recomputation_factor
        pass_telemetry = pullback.pass_telemetry
        peak_temporary_tape_bytes = max(
            (int(item["peak_temporary_tape_bytes"]) for item in pass_telemetry),
            default=0,
        )
        python_reverse_callback_count = pullback.reverse_python_callback_count
        del pullback
        rss_samples.append(_rss_bytes())
    (
        logical_bytes,
        resident_bytes,
        allocated_bytes,
        retained_allocation_bytes,
        checkpoint_bytes,
        peak_runtime_managed_bytes,
    ) = tape_samples[-1]
    elapsed = forward_seconds + backward_seconds
    rss_after = rss_samples[-1]
    measured_rss_samples = rss_samples[1:]
    stable_window_size = max(5, steps // 5)
    stable_window = measured_rss_samples[-stable_window_size:]
    stable_tolerance_bytes = max(1024 * 1024, max(stable_window) // 100)
    rss_stable_high_water_mark = (
        len(stable_window) == stable_window_size and max(stable_window) - min(stable_window) <= stable_tolerance_bytes
    )

    telemetry_errors = [
        name
        for name, invalid in (
            ("resident<logical", resident_bytes < logical_bytes),
            ("allocated<resident", allocated_bytes < resident_bytes),
            ("zero-logical-retains-tape", logical_bytes == 0 and (resident_bytes != 0 or allocated_bytes != 0)),
            (
                "min-memory-retained-tape",
                any(
                    logical != 0 or resident != 0 or allocated != 0
                    for logical, resident, allocated, _, _, _ in tape_samples
                ),
            ),
            ("peak<allocated+checkpoint", peak_runtime_managed_bytes < allocated_bytes + checkpoint_bytes),
            (
                "peak<temporary-tape+checkpoint",
                peak_runtime_managed_bytes < peak_temporary_tape_bytes + checkpoint_bytes,
            ),
            ("unexpected-context-limit", tape_context_limit_bytes != BENCHMARK_TAPE_CONTEXT_LIMIT),
            ("peak>context-limit", peak_runtime_managed_bytes > tape_context_limit_bytes),
            ("temporary-tape>context-limit", peak_temporary_tape_bytes > tape_context_limit_bytes),
            (
                "capture-without-temporary-tape",
                any(
                    item["residual_source_kind"].split("+", 1)[0] in {"static_capture", "dynamic_capture"}
                    and int(item["estimated_tape_bytes"]) > 0
                    and int(item["peak_temporary_tape_bytes"]) == 0
                    for item in pass_telemetry
                ),
            ),
            ("nonpositive-forward-time", forward_seconds <= 0),
            ("nonpositive-backward-time", backward_seconds <= 0),
            (f"callback-count={python_reverse_callback_count}", python_reverse_callback_count <= 0),
            (
                "logical-sum",
                sum(int(item["logical_residual_bytes"]) for item in pass_telemetry) != logical_bytes,
            ),
            ("resident-sum", sum(int(item["resident_tape_bytes"]) for item in pass_telemetry) != resident_bytes),
            ("allocated-sum", sum(int(item["allocated_tape_bytes"]) for item in pass_telemetry) != allocated_bytes),
            ("checkpoint-sum", sum(int(item["checkpoint_bytes"]) for item in pass_telemetry) != checkpoint_bytes),
            (
                "source-kind",
                any(
                    item["residual_source_kind"].split("+")[:1]
                    not in [["none"], ["static_capture"], ["dynamic_capture"]]
                    or any(source != "pure_rematerialization" for source in item["residual_source_kind"].split("+")[1:])
                    or item["control_history_kind"] not in {"none", "dynamic_capture"}
                    for item in pass_telemetry
                ),
            ),
        )
        if invalid
    ]
    if telemetry_errors:
        raise RuntimeError(f"smoke VJP reported inconsistent telemetry: {', '.join(telemetry_errors)}")
    return {
        "grid": grid,
        "pressure_iterations": pressure_iterations,
        "steps": steps,
        "logical_residual_bytes": logical_bytes,
        "resident_bytes": resident_bytes,
        "allocated_bytes": allocated_bytes,
        "retained_allocation_bytes": retained_allocation_bytes,
        "native_checkpoint_bytes": checkpoint_bytes,
        "peak_runtime_managed_bytes": peak_runtime_managed_bytes,
        "resident_to_logical_ratio": resident_bytes / logical_bytes if logical_bytes else 0.0,
        "rss_bytes": rss_after,
        "rss_before_bytes": rss_before,
        "rss_samples_bytes": measured_rss_samples,
        "rss_delta_bytes": max(rss_after - rss_before, 0),
        "rss_growth_bytes": max(rss_samples[-1] - rss_samples[1], 0) if len(rss_samples) > 2 else 0,
        "rss_stable_high_water_mark": rss_stable_high_water_mark,
        "rss_stability_window_steps": stable_window_size,
        "rss_stability_tolerance_bytes": stable_tolerance_bytes,
        "max_logical_residual_bytes": max(sample[0] for sample in tape_samples),
        "max_resident_bytes": max(sample[1] for sample in tape_samples),
        "max_allocated_bytes": max(sample[2] for sample in tape_samples),
        "max_retained_allocation_bytes": max(sample[3] for sample in tape_samples),
        "max_checkpoint_bytes": max(sample[4] for sample in tape_samples),
        "max_peak_runtime_managed_bytes": max(sample[5] for sample in tape_samples),
        "peak_temporary_tape_bytes": peak_temporary_tape_bytes,
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
        "# Cost-aware CPU smoke autodiff benchmark",
        "",
        "CPU, warmed forward/backward smoke runs. Pressure iterations and measured steps are recorded per row; "
        f"tape context limit={BENCHMARK_TAPE_CONTEXT_LIMIT} bytes.",
        "",
        f"Frozen baseline: {BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL:.1f} logical bytes per active cell; "
        f"compiler Tape hints per lane={BASELINE_COMPILER_TAPE_HINTS}.",
        "",
        "| Grid | Pressure | Steps | Logical residual | Logical/cell | Resident | Allocated | Retained allocation | "
        "Temporary Tape peak | "
        "Checkpoint | Peak managed | "
        "Resident/logical | RSS | "
        "RSS delta | Stable RSS | Forward | Backward | Recompute | Python callbacks |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {grid} | {pressure_iterations} | {steps} | {logical_residual_bytes} | "
            "{logical_bytes_per_active_cell:.3f} | "
            "{resident_bytes} | {allocated_bytes} | {retained_allocation_bytes} | {peak_temporary_tape_bytes} | "
            "{native_checkpoint_bytes} | {peak_runtime_managed_bytes} | {resident_to_logical_ratio:.3f} | "
            "{rss_bytes} | {rss_delta_bytes} | {rss_stable_high_water_mark} | "
            "{forward_seconds:.6f}s | {backward_seconds:.6f}s | "
            "{recomputation_factor:.3f} | {python_reverse_callback_count} |".format(**result)
        )
    lines.extend(
        (
            "",
            "Logical residual bytes are payload bytes retained by reusable pullbacks. A no-Tape or bounded-replay "
            "pullback reports zero retained Tape bytes; checkpoint, retained primal-version, transaction, and "
            "gradient staging memory remain visible in their separate counters. RSS is sampled from the worker "
            "process after backward; each grid runs in a fresh process.",
            "",
            "The JSON artifact also records maxima across all measured steps and RSS growth after the first "
            "measured step, plus every per-step RSS sample. Stable RSS means the final max(5, steps/5) samples "
            "fit within max(1 MiB, 1% of that window's high-water mark), so pressure runs distinguish stable "
            "selected residual/checkpoint storage from unconditional per-step Tape retention.",
            "",
            "Recomputation factor is 1 + compiler-estimated rematerialization cost / active operation count, plus "
            "any graph-checkpoint replay factor.",
            "",
            "CPU bounded replay uses complete-workgroup segments. A pure static balanced/min-runtime plan may retain "
            "whole-dispatch Tape only when its exact construction allocation fits the hard context budget.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cost-aware CPU autodiff smoke grids")
    parser.add_argument("--grids", type=int, nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--worker-grid", type=int)
    parser.add_argument("--pressure-iterations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--skip-gradient-parity",
        action="store_true",
        help="skip the finite-difference correctness gate (intended only for profiling iterations)",
    )
    arguments = parser.parse_args()
    if arguments.worker_grid is not None:
        print(
            json.dumps(
                _run_grid(arguments.worker_grid, arguments.pressure_iterations, arguments.steps),
                sort_keys=True,
            )
        )
        return
    if any(grid <= 0 for grid in arguments.grids) or arguments.pressure_iterations < 0 or arguments.steps <= 0:
        raise ValueError("grids and steps must be positive; pressure iterations must be non-negative")
    if not arguments.skip_gradient_parity:
        _run_gradient_parity_gate()

    results: list[dict[str, Any]] = []
    for grid in arguments.grids:
        completed = subprocess.run(
            (
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-grid",
                str(grid),
                "--pressure-iterations",
                str(arguments.pressure_iterations),
                "--steps",
                str(arguments.steps),
            ),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"grid {grid} benchmark failed:\n{completed.stderr.strip()}")
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append(result)
        print(json.dumps(result, sort_keys=True))
    report = {
        "schema_version": 3,
        "baseline": {
            "logical_bytes_per_active_cell": BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL,
            "compiler_tape_hints_per_lane": BASELINE_COMPILER_TAPE_HINTS,
            "gradient_parity_test": GRADIENT_PARITY_TEST,
        },
        "pressure_iterations": arguments.pressure_iterations,
        "steps": arguments.steps,
        "results": results,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(_markdown(results))


if __name__ == "__main__":
    main()
