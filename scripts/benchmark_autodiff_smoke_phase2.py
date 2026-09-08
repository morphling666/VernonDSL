from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import statistics
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
DEFAULT_ARCHITECTURES = ("cpu",)
SUPPORTED_ARCHITECTURES = ("cpu", "cuda", "vulkan", "directx", "metal", "opengl")
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


def _initial_state(grid: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    density = np.exp(-8.0 * ((x + 0.25) ** 2 + (y - 0.1) ** 2)).astype(np.float32)
    velocity = np.stack((-0.08 * y, 0.08 * x), axis=-1).astype(np.float32)
    return density, velocity


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    flat = array.reshape(-1)
    return {
        "shape": list(array.shape),
        "sum": float(np.sum(array)),
        "l2": float(np.linalg.norm(flat)),
        "maximum_absolute": float(np.max(np.abs(flat), initial=0.0)),
        "samples": [float(item) for item in flat[: min(flat.size, 16)]],
    }


def _latency_summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    percentile_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(ordered),
        "minimum_seconds": ordered[0],
        "median_seconds": statistics.median(ordered),
        "p95_seconds": ordered[percentile_index],
        "maximum_seconds": ordered[-1],
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


def _run_grid(
    grid: int,
    pressure_iterations: int,
    steps: int,
    architecture: str,
    planning_policy: str = "min_memory",
) -> dict[str, Any]:
    try:
        vd.init(arch=getattr(vd, architecture))
    except RuntimeError as error:
        return {
            "architecture": architecture,
            "planning_policy": planning_policy,
            "grid": grid,
            "skipped": True,
            "skip_reason": str(error),
        }
    simulation = SmokeFluidSimulation(
        grid=grid,
        pressure_iterations=pressure_iterations,
        differentiable=True,
        planning_policy=planning_policy,
    )
    target = v_target(grid)
    initial_density, initial_velocity = _initial_state(grid)
    simulation.set_state(initial_density, initial_velocity)

    warmup = simulation.step_vjp(target)
    warmup(_cotangents(grid))
    del warmup
    simulation.reset()
    simulation.set_state(initial_density, initial_velocity)
    gc.collect()

    rss_before = _rss_bytes()
    forward_seconds = 0.0
    backward_seconds = 0.0
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    rss_samples = [rss_before]
    tape_samples: list[tuple[int, int, int, int, int, int]] = []
    pass_telemetry: tuple[dict[str, Any], ...] = ()
    tape_context_limit_bytes = 0
    peak_temporary_tape_bytes = 0
    recomputation_factor = 0.0
    python_reverse_callback_count = 0
    submission_count = 0
    wait_count = 0
    readback_count = 0
    atomic_publication_count = 0
    temporary_allocation_traffic_bytes = 0
    device_wait_nanoseconds = 0
    gradient_summary: dict[str, Any] = {}
    forward_summary: dict[str, Any] = {}
    for _ in range(steps):
        forward_started = time.perf_counter()
        pullback = simulation.step_vjp(target)
        forward_sample = time.perf_counter() - forward_started
        forward_samples.append(forward_sample)
        forward_seconds += forward_sample
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
        forward_before_backward = (
            simulation.density_numpy().copy(),
            simulation.velocity_numpy().copy(),
            simulation.output_loss.to_numpy().copy(),
        )
        backward_started = time.perf_counter()
        gradients = pullback(_cotangents(grid))
        backward_sample = time.perf_counter() - backward_started
        backward_samples.append(backward_sample)
        backward_seconds += backward_sample
        forward_after_backward = (
            simulation.density_numpy(),
            simulation.velocity_numpy(),
            simulation.output_loss.to_numpy(),
        )
        if any(
            not np.array_equal(before, after)
            for before, after in zip(forward_before_backward, forward_after_backward, strict=True)
        ):
            raise RuntimeError("smoke pullback mutated published forward resources")
        gradient_summary = _array_summary(gradients["state_velocity"].to_numpy())
        forward_summary = {
            "density": _array_summary(forward_after_backward[0]),
            "velocity": _array_summary(forward_after_backward[1]),
            "loss": _array_summary(forward_after_backward[2]),
        }
        tape_samples[-1] = (*tape_samples[-1][:5], pullback.peak_runtime_managed_bytes)
        tape_context_limit_bytes = pullback.tape_context_limit_bytes
        recomputation_factor = pullback.recomputation_factor
        pass_telemetry = pullback.pass_telemetry
        peak_temporary_tape_bytes = max(
            (int(item["peak_temporary_tape_bytes"]) for item in pass_telemetry),
            default=0,
        )
        python_reverse_callback_count = pullback.reverse_python_callback_count
        submission_count += pullback.submission_count
        wait_count += pullback.wait_count
        readback_count += pullback.readback_count
        atomic_publication_count += pullback.atomic_publication_count
        temporary_allocation_traffic_bytes += pullback.temporary_allocation_traffic_bytes
        device_wait_nanoseconds += pullback.device_wait_nanoseconds
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
                planning_policy == "min_memory"
                and any(
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
                    and int(item["allocated_tape_bytes"]) == 0
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
        raise RuntimeError(
            "smoke VJP reported inconsistent telemetry: "
            f"{', '.join(telemetry_errors)} "
            f"(peak={peak_runtime_managed_bytes}, temporary={peak_temporary_tape_bytes}, "
            f"context_limit={tape_context_limit_bytes}, passes={pass_telemetry})"
        )
    return {
        "architecture": architecture,
        "planning_policy": planning_policy,
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
        "elapsed_latency": _latency_summary(
            [forward + backward for forward, backward in zip(forward_samples, backward_samples, strict=True)]
        ),
        "forward_latency": _latency_summary(forward_samples),
        "backward_latency": _latency_summary(backward_samples),
        "forward_samples_seconds": forward_samples,
        "backward_samples_seconds": backward_samples,
        "recomputation_factor": recomputation_factor,
        "python_reverse_callback_count": python_reverse_callback_count,
        "submission_count": submission_count,
        "wait_count": wait_count,
        "readback_count": readback_count,
        "atomic_publication_count": atomic_publication_count,
        "temporary_allocation_traffic_bytes": temporary_allocation_traffic_bytes,
        "device_wait_seconds": device_wait_nanoseconds / 1_000_000_000.0,
        "gpu_timestamp_seconds": None,
        "gpu_timestamp_skip_reason": (
            "RHI timestamp queries are unavailable" if architecture != "cpu" else "CPU backend has no GPU timestamp"
        ),
        "tape_context_limit_bytes": tape_context_limit_bytes,
        "logical_bytes_per_active_cell": logical_bytes / (grid * grid),
        "pass_telemetry": pass_telemetry,
        "gradient_summary": gradient_summary,
        "forward_summary": forward_summary,
    }


def _markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Cost-aware smoke autodiff benchmark",
        "",
        "Warmed forward/backward smoke runs. Pressure iterations and measured steps are recorded per row; "
        f"tape context limit={BENCHMARK_TAPE_CONTEXT_LIMIT} bytes.",
        "",
        f"Frozen baseline: {BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL:.1f} logical bytes per active cell; "
        f"compiler Tape hints per lane={BASELINE_COMPILER_TAPE_HINTS}.",
        "",
        "| Policy | Backend | Grid | Pressure | Steps | Logical residual | Logical/cell | "
        "Resident | Allocated | Retained allocation | "
        "Temporary Tape peak | "
        "Checkpoint | Peak managed | "
        "Resident/logical | RSS | "
        "RSS delta | Stable RSS | Forward | Backward | Recompute | Python callbacks |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if result.get("skipped") or result.get("failed"):
            state = "skipped" if result.get("skipped") else "failed"
            cells = [result["planning_policy"], result["architecture"], result["grid"], state, *([""] * 17)]
            lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
            continue
        lines.append(
            "| {planning_policy} | {architecture} | {grid} | {pressure_iterations} | {steps} | "
            "{logical_residual_bytes} | "
            "{logical_bytes_per_active_cell:.3f} | "
            "{resident_bytes} | {allocated_bytes} | {retained_allocation_bytes} | {peak_temporary_tape_bytes} | "
            "{native_checkpoint_bytes} | {peak_runtime_managed_bytes} | {resident_to_logical_ratio:.3f} | "
            "{rss_bytes} | {rss_delta_bytes} | {rss_stable_high_water_mark} | "
            "{forward_seconds:.6f}s | {backward_seconds:.6f}s | "
            "{recomputation_factor:.3f} | {python_reverse_callback_count} |".format(**result)
        )
    lines.extend(("", "## Control-plane telemetry", ""))
    for result in results:
        label = f"{result['planning_policy']} / {result['architecture']} / {result['grid']}"
        if result.get("skipped") or result.get("failed"):
            reason = result.get("skip_reason") or result.get("failure_reason", "unknown failure")
            state = "skipped" if result.get("skipped") else "failed"
            lines.append(f"- {label}: {state} — {reason}")
            continue
        lines.append(
            f"- {label}: submissions={result['submission_count']}, waits={result['wait_count']}, "
            f"readbacks={result['readback_count']}, atomic publications={result['atomic_publication_count']}, "
            f"temporary allocation traffic={result['temporary_allocation_traffic_bytes']} bytes, "
            f"device wait={result['device_wait_seconds']:.6f}s; GPU timestamp skipped "
            f"({result['gpu_timestamp_skip_reason']})."
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


def _summary_close(cpu: dict[str, Any], gpu: dict[str, Any], *, rtol: float = 2.0e-4, atol: float = 2.0e-5) -> bool:
    if cpu["shape"] != gpu["shape"] or len(cpu["samples"]) != len(gpu["samples"]):
        return False
    for name in ("l2", "maximum_absolute"):
        if not np.isclose(cpu[name], gpu[name], rtol=rtol, atol=atol):
            return False
    element_count = math.prod(cpu["shape"])
    if not np.isclose(cpu["sum"], gpu["sum"], rtol=rtol, atol=atol * math.sqrt(max(element_count, 1))):
        return False
    return bool(np.allclose(cpu["samples"], gpu["samples"], rtol=rtol, atol=atol))


def _compare_with_cpu(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cpu_by_policy_grid = {
        (result["planning_policy"], int(result["grid"])): result
        for result in results
        if result.get("architecture") == "cpu" and not result.get("skipped") and not result.get("failed")
    }
    comparisons: list[dict[str, Any]] = []
    for result in results:
        if result.get("architecture") == "cpu":
            continue
        grid = int(result["grid"])
        if result.get("skipped") or result.get("failed"):
            comparisons.append(
                {
                    "architecture": result["architecture"],
                    "planning_policy": result["planning_policy"],
                    "grid": grid,
                    "skipped": bool(result.get("skipped")),
                    "failed": bool(result.get("failed")),
                    "skip_reason": result.get("skip_reason"),
                    "failure_reason": result.get("failure_reason"),
                }
            )
            continue
        planning_policy = result["planning_policy"]
        cpu = cpu_by_policy_grid.get((planning_policy, grid))
        if cpu is None:
            raise RuntimeError(f"{planning_policy} GPU smoke grid {grid} has no CPU comparison")
        forward_match = all(
            _summary_close(cpu["forward_summary"][name], result["forward_summary"][name])
            for name in ("density", "velocity", "loss")
        )
        gradient_match = _summary_close(cpu["gradient_summary"], result["gradient_summary"])
        logical_match = result["logical_residual_bytes"] == cpu["logical_residual_bytes"]
        bounded_tape = result["resident_bytes"] == 0 and result["allocated_bytes"] == 0
        elapsed_ratio = result["elapsed_seconds"] / cpu["elapsed_seconds"]
        comparison = {
            "architecture": result["architecture"],
            "planning_policy": planning_policy,
            "grid": grid,
            "forward_matches_cpu": forward_match,
            "gradient_matches_cpu": gradient_match,
            "logical_residual_matches_cpu": logical_match,
            "bounded_device_tape": bounded_tape,
            "peak_temporary_tape_delta_bytes": result["peak_temporary_tape_bytes"] - cpu["peak_temporary_tape_bytes"],
            "elapsed_ratio_to_cpu": elapsed_ratio,
            "speedup_over_cpu": 1.0 / elapsed_ratio,
        }
        comparisons.append(comparison)
        failed = [name for name, value in comparison.items() if name.endswith(("_cpu", "_tape")) and not value]
        if failed:
            raise RuntimeError(f"{result['architecture']} grid {grid} failed CPU smoke comparison: {', '.join(failed)}")
    return comparisons


def _comparison_markdown(comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# GPU smoke autodiff comparison",
        "",
        "GPU results are checked against the CPU run at the same grid size.",
        "",
        "| Policy | Backend | Grid | Forward | Gradient | Logical residual | Bounded Tape | "
        "Temporary Tape delta | GPU time / CPU time | Speedup over CPU |",
        "| :--- | :--- | ---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: |",
    ]
    for comparison in comparisons:
        if comparison.get("skipped") or comparison.get("failed"):
            state = "skipped" if comparison.get("skipped") else "failed"
            lines.append(
                f"| {comparison['planning_policy']} | {comparison['architecture']} | {comparison['grid']} | "
                f"{state} |  |  |  |  |  |  |"
            )
            continue
        lines.append(
            "| {planning_policy} | {architecture} | {grid} | {forward_matches_cpu} | {gradient_matches_cpu} | "
            "{logical_residual_matches_cpu} | {bounded_device_tape} | "
            "{peak_temporary_tape_delta_bytes} | {elapsed_ratio_to_cpu:.3f} | {speedup_over_cpu:.3f}× |".format(
                **comparison
            )
        )
    return "\n".join(lines) + "\n"


def _performance_acceptance(
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> dict[str, Any]:
    def median_seconds(result: dict[str, Any]) -> float:
        latency = result.get("elapsed_latency")
        if latency is not None:
            return float(latency["median_seconds"])
        return float(result["elapsed_seconds"]) / int(result["steps"])

    checks: list[dict[str, Any]] = []
    for result in results:
        if result.get("failed"):
            checks.append(
                {
                    "name": "benchmark_case_completed",
                    "planning_policy": result["planning_policy"],
                    "architecture": result["architecture"],
                    "grid": result["grid"],
                    "passed": False,
                    "failure_reason": result["failure_reason"],
                }
            )
        if result.get("skipped") or result.get("failed") or result["architecture"] == "cpu":
            continue
        steps = int(result["steps"])
        pass_count = len(result["pass_telemetry"])
        maximum_control_events = 3 * pass_count * steps
        for metric in ("submission_count", "wait_count", "readback_count"):
            measured = int(result[metric])
            checks.append(
                {
                    "name": "bounded_gpu_replay_control_plane",
                    "metric": metric,
                    "planning_policy": result["planning_policy"],
                    "architecture": result["architecture"],
                    "grid": result["grid"],
                    "measured": measured,
                    "maximum": maximum_control_events,
                    "passed": measured <= maximum_control_events,
                }
            )
    for comparison in comparisons:
        if comparison.get("skipped") or comparison.get("failed") or comparison["grid"] < 256:
            continue
        passed = comparison["speedup_over_cpu"] > 1.0
        checks.append(
            {
                "name": "gpu_faster_than_same_policy_cpu",
                "planning_policy": comparison["planning_policy"],
                "architecture": comparison["architecture"],
                "grid": comparison["grid"],
                "measured_speedup": comparison["speedup_over_cpu"],
                "passed": passed,
            }
        )
    baseline_by_key = {
        (item.get("planning_policy"), item.get("architecture"), int(item["grid"])): item
        for item in baseline_results
        if not item.get("skipped") and not item.get("failed")
    }
    for result in results:
        if result.get("skipped") or result.get("failed") or result["architecture"] == "cpu":
            continue
        key = (result["planning_policy"], result["architecture"], int(result["grid"]))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        current_median = median_seconds(result)
        baseline_median = median_seconds(baseline)
        ratio = current_median / baseline_median
        checks.append(
            {
                "name": "within_ten_percent_of_reproducible_gpu_baseline",
                "planning_policy": result["planning_policy"],
                "architecture": result["architecture"],
                "grid": result["grid"],
                "measured_ratio": ratio,
                "passed": ratio <= 1.1,
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cost-aware autodiff smoke grids")
    parser.add_argument("--grids", type=int, nargs="+", default=DEFAULT_GRIDS)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURES,
        help="run CPU first, then compare each requested GPU backend against it",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--gpu-comparison-output", type=Path)
    parser.add_argument("--gpu-comparison-markdown", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--enforce-performance", action="store_true")
    parser.add_argument("--worker-grid", type=int)
    parser.add_argument("--worker-architecture", choices=SUPPORTED_ARCHITECTURES, default="cpu")
    parser.add_argument("--pressure-iterations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--planning-policies",
        nargs="+",
        choices=("min_memory", "balanced", "min_runtime"),
        default=("min_memory",),
    )
    parser.add_argument(
        "--worker-planning-policy",
        choices=("min_memory", "balanced", "min_runtime"),
        default="min_memory",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-gradient-parity",
        action="store_true",
        help="skip the finite-difference correctness gate (intended only for profiling iterations)",
    )
    arguments = parser.parse_args()
    if arguments.worker_grid is not None:
        print(
            json.dumps(
                _run_grid(
                    arguments.worker_grid,
                    arguments.pressure_iterations,
                    arguments.steps,
                    arguments.worker_architecture,
                    arguments.worker_planning_policy,
                ),
                sort_keys=True,
            )
        )
        return
    if any(grid <= 0 for grid in arguments.grids) or arguments.pressure_iterations < 0 or arguments.steps <= 0:
        raise ValueError("grids and steps must be positive; pressure iterations must be non-negative")
    if not arguments.skip_gradient_parity:
        _run_gradient_parity_gate()

    architectures = list(dict.fromkeys(arguments.architectures))
    if any(architecture != "cpu" for architecture in architectures) and "cpu" not in architectures:
        architectures.insert(0, "cpu")
    planning_policies = list(dict.fromkeys(arguments.planning_policies))
    results: list[dict[str, Any]] = []
    for planning_policy in planning_policies:
        for architecture in architectures:
            for grid in arguments.grids:
                completed = subprocess.run(
                    (
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-grid",
                        str(grid),
                        "--worker-architecture",
                        architecture,
                        "--pressure-iterations",
                        str(arguments.pressure_iterations),
                        "--steps",
                        str(arguments.steps),
                        "--worker-planning-policy",
                        planning_policy,
                    ),
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    result = {
                        "architecture": architecture,
                        "planning_policy": planning_policy,
                        "grid": grid,
                        "failed": True,
                        "failure_reason": detail,
                    }
                else:
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                results.append(result)
                print(json.dumps(result, sort_keys=True))
    comparisons = _compare_with_cpu(results)
    baseline_results: list[dict[str, Any]] = []
    if arguments.baseline is not None:
        baseline_results = json.loads(arguments.baseline.read_text())["results"]
    acceptance = _performance_acceptance(results, comparisons, baseline_results)
    report = {
        "schema_version": 5,
        "baseline": {
            "logical_bytes_per_active_cell": BASELINE_LOGICAL_BYTES_PER_ACTIVE_CELL,
            "compiler_tape_hints_per_lane": BASELINE_COMPILER_TAPE_HINTS,
            "gradient_parity_test": GRADIENT_PARITY_TEST,
        },
        "pressure_iterations": arguments.pressure_iterations,
        "steps": arguments.steps,
        "architectures": architectures,
        "planning_policies": planning_policies,
        "results": results,
        "cpu_gpu_comparisons": comparisons,
        "performance_acceptance": acceptance,
    }
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.markdown is not None:
        arguments.markdown.write_text(_markdown(results))
    comparison_report = {
        "schema_version": 2,
        "cpu_gpu_comparisons": comparisons,
    }
    if arguments.gpu_comparison_output is not None:
        arguments.gpu_comparison_output.write_text(json.dumps(comparison_report, indent=2, sort_keys=True) + "\n")
    if arguments.gpu_comparison_markdown is not None:
        arguments.gpu_comparison_markdown.write_text(_comparison_markdown(comparisons))
    if arguments.enforce_performance and not acceptance["passed"]:
        failed = [check for check in acceptance["checks"] if not check["passed"]]
        raise RuntimeError(f"performance acceptance failed: {json.dumps(failed, sort_keys=True)}")


if __name__ == "__main__":
    main()
