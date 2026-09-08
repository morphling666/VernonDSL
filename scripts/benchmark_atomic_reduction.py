from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "python"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np  # noqa: E402
import vernon_dsl as vd  # noqa: E402

WORKGROUP_SIZE = 64
DEFAULT_INVOCATIONS = (128, 256, 512, 1024)
DEFAULT_ARCHITECTURES = ("cpu", "metal", "vulkan")
PATTERNS = ("one_bin", "few_bin", "low_collision", "workgroup_publication")


@vd.kernel(workgroup_size=(WORKGROUP_SIZE, 1, 1))
def atomic_scatter(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],
    bin_count: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = vd.i32(gid[0]) % bin_count
    vd.atomic_add(values, index, 1.0)


@vd.kernel(workgroup_size=(WORKGROUP_SIZE, 1, 1))
def workgroup_publication(
    values: vd.TensorView[vd.f32, (1,), vd.read_write],
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
) -> None:
    partial = vd.workgroup_storage(vd.f32, shape=(1,))
    if lane_id[0] == 0:
        partial[0] = 0.0
    vd.workgroup_barrier()
    vd.atomic_add(partial, 0, 1.0)
    vd.workgroup_barrier()
    if lane_id[0] == 0:
        vd.atomic_add(values, 0, partial[0])


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    p95 = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(ordered),
        "median_seconds": statistics.median(ordered),
        "p95_seconds": ordered[p95],
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
    }


def _bin_count(pattern: str, invocations: int) -> int:
    if pattern in {"one_bin", "workgroup_publication"}:
        return 1
    if pattern == "few_bin":
        return min(8, invocations)
    return invocations


def _run_case(architecture: str, pattern: str, invocations: int, iterations: int) -> dict[str, Any]:
    try:
        vd.init(arch=getattr(vd, architecture))
    except RuntimeError as error:
        return {
            "architecture": architecture,
            "pattern": pattern,
            "invocations": invocations,
            "skipped": True,
            "skip_reason": str(error),
        }
    if invocations % WORKGROUP_SIZE:
        raise ValueError(f"invocations must be divisible by {WORKGROUP_SIZE}")
    bins = _bin_count(pattern, invocations)
    grid = (invocations // WORKGROUP_SIZE, 1, 1)

    def dispatch() -> np.ndarray:
        output = vd.storage.zeros(dtype=vd.f32, shape=(bins,))
        if pattern == "workgroup_publication":
            workgroup_publication(output, grid=grid)
        else:
            atomic_scatter(output, bins, grid=grid)
        return output.to_numpy()

    warmup = dispatch()
    if not np.isclose(float(np.sum(warmup, dtype=np.float64)), float(invocations), rtol=0.0, atol=1e-4):
        raise RuntimeError("atomic benchmark warmup failed numerical validation")
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = dispatch()
        samples.append(time.perf_counter() - started)
        if not np.isclose(float(np.sum(result, dtype=np.float64)), float(invocations), rtol=0.0, atol=1e-4):
            raise RuntimeError("atomic benchmark failed numerical validation")
    return {
        "architecture": architecture,
        "pattern": pattern,
        "invocations": invocations,
        "workgroup_size": WORKGROUP_SIZE,
        "bin_count": bins,
        "selected_strategy": "workgroup_publication" if pattern == "workgroup_publication" else "target_atomic",
        "latency": _summary(samples),
        "submission_count": iterations,
        "wait_count": iterations,
        "readback_count": iterations,
        "atomic_publication_count": (
            (invocations // WORKGROUP_SIZE) * iterations
            if pattern == "workgroup_publication"
            else invocations * iterations
        ),
        "temporary_allocation_traffic_bytes": bins * np.dtype(np.float32).itemsize * iterations,
        "gpu_timestamp_seconds": None,
        "gpu_timestamp_skip_reason": (
            "RHI timestamp queries are unavailable" if architecture != "cpu" else "CPU backend has no GPU timestamp"
        ),
        "sum": float(np.sum(result, dtype=np.float64)),
        "skipped": False,
    }


def _worker(args: argparse.Namespace) -> int:
    result = _run_case(args.worker_architecture, args.worker_pattern, args.worker_invocations, args.iterations)
    Path(args.worker_output).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Atomic and Workgroup Publication Benchmark",
        "",
        f"- Iterations per case: {payload['iterations']}",
        f"- Workgroup size: {payload['workgroup_size']}",
        "",
    ]
    for result in payload["results"]:
        label = f"{result['architecture']} / {result['pattern']} / {result['invocations']}"
        if result.get("skipped"):
            lines.append(f"- {label}: skipped — {result['skip_reason']}")
            continue
        latency = result["latency"]
        lines.append(
            f"- {label}: median {latency['median_seconds']:.6f}s, "
            f"p95 {latency['p95_seconds']:.6f}s, sum {result['sum']:.1f}, "
            f"submissions/waits/readbacks={result['submission_count']}/"
            f"{result['wait_count']}/{result['readback_count']}, "
            f"atomic publications={result['atomic_publication_count']}, "
            f"temporary allocation traffic={result['temporary_allocation_traffic_bytes']} bytes; "
            f"GPU timestamp skipped ({result['gpu_timestamp_skip_reason']})"
        )
    lines.extend(("", "## Uniform-contention crossover", ""))
    for comparison in payload["uniform_contention_comparisons"]:
        lines.append(
            f"- {comparison['architecture']} / {comparison['invocations']}: "
            f"workgroup/direct ratio {comparison['workgroup_to_direct_median_ratio']:.3f}, "
            f"recommended {comparison['recommended_strategy']}"
        )
    lines.append("")
    return "\n".join(lines)


def _uniform_contention_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (result["architecture"], result["invocations"], result["pattern"]): result
        for result in results
        if not result.get("skipped")
    }
    comparisons: list[dict[str, Any]] = []
    architectures = sorted({result["architecture"] for result in results})
    invocations = sorted({result["invocations"] for result in results})
    for architecture in architectures:
        for invocation_count in invocations:
            direct = indexed.get((architecture, invocation_count, "one_bin"))
            workgroup = indexed.get((architecture, invocation_count, "workgroup_publication"))
            if not direct or not workgroup:
                continue
            direct_median = direct["latency"]["median_seconds"]
            workgroup_median = workgroup["latency"]["median_seconds"]
            ratio = workgroup_median / direct_median
            comparisons.append(
                {
                    "architecture": architecture,
                    "invocations": invocation_count,
                    "workgroup_to_direct_median_ratio": ratio,
                    "recommended_strategy": "workgroup_reduction" if ratio <= 1.0 else "direct_atomic",
                }
            )
    return comparisons


def _parent(args: argparse.Namespace) -> int:
    results: list[dict[str, Any]] = []
    for architecture in args.architectures:
        for pattern in args.patterns:
            for invocations in args.invocations:
                with tempfile.TemporaryDirectory() as directory:
                    worker_output = Path(directory) / "result.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-architecture",
                        architecture,
                        "--worker-pattern",
                        pattern,
                        "--worker-invocations",
                        str(invocations),
                        "--worker-output",
                        str(worker_output),
                        "--iterations",
                        str(args.iterations),
                    ]
                    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
                    if completed.returncode != 0:
                        detail = completed.stderr.strip() or completed.stdout.strip()
                        raise RuntimeError(f"atomic benchmark worker failed: {detail}")
                    results.append(json.loads(worker_output.read_text()))
    comparisons = _uniform_contention_comparisons(results)
    measured_gpu_reduction = any(
        item["architecture"] != "cpu" and item["recommended_strategy"] == "workgroup_reduction" for item in comparisons
    )
    payload = {
        "schema_version": 2,
        "iterations": args.iterations,
        "workgroup_size": WORKGROUP_SIZE,
        "results": results,
        "uniform_contention_comparisons": comparisons,
        "cost_model": {
            "native_atomic_reduction_crossover": WORKGROUP_SIZE,
            "integer_cas_reduction_crossover": WORKGROUP_SIZE,
            "evidence": (
                "64 is the smallest measured workgroup and publication wins on available GPU backends"
                if measured_gpu_reduction
                else "no available GPU crossover; conservative smallest measured workgroup retained"
            ),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(_markdown(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS))
    parser.add_argument("--invocations", nargs="+", type=int, default=list(DEFAULT_INVOCATIONS))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="reports/benchmarks/atomic_reduction_benchmark.json")
    parser.add_argument("--markdown", default="reports/benchmarks/atomic_reduction_benchmark.md")
    parser.add_argument("--worker-architecture")
    parser.add_argument("--worker-pattern", choices=PATTERNS)
    parser.add_argument("--worker-invocations", type=int)
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.worker_architecture:
        if not args.worker_pattern or not args.worker_invocations or not args.worker_output:
            parser.error("worker mode requires pattern, invocations, and output")
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
