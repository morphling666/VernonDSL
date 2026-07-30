from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHOWCASES = ("terrain", "mandelbulb")
BACKENDS = ("vulkan", "directx", "opengl")


def _probe_backend(backend: str) -> tuple[bool, str]:
    initialization = (
        f"import vernon_dsl as vd; vd.init(arch=vd.{backend}, api_version={(4, 3) if backend == 'opengl' else None!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", initialization],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    detail = (completed.stderr or completed.stdout).strip()
    return completed.returncode == 0, detail


def _validate_result(name: str, backend: str, size: int, png: Path, result_json: Path) -> dict[str, Any]:
    result = json.loads(result_json.read_text(encoding="utf-8"))
    required = {
        "showcase",
        "backend",
        "preset",
        "frames",
        "size",
        "fps",
        "elapsed_seconds",
        "passes",
        "barriers",
        "image",
    }
    if set(result) != required:
        raise ValueError(f"{name} result fields differ: {sorted(set(result) ^ required)}")
    if result["showcase"] != name or result["backend"] != backend or result["preset"] != "smoke":
        raise ValueError(f"{name} result identity does not match the invocation")
    if result["frames"] < 1 or result["size"] != size or result["elapsed_seconds"] < 0.0:
        raise ValueError(f"{name} result contains invalid frame, size, or timing values")

    image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if image is None or list(image.shape) != [size, size, 4]:
        raise ValueError(f"{name} PNG must have shape {[size, size, 4]}")
    alpha = image[..., 3]
    rgb = image[..., :3].astype(np.float32)
    if np.count_nonzero(alpha) != size * size:
        raise ValueError(f"{name} PNG must have full nonzero alpha")
    if float(np.std(rgb)) < 0.25:
        raise ValueError(f"{name} PNG is effectively uniform")

    statistics = result["image"]
    if statistics.get("shape") != [size, size, 4]:
        raise ValueError(f"{name} JSON image shape does not match the PNG")
    if statistics.get("nonzero_alpha") != size * size or statistics.get("stddev", 0.0) < 0.25:
        raise ValueError(f"{name} JSON image statistics failed acceptance")
    return result


def _run_showcase(name: str, backend: str, size: int, output: Path) -> dict[str, Any]:
    png = output / f"{name}-{backend}.png"
    result_json = output / f"{name}-{backend}.json"
    command = [
        sys.executable,
        str(ROOT / "examples" / f"{name}_showcase.py"),
        "--arch",
        backend,
        "--preset",
        "smoke",
        "--size",
        str(size),
        "--frames",
        "1",
        "--headless",
        "--output",
        str(png),
        "--result-json",
        str(result_json),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return _validate_result(name, backend, size, png, result_json)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU-optional VernonDSL showcase acceptance smoke tests.")
    parser.add_argument("--backend", action="append", choices=BACKENDS, dest="backends")
    parser.add_argument("--required", action="store_true", help="fail if any requested backend is unavailable")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "showcase-smoke")
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()
    if arguments.size <= 0:
        parser.error("--size must be positive")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failed = False
    for backend in arguments.backends or BACKENDS:
        available, detail = _probe_backend(backend)
        if not available:
            records.append({"backend": backend, "status": "skipped", "reason": detail or "initialization failed"})
            failed = failed or arguments.required
            continue
        try:
            results = [_run_showcase(name, backend, arguments.size, arguments.output_dir) for name in SHOWCASES]
        except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
            records.append({"backend": backend, "status": "failed", "reason": str(error)})
            failed = True
            continue
        records.append(
            {
                "backend": backend,
                "status": "executed",
                "showcases": [result["showcase"] for result in results],
            }
        )

    summary = {"size": arguments.size, "backends": records}
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(encoded + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
