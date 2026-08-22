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
COMPUTE_BACKENDS = ("cpu", "vulkan", "metal")
GRAPHICS_BACKENDS = ("vulkan", "metal", "opengl", "directx")


def _probe(backend: str) -> tuple[bool, str]:
    initialization = (
        f"import vernon_dsl as vd; vd.init(arch=vd.{backend}, api_version={(4, 3) if backend == 'opengl' else None!r})"
    )
    completed = subprocess.run([sys.executable, "-c", initialization], cwd=ROOT, capture_output=True, text=True)
    return completed.returncode == 0, (completed.stderr or completed.stdout).strip()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)


def _require_image(path: Path, *, minimum_std: float = 0.25) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"{path} is not a readable image")
    rgb = image[..., :3].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    if float(np.std(rgb)) < minimum_std:
        raise ValueError(f"{path} is effectively uniform")


def _run_compute(backend: str, output: Path) -> dict[str, Any]:
    fractal = output / f"fractal-{backend}.webp"
    _run(
        [
            sys.executable,
            str(ROOT / "examples" / "fractal.py"),
            "--arch",
            backend,
            "--headless",
            "--frames",
            "1",
            "--animation-output",
            str(fractal),
        ]
    )
    if fractal.exists():
        _require_image(fractal)
    shared = _run([sys.executable, str(ROOT / "examples" / "shared_struct_methods.py"), "--arch", backend])
    if "host contribution:" not in shared.stdout:
        raise ValueError("shared_struct_methods.py did not print a host contribution")
    return {"fractal": str(fractal) if fractal.exists() else None, "shared": True}


def _run_graphics(name: str, backend: str, output: Path, extra: list[str] | None = None) -> dict[str, Any]:
    png = output / f"{name}-{backend}.png"
    command = [
        sys.executable,
        str(ROOT / "examples" / f"{name}.py"),
        "--arch",
        backend,
        "--headless",
        "--frames",
        "1",
        "--output",
        str(png),
        *(extra or []),
    ]
    _run(command)
    if png.exists():
        _require_image(png)
    return {"image": str(png) if png.exists() else None}


def _run_showcase(name: str, backend: str, output: Path) -> dict[str, Any]:
    from run_showcase_smoke import _run_showcase as run_one

    return run_one(name, backend, 64, output)


def _compile_only(script: Path, backend: str) -> None:
    _run([sys.executable, str(script), "--arch", backend, "--headless", "--frames", "1"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic non-AD VernonDSL example acceptance.")
    parser.add_argument("--backend", action="append", dest="backends")
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "examples-acceptance")
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))

    records: list[dict[str, Any]] = []
    failed = False
    backends = arguments.backends or sorted(set(COMPUTE_BACKENDS + GRAPHICS_BACKENDS))
    for backend in backends:
        available, detail = _probe(backend)
        record: dict[str, Any] = {"backend": backend}
        try:
            if available:
                if backend in COMPUTE_BACKENDS:
                    record["compute"] = _run_compute(backend, arguments.output_dir)
                if backend in GRAPHICS_BACKENDS:
                    record["graphics"] = {
                        name: _run_graphics(name, backend, arguments.output_dir)
                        for name in ("unified_pipeline", "complete_pipeline", "advanced_pipeline", "pbr")
                    }
                    record["showcases"] = [
                        _run_showcase(name, backend, arguments.output_dir)["showcase"]
                        for name in ("terrain", "mandelbulb")
                    ]
                record["status"] = "executed"
            else:
                record["status"] = "compiled"
                record["reason"] = detail or "initialization failed"
                for script in (
                    ROOT / "examples" / "fractal.py",
                    ROOT / "examples" / "unified_pipeline.py",
                ):
                    try:
                        _compile_only(script, backend)
                    except subprocess.CalledProcessError as error:
                        record["status"] = "failed"
                        record["reason"] = error.stderr or str(error)
                        failed = True
                        break
                failed = failed or arguments.required
        except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
            record["status"] = "failed"
            record["reason"] = str(error)
            failed = True
        records.append(record)

    summary = {"backends": records}
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(encoded + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
