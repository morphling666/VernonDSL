from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "tests"))

from backend_test_matrix import BACKEND_TEST_MATRIX  # noqa: E402

BACKENDS = tuple(row.runtime_backend.lower() for row in BACKEND_TEST_MATRIX)
LAYERS = {"binding", "program", "backend", "autodiff", "graphics", "application"}
SURFACES = ("python", "cpp")
MEASUREMENT_FIELDS = {"name", "unit", "sample_count", "median", "p95", "p99", "minimum", "maximum"}
MEASUREMENT_UNITS = {"ns", "bytes", "count", "frames_per_second"}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def _command(value: Any, description: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(part, str) and part for part in value):
        raise ValueError(f"{description} must be null or a non-empty string array")
    return value


def _validate_catalog(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("catalog_version") != 2:
        raise ValueError("fixture catalog must have catalog_version 2")
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixture catalog must contain a non-empty fixtures array")
    seen: set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("each fixture must be an object")
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or "." not in fixture_id or fixture_id in seen:
            raise ValueError(f"invalid or duplicate fixture id: {fixture_id!r}")
        seen.add(fixture_id)
        if fixture.get("layer") not in LAYERS:
            raise ValueError(f"{fixture_id}: invalid benchmark layer")
        drivers = fixture.get("drivers")
        measurements = fixture.get("measurements")
        if not isinstance(drivers, dict) or set(drivers) != set(SURFACES):
            raise ValueError(f"{fixture_id}: drivers must define python and cpp")
        if not isinstance(measurements, dict) or set(measurements) != set(SURFACES):
            raise ValueError(f"{fixture_id}: measurements must define python and cpp")
        for surface in SURFACES:
            _command(drivers[surface], f"{fixture_id}: {surface} driver")
            names = measurements[surface]
            if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
                raise ValueError(f"{fixture_id}: {surface} measurements must be a non-empty string array")
        variants = fixture.get("variants")
        if not isinstance(variants, list) or not variants or not all(isinstance(variant, dict) for variant in variants):
            raise ValueError(f"{fixture_id}: variants must be a non-empty object array")
    return fixtures


def _selected_fixtures(fixtures: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        fixture
        for fixture in fixtures
        if (not args.layer or fixture["layer"] == args.layer) and (not args.fixture or fixture["id"] == args.fixture)
    ]


def _print_catalog(fixtures: list[dict[str, Any]]) -> None:
    for fixture in fixtures:
        surfaces = ",".join(surface for surface in SURFACES if fixture["drivers"][surface])
        state = "runnable" if surfaces else "planned"
        print(f"{fixture['id']}\t{fixture['layer']}\t{state}\t{surfaces or '-'}")


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment(configuration: str, backend: str, surface: str) -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "revision": _git_output("rev-parse", "HEAD"),
        "dirty": bool(_git_output("status", "--short")),
        "platform": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "configuration": configuration,
        "backend": backend,
        "surface": surface,
        "device": None,
        "driver": None,
    }


def _validate_driver_result(fixture: dict[str, Any], surface: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("status") not in {"completed", "skipped"}:
        raise ValueError(f"{fixture['id']}/{surface}: invalid driver result")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError(f"{fixture['id']}/{surface}: measurements must be an array")
    if payload["status"] == "skipped":
        if measurements or not isinstance(payload.get("skip_reason"), str) or not payload["skip_reason"]:
            raise ValueError(f"{fixture['id']}/{surface}: skipped result requires a reason and no measurements")
        return payload
    names = [measurement.get("name") for measurement in measurements if isinstance(measurement, dict)]
    if len(names) != len(measurements) or len(set(names)) != len(names):
        raise ValueError(f"{fixture['id']}/{surface}: malformed or duplicate measurements")
    missing = set(fixture["measurements"][surface]) - set(names)
    if missing:
        raise ValueError(f"{fixture['id']}/{surface}: omitted measurements: {', '.join(sorted(missing))}")
    for measurement in measurements:
        if set(measurement) != MEASUREMENT_FIELDS or measurement["unit"] not in MEASUREMENT_UNITS:
            raise ValueError(f"{fixture['id']}/{surface}: malformed measurement")
        count = measurement["sample_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{fixture['id']}/{surface}: invalid sample count")
        for statistic in ("median", "p95", "p99", "minimum", "maximum"):
            value = measurement[statistic]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{fixture['id']}/{surface}: invalid {statistic}")
    counters = payload.get("counters", {})
    if not isinstance(counters, dict) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters.values()
    ):
        raise ValueError(f"{fixture['id']}/{surface}: counters must be non-negative integers")
    return payload


def _run_case(
    fixture: dict[str, Any],
    backend: str,
    surface: str,
    parameters: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
) -> None:
    driver = fixture["drivers"][surface]
    if driver is None:
        raise ValueError(f"{fixture['id']}: {surface} surface is planned but has no driver")
    substitutions = {
        "repository": str(REPOSITORY_ROOT),
        "build": str(args.build.resolve()),
        "python": sys.executable,
    }
    command = [part.format_map(substitutions) for part in driver]
    command.extend(
        (
            "--fixture",
            fixture["id"],
            "--backend",
            backend,
            "--parameters-json",
            json.dumps(parameters, separators=(",", ":"), sort_keys=True),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        )
    )
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{fixture['id']}/{surface}/{backend}: driver failed: {detail}")
    try:
        payload = _validate_driver_result(fixture, surface, json.loads(completed.stdout))
    except json.JSONDecodeError as error:
        raise ValueError(f"{fixture['id']}/{surface}/{backend}: driver did not emit one JSON object") from error
    result = {
        "schema_version": 2,
        "fixture": fixture["id"],
        "layer": fixture["layer"],
        "status": payload["status"],
        "environment": _environment(args.configuration, backend, surface),
        "parameters": parameters,
        "measurements": payload["measurements"],
    }
    for optional in ("skip_reason", "counters"):
        if optional in payload:
            result[optional] = payload[optional]
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run opt-in Vernon benchmarks")
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("fixtures.json"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--layer", choices=sorted(LAYERS))
    parser.add_argument("--fixture")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--surface", choices=SURFACES)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--configuration", default="release")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_arguments()
    try:
        fixtures = _selected_fixtures(_validate_catalog(_load_json(args.catalog)), args)
        if not fixtures:
            raise ValueError("no benchmark fixture matched the selection")
        if args.list:
            _print_catalog(fixtures)
            return 0
        if args.validate:
            return 0
        if args.build is None:
            raise ValueError("--build is required when running benchmarks")
        if args.warmup < 0 or args.iterations < 1:
            raise ValueError("--warmup must be non-negative and --iterations must be positive")
        selected_surfaces = (args.surface,) if args.surface else SURFACES
        planned = [
            f"{fixture['id']}/{surface}"
            for fixture in fixtures
            for surface in selected_surfaces
            if fixture["drivers"][surface] is None
        ]
        if planned:
            raise ValueError(f"selected benchmark surfaces have no drivers: {', '.join(planned)}")
        args.output.mkdir(parents=True, exist_ok=True)
        for fixture in fixtures:
            for backend in (args.backend,) if args.backend else BACKENDS:
                for surface in selected_surfaces:
                    for variant_index, parameters in enumerate(fixture["variants"]):
                        output = args.output / f"{fixture['id']}-{backend}-{surface}-{variant_index}.json"
                        _run_case(fixture, backend, surface, parameters, args, output)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
