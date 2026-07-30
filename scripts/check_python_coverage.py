from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def percentage(files: list[dict[str, Any]], covered: str, total: str) -> float:
    covered_count = sum(int(row["summary"][covered]) for row in files)
    total_count = sum(int(row["summary"][total]) for row in files)
    if total_count == 0:
        raise ValueError("coverage group has no measurable entries")
    return 100.0 * covered_count / total_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce Vernon Python frontend coverage groups.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--frontend-lines", type=float, default=90.0)
    parser.add_argument("--inference-branches", type=float, default=85.0)
    arguments = parser.parse_args()
    document = json.loads(arguments.report.read_text(encoding="utf-8"))
    rows = {name.replace("\\", "/"): value for name, value in document["files"].items()}
    frontend = [value for name, value in rows.items() if "/frontend/" in name or "/language/" in name]
    inference = [
        value
        for name, value in rows.items()
        if name.endswith(("/frontend/inference.py", "/frontend/type_parser.py", "/frontend/type_solver.py"))
    ]
    frontend_lines = percentage(frontend, "covered_lines", "num_statements")
    inference_branches = percentage(inference, "covered_branches", "num_branches")
    print(f"language/frontend line coverage: {frontend_lines:.2f}%")
    print(f"inference/type-parser branch coverage: {inference_branches:.2f}%")
    failures = []
    if frontend_lines < arguments.frontend_lines:
        failures.append(f"language/frontend lines require {arguments.frontend_lines:.2f}%")
    if inference_branches < arguments.inference_branches:
        failures.append(f"inference/type-parser branches require {arguments.inference_branches:.2f}%")
    if failures:
        for failure in failures:
            print(f"coverage gate failed: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
