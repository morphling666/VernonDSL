"""Build the external-engine example against Runtime sources from an installed wheel."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> int:
    repository = Path(__file__).resolve().parent.parent
    example = repository / "examples" / "external_engine"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(repository)

    runtime_source = Path(
        subprocess.check_output(
            [sys.executable, "-I", "-m", "vernon_dsl.runtime_source", "--cmake-dir"],
            text=True,
        ).strip()
    ).resolve()
    environment_prefix = Path(sys.prefix).resolve()
    if environment_prefix not in runtime_source.parents:
        raise RuntimeError(f"Runtime source is not from the active wheel environment: {runtime_source}")

    with tempfile.TemporaryDirectory(prefix="vernon-external-engine-") as temporary:
        build = Path(temporary) / "build"

        run(
            [
                "cmake",
                "-S",
                os.fspath(example),
                "-B",
                os.fspath(build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}",
            ],
            cwd=repository,
            environment=environment,
        )
        run(
            ["cmake", "--build", os.fspath(build), "--config", "Release", "--parallel", "3"],
            cwd=repository,
            environment=environment,
        )
        run(
            ["ctest", "--test-dir", os.fspath(build), "-C", "Release", "--output-on-failure"],
            cwd=repository,
            environment=environment,
        )

    print(f"External engine consumed installed Runtime sources from {runtime_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
