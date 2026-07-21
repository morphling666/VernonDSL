"""Locate the VernonRuntime sources bundled with this installation."""

from __future__ import annotations

import argparse
from pathlib import Path

_RUNTIME_SOURCE_DIR = Path(__file__).resolve().parent / "runtime_src"


def cmake_source_dir() -> Path:
    """Return the read-only CMake source directory for VernonRuntime."""
    cmake_file = _RUNTIME_SOURCE_DIR / "CMakeLists.txt"
    if not cmake_file.is_file():
        raise RuntimeError(
            "VernonRuntime sources are missing from this VernonDSL installation"
        )
    return _RUNTIME_SOURCE_DIR


def version() -> str:
    """Return the semantic version of the bundled runtime sources."""
    version_file = cmake_source_dir() / "VERSION"
    return version_file.read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the VernonRuntime sources bundled with VernonDSL"
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--cmake-dir",
        action="store_true",
        help="print the bundled VernonRuntime CMake source directory",
    )
    output.add_argument(
        "--version",
        action="store_true",
        help="print the bundled VernonRuntime source version",
    )
    args = parser.parse_args(argv)
    print(cmake_source_dir() if args.cmake_dir else version())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
