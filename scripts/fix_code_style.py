"""Format staged or tracked VernonDSL source files."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def find_tool(tool_name: str) -> str:
    executable = tool_name + (".exe" if os.name == "nt" else "")
    python_dir = Path(sys.executable).resolve().parent
    candidates = (
        python_dir / executable,
        python_dir / "Scripts" / executable,
        python_dir / "bin" / tool_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    discovered = shutil.which(tool_name)
    if discovered:
        return discovered
    raise FileNotFoundError(f"{tool_name} is not installed in the active Python environment")


def git_files(*args: str) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *args, "-z"],
        check=True,
        capture_output=True,
    )
    return [REPOSITORY_ROOT / os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


def staged_files() -> list[Path]:
    return git_files("diff", "--cached", "--name-only", "--diff-filter=ACMR")


def tracked_files() -> list[Path]:
    return git_files("ls-files")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_cmake_file(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix == ".cmake"


def run(command: list[str], description: str, path: Path) -> None:
    relative_path = path.relative_to(REPOSITORY_ROOT)
    print(f"{description}: {relative_path}")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def format_file(path: Path, clang_format: str, ruff: str, cmake_format: str) -> bool:
    if not path.is_file():
        return False

    before = file_hash(path)
    if path.suffix in CPP_SUFFIXES:
        run([clang_format, "-style", "file", "-i", str(path)], "Formatting C/C++", path)
    elif path.suffix == ".py":
        run([ruff, "check", "--fix", "--exit-zero", str(path)], "Linting Python", path)
        run([ruff, "format", str(path)], "Formatting Python", path)
    elif is_cmake_file(path):
        run([cmake_format, "-i", str(path)], "Formatting CMake", path)
    else:
        return False
    return before != file_hash(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply VernonDSL code formatting")
    parser.add_argument(
        "--all",
        action="store_true",
        help="format every tracked source file instead of only staged files",
    )
    args = parser.parse_args(argv)

    clang_format = find_tool("clang-format")
    ruff = find_tool("ruff")
    cmake_format = find_tool("cmake-format")
    files = tracked_files() if args.all else staged_files()
    if not files:
        print("No files to format.")
        return 0

    for path in files:
        changed = format_file(path, clang_format, ruff, cmake_format)
        if changed and not args.all:
            relative_path = path.relative_to(REPOSITORY_ROOT)
            print(f"Re-adding formatted file: {relative_path}")
            subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "add", "--", str(relative_path)],
                check=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
