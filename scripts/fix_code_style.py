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


def changed_files(base_revision: str) -> list[Path]:
    return git_files("diff", "--name-only", "--diff-filter=ACMR", f"{base_revision}...HEAD")


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


def format_file(
    path: Path,
    clang_format: str,
    ruff: str,
    cmake_format: str,
    check: bool,
) -> bool:
    if not path.is_file():
        return False

    before = file_hash(path)
    if path.suffix in CPP_SUFFIXES:
        command = (
            [clang_format, "-style", "file", "--dry-run", "--Werror", str(path)]
            if check
            else [clang_format, "-style", "file", "-i", str(path)]
        )
        run(command, "Checking C/C++" if check else "Formatting C/C++", path)
    elif path.suffix == ".py":
        if check:
            run([ruff, "check", "--no-fix", str(path)], "Linting Python", path)
            run([ruff, "format", "--check", str(path)], "Checking Python format", path)
        else:
            run([ruff, "check", "--fix", "--exit-zero", str(path)], "Linting Python", path)
            run([ruff, "format", str(path)], "Formatting Python", path)
    elif is_cmake_file(path):
        command = [cmake_format, "--check", str(path)] if check else [cmake_format, "-i", str(path)]
        run(command, "Checking CMake" if check else "Formatting CMake", path)
    else:
        return False
    return not check and before != file_hash(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply VernonDSL code formatting")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--all",
        action="store_true",
        help="format every tracked source file instead of only staged files",
    )
    scope.add_argument(
        "--since",
        metavar="REVISION",
        help="format files changed since the merge base with REVISION",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that require changes without rewriting them",
    )
    args = parser.parse_args(argv)

    clang_format = find_tool("clang-format")
    ruff = find_tool("ruff")
    cmake_format = find_tool("cmake-format")
    if args.all:
        files = tracked_files()
    elif args.since:
        files = changed_files(args.since)
    else:
        files = staged_files()
    if not files:
        print("No files to format.")
        return 0

    for path in files:
        changed = format_file(path, clang_format, ruff, cmake_format, args.check)
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
