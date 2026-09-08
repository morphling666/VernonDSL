"""Configure the pinned LLVM/MLIR build on Linux, macOS, or Windows."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURES = {
    "aarch64": ("AArch64", "ARM64"),
    "amd64": ("X86", "x64"),
    "arm64": ("AArch64", "ARM64"),
    "i386": ("X86", "Win32"),
    "i686": ("X86", "Win32"),
    "x86": ("X86", "Win32"),
    "x86_64": ("X86", "x64"),
}


def host_architecture() -> tuple[str, str]:
    machine = platform.machine().lower()
    try:
        return ARCHITECTURES[machine]
    except KeyError as error:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise RuntimeError(f"unsupported host architecture {machine!r}; known values: {supported}") from error


def cached_generator(build_dir: Path) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_GENERATOR:INTERNAL="):
            return line.partition("=")[2]
    return None


def default_generator(build_dir: Path) -> str:
    existing = cached_generator(build_dir)
    if existing:
        return existing
    if os.name == "nt":
        return "Visual Studio 17 2022"
    if shutil.which("ninja"):
        return "Ninja"
    return "Unix Makefiles"


def libdevice_candidates(repository_root: Path) -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("MLIR_NVVM_LIBDEVICE_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(
        repository_root
        / "llvm-project"
        / "nvidia-nvcc"
        / "nvidia"
        / "cuda_nvcc"
        / "nvvm"
        / "libdevice"
        / "libdevice.10.bc"
    )

    roots: list[Path] = []
    for variable in ("CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT", "CONDA_PREFIX"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())

    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(Path(nvcc).resolve().parent.parent)

    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        roots.extend(
            sorted(
                (program_files / "NVIDIA GPU Computing Toolkit" / "CUDA").glob("v*"),
                reverse=True,
            )
        )
    else:
        roots.extend(
            [
                Path("/usr/local/cuda"),
                Path("/opt/cuda"),
                Path("/usr/lib/cuda"),
                Path("/usr/lib/nvidia-cuda-toolkit"),
            ]
        )
        roots.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))

    for root in roots:
        candidates.extend(
            [
                root / "nvvm" / "libdevice" / "libdevice.10.bc",
                root / "lib" / "nvvm" / "libdevice" / "libdevice.10.bc",
            ]
        )
    return candidates


def find_libdevice(repository_root: Path, requested_path: Path | None) -> Path | None:
    candidates = [requested_path] if requested_path else libdevice_candidates(repository_root)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the host architecture and CUDA toolkit, then configure LLVM/MLIR",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPOSITORY_ROOT / "llvm-project" / "llvm",
        help="LLVM CMake source directory",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPOSITORY_ROOT / "llvm-project" / "build",
        help="LLVM build directory",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=REPOSITORY_ROOT / "llvm-project" / "install",
        help="LLVM installation directory",
    )
    parser.add_argument("--generator", help="override the automatically selected CMake generator")
    parser.add_argument(
        "--cuda",
        choices=("auto", "on", "off"),
        default="auto",
        help="discover libdevice and add NVPTX automatically, require libdevice, or disable discovery (default: auto)",
    )
    parser.add_argument("--libdevice", type=Path, help="explicit path to libdevice.10.bc")
    parser.add_argument(
        "--targets",
        help="override LLVM targets (comma- or semicolon-separated); Native resolves to the host target",
    )
    parser.add_argument("--build", action="store_true", help="build and install LLVM after configuring")
    parser.add_argument("--parallel", type=int, default=os.cpu_count() or 1, help="parallel build jobs")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    install_dir = args.install_dir.resolve()
    if not (source_dir / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            f"LLVM sources not found at {source_dir}; run 'git submodule update --init --depth 1 llvm-project' first"
        )
    if not shutil.which("cmake"):
        raise FileNotFoundError("cmake was not found on PATH")

    llvm_host_target, windows_platform = host_architecture()
    generator = args.generator or default_generator(build_dir)
    libdevice = None if args.cuda == "off" else find_libdevice(REPOSITORY_ROOT, args.libdevice)
    if args.libdevice is not None and libdevice is None:
        raise FileNotFoundError(f"the requested libdevice file does not exist: {args.libdevice}")
    if args.cuda == "on" and libdevice is None:
        raise FileNotFoundError(
            "CUDA was requested, but libdevice.10.bc was not found; pass --libdevice PATH or set CUDA_PATH/CUDA_HOME"
        )

    explicit_targets = bool(args.targets)
    if explicit_targets:
        targets = [
            llvm_host_target if item.lower() == "native" else item
            for item in args.targets.replace(",", ";").split(";")
            if item
        ]
    else:
        targets = [llvm_host_target]
    if libdevice and "NVPTX" not in targets:
        targets.append("NVPTX")
    if not libdevice and not explicit_targets:
        targets = [target for target in targets if target != "NVPTX"]
    targets = list(dict.fromkeys(targets))

    command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-G",
        generator,
    ]
    if generator.startswith("Visual Studio"):
        command.extend(["-A", windows_platform])
    else:
        command.append("-DCMAKE_BUILD_TYPE=Release")
    command.extend(
        [
            "-DLLVM_ENABLE_PROJECTS=mlir",
            f"-DLLVM_TARGETS_TO_BUILD={';'.join(targets)}",
            "-DLLVM_ENABLE_ASSERTIONS=OFF",
            "-DLLVM_INCLUDE_TESTS=OFF",
            "-DMLIR_INCLUDE_TESTS=OFF",
            f"-DMLIR_NVVM_EMBED_LIBDEVICE={'ON' if libdevice else 'OFF'}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
    )
    if libdevice:
        command.append(f"-DMLIR_NVVM_LIBDEVICE_PATH={libdevice}")

    print(f"Host:      {platform.system()} {platform.machine()}")
    print(f"Generator: {generator}")
    print(f"Targets:   {';'.join(targets)}")
    print(f"CUDA:      {libdevice if libdevice else 'disabled (libdevice not found)'}")
    print(f"Configure: {format_command(command)}")
    if not args.dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    if args.build:
        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "install",
            "--parallel",
            str(args.parallel),
        ]
        print(f"Build:     {format_command(build_command)}")
        if not args.dry_run:
            subprocess.run(build_command, cwd=REPOSITORY_ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
