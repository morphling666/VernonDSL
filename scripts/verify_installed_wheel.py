from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import vernon_dsl as vd
from vernon_dsl._versions import RELEASE_VERSION
from vernon_dsl.runtime_source import cmake_source_dir, version
from wheel_smoke_kernel import add_one


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _verify_runtime_source_contents(source: Path) -> None:
    required = (
        "CMakeLists.txt",
        "VERSION",
        "VernonRuntimeConfig.cmake.in",
        "cmake/EmbedBinary.cmake",
        "cmake/IncludeNlohmannJson.cmake",
        "cmake/IncludeVulkanHeaders.cmake",
        "cmake/VernonRuntimeTarget.cmake",
        "cmake/VernonVersions.cmake",
        "include/VernonRuntime.h",
        "include/VernonGpuAutodiffAbi.h",
        "lib/rhi/cuda_backend.cpp",
        "lib/rhi/directx12_backend.cpp",
        "lib/rhi/directx12_mipmap.hlsl",
        "lib/rhi/metal_backend.mm",
        "lib/rhi/rhi_metal.mm",
        "lib/rhi/vulkan_backend.cpp",
        "lib/runtime/rhi_adapter/adapter_metal.mm",
    )
    missing = [relative for relative in required if not (source / relative).is_file()]
    if missing:
        raise RuntimeError(f"bundled VernonRuntime source is incomplete: {', '.join(missing)}")


def _verify_runtime_source_build(source: Path, root: Path) -> None:
    build = root / "runtime-build"
    install = root / "runtime-install"
    system = platform.system()
    configure = [
        "cmake",
        "-S",
        os.fspath(source),
        "-B",
        os.fspath(build),
        "-DBUILD_TESTING=OFF",
        "-DVERNON_RUNTIME_PROFILE=desktop",
        "-DVERNON_RUNTIME_LIBRARY_TYPE=STATIC",
        "-DVERNON_ENABLE_CUDA_RUNTIME=OFF",
        f"-DVERNON_ENABLE_VULKAN_RUNTIME={'ON' if system == 'Linux' else 'OFF'}",
        f"-DVERNON_ENABLE_DIRECTX12_RUNTIME={'ON' if system == 'Windows' else 'OFF'}",
        f"-DVERNON_ENABLE_METAL_RUNTIME={'ON' if system == 'Darwin' else 'OFF'}",
    ]
    packaged_dxc = source.parent / ("dxc.exe" if system == "Windows" else "dxc")
    if packaged_dxc.is_file():
        configure.append(f"-DVERNON_DXC_EXECUTABLE={packaged_dxc}")
    _run(configure)
    _run(["cmake", "--build", os.fspath(build), "--config", "Release", "--target", "VernonRuntime"])
    _run(
        [
            "cmake",
            "--install",
            os.fspath(build),
            "--config",
            "Release",
            "--prefix",
            os.fspath(install),
            "--component",
            "VernonDevelopment",
        ]
    )

    consumer = root / "runtime-consumer"
    consumer.mkdir()
    (consumer / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(VernonRuntimeWheelSmoke LANGUAGES C)\n"
        "find_package(VernonRuntime CONFIG REQUIRED)\n"
        "get_target_property(runtime_type Vernon::Runtime TYPE)\n"
        'if(runtime_type STREQUAL "STATIC_LIBRARY")\n'
        "  enable_language(CXX)\n"
        "endif()\n"
        "enable_testing()\n"
        "add_executable(vernon-runtime-wheel-smoke main.c)\n"
        "target_link_libraries(vernon-runtime-wheel-smoke PRIVATE Vernon::Runtime)\n"
        "add_test(NAME runtime-source-consumer COMMAND vernon-runtime-wheel-smoke)\n",
        encoding="utf-8",
    )
    (consumer / "main.c").write_text(
        "#include <vernon-c/Runtime.h>\n"
        "int main(void) {\n"
        "  VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);\n"
        "  return capabilities.available ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    consumer_build = root / "runtime-consumer-build"
    _run(
        [
            "cmake",
            "-S",
            os.fspath(consumer),
            "-B",
            os.fspath(consumer_build),
            f"-DCMAKE_PREFIX_PATH={install}",
        ]
    )
    _run(["cmake", "--build", os.fspath(consumer_build), "--config", "Release"])
    _run(["ctest", "--test-dir", os.fspath(consumer_build), "-C", "Release", "--output-on-failure"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an installed VernonDSL wheel.")
    parser.add_argument(
        "--skip-package-layout",
        action="store_true",
        help="skip bundled-runtime checks when exercising the script from a source checkout",
    )
    arguments = parser.parse_args()
    if not arguments.skip_package_layout:
        runtime_source = cmake_source_dir()
        assert version() == RELEASE_VERSION
        _verify_runtime_source_contents(runtime_source)
    else:
        runtime_source = None

    vd.init(arch=vd.cpu)
    expected = np.arange(16, dtype=np.float32) + np.float32(1.0)
    output = vd.storage.zeros(dtype=vd.f32, shape=expected.shape)
    add_one(
        output,
        vd.storage.from_numpy(expected - np.float32(1.0)),
        grid=((expected.size + 7) // 8, 1, 1),
    )
    np.testing.assert_array_equal(output.to_numpy(), expected)
    with tempfile.TemporaryDirectory(prefix="vernon-wheel-") as directory:
        root = Path(directory)
        source = root / "smoke_shader.py"
        result = root / "smoke.mlir"
        source.write_text(
            "from vernon_dsl import *\n@fragment\ndef main(value: f32) -> f32:\n    return value + 1.0\n",
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "vernon_dsl.cli", str(source), "-o", str(result)],
            check=True,
        )
        assert result.stat().st_size > 0

        asset_source = root / "smoke_asset.py"
        asset_source.write_text(
            "from typing import Annotated\n"
            "import vernon_dsl as vd\n"
            "@vd.kernel(workgroup_size=(1, 1, 1))\n"
            "def fill(output: vd.TensorView[vd.f32, (vd.dyn,), vd.write], "
            "gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin('global_invocation_id')]) -> None:\n"
            "    output[gid[0]] = 1.0\n"
            "asset = vd.program_asset(id='release/smoke', program=fill)\n",
            encoding="utf-8",
        )
        cooked = root / "cooked"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vernon_dsl.program_asset_cli",
                f"{asset_source}:asset",
                "--target",
                "cpu",
                "-o",
                str(cooked),
            ],
            check=True,
        )
        assert (cooked / "cooked.program.json").stat().st_size > 0
        if runtime_source is not None:
            _verify_runtime_source_build(runtime_source, root)
    print(
        f"Installed VernonDSL {RELEASE_VERSION} CPU dispatch, frontend, cooker, "
        "and standalone Runtime source checks passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
