from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bundle import PipelineCompileError, make_target_options
from .diagnostics import CompileError
from .pipeline_assets import cook_pipeline_asset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vernon-cook-pipeline",
        description=(
            "Cook a Python pipeline_asset descriptor for one compiler target. "
            "The CPU target emits AOT host object code; it is not restricted to x86."
        ),
        epilog=(
            "CPU examples:\n"
            "  --target cpu --cpu-triple x86_64-pc-windows-msvc --cpu-name x86-64-v3\n"
            "  --target cpu --cpu-triple aarch64-apple-darwin --cpu-name apple-m1 --cpu-features +neon\n"
            "\n"
            "The target triple selects the ISA, OS, and ABI. --cpu-name then selects a processor model "
            "within that ISA, while --cpu-features applies explicit LLVM feature toggles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pipeline_asset",
        type=str,
        help="Python descriptor reference in source.py:descriptor_name form",
    )
    parser.add_argument(
        "--target",
        choices=("cpu", "cuda", "opengl", "opengles", "vulkan", "metal", "directx", "dx"),
        default="opengl",
        help=(
            "compiler target (default: opengl); dx is an alias for directx. "
            "A recognized target may still report unsupported when its complete lowering is not built."
        ),
    )
    opengl = parser.add_argument_group("OpenGL target options")
    opengl.add_argument(
        "--opengl-version",
        type=int,
        metavar="VERSION",
        help="three-digit GLSL version for opengl/opengles only, for example 330, 430, 300, or 310",
    )
    directx = parser.add_argument_group("DirectX target options")
    directx.add_argument(
        "--directx-shader-model",
        type=int,
        choices=(60,),
        metavar="MODEL",
        help="HLSL Shader Model for directx only (default: 60), encoded as major*10+minor",
    )
    metal = parser.add_argument_group("Metal target options")
    metal.add_argument(
        "--metal-platform",
        choices=("macos", "ios"),
        help="Apple platform for cooked Metal MSL only (default: macos); this does not cross-build the Runtime",
    )
    cpu = parser.add_argument_group("CPU target options")
    cpu.add_argument(
        "--cpu-triple",
        metavar="TRIPLE",
        help=(
            "LLVM target triple for --target cpu; selects architecture, platform, and ABI, "
            "for example x86_64-pc-windows-msvc or aarch64-apple-darwin; "
            "defaults to the host triple and currently must describe a 64-bit target"
        ),
    )
    cpu.add_argument(
        "--cpu-name",
        metavar="CPU_NAME",
        help=(
            "LLVM processor model for --target cpu, such as generic, x86-64-v3, skylake, or apple-m1; "
            "it refines the architecture selected by --cpu-triple and defaults to LLVM's target default"
        ),
    )
    cpu.add_argument(
        "--cpu-features",
        metavar="FEATURES",
        help=("comma-separated LLVM feature toggles for --target cpu, for example +sse2,-avx or +neon"),
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="output asset directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        option_values = {
            "cpu": {
                name: value
                for name, value in (
                    ("triple", arguments.cpu_triple),
                    ("processor", arguments.cpu_name),
                    (
                        "features",
                        tuple(value for value in (arguments.cpu_features or "").split(",") if value),
                    ),
                )
                if value
            },
            "opengl": {"version": arguments.opengl_version} if arguments.opengl_version is not None else {},
            "opengles": {"version": arguments.opengl_version} if arguments.opengl_version is not None else {},
            "metal": {"platform": arguments.metal_platform} if arguments.metal_platform is not None else {},
            "directx": (
                {"shader_model": arguments.directx_shader_model} if arguments.directx_shader_model is not None else {}
            ),
        }
        target = "directx" if arguments.target == "dx" else arguments.target
        selected_options = option_values.get(target, {})
        supplied_groups = {
            "cpu": any(
                value is not None for value in (arguments.cpu_triple, arguments.cpu_name, arguments.cpu_features)
            ),
            "opengl": arguments.opengl_version is not None,
            "metal": arguments.metal_platform is not None,
            "directx": arguments.directx_shader_model is not None,
        }
        invalid_groups = [name for name, supplied in supplied_groups.items() if supplied and name != target]
        if invalid_groups and not (target == "opengles" and invalid_groups == ["opengl"]):
            raise PipelineCompileError(f"{invalid_groups[0]} options do not apply to target '{target}'")
        cook_pipeline_asset(
            pipeline_asset=arguments.pipeline_asset,
            output=arguments.output,
            target=make_target_options(target, selected_options),
        )
    except (CompileError, PipelineCompileError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
