from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnostics import CompileError
from .pipeline_assets import cook_pipeline_asset
from .pipeline_compile import PipelineCompileError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vernon-cook-pipeline",
        description=(
            "Cook a Python pipeline_asset descriptor for one compiler target. "
            "The CPU target emits AOT host object code; it is not restricted to x86."
        ),
        epilog=(
            "CPU examples:\n"
            "  --target cpu --target-triple x86_64-pc-windows-msvc --cpu x86-64-v3\n"
            "  --target cpu --target-triple aarch64-apple-darwin --cpu apple-m1 --cpu-features +neon\n"
            "\n"
            "The target triple selects the ISA, OS, and ABI. --cpu then selects a processor model "
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
    parser.add_argument(
        "--glsl-version",
        type=int,
        metavar="VERSION",
        help="three-digit GLSL version for opengl/opengles only, for example 330, 430, 300, or 310",
    )
    parser.add_argument(
        "--hlsl-shader-model",
        type=int,
        choices=(60,),
        metavar="MODEL",
        help="HLSL Shader Model for directx only (default: 60), encoded as major*10+minor",
    )
    parser.add_argument(
        "--target-triple",
        metavar="TRIPLE",
        help=(
            "LLVM target triple for --target cpu; selects architecture, platform, and ABI, "
            "for example x86_64-pc-windows-msvc or aarch64-apple-darwin; "
            "defaults to the host triple and currently must describe a 64-bit target"
        ),
    )
    parser.add_argument(
        "--cpu",
        metavar="CPU_NAME",
        help=(
            "LLVM processor model for --target cpu, such as generic, x86-64-v3, skylake, or apple-m1; "
            "it refines the architecture selected by --target-triple and defaults to LLVM's target default"
        ),
    )
    parser.add_argument(
        "--cpu-features",
        metavar="FEATURES",
        help=("comma-separated LLVM feature toggles for --target cpu, for example +sse2,-avx or +neon"),
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="output asset directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    target_options = {
        name: value
        for name, value in (
            ("glsl_version", arguments.glsl_version),
            ("hlsl_shader_model", arguments.hlsl_shader_model),
            ("target_triple", arguments.target_triple),
            ("cpu", arguments.cpu),
            ("cpu_features", arguments.cpu_features),
        )
        if value is not None
    }
    try:
        cook_pipeline_asset(
            pipeline_asset=arguments.pipeline_asset,
            output=arguments.output,
            target=arguments.target,
            target_options=target_options,
        )
    except (CompileError, PipelineCompileError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
