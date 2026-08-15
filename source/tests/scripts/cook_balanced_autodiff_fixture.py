from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from vernon_dsl._shader_assets.cooking import cook_pipeline_asset
from vernon_dsl.frontend.compiler import Compiler
from vernon_dsl.frontend.request import FrontendCompileRequest, FrontendCompileResult


def _with_balanced_policy(mlir: str, entry: str) -> str:
    function = f"func.func @{entry}("
    function_offset = mlir.find(function)
    if function_offset < 0:
        raise RuntimeError(f"generated MLIR has no @{entry} entry")
    attributes_offset = mlir.find(" attributes {", function_offset)
    if attributes_offset < 0:
        raise RuntimeError(f"generated MLIR entry @{entry} has no attribute dictionary")
    insertion_offset = attributes_offset + len(" attributes {")
    return mlir[:insertion_offset] + 'vernon.ad.planning_policy = "balanced", ' + mlir[insertion_offset:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cook a compiler-generated balanced-policy CPU autodiff fixture")
    parser.add_argument("asset")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    original_compile_request = Compiler.compile_request

    def compile_request(compiler: Compiler, request: FrontendCompileRequest) -> FrontendCompileResult:
        result = original_compile_request(compiler, request)
        return replace(result, mlir=_with_balanced_policy(result.mlir, request.entry))

    Compiler.compile_request = compile_request
    try:
        cook_pipeline_asset(
            pipeline_asset=arguments.asset,
            output=arguments.output,
            target="cpu",
        )
    finally:
        Compiler.compile_request = original_compile_request


if __name__ == "__main__":
    main()
