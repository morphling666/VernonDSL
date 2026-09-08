"""Thin orchestration entry point for cooking typed Program Assets."""

from __future__ import annotations

from pathlib import Path

from ..bundle import OpenGLTargetOptions, ProgramCompileError, TargetOptions, make_target_options
from ..module_graph import load_project
from .artifact_io import write_bundle_artifacts
from .capture import CapturedProgram, capture_program
from .compile_orchestration import compile_captured_program
from .parsing import program_asset_reference
from .source import load_program_asset_declaration


def _validate_captured_target(captured: CapturedProgram, target: TargetOptions) -> None:
    if target.target == "cpu" and any(
        implementation.kind == "graphics"
        for variant in captured.variants
        for implementation in variant.ir.implementations
    ):
        raise ProgramCompileError("CPU Program Assets do not support graphics implementations")


def cook_program_asset(
    *,
    program_asset: str | Path,
    output: str | Path,
    target: TargetOptions | str | None = None,
) -> Path:
    source, descriptor_name = program_asset_reference(program_asset)
    declaration = load_program_asset_declaration(source, descriptor_name)
    captured = capture_program(declaration)
    if target is None:
        target = OpenGLTargetOptions()
    elif isinstance(target, str):
        target = make_target_options(target)

    declared_features = set(load_project(source).features)
    for variant in captured.variant_keys:
        unknown = set(variant) - declared_features
        if unknown:
            raise ProgramCompileError("variant requests undeclared feature(s): " + ", ".join(sorted(unknown)))

    _validate_captured_target(captured, target)
    plan = compile_captured_program(captured, target)
    return write_bundle_artifacts(plan, Path(output).resolve())


__all__ = ["cook_program_asset"]
