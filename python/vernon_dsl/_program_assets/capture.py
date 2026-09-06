"""Typed Program Asset capture.

This is the only layer that discriminates authored Program forms. Every branch
produces the same immutable CapturedProgram consumed by compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..diagnostics import ProgramCompileError
from ..program_frontend import ParsedProgram
from .declaration import ProgramAssetDeclaration

if TYPE_CHECKING:
    from .._runtime.kernel import Kernel
    from .._runtime.pipeline import Pipeline
    from ..ad import ProgramExpression
    from ..module import Module
    from ..program import ModuleVjpExpression


@dataclass(frozen=True)
class CapturedProgramVariant:
    key: tuple[str, ...]
    ir: ParsedProgram


@dataclass(frozen=True)
class CapturedProgram:
    id: str
    variants: tuple[CapturedProgramVariant, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.variants:
            raise ValueError("CapturedProgram requires an id and at least one variant")
        keys = tuple(variant.key for variant in self.variants)
        if len(set(keys)) != len(keys):
            raise ValueError("CapturedProgram variant keys must be unique")

    @property
    def variant_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(variant.key for variant in self.variants)


def _validate_module_vjp_resources(module: object) -> None:
    from ..ad import _capability_diagnostic
    from ..module import Module
    from ..types import TypeExpr

    assert isinstance(module, Module)
    for path, child in module.named_modules():
        definition = child._module_definition
        for name in definition.signature.parameters:
            annotation = definition.resolved_annotations.get(name)
            if isinstance(annotation, TypeExpr) and annotation.name in {"Texture", "Sampler"}:
                qualified = f"{path}.{name}" if path else name
                raise ProgramCompileError(
                    f"Module VJP does not support {annotation.name} parameter {qualified!r}; "
                    f"{_capability_diagnostic('opaque_resource_vjp')}"
                )


def _capture_kernel(kernel: Kernel, keys: tuple[tuple[str, ...], ...]) -> tuple[CapturedProgramVariant, ...]:
    from ..program import _parse_kernel_program

    return tuple(CapturedProgramVariant(key, _parse_kernel_program(kernel, key)) for key in keys)


def _capture_pipeline(pipeline: Pipeline, keys: tuple[tuple[str, ...], ...]) -> tuple[CapturedProgramVariant, ...]:
    from ..program import _parse_pipeline_program

    if pipeline._features:
        raise ProgramCompileError(
            "a Program Asset drives features from its variants, so a cooked vd.pipeline(...) must not carry its "
            "own features="
        )
    return tuple(CapturedProgramVariant(key, _parse_pipeline_program(pipeline, key)) for key in keys)


def _capture_kernel_vjp(
    expression: ProgramExpression,
    keys: tuple[tuple[str, ...], ...],
) -> tuple[CapturedProgramVariant, ...]:
    from ..ad import _capability_diagnostic
    from ..program import _parse_kernel_program

    kernel = expression.program
    if getattr(kernel, "__vernon_dsl__", (None,))[0] != "compute":
        raise ProgramCompileError(_capability_diagnostic("graphics_vjp"))
    return tuple(
        CapturedProgramVariant(key, _parse_kernel_program(kernel, key, transform=expression.transform)) for key in keys
    )


def _capture_module(module: Module, keys: tuple[tuple[str, ...], ...]) -> tuple[CapturedProgramVariant, ...]:
    from ..program import _parse_module_program

    parsed = _parse_module_program(module)
    return tuple(CapturedProgramVariant(key, parsed) for key in keys)


def _capture_module_vjp(
    expression: ModuleVjpExpression,
    keys: tuple[tuple[str, ...], ...],
) -> tuple[CapturedProgramVariant, ...]:
    from ..program import _parse_module_program

    _validate_module_vjp_resources(expression.module)
    try:
        parsed = _parse_module_program(
            expression.module,
            vjp_wrt=expression.wrt,
            vjp_outputs=expression.outputs,
            autodiff_planning_policy=expression.planning_policy,
        )
    except TypeError as error:
        if str(error) != "graphics Module programs do not support autodiff":
            raise
        from ..ad import _capability_diagnostic

        raise ProgramCompileError(_capability_diagnostic("graphics_vjp")) from None
    return tuple(CapturedProgramVariant(key, parsed) for key in keys)


def capture_program(declaration: ProgramAssetDeclaration) -> CapturedProgram:
    """Capture one typed declaration without exposing its authored form downstream."""

    from .._runtime.kernel import Kernel
    from .._runtime.pipeline import Pipeline
    from ..ad import ProgramExpression
    from ..module import Module
    from ..program import ModuleVjpExpression

    authored = declaration.program
    keys = declaration.variant_keys

    if isinstance(authored, ModuleVjpExpression):
        variants = _capture_module_vjp(authored, keys)
    elif isinstance(authored, ProgramExpression):
        variants = _capture_kernel_vjp(authored, keys)
    elif isinstance(authored, Pipeline):
        variants = _capture_pipeline(authored, keys)
    elif isinstance(authored, Module):
        variants = _capture_module(authored, keys)
    elif isinstance(authored, Kernel):
        variants = _capture_kernel(authored, keys)
    else:
        raise TypeError("Program Asset program must be a compute Kernel, Pipeline, Module, or VJP expression")

    return CapturedProgram(declaration.id, variants)


__all__ = ["CapturedProgram", "CapturedProgramVariant", "capture_program"]
