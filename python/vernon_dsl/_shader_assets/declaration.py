from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..ad import ProgramExpression
from ..language.stage_registry import GRAPHICS_STAGES, validate_graphics_topology
from ..types import Feature


@dataclass(frozen=True)
class PipelineAssetDeclaration:
    id: str
    program: Callable[..., Any] | tuple[Callable[..., Any], ...] | ProgramExpression
    variants: tuple[tuple[Feature, ...], ...]


def pipeline_asset(
    *,
    id: str,
    program: Callable[..., Any] | tuple[Callable[..., Any], ...] | ProgramExpression,
    variants: Iterable[Iterable[Feature]] = ((),),
) -> PipelineAssetDeclaration:
    """Declare one cookable compute or graphics pipeline."""

    primal = program.program if isinstance(program, ProgramExpression) else program
    if isinstance(primal, tuple):
        if not primal:
            raise ValueError("graphics pipeline program must contain at least one stage")
        kinds = [getattr(value, "__vernon_dsl__", (None,))[0] for value in primal]
        if any(kind not in GRAPHICS_STAGES for kind in kinds):
            raise TypeError("graphics pipeline program must contain only graphics entry stages")
        try:
            validate_graphics_topology(kinds)
        except ValueError as error:
            raise ValueError(str(error).replace("graphics pipeline", "graphics pipeline program")) from None
    elif getattr(primal, "__vernon_dsl__", (None,))[0] != "compute":
        raise TypeError("single-entry pipeline program must be a compute Kernel")
    return PipelineAssetDeclaration(
        id=id,
        program=program,
        variants=tuple(tuple(key) for key in variants),
    )
