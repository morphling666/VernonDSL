from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..language.stage_registry import GRAPHICS_STAGES, validate_graphics_topology
from ..types import Feature


@dataclass(frozen=True)
class PipelineAssetDeclaration:
    id: str
    program: Callable[..., Any] | tuple[Callable[..., Any], ...]
    variants: tuple[tuple[Feature, ...], ...]


def pipeline_asset(
    *,
    id: str,
    program: Callable[..., Any] | tuple[Callable[..., Any], ...],
    variants: Iterable[Iterable[Feature]] = ((),),
) -> PipelineAssetDeclaration:
    """Declare one cookable compute or graphics pipeline."""

    if isinstance(program, tuple):
        if not program:
            raise ValueError("graphics pipeline program must contain at least one stage")
        kinds = [getattr(value, "__vernon_dsl__", (None,))[0] for value in program]
        if any(kind not in GRAPHICS_STAGES for kind in kinds):
            raise TypeError("graphics pipeline program must contain only graphics entry stages")
        try:
            validate_graphics_topology(kinds)
        except ValueError as error:
            raise ValueError(str(error).replace("graphics pipeline", "graphics pipeline program")) from None
    elif getattr(program, "__vernon_dsl__", (None,))[0] != "compute":
        raise TypeError("single-entry pipeline program must be a compute Kernel")
    return PipelineAssetDeclaration(
        id=id,
        program=program,
        variants=tuple(tuple(key) for key in variants),
    )
