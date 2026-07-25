from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..decorators import GRAPHICS_STAGE_ORDER
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
        if any(kind in {None, "compute", "func", "struct"} for kind in kinds):
            raise TypeError("graphics pipeline program must contain only graphics entry stages")
        if len(set(kinds)) != len(kinds):
            raise ValueError("graphics pipeline program contains a duplicate stage kind")
        order = {kind: index for index, kind in enumerate(GRAPHICS_STAGE_ORDER)}
        if kinds != sorted(kinds, key=lambda kind: order.get(kind, len(order))):
            raise ValueError("graphics pipeline program stages are not in topology order")
    elif getattr(program, "__vernon_dsl__", (None,))[0] != "compute":
        raise TypeError("single-entry pipeline program must be a compute Kernel")
    return PipelineAssetDeclaration(
        id=id,
        program=program,
        variants=tuple(tuple(key) for key in variants),
    )
