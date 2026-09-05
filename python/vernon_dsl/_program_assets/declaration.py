from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, cast

from ..ad import ProgramExpression
from ..language.stage_registry import GRAPHICS_STAGES, validate_graphics_topology
from ..types import Feature

VARIANT_CAP = 16


@dataclass(frozen=True)
class ProgramAssetDeclaration:
    id: str
    program: Any
    variants: tuple[tuple[Feature, ...], ...]

    @property
    def variant_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(feature.name for feature in variant) for variant in self.variants)


def _validated_variants(variants: Iterable[Iterable[Feature]]) -> tuple[tuple[Feature, ...], ...]:
    result = tuple(tuple(key) for key in variants)
    if not result:
        raise ValueError("a Program Asset must declare at least one variant")
    if len(result) > VARIANT_CAP:
        raise ValueError(f"a Program Asset declares {len(result)} variants, exceeding variant cap {VARIANT_CAP}")
    seen: list[tuple[str, ...]] = []
    for variant in result:
        if any(not isinstance(feature, Feature) for feature in variant):
            raise TypeError("Program Asset variants must contain vd.feature(...) values")
        key = tuple(feature.name for feature in variant)
        if list(key) != sorted(key) or len(set(key)) != len(key):
            raise ValueError(f"Program Asset variant is not canonical: {list(key)}")
        if key in seen:
            raise ValueError(f"duplicate Program Asset variant: {list(key)}")
        seen.append(key)
    return result


def program_asset(
    *,
    id: str,
    program: Any,
    variants: Iterable[Iterable[Feature]] = ((),),
) -> ProgramAssetDeclaration:
    """Declare one cookable compute or graphics pipeline."""

    from .._runtime.pipeline import Pipeline
    from ..module import Module
    from ..program import ModuleVjpExpression

    if not isinstance(id, str) or not id:
        raise ValueError("Program Asset id must be a non-empty string")
    checked_variants = _validated_variants(variants)
    primal = program.program if isinstance(program, ProgramExpression) else program
    if isinstance(primal, Pipeline) and primal._targets is None:
        raise ValueError(
            "cooking a graphics pipeline requires vd.pipeline(..., targets=vd.target_formats(...)); attachment "
            "formats and sample count are pipeline state that every backend bakes into the pipeline object, so "
            "unlike the attachment extent they cannot be deferred to invocation"
        )
    if isinstance(primal, (Module, Pipeline)) or isinstance(program, ModuleVjpExpression):
        return ProgramAssetDeclaration(id=id, program=program, variants=checked_variants)
    if isinstance(primal, tuple):
        if not primal:
            raise ValueError("graphics pipeline program must contain at least one stage")
        kinds = [getattr(value, "__vernon_dsl__", (None,))[0] for value in primal]
        if any(kind not in GRAPHICS_STAGES for kind in kinds):
            raise TypeError("graphics pipeline program must contain only graphics entry stages")
        try:
            validate_graphics_topology(cast(list[str], kinds))
        except ValueError as error:
            raise ValueError(str(error).replace("graphics pipeline", "graphics pipeline program")) from None
    elif getattr(primal, "__vernon_dsl__", (None,))[0] != "compute":
        raise TypeError("single-entry pipeline program must be a compute Kernel")
    return ProgramAssetDeclaration(id=id, program=program, variants=checked_variants)
