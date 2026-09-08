from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .._runtime.kernel import Kernel
from .._runtime.pipeline import Pipeline
from ..ad import ProgramExpression
from ..module import Module
from ..program import ModuleVjpExpression
from ..types import Feature

VARIANT_CAP = 16
ProgramAssetProgram = Kernel | Pipeline | Module | ProgramExpression | ModuleVjpExpression


@dataclass(frozen=True)
class ProgramAssetDeclaration:
    id: str
    program: ProgramAssetProgram
    variants: tuple[tuple[Feature, ...], ...]

    @property
    def variant_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(feature.name for feature in variant) for variant in self.variants)


def _validated_variants(
    variants: Iterable[Iterable[Feature]],
) -> tuple[tuple[Feature, ...], ...]:
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
    program: ProgramAssetProgram,
    variants: Iterable[Iterable[Feature]] = ((),),
) -> ProgramAssetDeclaration:
    """Declare one cookable Program."""

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
        raise TypeError("Program Asset graphics must use vd.pipeline(...), not a tuple of entry functions")
    if getattr(primal, "__vernon_dsl__", (None,))[0] != "compute":
        raise TypeError("Program Asset program must be a compute Kernel, Pipeline, Module, or VJP expression")
    assert isinstance(program, (Kernel, ProgramExpression))
    return ProgramAssetDeclaration(id=id, program=program, variants=checked_variants)


__all__ = ["ProgramAssetDeclaration", "ProgramAssetProgram", "program_asset"]
