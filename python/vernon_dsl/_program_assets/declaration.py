from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .._runtime.kernel import Kernel
from .._runtime.pipeline import Pipeline
from ..ad import ProgramExpression
from ..module import Module
from ..program import ModuleVjpExpression
from ..types import Specialization, SpecializationAssignment, specialization_assignment

VARIANT_CAP = 16
ProgramAssetProgram = Kernel | Pipeline | Module | ProgramExpression | ModuleVjpExpression


@dataclass(frozen=True)
class ProgramAssetDeclaration:
    id: str
    program: ProgramAssetProgram
    variants: tuple[tuple[SpecializationAssignment, ...], ...]

    @property
    def variant_keys(self) -> tuple[tuple[SpecializationAssignment, ...], ...]:
        return self.variants


def _validated_variants(
    variants: Iterable[Mapping[Specialization, object]],
) -> tuple[tuple[SpecializationAssignment, ...], ...]:
    rows = tuple(variants)
    result: list[tuple[SpecializationAssignment, ...]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each Program Asset variant must be a mapping from specializations to values")
        if any(not isinstance(parameter, Specialization) for parameter in row):
            raise TypeError("Program Asset variant keys must be vd.specialization(...) or vd.feature(...) values")
        assignments = tuple(sorted((specialization_assignment(parameter, value) for parameter, value in row.items())))
        if len({assignment.name for assignment in assignments}) != len(assignments):
            raise ValueError("Program Asset variant contains duplicate specialization names")
        result.append(assignments)
    canonical = tuple(result)
    if not canonical:
        raise ValueError("a Program Asset must declare at least one variant")
    if len(canonical) > VARIANT_CAP:
        raise ValueError(f"a Program Asset declares {len(canonical)} variants, exceeding variant cap {VARIANT_CAP}")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Program Asset variants contain duplicate canonical keys")
    return canonical


def program_asset(
    *,
    id: str,
    program: ProgramAssetProgram,
    variants: Iterable[Mapping[Specialization, object]] = ({},),
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
