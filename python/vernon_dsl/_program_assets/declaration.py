from __future__ import annotations

import re
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .._markers import dsl_kind
from .._runtime.kernel import Kernel
from .._runtime.pipeline import Pipeline
from ..ad import ProgramExpression
from ..module import Module
from ..program import ModuleVjpExpression
from ..types import Specialization, SpecializationAssignment, check_type_identity_sources, specialization_assignment

VARIANT_CAP = 16
ProgramAssetProgram = (
    Kernel
    | Pipeline
    | Module
    | ProgramExpression
    | ModuleVjpExpression
    | types.ModuleType
    | tuple[types.ModuleType, ...]
)


@dataclass(frozen=True)
class ProgramAssetDeclaration:
    id: str
    program: ProgramAssetProgram
    variants: tuple[tuple[SpecializationAssignment, ...], ...]
    # The C symbol prefix of a Host module asset's exports and table; None for device Programs.
    prefix: str | None = None

    @property
    def variant_keys(self) -> tuple[tuple[SpecializationAssignment, ...], ...]:
        return self.variants

    @property
    def is_host(self) -> bool:
        """Whether the asset is a Host module: Python modules of `@vd.host` functions and classes."""

        return isinstance(self.program, (types.ModuleType, tuple))

    @property
    def host_sources(self) -> tuple[str, ...]:
        """The source files of a Host module asset, in declaration order; they cook into one object and table."""

        modules = self.program if isinstance(self.program, tuple) else (self.program,)
        return tuple(module.__file__ for module in modules if isinstance(module, types.ModuleType) and module.__file__)


def host_prefix(asset_id: str) -> str:
    """The default symbol prefix of a Host module asset: its id with every non-identifier character as `_`."""

    prefix = re.sub(r"\W", "_", asset_id, flags=re.ASCII) + "_"
    return prefix if prefix[0].isalpha() or prefix[0] == "_" else "_" + prefix


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
    check_type_identity_sources(canonical)
    if len(set(canonical)) != len(canonical):
        raise ValueError("Program Asset variants contain duplicate canonical keys")
    return canonical


def program_asset(
    *,
    id: str,
    program: ProgramAssetProgram,
    variants: Iterable[Mapping[Specialization, object]] = ({},),
    prefix: str | None = None,
) -> ProgramAssetDeclaration:
    """Declare one cookable Program. `program` may also be an imported Python module of `@vd.host` functions and
    classes, or a tuple of such modules that cook into one object and table; that Host module asset cooks for the
    cpu target, and `prefix` names its C symbols (by default, the id with non-identifier characters as `_`)."""

    if not isinstance(id, str) or not id:
        raise ValueError("Program Asset id must be a non-empty string")
    checked_variants = _validated_variants(variants)
    modules = program if isinstance(program, tuple) else (program,)
    if any(isinstance(module, types.ModuleType) for module in modules):
        if any(not isinstance(module, types.ModuleType) for module in modules):
            raise TypeError("a Host module Program Asset takes a Python module or a tuple of only modules")
        if any(getattr(module, "__file__", None) is None for module in modules):
            raise TypeError("a Host module Program Asset needs modules loaded from source files")
        if len({module.__file__ for module in modules}) != len(modules):
            raise ValueError("a Host module Program Asset lists a module twice")
        if checked_variants != ((),):
            raise ValueError("Host module Program Assets do not take variants yet")
        chosen = host_prefix(id) if prefix is None else prefix
        if not chosen.isidentifier():
            raise ValueError(f"Host symbol prefix '{chosen}' is not a C identifier prefix")
        return ProgramAssetDeclaration(id=id, program=program, variants=checked_variants, prefix=chosen)
    if prefix is not None:
        raise TypeError("prefix applies only to Host module Program Assets")
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
    if dsl_kind(primal) != "compute":
        raise TypeError("Program Asset program must be a compute Kernel, Pipeline, Module, or VJP expression")
    assert isinstance(program, (Kernel, ProgramExpression))
    return ProgramAssetDeclaration(id=id, program=program, variants=checked_variants)


__all__ = ["ProgramAssetDeclaration", "ProgramAssetProgram", "program_asset"]
