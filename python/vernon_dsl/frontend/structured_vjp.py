from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ..ad import ProgramTransformSpec
from .autodiff_profiles import (
    AutodiffProfilePlan,
    build_structured_scalar_profile_plan,
    structured_scalar_profile_symbols,
)
from .model import TypedFunctionInstance
from .request import FrontendCompileResult


@dataclass(frozen=True)
class StructuredScalarVjpBuild:
    transform: ProgramTransformSpec
    entry: TypedFunctionInstance
    plan: AutodiffProfilePlan
    profiles: Mapping[str, str]
    protocols: Mapping[str, str]
    uses_dynamic_tape: bool


def resolve_vjp_transform(
    transform: ProgramTransformSpec,
    output_cotangents: tuple[str, ...],
) -> ProgramTransformSpec:
    return replace(transform, output_cotangents=output_cotangents)


def is_structured_scalar_vjp_abi_eligible(
    frontend: FrontendCompileResult,
    transform: ProgramTransformSpec,
) -> bool:
    entry = next(
        (function for function in frontend.typed_functions if function.symbol == frontend.request.entry),
        None,
    )
    if entry is None or entry.result_type is None:
        return False
    if entry.result_type.kind != "scalar" or not entry.result_type.is_float:
        return False
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    return all(
        path in parameters
        and parameters[path].builtin is None
        and parameters[path].type.kind == "scalar"
        and parameters[path].type.is_float
        for path in transform.wrt
    )


def build_structured_scalar_vjp(
    native: Any,
    frontend: FrontendCompileResult,
    transform: ProgramTransformSpec,
) -> StructuredScalarVjpBuild:
    entry = next(
        (function for function in frontend.typed_functions if function.symbol == frontend.request.entry),
        None,
    )
    if entry is None:
        raise ValueError("structured VJP frontend produced no typed entry")
    if not is_structured_scalar_vjp_abi_eligible(frontend, transform):
        raise ValueError("structured scalar VJP requires one floating result and scalar floating wrt parameters")
    resolved = resolve_vjp_transform(transform, ("output",))
    _, forward_symbol, backward_symbol = structured_scalar_profile_symbols(
        resolved,
        entry,
        frontend.mlir,
    )
    transformed = native._build_structured_vjp(
        frontend.mlir,
        entry.symbol,
        list(resolved.wrt),
        forward_symbol,
        backward_symbol,
    )
    plan = build_structured_scalar_profile_plan(
        resolved,
        entry,
        frontend.mlir,
        int(transformed.tape_bytes),
        tuple(str(rule) for rule in transformed.derivative_rules),
        frontend.entry_workgroup_size or (1, 1, 1),
    )
    profiles = MappingProxyType(dict(transformed.profiles(plan.identity)))
    return StructuredScalarVjpBuild(
        resolved,
        entry,
        plan,
        profiles,
        MappingProxyType({"forward_with_tape": "dynamic_v2", "backward": "dynamic_v2"}),
        any("!vernon.ad_tape" in module for module in profiles.values()),
    )


__all__ = [
    "StructuredScalarVjpBuild",
    "build_structured_scalar_vjp",
    "is_structured_scalar_vjp_abi_eligible",
    "resolve_vjp_transform",
]
