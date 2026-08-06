from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ..ad import ProgramTransformSpec
from .autodiff_profiles import (
    AutodiffProfilePlan,
    _gradient_type,
    _leaves,
    _resolve_path,
    build_structured_profile_plan,
    structured_profile_symbols,
)
from .model import TypedFunctionInstance
from .request import FrontendCompileResult


@dataclass(frozen=True)
class StructuredVjpBuild:
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


def is_structured_vjp_abi_eligible(
    frontend: FrontendCompileResult,
    transform: ProgramTransformSpec,
) -> bool:
    entry = next(
        (function for function in frontend.typed_functions if function.symbol == frontend.request.entry),
        None,
    )
    if entry is None or entry.result_type is None:
        return False
    structs = dict(frontend.structs)
    if not _leaves(entry.result_type, structs, ("output",)):
        return False
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    try:
        for path in transform.wrt:
            root = path.split(".", 1)[0]
            if root not in parameters or parameters[root].builtin is not None:
                return False
            value_type = _resolve_path(path, {name: value.type for name, value in parameters.items()}, structs)
            leaves = _leaves(value_type, structs, tuple(path.split(".")))
            if not leaves:
                return False
            for _, leaf_type in leaves:
                _gradient_type(leaf_type)
    except (KeyError, ValueError, IndexError):
        return False
    return True


def build_structured_vjp(
    native: Any,
    frontend: FrontendCompileResult,
    transform: ProgramTransformSpec,
) -> StructuredVjpBuild:
    entry = next(
        (function for function in frontend.typed_functions if function.symbol == frontend.request.entry),
        None,
    )
    if entry is None:
        raise ValueError("structured VJP frontend produced no typed entry")
    if not is_structured_vjp_abi_eligible(frontend, transform):
        raise ValueError("structured VJP requires canonical differentiable result and wrt leaves")
    structs = dict(frontend.structs)
    output_cotangents = tuple(path for path, _ in _leaves(entry.result_type, structs, ("output",)))
    resolved = resolve_vjp_transform(transform, output_cotangents)
    _, forward_symbol, backward_symbol = structured_profile_symbols(
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
    plan = build_structured_profile_plan(
        resolved,
        entry,
        structs,
        frontend.mlir,
        int(transformed.tape_bytes),
        tuple(str(rule) for rule in transformed.derivative_rules),
        frontend.entry_workgroup_size or (1, 1, 1),
    )
    profiles = MappingProxyType(dict(transformed.profiles(plan.identity)))
    return StructuredVjpBuild(
        resolved,
        entry,
        plan,
        profiles,
        MappingProxyType({"forward_with_tape": "dynamic_v2", "backward": "dynamic_v2"}),
        any("!vernon.ad_tape" in module for module in profiles.values()),
    )


__all__ = [
    "StructuredVjpBuild",
    "build_structured_vjp",
    "is_structured_vjp_abi_eligible",
    "resolve_vjp_transform",
]
