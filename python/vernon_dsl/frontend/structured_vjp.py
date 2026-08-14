from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..ad import ProgramTransformSpec
from .autodiff_profiles import (
    AutodiffProfilePlan,
    _gradient_type,
    _resolve_path,
    _structured_leaves,
    build_structured_profile_plan,
    structured_profile_symbols,
)
from .model import AccessMode, ConcreteType, TypedFunctionInstance
from .request import FrontendCompileResult


@dataclass(frozen=True)
class StructuredVjpBuild:
    transform: ProgramTransformSpec
    entry: TypedFunctionInstance
    plan: AutodiffProfilePlan
    profiles: Mapping[str, str]
    protocols: Mapping[str, str]
    uses_dynamic_tape: bool
    residual_storage_kind: str
    active_operation_count: int
    recomputation_cost: int
    source_kind_counts: Mapping[str, int]
    cost_components: Mapping[str, int]
    selected_policy: str


def _validate_nonoverlapping_paths(paths: tuple[str, ...], role: str) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left.startswith(right + ".") or right.startswith(left + "."):
                raise ValueError(f"structured VJP {role} paths {left!r} and {right!r} overlap")


def resolve_vjp_transform(
    transform: ProgramTransformSpec,
    output_cotangents: tuple[str, ...],
) -> ProgramTransformSpec:
    if transform.output_cotangents != output_cotangents:
        raise ValueError("structured VJP output paths changed after declaration")
    return transform


def _validate_selected_outputs(
    entry: TypedFunctionInstance,
    transform: ProgramTransformSpec,
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
) -> tuple[tuple[str, ConcreteType], ...]:
    if entry.result_type is not None:
        raise ValueError("structured compute VJP requires a void kernel")
    if not transform.output_cotangents:
        raise ValueError("structured compute VJP requires writable Storage outputs")
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    selected: list[tuple[str, ConcreteType]] = []
    for path in transform.output_cotangents:
        root = path.split(".", 1)[0]
        parameter = parameters.get(root)
        if parameter is None:
            raise ValueError(f"VJP output path {path!r} names no entry parameter")
        if parameter.builtin is not None:
            raise ValueError(f"VJP output path {path!r} names a builtin parameter")
        try:
            value_type = _resolve_path(
                path,
                {name: value.type for name, value in parameters.items()},
                structs,
            )
        except (KeyError, ValueError, IndexError):
            raise ValueError(f"VJP output path {path!r} is not a canonical Storage projection") from None
        if value_type.kind != "tensor_view":
            raise ValueError(f"VJP output path {path!r} must resolve to TensorView Storage")
        if parameter.access is AccessMode.READ:
            raise ValueError(f"VJP output path {path!r} must be writable")
        if not _structured_leaves(value_type, structs, tuple(path.split("."))):
            raise ValueError(f"VJP output path {path!r} has no floating elements")
        selected.append((path, value_type))
    return tuple(selected)


def is_structured_vjp_abi_eligible(
    frontend: FrontendCompileResult,
    transform: ProgramTransformSpec,
) -> bool:
    entry = next(
        (function for function in frontend.typed_functions if function.symbol == frontend.request.entry),
        None,
    )
    if entry is None:
        return False
    structs = dict(frontend.structs)
    try:
        _validate_selected_outputs(entry, transform, structs)
    except ValueError:
        return False
    parameters = {parameter.name: parameter for parameter in entry.parameters}
    try:
        _validate_nonoverlapping_paths(transform.wrt, "wrt")
        _validate_nonoverlapping_paths(transform.output_cotangents, "output")
        for path in transform.wrt:
            root = path.split(".", 1)[0]
            if root not in parameters or parameters[root].builtin is not None:
                return False
            value_type = _resolve_path(path, {name: value.type for name, value in parameters.items()}, structs)
            leaves = _structured_leaves(value_type, structs, tuple(path.split(".")))
            if not leaves:
                return False
            for _, leaf_type in leaves:
                if leaf_type.kind == "tensor_view":
                    element = leaf_type.arguments[0]
                    if not isinstance(element, ConcreteType):
                        return False
                    _gradient_type(element)
                else:
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
    _validate_nonoverlapping_paths(transform.wrt, "wrt")
    _validate_nonoverlapping_paths(transform.output_cotangents, "output")
    if not is_structured_vjp_abi_eligible(frontend, transform):
        raise ValueError("structured VJP requires canonical differentiable result and wrt leaves")
    structs = dict(frontend.structs)
    selected_outputs = _validate_selected_outputs(entry, transform, structs)
    output_cotangents = tuple(path for path, _ in selected_outputs)
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
        list(resolved.output_cotangents),
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
        tuple(str(path) for path in transformed.required_primal_paths),
        tuple(sorted((str(kind), int(count)) for kind, count in transformed.source_kind_counts.items())),
        tuple(sorted((str(kind), int(cost)) for kind, cost in transformed.cost_components.items())),
        str(transformed.selected_policy),
        frontend.entry_workgroup_size or (1, 1, 1),
    )
    profiles = MappingProxyType(dict(transformed.profiles(plan.identity)))
    storage_kinds = {
        kind
        for kind in ("none", "static", "dynamic")
        if all(f'vernon.ad.residual_storage = "{kind}"' in module for module in profiles.values())
    }
    if len(storage_kinds) != 1:
        raise RuntimeError("structured VJP profiles disagree on residual storage classification")
    residual_storage_kind = storage_kinds.pop()
    return StructuredVjpBuild(
        resolved,
        entry,
        plan,
        profiles,
        MappingProxyType({"forward_with_tape": "dynamic_v2", "backward": "dynamic_v2"}),
        any("!vernon.ad_tape" in module for module in profiles.values()),
        residual_storage_kind,
        int(transformed.active_operation_count),
        int(transformed.recomputation_cost),
        MappingProxyType({str(kind): int(count) for kind, count in transformed.source_kind_counts.items()}),
        MappingProxyType({str(kind): int(cost) for kind, cost in transformed.cost_components.items()}),
        str(transformed.selected_policy),
    )


__all__ = [
    "StructuredVjpBuild",
    "build_structured_vjp",
    "is_structured_vjp_abi_eligible",
    "resolve_vjp_transform",
]
