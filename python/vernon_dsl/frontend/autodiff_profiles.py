from __future__ import annotations

import ast
import hashlib
import itertools
from dataclasses import dataclass
from typing import Any

from ..ad import ProgramTransformSpec
from ..bundle import canonical_json
from .autodiff import (
    AccessPatternEvidence,
    AccumulationMode,
    AccumulationPlan,
    AutodiffProgram,
    LaunchPlan,
    OpCode,
)
from .model import ConcreteType, TypedFunctionInstance


@dataclass(frozen=True)
class AutodiffBinding:
    path: str
    type: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "type": self.type, "role": self.role}


@dataclass(frozen=True)
class DerivativeGroup:
    role: str
    declared_path: str
    leaf_paths: tuple[str, ...]

    @property
    def parameter_root(self) -> str:
        return self.declared_path.split(".", 1)[0]


def derivative_groups_from_paths(
    role: str,
    declared_paths: tuple[str, ...],
    leaf_paths: tuple[str, ...],
) -> tuple[DerivativeGroup, ...]:
    if len(set(declared_paths)) != len(declared_paths):
        raise ValueError(f"{role} derivative group paths must be unique")
    if len(set(leaf_paths)) != len(leaf_paths):
        raise ValueError(f"{role} derivative leaf paths must be unique")
    groups: list[DerivativeGroup] = []
    consumed: set[str] = set()
    for declared_path in declared_paths:
        components = declared_path.split(".")
        leaves = tuple(path for path in leaf_paths if path.split(".")[: len(components)] == components)
        if not leaves:
            raise ValueError(f"{role} path '{declared_path}' has no reflected derivative leaves")
        overlap = consumed.intersection(leaves)
        if overlap:
            raise ValueError(f"{role} derivative leaf '{next(iter(overlap))}' belongs to multiple groups")
        consumed.update(leaves)
        groups.append(DerivativeGroup(role, declared_path, leaves))
    if consumed != set(leaf_paths):
        unexpected = sorted(set(leaf_paths) - consumed)
        raise ValueError(f"unowned {role} derivative leaf '{unexpected[0]}'")
    return tuple(groups)


@dataclass(frozen=True)
class AutodiffProfile:
    name: str
    symbol: str
    inputs: tuple[AutodiffBinding, ...]
    outputs: tuple[AutodiffBinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "inputs": [binding.to_dict() for binding in self.inputs],
            "outputs": [binding.to_dict() for binding in self.outputs],
        }


@dataclass(frozen=True)
class AutodiffProfilePlan:
    transform_identity: str
    program_graph_identity: str
    tape_bytes: int
    derivative_rules: tuple[str, ...]
    derivative_rules_version: int
    launch: LaunchPlan
    profiles: tuple[AutodiffProfile, ...]
    derivative_groups: tuple[DerivativeGroup, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_identity": self.transform_identity,
            "program_graph_identity": self.program_graph_identity,
            "tape_bytes": self.tape_bytes,
            "derivative_rules": list(self.derivative_rules),
            "derivative_rules_version": self.derivative_rules_version,
            "launch": self.launch.to_dict(),
            "profiles": [profile.to_dict() for profile in self.profiles],
        }

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def manifest_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result["identity"] = self.identity
        return result


def _gradient_type(value_type: ConcreteType) -> ConcreteType:
    if value_type.kind == "scalar":
        if not value_type.is_float:
            raise ValueError(f"{value_type.mlir} is not differentiable")
        return ConcreteType("scalar", "f64" if value_type.name == "f64" else "f32")
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            raise ValueError("Tensor element type is unresolved")
        return ConcreteType("tensor", "Tensor", (_gradient_type(element), *value_type.arguments[1:]))
    if value_type.kind == "tensor_view":
        element, shape, _, _ = value_type.arguments
        if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
            raise ValueError("TensorView gradient type is unresolved")
        if any(not isinstance(extent, int) or extent <= 0 for extent in shape):
            raise ValueError("initial TensorView gradients require a positive static shape")
        return ConcreteType("tensor", "Tensor", (_gradient_type(element), *shape))
    raise ValueError(f"{value_type.mlir} is not one differentiable ABI leaf")


def _structured_derivative_type(value_type: ConcreteType, access: str) -> ConcreteType:
    if value_type.kind != "tensor_view":
        return _gradient_type(value_type)
    element, shape, _, address_space = value_type.arguments
    if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
        raise ValueError("TensorView derivative type is unresolved")
    return ConcreteType(
        "tensor_view",
        "TensorView",
        (_gradient_type(element), shape, access, address_space),
    )


def _leaves(
    value_type: ConcreteType,
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
    prefix: tuple[str, ...],
) -> tuple[tuple[str, ConcreteType], ...]:
    if value_type.kind == "scalar":
        return ((".".join(prefix), value_type),) if value_type.is_float else ()
    if value_type.kind in {"tensor", "tensor_view"}:
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            return ()
        return ((".".join(prefix), value_type),) if _leaves(element, structs, prefix) else ()
    if value_type.kind == "tuple":
        return tuple(
            leaf
            for index, element in enumerate(value_type.arguments)
            if isinstance(element, ConcreteType)
            for leaf in _leaves(element, structs, (*prefix, str(index)))
        )
    if value_type.kind == "struct":
        return tuple(
            leaf for name, field in structs[value_type.name] for leaf in _leaves(field, structs, (*prefix, name))
        )
    return ()


def _structured_leaves(
    value_type: ConcreteType,
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
    prefix: tuple[str, ...],
) -> tuple[tuple[str, ConcreteType], ...]:
    if value_type.kind != "tensor_view":
        return _leaves(value_type, structs, prefix)
    element, shape, access, address_space = value_type.arguments
    if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
        return ()

    def element_leaves(current: ConcreteType, path: tuple[str, ...]) -> tuple[tuple[str, ConcreteType], ...]:
        if current.kind == "tensor":
            child = current.arguments[0]
            dimensions = current.arguments[1:]
            if not isinstance(child, ConcreteType) or any(not isinstance(extent, int) for extent in dimensions):
                return ()
            static_dimensions = tuple(extent for extent in dimensions if isinstance(extent, int))
            terminal = child
            while terminal.kind == "tensor" and isinstance(terminal.arguments[0], ConcreteType):
                terminal = terminal.arguments[0]
            if terminal.kind == "scalar":
                return ((".".join(path), current),) if terminal.is_float else ()
            return tuple(
                leaf
                for coordinate in itertools.product(*(range(extent) for extent in static_dimensions))
                for leaf in element_leaves(child, (*path, *(str(index) for index in coordinate)))
            )
        if current.kind == "tuple":
            return tuple(
                leaf
                for index, child in enumerate(current.arguments)
                if isinstance(child, ConcreteType)
                for leaf in element_leaves(child, (*path, str(index)))
            )
        if current.kind == "struct":
            return tuple(leaf for name, child in structs[current.name] for leaf in element_leaves(child, (*path, name)))
        return ((".".join(path), current),) if current.kind == "scalar" and current.is_float else ()

    resolved_element_leaves = element_leaves(element, prefix)
    return tuple(
        (
            path,
            ConcreteType("tensor_view", "TensorView", (leaf_type, shape, access, address_space)),
        )
        for path, leaf_type in resolved_element_leaves
    )


def _resolve_path(
    path: str,
    parameters: dict[str, ConcreteType],
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
) -> ConcreteType:
    components = path.split(".")
    value_type = parameters[components[0]]
    view_arguments = value_type.arguments[1:] if value_type.kind == "tensor_view" else None
    if view_arguments is not None and len(components) > 1:
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            raise ValueError(f"unresolved TensorView path {path!r}")
        value_type = element
    for component in components[1:]:
        if value_type.kind == "struct":
            value_type = dict(structs[value_type.name])[component]
        elif value_type.kind == "tuple":
            nested = value_type.arguments[int(component)]
            if not isinstance(nested, ConcreteType):
                raise ValueError(f"unresolved Tuple path {path!r}")
            value_type = nested
        else:
            raise ValueError(f"path {path!r} does not resolve to an aggregate Value")
    if view_arguments is not None and len(components) > 1:
        value_type = ConcreteType("tensor_view", "TensorView", (value_type, *view_arguments))
    return value_type


def _tensor_view_evidence(
    entry: TypedFunctionInstance, parameter_name: str
) -> tuple[tuple[AccessPatternEvidence, ...], tuple[int, ...]]:
    builtin_names = {parameter.name for parameter in entry.parameters if parameter.builtin == "global_invocation_id"}
    mappings: list[tuple[tuple[str, int], ...]] = []
    direct_storage_references: set[int] = set()
    static = False
    non_injective = False
    axes = {"x": 0, "y": 1, "z": 2}
    for node in ast.walk(entry.source):
        if (
            not isinstance(node, ast.Subscript)
            or not isinstance(node.value, ast.Name)
            or node.value.id != parameter_name
        ):
            continue
        direct_storage_references.add(id(node.value))
        components = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        mapping: list[tuple[str, int]] = []
        for component in components:
            if isinstance(component, ast.Constant) and isinstance(component.value, int):
                static = True
                mapping.append(("constant", component.value))
            elif (
                isinstance(component, ast.Attribute)
                and isinstance(component.value, ast.Name)
                and component.value.id in builtin_names
                and component.attr in axes
            ):
                mapping.append(("gid", axes[component.attr]))
            elif (
                isinstance(component, ast.Subscript)
                and isinstance(component.value, ast.Name)
                and component.value.id in builtin_names
                and isinstance(component.slice, ast.Constant)
                and isinstance(component.slice.value, int)
                and 0 <= component.slice.value < 3
            ):
                mapping.append(("gid", component.slice.value))
            else:
                non_injective = True
                mapping = []
                break
        if mapping:
            mappings.append(tuple(mapping))
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == parameter_name
        and id(node) not in direct_storage_references
        for node in ast.walk(entry.source)
    ):
        non_injective = True
    evidence: set[AccessPatternEvidence] = set()
    if static:
        evidence.add(AccessPatternEvidence.STATIC_INDEX_CONFLICT)
    invocation_axes = {index for mapping in mappings for kind, index in mapping if kind == "gid"}
    if invocation_axes:
        evidence.add(AccessPatternEvidence.INJECTIVE_GLOBAL_INDEX)
    if non_injective or not mappings:
        evidence.add(AccessPatternEvidence.NON_INJECTIVE_INDEX)
    if (
        invocation_axes == {0, 1, 2}
        and mappings
        and all(mapping == mappings[0] for mapping in mappings[1:])
        and not static
        and not non_injective
    ):
        evidence.add(AccessPatternEvidence.DISJOINT_SCATTER)
    return tuple(sorted(evidence, key=lambda item: item.value)), tuple(sorted(invocation_axes))


def build_autodiff_profile_plan(
    transform: ProgramTransformSpec,
    program: AutodiffProgram,
) -> AutodiffProfilePlan:
    graph = program.semantic
    structs = dict(graph.structs)
    parameter_nodes = tuple(node for node in graph.nodes if node.operation is OpCode.PARAMETER)
    parameters = {node.source_name: node.type for node in parameter_nodes if node.source_name is not None}
    output_type = next(node.type for node in graph.nodes if node.id == graph.outputs[0])
    cotangents = tuple(
        AutodiffBinding(path, _gradient_type(value_type).mlir, "cotangent")
        for path, value_type in _leaves(output_type, structs, ("output",))
    )
    gradients = tuple(
        AutodiffBinding(leaf_path, _gradient_type(value_type).mlir, "gradient")
        for leaf_path, value_type in sorted(
            (
                leaf
                for wrt_path in transform.wrt
                for leaf in _leaves(
                    _resolve_path(wrt_path, parameters, structs),
                    structs,
                    tuple(wrt_path.split(".")),
                )
            ),
            key=lambda leaf: leaf[0],
        )
    )
    primal_inputs = tuple(AutodiffBinding(node.source_name or "", node.type.mlir, "primal") for node in parameter_nodes)
    primal_outputs = (AutodiffBinding("output", output_type.mlir, "primal"),)
    tape = AutodiffBinding("tape", f"!vernon.ad_tape<{program.tape.bytes}>", "tape")
    identity_prefix = hashlib.sha256(f"{transform.identity}:{program.identity}".encode("utf-8")).hexdigest()[:20]
    profiles = (
        AutodiffProfile("primal", graph.entry, primal_inputs, primal_outputs),
        AutodiffProfile(
            "forward_with_tape",
            f"vernon_ad_{identity_prefix}_forward",
            primal_inputs,
            (*primal_outputs, tape),
        ),
        AutodiffProfile(
            "backward",
            f"vernon_ad_{identity_prefix}_backward",
            (tape, *cotangents),
            gradients,
        ),
    )
    return AutodiffProfilePlan(
        transform.identity,
        program.identity,
        program.tape.bytes,
        program.reverse.derivative_rules,
        program.reverse.derivative_rules_version,
        program.launch,
        profiles,
    )


def build_structured_profile_plan(
    transform: ProgramTransformSpec,
    entry: TypedFunctionInstance,
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
    primal_mlir: str,
    tape_bytes: int,
    derivative_rules: tuple[str, ...],
    workgroup_size: tuple[int, int, int],
) -> AutodiffProfilePlan:
    """Build profile metadata without constructing the legacy AD program graph."""
    parameters = {parameter.name: parameter.type for parameter in entry.parameters}
    cotangent_leaves = tuple(
        (
            path,
            _structured_leaves(
                _resolve_path(path, parameters, structs),
                structs,
                tuple(path.split(".")),
            ),
        )
        for path in transform.output_cotangents
    )
    cotangents = tuple(
        AutodiffBinding(leaf_path, _structured_derivative_type(leaf_type, "read").mlir, "cotangent")
        for _, leaves in cotangent_leaves
        for leaf_path, leaf_type in leaves
    )
    gradient_leaves = tuple(
        (
            path,
            _structured_leaves(
                _resolve_path(path, parameters, structs),
                structs,
                tuple(path.split(".")),
            ),
        )
        for path in transform.wrt
    )
    gradients = tuple(
        AutodiffBinding(
            leaf_path,
            _structured_derivative_type(value_type, "write").mlir,
            "gradient",
        )
        for leaf_path, value_type in sorted(
            (leaf for _, leaves in gradient_leaves for leaf in leaves),
            key=lambda leaf: leaf[0],
        )
    )
    derivative_groups = derivative_groups_from_paths(
        "gradient",
        transform.wrt,
        tuple(binding.path for binding in gradients),
    ) + derivative_groups_from_paths(
        "cotangent",
        transform.output_cotangents,
        tuple(binding.path for binding in cotangents),
    )
    if not cotangents:
        raise ValueError("structured VJP result has no differentiable leaves")
    if not gradients:
        raise ValueError("structured VJP wrt paths have no differentiable leaves")
    gradient_group_by_leaf = {leaf_path: path for path, leaves in gradient_leaves for leaf_path, _ in leaves}

    def gradient_source(path: str) -> ConcreteType:
        return _resolve_path(gradient_group_by_leaf[path], parameters, structs)

    def gradient_evidence(path: str) -> tuple[tuple[AccessPatternEvidence, ...], tuple[int, ...]]:
        source = gradient_source(path)
        if source.kind != "tensor_view":
            return (AccessPatternEvidence.SHARED_VALUE,), ()
        wrt_path = gradient_group_by_leaf[path]
        return _tensor_view_evidence(entry, wrt_path.split(".", 1)[0])

    def accumulation_plan(wrt_path: str) -> AccumulationPlan:
        source = _resolve_path(wrt_path, parameters, structs)
        group_leaf_paths = dict(gradient_leaves)[wrt_path]
        evidence_sets = tuple(gradient_evidence(path) for path, _ in group_leaf_paths)
        evidence = tuple(
            sorted(
                {item for items, _ in evidence_sets for item in items},
                key=lambda item: item.value,
            )
        )
        axes = tuple(sorted({axis for _, item_axes in evidence_sets for axis in item_axes}))
        return AccumulationPlan(
            wrt_path,
            AccumulationMode.SCATTER_ADD if source.kind == "tensor_view" else AccumulationMode.REDUCE_SUM,
            evidence,
            axes,
        )

    program_identity = hashlib.sha256(primal_mlir.encode("utf-8")).hexdigest()
    primal_inputs = tuple(
        AutodiffBinding(parameter.name, parameter.type.mlir, "primal")
        for parameter in entry.parameters
        if parameter.builtin is None
    )
    profile_symbols = structured_profile_symbols(transform, entry, primal_mlir)
    profiles = (
        AutodiffProfile(
            "primal",
            profile_symbols[0],
            primal_inputs,
            (),
        ),
        AutodiffProfile(
            "forward_with_tape",
            profile_symbols[1],
            primal_inputs,
            (AutodiffBinding("tape", f"!vernon.ad_tape<{tape_bytes}>", "tape"),),
        ),
        AutodiffProfile(
            "backward",
            profile_symbols[2],
            (
                AutodiffBinding("tape", f"!vernon.ad_tape<{tape_bytes}>", "tape"),
                *cotangents,
            ),
            gradients,
        ),
    )
    return AutodiffProfilePlan(
        transform.identity,
        program_identity,
        tape_bytes,
        derivative_rules,
        transform.derivative_rules_version,
        LaunchPlan(
            workgroup_size,
            tuple(accumulation_plan(wrt_path) for wrt_path in transform.wrt),
        ),
        profiles,
        derivative_groups,
    )


def structured_profile_symbols(
    transform: ProgramTransformSpec,
    entry: TypedFunctionInstance,
    primal_mlir: str,
) -> tuple[str, str, str]:
    program_identity = hashlib.sha256(primal_mlir.encode("utf-8")).hexdigest()
    prefix = hashlib.sha256(f"{transform.identity}:{program_identity}".encode("utf-8")).hexdigest()[:20]
    return entry.symbol, f"vernon_ad_{prefix}_forward", f"vernon_ad_{prefix}_backward"


__all__ = [
    "AutodiffBinding",
    "AutodiffProfile",
    "AutodiffProfilePlan",
    "DerivativeGroup",
    "build_autodiff_profile_plan",
    "build_structured_profile_plan",
    "derivative_groups_from_paths",
    "structured_profile_symbols",
]
