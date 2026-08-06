from __future__ import annotations

import ast
import hashlib
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


def _resolve_path(
    path: str,
    parameters: dict[str, ConcreteType],
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
) -> ConcreteType:
    components = path.split(".")
    value_type = parameters[components[0]]
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
    if entry.result_type is None:
        raise ValueError("structured VJP requires a result")
    parameters = {parameter.name: parameter.type for parameter in entry.parameters}
    cotangents = tuple(
        AutodiffBinding(path, _gradient_type(value_type).mlir, "cotangent")
        for path, value_type in _leaves(entry.result_type, structs, ("output",))
    )
    if not cotangents:
        raise ValueError("structured VJP result has no differentiable leaves")
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
    if not gradients:
        raise ValueError("structured VJP wrt paths have no differentiable leaves")

    def gradient_source(path: str) -> ConcreteType:
        wrt_path = next(
            candidate for candidate in transform.wrt if path == candidate or path.startswith(candidate + ".")
        )
        return _resolve_path(wrt_path, parameters, structs)

    def gradient_evidence(path: str) -> tuple[tuple[AccessPatternEvidence, ...], tuple[int, ...]]:
        source = gradient_source(path)
        if source.kind != "tensor_view":
            return (AccessPatternEvidence.SHARED_VALUE,), ()
        wrt_path = next(
            candidate for candidate in transform.wrt if path == candidate or path.startswith(candidate + ".")
        )
        return _tensor_view_evidence(entry, wrt_path.split(".", 1)[0])

    def accumulation_plan(gradient: AutodiffBinding) -> AccumulationPlan:
        source = gradient_source(gradient.path)
        evidence, invocation_axes = gradient_evidence(gradient.path)
        return AccumulationPlan(
            gradient.path,
            AccumulationMode.SCATTER_ADD if source.kind == "tensor_view" else AccumulationMode.REDUCE_SUM,
            evidence,
            invocation_axes,
        )

    program_identity = hashlib.sha256(primal_mlir.encode("utf-8")).hexdigest()
    primal_inputs = tuple(
        AutodiffBinding(parameter.name, parameter.type.mlir, "primal")
        for parameter in entry.parameters
        if parameter.builtin is None
    )
    output_type = entry.result_type
    profile_symbols = structured_profile_symbols(transform, entry, primal_mlir)
    profiles = (
        AutodiffProfile(
            "primal",
            profile_symbols[0],
            primal_inputs,
            (AutodiffBinding("output", output_type.mlir, "primal"),),
        ),
        AutodiffProfile(
            "forward_with_tape",
            profile_symbols[1],
            primal_inputs,
            (
                AutodiffBinding("output", output_type.mlir, "primal"),
                AutodiffBinding("tape", f"!vernon.ad_tape<{tape_bytes}>", "tape"),
            ),
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
            tuple(accumulation_plan(gradient) for gradient in gradients),
        ),
        profiles,
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
    "build_autodiff_profile_plan",
    "build_structured_profile_plan",
    "structured_profile_symbols",
]
