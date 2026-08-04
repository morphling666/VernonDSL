from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..ad import ProgramTransformSpec
from ..bundle import canonical_json
from .autodiff import AutodiffProgram, LaunchPlan, OpCode
from .model import ConcreteType


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
        for path, value_type in sorted(_leaves(output_type, structs, ("output",)), key=lambda leaf: leaf[0])
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


__all__ = [
    "AutodiffBinding",
    "AutodiffProfile",
    "AutodiffProfilePlan",
    "build_autodiff_profile_plan",
]
