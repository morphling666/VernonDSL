from __future__ import annotations

from dataclasses import dataclass

from .autodiff import AutodiffProgram, OpCode, ProgramGraphNode
from .model import ConcreteType


class NativeAbiError(ValueError):
    pass


@dataclass(frozen=True)
class ResourceBindingPlan:
    name: str
    value_type: ConcreteType
    access: str
    role: str
    runtime_carrier: bool = False

    def resource_type(self) -> str:
        return resource_type(self.value_type, self.access, self.runtime_carrier)

    def argument(self, binding: int) -> str:
        return resource_argument(self, binding)


def value_dtype(value_type: ConcreteType) -> str:
    element = value_type.arguments[0] if value_type.kind in {"tensor", "tensor_view"} else value_type
    if not isinstance(element, ConcreteType) or element.kind != "scalar":
        raise NativeAbiError(f"{value_type.mlir} has no native scalar element type")
    return element.name


def dense_value_type(value_type: ConcreteType) -> ConcreteType:
    if value_type.kind != "tensor_view":
        return value_type
    element, shape, _, _ = value_type.arguments
    if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
        raise NativeAbiError("TensorView type is unresolved")
    return ConcreteType("tensor", "Tensor", (element, *shape))


def native_type(value_type: ConcreteType) -> str:
    return dense_value_type(value_type).mlir


def resource_type(value_type: ConcreteType, access: str, runtime_carrier: bool = False) -> str:
    if runtime_carrier:
        return f'!vernon.tensor_view<{native_type(value_type)}, [-1, -1, -1], "{access}", "device">'
    if value_type.kind == "tensor_view":
        element, shape, _, address_space = value_type.arguments
        if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
            raise NativeAbiError("TensorView resource type is unresolved")
        dimensions = ", ".join(str(extent) for extent in shape)
        return f'!vernon.tensor_view<{element.mlir}, [{dimensions}], "{access}", "{address_space}">'
    return f'!vernon.tensor_view<{value_type.mlir}, [1], "{access}", "device">'


def resource_argument(resource: ResourceBindingPlan, binding: int) -> str:
    return (
        f"%resource{binding}: {resource.resource_type()} "
        f'{{vernon.interface = "resource", vernon.source_name = "{resource.name}", '
        f'vernon.autodiff_role = "{resource.role}", '
        f'vernon.element_abi_leaf_dtypes = ["{value_dtype(resource.value_type)}"], '
        f"vernon.set = 0 : i64, vernon.binding = {binding} : i64}}"
    )


def _parameters(program: AutodiffProgram) -> tuple[ProgramGraphNode, ...]:
    return tuple(node for node in program.semantic.nodes if node.operation is OpCode.PARAMETER)


def launch_value_type() -> ConcreteType:
    return ConcreteType(
        "tensor",
        "Tensor",
        (ConcreteType("scalar", "u32"), 3),
    )


def forward_resources(program: AutodiffProgram) -> tuple[ResourceBindingPlan, ...]:
    graph = program.semantic
    nodes = {node.id: node for node in graph.nodes}
    output = nodes[graph.outputs[0]]
    return (
        *(
            ResourceBindingPlan(
                node.source_name or "",
                node.type,
                str(node.type.arguments[2]) if node.type.kind == "tensor_view" else "read",
                "storage" if node.type.kind == "tensor_view" and node.type.arguments[2] != "read" else "input",
            )
            for node in _parameters(program)
        ),
        ResourceBindingPlan("__vernon_launch", launch_value_type(), "read", "input"),
        ResourceBindingPlan("output", output.type, "write", "output", True),
        *(
            ResourceBindingPlan(f"tape.{index}", nodes[value].type, "write", "tape", True)
            for index, value in enumerate(program.reverse.saved_values)
        ),
    )


def backward_resources(program: AutodiffProgram) -> tuple[ResourceBindingPlan, ...]:
    graph = program.semantic
    nodes = {node.id: node for node in graph.nodes}
    output = nodes[graph.outputs[0]]
    gradient_nodes = {node.source_name: node for node in graph.nodes if node.operation is OpCode.PARAMETER}
    resources = [
        ResourceBindingPlan("__vernon_launch", launch_value_type(), "read", "input"),
        *[
            ResourceBindingPlan(f"tape.{index}", nodes[value].type, "read", "tape", True)
            for index, value in enumerate(program.reverse.saved_values)
        ],
    ]
    resources.append(ResourceBindingPlan("output", output.type, "read", "cotangent", True))
    for path in graph.wrt:
        node = gradient_nodes[path]
        gradient_type = node.type
        if node.type.kind == "tensor":
            element = node.type.arguments[0]
            assert isinstance(element, ConcreteType)
            gradient_type = ConcreteType(
                "tensor_view",
                "TensorView",
                (element, tuple(node.type.arguments[1:]), "read_write", "device"),
            )
        resources.append(
            ResourceBindingPlan(
                path,
                gradient_type,
                "read_write",
                "gradient",
            )
        )
    return tuple(resources)


__all__ = [
    "NativeAbiError",
    "ResourceBindingPlan",
    "backward_resources",
    "dense_value_type",
    "forward_resources",
    "launch_value_type",
    "native_type",
    "resource_argument",
    "resource_type",
    "value_dtype",
]
