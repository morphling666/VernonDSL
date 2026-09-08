"""Typed, resource-versioned program graph used above execution passes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .frontend.model import ConcreteType
from .frontend.runtime_types import RuntimeParameterDescriptor


class OperationKind(Enum):
    ALLOC = "alloc"
    KERNEL_CALL = "kernel_call"
    GRAPHICS_CALL = "graphics_call"


@dataclass
class GraphBuffer:
    """SSA symbol for a Program Storage full view. Not a runtime allocation."""

    dtype: Any
    shape: tuple[int, ...]
    access: str
    as_view: bool = False
    logical: ConcreteType | None = None


@dataclass(frozen=True, eq=False)
class GraphValueInput:
    """Typed invocation Value that remains a Program argument."""

    name: str
    descriptor: RuntimeParameterDescriptor

    @property
    def annotation(self) -> Any:
        return self.descriptor.annotation

    @property
    def logical(self) -> ConcreteType:
        return self.descriptor.logical


@dataclass(frozen=True, eq=False)
class GraphResourceInput:
    """Typed non-storage resource imported by one Program invocation."""

    name: str
    descriptor: RuntimeParameterDescriptor

    @property
    def annotation(self) -> Any:
        return self.descriptor.annotation

    @property
    def logical(self) -> ConcreteType:
        return self.descriptor.logical


@dataclass(frozen=True, eq=False)
class GraphControlInput:
    """A typed Program invocation control, never a public boundary value."""

    name: str
    kind: str
    prototype: Any = None

    @property
    def ref(self) -> ProgramControlRef:
        return ProgramControlRef(self.kind, ProgramControlSource.ARGUMENT, self.name)


@dataclass(frozen=True)
class ProgramControlDescriptor:
    kind: str
    annotation: Any
    prototype: Any = None


class ProgramControlSource(Enum):
    ARGUMENT = "argument"
    CAPTURE = "capture"


@dataclass(frozen=True)
class ProgramControlRef:
    """Stable identity of one invocation control in a captured Program."""

    kind: str
    source: ProgramControlSource
    identifier: str | int


@dataclass(frozen=True)
class AttachmentProjection:
    """One attachment projected from a RenderPass control."""

    control: ProgramControlRef
    aspect: str
    location: int | None


@dataclass(frozen=True)
class GraphAttachmentOutput:
    """Symbolic output version selected from a RenderPass control."""

    render_pass: GraphControlInput
    projection: AttachmentProjection


class DispatchControlKind(Enum):
    STATIC = "static"
    VALUE = "value"


@dataclass(frozen=True)
class DispatchControl:
    """One dispatch axis, either compile-time static or invocation-resolved."""

    kind: DispatchControlKind
    static_value: int | None = None
    value: GraphValueInput | None = None

    def __post_init__(self) -> None:
        if self.kind is DispatchControlKind.STATIC:
            if self.static_value is None or self.static_value <= 0 or self.value is not None:
                raise ValueError("static dispatch control requires one positive integer")
        elif self.kind is DispatchControlKind.VALUE:
            if self.static_value is not None or not isinstance(self.value, GraphValueInput):
                raise ValueError("dynamic dispatch control requires one Program Value")
        else:
            raise ValueError(f"unknown dispatch control kind {self.kind!r}")

    @classmethod
    def from_value(cls, value: int | GraphValueInput) -> DispatchControl:
        if isinstance(value, GraphValueInput):
            return cls(DispatchControlKind.VALUE, value=value)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("static dispatch controls must be positive integers")
        return cls(DispatchControlKind.STATIC, static_value=value)


@dataclass(frozen=True, eq=False)
class GraphControlResource:
    """An attachment resource projected from a RenderPass control."""

    name: str
    shape: tuple[int, ...]
    logical: ConcreteType
    projection: AttachmentProjection


@dataclass(frozen=True)
class ResourceType:
    shape: tuple[int, ...]
    dtype: str
    logical: ConcreteType | None = None


@dataclass(frozen=True)
class ResourceVersion:
    id: int
    owner: int
    version: int
    type: ResourceType
    producer: int | None


@dataclass(frozen=True)
class KernelParameter:
    name: str
    access: str
    differentiable_input: bool
    differentiable_output: bool


@dataclass(frozen=True)
class AllocOp:
    id: int
    kind: OperationKind
    name: str
    like: int | None
    values: Any
    result: int


@dataclass(frozen=True)
class KernelCallOp:
    id: int
    kind: OperationKind
    name: str
    kernel: Any
    binding_slots: Mapping[str, int]
    parameters: tuple[KernelParameter, ...]
    inputs: Mapping[str, int]
    outputs: Mapping[str, int]
    grid: tuple[DispatchControl, DispatchControl, DispatchControl]
    features: tuple[str, ...]


@dataclass(frozen=True)
class GraphicsCallOp:
    id: int
    kind: OperationKind
    name: str
    pipeline: Any
    binding_slots: Mapping[str, int]
    control_slots: Mapping[str, int]
    parameters: tuple[KernelParameter, ...]
    inputs: Mapping[str, int]
    outputs: Mapping[str, int]
    attachment_names: tuple[str, ...]
    color_count: int
    features: tuple[str, ...]


def _resource_owner(value: Any) -> Any | None:
    return value if isinstance(value, (GraphBuffer, GraphResourceInput, GraphControlResource)) else None


def _resource_type(value: Any) -> ResourceType:
    if isinstance(value, GraphResourceInput):
        return ResourceType((), value.logical.mlir, value.logical)
    if isinstance(value, GraphControlResource):
        return ResourceType(value.shape, value.logical.mlir, value.logical)
    return ResourceType(
        tuple(value.shape),
        str(getattr(value, "dtype", value)),
        value.logical,
    )


class OperationGraph:
    """Primal Module graph with owner-level resource versioning."""

    def __init__(self):
        self._values: list[ResourceVersion] = []
        self._nodes: list[AllocOp | KernelCallOp | GraphicsCallOp] = []
        self._current: dict[int, int] = {}
        self._owner_ids: dict[int, int] = {}
        self._owner_objects: dict[int, Any] = {}
        self._inputs: dict[str, int] = {}
        self._outputs: dict[str, int] = {}

    @property
    def values(self) -> tuple[ResourceVersion, ...]:
        return tuple(self._values)

    @property
    def nodes(self) -> tuple[AllocOp | KernelCallOp | GraphicsCallOp, ...]:
        return tuple(self._nodes)

    @property
    def operations(self) -> tuple[KernelCallOp | GraphicsCallOp, ...]:
        return tuple(node for node in self._nodes if isinstance(node, (KernelCallOp, GraphicsCallOp)))

    @property
    def inputs(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._inputs))

    @property
    def outputs(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._outputs))

    def import_input(self, name: str, value: Any) -> None:
        owner = _resource_owner(value)
        if owner is None:
            return
        version = self._define_version(value, producer=None)
        existing = self._inputs.get(name)
        if existing is not None and existing != version:
            raise ValueError(f"Module input {name!r} was imported with inconsistent resource versions")
        self._inputs[name] = version

    def import_control_resource(self, value: GraphControlResource) -> int:
        return self._define_version(value, producer=None)

    def append_alloc(
        self,
        *,
        buffer: GraphBuffer,
        name: str,
        like: GraphBuffer | None = None,
        values: Any = None,
    ) -> AllocOp:
        operation_id = len(self._nodes)
        like_id = None if like is None else self._current_version(like)
        result = self._define_version(buffer, producer=operation_id)
        operation = AllocOp(operation_id, OperationKind.ALLOC, name, like_id, values, result)
        self._nodes.append(operation)
        return operation

    def append_kernel(
        self,
        *,
        name: str,
        kernel: Any,
        bindings: Mapping[str, Any],
        binding_slots: Mapping[str, int],
        parameters: tuple[KernelParameter, ...],
        grid: tuple[int | GraphValueInput, int | GraphValueInput, int | GraphValueInput],
        features: tuple[str, ...],
    ) -> KernelCallOp:
        operation_id = len(self._nodes)
        inputs: dict[str, int] = {}
        outputs: dict[str, int] = {}
        for parameter in parameters:
            value = bindings[parameter.name]
            owner = _resource_owner(value)
            if owner is None:
                continue
            before = self._current_version(value)
            inputs[parameter.name] = before
            if parameter.access in {"write", "read_write"}:
                outputs[parameter.name] = self._advance(value, operation_id)
        operation = KernelCallOp(
            operation_id,
            OperationKind.KERNEL_CALL,
            name,
            kernel,
            MappingProxyType(dict(binding_slots)),
            parameters,
            MappingProxyType(inputs),
            MappingProxyType(outputs),
            tuple(DispatchControl.from_value(component) for component in grid),
            features,
        )
        self._nodes.append(operation)
        return operation

    def append_graphics(
        self,
        *,
        name: str,
        pipeline: Any,
        bindings: Mapping[str, Any],
        binding_slots: Mapping[str, int],
        control_slots: Mapping[str, int],
        parameters: tuple[KernelParameter, ...],
        attachments: tuple[GraphControlResource, ...],
        color_count: int,
        features: tuple[str, ...],
    ) -> GraphicsCallOp:
        operation_id = len(self._nodes)
        inputs: dict[str, int] = {attachment.name: self._current_version(attachment) for attachment in attachments}
        outputs: dict[str, int] = {
            attachment.name: self._advance(attachment, operation_id) for attachment in attachments
        }
        for parameter in parameters:
            value = bindings[parameter.name]
            owner = _resource_owner(value)
            if owner is None:
                continue
            inputs[parameter.name] = self._current_version(value)
            if parameter.access in {"write", "read_write"}:
                outputs[parameter.name] = self._advance(value, operation_id)
        operation = GraphicsCallOp(
            operation_id,
            OperationKind.GRAPHICS_CALL,
            name,
            pipeline,
            MappingProxyType(dict(binding_slots)),
            MappingProxyType(dict(control_slots)),
            parameters,
            MappingProxyType(inputs),
            MappingProxyType(outputs),
            tuple(attachment.name for attachment in attachments),
            color_count,
            features,
        )
        self._nodes.append(operation)
        return operation

    def set_outputs(self, outputs: Mapping[str, Any]) -> None:
        resolved: dict[str, int] = {}
        for path, value in outputs.items():
            if _resource_owner(value) is not None:
                resolved[path] = self._current_version(value)
        if not resolved and not any(isinstance(node, (KernelCallOp, GraphicsCallOp)) for node in self._nodes):
            raise ValueError("Module.forward() must return at least one Program buffer")
        self._outputs = resolved
        self._validate()

    def owner_id(self, value: Any) -> int:
        owner = _resource_owner(value)
        if owner is None or id(owner) not in self._owner_ids:
            raise ValueError("resource is not present in this Program graph")
        return self._owner_ids[id(owner)]

    def refine_logical_type(self, value_id: int, logical: ConcreteType) -> None:
        """Attach compiler-resolved type information to every version of one owner."""

        if value_id < 0 or value_id >= len(self._values):
            raise ValueError(f"Program value {value_id} is not present in this graph")
        owner = self._values[value_id].owner
        for index, value in enumerate(self._values):
            if value.owner != owner:
                continue
            resource_type = ResourceType(value.type.shape, value.type.dtype, logical)
            self._values[index] = ResourceVersion(
                value.id,
                value.owner,
                value.version,
                resource_type,
                value.producer,
            )

    def _owner(self, value: Any) -> int:
        owner = _resource_owner(value)
        assert owner is not None
        identity = id(owner)
        owner_id = self._owner_ids.get(identity)
        if owner_id is None:
            owner_id = len(self._owner_ids)
            self._owner_ids[identity] = owner_id
            self._owner_objects[identity] = owner
        elif self._owner_objects[identity] is not owner:
            raise RuntimeError("Python resource identity collision during Program capture")
        return owner_id

    def _define_version(self, value: Any, *, producer: int | None) -> int:
        owner_id = self._owner(value)
        existing = self._current.get(owner_id)
        if existing is not None:
            return existing
        value_id = len(self._values)
        self._values.append(ResourceVersion(value_id, owner_id, 0, _resource_type(value), producer))
        self._current[owner_id] = value_id
        return value_id

    def _current_version(self, value: Any) -> int:
        owner_id = self._owner(value)
        existing = self._current.get(owner_id)
        if existing is None:
            raise ValueError("resource version is not defined in this Program graph")
        return existing

    def _advance(self, value: Any, producer: int) -> int:
        previous = self._values[self._current_version(value)]
        value_id = len(self._values)
        self._values.append(
            ResourceVersion(
                value_id,
                previous.owner,
                previous.version + 1,
                _resource_type(value),
                producer,
            )
        )
        self._current[previous.owner] = value_id
        return value_id

    def _validate(self) -> None:
        produced: set[int] = set(self._inputs.values())
        for node in self._nodes:
            if isinstance(node, AllocOp):
                if node.like is not None and node.like not in produced:
                    raise ValueError(f"AllocOp {node.name!r} copies an unavailable resource version")
                produced.add(node.result)
                continue
            control_values = {
                value
                for name, value in node.inputs.items()
                if isinstance(node, GraphicsCallOp) and name in node.attachment_names
            }
            produced.update(control_values)
            if any(value not in produced for value in node.inputs.values()):
                raise ValueError(f"{type(node).__name__} {node.name!r} consumes an unavailable resource version")
            produced.update(node.outputs.values())
        if any(value not in produced for value in self._outputs.values()):
            raise ValueError("Module output refers to an unavailable resource version")


__all__ = [
    "AllocOp",
    "DispatchControl",
    "DispatchControlKind",
    "GraphBuffer",
    "GraphControlInput",
    "GraphControlResource",
    "GraphResourceInput",
    "GraphValueInput",
    "GraphicsCallOp",
    "KernelCallOp",
    "KernelParameter",
    "OperationGraph",
    "OperationKind",
    "ProgramControlDescriptor",
    "ResourceType",
    "ResourceVersion",
]
