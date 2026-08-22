"""Typed, resource-versioned program graph used above execution passes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ._runtime.resources import TensorStorage, TensorView


class OperationKind(Enum):
    KERNEL_CALL = "kernel_call"


@dataclass(frozen=True)
class ResourceType:
    shape: tuple[int, ...]
    dtype: str
    byte_size: int


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
class KernelCallOp:
    id: int
    kind: OperationKind
    name: str
    kernel: Any
    binding_slots: Mapping[str, int]
    parameters: tuple[KernelParameter, ...]
    inputs: Mapping[str, int]
    outputs: Mapping[str, int]
    grid: tuple[int, int, int]
    features: tuple[str, ...]


def _resource_owner(value: Any) -> TensorStorage | None:
    if isinstance(value, TensorStorage):
        return value
    if isinstance(value, TensorView) and isinstance(value.owner, TensorStorage):
        return value.owner
    return None


def _resource_type(value: TensorStorage | TensorView) -> ResourceType:
    return ResourceType(
        tuple(value.shape),
        str(value.dtype),
        int(value.dtype.itemsize) * math.prod(value.shape),
    )


class OperationGraph:
    """Primal Module graph with conservative owner-level resource versioning."""

    def __init__(self):
        self._values: list[ResourceVersion] = []
        self._operations: list[KernelCallOp] = []
        self._current: dict[int, int] = {}
        self._owner_ids: dict[int, int] = {}
        self._owner_objects: dict[int, TensorStorage] = {}
        self._inputs: dict[str, int] = {}
        self._outputs: dict[str, int] = {}

    @property
    def values(self) -> tuple[ResourceVersion, ...]:
        return tuple(self._values)

    @property
    def operations(self) -> tuple[KernelCallOp, ...]:
        return tuple(self._operations)

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
        version = self._current_version(value)
        existing = self._inputs.get(name)
        if existing is not None and existing != version:
            raise ValueError(f"Module input {name!r} was imported with inconsistent resource versions")
        self._inputs[name] = version

    def append_kernel(
        self,
        *,
        name: str,
        kernel: Any,
        bindings: Mapping[str, Any],
        binding_slots: Mapping[str, int],
        parameters: tuple[KernelParameter, ...],
        grid: tuple[int, int, int],
        features: tuple[str, ...],
    ) -> KernelCallOp:
        operation_id = len(self._operations)
        inputs: dict[str, int] = {}
        outputs: dict[str, int] = {}
        for parameter in parameters:
            value = bindings[parameter.name]
            owner = _resource_owner(value)
            if owner is None:
                continue
            before = self._current_version(value)
            if parameter.access in {"read", "read_write"}:
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
            grid,
            features,
        )
        self._operations.append(operation)
        return operation

    def set_outputs(self, outputs: Mapping[str, Any]) -> None:
        resolved: dict[str, int] = {}
        for path, value in outputs.items():
            if _resource_owner(value) is not None:
                resolved[path] = self._current_version(value)
        if not resolved:
            raise ValueError("Module.forward() must return at least one TensorStorage or TensorView")
        self._outputs = resolved
        self._validate()

    def owner_id(self, value: TensorStorage | TensorView) -> int:
        """Return the stable owner ID assigned during this graph capture."""

        owner = _resource_owner(value)
        if owner is None or id(owner) not in self._owner_ids:
            raise ValueError("resource is not present in this Program graph")
        return self._owner_ids[id(owner)]

    def _current_version(self, value: TensorStorage | TensorView) -> int:
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
        existing = self._current.get(owner_id)
        if existing is not None:
            return existing
        value_id = len(self._values)
        self._values.append(ResourceVersion(value_id, owner_id, 0, _resource_type(value), None))
        self._current[owner_id] = value_id
        return value_id

    def _advance(self, value: TensorStorage | TensorView, producer: int) -> int:
        owner = _resource_owner(value)
        assert owner is not None
        previous = self._values[self._current_version(value)]
        value_id = len(self._values)
        self._values.append(
            ResourceVersion(value_id, previous.owner, previous.version + 1, _resource_type(value), producer)
        )
        self._current[previous.owner] = value_id
        return value_id

    def _validate(self) -> None:
        produced: set[int] = {value.id for value in self._values if value.producer is None}
        for operation in self._operations:
            if any(value not in produced for value in operation.inputs.values()):
                raise ValueError(f"KernelCallOp {operation.name!r} consumes an unavailable resource version")
            produced.update(operation.outputs.values())
        if any(value not in produced for value in self._outputs.values()):
            raise ValueError("Module output refers to an unavailable resource version")


__all__ = [
    "KernelCallOp",
    "KernelParameter",
    "OperationGraph",
    "OperationKind",
    "ResourceType",
    "ResourceVersion",
]
