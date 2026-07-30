from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..language.scalar_types import SCALAR_TYPES


@dataclass(frozen=True)
class SourceType:
    spelling: str


@dataclass(frozen=True)
class LiteralType:
    category: str
    value: int | float


@dataclass(frozen=True)
class ConcreteType:
    kind: str
    name: str
    arguments: tuple["ConcreteType | tuple[int | str, ...] | int | str", ...] = ()

    def __post_init__(self) -> None:
        if self.kind != "tensor" or not self.arguments:
            return
        element = self.arguments[0]
        if not isinstance(element, ConcreteType) or element.kind != "tensor":
            return
        # Tensor nesting is logical shape composition. Structs remain nominal
        # because only an immediately nested Tensor is flattened here.
        object.__setattr__(
            self,
            "arguments",
            (element.arguments[0], *self.arguments[1:], *element.arguments[1:]),
        )

    @property
    def mlir(self) -> str:
        if self.kind == "scalar":
            return SCALAR_TYPES[self.name].mlir
        if self.kind == "index":
            return "index"
        if self.kind == "void":
            return "none"
        if self.kind == "tensor":
            element = self.arguments[0]
            assert isinstance(element, ConcreteType)
            if element.kind != "scalar":
                shape = ", ".join(str(value) for value in self.arguments[1:])
                return f"!vernon.tensor<{element.mlir}, [{shape}]>"
            dimensions = "x".join(str(value) for value in self.arguments[1:])
            return f"tensor<{dimensions}x{element.mlir}>"
        if self.kind == "tuple":
            elements = self.arguments
            assert all(isinstance(element, ConcreteType) for element in elements)
            return f"tuple<{', '.join(element.mlir for element in elements if isinstance(element, ConcreteType))}>"
        if self.kind == "struct":
            return f'!vernon.struct<"{self.name}">'
        if self.kind == "tensor_view_abi":
            element, shape, access, address_space = self.arguments
            assert isinstance(element, ConcreteType)
            assert isinstance(shape, tuple)
            dimensions = ", ".join(str(-1 if extent == "?" else extent) for extent in shape)
            return f'!vernon.tensor_view<{element.mlir}, [{dimensions}], "{access}", "{address_space}">'
        if self.kind == "tensor_storage":
            raise ValueError("TensorStorage is host-runtime only and has no device IR type")
        if self.kind == "tensor_view":
            element, shape, access, address_space = self.arguments
            assert isinstance(element, ConcreteType)
            assert isinstance(shape, tuple)
            dimensions = ", ".join(str(-1 if extent == "?" else extent) for extent in shape)
            return f'!vernon.tensor_view<{element.mlir}, [{dimensions}], "{access}", "{address_space}">'
        if self.kind == "texture":
            dimension, element = self.arguments
            assert isinstance(element, ConcreteType)
            return f'!vernon.texture<"{dimension}", {element.mlir}>'
        if self.kind == "sampler":
            return "!vernon.sampler"
        raise AssertionError(f"unknown type kind {self.kind}")

    @property
    def is_float(self) -> bool:
        if self.kind == "scalar":
            return self.name.startswith("f")
        return self.kind == "tensor" and isinstance(self.arguments[0], ConcreteType) and self.arguments[0].is_float

    @property
    def is_integer(self) -> bool:
        if self.kind == "index":
            return True
        if self.kind == "scalar":
            return self.name in {"i32", "u32"}
        return self.kind == "tensor" and isinstance(self.arguments[0], ConcreteType) and self.arguments[0].is_integer


class SemanticCategory(Enum):
    VALUE = "value"
    STORAGE = "storage"
    RESOURCE = "resource"


def semantic_category(value_type: ConcreteType) -> SemanticCategory | None:
    if value_type.kind in {"scalar", "tensor", "tuple", "struct"}:
        return SemanticCategory.VALUE
    if value_type.kind in {"tensor_storage", "tensor_view"}:
        return SemanticCategory.STORAGE
    if value_type.kind in {"texture", "sampler"}:
        return SemanticCategory.RESOURCE
    return None


def is_abi_stable_value(
    value_type: ConcreteType,
    struct_fields: Callable[[str], tuple[ConcreteType, ...]] | None = None,
    active_structs: frozenset[str] = frozenset(),
) -> bool:
    """Return whether a type has a finite, deterministic Value ABI."""
    if value_type.kind == "scalar":
        return True
    if value_type.kind == "tensor":
        element = value_type.arguments[0] if value_type.arguments else None
        return isinstance(element, ConcreteType) and is_abi_stable_value(element, struct_fields, active_structs)
    if value_type.kind == "tuple":
        return all(
            isinstance(element, ConcreteType) and is_abi_stable_value(element, struct_fields, active_structs)
            for element in value_type.arguments
        )
    if value_type.kind == "struct":
        if struct_fields is None:
            # Nominal declarations are resolved after all Struct bodies are collected.
            return True
        if value_type.name in active_structs:
            return False
        nested = active_structs | {value_type.name}
        return all(is_abi_stable_value(field, struct_fields, nested) for field in struct_fields(value_type.name))
    return False


class AccessMode(Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class TypedExpression:
    source: ast.expr
    type: ConcreteType
    operation: str | None = None
    operand_types: tuple[ConcreteType, ...] = ()
    access: AccessMode = AccessMode.READ


@dataclass(frozen=True)
class TypedParameter:
    name: str
    type: ConcreteType
    access: AccessMode = AccessMode.READ


class Effect(Enum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"


class StorageEffectKind(Enum):
    READ = "read"
    WRITE = "write"


class StorageRegionKind(Enum):
    ELEMENT = "element"
    UNKNOWN = "unknown"


class StorageOwnerKind(Enum):
    PARAMETER = "parameter"
    WORKGROUP_LOCAL = "workgroup_local"


@dataclass(frozen=True)
class StorageOwner:
    kind: StorageOwnerKind
    name: str


@dataclass(frozen=True)
class StorageRegion:
    kind: StorageRegionKind
    indices: tuple[int, ...] = ()

    def overlaps(self, other: "StorageRegion") -> bool:
        if self.kind is StorageRegionKind.UNKNOWN or other.kind is StorageRegionKind.UNKNOWN:
            return True
        return self.indices == other.indices


@dataclass(frozen=True)
class StorageEffect:
    kind: StorageEffectKind
    owner: StorageOwner
    region: StorageRegion


@dataclass(frozen=True)
class ResourceEffect:
    operation: str
    owner: str
    stages: frozenset[str] = frozenset()
    has_lod: bool = False


class MemoryOrdering(Enum):
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQUIRE_RELEASE = "acquire_release"
    SEQUENTIAL = "sequential"


class EffectScope(Enum):
    INVOCATION = "invocation"
    WORKGROUP = "workgroup"
    DEVICE = "device"


@dataclass(frozen=True)
class AtomicEffect:
    operation: str
    owner: StorageOwner
    region: StorageRegion
    ordering: MemoryOrdering
    scope: EffectScope


@dataclass(frozen=True)
class BarrierEffect:
    ordering: MemoryOrdering
    scope: EffectScope


TypedEffect = StorageEffect | ResourceEffect | AtomicEffect | BarrierEffect


class Termination(Enum):
    FALLTHROUGH = "fallthrough"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"


@dataclass(frozen=True)
class LValue:
    kind: str
    name: str
    type: ConcreteType
    access: AccessMode = AccessMode.READ_WRITE


@dataclass(frozen=True)
class BranchMerge:
    name: str
    type: ConcreteType


@dataclass(frozen=True)
class TypedStatement:
    source: ast.stmt
    termination: Termination
    effects: tuple[TypedEffect, ...] = ()
    expressions: tuple[TypedExpression, ...] = ()
    lvalues: tuple[LValue, ...] = ()
    branch_merges: tuple[BranchMerge, ...] = ()
    children: tuple["TypedStatement", ...] = ()
    loop_depth: int = 0
    return_type: ConcreteType | None = None

    @property
    def effect(self) -> Effect:
        nested_effects = self._nested_effects()
        storage_effects = tuple(effect for effect in nested_effects if isinstance(effect, StorageEffect))
        if any(effect.kind is StorageEffectKind.WRITE for effect in storage_effects) or any(
            isinstance(effect, (AtomicEffect, BarrierEffect)) for effect in nested_effects
        ):
            return Effect.WRITE
        if any(effect.kind is StorageEffectKind.READ for effect in storage_effects):
            return Effect.READ
        if any(isinstance(effect, ResourceEffect) for effect in nested_effects):
            return Effect.READ
        return Effect.PURE

    def _nested_effects(self) -> tuple[TypedEffect, ...]:
        return (
            *self.effects,
            *(effect for child in self.children for effect in child._nested_effects()),
        )

    def _nested_storage_effects(self) -> tuple[StorageEffect, ...]:
        return tuple(effect for effect in self._nested_effects() if isinstance(effect, StorageEffect))


@dataclass(frozen=True)
class TypedFunctionInstance:
    qualified_name: str
    symbol: str
    argument_types: tuple[ConcreteType, ...]
    result_type: ConcreteType | None
    enabled_features: tuple[str, ...]
    source: ast.FunctionDef
    body: tuple[TypedStatement, ...] = ()
    parameters: tuple[TypedParameter, ...] = ()
    effects: tuple[TypedEffect, ...] = ()

    @property
    def specialization_key(self) -> tuple[str, tuple[ConcreteType, ...], tuple[str, ...]]:
        return self.qualified_name, self.argument_types, self.enabled_features


SemanticType = SourceType | LiteralType | ConcreteType
SemanticValue = (
    SemanticType
    | AtomicEffect
    | BarrierEffect
    | EffectScope
    | MemoryOrdering
    | TypedExpression
    | TypedParameter
    | StorageOwner
    | StorageOwnerKind
    | StorageRegion
    | StorageEffect
    | TypedEffect
    | LValue
    | BranchMerge
    | TypedStatement
    | TypedFunctionInstance
    | Any
)
