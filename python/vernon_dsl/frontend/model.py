from __future__ import annotations

import ast
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
    arguments: tuple["ConcreteType | int | str", ...] = ()

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
            dimensions = "x".join(str(value) for value in self.arguments[1:])
            return f"tensor<{dimensions}x{element.mlir}>"
        if self.kind == "struct":
            return f'!vernon.struct<"{self.name}">'
        if self.kind == "buffer":
            element = self.arguments[0]
            access = self.arguments[1] if len(self.arguments) > 1 else "read_write"
            assert isinstance(element, ConcreteType)
            return f'!vernon.buffer<{element.mlir}, "{access}">'
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


class StorageClass(Enum):
    VALUE = "value"
    ADDRESSABLE = "addressable"


class AccessMode(Enum):
    READ = "read"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class TypedExpression:
    source: ast.expr
    type: ConcreteType
    operation: str | None = None
    operand_types: tuple[ConcreteType, ...] = ()
    storage: StorageClass = StorageClass.VALUE
    access: AccessMode = AccessMode.READ


@dataclass(frozen=True)
class TypedParameter:
    name: str
    type: ConcreteType
    storage: StorageClass = StorageClass.VALUE
    access: AccessMode = AccessMode.READ


class Effect(Enum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"


class Termination(Enum):
    FALLTHROUGH = "fallthrough"
    RETURN = "return"


@dataclass(frozen=True)
class LValue:
    kind: str
    name: str
    type: ConcreteType
    storage: StorageClass = StorageClass.VALUE
    access: AccessMode = AccessMode.READ_WRITE


@dataclass(frozen=True)
class BranchMerge:
    name: str
    type: ConcreteType


@dataclass(frozen=True)
class TypedStatement:
    source: ast.stmt
    effect: Effect
    termination: Termination
    expressions: tuple[TypedExpression, ...] = ()
    lvalues: tuple[LValue, ...] = ()
    branch_merges: tuple[BranchMerge, ...] = ()
    children: tuple["TypedStatement", ...] = ()


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

    @property
    def specialization_key(self) -> tuple[str, tuple[ConcreteType, ...], tuple[str, ...]]:
        return self.qualified_name, self.argument_types, self.enabled_features


SemanticType = SourceType | LiteralType | ConcreteType
SemanticValue = (
    SemanticType
    | TypedExpression
    | TypedParameter
    | LValue
    | BranchMerge
    | TypedStatement
    | TypedFunctionInstance
    | Any
)
