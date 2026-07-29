from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..diagnostics import CompileError, SourceLocation
from .model import AccessMode, ConcreteType, TypedFunctionInstance
from .type_parser import AnnotatedType

DslType = ConcreteType


def element_type(value_type: DslType) -> DslType:
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        assert isinstance(element, DslType)
        return element
    return value_type


@dataclass(frozen=True)
class FunctionSignature:
    arguments: tuple[DslType, ...]
    result: DslType | None


@dataclass(frozen=True)
class ViewLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int

    def __post_init__(self) -> None:
        if not self.shape or any(
            not isinstance(extent, int) or isinstance(extent, bool) or extent < 0 for extent in self.shape
        ):
            raise ValueError("TensorView specialization shape must contain non-negative integer extents")
        if len(self.strides) != len(self.shape) or any(
            not isinstance(stride, int) or isinstance(stride, bool) for stride in self.strides
        ):
            raise ValueError("TensorView specialization requires one signed integer stride per dimension")
        if not isinstance(self.offset, int) or isinstance(self.offset, bool) or self.offset < 0:
            raise ValueError("TensorView specialization offset must be a non-negative integer")


@dataclass(frozen=True)
class Value:
    name: str
    type: DslType
    fields: tuple["Value", ...] | None = None
    access: AccessMode = AccessMode.READ
    view_layout: ViewLayout | None = None

    @property
    def abi_type(self) -> DslType:
        if self.type.kind == "tensor_view":
            element, rank, _ = self.type.arguments
            assert isinstance(element, DslType)
            return DslType("tensor_view_abi", "TensorViewAbi", (element, rank, self.access.value))
        return self.type


@dataclass
class ModuleContext:
    filename: str
    runtime_entry: str | None = None
    tensor_view_layouts: dict[str, ViewLayout] = field(default_factory=dict)
    structs: dict[str, tuple[tuple[str, AnnotatedType], ...]] = field(default_factory=dict)
    signatures: dict[str, FunctionSignature] = field(default_factory=dict)
    result_annotations: dict[str, AnnotatedType | None] = field(default_factory=dict)
    shared_functions: set[str] = field(default_factory=set)
    typed_functions: dict[str, TypedFunctionInstance] = field(default_factory=dict)

    def error(self, node: ast.AST, message: str) -> CompileError:
        return CompileError(
            message,
            SourceLocation(
                self.filename,
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
            ),
        )
