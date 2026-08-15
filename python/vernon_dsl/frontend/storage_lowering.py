from __future__ import annotations

import ast
from typing import Protocol

from .lowering_types import DslType, ModuleContext, Value
from .model import AccessMode


class StorageEmitter(Protocol):
    context: ModuleContext

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value: ...

    def _fresh(self) -> str: ...

    def _line(self, text: str) -> None: ...

    def _coerce_implicit(self, node: ast.AST, value: Value, target: DslType) -> Value: ...


def lower_buffer_index(emitter: StorageEmitter, node: ast.AST) -> Value:
    if isinstance(node, ast.Tuple):
        raise emitter.context.error(node, "buffers require exactly one index")
    index = emitter._expression(node)
    if not index.type.is_integer:
        raise emitter.context.error(node, "buffer index must be an integer")
    if index.canonical_index is not None:
        return Value(index.canonical_index, DslType("index", "index"))
    if index.type.mlir == "index":
        return index
    cast = emitter._fresh()
    emitter._line(f"{cast} = arith.index_cast {index.name} : {index.type.mlir} to index")
    return Value(cast, DslType("index", "index"))


def lower_tensor_view_indices(
    emitter: StorageEmitter,
    node: ast.Subscript,
    value: Value,
) -> tuple[Value, ...]:
    shape = value.type.arguments[1]
    assert isinstance(shape, tuple)
    rank = len(shape)
    index_nodes = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if len(index_nodes) != rank:
        raise emitter.context.error(node, "TensorView indexing requires one index per dimension")
    return tuple(lower_buffer_index(emitter, index_node) for index_node in index_nodes)


def lower_storage_store(emitter: StorageEmitter, target: ast.Subscript, value: Value) -> None:
    buffer = emitter._expression(target.value)
    if buffer.type.kind != "tensor_view":
        raise emitter.context.error(target, "indexed assignment requires TensorView storage")
    element = buffer.type.arguments[0]
    assert isinstance(element, DslType)
    if buffer.access is AccessMode.READ:
        raise emitter.context.error(target, "cannot assign through a read-only TensorView")
    value = emitter._coerce_implicit(target, value, element)
    indices = lower_tensor_view_indices(emitter, target, buffer)
    operands = ", ".join((value.name, buffer.name, *(index.name for index in indices)))
    operand_types = ", ".join((element.mlir, buffer.abi_type.mlir, *(["index"] * len(indices))))
    emitter._line(f'"vernon.store"({operands}) : ({operand_types}) -> ()')
