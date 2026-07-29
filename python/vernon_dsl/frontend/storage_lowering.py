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
    if index.type.mlir == "index":
        return index
    cast = emitter._fresh()
    emitter._line(f"{cast} = arith.index_cast {index.name} : {index.type.mlir} to index")
    return Value(cast, DslType("index", "index"))


def lower_tensor_view_index(emitter: StorageEmitter, node: ast.Subscript, value: Value) -> Value:
    rank = value.type.arguments[1]
    index_nodes = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if len(index_nodes) != rank:
        raise emitter.context.error(node, "TensorView indexing requires one index per dimension")
    if value.view_layout is None:
        if rank == 1:
            return lower_buffer_index(emitter, index_nodes[0])
        raise emitter.context.error(
            node,
            "multi-dimensional TensorView lowering requires runtime stride descriptors and is not implemented",
        )
    layout = value.view_layout
    if len(layout.shape) != rank or len(layout.strides) != rank:
        raise emitter.context.error(node, "runtime TensorView layout rank does not match its annotation")

    physical: Value | None = None
    if layout.offset:
        offset = emitter._fresh()
        emitter._line(f"{offset} = arith.constant {layout.offset} : index")
        physical = Value(offset, DslType("index", "index"))
    for index_node, stride in zip(index_nodes, layout.strides, strict=True):
        term = lower_buffer_index(emitter, index_node)
        if stride != 1:
            stride_value = emitter._fresh()
            emitter._line(f"{stride_value} = arith.constant {stride} : index")
            multiplied = emitter._fresh()
            emitter._line(f"{multiplied} = arith.muli {term.name}, {stride_value} : index")
            term = Value(multiplied, DslType("index", "index"))
        if physical is None:
            physical = term
        else:
            added = emitter._fresh()
            emitter._line(f"{added} = arith.addi {physical.name}, {term.name} : index")
            physical = Value(added, DslType("index", "index"))
    assert physical is not None
    return physical


def lower_storage_store(emitter: StorageEmitter, target: ast.Subscript, value: Value) -> None:
    buffer = emitter._expression(target.value)
    if buffer.type.kind not in {"tensor_view", "workgroup"}:
        raise emitter.context.error(target, "indexed assignment requires TensorView or workgroup storage")
    element = buffer.type.arguments[0]
    assert isinstance(element, DslType)
    if buffer.type.kind == "tensor_view" and buffer.access is AccessMode.READ:
        raise emitter.context.error(target, "cannot assign through a read-only TensorView")
    value = emitter._coerce_implicit(target, value, element)
    if buffer.type.kind == "workgroup":
        index = lower_buffer_index(emitter, target.slice)
        emitter._line(
            f'"vernon.workgroup_store"({value.name}, {buffer.name}, {index.name}) '
            f": ({element.mlir}, {buffer.type.mlir}, index) -> ()"
        )
        return
    index = lower_tensor_view_index(emitter, target, buffer)
    emitter._line(
        f'"vernon.intrinsic"({buffer.name}, {index.name}, {value.name}) '
        f'{{name = "tensor_view_store"}} : ({buffer.abi_type.mlir}, index, {element.mlir}) -> ()'
    )
