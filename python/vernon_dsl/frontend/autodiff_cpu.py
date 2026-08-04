from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .autodiff import (
    AutodiffProgram,
    BuiltinOp,
    LiteralOp,
    NamedOp,
    NodeId,
    OpCode,
    ProgramGraphNode,
    StaticIndicesOp,
)
from .model import ConcreteType

_DTYPES = {
    "bool": np.dtype(np.bool_),
    "i32": np.dtype(np.int32),
    "u32": np.dtype(np.uint32),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


def _gradient_dtype(value_type: ConcreteType) -> np.dtype[Any]:
    element = value_type.arguments[0] if value_type.kind in {"tensor", "tensor_view"} else value_type
    if not isinstance(element, ConcreteType) or not element.is_float:
        raise TypeError(f"{value_type.mlir} has no floating cotangent")
    return np.dtype(np.float64 if element.name == "f64" else np.float32)


def _primal_dtype(value_type: ConcreteType) -> np.dtype[Any]:
    element = value_type.arguments[0] if value_type.kind in {"tensor", "tensor_view"} else value_type
    if not isinstance(element, ConcreteType) or element.kind != "scalar":
        raise TypeError(f"{value_type.mlir} is not a numeric CPU reference Value")
    return _DTYPES[element.name]


def _copy_primal(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_copy_primal(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _copy_primal(item) for key, item in value.items()}
    return value.item() if isinstance(value, np.generic) else value


def _coerce(
    value: Any,
    value_type: ConcreteType,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> Any:
    if value_type.kind in {"scalar", "tensor", "tensor_view"}:
        array = np.array(value, dtype=_primal_dtype(value_type), copy=True)
        expected_shape = (
            tuple(value_type.arguments[1:])
            if value_type.kind == "tensor"
            else value_type.arguments[1]
            if value_type.kind == "tensor_view"
            else ()
        )
        if value_type.kind != "scalar" and array.shape != expected_shape:
            raise TypeError(f"Value shape {array.shape} does not match {value_type.mlir}")
        return array
    if value_type.kind == "tuple":
        if not isinstance(value, tuple) or len(value) != len(value_type.arguments):
            raise TypeError(f"Value does not match {value_type.mlir}")
        return tuple(
            _coerce(item, item_type, structs)
            for item, item_type in zip(value, value_type.arguments, strict=True)
            if isinstance(item_type, ConcreteType)
        )
    if value_type.kind == "struct":
        fields = structs[value_type.name]
        if isinstance(value, Mapping):
            if set(value) != {name for name, _ in fields}:
                raise TypeError(f"Value does not match {value_type.mlir}")
            return {name: _coerce(value[name], field_type, structs) for name, field_type in fields}
        return {name: _coerce(getattr(value, name), field_type, structs) for name, field_type in fields}
    return _copy_primal(value)


def _zeros_like(
    value: Any,
    value_type: ConcreteType,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> Any:
    if value_type.kind in {"scalar", "tensor", "tensor_view"}:
        return np.zeros_like(value, dtype=_gradient_dtype(value_type))
    if value_type.kind == "tuple":
        return tuple(
            _zeros_like(item, item_type, structs)
            for item, item_type in zip(value, value_type.arguments, strict=True)
            if isinstance(item_type, ConcreteType)
        )
    if value_type.kind == "struct":
        return {name: _zeros_like(value[name], field_type, structs) for name, field_type in structs[value_type.name]}
    raise TypeError(f"{value_type.mlir} cotangent is not supported by the CPU reference")


def _add(left: Any, right: Any) -> Any:
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise TypeError("cotangent structures do not match")
        return tuple(_add(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            raise TypeError("cotangent structures do not match")
        return {key: _add(left[key], right[key]) for key in left}
    return left + right


def _unbroadcast(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, extent in enumerate(shape):
        if extent == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value


def _seed(
    value: Any,
    value_type: ConcreteType,
    cotangent: Any | None,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
    path: tuple[str, ...] = ("output",),
) -> Any:
    if value_type.kind == "scalar" and not value_type.is_float:
        return np.zeros_like(value)
    if cotangent is None:
        if value_type.kind != "scalar" or not value_type.is_float:
            raise TypeError("an explicit cotangent is required for non-scalar or structured output")
        return np.ones((), dtype=_gradient_dtype(value_type))
    if value_type.kind in {"scalar", "tensor", "tensor_view"}:
        if isinstance(cotangent, Mapping):
            cotangent = cotangent[".".join(path)]
        seed = np.asarray(cotangent, dtype=_gradient_dtype(value_type))
        if seed.shape != np.asarray(value).shape:
            raise TypeError("cotangent shape does not match program output")
        return seed
    if value_type.kind == "tuple":
        if not isinstance(cotangent, Mapping):
            raise TypeError("cotangent structure does not match program output")
        return tuple(
            _seed(item, item_type, cotangent, structs, (*path, str(index)))
            for index, (item, item_type) in enumerate(zip(value, value_type.arguments, strict=True))
            if isinstance(item_type, ConcreteType)
        )
    if value_type.kind == "struct":
        fields = structs[value_type.name]
        if not isinstance(cotangent, Mapping):
            raise TypeError("cotangent structure does not match program output")
        return {name: _seed(value[name], field_type, cotangent, structs, (*path, name)) for name, field_type in fields}
    raise TypeError(f"{value_type.mlir} cotangent is unsupported")


def _forward(
    node: ProgramGraphNode,
    inputs: tuple[Any, ...],
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> Any:
    operation = node.operation
    if operation is OpCode.CONSTANT:
        assert isinstance(node.payload, LiteralOp)
        return _coerce(node.payload.value, node.type, structs)
    if operation in {OpCode.TUPLE, OpCode.TYPED_TUPLE}:
        return inputs
    if operation in {OpCode.TENSOR, OpCode.VECTOR, OpCode.MATRIX}:
        return np.asarray(inputs, dtype=_primal_dtype(node.type)).reshape(node.type.arguments[1:])
    if operation is OpCode.STRUCT:
        assert isinstance(node.payload, NamedOp)
        fields = structs[node.payload.name]
        return {field_name: value for (field_name, _), value in zip(fields, inputs, strict=True)}
    if operation is OpCode.FIELD:
        assert isinstance(node.payload, NamedOp)
        return inputs[0][node.payload.name]
    if operation is OpCode.INDEX:
        assert isinstance(node.payload, StaticIndicesOp)
        indices = node.payload.indices
        index = indices[0] if len(indices) == 1 else indices
        return inputs[0][index]
    if operation is OpCode.INDEX_DYNAMIC:
        indices = tuple(int(value) for value in inputs[1:])
        index = indices[0] if len(indices) == 1 else indices
        return inputs[0][index]
    if operation is OpCode.STORE:
        assert isinstance(node.payload, StaticIndicesOp)
        indices = node.payload.indices
        index = indices[0] if len(indices) == 1 else indices
        storage = np.array(inputs[0], copy=True)
        storage[index] = inputs[1]
        return storage
    if operation is OpCode.STORE_DYNAMIC:
        indices = tuple(int(value) for value in inputs[1:-1])
        index = indices[0] if len(indices) == 1 else indices
        storage = np.array(inputs[0], copy=True)
        storage[index] = inputs[-1]
        return storage
    if operation is OpCode.COMPARE:
        assert isinstance(node.payload, NamedOp)
        predicates = {
            "Eq": np.equal,
            "NotEq": np.not_equal,
            "Lt": np.less,
            "LtE": np.less_equal,
            "Gt": np.greater,
            "GtE": np.greater_equal,
        }
        return predicates[node.payload.name](inputs[0], inputs[1])
    if operation is OpCode.CONDITIONAL:
        return inputs[1] if bool(inputs[0]) else inputs[2]
    if operation in {OpCode.SPLAT, OpCode.BROADCAST}:
        return np.broadcast_to(inputs[0], tuple(node.type.arguments[1:]))
    if operation is OpCode.ADD:
        return inputs[0] + inputs[1]
    if operation is OpCode.SUB:
        return inputs[0] - inputs[1]
    if operation is OpCode.MUL:
        return inputs[0] * inputs[1]
    if operation is OpCode.DIV:
        return inputs[0] / inputs[1]
    if operation is OpCode.POW:
        return np.power(inputs[0], inputs[1])
    if operation is OpCode.NEG:
        return -inputs[0]
    if operation is OpCode.IDENTITY:
        return inputs[0]
    unary = {
        OpCode.SIN: np.sin,
        OpCode.COS: np.cos,
        OpCode.ACOS: np.arccos,
        OpCode.EXP: np.exp,
        OpCode.LOG: np.log,
        OpCode.SQRT: np.sqrt,
        OpCode.ABS: np.abs,
        OpCode.NORM: np.linalg.norm,
        OpCode.NORMALIZE: lambda value: value / np.linalg.norm(value),
    }
    if operation in unary:
        return unary[operation](inputs[0])
    if operation is OpCode.ATAN2:
        return np.arctan2(inputs[0], inputs[1])
    if operation is OpCode.DOT:
        return np.dot(inputs[0], inputs[1])
    if operation is OpCode.CROSS:
        return np.cross(inputs[0], inputs[1])
    if operation is OpCode.MATMUL:
        return np.matmul(inputs[0], inputs[1])
    if operation is OpCode.REFLECT:
        return inputs[0] - 2 * np.dot(inputs[0], inputs[1]) * inputs[1]
    raise AssertionError(f"missing CPU forward rule for {operation.value}")


def _matmul_vjp(seed: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.ndim == right.ndim == 1:
        return seed * right, seed * left
    if left.ndim == 1:
        return np.matmul(seed, np.swapaxes(right, -1, -2)), np.expand_dims(left, -1) * np.expand_dims(seed, -2)
    if right.ndim == 1:
        return np.expand_dims(seed, -1) * np.expand_dims(right, -2), np.matmul(np.swapaxes(left, -1, -2), seed)
    return (
        np.matmul(seed, np.swapaxes(right, -1, -2)),
        np.matmul(np.swapaxes(left, -1, -2), seed),
    )


def _backward(
    node: ProgramGraphNode,
    seed: Any,
    inputs: tuple[Any, ...],
    input_types: tuple[ConcreteType, ...],
    output: Any,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> tuple[Any, ...]:
    operation = node.operation
    if operation is OpCode.ADD:
        return seed, seed
    if operation is OpCode.SUB:
        return seed, -seed
    if operation is OpCode.MUL:
        return seed * inputs[1], seed * inputs[0]
    if operation is OpCode.DIV:
        return seed / inputs[1], -seed * inputs[0] / np.square(inputs[1])
    if operation is OpCode.POW:
        return (
            seed * inputs[1] * np.power(inputs[0], inputs[1] - 1),
            seed * output * np.log(inputs[0]),
        )
    if operation is OpCode.NEG:
        return (-seed,)
    if operation is OpCode.IDENTITY:
        return (seed,)
    if operation in {OpCode.SPLAT, OpCode.BROADCAST}:
        input_shape = np.asarray(inputs[0]).shape
        return (_unbroadcast(np.asarray(seed), input_shape),)
    if operation is OpCode.SIN:
        return (seed * np.cos(inputs[0]),)
    if operation is OpCode.COS:
        return (-seed * np.sin(inputs[0]),)
    if operation is OpCode.ACOS:
        return (-seed / np.sqrt(1 - np.square(inputs[0])),)
    if operation is OpCode.ATAN2:
        denominator = np.square(inputs[0]) + np.square(inputs[1])
        return seed * inputs[1] / denominator, -seed * inputs[0] / denominator
    if operation is OpCode.EXP:
        return (seed * output,)
    if operation is OpCode.LOG:
        return (seed / inputs[0],)
    if operation is OpCode.SQRT:
        return (seed * 0.5 / output,)
    if operation is OpCode.ABS:
        return (seed * np.sign(inputs[0]),)
    if operation in {OpCode.TUPLE, OpCode.TYPED_TUPLE}:
        return tuple(seed)
    if operation in {OpCode.TENSOR, OpCode.VECTOR, OpCode.MATRIX}:
        flat = np.asarray(seed).reshape(-1)
        return tuple(flat[index] for index in range(len(inputs)))
    if operation is OpCode.STRUCT:
        assert isinstance(node.payload, NamedOp)
        return tuple(seed[field_name] for field_name, _ in structs[node.payload.name])
    if operation is OpCode.FIELD:
        assert isinstance(node.payload, NamedOp)
        field_name = node.payload.name
        gradient = _zeros_like(inputs[0], input_types[0], structs)
        gradient[field_name] = seed
        return (gradient,)
    if operation is OpCode.INDEX:
        assert isinstance(node.payload, StaticIndicesOp)
        indices = node.payload.indices
        index = indices[0] if len(indices) == 1 else indices
        gradient = _zeros_like(inputs[0], input_types[0], structs)
        if isinstance(gradient, tuple):
            mutable = list(gradient)
            assert isinstance(index, int)
            mutable[index] = _add(mutable[index], seed)
            gradient = tuple(mutable)
        else:
            gradient[index] += seed
        return (gradient,)
    if operation is OpCode.INDEX_DYNAMIC:
        indices = tuple(int(value) for value in inputs[1:])
        index = indices[0] if len(indices) == 1 else indices
        gradient = _zeros_like(inputs[0], input_types[0], structs)
        np.add.at(gradient, index, seed)
        return (gradient, *(None for _ in inputs[1:]))
    if operation is OpCode.STORE:
        assert isinstance(node.payload, StaticIndicesOp)
        indices = node.payload.indices
        index = indices[0] if len(indices) == 1 else indices
        storage_gradient = np.array(seed, copy=True)
        value_gradient = np.array(storage_gradient[index], copy=True)
        storage_gradient[index] = 0
        return storage_gradient, value_gradient
    if operation is OpCode.STORE_DYNAMIC:
        indices = tuple(int(value) for value in inputs[1:-1])
        index = indices[0] if len(indices) == 1 else indices
        storage_gradient = np.array(seed, copy=True)
        value_gradient = np.array(storage_gradient[index], copy=True)
        storage_gradient[index] = 0
        index_gradients = (None,) * len(inputs[1:-1])
        return storage_gradient, *index_gradients, value_gradient
    if operation is OpCode.CONDITIONAL:
        return (None, seed, None) if bool(inputs[0]) else (None, None, seed)
    if operation in {OpCode.DOT, OpCode.MATMUL}:
        return _matmul_vjp(np.asarray(seed), np.asarray(inputs[0]), np.asarray(inputs[1]))
    if operation is OpCode.CROSS:
        return np.cross(inputs[1], seed), np.cross(seed, inputs[0])
    if operation is OpCode.NORM:
        return (seed * inputs[0] / output,)
    if operation is OpCode.NORMALIZE:
        norm = np.linalg.norm(inputs[0])
        return (seed / norm - inputs[0] * np.sum(seed * inputs[0]) / (norm**3),)
    if operation is OpCode.REFLECT:
        direction, normal = inputs
        return (
            seed - 2 * np.dot(seed, normal) * normal,
            -2 * (np.dot(seed, normal) * direction + np.dot(direction, normal) * seed),
        )
    raise AssertionError(f"missing CPU backward rule for {operation}")


def _execute_single_invocation(
    program: AutodiffProgram,
    bindings: Mapping[str, Any],
) -> tuple[Any, Callable[[Any | None], dict[str, Any]]]:
    graph = program.semantic
    reverse = program.reverse

    nodes = {node.id: node for node in graph.nodes}
    structs = dict(graph.structs)
    values: dict[NodeId, Any] = {}
    parameter_nodes: dict[str, NodeId] = {}
    storage_bindings = [
        (node.source_name, np.asarray(bindings[node.source_name]))
        for node in graph.nodes
        if node.operation is OpCode.PARAMETER
        and node.type.kind == "tensor_view"
        and node.source_name is not None
        and node.source_name in bindings
    ]
    written_storage = {
        node.source_name
        for node in graph.nodes
        if node.source_name is not None and node.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC}
    }
    for index, (left_name, left) in enumerate(storage_bindings):
        for right_name, right in storage_bindings[index + 1 :]:
            if np.shares_memory(left, right) and (left_name in written_storage or right_name in written_storage):
                raise TypeError(
                    f"writable aliased TensorView bindings {left_name!r} and {right_name!r} "
                    "cannot prove a legal reverse scatter"
                )
    for node in graph.nodes:
        if node.operation in {OpCode.PARAMETER, OpCode.BUILTIN}:
            assert node.source_name is not None
            if node.source_name not in bindings:
                raise TypeError(f"missing program argument {node.source_name!r}")
            values[node.id] = _coerce(bindings[node.source_name], node.type, structs)
            parameter_nodes[node.source_name] = node.id

    def evaluate(node_id: NodeId) -> Any:
        if node_id in values:
            return values[node_id]
        node = nodes[node_id]
        if node.operation is OpCode.CONDITIONAL:
            condition = evaluate(node.inputs[0])
            selected = node.inputs[1] if bool(condition) else node.inputs[2]
            inputs = (condition, evaluate(selected), None)
            if not bool(condition):
                inputs = (condition, None, inputs[1])
            values[node_id] = inputs[1] if bool(condition) else inputs[2]
        else:
            values[node_id] = _forward(
                node,
                tuple(evaluate(value) for value in node.inputs),
                structs,
            )
        return values[node_id]

    unexpected = set(bindings) - set(parameter_nodes)
    if unexpected:
        raise TypeError("unexpected program argument(s): " + ", ".join(sorted(unexpected)))
    if len(graph.outputs) != 1:
        raise RuntimeError("CPU reference currently requires one structured output")
    output_id = graph.outputs[0]
    evaluate(output_id)
    for parameter, value_id in graph.storage_outputs:
        evaluate(value_id)
        binding = bindings[parameter]
        if not isinstance(binding, np.ndarray):
            raise TypeError(f"writable TensorView binding {parameter!r} must be a NumPy array")
        destination = binding
        if not destination.flags.writeable:
            raise TypeError(f"TensorView binding {parameter!r} is not writable")
        np.copyto(destination, values[value_id], casting="no")
    output_node = nodes[output_id]

    def pullback(cotangent: Any | None = None) -> dict[str, Any]:
        if (
            cotangent is not None
            and output_node.type.kind in {"tuple", "struct"}
            and (not isinstance(cotangent, Mapping) or set(cotangent) != set(reverse.cotangent_paths))
        ):
            raise TypeError("cotangent structure does not match program output")
        gradients: dict[NodeId, Any] = {output_id: _seed(values[output_id], output_node.type, cotangent, structs)}
        for node_id in reverse.reverse_order:
            seed = gradients.get(node_id)
            if seed is None:
                continue
            node = nodes[node_id]
            input_gradients = _backward(
                node,
                seed,
                tuple(values.get(value) for value in node.inputs),
                tuple(nodes[value].type for value in node.inputs),
                values[node_id],
                structs,
            )
            for input_id, gradient in zip(node.inputs, input_gradients, strict=True):
                if gradient is None:
                    continue
                input_value = np.asarray(values[input_id])
                if isinstance(gradient, np.ndarray):
                    gradient = _unbroadcast(gradient, input_value.shape)
                gradients[input_id] = _add(gradients[input_id], gradient) if input_id in gradients else gradient
        result: dict[str, Any] = {}
        for path in graph.wrt:
            components = path.split(".")
            parameter_id = parameter_nodes[components[0]]
            gradient = gradients.get(
                parameter_id,
                _zeros_like(values[parameter_id], nodes[parameter_id].type, structs),
            )
            for component in components[1:]:
                gradient = (
                    gradient[int(component)]
                    if isinstance(gradient, tuple)
                    else gradient[component]
                    if isinstance(gradient, Mapping)
                    else getattr(gradient, component)
                )
            result[path] = _copy_primal(gradient)
        return result

    return _copy_primal(values[output_id]), pullback


def execute_program_graph(
    program: AutodiffProgram,
    bindings: Mapping[str, Any],
    grid: tuple[int, int, int],
) -> tuple[Any, Callable[[Any | None], dict[str, Any]]]:
    """Execute a typed autodiff program over an explicit x-fastest 3D grid."""

    if len(grid) != 3 or any(not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0 for extent in grid):
        raise ValueError("CPU autodiff grid requires three positive integer extents")
    invocation_count = grid[0] * grid[1] * grid[2]
    if invocation_count == 1:
        return _execute_single_invocation(program, bindings)

    builtin_names = {
        node.source_name: node.payload.name
        for node in program.semantic.nodes
        if node.operation is OpCode.BUILTIN and node.source_name is not None and isinstance(node.payload, BuiltinOp)
    }
    outputs: list[Any] = []
    pullbacks: list[Callable[[Any | None], dict[str, Any]]] = []
    grid_x, grid_y, grid_z = grid
    for z in range(grid_z):
        for y in range(grid_y):
            for x in range(grid_x):
                invocation_bindings = dict(bindings)
                for parameter, builtin in builtin_names.items():
                    if builtin == "global_invocation_id":
                        invocation_bindings[parameter] = np.array([x, y, z], dtype=np.uint32)
                    else:
                        raise TypeError(f"CPU parallel autodiff does not support builtin {builtin!r}")
                output, invocation_pullback = _execute_single_invocation(program, invocation_bindings)
                outputs.append(output)
                pullbacks.append(invocation_pullback)

    def parallel_pullback(cotangent: Any | None = None) -> dict[str, Any]:
        seeds = [None] * invocation_count if cotangent is None else list(np.asarray(cotangent))
        if len(seeds) != invocation_count:
            raise TypeError("parallel cotangent leading dimension does not match the runtime grid")
        result: dict[str, Any] = {}
        for invocation_pullback, seed in zip(pullbacks, seeds, strict=True):
            for path, gradient in invocation_pullback(seed).items():
                if path not in result:
                    result[path] = gradient
                else:
                    result[path] = _add(result[path], gradient)
        return result

    return np.stack(outputs), parallel_pullback


__all__ = ["execute_program_graph"]
