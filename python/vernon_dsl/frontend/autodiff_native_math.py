from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Protocol

from .autodiff import NodeId, OpCode, ProgramGraphNode
from .autodiff_native_abi import gradient_type, native_type
from .model import ConcreteType
from .tensor_shapes import broadcast_shape, matmul_shape


class NameSource(Protocol):
    def fresh(self) -> str: ...


def _element_type(value_type: ConcreteType) -> ConcreteType:
    element = value_type.arguments[0] if value_type.kind == "tensor" else value_type
    if not isinstance(element, ConcreteType) or element.kind != "scalar":
        raise ValueError(f"{value_type.mlir} has no scalar element type")
    return element


def _shape(value_type: ConcreteType) -> tuple[int, ...]:
    if value_type.kind != "tensor" or not all(isinstance(extent, int) for extent in value_type.arguments[1:]):
        raise ValueError(f"{value_type.mlir} is not a static Tensor")
    return tuple(value_type.arguments[1:])


def _constant(value_type: ConcreteType, value: str, names: NameSource, lines: list[str]) -> str:
    result = names.fresh()
    literal = f"dense<{value}>" if value_type.kind == "tensor" else value
    lines.append(f"    {result} = arith.constant {literal} : {native_type(value_type)}")
    return result


def _binary(
    operation: str,
    left: str,
    right: str,
    value_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    result = names.fresh()
    lines.append(f"    {result} = arith.{operation} {left}, {right} : {native_type(value_type)}")
    return result


def _unary(
    operation: str,
    operand: str,
    value_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    result = names.fresh()
    lines.append(f"    {result} = math.{operation} {operand} : {native_type(value_type)}")
    return result


def _cast(
    value: str,
    source_type: ConcreteType,
    result_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    if source_type == result_type:
        return value
    result = names.fresh()
    source_element = _element_type(source_type)
    result_element = _element_type(result_type)
    conversion = (source_element.mlir, result_element.mlir)
    if conversion == ("f16", "f32"):
        operation = "extf"
    elif conversion == ("f32", "f16"):
        operation = "truncf"
    else:
        raise ValueError(f"unsupported native autodiff cast from {source_type.mlir} to {result_type.mlir}")
    lines.append(f"    {result} = arith.{operation} {value} : {native_type(source_type)} to {native_type(result_type)}")
    return result


def _intrinsic(
    operation: str,
    operands: tuple[str, ...],
    operand_types: tuple[ConcreteType, ...],
    result_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    result = names.fresh()
    lines.append(
        f'    {result} = "vernon.intrinsic"({", ".join(operands)}) {{name = "{operation}"}} : '
        f"({', '.join(native_type(value_type) for value_type in operand_types)}) -> {native_type(result_type)}"
    )
    return result


def _splat(
    value: str,
    tensor_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    result = names.fresh()
    lines.append(f"    {result} = tensor.splat {value} : {native_type(tensor_type)}")
    return result


def _scale(
    tensor: str,
    scalar: str,
    tensor_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    factor = _splat(scalar, tensor_type, names, lines)
    return _binary("mulf", tensor, factor, tensor_type, names, lines)


def _extract(
    value: str,
    value_type: ConcreteType,
    indices: tuple[int, ...],
    names: NameSource,
    lines: list[str],
) -> str:
    if value_type.kind == "scalar":
        if indices:
            raise ValueError("scalar extraction cannot have indices")
        return value
    index_values = tuple(names.fresh() for _ in indices)
    for index_value, index in zip(index_values, indices, strict=True):
        lines.append(f"    {index_value} = arith.constant {index} : index")
    result = names.fresh()
    lines.append(f"    {result} = tensor.extract {value}[{', '.join(index_values)}] : {native_type(value_type)}")
    return result


def _construct(
    elements: list[str],
    result_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> str:
    result = names.fresh()
    element_type = _element_type(result_type)
    lines.append(
        f'    {result} = "vernon.intrinsic"({", ".join(elements)}) {{name = "construct"}} : '
        f"({', '.join(element_type.mlir for _ in elements)}) -> {native_type(result_type)}"
    )
    return result


def emit_forward_math(
    node: ProgramGraphNode,
    operands: tuple[str, ...],
    operand_types: tuple[ConcreteType, ...],
    names: NameSource,
    lines: list[str],
) -> str | None:
    operation = node.operation
    compute_operand_types = tuple(gradient_type(value_type) for value_type in operand_types)
    compute_operands = tuple(
        _cast(operand, source_type, compute_type, names, lines)
        for operand, source_type, compute_type in zip(
            operands,
            operand_types,
            compute_operand_types,
            strict=True,
        )
    )
    compute_result_type = gradient_type(node.type)
    if operation in {OpCode.ACOS, OpCode.ABS}:
        intrinsic = "acos" if operation is OpCode.ACOS else "absf"
        result = _unary(intrinsic, compute_operands[0], compute_result_type, names, lines)
        return _cast(result, compute_result_type, node.type, names, lines)
    if operation is OpCode.ATAN2:
        result = names.fresh()
        lines.append(
            f"    {result} = math.atan2 {compute_operands[0]}, {compute_operands[1]} : "
            f"{native_type(compute_result_type)}"
        )
        return _cast(result, compute_result_type, node.type, names, lines)
    if operation is OpCode.POW:
        result = _intrinsic("pow", compute_operands, compute_operand_types, compute_result_type, names, lines)
        return _cast(result, compute_result_type, node.type, names, lines)
    if operation in {OpCode.DOT, OpCode.CROSS, OpCode.MATMUL, OpCode.NORMALIZE, OpCode.REFLECT}:
        result = _intrinsic(
            operation.value,
            compute_operands,
            compute_operand_types,
            compute_result_type,
            names,
            lines,
        )
        return _cast(result, compute_result_type, node.type, names, lines)
    if operation is OpCode.NORM:
        element = _element_type(compute_operand_types[0])
        squared = _intrinsic(
            "dot",
            (compute_operands[0], compute_operands[0]),
            (compute_operand_types[0], compute_operand_types[0]),
            element,
            names,
            lines,
        )
        result = _unary("sqrt", squared, element, names, lines)
        return _cast(result, compute_result_type, node.type, names, lines)
    return None


def _broadcast_batch_coordinate(
    coordinate: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> tuple[int, ...]:
    offset = len(coordinate) - len(source_shape)
    return tuple(0 if extent == 1 else coordinate[offset + index] for index, extent in enumerate(source_shape))


def _matmul_vjp(
    seed: str,
    left: str,
    right: str,
    left_type: ConcreteType,
    right_type: ConcreteType,
    result_type: ConcreteType,
    names: NameSource,
    lines: list[str],
) -> tuple[str, str]:
    left_shape = _shape(left_type)
    right_shape = _shape(right_type)
    if matmul_shape(left_shape, right_shape) is None:
        raise ValueError("invalid static matmul shape")
    left_vector = len(left_shape) == 1
    right_vector = len(right_shape) == 1
    left_batch = () if left_vector else left_shape[:-2]
    right_batch = () if right_vector else right_shape[:-2]
    batch_shape = broadcast_shape(left_batch, right_batch)
    if batch_shape is None:
        raise ValueError("invalid static matmul batch shape")
    rows = 1 if left_vector else left_shape[-2]
    reduction = left_shape[-1]
    columns = 1 if right_vector else right_shape[-1]
    left_terms: dict[tuple[int, ...], list[str]] = {
        indices: [] for indices in product(*(range(extent) for extent in left_shape))
    }
    right_terms: dict[tuple[int, ...], list[str]] = {
        indices: [] for indices in product(*(range(extent) for extent in right_shape))
    }
    for batch in product(*(range(extent) for extent in batch_shape)):
        left_prefix = _broadcast_batch_coordinate(batch, left_batch)
        right_prefix = _broadcast_batch_coordinate(batch, right_batch)
        for row in range(rows):
            for column in range(columns):
                output_indices = (
                    *batch,
                    *((row,) if not left_vector else ()),
                    *((column,) if not right_vector else ()),
                )
                output_seed = _extract(seed, result_type, output_indices, names, lines)
                for inner in range(reduction):
                    left_indices = (inner,) if left_vector else (*left_prefix, row, inner)
                    right_indices = (inner,) if right_vector else (*right_prefix, inner, column)
                    left_value = _extract(left, left_type, left_indices, names, lines)
                    right_value = _extract(right, right_type, right_indices, names, lines)
                    left_terms[left_indices].append(
                        _binary("mulf", output_seed, right_value, _element_type(left_type), names, lines)
                    )
                    right_terms[right_indices].append(
                        _binary("mulf", output_seed, left_value, _element_type(right_type), names, lines)
                    )

    def sum_terms(terms: dict[tuple[int, ...], list[str]], value_type: ConcreteType) -> str:
        elements: list[str] = []
        element_type = _element_type(value_type)
        for indices in product(*(range(extent) for extent in _shape(value_type))):
            contributions = terms[indices]
            value = contributions[0]
            for contribution in contributions[1:]:
                value = _binary("addf", value, contribution, element_type, names, lines)
            elements.append(value)
        return _construct(elements, value_type, names, lines)

    return sum_terms(left_terms, left_type), sum_terms(right_terms, right_type)


def emit_math_vjp(
    node: ProgramGraphNode,
    seed: str,
    primal: Callable[[NodeId], str],
    nodes: dict[NodeId, ProgramGraphNode],
    names: NameSource,
    lines: list[str],
    derivative_type: Callable[[ConcreteType], ConcreteType] = lambda value: value,
) -> tuple[str | None, ...] | None:
    operation = node.operation
    operands = tuple(primal(value) for value in node.inputs)
    operand_types = tuple(derivative_type(nodes[value].type) for value in node.inputs)
    value_type = derivative_type(node.type)
    if operation is OpCode.ACOS:
        one = _constant(value_type, "1.0", names, lines)
        square = _binary("mulf", operands[0], operands[0], value_type, names, lines)
        radicand = _binary("subf", one, square, value_type, names, lines)
        root = _unary("sqrt", radicand, value_type, names, lines)
        quotient = _binary("divf", seed, root, value_type, names, lines)
        zero = _constant(value_type, "0.0", names, lines)
        return (_binary("subf", zero, quotient, value_type, names, lines),)
    if operation is OpCode.ATAN2:
        y_square = _binary("mulf", operands[0], operands[0], value_type, names, lines)
        x_square = _binary("mulf", operands[1], operands[1], value_type, names, lines)
        denominator = _binary("addf", y_square, x_square, value_type, names, lines)
        y_gradient = _binary("mulf", seed, operands[1], value_type, names, lines)
        y_gradient = _binary("divf", y_gradient, denominator, value_type, names, lines)
        x_gradient = _binary("mulf", seed, operands[0], value_type, names, lines)
        x_gradient = _binary("divf", x_gradient, denominator, value_type, names, lines)
        zero = _constant(value_type, "0.0", names, lines)
        x_gradient = _binary("subf", zero, x_gradient, value_type, names, lines)
        return y_gradient, x_gradient
    if operation is OpCode.ABS:
        zero = _constant(value_type, "0.0", names, lines)
        one = _constant(value_type, "1.0", names, lines)
        minus_one = _constant(value_type, "-1.0", names, lines)
        positive = names.fresh()
        negative = names.fresh()
        sign = names.fresh()
        lines.append(f"    {positive} = arith.cmpf ogt, {operands[0]}, {zero} : {native_type(value_type)}")
        lines.append(f"    {negative} = arith.cmpf olt, {operands[0]}, {zero} : {native_type(value_type)}")
        lines.append(f"    {sign} = arith.select {negative}, {minus_one}, {zero} : {native_type(value_type)}")
        selected = names.fresh()
        lines.append(f"    {selected} = arith.select {positive}, {one}, {sign} : {native_type(value_type)}")
        return (_binary("mulf", seed, selected, value_type, names, lines),)
    if operation is OpCode.POW:
        one = _constant(value_type, "1.0", names, lines)
        exponent_minus_one = _binary("subf", operands[1], one, value_type, names, lines)
        power = _intrinsic("pow", (operands[0], exponent_minus_one), (value_type, value_type), value_type, names, lines)
        left = _binary("mulf", seed, operands[1], value_type, names, lines)
        left = _binary("mulf", left, power, value_type, names, lines)
        logarithm = _unary("log", operands[0], value_type, names, lines)
        right = _binary("mulf", seed, primal(node.id), value_type, names, lines)
        right = _binary("mulf", right, logarithm, value_type, names, lines)
        return left, right
    if operation is OpCode.DOT:
        return (
            _scale(operands[1], seed, operand_types[0], names, lines),
            _scale(operands[0], seed, operand_types[1], names, lines),
        )
    if operation is OpCode.CROSS:
        left = _intrinsic("cross", (operands[1], seed), (operand_types[1], value_type), operand_types[0], names, lines)
        right = _intrinsic("cross", (seed, operands[0]), (value_type, operand_types[0]), operand_types[1], names, lines)
        return left, right
    if operation is OpCode.MATMUL:
        return _matmul_vjp(
            seed,
            operands[0],
            operands[1],
            operand_types[0],
            operand_types[1],
            value_type,
            names,
            lines,
        )
    if operation is OpCode.NORM:
        scaled = _scale(operands[0], seed, operand_types[0], names, lines)
        norm = _splat(primal(node.id), operand_types[0], names, lines)
        return (_binary("divf", scaled, norm, operand_types[0], names, lines),)
    if operation is OpCode.NORMALIZE:
        element = _element_type(value_type)
        squared = _intrinsic("dot", (operands[0], operands[0]), (value_type, value_type), element, names, lines)
        norm = _unary("sqrt", squared, element, names, lines)
        norm_tensor = _splat(norm, value_type, names, lines)
        first = _binary("divf", seed, norm_tensor, value_type, names, lines)
        projection = _intrinsic("dot", (seed, operands[0]), (value_type, value_type), element, names, lines)
        norm_squared = _binary("mulf", norm, norm, element, names, lines)
        norm_cubed = _binary("mulf", norm_squared, norm, element, names, lines)
        factor = _binary("divf", projection, norm_cubed, element, names, lines)
        second = _scale(operands[0], factor, value_type, names, lines)
        return (_binary("subf", first, second, value_type, names, lines),)
    if operation is OpCode.REFLECT:
        direction, normal = operands
        direction_type, normal_type = operand_types
        element = _element_type(direction_type)
        two = _constant(element, "2.0", names, lines)
        seed_dot_normal = _intrinsic("dot", (seed, normal), (value_type, normal_type), element, names, lines)
        direction_gradient = _scale(
            normal, _binary("mulf", two, seed_dot_normal, element, names, lines), normal_type, names, lines
        )
        direction_gradient = _binary("subf", seed, direction_gradient, direction_type, names, lines)
        direction_dot_normal = _intrinsic(
            "dot", (direction, normal), (direction_type, normal_type), element, names, lines
        )
        first = _scale(direction, seed_dot_normal, direction_type, names, lines)
        second = _scale(seed, direction_dot_normal, value_type, names, lines)
        normal_gradient = _binary("addf", first, second, normal_type, names, lines)
        minus_two = _constant(element, "-2.0", names, lines)
        normal_gradient = _scale(normal_gradient, minus_two, normal_type, names, lines)
        return direction_gradient, normal_gradient
    return None


__all__ = ["emit_forward_math", "emit_math_vjp"]
