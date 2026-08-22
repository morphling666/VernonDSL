"""Canonical MLIR emission for program-level graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ..frontend.abi import value_leaves
from ..frontend.model import ConcreteType
from .model import GraphOperation, GraphValue, ProgramGraph


def _string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _type(value: GraphValue) -> str:
    return value.type.logical.mlir


def _interface_attributes(
    value: GraphValue,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> str:
    source_name = value.role.split(".", 1)[-1]
    logical = value.type.logical
    element = logical.arguments[0] if logical.kind in {"tensor", "tensor_view", "tensor_view_abi"} else logical
    assert isinstance(element, ConcreteType)
    leaves = value_leaves(element, structs.__getitem__)
    dtypes = ", ".join(_string(leaf.dtype) for leaf in leaves)
    attributes = [f"vernon.source_name = {_string(source_name)}", f"vernon.abi_leaf_dtypes = [{dtypes}]"]
    if len(leaves) == 1:
        attributes.insert(1, f"vernon.dtype = {_string(leaves[0].dtype)}")
    return f" {{{', '.join(attributes)}}}"


def _leaf_dtypes(
    value: GraphValue,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> str:
    logical = value.type.logical
    element = logical.arguments[0] if logical.kind in {"tensor", "tensor_view", "tensor_view_abi"} else logical
    assert isinstance(element, ConcreteType)
    return "[" + ", ".join(_string(leaf.dtype) for leaf in value_leaves(element, structs.__getitem__)) + "]"


def _operation(
    operation: GraphOperation,
    types: dict[str, GraphValue],
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> str:
    operands = ", ".join(name for _, name in operation.operands)
    operand_types = ", ".join(_type(types[name]) for _, name in operation.operands)
    result_types = ", ".join(_type(types[name]) for _, name in operation.results)
    result_prefix = ", ".join(name for _, name in operation.results)
    if result_prefix:
        result_prefix += " = "
    attributes = [
        f"source_id = {operation.id} : i64",
        f"debug_name = {_string(operation.name)}",
        "vernon_program.result_abi_leaf_dtypes = ["
        + ", ".join(_leaf_dtypes(types[name], structs) for _, name in operation.results)
        + "]",
        *(f"{name} = {value}" for name, value in operation.attributes),
    ]
    return (
        f'    {result_prefix}"{operation.kind}"({operands}) '
        f"{{{', '.join(attributes)}}} : ({operand_types}) -> ({result_types})"
    )


def emit_graph(
    graph: ProgramGraph,
    structs: Mapping[str, tuple[tuple[str, ConcreteType], ...]],
) -> str:
    all_values = {value.name: value for value in graph.values}
    if len(all_values) != len(graph.values):
        raise ValueError("Program graph contains duplicate SSA value names")
    available = {value.name for value in graph.arguments}
    for operation in graph.operations:
        for _, operand in operation.operands:
            if operand not in available:
                raise ValueError(f"Program operation {operation.name!r} consumes unavailable value {operand!r}")
        for _, result in operation.results:
            if result not in all_values:
                raise ValueError(f"missing type for Program result {result!r}")
            if result in available:
                raise ValueError(f"Program SSA value {result!r} has multiple definitions")
            available.add(result)
    if any(value.name not in available for value in graph.results):
        raise ValueError("Program graph returns an unavailable SSA value")

    arguments = ", ".join(
        f"{value.name}: {_type(value)}{_interface_attributes(value, structs)}" for value in graph.arguments
    )
    argument_names = ", ".join(_string(value.role) for value in graph.arguments)
    result_names = ", ".join(_string(value.role) for value in graph.results)
    result_values = ", ".join(value.name for value in graph.results)
    result_types = ", ".join(_type(value) for value in graph.results)
    attributed_result_types = ", ".join(
        f"{_type(value)}{_interface_attributes(value, structs)}" for value in graph.results
    )
    function_results = f" -> ({attributed_result_types})" if graph.results else ""
    lines = [
        f"  func.func @{graph.name}({arguments}){function_results} attributes "
        + "{"
        + f"vernon_program.graph = {_string(graph.direction)}, "
        + f"vernon_program.argument_names = [{argument_names}], "
        + f"vernon_program.result_names = [{result_names}]"
        + "} {",
        *(_operation(operation, all_values, structs) for operation in graph.operations),
        f"    func.return {result_values} : {result_types}" if graph.results else "    func.return",
        "  }",
    ]
    return "\n".join(lines)


def emit_program(
    forward: ProgramGraph,
    backward: ProgramGraph | None,
    *,
    vjp_wrt: tuple[str, ...] = (),
    structs: tuple[tuple[str, tuple[tuple[str, ConcreteType], ...]], ...] = (),
) -> str:
    struct_map = dict(structs)
    graphs = [emit_graph(forward, struct_map)]
    if backward is not None:
        graphs.append(emit_graph(backward, struct_map))
    attributes = (
        " attributes {vernon_program.vjp_wrt = [" + ", ".join(_string(name) for name in vjp_wrt) + "]}"
        if vjp_wrt
        else ""
    )
    declarations = [
        '  "vernon.struct"() {'
        + f"sym_name = {_string(name)}, "
        + "fields = ["
        + ", ".join(_string(f"{field_name}:{field_type.mlir}") for field_name, field_type in fields)
        + "], abi_leaf_dtypes = ["
        + ", ".join(_string(leaf.dtype) for leaf in value_leaves(ConcreteType("struct", name), struct_map.__getitem__))
        + "]} : () -> ()"
        for name, fields in structs
    ]
    body = "\n".join([*declarations, *graphs])
    return f"module{attributes} {{\n{body}\n}}\n"


__all__ = ["emit_graph", "emit_program"]
