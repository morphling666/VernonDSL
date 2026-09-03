"""Program-level parser from captured Module calls to schedulable graphs."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .._dtypes import scalar_name
from ..frontend.model import ConcreteType
from .mlir import emit_program
from .model import MlirOperation, MlirValue, ParsedProgram, ProgramImplementation, ProgramType


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _string_array(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_quoted(value) for value in values) + "]"


def _i64_array(values: tuple[int, ...] | list[str]) -> str:
    return "array<i64: " + ", ".join(str(value) for value in values) + ">" if values else "array<i64>"


def _program_resource_type(value_type: ConcreteType) -> ProgramType:
    if value_type.kind in {"tensor_view", "tensor_view_abi"}:
        element, shape, *_ = value_type.arguments
        if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
            raise TypeError("Program resource parameter has an invalid logical TensorView type")
        # Access lives on the resource edge, not the Value type, so SSA versions
        # of one owner stay the same TensorView cell + view shape.
        return ProgramType(ConcreteType("tensor_view", "TensorView", (element, shape, "read_write", "device")))
    if value_type.kind == "tensor":
        return ProgramType(value_type)
    if value_type.kind in {"texture", "sampler"}:
        return ProgramType(value_type)
    raise TypeError(f"Program resource parameter has unsupported logical type {value_type.kind!r}")


def _scalar_type(value: Any) -> ProgramType:
    if isinstance(value, np.generic):
        name = scalar_name(value.dtype)
        if name is None:
            raise TypeError(f"Program scalar value has unsupported dtype {value.dtype}")
        return ProgramType(ConcreteType("scalar", name))
    if isinstance(value, bool):
        return ProgramType(ConcreteType("scalar", "bool"))
    if isinstance(value, int):
        return ProgramType(ConcreteType("scalar", "i32"))
    if isinstance(value, float):
        return ProgramType(ConcreteType("scalar", "f32"))
    raise TypeError(f"Program scalar value has unsupported type {type(value).__name__}")


def _host_constant_value(value: Any) -> int | float | bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    raise TypeError(f"Program constant has unsupported type {type(value).__name__}")


def _constant_attribute(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)} : i64"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g} : f64"
    raise TypeError(f"Program constant has unsupported type {type(value).__name__}")


def _value_input_types(invocation: Any) -> dict[int, ProgramType]:
    from ..operation_graph import GraphValueInput

    return {
        id(value): ProgramType(value.logical)
        for value in invocation.inputs.values()
        if isinstance(value, GraphValueInput)
    }


def _operation_host_constants(
    node: Any,
    invocation: Any,
    value_input_ids: set[int],
) -> tuple[tuple[str, int | float | bool], ...]:
    constants: list[tuple[str, int | float | bool]] = []
    for parameter in node.parameters:
        if parameter.name in node.inputs:
            continue
        supplied = invocation.slots[node.binding_slots[parameter.name]]
        if id(supplied) in value_input_ids:
            continue
        constants.append((parameter.name, _host_constant_value(supplied)))
    return tuple(constants)


def _lower_forward_function(
    invocation: Any,
    resource_types: Mapping[int, ProgramType],
) -> tuple[
    tuple[MlirValue, ...],
    tuple[MlirValue, ...],
    tuple[MlirValue, ...],
    tuple[MlirOperation, ...],
]:
    from ..operation_graph import AllocOp, KernelCallOp

    graph = invocation.graph
    values: dict[str, MlirValue] = {
        f"%v{value.id}": MlirValue(f"%v{value.id}", resource_types[value.id], "resource", value.id)
        for value in graph.values
    }
    arguments: list[MlirValue] = []
    for name, value_id in graph.inputs.items():
        ssa = f"%v{value_id}"
        if all(argument.name != ssa for argument in arguments):
            arguments.append(MlirValue(ssa, values[ssa].type, f"input.{name}", value_id))

    value_input_types = _value_input_types(invocation)
    value_inputs: dict[int, str] = {}
    for index, (name, value) in enumerate(invocation.inputs.items()):
        if name in graph.inputs:
            continue
        value_type = value_input_types.get(id(value))
        if value_type is None:
            continue
        ssa = f"%arg{index}"
        graph_value = MlirValue(ssa, value_type, f"input.{name}", -1)
        values[ssa] = graph_value
        arguments.append(graph_value)
        value_inputs[id(value)] = ssa

    operations: list[MlirOperation] = []
    for node in graph.nodes:
        if isinstance(node, AllocOp):
            operands = (("source", f"%v{node.like}"),) if node.like is not None else ()
            attributes: list[tuple[str, str]] = [("name", _quoted(node.name))]
            if node.values is not None:
                attributes.append(("payload", _quoted(json.dumps(node.values))))
            operations.append(
                MlirOperation(
                    node.id,
                    "vernon.intrinsic",
                    node.name,
                    operands,
                    (("result", f"%v{node.result}"),),
                    tuple(attributes),
                )
            )
            continue
        if not isinstance(node, KernelCallOp):
            continue
        operands: list[tuple[str, str]] = []
        constant_names: list[str] = []
        constant_values: list[str] = []
        access_by_name = {parameter.name: parameter.access for parameter in node.parameters}
        for parameter in node.parameters:
            if parameter.name in node.inputs:
                operands.append((parameter.name, f"%v{node.inputs[parameter.name]}"))
                continue
            slot = node.binding_slots[parameter.name]
            supplied = invocation.slots[slot]
            ssa = value_inputs.get(id(supplied))
            if ssa is None:
                constant_names.append(parameter.name)
                constant_values.append(_constant_attribute(supplied))
            else:
                operands.append((parameter.name, ssa))
        results = tuple((name, f"%v{value}") for name, value in node.outputs.items())
        operand_names = tuple(name for name, _ in operands)
        result_names = tuple(name for name, _ in results)
        resource_sources = []
        for result_name in result_names:
            resource_sources.append(str(operand_names.index(result_name)) if result_name in operand_names else "-1")
        operations.append(
            MlirOperation(
                node.id,
                "vernon_program.compute",
                node.name,
                tuple(operands),
                results,
                (
                    ("callee", _quoted(node.name)),
                    ("grid", _i64_array(node.grid)),
                    ("features", _string_array(node.features)),
                    ("operand_names", _string_array(operand_names)),
                    ("result_names", _string_array(result_names)),
                    (
                        "vernon_program.operand_accesses",
                        _string_array(tuple(access_by_name.get(name, "read") for name in operand_names)),
                    ),
                    (
                        "vernon_program.result_resource_sources",
                        _i64_array(resource_sources),
                    ),
                    ("constant_names", _string_array(tuple(constant_names))),
                    ("constant_values", "[" + ", ".join(constant_values) + "]"),
                ),
            )
        )

    results = [
        MlirValue(f"%v{value_id}", values[f"%v{value_id}"].type, f"output.{path}", value_id)
        for path, value_id in graph.outputs.items()
    ]
    return (
        tuple(values.values()),
        tuple(arguments),
        tuple(results),
        tuple(operations),
    )


def parse_program(
    invocation: Any,
    *,
    vjp_wrt: tuple[str, ...] = (),
    vjp_outputs: tuple[str, ...] | None = None,
    autodiff_planning_policy: str | None = None,
) -> ParsedProgram:
    """Parse one captured Module specialization into a primal Program."""

    implementations: dict[str, ProgramImplementation] = {}
    missing_types = tuple(value.id for value in invocation.graph.values if value.type.logical is None)
    if missing_types:
        raise ValueError(f"Program values have no capture-authoritative logical types: {missing_types}")
    resource_types: dict[int, ProgramType] = {
        value.id: _program_resource_type(value.type.logical)
        for value in invocation.graph.values
        if value.type.logical is not None
    }
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]] = {}
    value_input_ids = set(_value_input_types(invocation))
    for operation in invocation.graph.operations:
        frontend = operation.kernel._lower(
            operation.features,
            autodiff_planning_policy=autodiff_planning_policy,
        ).frontend
        host_constants = _operation_host_constants(operation, invocation, value_input_ids)
        implementation = ProgramImplementation(
            operation.name,
            operation.kernel._entry,
            "compute",
            frontend.mlir,
            host_constants,
        )
        previous = implementations.get(operation.name)
        if previous is not None and (
            previous.mlir != implementation.mlir or previous.host_constants != implementation.host_constants
        ):
            raise ValueError(
                f"Program callee {operation.name!r} resolves to multiple specialized implementations; "
                "callee names must identify one implementation per Program specialization"
            )
        implementations[operation.name] = implementation
        for name, fields in frontend.structs:
            previous_fields = structs.get(name)
            if previous_fields is not None and previous_fields != fields:
                raise ValueError(f"Program struct {name!r} has conflicting canonical declarations")
            structs[name] = fields
        entry = next(
            (function for function in frontend.typed_functions if function.symbol == operation.kernel._entry),
            None,
        )
        if entry is None:
            raise ValueError(f"Program callee {operation.name!r} has no typed entry function")
        by_name = {parameter.name: parameter for parameter in entry.parameters}
        for parameter in operation.parameters:
            value_id = operation.inputs.get(parameter.name, operation.outputs.get(parameter.name))
            if value_id is None:
                continue
            typed = by_name.get(parameter.name)
            if typed is None:
                raise ValueError(
                    f"Program callee {operation.name!r} has no logical type for resource {parameter.name!r}"
                )
            value_type = _program_resource_type(typed.type)
            invocation.graph.refine_logical_type(value_id, value_type.logical)
    resource_types = {
        value.id: _program_resource_type(value.type.logical)
        for value in invocation.graph.values
        if value.type.logical is not None
    }

    values, arguments, results, operations = _lower_forward_function(invocation, resource_types)
    sources = {
        inspect.getsourcefile(function)
        for operation in invocation.graph.operations
        if (function := getattr(operation.kernel, "_function", None)) is not None
    }
    provenance = tuple(sorted(source for source in sources if source is not None))
    canonical_structs = tuple(sorted(structs.items()))
    mlir = emit_program(
        values,
        arguments,
        results,
        operations,
        direction="primal" if vjp_wrt else "forward",
        vjp_wrt=vjp_wrt,
        vjp_outputs=vjp_outputs,
        structs=canonical_structs,
    )
    return ParsedProgram(
        mlir,
        tuple(implementations[name] for name in sorted(implementations)),
        provenance,
        vjp_wrt,
        canonical_structs,
    )


__all__ = ["parse_program"]
