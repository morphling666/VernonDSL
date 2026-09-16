"""Typed validation for canonical, invocation-independent Program deployments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..bundle import ProgramCompileError


def _is_control_component(value: object) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value > 0
    if not isinstance(value, Mapping):
        return False
    if set(value) == {"dimension"}:
        dimension = value["dimension"]
        return (
            isinstance(dimension, Mapping)
            and set(dimension) == {"control", "axis"}
            and isinstance(dimension["axis"], int)
            and not isinstance(dimension["axis"], bool)
            and dimension["axis"] >= 0
            and _is_control_reference(dimension["control"])
        )
    return set(value) == {"control"} and _is_control_reference(value["control"])


def _is_control_reference(control: object) -> bool:
    return (
        isinstance(control, Mapping)
        and set(control) == {"value"}
        and isinstance(control["value"], int)
        and not isinstance(control["value"], bool)
        and control["value"] >= 0
    )


def _validate_storage(storage: Mapping[str, Any], index: int) -> None:
    descriptor = storage.get("descriptor")
    if not isinstance(descriptor, Mapping) or descriptor.get("tag") != "image":
        return
    ownership = storage.get("ownership")
    extent = descriptor.get("extent")
    if ownership == "borrowed":
        if "extent" in descriptor:
            raise ProgramCompileError(f"canonical Program Storage {index} captures a borrowed image invocation extent")
        return
    if ownership != "owned":
        raise ProgramCompileError(f"canonical Program Storage {index} has invalid image ownership")
    if (
        not isinstance(extent, Sequence)
        or isinstance(extent, (str, bytes))
        or len(extent) != 3
        or not all(_is_control_component(component) for component in extent)
    ):
        raise ProgramCompileError(
            f"canonical owned image Storage {index} requires three positive static or Value-controlled extents"
        )


def _validate_graph(graph: Mapping[str, Any], graph_index: int) -> None:
    nodes = graph.get("nodes")
    if not isinstance(nodes, Sequence):
        raise ProgramCompileError(f"canonical Program graph {graph_index} has no typed nodes")
    for node_index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise ProgramCompileError(f"canonical Program graph {graph_index} node {node_index} is not typed")
        operation = node.get("operation")
        if not isinstance(operation, Mapping):
            raise ProgramCompileError(f"canonical Program graph {graph_index} node {node_index} has no operation")
        if operation.get("tag") == "compute":
            workgroups = operation.get("workgroups")
            if (
                not isinstance(workgroups, Sequence)
                or isinstance(workgroups, (str, bytes))
                or len(workgroups) != 3
                or not all(_is_control_component(component) for component in workgroups)
            ):
                raise ProgramCompileError(
                    f"canonical Program graph {graph_index} node {node_index} captures an invocation grid"
                )
        if operation.get("tag") == "graphics":
            render_pass = operation.get("render_pass")
            draw = operation.get("draw")
            dynamic_state = operation.get("dynamic_state")
            if (
                not isinstance(render_pass, Mapping)
                or not isinstance(render_pass.get("control"), int)
                or not isinstance(draw, Mapping)
                or not isinstance(draw.get("control"), int)
                or not isinstance(dynamic_state, Mapping)
                or set(dynamic_state) != {"control"}
                or not isinstance(dynamic_state["control"], int)
            ):
                raise ProgramCompileError(
                    f"canonical Program graph {graph_index} node {node_index} captures invocation graphics state"
                )


def validate_canonical_deployment(program: Mapping[str, Any]) -> None:
    """Reject invocation snapshots using Program ownership and control types."""

    storages = program.get("storages")
    values = program.get("values")
    graphs = program.get("graphs")
    if not isinstance(storages, Sequence) or not isinstance(values, Sequence) or not isinstance(graphs, Sequence):
        raise ProgramCompileError("canonical Program deployment is missing typed Storage, Value, or graph records")
    for index, storage in enumerate(storages):
        if not isinstance(storage, Mapping):
            raise ProgramCompileError(f"canonical Program Storage {index} is not typed")
        _validate_storage(storage, index)
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or not isinstance(value.get("origin"), Mapping):
            raise ProgramCompileError(f"canonical Program Value {index} has no typed origin")
        origin = value["origin"]
        if origin.get("tag") == "argument" and "view" in value:
            raise ProgramCompileError(f"canonical Program argument Value {index} captures an invocation TensorView")
    for index, graph in enumerate(graphs):
        if not isinstance(graph, Mapping):
            raise ProgramCompileError(f"canonical Program graph {index} is not typed")
        _validate_graph(graph, index)


__all__ = ["validate_canonical_deployment"]
