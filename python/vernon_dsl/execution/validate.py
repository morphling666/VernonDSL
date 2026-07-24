from __future__ import annotations

import importlib
from dataclasses import dataclass

from .._runtime.resources import TensorStorage, TensorView, _views_overlap
from .graph import ComputeDispatch, ExecutionGraph, GraphicsDraw, ResourceBarrier


@dataclass(frozen=True)
class BoundAccess:
    node: int
    name: str
    value: TensorStorage | TensorView
    mode: str

    @property
    def view(self) -> TensorView:
        return self.value if isinstance(self.value, TensorView) else self.value.view(access=self.mode)


def _writes(mode: str) -> bool:
    return mode in {"write", "read_write"}


def _reachable(graph: ExecutionGraph, source: int, destination: int) -> bool:
    pending = [destination]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current == source:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(value.index for value in graph.nodes[current].dependencies)
    return False


def _compute_accesses(node: ComputeDispatch) -> tuple[BoundAccess, ...]:
    frontend, function, builtins, _ = node.kernel._lower(node.arguments)
    del frontend
    names = [value.arg for value in function.args.args if value.arg not in builtins]
    writable = node.kernel._writable_parameters(function)
    accesses: list[BoundAccess] = []
    for name, value in zip(names, node.arguments, strict=True):
        if not isinstance(value, (TensorStorage, TensorView)):
            continue
        if isinstance(value, TensorView):
            mode = value.access
        else:
            mode = "write" if name in writable else "read"
        accesses.append(BoundAccess(node.handle.index, name, value, mode))
    return tuple(accesses)


def _draw_accesses(node: GraphicsDraw) -> tuple[BoundAccess, ...]:
    accesses: list[BoundAccess] = []
    for name, value in sorted(node.arguments.items()):
        if not isinstance(value, (TensorStorage, TensorView)):
            continue
        mode = value.access if isinstance(value, TensorView) else "read"
        if name in {"target", "targets"}:
            mode = "write"
        accesses.append(BoundAccess(node.handle.index, name, value, mode))
    return tuple(accesses)


def graph_accesses(graph: ExecutionGraph) -> tuple[BoundAccess, ...]:
    accesses: list[BoundAccess] = []
    for node in graph.nodes:
        if isinstance(node, ComputeDispatch):
            accesses.extend(_compute_accesses(node))
        elif isinstance(node, GraphicsDraw):
            accesses.extend(_draw_accesses(node))
    return tuple(accesses)


def _barrier_orders(
    graph: ExecutionGraph,
    producer: BoundAccess,
    consumer: BoundAccess,
) -> bool:
    for node in graph.nodes:
        if not isinstance(node, ResourceBarrier):
            continue
        if not _reachable(graph, producer.node, node.handle.index):
            continue
        if not _reachable(graph, node.handle.index, consumer.node):
            continue
        barrier_views = [
            value if isinstance(value, TensorView) else value.view()
            for value in node.resources
            if isinstance(value, (TensorStorage, TensorView))
        ]
        if any(_views_overlap(producer.view, value) for value in barrier_views):
            return True
    return False


def validate_graph(graph: ExecutionGraph) -> None:
    if not graph.nodes:
        raise ValueError("execution graph is empty")
    state = importlib.import_module("vernon_dsl._runtime.session")
    if graph._generation != state._runtime_generation:
        raise RuntimeError("execution graph belongs to a different runtime generation")
    for expected, node in enumerate(graph.nodes):
        if node.handle.index != expected:
            raise ValueError("execution graph node handles must be dense and ordered")
        if any(value.index >= node.handle.index for value in node.dependencies):
            raise ValueError("execution graph contains a cycle or forward dependency")
    accesses = graph_accesses(graph)
    for index, left in enumerate(accesses):
        for right in accesses[index + 1 :]:
            if left.node == right.node or not (_writes(left.mode) or _writes(right.mode)):
                continue
            if not _views_overlap(left.view, right.view):
                continue
            if not _reachable(graph, left.node, right.node):
                raise ValueError(
                    f"unordered graph hazard between node {left.node} {left.name!r} "
                    f"and node {right.node} {right.name!r}"
                )
            if not graph.automatic_transitions and not _barrier_orders(graph, left, right):
                raise ValueError(f"missing resource transition between node {left.node} and node {right.node}")


__all__ = ["BoundAccess", "graph_accesses", "validate_graph"]
