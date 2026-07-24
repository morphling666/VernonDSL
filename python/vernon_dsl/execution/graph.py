from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, order=True)
class NodeHandle:
    index: int


@dataclass(frozen=True)
class ComputeDispatch:
    handle: NodeHandle
    kernel: Any
    arguments: tuple[Any, ...]
    grid: tuple[int, int, int] | None
    dependencies: tuple[NodeHandle, ...]


@dataclass(frozen=True)
class GraphicsDraw:
    handle: NodeHandle
    pipeline: Any
    arguments: Mapping[str, Any]
    dependencies: tuple[NodeHandle, ...]


@dataclass(frozen=True)
class ResourceBarrier:
    handle: NodeHandle
    resources: tuple[Any, ...]
    source: str
    destination: str
    dependencies: tuple[NodeHandle, ...]


ExecutionNode = ComputeDispatch | GraphicsDraw | ResourceBarrier


class ExecutionGraph:
    """Typed, forward-only host execution graph for one synchronous run."""

    def __init__(self, *, automatic_transitions: bool = True):
        self._nodes: list[ExecutionNode] = []
        self._automatic_transitions = bool(automatic_transitions)
        self._generation = importlib.import_module("vernon_dsl._runtime.session")._runtime_generation

    @property
    def nodes(self) -> tuple[ExecutionNode, ...]:
        return tuple(self._nodes)

    @property
    def automatic_transitions(self) -> bool:
        return self._automatic_transitions

    def _dependencies(self, values: Iterable[NodeHandle] | None) -> tuple[NodeHandle, ...]:
        if values is None:
            return (NodeHandle(len(self._nodes) - 1),) if self._nodes else ()
        dependencies = tuple(sorted(set(values)))
        if any(value.index < 0 or value.index >= len(self._nodes) for value in dependencies):
            raise ValueError("graph dependency must reference an existing node")
        return dependencies

    def dispatch(
        self,
        kernel: Any,
        *arguments: Any,
        grid: tuple[int, int, int] | None = None,
        depends_on: Iterable[NodeHandle] | None = None,
    ) -> NodeHandle:
        if getattr(kernel, "__class__", None).__name__ != "Kernel":
            raise TypeError("dispatch requires a Vernon Kernel")
        handle = NodeHandle(len(self._nodes))
        self._nodes.append(ComputeDispatch(handle, kernel, tuple(arguments), grid, self._dependencies(depends_on)))
        return handle

    def draw(
        self,
        pipeline: Any,
        *,
        depends_on: Iterable[NodeHandle] | None = None,
        **arguments: Any,
    ) -> NodeHandle:
        if getattr(pipeline, "__class__", None).__name__ != "Pipeline":
            raise TypeError("draw requires a Vernon Pipeline")
        handle = NodeHandle(len(self._nodes))
        self._nodes.append(GraphicsDraw(handle, pipeline, dict(arguments), self._dependencies(depends_on)))
        return handle

    def barrier(
        self,
        *resources: Any,
        source: str,
        destination: str,
        depends_on: Iterable[NodeHandle] | None = None,
    ) -> NodeHandle:
        if not resources:
            raise ValueError("barrier requires at least one resource")
        handle = NodeHandle(len(self._nodes))
        self._nodes.append(
            ResourceBarrier(
                handle,
                tuple(resources),
                source,
                destination,
                self._dependencies(depends_on),
            )
        )
        return handle

    @property
    def semantic_inputs(self) -> Mapping[str, Any]:
        from .lower import graph_semantic_inputs

        return graph_semantic_inputs(self)

    def validate(self) -> None:
        from .validate import validate_graph

        validate_graph(self)

    def run(self) -> None:
        from .executor import run_graph

        run_graph(self)

    def cook(
        self,
        output: str | Path,
        *,
        target: str | None = None,
        target_options: Mapping[str, Any] | None = None,
    ) -> Path:
        from .cooking import cook_execution_graph

        state = importlib.import_module("vernon_dsl._runtime.session")
        resolved_target = state._architecture.name if target is None else target
        resolved_options = target_options
        if (
            resolved_options is None
            and resolved_target in {"opengl", "opengles"}
            and state._architecture.name == resolved_target
        ):
            resolved_options = {"glsl_version": state._interactive_glsl_version()}
        return cook_execution_graph(
            self,
            output=output,
            target=resolved_target,
            target_options=resolved_options,
        )


__all__ = [
    "ComputeDispatch",
    "ExecutionGraph",
    "ExecutionNode",
    "GraphicsDraw",
    "NodeHandle",
    "ResourceBarrier",
]
