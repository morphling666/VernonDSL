from __future__ import annotations

from typing import Any, Mapping

from .._runtime.resources import TensorStorage, TensorView, Texture
from .graph import ComputeDispatch, ExecutionGraph, GraphicsDraw


def _resource_descriptor(value: Any) -> Mapping[str, Any]:
    if isinstance(value, (TensorStorage, TensorView)):
        layout = value.layout
        return {
            "kind": "tensor",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "byte_strides": list(layout.byte_strides),
            "byte_offset": layout.byte_offset,
            "access": value.access if isinstance(value, TensorView) else "read_write",
        }
    if isinstance(value, Texture):
        return {
            "kind": "texture",
            "width": value.width,
            "height": value.height,
        }
    if isinstance(value, bool):
        return {"kind": "scalar", "type": "bool"}
    if isinstance(value, int):
        return {"kind": "scalar", "type": "i32"}
    if isinstance(value, float):
        return {"kind": "scalar", "type": "f32"}
    return {"kind": type(value).__name__}


def graph_semantic_inputs(graph: ExecutionGraph) -> Mapping[str, Any]:
    nodes: list[dict[str, Any]] = []
    for node in graph.nodes:
        common = {
            "id": node.handle.index,
            "dependencies": [value.index for value in node.dependencies],
        }
        if isinstance(node, ComputeDispatch):
            frontend, _, _, _ = node.kernel._lower(node.arguments)
            nodes.append(
                {
                    **common,
                    "kind": "dispatch",
                    "entry": node.kernel._entry,
                    "frontend": frontend.semantic_inputs,
                    "grid": list(node.grid) if node.grid is not None else None,
                    "bindings": [_resource_descriptor(value) for value in node.arguments],
                }
            )
        elif isinstance(node, GraphicsDraw):
            stage_functions = [
                getattr(stage, "_function", getattr(stage, "function", None)) for stage in node.pipeline._stages
            ]
            nodes.append(
                {
                    **common,
                    "kind": "draw",
                    "stages": [
                        value.__name__ if value is not None else type(stage).__name__
                        for stage, value in zip(node.pipeline._stages, stage_functions, strict=True)
                    ],
                    "features": list(node.pipeline._features),
                    "bindings": {name: _resource_descriptor(value) for name, value in sorted(node.arguments.items())},
                }
            )
        else:
            nodes.append(
                {
                    **common,
                    "kind": "barrier",
                    "source": node.source,
                    "destination": node.destination,
                    "resources": [_resource_descriptor(value) for value in node.resources],
                }
            )
    return {
        "version": 1,
        "automatic_transitions": graph.automatic_transitions,
        "nodes": nodes,
    }


def infer_dispatch_grid(
    node: ComputeDispatch,
    function: Any,
    builtins: set[str],
    writable: set[str],
) -> tuple[int, int, int]:
    if node.grid is not None:
        grid = node.grid
    else:
        names = [value.arg for value in function.args.args if value.arg not in builtins]
        shapes = {
            value.shape
            for name, value in zip(names, node.arguments, strict=True)
            if name in writable and isinstance(value, (TensorStorage, TensorView))
        }
        if len(shapes) != 1:
            raise ValueError("graph dispatch grid cannot be inferred from writable Tensor arguments")
        shape = next(iter(shapes))
        if not 1 <= len(shape) <= 3:
            raise ValueError("inferred graph grids require Tensor rank one through three")
        grid = tuple(reversed(shape)) + (1,) * (3 - len(shape))
    if len(grid) != 3 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid):
        raise ValueError("graph dispatch grid must contain three positive integers")
    return grid


__all__ = ["graph_semantic_inputs", "infer_dispatch_grid"]
