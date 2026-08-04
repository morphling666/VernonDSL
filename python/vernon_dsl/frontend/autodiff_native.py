from __future__ import annotations

from .autodiff import AutodiffProgram, NodeId, OpCode, SemanticProgramGraph, StaticIndicesOp
from .autodiff_native_abi import value_dtype
from .autodiff_native_common import AutodiffNativeLoweringError
from .autodiff_profiles import AutodiffProfilePlan

_GPU_TARGETS = frozenset({"cuda", "vulkan", "metal", "directx", "opengl", "opengles"})


def _static_indices(node):
    if not isinstance(node.payload, StaticIndicesOp) or not node.payload.indices:
        raise AutodiffNativeLoweringError(
            f"native autodiff requires static integer indices for '{node.operation.value}'"
        )
    return node.payload.indices


def _validate(graph: SemanticProgramGraph) -> None:
    if len(graph.outputs) != 1:
        raise AutodiffNativeLoweringError("native autodiff requires one structured output")
    nodes = {node.id: node for node in graph.nodes}
    active: set[NodeId] = set()

    def activate(value: NodeId) -> None:
        if value in active:
            return
        active.add(value)
        for operand in nodes[value].inputs:
            activate(operand)

    for output in graph.outputs:
        activate(output)
    for _, output in graph.storage_outputs:
        activate(output)
    storage_parameters = [
        node for node in graph.nodes if node.operation is OpCode.PARAMETER and node.type.kind == "tensor_view"
    ]
    if storage_parameters and any(
        node.operation is OpCode.CONSTANT and node.source_name is not None for node in graph.nodes
    ):
        raise AutodiffNativeLoweringError("native Storage autodiff does not support storage control flow")
    written_parameters = {
        node.source_name
        for node in graph.nodes
        if node.id in active and node.operation is OpCode.STORE and node.source_name is not None
    }
    if written_parameters and len(storage_parameters) > 1:
        raise AutodiffNativeLoweringError(
            "native Storage autodiff cannot prove non-aliasing between writable TensorView parameters"
        )
    supported = {
        OpCode.PARAMETER,
        OpCode.BUILTIN,
        OpCode.CONSTANT,
        OpCode.ADD,
        OpCode.SUB,
        OpCode.MUL,
        OpCode.DIV,
        OpCode.NEG,
        OpCode.IDENTITY,
        OpCode.SIN,
        OpCode.COS,
        OpCode.EXP,
        OpCode.LOG,
        OpCode.SQRT,
        OpCode.SPLAT,
        OpCode.BROADCAST,
        OpCode.CONDITIONAL,
        OpCode.COMPARE,
        OpCode.INDEX,
        OpCode.INDEX_DYNAMIC,
        OpCode.STORE,
        OpCode.STORE_DYNAMIC,
    }
    for node in graph.nodes:
        if node.id not in active:
            continue
        native_value = (node.type.kind == "scalar" and node.type.name in {"bool", "i32", "u32", "f32", "f64"}) or (
            node.type.kind in {"tensor", "tensor_view"}
            and value_dtype(node.type) in {"f32", "f64"}
            and all(
                isinstance(extent, int) and extent > 0
                for extent in (
                    node.type.arguments[1]
                    if node.type.kind == "tensor_view" and isinstance(node.type.arguments[1], tuple)
                    else node.type.arguments[1:]
                )
            )
        )
        builtin_tensor = (
            node.operation is OpCode.BUILTIN and node.type.kind == "tensor" and value_dtype(node.type) == "u32"
        )
        if not native_value and not builtin_tensor:
            raise AutodiffNativeLoweringError(
                "native autodiff lowering supports only native Scalars, static Tensor Values, "
                "and static-shape f32/f64 TensorView parameters"
            )
        if node.type.kind == "tensor_view" and node.operation not in {
            OpCode.PARAMETER,
            OpCode.CONDITIONAL,
            OpCode.STORE,
            OpCode.STORE_DYNAMIC,
        }:
            raise AutodiffNativeLoweringError(
                f"native autodiff lowering does not permit TensorView value operation '{node.operation.value}'"
            )
        if node.operation is OpCode.CONDITIONAL and node.type.kind == "tensor_view":
            raise AutodiffNativeLoweringError("native Storage autodiff does not support storage control flow")
        if node.operation not in supported:
            raise AutodiffNativeLoweringError(f"native autodiff lowering has no rule for '{node.operation.value}'")
        if node.operation in {OpCode.INDEX, OpCode.STORE}:
            indices = _static_indices(node)
            storage_type = nodes[node.inputs[0]].type
            if node.operation is OpCode.STORE and storage_type.kind != "tensor_view":
                raise AutodiffNativeLoweringError(
                    f"native Storage autodiff requires TensorView storage for '{node.operation.value}'"
                )
            if storage_type.kind not in {"tensor", "tensor_view"}:
                raise AutodiffNativeLoweringError(
                    f"native autodiff indexing requires a static Tensor Value or TensorView, got {storage_type.mlir}"
                )
            shape = storage_type.arguments[1] if storage_type.kind == "tensor_view" else storage_type.arguments[1:]
            if not isinstance(shape, tuple) or len(indices) != len(shape):
                raise AutodiffNativeLoweringError(
                    f"native Storage autodiff index rank does not match {storage_type.mlir}"
                )
            if any(index < 0 or index >= extent for index, extent in zip(indices, shape, strict=True)):
                raise AutodiffNativeLoweringError(
                    f"native Storage autodiff index is out of bounds for {storage_type.mlir}"
                )
        if node.operation in {OpCode.INDEX_DYNAMIC, OpCode.STORE_DYNAMIC}:
            storage_type = nodes[node.inputs[0]].type
            index_inputs = node.inputs[1:] if node.operation is OpCode.INDEX_DYNAMIC else node.inputs[1:-1]
            shape = storage_type.arguments[1] if storage_type.kind == "tensor_view" else storage_type.arguments[1:]
            if not isinstance(shape, tuple) or len(index_inputs) != len(shape):
                raise AutodiffNativeLoweringError(
                    f"native Storage autodiff dynamic index rank does not match {storage_type.mlir}"
                )
    parameter_names = {node.source_name for node in graph.nodes if node.operation is OpCode.PARAMETER}
    if any("." in path or path not in parameter_names for path in graph.wrt):
        raise AutodiffNativeLoweringError("initial native autodiff lowering requires whole-parameter wrt paths")


def emit_native_autodiff_modules(
    program: AutodiffProgram,
    profiles: AutodiffProfilePlan,
    *,
    target: str = "cpu",
) -> dict[str, str]:
    """Validate and dispatch native VJP lowering to the selected emitter."""

    _validate(program.semantic)
    if target == "cpu":
        from .autodiff_native_cpu import emit_backward, emit_forward
    elif target in _GPU_TARGETS:
        from .autodiff_native_gpu import emit_backward, emit_forward
    else:
        raise AutodiffNativeLoweringError(f"native autodiff lowering does not support target '{target}'")
    return {
        "forward_with_tape": emit_forward(program, profiles),
        "backward": (
            emit_backward(program, profiles, target=target)
            if target in _GPU_TARGETS
            else emit_backward(program, profiles)
        ),
    }


__all__ = ["AutodiffNativeLoweringError", "emit_native_autodiff_modules"]
