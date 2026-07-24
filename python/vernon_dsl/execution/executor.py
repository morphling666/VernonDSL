from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .._runtime.resources import TensorStorage, TensorView, _dispatch_borrow_scope
from ..bundle.parameters import external_parameters
from ..bundle.planner import compiled_stage_from_program, plan_variant
from ..bundle.serialize import (
    canonical_json,
    inline_artifact_descriptor,
    materialize_bundle,
    serialize_bundle,
)
from ..bundle.types import BundlePlan, CompiledStage, TargetOptions, VariantPlan
from .graph import ComputeDispatch, ExecutionGraph, GraphicsDraw, ResourceBarrier
from .lower import infer_dispatch_grid
from .validate import graph_accesses, validate_graph


@dataclass
class _CompiledGraph:
    native: Any
    bundle: bytes
    generation: int
    bindings: dict[str, Any]


class _GraphCache:
    values: ClassVar[dict[str, _CompiledGraph]] = {}


def _target() -> tuple[Any, TargetOptions]:
    state = importlib.import_module("vernon_dsl._runtime.session")
    if state._native is None or state._native_runtime is None:
        raise RuntimeError("execution graph requires the native runtime")
    native_target, name = {
        state.cpu: (state._native.Target.CPU, "cpu"),
        state.cuda: (state._native.Target.CUDA, "cuda"),
        state.vulkan: (state._native.Target.VULKAN, "vulkan"),
        state.opengl: (state._native.Target.OPENGL, "opengl"),
        state.opengles: (state._native.Target.OPENGL_ES, "opengles"),
    }[state._architecture]
    options = TargetOptions(
        name,
        {"glsl_version": state._interactive_glsl_version()}
        if state._architecture in {state.opengl, state.opengles}
        else {},
    )
    return native_target, options


def _compile_compute_graph(graph: ExecutionGraph) -> _CompiledGraph:
    state = importlib.import_module("vernon_dsl._runtime.session")
    _, target = _target()
    stages: dict[str, CompiledStage] = {}
    parameters: list[dict[str, Any]] = []
    internal_parameters: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    slot = 0
    previous_dispatch = False

    for node in graph.nodes:
        if isinstance(node, ResourceBarrier):
            steps.append(
                {
                    "kind": "barrier",
                    "source": node.source,
                    "destination": node.destination,
                }
            )
            previous_dispatch = False
            continue
        if not isinstance(node, ComputeDispatch):
            raise TypeError("native compute graph cannot contain draw nodes")
        frontend, function, builtins, _ = node.kernel._lower(node.arguments)
        writable = node.kernel._writable_parameters(function)
        grid = infer_dispatch_grid(node, function, builtins, writable)
        compiled_kernel = node.kernel._compile(node.arguments)
        program = compiled_kernel.program
        stage = compiled_stage_from_program(
            program,
            module=f"graph/node_{node.handle.index}",
            module_manifest=canonical_json(frontend.semantic_inputs),
            entry=node.kernel._entry,
            target=target,
        )
        stages[stage.id] = stage
        record = stage.logical_record()
        external = external_parameters({"compute": record})
        local_slots = {name: index for index, name in enumerate(sorted(external))}
        local_plan = plan_variant((), {"compute": record}, local_slots)
        user_names = [value.arg for value in function.args.args if value.arg not in builtins]
        values = dict(zip(user_names, node.arguments, strict=True))
        for row_value in local_plan.parameters:
            row = dict(row_value)
            source_name = str(row["name"])
            alias = f"node_{node.handle.index}_{source_name}"
            row["name"] = alias
            row["slot"] = slot
            row["uses"] = [{**use, "stage": stage.id} for use in row["uses"]]
            if isinstance(values[source_name], (TensorStorage, TensorView)):
                row["shape"] = [0] * len(values[source_name].shape)
            parameters.append(row)
            bindings[alias] = values[source_name]
            slot += 1
        for value in local_plan.internal_parameters:
            row = dict(value)
            name = str(row["name"])
            row["uses"] = [{**use, "stage": stage.id} for use in row["uses"]]
            existing = internal_parameters.get(name)
            if existing is None:
                internal_parameters[name] = row
            else:
                existing["uses"] = [*existing["uses"], *row["uses"]]
        if graph.automatic_transitions and previous_dispatch:
            steps.append(
                {
                    "kind": "barrier",
                    "source": "compute_write",
                    "destination": "compute_read_write",
                }
            )
        steps.append({"kind": "dispatch", "stage": stage.id, "grid": list(grid)})
        previous_dispatch = True

    identity = canonical_json(graph.semantic_inputs)
    pipeline_id = f"graph/{hashlib.sha256(identity.encode()).hexdigest()}"
    plan = BundlePlan(
        pipeline_id,
        target,
        (),
        (
            VariantPlan(
                (),
                {},
                tuple(parameters),
                tuple(internal_parameters[name] for name in sorted(internal_parameters)),
                (),
                tuple(steps),
            ),
        ),
        tuple(stages[key] for key in sorted(stages)),
    )
    bundle = materialize_bundle(
        plan,
        {stage.id: inline_artifact_descriptor(stage.artifact) for stage in stages.values()},
    )
    bundle_bytes = serialize_bundle(bundle)
    key = hashlib.sha256(bundle_bytes).hexdigest()
    cached = _GraphCache.values.get(key)
    if cached is None:
        cached = _CompiledGraph(
            state._native_runtime.load_pipeline(bundle_bytes, []),
            bundle_bytes,
            state._runtime_generation,
            bindings,
        )
        _GraphCache.values[key] = cached
    elif cached.generation != state._runtime_generation:
        cached.native = state._native_runtime.load_pipeline(cached.bundle, [])
        cached.generation = state._runtime_generation
    cached.bindings = bindings
    return cached


def _bind_compute_graph(compiled: _CompiledGraph) -> None:
    state = importlib.import_module("vernon_dsl._runtime.session")
    builder = compiled.native.invocation_builder()
    dtype_codes = {
        np.dtype(np.bool_): state._native.DATA_BOOL,
        np.dtype(np.int32): state._native.DATA_I32,
        np.dtype(np.uint32): state._native.DATA_U32,
        np.dtype(np.float16): state._native.DATA_F16,
        np.dtype(np.float32): state._native.DATA_F32,
        np.dtype(np.float64): state._native.DATA_F64,
    }
    parameters = {parameter.name: parameter for parameter in compiled.native.parameters}
    for name, value in compiled.bindings.items():
        parameter = parameters[name]
        if isinstance(value, (TensorStorage, TensorView)):
            layout = value.layout
            builder.device_tensor(
                name,
                value._resident_buffer(),
                dtype_codes[value.dtype],
                parameter.access,
                list(value.shape),
                list(layout.byte_strides),
                layout.byte_offset,
            )
        else:
            scalar = np.asarray(value)
            if scalar.dtype.kind == "f":
                scalar = np.asarray(value, dtype=np.float32)
            elif scalar.dtype.kind == "u":
                scalar = np.asarray(value, dtype=np.uint32)
            elif scalar.dtype.kind == "b":
                scalar = np.asarray(value, dtype=np.bool_)
            else:
                scalar = np.asarray(value, dtype=np.int32)
            builder.host_tensor(name, scalar)
    builder.grid(1, 1, 1)
    builder.invoke()


def run_graph(graph: ExecutionGraph) -> None:
    validate_graph(graph)
    state = importlib.import_module("vernon_dsl._runtime.session")
    if any(isinstance(node, GraphicsDraw) for node in graph.nodes):
        for node in graph.nodes:
            if isinstance(node, ComputeDispatch):
                node.kernel._invoke_direct(node.arguments, node.grid)
            elif isinstance(node, GraphicsDraw):
                node.pipeline._invoke_direct(dict(node.arguments))
            elif isinstance(node, ResourceBarrier):
                state._native_runtime.synchronize()
        return
    if state._architecture == state.cpu:
        for node in graph.nodes:
            if isinstance(node, ComputeDispatch):
                node.kernel._invoke_direct(node.arguments, node.grid)
        return

    accesses = graph_accesses(graph)
    if any(
        value.value.dtype.fields is not None or getattr(value.value, "_element_type", None) is not None
        for value in accesses
    ):
        for node in graph.nodes:
            if isinstance(node, ComputeDispatch):
                node.kernel._invoke_direct(node.arguments, node.grid)
        return
    owner_borrows: dict[int, tuple[str, TensorStorage | TensorView, str]] = {}
    for access in accesses:
        owner = access.value.owner if isinstance(access.value, TensorView) else access.value
        key = id(owner)
        existing = owner_borrows.get(key)
        mode = access.mode
        if existing is not None and existing[2] != mode:
            mode = "read_write"
        borrowed = owner if isinstance(owner, TensorStorage) else access.value
        owner_borrows[key] = (f"graph_owner_{len(owner_borrows)}", borrowed, mode)
    borrows = list(owner_borrows.values())
    compiled = _compile_compute_graph(graph)
    with _dispatch_borrow_scope(borrows):
        _bind_compute_graph(compiled)
        state._native_runtime.synchronize()
        for access in accesses:
            if access.mode in {"write", "read_write"}:
                access.value._mark_device_dirty()


__all__ = ["run_graph"]
