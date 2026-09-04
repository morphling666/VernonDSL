from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .._shader_assets.artifact_io import write_external_artifact
from .._shader_assets.cooking import (
    _canonical_deployment,
    _compile_program_bundle_plan,
    _native_target,
)
from ..bundle import canonical_json, make_target_options
from ..storage import TensorStorage, TensorView
from . import session as state
from .autodiff import _pipeline_derivative_groups
from .binding import _DispatchBorrowLease, _PersistentBindingTable
from .sampler import SamplerState
from .texture import _TextureResource


@dataclass
class _ProgramDeployment:
    directory: tempfile.TemporaryDirectory[str]
    canonical_program: bytes
    artifact_system: bytes
    stage_bindings: Any
    compiled_stages: list[tuple[str, str, Any]]

    def load(self) -> Any:
        if state._native_runtime is None:
            raise RuntimeError(f"{state._architecture.name} Program execution requires the native runtime")
        return state._native_runtime.load_canonical_program(
            self.canonical_program,
            self.artifact_system,
            self.directory.name,
            self.stage_bindings,
            self.compiled_stages,
        )


def _bind_program_graphics_controls(pipeline: Any, cache: _PersistentBindingTable, invocation: Any) -> list[Any]:
    from ..render import ColorBlendState, LoadOperation, StoreOperation, lines, points, triangles

    load_values = {
        LoadOperation.CLEAR: state._native.ATTACHMENT_CLEAR,
        LoadOperation.PRESERVE: state._native.ATTACHMENT_PRESERVE,
        LoadOperation.DISCARD: state._native.ATTACHMENT_DISCARD,
    }
    store_values = {
        StoreOperation.PRESERVE: state._native.ATTACHMENT_STORE,
        StoreOperation.DISCARD: state._native.ATTACHMENT_DONT_CARE,
    }
    written: list[Any] = []
    operations = {operation.id: operation for operation in invocation.graph.operations}
    for node_id, controls in invocation.graphics_controls.items():
        operation = operations[node_id]
        render_slot, render_pass = controls["render_pass"]
        draw_slot, draw = controls["draw"]
        dynamic_slot, dynamic = controls["dynamic_state"]
        builder = pipeline.invocation_builder()
        colors = tuple(render_pass.target._color_attachments())
        color_operations = dict(render_pass.colors)
        for location, texture in colors:
            attachment = color_operations[location]
            clear_value = (
                tuple(float(component) for component in attachment.clear_value)
                if attachment.load is LoadOperation.CLEAR
                else (0.0, 0.0, 0.0, 0.0)
            )
            builder.rhi_color_attachment(
                location,
                texture._resident_view(),
                load_values[attachment.load],
                store_values[attachment.store],
                list(clear_value),
            )
            written.append(texture)
        depth_texture = render_pass.target._depth_attachment()
        if depth_texture is not None:
            if render_pass.depth is None:
                raise ValueError("depth attachment operations are required for a depth RenderTarget")
            attachment = render_pass.depth
            clear_depth = float(attachment.clear_value) if attachment.load is LoadOperation.CLEAR else 1.0
            builder.rhi_depth_attachment(
                depth_texture._resident_view(),
                load_values[attachment.load],
                store_values[attachment.store],
                clear_depth,
            )
            written.append(depth_texture)
        if draw is not None and draw.index_buffer is not None:
            index_view = draw.index_buffer.view
            builder.rhi_index_binding(
                index_view._resident_buffer(),
                draw.index_buffer.count,
                index_view.layout.byte_offset,
            )
        topology = {
            triangles: state._native.TOPOLOGY_TRIANGLE_LIST,
            lines: state._native.TOPOLOGY_LINE_LIST,
            points: state._native.TOPOLOGY_POINT_LIST,
        }[operation.pipeline._topology]
        builder.topology(topology)
        builder.counts(
            0 if draw is None or draw.vertex_count is None else draw.vertex_count,
            1 if draw is None else draw.instance_count,
        )
        if render_pass.render_area is not None:
            builder.viewport(*render_pass.render_area)
            builder.scissor(*render_pass.render_area)
        if dynamic is not None:
            if dynamic.viewport is not None:
                builder.viewport(*dynamic.viewport)
            if dynamic.scissor is not None:
                builder.scissor(*dynamic.scissor)
            builder.stencil_reference(dynamic.stencil_reference)
        configured_blends = dict(operation.pipeline._graphics_state.color_blends)
        graphics_state = replace(
            operation.pipeline._graphics_state,
            color_blends=tuple(
                (location, configured_blends.get(location, ColorBlendState())) for location, _ in colors
            ),
        )
        builder.graphics_state(graphics_state)
        render_token = (
            "render-pass",
            node_id,
            repr(render_pass).encode(),
            *(id(texture._resident_view()) for _, texture in colors),
        )
        draw_token = ("draw-command", node_id, repr(draw).encode())
        dynamic_token = ("dynamic-state", node_id, repr(dynamic).encode())
        cache.bind_render_pass_control(render_slot, render_token, builder)
        cache.bind_draw_command_control(draw_slot, draw_token, builder)
        cache.bind_dynamic_state_control(dynamic_slot, dynamic_token, builder)
    return written


def _program_storage_leaf(value: Any, root: str, leaf_path: str) -> Any:
    if leaf_path == root:
        return value
    projection = value
    for component in leaf_path[len(root) + 1 :].split("."):
        projection = projection[int(component)] if component.isdigit() else projection.field(component)
    return projection


def _program_host_array(value: Any) -> np.ndarray:
    if hasattr(value, "_native_host_array"):
        return np.asarray(value._native_host_array())
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy())
    return np.asarray(value)


@contextmanager
def _bound_program_invocation(
    pipeline: Any,
    cache: _PersistentBindingTable,
    invocation: Any,
    targets: Mapping[str, Any],
    *,
    retain_borrows: bool = False,
) -> Iterator[tuple[Any, list[Any], _DispatchBorrowLease]]:
    parameters = tuple(pipeline.parameters)
    parameters_by_slot = {parameter.slot: parameter for parameter in parameters}
    if len(parameters_by_slot) != len(parameters):
        raise RuntimeError("Program compiler ABI contains duplicate public slots")
    boundary_slots = tuple(
        slot for slot in pipeline.program_abi["boundary_slots"] if slot["role"] in {"input", "output"}
    )
    if set(parameters_by_slot) != {slot["slot"] for slot in boundary_slots}:
        raise RuntimeError("Program invocation parameters do not match compiler-emitted ProgramABI")

    def binding(slot: Mapping[str, Any]) -> tuple[str, Any]:
        path = slot["path"]
        values = invocation.inputs if slot["role"] == "input" else targets
        if path not in values:
            raise RuntimeError(f"Program invocation is missing boundary value {path!r}")
        return path, values[path]

    access_names = {
        state._native.ACCESS_READ: "read",
        state._native.ACCESS_WRITE: "write",
        state._native.ACCESS_READ_WRITE: "read_write",
    }
    resolved = [(parameters_by_slot[slot["slot"]], slot, *binding(slot)) for slot in boundary_slots]
    borrows = [
        (path, value, access_names[parameter.access])
        for parameter, _, path, value in resolved
        if isinstance(value, (TensorStorage, TensorView, _TextureResource))
    ]
    for controls in invocation.graphics_controls.values():
        render_pass = controls["render_pass"][1]
        borrows.extend(
            (f"render_attachment_{location}", texture, "write")
            for location, texture in render_pass.target._color_attachments()
        )
        depth = render_pass.target._depth_attachment()
        if depth is not None:
            borrows.append(("depth_attachment", depth, "write"))
        draw = controls["draw"][1]
        if draw is not None and draw.index_buffer is not None:
            borrows.append(("index_buffer", draw.index_buffer.view, "read"))
    written: list[Any] = []
    lease = _DispatchBorrowLease(borrows)
    succeeded = False
    try:
        with cache.invocation(pipeline) as builder:
            for parameter, slot, path, value in resolved:
                if parameter.kind == state._native.PIPELINE_SAMPLER:
                    if not isinstance(value, SamplerState):
                        raise TypeError(f"sampler {parameter.name!r} must be a SamplerState")
                    cache.bind_sampler(builder, pipeline, parameter, value)
                else:
                    cache.bind_argument(
                        builder,
                        pipeline,
                        parameter,
                        value,
                        host_value=slot.get("category") == "value",
                        annotation=invocation.input_annotations.get(path),
                    )
                if (
                    state._architecture != state.cpu
                    and parameter.access != state._native.ACCESS_READ
                    and hasattr(value, "_mark_device_dirty")
                ):
                    written.append(value)
            written.extend(_bind_program_graphics_controls(pipeline, cache, invocation))
            yield builder, written, lease
        succeeded = True
    finally:
        if retain_borrows and succeeded:
            lease.release_writes()
        else:
            lease.release()


class ProgramNativePullback:
    def __init__(
        self,
        native: Any,
        signature: Mapping[str, Any],
        outputs: Mapping[str, Any],
        inputs: Mapping[str, Any],
        derivative_groups: tuple[Any, ...],
        lease: _DispatchBorrowLease,
    ) -> None:
        self._native = native
        self._signature = signature
        self._outputs = dict(outputs)
        self._inputs = dict(inputs)
        self._lease = lease
        self._storage_gradients = {
            path for path, value in inputs.items() if isinstance(value, (TensorStorage, TensorView))
        }
        self._gradient_groups = tuple(group for group in derivative_groups if group.role == "gradient")
        self._cotangent_groups = tuple(group for group in derivative_groups if group.role == "cotangent")

    def __call__(self, cotangents: Any = None) -> dict[str, Any]:
        paths = tuple(row["path"] for row in self._signature["cotangents"])
        if cotangents is None:
            if len(paths) != 1:
                raise ValueError("implicit Program cotangent requires exactly one output")
            output = self._outputs[paths[0]]
            host = _program_host_array(output)
            supplied: Any = np.ones(host.shape, dtype=host.dtype)
        elif isinstance(cotangents, Mapping):
            if set(cotangents) != set(paths):
                raise ValueError("Program pullback requires exactly one cotangent per output")
            supplied = dict(cotangents)
        elif len(paths) == 1:
            supplied = cotangents
        else:
            raise TypeError("Program pullback requires a cotangent mapping for multiple outputs")
        if isinstance(supplied, Mapping):
            grouped_cotangents: dict[str, Any] = dict(supplied)
            for group in self._cotangent_groups:
                root = group.declared_path
                value = grouped_cotangents[root]
                if isinstance(value, TensorStorage) and any(leaf != root for leaf in group.leaf_paths):
                    grouped_cotangents[root] = {
                        leaf: _program_storage_leaf(value, root, leaf)._native_host_array() for leaf in group.leaf_paths
                    }
            supplied = grouped_cotangents
        try:
            gradients = dict(
                self._native.apply_grouped(
                    supplied,
                    self._gradient_groups,
                    self._cotangent_groups,
                    (),
                    False,
                )
            )
        finally:
            self._lease.release()
        return {
            path: (
                value
                if isinstance(value, TensorStorage) or path not in self._storage_gradients
                else TensorStorage.from_numpy(np.asarray(value))
            )
            for path, value in gradients.items()
        }

    def __del__(self) -> None:
        lease = getattr(self, "_lease", None)
        if lease is not None:
            lease.release()

    @property
    def logical_residual_bytes(self) -> int:
        return int(self._native.logical_residual_bytes)

    @property
    def estimated_tape_bytes(self) -> int:
        return int(self._native.estimated_tape_bytes)

    @property
    def resident_tape_bytes(self) -> int:
        return int(self._native.resident_bytes)

    @property
    def allocated_tape_bytes(self) -> int:
        return int(self._native.allocated_bytes)

    @property
    def recomputation_factor(self) -> float:
        return float(self._native.recomputation_factor)

    @property
    def peak_temporary_bytes(self) -> int:
        return int(self._native.peak_temporary_bytes)

    @property
    def tape_context_limit_bytes(self) -> int:
        return int(self._native.tape_context_limit_bytes)

    @property
    def peak_runtime_managed_bytes(self) -> int:
        return int(self._native.peak_runtime_managed_bytes)

    @property
    def checkpoint_plan(self) -> Mapping[str, Any] | None:
        plan = self._native.checkpoint_plan
        return dict(plan) if plan is not None else None

    @property
    def pass_telemetry(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._native.pass_telemetry)


@dataclass(eq=False)
class ProgramAutodiffSpecialization:
    template: Any
    deployment: _ProgramDeployment
    pipeline: Any
    binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable, init=False, repr=False)

    def __post_init__(self) -> None:
        state._runtime_children.add(self)

    def _release_runtime_native(self) -> None:
        self.binding_cache.clear()
        self.pipeline = None

    def _loaded_pipeline(self) -> Any:
        if self.pipeline is None:
            self.pipeline = self.deployment.load()
        return self.pipeline

    @property
    def binding_telemetry(self) -> Mapping[str, int]:
        return self.binding_cache.telemetry

    def invoke(
        self,
        invocation: Any,
        *,
        checkpoint_memory_budget: int | None = None,
        checkpoint_policy: str = "",
    ) -> tuple[Any, ProgramNativePullback]:
        pipeline = self._loaded_pipeline()
        signature = pipeline.program_ad_signature
        expected_inputs = {row["path"] for row in signature["inputs"]}
        if set(invocation.inputs) != expected_inputs:
            raise RuntimeError(
                "Program invocation inputs do not match compiler ABI: "
                f"expected={sorted(expected_inputs)}, actual={sorted(invocation.inputs)}"
            )
        from ..program import flatten_program_outputs

        targets = flatten_program_outputs(invocation.outputs)
        program_bindings = dict(invocation.inputs)
        program_bindings.update(targets)
        with _bound_program_invocation(
            pipeline,
            self.binding_cache,
            invocation,
            targets,
            retain_borrows=True,
        ) as (builder, written, lease):
            native_outputs, native_pullback = pipeline.program_vjp_bound(
                builder,
                program_bindings,
                checkpoint_memory_budget=checkpoint_memory_budget,
                checkpoint_policy=checkpoint_policy,
            )
            for value in written:
                value._mark_device_dirty()
        derivative_groups = _pipeline_derivative_groups(pipeline)
        pullback = ProgramNativePullback(
            native_pullback,
            signature,
            targets,
            invocation.inputs,
            derivative_groups,
            lease,
        )
        expected_output_leaves = {row["path"] for row in signature["outputs"]}
        if native_outputs and set(native_outputs) != expected_output_leaves:
            raise RuntimeError(
                "Program outputs do not match compiler ABI: "
                f"expected={sorted(expected_output_leaves)}, actual={sorted(native_outputs)}"
            )
        return invocation.outputs, pullback


@dataclass(eq=False)
class ProgramSpecialization:
    template: Any
    deployment: _ProgramDeployment
    pipeline: Any
    binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable, init=False, repr=False)

    def __post_init__(self) -> None:
        state._runtime_children.add(self)

    def _release_runtime_native(self) -> None:
        self.binding_cache.clear()
        self.pipeline = None

    def _loaded_pipeline(self) -> Any:
        if self.pipeline is None:
            self.pipeline = self.deployment.load()
        return self.pipeline

    @property
    def binding_telemetry(self) -> Mapping[str, int]:
        return self.binding_cache.telemetry

    def invoke(self, invocation: Any) -> Any:
        from ..program import flatten_program_outputs

        pipeline = self._loaded_pipeline()
        targets = flatten_program_outputs(invocation.outputs)
        with _bound_program_invocation(pipeline, self.binding_cache, invocation, targets) as (
            builder,
            written,
            _,
        ):
            self.binding_cache.forward()
            for value in written:
                value._mark_device_dirty()
        return invocation.outputs


def _compile_program(parsed: Any) -> _ProgramDeployment:
    if state._native_runtime is None:
        raise RuntimeError(f"{state._architecture.name} Program execution requires the native runtime")
    native = state._native
    target = make_target_options(
        state._architecture.name,
        {"version": state._interactive_glsl_version()} if state._architecture in {state.opengl, state.opengles} else {},
    )
    compiler = native.Compiler()
    retained_programs: list[tuple[Any, Any]] = []
    plan = _compile_program_bundle_plan(
        parsed,
        pipeline_id=f"interactive/program-ad/{parsed.identity}",
        variant=(),
        target=target,
        compiler=compiler,
        native=native,
        native_target=_native_target(native, target.target),
        retained_programs=retained_programs if state._architecture == state.cpu else None,
    )
    directory = tempfile.TemporaryDirectory(prefix="vernon-program-ad-")
    root = Path(directory.name)
    descriptors = {
        stage.id: write_external_artifact(
            root,
            stage.artifact.data,
            stage.artifact.format,
            stage.stage,
            stage.artifact.filename,
        )
        for stage in plan.stages
    }
    canonical_program, artifact_system, stage_bindings = _canonical_deployment(plan, descriptors)
    return _ProgramDeployment(
        directory,
        canonical_json(dict(canonical_program)).encode(),
        canonical_json(dict(artifact_system)).encode(),
        stage_bindings,
        [(stage.metadata["symbol"], stage.entry, result) for stage, result in retained_programs],
    )


def compile_program(parsed: Any, template: Any) -> ProgramSpecialization:
    deployment = _compile_program(parsed)
    return ProgramSpecialization(template, deployment, deployment.load())


def compile_program_autodiff(parsed: Any, template: Any) -> ProgramAutodiffSpecialization:
    deployment = _compile_program(parsed)
    return ProgramAutodiffSpecialization(template, deployment, deployment.load())


__all__ = [
    "ProgramAutodiffSpecialization",
    "ProgramSpecialization",
    "ProgramNativePullback",
    "compile_program",
    "compile_program_autodiff",
]
