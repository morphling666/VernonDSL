from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .._program_assets.artifact_io import write_external_artifact
from .._program_assets.compile_orchestration import (
    _compile_program_variant,
    _native_target,
)
from ..bundle import build_program_manifest, build_program_plan, canonical_json, make_target_options
from ..storage import TensorStorage, TensorView
from .autodiff import _program_derivative_groups
from .binding import (
    _argument_preparation,
    _DispatchBorrowLease,
    _sampler_preparation,
)
from .sampler import SamplerState
from .session import (
    _execution_context,
    _interactive_target_options,
    _InvocationContext,
    _session_state,
    _use_invocation_context,
    cpu,
)
from .tensor import RawBuffer, _owner_controls, _PreservedOwners
from .texture import _TextureResource


@dataclass
class _ProgramDeployment:
    directory: tempfile.TemporaryDirectory[str]
    manifest: bytes
    compiled_stages: list[tuple[str, str, Any]]
    # (entry, reflection JSON, artifact) of each GPU Stage, which a linked schedule loads by itself.
    device_stages: list[tuple[str, str, bytes]] = field(default_factory=list)
    # The variant's Program JSON and, per logical Stage, (name, entry, CPU program, GPU reflection JSON, GPU artifact),
    # from which the Program links as a schedule (specs/runtime/architecture_v2.md section 16.1).
    program_json: str = ""
    linked_stages: list[tuple[str, str, Any, str | None, bytes | None]] = field(default_factory=list)

    def load(self) -> Any:
        state = _session_state()
        return state.native_runtime.load_in_memory_program(
            self.manifest,
            self.directory.name,
            self.compiled_stages,
        )


def _configure_graphics_control(builder: Any, invocation: Any, node_id: int, context: Any) -> None:
    """Writes the attachments, draw and dynamic state of graphics node `node_id` into a control builder."""
    from ..render import ColorBlendState, LoadOperation, StoreOperation, lines, points, triangles

    state = context.session
    load_values = {
        LoadOperation.CLEAR: state.native.ATTACHMENT_CLEAR,
        LoadOperation.PRESERVE: state.native.ATTACHMENT_PRESERVE,
        LoadOperation.DISCARD: state.native.ATTACHMENT_DISCARD,
    }
    store_values = {
        StoreOperation.PRESERVE: state.native.ATTACHMENT_STORE,
        StoreOperation.DISCARD: state.native.ATTACHMENT_DONT_CARE,
    }
    operation = next(operation for operation in invocation.graph.operations if operation.id == node_id)
    controls = invocation.graphics_controls[node_id]
    render_pass = controls["render_pass"][1]
    draw = controls["draw"][1]
    dynamic = controls["dynamic_state"][1]
    colors = tuple(render_pass.target._color_attachments())
    color_operations = dict(render_pass.colors)
    for location, texture in colors:
        attachment = color_operations[location]
        component = int if texture.format.integer else float
        clear_value = (
            [component(value) for value in attachment.clear_value]
            if attachment.load is LoadOperation.CLEAR
            else [component(0)] * 4
        )
        builder.color_attachment(
            location,
            texture._resident_view(context),
            load_values[attachment.load],
            store_values[attachment.store],
            clear_value,
        )
    depth_texture = render_pass.target._depth_attachment()
    if depth_texture is not None:
        if render_pass.depth is None:
            raise ValueError("depth attachment operations are required for a depth RenderTarget")
        attachment = render_pass.depth
        clear_depth = float(attachment.clear_value) if attachment.load is LoadOperation.CLEAR else 1.0
        builder.depth_attachment(
            depth_texture._resident_view(context),
            load_values[attachment.load],
            store_values[attachment.store],
            clear_depth,
        )
    if draw is not None and draw.index_buffer is not None:
        index_view = draw.index_buffer.view
        builder.index_binding(
            index_view._resident_buffer(context),
            draw.index_buffer.count,
            index_view.layout.byte_offset,
        )
    topology = {
        triangles: state.native.TOPOLOGY_TRIANGLE_LIST,
        lines: state.native.TOPOLOGY_LINE_LIST,
        points: state.native.TOPOLOGY_POINT_LIST,
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
        color_blends=tuple((location, configured_blends.get(location, ColorBlendState())) for location, _ in colors),
    )
    builder.graphics_state(graphics_state)


def _program_storage_leaf(value: Any, root: str, leaf_path: str) -> Any:
    if leaf_path == root:
        return value
    projection = value
    for component in leaf_path[len(root) + 1 :].split("."):
        projection = projection[int(component)] if component.isdigit() else projection.field(component)
    return projection


def _program_boundary_values(
    executable: Any, invocation: Any, targets: Mapping[str, Any], state: Any
) -> tuple[list[tuple[Any, Mapping[str, Any], str, Any]], list[tuple[Any, Any, str]]]:
    """Each parameter of the executable with its boundary slot, path and value, and the resource borrows of a call:
    each resource argument with its declared access, and the attachments and index buffers its graphics nodes use."""
    parameters = tuple(executable.parameters)
    parameters_by_slot = {parameter.slot: parameter for parameter in parameters}
    if len(parameters_by_slot) != len(parameters):
        raise RuntimeError("Program compiler ABI contains duplicate public slots")
    boundary_slots = tuple(
        slot for slot in executable.program_abi["boundary_slots"] if slot["role"] in {"input", "output"}
    )
    boundary_slots_by_slot = {slot["slot"]: slot for slot in boundary_slots}
    if not set(parameters_by_slot) <= set(boundary_slots_by_slot):
        raise RuntimeError(
            "Program invocation parameters do not match compiler-emitted ProgramABI: "
            f"parameters={sorted(parameters_by_slot)}, "
            f"boundaries={sorted(boundary_slots_by_slot)}"
        )

    def binding(slot: Mapping[str, Any]) -> tuple[str, Any]:
        path = slot["path"]
        values = invocation.inputs if slot["role"] == "input" else targets
        if path not in values:
            raise RuntimeError(f"Program invocation is missing boundary value {path!r}")
        return path, values[path]

    access_names = {
        state.native.ACCESS_READ: "read",
        state.native.ACCESS_WRITE: "write",
        state.native.ACCESS_READ_WRITE: "read_write",
    }
    resolved = [
        (parameter, boundary_slots_by_slot[parameter.slot], *binding(boundary_slots_by_slot[parameter.slot]))
        for parameter in parameters
    ]
    borrows: list[tuple[Any, TensorStorage | RawBuffer | TensorView | _TextureResource, str]] = [
        (parameter.slot, value, access_names[parameter.access])
        for parameter, _, path, value in resolved
        if isinstance(value, (TensorStorage, TensorView, _TextureResource))
    ]
    for controls in invocation.graphics_controls.values():
        render_slot = int(controls["render_pass"][0])
        render_pass = controls["render_pass"][1]
        borrows.extend(
            (("render_pass", render_slot), texture, "write")
            for location, texture in render_pass.target._color_attachments()
        )
        depth = render_pass.target._depth_attachment()
        if depth is not None:
            borrows.append((("render_pass", render_slot), depth, "write"))
        draw = controls["draw"][1]
        if draw is not None and draw.index_buffer is not None:
            borrows.append(("index_buffer", draw.index_buffer.view, "read"))
    return resolved, borrows


def _bind_schedule_arguments(builder: Any, context: Any, invocation: Any, resolved: Any) -> None:
    for parameter, slot, path, value in resolved:
        if parameter.kind == context.session.native.PROGRAM_SAMPLER:
            if not isinstance(value, SamplerState):
                raise TypeError(f"sampler {parameter.name!r} must be a SamplerState")
            preparation = _sampler_preparation(builder, context, parameter, value)
        else:
            preparation = _argument_preparation(
                builder,
                context,
                parameter,
                value,
                host_value=slot.get("category") == "value",
                annotation=invocation.input_annotations.get(path),
            )
        builder.argument(preparation())


def _call_program_vjp(
    vjp: Any, executable: Any, invocation: Any, targets: Mapping[str, Any], bindings: Mapping[str, Any]
) -> Any:
    """Runs the forward call of the executable's linked VJP and returns its native pullback."""
    context = _execution_context()
    resolved, borrows = _program_boundary_values(executable, invocation, targets, context.session)
    lease = _DispatchBorrowLease(borrows, context)
    try:
        builder = vjp.builder()
        _bind_schedule_arguments(builder, context, invocation, resolved)
        try:
            pullback, point = vjp.forward(builder, dict(bindings))
        except BaseException:
            lease.resolve_mutations({})
            raise
        lease.resolve_linked_call(vjp.timeline, point)
        return pullback
    finally:
        lease.release()


def _call_program_schedule(schedule: Any, executable: Any, invocation: Any, targets: Mapping[str, Any]) -> None:
    """Runs one call of the executable's linked schedule. A write the call makes is published as its accesses
    declare, once the call is submitted."""
    context = _execution_context()
    resolved, borrows = _program_boundary_values(executable, invocation, targets, context.session)
    lease = _DispatchBorrowLease(borrows, context)
    try:
        builder = schedule.builder()
        _bind_schedule_arguments(builder, context, invocation, resolved)
        passes = []
        for node_id in invocation.graphics_controls:
            control = schedule.builder()
            _configure_graphics_control(control, invocation, node_id, context)
            passes.append(control)
        try:
            point = schedule.call(builder, passes)
        except BaseException:
            lease.resolve_mutations({})
            raise
        lease.resolve_linked_call(schedule.timeline, point)
    finally:
        lease.release()


def _admit_pullback_accesses(requests: Any, context: _InvocationContext) -> _DispatchBorrowLease:
    return _DispatchBorrowLease(list(requests), context)


class ProgramNativePullback:
    def __init__(
        self,
        native: Any,
        signature: Mapping[str, Any],
        inputs: Mapping[str, Any],
        derivative_groups: tuple[Any, ...],
        context: _InvocationContext,
    ) -> None:
        self._native = native
        self._signature = signature
        self._inputs = dict(inputs)
        self._context = context
        # The residuals point into the memory of the forward call's inputs, which a later write leaves to the pullback.
        self._preserved = _PreservedOwners(_owner_controls(inputs.values()))
        self._storage_gradients = {
            path for path, value in inputs.items() if isinstance(value, (TensorStorage, TensorView))
        }
        self._gradient_groups = tuple(group for group in derivative_groups if group.role == "gradient")
        self._cotangent_groups = tuple(group for group in derivative_groups if group.role == "cotangent")

    def __call__(self, cotangents: Any = None) -> dict[str, Any]:
        return self.apply_with_carrier(cotangents, ())

    def apply_with_carrier(
        self,
        cotangents: Any,
        carrier_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        with _use_invocation_context(self._context):
            return self._apply_with_carrier(cotangents, carrier_shape)

    def _apply_with_carrier(
        self,
        cotangents: Any,
        carrier_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        paths = tuple(row["path"] for row in self._signature["cotangents"])
        if cotangents is None:
            if len(paths) != 1:
                raise ValueError("implicit Program cotangent requires exactly one output")
            supplied: Any = None
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
                        leaf: _program_storage_leaf(value, root, leaf).to_numpy() for leaf in group.leaf_paths
                    }
            supplied = grouped_cotangents
        context = _execution_context()
        gradients = dict(
            self._native.apply_grouped(
                supplied,
                self._gradient_groups,
                self._cotangent_groups,
                carrier_shape,
                context,
                lambda requests: _admit_pullback_accesses(requests, context),
            )
        )
        return {
            path: (
                value
                if isinstance(value, TensorStorage) or path not in self._storage_gradients
                else TensorStorage.from_numpy(np.asarray(value))
            )
            for path, value in gradients.items()
        }

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
    executable: Any
    # The executable's forward and backward graphs linked as a VJP per checkpoint memory budget and policy. An
    # executable that does not link raises the linker's ValueError on every call.
    _vjps: dict[tuple[int | None, str], Any] = field(default_factory=dict, init=False, repr=False)
    _vjp_executable: Any = field(default=None, init=False, repr=False)

    def _loaded_executable(self) -> Any:
        return self.executable

    def _linked_vjp(self, executable: Any, budget: int | None = None, policy: str = "") -> Any:
        if self._vjp_executable is not executable:
            self._vjp_executable = executable
            self._vjps.clear()
        key = (budget, policy)
        if key not in self._vjps:
            state = _session_state()
            self._vjps[key] = state.native._ProgramVjp(
                executable,
                checkpoint_memory_budget=budget,
                checkpoint_policy=policy,
                storage_type=TensorStorage,
                view_type=TensorView,
                residency_key=None if state.arch == cpu else state.identity,
            )
        return self._vjps[key]

    def invoke(
        self,
        invocation: Any,
        *,
        checkpoint_memory_budget: int | None = None,
        checkpoint_policy: str = "",
    ) -> tuple[Any, ProgramNativePullback]:
        context = _execution_context()
        executable = self._loaded_executable()
        signature = executable.program_ad_signature
        expected_inputs = {row["path"] for row in signature["inputs"]}
        actual_inputs = set(invocation.inputs)
        if actual_inputs != expected_inputs:
            raise RuntimeError(
                "Program invocation inputs do not match compiler ABI: "
                f"expected={sorted(expected_inputs)}, actual={sorted(actual_inputs)}"
            )
        from ..program import flatten_program_outputs

        targets = flatten_program_outputs(invocation.outputs)
        program_bindings = dict(invocation.inputs)
        program_bindings.update(targets)
        vjp = self._linked_vjp(executable, checkpoint_memory_budget, checkpoint_policy)
        native_pullback = _call_program_vjp(vjp, executable, invocation, targets, program_bindings)
        pullback = ProgramNativePullback(
            native_pullback,
            signature,
            invocation.inputs,
            _program_derivative_groups(executable),
            context,
        )
        return invocation.outputs, pullback


@dataclass(eq=False)
class ProgramSpecialization:
    template: Any
    deployment: _ProgramDeployment
    executable: Any
    # The executable's forward graph linked as a schedule. An executable that does not link raises the linker's
    # ValueError on every call.
    _schedule: Any = field(default=None, init=False, repr=False)
    _schedule_executable: Any = field(default=None, init=False, repr=False)

    def _loaded_executable(self) -> Any:
        return self.executable

    def _linked_schedule(self, executable: Any) -> Any:
        if self._schedule_executable is not executable:
            self._schedule = _session_state().native._ProgramSchedule(executable)
            self._schedule_executable = executable
        return self._schedule

    def invoke(self, invocation: Any) -> Any:
        from ..program import flatten_program_outputs

        executable = self._loaded_executable()
        targets = flatten_program_outputs(invocation.outputs)
        _call_program_schedule(self._linked_schedule(executable), executable, invocation, targets)
        return invocation.outputs


def _compile_program(parsed: Any) -> _ProgramDeployment:
    state = _session_state()
    native = state.native
    target = make_target_options(state.arch.name, _interactive_target_options(state))
    compiler = native.Compiler()
    retained_programs: list[tuple[Any, Any]] = []
    reflected_target, variant = _compile_program_variant(
        parsed,
        program_id=f"interactive/program-ad/{parsed.identity}",
        variant=(),
        target=target,
        compiler=compiler,
        native=native,
        native_target=_native_target(native, target.target),
        retained_programs=retained_programs if state.arch == cpu else None,
    )
    plan = build_program_plan(
        f"interactive/program-ad/{parsed.identity}",
        reflected_target,
        (variant,),
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
        for stage in plan.compiled_stages
    }
    manifest = build_program_manifest(plan, descriptors)
    compiled_by_id = {stage.id: stage for stage in plan.compiled_stages}
    results_by_id = {stage.id: result for stage, result in retained_programs}
    linked_stages: list[tuple[str, str, Any, str | None, bytes | None]] = []
    for logical, implementation in plan.variants[0].stage_implementations.items():
        stage = compiled_by_id[implementation]
        if stage.stage != "compute":
            linked_stages = []
            break
        if state.arch == cpu:
            linked_stages.append((logical, stage.entry, results_by_id.get(implementation), None, None))
        else:
            linked_stages.append(
                (logical, stage.entry, None, canonical_json(dict(stage.reflection)), stage.artifact.data)
            )
    return _ProgramDeployment(
        directory,
        canonical_json(manifest).encode(),
        [(stage.metadata["symbol"], stage.entry, result) for stage, result in retained_programs],
        []
        if state.arch == cpu
        else [
            (stage.entry, canonical_json(dict(stage.reflection)), stage.artifact.data)
            for stage in plan.compiled_stages
            if stage.stage == "compute"
        ],
        canonical_json(dict(plan.variants[0].program)),
        linked_stages,
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
