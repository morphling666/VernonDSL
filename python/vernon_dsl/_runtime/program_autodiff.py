from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    written: list[Any] = []
    lease = _DispatchBorrowLease(borrows)
    succeeded = False
    try:
        with cache.invocation(pipeline) as builder:
            for parameter, _, path, value in resolved:
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
                        annotation=invocation.input_annotations.get(path),
                    )
                if (
                    state._architecture != state.cpu
                    and parameter.access != state._native.ACCESS_READ
                    and hasattr(value, "_mark_device_dirty")
                ):
                    written.append(value)
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


@dataclass
class ProgramAutodiffSpecialization:
    template: Any
    pipeline: Any
    directory: tempfile.TemporaryDirectory[str]
    binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable, init=False, repr=False)

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
        signature = self.pipeline.program_ad_signature
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
            self.pipeline,
            self.binding_cache,
            invocation,
            targets,
            retain_borrows=True,
        ) as (builder, written, lease):
            native_outputs, native_pullback = self.pipeline.program_vjp_bound(
                builder,
                program_bindings,
                checkpoint_memory_budget=checkpoint_memory_budget,
                checkpoint_policy=checkpoint_policy,
            )
            for value in written:
                value._mark_device_dirty()
        derivative_groups = _pipeline_derivative_groups(self.pipeline)
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


@dataclass
class ProgramSpecialization:
    template: Any
    pipeline: Any
    directory: tempfile.TemporaryDirectory[str]
    binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable, init=False, repr=False)

    @property
    def binding_telemetry(self) -> Mapping[str, int]:
        return self.binding_cache.telemetry

    def invoke(self, invocation: Any) -> Any:
        from ..program import flatten_program_outputs

        targets = flatten_program_outputs(invocation.outputs)
        with _bound_program_invocation(self.pipeline, self.binding_cache, invocation, targets) as (
            builder,
            written,
            _,
        ):
            self.pipeline.program_forward_bound(builder)
            for value in written:
                value._mark_device_dirty()
        return invocation.outputs


def _compile_program(parsed: Any) -> tuple[Any, tempfile.TemporaryDirectory[str]]:
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
    pipeline = state._native_runtime.load_canonical_program(
        canonical_json(dict(canonical_program)).encode(),
        canonical_json(dict(artifact_system)).encode(),
        directory.name,
        stage_bindings,
        [(stage.metadata["symbol"], stage.entry, result) for stage, result in retained_programs],
    )
    return pipeline, directory


def compile_program(parsed: Any, template: Any) -> ProgramSpecialization:
    pipeline, directory = _compile_program(parsed)
    return ProgramSpecialization(template, pipeline, directory)


def compile_program_autodiff(parsed: Any, template: Any) -> ProgramAutodiffSpecialization:
    pipeline, directory = _compile_program(parsed)
    return ProgramAutodiffSpecialization(template, pipeline, directory)


__all__ = [
    "ProgramAutodiffSpecialization",
    "ProgramSpecialization",
    "ProgramNativePullback",
    "compile_program",
    "compile_program_autodiff",
]
