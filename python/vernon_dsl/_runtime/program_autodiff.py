from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .._shader_assets.artifact_io import write_external_artifact
from .._shader_assets.cooking import (
    _canonical_deployment,
    _compile_program_bundle_plan,
    _native_target,
)
from ..bundle import canonical_json, make_target_options
from ..storage import TensorStorage
from . import session as state
from .autodiff import _pipeline_derivative_groups


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


class ProgramNativePullback:
    def __init__(
        self,
        native: Any,
        signature: Mapping[str, Any],
        outputs: Mapping[str, Any],
        derivative_groups: tuple[Any, ...],
    ) -> None:
        self._native = native
        self._signature = signature
        self._outputs = dict(outputs)
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
        gradients = dict(
            self._native.apply_grouped(
                supplied,
                self._gradient_groups,
                self._cotangent_groups,
                (),
                False,
            )
        )
        return {
            path: (value if isinstance(value, TensorStorage) else TensorStorage.from_numpy(np.asarray(value)))
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


@dataclass
class ProgramAutodiffSpecialization:
    template: Any
    pipeline: Any
    directory: tempfile.TemporaryDirectory[str]

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
        native_inputs = {
            path: (value._native_host_array() if hasattr(value, "_native_host_array") else value)
            for path, value in invocation.inputs.items()
        }
        from ..program import flatten_program_outputs

        targets = flatten_program_outputs(invocation.outputs)
        program_bindings = dict(invocation.inputs)
        program_bindings.update(targets)
        native_outputs, native_pullback = self.pipeline.program_vjp(
            native_inputs,
            program_bindings,
            checkpoint_memory_budget=checkpoint_memory_budget,
            checkpoint_policy=checkpoint_policy,
        )
        derivative_groups = _pipeline_derivative_groups(self.pipeline)
        output_groups = tuple(group for group in derivative_groups if group.role == "cotangent")
        expected_output_leaves = {leaf for group in output_groups for leaf in group.leaf_paths}
        if set(native_outputs) != expected_output_leaves:
            raise RuntimeError(
                "Program outputs do not match compiler ABI: "
                f"expected={sorted(expected_output_leaves)}, actual={sorted(native_outputs)}"
            )
        for group in output_groups:
            path = group.declared_path
            target = targets[path]
            for leaf_path in group.leaf_paths:
                source = np.asarray(native_outputs[leaf_path])
                destination = _program_storage_leaf(target, path, leaf_path)
                if isinstance(destination, TensorStorage) or hasattr(destination, "copy_from_numpy"):
                    destination.copy_from_numpy(source)
                else:
                    np.copyto(destination._native_host_array(), source)
        return invocation.outputs, ProgramNativePullback(
            native_pullback,
            signature,
            targets,
            derivative_groups,
        )


@dataclass
class ProgramSpecialization:
    template: Any
    pipeline: Any
    directory: tempfile.TemporaryDirectory[str]

    def invoke(self, invocation: Any) -> Any:
        from ..program import flatten_program_outputs

        native_inputs = {
            path: (value._native_host_array() if hasattr(value, "_native_host_array") else value)
            for path, value in invocation.inputs.items()
        }
        targets = flatten_program_outputs(invocation.outputs)
        bindings = dict(invocation.inputs)
        bindings.update(targets)
        self.pipeline.program_forward(native_inputs, bindings)
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
