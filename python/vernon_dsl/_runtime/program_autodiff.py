from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .._shader_assets.artifact_io import write_external_artifact
from .._shader_assets.cooking import _compile_program_bundle_plan, _native_target
from ..bundle import CpuTargetOptions, materialize_bundle, serialize_bundle
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
            supplied: Any = np.ones(np.asarray(output).shape, dtype=np.asarray(output).dtype)
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
            path: value if isinstance(value, TensorStorage) else TensorStorage.from_numpy(np.asarray(value))
            for path, value in gradients.items()
        }

    @property
    def logical_residual_bytes(self) -> int:
        return int(self._native.logical_residual_bytes)

    @property
    def resident_tape_bytes(self) -> int:
        return int(self._native.resident_bytes)

    @property
    def allocated_tape_bytes(self) -> int:
        return int(self._native.allocated_bytes)

    @property
    def peak_temporary_bytes(self) -> int:
        return int(self._native.peak_temporary_bytes)


@dataclass
class ProgramAutodiffSpecialization:
    template: Any
    pipeline: Any
    directory: tempfile.TemporaryDirectory[str]

    def invoke(self, invocation: Any) -> tuple[Any, ProgramNativePullback]:
        signature = self.pipeline.program_ad_signature
        expected_inputs = {row["path"] for row in signature["inputs"]}
        if set(invocation.inputs) != expected_inputs:
            raise RuntimeError(
                "Program invocation inputs do not match compiler ABI: "
                f"expected={sorted(expected_inputs)}, actual={sorted(invocation.inputs)}"
            )
        native_inputs = {
            path: value._native_host_array() if hasattr(value, "_native_host_array") else value
            for path, value in invocation.inputs.items()
        }
        from ..program import flatten_program_outputs

        targets = flatten_program_outputs(invocation.outputs)
        program_bindings = dict(invocation.inputs)
        program_bindings.update(targets)
        native_outputs, native_pullback = self.pipeline.program_vjp(native_inputs, program_bindings)
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


def compile_program_autodiff(parsed: Any, template: Any) -> ProgramAutodiffSpecialization:
    if state._architecture != state.cpu or state._native_runtime is None:
        raise RuntimeError("interactive Program autodiff currently requires the CPU runtime")
    native = state._native
    target = CpuTargetOptions()
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
        retained_programs=retained_programs,
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
    pipeline = state._native_runtime.load_program_pipeline_asset(
        serialize_bundle(materialize_bundle(plan, descriptors)),
        directory.name,
        [],
        [(stage.metadata["symbol"], stage.entry, result) for stage, result in retained_programs],
    )
    return ProgramAutodiffSpecialization(template, pipeline, directory)


__all__ = [
    "ProgramAutodiffSpecialization",
    "ProgramNativePullback",
    "compile_program_autodiff",
]
