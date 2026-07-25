from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..pipeline_compile import (
    CompiledStage,
    PipelineCompileError,
    TargetOptions,
    build_bundle_plan,
    canonical_json,
    compiled_stage_from_program,
    inline_artifact_descriptor,
    materialize_bundle,
    serialize_bundle,
)
from .resources import TensorStorage, TensorView, Texture, _dispatch_borrow_scope


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


@dataclass(frozen=True)
class PrimitiveTopology:
    name: str
    vertices_per_primitive: int


triangles = PrimitiveTopology("triangles", 3)
lines = PrimitiveTopology("lines", 2)
points = PrimitiveTopology("points", 1)


@dataclass
class _CompiledPipeline:
    native: Any
    key: str
    bundle: bytes
    target_identity: str
    native_generation: int


@dataclass(frozen=True)
class _PipelineStageRequest:
    frontend: FrontendCompileResult
    module: str
    module_manifest: str
    entry: str
    target: TargetOptions
    native_target: Any


class Pipeline:
    """Callable specialized graphics pipeline."""

    _cache: ClassVar[dict[str, _CompiledPipeline]] = {}

    def __init__(self, *stages: Any, features: Iterable[str] = ()):
        kinds = tuple(getattr(stage, "__vernon_dsl__", (None,))[0] for stage in stages)
        if kinds != ("vertex", "fragment"):
            raise ValueError("pipeline stages must be (vertex, fragment)")
        feature_values = tuple(features)
        if any(not isinstance(value, str) or not value for value in feature_values):
            raise TypeError("pipeline features must be non-empty strings")
        self._features = tuple(sorted(set(feature_values)))
        self._stages = stages
        self._vertex, self._fragment = stages
        self._compiled: _CompiledPipeline | None = None
        self._compiled_generation = -1
        self.compile_count = 0
        _session_state()._runtime_children.add(self)

    def _stage_request(
        self,
        stage_value: Any,
        target: TargetOptions,
        native_target: Any,
    ) -> _PipelineStageRequest:
        function = getattr(stage_value, "_function", getattr(stage_value, "function", None))
        if function is None:
            raise RuntimeError("pipeline stage has no Python function")
        entry = function.__name__
        path = Path(inspect.getsourcefile(function) or "").resolve()
        frontend = Compiler().compile_request(FrontendCompileRequest(path, entry, self._features))
        return _PipelineStageRequest(
            frontend,
            f"python/{path.stem}",
            canonical_json(frontend.semantic_inputs),
            entry,
            target,
            native_target,
        )

    def _compile_pipeline_bundle(self) -> _CompiledPipeline:
        state = _session_state()
        if state._native is None or state._native_runtime is None:
            raise RuntimeError("graphics requires the native pipeline runtime")
        native_target, target_name = {
            state.vulkan: (state._native.Target.VULKAN, "vulkan"),
            state.directx: (state._native.Target.DIRECTX, "directx"),
            state.opengl: (state._native.Target.OPENGL, "opengl"),
            state.opengles: (state._native.Target.OPENGL_ES, "opengles"),
        }[state._architecture]
        options = TargetOptions(
            target_name,
            {"glsl_version": state._interactive_glsl_version()}
            if state._architecture in {state.opengl, state.opengles}
            else {},
        )
        requests = [self._stage_request(stage, options, native_target) for stage in self._stages]
        compiled_stages: list[CompiledStage] = []
        compiler = state._native.Compiler()
        for request in requests:
            program = compiler.compile_program_result(
                request.frontend.mlir,
                request.native_target,
                **request.target.native_options,
            )
            try:
                compiled_stages.append(
                    compiled_stage_from_program(
                        program,
                        module=request.module,
                        module_manifest=request.module_manifest,
                        entry=request.entry,
                        target=request.target,
                    )
                )
            except PipelineCompileError as error:
                raise RuntimeError(str(error)) from None
        pipeline_id = (
            "interactive/"
            + hashlib.sha256(
                canonical_json(
                    {
                        "stages": [stage.id for stage in compiled_stages],
                        "features": self._features,
                        "target": target_name,
                        "target_options": dict(options.options),
                    }
                ).encode()
            ).hexdigest()
        )
        try:
            plan = build_bundle_plan(
                pipeline_id,
                options,
                self._features,
                [(self._features, {stage.stage: stage for stage in compiled_stages})],
            )
            bundle = materialize_bundle(
                plan,
                {stage.id: inline_artifact_descriptor(stage.artifact) for stage in compiled_stages},
            )
        except PipelineCompileError as error:
            raise TypeError(str(error)) from None
        bundle_bytes = serialize_bundle(bundle)
        key = hashlib.sha256(bundle_bytes).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            cached = _CompiledPipeline(
                state._native_runtime.load_pipeline(bundle_bytes, list(self._features)),
                key,
                bundle_bytes,
                canonical_json({"target": target_name, "target_options": dict(options.options)}),
                state._runtime_generation,
            )
            self._cache[key] = cached
        elif cached.native_generation != state._runtime_generation:
            cached.native = state._native_runtime.load_pipeline(cached.bundle, list(self._features))
            cached.native_generation = state._runtime_generation
        self._compiled = cached
        self._compiled_generation = state._runtime_generation
        if self.compile_count == 0:
            self.compile_count = 1
        return cached

    def _compile(self, call_arguments: Mapping[str, Any] | None = None) -> _CompiledPipeline:
        state = _session_state()
        if self._compiled is not None and self._compiled_generation == state._runtime_generation:
            return self._compiled
        if state._architecture in {state.vulkan, state.directx, state.opengl, state.opengles}:
            target_identity = canonical_json(
                {
                    "target": state._architecture.name,
                    "target_options": (
                        {"glsl_version": state._interactive_glsl_version()}
                        if state._architecture in {state.opengl, state.opengles}
                        else {}
                    ),
                }
            )
            if self._compiled is not None and self._compiled.target_identity == target_identity:
                assert state._native_runtime is not None
                self._compiled.native = state._native_runtime.load_pipeline(
                    self._compiled.bundle,
                    list(self._features),
                )
                self._compiled.native_generation = state._runtime_generation
                self._compiled_generation = state._runtime_generation
                return self._compiled
            return self._compile_pipeline_bundle()
        if state._architecture == state.cpu:
            raise RuntimeError(
                "CPU graphics pipelines require a software rasterizer, which "
                "Vernon does not provide; CPU supports compute kernels only"
            )
        raise RuntimeError("unsupported graphics backend")

    def _invoke_direct(self, arguments: dict[str, Any]) -> None:
        state = _session_state()
        target = arguments.pop("target", None)
        targets = arguments.pop("targets", None)
        indices = arguments.pop("indices", None)
        topology = arguments.pop("topology", triangles)
        if target is not None and targets is not None:
            raise TypeError("pipeline call cannot use both target and targets")
        compiled = self._compile(arguments)
        parameters = tuple(compiled.native.parameters)
        expected = {parameter.name for parameter in parameters}
        missing = expected - set(arguments)
        if missing:
            raise TypeError(f"missing pipeline argument(s): {', '.join(sorted(missing))}")
        unexpected = set(arguments) - expected
        if unexpected:
            raise TypeError(f"unexpected pipeline argument(s): {', '.join(sorted(unexpected))}")
        access_names = {
            state._native.ACCESS_READ: "read",
            state._native.ACCESS_WRITE: "write",
            state._native.ACCESS_READ_WRITE: "read_write",
        }
        dispatch_borrows = [
            (parameter.name, arguments[parameter.name], access_names[parameter.access])
            for parameter in parameters
            if parameter.kind == state._native.PIPELINE_TENSOR
            and isinstance(arguments[parameter.name], (TensorStorage, TensorView))
        ]
        builder = compiled.native.invocation_builder()
        dtype_codes = {
            np.dtype(np.bool_): state._native.DATA_BOOL,
            np.dtype(np.int32): state._native.DATA_I32,
            np.dtype(np.uint32): state._native.DATA_U32,
            np.dtype(np.float16): state._native.DATA_F16,
            np.dtype(np.float32): state._native.DATA_F32,
            np.dtype(np.float64): state._native.DATA_F64,
        }
        for parameter in parameters:
            value = arguments[parameter.name]
            if parameter.kind == state._native.PIPELINE_TEXTURE:
                if not isinstance(value, Texture):
                    raise TypeError(f"texture {parameter.name!r} must be a Texture")
                builder.rhi_texture(parameter.name, value._resident_texture())
                continue
            if parameter.kind == state._native.PIPELINE_SAMPLER:
                builder.rhi_sampler(parameter.name, value)
                continue
            if parameter.kind != state._native.PIPELINE_TENSOR:
                raise TypeError(f"pipeline parameter {parameter.name!r} has unsupported kind")
            if not isinstance(value, (TensorStorage, TensorView)):
                scalar = np.asarray(value)
                if scalar.dtype.kind == "f":
                    scalar = np.asarray(value, dtype=np.float32)
                elif scalar.dtype.kind == "u":
                    scalar = np.asarray(value, dtype=np.uint32)
                else:
                    scalar = np.asarray(value, dtype=np.int32)
                builder.host_tensor(parameter.name, scalar)
                continue
            if tuple(value.shape) == tuple(parameter.shape):
                builder.host_tensor(parameter.name, value._borrowed_array())
                continue
            dtype = dtype_codes.get(value.dtype)
            if dtype is None:
                raise TypeError(f"pipeline does not support dtype {value.dtype}")
            layout = value.layout
            if state._rhi_host is None:
                raise RuntimeError("device pipeline Tensor arguments require a GPU RHI host")
            builder.rhi_tensor(
                parameter.name,
                value._resident_buffer(),
                dtype,
                parameter.access,
                list(value.shape),
                list(layout.byte_strides),
                layout.byte_offset,
            )
        rendered_targets: list[Texture] = []
        outputs = tuple(compiled.native.outputs)
        named_outputs = {output.name: output for output in outputs if output.name != f"output_{output.location}"}
        if targets is not None:
            if not isinstance(targets, Mapping):
                raise TypeError("targets must be a mapping of output names")
            if len(named_outputs) != len(outputs):
                raise TypeError("targets= requires named fragment struct outputs")
            if set(targets) != set(named_outputs):
                raise ValueError("target keys must exactly match fragment output names")
            for name, output in sorted(named_outputs.items(), key=lambda item: item[1].location):
                texture = targets[name]
                if not isinstance(texture, Texture):
                    raise TypeError(f"target {name!r} must be a Texture")
                builder.rhi_color_attachment(output.location, texture._resident_texture())
                rendered_targets.append(texture)
        else:
            if (
                not isinstance(target, Texture)
                or len(outputs) != 1
                or outputs[0].location != 0
                or outputs[0].name != "output_0"
            ):
                raise TypeError("target=Texture requires one unnamed fragment output at location zero")
            builder.rhi_color_attachment(0, target._resident_texture())
            rendered_targets.append(target)
        if indices is not None:
            if (
                not isinstance(indices, TensorStorage)
                or indices.dtype != np.dtype(np.uint32)
                or len(indices.shape) != 1
                or not indices.shape[0]
            ):
                raise TypeError("indices must be a non-empty rank-one u32 TensorStorage")
            builder.rhi_index_binding(indices._resident_buffer(), indices.shape[0])
            dispatch_borrows.append(("indices", indices, "read"))
        native_topology = {
            triangles: state._native.TOPOLOGY_TRIANGLE_LIST,
            lines: state._native.TOPOLOGY_LINE_LIST,
            points: state._native.TOPOLOGY_POINT_LIST,
        }.get(topology)
        if native_topology is None:
            raise TypeError("topology must be triangles, lines, or points")
        builder.topology(native_topology)
        assert state._native_runtime is not None
        with _dispatch_borrow_scope(dispatch_borrows):
            compiled.native.invoke(builder)
        for texture in rendered_targets:
            texture._mark_device_dirty()

    def __call__(self, **arguments: Any) -> None:
        self._invoke_direct(arguments)


def pipeline(*stages: Any, features: Iterable[str] = ()) -> Pipeline:
    return Pipeline(*stages, features=features)


__all__ = ["Pipeline", "PrimitiveTopology", "lines", "pipeline", "points", "triangles"]
