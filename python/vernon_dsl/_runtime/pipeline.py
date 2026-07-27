from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..bundle import (
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
from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from .execution_graph import (
    ExecutionGraph,
    ExecutionResources,
    GraphicsEncoder,
    LoadOperation,
    PipelineInvocation,
    RenderPass,
    StoreOperation,
)
from .resources import (
    RenderTarget,
    TensorStorage,
    TensorView,
    Texture,
    _bind_native_argument,
    _dispatch_borrow_scope,
)


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


@dataclass(frozen=True)
class PrimitiveTopology:
    name: str
    vertices_per_primitive: int


triangles = PrimitiveTopology("triangles", 3)
lines = PrimitiveTopology("lines", 2)
points = PrimitiveTopology("points", 1)


class _ImmediateRenderPass(RenderPass):
    def __init__(self, name: str, target: RenderTarget, invocation: PipelineInvocation):
        super().__init__(name)
        self._immediate_target = target
        self._invocation = invocation

    def declare(self) -> None:
        self.attachments(self._immediate_target)
        self._invocation.declare(self)

    def execute(self, encoder: GraphicsEncoder, resources: ExecutionResources) -> None:
        self._invocation.encode(encoder, resources)


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

    def _invoke_direct(self, arguments: dict[str, Any], encoder: GraphicsEncoder) -> None:
        state = _session_state()
        target = encoder.target
        indices = arguments.pop("indices", None)
        topology = arguments.pop("topology", triangles)
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
            leaves = tuple(parameter.element_leaves)
            host_value = isinstance(value, (TensorStorage, TensorView)) and (
                tuple(value.shape) == tuple(parameter.shape) and len(leaves) == 1 and leaves[0][1:] == (1, 0)
            )
            _bind_native_argument(builder, parameter, value, host_value=host_value)
        outputs = tuple(compiled.native.outputs)
        color_attachments = target._color_attachments()
        output_locations = {output.location for output in outputs}
        attachment_locations = {location for location, _ in color_attachments}
        if attachment_locations != output_locations:
            raise ValueError("RenderTarget color locations must exactly match fragment output locations")
        operations = encoder.attachment_operations
        first_in_scope = operations["first_in_scope"]
        last_in_scope = operations["last_in_scope"]
        load_values = {
            LoadOperation.CLEAR: state._native.ATTACHMENT_CLEAR,
            LoadOperation.PRESERVE: state._native.ATTACHMENT_PRESERVE,
            LoadOperation.DISCARD: state._native.ATTACHMENT_DISCARD,
        }
        store_values = {
            StoreOperation.PRESERVE: state._native.ATTACHMENT_STORE,
            StoreOperation.DISCARD: state._native.ATTACHMENT_DONT_CARE,
        }
        for location, texture in color_attachments:
            attachment = operations["colors"][location]
            load = (
                attachment.load if first_in_scope or attachment.load is LoadOperation.CLEAR else LoadOperation.PRESERVE
            )
            store = attachment.store if last_in_scope else StoreOperation.PRESERVE
            builder.rhi_color_attachment(
                location,
                texture._resident_texture(),
                load_values[load],
                store_values[store],
                list(attachment.clear_value),
            )
        depth_attachment = target._resident_depth_attachment()
        if depth_attachment is not None:
            attachment = operations["depth"]
            assert attachment is not None
            load = (
                attachment.depth_load
                if first_in_scope or attachment.depth_load is LoadOperation.CLEAR
                else LoadOperation.PRESERVE
            )
            store = attachment.depth_store if last_in_scope else StoreOperation.PRESERVE
            builder.rhi_depth_attachment(
                depth_attachment,
                load_values[load],
                store_values[store],
                attachment.clear_depth,
            )
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
        if encoder.viewport is not None:
            builder.viewport(*encoder.viewport)
        if encoder.scissor is not None:
            builder.scissor(*encoder.scissor)
        assert state._native_runtime is not None
        with _dispatch_borrow_scope(dispatch_borrows):
            builder.encode(encoder._native)
        for _, texture in color_attachments:
            texture._mark_device_dirty()

    def _declare_invocation(self, arguments: dict[str, Any], execution_pass: RenderPass) -> None:
        state = _session_state()
        indices = arguments.pop("indices", None)
        arguments.pop("topology", None)
        compiled = self._compile(arguments)
        for parameter in compiled.native.parameters:
            value = arguments[parameter.name]
            if isinstance(value, Texture):
                execution_pass.read(value)
            elif isinstance(value, (TensorStorage, TensorView)):
                if parameter.access == state._native.ACCESS_READ:
                    execution_pass.read(value)
                elif parameter.access == state._native.ACCESS_WRITE:
                    execution_pass.write(value)
                else:
                    execution_pass.read_write(value)
        if indices is not None:
            execution_pass.read(indices)

    def invocation(self, **arguments: Any) -> PipelineInvocation:
        captured = dict(arguments)
        return PipelineInvocation(
            "graphics",
            lambda encoder: self._invoke_direct(dict(captured), encoder),
            lambda execution_pass: self._declare_invocation(dict(captured), execution_pass),
        )

    def __call__(self, **arguments: Any) -> None:
        target = arguments.pop("target", None)
        if not isinstance(target, RenderTarget):
            raise TypeError("immediate graphics execution requires target=RenderTarget")
        state = _session_state()
        if state._architecture == state.cpu:
            raise RuntimeError(
                "CPU graphics pipelines require a software rasterizer, which "
                "Vernon does not provide; CPU supports compute kernels only"
            )
        invocation = self.invocation(**arguments)

        graph = ExecutionGraph()
        graph.add_pass(_ImmediateRenderPass(f"{self._fragment.__name__} immediate", target, invocation))
        try:
            graph.execute()
        finally:
            graph._dispose_native()


def pipeline(*stages: Any, features: Iterable[str] = ()) -> Pipeline:
    return Pipeline(*stages, features=features)


__all__ = ["Pipeline", "PrimitiveTopology", "lines", "pipeline", "points", "triangles"]
