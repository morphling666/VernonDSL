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
    make_target_options,
    materialize_bundle,
    serialize_bundle,
)
from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..language.stage_registry import validate_graphics_topology, validate_stage_target
from .execution_graph import (
    ColorAttachmentUse,
    DepthStencilAttachmentUse,
    ExecutionParameter,
    ExecutionResources,
    GraphicsEncoder,
    LoadOperation,
    PipelineInvocation,
    RenderPass,
    StoreOperation,
)
from .resources import (
    RenderTarget,
    SamplerState,
    TensorStorage,
    TensorView,
    _DispatchBorrowLease,
    _NativeBindingCache,
    _TextureResource,
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


@dataclass
class _CompiledPipeline:
    native: Any
    key: str
    bundle: bytes
    target_identity: str
    native_generation: int


@dataclass
class _GraphicsInvocationPlan:
    state: Any
    arguments: dict[str, Any]
    target: RenderTarget
    operations: dict[str, Any]
    indices: TensorStorage | None
    topology: PrimitiveTopology
    compiled: _CompiledPipeline
    parameters: tuple[Any, ...]
    color_attachments: tuple[tuple[int, _TextureResource], ...]
    lease: _DispatchBorrowLease | None


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
        validate_graphics_topology(kinds)
        feature_values = tuple(features)
        if any(not isinstance(value, str) or not value for value in feature_values):
            raise TypeError("pipeline features must be non-empty strings")
        self._features = tuple(sorted(set(feature_values)))
        self._stages = stages
        self._vertex = stages[0]
        self._fragment = stages[-1]
        self._compiled: _CompiledPipeline | None = None
        self._compiled_generation = -1
        self._direct_binding_cache = _NativeBindingCache()
        self.compile_count = 0
        _session_state()._runtime_children.add(self)

    def _release_runtime_native(self) -> None:
        self._compiled = None
        self._compiled_generation = -1
        self._direct_binding_cache.clear()

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
            state.metal: (state._native.Target.METAL, "metal"),
            state.opengl: (state._native.Target.OPENGL, "opengl"),
            state.opengles: (state._native.Target.OPENGL_ES, "opengles"),
        }[state._architecture]
        options = make_target_options(
            target_name,
            {"version": state._interactive_glsl_version()}
            if state._architecture in {state.opengl, state.opengles}
            else {},
        )
        for stage in (getattr(value, "__vernon_dsl__", (None,))[0] for value in self._stages):
            try:
                validate_stage_target(stage, target_name)
            except ValueError as error:
                raise RuntimeError(str(error)) from None
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
                        "target": options.spec,
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
                canonical_json(options.spec),
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
        if state._architecture in {state.vulkan, state.directx, state.metal, state.opengl, state.opengles}:
            identity_options = make_target_options(
                state._architecture.name,
                {"version": state._interactive_glsl_version()}
                if state._architecture in {state.opengl, state.opengles}
                else {},
            )
            target_identity = canonical_json(identity_options.spec)
            if self._compiled is not None and self._compiled.target_identity == target_identity:
                if state._native_runtime is None:
                    raise RuntimeError(f"{state._architecture.name} pipeline execution requires the native runtime")
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

    def _invoke_direct(
        self,
        arguments: dict[str, Any],
        encoder: GraphicsEncoder | None,
        binding_cache: _NativeBindingCache,
        binding_tokens: dict[str, int] | None = None,
        *,
        immediate_target: RenderTarget | None = None,
    ) -> tuple[Any, _DispatchBorrowLease] | None:
        plan = self._prepare_graphics_invocation(arguments, encoder, immediate_target)
        try:
            with binding_cache.invocation(plan.compiled.native) as builder:
                self._bind_graphics_arguments(plan, builder, binding_cache, binding_tokens)
                return self._encode_graphics_invocation(plan, builder, encoder)
        except Exception:
            if plan.lease is not None:
                plan.lease.release()
            raise

    def _prepare_graphics_invocation(
        self,
        arguments: dict[str, Any],
        encoder: GraphicsEncoder | None,
        immediate_target: RenderTarget | None,
    ) -> _GraphicsInvocationPlan:
        state = _session_state()
        if encoder is None:
            if immediate_target is None:
                raise RuntimeError("direct graphics submission requires a render target")
            target = immediate_target
            operations = {
                "colors": {location: ColorAttachmentUse(texture) for location, texture in target._color_attachments()},
                "depth": DepthStencilAttachmentUse(target) if target._depth_texture is not None else None,
                "first_in_scope": True,
                "last_in_scope": True,
            }
        else:
            target = encoder.target
            operations = encoder.attachment_operations
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
            if parameter.kind in {state._native.PIPELINE_TENSOR, state._native.PIPELINE_IMAGE}
            and isinstance(arguments[parameter.name], (TensorStorage, TensorView, _TextureResource))
        ]
        outputs = tuple(compiled.native.outputs)
        color_attachments = tuple(target._color_attachments())
        dispatch_borrows.extend(
            (f"color_attachment_{location}", texture, "write") for location, texture in color_attachments
        )
        if target._depth_texture is not None:
            dispatch_borrows.append(("depth_attachment", target._depth_texture, "write"))
        output_locations = {output.location for output in outputs}
        attachment_locations = {location for location, _ in color_attachments}
        if attachment_locations != output_locations:
            raise ValueError("RenderTarget color locations must exactly match fragment output locations")
        if indices is not None:
            if (
                not isinstance(indices, TensorStorage)
                or indices.dtype != np.dtype(np.uint32)
                or len(indices.shape) != 1
                or not indices.shape[0]
            ):
                raise TypeError("indices must be a non-empty rank-one u32 TensorStorage")
            dispatch_borrows.append(("indices", indices, "read"))
        if topology not in {triangles, lines, points}:
            raise TypeError("topology must be triangles, lines, or points")
        if state._native_runtime is None:
            raise RuntimeError(f"{state._architecture.name} pipeline execution requires the native runtime")
        lease = _DispatchBorrowLease(dispatch_borrows) if encoder is None else None
        return _GraphicsInvocationPlan(
            state,
            arguments,
            target,
            operations,
            indices,
            topology,
            compiled,
            parameters,
            color_attachments,
            lease,
        )

    def _bind_graphics_arguments(
        self,
        plan: _GraphicsInvocationPlan,
        builder: Any,
        binding_cache: _NativeBindingCache,
        binding_tokens: dict[str, int] | None,
    ) -> None:
        state = plan.state
        for parameter in plan.parameters:
            value = plan.arguments[parameter.name]
            if parameter.kind == state._native.PIPELINE_IMAGE:
                if not isinstance(value, _TextureResource):
                    raise TypeError(f"texture {parameter.name!r} must be a Texture")
                binding_cache.bind_argument(builder, plan.compiled.native, parameter, value)
                continue
            if parameter.kind == state._native.PIPELINE_SAMPLER:
                if not isinstance(value, SamplerState):
                    raise TypeError(f"sampler {parameter.name!r} must be a SamplerState")
                binding_cache.bind_sampler(builder, plan.compiled.native, parameter, value)
                continue
            if parameter.kind != state._native.PIPELINE_TENSOR:
                raise TypeError(f"pipeline parameter {parameter.name!r} has unsupported kind")
            leaves = tuple(parameter.element_leaves)
            host_value = isinstance(value, (TensorStorage, TensorView)) and (
                tuple(value.shape) == tuple(parameter.shape) and len(leaves) == 1 and leaves[0][1:] == (1, 0)
            )
            binding_cache.bind_argument(
                builder,
                plan.compiled.native,
                parameter,
                value,
                host_value=host_value,
                binding_token=None if binding_tokens is None else binding_tokens.get(parameter.name),
            )

    def _encode_graphics_invocation(
        self,
        plan: _GraphicsInvocationPlan,
        builder: Any,
        encoder: GraphicsEncoder | None,
    ) -> tuple[Any, _DispatchBorrowLease] | None:
        state = plan.state
        first_in_scope = plan.operations["first_in_scope"]
        last_in_scope = plan.operations["last_in_scope"]
        load_values = {
            LoadOperation.CLEAR: state._native.ATTACHMENT_CLEAR,
            LoadOperation.PRESERVE: state._native.ATTACHMENT_PRESERVE,
            LoadOperation.DISCARD: state._native.ATTACHMENT_DISCARD,
        }
        store_values = {
            StoreOperation.PRESERVE: state._native.ATTACHMENT_STORE,
            StoreOperation.DISCARD: state._native.ATTACHMENT_DONT_CARE,
        }
        for location, texture in plan.color_attachments:
            attachment = plan.operations["colors"][location]
            load = (
                attachment.load if first_in_scope or attachment.load is LoadOperation.CLEAR else LoadOperation.PRESERVE
            )
            store = attachment.store if last_in_scope else StoreOperation.PRESERVE
            builder.rhi_color_attachment(
                location,
                texture._resident_view(),
                load_values[load],
                store_values[store],
                list(attachment.clear_value),
            )
        depth_attachment = plan.target._resident_depth_attachment()
        if depth_attachment is not None:
            attachment = plan.operations["depth"]
            if attachment is None:
                raise RuntimeError("depth attachment operations are missing while a depth target is bound")
            load = (
                attachment.depth_load
                if first_in_scope or attachment.depth_load is LoadOperation.CLEAR
                else LoadOperation.PRESERVE
            )
            store = attachment.depth_store if last_in_scope else StoreOperation.PRESERVE
            assert plan.target._depth_texture is not None
            builder.rhi_depth_attachment(
                plan.target._depth_texture._resident_view(),
                load_values[load],
                store_values[store],
                attachment.clear_depth,
            )
        if plan.indices is not None:
            builder.rhi_index_binding(plan.indices._resident_buffer(), plan.indices.shape[0])
        native_topology = {
            triangles: state._native.TOPOLOGY_TRIANGLE_LIST,
            lines: state._native.TOPOLOGY_LINE_LIST,
            points: state._native.TOPOLOGY_POINT_LIST,
        }[plan.topology]
        builder.topology(native_topology)
        if encoder is not None and encoder.viewport is not None:
            builder.viewport(*encoder.viewport)
        if encoder is not None and encoder.scissor is not None:
            builder.scissor(*encoder.scissor)
        if encoder is None:
            native_submission = builder.submit()
        else:
            builder.encode(encoder._native)
        for _, texture in plan.color_attachments:
            texture._mark_device_dirty()
        if plan.target._depth_texture is not None:
            plan.target._depth_texture._mark_device_dirty()
        if encoder is None:
            assert plan.lease is not None
            return native_submission, plan.lease
        return None

    def _declare_invocation(self, arguments: dict[str, Any], execution_pass: RenderPass) -> None:
        state = _session_state()
        indices = arguments.pop("indices", None)
        arguments.pop("topology", None)
        compiled = self._compile(arguments)
        for parameter in compiled.native.parameters:
            value = arguments[parameter.name]
            if isinstance(value, _TextureResource):
                if parameter.access == state._native.ACCESS_READ:
                    execution_pass.read(value)
                elif parameter.access == state._native.ACCESS_WRITE:
                    execution_pass.write(value)
                else:
                    execution_pass.read_write(value)
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
        return self._invocation(dict(arguments), _NativeBindingCache())

    def _invocation(
        self,
        captured: dict[str, Any],
        binding_cache: _NativeBindingCache,
    ) -> PipelineInvocation:
        parameterized = any(isinstance(value, ExecutionParameter) for value in captured.values())

        def invoke(encoder: GraphicsEncoder, resources: ExecutionResources | None) -> None:
            binding_tokens: dict[str, int] | None = None
            if parameterized:
                if resources is None:
                    raise RuntimeError("parameterized pipeline invocation requires execution resources")
                resolved = {}
                binding_tokens = {}
                for name, value in captured.items():
                    if isinstance(value, ExecutionParameter):
                        resolved[name], binding_tokens[name] = resources._resolve_parameter_with_token(value)
                    else:
                        resolved[name] = value
            else:
                resolved = dict(captured)
            self._invoke_direct(resolved, encoder, binding_cache, binding_tokens)

        invocation = PipelineInvocation(
            "graphics",
            invoke,
            lambda execution_pass: self._declare_invocation(dict(captured), execution_pass),
        )
        invocation._binding_cache = binding_cache
        return invocation

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
        result = self._invoke_direct(
            dict(arguments),
            None,
            self._direct_binding_cache,
            immediate_target=target,
        )
        if result is None:
            raise RuntimeError("direct graphics invocation did not produce a submission")
        native_submission, lease = result
        try:
            native_submission.wait()
        finally:
            lease.release()


def pipeline(*stages: Any, features: Iterable[str] = ()) -> Pipeline:
    return Pipeline(*stages, features=features)


__all__ = ["Pipeline", "PrimitiveTopology", "lines", "pipeline", "points", "triangles"]
