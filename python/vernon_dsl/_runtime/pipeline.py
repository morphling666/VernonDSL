from __future__ import annotations

import hashlib
import importlib
import inspect
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..bundle import (
    CompiledStage,
    PipelineCompileError,
    TargetOptions,
    canonical_json,
    compiled_stage_from_program,
    make_target_options,
    parse_reflection_json,
)
from ..compiler import Compiler, FrontendCompileRequest, FrontendCompileResult
from ..language.stage_registry import validate_graphics_topology
from ..render import RenderTargetUse
from ..render import clear as clear_attachment
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
    canonical_directory: tempfile.TemporaryDirectory[str] | None = None
    canonical_program: bytes | None = None
    canonical_artifact_system: bytes | None = None
    canonical_stage_bindings: dict[str, str] | None = None
    invocation_identity: str | None = None


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

    def __init__(
        self,
        *stages: Any,
        topology: PrimitiveTopology = triangles,
        features: Iterable[str] = (),
    ):
        kinds = tuple(getattr(stage, "__vernon_dsl__", (None,))[0] for stage in stages)
        validate_graphics_topology(kinds)
        feature_values = tuple(features)
        if any(not isinstance(value, str) or not value for value in feature_values):
            raise TypeError("pipeline features must be non-empty strings")
        self._features = tuple(sorted(set(feature_values)))
        if topology not in {triangles, lines, points}:
            raise TypeError("topology must be triangles, lines, or points")
        self._topology = topology
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

    def _compile_pipeline_bundle(
        self,
        call_arguments: Mapping[str, Any],
        target: RenderTarget,
    ) -> _CompiledPipeline:
        colors = tuple(target._color_attachments())
        if not colors:
            raise RuntimeError("graphics Program requires at least one color attachment")
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
        requests = [self._stage_request(stage, options, native_target) for stage in self._stages]
        compiler = state._native.Compiler()
        logical_entries: list[tuple[str, Mapping[str, Any]]] = []
        for request in requests:
            analyzed = compiler.analyze_program_result(request.frontend.mlir)
            if not bool(analyzed.ok):
                raise RuntimeError(str(analyzed.diagnostics) or "graphics frontend analysis failed")
            reflection = parse_reflection_json(analyzed.reflection)
            entries = reflection.get("entries")
            if not isinstance(entries, list):
                raise RuntimeError("graphics frontend reflection has no entries")
            entry = next(
                (row for row in entries if isinstance(row, Mapping) and row.get("name") == request.entry),
                None,
            )
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"graphics frontend reflection has no entry {request.entry!r}")
            logical_entries.append((str(entry.get("stage")), entry))

        reflected_arguments: dict[str, tuple[str, Mapping[str, Any]]] = {}
        argument_order: list[str] = []
        for role, entry in logical_entries:
            for row in entry.get("arguments", []):
                if not isinstance(row, Mapping):
                    continue
                name = row.get("vernon.source_name")
                if (
                    not isinstance(name, str)
                    or row.get("vernon.builtin") is not None
                    or row.get("vernon.implicit") is not None
                    or (role == "fragment" and row.get("vernon.location") is not None)
                ):
                    continue
                previous = reflected_arguments.get(name)
                if previous is None:
                    reflected_arguments[name] = (role, row)
                    argument_order.append(name)
                elif previous[1].get("type") != row.get("type"):
                    raise RuntimeError(f"graphics stages disagree on logical argument {name!r}")
        if set(reflected_arguments) != set(call_arguments):
            missing = set(reflected_arguments) - set(call_arguments)
            unexpected = set(call_arguments) - set(reflected_arguments)
            if missing:
                raise TypeError(f"missing pipeline argument(s): {', '.join(sorted(missing))}")
            raise TypeError(f"unexpected pipeline argument(s): {', '.join(sorted(unexpected))}")

        def scalar_mlir_type(dtype: np.dtype[Any]) -> str:
            resolved = np.dtype(dtype)
            names = {
                np.dtype(np.float16): "f16",
                np.dtype(np.float32): "f32",
                np.dtype(np.float64): "f64",
                np.dtype(np.int32): "i32",
                np.dtype(np.uint32): "i32",
            }
            if resolved not in names:
                raise TypeError(f"graphics Program does not support runtime dtype {resolved}")
            return names[resolved]

        def peel_tensor_element(spelling: str) -> str | None:
            text = spelling.strip()
            if not text.startswith("!vernon.tensor<") or not text.endswith(">"):
                return None
            body = text[len("!vernon.tensor<") : -1]
            depth = 0
            split = -1
            for index, character in enumerate(body):
                if character in "<[":
                    depth += 1
                elif character in ">]":
                    depth -= 1
                elif character == "," and depth == 0:
                    split = index
            if split < 0:
                return None
            return body[:split].strip()

        def logical_type(name: str) -> str:
            role, row = reflected_arguments[name]
            value = call_arguments[name]
            if role == "vertex" and row.get("attribute_leaves") is not None:
                if not isinstance(value, (TensorStorage, TensorView)):
                    raise TypeError(f"vertex input {name!r} requires TensorStorage or TensorView")
                reflected = row.get("type")
                if not isinstance(reflected, str) or not reflected:
                    raise RuntimeError(f"graphics argument {name!r} has no logical type")
                if not value.shape:
                    raise TypeError(f"vertex input {name!r} requires a leading invocation dimension")
                cell = reflected
                while True:
                    peeled = peel_tensor_element(cell)
                    if peeled is None:
                        break
                    cell = peeled
                if value.dtype.fields:
                    shape = ", ".join(str(int(extent)) for extent in value.shape)
                    return f'!vernon.tensor_view<{cell}, [{shape}], "read", "device">'
                shape = ", ".join(str(int(extent)) for extent in value.shape)
                return f'!vernon.tensor_view<{scalar_mlir_type(value.dtype)}, [{shape}], "read", "device">'
            reflected = row.get("type")
            if not isinstance(reflected, str) or not reflected:
                raise RuntimeError(f"graphics argument {name!r} has no logical type")
            return reflected

        def attachment_type(texture: Any) -> str:
            format_name = getattr(getattr(texture, "format", None), "name", None) or "d32_float"
            return f'!vernon.texture<"2d", f32, "{format_name}", "read_write">'

        attachment_types = [attachment_type(texture) for _, texture in colors]
        if target._depth_texture is not None:
            attachment_types.append(attachment_type(target._depth_texture))
        topology = {"triangles": "triangle_list", "lines": "line_list", "points": "point_list"}[self._topology.name]
        planned = compiler.plan_graphics_result(
            [request.frontend.mlir for request in requests],
            topology,
            list(self._features),
            attachment_types,
            len(colors),
            [(name, logical_type(name)) for name in argument_order],
        )
        if not bool(planned.ok):
            raise RuntimeError(str(planned.diagnostics) or "graphics Program planning failed")
        planned_reflection = parse_reflection_json(planned.reflection)
        compile_requests = planned_reflection.get("kernel_compile_requests")
        if not isinstance(compile_requests, list) or len(compile_requests) != 1:
            raise RuntimeError("graphics Program planner did not return one render request")
        request_id = compile_requests[0].get("id")
        if not isinstance(request_id, str):
            raise RuntimeError("graphics Program compile request has no identity")

        compiled_stages: list[CompiledStage] = []
        compiled_results: list[Any] = []
        for request in requests:
            result = compiler.compile_program_result(
                request.frontend.mlir,
                request.native_target,
                **request.target.native_options,
            )
            if not bool(result.ok):
                raise RuntimeError(str(result.diagnostics) or "graphics target compilation failed")
            compiled_results.append(result)
            try:
                compiled_stages.append(
                    compiled_stage_from_program(
                        result,
                        module=request.module,
                        module_manifest=request.module_manifest,
                        entry=request.entry,
                        target=request.target,
                    )
                )
            except PipelineCompileError as error:
                raise RuntimeError(str(error)) from None
        artifact_id = hashlib.sha256(
            canonical_json(
                {
                    "modules": [stage.id for stage in compiled_stages],
                    "program": planned.reflection,
                    "target": options.spec,
                }
            ).encode()
        ).hexdigest()
        # Graphics image/attachment extents only. Compute TensorView dyn is bound later.
        shape_facts = [
            (
                request_id,
                "target" if len(colors) == 1 else f"color{index}",
                list(texture.shape),
            )
            for index, (_, texture) in enumerate(colors)
        ]
        if target._depth_texture is not None:
            shape_facts.append((request_id, "depth", list(target._depth_texture.shape)))
        for name, value in call_arguments.items():
            if isinstance(value, _TextureResource):
                shape_facts.append((request_id, name, list(value.shape)))
        finalized = compiler.finalize_program_result(
            planned.reflection,
            [
                (request_id, artifact_id, stage.entry, canonical_json(dict(stage.reflection)))
                for stage in compiled_stages
            ],
            shape_facts,
        )
        if not bool(finalized.ok):
            raise RuntimeError(str(finalized.diagnostics) or "graphics Program finalization failed")
        from .._shader_assets.cooking import _canonical_graphics_deployment

        directory = tempfile.TemporaryDirectory(prefix="vernon-graphics-")
        canonical_program, artifact_system, stage_bindings = _canonical_graphics_deployment(
            compiled_stages,
            parse_reflection_json(finalized.reflection),
            target=options,
            request_id=request_id,
            artifact_id=artifact_id,
            output=Path(directory.name),
        )
        program_bytes = canonical_json(dict(canonical_program)).encode()
        artifact_bytes = canonical_json(dict(artifact_system)).encode()
        key = hashlib.sha256(program_bytes + artifact_bytes).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            native = state._native_runtime.load_canonical_program(
                program_bytes,
                artifact_bytes,
                directory.name,
                stage_bindings,
                [],
            )
            cached = _CompiledPipeline(
                native,
                key,
                b"",
                canonical_json(options.spec),
                state._runtime_generation,
                directory,
                program_bytes,
                artifact_bytes,
                dict(stage_bindings),
                canonical_json(
                    {
                        "target": [(location, value.shape, value.format.name) for location, value in colors],
                        "depth": None if target._depth_texture is None else tuple(target._depth_texture.shape),
                        "arguments": {
                            name: (tuple(value.shape), str(value.dtype))
                            for name, value in call_arguments.items()
                            if isinstance(value, (TensorStorage, TensorView))
                        },
                    }
                ),
            )
            self._cache[key] = cached
        elif cached.native_generation != state._runtime_generation:
            if cached.canonical_program is None or cached.canonical_artifact_system is None:
                raise RuntimeError("graphics pipeline cache is missing its canonical Program")
            assert cached.canonical_directory is not None
            assert cached.canonical_stage_bindings is not None
            cached.native = state._native_runtime.load_canonical_program(
                cached.canonical_program,
                cached.canonical_artifact_system,
                cached.canonical_directory.name,
                cached.canonical_stage_bindings,
                [],
            )
            cached.native_generation = state._runtime_generation
            directory.cleanup()
        else:
            directory.cleanup()
        self._compiled = cached
        self._compiled_generation = state._runtime_generation
        if self.compile_count == 0:
            self.compile_count = 1
        return cached

    def _compile(
        self,
        call_arguments: Mapping[str, Any] | None = None,
        target: RenderTarget | None = None,
    ) -> _CompiledPipeline:
        state = _session_state()
        invocation_identity = (
            canonical_json(
                {
                    "target": [
                        (location, value.shape, value.format.name) for location, value in target._color_attachments()
                    ],
                    "depth": None if target._depth_texture is None else tuple(target._depth_texture.shape),
                    "arguments": {
                        name: (tuple(value.shape), str(value.dtype))
                        for name, value in (call_arguments or {}).items()
                        if isinstance(value, (TensorStorage, TensorView))
                    },
                }
            )
            if target is not None
            else None
        )
        if (
            self._compiled is not None
            and self._compiled_generation == state._runtime_generation
            and self._compiled.invocation_identity == invocation_identity
        ):
            return self._compiled
        if state._architecture in {state.vulkan, state.directx, state.metal, state.opengl, state.opengles}:
            identity_options = make_target_options(
                state._architecture.name,
                {"version": state._interactive_glsl_version()}
                if state._architecture in {state.opengl, state.opengles}
                else {},
            )
            target_identity = canonical_json(identity_options.spec)
            if (
                self._compiled is not None
                and self._compiled.target_identity == target_identity
                and (
                    self._compiled.canonical_program is None
                    or self._compiled.invocation_identity == invocation_identity
                )
            ):
                if state._native_runtime is None:
                    raise RuntimeError(f"{state._architecture.name} pipeline execution requires the native runtime")
                if self._compiled.canonical_program is None:
                    raise RuntimeError("graphics pipeline cache is missing its canonical Program")
                assert self._compiled.canonical_artifact_system is not None
                assert self._compiled.canonical_directory is not None
                assert self._compiled.canonical_stage_bindings is not None
                self._compiled.native = state._native_runtime.load_canonical_program(
                    self._compiled.canonical_program,
                    self._compiled.canonical_artifact_system,
                    self._compiled.canonical_directory.name,
                    self._compiled.canonical_stage_bindings,
                    [],
                )
                self._compiled.native_generation = state._runtime_generation
                self._compiled_generation = state._runtime_generation
                return self._compiled
            if target is not None and call_arguments is not None:
                return self._compile_pipeline_bundle(call_arguments, target)
            raise RuntimeError("graphics compilation requires invocation arguments and a RenderTarget")
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
        immediate_render: RenderTargetUse | None = None,
    ) -> tuple[Any, _DispatchBorrowLease] | None:
        plan = self._prepare_graphics_invocation(arguments, encoder, immediate_render)
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
        immediate_render: RenderTargetUse | None,
    ) -> _GraphicsInvocationPlan:
        state = _session_state()
        if encoder is None:
            if immediate_render is None:
                raise RuntimeError("direct graphics submission requires a render target")
            target = immediate_render.target
            color_operations = dict(immediate_render.colors)
            target_locations = {location for location, _ in target._color_attachments()}
            if set(color_operations) != target_locations:
                raise ValueError("render color operations must exactly match RenderTarget color locations")
            colors = {}
            for location, texture in target._color_attachments():
                operation = color_operations.get(location)
                if operation is None:
                    raise ValueError(f"render use has no operation for color attachment {location}")
                clear_value = operation.clear_value
                if operation.load is LoadOperation.CLEAR:
                    if (
                        not isinstance(clear_value, (tuple, list))
                        or len(clear_value) != 4
                        or any(not isinstance(component, (int, float)) for component in clear_value)
                    ):
                        raise TypeError("color clear values must contain four numeric components")
                    normalized_clear = tuple(float(component) for component in clear_value)
                else:
                    normalized_clear = (0.0, 0.0, 0.0, 0.0)
                colors[location] = ColorAttachmentUse(
                    texture,
                    operation.load,
                    operation.store,
                    normalized_clear,
                )
            depth_operation = immediate_render.depth
            operations = {
                "colors": colors,
                "depth": (
                    DepthStencilAttachmentUse(
                        target,
                        depth_load=depth_operation.load,
                        depth_store=depth_operation.store,
                        clear_depth=float(depth_operation.clear_value)
                        if depth_operation.load is LoadOperation.CLEAR
                        else 1.0,
                    )
                    if target._depth_texture is not None and depth_operation is not None
                    else None
                ),
                "first_in_scope": True,
                "last_in_scope": True,
                "render_area": immediate_render.render_area,
            }
        else:
            target = encoder.target
            operations = encoder.attachment_operations
        indices = arguments.pop("indices", None)
        topology = arguments.pop("topology", self._topology)
        if topology is not self._topology:
            raise ValueError("graphics topology is static Pipeline state; construct a separate Pipeline")
        compiled = self._compile(arguments, target)
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
            cell_shape = tuple(parameter.shape)
            vertex_stream = isinstance(value, (TensorStorage, TensorView)) and (
                (cell_shape and len(value.shape) == len(cell_shape) + 1)
                or (not cell_shape and len(value.shape) == 1 and (len(leaves) != 1 or leaves[0][1:] == (1, 0)))
            )
            host_value = (
                isinstance(value, (TensorStorage, TensorView))
                and not vertex_stream
                and (
                    not cell_shape
                    or (tuple(value.shape) == cell_shape and len(leaves) == 1 and leaves[0][1:] == (1, 0))
                )
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
        if encoder is None and plan.operations.get("render_area") is not None:
            builder.viewport(*plan.operations["render_area"])
            builder.scissor(*plan.operations["render_area"])
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
        indices = arguments.pop("indices", None)
        arguments.pop("topology", None)
        arguments.pop("target", None)
        for value in arguments.values():
            if isinstance(value, (_TextureResource, TensorStorage, TensorView)):
                execution_pass.read(value)
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
        render = arguments.pop("render", None)
        legacy_target = arguments.pop("target", None)
        if render is not None and legacy_target is not None:
            raise TypeError("graphics execution accepts render= or legacy target=, not both")
        if render is None and isinstance(legacy_target, RenderTarget):
            render = RenderTargetUse(
                legacy_target,
                tuple(
                    (location, clear_attachment((0.0, 0.0, 0.0, 0.0)))
                    for location, _ in legacy_target._color_attachments()
                ),
                clear_attachment(1.0) if legacy_target._depth_texture is not None else None,
            )
        if not isinstance(render, RenderTargetUse):
            raise TypeError("immediate graphics execution requires render=RenderTargetUse")
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
            immediate_render=render,
        )
        if result is None:
            raise RuntimeError("direct graphics invocation did not produce a submission")
        native_submission, lease = result
        try:
            native_submission.wait()
        finally:
            lease.release()


def pipeline(
    *stages: Any,
    topology: PrimitiveTopology = triangles,
    features: Iterable[str] = (),
) -> Pipeline:
    return Pipeline(*stages, topology=topology, features=features)


__all__ = ["Pipeline", "PrimitiveTopology", "lines", "pipeline", "points", "triangles"]
