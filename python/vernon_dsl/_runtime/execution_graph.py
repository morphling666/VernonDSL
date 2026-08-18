from __future__ import annotations

import copy
import importlib
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from .resources import (
    RawBuffer,
    RenderTarget,
    TensorStorage,
    TensorView,
    Texture,
    TextureView,
    _DispatchBorrowLease,
    _TextureResource,
)


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


def _resource_identity(value: Any) -> object:
    if isinstance(value, TensorView):
        return value.owner
    if isinstance(value, _TextureResource):
        return value
    return value


class LoadOperation(Enum):
    CLEAR = "clear"
    PRESERVE = "preserve"
    DISCARD = "discard"


class StoreOperation(Enum):
    PRESERVE = "preserve"
    DISCARD = "discard"


class SubmissionState(Enum):
    PENDING = 0
    SUCCEEDED = 1
    FAILED = 2


@dataclass(frozen=True)
class GraphResource:
    id: int
    value: Any
    exported: bool
    _owner: object
    _native: Any
    _generation: int


@dataclass(frozen=True, eq=False)
class ExecutionParameter:
    name: str
    _owner: object
    _native: Any
    _generation: int


def _snapshot_parameter_value(value: Any) -> Any:
    if isinstance(value, (GraphResource, TensorStorage, TensorView, RawBuffer, _TextureResource, RenderTarget)):
        raise TypeError("execution value parameters cannot contain graph resources")
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ColorAttachmentUse:
    texture: Texture | TextureView
    load: LoadOperation = LoadOperation.CLEAR
    store: StoreOperation = StoreOperation.PRESERVE
    clear_value: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DepthStencilAttachmentUse:
    target: RenderTarget
    depth_load: LoadOperation = LoadOperation.CLEAR
    depth_store: StoreOperation = StoreOperation.PRESERVE
    clear_depth: float = 1.0
    stencil_load: LoadOperation = LoadOperation.DISCARD
    stencil_store: StoreOperation = StoreOperation.DISCARD
    clear_stencil: int = 0
    read_only_depth: bool = False
    read_only_stencil: bool = False


class PipelineInvocation:
    """Prepared graphics or compute work that encodes into a graph-provided encoder."""

    def __init__(
        self,
        kind: str,
        callback: Callable[[GraphicsEncoder | ComputeEncoder, ExecutionResources | None], None],
        declare_callback: Callable[[ExecutionPass], None] | None = None,
    ):
        if kind not in {"graphics", "compute"}:
            raise ValueError("pipeline invocation kind must be graphics or compute")
        self.kind = kind
        self._callback = callback
        self._declare_callback = declare_callback
        self._binding_cache: Any | None = None

    def declare(self, execution_pass: ExecutionPass) -> None:
        if self._declare_callback is not None:
            self._declare_callback(execution_pass)

    def encode(
        self,
        encoder: GraphicsEncoder | ComputeEncoder,
        resources: ExecutionResources | None = None,
    ) -> None:
        if self.kind != encoder.kind:
            raise TypeError(f"{self.kind} invocation cannot encode into a {encoder.kind} encoder")
        self._callback(encoder, resources)


class ExecutionResources:
    def __init__(self, graph: ExecutionGraph | CompiledExecutionGraph, native_bindings: Any | None = None):
        self._graph = graph
        self._native_bindings = native_bindings

    def resolve(self, value: GraphResource | ExecutionParameter) -> Any:
        if isinstance(value, GraphResource):
            return self._graph._resolve(value)
        if isinstance(value, ExecutionParameter):
            self._graph._validate_parameter(value)
            if self._native_bindings is None:
                raise RuntimeError("execution parameter resolution requires submission bindings")
            return self._native_bindings.get(value._native)
        raise TypeError("execution resources resolve only graph resources and execution parameters")

    def _resolve_parameter_with_token(self, parameter: ExecutionParameter) -> tuple[Any, int]:
        self._graph._validate_parameter(parameter)
        if self._native_bindings is None:
            raise RuntimeError("execution parameter resolution requires submission bindings")
        return self._native_bindings.get(parameter._native), self._native_bindings.token(parameter._native)


class GraphicsEncoder:
    kind = "graphics"

    def __init__(
        self,
        render_pass: RenderPass,
        native: Any,
        *,
        first_in_scope: bool,
        last_in_scope: bool,
    ):
        self._pass = render_pass
        self._native = native
        self._first_in_scope = first_in_scope
        self._last_in_scope = last_in_scope
        self.viewport: tuple[int, int, int, int] | None = None
        self.scissor: tuple[int, int, int, int] | None = None

    @property
    def target(self) -> RenderTarget:
        if self._pass._target is None:
            raise RuntimeError("render pass did not declare attachments")
        return self._pass._target

    @property
    def attachment_operations(self) -> dict[str, Any]:
        return {
            "colors": self._pass._colors,
            "depth": self._pass._depth,
            "first_in_scope": self._first_in_scope,
            "last_in_scope": self._last_in_scope,
        }

    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        self.viewport = _checked_rectangle("viewport", x, y, width, height)

    def set_scissor(self, x: int, y: int, width: int, height: int) -> None:
        self.scissor = _checked_rectangle("scissor", x, y, width, height)


class ComputeEncoder:
    kind = "compute"

    def __init__(self, native: Any):
        self._native = native


def _checked_rectangle(name: str, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
    values = (x, y, width, height)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values) or width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain integer coordinates and positive dimensions")
    return values


class ExecutionPass:
    def __init__(self, name: str | None = None):
        self.name = name or type(self).__name__
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("execution pass name must be a non-empty string")
        self._never_cull = False
        self._side_effect = False
        self._no_merge = False
        self._derivative = False
        self._dependencies: list[ExecutionPass] = []
        self._graph_ref: weakref.ReferenceType[Any] | None = None
        self._native_pass: Any | None = None
        self._declaring = False
        self._first_in_scope = False
        self._last_in_scope = False
        self._borrow_uses: list[tuple[str, Any, str]] = []
        self._frozen = False

    @property
    def _graph(self) -> Any | None:
        return self._graph_ref() if self._graph_ref is not None else None

    @_graph.setter
    def _graph(self, value: Any | None) -> None:
        self._graph_ref = weakref.ref(value) if value is not None else None

    @property
    def never_cull(self) -> bool:
        return self._never_cull

    @never_cull.setter
    def never_cull(self, value: bool) -> None:
        self._set_flag("_never_cull", value)

    @property
    def side_effect(self) -> bool:
        return self._side_effect

    @side_effect.setter
    def side_effect(self, value: bool) -> None:
        self._set_flag("_side_effect", value)

    @property
    def no_merge(self) -> bool:
        return self._no_merge

    @no_merge.setter
    def no_merge(self, value: bool) -> None:
        self._set_flag("_no_merge", value)

    @property
    def derivative(self) -> bool:
        return self._derivative

    @derivative.setter
    def derivative(self, value: bool) -> None:
        self._set_flag("_derivative", value)

    def _set_flag(self, field: str, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("execution pass flags must be bool")
        if self._frozen:
            raise RuntimeError("cannot mutate a pass after its execution graph has been compiled")
        if getattr(self, field) == value:
            return
        setattr(self, field, value)
        if self._graph is not None:
            self._sync_flags()

    def _sync_flags(self) -> None:
        if self._native_pass is None:
            return
        native = _session_state()._native
        flags = 0
        if self._never_cull:
            flags |= native.GRAPH_PASS_NEVER_CULL
        if self._no_merge:
            flags |= native.GRAPH_PASS_NO_MERGE
        if self._side_effect:
            flags |= native.GRAPH_PASS_SIDE_EFFECT
        if self._derivative:
            flags |= native.GRAPH_PASS_DERIVATIVE
        self._native_pass.set_flags(flags)

    def declare(self) -> None:
        raise NotImplementedError

    def depends_on(self, dependency: ExecutionPass) -> ExecutionPass:
        if not isinstance(dependency, ExecutionPass):
            raise TypeError("dependency must be an ExecutionPass")
        if self._frozen:
            raise RuntimeError("cannot mutate a pass after its execution graph has been compiled")
        if dependency is self:
            raise ValueError(f"pass {self.name!r} cannot depend on itself")
        if dependency not in self._dependencies:
            self._dependencies.append(dependency)
            if self._graph is not None:
                self._graph._sync_dependencies()
        return self

    def read(self, resource: GraphResource | Any) -> None:
        self._use(resource, "read")

    def write(self, resource: GraphResource | Any) -> None:
        self._use(resource, "write")

    def read_write(self, resource: GraphResource | Any) -> None:
        self._use(resource, "read_write")

    def _use(self, resource: GraphResource | Any, access: str) -> None:
        if not self._declaring or self._graph is None or self._native_pass is None:
            raise RuntimeError("resource declarations are only valid while the pass is being declared")
        graph_resource = resource if isinstance(resource, GraphResource) else self._graph.import_resource(resource)
        self._graph._validate_resource(graph_resource)
        self._borrow_uses.append((self.name, graph_resource.value, access))
        native = _session_state()._native
        stage_mask = (
            native.GRAPH_STAGE_COMPUTE
            if isinstance(self, ComputePass)
            else native.GRAPH_STAGE_VERTEX | native.GRAPH_STAGE_FRAGMENT
        )
        self._native_pass.use(
            graph_resource._native,
            {
                "read": native.GRAPH_READ,
                "write": native.GRAPH_WRITE,
                "read_write": native.GRAPH_READ_WRITE,
            }[access],
            native.GRAPH_SHADER_READ if access == "read" else native.GRAPH_SHADER_WRITE,
            stage_mask,
        )

    def _native_declare(self) -> None:
        self._borrow_uses.clear()
        self._declaring = True
        try:
            self.declare()
        finally:
            self._declaring = False


class RenderPass(ExecutionPass):
    def __init__(self, name: str | None = None):
        super().__init__(name)
        self._target: RenderTarget | None = None
        self._colors: dict[int, ColorAttachmentUse] = {}
        self._depth: DepthStencilAttachmentUse | None = None
        self._render_area: tuple[int, int, int, int] | None = None

    def execute(self, encoder: GraphicsEncoder, resources: ExecutionResources) -> None:
        raise NotImplementedError

    def attachments(
        self,
        target: RenderTarget,
        *,
        color_load: LoadOperation = LoadOperation.CLEAR,
        color_store: StoreOperation = StoreOperation.PRESERVE,
        clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        depth_load: LoadOperation = LoadOperation.CLEAR,
        depth_store: StoreOperation = StoreOperation.PRESERVE,
        clear_depth: float = 1.0,
    ) -> None:
        if not isinstance(target, RenderTarget):
            raise TypeError("target must be a RenderTarget")
        self._target = target
        for location, texture in target._color_attachments():
            self.color(
                location,
                ColorAttachmentUse(texture, color_load, color_store, _checked_clear_color(clear_color)),
            )
        if target._has_depth:
            self.depth(DepthStencilAttachmentUse(target, depth_load, depth_store, clear_depth))

    def color(self, location: int, attachment: ColorAttachmentUse) -> None:
        if not self._declaring or self._graph is None or self._native_pass is None:
            raise RuntimeError("attachments are only valid while the pass is being declared")
        if not isinstance(location, int) or isinstance(location, bool) or location < 0:
            raise ValueError("color attachment location must be a non-negative integer")
        if not isinstance(attachment, ColorAttachmentUse):
            raise TypeError("attachment must be a ColorAttachmentUse")
        if location in self._colors:
            raise ValueError(f"color attachment location {location} is already declared")
        if self._target is None:
            self._target = RenderTarget(shape=attachment.texture.shape)
        if attachment.texture.shape != self._target.shape:
            raise ValueError("all render attachments must have matching dimensions")
        if location not in dict(self._target._color_attachments()):
            self._target.attach_color(location, attachment.texture)
        resource = self._graph.import_resource(attachment.texture)
        self._colors[location] = attachment
        self._borrow_uses.append((f"{self.name}.color[{location}]", attachment.texture, "write"))
        self._native_pass.color(
            location,
            resource._native,
            _native_load(attachment.load),
            _native_store(attachment.store),
            attachment.clear_value,
        )

    def depth(self, attachment: DepthStencilAttachmentUse) -> None:
        if not self._declaring or self._graph is None or self._native_pass is None:
            raise RuntimeError("attachments are only valid while the pass is being declared")
        if not isinstance(attachment, DepthStencilAttachmentUse):
            raise TypeError("attachment must be a DepthStencilAttachmentUse")
        if self._depth is not None:
            raise ValueError("depth/stencil attachment is already declared")
        if not 0.0 <= attachment.clear_depth <= 1.0:
            raise ValueError("clear depth must be between zero and one")
        if self._target is not None and attachment.target is not self._target:
            raise ValueError("depth and color attachments must belong to one RenderTarget")
        self._target = attachment.target
        self._depth = attachment
        resource = self._graph.import_resource(attachment.target)
        if attachment.target._depth_texture is not None:
            self._borrow_uses.append(
                (
                    f"{self.name}.depth",
                    attachment.target._depth_texture,
                    "read" if attachment.read_only_depth else "write",
                )
            )
        self._native_pass.depth(
            resource._native,
            _native_load(attachment.depth_load),
            _native_store(attachment.depth_store),
            attachment.clear_depth,
            _native_load(attachment.stencil_load),
            _native_store(attachment.stencil_store),
            attachment.clear_stencil,
            attachment.read_only_depth,
            attachment.read_only_stencil,
        )

    def render_area(self, x: int, y: int, width: int, height: int) -> None:
        if not self._declaring or self._native_pass is None:
            raise RuntimeError("render area is only valid while the pass is being declared")
        self._render_area = _checked_rectangle("render area", x, y, width, height)
        self._native_pass.render_area(*self._render_area)

    def _native_declare(self) -> None:
        self._target = None
        self._colors.clear()
        self._depth = None
        self._render_area = None
        super()._native_declare()

    def _native_execute(self, native_encoder: Any, native_bindings: Any | None = None) -> None:
        if self._graph is None:
            raise RuntimeError("render pass execution requires a declared execution graph")
        encoder = GraphicsEncoder(
            self,
            native_encoder,
            first_in_scope=self._first_in_scope,
            last_in_scope=self._last_in_scope,
        )
        try:
            self.execute(encoder, ExecutionResources(self._graph, native_bindings))
        finally:
            encoder._native = None


class ComputePass(ExecutionPass):
    def execute(self, encoder: ComputeEncoder, resources: ExecutionResources) -> None:
        raise NotImplementedError

    def _native_execute(self, native_encoder: Any, native_bindings: Any | None = None) -> None:
        if self._graph is None:
            raise RuntimeError("compute pass execution requires a declared execution graph")
        encoder = ComputeEncoder(native_encoder)
        try:
            self.execute(encoder, ExecutionResources(self._graph, native_bindings))
        finally:
            encoder._native = None


def _checked_clear_color(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(not isinstance(component, (int, float)) for component in value)
    ):
        raise ValueError("clear color must contain four numeric components")
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _native_load(operation: LoadOperation) -> int:
    native = _session_state()._native
    return {
        LoadOperation.CLEAR: native.ATTACHMENT_CLEAR,
        LoadOperation.PRESERVE: native.ATTACHMENT_PRESERVE,
        LoadOperation.DISCARD: native.ATTACHMENT_DISCARD,
    }[operation]


def _native_store(operation: StoreOperation) -> int:
    native = _session_state()._native
    return {
        StoreOperation.PRESERVE: native.ATTACHMENT_STORE,
        StoreOperation.DISCARD: native.ATTACHMENT_DONT_CARE,
    }[operation]


@dataclass(frozen=True)
class CompiledBarrier:
    source_stage_mask: int
    destination_stage_mask: int
    source_access: int
    destination_access: int
    old_state: int
    new_state: int
    is_image: bool
    base_mip_level: int
    mip_level_count: int
    base_array_layer: int
    array_layer_count: int
    aspects: int


@dataclass(frozen=True)
class CompiledScope:
    kind: str
    passes: tuple[ExecutionPass, ...]
    barriers: tuple[CompiledBarrier, ...] = ()


class ExecutionSubmission:
    def __init__(self, native: Any, owner: Any, lease: _DispatchBorrowLease | None = None):
        self._native = native
        self._owner = owner
        self._lease = lease
        _session_state()._runtime_submissions.add(self)

    def _release_runtime_native(self) -> None:
        try:
            self._native.wait()
        except Exception:
            pass
        finally:
            if self._lease is not None:
                self._lease.release()
                self._lease = None

    def _release_if_complete(self) -> None:
        if self._lease is not None and SubmissionState(self._native.state) is not SubmissionState.PENDING:
            self._lease.release()
            self._lease = None

    @property
    def state(self) -> SubmissionState:
        state = SubmissionState(self._native.state)
        if state is not SubmissionState.PENDING and self._lease is not None:
            self._lease.release()
            self._lease = None
        return state

    def wait(self) -> None:
        try:
            self._native.wait()
        finally:
            if self._lease is not None:
                self._lease.release()
                self._lease = None

    def __del__(self) -> None:
        lease = getattr(self, "_lease", None)
        if lease is None:
            return
        try:
            self._native.wait()
        except Exception:
            pass
        lease.release()
        self._lease = None


class ExecutionBindings:
    def __init__(self, plan: CompiledExecutionGraph, native: Any, values: Mapping[ExecutionParameter, Any]):
        self._plan = plan
        self._native = native
        self._values = dict(values)

    def update(self, changes: Mapping[ExecutionParameter, Any]) -> ExecutionBindings:
        self._plan._ensure_current()
        if not isinstance(changes, Mapping):
            raise TypeError("execution binding changes must be a mapping")
        snapshots: list[tuple[ExecutionParameter, Any]] = []
        for parameter, value in changes.items():
            self._plan._validate_parameter(parameter)
            snapshots.append((parameter, _snapshot_parameter_value(value)))
        for parameter, value in snapshots:
            self._native.set(parameter._native, value)
            self._values[parameter] = value
        return self


class CompiledExecutionGraph:
    def __init__(self, builder: ExecutionGraph, native: Any):
        self._native = native
        self._generation = builder._generation
        self._owner = builder._owner
        self._passes = tuple(builder._passes)
        self._resources = tuple(builder._resources)
        self._parameters = tuple(builder._parameters)
        self._differentiable_inputs = dict(builder._differentiable_inputs)
        self._objectives = dict(builder._objectives)
        self._schedule = tuple(self._passes[index] for index in native.schedule)
        self._borrows = [
            borrow
            for execution_pass in self._schedule
            if not getattr(execution_pass, "_manages_borrows", False)
            for borrow in execution_pass._borrow_uses
            if isinstance(borrow[1], (TensorStorage, RawBuffer, TensorView, _TextureResource))
        ]
        scopes: list[CompiledScope] = []
        for execution_pass in self._passes:
            execution_pass._frozen = True
            execution_pass._graph = self
            execution_pass._first_in_scope = False
            execution_pass._last_in_scope = False
        for native_scope in native.scopes:
            passes = tuple(self._passes[index] for index in native_scope.pass_indices)
            if passes:
                passes[0]._first_in_scope = True
                passes[-1]._last_in_scope = True
            scopes.append(
                CompiledScope(
                    "render" if native_scope.rendering else "compute",
                    passes,
                    tuple(
                        CompiledBarrier(
                            barrier.source_stage_mask,
                            barrier.destination_stage_mask,
                            barrier.source_access,
                            barrier.destination_access,
                            barrier.old_state,
                            barrier.new_state,
                            barrier.is_image,
                            barrier.base_mip_level,
                            barrier.mip_level_count,
                            barrier.base_array_layer,
                            barrier.array_layer_count,
                            barrier.aspects,
                        )
                        for barrier in native_scope.barriers
                    ),
                )
            )
        self._scopes = tuple(scopes)
        _session_state()._runtime_children.add(self)

    def _release_runtime_native(self) -> None:
        self._native = None

    def _ensure_current(self) -> Any:
        if self._native is None or self._generation != _session_state()._runtime_generation:
            raise RuntimeError("CompiledExecutionGraph requires the runtime generation in which it was compiled")
        return self._native

    @property
    def schedule(self) -> tuple[ExecutionPass, ...]:
        return self._schedule

    @property
    def scopes(self) -> tuple[CompiledScope, ...]:
        return self._scopes

    @property
    def autodiff_checkpoint_plan(self) -> Mapping[str, Any] | None:
        return self._ensure_current().autodiff_checkpoint_plan

    def _validate_resource(self, resource: GraphResource) -> None:
        if (
            not isinstance(resource, GraphResource)
            or resource._owner is not self._owner
            or resource._generation != self._generation
            or resource.id != resource._native.id
            or not any(candidate is resource for candidate in self._resources)
        ):
            raise ValueError("resource does not belong to this compiled execution graph")

    def _resolve(self, resource: GraphResource) -> Any:
        self._validate_resource(resource)
        return resource.value

    def _validate_parameter(self, parameter: ExecutionParameter) -> None:
        if (
            not isinstance(parameter, ExecutionParameter)
            or parameter._owner is not self._owner
            or parameter._generation != self._generation
            or not any(candidate is parameter for candidate in self._parameters)
        ):
            raise ValueError("parameter does not belong to this compiled execution graph")

    def create_bindings(self, initial: Mapping[ExecutionParameter, Any]) -> ExecutionBindings:
        native = self._ensure_current()
        if not isinstance(initial, Mapping):
            raise TypeError("initial execution bindings must be a mapping")
        for parameter in initial:
            self._validate_parameter(parameter)
        if len(initial) != len(self._parameters) or any(parameter not in initial for parameter in self._parameters):
            raise ValueError("initial execution bindings must bind every parameter exactly once")
        entries = [(parameter._native, _snapshot_parameter_value(initial[parameter])) for parameter in self._parameters]
        return ExecutionBindings(
            self,
            native.create_bindings(entries),
            {
                parameter: value
                for parameter, value in zip(self._parameters, (entry[1] for entry in entries), strict=True)
            },
        )

    def submit(self, bindings: ExecutionBindings | None = None) -> ExecutionSubmission:
        native = self._ensure_current()
        if bindings is not None and (not isinstance(bindings, ExecutionBindings) or bindings._plan is not self):
            raise ValueError("execution bindings do not belong to this compiled execution graph")
        if bindings is None and self._parameters:
            raise ValueError("compiled execution graph requires parameter bindings")
        lease = _DispatchBorrowLease(self._borrows)
        try:
            native_submission = native.submit(None if bindings is None else bindings._native)
        except Exception:
            if lease is not None:
                lease.release()
            raise
        submission = ExecutionSubmission(native_submission, self, lease)
        submission._release_if_complete()
        return submission

    def vjp(self, bindings: ExecutionBindings | None = None) -> Any:
        from .graph_autodiff import GraphPullback

        native = self._ensure_current()
        if bindings is not None and (not isinstance(bindings, ExecutionBindings) or bindings._plan is not self):
            raise ValueError("execution bindings do not belong to this compiled execution graph")
        if bindings is None and self._parameters:
            raise ValueError("compiled execution graph requires parameter bindings")
        native_bindings = None if bindings is None else bindings._native
        for execution_pass in self._schedule:
            prepare = getattr(execution_pass, "_prepare_vjp_recording", None)
            if prepare is not None:
                prepare(native_bindings)
        return GraphPullback(native.vjp(native_bindings), self)


class ExecutionGraph:
    def __init__(self):
        state = _session_state()
        self._generation = state._runtime_generation
        self._owner = object()
        self._passes: list[ExecutionPass] = []
        self._resources: list[GraphResource] = []
        self._resource_by_identity: dict[int, GraphResource] = {}
        self._parameters: list[ExecutionParameter] = []
        self._parameter_by_name: dict[str, ExecutionParameter] = {}
        self._differentiable_inputs: dict[str, GraphResource | ExecutionParameter] = {}
        self._objectives: dict[str, GraphResource] = {}
        self._native_graph = (
            state._native_runtime.create_execution_graph() if state._native_runtime is not None else None
        )
        state._runtime_children.add(self)

    def _release_runtime_native(self) -> None:
        for execution_pass in getattr(self, "_passes", ()):
            execution_pass._native_pass = None
        self._native_graph = None

    def _ensure_current(self) -> Any:
        state = _session_state()
        if self._native_graph is None or self._generation != state._runtime_generation:
            raise RuntimeError("ExecutionGraph requires the current initialized runtime")
        return self._native_graph

    def import_resource(self, value: Any, *, exported: bool = True) -> GraphResource:
        native_graph = self._ensure_current()
        identity_value = _resource_identity(value)
        identity = id(identity_value)
        existing = self._resource_by_identity.get(identity)
        if existing is not None:
            existing_identity = _resource_identity(existing.value)
            if existing_identity is not identity_value:
                raise RuntimeError("Python resource identity collision")
            return existing
        state = _session_state()
        if state._architecture == state.cpu and isinstance(value, (TensorStorage, TensorView, RawBuffer)):
            checkpoint_owner = _resource_identity(value)
            native_resource = native_graph.import_host_buffer(identity, checkpoint_owner._array, exported)
        elif isinstance(value, _TextureResource):
            native_resource = native_graph.import_image(value._resident_view(), exported)
        elif isinstance(value, RenderTarget):
            if value._depth_texture is None:
                raise ValueError("RenderTarget graph resources require a depth attachment")
            native_resource = native_graph.import_image(value._depth_texture._resident_view(), exported)
        elif isinstance(value, (TensorStorage, TensorView, RawBuffer)):
            native_resource = native_graph.import_buffer(value._resident_buffer(), exported)
        else:
            raise TypeError("execution graph resources must be Tensor, RawBuffer, Texture, or depth RenderTarget")
        resource = GraphResource(native_resource.id, value, exported, self._owner, native_resource, self._generation)
        self._resources.append(resource)
        self._resource_by_identity[identity] = resource
        return resource

    def parameter(self, name: str) -> ExecutionParameter:
        native_graph = self._ensure_current()
        if not isinstance(name, str) or not name:
            raise ValueError("execution parameter name must be a non-empty string")
        if name in self._parameter_by_name:
            raise ValueError(f"execution parameter name {name!r} is already defined")
        parameter = ExecutionParameter(name, self._owner, native_graph.parameter(name), self._generation)
        self._parameters.append(parameter)
        self._parameter_by_name[name] = parameter
        return parameter

    def differentiable_input(
        self,
        name: str,
        value: GraphResource | ExecutionParameter | Any,
    ) -> GraphResource | ExecutionParameter:
        if not isinstance(name, str) or not name:
            raise ValueError("differentiable input name must be a non-empty string")
        if name in self._differentiable_inputs:
            raise ValueError(f"differentiable input name {name!r} is already defined")
        endpoint = value if isinstance(value, (GraphResource, ExecutionParameter)) else self.import_resource(value)
        if isinstance(endpoint, GraphResource):
            self._validate_resource(endpoint)
        else:
            self._validate_parameter(endpoint)
        if any(
            type(candidate) is type(endpoint) and candidate._native is endpoint._native
            for candidate in self._differentiable_inputs.values()
        ):
            raise ValueError("a graph value cannot be declared as two differentiable inputs")
        self._differentiable_inputs[name] = endpoint
        return endpoint

    def objective(self, name: str, value: GraphResource | Any) -> GraphResource:
        if not isinstance(name, str) or not name:
            raise ValueError("objective name must be a non-empty string")
        if name in self._objectives:
            raise ValueError(f"objective name {name!r} is already defined")
        resource = value if isinstance(value, GraphResource) else self.import_resource(value)
        self._validate_resource(resource)
        if any(candidate.id == resource.id for candidate in self._objectives.values()):
            raise ValueError("a graph resource cannot be declared as two objectives")
        self._objectives[name] = resource
        return resource

    def plan_autodiff_checkpoints(self, *, memory_budget: int) -> None:
        native_graph = self._ensure_current()
        if isinstance(memory_budget, bool) or not isinstance(memory_budget, int) or memory_budget < 0:
            raise ValueError("autodiff checkpoint memory budget must be a non-negative integer")
        native_graph.plan_autodiff_checkpoints(memory_budget)

    def add_pass(self, execution_pass: ExecutionPass) -> ExecutionPass:
        native_graph = self._ensure_current()
        if not isinstance(execution_pass, ExecutionPass):
            raise TypeError("execution graph accepts only ExecutionPass instances")
        if execution_pass in self._passes or execution_pass._graph not in {None, self}:
            raise ValueError("execution pass already belongs to a graph")
        execution_pass._graph = self
        if isinstance(execution_pass, RenderPass):
            execution_pass._native_pass = native_graph.add_render_pass(execution_pass.name, execution_pass)
        elif isinstance(execution_pass, ComputePass):
            execution_pass._native_pass = native_graph.add_compute_pass(execution_pass.name, execution_pass)
        else:
            raise TypeError("execution graph accepts only RenderPass or ComputePass instances")
        self._passes.append(execution_pass)
        execution_pass._sync_flags()
        self._sync_dependencies()
        return execution_pass

    def _sync_dependencies(self) -> None:
        for execution_pass in self._passes:
            if execution_pass._native_pass is None:
                continue
            for dependency in execution_pass._dependencies:
                if dependency._graph is self and dependency._native_pass is not None:
                    execution_pass._native_pass.depends_on(dependency._native_pass)

    def _validate_resource(self, resource: GraphResource) -> None:
        if (
            not isinstance(resource, GraphResource)
            or resource._owner is not self._owner
            or resource._generation != self._generation
            or resource.id != resource._native.id
            or not any(candidate is resource for candidate in self._resources)
        ):
            raise ValueError("resource does not belong to this execution graph")

    def _resolve(self, resource: GraphResource) -> Any:
        self._ensure_current()
        self._validate_resource(resource)
        return resource.value

    def _validate_parameter(self, parameter: ExecutionParameter) -> None:
        self._ensure_current()
        if (
            not isinstance(parameter, ExecutionParameter)
            or parameter._owner is not self._owner
            or parameter._generation != self._generation
            or not any(candidate is parameter for candidate in self._parameters)
        ):
            raise ValueError("parameter does not belong to this execution graph")

    def _validate_structure(self) -> None:
        if len({execution_pass.name for execution_pass in self._passes}) != len(self._passes):
            raise ValueError("execution graph pass names must be unique")
        for execution_pass in self._passes:
            for dependency in execution_pass._dependencies:
                if dependency._graph is not self:
                    raise ValueError(f"pass {execution_pass.name!r} depends on a pass outside this graph")
        for resource in self._resources:
            self._validate_resource(resource)

    def validate(self) -> None:
        native_graph = self._ensure_current()
        self._validate_structure()
        native_graph.validate()

    def compile(self) -> CompiledExecutionGraph:
        native_graph = self._ensure_current()
        self._validate_structure()
        self._sync_dependencies()
        for execution_pass in self._passes:
            execution_pass._sync_flags()
        native_graph.set_autodiff_endpoints(
            [
                (
                    name,
                    0 if isinstance(endpoint, GraphResource) else 1,
                    endpoint.id if isinstance(endpoint, GraphResource) else endpoint._native.id,
                )
                for name, endpoint in self._differentiable_inputs.items()
            ],
            [(name, 0, resource.id) for name, resource in self._objectives.items()],
        )
        native_plan = native_graph.compile()
        plan = CompiledExecutionGraph(self, native_plan)
        self._native_graph = None
        self._passes = []
        self._resources = []
        self._resource_by_identity = {}
        self._parameters = []
        self._parameter_by_name = {}
        self._differentiable_inputs = {}
        self._objectives = {}
        return plan


__all__ = [
    "ColorAttachmentUse",
    "CompiledBarrier",
    "CompiledExecutionGraph",
    "CompiledScope",
    "ComputeEncoder",
    "ComputePass",
    "DepthStencilAttachmentUse",
    "ExecutionBindings",
    "ExecutionGraph",
    "ExecutionParameter",
    "ExecutionPass",
    "ExecutionResources",
    "ExecutionSubmission",
    "GraphResource",
    "GraphicsEncoder",
    "LoadOperation",
    "PipelineInvocation",
    "RenderPass",
    "StoreOperation",
    "SubmissionState",
]
