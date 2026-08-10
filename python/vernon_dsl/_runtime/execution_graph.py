from __future__ import annotations

import importlib
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .resources import RawBuffer, RenderTarget, TensorStorage, TensorView, Texture, _TextureResource


def _session_state() -> Any:
    return importlib.import_module("vernon_dsl._runtime.session")


def _resource_identity(value: Any) -> object:
    if isinstance(value, TensorView):
        return value.owner
    if isinstance(value, _TextureResource):
        return value._graph_identity()
    return value


class LoadOperation(Enum):
    CLEAR = "clear"
    PRESERVE = "preserve"
    DISCARD = "discard"


class StoreOperation(Enum):
    PRESERVE = "preserve"
    DISCARD = "discard"


@dataclass(frozen=True)
class GraphResource:
    id: int
    value: Any
    exported: bool
    _owner: object
    _native: Any
    _generation: int


@dataclass(frozen=True)
class ColorAttachmentUse:
    texture: Texture
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
        callback: Callable[[GraphicsEncoder | ComputeEncoder], None],
        declare_callback: Callable[[ExecutionPass], None] | None = None,
    ):
        if kind not in {"graphics", "compute"}:
            raise ValueError("pipeline invocation kind must be graphics or compute")
        self.kind = kind
        self._callback = callback
        self._declare_callback = declare_callback

    def declare(self, execution_pass: ExecutionPass) -> None:
        if self._declare_callback is not None:
            self._declare_callback(execution_pass)

    def encode(
        self,
        encoder: GraphicsEncoder | ComputeEncoder,
        resources: ExecutionResources | None = None,
    ) -> None:
        del resources
        if self.kind != encoder.kind:
            raise TypeError(f"{self.kind} invocation cannot encode into a {encoder.kind} encoder")
        self._callback(encoder)


class ExecutionResources:
    def __init__(self, graph: ExecutionGraph):
        self._graph = graph

    def resolve(self, resource: GraphResource) -> Any:
        return self._graph._resolve(resource)


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
        self._dependencies: list[ExecutionPass] = []
        self._graph_ref: weakref.ReferenceType[ExecutionGraph] | None = None
        self._native_pass: Any | None = None
        self._declaring = False
        self._first_in_scope = False
        self._last_in_scope = False

    @property
    def _graph(self) -> ExecutionGraph | None:
        return self._graph_ref() if self._graph_ref is not None else None

    @_graph.setter
    def _graph(self, value: ExecutionGraph | None) -> None:
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

    def _set_flag(self, field: str, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("execution pass flags must be bool")
        if getattr(self, field) == value:
            return
        setattr(self, field, value)
        if self._graph is not None:
            self._graph._dirty = True
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
        self._native_pass.set_flags(flags)

    def declare(self) -> None:
        raise NotImplementedError

    def depends_on(self, dependency: ExecutionPass) -> ExecutionPass:
        if not isinstance(dependency, ExecutionPass):
            raise TypeError("dependency must be an ExecutionPass")
        if dependency is self:
            raise ValueError(f"pass {self.name!r} cannot depend on itself")
        if dependency not in self._dependencies:
            self._dependencies.append(dependency)
            if self._graph is not None:
                self._graph._dirty = True
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

    def _native_execute(self, native_encoder: Any) -> None:
        if self._graph is None:
            raise RuntimeError("render pass execution requires a declared execution graph")
        encoder = GraphicsEncoder(
            self,
            native_encoder,
            first_in_scope=self._first_in_scope,
            last_in_scope=self._last_in_scope,
        )
        try:
            self.execute(encoder, ExecutionResources(self._graph))
        finally:
            encoder._native = None


class ComputePass(ExecutionPass):
    def execute(self, encoder: ComputeEncoder, resources: ExecutionResources) -> None:
        raise NotImplementedError

    def _native_execute(self, native_encoder: Any) -> None:
        if self._graph is None:
            raise RuntimeError("compute pass execution requires a declared execution graph")
        encoder = ComputeEncoder(native_encoder)
        try:
            self.execute(encoder, ExecutionResources(self._graph))
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


@dataclass(frozen=True)
class CompiledScope:
    kind: str
    passes: tuple[ExecutionPass, ...]
    barriers: tuple[CompiledBarrier, ...] = ()


class ExecutionGraph:
    def __init__(self):
        state = _session_state()
        self._generation = state._runtime_generation
        self._owner = object()
        self._passes: list[ExecutionPass] = []
        self._resources: list[GraphResource] = []
        self._resource_by_identity: dict[int, GraphResource] = {}
        self._schedule: tuple[ExecutionPass, ...] = ()
        self._scopes: tuple[CompiledScope, ...] = ()
        self._dirty = True
        self._native_graph = (
            state._native_runtime.create_execution_graph() if state._native_runtime is not None else None
        )
        state._runtime_children.add(self)

    def _dispose_native(self) -> None:
        for execution_pass in getattr(self, "_passes", ()):
            execution_pass._native_pass = None
        self._native_graph = None

    @property
    def schedule(self) -> tuple[ExecutionPass, ...]:
        return self._schedule

    @property
    def scopes(self) -> tuple[CompiledScope, ...]:
        return self._scopes

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
            native_resource = native_graph.import_host_buffer(identity, exported)
        elif isinstance(value, _TextureResource):
            native_resource = native_graph.import_image(value._resident_texture(), exported)
        elif isinstance(value, RenderTarget):
            image = value._resident_depth_attachment()
            if image is None:
                raise ValueError("RenderTarget graph resources require a depth attachment")
            native_resource = native_graph.import_image(image, exported)
        elif isinstance(value, (TensorStorage, TensorView, RawBuffer)):
            native_resource = native_graph.import_buffer(value._resident_buffer(), exported)
        else:
            raise TypeError("execution graph resources must be Tensor, RawBuffer, Texture, or depth RenderTarget")
        resource = GraphResource(native_resource.id, value, exported, self._owner, native_resource, self._generation)
        self._resources.append(resource)
        self._resource_by_identity[identity] = resource
        self._dirty = True
        return resource

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
        self._dirty = True
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

    def validate(self) -> None:
        native_graph = self._ensure_current()
        if len({execution_pass.name for execution_pass in self._passes}) != len(self._passes):
            raise ValueError("execution graph pass names must be unique")
        for execution_pass in self._passes:
            for dependency in execution_pass._dependencies:
                if dependency._graph is not self:
                    raise ValueError(f"pass {execution_pass.name!r} depends on a pass outside this graph")
        for resource in self._resources:
            self._validate_resource(resource)
        native_graph.validate()

    def compile(self) -> None:
        native_graph = self._ensure_current()
        self._sync_dependencies()
        for execution_pass in self._passes:
            execution_pass._sync_flags()
        native_graph.compile()
        schedule_indices = tuple(native_graph.schedule)
        self._schedule = tuple(self._passes[index] for index in schedule_indices)
        scopes: list[CompiledScope] = []
        for execution_pass in self._passes:
            execution_pass._first_in_scope = False
            execution_pass._last_in_scope = False
        for native_scope in native_graph.scopes:
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
                        )
                        for barrier in native_scope.barriers
                    ),
                )
            )
        self._scopes = tuple(scopes)
        self._dirty = False
        self.validate()

    def execute(self) -> None:
        native_graph = self._ensure_current()
        if self._dirty:
            self.compile()
        native_graph.execute()


__all__ = [
    "ColorAttachmentUse",
    "CompiledBarrier",
    "CompiledScope",
    "ComputeEncoder",
    "ComputePass",
    "DepthStencilAttachmentUse",
    "ExecutionGraph",
    "ExecutionPass",
    "ExecutionResources",
    "GraphResource",
    "GraphicsEncoder",
    "LoadOperation",
    "PipelineInvocation",
    "RenderPass",
    "StoreOperation",
]
