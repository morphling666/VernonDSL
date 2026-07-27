from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .resources import RenderTarget, Texture


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


@dataclass(frozen=True)
class _ResourceUse:
    resource: GraphResource
    access: str


class PipelineInvocation:
    """Prepared graphics or compute work that encodes into a graph-provided encoder."""

    def __init__(self, kind: str, callback: Callable[[GraphicsEncoder | ComputeEncoder], None]):
        if kind not in {"graphics", "compute"}:
            raise ValueError("pipeline invocation kind must be graphics or compute")
        self.kind = kind
        self._callback = callback

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
    def __init__(self, resources: tuple[GraphResource, ...]):
        self._resources = resources

    def resolve(self, resource: GraphResource) -> Any:
        if not isinstance(resource, GraphResource) or resource.id >= len(self._resources):
            raise ValueError("resource does not belong to this execution graph")
        current = self._resources[resource.id]
        if current is not resource:
            raise ValueError("resource does not belong to this execution graph")
        return current.value


class GraphicsEncoder:
    kind = "graphics"

    def __init__(self, render_pass: RenderPass, *, first_in_scope: bool, last_in_scope: bool):
        self._pass = render_pass
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
        self.never_cull = False
        self.side_effect = False
        self.no_merge = False
        self._dependencies: list[ExecutionPass] = []
        self._uses: list[_ResourceUse] = []
        self._graph: ExecutionGraph | None = None

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
        return self

    def read(self, resource: GraphResource | Any) -> None:
        self._uses.append(_ResourceUse(self._resource(resource), "read"))

    def write(self, resource: GraphResource | Any) -> None:
        self._uses.append(_ResourceUse(self._resource(resource), "write"))

    def read_write(self, resource: GraphResource | Any) -> None:
        self._uses.append(_ResourceUse(self._resource(resource), "read_write"))

    def _resource(self, resource: GraphResource | Any) -> GraphResource:
        if self._graph is None:
            raise RuntimeError("resource declarations are only valid while the pass is being declared")
        return resource if isinstance(resource, GraphResource) else self._graph.import_resource(resource)

    def _reset_declaration(self, graph: ExecutionGraph) -> None:
        self._graph = graph
        self._uses.clear()


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
        if target._depth_format is not None:
            self.depth(DepthStencilAttachmentUse(target, depth_load, depth_store, clear_depth))

    def color(self, location: int, attachment: ColorAttachmentUse) -> None:
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
        self._colors[location] = attachment
        self.write(attachment.texture)

    def depth(self, attachment: DepthStencilAttachmentUse) -> None:
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
        if attachment.read_only_depth and attachment.read_only_stencil:
            self.read(attachment.target)
        else:
            self.write(attachment.target)

    def render_area(self, x: int, y: int, width: int, height: int) -> None:
        self._render_area = _checked_rectangle("render area", x, y, width, height)

    def _reset_declaration(self, graph: ExecutionGraph) -> None:
        super()._reset_declaration(graph)
        self._target = None
        self._colors.clear()
        self._depth = None
        self._render_area = None


class ComputePass(ExecutionPass):
    def execute(self, encoder: ComputeEncoder, resources: ExecutionResources) -> None:
        raise NotImplementedError


def _checked_clear_color(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(not isinstance(component, (int, float)) for component in value)
    ):
        raise ValueError("clear color must contain four numeric components")
    return tuple(float(component) for component in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class CompiledBarrier:
    resource: GraphResource
    before: str
    after: str


@dataclass(frozen=True)
class CompiledScope:
    kind: str
    passes: tuple[ExecutionPass, ...]
    barriers: tuple[CompiledBarrier, ...] = ()


class ExecutionGraph:
    def __init__(self):
        self._passes: list[ExecutionPass] = []
        self._resources: list[GraphResource] = []
        self._resource_by_identity: dict[int, GraphResource] = {}
        self._schedule: tuple[ExecutionPass, ...] = ()
        self._scopes: tuple[CompiledScope, ...] = ()
        self._dirty = True

    @property
    def schedule(self) -> tuple[ExecutionPass, ...]:
        return self._schedule

    @property
    def scopes(self) -> tuple[CompiledScope, ...]:
        return self._scopes

    def import_resource(self, value: Any, *, exported: bool = True) -> GraphResource:
        identity = id(value)
        existing = self._resource_by_identity.get(identity)
        if existing is not None:
            if existing.value is not value:
                raise RuntimeError("Python resource identity collision")
            return existing
        resource = GraphResource(len(self._resources), value, exported)
        self._resources.append(resource)
        self._resource_by_identity[identity] = resource
        self._dirty = True
        return resource

    def add_pass(self, execution_pass: ExecutionPass) -> ExecutionPass:
        if not isinstance(execution_pass, ExecutionPass):
            raise TypeError("execution graph accepts only ExecutionPass instances")
        if execution_pass in self._passes or execution_pass._graph not in {None, self}:
            raise ValueError("execution pass already belongs to a graph")
        self._passes.append(execution_pass)
        execution_pass._graph = self
        self._dirty = True
        return execution_pass

    def validate(self) -> None:
        if len({execution_pass.name for execution_pass in self._passes}) != len(self._passes):
            raise ValueError("execution graph pass names must be unique")
        owned = set(self._passes)
        for execution_pass in self._passes:
            for dependency in execution_pass._dependencies:
                if dependency not in owned:
                    raise ValueError(f"pass {execution_pass.name!r} depends on a pass outside this graph")
            for use in execution_pass._uses:
                if use.resource.id >= len(self._resources) or self._resources[use.resource.id] is not use.resource:
                    raise ValueError(f"pass {execution_pass.name!r} uses a resource outside this graph")
        for scope in self._scopes:
            if not scope.passes or any(execution_pass not in owned for execution_pass in scope.passes):
                raise RuntimeError("compiled execution scope is invalid")

    def compile(self) -> None:
        for execution_pass in self._passes:
            execution_pass._reset_declaration(self)
            execution_pass.declare()
            execution_pass._graph = self
            if isinstance(execution_pass, RenderPass) and execution_pass._target is None:
                raise ValueError(f"render pass {execution_pass.name!r} declares no attachments")
        self.validate()
        count = len(self._passes)
        indices = {execution_pass: index for index, execution_pass in enumerate(self._passes)}
        edges: list[set[int]] = [set() for _ in self._passes]
        reverse: list[set[int]] = [set() for _ in self._passes]

        def add_edge(before: int, after: int) -> None:
            edges[before].add(after)
            reverse[after].add(before)

        for index, execution_pass in enumerate(self._passes):
            for dependency in execution_pass._dependencies:
                add_edge(indices[dependency], index)
        for before in range(count):
            for after in range(before + 1, count):
                for left in self._passes[before]._uses:
                    for right in self._passes[after]._uses:
                        if left.resource is right.resource and (left.access != "read" or right.access != "read"):
                            add_edge(before, after)

        roots = {
            index
            for index, execution_pass in enumerate(self._passes)
            if execution_pass.never_cull
            or execution_pass.side_effect
            or any(use.access != "read" and use.resource.exported for use in execution_pass._uses)
        }
        live = set(roots)
        pending = list(roots)
        while pending:
            current = pending.pop()
            for dependency in reverse[current] - live:
                live.add(dependency)
                pending.append(dependency)
        indegree = {index: len(reverse[index] & live) for index in live}
        ready = [index for index in live if indegree[index] == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            current = heapq.heappop(ready)
            order.append(current)
            for following in edges[current] & live:
                indegree[following] -= 1
                if indegree[following] == 0:
                    heapq.heappush(ready, following)
        if len(order) != len(live):
            cyclic = ", ".join(self._passes[index].name for index in sorted(live) if indegree[index])
            raise ValueError(f"execution graph contains a dependency cycle involving: {cyclic}")
        self._schedule = tuple(self._passes[index] for index in order)
        scopes: list[CompiledScope] = []
        for execution_pass in self._schedule:
            kind = "render" if isinstance(execution_pass, RenderPass) else "compute"
            if (
                kind == "render"
                and scopes
                and scopes[-1].kind == "render"
                and _can_fuse(scopes[-1].passes[-1], execution_pass)
            ):
                scopes[-1] = CompiledScope("render", (*scopes[-1].passes, execution_pass))
            else:
                scopes.append(CompiledScope(kind, (execution_pass,)))
        last_uses: dict[int, _ResourceUse] = {}
        compiled_scopes: list[CompiledScope] = []
        for scope in scopes:
            first_uses: dict[int, _ResourceUse] = {}
            final_uses: dict[int, _ResourceUse] = {}
            for execution_pass in scope.passes:
                for use in execution_pass._uses:
                    first_uses.setdefault(use.resource.id, use)
                    final_uses[use.resource.id] = use
            barriers = tuple(
                CompiledBarrier(use.resource, last_uses[resource_id].access, use.access)
                for resource_id, use in first_uses.items()
                if resource_id in last_uses and (last_uses[resource_id].access != "read" or use.access != "read")
            )
            compiled_scopes.append(CompiledScope(scope.kind, scope.passes, barriers))
            last_uses.update(final_uses)
        self._scopes = tuple(compiled_scopes)
        self._dirty = False
        self.validate()

    def execute(self) -> None:
        if self._dirty:
            self.compile()
        resources = ExecutionResources(tuple(self._resources))
        for scope in self._scopes:
            if scope.kind == "compute":
                compute_pass = scope.passes[0]
                assert isinstance(compute_pass, ComputePass)
                compute_pass.execute(ComputeEncoder(), resources)
                continue
            for index, execution_pass in enumerate(scope.passes):
                assert isinstance(execution_pass, RenderPass)
                encoder = GraphicsEncoder(
                    execution_pass,
                    first_in_scope=index == 0,
                    last_in_scope=index == len(scope.passes) - 1,
                )
                execution_pass.execute(encoder, resources)


def _can_fuse(left: ExecutionPass, right: ExecutionPass) -> bool:
    if not isinstance(left, RenderPass) or not isinstance(right, RenderPass) or left.no_merge or right.no_merge:
        return False
    if left._target is None or right._target is None:
        return False
    left_colors = tuple((location, id(use.texture)) for location, use in sorted(left._colors.items()))
    right_colors = tuple((location, id(use.texture)) for location, use in sorted(right._colors.items()))
    return (
        left_colors == right_colors
        and bool(left._depth) == bool(right._depth)
        and (left._depth is None or left._depth.target is right._depth.target)
        and left._target.shape == right._target.shape
        and left._render_area == right._render_area
    )


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
