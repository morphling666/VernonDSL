"""Immutable render-target use values for graphics Program calls."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from enum import Enum, IntEnum, IntFlag
from typing import Any, Mapping

import numpy as np

from ._runtime.tensor import TensorStorage, TensorView
from ._runtime.texture import RenderTarget, TextureView


class LoadOperation(Enum):
    CLEAR = "clear"
    PRESERVE = "preserve"
    DISCARD = "discard"


class StoreOperation(Enum):
    PRESERVE = "preserve"
    DISCARD = "discard"


class PrimitiveTopology(Enum):
    triangles = ("triangles", 3)
    lines = ("lines", 2)
    points = ("points", 1)

    @property
    def vertices_per_primitive(self) -> int:
        return self.value[1]


triangles = PrimitiveTopology.triangles
lines = PrimitiveTopology.lines
points = PrimitiveTopology.points


class CompareOperation(IntEnum):
    NEVER = 0
    LESS = 1
    EQUAL = 2
    LESS_EQUAL = 3
    GREATER = 4
    NOT_EQUAL = 5
    GREATER_EQUAL = 6
    ALWAYS = 7


class StencilOperation(IntEnum):
    KEEP = 0
    ZERO = 1
    REPLACE = 2
    INCREMENT_CLAMP = 3
    DECREMENT_CLAMP = 4
    INVERT = 5
    INCREMENT_WRAP = 6
    DECREMENT_WRAP = 7


class CullMode(IntEnum):
    NONE = 0
    FRONT = 1
    BACK = 2


class FrontFace(IntEnum):
    COUNTER_CLOCKWISE = 0
    CLOCKWISE = 1


class BlendFactor(IntEnum):
    ZERO = 0
    ONE = 1
    SOURCE_COLOR = 2
    ONE_MINUS_SOURCE_COLOR = 3
    DESTINATION_COLOR = 4
    ONE_MINUS_DESTINATION_COLOR = 5
    SOURCE_ALPHA = 6
    ONE_MINUS_SOURCE_ALPHA = 7
    DESTINATION_ALPHA = 8
    ONE_MINUS_DESTINATION_ALPHA = 9


class BlendOperation(IntEnum):
    ADD = 0
    SUBTRACT = 1
    REVERSE_SUBTRACT = 2
    MINIMUM = 3
    MAXIMUM = 4


class ColorWrite(IntFlag):
    RED = 1 << 0
    GREEN = 1 << 1
    BLUE = 1 << 2
    ALPHA = 1 << 3
    ALL = RED | GREEN | BLUE | ALPHA


@dataclass(frozen=True)
class RasterizationState:
    cull_mode: CullMode = CullMode.NONE
    front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE
    depth_clamp: bool = False
    depth_bias_constant: float = 0.0
    depth_bias_slope: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.cull_mode, CullMode) or not isinstance(self.front_face, FrontFace):
            raise TypeError("rasterization state requires CullMode and FrontFace values")
        if not np.isfinite(self.depth_bias_constant) or not np.isfinite(self.depth_bias_slope):
            raise ValueError("depth bias values must be finite")


@dataclass(frozen=True)
class StencilFaceState:
    stencil_fail: StencilOperation = StencilOperation.KEEP
    depth_fail: StencilOperation = StencilOperation.KEEP
    pass_operation: StencilOperation = StencilOperation.KEEP
    compare: CompareOperation = CompareOperation.ALWAYS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stencil_fail, StencilOperation)
            or not isinstance(self.depth_fail, StencilOperation)
            or not isinstance(self.pass_operation, StencilOperation)
            or not isinstance(self.compare, CompareOperation)
        ):
            raise TypeError("stencil face state contains an invalid operation")


@dataclass(frozen=True)
class DepthStencilState:
    depth_test: bool = False
    depth_write: bool = False
    depth_compare: CompareOperation = CompareOperation.LESS
    stencil_test: bool = False
    front: StencilFaceState = StencilFaceState()
    back: StencilFaceState = StencilFaceState()
    stencil_read_mask: int = 0xFF
    stencil_write_mask: int = 0xFF

    def __post_init__(self) -> None:
        if not isinstance(self.depth_compare, CompareOperation):
            raise TypeError("depth_compare must be a CompareOperation")
        if not isinstance(self.front, StencilFaceState) or not isinstance(self.back, StencilFaceState):
            raise TypeError("front and back must be StencilFaceState values")
        if not 0 <= self.stencil_read_mask <= 0xFF or not 0 <= self.stencil_write_mask <= 0xFF:
            raise ValueError("stencil masks must be in [0, 255]")


@dataclass(frozen=True)
class ColorBlendState:
    enabled: bool = False
    source_color: BlendFactor = BlendFactor.ONE
    destination_color: BlendFactor = BlendFactor.ZERO
    color_operation: BlendOperation = BlendOperation.ADD
    source_alpha: BlendFactor = BlendFactor.ONE
    destination_alpha: BlendFactor = BlendFactor.ZERO
    alpha_operation: BlendOperation = BlendOperation.ADD
    write_mask: ColorWrite = ColorWrite.ALL

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_color, BlendFactor)
            or not isinstance(self.destination_color, BlendFactor)
            or not isinstance(self.source_alpha, BlendFactor)
            or not isinstance(self.destination_alpha, BlendFactor)
            or not isinstance(self.color_operation, BlendOperation)
            or not isinstance(self.alpha_operation, BlendOperation)
            or not isinstance(self.write_mask, ColorWrite)
        ):
            raise TypeError("color blend state contains an invalid enum value")


@dataclass(frozen=True)
class GraphicsPipelineState:
    topology: PrimitiveTopology = triangles
    rasterization: RasterizationState = RasterizationState()
    depth_stencil: DepthStencilState = DepthStencilState()
    color_blends: tuple[tuple[int, ColorBlendState], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.topology, PrimitiveTopology):
            raise TypeError("graphics topology must be a PrimitiveTopology")
        if not isinstance(self.rasterization, RasterizationState) or not isinstance(
            self.depth_stencil, DepthStencilState
        ):
            raise TypeError("graphics state requires typed rasterization and depth/stencil state")
        locations = tuple(location for location, _ in self.color_blends)
        if locations != tuple(sorted(set(locations))):
            raise ValueError("color blend locations must be unique and sorted")
        if any(location < 0 or not isinstance(blend, ColorBlendState) for location, blend in self.color_blends):
            raise ValueError("color blends require non-negative locations and ColorBlendState values")


def graphics_state(
    *,
    topology: PrimitiveTopology = triangles,
    rasterization: RasterizationState | None = None,
    depth_stencil: DepthStencilState | None = None,
    color_blends: Mapping[int, ColorBlendState] | None = None,
) -> GraphicsPipelineState:
    if not isinstance(topology, PrimitiveTopology):
        raise TypeError("graphics topology must be a PrimitiveTopology")
    return GraphicsPipelineState(
        topology,
        RasterizationState() if rasterization is None else rasterization,
        DepthStencilState() if depth_stencil is None else depth_stencil,
        tuple(sorted((color_blends or {}).items())),
    )


def _graphics_pipeline_state_data(value: GraphicsPipelineState) -> dict[str, Any]:
    """Return the canonical compiler-facing representation of immutable pipeline state."""

    if not isinstance(value, GraphicsPipelineState):
        raise TypeError("graphics pipeline state must be a GraphicsPipelineState")

    def normalize(member: Any) -> Any:
        if isinstance(member, IntFlag):
            return int(member)
        if isinstance(member, Enum):
            return member.name
        if is_dataclass(member) and not isinstance(member, type):
            return {field.name: normalize(getattr(member, field.name)) for field in dataclass_fields(member)}
        if isinstance(member, tuple):
            return [normalize(item) for item in member]
        return member

    return normalize(value)


@dataclass(frozen=True)
class AttachmentOperation:
    load: LoadOperation
    store: StoreOperation
    clear_value: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.load, LoadOperation) or not isinstance(self.store, StoreOperation):
            raise TypeError("attachment operations require LoadOperation and StoreOperation values")
        if self.load is not LoadOperation.CLEAR and self.clear_value is not None:
            raise ValueError("clear_value is valid only for a CLEAR load operation")


@dataclass(frozen=True)
class RenderPass:
    target: RenderTarget
    colors: tuple[tuple[int, AttachmentOperation], ...]
    depth: AttachmentOperation | None = None
    render_area: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        locations = tuple(location for location, _ in self.colors)
        if locations != tuple(sorted(set(locations))):
            raise ValueError("render color attachment locations must be unique and sorted")
        if self.render_area is not None and (
            len(self.render_area) != 4
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.render_area)
            or self.render_area[2] == 0
            or self.render_area[3] == 0
        ):
            raise ValueError("render_area must contain x, y, non-zero width, and non-zero height")
        for _, operation in self.colors:
            if not isinstance(operation, AttachmentOperation):
                raise TypeError("color attachments require AttachmentOperation values")
            if operation.load is LoadOperation.CLEAR and (
                not isinstance(operation.clear_value, (tuple, list, np.ndarray))
                or len(operation.clear_value) != 4
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(float(value))
                    for value in operation.clear_value
                )
            ):
                raise ValueError("color clear values must contain four finite numbers")
        if self.depth is not None:
            if not isinstance(self.depth, AttachmentOperation):
                raise TypeError("depth attachment requires an AttachmentOperation")
            if self.depth.load is LoadOperation.CLEAR and (
                not isinstance(self.depth.clear_value, (int, float))
                or isinstance(self.depth.clear_value, bool)
                or not 0.0 <= float(self.depth.clear_value) <= 1.0
            ):
                raise ValueError("depth clear value must be in [0, 1]")


def color_output(render_pass: RenderPass, *, location: int = 0) -> TextureView:
    if not isinstance(render_pass, RenderPass):
        raise TypeError("color_output requires a RenderPass")
    if not isinstance(location, int) or isinstance(location, bool) or location < 0:
        raise ValueError("color_output location must be a non-negative integer")
    colors = dict(render_pass.target._color_attachments())
    if location not in colors:
        raise ValueError(f"RenderPass has no color attachment at location {location}")
    return colors[location]


color_output.__vernon_attachment_output__ = "color"  # type: ignore[attr-defined]


def depth_output(render_pass: RenderPass) -> TextureView:
    if not isinstance(render_pass, RenderPass):
        raise TypeError("depth_output requires a RenderPass")
    depth = render_pass.target._depth_attachment()
    if depth is None:
        raise ValueError("RenderPass has no depth attachment")
    return depth


depth_output.__vernon_attachment_output__ = "depth"  # type: ignore[attr-defined]


def clear(value: Any, *, store: StoreOperation = StoreOperation.PRESERVE) -> AttachmentOperation:
    return AttachmentOperation(LoadOperation.CLEAR, store, value)


def load(*, store: StoreOperation = StoreOperation.PRESERVE) -> AttachmentOperation:
    return AttachmentOperation(LoadOperation.PRESERVE, store)


def discard(*, store: StoreOperation = StoreOperation.PRESERVE) -> AttachmentOperation:
    return AttachmentOperation(LoadOperation.DISCARD, store)


def preserve() -> AttachmentOperation:
    return load()


def render_pass(
    target: RenderTarget,
    *,
    color: AttachmentOperation | None = None,
    colors: Mapping[int, AttachmentOperation] | None = None,
    depth: AttachmentOperation | None = None,
    render_area: tuple[int, int, int, int] | None = None,
) -> RenderPass:
    if not isinstance(target, RenderTarget):
        raise TypeError("render target must be a RenderTarget")
    if color is not None and colors is not None:
        raise TypeError("render_pass accepts either color or colors, not both")
    selected = {0: color} if color is not None else dict(colors or {})
    if any(
        not isinstance(location, int)
        or isinstance(location, bool)
        or location < 0
        or not isinstance(operation, AttachmentOperation)
        for location, operation in selected.items()
    ):
        raise TypeError("render_pass colors must map non-negative locations to attachment operations")
    if depth is not None and not isinstance(depth, AttachmentOperation):
        raise TypeError("render_pass depth must be an attachment operation")
    return RenderPass(target, tuple(sorted(selected.items())), depth, render_area)


def clear_depth(
    value: float,
    *,
    store: StoreOperation = StoreOperation.PRESERVE,
) -> AttachmentOperation:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("depth clear value must be in [0, 1]")
    return AttachmentOperation(LoadOperation.CLEAR, store, float(value))


@dataclass(frozen=True)
class IndexBufferView:
    view: TensorView

    def __post_init__(self) -> None:
        if not isinstance(self.view, TensorView):
            raise TypeError("IndexBufferView requires a TensorView")
        if self.view.dtype != np.dtype(np.uint32) or len(self.view.shape) != 1 or not self.view.shape[0]:
            raise TypeError("index buffer must be a non-empty rank-one u32 TensorView")
        if self.view.layout.element_strides != (1,):
            raise ValueError("index buffer must be contiguous")
        if self.view.access == "write":
            raise ValueError("index buffer must be readable")

    @property
    def count(self) -> int:
        return self.view.shape[0]


def index_buffer(value: TensorStorage | TensorView) -> IndexBufferView:
    if isinstance(value, TensorStorage):
        value = value.view(access="read")
    elif isinstance(value, TensorView):
        value = value._with_access("read")
    else:
        raise TypeError("index_buffer requires TensorStorage or TensorView")
    return IndexBufferView(value)


@dataclass(frozen=True)
class DrawCommand:
    vertex_count: int | None = None
    index_buffer: IndexBufferView | None = None
    instance_count: int = 1

    def __post_init__(self) -> None:
        if self.index_buffer is not None and not isinstance(self.index_buffer, IndexBufferView):
            raise TypeError("index_buffer must be an IndexBufferView")
        if self.vertex_count is not None and (
            not isinstance(self.vertex_count, int) or isinstance(self.vertex_count, bool) or self.vertex_count <= 0
        ):
            raise ValueError("vertex_count must be a positive integer")
        if (
            not isinstance(self.instance_count, int)
            or isinstance(self.instance_count, bool)
            or self.instance_count <= 0
        ):
            raise ValueError("instance_count must be a positive integer")
        if self.index_buffer is not None and self.vertex_count is not None:
            raise ValueError("indexed draw cannot also specify vertex_count")


def draw(
    *,
    vertex_count: int | None = None,
    index_buffer: IndexBufferView | None = None,
    instance_count: int = 1,
) -> DrawCommand:
    return DrawCommand(vertex_count, index_buffer, instance_count)


@dataclass(frozen=True)
class DynamicState:
    viewport: tuple[int, int, int, int] | None = None
    scissor: tuple[int, int, int, int] | None = None
    stencil_reference: int = 0

    def __post_init__(self) -> None:
        for name, rectangle in (("viewport", self.viewport), ("scissor", self.scissor)):
            if rectangle is not None and (
                len(rectangle) != 4
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in rectangle)
                or rectangle[2] == 0
                or rectangle[3] == 0
            ):
                raise ValueError(f"{name} must contain x, y, non-zero width, and non-zero height")
        if not isinstance(self.stencil_reference, int) or not 0 <= self.stencil_reference <= 0xFF:
            raise ValueError("stencil_reference must be in [0, 255]")


def dynamic_state(
    *,
    viewport: tuple[int, int, int, int] | None = None,
    scissor: tuple[int, int, int, int] | None = None,
    stencil_reference: int = 0,
) -> DynamicState:
    return DynamicState(viewport, scissor, stencil_reference)


__all__ = [
    "AttachmentOperation",
    "BlendFactor",
    "BlendOperation",
    "ColorBlendState",
    "ColorWrite",
    "CompareOperation",
    "CullMode",
    "DepthStencilState",
    "DrawCommand",
    "DynamicState",
    "FrontFace",
    "GraphicsPipelineState",
    "IndexBufferView",
    "LoadOperation",
    "PrimitiveTopology",
    "RasterizationState",
    "RenderPass",
    "StencilFaceState",
    "StencilOperation",
    "StoreOperation",
    "clear",
    "clear_depth",
    "color_output",
    "discard",
    "depth_output",
    "draw",
    "dynamic_state",
    "graphics_state",
    "index_buffer",
    "lines",
    "load",
    "points",
    "preserve",
    "render_pass",
    "triangles",
]
