"""Immutable render-target use values for graphics Program calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._runtime.execution_graph import LoadOperation, StoreOperation
from ._runtime.texture import RenderTarget


@dataclass(frozen=True)
class AttachmentOperation:
    load: LoadOperation
    store: StoreOperation
    clear_value: Any = None


@dataclass(frozen=True)
class RenderTargetUse:
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
        ):
            raise ValueError("render_area must contain four non-negative integers")


def clear(value: Any) -> AttachmentOperation:
    return AttachmentOperation(LoadOperation.CLEAR, StoreOperation.PRESERVE, value)


def load() -> AttachmentOperation:
    return AttachmentOperation(LoadOperation.PRESERVE, StoreOperation.PRESERVE)


def preserve() -> AttachmentOperation:
    return AttachmentOperation(LoadOperation.PRESERVE, StoreOperation.PRESERVE)


def render(
    target: RenderTarget,
    *,
    color: AttachmentOperation | None = None,
    colors: Mapping[int, AttachmentOperation] | None = None,
    depth: AttachmentOperation | None = None,
    render_area: tuple[int, int, int, int] | None = None,
) -> RenderTargetUse:
    if not isinstance(target, RenderTarget):
        raise TypeError("render target must be a RenderTarget")
    if color is not None and colors is not None:
        raise TypeError("render accepts either color or colors, not both")
    selected = {0: color} if color is not None else dict(colors or {})
    if any(
        not isinstance(location, int)
        or isinstance(location, bool)
        or location < 0
        or not isinstance(operation, AttachmentOperation)
        for location, operation in selected.items()
    ):
        raise TypeError("render colors must map non-negative locations to attachment operations")
    if depth is not None and not isinstance(depth, AttachmentOperation):
        raise TypeError("render depth must be an attachment operation")
    return RenderTargetUse(target, tuple(sorted(selected.items())), depth, render_area)


__all__ = ["AttachmentOperation", "RenderTargetUse", "clear", "load", "preserve", "render"]
