"""Numpy-style selection of Tensor values and TensorStorages, shared by the frontends and the Python runtime.

A selection has one item per axis. An integer index picks one position and drops the axis. A slice keeps the axis
with a length known at compile time, since Tensor values have static shapes:

- a slice of constants follows numpy (`1:3`, `:-1`, `::2`, `::-1`, `:`), on an axis of static extent;
- `a:b` with `b > a` is the *window* of `b - a` elements starting at `a`, a negative `a` counting from the end.
  A dynamic window is written `i:i+n` with `n` a positive literal.

The window reading is what lets Python mode, which sees only the integers, agree with compiled code for dynamic
bounds: where numpy would shorten a window that runs off the end, a Tensor value clamps its start (as JAX's
`dynamic_slice` does) and a TensorStorage raises. A constant window that does not fit is a compile error, so
numpy and the window reading agree on every constant slice that compiles.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AxisIndex:
    """An integer index: one position, and the axis is dropped."""

    node: ast.expr


@dataclass(frozen=True)
class AxisRange:
    """Positions `start + k * step` for `k < length`, all known at compile time."""

    start: int
    step: int
    length: int


@dataclass(frozen=True)
class AxisWindow:
    """`length` consecutive positions from a start known only at run time, or on an axis of `dyn` extent."""

    start: ast.expr
    length: int


AxisSelection = AxisIndex | AxisRange | AxisWindow


class SelectionError(ValueError):
    def __init__(self, node: ast.AST, message: str) -> None:
        super().__init__(message)
        self.node = node


def constant_integer(node: ast.expr | None) -> int | None:
    """The value of an integer literal, `-` applied to one included; None for anything else."""

    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and (operand := constant_integer(node.operand)) is not None
    ):
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def is_slicing(indices: Sequence[ast.expr], rank: int) -> bool:
    """Whether `indices` select more than one element: a slice, an `...`, or fewer indices than axes."""

    return len(indices) < rank or any(isinstance(index, ast.Slice) or _is_ellipsis(index) for index in indices)


def plan_selection(
    subscript: ast.expr, indices: Sequence[ast.expr], extents: Sequence[int | None]
) -> list[AxisSelection]:
    """One selection per axis of `extents` (None for a `dyn` axis). Missing trailing axes and `...` select whole
    axes."""

    ellipses = [index for index in indices if _is_ellipsis(index)]
    if len(ellipses) > 1:
        raise SelectionError(ellipses[1], "an index has at most one ...")
    explicit = len(indices) - len(ellipses)
    if explicit > len(extents):
        raise SelectionError(subscript, f"{explicit} indices for {len(extents)} axes; there is at most one per axis")
    expanded: list[ast.expr | None] = []
    for index in indices:
        if _is_ellipsis(index):
            expanded.extend([None] * (len(extents) - explicit))
        else:
            expanded.append(index)
    expanded.extend([None] * (len(extents) - len(expanded)))
    return [_plan_axis(subscript, index, extent) for index, extent in zip(expanded, extents, strict=True)]


def selection_shape(plan: Sequence[AxisSelection]) -> tuple[int, ...]:
    return tuple(axis.length for axis in plan if not isinstance(axis, AxisIndex))


def _plan_axis(subscript: ast.expr, index: ast.expr | None, extent: int | None) -> AxisSelection:
    if index is None:
        if extent is None:
            raise SelectionError(subscript, "a dyn axis is selected by an index or a window i:i+n, not as a whole")
        return AxisRange(0, 1, extent)
    if not isinstance(index, ast.Slice):
        return AxisIndex(index)
    step = 1 if index.step is None else constant_integer(index.step)
    if step is None or step == 0:
        raise SelectionError(index.step or index, "a slice step is a nonzero integer literal")
    lower, upper = constant_integer(index.lower), constant_integer(index.upper)
    if index.lower is not None and index.upper is not None and step == 1:
        if lower is not None and upper is not None and upper > lower:
            if extent is None:
                return AxisWindow(index.lower, upper - lower)
            start = window_start(lower, upper - lower, extent, clamp=False)
            if start is None:
                raise SelectionError(index, f"the window {lower}:{upper} does not fit an axis of extent {extent}")
            return AxisRange(start, 1, upper - lower)
        if lower is None:
            length = _window_length(index.lower, index.upper)
            if length is None:
                raise SelectionError(
                    index, "a slice with a dynamic start is a window i:i+n, with n a positive integer literal"
                )
            if extent is not None and length > extent:
                raise SelectionError(index, f"a window of {length} does not fit an axis of extent {extent}")
            return AxisWindow(index.lower, length)
    for bound in (index.lower, index.upper):
        if bound is not None and constant_integer(bound) is None:
            raise SelectionError(bound, "a slice bound is an integer literal, or the slice is a window i:i+n")
    if extent is None:
        raise SelectionError(index, "a dyn axis is selected by an index or a window i:i+n")
    positions = range(*slice(lower, upper, step).indices(extent))
    if not positions:
        raise SelectionError(index, f"the slice {ast.unparse(index)} selects nothing from an axis of extent {extent}")
    return AxisRange(positions.start, positions.step, len(positions))


def _window_length(start: ast.expr, stop: ast.expr) -> int | None:
    """`n` when `stop` is `start + n` or `n + start` for a positive literal `n`."""

    if not isinstance(stop, ast.BinOp) or not isinstance(stop.op, ast.Add):
        return None
    same = ast.dump(start)
    for base, offset in ((stop.left, stop.right), (stop.right, stop.left)):
        length = constant_integer(offset)
        if length is not None and length > 0 and ast.dump(base) == same:
            return length
    return None


def _is_ellipsis(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def window_start(start: int, length: int, extent: int, *, clamp: bool) -> int | None:
    """Where a window of `length` starting at `start` begins: a negative start counts from the end. Out of range,
    a clamped window moves to the nearest place it fits, and an unclamped one is None."""

    if start < 0:
        start += extent
    if clamp:
        return min(max(start, 0), extent - length)
    return start if 0 <= start <= extent - length else None


def runtime_key(key: Any, shape: Sequence[int], *, clamp: bool) -> Any:
    """`key` with each window (`slice(a, b)` with integers `b > a`) made explicit, for numpy to index with.

    Python mode sees only integers, so a constant window and a dynamic one read the same; a constant one that does
    not fit never compiles, so this only changes windows with a dynamic start."""

    items = key if isinstance(key, tuple) else (key,)
    if not any(isinstance(item, slice) for item in items):
        return key
    explicit = sum(1 for item in items if item is not Ellipsis)
    axis = 0
    resolved: list[Any] = []
    for item in items:
        if item is Ellipsis:
            resolved.append(item)
            axis += len(shape) - explicit
            continue
        if isinstance(item, slice) and axis < len(shape) and (window := _runtime_window(item)) is not None:
            start, length = window
            extent = shape[axis]
            begin = window_start(start, length, extent, clamp=clamp) if length <= extent else None
            if begin is None:
                raise IndexError(f"the window {item.start}:{item.stop} does not fit an axis of extent {extent}")
            item = slice(begin, begin + length)
        resolved.append(item)
        axis += 1
    return tuple(resolved) if isinstance(key, tuple) else resolved[0]


def _runtime_window(item: slice) -> tuple[int, int] | None:
    if item.step not in (None, 1) or item.start is None or item.stop is None:
        return None
    try:
        start, stop = operator.index(item.start), operator.index(item.stop)
    except TypeError:
        return None
    return (start, stop - start) if stop > start else None
