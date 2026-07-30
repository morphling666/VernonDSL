from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest


def broadcast_shape(*shapes: Sequence[int]) -> tuple[int, ...] | None:
    """Return NumPy's broadcast result for positive static shapes."""
    if any(extent <= 0 for shape in shapes for extent in shape):
        return None
    result: list[int] = []
    for dimensions in zip_longest(*(reversed(tuple(shape)) for shape in shapes), fillvalue=1):
        extent = max(dimensions)
        if any(value not in {1, extent} for value in dimensions):
            return None
        result.append(extent)
    return tuple(reversed(result))


def matmul_shape(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...] | None:
    """Return NumPy's matmul result shape for positive static shapes."""
    left_shape = tuple(left)
    right_shape = tuple(right)
    if not left_shape or not right_shape or any(extent <= 0 for extent in (*left_shape, *right_shape)):
        return None

    left_vector = len(left_shape) == 1
    right_vector = len(right_shape) == 1
    left_matrix = (1, left_shape[0]) if left_vector else left_shape[-2:]
    right_matrix = (right_shape[0], 1) if right_vector else right_shape[-2:]
    if left_matrix[1] != right_matrix[0]:
        return None

    left_batch = () if left_vector else left_shape[:-2]
    right_batch = () if right_vector else right_shape[:-2]
    batch = broadcast_shape(left_batch, right_batch)
    if batch is None:
        return None

    result = (*batch, left_matrix[0], right_matrix[1])
    if left_vector:
        result = result[:-2] + result[-1:]
    if right_vector:
        result = result[:-1]
    return result


__all__ = ["broadcast_shape", "matmul_shape"]
