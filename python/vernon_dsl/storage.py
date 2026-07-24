"""Host-owning TensorStorage allocation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._runtime.resources import TensorStorage


def zeros(*, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
    return TensorStorage.zeros(dtype=dtype, shape=shape)


def empty(*, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
    return TensorStorage.empty(dtype=dtype, shape=shape)


def from_numpy(array: np.ndarray[Any, Any]) -> TensorStorage:
    return TensorStorage.from_numpy(array)


def from_values(values: Any, *, dtype: Any) -> TensorStorage:
    return TensorStorage.from_values(values, dtype=dtype)


__all__ = ["empty", "from_numpy", "from_values", "zeros"]
