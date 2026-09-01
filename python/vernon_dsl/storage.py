"""Host-owning TensorStorage allocation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._runtime.tensor import TensorStorage, TensorView
from ._runtime.tensor import _logical_collection_shape as _resource_collection_shape


def _logical_collection_shape(values: Any, element_type: Any) -> tuple[int, ...]:
    return _resource_collection_shape(values, element_type)


def empty_like(value: TensorStorage | TensorView) -> TensorStorage:
    return TensorStorage.empty_like(value)


def zeros_like(value: TensorStorage | TensorView) -> TensorStorage:
    return TensorStorage.zeros_like(value)


def zeros(*, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
    return TensorStorage.zeros(dtype=dtype, shape=shape)


def empty(*, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
    return TensorStorage.empty(dtype=dtype, shape=shape)


def from_numpy(array: np.ndarray[Any, Any]) -> TensorStorage:
    return TensorStorage.from_numpy(array)


def from_values(values: Any, *, dtype: Any) -> TensorStorage:
    return TensorStorage.from_values(values, dtype=dtype)


def tangent_zeros(*, dtype: Any, shape: tuple[int, ...]) -> TensorStorage:
    return TensorStorage.tangent_zeros(dtype=dtype, shape=shape)


__all__ = ["empty", "empty_like", "from_numpy", "from_values", "tangent_zeros", "zeros", "zeros_like"]
