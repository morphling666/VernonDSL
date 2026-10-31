"""Editor-visible declarations for compiler-recognized DSL intrinsics."""

from typing import Any, overload

import numpy as np

from .shader_contracts import DEVICE_ONLY_OPERATION_NAMES
from .types import Tensor


def _like(result: Any, *operands: Any) -> Any:
    """A Tensor result as the Tensor class of its operands, immutable like them; numpy returns a bare array."""

    if isinstance(result, np.ndarray) and result.ndim:
        for operand in operands:
            if isinstance(operand, Tensor):
                return type(operand)._result(result)
    return result


def _raise_device_only(name: str) -> Any:
    assert name in DEVICE_ONLY_OPERATION_NAMES
    raise TypeError(f"{name} is device-only and cannot execute on host")


def sin(value: Any) -> Any:
    return _like(np.sin(value), value)


def cos(value: Any) -> Any:
    return _like(np.cos(value), value)


def acos(value: Any) -> Any:
    return _like(np.arccos(value), value)


def atan2(y: Any, x: Any) -> Any:
    return _like(np.arctan2(y, x), y, x)


def exp(value: Any) -> Any:
    return _like(np.exp(value), value)


def log(value: Any) -> Any:
    return _like(np.log(value), value)


def sqrt(value: Any) -> Any:
    return _like(np.sqrt(value), value)


def floor(value: Any) -> Any:
    return _like(np.floor(value), value)


def abs(value: Any) -> Any:
    return _like(np.abs(value), value)


def dot(left: Any, right: Any) -> Any:
    return np.sum(np.asarray(left) * np.asarray(right))


def cross(left: Any, right: Any) -> Any:
    return _like(np.cross(left, right), left, right)


def normalize(value: Any) -> Any:
    return _like(np.asarray(value) / np.linalg.norm(value), value)


def norm(value: Any) -> Any:
    return np.linalg.norm(value)


def reflect(direction: Any, normal: Any) -> Any:
    return _like(np.asarray(direction) - 2 * np.dot(direction, normal) * np.asarray(normal), direction, normal)


def min(left: Any, right: Any) -> Any:
    return _like(np.minimum(left, right), left, right)


def max(left: Any, right: Any) -> Any:
    return _like(np.maximum(left, right), left, right)


def pow(left: Any, right: Any) -> Any:
    return _like(np.power(left, right), left, right)


def clamp(value: Any, minimum: Any, maximum: Any) -> Any:
    return _like(np.clip(value, minimum, maximum), value, minimum, maximum)


def matmul(left: Any, right: Any) -> Any:
    return np.matmul(left, right)


@overload
def texture_sample(texture: Any, coordinates: Any) -> Any: ...


@overload
def texture_sample(texture: Any, coordinates: Any, lod: Any) -> Any: ...


@overload
def texture_sample(texture: Any, sampler: Any, coordinates: Any) -> Any: ...


@overload
def texture_sample(texture: Any, sampler: Any, coordinates: Any, lod: Any) -> Any: ...


def texture_sample(texture: Any, *arguments: Any) -> Any:
    del texture, arguments
    return _raise_device_only("texture_sample")


def texture_size(texture: Any, lod: Any | None = None) -> Any:
    del texture, lod
    return _raise_device_only("texture_size")


def texture_load(texture: Any, coordinates: Any) -> Any:
    del texture, coordinates
    return _raise_device_only("texture_load")


def texture_store(texture: Any, coordinates: Any, value: Any) -> None:
    del texture, coordinates, value
    _raise_device_only("texture_store")


def workgroup_storage(element_type: Any, *, shape: tuple[int, ...]) -> Any:
    del element_type, shape
    return _raise_device_only("workgroup_storage")


def atomic_add(storage: Any, index: int | tuple[int, ...], value: Any) -> Any:
    del storage, index, value
    return _raise_device_only("atomic_add")


def atomic_min(storage: Any, index: int | tuple[int, ...], value: Any) -> Any:
    del storage, index, value
    return _raise_device_only("atomic_min")


def atomic_max(storage: Any, index: int | tuple[int, ...], value: Any) -> Any:
    del storage, index, value
    return _raise_device_only("atomic_max")


def atomic_exchange(storage: Any, index: int | tuple[int, ...], value: Any) -> Any:
    del storage, index, value
    return _raise_device_only("atomic_exchange")


def workgroup_barrier() -> None:
    _raise_device_only("workgroup_barrier")


def storage_barrier() -> None:
    _raise_device_only("storage_barrier")


def discard_fragment() -> None:
    _raise_device_only("discard_fragment")


def resolution() -> Any:
    return _raise_device_only("resolution")


def fragment_coord() -> Any:
    return _raise_device_only("fragment_coord")


def front_facing() -> Any:
    return _raise_device_only("front_facing")


def vertex_id() -> Any:
    return _raise_device_only("vertex_id")


def instance_id() -> Any:
    return _raise_device_only("instance_id")
