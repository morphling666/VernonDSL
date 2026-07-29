"""Editor-visible declarations for compiler-recognized DSL intrinsics."""

from typing import Any, overload

import numpy as np

from .shader_contracts import DEVICE_ONLY_OPERATION_NAMES


def _raise_device_only(name: str) -> Any:
    assert name in DEVICE_ONLY_OPERATION_NAMES
    raise TypeError(f"{name} is device-only and cannot execute on host")


def sin(value: Any) -> Any:
    return np.sin(value)


def cos(value: Any) -> Any:
    return np.cos(value)


def exp(value: Any) -> Any:
    return np.exp(value)


def log(value: Any) -> Any:
    return np.log(value)


def sqrt(value: Any) -> Any:
    return np.sqrt(value)


def abs(value: Any) -> Any:
    return np.abs(value)


def dot(left: Any, right: Any) -> Any:
    return np.dot(left, right)


def cross(left: Any, right: Any) -> Any:
    return np.cross(left, right)


def normalize(value: Any) -> Any:
    return np.asarray(value) / np.linalg.norm(value)


def norm(value: Any) -> Any:
    return np.linalg.norm(value)


def reflect(direction: Any, normal: Any) -> Any:
    return np.asarray(direction) - 2 * np.dot(direction, normal) * np.asarray(normal)


def min(left: Any, right: Any) -> Any:
    return np.minimum(left, right)


def max(left: Any, right: Any) -> Any:
    return np.maximum(left, right)


def pow(left: Any, right: Any) -> Any:
    return np.power(left, right)


def clamp(value: Any, minimum: Any, maximum: Any) -> Any:
    return np.clip(value, minimum, maximum)


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
