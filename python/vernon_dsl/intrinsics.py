"""Editor-visible declarations for compiler-recognized DSL intrinsics."""

from typing import Any

import numpy as np


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
    return np.asarray(direction) - 2 * np.dot(direction,
                                              normal) * np.asarray(normal)


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


def texture_sample(texture: Any, sampler: Any, coordinates: Any) -> Any:
    del texture, sampler, coordinates
    raise TypeError("texture_sample is device-only and cannot execute on host")
