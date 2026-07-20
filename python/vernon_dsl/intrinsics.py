"""Editor-visible declarations for compiler-recognized DSL intrinsics."""

from typing import Any


def sin(value: Any) -> Any:
    ...


def cos(value: Any) -> Any:
    ...


def exp(value: Any) -> Any:
    ...


def log(value: Any) -> Any:
    ...


def sqrt(value: Any) -> Any:
    ...


def abs(value: Any) -> Any:
    ...


def dot(left: Any, right: Any) -> Any:
    ...


def cross(left: Any, right: Any) -> Any:
    ...


def normalize(value: Any) -> Any:
    ...


def norm(value: Any) -> Any:
    ...


def reflect(direction: Any, normal: Any) -> Any:
    ...


def min(left: Any, right: Any) -> Any:
    ...


def max(left: Any, right: Any) -> Any:
    ...


def pow(left: Any, right: Any) -> Any:
    ...


def clamp(value: Any, minimum: Any, maximum: Any) -> Any:
    ...


def matmul(left: Any, right: Any) -> Any:
    ...


def texture_sample(texture: Any, sampler: Any, coordinates: Any) -> Any:
    ...
