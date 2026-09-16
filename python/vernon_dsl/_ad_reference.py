"""Small reverse-mode CPU oracle for pure Value programs.

This is intentionally independent from Runtime and is used to validate
compiler-generated VJPs against host execution.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def _unbroadcast(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, extent in enumerate(shape):
        if extent == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value


class _Value:
    __array_priority__ = 1000

    def __init__(self, value: Any, parents: tuple[tuple["_Value", Callable[[np.ndarray], np.ndarray]], ...] = ()):
        self.value = np.asarray(value)
        self.parents = parents
        self.gradient = np.zeros_like(self.value, dtype=_gradient_dtype(self.value.dtype))

    @staticmethod
    def coerce(value: Any) -> "_Value":
        return value if isinstance(value, _Value) else _Value(value)

    def _binary(
        self,
        other: Any,
        forward: Callable[[np.ndarray, np.ndarray], np.ndarray],
        left: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        right: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> "_Value":
        other = self.coerce(other)
        output = forward(self.value, other.value)
        return _Value(
            output,
            (
                (self, lambda seed: _unbroadcast(left(seed, self.value, other.value), self.value.shape)),
                (other, lambda seed: _unbroadcast(right(seed, self.value, other.value), other.value.shape)),
            ),
        )

    def __add__(self, other: Any) -> "_Value":
        return self._binary(other, np.add, lambda g, _x, _y: g, lambda g, _x, _y: g)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_Value":
        return self._binary(other, np.subtract, lambda g, _x, _y: g, lambda g, _x, _y: -g)

    def __rsub__(self, other: Any) -> "_Value":
        return self.coerce(other).__sub__(self)

    def __mul__(self, other: Any) -> "_Value":
        return self._binary(other, np.multiply, lambda g, _x, y: g * y, lambda g, x, _y: g * x)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "_Value":
        return self._binary(
            other,
            np.divide,
            lambda g, _x, y: g / y,
            lambda g, x, y: -g * x / (y * y),
        )

    def __rtruediv__(self, other: Any) -> "_Value":
        return self.coerce(other).__truediv__(self)

    def __pow__(self, other: Any) -> "_Value":
        return self._binary(
            other,
            np.power,
            lambda g, x, y: g * y * np.power(x, y - 1),
            lambda g, x, y: g * np.power(x, y) * np.log(x),
        )

    def __rpow__(self, other: Any) -> "_Value":
        return self.coerce(other).__pow__(self)

    def __neg__(self) -> "_Value":
        return _Value(-self.value, ((self, lambda seed: -seed),))

    def __pos__(self) -> "_Value":
        return self

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        del copy
        return np.asarray(self.value, dtype=dtype)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__" or kwargs:
            return NotImplemented
        unary = {
            np.sin: (np.sin, np.cos),
            np.cos: (np.cos, lambda x: -np.sin(x)),
            np.exp: (np.exp, np.exp),
            np.log: (np.log, lambda x: 1 / x),
            np.sqrt: (np.sqrt, lambda x: 0.5 / np.sqrt(x)),
            np.absolute: (np.abs, np.sign),
            np.negative: (np.negative, lambda x: -np.ones_like(x)),
        }
        if ufunc in unary and len(inputs) == 1:
            value = self.coerce(inputs[0])
            forward, derivative = unary[ufunc]
            return _Value(forward(value.value), ((value, lambda seed: seed * derivative(value.value)),))
        binary = {
            np.add: "__add__",
            np.subtract: "__sub__",
            np.multiply: "__mul__",
            np.divide: "__truediv__",
            np.power: "__pow__",
        }
        operation = binary.get(ufunc)
        if operation is not None and len(inputs) == 2:
            return getattr(self.coerce(inputs[0]), operation)(inputs[1])
        return NotImplemented


def _gradient_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    if dtype == np.dtype(np.float16):
        return np.dtype(np.float32)
    if dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.float64)
    raise TypeError(f"{dtype} is not a differentiable floating Value")


def _leaves(value: Any) -> list[_Value]:
    if isinstance(value, _Value):
        return [value]
    if isinstance(value, tuple):
        return [leaf for item in value for leaf in _leaves(item)]
    if isinstance(value, Mapping):
        return [leaf for key in sorted(value) for leaf in _leaves(value[key])]
    raise TypeError("reference VJP outputs must contain only floating Scalar, Tensor, Tuple, or mapping leaves")


def _cotangent_leaves(value: Any, cotangent: Any | None) -> list[np.ndarray]:
    outputs = _leaves(value)
    if cotangent is None:
        if len(outputs) != 1 or outputs[0].value.shape:
            raise TypeError("an explicit cotangent is required for non-scalar or structured output")
        return [np.ones((), dtype=_gradient_dtype(outputs[0].value.dtype))]
    pairs: list[tuple[_Value, Any]] = []

    def visit(output: Any, seed: Any) -> None:
        if isinstance(output, _Value):
            if isinstance(seed, (tuple, Mapping)):
                raise TypeError("cotangent structure does not match program output")
            pairs.append((output, seed))
            return
        if isinstance(output, tuple):
            if not isinstance(seed, tuple) or len(output) != len(seed):
                raise TypeError("cotangent structure does not match program output")
            for output_item, seed_item in zip(output, seed, strict=True):
                visit(output_item, seed_item)
            return
        if isinstance(output, Mapping):
            if not isinstance(seed, Mapping) or set(output) != set(seed):
                raise TypeError("cotangent structure does not match program output")
            for key in sorted(output):
                visit(output[key], seed[key])
            return
        raise TypeError("reference VJP output contains an unsupported leaf")

    visit(value, cotangent)
    result = [np.asarray(seed, dtype=_gradient_dtype(output.value.dtype)) for output, seed in pairs]
    if any(seed.shape != output.value.shape for seed, output in zip(result, outputs, strict=True)):
        raise TypeError("cotangent shape does not match program output")
    return result


def _topological(outputs: list[_Value]) -> list[_Value]:
    ordered: list[_Value] = []
    seen: set[int] = set()

    def visit(value: _Value) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        for parent, _ in value.parents:
            visit(parent)
        ordered.append(value)

    for output in outputs:
        visit(output)
    return ordered


def reference_vjp(
    program: Callable[..., Any],
    bindings: Mapping[str, Any],
    wrt: tuple[str, ...],
) -> tuple[Any, Callable[[Any | None], dict[str, Any]]]:
    """Execute a pure host-callable Value program and construct its pullback."""

    function = getattr(program, "_function", getattr(program, "function", program))
    if not callable(function):
        raise TypeError("reference VJP program is not callable")
    signature = inspect.signature(function)
    signature.bind(**bindings)
    unknown = set(wrt) - set(bindings)
    if unknown:
        raise ValueError("unknown wrt input(s): " + ", ".join(sorted(unknown)))
    traced = dict(bindings)
    inputs: dict[str, _Value] = {}
    for name in wrt:
        value = np.asarray(bindings[name])
        _gradient_dtype(value.dtype)
        inputs[name] = _Value(value)
        traced[name] = inputs[name]
    traced_output = function(**traced)
    output_leaves = _leaves(traced_output)
    topology = _topological(output_leaves)

    def primal(value: Any) -> Any:
        if isinstance(value, _Value):
            result = value.value.copy()
            return result.item() if not result.shape else result
        if isinstance(value, tuple):
            return tuple(primal(item) for item in value)
        if isinstance(value, Mapping):
            return {key: primal(item) for key, item in value.items()}
        return value

    def pullback(cotangent: Any | None = None) -> dict[str, Any]:
        for value in topology:
            value.gradient.fill(0)
        for output, seed in zip(output_leaves, _cotangent_leaves(traced_output, cotangent), strict=True):
            output.gradient += seed
        for value in reversed(topology):
            for parent, rule in value.parents:
                parent.gradient += rule(value.gradient)
        return {
            name: gradient.item() if not gradient.shape else gradient.copy()
            for name, value in inputs.items()
            for gradient in (value.gradient,)
        }

    return primal(traced_output), pullback


__all__ = ["reference_vjp"]
