"""Python DSL providers for compiler-planned Program kernel regions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import vernon_dsl as vd

from .._runtime.operation_implementations import ImplementationUnavailable, program_add_invocation
from .._runtime.resources import TensorStorage
from .model import ProgramImplementation


class ProgramDslProvider(Protocol):
    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None: ...


class CapturedDslProvider:
    def __init__(self, implementations: Sequence[ProgramImplementation]):
        self._implementations = {implementation.callee: implementation for implementation in implementations}

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        del values
        hint = request.get("implementation_hint")
        return self._implementations.get(hint) if isinstance(hint, str) else None


class DirectKernelDslProvider:
    """Returns the original frontend implementation for a planned direct Kernel."""

    def __init__(self, mlir: str, entry: str):
        self._implementation = ProgramImplementation(entry, entry, "compute", mlir)

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        del values
        if request.get("kind") != "compute" or request.get("implementation_hint") != self._implementation.callee:
            return None
        return self._implementation


class CapturedVjpDslProvider:
    def __init__(self, implementations: Sequence[ProgramImplementation], native: Any):
        self._implementations = {implementation.callee: implementation for implementation in implementations}
        self._native = native

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        del values
        hint = request.get("implementation_hint")
        if not isinstance(hint, str) or not hint.endswith(".vjp"):
            return None
        primal = self._implementations.get(hint.removesuffix(".vjp"))
        if primal is None:
            return None
        raw_bindings = request.get("bindings")
        raw_results = request.get("results")
        if not isinstance(raw_bindings, list) or not isinstance(raw_results, list):
            raise ImplementationUnavailable("Program VJP request has no canonical ABI bindings")
        result_ids = {value for value in raw_results if isinstance(value, int)}
        bindings = [
            (binding.get("parameter"), binding.get("value"))
            for binding in raw_bindings
            if isinstance(binding, Mapping)
            and isinstance(binding.get("parameter"), str)
            and isinstance(binding.get("value"), int)
        ]
        wrt = tuple(name for name, value in bindings if value in result_ids)
        outputs = tuple(name.removeprefix("cotangent.") for name, _ in bindings if name.startswith("cotangent."))
        if not wrt or not outputs:
            raise ImplementationUnavailable("Program VJP request has no gradient or cotangent bindings")
        identity = hashlib.sha256(
            (primal.mlir + "\0" + primal.entry + "\0" + "\0".join((*wrt, *outputs))).encode()
        ).hexdigest()
        forward_symbol = f"vernon_program_{identity[:20]}_forward"
        backward_symbol = f"vernon_program_{identity[:20]}_backward"
        transformed = self._native._build_structured_vjp(
            primal.mlir,
            primal.entry,
            list(wrt),
            list(outputs),
            forward_symbol,
            backward_symbol,
        )
        profiles = transformed.profiles(identity)
        return ProgramImplementation(hint, backward_symbol, "compute", str(profiles["backward"]))


BuiltinLowerer = Callable[
    [Mapping[str, Any], Mapping[int, Mapping[str, Any]]],
    ProgramImplementation | None,
]


class BuiltinDslProvider:
    """Provider for built-in Program semantic operations."""

    def __init__(self, lowerers: Sequence[BuiltinLowerer] = ()):
        self._lowerers = (*lowerers, _lower_builtin_add)

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        for lowerer in self._lowerers:
            implementation = lowerer(request, values)
            if implementation is not None:
                return implementation
        return None


class ProviderChain:
    def __init__(self, providers: Sequence[ProgramDslProvider]):
        self._providers = tuple(providers)

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        for provider in self._providers:
            implementation = provider.lower(request, values)
            if implementation is not None:
                return implementation
        return None


def _request_value(
    bindings: Mapping[str, int],
    values: Mapping[int, Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any]:
    value_id = bindings.get(name)
    value = values.get(value_id) if value_id is not None else None
    if value is None:
        raise ImplementationUnavailable(f"Program request has no {name!r} value")
    return value


def _lower_builtin_add(
    request: Mapping[str, Any],
    values: Mapping[int, Mapping[str, Any]],
) -> ProgramImplementation | None:
    if request.get("implementation_hint") != "vernon.builtin.add":
        return None
    raw_bindings = request.get("bindings")
    if not isinstance(raw_bindings, list):
        raise ImplementationUnavailable("built-in Add request has no ABI bindings")
    bindings = {
        binding["parameter"]: binding["value"]
        for binding in raw_bindings
        if isinstance(binding, Mapping)
        and isinstance(binding.get("parameter"), str)
        and isinstance(binding.get("value"), int)
    }
    reflected = {name: _request_value(bindings, values, name) for name in ("output", "left", "right")}
    shapes = {tuple(value.get("shape", ())) for value in reflected.values()}
    dtypes = {value.get("dtype") for value in reflected.values()}
    if len(shapes) != 1 or dtypes != {"f32"}:
        raise ImplementationUnavailable("built-in Add requires equal-shape f32 tensors")
    shape = next(iter(shapes))
    if not all(isinstance(extent, int) and extent >= 0 for extent in shape):
        raise ImplementationUnavailable("built-in Add currently requires a static shape")
    output = TensorStorage.empty(dtype=vd.f32, shape=shape)._full_view("write")
    left = TensorStorage.empty(dtype=vd.f32, shape=shape)._full_view("read")
    right = TensorStorage.empty(dtype=vd.f32, shape=shape)._full_view("read")
    kernel, _arguments, _grid = program_add_invocation(output, left, right)
    frontend = kernel._lower().frontend
    return ProgramImplementation("vernon.builtin.add", kernel._entry, "compute", frontend.mlir)


__all__ = [
    "CapturedDslProvider",
    "CapturedVjpDslProvider",
    "DirectKernelDslProvider",
    "BuiltinDslProvider",
    "BuiltinLowerer",
    "ProgramDslProvider",
    "ProviderChain",
]
