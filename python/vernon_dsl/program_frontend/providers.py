"""Python DSL providers for compiler-planned Program kernel regions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Any, Protocol

from .._runtime.operators import (
    ImplementationUnavailable,
    program_element_type,
)
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


class CapturedVjpDslProvider:
    def __init__(self, implementations: Sequence[ProgramImplementation], native: Any):
        self._implementations = {implementation.callee: implementation for implementation in implementations}
        self._native = native
        self._pairs: dict[str, tuple[ProgramImplementation, ProgramImplementation]] = {}

    def _pair_for_vjp(
        self,
        hint: str,
        request: Mapping[str, Any],
    ) -> tuple[ProgramImplementation, ProgramImplementation]:
        primal_name = hint.removesuffix(".vjp")
        cached = self._pairs.get(primal_name)
        if cached is not None:
            return cached
        primal = self._implementations.get(primal_name)
        if primal is None:
            raise ImplementationUnavailable(f"Program VJP request has no primal implementation for {primal_name!r}")
        raw_bindings = request.get("bindings")
        raw_results = request.get("results")
        if not isinstance(raw_bindings, list) or not isinstance(raw_results, list):
            raise ImplementationUnavailable("Program VJP request has no canonical ABI bindings")
        wrt: list[str] = []
        outputs: list[str] = []
        seen_wrt: set[str] = set()
        seen_outputs: set[str] = set()
        for binding in raw_bindings:
            if not isinstance(binding, Mapping):
                continue
            role = binding.get("autodiff_role")
            source = binding.get("autodiff_source")
            parameter = binding.get("parameter")
            if role == "cotangent":
                path = (
                    source
                    if isinstance(source, str)
                    else (parameter.removeprefix("cotangent.") if isinstance(parameter, str) else None)
                )
                if path and path not in seen_outputs:
                    seen_outputs.add(path)
                    outputs.append(path)
            elif role == "gradient":
                path = (
                    source
                    if isinstance(source, str)
                    else (parameter.removeprefix("gradient.") if isinstance(parameter, str) else None)
                )
                if path and path not in seen_wrt:
                    seen_wrt.add(path)
                    wrt.append(path)
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
        pair = (
            ProgramImplementation(
                f"{primal_name}.forward_with_tape",
                forward_symbol,
                "compute",
                str(profiles["forward_with_tape"]),
            ),
            ProgramImplementation(hint, backward_symbol, "compute", str(profiles["backward"])),
        )
        self._pairs[primal_name] = pair
        return pair

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        del values
        hint = request.get("implementation_hint")
        if not isinstance(hint, str):
            return None
        if hint.endswith(".forward_with_tape"):
            primal_name = hint.removesuffix(".forward_with_tape")
            pair = self._pairs.get(primal_name)
            if pair is None:
                raise ImplementationUnavailable("Program forward_with_tape request has no matching structured VJP pair")
            return pair[0]
        if not hint.endswith(".vjp"):
            return None
        if self._implementations.get(hint.removesuffix(".vjp")) is None:
            return None
        return self._pair_for_vjp(hint, request)[1]


BuiltinLowerer = Callable[
    [Mapping[str, Any], Mapping[int, Mapping[str, Any]]],
    ProgramImplementation | None,
]


class BuiltinDslProvider:
    """Provider for built-in Program semantic operations."""

    def __init__(self, lowerers: Sequence[BuiltinLowerer] = (), native: Any | None = None):
        if native is None:
            native = import_module("vernon_dsl._native")

        self._lowerers = tuple(lowerers)
        self._native = native

    def lower(
        self,
        request: Mapping[str, Any],
        values: Mapping[int, Mapping[str, Any]],
    ) -> ProgramImplementation | None:
        for lowerer in self._lowerers:
            implementation = lowerer(request, values)
            if implementation is not None:
                return implementation
        return _lower_builtin_add(request, values, self._native) or _lower_builtin_copy(request, values, self._native)


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


def _abi_bindings(request: Mapping[str, Any], operation: str) -> dict[str, int]:
    raw_bindings = request.get("bindings")
    if not isinstance(raw_bindings, list):
        raise ImplementationUnavailable(f"built-in {operation} request has no ABI bindings")
    return {
        binding["parameter"]: binding["value"]
        for binding in raw_bindings
        if isinstance(binding, Mapping)
        and isinstance(binding.get("parameter"), str)
        and isinstance(binding.get("value"), int)
    }


def _lower_builtin_elementwise(
    request: Mapping[str, Any],
    values: Mapping[int, Mapping[str, Any]],
    *,
    callee: str,
    operation: str,
    parameters: tuple[str, ...],
    native: Any,
) -> ProgramImplementation | None:
    if request.get("implementation_hint") != callee:
        return None
    bindings = _abi_bindings(request, operation)
    reflected = [_request_value(bindings, values, name) for name in parameters]
    shapes = {tuple(value.get("shape", ())) for value in reflected}
    element_types = {program_element_type(value) for value in reflected}
    if len(shapes) != 1 or len(element_types) != 1 or None in element_types:
        raise ImplementationUnavailable(f"built-in {operation} requires equal-shape tensors of one element type")
    shape = next(iter(shapes))
    element_type = next(iter(element_types))
    assert element_type is not None
    if not all(isinstance(extent, int) for extent in shape):
        raise ImplementationUnavailable(f"built-in {operation} requires a ranked TensorView")
    layouts = [value.get("value_layout") for value in reflected]
    layout_identity = {
        layout.get("layout_hash")
        for layout in layouts
        if isinstance(layout, Mapping) and isinstance(layout.get("layout_hash"), str)
    }
    if len(layout_identity) != 1 or any(not isinstance(layout, Mapping) for layout in layouts):
        raise ImplementationUnavailable(f"built-in {operation} requires one canonical ValueLayout")
    leaves = next(layout["leaves"] for layout in layouts if isinstance(layout, Mapping))
    if not isinstance(leaves, list) or any(
        not isinstance(leaf, Mapping) or not isinstance(leaf.get("dtype"), str) for leaf in leaves
    ):
        raise ImplementationUnavailable(f"built-in {operation} has invalid ValueLayout leaves")
    leaf_dtypes = tuple(str(leaf["dtype"]) for leaf in leaves)
    if operation == "add" and any(not dtype.startswith("f") for dtype in leaf_dtypes):
        raise ImplementationUnavailable("built-in add requires floating tangent leaves")
    entry, mlir = native._build_program_builtin(operation, element_type, len(shape), leaf_dtypes)
    return ProgramImplementation(callee, entry, "compute", mlir)


def _lower_builtin_add(
    request: Mapping[str, Any],
    values: Mapping[int, Mapping[str, Any]],
    native: Any,
) -> ProgramImplementation | None:
    return _lower_builtin_elementwise(
        request,
        values,
        callee="vernon.builtin.add",
        operation="add",
        parameters=("output", "left", "right"),
        native=native,
    )


def _lower_builtin_copy(
    request: Mapping[str, Any],
    values: Mapping[int, Mapping[str, Any]],
    native: Any,
) -> ProgramImplementation | None:
    return _lower_builtin_elementwise(
        request,
        values,
        callee="vernon.builtin.copy",
        operation="copy",
        parameters=("output", "source"),
        native=native,
    )


__all__ = [
    "CapturedDslProvider",
    "CapturedVjpDslProvider",
    "BuiltinDslProvider",
    "BuiltinLowerer",
    "ProgramDslProvider",
    "ProviderChain",
]
