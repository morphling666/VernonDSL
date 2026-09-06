from __future__ import annotations

import dataclasses
import math
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ad import ProgramExpression
from ..bundle import make_target_options
from ..frontend.autodiff_profiles import DerivativeGroup
from .kernel import Kernel, _session_state


def _validate_grid(grid: tuple[int, int, int]) -> None:
    if len(grid) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
        raise ValueError("grid must contain three positive integers")


def _pipeline_derivative_groups(pipeline: Any) -> tuple[DerivativeGroup, ...]:
    groups = tuple(
        DerivativeGroup(str(role), str(path), tuple(str(leaf) for leaf in leaves))
        for role, path, leaves in pipeline.derivative_groups
    )
    if not groups or {group.role for group in groups} != {"gradient", "cotangent"}:
        raise RuntimeError("structured VJP Program has no validated derivative groups")
    return groups


@dataclass
class _CompiledKernelVjp:
    compiled: Any
    invocation: Any
    parameter_names: tuple[str, ...]
    output_recipe: Any
    workgroup_size: tuple[int, int, int]
    dependency_hashes: tuple[tuple[Any, str], ...]


@dataclass
class _KernelVjpState:
    compiled: dict[tuple[Any, ...], _CompiledKernelVjp] = field(default_factory=dict)


_kernel_states: weakref.WeakKeyDictionary[Kernel, dict[str, _KernelVjpState]] = weakref.WeakKeyDictionary()


def _kernel_state(kernel: Kernel, expression: ProgramExpression) -> _KernelVjpState:
    transforms = _kernel_states.setdefault(kernel, {})
    return transforms.setdefault(expression.transform.identity, _KernelVjpState())


def invalidate_loaded_vjps() -> None:
    for transforms in list(_kernel_states.values()):
        for state in transforms.values():
            for specialization in state.compiled.values():
                specialization.compiled._release_runtime_native()


def clear_vjp_cache() -> None:
    _kernel_states.clear()


def _compile_kernel_vjp(expression: ProgramExpression) -> _CompiledKernelVjp:
    kernel = expression.program
    if not isinstance(kernel, Kernel):
        raise TypeError("single-entry VJP execution requires one compute Kernel")
    runtime_state = _session_state()
    if runtime_state._native is None or runtime_state._native_runtime is None:
        raise RuntimeError("VJP execution requires the native compiler and runtime")
    options = make_target_options(
        runtime_state._architecture.name,
        {"version": runtime_state._interactive_glsl_version()}
        if runtime_state._architecture in {runtime_state.opengl, runtime_state.opengles}
        else {},
    )
    key = kernel._specialization_key((), options.target, tuple(sorted(options.options.items())))
    state = _kernel_state(kernel, expression)
    specialization = state.compiled.get(key)
    if specialization is not None and not kernel._dependencies_current(specialization):
        del state.compiled[key]
        specialization = None
    if specialization is None:
        from ..program import _capture_kernel_program, _capture_recipes
        from .program_autodiff import compile_program_autodiff

        capture, template, invocation, parsed, parameter_names, frontend = _capture_kernel_program(
            kernel,
            transform=expression.transform,
        )
        _, _, output_recipe = _capture_recipes(capture, invocation.outputs)
        specialization = _CompiledKernelVjp(
            compile_program_autodiff(parsed, template),
            invocation,
            parameter_names,
            output_recipe,
            kernel._workgroup_size,
            kernel._dependency_hashes(frontend),
        )
        state.compiled[key] = specialization
    return specialization


def execute_direct_vjp(
    expression: ProgramExpression,
    arguments: tuple[Any, ...],
    grid: tuple[int, int, int] | None,
) -> tuple[Any, Any]:
    if grid is None:
        raise TypeError("grid is required for single-entry structured VJP execution")
    _validate_grid(grid)
    specialization = _compile_kernel_vjp(expression)
    if len(arguments) != len(specialization.parameter_names):
        raise TypeError(f"single-entry VJP expects {len(specialization.parameter_names)} launch arguments")
    inputs = dict(zip(specialization.parameter_names, arguments, strict=True))
    inputs.update(dict(zip(("__grid_x", "__grid_y", "__grid_z"), grid, strict=True)))
    invocation = dataclasses.replace(
        specialization.invocation,
        inputs=inputs,
        outputs=specialization.output_recipe.resolve(inputs, ()),
    )
    _, pullback = specialization.compiled.invoke(invocation)
    pullback._lease.release()
    extent = tuple(count * size for count, size in zip(grid, specialization.workgroup_size, strict=True))
    carrier_shape = () if extent == (1, 1, 1) else tuple(reversed(extent))
    return None, _KernelPullback(pullback, carrier_shape)


@dataclass
class _KernelPullback:
    pullback: Any
    carrier_shape: tuple[int, ...]

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        return self.pullback.apply_with_carrier(self._canonical_cotangent(cotangent), ())

    def apply_logical(self, cotangent: Any) -> dict[str, Any]:
        return self.pullback.apply_with_carrier(self._canonical_cotangent(cotangent), ())

    def _canonical_cotangent(self, cotangent: Any) -> Any:
        if cotangent is None or not self.carrier_shape:
            return cotangent
        if isinstance(cotangent, dict):
            return {path: self._canonical_cotangent(value) for path, value in cotangent.items()}
        source = cotangent._native_host_array() if hasattr(cotangent, "_native_host_array") else cotangent
        array = np.asarray(source)
        rank = len(self.carrier_shape)
        if tuple(array.shape[:rank]) == self.carrier_shape:
            if math.prod(array.shape[rank:]) == math.prod(self.carrier_shape):
                return array.sum(axis=tuple(range(rank)))
            return array[(0,) * rank]
        return cotangent

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pullback, name)


__all__ = [
    "clear_vjp_cache",
    "execute_direct_vjp",
    "invalidate_loaded_vjps",
]
