from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ad import ProgramExpression
from ..bundle import make_target_options
from ..frontend.structured_vjp import build_structured_vjp
from .kernel import Kernel, _session_state


@dataclass
class _CompiledDirectVjp:
    primal: Any
    forward: Any
    backward: Any
    primal_symbol: str
    forward_symbol: str
    backward_symbol: str
    forward_protocol: str
    backward_protocol: str
    gradient_paths: tuple[str, ...]
    user_parameters: tuple[str, ...]
    pipeline: Any
    runtime_generation: int
    dependency_hashes: tuple[tuple[Path, str], ...]


@dataclass
class _DirectVjpState:
    compiled: dict[tuple[Any, ...], _CompiledDirectVjp] = field(default_factory=dict)


_kernel_states: weakref.WeakKeyDictionary[Kernel, dict[str, _DirectVjpState]] = weakref.WeakKeyDictionary()


def _direct_state(kernel: Kernel, expression: ProgramExpression) -> _DirectVjpState:
    transforms = _kernel_states.setdefault(kernel, {})
    return transforms.setdefault(expression.transform.identity, _DirectVjpState())


def invalidate_loaded_vjps() -> None:
    for transforms in list(_kernel_states.values()):
        for state in transforms.values():
            for compiled in state.compiled.values():
                compiled.pipeline = None
                compiled.runtime_generation = -1


def clear_vjp_cache() -> None:
    _kernel_states.clear()


def _load(compiled: _CompiledDirectVjp, runtime_state: Any) -> None:
    if compiled.pipeline is not None and compiled.runtime_generation == runtime_state._runtime_generation:
        return
    if runtime_state._native_runtime is None:
        raise RuntimeError("CPU VJP execution requires the native runtime")
    compiled.pipeline = runtime_state._native_runtime.load_cpu_autodiff(
        compiled.primal,
        compiled.primal_symbol,
        compiled.forward,
        compiled.forward_symbol,
        compiled.backward,
        compiled.backward_symbol,
        compiled.forward_protocol,
        compiled.backward_protocol,
        list(compiled.gradient_paths),
    )
    compiled.runtime_generation = runtime_state._runtime_generation


def execute_direct_vjp(
    expression: ProgramExpression,
    arguments: tuple[Any, ...],
    grid: tuple[int, int, int] | None,
) -> tuple[Any, Any]:
    kernel = expression.program
    if not isinstance(kernel, Kernel):
        raise TypeError("direct VJP execution requires one compute Kernel")
    if expression.transform.protocol != "dynamic_v2":
        raise RuntimeError("direct structured VJP execution requires protocol='dynamic_v2'")
    runtime_state = _session_state()
    if runtime_state._architecture != runtime_state.cpu:
        raise RuntimeError("direct structured VJP execution currently supports only the CPU runtime")
    if runtime_state._native is None or runtime_state._native_runtime is None:
        raise RuntimeError("CPU VJP execution requires the native compiler and runtime")
    if grid is None:
        raise TypeError("grid is required for direct structured VJP execution")
    if len(grid) != 3 or any(not isinstance(value, int) or value <= 0 for value in grid):
        raise ValueError("grid must contain three positive integers")

    options = make_target_options("cpu")
    key = kernel._dispatch_key(arguments, (), options.target, tuple(sorted(options.options.items())))
    direct = _direct_state(kernel, expression)
    compiled = direct.compiled.get(key)
    if compiled is not None and not kernel._dependencies_current(compiled):
        del direct.compiled[key]
        compiled = None
    if compiled is None:
        frontend, function, builtins, _ = kernel._lower(arguments)
        structured = build_structured_vjp(
            runtime_state._native,
            frontend,
            expression.transform,
        )
        profiles = {profile.name: profile for profile in structured.plan.profiles}
        compiler = runtime_state._native.Compiler()

        def compile_profile(mlir: str) -> Any:
            program = compiler.compile_program_result(
                mlir,
                runtime_state._native.Target.CPU,
                **options.native_options,
            )
            if not program.ok:
                raise RuntimeError(program.diagnostics)
            return program

        primal = compile_profile(frontend.mlir)
        forward = compile_profile(structured.profiles["forward_with_tape"])
        backward = compile_profile(structured.profiles["backward"])
        compiled = _CompiledDirectVjp(
            primal=primal,
            forward=forward,
            backward=backward,
            primal_symbol=structured.entry.symbol,
            forward_symbol=profiles["forward_with_tape"].symbol,
            backward_symbol=profiles["backward"].symbol,
            forward_protocol=structured.protocols["forward_with_tape"],
            backward_protocol=structured.protocols["backward"],
            gradient_paths=tuple(binding.path for binding in profiles["backward"].outputs),
            user_parameters=tuple(argument.arg for argument in function.args.args if argument.arg not in builtins),
            pipeline=None,
            runtime_generation=-1,
            dependency_hashes=kernel._dependency_hashes(frontend),
        )
        direct.compiled[key] = compiled
        _load(compiled, runtime_state)
    else:
        _load(compiled, runtime_state)
    if len(arguments) != len(compiled.user_parameters):
        raise TypeError(f"{kernel.__name__} expects {len(compiled.user_parameters)} launch arguments")
    bindings = dict(zip(compiled.user_parameters, arguments, strict=True))
    return compiled.pipeline.vjp(bindings, grid)


__all__ = ["clear_vjp_cache", "execute_direct_vjp", "invalidate_loaded_vjps"]
