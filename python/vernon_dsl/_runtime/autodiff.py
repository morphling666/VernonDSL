from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..ad import ProgramExpression
from ..bundle import make_target_options
from ..frontend.autodiff_profiles import DerivativeGroup
from ..frontend.structured_vjp import build_structured_vjp
from .kernel import Kernel, _session_state
from .resources import TensorStorage, TensorView, _dispatch_borrow_scope, _NativeBindingCache


@dataclass
class _CompiledDirectVjp:
    primal: Any
    forward: Any
    backward: Any
    primal_symbol: str
    forward_symbol: str
    backward_symbol: str
    derivative_groups: tuple[DerivativeGroup, ...]
    tape_bytes_per_invocation: int
    residual_storage_kind: str
    selected_policy: str
    whole_dispatch_retention_permitted: bool
    active_operation_count: int
    recomputation_cost: int
    resource_reload_cost: int
    deterministic_reduction_legal: bool
    required_primal_paths: tuple[str, ...]
    user_parameters: tuple[str, ...]
    pipeline: Any
    runtime_generation: int
    dependency_hashes: tuple[tuple[Path, str], ...]


@dataclass
class _DirectVjpState:
    compiled: dict[tuple[Any, ...], _CompiledDirectVjp] = field(default_factory=dict)


_kernel_states: weakref.WeakKeyDictionary[Kernel, dict[str, _DirectVjpState]] = weakref.WeakKeyDictionary()


def _validate_grid(grid: tuple[int, int, int]) -> None:
    if len(grid) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
        raise ValueError("grid must contain three positive integers")


@dataclass(frozen=True)
class _StructuredPullback:
    native: Any
    gradient_groups: tuple[DerivativeGroup, ...]
    cotangent_groups: tuple[DerivativeGroup, ...]
    bindings: dict[str, Any]
    carrier_shape: tuple[int, ...]
    estimated_tape_bytes: int
    active_operation_count: int
    recomputation_cost: int
    residual_source_kind: str
    control_history_kind: str

    @property
    def logical_residual_bytes(self) -> int:
        return int(self.native.logical_residual_bytes)

    @property
    def resident_tape_bytes(self) -> int:
        return int(self.native.resident_bytes)

    @property
    def allocated_tape_bytes(self) -> int:
        return int(self.native.allocated_bytes)

    @property
    def peak_temporary_tape_bytes(self) -> int:
        return int(self.native.peak_temporary_bytes)

    @property
    def recomputation_factor(self) -> float:
        return 1.0 + self.recomputation_cost / max(self.active_operation_count, 1)

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        return self._apply(cotangent, logical=False)

    def apply_logical(self, cotangent: Any) -> dict[str, Any]:
        return self._apply(cotangent, logical=True)

    def _apply(self, cotangent: Any, *, logical: bool) -> dict[str, Any]:
        return dict(
            self.native.apply_grouped(
                cotangent,
                self.gradient_groups,
                self.cotangent_groups,
                self.carrier_shape,
                logical,
            )
        )


@dataclass
class CookedVjpPipeline:
    _bundle: bytes
    _directory: str
    _features: tuple[str, ...]
    _native: Any = None
    _runtime_generation: int = -1
    _binding_cache: _NativeBindingCache = field(default_factory=_NativeBindingCache, init=False, repr=False)

    def _load(self) -> None:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("cooked structured VJP assets require the native runtime")
        if self._native is not None and self._runtime_generation == state._runtime_generation:
            return
        native = state._native_runtime.load_pipeline_asset(
            self._bundle,
            self._directory,
            list(self._features),
        )
        _pipeline_derivative_groups(native)
        self._native = native
        self._runtime_generation = state._runtime_generation

    def vjp(
        self,
        bindings: dict[str, Any],
        grid: tuple[int, int, int],
    ) -> tuple[Any, _StructuredPullback]:
        self._load()
        return _invoke_structured_pipeline(self._native, bindings, grid)

    def primal(self, bindings: dict[str, Any], grid: tuple[int, int, int]) -> None:
        self._load()
        parameters = tuple(self._native.parameters)
        if set(bindings) != {parameter.name for parameter in parameters}:
            raise ValueError("cooked VJP primal bindings do not match pipeline parameters")
        state = _session_state()
        access_names = {
            state._native.ACCESS_READ: "read",
            state._native.ACCESS_WRITE: "write",
            state._native.ACCESS_READ_WRITE: "read_write",
        }
        borrows = [
            (parameter.name, bindings[parameter.name], access_names[parameter.access])
            for parameter in parameters
            if isinstance(bindings[parameter.name], (TensorStorage, TensorView))
        ]
        with _dispatch_borrow_scope(borrows), self._binding_cache.invocation(self._native) as builder:
            for parameter in parameters:
                self._binding_cache.bind_argument(
                    builder,
                    self._native,
                    parameter,
                    bindings[parameter.name],
                )
            builder.grid(*grid)
            builder.submit().wait()


def _pipeline_derivative_groups(pipeline: Any) -> tuple[DerivativeGroup, ...]:
    groups = tuple(
        DerivativeGroup(str(role), str(path), tuple(str(leaf) for leaf in leaves))
        for role, path, leaves in pipeline.derivative_groups
    )
    if not groups or {group.role for group in groups} != {"gradient", "cotangent"}:
        raise RuntimeError("structured VJP pipeline has no validated derivative groups")
    return groups


def _retained_primal_allocation_bytes(bindings: dict[str, Any], required_paths: tuple[str, ...]) -> int:
    total = 0
    retained_roots: set[str] = set()
    for path in required_paths:
        root = path.removeprefix("primal.").split(".", 1)[0]
        if root in retained_roots:
            continue
        value = bindings.get(root)
        if isinstance(value, TensorView):
            total += len(value.shape) * 8 + math.prod(value.shape) * value.dtype.itemsize
            retained_roots.add(root)
        elif isinstance(value, TensorStorage):
            total += len(value.shape) * 8 + value._array.nbytes
            retained_roots.add(root)
        elif isinstance(value, np.ndarray):
            total += value.nbytes
            retained_roots.add(root)
        elif isinstance(value, np.generic):
            total += value.dtype.itemsize
            retained_roots.add(root)
        elif isinstance(value, (bool, int, float)):
            total += 8
            retained_roots.add(root)
    return total


def _invoke_structured_pipeline(
    pipeline: Any,
    bindings: dict[str, Any],
    grid: tuple[int, int, int],
    tape_bytes_per_invocation: int = 0,
    active_operation_count: int = 0,
    recomputation_cost: int = 0,
    residual_storage_kind: str = "unknown",
    *,
    encoder: Any | None = None,
    command_plan: Any | None = None,
    binding_cache: _NativeBindingCache | None = None,
) -> tuple[Any, _StructuredPullback]:
    _validate_grid(grid)
    derivative_groups = _pipeline_derivative_groups(pipeline)
    state = _session_state()
    access_names = {
        state._native.ACCESS_READ: "read",
        state._native.ACCESS_WRITE: "write",
        state._native.ACCESS_READ_WRITE: "read_write",
    }
    access_by_name = {parameter.name: access_names[parameter.access] for parameter in pipeline.parameters}
    borrows: list[tuple[str, Any, str]] = [
        (name, value, access_by_name[name])
        for name, value in bindings.items()
        if name in access_by_name and isinstance(value, (TensorStorage, TensorView))
    ]
    if encoder is None and command_plan is None:
        with _dispatch_borrow_scope(borrows):
            output, pullback = pipeline.vjp(bindings, grid)
    else:
        if encoder is not None and command_plan is not None:
            raise RuntimeError("VJP cannot use an encoder and command plan together")
        if binding_cache is None:
            raise RuntimeError("encoded VJP requires a binding cache")
        parameters = tuple(pipeline.parameters)
        if set(bindings) != {parameter.name for parameter in parameters}:
            raise ValueError("encoded VJP bindings do not match pipeline parameters")
        with binding_cache.invocation(pipeline) as builder:
            for parameter in parameters:
                binding_cache.bind_argument(builder, pipeline, parameter, bindings[parameter.name])
            builder.grid(*grid)
            if command_plan is None:
                if encoder is None:
                    raise RuntimeError("encoded VJP requires a command encoder")
                output, pullback = pipeline.vjp_encode(builder, encoder._native, bindings, grid)
            else:
                output, pullback = pipeline.vjp_plan(builder, command_plan, bindings, grid)
        for parameter in parameters:
            value = bindings[parameter.name]
            if parameter.access != state._native.ACCESS_READ and isinstance(value, (TensorStorage, TensorView)):
                value._mark_device_dirty()
    workgroup = tuple(pipeline.workgroup_size)
    extent = tuple(count * size for count, size in zip(grid, workgroup, strict=True))
    carrier_shape = () if extent == (1, 1, 1) else tuple(reversed(extent))
    invocation_count = extent[0] * extent[1] * extent[2]
    return output, _StructuredPullback(
        pullback,
        tuple(group for group in derivative_groups if group.role == "gradient"),
        tuple(group for group in derivative_groups if group.role == "cotangent"),
        bindings,
        carrier_shape,
        tape_bytes_per_invocation * invocation_count,
        active_operation_count * invocation_count,
        recomputation_cost * invocation_count,
        (
            (f"{residual_storage_kind}_capture" if residual_storage_kind in {"static", "dynamic"} else "capture")
            + ("+pure_rematerialization" if recomputation_cost else "")
        ),
        "dynamic_capture"
        if residual_storage_kind == "dynamic"
        else "none"
        if residual_storage_kind == "static"
        else "unknown",
    )


def load_cooked_vjp_asset(
    manifest: str | Path,
    *,
    features: tuple[str, ...] = (),
) -> CookedVjpPipeline:
    manifest_path = Path(manifest).resolve()
    if any(not isinstance(feature, str) or not feature for feature in features):
        raise ValueError("pipeline asset features must be non-empty strings")
    selected_features = tuple(sorted(set(features)))
    pipeline = CookedVjpPipeline(
        manifest_path.read_bytes(),
        str(manifest_path.parent),
        selected_features,
    )
    pipeline._load()
    return pipeline


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
        raise RuntimeError("VJP execution requires the native runtime")
    compiled.pipeline = runtime_state._native_runtime.load_autodiff(
        compiled.primal,
        compiled.primal_symbol,
        compiled.forward,
        compiled.forward_symbol,
        compiled.backward,
        compiled.backward_symbol,
        compiled.tape_bytes_per_invocation,
        compiled.residual_storage_kind,
        compiled.selected_policy,
        compiled.whole_dispatch_retention_permitted,
        [(group.role, group.declared_path, list(group.leaf_paths)) for group in compiled.derivative_groups],
    )
    compiled.runtime_generation = runtime_state._runtime_generation


def _compile_direct_vjp(expression: ProgramExpression, arguments: tuple[Any, ...]) -> _CompiledDirectVjp:
    kernel = expression.program
    if not isinstance(kernel, Kernel):
        raise TypeError("direct VJP execution requires one compute Kernel")
    runtime_state = _session_state()
    if runtime_state._native is None or runtime_state._native_runtime is None:
        raise RuntimeError("VJP execution requires the native compiler and runtime")
    target = {
        runtime_state.cpu: runtime_state._native.Target.CPU,
        runtime_state.cuda: runtime_state._native.Target.CUDA,
        runtime_state.vulkan: runtime_state._native.Target.VULKAN,
        runtime_state.directx: runtime_state._native.Target.DIRECTX,
        runtime_state.metal: runtime_state._native.Target.METAL,
        runtime_state.opengl: runtime_state._native.Target.OPENGL,
        runtime_state.opengles: runtime_state._native.Target.OPENGL_ES,
    }.get(runtime_state._architecture)
    if target is None:
        raise RuntimeError(f"unsupported VJP architecture {runtime_state._architecture.name!r}")
    options = make_target_options(
        runtime_state._architecture.name,
        {"version": runtime_state._interactive_glsl_version()}
        if runtime_state._architecture in {runtime_state.opengl, runtime_state.opengles}
        else {},
    )
    key = kernel._dispatch_key(
        arguments,
        (),
        options.target,
        tuple(sorted(options.options.items())),
    )
    direct = _direct_state(kernel, expression)
    compiled = direct.compiled.get(key)
    if compiled is not None and not kernel._dependencies_current(compiled):
        del direct.compiled[key]
        compiled = None
    if compiled is None:
        lowered = kernel._lower()
        frontend, function, builtins = lowered.frontend, lowered.function, lowered.builtins
        structured = build_structured_vjp(
            runtime_state._native,
            frontend,
            expression.transform,
        )
        profiles = {profile.name: profile for profile in structured.plan.profiles}
        modules = [
            frontend.mlir,
            structured.profiles["forward_with_tape"],
            structured.profiles["backward"],
        ]
        if runtime_state._architecture == runtime_state.cpu:
            programs = runtime_state._native._compile_cpu_program_results(
                modules,
                options.native_options["options"],
            )
        else:
            compiler = runtime_state._native.Compiler()
            programs = [compiler.compile_program_result(module, target, **options.native_options) for module in modules]
        if len(programs) != 3:
            raise RuntimeError("VJP profile compilation returned an invalid result count")
        for program in programs:
            if not program.ok:
                raise RuntimeError(program.diagnostics)
        primal, forward, backward = programs
        derivative_groups = structured.plan.derivative_groups
        compiled = _CompiledDirectVjp(
            primal=primal,
            forward=forward,
            backward=backward,
            primal_symbol=structured.entry.symbol,
            forward_symbol=profiles["forward_with_tape"].symbol,
            backward_symbol=profiles["backward"].symbol,
            derivative_groups=derivative_groups,
            tape_bytes_per_invocation=int(structured.plan.tape_bytes),
            residual_storage_kind=structured.residual_storage_kind,
            selected_policy=structured.selected_policy,
            whole_dispatch_retention_permitted=structured.whole_dispatch_retention_permitted,
            active_operation_count=structured.active_operation_count,
            recomputation_cost=structured.recomputation_cost,
            resource_reload_cost=int(structured.cost_components.get("resource_reload_cost", 0)),
            deterministic_reduction_legal=True,
            required_primal_paths=tuple(structured.plan.required_primal_paths),
            user_parameters=tuple(argument.arg for argument in function.args.args if argument.arg not in builtins),
            pipeline=None,
            runtime_generation=-1,
            dependency_hashes=kernel._dependency_hashes(frontend),
        )
        direct.compiled[key] = compiled
        _load(compiled, runtime_state)
    else:
        _load(compiled, runtime_state)
    return compiled


def execute_direct_vjp(
    expression: ProgramExpression,
    arguments: tuple[Any, ...],
    grid: tuple[int, int, int] | None,
) -> tuple[Any, Any]:
    if grid is None:
        raise TypeError("grid is required for direct structured VJP execution")
    _validate_grid(grid)
    compiled = _compile_direct_vjp(expression, arguments)
    if len(arguments) != len(compiled.user_parameters):
        raise TypeError(f"direct VJP expects {len(compiled.user_parameters)} launch arguments")
    bindings = dict(zip(compiled.user_parameters, arguments, strict=True))
    return _invoke_structured_pipeline(
        compiled.pipeline,
        bindings,
        grid,
        compiled.tape_bytes_per_invocation,
        compiled.active_operation_count,
        compiled.recomputation_cost,
        compiled.residual_storage_kind,
    )


__all__ = [
    "CookedVjpPipeline",
    "clear_vjp_cache",
    "execute_direct_vjp",
    "invalidate_loaded_vjps",
    "load_cooked_vjp_asset",
]
