from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ad import ProgramExpression
from ..bundle import make_target_options
from ..frontend.autodiff_profiles import DerivativeGroup
from ..frontend.structured_vjp import build_structured_vjp
from ..host_values import TangentLayout, tangent_layout
from .kernel import Kernel, _session_state
from .resources import TensorStorage, TensorView, _dispatch_borrow_scope


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
    derivative_groups: tuple[DerivativeGroup, ...]
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

    def _single_storage_cotangent(self, group: DerivativeGroup, value: TensorStorage) -> Any:
        primal = self.bindings[group.parameter_root]
        if not isinstance(primal, (TensorStorage, TensorView)):
            raise TypeError(f"output cotangent '{group.declared_path}' is not Storage-backed")
        if not isinstance(value.element_layout, TangentLayout):
            raise TypeError(f"output cotangent '{group.declared_path}' requires packed tangent TensorStorage")
        expected_shape = (*self.carrier_shape, *primal.shape)
        primal_dtype = TensorStorage._dtype(value.element_layout.primal_element_type)
        if primal_dtype != primal.dtype or value.shape != expected_shape:
            raise ValueError(
                f"output cotangent '{group.declared_path}' has an incompatible tangent dtype or owner shape"
            )
        if not isinstance(primal, TensorView):
            return value.to_numpy()
        projection = value._tangent_view(
            "",
            shape=expected_shape,
            strides=(
                *tuple(
                    stride // value.element_layout.size for stride in value._array.strides[: len(self.carrier_shape)]
                ),
                *primal._strides,
            ),
            offset=primal._offset,
            access="read",
        )
        return projection.to_numpy()

    def _packed_cotangent(self, group: DerivativeGroup, value: Any) -> dict[str, Any]:
        primal = self.bindings[group.parameter_root]
        owner = primal.owner if isinstance(primal, TensorView) else primal
        if not isinstance(value, TensorStorage) or not isinstance(value.element_layout, TangentLayout):
            raise TypeError(f"output cotangent '{group.declared_path}' requires packed tangent TensorStorage")
        if not isinstance(owner, TensorStorage) or owner._element_type is None:
            raise TypeError(f"output cotangent '{group.declared_path}' is not aggregate Storage")
        expected = tangent_layout(owner._element_type)
        expected_shape = (*self.carrier_shape, *owner.shape)
        if value.element_layout.layout_hash != expected.layout_hash or value.shape != expected_shape:
            raise ValueError(
                f"output cotangent '{group.declared_path}' has an incompatible tangent layout or owner shape"
            )
        packed: dict[str, Any] = {}
        for leaf_path in group.leaf_paths:
            suffix = leaf_path[len(group.parameter_root) + 1 :] if leaf_path != group.parameter_root else ""
            projection = (
                value._tangent_view(
                    suffix,
                    shape=(*self.carrier_shape, *primal.shape),
                    strides=(
                        *tuple(
                            stride // value.element_layout.size
                            for stride in value._array.strides[: len(self.carrier_shape)]
                        ),
                        *primal._strides,
                    ),
                    offset=primal._offset,
                    access="read",
                )
                if isinstance(primal, TensorView)
                else value[suffix]
            )
            packed[leaf_path] = projection.to_numpy()
        return packed

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        if cotangent is None:
            native_cotangent = None
        else:
            supplied = (
                cotangent
                if isinstance(cotangent, dict)
                else {self.cotangent_groups[0].declared_path: cotangent}
                if len(self.cotangent_groups) == 1
                else None
            )
            if supplied is None or set(supplied) != {group.declared_path for group in self.cotangent_groups}:
                raise ValueError("pullback requires exactly one cotangent per declared output path")
            leaves: dict[str, Any] = {}
            for group in self.cotangent_groups:
                value = supplied[group.declared_path]
                primal = self.bindings[group.parameter_root]
                owner = primal.owner if isinstance(primal, TensorView) else primal
                aggregate_storage = isinstance(owner, TensorStorage) and owner._element_type is not None
                if isinstance(value, TensorStorage) and aggregate_storage:
                    leaves.update(self._packed_cotangent(group, value))
                elif len(group.leaf_paths) == 1:
                    leaves[group.leaf_paths[0]] = (
                        self._single_storage_cotangent(group, value) if isinstance(value, TensorStorage) else value
                    )
                else:
                    leaves.update(self._packed_cotangent(group, value))
            native_cotangent = next(iter(leaves.values())) if len(leaves) == 1 else leaves
        leaf_results = self.native(native_cotangent)
        grouped: dict[str, Any] = {}
        for group in self.gradient_groups:
            values = tuple(leaf_results[path] for path in group.leaf_paths)
            first = values[0]
            if any(value is not first for value in values[1:]):
                raise RuntimeError(f"gradient leaves for '{group.declared_path}' did not materialize into one owner")
            grouped[group.declared_path] = first
        return grouped


@dataclass
class CookedVjpPipeline:
    _bundle: bytes
    _directory: str
    _features: tuple[str, ...]
    _native: Any = None
    _runtime_generation: int = -1

    def _load(self) -> None:
        state = _session_state()
        if state._architecture != state.cpu or state._native_runtime is None:
            raise RuntimeError("cooked structured VJP assets require the CPU runtime")
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


def _pipeline_derivative_groups(pipeline: Any) -> tuple[DerivativeGroup, ...]:
    groups = tuple(
        DerivativeGroup(str(role), str(path), tuple(str(leaf) for leaf in leaves))
        for role, path, leaves in pipeline.derivative_groups
    )
    if not groups or {group.role for group in groups} != {"gradient", "cotangent"}:
        raise RuntimeError("structured VJP pipeline has no validated derivative groups")
    return groups


def _invoke_structured_pipeline(
    pipeline: Any,
    bindings: dict[str, Any],
    grid: tuple[int, int, int],
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
    borrows = [
        (name, value, access_by_name[name])
        for name, value in bindings.items()
        if name in access_by_name and isinstance(value, (TensorStorage, TensorView))
    ]
    with _dispatch_borrow_scope(borrows):
        output, pullback = pipeline.vjp(bindings, grid)
    return output, _StructuredPullback(
        pullback,
        tuple(group for group in derivative_groups if group.role == "gradient"),
        tuple(group for group in derivative_groups if group.role == "cotangent"),
        bindings,
        (grid[2], grid[1], grid[0]) if grid[0] * grid[1] * grid[2] > 1 else (),
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
        [(group.role, group.declared_path, list(group.leaf_paths)) for group in compiled.derivative_groups],
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
    _validate_grid(grid)

    options = make_target_options("cpu")
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
        frontend, function, builtins, _ = kernel._lower(arguments)
        structured = build_structured_vjp(
            runtime_state._native,
            frontend,
            expression.transform,
        )
        profiles = {profile.name: profile for profile in structured.plan.profiles}
        programs = runtime_state._native._compile_cpu_program_results(
            [
                frontend.mlir,
                structured.profiles["forward_with_tape"],
                structured.profiles["backward"],
            ],
            options.native_options["options"],
        )
        if len(programs) != 3:
            raise RuntimeError("CPU VJP profile compilation returned an invalid result count")
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
            forward_protocol=structured.protocols["forward_with_tape"],
            backward_protocol=structured.protocols["backward"],
            derivative_groups=derivative_groups,
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
    return _invoke_structured_pipeline(
        compiled.pipeline,
        bindings,
        grid,
    )


__all__ = [
    "CookedVjpPipeline",
    "clear_vjp_cache",
    "execute_direct_vjp",
    "invalidate_loaded_vjps",
    "load_cooked_vjp_asset",
]
