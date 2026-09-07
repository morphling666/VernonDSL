"""Loading and invocation for existing cooked pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..frontend.autodiff_profiles import DerivativeGroup
from .binding import _dispatch_borrow_scope, _PersistentBindingTable
from .resource_common import _session_state
from .sampler import SamplerState
from .tensor import TensorStorage, TensorView
from .texture import _TextureResource


@dataclass(frozen=True)
class _CookedProgramPullback:
    native: Any
    gradient_groups: tuple[DerivativeGroup, ...]
    cotangent_groups: tuple[DerivativeGroup, ...]
    carrier_shape: tuple[int, ...]

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        return dict(
            self.native.apply_grouped(
                cotangent,
                self.gradient_groups,
                self.cotangent_groups,
                self.carrier_shape,
                False,
            )
        )

    def apply_logical(self, cotangent: Any) -> dict[str, Any]:
        return dict(
            self.native.apply_grouped(
                cotangent,
                self.gradient_groups,
                self.cotangent_groups,
                self.carrier_shape,
                True,
            )
        )

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
        return float(self.native.recomputation_factor)


@dataclass
class CookedProgram:
    """Callable compute pipeline loaded through the native asset loader."""

    _bundle: bytes
    _directory: str
    _features: tuple[str, ...]
    _native: Any = None
    _runtime_generation: int = -1
    _binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable, init=False, repr=False)

    def _load(self) -> None:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("cooked pipeline assets require the native runtime")
        if self._native is not None and self._runtime_generation == state._runtime_generation:
            return
        self._native = state._native_runtime.load_program(
            self._bundle,
            self._directory,
            list(self._features),
        )
        self._runtime_generation = state._runtime_generation

    @property
    def parameter_names(self) -> tuple[str, ...]:
        self._load()
        return tuple(
            parameter.name for parameter in self._native.parameters if not parameter.name.startswith("__grid_")
        )

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] = (1, 1, 1),
        **keywords: Any,
    ) -> None:
        self._load()
        names = self.parameter_names
        if len(arguments) > len(names):
            raise TypeError(f"pipeline expects at most {len(names)} positional arguments")
        bindings = dict(zip(names, arguments, strict=False))
        duplicate = set(bindings) & set(keywords)
        if duplicate:
            raise TypeError(f"pipeline received duplicate binding(s): {', '.join(sorted(duplicate))}")
        bindings.update(keywords)
        missing = set(names) - set(bindings)
        unknown = set(bindings) - set(names)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise TypeError("pipeline bindings do not match its signature: " + "; ".join(details))
        if (
            not isinstance(grid, tuple)
            or len(grid) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid)
        ):
            raise ValueError("grid must contain three positive integers")

        state = _session_state()
        bindings.update({"__grid_x": grid[0], "__grid_y": grid[1], "__grid_z": grid[2]})
        parameters = {parameter.slot: parameter for parameter in self._native.parameters}
        slots = tuple(
            slot for slot in self._native.program_abi["boundary_slots"] if slot["role"] in {"input", "output"}
        )
        if set(parameters) != {slot["slot"] for slot in slots}:
            raise RuntimeError("cooked Program parameters do not match compiler-emitted ProgramABI")
        access_names = {
            state._native.ACCESS_READ: "read",
            state._native.ACCESS_WRITE: "write",
            state._native.ACCESS_READ_WRITE: "read_write",
        }
        resolved = [(parameters[slot["slot"]], bindings[slot["path"]]) for slot in slots]
        borrows = [
            (parameter.name, value, access_names[parameter.access])
            for parameter, value in resolved
            if isinstance(value, (TensorStorage, TensorView, _TextureResource))
        ]
        with _dispatch_borrow_scope(borrows), self._binding_cache.invocation(self._native) as builder:
            for parameter, value in resolved:
                if parameter.kind == state._native.PIPELINE_SAMPLER:
                    if not isinstance(value, SamplerState):
                        raise TypeError(f"sampler {parameter.name!r} must be a SamplerState")
                    self._binding_cache.bind_sampler(builder, self._native, parameter, value)
                else:
                    self._binding_cache.bind_argument(builder, self._native, parameter, value)
            self._native.program_forward_bound(builder)
        if state._architecture != state.cpu:
            for parameter, value in resolved:
                if parameter.access != state._native.ACCESS_READ and hasattr(value, "_mark_device_dirty"):
                    value._mark_device_dirty()

    def vjp(
        self,
        bindings: dict[str, Any],
        grid: tuple[int, int, int],
    ) -> tuple[Any, _CookedProgramPullback]:
        self._load()
        if len(grid) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
            raise ValueError("grid must contain three positive integers")
        parameters = tuple(self._native.parameters)
        grid_values = dict(zip(("__grid_x", "__grid_y", "__grid_z"), grid, strict=True))
        grid_parameters = {parameter.name for parameter in parameters if parameter.name in grid_values}
        if grid_parameters != set(grid_values):
            raise RuntimeError("cooked Program is missing compute workgroup boundary Values")
        if set(bindings) != {parameter.name for parameter in parameters} - grid_parameters:
            raise ValueError("autodiff bindings do not match pipeline parameters")
        state = _session_state()
        access_names = {
            state._native.ACCESS_READ: "read",
            state._native.ACCESS_WRITE: "write",
            state._native.ACCESS_READ_WRITE: "read_write",
        }
        borrows = [
            (parameter.name, bindings[parameter.name], access_names[parameter.access])
            for parameter in parameters
            if parameter.name not in grid_values
            if isinstance(bindings[parameter.name], (TensorStorage, TensorView))
        ]
        with _dispatch_borrow_scope(borrows), self._binding_cache.invocation(self._native) as builder:
            for parameter in parameters:
                value = grid_values[parameter.name] if parameter.name in grid_values else bindings[parameter.name]
                self._binding_cache.bind_argument(builder, self._native, parameter, value)
            output, native_pullback = self._native.program_vjp_bound(builder, bindings)
        if state._architecture != state.cpu:
            for parameter in parameters:
                if parameter.name in grid_values:
                    continue
                value = bindings[parameter.name]
                if parameter.access != state._native.ACCESS_READ and hasattr(value, "_mark_device_dirty"):
                    value._mark_device_dirty()
        groups = tuple(
            DerivativeGroup(str(role), str(path), tuple(str(leaf) for leaf in leaves))
            for role, path, leaves in self._native.derivative_groups
        )
        if not groups or {group.role for group in groups} != {"gradient", "cotangent"}:
            raise RuntimeError("cooked Program has no validated derivative groups")
        workgroup = tuple(getattr(self._native, "workgroup_size", (1, 1, 1)))
        extent = tuple(count * size for count, size in zip(grid, workgroup, strict=True))
        carrier_shape = () if extent == (1, 1, 1) else tuple(reversed(extent))
        return output, _CookedProgramPullback(
            native_pullback,
            tuple(group for group in groups if group.role == "gradient"),
            tuple(group for group in groups if group.role == "cotangent"),
            carrier_shape,
        )


def load_program(
    manifest: str | Path,
    *,
    features: tuple[str, ...] = (),
) -> CookedProgram:
    manifest_path = Path(manifest).resolve()
    if any(not isinstance(feature, str) or not feature for feature in features):
        raise ValueError("pipeline asset features must be non-empty strings")
    bundle = manifest_path.read_bytes()
    pipeline = CookedProgram(
        bundle,
        str(manifest_path.parent),
        tuple(sorted(set(features))),
    )
    pipeline._load()
    return pipeline


__all__ = ["CookedProgram", "load_program"]
