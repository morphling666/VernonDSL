"""Loading and invocation for cooked Program assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..frontend.autodiff_profiles import DerivativeGroup
from ..types import Specialization, SpecializationAssignment, specialization_key
from .binding import _dispatch_borrow_scope, _DispatchBorrowLease, _PersistentBindingTable, _raise_invocation_error
from .sampler import SamplerState
from .session import (
    _execution_context,
    _invocation_context,
    _InvocationContext,
    _SessionArtifactCache,
    _use_invocation_context,
)
from .tensor import TensorStorage, TensorView
from .texture import _TextureResource


@dataclass(frozen=True)
class _CookedProgramPullback:
    native: Any
    gradient_groups: tuple[DerivativeGroup, ...]
    cotangent_groups: tuple[DerivativeGroup, ...]
    carrier_shape: tuple[int, ...]
    context: _InvocationContext

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        with _use_invocation_context(self.context):
            context = _execution_context()
            return dict(
                self.native.apply_grouped(
                    cotangent,
                    self.gradient_groups,
                    self.cotangent_groups,
                    self.carrier_shape,
                    False,
                    context,
                    lambda requests: _DispatchBorrowLease(list(requests), context),
                )
            )

    def apply_logical(self, cotangent: Any) -> dict[str, Any]:
        with _use_invocation_context(self.context):
            context = _execution_context()
            return dict(
                self.native.apply_grouped(
                    cotangent,
                    self.gradient_groups,
                    self.cotangent_groups,
                    self.carrier_shape,
                    True,
                    context,
                    lambda requests: _DispatchBorrowLease(list(requests), context),
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
class _LoadedCookedProgram:
    native: Any
    binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable)


@dataclass(eq=False)
class CookedProgram:
    """Callable Program executable loaded through the canonical asset loader."""

    _bundle: bytes
    _directory: str
    _specializations: tuple[SpecializationAssignment, ...]
    _loaded: _SessionArtifactCache = field(default_factory=_SessionArtifactCache, init=False, repr=False)

    def _load(self) -> _LoadedCookedProgram:
        context = _execution_context()
        state = context.session
        return self._loaded.get_or_create(
            state,
            "program",
            lambda: _LoadedCookedProgram(
                state.native_runtime.load_program(
                    self._bundle,
                    self._directory,
                    [assignment.manifest for assignment in self._specializations],
                )
            ),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        with _invocation_context():
            native = self._load().native
            return tuple(parameter.name for parameter in native.parameters if not parameter.name.startswith("__grid_"))

    def __call__(
        self,
        *arguments: Any,
        grid: tuple[int, int, int] = (1, 1, 1),
        **keywords: Any,
    ) -> None:
        with _invocation_context():
            self._invoke(arguments, grid, keywords)

    def _invoke(
        self,
        arguments: tuple[Any, ...],
        grid: tuple[int, int, int],
        keywords: Mapping[str, Any],
    ) -> None:
        loaded = self._load()
        native = loaded.native
        names = tuple(parameter.name for parameter in native.parameters if not parameter.name.startswith("__grid_"))
        if len(arguments) > len(names):
            raise TypeError(f"Program executable expects at most {len(names)} positional arguments")
        bindings = dict(zip(names, arguments, strict=False))
        duplicate = set(bindings) & set(keywords)
        if duplicate:
            raise TypeError(f"Program executable received duplicate binding(s): {', '.join(sorted(duplicate))}")
        bindings.update(keywords)
        missing = set(names) - set(bindings)
        unknown = set(bindings) - set(names)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise TypeError("Program executable bindings do not match its signature: " + "; ".join(details))
        if (
            not isinstance(grid, tuple)
            or len(grid) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid)
        ):
            raise ValueError("grid must contain three positive integers")

        context = _execution_context()
        state = context.session
        bindings.update({"__grid_x": grid[0], "__grid_y": grid[1], "__grid_z": grid[2]})
        parameters = {parameter.slot: parameter for parameter in native.parameters}
        slots = tuple(slot for slot in native.program_abi["boundary_slots"] if slot["role"] in {"input", "output"})
        if set(parameters) != {slot["slot"] for slot in slots}:
            raise RuntimeError("cooked Program parameters do not match compiler-emitted ProgramABI")
        access_names = {
            state.native.ACCESS_READ: "read",
            state.native.ACCESS_WRITE: "write",
            state.native.ACCESS_READ_WRITE: "read_write",
        }
        resolved = [(parameters[slot["slot"]], bindings[slot["path"]]) for slot in slots]
        borrows = [
            (parameter.slot, value, access_names[parameter.access])
            for parameter, value in resolved
            if isinstance(value, (TensorStorage, TensorView, _TextureResource))
        ]
        with (
            _dispatch_borrow_scope(borrows, context) as lease,
            loaded.binding_cache.invocation(native, context) as native_invocation,
        ):
            builder = native_invocation.builder
            for parameter, value in resolved:
                if parameter.kind == state.native.PROGRAM_SAMPLER:
                    if not isinstance(value, SamplerState):
                        raise TypeError(f"sampler {parameter.name!r} must be a SamplerState")
                    loaded.binding_cache.bind_sampler(builder, native, parameter, value)
                else:
                    loaded.binding_cache.bind_argument(builder, native, parameter, value)
            outcome = native_invocation.forward()
            lease.resolve(outcome)
            if not outcome.ok:
                _raise_invocation_error(outcome, "cooked Program forward failed", context)

    def vjp(
        self,
        bindings: dict[str, Any],
        grid: tuple[int, int, int] | None = None,
    ) -> tuple[Any, _CookedProgramPullback]:
        with _invocation_context():
            return self._vjp(bindings, grid)

    def _vjp(
        self,
        bindings: dict[str, Any],
        grid: tuple[int, int, int] | None,
    ) -> tuple[Any, _CookedProgramPullback]:
        loaded = self._load()
        native = loaded.native
        parameters = tuple(native.parameters)
        grid_names = ("__grid_x", "__grid_y", "__grid_z")
        grid_parameters = {parameter.name for parameter in parameters if parameter.name in grid_names}
        if grid_parameters == set(grid_names):
            if (
                not isinstance(grid, tuple)
                or len(grid) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid)
            ):
                raise ValueError("grid must contain three positive integers")
            grid_values = dict(zip(grid_names, grid, strict=True))
        elif grid_parameters:
            raise RuntimeError("cooked Program has an incomplete compute workgroup boundary")
        else:
            if grid is not None:
                raise ValueError("this cooked Program owns its launch grids and does not accept a grid")
            grid_values = {}
        if set(bindings) != {parameter.name for parameter in parameters} - grid_parameters:
            raise ValueError("autodiff bindings do not match Program parameters")
        context = _execution_context()
        state = context.session
        access_names = {
            state.native.ACCESS_READ: "read",
            state.native.ACCESS_WRITE: "write",
            state.native.ACCESS_READ_WRITE: "read_write",
        }
        borrows = [
            (parameter.slot, bindings[parameter.name], access_names[parameter.access])
            for parameter in parameters
            if parameter.name not in grid_values
            if isinstance(bindings[parameter.name], (TensorStorage, TensorView, _TextureResource))
        ]
        lease = _DispatchBorrowLease(borrows, context)
        try:
            with loaded.binding_cache.invocation(native, context) as native_invocation:
                builder = native_invocation.builder
                for parameter in parameters:
                    value = grid_values[parameter.name] if parameter.name in grid_values else bindings[parameter.name]
                    loaded.binding_cache.bind_argument(builder, native, parameter, value)
                outcome, output, native_pullback = native.program_vjp_bound(native_invocation, bindings)
                lease.resolve(outcome)
                if not outcome.ok:
                    _raise_invocation_error(outcome, "cooked Program autodiff forward failed", context)
        except BaseException:
            raise
        finally:
            lease.release()
        groups = tuple(
            DerivativeGroup(str(role), str(path), tuple(str(leaf) for leaf in leaves))
            for role, path, leaves in native.derivative_groups
        )
        if not groups or {group.role for group in groups} != {"gradient", "cotangent"}:
            raise RuntimeError("cooked Program has no validated derivative groups")
        if grid is None:
            carrier_shape = ()
        else:
            workgroup = tuple(getattr(native, "workgroup_size", (1, 1, 1)))
            extent = tuple(count * size for count, size in zip(grid, workgroup, strict=True))
            carrier_shape = () if extent == (1, 1, 1) else tuple(reversed(extent))
        return output, _CookedProgramPullback(
            native_pullback,
            tuple(group for group in groups if group.role == "gradient"),
            tuple(group for group in groups if group.role == "cotangent"),
            carrier_shape,
            context,
        )


def load_program(
    manifest: str | Path,
    *,
    specializations: Mapping[Specialization, object] | None = None,
) -> CookedProgram:
    with _invocation_context():
        manifest_path = Path(manifest).resolve()
        assignments = specialization_key(specializations)
        bundle = manifest_path.read_bytes()
        executable = CookedProgram(
            bundle,
            str(manifest_path.parent),
            assignments,
        )
        executable._load()
        return executable


__all__ = ["CookedProgram", "load_program"]
