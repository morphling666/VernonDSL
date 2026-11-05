"""Loading and invocation for cooked Program assets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..frontend.autodiff_profiles import DerivativeGroup
from ..types import Specialization, SpecializationAssignment, specialization_key
from .binding import _DispatchBorrowLease
from .session import (
    _execution_context,
    _invocation_context,
    _InvocationContext,
    _SessionArtifactCache,
    _use_invocation_context,
)


@dataclass(frozen=True)
class _CookedInvocation:
    """The boundary values of one cooked Program call, as the linked schedule helpers read a Program invocation."""

    inputs: Mapping[str, Any]
    input_annotations: Mapping[str, Any] = field(default_factory=dict)
    graphics_controls: Mapping[int, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _CookedProgramPullback:
    native: Any
    gradient_groups: tuple[DerivativeGroup, ...]
    cotangent_groups: tuple[DerivativeGroup, ...]
    carrier_shape: tuple[int, ...]
    context: _InvocationContext
    # A linked pullback's residuals point into the memory of its forward call's inputs, which a later write leaves to
    # the pullback.
    preserved: Any = None

    def __call__(self, cotangent: Any = None) -> dict[str, Any]:
        with _use_invocation_context(self.context):
            context = _execution_context()
            return dict(
                self.native.apply_grouped(
                    cotangent,
                    self.gradient_groups,
                    self.cotangent_groups,
                    self.carrier_shape,
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
    # The executable linked as a schedule and as a VJP by its first call. An executable that does not link raises the
    # linker's ValueError on every call.
    _schedule: Any = None
    _vjp: Any = None

    def linked_schedule(self, state: Any) -> Any:
        if self._schedule is None:
            self._schedule = state.native._ProgramSchedule(self.native)
        return self._schedule

    def linked_vjp(self, state: Any) -> Any:
        if self._vjp is None:
            self._vjp = state.native._ProgramVjp(self.native)
        return self._vjp


def _cooked_invocation(native: Any, bindings: Mapping[str, Any]) -> tuple[_CookedInvocation, dict[str, Any]]:
    slots = tuple(slot for slot in native.program_abi["boundary_slots"] if slot["role"] in {"input", "output"})
    inputs = {slot["path"]: bindings[slot["path"]] for slot in slots if slot["role"] == "input"}
    targets = {slot["path"]: bindings[slot["path"]] for slot in slots if slot["role"] == "output"}
    return _CookedInvocation(inputs), targets


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
        from .program_autodiff import _call_program_schedule

        invocation, targets = _cooked_invocation(native, bindings)
        _call_program_schedule(loaded.linked_schedule(state), native, invocation, targets)

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
        from .program_autodiff import _call_program_vjp
        from .tensor import _owner_controls, _PreservedOwners

        context = _execution_context()
        invocation, targets = _cooked_invocation(native, {**bindings, **grid_values})
        native_pullback = _call_program_vjp(loaded.linked_vjp(context.session), native, invocation, targets, bindings)
        preserved = _PreservedOwners(_owner_controls(invocation.inputs.values()))
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
        return {}, _CookedProgramPullback(
            native_pullback,
            tuple(group for group in groups if group.role == "gradient"),
            tuple(group for group in groups if group.role == "cotangent"),
            carrier_shape,
            context,
            preserved,
        )


def load_program(
    manifest: str | Path,
    *,
    specializations: Mapping[Specialization, object] | None = None,
) -> CookedProgram:
    manifest_path = Path(manifest).resolve()
    bundle = manifest_path.read_bytes()
    if json.loads(bundle).get("type") == "host_module":
        raise ValueError(
            f"{manifest_path.name} is a Host module asset: link its object and find its table by id "
            "(vernonRuntimeFindHostModule), or load it in Python mode with HostModule.from_asset"
        )
    with _invocation_context():
        assignments = specialization_key(specializations)
        executable = CookedProgram(
            bundle,
            str(manifest_path.parent),
            assignments,
        )
        executable._load()
        return executable


__all__ = ["CookedProgram", "load_program"]
