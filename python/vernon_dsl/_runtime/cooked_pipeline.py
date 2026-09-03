"""Loading and invocation for existing cooked pipeline artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .binding import _dispatch_borrow_scope, _PersistentBindingTable
from .resource_common import _session_state
from .sampler import SamplerState
from .tensor import TensorStorage, TensorView
from .texture import _TextureResource


@dataclass
class CookedPipeline:
    """Callable compute pipeline loaded through the native asset loader."""

    _bundle: bytes
    _directory: str
    _features: tuple[str, ...]
    _source_parameters: tuple[str, ...] = ()
    _native: Any = None
    _runtime_generation: int = -1
    _program_bundle: bool = False
    _binding_cache: _PersistentBindingTable = field(default_factory=_PersistentBindingTable, init=False, repr=False)

    def _load(self) -> None:
        state = _session_state()
        if state._native_runtime is None:
            raise RuntimeError("cooked pipeline assets require the native runtime")
        if self._native is not None and self._runtime_generation == state._runtime_generation:
            return
        self._native = state._native_runtime.load_cooked_asset(
            self._bundle,
            self._directory,
            list(self._features),
        )
        self._program_bundle = self._native.is_managed_program
        self._runtime_generation = state._runtime_generation

    @property
    def parameter_names(self) -> tuple[str, ...]:
        self._load()
        if self._source_parameters:
            return self._source_parameters
        return tuple(parameter.name for parameter in self._native.parameters)

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
        if self._program_bundle:
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
            return
        access_names = {
            state._native.ACCESS_READ: "read",
            state._native.ACCESS_WRITE: "write",
            state._native.ACCESS_READ_WRITE: "read_write",
        }
        parameters = tuple(self._native.parameters)
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
        if state._architecture != state.cpu:
            for parameter in parameters:
                value = bindings[parameter.name]
                if parameter.access in {state._native.ACCESS_WRITE, state._native.ACCESS_READ_WRITE} and isinstance(
                    value, (TensorStorage, TensorView)
                ):
                    value._mark_device_dirty()


def load_pipeline(
    manifest: str | Path,
    *,
    features: tuple[str, ...] = (),
) -> CookedPipeline:
    manifest_path = Path(manifest).resolve()
    if any(not isinstance(feature, str) or not feature for feature in features):
        raise ValueError("pipeline asset features must be non-empty strings")
    bundle = manifest_path.read_bytes()
    pipeline = CookedPipeline(
        bundle,
        str(manifest_path.parent),
        tuple(sorted(set(features))),
        _source_parameter_order(bundle),
    )
    pipeline._load()
    return pipeline


def _source_parameter_order(bundle: bytes) -> tuple[str, ...]:
    try:
        manifest = json.loads(bundle)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ()
    artifacts = manifest.get("stage_artifacts")
    if manifest.get("type") == "program_bundle":
        return ()
    if not isinstance(artifacts, dict):
        return ()
    for artifact in artifacts.values():
        reflection = artifact.get("reflection") if isinstance(artifact, dict) else None
        entries = reflection.get("entries") if isinstance(reflection, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("stage") != "compute":
                continue
            arguments = entry.get("arguments")
            if not isinstance(arguments, list):
                continue
            ordered = sorted(
                (
                    (int(argument["index"]), str(argument["vernon.source_name"]))
                    for argument in arguments
                    if isinstance(argument, dict)
                    and argument.get("kind") != "builtin"
                    and isinstance(argument.get("index"), int)
                    and isinstance(argument.get("vernon.source_name"), str)
                ),
                key=lambda item: item[0],
            )
            return tuple(name for _, name in ordered)
    return ()


__all__ = ["CookedPipeline", "load_pipeline"]
