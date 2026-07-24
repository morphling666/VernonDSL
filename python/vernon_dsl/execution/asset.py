from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .._runtime.resources import TensorStorage, TensorView, _dispatch_borrow_scope


class ExecutionGraphAsset:
    """Reloadable target-specific cooked execution graph."""

    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest).resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self._native: Any | None = None
        self._generation = -1

    def _load(self) -> Any:
        state = importlib.import_module("vernon_dsl._runtime.session")
        if state._native_runtime is None:
            raise RuntimeError("execution graph asset requires an initialized native runtime")
        if self._native is None or self._generation != state._runtime_generation:
            self._native = state._native_runtime.load_pipeline_asset(
                self.manifest.read_bytes(),
                str(self.manifest.parent),
                [],
            )
            self._generation = state._runtime_generation
        return self._native

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self._load().parameters)

    @property
    def steps(self) -> tuple[Any, ...]:
        return tuple(self._load().steps)

    def run(self, bindings: Mapping[str, Any] | None = None, /, **values: Any) -> None:
        state = importlib.import_module("vernon_dsl._runtime.session")
        supplied = dict(bindings or {})
        overlap = supplied.keys() & values.keys()
        if overlap:
            raise TypeError("duplicate execution graph binding(s): " + ", ".join(sorted(overlap)))
        supplied.update(values)
        native = self._load()
        parameters = {parameter.name: parameter for parameter in native.parameters}
        missing = parameters.keys() - supplied.keys()
        unexpected = supplied.keys() - parameters.keys()
        if missing:
            raise TypeError("missing execution graph binding(s): " + ", ".join(sorted(missing)))
        if unexpected:
            raise TypeError("unexpected execution graph binding(s): " + ", ".join(sorted(unexpected)))

        dtype_codes = {
            np.dtype(np.bool_): state._native.DATA_BOOL,
            np.dtype(np.int32): state._native.DATA_I32,
            np.dtype(np.uint32): state._native.DATA_U32,
            np.dtype(np.float16): state._native.DATA_F16,
            np.dtype(np.float32): state._native.DATA_F32,
            np.dtype(np.float64): state._native.DATA_F64,
        }
        access_names = {
            state._native.ACCESS_READ: "read",
            state._native.ACCESS_WRITE: "write",
            state._native.ACCESS_READ_WRITE: "read_write",
        }
        owner_borrows: dict[int, tuple[str, TensorStorage | TensorView, str]] = {}
        for name, parameter in parameters.items():
            value = supplied[name]
            if not isinstance(value, (TensorStorage, TensorView)):
                continue
            owner = value.owner if isinstance(value, TensorView) else value
            mode = access_names[parameter.access]
            existing = owner_borrows.get(id(owner))
            if existing is not None and existing[2] != mode:
                mode = "read_write"
            borrowed = owner if isinstance(owner, TensorStorage) else value
            owner_borrows[id(owner)] = (name, borrowed, mode)
        borrows = list(owner_borrows.values())
        builder = native.invocation_builder()
        with _dispatch_borrow_scope(borrows):
            for name, parameter in parameters.items():
                value = supplied[name]
                if isinstance(value, (TensorStorage, TensorView)):
                    layout = value.layout
                    builder.device_tensor(
                        name,
                        value._resident_buffer(),
                        dtype_codes[value.dtype],
                        parameter.access,
                        list(value.shape),
                        list(layout.byte_strides),
                        layout.byte_offset,
                    )
                    continue
                scalar = np.asarray(value)
                if scalar.dtype.kind == "f":
                    scalar = np.asarray(value, dtype=np.float32)
                elif scalar.dtype.kind == "u":
                    scalar = np.asarray(value, dtype=np.uint32)
                elif scalar.dtype.kind == "b":
                    scalar = np.asarray(value, dtype=np.bool_)
                else:
                    scalar = np.asarray(value, dtype=np.int32)
                builder.host_tensor(name, scalar)
            builder.grid(1, 1, 1)
            builder.invoke()
            state._native_runtime.synchronize()
            for name, parameter in parameters.items():
                value = supplied[name]
                if parameter.access in {state._native.ACCESS_WRITE, state._native.ACCESS_READ_WRITE} and isinstance(
                    value, (TensorStorage, TensorView)
                ):
                    value._mark_device_dirty()


def load_execution_graph_asset(manifest: str | Path) -> ExecutionGraphAsset:
    return ExecutionGraphAsset(manifest)


__all__ = ["ExecutionGraphAsset", "load_execution_graph_asset"]
