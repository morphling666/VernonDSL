from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..language.syntax import FRONTEND_VERSION
from .analysis import typed_effect_data
from .model import TypedFunctionInstance


@dataclass(frozen=True)
class FrontendCompileRequest:
    """Complete, deterministic inputs to one Python frontend specialization."""

    source_path: Path
    entry: str
    enabled_features: tuple[str, ...] = ()
    tensor_shapes: tuple[tuple[str, str, tuple[int, ...]], ...] = ()
    tensor_view_layouts: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...], int], ...] = ()
    captured_constants: tuple[tuple[str, int | float | bool], ...] = ()
    workgroup_size: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        object.__setattr__(self, "enabled_features", tuple(sorted(set(self.enabled_features))))
        object.__setattr__(
            self,
            "tensor_shapes",
            tuple(sorted((name, dtype, tuple(shape)) for name, dtype, shape in self.tensor_shapes)),
        )
        object.__setattr__(
            self,
            "tensor_view_layouts",
            tuple(
                sorted(
                    (
                        name,
                        dtype,
                        tuple(shape),
                        tuple(strides),
                        offset,
                    )
                    for name, dtype, shape, strides, offset in self.tensor_view_layouts
                )
            ),
        )
        object.__setattr__(
            self,
            "captured_constants",
            tuple(sorted(self.captured_constants, key=lambda value: value[0])),
        )


@dataclass(frozen=True)
class FrontendCompileResult:
    mlir: str
    specialized_source: str
    dependencies: tuple[tuple[str, str], ...]
    declared_features: tuple[str, ...]
    request: FrontendCompileRequest
    helper_specializations: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    typed_functions: tuple[TypedFunctionInstance, ...] = ()

    @property
    def semantic_inputs(self) -> dict[str, Any]:
        entry = next(
            (function for function in self.typed_functions if function.symbol == self.request.entry),
            None,
        )
        return {
            "frontend_version": FRONTEND_VERSION,
            "entry": self.request.entry,
            "enabled_features": list(self.request.enabled_features),
            "tensor_shapes": [[name, dtype, list(shape)] for name, dtype, shape in self.request.tensor_shapes],
            "tensor_view_layouts": [
                [name, dtype, list(shape), list(strides), offset]
                for name, dtype, shape, strides, offset in self.request.tensor_view_layouts
            ],
            "captured_constants": [
                [name, type(value).__name__, value] for name, value in self.request.captured_constants
            ],
            "workgroup_size": list(self.request.workgroup_size) if self.request.workgroup_size else None,
            "helper_specializations": [
                [name, list(argument_types), list(features)]
                for name, argument_types, features in self.helper_specializations
            ],
            "entry_effects": ([typed_effect_data(effect) for effect in entry.effects] if entry is not None else []),
            "dependencies": [[path, digest] for path, digest in self.dependencies],
        }
