from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._versions import COMPILER_CONTRACT_VERSION, PROGRAM_VERSION
from ..ad import ProgramTransformSpec
from .analysis import typed_effect_data
from .autodiff_profiles import AutodiffProfilePlan
from .model import ConcreteType, TypedFunctionInstance


@dataclass(frozen=True)
class FrontendCompileRequest:
    """Complete, deterministic inputs to one Python frontend specialization."""

    source_path: Path
    entry: str
    enabled_features: tuple[str, ...] = ()
    tensor_shapes: tuple[tuple[str, str, tuple[int, ...]], ...] = ()
    captured_constants: tuple[tuple[str, int | float | bool], ...] = ()
    workgroup_size: tuple[int, int, int] | None = None
    program_transform: ProgramTransformSpec | None = None
    autodiff_planning_policy: str | None = None

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
            "captured_constants",
            tuple(sorted(self.captured_constants, key=lambda value: value[0])),
        )
        if self.autodiff_planning_policy not in {None, "min_memory", "balanced", "min_runtime"}:
            raise ValueError("autodiff planning policy must be 'min_memory', 'balanced', or 'min_runtime'")


@dataclass(frozen=True)
class FrontendCompileResult:
    mlir: str
    specialized_source: str
    dependencies: tuple[tuple[str, str], ...]
    declared_features: tuple[str, ...]
    request: FrontendCompileRequest
    helper_specializations: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    typed_functions: tuple[TypedFunctionInstance, ...] = ()
    autodiff_profiles: AutodiffProfilePlan | None = None
    entry_workgroup_size: tuple[int, int, int] | None = None
    structs: tuple[tuple[str, tuple[tuple[str, ConcreteType], ...]], ...] = ()

    @property
    def semantic_inputs(self) -> dict[str, Any]:
        entry = next(
            (function for function in self.typed_functions if function.symbol == self.request.entry),
            None,
        )
        return {
            "compiler_contract_version": COMPILER_CONTRACT_VERSION,
            "program_version": PROGRAM_VERSION,
            "entry": self.request.entry,
            "enabled_features": list(self.request.enabled_features),
            "tensor_shapes": [[name, dtype, list(shape)] for name, dtype, shape in self.request.tensor_shapes],
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
            "program_transform": (
                self.request.program_transform.to_dict() if self.request.program_transform is not None else None
            ),
            "autodiff_planning_policy": self.request.autodiff_planning_policy,
            "autodiff_profiles": (
                self.autodiff_profiles.manifest_dict() if self.autodiff_profiles is not None else None
            ),
        }
