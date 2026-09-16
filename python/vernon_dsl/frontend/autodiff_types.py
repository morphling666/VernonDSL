from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AccumulationMode(Enum):
    """Grid-independent operation for combining invocation gradients."""

    REDUCE_SUM = "reduce_sum"
    SCATTER_ADD = "scatter_add"


class AccessPatternEvidence(Enum):
    DISJOINT_SCATTER = "disjoint_scatter"
    INJECTIVE_GLOBAL_INDEX = "injective_global_index"
    NON_INJECTIVE_INDEX = "non_injective_index"
    STATIC_INDEX_CONFLICT = "static_index_conflict"
    SHARED_VALUE = "shared_value"


@dataclass(frozen=True)
class AccumulationPlan:
    path: str
    mode: AccumulationMode
    evidence: tuple[AccessPatternEvidence, ...]
    invocation_axes: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode.value,
            "evidence": [item.value for item in self.evidence],
            "invocation_axes": list(self.invocation_axes),
        }


@dataclass(frozen=True)
class LaunchPlan:
    workgroup_size: tuple[int, int, int]
    accumulation_plans: tuple[AccumulationPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workgroup_size": list(self.workgroup_size),
            "accumulation_plans": [plan.to_dict() for plan in self.accumulation_plans],
        }


__all__ = [
    "AccessPatternEvidence",
    "AccumulationMode",
    "AccumulationPlan",
    "LaunchPlan",
]
