from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

STAGE_REGISTRY_VERSION = 2


@dataclass(frozen=True)
class StageDefinition:
    decorator: str
    kind: str
    graphics_order: int | None = None
    supported_targets: frozenset[str] | None = None

    @property
    def is_graphics(self) -> bool:
        return self.graphics_order is not None

    def supports_target(self, target: str) -> bool:
        return self.supported_targets is None or target in self.supported_targets


STAGES = (
    StageDefinition("kernel", "compute"),
    StageDefinition("vertex", "vertex", 0),
    StageDefinition("fragment", "fragment", 1),
)

STAGE_BY_DECORATOR = {stage.decorator: stage for stage in STAGES}
STAGE_BY_KIND = {stage.kind: stage for stage in STAGES}
ENTRY_DECORATOR_STAGES = {stage.decorator: stage.kind for stage in STAGES}
ENTRY_DECORATORS = frozenset(ENTRY_DECORATOR_STAGES)
GRAPHICS_STAGES = frozenset(stage.kind for stage in STAGES if stage.is_graphics)
GRAPHICS_STAGE_ORDER = tuple(
    stage.kind
    for stage in sorted(
        (stage for stage in STAGES if stage.is_graphics),
        key=lambda stage: stage.graphics_order,
    )
)
GRAPHICS_TOPOLOGIES = frozenset(
    {
        ("vertex", "fragment"),
    }
)


def validate_graphics_topology(stages: Iterable[str]) -> tuple[str, ...]:
    topology = tuple(stages)
    unknown = set(topology) - GRAPHICS_STAGES
    if unknown:
        expected = ", ".join(GRAPHICS_STAGE_ORDER)
        raise ValueError(
            "unknown graphics pipeline stages; expected " + expected + "; got " + ", ".join(sorted(unknown))
        )
    if len(set(topology)) != len(topology):
        raise ValueError("graphics pipeline contains a duplicate stage kind")
    if topology not in GRAPHICS_TOPOLOGIES:
        ordered = tuple(sorted(topology, key=lambda stage: STAGE_BY_KIND[stage].graphics_order))
        if ordered in GRAPHICS_TOPOLOGIES:
            raise ValueError("graphics pipeline stages are not in topology order")
        expected = " or ".join("→".join(candidate) for candidate in sorted(GRAPHICS_TOPOLOGIES))
        raise ValueError(f"unsupported graphics pipeline stages topology; expected {expected}")
    return topology


def validate_stage_target(stage: str, target: str) -> None:
    definition = STAGE_BY_KIND.get(stage)
    if definition is None:
        raise ValueError(f"unknown shader stage: {stage}")
    if not definition.supports_target(target):
        supported = ", ".join(sorted(definition.supported_targets or ()))
        raise ValueError(f"{stage} stage is unsupported for target {target}; supported targets: {supported}")


__all__ = [
    "ENTRY_DECORATORS",
    "ENTRY_DECORATOR_STAGES",
    "GRAPHICS_STAGES",
    "GRAPHICS_STAGE_ORDER",
    "GRAPHICS_TOPOLOGIES",
    "STAGES",
    "STAGE_BY_DECORATOR",
    "STAGE_BY_KIND",
    "STAGE_REGISTRY_VERSION",
    "StageDefinition",
    "validate_graphics_topology",
    "validate_stage_target",
]
