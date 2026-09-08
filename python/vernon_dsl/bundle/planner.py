from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..language.stage_registry import validate_stage_target
from .reflection import (
    compiled_stage_from_program,
    parse_reflection_json,
    select_artifact,
    select_artifact_bytes,
    select_entry,
)
from .serialize import (
    canonical_json,
    content_hash,
    inline_artifact_descriptor,
    serialize_bundle,
    with_content_hash,
)
from .types import BundlePlan, CompiledArtifact, CompiledStage, ProgramCompileError, ProgramVariantPlan, TargetOptions


def build_program_plan(
    program_id: str,
    target: TargetOptions,
    variants: Sequence[
        tuple[
            Sequence[str],
            Mapping[str, CompiledStage],
            Mapping[str, Any],
        ]
    ],
) -> BundlePlan:
    """Build one immutable plan containing every canonical Program variant."""

    variant_plans: list[ProgramVariantPlan] = []
    unique_stages: dict[str, CompiledStage] = {}
    for key, stages, program in variants:
        canonical_stages = program.get("stages")
        if not isinstance(canonical_stages, Mapping):
            raise ProgramCompileError("canonical Program has no stage contracts")
        if set(canonical_stages) != set(stages):
            raise ProgramCompileError("Program implementation stages do not exactly match its contracts")
        for name, stage in stages.items():
            contract = canonical_stages[name]
            operation = contract.get("operation") if isinstance(contract, Mapping) else None
            expected = "compute" if operation == "compute" else "graphics" if operation == "graphics" else None
            if expected is None:
                raise ProgramCompileError(f"canonical Program stage {name!r} has an invalid operation")
            if stage.stage != expected:
                raise ProgramCompileError(
                    f"Program stage {name!r} requires {expected}, compiler produced {stage.stage}"
                )
            try:
                if stage.stage == "graphics":
                    graphics_stages = stage.metadata.get("graphics_compiled_stages")
                    if not isinstance(graphics_stages, tuple) or not graphics_stages:
                        raise ValueError("graphics Program stage has no compiled shader modules")
                    for graphics_stage in graphics_stages:
                        validate_stage_target(graphics_stage.stage, target.target)
                else:
                    validate_stage_target(stage.stage, target.target)
            except ValueError as error:
                raise ProgramCompileError(str(error)) from None
            if stage.target.spec != target.spec:
                raise ProgramCompileError(f"Program stage {name!r} target does not match the bundle target")
            existing = unique_stages.get(stage.id)
            if existing is None:
                unique_stages[stage.id] = stage
            else:
                metadata = dict(existing.metadata)
                for metadata_name in ("program_contracts", "program_implementations"):
                    merged = dict(metadata.get(metadata_name, {}))
                    additions = stage.metadata.get(metadata_name, {})
                    if isinstance(additions, Mapping):
                        for logical_stage, value in additions.items():
                            if logical_stage in merged and merged[logical_stage] != value:
                                raise ProgramCompileError(f"compiled stage {stage.id} has conflicting {metadata_name}")
                            merged[logical_stage] = value
                    metadata[metadata_name] = merged
                unique_stages[stage.id] = replace(existing, metadata=metadata)
        variant_plans.append(
            ProgramVariantPlan(
                tuple(key),
                copy.deepcopy(dict(program)),
                {name: stage.id for name, stage in stages.items()},
            )
        )
    variant_plans.sort(key=lambda variant: canonical_json(list(variant.key)))
    return BundlePlan(
        program_id,
        target,
        tuple(variant_plans),
        tuple(unique_stages[stage_id] for stage_id in sorted(unique_stages)),
    )


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "ProgramVariantPlan",
    "ProgramCompileError",
    "TargetOptions",
    "build_program_plan",
    "canonical_json",
    "compiled_stage_from_program",
    "content_hash",
    "inline_artifact_descriptor",
    "parse_reflection_json",
    "select_artifact",
    "select_artifact_bytes",
    "select_entry",
    "serialize_bundle",
    "with_content_hash",
]
