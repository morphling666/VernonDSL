from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..language.stage_registry import (
    GRAPHICS_STAGES,
    STAGE_BY_KIND,
    validate_graphics_topology,
    validate_stage_target,
)
from .parameters import (
    assign_parameter_slots,
    external_parameters,
    fragment_outputs,
    internal_parameters,
    merge_internal_parameter_uses,
    merge_parameter_uses,
    validate_graphics_interfaces,
)
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
    materialize_bundle,
    serialize_bundle,
    with_content_hash,
)
from .types import BundlePlan, CompiledArtifact, CompiledStage, PipelineCompileError, TargetOptions, VariantPlan


def plan_variant(
    key: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    slots: Mapping[str, int],
) -> VariantPlan:
    graphics = tuple(
        sorted(
            (stage for stage in records if stage in GRAPHICS_STAGES),
            key=lambda stage: STAGE_BY_KIND[stage].graphics_order,
        )
    )
    if ("compute" in records) == bool(graphics):
        raise PipelineCompileError("pipeline variant must contain either one compute program or one graphics program")
    if graphics:
        try:
            validate_graphics_topology(graphics)
        except ValueError as error:
            raise PipelineCompileError(str(error)) from None
        for producer, consumer in zip(graphics, graphics[1:], strict=False):
            validate_graphics_interfaces(producer, records[producer], consumer, records[consumer])
    external = external_parameters(records)
    internal = internal_parameters(records)
    parameters = []
    for name in sorted(external, key=lambda value: slots[value]):
        parameter = merge_parameter_uses(name, external[name])
        parameter["slot"] = slots[name]
        parameters.append(parameter)
    internal_rows = [merge_internal_parameter_uses(name, internal[name]) for name in sorted(internal)]
    stage_ids = {stage: str(record["id"]) for stage, record in records.items()}
    return VariantPlan(
        tuple(key),
        stage_ids,
        tuple(parameters),
        tuple(internal_rows),
        tuple(fragment_outputs(records)),
    )


def build_bundle_plan(
    pipeline_id: str,
    target: TargetOptions,
    features: Sequence[str],
    variants: Sequence[tuple[Sequence[str], Mapping[str, CompiledStage]]],
    transform: Mapping[str, Any] | None = None,
    autodiff_profiles: Mapping[str, Any] | None = None,
) -> BundlePlan:
    records_by_variant = [{name: stage.logical_record() for name, stage in stages.items()} for _, stages in variants]
    slots = assign_parameter_slots(records_by_variant)
    variant_plans = tuple(
        plan_variant(key, records, slots) for (key, _), records in zip(variants, records_by_variant, strict=True)
    )
    unique_stages = {stage.id: stage for _, stages in variants for stage in stages.values()}
    for stage in unique_stages.values():
        try:
            validate_stage_target(stage.stage, target.target)
        except ValueError as error:
            raise PipelineCompileError(str(error)) from None
    return BundlePlan(
        pipeline_id,
        target,
        tuple(sorted(set(features))),
        variant_plans,
        tuple(unique_stages[key] for key in sorted(unique_stages)),
        transform,
        autodiff_profiles,
    )


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "PipelineCompileError",
    "TargetOptions",
    "VariantPlan",
    "build_bundle_plan",
    "canonical_json",
    "compiled_stage_from_program",
    "content_hash",
    "inline_artifact_descriptor",
    "materialize_bundle",
    "parse_reflection_json",
    "plan_variant",
    "select_artifact",
    "select_artifact_bytes",
    "select_entry",
    "serialize_bundle",
    "with_content_hash",
]
