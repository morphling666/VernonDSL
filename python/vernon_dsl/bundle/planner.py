from __future__ import annotations

import copy
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
            key=lambda stage: STAGE_BY_KIND[stage].graphics_order or 0,
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
        None,
    )


def build_bundle_plan(
    pipeline_id: str,
    target: TargetOptions,
    features: Sequence[str],
    variants: Sequence[tuple[Sequence[str], Mapping[str, CompiledStage]]],
    transform: Mapping[str, Any] | None = None,
    autodiff_profiles: Mapping[str, Any] | None = None,
) -> BundlePlan:
    records_by_variant = [
        {
            name: {
                "id": stage.id,
                "entry": stage.entry,
                "target": stage.target.target,
                "interface": dict(stage.interface),
            }
            for name, stage in stages.items()
        }
        for _, stages in variants
    ]
    slots = assign_parameter_slots(records_by_variant)
    variant_plans = tuple(
        plan_variant(key, records, slots) for (key, _), records in zip(variants, records_by_variant, strict=True)
    )
    unique_stages = {stage.id: stage for _, stages in variants for stage in stages.values()}
    for stage in unique_stages.values():
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


def build_program_bundle_plan(
    pipeline_id: str,
    target: TargetOptions,
    key: Sequence[str],
    stages: Mapping[str, CompiledStage],
    canonical_program: Mapping[str, Any],
) -> BundlePlan:
    """Package one compiler-finalized Program variant and its stage artifacts."""

    canonical_stages = canonical_program.get("stages")
    if not isinstance(canonical_stages, Mapping):
        raise PipelineCompileError("canonical Program has no stage contracts")
    if set(canonical_stages) != set(stages):
        missing = sorted(set(canonical_stages) - set(stages))
        unused = sorted(set(stages) - set(canonical_stages))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(repr(value) for value in missing))
        if unused:
            detail.append("unused " + ", ".join(repr(value) for value in unused))
        raise PipelineCompileError("Program implementation stages do not match its contracts: " + "; ".join(detail))

    stage_bindings: dict[str, str] = {}
    unique_stages: dict[str, CompiledStage] = {}
    for name, stage in stages.items():
        contract = canonical_stages[name]
        operation = contract.get("operation") if isinstance(contract, Mapping) else None
        expected = "compute" if operation == "compute" else "graphics" if operation == "graphics" else None
        if expected is None:
            raise PipelineCompileError(f"canonical Program stage {name!r} has an invalid operation")
        if stage.stage != expected:
            raise PipelineCompileError(f"Program stage {name!r} requires {expected}, compiler produced {stage.stage}")
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
            raise PipelineCompileError(str(error)) from None
        if stage.target.spec != target.spec:
            raise PipelineCompileError(f"Program stage {name!r} target does not match the bundle target")
        stage_bindings[name] = stage.id
        unique_stages[stage.id] = stage

    variant = VariantPlan(
        tuple(key),
        stage_bindings,
        (),
        (),
        (),
        copy.deepcopy(dict(canonical_program)),
    )
    return BundlePlan(
        pipeline_id,
        target,
        tuple(sorted(set(key))),
        (variant,),
        tuple(unique_stages[stage_id] for stage_id in sorted(unique_stages)),
    )


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "PipelineCompileError",
    "TargetOptions",
    "VariantPlan",
    "build_bundle_plan",
    "build_program_bundle_plan",
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
