"""Construction of the canonical Program deployment.

This belongs to the deployment layer rather than the cook orchestrator. It reads a plan and its artifact descriptors
and returns the deployed form; it decides nothing about how the plan was produced and imports nothing from above.

Every variant is deployed in one pass. There is no single-variant restriction, and callers do not split a plan into
one plan per variant to work around one.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from .._versions import COMPILER_CONTRACT_VERSION, PROGRAM_VERSION
from .requirements import runtime_requirements
from .types import BundlePlan, ProgramCompileError, ProgramVariantPlan, canonical_json


def _stage_artifact(
    logical_stage: str,
    stage: Any,
    descriptor: Mapping[str, Any],
    contract: Mapping[str, Any],
    target: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The blob, artifact row, and contract hash one logical Program stage deploys to."""

    path = descriptor.get("path")
    digest = descriptor.get("sha256")
    byte_length = descriptor.get("size")
    symbol = stage.metadata.get("symbol")
    entry_point = symbol if isinstance(symbol, str) else stage.entry
    if (
        not isinstance(path, str)
        or not isinstance(digest, str)
        or not isinstance(byte_length, int)
        or not isinstance(entry_point, str)
    ):
        raise ProgramCompileError(f"canonical stage {logical_stage} has invalid artifact metadata")
    blob = {"byte_length": byte_length, "sha256": digest, "location": {"tag": "external", "uri": path}}

    graphics_stages = stage.metadata.get("graphics_compiled_stages")
    requirement_stages = tuple(graphics_stages) if isinstance(graphics_stages, tuple) else (stage,)
    requirements = runtime_requirements(target, requirement_stages)
    if requirements is None:
        raise ProgramCompileError(f"canonical stage {logical_stage} has no runtime requirements")
    requirements = dict(requirements)
    requirements.pop("compute_workgroup_size", None)
    reflection = contract.get("reflection")
    if not isinstance(reflection, Mapping):
        raise ProgramCompileError(f"canonical stage {logical_stage} has no portable reflection")
    contract_hash = hashlib.sha256(canonical_json(dict(contract)).encode("utf-8")).hexdigest()

    graphics_modules = stage.metadata.get("graphics_modules")
    if isinstance(graphics_modules, tuple):
        modules = [{**dict(module), "blob": digest} for module in graphics_modules]
    else:
        modules = [
            {
                "role": "compute",
                "format": stage.artifact.format,
                "entry_point": entry_point,
                "blob": digest,
                "offset": 0,
                "byte_length": byte_length,
                "sha256": digest,
            }
        ]
    artifact: dict[str, Any] = {
        "tag": "stage",
        "operation": "graphics" if isinstance(graphics_modules, tuple) else "compute",
        "contract_hash": contract_hash,
        "runtime_requirements": requirements,
        "modules": modules,
        "reflection": copy.deepcopy(dict(reflection)),
    }
    implementations = stage.metadata.get("program_implementations")
    implementation = implementations.get(logical_stage) if isinstance(implementations, Mapping) else None
    if isinstance(implementation, Mapping):
        artifact["implementation"] = copy.deepcopy(dict(implementation))
    return blob, artifact


def _deploy_program_variant(
    plan: BundlePlan,
    variant: ProgramVariantPlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_stage_rows = variant.program.get("stages")
    if not isinstance(canonical_stage_rows, Mapping):
        raise ProgramCompileError("canonical deployment has no logical stages")
    stages = {stage.id: stage for stage in plan.compiled_stages}
    blobs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    aggregate_stages = []
    for logical_stage, implementation_stage in variant.stage_implementations.items():
        stage = stages.get(implementation_stage)
        descriptor = artifact_descriptors.get(implementation_stage)
        contracts = stage.metadata.get("program_contracts") if stage is not None else None
        contract = contracts.get(logical_stage) if isinstance(contracts, Mapping) else None
        canonical_stage = canonical_stage_rows.get(logical_stage)
        if not isinstance(descriptor, Mapping) or not isinstance(contract, Mapping):
            raise ProgramCompileError(f"canonical stage {logical_stage} is incomplete")
        if not isinstance(canonical_stage, Mapping):
            raise ProgramCompileError(f"canonical Program has no logical stage {logical_stage!r}")
        blob, artifact = _stage_artifact(
            logical_stage,
            stage,
            descriptor,
            contract,
            plan.target.target,
        )
        if canonical_stage.get("contract_hash") != artifact["contract_hash"]:
            raise ProgramCompileError(f"canonical stage {logical_stage} contract hash disagrees with its Program")
        blobs[blob["sha256"]] = blob
        artifacts[logical_stage] = artifact
        graphics_stages = stage.metadata.get("graphics_compiled_stages")
        aggregate_stages.extend(graphics_stages if isinstance(graphics_stages, tuple) else (stage,))
    requirements = runtime_requirements(plan.target.target, aggregate_stages)
    if requirements is None:
        raise ProgramCompileError("canonical Program variant has no runtime requirements")
    requirements = dict(requirements)
    requirements.pop("compute_workgroup_size", None)
    return (
        {
            "key": list(variant.key),
            "program": copy.deepcopy(dict(variant.program)),
            "artifact_system": {
                "runtime_requirements": requirements,
                "artifacts": artifacts,
            },
        },
        blobs,
    )


def build_program_deployment(
    plan: BundlePlan,
    variant: ProgramVariantPlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Build one canonical Program and its exact variant ArtifactSystem."""

    deployed, _ = _deploy_program_variant(plan, variant, artifact_descriptors)
    return deployed["program"], deployed["artifact_system"]


def build_program_manifest(
    plan: BundlePlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize every planned variant into the exact Program bundle envelope."""

    stages = {stage.id: stage for stage in plan.compiled_stages}
    if set(artifact_descriptors) != set(stages):
        raise ProgramCompileError("artifact descriptors do not exactly match planned compiled stages")
    for stage_id, descriptor in artifact_descriptors.items():
        if descriptor.get("sha256") != stages[stage_id].artifact.sha256:
            raise ProgramCompileError(f"artifact descriptor digest does not match compiled stage {stage_id}")

    variants: list[dict[str, Any]] = []
    blobs: dict[str, Any] = {}
    for variant in plan.variants:
        deployed, variant_blobs = _deploy_program_variant(plan, variant, artifact_descriptors)
        variants.append(deployed)
        for blob_id, blob in variant_blobs.items():
            previous = blobs.get(blob_id)
            if previous is not None and previous != blob:
                raise ProgramCompileError(f"Program variants disagree on Blob {blob_id}")
            blobs[blob_id] = blob
    manifest: dict[str, Any] = {
        "compiler_contract_version": COMPILER_CONTRACT_VERSION,
        "program_version": PROGRAM_VERSION,
        "type": "program",
        "id": plan.program_id,
        "target": copy.deepcopy(plan.target.spec),
        "blobs": blobs,
        "variants": variants,
    }
    manifest["content_hash"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


__all__ = ["build_program_deployment", "build_program_manifest"]
