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

from .requirements import runtime_requirements
from .types import BundlePlan, ProgramCompileError, canonical_json


def _stage_artifact(
    logical_stage: str,
    stage: Any,
    descriptor: Mapping[str, Any],
    contract: Mapping[str, Any],
    target: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
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
    return blob, artifact, contract_hash


def deploy_program_variant(
    plan: BundlePlan,
    variant: Any,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, str]]:
    if variant.canonical_program is None:
        raise ProgramCompileError("canonical deployment has no Program")
    canonical_stage_rows = variant.canonical_program.get("stages")
    if not isinstance(canonical_stage_rows, Mapping):
        raise ProgramCompileError("canonical deployment has no logical stages")
    stages = {stage.id: stage for stage in plan.stages}
    blobs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    stage_bindings: dict[str, str] = {}
    for logical_stage, implementation_stage in variant.program.items():
        stage = stages.get(implementation_stage)
        descriptor = artifact_descriptors.get(implementation_stage)
        contracts = stage.metadata.get("program_contracts") if stage is not None else None
        contract = contracts.get(logical_stage) if isinstance(contracts, Mapping) else None
        canonical_stage = canonical_stage_rows.get(logical_stage)
        if not isinstance(descriptor, Mapping) or not isinstance(contract, Mapping):
            raise ProgramCompileError(f"canonical stage {logical_stage} is incomplete")
        if not isinstance(canonical_stage, Mapping):
            raise ProgramCompileError(f"canonical Program has no logical stage {logical_stage!r}")
        blob, artifact, contract_hash = _stage_artifact(
            logical_stage,
            stage,
            descriptor,
            contract,
            plan.target.target,
        )
        if canonical_stage.get("contract_hash") != contract_hash:
            raise ProgramCompileError(f"canonical stage {logical_stage} contract hash disagrees with its Program")
        blobs[blob["sha256"]] = blob
        artifacts[logical_stage] = artifact
        stage_bindings[logical_stage] = logical_stage
    return (
        copy.deepcopy(dict(variant.canonical_program)),
        {"target": copy.deepcopy(plan.target.spec), "blobs": blobs, "artifacts": artifacts},
        stage_bindings,
    )


def deploy_program_variants(
    plan: BundlePlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deploy every variant of a canonical Program plan."""

    variants = []
    for variant in plan.variants:
        program, artifact_system, stage_bindings = deploy_program_variant(plan, variant, artifact_descriptors)
        variants.append(
            {
                "key": list(variant.key),
                "program": dict(program),
                "artifact_system": dict(artifact_system),
                "stage_bindings": dict(stage_bindings),
            }
        )
    return variants


__all__ = ["deploy_program_variant", "deploy_program_variants"]
