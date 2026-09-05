from __future__ import annotations

import base64
import hashlib
from typing import Any, Mapping

from .deployment import deploy_program_variants
from .types import PIPELINE_VERSION, BundlePlan, CompiledArtifact, ProgramCompileError, canonical_json


def content_hash(value: Mapping[str, Any]) -> str:
    unhashed = dict(value)
    unhashed.pop("content_hash", None)
    return hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()


def with_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_hash", None)
    result["content_hash"] = content_hash(result)
    return result


def inline_artifact_descriptor(artifact: CompiledArtifact) -> dict[str, Any]:
    encoding = "base64" if artifact.format in {"spirv", "dxil"} else "utf8"
    try:
        data = (
            base64.b64encode(artifact.data).decode("ascii") if encoding == "base64" else artifact.data.decode("utf-8")
        )
    except UnicodeDecodeError:
        raise ProgramCompileError(f"{artifact.format} runtime artifact is not UTF-8") from None
    return {
        "format": artifact.format,
        "storage": "inline",
        "encoding": encoding,
        "data": data,
        "size": len(artifact.data),
        "sha256": artifact.sha256,
    }


def materialize_bundle(
    plan: BundlePlan,
    artifact_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if plan.variants and all(
        variant.canonical_program is not None and isinstance(variant.canonical_program.get("stages"), Mapping)
        for variant in plan.variants
    ):
        return with_content_hash(
            {
                "pipeline_version": PIPELINE_VERSION,
                "type": "program_bundle",
                "id": plan.pipeline_id,
                "target": plan.target.spec,
                "variants": deploy_program_variants(plan, artifact_descriptors),
            }
        )
    document = plan.logical_dict()
    records = document["stage_artifacts"]
    if set(records) != set(artifact_descriptors):
        raise ProgramCompileError("artifact descriptors do not match planned stages")
    for stage_id, descriptor in artifact_descriptors.items():
        stage = next(stage for stage in plan.stages if stage.id == stage_id)
        if descriptor.get("sha256") != stage.artifact.sha256:
            raise ProgramCompileError(f"artifact descriptor digest does not match stage {stage_id}")
        records[stage_id]["artifact"] = dict(descriptor)
        if stage.target.target == "cpu":
            symbol = stage.metadata.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                raise ProgramCompileError(f"CPU stage {stage_id} has no exported symbol")
            records[stage_id]["symbol"] = symbol
    return with_content_hash(document)


def serialize_bundle(bundle: Mapping[str, Any]) -> bytes:
    return (canonical_json(with_content_hash(bundle)) + "\n").encode("utf-8")


__all__ = [
    "canonical_json",
    "content_hash",
    "inline_artifact_descriptor",
    "materialize_bundle",
    "serialize_bundle",
    "with_content_hash",
]
