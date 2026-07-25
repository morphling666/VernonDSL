from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..pipeline_compile import CompiledArtifact, PipelineCompileError, inline_artifact_descriptor


def encode_runtime_stage(record: dict[str, Any], artifact: bytes) -> dict[str, Any]:
    artifact_format = record.get("format")
    if not isinstance(artifact_format, str) or not artifact_format:
        raise PipelineCompileError("runtime stage artifact format is missing")
    return {
        **record,
        "artifact": inline_artifact_descriptor(CompiledArtifact(artifact_format, artifact)),
    }


def artifact_extension(artifact_format: str, stage: str, original_name: str = "") -> str:
    if artifact_format == "glsl":
        return {
            "vertex": ".vert.glsl",
            "fragment": ".frag.glsl",
            "compute": ".comp.glsl",
        }.get(stage, ".glsl")
    if artifact_format == "gles":
        return {
            "vertex": ".vert.gles",
            "fragment": ".frag.gles",
            "compute": ".comp.gles",
        }.get(stage, ".gles")
    if artifact_format == "msl":
        return {
            "vertex": ".vert.metal",
            "fragment": ".frag.metal",
            "compute": ".comp.metal",
        }.get(stage, ".metal")
    if artifact_format == "hlsl":
        return {
            "vertex": ".vert.hlsl",
            "fragment": ".frag.hlsl",
            "compute": ".comp.hlsl",
        }.get(stage, ".hlsl")
    if artifact_format == "ptx":
        return ".ptx"
    if artifact_format == "spirv":
        return ".spv"
    if artifact_format == "native_library":
        return Path(original_name).suffix or ".native"
    if artifact_format == "relocatable_object":
        suffix = Path(original_name).suffix
        return suffix if suffix in {".o", ".obj"} else ".o"
    return Path(original_name).suffix or f".{artifact_format}"


def write_external_artifact(
    output: Path,
    artifact: bytes,
    artifact_format: str,
    stage: str,
    original_name: str = "",
) -> dict[str, Any]:
    digest = hashlib.sha256(artifact).hexdigest()
    relative_path = Path("artifacts") / f"{digest}{artifact_extension(artifact_format, stage, original_name)}"
    artifact_path = output / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        if artifact_path.read_bytes() != artifact:
            raise PipelineCompileError(f"content-addressed artifact collision: {relative_path}")
    else:
        artifact_path.write_bytes(artifact)
    return {
        "format": artifact_format,
        "storage": "external",
        "path": relative_path.as_posix(),
        "size": len(artifact),
        "sha256": digest,
    }


__all__ = ["artifact_extension", "encode_runtime_stage", "write_external_artifact"]
