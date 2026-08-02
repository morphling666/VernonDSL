from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .types import CompiledArtifact, CompiledStage, PipelineCompileError, TargetOptions, make_target_options


def parse_reflection_json(reflection: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(reflection, Mapping):
        return dict(reflection)
    try:
        value = json.loads(reflection)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineCompileError(f"compiler reflection is invalid JSON: {error}") from None
    if not isinstance(value, dict):
        raise PipelineCompileError("compiler reflection must be an object")
    return value


def select_entry(reflection: Mapping[str, Any], entry: str) -> dict[str, Any]:
    matches = [
        value for value in reflection.get("entries", []) if isinstance(value, dict) and value.get("name") == entry
    ]
    if len(matches) != 1:
        raise PipelineCompileError(f"compiler reflection does not contain exactly one '{entry}' entry")
    return matches[0]


def select_artifact(reflection: Mapping[str, Any], entry: str, stage: str) -> dict[str, Any]:
    matches = [
        value
        for value in reflection.get("artifacts", [])
        if isinstance(value, dict) and value.get("entry_point") == entry and value.get("stage") == stage
    ]
    if len(matches) != 1:
        raise PipelineCompileError(f"compiler did not emit exactly one {stage} artifact for {entry}")
    return matches[0]


def select_artifact_bytes(
    artifacts: Sequence[tuple[str, bytes]],
    artifact_row: Mapping[str, Any],
) -> tuple[str, bytes]:
    filename = artifact_row.get("filename")
    if not isinstance(filename, str) or not filename:
        raise PipelineCompileError("compiler artifact filename is invalid")
    matches = [(name, bytes(data)) for name, data in artifacts if name == filename]
    if len(matches) != 1:
        raise PipelineCompileError(f"compiler produced no unique artifact {filename!r}")
    return matches[0]


def compiled_stage_from_program(
    program: Any,
    *,
    module: str,
    module_manifest: str,
    entry: str,
    target: TargetOptions,
    metadata: Mapping[str, Any] = MappingProxyType({}),
) -> CompiledStage:
    """Normalize one owning compiler result into the shared stage model."""
    if not bool(program.ok):
        raise PipelineCompileError(str(program.diagnostics) or f"native compilation failed for {entry}")
    reflection = parse_reflection_json(program.reflection)
    interface = select_entry(reflection, entry)
    stage = interface.get("stage")
    if not isinstance(stage, str) or not stage:
        raise PipelineCompileError(f"compiler reflection has no stage for {entry}")
    artifact_row = select_artifact(reflection, entry, stage)
    artifact_name, artifact_data = select_artifact_bytes(program.artifacts, artifact_row)
    artifact_format = artifact_row.get("format")
    if not isinstance(artifact_format, str) or not artifact_format:
        raise PipelineCompileError("compiler artifact format is invalid")
    reflected_target = reflection.get("target")
    if isinstance(reflected_target, Mapping):
        kind = reflected_target.get("kind")
        options = reflected_target.get("options")
        if kind != target.target or not isinstance(options, Mapping):
            raise PipelineCompileError("compiler reflection target does not match the requested target")
        target = make_target_options(kind, options)
    return CompiledStage(
        module,
        module_manifest,
        entry,
        stage,
        target,
        reflection,
        interface,
        CompiledArtifact(artifact_format, artifact_data, artifact_name),
        metadata,
    )


__all__ = [
    "compiled_stage_from_program",
    "parse_reflection_json",
    "select_artifact",
    "select_artifact_bytes",
    "select_entry",
]
