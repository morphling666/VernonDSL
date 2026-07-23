from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


class PipelineCompileError(ValueError):
    pass


def frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: value[key] for key in sorted(value)})


@dataclass(frozen=True)
class TargetOptions:
    target: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target:
            raise PipelineCompileError("target must be a non-empty string")
        options = dict(self.options)
        glsl_version = options.get("glsl_version")
        if glsl_version is not None:
            if self.target not in {"opengl", "opengles"}:
                raise PipelineCompileError("glsl_version is valid only for OpenGL targets")
            if not isinstance(glsl_version, int) or isinstance(glsl_version, bool) or glsl_version <= 0:
                raise PipelineCompileError("glsl_version must be a positive integer")
        for name in ("target_triple", "cpu", "cpu_features"):
            value = options.get(name)
            if value is not None and not isinstance(value, str):
                raise PipelineCompileError(f"{name} must be a string")
        object.__setattr__(self, "options", frozen_mapping(options))

    @property
    def native_options(self) -> dict[str, Any]:
        return {
            "glsl_version": int(self.options.get("glsl_version", 0)),
            "target_triple": str(self.options.get("target_triple", "")),
            "cpu": str(self.options.get("cpu", "")),
            "cpu_features": str(self.options.get("cpu_features", "")),
        }


@dataclass(frozen=True)
class CompiledArtifact:
    format: str
    data: bytes
    filename: str = ""

    def __post_init__(self) -> None:
        if not self.format:
            raise PipelineCompileError("compiled artifact format is missing")
        object.__setattr__(self, "data", bytes(self.data))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class CompiledStage:
    module: str
    module_manifest: str
    entry: str
    stage: str
    target: TargetOptions
    reflection: Mapping[str, Any]
    interface: Mapping[str, Any]
    artifact: CompiledArtifact
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entry or not self.stage:
            raise PipelineCompileError("compiled stage requires entry and stage")
        object.__setattr__(self, "reflection", frozen_mapping(self.reflection))
        object.__setattr__(self, "interface", frozen_mapping(self.interface))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "cache_version": 2,
            "compiler_version": 1,
            "module": self.module,
            "entry": self.entry,
            "stage": self.stage,
            "target": self.target.target,
            "target_options": dict(self.target.options),
            "dependencies": self.reflection.get("dependencies", []),
            "interface": dict(self.interface),
            "artifact_sha256": self.artifact.sha256,
        }

    @property
    def id(self) -> str:
        from .serialize import canonical_json

        return hashlib.sha256(canonical_json(self.identity).encode("utf-8")).hexdigest()

    def logical_record(self) -> dict[str, Any]:
        record = {
            "id": self.id,
            "module": self.module,
            "entry": self.entry,
            "stage": self.stage,
            "target": self.target.target,
            "format": self.artifact.format,
            "module_hash": self.reflection.get("module_hash"),
            "dependencies": self.reflection.get("dependencies", []),
            "interface": dict(self.interface),
            "reflection": dict(self.reflection),
            **dict(self.metadata),
        }
        if not self.module:
            record.pop("module")
        return record


@dataclass(frozen=True)
class VariantPlan:
    key: tuple[str, ...]
    stages: Mapping[str, str]
    parameters: tuple[Mapping[str, Any], ...]
    internal_parameters: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    steps: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", frozen_mapping(self.stages))
        for name in ("parameters", "internal_parameters", "outputs", "steps"):
            values = tuple(frozen_mapping(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "key": list(self.key),
            "parameters": [dict(value) for value in self.parameters],
            "outputs": [dict(value) for value in self.outputs],
            "steps": [dict(value) for value in self.steps],
        }
        if self.internal_parameters:
            result["internal_parameters"] = [dict(value) for value in self.internal_parameters]
        return result


@dataclass(frozen=True)
class BundlePlan:
    pipeline_id: str
    target: TargetOptions
    features: tuple[str, ...]
    variants: tuple[VariantPlan, ...]
    stages: tuple[CompiledStage, ...]

    def logical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "invocation_abi_version": 3,
            "type": "pipeline",
            "id": self.pipeline_id,
            "target": self.target.target,
            "target_options": dict(self.target.options),
            "features": list(self.features),
            "variants": [variant.to_dict() for variant in self.variants],
            "stage_artifacts": {
                stage.id: stage.logical_record() for stage in sorted(self.stages, key=lambda value: value.id)
            },
        }


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "PipelineCompileError",
    "TargetOptions",
    "VariantPlan",
]
