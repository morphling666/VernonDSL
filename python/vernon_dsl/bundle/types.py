from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION


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
        hlsl_shader_model = options.get("hlsl_shader_model")
        if hlsl_shader_model is not None:
            if self.target != "directx":
                raise PipelineCompileError("hlsl_shader_model is valid only for the DirectX target")
            if not isinstance(hlsl_shader_model, int) or isinstance(hlsl_shader_model, bool) or hlsl_shader_model < 60:
                raise PipelineCompileError("hlsl_shader_model must be Shader Model 6.0 or newer")
        apple_platform = options.get("apple_platform")
        if apple_platform is not None:
            if self.target != "metal":
                raise PipelineCompileError("apple_platform is valid only for the Metal target")
            if apple_platform not in {"macos", "ios"}:
                raise PipelineCompileError("apple_platform must be 'macos' or 'ios'")
        for name in ("msl_version", "minimum_os_version"):
            value = options.get(name)
            if value is not None:
                if self.target != "metal":
                    raise PipelineCompileError(f"{name} is valid only for the Metal target")
                if (
                    not isinstance(value, (list, tuple))
                    or len(value) != 2
                    or any(not isinstance(part, int) or isinstance(part, bool) or part < 0 for part in value)
                ):
                    raise PipelineCompileError(f"{name} must be a two-component non-negative integer version")
                options[name] = tuple(value)
        cpu_option_names = ("target_triple", "cpu", "cpu_features")
        for name in cpu_option_names:
            value = options.get(name)
            if value is not None and not isinstance(value, str):
                raise PipelineCompileError(f"{name} must be a string")
        if self.target != "cpu" and any(name in options for name in cpu_option_names):
            raise PipelineCompileError("target_triple, cpu, and cpu_features are valid only for the CPU target")
        object.__setattr__(self, "options", frozen_mapping(options))

    @property
    def native_options(self) -> dict[str, Any]:
        if self.target in {"opengl", "opengles"}:
            return {"glsl_version": self.options["glsl_version"]} if "glsl_version" in self.options else {}
        if self.target == "cpu":
            return {
                name: self.options[name] for name in ("target_triple", "cpu", "cpu_features") if name in self.options
            }
        if self.target == "directx":
            return {"hlsl_shader_model": self.options.get("hlsl_shader_model", 60)}
        if self.target == "metal":
            return {"apple_platform": self.options.get("apple_platform", "macos")}
        return {}


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
            "compiler_contract_version": COMPILER_CONTRACT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
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
    program: Mapping[str, str]
    parameters: tuple[Mapping[str, Any], ...]
    internal_parameters: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "program", frozen_mapping(self.program))
        for name in ("parameters", "internal_parameters", "outputs"):
            values = tuple(frozen_mapping(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "key": list(self.key),
            "program": dict(self.program),
            "parameters": [dict(value) for value in self.parameters],
            "outputs": [dict(value) for value in self.outputs],
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
        from .requirements import runtime_requirements

        result = {
            "pipeline_version": PIPELINE_VERSION,
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
        requirements = runtime_requirements(self.target.target, self.stages)
        if requirements is not None:
            result["runtime_requirements"] = requirements
        return result


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "PipelineCompileError",
    "TargetOptions",
    "VariantPlan",
]
