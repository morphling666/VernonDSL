from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, TypeAlias

from .._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION


class ProgramCompileError(ValueError):
    pass


def frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: value[key] for key in sorted(value)})


class _TargetOptionsBase:
    target: ClassVar[str]

    @property
    def options(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @property
    def spec(self) -> dict[str, Any]:
        return {"kind": self.target, "options": dict(self.options)}

    @property
    def native_options(self) -> dict[str, Any]:
        return {"options": dict(self.options)}


@dataclass(frozen=True)
class CpuTargetOptions(_TargetOptionsBase):
    target: ClassVar[str] = "cpu"
    triple: str = ""
    processor: str = ""
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.triple, str) or not isinstance(self.processor, str):
            raise ProgramCompileError("CPU triple and processor must be strings")
        if not isinstance(self.features, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in self.features
        ):
            raise ProgramCompileError("CPU features must be a sequence of non-empty strings")
        object.__setattr__(self, "features", tuple(self.features))

    @property
    def options(self) -> Mapping[str, Any]:
        return frozen_mapping(
            {
                name: value
                for name, value in (
                    ("triple", self.triple),
                    ("processor", self.processor),
                    ("features", list(self.features)),
                )
                if value
            }
        )

    @property
    def native_options(self) -> dict[str, Any]:
        options = dict(self.options)
        if self.features:
            options["features"] = ",".join(self.features)
        return {"options": options}


@dataclass(frozen=True)
class OpenGLTargetOptions(_TargetOptionsBase):
    target: ClassVar[str] = "opengl"
    version: int | None = None

    def __post_init__(self) -> None:
        if self.version is not None and (
            not isinstance(self.version, int) or isinstance(self.version, bool) or not 100 <= self.version <= 999
        ):
            raise ProgramCompileError("OpenGL version must be a three-digit GLSL version number")

    @property
    def options(self) -> Mapping[str, Any]:
        return frozen_mapping({"version": self.version} if self.version is not None else {})


@dataclass(frozen=True)
class OpenGLESTargetOptions(OpenGLTargetOptions):
    target: ClassVar[str] = "opengles"


@dataclass(frozen=True)
class VulkanTargetOptions(_TargetOptionsBase):
    target: ClassVar[str] = "vulkan"

    @property
    def options(self) -> Mapping[str, Any]:
        return frozen_mapping({})


@dataclass(frozen=True)
class MetalTargetOptions(_TargetOptionsBase):
    target: ClassVar[str] = "metal"
    platform: str = "macos"

    def __post_init__(self) -> None:
        if self.platform not in {"macos", "ios"}:
            raise ProgramCompileError("Metal platform must be 'macos' or 'ios'")

    @property
    def options(self) -> Mapping[str, Any]:
        return frozen_mapping({"platform": self.platform})


@dataclass(frozen=True)
class DirectXTargetOptions(_TargetOptionsBase):
    target: ClassVar[str] = "directx"
    shader_model: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.shader_model, int) or isinstance(self.shader_model, bool) or self.shader_model < 60:
            raise ProgramCompileError("DirectX shader model must be 6.0 or newer")

    @property
    def options(self) -> Mapping[str, Any]:
        return frozen_mapping({"shader_model": self.shader_model})


@dataclass(frozen=True)
class CudaTargetOptions(_TargetOptionsBase):
    target: ClassVar[str] = "cuda"

    @property
    def options(self) -> Mapping[str, Any]:
        return frozen_mapping({})


TargetOptions: TypeAlias = (
    CpuTargetOptions
    | OpenGLTargetOptions
    | OpenGLESTargetOptions
    | VulkanTargetOptions
    | MetalTargetOptions
    | DirectXTargetOptions
    | CudaTargetOptions
)


def make_target_options(target: str, options: Mapping[str, Any] | None = None) -> TargetOptions:
    values = dict(options or {})
    target = "directx" if target == "dx" else target
    constructors = {
        "cpu": CpuTargetOptions,
        "opengl": OpenGLTargetOptions,
        "opengles": OpenGLESTargetOptions,
        "vulkan": VulkanTargetOptions,
        "metal": MetalTargetOptions,
        "directx": DirectXTargetOptions,
        "cuda": CudaTargetOptions,
    }
    try:
        constructor = constructors[target]
    except KeyError:
        raise ProgramCompileError(f"unknown compiler target '{target}'") from None
    try:
        return constructor(**values)
    except TypeError as error:
        raise ProgramCompileError(f"invalid {target} target options: {error}") from None


@dataclass(frozen=True)
class CompiledArtifact:
    format: str
    data: bytes
    filename: str = ""

    def __post_init__(self) -> None:
        if not self.format:
            raise ProgramCompileError("compiled artifact format is missing")
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
            raise ProgramCompileError("compiled stage requires entry and stage")
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
            "target": self.target.spec,
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
            "entry": self.entry,
            "stage": self.stage,
            "reflection": dict(self.reflection),
        }
        return record


@dataclass(frozen=True)
class VariantPlan:
    key: tuple[str, ...]
    program: Mapping[str, str]
    parameters: tuple[Mapping[str, Any], ...]
    internal_parameters: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]
    canonical_program: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if any(not feature for feature in self.key) or tuple(sorted(set(self.key))) != self.key:
            raise ProgramCompileError("pipeline variant key must contain unique non-empty features in sorted order")
        object.__setattr__(self, "program", frozen_mapping(self.program))
        if self.canonical_program is not None:
            object.__setattr__(self, "canonical_program", frozen_mapping(self.canonical_program))
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
    transform: Mapping[str, Any] | None = None
    autodiff_profiles: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.transform is None) != (self.autodiff_profiles is None):
            raise ProgramCompileError("differentiated bundles require both program_transform and autodiff_profiles")
        if self.transform is not None:
            object.__setattr__(self, "transform", frozen_mapping(self.transform))
        if self.autodiff_profiles is not None:
            object.__setattr__(self, "autodiff_profiles", frozen_mapping(self.autodiff_profiles))

    def logical_dict(self) -> dict[str, Any]:
        from .requirements import runtime_requirements

        variant_records = [variant.to_dict() for variant in self.variants]
        variant_keys = [variant.key for variant in self.variants]
        if len(set(variant_keys)) != len(variant_keys):
            raise ProgramCompileError("pipeline variants contain duplicate canonical keys")
        if self.transform is not None:
            profile_rows = (self.autodiff_profiles or {})["variants"]
            profile_keys = [tuple(profile["key"]) for profile in profile_rows]
            if any(any(not feature for feature in key) or tuple(sorted(set(key))) != key for key in profile_keys):
                raise ProgramCompileError("autodiff profile variant keys are not canonical")
            if len(set(profile_keys)) != len(profile_keys):
                raise ProgramCompileError("autodiff profiles contain duplicate variant keys")
            profiles = {
                tuple(profile["key"]): {name: value for name, value in profile.items() if name != "key"}
                for profile in profile_rows
            }
            if set(profiles) != {variant.key for variant in self.variants}:
                raise ProgramCompileError("autodiff profiles do not exactly cover pipeline variants")
        result = {
            "pipeline_version": PIPELINE_VERSION,
            "type": "pipeline",
            "id": self.pipeline_id,
            "target": self.target.spec,
            "variants": variant_records,
            "stage_artifacts": {
                stage.id: stage.logical_record() for stage in sorted(self.stages, key=lambda value: value.id)
            },
        }
        if self.transform is not None:
            result["autodiff"] = {
                "kind": self.transform["kind"],
                "wrt": list(self.transform["wrt"]),
                "output_cotangents": list(self.transform["output_cotangents"]),
                "profiles": [
                    {
                        "variant_key": list(variant.key),
                        **{name: value for name, value in profiles[variant.key].items() if name != "profiles"},
                        **profiles[variant.key]["profiles"],
                    }
                    for variant in self.variants
                ],
            }
        requirements = runtime_requirements(self.target.target, self.stages)
        if requirements is not None:
            result["runtime_requirements"] = requirements
        return result


__all__ = [
    "BundlePlan",
    "CompiledArtifact",
    "CompiledStage",
    "CpuTargetOptions",
    "CudaTargetOptions",
    "DirectXTargetOptions",
    "MetalTargetOptions",
    "OpenGLESTargetOptions",
    "OpenGLTargetOptions",
    "ProgramCompileError",
    "TargetOptions",
    "VulkanTargetOptions",
    "VariantPlan",
    "make_target_options",
]
