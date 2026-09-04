from __future__ import annotations

import re
import struct
from typing import Any, Iterable

_PTX_VERSION = re.compile(r"(?m)^\.version\s+(\d+)\.(\d+)\s*$")
_PTX_TARGET = re.compile(r"(?m)^\.target\s+(?:sm_|compute_)(\d+)(?:\s|,|$)")
_PTX_ADDRESS_SIZE = re.compile(r"(?m)^\.address_size\s+(32|64)\s*$")
_GLSL_VERSION = re.compile(r"^#version\s+(\d{3})(?:\s+(es|core|compatibility))?\s*$")


def _single(values: Iterable[Any], name: str) -> Any:
    unique = {value for value in values}
    if len(unique) != 1:
        from .types import ProgramCompileError

        raise ProgramCompileError(f"pipeline stages have inconsistent {name}")
    return next(iter(unique))


def _features(stages: Iterable[Any]) -> list[str]:
    values: set[str] = set()
    for stage in stages:
        reflected = stage.reflection.get("required_features", [])
        if isinstance(reflected, list):
            values.update(value for value in reflected if isinstance(value, str))
    return sorted(values)


def _glsl_requirement(stage: Any) -> tuple[int, str]:
    try:
        first_line = stage.artifact.data.decode("utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        first_line = ""
    match = _GLSL_VERSION.fullmatch(first_line)
    if not match:
        from .types import ProgramCompileError

        raise ProgramCompileError(f"{stage.target.target} artifact has no canonical #version directive")
    profile = match.group(2) or ("es" if stage.target.target == "opengles" else "core")
    if (stage.target.target == "opengles") != (profile == "es"):
        from .types import ProgramCompileError

        raise ProgramCompileError(f"{stage.target.target} artifact has incompatible GLSL profile {profile}")
    return int(match.group(1)), profile


def _spirv_version(stage: Any) -> tuple[int, int]:
    data = stage.artifact.data
    if len(data) < 8:
        from .types import ProgramCompileError

        raise ProgramCompileError("Vulkan artifact has no SPIR-V header")
    magic, version = struct.unpack_from("<II", data)
    if magic != 0x07230203:
        from .types import ProgramCompileError

        raise ProgramCompileError("Vulkan artifact has an invalid SPIR-V magic number")
    return ((version >> 16) & 0xFF, (version >> 8) & 0xFF)


def _ptx_requirements(stage: Any) -> tuple[tuple[int, int], tuple[int, int], int]:
    try:
        source = stage.artifact.data.decode("utf-8")
    except UnicodeDecodeError:
        source = ""
    version = _PTX_VERSION.search(source)
    target = _PTX_TARGET.search(source)
    address_size = _PTX_ADDRESS_SIZE.search(source)
    if not version or not target or not address_size:
        from .types import ProgramCompileError

        raise ProgramCompileError("CUDA artifact has an incomplete PTX header")
    sm = int(target.group(1))
    return (
        (int(version.group(1)), int(version.group(2))),
        (sm // 10, sm % 10),
        int(address_size.group(1)),
    )


def _metal_version(stage: Any, name: str) -> tuple[int, int]:
    reflected_target = stage.reflection.get("target", {})
    output = reflected_target.get("output", {}) if isinstance(reflected_target, dict) else {}
    output_name = "version" if name == "msl_version" else name
    value = output.get(output_name) if isinstance(output, dict) else None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) or part < 0 for part in value)
    ):
        from .types import ProgramCompileError

        raise ProgramCompileError(f"Metal compiler reflection has no valid {name}")
    return (value[0], value[1])


def runtime_requirements(target: str, stages: Iterable[Any]) -> dict[str, Any] | None:
    stage_values = tuple(stages)
    if target not in {"cpu", "cuda", "vulkan", "opengl", "opengles", "directx", "metal"}:
        return None
    result: dict[str, Any] = {"backend": target, "features": _features(stage_values)}
    if target == "cpu":
        result.update(
            {
                "target_triple": _single(
                    (stage.metadata.get("target_triple") for stage in stage_values), "CPU target triples"
                ),
                "object_format": _single(
                    (stage.metadata.get("object_format") for stage in stage_values), "CPU object formats"
                ),
            }
        )
    elif target in {"opengl", "opengles"}:
        glsl = [_glsl_requirement(stage) for stage in stage_values]
        version = max(value[0] for value in glsl)
        result["glsl_version"] = version
        result["profile"] = _single((value[1] for value in glsl), "GLSL profiles")
        result["api_version"] = [version // 100, (version % 100) // 10]
    elif target == "vulkan":
        version = max(_spirv_version(stage) for stage in stage_values)
        workgroups = [
            tuple(stage.interface.get("workgroup_size", (1, 1, 1)))
            for stage in stage_values
            if stage.stage == "compute"
        ]
        result["api_version"] = [1, 1]
        result["spirv_version"] = list(version)
        if workgroups:
            result["compute_workgroup_size"] = [max(value[index] for value in workgroups) for index in range(3)]
    elif target == "directx":
        for stage in stage_values:
            if len(stage.artifact.data) < 4 or stage.artifact.data[:4] != b"DXBC" or stage.artifact.format != "dxil":
                from .types import ProgramCompileError

                raise ProgramCompileError("DirectX runtime artifact is not a DXIL container")
        shader_model = _single(
            (stage.target.options.get("shader_model", 60) for stage in stage_values), "HLSL Shader Models"
        )
        if not isinstance(shader_model, int) or shader_model < 60:
            from .types import ProgramCompileError

            raise ProgramCompileError("DirectX runtime requires Shader Model 6.0 or newer")
        result["api_version"] = [12, 0]
        result["minimum_feature_level"] = [11, 0]
        result["shader_model"] = [shader_model // 10, shader_model % 10]
        result["root_signature_version"] = [1, 0]
        workgroups = [
            tuple(stage.interface.get("workgroup_size", (1, 1, 1)))
            for stage in stage_values
            if stage.stage == "compute"
        ]
        if workgroups:
            result["compute_workgroup_size"] = [max(value[index] for value in workgroups) for index in range(3)]
    elif target == "metal":
        platform = _single((stage.target.options.get("platform") for stage in stage_values), "Apple Metal platforms")
        if platform not in {"macos", "ios"}:
            from .types import ProgramCompileError

            raise ProgramCompileError("Metal runtime requires apple_platform 'macos' or 'ios'")
        msl_version = _single((_metal_version(stage, "msl_version") for stage in stage_values), "MSL versions")
        minimum_os_version = _single(
            (_metal_version(stage, "minimum_os_version") for stage in stage_values), "Metal minimum OS versions"
        )
        required_os = (15, 0) if platform == "ios" else (11, 0)
        if msl_version != (2, 4) or minimum_os_version < required_os:
            from .types import ProgramCompileError

            raise ProgramCompileError("Metal compiler reflection contains unsupported runtime requirements")
        result["apple_platform"] = platform
        result["msl_version"] = list(msl_version)
        result["minimum_os_version"] = list(minimum_os_version)
    else:
        ptx = [_ptx_requirements(stage) for stage in stage_values]
        result["ptx_version"] = list(max(value[0] for value in ptx))
        result["minimum_compute_capability"] = list(max(value[1] for value in ptx))
        result["address_size"] = _single((value[2] for value in ptx), "PTX address sizes")
    return result


__all__ = ["runtime_requirements"]
