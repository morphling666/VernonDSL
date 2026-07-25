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
        from .types import PipelineCompileError

        raise PipelineCompileError(f"pipeline stages have inconsistent {name}")
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
        from .types import PipelineCompileError

        raise PipelineCompileError(f"{stage.target.target} artifact has no canonical #version directive")
    profile = match.group(2) or ("es" if stage.target.target == "opengles" else "core")
    if (stage.target.target == "opengles") != (profile == "es"):
        from .types import PipelineCompileError

        raise PipelineCompileError(f"{stage.target.target} artifact has incompatible GLSL profile {profile}")
    return int(match.group(1)), profile


def _spirv_version(stage: Any) -> tuple[int, int]:
    data = stage.artifact.data
    if len(data) < 8:
        from .types import PipelineCompileError

        raise PipelineCompileError("Vulkan artifact has no SPIR-V header")
    magic, version = struct.unpack_from("<II", data)
    if magic != 0x07230203:
        from .types import PipelineCompileError

        raise PipelineCompileError("Vulkan artifact has an invalid SPIR-V magic number")
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
        from .types import PipelineCompileError

        raise PipelineCompileError("CUDA artifact has an incomplete PTX header")
    sm = int(target.group(1))
    return (
        (int(version.group(1)), int(version.group(2))),
        (sm // 10, sm % 10),
        int(address_size.group(1)),
    )


def runtime_requirements(target: str, stages: Iterable[Any]) -> dict[str, Any] | None:
    stage_values = tuple(stages)
    if target not in {"cpu", "cuda", "vulkan", "opengl", "opengles"}:
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
                "invocation_abi_version": _single(
                    (stage.metadata.get("cpu_invocation_abi_version") for stage in stage_values),
                    "CPU invocation ABI versions",
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
    else:
        ptx = [_ptx_requirements(stage) for stage in stage_values]
        result["ptx_version"] = list(max(value[0] for value in ptx))
        result["minimum_compute_capability"] = list(max(value[1] for value in ptx))
        result["address_size"] = _single((value[2] for value in ptx), "PTX address sizes")
    return result


__all__ = ["runtime_requirements"]
