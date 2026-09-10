from __future__ import annotations

from typing import Any

from vernon_dsl import _native
from vernon_dsl.bundle import make_target_options

_TARGETS = {
    "cpu": _native.Target.CPU,
    "cuda": _native.Target.CUDA,
    "vulkan": _native.Target.VULKAN,
    "directx": _native.Target.DIRECTX,
    "metal": _native.Target.METAL,
    "opengl": _native.Target.OPENGL,
    "opengles": _native.Target.OPENGL_ES,
}


def compile_kernel_artifact(kernel: Any, target: str) -> tuple[bytes, str]:
    """Compile lowered kernel IR for compiler tests without creating a runtime execution surface."""

    options = make_target_options(
        target,
        {"version": 430} if target == "opengl" else {"version": 310} if target == "opengles" else {},
    )
    result = _native.Compiler().compile_program_result(
        kernel._lower(()).frontend.mlir,
        _TARGETS[target],
        **options.native_options,
    )
    if not result.ok:
        raise RuntimeError(result.diagnostics)
    if len(result.artifacts) != 1:
        raise RuntimeError("kernel compilation must produce exactly one artifact")
    return bytes(result.artifacts[0][1]), str(result.reflection)


__all__ = ["compile_kernel_artifact"]
