"""Canonical compiler/runtime backend matrix for backend-independent tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import vernon_dsl as vd
import vernon_dsl._runtime.session as runtime_session
from vernon_dsl._runtime.session import RuntimeUnavailableError

try:
    from vernon_dsl import _native
except (ImportError, OSError):
    _native = None


@dataclass(frozen=True)
class BackendRow:
    name: str
    architecture: object
    compiler_target: str
    runtime_backend: str


BACKEND_TEST_MATRIX = (
    BackendRow("CPU", vd.cpu, "CPU", "CPU"),
    BackendRow("CUDA", vd.cuda, "CUDA", "CUDA"),
    BackendRow("Vulkan", vd.vulkan, "VULKAN", "VULKAN"),
    BackendRow("DirectX12", vd.directx, "DIRECTX", "DIRECTX12"),
    BackendRow("Metal", vd.metal, "METAL", "METAL"),
    BackendRow("OpenGL", vd.opengl, "OPENGL", "OPENGL"),
    BackendRow("OpenGLES", vd.opengles, "OPENGL_ES", "OPENGL_ES"),
)


@dataclass(frozen=True)
class BackendRequirements:
    compute: bool = False
    graphics: bool = False
    storage_buffers: bool = False
    device_atomics: bool = False
    f32_atomic_add: bool = False
    f64_atomic_add: bool = False
    texture_sampler_operations: bool = False
    minimum_api_version: tuple[int, int] | None = None
    native_interop_backend: str | None = None


class ProbeKind(Enum):
    AVAILABLE = "available"
    PLATFORM_NOT_BUILT = "platform-not-built"
    DEVICE_OR_CONTEXT_UNAVAILABLE = "device-or-context-unavailable"
    CAPABILITY_UNSUPPORTED = "capability-unsupported"
    PROBE_FAILURE = "probe-failure"


@dataclass(frozen=True)
class ProbeResult:
    kind: ProbeKind
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.kind is ProbeKind.AVAILABLE


@dataclass(frozen=True)
class BackendSelection:
    available: tuple[BackendRow, ...]
    unavailable: tuple[tuple[BackendRow, ProbeResult], ...]


def _unsupported(row: BackendRow, capability: str) -> ProbeResult:
    return ProbeResult(
        ProbeKind.CAPABILITY_UNSUPPORTED,
        f"{row.name} does not support required capability {capability!r}",
    )


def probe_compiler(row: BackendRow, requirements: BackendRequirements) -> ProbeResult:
    if _native is None:
        return ProbeResult(
            ProbeKind.PLATFORM_NOT_BUILT,
            "native Vernon compiler extension is not built",
        )
    target = getattr(_native.Target, row.compiler_target)
    capabilities: dict[str, Any] = dict(_native.target_capabilities(target))
    if not capabilities["available"]:
        return ProbeResult(
            ProbeKind.PLATFORM_NOT_BUILT,
            f"{row.name} compiler target is not built on this platform",
        )
    for required, key in (
        (requirements.compute, "compute"),
        (requirements.graphics, "graphics"),
        (requirements.device_atomics, "device_storage_atomics"),
        (requirements.f32_atomic_add, "f32_device_atomic_add"),
    ):
        if required and not capabilities[key]:
            return _unsupported(row, key)
    # The public compiler capability record has no f64 field. Its current
    # target profile exposes a legal f64 atomic implementation only for CPU.
    if requirements.f64_atomic_add and row.compiler_target != "CPU":
        return _unsupported(row, "f64_atomic_add")
    if requirements.texture_sampler_operations:
        capability = "graphics_texture_sampling" if requirements.graphics else "compute_sampler_binding"
        program_capability = dict(_native._program_capability(capability))
        if not program_capability["supported"]:
            return _unsupported(row, capability)
    if requirements.native_interop_backend is not None and requirements.native_interop_backend != row.name:
        return _unsupported(row, f"{requirements.native_interop_backend}_native_interop")
    return ProbeResult(ProbeKind.AVAILABLE)


def _requested_api_version(row: BackendRow, requirements: BackendRequirements) -> tuple[int, int] | None:
    if requirements.minimum_api_version is not None:
        return requirements.minimum_api_version
    if requirements.compute and row.architecture == vd.opengl:
        return (4, 3)
    if requirements.compute and row.architecture == vd.opengles:
        return (3, 1)
    return None


def probe_runtime(row: BackendRow, requirements: BackendRequirements) -> ProbeResult:
    if _native is None:
        return ProbeResult(
            ProbeKind.DEVICE_OR_CONTEXT_UNAVAILABLE,
            "native Vernon runtime extension is not built",
        )
    options: dict[str, object] = {}
    api_version = _requested_api_version(row, requirements)
    if api_version is not None and row.architecture in {vd.opengl, vd.opengles}:
        options["api_version"] = api_version
    try:
        vd.init(arch=row.architecture, **options)  # type: ignore[arg-type]
    except RuntimeUnavailableError as error:
        vd.init(arch=vd.cpu)
        return ProbeResult(ProbeKind.DEVICE_OR_CONTEXT_UNAVAILABLE, str(error))

    native_runtime = runtime_session._native_runtime
    if native_runtime is None:
        raise RuntimeError(f"{row.name} initialization succeeded without creating a runtime context")
    capabilities: dict[str, Any] = dict(native_runtime.capabilities)
    if not capabilities["available"]:
        raise RuntimeError(f"{row.name} created context reported unavailable capabilities")
    for required, key in (
        (requirements.compute, "compute"),
        (requirements.graphics, "graphics"),
        (requirements.storage_buffers, "storage_buffers"),
    ):
        if required and not capabilities[key]:
            return _unsupported(row, key)
    if (
        requirements.minimum_api_version is not None
        and tuple(capabilities["api_version"]) < requirements.minimum_api_version
    ):
        return _unsupported(row, f"api_version>={requirements.minimum_api_version}")
    return ProbeResult(ProbeKind.AVAILABLE)


def probe_backend(row: BackendRow, requirements: BackendRequirements) -> ProbeResult:
    compiler = probe_compiler(row, requirements)
    return compiler if not compiler.available else probe_runtime(row, requirements)


def select_backends(requirements: BackendRequirements) -> BackendSelection:
    available: list[BackendRow] = []
    unavailable: list[tuple[BackendRow, ProbeResult]] = []
    for row in BACKEND_TEST_MATRIX:
        result = probe_backend(row, requirements)
        if result.available:
            available.append(row)
        else:
            unavailable.append((row, result))
    return BackendSelection(tuple(available), tuple(unavailable))


def unavailable_summary(selection: BackendSelection) -> str:
    return "; ".join(f"{row.name} [{result.kind.value}]: {result.reason}" for row, result in selection.unavailable)
