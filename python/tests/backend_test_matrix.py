"""Canonical compiler/runtime backend matrix for backend-independent tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any
from unittest import SkipTest

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
    rhi_backend: str | None


BACKEND_TEST_MATRIX = (
    BackendRow("CPU", vd.cpu, "CPU", "CPU", None),
    BackendRow("CUDA", vd.cuda, "CUDA", "CUDA", "CUDA"),
    BackendRow("Vulkan", vd.vulkan, "VULKAN", "VULKAN", "VULKAN"),
    BackendRow("DirectX12", vd.directx, "DIRECTX", "DIRECTX12", "DIRECTX12"),
    BackendRow("Metal", vd.metal, "METAL", "METAL", "METAL"),
    BackendRow("OpenGL", vd.opengl, "OPENGL", "OPENGL", "OPENGL"),
    BackendRow("OpenGLES", vd.opengles, "OPENGL_ES", "OPENGL_ES", "OPENGL_ES"),
)


@dataclass(frozen=True)
class BackendRequirements:
    gpu: bool = False
    rhi: bool = False
    dynamic_range_step: bool = False
    f16: bool = False
    f64: bool = False
    compute: bool = False
    graphics: bool = False
    storage_buffers: bool = False
    storage_texture: bool = False
    device_atomics: bool = False
    f32_atomic_add: bool = False
    f64_atomic_add: bool = False
    texture_sampler_operations: bool = False
    program_vjp: bool = False
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


class CapabilityUnavailable(SkipTest):
    def __init__(self, result: ProbeResult) -> None:
        if result.kind not in {
            ProbeKind.PLATFORM_NOT_BUILT,
            ProbeKind.DEVICE_OR_CONTEXT_UNAVAILABLE,
            ProbeKind.CAPABILITY_UNSUPPORTED,
        }:
            raise ValueError(f"{result.kind.value} is not a skippable capability result")
        super().__init__(f"{result.kind.value}: {result.reason}")
        self.result = result


def require_available(result: ProbeResult) -> None:
    if result.available:
        return
    if result.kind is ProbeKind.PROBE_FAILURE:
        raise RuntimeError(result.reason)
    raise CapabilityUnavailable(result)


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
    if requirements.gpu and row.architecture == vd.cpu:
        return _unsupported(row, "gpu")
    if requirements.rhi and row.rhi_backend is None:
        return _unsupported(row, "rhi")
    for required, key in (
        (requirements.dynamic_range_step, "dynamic_range_step"),
        (requirements.f16, "f16"),
        (requirements.f64, "f64"),
        (requirements.compute, "compute"),
        (requirements.graphics, "graphics"),
        (requirements.device_atomics, "device_storage_atomics"),
        (requirements.f32_atomic_add, "f32_device_atomic_add"),
        (requirements.f64_atomic_add, "f64_device_atomic_add"),
    ):
        if required and not capabilities[key]:
            return _unsupported(row, key)
    if requirements.texture_sampler_operations:
        capability = "graphics_texture_sampling" if requirements.graphics else "compute_sampler_binding"
        program_capability = dict(_native._program_capability(capability))
        if not program_capability["supported"]:
            return _unsupported(row, capability)
    for required, capability in (
        (requirements.storage_texture, "compute_texture_binding"),
        (requirements.program_vjp, "compute_vjp"),
    ):
        if required and not dict(_native._program_capability(capability))["supported"]:
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

    selected_session = runtime_session.current_session()
    if selected_session is None:
        raise RuntimeError(f"{row.name} initialization succeeded without creating a runtime context")
    native_runtime = selected_session.native_runtime
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
    if api_version is not None and tuple(capabilities["api_version"]) < api_version:
        return _unsupported(row, f"api_version>={api_version}")
    return ProbeResult(ProbeKind.AVAILABLE)


def probe_backend(row: BackendRow, requirements: BackendRequirements) -> ProbeResult:
    compiler = probe_compiler(row, requirements)
    return compiler if not compiler.available else probe_runtime(row, requirements)


def backend_row(architecture: object) -> BackendRow:
    for row in BACKEND_TEST_MATRIX:
        if row.architecture == architecture:
            return row
    raise ValueError(f"architecture is not in the canonical backend matrix: {architecture!r}")


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


_MATRIX_REQUIREMENTS_ATTRIBUTE = "__vernon_backend_matrix_requirements__"


def backend_matrix_test(requirements: BackendRequirements):
    def decorate(function):
        setattr(function, _MATRIX_REQUIREMENTS_ATTRIBUTE, requirements)
        return function

    return decorate


def expand_backend_matrix_tests(test_class):
    for name, function in tuple(vars(test_class).items()):
        requirements = getattr(function, _MATRIX_REQUIREMENTS_ATTRIBUTE, None)
        if requirements is None:
            continue
        delattr(test_class, name)
        for row in BACKEND_TEST_MATRIX:
            test_name = f"{name}_{row.runtime_backend.lower()}"

            @wraps(function)
            def run(self, _function=function, _requirements=requirements, _row=row):
                try:
                    require_available(probe_backend(_row, _requirements))
                    return _function(self, _row)
                finally:
                    if _row.architecture != vd.cpu:
                        vd.init(arch=vd.cpu)

            run.__name__ = test_name
            run.__qualname__ = f"{test_class.__qualname__}.{test_name}"
            setattr(test_class, test_name, run)
    return test_class
