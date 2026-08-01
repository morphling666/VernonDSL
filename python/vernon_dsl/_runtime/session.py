from __future__ import annotations

import atexit
import weakref
from dataclasses import dataclass
from typing import Any

from .kernel import Kernel
from .pipeline import Pipeline, PrimitiveTopology, lines, pipeline, points, triangles
from .resources import TensorLayout, TensorStorage, TensorView, Texture

try:
    from .. import _native
except (ImportError, OSError):
    _native = None
try:
    from .. import _gl_context
except (ImportError, OSError):
    _gl_context = None


@dataclass(frozen=True)
class _Architecture:
    name: str


cpu = _Architecture("cpu")
cuda = _Architecture("cuda")
vulkan = _Architecture("vulkan")
directx = _Architecture("directx")
metal = _Architecture("metal")
opengl = _Architecture("opengl")
opengles = _Architecture("opengles")
_architecture = cpu
_native_runtime: Any | None = None
_rhi_host: Any | None = None
_owned_opengl_context: Any | None = None
_runtime_generation = 0
_api_version: tuple[int, int] | None = None
_external_opengl_contexts: dict[_Architecture, tuple[int, int, int, tuple[int, int]]] = {}
_runtime_children: weakref.WeakSet[Any] = weakref.WeakSet()


class RuntimeUnavailableError(RuntimeError):
    """Raised when the requested runtime cannot be created on this host."""


def _release_runtime() -> None:
    global _native_runtime, _rhi_host, _owned_opengl_context
    Kernel.invalidate_loaded()
    for compiled in Pipeline._cache.values():
        compiled.native = None
    for child in list(_runtime_children):
        if hasattr(child, "_native_buffer"):
            child._native_buffer = None
        if hasattr(child, "_native_texture"):
            child._native_texture = None
        if hasattr(child, "_native_sampler"):
            child._native_sampler = None
        if hasattr(child, "_dispose_native"):
            child._dispose_native()
        if hasattr(child, "_compiled"):
            child._compiled = None
            child._compiled_generation = -1
    _native_runtime = None
    _rhi_host = None
    _owned_opengl_context = None


atexit.register(_release_runtime)


def init(*, arch: _Architecture = cpu, api_version: tuple[int, int] | None = None) -> None:
    global _architecture, _native_runtime, _rhi_host, _owned_opengl_context
    global _runtime_generation, _api_version
    if arch not in {cpu, cuda, vulkan, directx, metal, opengl, opengles}:
        raise ValueError("unsupported VernonDSL runtime architecture")
    if api_version is not None and (
        arch not in {opengl, opengles}
        or len(api_version) != 2
        or any(not isinstance(value, int) or value < 0 for value in api_version)
    ):
        raise ValueError("api_version is a (major, minor) pair for OpenGL runtimes")
    _release_runtime()
    if _native is None:
        raise RuntimeUnavailableError(
            f"{arch.name} requires vernon_dsl._native; build the Release native "
            "targets or install a wheel containing the native module"
        )
    backend_name = {
        cpu: "CPU",
        cuda: "CUDA",
        vulkan: "VULKAN",
        directx: "DIRECTX12",
        metal: "METAL",
        opengl: "OPENGL",
        opengles: "OPENGL_ES",
    }[arch]
    backend = getattr(_native.RuntimeBackend, backend_name)
    use_rhi_host = arch != cpu and hasattr(_native, "RhiHost") and hasattr(_native, "RhiBackend")
    if arch != cpu and not use_rhi_host:
        raise RuntimeUnavailableError(f"{arch.name} requires Vernon RHI support in vernon_dsl._native")
    rhi_backend = getattr(_native.RhiBackend, backend_name) if use_rhi_host else None
    if arch in {opengl, opengles}:
        external = _external_opengl_contexts.get(arch)
        default_version = (4, 3) if arch == opengl else (3, 1)
        if external is not None and use_rhi_host:
            user_data, make_current, get_proc_address, registered_version = external
            requested = api_version or registered_version
            _rhi_host = _native.RhiHost.create_external_opengl(
                rhi_backend,
                user_data,
                make_current,
                get_proc_address,
                *requested,
            )
            _native_runtime = _rhi_host.create_runtime()
        elif external is None and use_rhi_host:
            if _gl_context is None:
                raise RuntimeUnavailableError(
                    f"{arch.name} requires vernon_dsl._gl_context or a registered external context"
                )
            requested = api_version or default_version
            try:
                owned_context = _gl_context.Context(arch.name, *requested)
            except RuntimeError as error:
                raise RuntimeUnavailableError(
                    f"{arch.name} {requested[0]}.{requested[1]} context is unavailable"
                ) from error
            _rhi_host = _native.RhiHost.create_external_opengl(
                rhi_backend,
                owned_context.user_data,
                owned_context.make_current,
                owned_context.get_proc_address,
                *requested,
            )
            _native_runtime = _rhi_host.create_runtime()
            _owned_opengl_context = owned_context
    elif arch == cpu:
        _native_runtime = _native.Runtime(backend)
    elif not _native.runtime_available(backend):
        raise RuntimeUnavailableError(f"{arch.name} loader or a usable device is unavailable")
    else:
        _rhi_host = _native.RhiHost(rhi_backend)
        _native_runtime = _rhi_host.create_runtime()
    _architecture = arch
    _api_version = api_version or ((4, 3) if arch == opengl else (3, 1) if arch == opengles else None)
    _runtime_generation += 1


def register_external_opengl_context(
    *,
    arch: _Architecture,
    user_data: int,
    make_current: int,
    get_proc_address: int,
    api_version: tuple[int, int],
) -> None:
    if arch not in {opengl, opengles}:
        raise ValueError("external contexts are only valid for OpenGL backends")
    if (
        not all(isinstance(value, int) and value >= 0 for value in (user_data, make_current, get_proc_address))
        or not make_current
        or not get_proc_address
    ):
        raise ValueError("external context callbacks must be non-zero addresses")
    if len(api_version) != 2 or any(not isinstance(value, int) or value < 0 for value in api_version):
        raise ValueError("api_version must be a non-negative major/minor pair")
    _external_opengl_contexts[arch] = (user_data, make_current, get_proc_address, api_version)


def _interactive_glsl_version() -> int:
    if _architecture not in {opengl, opengles}:
        return 0
    major, minor = _api_version or ((4, 3) if _architecture == opengl else (3, 1))
    return major * 100 + minor * 10


__all__ = [
    "Kernel",
    "Pipeline",
    "PrimitiveTopology",
    "TensorLayout",
    "TensorStorage",
    "TensorView",
    "Texture",
    "cpu",
    "cuda",
    "directx",
    "init",
    "lines",
    "metal",
    "opengl",
    "opengles",
    "pipeline",
    "points",
    "register_external_opengl_context",
    "triangles",
    "vulkan",
]
