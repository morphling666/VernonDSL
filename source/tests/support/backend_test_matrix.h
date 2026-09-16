#ifndef VERNON_TESTS_SUPPORT_BACKEND_TEST_MATRIX_H
#define VERNON_TESTS_SUPPORT_BACKEND_TEST_MATRIX_H

#include "VernonCompiler.h"
#include "VernonProgramCapabilities.h"
#include "VernonRHI.h"
#include "VernonRuntime.h"

#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::tests {

struct BackendTestRow {
    std::string_view name;
    VernonTarget compiler;
    VernonRuntimeBackend runtime;
    std::optional<VernonRhiBackend> rhi;
};

inline void PrintTo(const BackendTestRow &backend, std::ostream *stream) { *stream << backend.name; }

inline constexpr std::array<BackendTestRow, 7> backendTestMatrix{{
    {"CPU", VERNON_TARGET_CPU, VERNON_RUNTIME_CPU, std::nullopt},
    {"CUDA", VERNON_TARGET_CUDA, VERNON_RUNTIME_CUDA, VERNON_RHI_BACKEND_CUDA},
    {"Vulkan", VERNON_TARGET_VULKAN, VERNON_RUNTIME_VULKAN, VERNON_RHI_BACKEND_VULKAN},
    {"DirectX12", VERNON_TARGET_DIRECTX, VERNON_RUNTIME_DIRECTX12, VERNON_RHI_BACKEND_DIRECTX12},
    {"Metal", VERNON_TARGET_METAL, VERNON_RUNTIME_METAL, VERNON_RHI_BACKEND_METAL},
    {"OpenGL", VERNON_TARGET_OPENGL, VERNON_RUNTIME_OPENGL, VERNON_RHI_BACKEND_OPENGL},
    {"OpenGLES", VERNON_TARGET_OPENGL_ES, VERNON_RUNTIME_OPENGL_ES, VERNON_RHI_BACKEND_OPENGL_ES},
}};

inline std::vector<BackendTestRow> rhiBackendCases() {
    std::vector<BackendTestRow> result;
    for (const BackendTestRow &backend : backendTestMatrix)
        if (backend.rhi)
            result.push_back(backend);
    return result;
}

struct BackendTestRequirements {
    bool compute{};
    bool graphics{};
    bool storageBuffers{};
    bool storageTexture{};
    bool nonR32ReadWriteStorageImages{};
    bool deviceAtomics{};
    bool f32AtomicAdd{};
    bool f64AtomicAdd{};
    bool textureSamplerOperations{};
    bool programVjp{};
    uint16_t minimumApiMajor{};
    uint16_t minimumApiMinor{};
    std::string_view nativeInteropBackend;
};

enum class BackendProbeKind {
    Available,
    PlatformNotBuilt,
    DeviceOrContextUnavailable,
    CapabilityUnsupported,
    ProbeFailure,
};

struct BackendProbeResult {
    BackendProbeKind kind{BackendProbeKind::Available};
    std::string reason;

    bool available() const { return kind == BackendProbeKind::Available; }
    bool skippable() const {
        return kind == BackendProbeKind::PlatformNotBuilt || kind == BackendProbeKind::DeviceOrContextUnavailable ||
               kind == BackendProbeKind::CapabilityUnsupported;
    }
};

inline std::string runtimeDiagnostic(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string{};
}

inline BackendProbeResult unsupported(std::string_view backend, std::string_view capability) {
    return {BackendProbeKind::CapabilityUnsupported,
            std::string(backend) + " does not support required capability '" + std::string(capability) + "'"};
}

inline BackendProbeResult probeCompilerBackend(const VernonCompilerContext *compiler, const BackendTestRow &backend,
                                               const BackendTestRequirements &requirements) {
    VernonTargetCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    capabilities.abi_version = VERNON_TARGET_CAPABILITIES_VERSION;
    if (vernonCompilerQueryTargetCapabilities(compiler, backend.compiler, &capabilities) != VERNON_STATUS_OK)
        return {BackendProbeKind::ProbeFailure, std::string(backend.name) + " compiler capability query failed"};
    if (!capabilities.available)
        return {BackendProbeKind::PlatformNotBuilt,
                std::string(backend.name) + " compiler target is not built on this platform"};
    if (requirements.compute && !capabilities.supports_compute)
        return unsupported(backend.name, "compute");
    if (requirements.graphics && !capabilities.supports_graphics)
        return unsupported(backend.name, "graphics");
    if (requirements.deviceAtomics && !capabilities.supports_device_storage_atomics)
        return unsupported(backend.name, "device_storage_atomics");
    if (requirements.f32AtomicAdd && !capabilities.supports_f32_device_atomic_add)
        return unsupported(backend.name, "f32_atomic_add");
    if (requirements.f64AtomicAdd && !capabilities.supports_f64_device_atomic_add)
        return unsupported(backend.name, "f64_atomic_add");
    if (requirements.textureSamplerOperations) {
        using namespace vernon::program_capabilities;
        if (requirements.compute && !get(Id::ComputeSamplerBinding).supported)
            return unsupported(backend.name, "compute_sampler_binding");
        if (requirements.graphics && !get(Id::GraphicsTextureSampling).supported)
            return unsupported(backend.name, "graphics_texture_sampling");
    }
    if (requirements.storageTexture && !capabilities.supports_compute_texture_binding)
        return unsupported(backend.name, "compute_texture_binding");
    if (requirements.nonR32ReadWriteStorageImages && !capabilities.supports_non_r32_read_write_storage_images)
        return unsupported(backend.name, "non_r32_read_write_storage_images");
    if (requirements.programVjp &&
        !vernon::program_capabilities::get(vernon::program_capabilities::Id::ComputeVjp).supported)
        return unsupported(backend.name, "program_vjp");
    if (!requirements.nativeInteropBackend.empty() && requirements.nativeInteropBackend != backend.name)
        return unsupported(backend.name, std::string(requirements.nativeInteropBackend) + "_native_interop");
    return {};
}

inline bool apiVersionAtLeast(const VernonRuntimeCapabilities &capabilities, uint16_t major, uint16_t minor) {
    return capabilities.api_version_major > major ||
           (capabilities.api_version_major == major && capabilities.api_version_minor >= minor);
}

inline BackendProbeResult probeRuntimeBackend(const BackendTestRow &backend,
                                              const BackendTestRequirements &requirements,
                                              const VernonRuntimeContext *context = nullptr) {
    const bool requiresContext =
        backend.runtime == VERNON_RUNTIME_OPENGL || backend.runtime == VERNON_RUNTIME_OPENGL_ES;
    const VernonRuntimeCapabilities baseline = vernonRuntimeGetCapabilities(backend.runtime);
    if (!context) {
        std::string reason = runtimeDiagnostic(baseline.diagnostic);
        if (!requiresContext && baseline.available) {
            if (reason.empty())
                reason = std::string(backend.name) + " runtime context creation failed after a successful device probe";
            return {BackendProbeKind::ProbeFailure, std::move(reason)};
        }
        if (reason.empty()) {
            reason = requiresContext ? std::string(backend.name) + " requires a host-owned current graphics context"
                                     : std::string(backend.name) + " runtime context could not be created";
        }
        return {BackendProbeKind::DeviceOrContextUnavailable, std::move(reason)};
    }

    const VernonRuntimeCapabilities capabilities = vernonRuntimeGetContextCapabilities(context);
    if (!capabilities.available) {
        std::string reason = runtimeDiagnostic(capabilities.diagnostic);
        if (reason.empty())
            reason = std::string(backend.name) + " created runtime reported unavailable context capabilities";
        return {BackendProbeKind::ProbeFailure, std::move(reason)};
    }
    if (requirements.compute && !capabilities.supports_compute)
        return unsupported(backend.name, "compute");
    if (requirements.graphics && !capabilities.supports_graphics)
        return unsupported(backend.name, "graphics");
    if (requirements.storageBuffers && !capabilities.supports_storage_buffers)
        return unsupported(backend.name, "storage_buffers");
    if ((requirements.minimumApiMajor || requirements.minimumApiMinor) &&
        !apiVersionAtLeast(capabilities, requirements.minimumApiMajor, requirements.minimumApiMinor)) {
        return {BackendProbeKind::CapabilityUnsupported, std::string(backend.name) + " context API version is below " +
                                                             std::to_string(requirements.minimumApiMajor) + "." +
                                                             std::to_string(requirements.minimumApiMinor)};
    }
    return {};
}

} // namespace vernon::tests

#endif
