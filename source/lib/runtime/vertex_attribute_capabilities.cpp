#include "vertex_attribute_capabilities.h"

#include <string_view>

namespace vernon::runtime {
namespace {

std::string_view backendName(VertexAttributeBackend backend) {
    switch (backend) {
    case VertexAttributeBackend::OpenGL:
        return "OpenGL";
    case VertexAttributeBackend::Vulkan:
        return "Vulkan";
    case VertexAttributeBackend::DirectX12:
        return "D3D12";
    }
    return "unknown";
}

std::string_view dtypeName(uint32_t dtype) {
    switch (dtype) {
    case VERNON_RUNTIME_PROVIDER_I32:
        return "i32";
    case VERNON_RUNTIME_PROVIDER_U32:
        return "u32";
    case VERNON_RUNTIME_PROVIDER_F16:
        return "f16";
    case VERNON_RUNTIME_PROVIDER_F32:
        return "f32";
    case VERNON_RUNTIME_PROVIDER_F64:
        return "f64";
    case 0:
        return "bool";
    case 6:
        return "u8";
    }
    return "unknown";
}

} // namespace

bool validateVertexAttributeCapability(VertexAttributeBackend backend,
                                       const VernonRuntimeProviderVertexAttribute &attribute, uint32_t maximumLocations,
                                       bool supportsFloat64, std::string &diagnostic) {
    std::string reason;
    if (!attribute.component_count || attribute.component_count > 4)
        reason = "component count must be between 1 and 4";
    else if (attribute.location >= maximumLocations)
        reason = "location exceeds the device limit";
    else {
        const bool baseNumeric =
            attribute.dtype == VERNON_RUNTIME_PROVIDER_I32 || attribute.dtype == VERNON_RUNTIME_PROVIDER_U32 ||
            attribute.dtype == VERNON_RUNTIME_PROVIDER_F16 || attribute.dtype == VERNON_RUNTIME_PROVIDER_F32;
        if (!baseNumeric && attribute.dtype != VERNON_RUNTIME_PROVIDER_F64)
            reason = "dtype is not a numeric vertex format";
        else if (attribute.dtype == VERNON_RUNTIME_PROVIDER_F64 &&
                 (!supportsFloat64 || backend == VertexAttributeBackend::DirectX12 ||
                  (backend == VertexAttributeBackend::Vulkan && attribute.component_count > 2)))
            reason = "f64 format is unsupported";
        else if (backend == VertexAttributeBackend::DirectX12 && attribute.dtype == VERNON_RUNTIME_PROVIDER_F16 &&
                 attribute.component_count == 3)
            reason = "three-component f16 format is unsupported";
    }
    if (reason.empty()) {
        diagnostic.clear();
        return true;
    }
    diagnostic = std::string(backendName(backend)) + " vertex attribute location " +
                 std::to_string(attribute.location) + " dtype " + std::string(dtypeName(attribute.dtype)) + "x" +
                 std::to_string(attribute.component_count) +
                 " exceeds device location or format capabilities: " + reason;
    return false;
}

} // namespace vernon::runtime
