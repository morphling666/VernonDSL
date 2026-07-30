#ifndef VERNON_RUNTIME_VERTEX_ATTRIBUTE_CAPABILITIES_H
#define VERNON_RUNTIME_VERTEX_ATTRIBUTE_CAPABILITIES_H

#include "VernonRuntimeProvider.h"

#include <cstdint>
#include <string>

namespace vernon::runtime {

enum class VertexAttributeBackend { OpenGL, Vulkan, DirectX12 };

bool validateVertexAttributeCapability(VertexAttributeBackend backend,
                                       const VernonRuntimeProviderVertexAttribute &attribute, uint32_t maximumLocations,
                                       bool supportsFloat64, std::string &diagnostic);

} // namespace vernon::runtime

#endif
