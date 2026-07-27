#include "runtime/vertex_attribute_capabilities.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace {

using vernon::runtime::validateVertexAttributeCapability;
using vernon::runtime::VertexAttributeBackend;

VernonRuntimeProviderVertexAttribute attribute(uint32_t location, uint32_t dtype, uint32_t components) {
    return {0, location, dtype, components, 0};
}

TEST(VertexAttributeCapabilities, CoversAggregateLeafDtypeMatrix) {
    struct Case {
        VertexAttributeBackend backend;
        uint32_t dtype;
        uint32_t components;
        bool supportsFloat64;
        bool accepted;
    };
    constexpr std::array cases{
        Case{VertexAttributeBackend::OpenGL, VERNON_RUNTIME_PROVIDER_I32, 4, true, true},
        Case{VertexAttributeBackend::OpenGL, VERNON_RUNTIME_PROVIDER_U32, 3, true, true},
        Case{VertexAttributeBackend::OpenGL, VERNON_RUNTIME_PROVIDER_F16, 4, true, true},
        Case{VertexAttributeBackend::OpenGL, VERNON_RUNTIME_PROVIDER_F32, 4, true, true},
        Case{VertexAttributeBackend::OpenGL, VERNON_RUNTIME_PROVIDER_F64, 2, true, true},
        Case{VertexAttributeBackend::OpenGL, VERNON_RUNTIME_PROVIDER_F64, 2, false, false},
        Case{VertexAttributeBackend::OpenGL, 0, 1, true, false},
        Case{VertexAttributeBackend::OpenGL, 6, 1, true, false},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_I32, 4, true, true},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_U32, 4, true, true},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_F16, 3, true, true},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_F32, 4, true, true},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_F64, 1, true, true},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_F64, 2, true, true},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_F64, 3, true, false},
        Case{VertexAttributeBackend::Vulkan, VERNON_RUNTIME_PROVIDER_F64, 4, true, false},
        Case{VertexAttributeBackend::Vulkan, 0, 1, true, false},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_I32, 3, false, true},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_U32, 4, false, true},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_F16, 1, false, true},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_F16, 2, false, true},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_F16, 3, false, false},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_F16, 4, false, true},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_F32, 4, false, true},
        Case{VertexAttributeBackend::DirectX12, VERNON_RUNTIME_PROVIDER_F64, 1, false, false},
        Case{VertexAttributeBackend::DirectX12, 0, 1, false, false},
    };

    for (const Case &test : cases) {
        std::string diagnostic;
        EXPECT_EQ(validateVertexAttributeCapability(test.backend, attribute(3, test.dtype, test.components), 16,
                                                    test.supportsFloat64, diagnostic),
                  test.accepted)
            << "dtype=" << test.dtype << " components=" << test.components;
        EXPECT_EQ(diagnostic.empty(), test.accepted);
    }
}

TEST(VertexAttributeCapabilities, DiagnosesExpandedAggregateLocationOverflow) {
    constexpr std::array aggregateLeaves{
        VernonRuntimeProviderVertexAttribute{0, 13, VERNON_RUNTIME_PROVIDER_F32, 4, 0},
        VernonRuntimeProviderVertexAttribute{0, 14, VERNON_RUNTIME_PROVIDER_I32, 1, 16},
        VernonRuntimeProviderVertexAttribute{0, 15, VERNON_RUNTIME_PROVIDER_F64, 2, 24},
        VernonRuntimeProviderVertexAttribute{0, 16, VERNON_RUNTIME_PROVIDER_F32, 2, 40},
    };
    for (VertexAttributeBackend backend :
         {VertexAttributeBackend::OpenGL, VertexAttributeBackend::Vulkan, VertexAttributeBackend::DirectX12}) {
        for (size_t index = 0; index < aggregateLeaves.size(); ++index) {
            std::string diagnostic;
            const bool supportsFloat64 = backend != VertexAttributeBackend::DirectX12;
            const bool accepted =
                validateVertexAttributeCapability(backend, aggregateLeaves[index], 16, supportsFloat64, diagnostic);
            if (index < 3 && !(backend == VertexAttributeBackend::DirectX12 && index == 2))
                EXPECT_TRUE(accepted) << diagnostic;
            else {
                EXPECT_FALSE(accepted);
                EXPECT_NE(diagnostic.find(index == 3 ? "location 16" : "f64"), std::string::npos);
                EXPECT_NE(diagnostic.find("location or format capabilities"), std::string::npos);
            }
        }
    }
}

TEST(VertexAttributeCapabilities, RejectsInvalidExpandedComponentCounts) {
    for (uint32_t components : {0u, 5u}) {
        std::string diagnostic;
        EXPECT_FALSE(validateVertexAttributeCapability(VertexAttributeBackend::Vulkan,
                                                       attribute(0, VERNON_RUNTIME_PROVIDER_F32, components), 16, true,
                                                       diagnostic));
        EXPECT_NE(diagnostic.find("component count"), std::string::npos);
    }
}

} // namespace
