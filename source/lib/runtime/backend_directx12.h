#ifndef VERNON_RUNTIME_BACKEND_DIRECTX12_H
#define VERNON_RUNTIME_BACKEND_DIRECTX12_H

#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "../rhi/directx12_backend.h"
#include <d3d12.h>
#endif

#include <string>
#include <vector>

struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
struct DirectX12ContextState {
    VernonRuntimeRhiAdapter *adapter{};
    D3D_FEATURE_LEVEL featureLevel{};
    D3D_SHADER_MODEL shaderModel{};
    D3D_ROOT_SIGNATURE_VERSION rootSignatureVersion{};
    D3D12_RESOURCE_BINDING_TIER resourceBindingTier{};
    uint32_t maxComputeInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
};
struct DirectX12PipelineState {
    struct GraphicsBinding {
        enum Source {
            EXTERNAL_UNIFORM,
            EXTERNAL_VERTEX,
            EXTERNAL_TEXTURE,
            EXTERNAL_SAMPLER,
            EXTERNAL_STORAGE,
            IMPLICIT_SAMPLER,
            RESOLUTION
        };
        Source source{};
        uint32_t externalSlot{};
        TensorPackingLayout packing;
        std::vector<uint8_t> storage;
    };

    VernonRuntimeCorePipeline *rhiComputePipeline{};
    VernonRuntimeCoreBindings *rhiComputeBindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiComputeLayout;
    std::vector<VernonRuntimeProviderBindingValue> rhiComputeValues;
    std::vector<uint64_t> rhiComputeResourceOffsets;
    std::vector<ComputeBindingSource> rhiComputeBindingSources;
    std::vector<int64_t> rhiComputeDescriptorValues;
    uint32_t rhiComputeWorkgroup[3]{1, 1, 1};
    VernonRuntimeCorePipeline *rhiGraphicsPipeline{};
    VernonRuntimeCoreBindings *rhiGraphicsBindings{};
    PreparedGraphicsVariant rhiGraphicsVariant;
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiGraphicsLayout;
    std::vector<VernonRuntimeProviderVertexAttribute> rhiGraphicsVertexAttributes;
    std::vector<VernonRuntimeProviderBindingValue> rhiGraphicsValues;
    std::vector<GraphicsBinding> rhiGraphicsBindingsPlan;
};

inline DirectX12ContextState &directX12State(VernonRuntimeContext &context) {
    return runtimeBackendState<DirectX12ContextState>(context);
}
inline const DirectX12ContextState &directX12State(const VernonRuntimeContext &context) {
    return runtimeBackendState<DirectX12ContextState>(context);
}
#endif

} // namespace vernon::runtime

#endif
