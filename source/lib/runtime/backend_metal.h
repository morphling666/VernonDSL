#ifndef VERNON_RUNTIME_BACKEND_METAL_H
#define VERNON_RUNTIME_BACKEND_METAL_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "metal_runtime_capabilities.h"
#include "rhi_provider_pipeline.h"
#include "runtime_state.h"

#include <cstdint>

struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

struct MetalContextState {
    VernonRuntimeRhiAdapter *adapter{};
    uint32_t maxComputeInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    RuntimeVersion operatingSystemVersion;
    uint32_t argumentBuffersTier{};
    bool argumentBufferEncodingSupported{};
};

inline MetalContextState &metalState(VernonRuntimeContext &context) {
    return runtimeBackendState<MetalContextState>(context);
}

inline const MetalContextState &metalState(const VernonRuntimeContext &context) {
    return runtimeBackendState<MetalContextState>(context);
}

struct MetalPipelineState {
    RhiComputePipeline rhiCompute;
    RhiGraphicsPipeline rhiGraphics;
};

} // namespace vernon::runtime

#endif
