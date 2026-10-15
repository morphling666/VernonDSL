#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_METAL_RUNTIME)
#include "backend_metal.h"
#include "rhi/rhi_internal.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_METAL_RUNTIME)
namespace {

std::optional<uint32_t> metalAttachmentFormat(VernonTextureFormat format) {
    const uint32_t pixelFormat = rhi::metalTexturePixelFormat(format);
    return pixelFormat ? std::optional<uint32_t>(pixelFormat) : std::nullopt;
}

constexpr RhiComputeTarget kMetalCompute{"Metal", "msl"};
constexpr RhiGraphicsTarget kMetalGraphics{"Metal", "msl", metalAttachmentFormat, metalAttachmentFormat};

struct MetalResourceLocation {
    uint32_t argumentBufferIndex{UINT32_MAX};
    uint32_t memberId{UINT32_MAX};
    uint32_t directBufferIndex{UINT32_MAX};
    uint32_t count{};
};

bool resolveMetalResourceLocation(const std::vector<NativeResourceSlot> &slots, const std::string &entry,
                                  const char *stage, const char *kind, uint32_t set, uint32_t binding,
                                  MetalResourceLocation &output, std::string &error,
                                  const std::string *name = nullptr) {
    bool found = false;
    for (const NativeResourceSlot &slot : slots) {
        if (slot.entry != entry || slot.stage != stage || slot.kind != kind || (name && slot.name != *name) ||
            (set != UINT32_MAX && slot.set != set) || (binding != UINT32_MAX && slot.binding != binding))
            continue;
        MetalResourceLocation location;
        location.argumentBufferIndex = slot.argumentBufferIndex;
        location.memberId = slot.memberId;
        location.directBufferIndex = slot.directBufferIndex;
        location.count = slot.count;
        const bool descriptor = set != UINT32_MAX;
        const bool validDescriptor = location.argumentBufferIndex < 8 && location.memberId != UINT32_MAX &&
                                     location.directBufferIndex == UINT32_MAX;
        const bool validDirect = location.argumentBufferIndex == UINT32_MAX && location.memberId == UINT32_MAX &&
                                 location.directBufferIndex < 31;
        if (found || !location.count || (descriptor ? !validDescriptor : !validDirect)) {
            error = "Metal reflection contains an ambiguous or invalid argument-buffer resource";
            return false;
        }
        output = location;
        found = true;
    }
    if (!found)
        error = "Metal reflection has no compiled " + std::string(kind) + " location for " + stage + " binding (" +
                std::to_string(set) + ", " + std::to_string(binding) + ")";
    return found;
}

bool resolveMetalComputeNativeBinding(const void *userData, const PreparedBindingSource &source,
                                      VernonRuntimeProviderBindingLayoutEntry &layout, std::string &error) {
    const auto &stage = *static_cast<const LoadedStageArtifact *>(userData);
    const char *resourceKind = source.metadataCarrier                                 ? "uniform_buffer"
                               : layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ? "sampled_image"
                               : layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ? "storage_image"
                               : layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER       ? "sampler"
                                                                                      : "storage_buffer";
    MetalResourceLocation location;
    if (!resolveMetalResourceLocation(stage.nativeSlots, stage.entry, "compute", resourceKind, layout.set,
                                      layout.binding, location, error))
        return false;
    layout.set = location.argumentBufferIndex;
    layout.binding = location.memberId;
    layout.array_count = location.count;
    return true;
}

struct MetalGraphicsNativeBindingContext {
    const LoadedStageArtifact &vertex;
    const LoadedStageArtifact &fragment;
};

bool resolveMetalGraphicsNativeBinding(void *userData, std::string_view stage, std::string_view kind,
                                       uint32_t descriptorSet, uint32_t descriptorBinding, const std::string *name,
                                       VernonRuntimeProviderBindingLayoutEntry &layout, std::string &error) {
    const auto &context = *static_cast<const MetalGraphicsNativeBindingContext *>(userData);
    const LoadedStageArtifact *artifact = stage == "vertex"     ? &context.vertex
                                          : stage == "fragment" ? &context.fragment
                                                                : nullptr;
    if (!artifact) {
        error = "Metal native resource resolver received an unsupported graphics stage";
        return false;
    }
    MetalResourceLocation location;
    if (!resolveMetalResourceLocation(artifact->nativeSlots, artifact->entry, std::string(stage).c_str(),
                                      std::string(kind).c_str(), descriptorSet, descriptorBinding, location, error,
                                      name))
        return false;
    layout.set = descriptorSet == UINT32_MAX ? UINT32_MAX : location.argumentBufferIndex;
    layout.binding = descriptorSet == UINT32_MAX ? location.directBufferIndex : location.memberId;
    layout.array_count = descriptorSet == UINT32_MAX ? 1 : location.count;
    return true;
}

struct MetalArgumentBufferUsage {
    uint64_t buffers{};
    uint64_t textures{};
    uint64_t samplers{};
    bool writableTexture{};
};

bool collectMetalArgumentBufferUsage(const std::vector<NativeResourceSlot> &slots, const std::string &entry,
                                     const char *stage, MetalArgumentBufferUsage &usage, std::string &error) {
    for (const NativeResourceSlot &slot : slots) {
        if (slot.entry != entry || slot.stage != stage)
            continue;
        if (!slot.count) {
            error = "Metal reflection contains a zero-sized resource binding";
            return false;
        }
        if (slot.kind == "uniform_buffer" || slot.kind == "storage_buffer")
            usage.buffers += slot.count;
        else if (slot.kind == "sampled_image" || slot.kind == "storage_image") {
            usage.textures += slot.count;
            usage.writableTexture |= slot.kind == "storage_image";
        } else if (slot.kind == "sampler")
            usage.samplers += slot.count;
    }
    return true;
}

bool validateMetalArgumentBufferUsage(const MetalArgumentBufferUsage &usage, uint32_t deviceTier,
                                      bool encodingSupported, std::string &error) {
    if ((usage.buffers || usage.textures || usage.samplers) && !encodingSupported) {
        error = "Metal argument-buffer encoding is unavailable on this device";
        return false;
    }
    if (deviceTier < 1 && (usage.buffers > 31 || usage.textures > 31 || usage.samplers > 16 || usage.writableTexture)) {
        error = "Metal pipeline requires Argument Buffers Tier 2 (resources: " + std::to_string(usage.buffers) +
                " buffers, " + std::to_string(usage.textures) + " textures, " + std::to_string(usage.samplers) +
                " samplers" + (usage.writableTexture ? ", writable texture" : "") + ")";
        return false;
    }
    return true;
}

} // namespace
#endif

bool validateMetalArgumentBufferLimitsForTesting(uint64_t buffers, uint64_t textures, uint64_t samplers,
                                                 bool writableTexture, uint32_t deviceTier) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    const MetalArgumentBufferUsage usage{buffers, textures, samplers, writableTexture};
    std::string error;
    return validateMetalArgumentBufferUsage(usage, deviceTier, true, error);
#else
    (void)buffers;
    (void)textures;
    (void)samplers;
    (void)writableTexture;
    (void)deviceTier;
    return false;
#endif
}

BackendPipelineResult resolveMetalPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                           VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_METAL_RUNTIME)
        if (plan.compute.empty()) {
            const LoadedStageArtifact &vertex = inputs.artifacts.at(plan.vertex);
            const LoadedStageArtifact &fragment = inputs.artifacts.at(plan.fragment);
            MetalArgumentBufferUsage argumentBufferUsage;
            if (!collectMetalArgumentBufferUsage(vertex.nativeSlots, vertex.entry, "vertex", argumentBufferUsage,
                                                 invocationDiagnostic(*inputs.context)) ||
                !collectMetalArgumentBufferUsage(fragment.nativeSlots, fragment.entry, "fragment", argumentBufferUsage,
                                                 invocationDiagnostic(*inputs.context)) ||
                !validateMetalArgumentBufferUsage(argumentBufferUsage, metalState(*inputs.context).argumentBuffersTier,
                                                  metalState(*inputs.context).argumentBufferEncodingSupported,
                                                  invocationDiagnostic(*inputs.context)))
                return false;
            MetalGraphicsNativeBindingContext nativeBindingContext{vertex, fragment};
            auto state = std::make_unique<MetalPipelineState>();
            const PreparedGraphicsBindingOptions bindingOptions{VERNON_RUNTIME_METAL, resolveMetalGraphicsNativeBinding,
                                                                &nativeBindingContext};
            if (!prepareRhiGraphicsPipeline(*inputs.context, metalState(*inputs.context).adapter, kMetalGraphics,
                                            vertex, fragment, plan, bindingOptions, state->rhiGraphics))
                return false;
            installRuntimeBackendState(pipeline, state.release());
            return true;
        }
        const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
        MetalArgumentBufferUsage argumentBufferUsage;
        if (!collectMetalArgumentBufferUsage(stage.nativeSlots, stage.entry, "compute", argumentBufferUsage,
                                             invocationDiagnostic(*inputs.context)) ||
            !validateMetalArgumentBufferUsage(argumentBufferUsage, metalState(*inputs.context).argumentBuffersTier,
                                              metalState(*inputs.context).argumentBufferEncodingSupported,
                                              invocationDiagnostic(*inputs.context)))
            return false;
        auto state = std::make_unique<MetalPipelineState>();
        const RhiComputeNativeBindingResolver resolver{resolveMetalComputeNativeBinding, &stage};
        if (!prepareRhiComputePipeline(*inputs.context, metalState(*inputs.context).adapter, kMetalCompute, stage, plan,
                                       state->rhiCompute, resolver))
            return false;
        installRuntimeBackendState(pipeline, state.release());
        return true;
#else
        (void)inputs;
        (void)plan;
        (void)pipeline;
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

VernonStatus invokeMetalGraphicsPipeline(VernonStageExecutable &pipeline,
                                         const VernonStageInvocationDescriptor &invocation,
                                         const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    return invokeRhiGraphicsPipeline(*pipeline.context, metalState(*pipeline.context).adapter, kMetalGraphics,
                                     invocation, plan, runtimeBackendState<MetalPipelineState>(pipeline).rhiGraphics);
#else
    (void)pipeline;
    (void)invocation;
    (void)plan;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus invokeMetalComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    return invokeRhiComputePipeline(*pipeline.context, metalState(*pipeline.context).adapter, kMetalCompute,
                                    pipeline.bindingProjection, launch,
                                    runtimeBackendState<MetalPipelineState>(pipeline).rhiCompute);
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
