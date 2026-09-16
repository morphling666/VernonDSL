#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_METAL_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_metal.h"
#include "pipeline_metadata.h"
#include "prepared_graphics_draw.h"
#include "rhi/rhi_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_METAL_RUNTIME)
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

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
            if (!buildPreparedGraphicsBindingPlan(plan, bindingOptions, state->rhiGraphicsBindingPlan,
                                                  invocationDiagnostic(*inputs.context)))
                return false;
            state->rhiGraphicsValues.resize(state->rhiGraphicsBindingPlan.size());
            const VernonRuntimeProviderShaderDescriptor shaders[2]{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                    VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                    {"msl", 3},
                                                                    vertex.source.data(),
                                                                    vertex.source.size(),
                                                                    {vertex.entry.data(), vertex.entry.size()},
                                                                    {},
                                                                    {0, 0, 0, 0}},
                                                                   {sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                    VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                                                                    {"msl", 3},
                                                                    fragment.source.data(),
                                                                    fragment.source.size(),
                                                                    {fragment.entry.data(), fragment.entry.size()},
                                                                    {},
                                                                    {0, 0, 0, 0}}};
            VernonRuntimeCorePipelineDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
            descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
            descriptor.shaders = shaders;
            descriptor.shader_count = 2;
            descriptor.bindings = state->rhiGraphicsBindingPlan.layouts.data();
            descriptor.binding_count = state->rhiGraphicsBindingPlan.size();
            descriptor.vertex_attributes = state->rhiGraphicsBindingPlan.vertexAttributes.data();
            descriptor.vertex_attribute_count = state->rhiGraphicsBindingPlan.vertexAttributes.size();
            descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
            descriptor.sample_count = 1;
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(metalState(*inputs.context).adapter), &descriptor,
                &state->rhiGraphicsPipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(metalState(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare Metal provider graphics pipeline";
                return false;
            }
            installRuntimeBackendState(pipeline, state.release());
            return true;
        }
        const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
        ReflectedEntry reflection;
        if (!resolveStageReflection(stage, VERNON_RUNTIME_METAL, reflection, invocationDiagnostic(*inputs.context)))
            return false;
        MetalArgumentBufferUsage argumentBufferUsage;
        if (!collectMetalArgumentBufferUsage(stage.nativeSlots, stage.entry, "compute", argumentBufferUsage,
                                             invocationDiagnostic(*inputs.context)) ||
            !validateMetalArgumentBufferUsage(argumentBufferUsage, metalState(*inputs.context).argumentBuffersTier,
                                              metalState(*inputs.context).argumentBufferEncodingSupported,
                                              invocationDiagnostic(*inputs.context)))
            return false;

        auto state = std::make_unique<MetalPipelineState>();
        if (!buildPreparedComputeBindingPlan(plan, reflection, VERNON_RUNTIME_METAL, state->rhiComputeBindingPlan,
                                             invocationDiagnostic(*inputs.context)))
            return false;
        for (size_t index = 0; index < state->rhiComputeBindingPlan.size(); ++index) {
            auto &layout = state->rhiComputeBindingPlan.layouts[index];
            const PreparedBindingSource &source = state->rhiComputeBindingPlan.sources[index];
            const char *resourceKind = source.metadataCarrier                                 ? "uniform_buffer"
                                       : layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ? "sampled_image"
                                       : layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ? "storage_image"
                                                                                              : "storage_buffer";
            MetalResourceLocation location;
            if (!resolveMetalResourceLocation(stage.nativeSlots, stage.entry, "compute", resourceKind, layout.set,
                                              layout.binding, location, invocationDiagnostic(*inputs.context)))
                return false;
            layout.set = location.argumentBufferIndex;
            layout.binding = location.memberId;
            layout.array_count = location.count;
        }
        state->rhiComputeValues.resize(state->rhiComputeBindingPlan.size());
        std::copy_n(stage.workgroup, 3, state->rhiComputeWorkgroup);
        const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                           VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                           {"msl", 3},
                                                           stage.source.data(),
                                                           stage.source.size(),
                                                           {stage.entry.data(), stage.entry.size()},
                                                           {},
                                                           {0, 0, 0, 0}};
        VernonRuntimeCorePipelineDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
        descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
        descriptor.shaders = &shader;
        descriptor.shader_count = 1;
        descriptor.bindings = state->rhiComputeBindingPlan.layouts.data();
        descriptor.binding_count = state->rhiComputeBindingPlan.size();
        std::copy_n(state->rhiComputeWorkgroup, 3, descriptor.workgroup_size);
        const VernonStatus status =
            vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(metalState(*inputs.context).adapter),
                                             &descriptor, &state->rhiComputePipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(metalState(*inputs.context).adapter);
            invocationDiagnostic(*inputs.context) = providerError.data
                                                        ? std::string(providerError.data, providerError.size)
                                                        : "failed to prepare Metal provider compute pipeline";
            return false;
        }
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

void destroyMetalPipeline(VernonStageExecutable &pipeline) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    MetalPipelineState &state = runtimeBackendState<MetalPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiComputeBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiComputePipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiGraphicsBindings);
    destroyGraphicsVariant(state.rhiGraphicsVariant);
    vernonRuntimeCorePipelineDestroy(state.rhiGraphicsPipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeMetalGraphicsPipeline(VernonStageExecutable &pipeline,
                                         const VernonStageInvocationDescriptor &invocation,
                                         const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    MetalPipelineState &state = runtimeBackendState<MetalPipelineState>(pipeline);
    if (!state.rhiGraphicsPipeline)
        return VERNON_STATUS_OK;
    VernonRuntimeRhiAdapter &adapter = *metalState(*pipeline.context).adapter;
    if (!fillPreparedGraphicsBindingValues(state.rhiGraphicsBindingPlan, plan, state.rhiGraphicsValues,
                                           invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonStatus status =
        state.rhiGraphicsBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiGraphicsBindings, state.rhiGraphicsValues.data(),
                                              state.rhiGraphicsValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiGraphicsPipeline, state.rhiGraphicsValues.data(),
                                              state.rhiGraphicsValues.size(), &state.rhiGraphicsBindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data && providerError.size ? std::string(providerError.data, providerError.size)
                                                             : "failed to update Metal RHI graphics bindings",
                    status);
    }
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachments.size());
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const uint32_t format = rhi::metalTexturePixelFormat(plan.attachmentFormats[index]);
        if (!format)
            return fail(*pipeline.context, "Metal RHI color attachment is stale");
        formats.push_back(format);
    }
    uint32_t depthFormat = 0;
    if (plan.depthAttachment) {
        depthFormat = rhi::metalTexturePixelFormat(plan.depthFormat);
        if (!depthFormat)
            return fail(*pipeline.context, "Metal RHI depth attachment is stale");
    }
    PreparedGraphicsDraw prepared;
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, state.rhiGraphicsBindingPlan.layouts,
                             state.rhiGraphicsValues, prepared, invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    status = ensureGraphicsVariant(state.rhiGraphicsPipeline, prepared.variantKey, state.rhiGraphicsVariant);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare Metal RHI graphics variant",
                    status);
    }
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(state.rhiGraphicsVariant.handle,
                                                                  state.rhiGraphicsBindings, &prepared.invocation);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data && providerError.size ? std::string(providerError.data, providerError.size)
                                                             : "failed to encode Metal RHI draw",
                    status);
    }
    return VERNON_STATUS_OK;
#else
    (void)pipeline;
    (void)invocation;
    (void)plan;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus invokeMetalComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_METAL_RUNTIME)
    MetalPipelineState &state = runtimeBackendState<MetalPipelineState>(pipeline);
    VernonRuntimeRhiAdapter &adapter = *metalState(*pipeline.context).adapter;
    for (size_t index = 0; index < state.rhiComputeBindingPlan.size(); ++index) {
        const auto &layout = state.rhiComputeBindingPlan.layouts[index];
        const PreparedBindingSource &preparedSource = state.rhiComputeBindingPlan.sources[index];
        auto &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (preparedSource.metadataCarrier) {
            if (!bindComputeMetadataCarrier(layout, preparedSource, launch, value))
                return fail(*pipeline.context, "Metal metadata carrier payload does not match its compiled ABI");
            continue;
        }
        if (preparedSource.argumentIndex >= launch.arguments.size())
            return fail(*pipeline.context, "Metal prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[preparedSource.argumentIndex];
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
                if (!bindComputeValueStorage(layout, argument, value))
                    return fail(*pipeline.context, "Metal prepared Value storage binding has invalid bytes");
                continue;
            }
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "Metal prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
            if (value.payload.buffer.resource.offset > value.payload.buffer.resource.size ||
                preparedSource.resourceOffset >
                    value.payload.buffer.resource.size - value.payload.buffer.resource.offset)
                return fail(*pipeline.context, "Metal aggregate storage leaf exceeds its Tensor resource");
            value.payload.buffer.resource.offset += preparedSource.resourceOffset;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                   layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto *image = std::get_if<ComputeImageArgument>(&argument);
            if (!image || !image->view.resource.value)
                return fail(*pipeline.context, "Metal prepared image binding requires an RHI Texture");
            value.payload.image.view = image->view;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar || !scalar->data || !scalar->size)
                return fail(*pipeline.context, "Metal prepared inline binding requires packed host data");
            value.payload.inline_value.data = scalar->data;
            value.payload.inline_value.size = scalar->size;
        }
    }
    VernonStatus status =
        state.rhiComputeBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiComputeBindings, state.rhiComputeValues.data(),
                                              state.rhiComputeValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiComputePipeline, state.rhiComputeValues.data(),
                                              state.rhiComputeValues.size(), &state.rhiComputeBindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare Metal invocation bindings",
                    status);
    }
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiComputePipeline, state.rhiComputeBindings, launch.commandEncoder,
                                             groups, nullptr, 0);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = vernonRuntimeRhiAdapterGetLastError(&adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode Metal provider dispatch",
                    status);
    }
    return VERNON_STATUS_OK;
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
