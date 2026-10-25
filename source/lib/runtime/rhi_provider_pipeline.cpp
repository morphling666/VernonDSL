#include "rhi_provider_pipeline.h"

#include "VernonRuntimeRHIAdapter.h"
#include "pipeline_metadata.h"
#include "prepared_graphics_draw.h"
#include "runtime_state.h"
#include <algorithm>
#include <utility>

namespace vernon::runtime {

namespace {

std::string providerFailure(const VernonRuntimeRhiAdapter *adapter, std::string_view target, std::string_view action,
                            std::string_view object) {
    return providerDiagnostic(vernonRuntimeRhiAdapterGetLastError(adapter), "failed to " + std::string(action) + " " +
                                                                                std::string(target) + " provider " +
                                                                                std::string(object));
}

VernonStatus fail(VernonRuntimeContext &context, std::string error, VernonStatus status) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

// Stage loaders fill `source` for backends that consume text and `binary` for the rest, never both.
VernonRuntimeProviderShaderDescriptor shaderDescriptor(uint32_t stage, std::string_view artifactKind,
                                                       const LoadedStageArtifact &artifact) {
    const bool binary = !artifact.binary.empty();
    return {sizeof(VernonRuntimeProviderShaderDescriptor),
            stage,
            {artifactKind.data(), artifactKind.size()},
            binary ? static_cast<const void *>(artifact.binary.data()) : artifact.source.data(),
            binary ? artifact.binary.size() : artifact.source.size(),
            {artifact.entry.data(), artifact.entry.size()},
            {},
            {0, 0, 0, 0}};
}

struct ComputeGroupValues {
    const RhiComputeTarget &target;
    const PreparedComputeBindingPlan &plan;
    const StageBindingPlan &projection;
    const PlannedComputeLaunch &launch;
};

bool fillComputeGroupValue(const void *userData, size_t slot, VernonRuntimeProviderBindingValue &value,
                           std::vector<uint8_t> &, std::string &error) {
    const auto &values = *static_cast<const ComputeGroupValues *>(userData);
    return values.target.fillValue(values.plan, values.projection, values.launch, slot, value, error);
}

struct GraphicsGroupValues {
    const PreparedGraphicsBindingPlan &plan;
    GraphicsSlotInputs inputs;
};

bool fillGraphicsGroupValue(const void *userData, size_t slot, VernonRuntimeProviderBindingValue &value,
                            std::vector<uint8_t> &storage, std::string &error) {
    const auto &values = *static_cast<const GraphicsGroupValues *>(userData);
    return fillPreparedGraphicsBindingValue(values.plan, values.inputs, slot, value, storage, error);
}

struct GraphicsGroupsMade {
    const GraphicsGroupSink &sink;
    const PreparedGraphicsBindingPlan &plan;
};

VernonStatus forwardGraphicsGroup(void *userData, VernonRuntimeCoreGroup *group, const std::vector<uint32_t> &slots,
                                  const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    const auto &made = *static_cast<const GraphicsGroupsMade *>(userData);
    return made.sink.made(made.sink.userData, group, made.plan, slots, values, valueCount);
}

// Resolves the vertex buffers of `command` into `stream`, then chooses and binds its variant when the pipeline, its
// state or the vertex strides differ from the bound variant of `stream`.
VernonStatus bindGraphicsVariant(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                 const RhiGraphicsTarget &target, VernonRuntimeCorePackets *packets,
                                 const GraphicsCommand &command, GraphicsStream &stream,
                                 RhiGraphicsPipeline &pipeline) {
    const size_t vertexBindings = pipeline.bindingPlan.vertexInputs.size();
    stream.nextVertexStrides.resize(vertexBindings);
    stream.vertexBuffers.resize(vertexBindings);
    const GraphicsSlotInputs inputs{command.arguments, command.draw.resolution};
    for (size_t binding = 0; binding < vertexBindings; ++binding) {
        std::string error;
        if (!preparedGraphicsVertexBuffer(pipeline.bindingPlan, inputs, binding, stream.vertexBuffers[binding],
                                          stream.nextVertexStrides[binding], error))
            return fail(context, std::move(error), VERNON_STATUS_INVALID_ARGUMENT);
    }
    if (stream.pipeline == &pipeline && stream.graphicsState == command.graphicsState &&
        stream.vertexStrides == stream.nextVertexStrides)
        return VERNON_STATUS_OK;
    stream.pipeline = nullptr;
    const PlannedRenderTargets &targets = *command.targets;
    std::vector<uint32_t> formats;
    formats.reserve(targets.attachmentFormats.size());
    for (VernonTextureFormat format : targets.attachmentFormats) {
        const std::optional<uint32_t> native = target.colorFormat(format);
        if (!native)
            return fail(context, std::string(target.name) + " color attachment format is unsupported",
                        VERNON_STATUS_INVALID_ARGUMENT);
        formats.push_back(*native);
    }
    uint32_t depthFormat = 0;
    if (targets.depthAttachment) {
        const std::optional<uint32_t> native = target.depthFormat(targets.depthFormat);
        if (!native)
            return fail(context, std::string(target.name) + " depth attachment format is unsupported",
                        VERNON_STATUS_INVALID_ARGUMENT);
        depthFormat = *native;
    }
    GraphicsVariantKey key;
    if (!prepareGraphicsVariantKey(*command.graphicsState, targets, std::move(formats), depthFormat,
                                   stream.nextVertexStrides, key, invocationDiagnostic(context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonStatus status = ensureGraphicsVariant(pipeline.pipeline, key, pipeline.variant);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "prepare", "graphics variant"), status);
    status = vernonRuntimeCorePacketsBindGraphicsVariant(packets, pipeline.variant.handle);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "bind", "graphics variant"), status);
    stream.pipeline = &pipeline;
    stream.graphicsState = command.graphicsState;
    std::swap(stream.vertexStrides, stream.nextVertexStrides);
    return VERNON_STATUS_OK;
}

} // namespace

bool prepareRhiComputePipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                               const RhiComputeTarget &target, const LoadedStageArtifact &stage,
                               const StageBindingPlan &plan, RhiComputePipeline &output,
                               RhiComputeNativeBindingResolver resolver) {
    std::string &diagnostic = invocationDiagnostic(context);
    ReflectedEntry reflection;
    if (!resolveStageReflection(stage, context.backend, reflection, diagnostic) ||
        !buildPreparedComputeBindingPlan(plan, reflection, context.backend, output.bindingPlan, diagnostic))
        return false;
    if (resolver.resolve) {
        for (size_t index = 0; index < output.bindingPlan.size(); ++index)
            if (!resolver.resolve(resolver.userData, output.bindingPlan.sources[index],
                                  output.bindingPlan.layouts[index], diagnostic))
                return false;
    }
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    if (!output.groups.prepare(*provider, output.bindingPlan.layouts,
                               preparedComputeSlotDependencies(output.bindingPlan, plan), diagnostic))
        return false;
    const VernonRuntimeProviderShaderDescriptor shader =
        shaderDescriptor(VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, target.artifactKind, stage);
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.group_layouts = output.groups.layouts();
    descriptor.group_layout_count = output.groups.size();
    std::copy_n(stage.workgroup, 3, descriptor.workgroup_size);
    if (vernonRuntimeCorePreparePipeline(provider, &descriptor, &output.pipeline) == VERNON_STATUS_OK)
        return true;
    diagnostic = providerFailure(adapter, target.name, "prepare", "compute pipeline");
    return false;
}

VernonStatus invokeRhiComputePipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                      const RhiComputeTarget &target, const StageBindingPlan &projection,
                                      VernonRuntimeCorePackets *packets, GroupStore &transient,
                                      const PlannedComputeLaunch &launch, const ArgumentOwner *owners,
                                      RhiComputePipeline &pipeline) {
    const ComputeGroupValues values{target, pipeline.bindingPlan, projection, launch};
    const GroupSources sources{owners, &launch, &transient, {fillComputeGroupValue, &values}};
    VernonStatus status = vernonRuntimeCorePacketsBindPipeline(packets, pipeline.pipeline);
    std::string error;
    if (status == VERNON_STATUS_OK)
        status = pipeline.groups.bind(packets, launch.commandEncoder, sources, error);
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    if (status == VERNON_STATUS_OK)
        status = vernonRuntimeCorePacketsDispatch(packets, groups);
    if (!error.empty())
        return fail(context, std::string(target.name) + " " + error, VERNON_STATUS_INVALID_ARGUMENT);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "encode", "compute dispatch"), status);
    return VERNON_STATUS_OK;
}

RhiComputePipeline::~RhiComputePipeline() { vernonRuntimeCorePipelineDestroy(pipeline); }

bool prepareRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                const RhiGraphicsTarget &target, const LoadedStageArtifact &vertex,
                                const LoadedStageArtifact &fragment, const StageBindingPlan &plan,
                                const PreparedGraphicsBindingOptions &options, RhiGraphicsPipeline &output) {
    std::string &diagnostic = invocationDiagnostic(context);
    if (!buildPreparedGraphicsBindingPlan(plan, options, output.bindingPlan, diagnostic))
        return false;
    const VernonRuntimeDeviceProvider *provider = vernonRuntimeRhiAdapterGetProvider(adapter);
    if (!output.groups.prepare(*provider, output.bindingPlan.layouts,
                               preparedGraphicsSlotDependencies(output.bindingPlan), diagnostic))
        return false;
    const VernonRuntimeProviderShaderDescriptor shaders[2]{
        shaderDescriptor(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX, target.artifactKind, vertex),
        shaderDescriptor(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT, target.artifactKind, fragment)};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
    descriptor.shaders = shaders;
    descriptor.shader_count = 2;
    descriptor.group_layouts = output.groups.layouts();
    descriptor.group_layout_count = output.groups.size();
    descriptor.vertex_attributes = output.bindingPlan.vertexAttributes.data();
    descriptor.vertex_attribute_count = output.bindingPlan.vertexAttributes.size();
    descriptor.vertex_bindings = output.bindingPlan.vertexBindings.data();
    descriptor.vertex_binding_count = output.bindingPlan.vertexBindings.size();
    descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    descriptor.sample_count = 1;
    if (vernonRuntimeCorePreparePipeline(provider, &descriptor, &output.pipeline) == VERNON_STATUS_OK)
        return true;
    diagnostic = providerFailure(adapter, target.name, "prepare", "graphics pipeline");
    return false;
}

VernonStatus invokeRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                       const RhiGraphicsTarget &target, VernonRuntimeCorePackets *packets,
                                       const GraphicsCommand &command, GraphicsStream &stream,
                                       RhiGraphicsPipeline &pipeline) {
    if (!pipeline.pipeline)
        return VERNON_STATUS_OK;
    VernonStatus status = bindGraphicsVariant(context, adapter, target, packets, command, stream, pipeline);
    if (status != VERNON_STATUS_OK)
        return status;
    const GraphicsGroupValues values{pipeline.bindingPlan, {command.arguments, command.draw.resolution}};
    GraphicsGroupsMade made{command.made, pipeline.bindingPlan};
    GroupSources sources = command.groups;
    sources.values = {fillGraphicsGroupValue, &values};
    if (command.made.made)
        sources.made = {forwardGraphicsGroup, &made};
    std::string error;
    status = pipeline.groups.bind(packets, command.encoder, sources, error);
    if (!error.empty())
        return fail(context, std::move(error), VERNON_STATUS_INVALID_ARGUMENT);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "bind", "graphics groups"), status);
    status =
        vernonRuntimeCorePacketsBindVertexBuffers(packets, 0, stream.vertexBuffers.data(), stream.vertexBuffers.size());
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "bind", "vertex buffers"), status);
    VernonRuntimeCoreDrawInvocation draw;
    if (!prepareGraphicsDrawInvocation(command.draw, command.graphicsState->topology, draw,
                                       invocationDiagnostic(context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    status = vernonRuntimeCorePacketsDraw(packets, &draw);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "encode", "draw"), status);
    return VERNON_STATUS_OK;
}

RhiGraphicsPipeline::~RhiGraphicsPipeline() {
    destroyGraphicsVariant(variant);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

} // namespace vernon::runtime
