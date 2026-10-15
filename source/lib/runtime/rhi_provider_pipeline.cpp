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

VernonStatus updateBindings(VernonRuntimeCorePipeline *pipeline, VernonRuntimeCoreBindings *&bindings,
                            const std::vector<VernonRuntimeProviderBindingValue> &values) {
    // Providers take a null binding set for a pipeline without layouts, and the OpenGL draw rejects any other.
    if (values.empty())
        return VERNON_STATUS_OK;
    return bindings ? vernonRuntimeCoreUpdateBindings(bindings, values.data(), values.size())
                    : vernonRuntimeCoreCreateBindings(pipeline, values.data(), values.size(), &bindings);
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
    output.values.resize(output.bindingPlan.size());
    const VernonRuntimeProviderShaderDescriptor shader =
        shaderDescriptor(VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE, target.artifactKind, stage);
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
    descriptor.shaders = &shader;
    descriptor.shader_count = 1;
    descriptor.bindings = output.bindingPlan.layouts.data();
    descriptor.binding_count = output.bindingPlan.size();
    std::copy_n(stage.workgroup, 3, descriptor.workgroup_size);
    if (vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(adapter), &descriptor, &output.pipeline) ==
        VERNON_STATUS_OK)
        return true;
    diagnostic = providerFailure(adapter, target.name, "prepare", "compute pipeline");
    return false;
}

VernonStatus invokeRhiComputePipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                      const RhiComputeTarget &target, const StageBindingPlan &projection,
                                      const PlannedComputeLaunch &launch, RhiComputePipeline &pipeline) {
    std::string error;
    if (!target.fillValues(pipeline.bindingPlan, projection, launch, pipeline.values, error))
        return fail(context, std::string(target.name) + " " + error, VERNON_STATUS_INVALID_ARGUMENT);
    VernonStatus status = updateBindings(pipeline.pipeline, pipeline.bindings, pipeline.values);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "prepare", "compute bindings"), status);
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(pipeline.pipeline, pipeline.bindings, launch.commandEncoder, groups,
                                             nullptr, 0);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "encode", "compute dispatch"), status);
    return VERNON_STATUS_OK;
}

RhiComputePipeline::~RhiComputePipeline() {
    vernonRuntimeCoreBindingsDestroy(bindings);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

bool prepareRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                const RhiGraphicsTarget &target, const LoadedStageArtifact &vertex,
                                const LoadedStageArtifact &fragment, const StageBindingPlan &plan,
                                const PreparedGraphicsBindingOptions &options, RhiGraphicsPipeline &output) {
    std::string &diagnostic = invocationDiagnostic(context);
    if (!buildPreparedGraphicsBindingPlan(plan, options, output.bindingPlan, diagnostic))
        return false;
    output.values.resize(output.bindingPlan.size());
    const VernonRuntimeProviderShaderDescriptor shaders[2]{
        shaderDescriptor(VERNON_RUNTIME_PROVIDER_STAGE_VERTEX, target.artifactKind, vertex),
        shaderDescriptor(VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT, target.artifactKind, fragment)};
    VernonRuntimeCorePipelineDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
    descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
    descriptor.shaders = shaders;
    descriptor.shader_count = 2;
    descriptor.bindings = output.bindingPlan.layouts.data();
    descriptor.binding_count = output.bindingPlan.size();
    descriptor.vertex_attributes = output.bindingPlan.vertexAttributes.data();
    descriptor.vertex_attribute_count = output.bindingPlan.vertexAttributes.size();
    descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    descriptor.sample_count = 1;
    if (vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(adapter), &descriptor, &output.pipeline) ==
        VERNON_STATUS_OK)
        return true;
    diagnostic = providerFailure(adapter, target.name, "prepare", "graphics pipeline");
    return false;
}

VernonStatus invokeRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                       const RhiGraphicsTarget &target,
                                       const VernonStageInvocationDescriptor &invocation,
                                       const PlannedGraphicsInvocation &plan, RhiGraphicsPipeline &pipeline) {
    if (!pipeline.pipeline)
        return VERNON_STATUS_OK;
    std::string &diagnostic = invocationDiagnostic(context);
    if (!fillPreparedGraphicsBindingValues(pipeline.bindingPlan, plan, pipeline.values, diagnostic))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonStatus status = updateBindings(pipeline.pipeline, pipeline.bindings, pipeline.values);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "prepare", "graphics bindings"), status);
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachmentFormats.size());
    for (VernonTextureFormat format : plan.attachmentFormats) {
        const std::optional<uint32_t> native = target.colorFormat(format);
        if (!native)
            return fail(context, std::string(target.name) + " color attachment format is unsupported",
                        VERNON_STATUS_INVALID_ARGUMENT);
        formats.push_back(*native);
    }
    uint32_t depthFormat = 0;
    if (plan.depthAttachment) {
        const std::optional<uint32_t> native = target.depthFormat(plan.depthFormat);
        if (!native)
            return fail(context, std::string(target.name) + " depth attachment format is unsupported",
                        VERNON_STATUS_INVALID_ARGUMENT);
        depthFormat = *native;
    }
    PreparedGraphicsDraw prepared;
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, pipeline.bindingPlan.layouts,
                             pipeline.values, prepared, diagnostic))
        return VERNON_STATUS_INVALID_ARGUMENT;
    status = ensureGraphicsVariant(pipeline.pipeline, prepared.variantKey, pipeline.variant);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "prepare", "graphics variant"), status);
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(pipeline.variant.handle, pipeline.bindings,
                                                                  &prepared.invocation);
    if (status != VERNON_STATUS_OK)
        return fail(context, providerFailure(adapter, target.name, "encode", "draw"), status);
    return VERNON_STATUS_OK;
}

RhiGraphicsPipeline::~RhiGraphicsPipeline() {
    vernonRuntimeCoreBindingsDestroy(bindings);
    destroyGraphicsVariant(variant);
    vernonRuntimeCorePipelineDestroy(pipeline);
}

} // namespace vernon::runtime
