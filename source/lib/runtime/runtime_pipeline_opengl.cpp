#include "runtime_pipeline_backend.h"

#include "VernonRuntimeRHIAdapter.h"
#include "backend_opengl.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "prepared_binding_plan.h"
#include "prepared_graphics_draw.h"
#include "tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::runtime {

namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

} // namespace

BackendPipelineResult resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
        auto *state = new OpenGLPipelineState();
        if (!plan.compute.empty()) {
            const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
            ReflectedEntry reflection;
            if (!resolveStageReflection(stage, inputs.context->backend, reflection,
                                        invocationDiagnostic(*inputs.context))) {
                delete state;
                return false;
            }
            if (!buildPreparedComputeBindingPlan(plan, reflection, inputs.context->backend,
                                                 state->rhiComputeBindingPlan, invocationDiagnostic(*inputs.context))) {
                delete state;
                return false;
            }
            const auto &layouts = state->rhiComputeBindingPlan.layouts;
            state->rhiValues.resize(layouts.size());
            state->rhiComputeBindingStorage.resize(layouts.size());
            for (size_t index = 0; index < layouts.size(); ++index) {
                state->rhiValues[index].slot = layouts[index].slot;
                state->rhiValues[index].kind = layouts[index].kind;
                const bool runtimeManagedBytes =
                    layouts[index].kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                    layouts[index].interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                if (runtimeManagedBytes) {
                    auto &storage = state->rhiComputeBindingStorage[index];
                    storage.resize(layouts[index].element_size);
                    state->rhiValues[index].payload.inline_value.data = storage.data();
                    state->rhiValues[index].payload.inline_value.size = storage.size();
                } else {
                    state->rhiValues[index].flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
                }
            }
            const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                               VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                               {"glsl", 4},
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
            descriptor.bindings = layouts.data();
            descriptor.binding_count = layouts.size();
            std::copy_n(stage.workgroup, 3, descriptor.workgroup_size);
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(openGLState(*inputs.context).adapter), &descriptor,
                &state->rhiPipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare OpenGL provider compute pipeline";
                delete state;
                return false;
            }
            if (!layouts.empty()) {
                const VernonStatus bindingStatus = vernonRuntimeCoreCreateBindings(
                    state->rhiPipeline, state->rhiValues.data(), state->rhiValues.size(), &state->rhiBindings);
                if (bindingStatus != VERNON_STATUS_OK) {
                    const VernonStringView providerError =
                        vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
                    invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                                ? std::string(providerError.data, providerError.size)
                                                                : "failed to prepare OpenGL provider compute bindings";
                    vernonRuntimeCorePipelineDestroy(state->rhiPipeline);
                    delete state;
                    return false;
                }
            }
            std::copy_n(stage.workgroup, 3, state->workgroup);
            installRuntimeBackendState(pipeline, state);
            return true;
        }
        const bool useRhiGraphics = plan.compute.empty() && !plan.vertex.empty() && !plan.fragment.empty();
        std::string representationError;
        if (useRhiGraphics && !buildPreparedGraphicsBindingPlan(plan, {inputs.context->backend},
                                                                state->rhiGraphicsBindingPlan, representationError)) {
            invocationDiagnostic(*inputs.context) = std::move(representationError);
            delete state;
            return false;
        }
        if (useRhiGraphics) {
            const LoadedStageArtifact &vertex = inputs.artifacts.at(plan.vertex);
            const LoadedStageArtifact &fragment = inputs.artifacts.at(plan.fragment);
            std::vector<VernonRuntimeProviderShaderDescriptor> shaders{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                        VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                        {"glsl", 4},
                                                                        vertex.source.data(),
                                                                        vertex.source.size(),
                                                                        {vertex.entry.data(), vertex.entry.size()},
                                                                        {nullptr, 0},
                                                                        {0, 0, 0, 0}}};
            shaders.push_back({sizeof(VernonRuntimeProviderShaderDescriptor),
                               VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                               {"glsl", 4},
                               fragment.source.data(),
                               fragment.source.size(),
                               {fragment.entry.data(), fragment.entry.size()},
                               {nullptr, 0},
                               {0, 0, 0, 0}});
            VernonRuntimeCorePipelineDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
            descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
            descriptor.shaders = shaders.data();
            descriptor.shader_count = shaders.size();
            descriptor.bindings = state->rhiGraphicsBindingPlan.layouts.data();
            descriptor.binding_count = state->rhiGraphicsBindingPlan.size();
            descriptor.vertex_attributes = state->rhiGraphicsBindingPlan.vertexAttributes.data();
            descriptor.vertex_attribute_count = state->rhiGraphicsBindingPlan.vertexAttributes.size();
            descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
            descriptor.sample_count = 1;
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(openGLState(*inputs.context).adapter), &descriptor,
                &state->rhiPipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare OpenGL RHI provider pipeline";
                delete state;
                return false;
            }
            if (!state->rhiGraphicsBindingPlan.empty()) {
                state->rhiValues.resize(state->rhiGraphicsBindingPlan.size());
                for (size_t index = 0; index < state->rhiGraphicsBindingPlan.size(); ++index) {
                    const auto &layout = state->rhiGraphicsBindingPlan.layouts[index];
                    auto &value = state->rhiValues[index];
                    value.slot = layout.slot;
                    value.kind = layout.kind;
                    if (layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                        layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                        (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
                         layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM)) {
                        auto &storage = state->rhiGraphicsBindingPlan.sources[index].storage;
                        value.payload.inline_value.data = storage.data();
                        value.payload.inline_value.size = storage.size();
                    } else {
                        value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
                    }
                }
                const VernonStatus bindingStatus = vernonRuntimeCoreCreateBindings(
                    state->rhiPipeline, state->rhiValues.data(), state->rhiValues.size(), &state->rhiBindings);
                if (bindingStatus != VERNON_STATUS_OK) {
                    const VernonStringView providerError =
                        vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
                    invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                                ? std::string(providerError.data, providerError.size)
                                                                : "failed to prepare OpenGL RHI bindings";
                    vernonRuntimeCorePipelineDestroy(state->rhiPipeline);
                    delete state;
                    return false;
                }
            }
            installRuntimeBackendState(pipeline, state);
            return true;
        }
        invocationDiagnostic(*inputs.context) =
            representationError.empty() ? "OpenGL ProgramAsset is not representable by the RuntimeCore provider"
                                        : std::move(representationError);
        delete state;
        return false;
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

void destroyOpenGLPipeline(VernonStageExecutable &pipeline) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    destroyGraphicsVariant(state.rhiGraphicsVariant);
    vernonRuntimeCoreBindingsDestroy(state.rhiBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiPipeline);
}

VernonStatus invokeOpenGLGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.rhiPipeline)
        return VERNON_STATUS_OK;
    if (!fillPreparedGraphicsBindingValues(state.rhiGraphicsBindingPlan, plan, state.rhiValues,
                                           invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (state.rhiBindings) {
        const VernonStatus bindingStatus =
            vernonRuntimeCoreUpdateBindings(state.rhiBindings, state.rhiValues.data(), state.rhiValues.size());
        if (bindingStatus != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(openGLState(*pipeline.context).adapter);
            return fail(*pipeline.context,
                        providerError.data ? std::string(providerError.data, providerError.size)
                                           : "failed to update OpenGL RHI bindings",
                        bindingStatus);
        }
    }
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachments.size());
    for (VernonTextureFormat format : plan.attachmentFormats)
        formats.push_back(static_cast<uint32_t>(format));
    const uint32_t depthFormat = !plan.depthAttachment ? 0
                                 : plan.depthFormat == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                     ? rhi::opengl::kDepth32fStencil8
                                     : rhi::opengl::kDepthComponent32f;
    PreparedGraphicsDraw prepared;
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, state.rhiGraphicsBindingPlan.layouts,
                             state.rhiValues, prepared, invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus variantStatus =
        ensureGraphicsVariant(state.rhiPipeline, prepared.variantKey, state.rhiGraphicsVariant);
    if (variantStatus != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(openGLState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare OpenGL RHI graphics variant",
                    variantStatus);
    }
    const VernonStatus status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(
        state.rhiGraphicsVariant.handle, state.rhiBindings, &prepared.invocation);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(openGLState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode OpenGL RHI draw",
                    status);
    }
    return VERNON_STATUS_OK;
}

VernonStatus invokeOpenGLComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.rhiPipeline)
        return fail(*pipeline.context, "OpenGL provider compute pipeline is not loaded");
    const PreparedComputeBindingPlan &bindingPlan = state.rhiComputeBindingPlan;
    for (size_t index = 0; index < bindingPlan.size(); ++index) {
        const auto &layout = bindingPlan.layouts[index];
        const PreparedBindingSource &preparedSource = bindingPlan.sources[index];
        auto &value = state.rhiValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (preparedSource.metadataCarrier) {
            if (!bindComputeMetadataCarrier(layout, preparedSource, launch, value))
                return fail(*pipeline.context, "OpenGL metadata carrier payload does not match its compiled ABI");
            continue;
        }
        if (preparedSource.argumentIndex >= launch.arguments.size())
            return fail(*pipeline.context, "OpenGL prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[preparedSource.argumentIndex];
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
                if (!bindComputeValueStorage(layout, argument, value))
                    return fail(*pipeline.context, "OpenGL prepared Value storage binding has invalid bytes");
                continue;
            }
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "OpenGL prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
            if (value.payload.buffer.resource.offset > value.payload.buffer.resource.size ||
                preparedSource.resourceOffset >
                    value.payload.buffer.resource.size - value.payload.buffer.resource.offset)
                return fail(*pipeline.context, "OpenGL aggregate storage leaf exceeds its Tensor resource");
            value.payload.buffer.resource.offset += preparedSource.resourceOffset;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                   layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto *image = std::get_if<ComputeImageArgument>(&argument);
            if (!image || !image->view.resource.value)
                return fail(*pipeline.context, "OpenGL prepared image binding requires an RHI Texture");
            value.payload.image.view = image->view;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar || !scalar->data || !scalar->size)
                return fail(*pipeline.context, "OpenGL prepared inline binding requires host data");
            value.payload.inline_value.data = scalar->data;
            value.payload.inline_value.size = scalar->size;
        }
    }
    VernonStatus status =
        state.rhiBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiBindings, state.rhiValues.data(), state.rhiValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiPipeline, state.rhiValues.data(), state.rhiValues.size(),
                                              &state.rhiBindings);
    if (status != VERNON_STATUS_OK)
        return fail(*pipeline.context, "failed to prepare OpenGL provider bindings", status);
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiPipeline, state.rhiBindings, launch.commandEncoder, groups,
                                             nullptr, 0);
    return status == VERNON_STATUS_OK ? status
                                      : fail(*pipeline.context, "failed to encode OpenGL provider dispatch", status);
}

} // namespace vernon::runtime
