#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_METAL_RUNTIME)
#include "backend_metal.h"

#include <memory>
#include <string>
#include <string_view>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_METAL_RUNTIME)
namespace {

enum class ResourceClass { Buffer, Texture, Sampler };

ResourceClass resourceClass(std::string_view kind) {
    return kind == "sampled_image" || kind == "storage_image" ? ResourceClass::Texture
           : kind == "sampler"                                ? ResourceClass::Sampler
                                                              : ResourceClass::Buffer;
}

ResourceClass resourceClass(VernonRuntimeProviderBindingKind kind) {
    return kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
               ? ResourceClass::Texture
           : kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? ResourceClass::Sampler
                                                     : ResourceClass::Buffer;
}

// A combined image sampler is one texture and one sampler member at the same set and binding.
const NativeResourceSlot *argumentSlot(const LoadedStageArtifact &stage, ResourceClass wanted, uint32_t set,
                                       uint32_t binding) {
    for (const NativeResourceSlot &slot : stage.nativeSlots)
        if (slot.argumentBufferIndex != UINT32_MAX && slot.set == set && slot.binding == binding &&
            resourceClass(slot.kind) == wanted)
            return &slot;
    return nullptr;
}

bool hasPushConstants(const LoadedStageArtifact &stage) {
    for (const NativeResourceSlot &slot : stage.nativeSlots)
        if (slot.directBufferIndex == kMetalPushConstantBuffer)
            return true;
    return false;
}

struct GraphicsStages {
    const LoadedStageArtifact *vertex{};
    const LoadedStageArtifact *fragment{};
};

// The layouts keep their canonical sets and bindings, which the device maps to argument buffer members through
// MetalPipelineState::slots; this only checks that the binding's stage exists.
bool resolveNativeBinding(void *userData, std::string_view stage, std::string_view, uint32_t, uint32_t,
                          const std::string *, VernonRuntimeProviderBindingLayoutEntry &, std::string &error) {
    const auto &stages = *static_cast<const GraphicsStages *>(userData);
    if ((stage == "vertex" && stages.vertex) || (stage == "fragment" && stages.fragment))
        return true;
    error = "Metal graphics binding names stage '" + std::string(stage) + "', which the Stage has no MSL for";
    return false;
}

bool assignSlots(const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layouts,
                 const LoadedStageArtifact *const (&stages)[3], bool graphics, MetalPipelineState &state,
                 std::string &error) {
    for (uint32_t role = 0; role < 3; ++role)
        if (stages[role])
            for (const NativeResourceSlot &slot : stages[role]->nativeSlots)
                if (slot.argumentBufferIndex < 32)
                    state.argumentBuffers[role] |= 1u << slot.argumentBufferIndex;
    state.slots.assign(layouts.size(), {});
    for (size_t index = 0; index < layouts.size(); ++index) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = layouts[index];
        const uint32_t role = !graphics                                                     ? kMetalCompute
                              : layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX   ? kMetalVertex
                              : layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT ? kMetalFragment
                                                                                            : UINT32_MAX;
        if (role == UINT32_MAX || !stages[role]) {
            error = "a Metal graphics binding is not read by exactly one stage";
            return false;
        }
        MetalBindingSlot &slot = state.slots[index];
        if (graphics && layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
            if (hasPushConstants(*stages[role]))
                slot.directBuffer = kMetalPushConstantBuffer;
            continue;
        }
        if (const NativeResourceSlot *native =
                argumentSlot(*stages[role], resourceClass(layout.kind), layout.set, layout.binding))
            slot.member = native->memberId;
    }
    return true;
}

} // namespace
#endif

BackendPipelineResult resolveMetalPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                           VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_METAL_RUNTIME)
        VernonRuntimeContext &context = *inputs.context;
        std::string &diagnostic = invocationDiagnostic(context);
        auto state = std::make_unique<MetalPipelineState>();
        const auto artifact = [&](const std::string &name) {
            return name.empty() ? nullptr : &inputs.artifacts.at(name);
        };
        const LoadedStageArtifact *const stages[3]{artifact(plan.vertex), artifact(plan.fragment),
                                                   artifact(plan.compute)};
        if (stages[kMetalCompute]) {
            if (!prepareComputeBindingPlan(context, *stages[kMetalCompute], plan, state->computePlan) ||
                !assignSlots(state->computePlan.layouts, stages, false, *state, diagnostic))
                return false;
        }
        if (stages[kMetalVertex]) {
            GraphicsStages graphics{stages[kMetalVertex], stages[kMetalFragment]};
            const PreparedGraphicsBindingOptions options{VERNON_RUNTIME_METAL, resolveNativeBinding, &graphics};
            if (!buildPreparedGraphicsBindingPlan(plan, options, state->graphicsPlan, diagnostic) ||
                !assignSlots(state->graphicsPlan.layouts, stages, true, *state, diagnostic))
                return false;
        }
        const auto code = [&](const LoadedStageArtifact *stage, MetalShaderCode &output) {
            if (!stage)
                return true;
            if (stage->source.empty()) {
                diagnostic = "Metal stage artifact has no MSL";
                return false;
            }
            output.source = stage->source;
            output.entry = stage->entry;
            return true;
        };
        if (!code(stages[kMetalCompute], state->compute) || !code(stages[kMetalVertex], state->vertex) ||
            !code(stages[kMetalFragment], state->fragment))
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

} // namespace vernon::runtime
