#include "host_tape_allocator.h"
#include "runtime_autodiff_internal.h"
#include "runtime_gpu_bindings.h"
#include "runtime_gpu_executable.h"
#include "runtime_gpu_resources.h"
#include "runtime_gpu_signature.h"

#include "runtime/runtime_state.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vernon::runtime::ad {
namespace {

using gpu::BindingSpecPlan;
using gpu::OwnedPipeline;

bool isReplaySegment(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::ReplaySegment;
}

bool isReplayStatus(const Parameter &parameter) { return parameter.autodiffRole == AutodiffResourceRole::ReplayStatus; }

bool isTape(const Parameter &parameter) {
    return parameter.autodiffRole == AutodiffResourceRole::Tape && !isReplaySegment(parameter) &&
           !isReplayStatus(parameter);
}

const Parameter *findReplaySegment(const Variant &variant) {
    const auto found = std::find_if(variant.parameters.begin(), variant.parameters.end(), isReplaySegment);
    return found == variant.parameters.end() ? nullptr : &*found;
}

} // namespace

bool createGpuExecutable(VernonRuntimeContext &context, const Stage &primal, const Stage &forward,
                         const Stage &backward, const std::vector<std::string> &gradientPaths,
                         uint64_t staticTapeBytesHint, const std::string &residualStorage,
                         const std::string &selectedPolicy, std::shared_ptr<Executable> &executable) {
    if (context.rhiDevice.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        invocationDiagnostic(context) = "GPU autodiff requires an RHI-backed runtime context";
        return false;
    }
    if (!context.autodiffMemoryPolicy)
        context.autodiffMemoryPolicy = std::make_shared<AutodiffMemoryPolicy>();
    PlanningPolicy planningPolicy{};
    if (!parsePlanningPolicy(selectedPolicy, planningPolicy)) {
        invocationDiagnostic(context) = "GPU autodiff profile has invalid planning policy metadata";
        return false;
    }
    const bool noTape = residualStorage == "none";
    if ((noTape != (staticTapeBytesHint == 0)) ||
        (residualStorage != "none" && residualStorage != "static" && residualStorage != "dynamic") ||
        staticTapeBytesHint > std::numeric_limits<size_t>::max()) {
        invocationDiagnostic(context) = "GPU autodiff profile has inconsistent residual storage metadata";
        return false;
    }
    auto forwardPipeline = std::make_shared<OwnedPipeline>();
    auto backwardPipeline = std::make_shared<OwnedPipeline>();
    if (!forwardPipeline->create(context, forward)) {
        invocationDiagnostic(context) =
            "cannot load GPU autodiff forward stage '" + forward.entry + "': " + invocationDiagnostic(context);
        return false;
    }
    if (!backwardPipeline->create(context, backward)) {
        invocationDiagnostic(context) =
            "cannot load GPU autodiff backward stage '" + backward.entry + "': " + invocationDiagnostic(context);
        return false;
    }
    auto signature = std::make_shared<Signature>();
    const Variant &forwardProjection = (**forwardPipeline).bindingProjection;
    const Variant &backwardProjection = (**backwardPipeline).bindingProjection;
    if (!gpu::buildSignature(context, forwardProjection, backwardProjection, gradientPaths, *signature))
        return false;
    if (noTape) {
        auto backwardBindingSpecs = std::make_shared<BindingSpecPlan>();
        if (!gpu::buildBindingSpecPlan(backwardProjection, *signature, *backwardBindingSpecs,
                                       invocationDiagnostic(context)))
            return false;
        executable = gpu::createNoTapeExecutable(context, std::move(forwardPipeline), std::move(backwardPipeline),
                                                 std::move(signature), std::move(backwardBindingSpecs));
        return true;
    }
    const auto tapeCount = [](const Variant &variant) {
        return static_cast<size_t>(std::count_if(variant.parameters.begin(), variant.parameters.end(), isTape));
    };
    if (tapeCount(forwardProjection) != 1 || tapeCount(backwardProjection) != 1 ||
        !findReplaySegment(forwardProjection) || !findReplaySegment(backwardProjection)) {
        invocationDiagnostic(context) =
            "GPU captured Tape profile does not expose one Tape and one bounded-replay segment resource";
        return false;
    }
    auto forwardBindingSpecs = std::make_shared<BindingSpecPlan>();
    auto backwardBindingSpecs = std::make_shared<BindingSpecPlan>();
    if (!gpu::buildBindingSpecPlan(forwardProjection, *signature, *forwardBindingSpecs,
                                   invocationDiagnostic(context)) ||
        !gpu::buildBindingSpecPlan(backwardProjection, *signature, *backwardBindingSpecs,
                                   invocationDiagnostic(context)))
        return false;
    auto primalPipeline = std::make_shared<OwnedPipeline>();
    if (!primalPipeline->create(context, primal)) {
        invocationDiagnostic(context) =
            "cannot load GPU autodiff primal stage '" + primal.entry + "': " + invocationDiagnostic(context);
        return false;
    }
    // The current artifact contract exposes only a hint: even "static" profiles may
    // report a larger required stride at runtime, so every Tape profile must inspect status.
    executable = gpu::createTapeExecutable(context, std::move(primalPipeline), std::move(forwardPipeline),
                                           std::move(backwardPipeline), std::move(signature),
                                           std::move(forwardBindingSpecs), std::move(backwardBindingSpecs),
                                           static_cast<size_t>(staticTapeBytesHint), planningPolicy, true);
    return true;
}

} // namespace vernon::runtime::ad
