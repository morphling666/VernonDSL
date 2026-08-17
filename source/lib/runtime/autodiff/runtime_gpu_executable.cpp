#include "runtime_gpu_executable.h"

#include "runtime/runtime_state.h"
#include "runtime_gpu_preparation.h"
#include "runtime_gpu_pullback.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace vernon::runtime::ad::gpu {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

std::vector<std::string> retainedNames(const BindingSpecPlan &first, const BindingSpecPlan *second = nullptr) {
    std::vector<std::string> result;
    const auto append = [&result](const std::vector<std::string> &names) {
        for (const std::string &name : names)
            if (std::find(result.begin(), result.end(), name) == result.end())
                result.push_back(name);
    };
    append(first.retainedNames);
    append(first.primalNames);
    if (second) {
        append(second->retainedNames);
        append(second->primalNames);
    }
    return result;
}

class TapeExecutable final : public Executable {
public:
    TapeExecutable(VernonRuntimeContext &context, std::shared_ptr<OwnedPipeline> primal,
                   std::shared_ptr<OwnedPipeline> forward, std::shared_ptr<OwnedPipeline> backward,
                   std::shared_ptr<Signature> signature, std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
                   std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs, size_t staticTapeBytesHint,
                   PlanningPolicy planningPolicy)
        : context_(context), primal_(std::move(primal)), forward_(std::move(forward)), backward_(std::move(backward)),
          signature_(std::move(signature)), forwardBindingSpecs_(std::move(forwardBindingSpecs)),
          backwardBindingSpecs_(std::move(backwardBindingSpecs)), staticTapeBytesHint_(staticTapeBytesHint),
          planningPolicy_(planningPolicy),
          retainedNames_(retainedNames(*forwardBindingSpecs_, backwardBindingSpecs_.get())) {}

    const Signature &signature() const override { return *signature_; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize computeGrid,
                         const VernonAdValueSet &inputs, VernonAdValueSet *outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        if (inputs.value_count != signature_->inputs.size() ||
            (outputs && outputs->value_count != 0 && outputs->value_count != signature_->outputs.size()))
            return fail(context_, "GPU autodiff forward values do not match reflection");
        PreparedForwardExecution prepared;
        VernonStatus status = prepareForward(context_, *primal_, signature_, computeGrid, inputs, target,
                                             retainedNames_, "primal", prepared);
        if (status != VERNON_STATUS_OK)
            return status;
        auto candidate = createTapePullback(context_, prepared.signature, forward_, backward_, forwardBindingSpecs_,
                                            backwardBindingSpecs_, computeGrid, std::move(prepared.retainedDevices),
                                            std::move(prepared.retainedHosts), staticTapeBytesHint_, planningPolicy_);
        if (!candidate)
            return fail(context_, "GPU autodiff retained values exceed the context memory budget");
        if (!target.externalEncoder()) {
            if (!outputs)
                return fail(context_, "standalone GPU autodiff forward has no outputs");
            PreparedForwardPublication publication;
            if (!stageForwardResults(*prepared.signature, *primal_, prepared.working, inputs, *outputs, publication) ||
                !publishForwardResults(publication))
                return fail(context_, "cannot publish GPU autodiff forward outputs", VERNON_STATUS_INTERNAL_ERROR);
        }
        pullback = std::move(candidate);
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<OwnedPipeline> primal_;
    std::shared_ptr<OwnedPipeline> forward_;
    std::shared_ptr<OwnedPipeline> backward_;
    std::shared_ptr<Signature> signature_;
    std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs_;
    std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs_;
    size_t staticTapeBytesHint_{};
    PlanningPolicy planningPolicy_{};
    std::vector<std::string> retainedNames_;
};

class NoTapeExecutable final : public Executable {
public:
    NoTapeExecutable(VernonRuntimeContext &context, std::shared_ptr<OwnedPipeline> forward,
                     std::shared_ptr<OwnedPipeline> backward, std::shared_ptr<Signature> signature,
                     std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs)
        : context_(context), forward_(std::move(forward)), backward_(std::move(backward)),
          signature_(std::move(signature)), backwardBindingSpecs_(std::move(backwardBindingSpecs)),
          retainedNames_(retainedNames(*backwardBindingSpecs_)) {}

    const Signature &signature() const override { return *signature_; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize computeGrid,
                         const VernonAdValueSet &inputs, VernonAdValueSet *outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        if (inputs.value_count != signature_->inputs.size() ||
            (outputs && outputs->value_count != 0 && outputs->value_count != signature_->outputs.size()))
            return fail(context_, "GPU autodiff forward values do not match reflection");
        PreparedForwardExecution prepared;
        VernonStatus status = prepareForward(context_, *forward_, signature_, computeGrid, inputs, target,
                                             retainedNames_, "forward", prepared);
        if (status != VERNON_STATUS_OK)
            return status;
        auto candidate =
            createNoTapePullback(context_, prepared.signature, backward_, backwardBindingSpecs_, computeGrid,
                                 std::move(prepared.retainedDevices), std::move(prepared.retainedHosts));
        if (!candidate)
            return fail(context_, "GPU autodiff retained values exceed the context memory budget");
        if (!target.externalEncoder()) {
            if (!outputs)
                return fail(context_, "standalone GPU autodiff forward has no outputs");
            PreparedForwardPublication publication;
            if (!stageForwardResults(*prepared.signature, *forward_, prepared.working, inputs, *outputs, publication) ||
                !publishForwardResults(publication))
                return fail(context_, "cannot publish GPU autodiff forward outputs", VERNON_STATUS_INTERNAL_ERROR);
        }
        pullback = std::move(candidate);
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<OwnedPipeline> forward_;
    std::shared_ptr<OwnedPipeline> backward_;
    std::shared_ptr<Signature> signature_;
    std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs_;
    std::vector<std::string> retainedNames_;
};

} // namespace

std::shared_ptr<Executable> createNoTapeExecutable(VernonRuntimeContext &context,
                                                   std::shared_ptr<OwnedPipeline> forward,
                                                   std::shared_ptr<OwnedPipeline> backward,
                                                   std::shared_ptr<Signature> signature,
                                                   std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs) {
    return std::make_shared<NoTapeExecutable>(context, std::move(forward), std::move(backward), std::move(signature),
                                              std::move(backwardBindingSpecs));
}

std::shared_ptr<Executable> createTapeExecutable(VernonRuntimeContext &context, std::shared_ptr<OwnedPipeline> primal,
                                                 std::shared_ptr<OwnedPipeline> forward,
                                                 std::shared_ptr<OwnedPipeline> backward,
                                                 std::shared_ptr<Signature> signature,
                                                 std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
                                                 std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs,
                                                 size_t staticTapeBytesHint, PlanningPolicy planningPolicy) {
    return std::make_shared<TapeExecutable>(context, std::move(primal), std::move(forward), std::move(backward),
                                            std::move(signature), std::move(forwardBindingSpecs),
                                            std::move(backwardBindingSpecs), staticTapeBytesHint, planningPolicy);
}

} // namespace vernon::runtime::ad::gpu
