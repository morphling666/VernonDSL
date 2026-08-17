#include "runtime_gpu_pullback.h"

#include "runtime/runtime_state.h"
#include "runtime_autodiff_memory_usage.h"
#include "runtime_gpu_argument_binding.h"
#include "runtime_gpu_commands.h"
#include "runtime_gpu_derivatives.h"
#include "runtime_gpu_failure_injection.h"
#include "runtime_gpu_replay.h"
#include "runtime_gpu_replay_execution.h"
#include "runtime_gpu_telemetry.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace vernon::runtime::ad::gpu {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool checkedAdd(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

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

const Parameter *findTape(const Variant &variant) {
    const auto found = std::find_if(variant.parameters.begin(), variant.parameters.end(), isTape);
    return found == variant.parameters.end() ? nullptr : &*found;
}

class NoTapePullback final : public PullbackExecution {
public:
    NoTapePullback(VernonRuntimeContext &context, std::shared_ptr<const Signature> signature,
                   std::shared_ptr<OwnedPipeline> backward, std::shared_ptr<const BindingSpecPlan> bindingSpecs,
                   VernonLaunchSize grid, DeviceValues retainedDevices, HostValues retainedHosts)
        : context_(context), signature_(std::move(signature)), backward_(std::move(backward)),
          bindingSpecs_(std::move(bindingSpecs)), grid_(grid), retainedDevices_(std::move(retainedDevices)),
          retainedHosts_(std::move(retainedHosts)) {
        retainedReservation_ =
            reserveRetainedValues(context_, retainedDevices_, retainedHosts_, memory_.retainedAllocationBytes);
    }

    bool valid() const { return retainedReservation_ != nullptr; }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        control_ = {};
        size_t temporaryLimit = 0;
        std::shared_ptr<AutodiffMemoryReservation> applyReservation =
            reserveApplyMemory(context_, options.maximumTemporaryBytes, temporaryLimit);
        if (!applyReservation)
            return fail(context_, "GPU pullback cannot reserve apply-time memory budget");
        PreparedDerivativeValues derivatives;
        std::string derivativeError;
        VernonLaunchSize extent{};
        if (!invocationExtent(grid_, (*backward_)->workgroupSize, extent))
            return fail(context_, "GPU pullback invocation extent overflows");
        if (!derivatives.prepare(context_, *signature_, cotangents, gradients, *bindingSpecs_, extent,
                                 3 * sizeof(uint32_t), temporaryLimit, derivativeError))
            return fail(context_, std::move(derivativeError), derivatives.failureStatus);
        control_.temporaryAllocationBytes = derivatives.temporaryBytes;

        DeviceValues working;
        BindingPlan bindings;
        std::string bindingError;
        if (!materializeBindingPlan((*backward_)->variant, *bindingSpecs_, retainedDevices_, retainedHosts_, working,
                                    derivatives.devices, bindings, bindingError))
            return fail(context_, std::move(bindingError), VERNON_STATUS_INTERNAL_ERROR);
        std::vector<VernonPipelineArgument> arguments;
        arguments.reserve(bindings.size());
        const uint32_t launchData[3]{grid_.x, grid_.y, grid_.z};
        DeviceBuffer launchBuffer(context_, sizeof(launchData));
        InternalBufferView launchView;
        if (!launchBuffer.upload(launchData, sizeof(launchData)))
            return fail(context_, "cannot upload GPU pullback launch metadata", VERNON_STATUS_INTERNAL_ERROR);
        for (const Binding &binding : bindings) {
            const Parameter &parameter = *binding.parameter;
            if (binding.source == BindingSource::Launch) {
                if (!appendInternalBufferArgument(context_, parameter, launchBuffer, sizeof(launchData), launchView,
                                                  arguments))
                    return fail(context_, "cannot bind GPU pullback launch metadata", VERNON_STATUS_INTERNAL_ERROR);
                continue;
            }
            if (!appendBindingArgument(binding, arguments))
                return fail(context_, "cannot bind GPU pullback argument", VERNON_STATUS_INTERNAL_ERROR);
        }
        VernonStatus status = submitAndWait(**backward_, grid_, arguments, &control_);
        if (status != VERNON_STATUS_OK)
            return status;
        if (!derivatives.stageGradients(*signature_, derivativeError))
            return fail(context_, std::move(derivativeError), VERNON_STATUS_INTERNAL_ERROR);
        control_.readbacks += signature_->gradients.size();
        if (!derivatives.publishGradients())
            return fail(context_, "cannot publish GPU pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
        memory_.observeTemporary(derivatives.temporaryBytes);
        return VERNON_STATUS_OK;
    }

    PullbackMemoryUsage memoryUsage() const override { return pullbackMemoryUsage(memory_); }
    PullbackControlPlaneUsage controlPlaneUsage() const override { return control_; }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const Signature> signature_;
    std::shared_ptr<OwnedPipeline> backward_;
    std::shared_ptr<const BindingSpecPlan> bindingSpecs_;
    VernonLaunchSize grid_{};
    DeviceValues retainedDevices_;
    HostValues retainedHosts_;
    std::shared_ptr<AutodiffMemoryReservation> retainedReservation_;
    MemoryAccounting memory_;
    PullbackControlPlaneUsage control_;
};

class TapePullback final : public PullbackExecution {
public:
    TapePullback(VernonRuntimeContext &context, std::shared_ptr<const Signature> signature,
                 std::shared_ptr<OwnedPipeline> forward, std::shared_ptr<OwnedPipeline> backward,
                 std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
                 std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs, VernonLaunchSize grid,
                 DeviceValues retainedDevices, HostValues retainedHosts, size_t staticTapeBytesHint,
                 PlanningPolicy planningPolicy)
        : context_(context), signature_(std::move(signature)), forward_(std::move(forward)),
          backward_(std::move(backward)), forwardBindingSpecs_(std::move(forwardBindingSpecs)),
          backwardBindingSpecs_(std::move(backwardBindingSpecs)), grid_(grid),
          retainedDevices_(std::move(retainedDevices)), retainedHosts_(std::move(retainedHosts)),
          staticTapeBytesHint_(staticTapeBytesHint), planningPolicy_(planningPolicy) {
        retainedReservation_ =
            reserveRetainedValues(context_, retainedDevices_, retainedHosts_, memory_.retainedAllocationBytes);
    }

    bool valid() const { return retainedReservation_ != nullptr; }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        control_ = {};
        size_t temporaryLimit = 0;
        std::shared_ptr<AutodiffMemoryReservation> applyReservation =
            reserveApplyMemory(context_, options.maximumTemporaryBytes, temporaryLimit);
        if (!applyReservation)
            return fail(context_, "GPU bounded replay cannot reserve apply-time memory budget");
        PreparedDerivativeValues derivatives;
        std::string derivativeError;
        VernonLaunchSize extent{};
        if (!invocationExtent(grid_, (*backward_)->workgroupSize, extent))
            return fail(context_, "GPU pullback invocation extent overflows");
        if (!derivatives.prepare(context_, *signature_, cotangents, gradients, *backwardBindingSpecs_, extent,
                                 3 * sizeof(uint32_t), temporaryLimit, derivativeError))
            return fail(context_, std::move(derivativeError), derivatives.failureStatus);
        control_.temporaryAllocationBytes = derivatives.temporaryBytes;

        DeviceValues working;
        std::vector<DeviceBufferCopy> initialCopies;
        initialCopies.reserve(retainedDevices_.size());
        for (auto &[name, retained] : retainedDevices_) {
            if (!checkedAdd(derivatives.temporaryBytes, retained.buffer.size(), derivatives.temporaryBytes))
                return fail(context_, "GPU bounded-replay restoration storage overflows", VERNON_STATUS_INTERNAL_ERROR);
            control_.temporaryAllocationBytes += retained.buffer.size();
            if (derivatives.temporaryBytes > temporaryLimit)
                return fail(context_, "GPU bounded-replay restoration exceeds the apply-time budget");
            DeviceValue current(context_, retained.buffer.size(), retained.dtype, retained.shape, retained.strides,
                                retained.byteOffset);
            if (!current.buffer.valid())
                return fail(context_, "cannot create GPU replay primal shadow", VERNON_STATUS_INTERNAL_ERROR);
            auto [inserted, ok] = working.emplace(name, std::move(current));
            if (!ok)
                return fail(context_, "cannot index GPU replay primal shadow", VERNON_STATUS_INTERNAL_ERROR);
            initialCopies.push_back(
                {retained.buffer.handle(), inserted->second.buffer.handle(), 0, retained.buffer.size()});
        }
        BindingPlan forwardBindings;
        BindingPlan backwardBindings;
        std::string bindingError;
        if (!materializeBindingPlan((*forward_)->variant, *forwardBindingSpecs_, retainedDevices_, retainedHosts_,
                                    working, derivatives.devices, forwardBindings, bindingError) ||
            !materializeBindingPlan((*backward_)->variant, *backwardBindingSpecs_, retainedDevices_, retainedHosts_,
                                    working, derivatives.devices, backwardBindings, bindingError))
            return fail(context_, std::move(bindingError), VERNON_STATUS_INTERNAL_ERROR);

        size_t workgroupVolume = 1;
        for (uint32_t extent :
             {(*forward_)->workgroupSize.x, (*forward_)->workgroupSize.y, (*forward_)->workgroupSize.z})
            if (!checkedMultiply(workgroupVolume, static_cast<size_t>(extent), workgroupVolume))
                return fail(context_, "GPU bounded-replay workgroup volume overflows", VERNON_STATUS_INTERNAL_ERROR);
        constexpr size_t statusBytes = sizeof(BatchSummary);
        size_t tapeStride = staticTapeBytesHint_;
        if (!normalizeTapeStride(tapeStride, tapeStride))
            return fail(context_, "GPU bounded-replay Tape size overflows", VERNON_STATUS_INTERNAL_ERROR);
        size_t groupTapeBytes = 0;
        if (!checkedMultiply(tapeStride, workgroupVolume, groupTapeBytes))
            return fail(context_, "GPU bounded-replay Tape size overflows", VERNON_STATUS_INTERNAL_ERROR);

        size_t groupCount = 1;
        for (uint32_t extent : {grid_.x, grid_.y, grid_.z})
            if (!checkedMultiply(groupCount, static_cast<size_t>(extent), groupCount))
                return fail(context_, "GPU bounded-replay group count overflows", VERNON_STATUS_INTERNAL_ERROR);
        BatchBudget batchBudget;
        if (!planBatchBudget(planningPolicy_, derivatives.temporaryBytes, temporaryLimit, groupCount, groupTapeBytes,
                             statusBytes, batchBudget))
            return fail(context_, "GPU bounded-replay temporary memory exceeds the apply-time budget");
        size_t batchCapacity = batchBudget.capacity;
        size_t tapeBytes = batchBudget.tapeBytes;
        size_t segmentBytes = batchBudget.segmentBytes;
        size_t statusBufferBytes = batchBudget.statusBytes;
        size_t peakTemporaryBytes = batchBudget.memory.peakTemporaryBytes;

        if (!findTape((*forward_)->variant) || !findTape((*backward_)->variant) ||
            !findReplaySegment((*forward_)->variant) || !findReplaySegment((*backward_)->variant))
            return fail(context_, "GPU Tape profile has no bounded-replay Tape and segment resources",
                        VERNON_STATUS_INTERNAL_ERROR);

        DeviceBuffer tape(context_, tapeBytes);
        DeviceBuffer segment(context_, segmentBytes);
        DeviceBuffer replayStatus(context_, statusBufferBytes);
        const uint32_t launchData[3]{grid_.x, grid_.y, grid_.z};
        DeviceBuffer launch(context_, sizeof(launchData));
        if (!tape.valid() || !segment.valid() || !replayStatus.valid() ||
            !launch.upload(launchData, sizeof(launchData)))
            return fail(context_, "cannot allocate GPU bounded-replay resources", VERNON_STATUS_INTERNAL_ERROR);
        control_.temporaryAllocationBytes += tapeBytes + segmentBytes + statusBufferBytes + sizeof(launchData);

        std::string failedParameter;
        bool pristineWorking = true;
        std::vector<DeviceBufferCopy> pendingCopies = std::move(initialCopies);
        ReverseBatchScheduler scheduler(groupCount, batchCapacity);
        if (!scheduler.valid())
            return fail(context_, "GPU replay scheduler has zero batch capacity", VERNON_STATUS_INTERNAL_ERROR);
        while (!scheduler.empty()) {
            BatchRange batch = scheduler.current();
            size_t batchGroups = batch.count;
            size_t batchBegin = batch.begin;
            if (!pristineWorking && !planReplayRestoreCopies(**forward_, working, retainedDevices_, pendingCopies))
                return fail(context_, "cannot restore GPU replay primal shadow", VERNON_STATUS_INTERNAL_ERROR);
            pristineWorking = false;

            for (;;) {
                std::vector<Segment> metadata(batchGroups);
                for (size_t index = 0; index < batchGroups; ++index)
                    if (!initializeBatchSegment(grid_, (*forward_)->workgroupSize, batchBegin + index, index,
                                                tapeStride, workgroupVolume, tapeBytes, metadata[index]))
                        return fail(context_, "cannot initialize GPU replay segment", VERNON_STATUS_INTERNAL_ERROR);
                if (!segment.upload(0, metadata.data(), metadata.size() * sizeof(Segment)))
                    return fail(context_, "cannot upload GPU replay segment", VERNON_STATUS_INTERNAL_ERROR);
                const BatchSummary emptySummary{};
                if (!replayStatus.upload(0, &emptySummary, sizeof(emptySummary)))
                    return fail(context_, "cannot clear GPU replay batch summary", VERNON_STATUS_INTERNAL_ERROR);
                std::vector<VernonPipelineArgument> arguments;
                ReplayArgumentViews forwardViews;
                if (!appendReplayArguments(context_, forwardBindings, tape, tapeBytes, segment, segmentBytes,
                                           replayStatus, statusBufferBytes, launch, sizeof(launchData), forwardViews,
                                           arguments, failedParameter))
                    return fail(context_, "cannot bind GPU replay forward resource '" + failedParameter + "'",
                                VERNON_STATUS_INTERNAL_ERROR);
                VernonStatus status =
                    pendingCopies.empty()
                        ? submitAndWait(**forward_, {static_cast<uint32_t>(batchGroups), 1, 1}, arguments, &control_)
                        : submitWithCopiesAndWait(context_, pendingCopies, **forward_, arguments,
                                                  {static_cast<uint32_t>(batchGroups), 1, 1}, &control_);
                if (status != VERNON_STATUS_OK)
                    return status;
                pendingCopies.clear();
                size_t batchLanes = 0;
                if (!checkedMultiply(batchGroups, workgroupVolume, batchLanes))
                    return fail(context_, "GPU replay batch lane count overflows", VERNON_STATUS_INTERNAL_ERROR);
                BatchSummary summary{};
                if (!replayStatus.download(0, &summary, sizeof(summary)))
                    return fail(context_, "cannot read GPU replay batch summary", VERNON_STATUS_INTERNAL_ERROR);
                ++control_.readbacks;
                size_t requiredStride = 0;
                size_t required = 0;
                uint32_t tapeStatus = 0;
                if (!requiredTapeBytes(summary, batchLanes, tapeStride, requiredStride, required, tapeStatus))
                    return fail(context_, "GPU dynamic Tape required bytes overflow", VERNON_STATUS_INTERNAL_ERROR);
                (void)required;
                if (tapeStatus == 0)
                    break;
                if (tapeStatus != 1 || requiredStride <= tapeStride)
                    return fail(context_, "GPU Tape construction failed", VERNON_STATUS_INTERNAL_ERROR);
                if (!normalizeTapeStride(requiredStride, requiredStride))
                    return fail(context_, "GPU dynamic Tape stride overflows", VERNON_STATUS_INTERNAL_ERROR);
                size_t requiredGroupBytes = 0;
                BatchBudget replacementBudget;
                if (!checkedMultiply(requiredStride, workgroupVolume, requiredGroupBytes) ||
                    !planBatchBudget(planningPolicy_, derivatives.temporaryBytes, temporaryLimit, groupCount,
                                     requiredGroupBytes, statusBytes, replacementBudget))
                    return fail(context_, "GPU dynamic Tape required bytes exceed the apply-time budget");
                if (injectFailure(FailureBoundary::Resize))
                    return fail(context_, "injected GPU dynamic replay resize failure", VERNON_STATUS_INTERNAL_ERROR);
                std::vector<Segment>().swap(metadata);
                tape = DeviceBuffer{};
                segment = DeviceBuffer{};
                replayStatus = DeviceBuffer{};
                DeviceBuffer replacementTape(context_, replacementBudget.tapeBytes);
                DeviceBuffer replacementSegment(context_, replacementBudget.segmentBytes);
                DeviceBuffer replacementStatus(context_, replacementBudget.statusBytes);
                if (!replacementTape.valid() || !replacementSegment.valid() || !replacementStatus.valid())
                    return fail(context_, "cannot rebuild GPU dynamic replay batch", VERNON_STATUS_INTERNAL_ERROR);
                control_.temporaryAllocationBytes +=
                    replacementBudget.tapeBytes + replacementBudget.segmentBytes + replacementBudget.statusBytes;
                tape = std::move(replacementTape);
                segment = std::move(replacementSegment);
                replayStatus = std::move(replacementStatus);
                tapeStride = requiredStride;
                groupTapeBytes = requiredGroupBytes;
                batchCapacity = replacementBudget.capacity;
                tapeBytes = replacementBudget.tapeBytes;
                segmentBytes = replacementBudget.segmentBytes;
                statusBufferBytes = replacementBudget.statusBytes;
                peakTemporaryBytes = replacementBudget.memory.peakTemporaryBytes;
                if (!scheduler.setCapacity(batchCapacity))
                    return fail(context_, "GPU replay resize produced zero batch capacity",
                                VERNON_STATUS_INTERNAL_ERROR);
                batch = scheduler.current();
                batchGroups = batch.count;
                batchBegin = batch.begin;
                if (!planReplayRestoreCopies(**forward_, working, retainedDevices_, pendingCopies))
                    return fail(context_, "cannot roll back GPU replay after Tape retry", VERNON_STATUS_INTERNAL_ERROR);
            }

            std::vector<VernonPipelineArgument> arguments;
            ReplayArgumentViews backwardViews;
            if (!appendReplayArguments(context_, backwardBindings, tape, tapeBytes, segment, segmentBytes, replayStatus,
                                       statusBufferBytes, launch, sizeof(launchData), backwardViews, arguments,
                                       failedParameter))
                return fail(context_, "cannot bind GPU replay backward resource '" + failedParameter + "'",
                            VERNON_STATUS_INTERNAL_ERROR);
            VernonStatus status =
                submitAndWait(**backward_, {static_cast<uint32_t>(batchGroups), 1, 1}, arguments, &control_);
            if (status != VERNON_STATUS_OK)
                return status;
            scheduler.commit();
        }

        if (!derivatives.stageGradients(*signature_, derivativeError))
            return fail(context_, std::move(derivativeError), VERNON_STATUS_INTERNAL_ERROR);
        control_.readbacks += signature_->gradients.size();
        if (!derivatives.publishGradients())
            return fail(context_, "cannot publish GPU pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
        memory_.observeTemporary(peakTemporaryBytes);
        return VERNON_STATUS_OK;
    }

    PullbackMemoryUsage memoryUsage() const override { return pullbackMemoryUsage(memory_); }
    PullbackControlPlaneUsage controlPlaneUsage() const override { return control_; }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const Signature> signature_;
    std::shared_ptr<OwnedPipeline> forward_;
    std::shared_ptr<OwnedPipeline> backward_;
    std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs_;
    std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs_;
    VernonLaunchSize grid_{};
    DeviceValues retainedDevices_;
    HostValues retainedHosts_;
    std::shared_ptr<AutodiffMemoryReservation> retainedReservation_;
    MemoryAccounting memory_;
    size_t staticTapeBytesHint_{};
    PlanningPolicy planningPolicy_{};
    PullbackControlPlaneUsage control_;
};

} // namespace

std::unique_ptr<PullbackExecution>
createNoTapePullback(VernonRuntimeContext &context, std::shared_ptr<const Signature> signature,
                     std::shared_ptr<OwnedPipeline> backward, std::shared_ptr<const BindingSpecPlan> bindingSpecs,
                     VernonLaunchSize grid, DeviceValues retainedDevices, HostValues retainedHosts) {
    auto pullback =
        std::make_unique<NoTapePullback>(context, std::move(signature), std::move(backward), std::move(bindingSpecs),
                                         grid, std::move(retainedDevices), std::move(retainedHosts));
    return pullback->valid() ? std::move(pullback) : nullptr;
}

std::unique_ptr<PullbackExecution> createTapePullback(
    VernonRuntimeContext &context, std::shared_ptr<const Signature> signature, std::shared_ptr<OwnedPipeline> forward,
    std::shared_ptr<OwnedPipeline> backward, std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
    std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs, VernonLaunchSize grid, DeviceValues retainedDevices,
    HostValues retainedHosts, size_t staticTapeBytesHint, PlanningPolicy planningPolicy) {
    auto pullback = std::make_unique<TapePullback>(context, std::move(signature), std::move(forward),
                                                   std::move(backward), std::move(forwardBindingSpecs),
                                                   std::move(backwardBindingSpecs), grid, std::move(retainedDevices),
                                                   std::move(retainedHosts), staticTapeBytesHint, planningPolicy);
    return pullback->valid() ? std::move(pullback) : nullptr;
}

} // namespace vernon::runtime::ad::gpu
