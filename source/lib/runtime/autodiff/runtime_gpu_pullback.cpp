#include "runtime_gpu_pullback.h"

#include "execution_graph/execution_graph_internal.h"
#include "rhi/rhi_internal.h"
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

bool isMemoryBudgetFailure(const std::string &error) {
    return error.find("budget") != std::string::npos || error.find("reserve") != std::string::npos;
}

bool flushForMemoryRetry(VernonRuntimeContext &context, execution::detail::RhiCommandPlanSink &sink) {
    if (sink.flush() != VERNON_RHI_STATUS_OK) {
        invocationDiagnostic(context) = "GPU pullback cannot flush pending commands for memory pressure";
        return false;
    }
    invocationDiagnostic(context).clear();
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

class NoTapePullback final : public DevicePullbackExecution {
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
        if (!invocationExtent(grid_, (**backward_).workgroupSize, extent))
            return fail(context_, "GPU pullback invocation extent overflows");
        if (!derivatives.prepare(context_, *signature_, cotangents, gradients, *bindingSpecs_, extent,
                                 3 * sizeof(uint32_t), temporaryLimit, derivativeError))
            return fail(context_, std::move(derivativeError), derivatives.failureStatus);
        return applyPrepared(derivatives, derivativeError, false);
    }

    VernonStatus applyDevice(const VernonAdDeviceValueSet *cotangents, VernonAdDeviceValueSet &gradients,
                             const PullbackApplyOptions &options,
                             execution::detail::RhiCommandPlanSink *sink) override {
        VernonLaunchSize extent{};
        if (!invocationExtent(grid_, (**backward_).workgroupSize, extent))
            return fail(context_, "GPU device pullback invocation extent overflows");
        for (bool retry = sink != nullptr;; retry = false) {
            control_ = {};
            size_t temporaryLimit = 0;
            std::shared_ptr<AutodiffMemoryReservation> applyReservation =
                reserveApplyMemory(context_, options.maximumTemporaryBytes, temporaryLimit);
            if (!applyReservation) {
                if (retry) {
                    if (!flushForMemoryRetry(context_, *sink))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    continue;
                }
                return fail(context_, "GPU device pullback cannot reserve apply-time memory budget");
            }
            PreparedDerivativeValues derivatives;
            std::string derivativeError;
            if (!derivatives.prepareDevice(context_, *signature_, cotangents, gradients, *bindingSpecs_, extent,
                                           3 * sizeof(uint32_t), temporaryLimit, derivativeError)) {
                if (retry && isMemoryBudgetFailure(derivativeError)) {
                    if (!flushForMemoryRetry(context_, *sink))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    continue;
                }
                return fail(context_, std::move(derivativeError), derivatives.failureStatus);
            }
            size_t actualTemporaryBytes = 0;
            const VernonStatus status = applyPrepared(derivatives, derivativeError, true, sink, &actualTemporaryBytes);
            if (status != VERNON_STATUS_OK) {
                if (retry && isMemoryBudgetFailure(invocationDiagnostic(context_))) {
                    if (!flushForMemoryRetry(context_, *sink))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    continue;
                }
                return status;
            }
            if (!applyReservation->shrink(actualTemporaryBytes))
                return fail(context_, "GPU device pullback reservation accounting failed",
                            VERNON_STATUS_INTERNAL_ERROR);
            if (sink) {
                sink->retain(std::make_shared<PreparedDerivativeValues>(std::move(derivatives)));
                sink->retain(std::move(applyReservation));
            }
            return VERNON_STATUS_OK;
        }
    }

    PullbackMemoryUsage memoryUsage() const override { return pullbackMemoryUsage(memory_); }
    PullbackControlPlaneUsage controlPlaneUsage() const override { return control_; }

private:
    VernonStatus applyPrepared(PreparedDerivativeValues &derivatives, std::string &derivativeError,
                               bool deviceGradients, execution::detail::RhiCommandPlanSink *sink = nullptr,
                               size_t *actualTemporaryBytes = nullptr) {
        control_.temporaryAllocationBytes = derivatives.temporaryBytes;

        DeviceValues working;
        BindingPlan bindings;
        std::string bindingError;
        if (!materializeBindingPlan((**backward_).bindingProjection, *bindingSpecs_, retainedDevices_, retainedHosts_,
                                    working, derivatives.devices, bindings, bindingError))
            return fail(context_, std::move(bindingError), VERNON_STATUS_INTERNAL_ERROR);
        std::vector<VernonProgramArgument> arguments;
        arguments.reserve(bindings.size());
        const uint32_t launchData[3]{grid_.x, grid_.y, grid_.z};
        DeviceBuffer launchBuffer(context_, sizeof(launchData));
        InternalBufferView launchView;
        if (!launchBuffer.valid())
            return fail(context_, "cannot allocate GPU pullback launch metadata", VERNON_STATUS_INTERNAL_ERROR);
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
        std::vector<DeviceBufferCopy> publicationCopies;
        const std::vector<DeviceBufferUpload> uploads{{launchBuffer.handle(), 0, launchData, sizeof(launchData)}};
        if (deviceGradients && (!derivatives.devicePublicationCopies(publicationCopies, derivativeError) ||
                                injectFailure(FailureBoundary::Publication)))
            return fail(context_,
                        derivativeError.empty() ? "cannot publish GPU device gradients" : std::move(derivativeError),
                        VERNON_STATUS_INTERNAL_ERROR);
        VernonStatus status =
            deviceGradients
                ? executePipelineCommandDagAndWait(context_, {}, uploads, **backward_, arguments, grid_,
                                                   publicationCopies, execution::detail::CommandNodeKind::Derivative,
                                                   &control_, sink)
                : executePipelineCommandDagAndWait(**backward_, grid_, arguments, uploads,
                                                   execution::detail::CommandNodeKind::Derivative, &control_, sink);
        if (status != VERNON_STATUS_OK)
            return status;
        if (!deviceGradients) {
            if (!derivatives.stageGradients(*signature_, derivativeError))
                return fail(context_, std::move(derivativeError), VERNON_STATUS_INTERNAL_ERROR);
            control_.readbacks += signature_->gradients.size();
            if (!derivatives.publishGradients())
                return fail(context_, "cannot publish GPU pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
        }
        if (sink) {
            sink->retain(std::make_shared<DeviceValues>(std::move(working)));
            sink->retain(std::make_shared<DeviceBuffer>(std::move(launchBuffer)));
        }
        if (actualTemporaryBytes)
            *actualTemporaryBytes = derivatives.temporaryBytes;
        memory_.observeTemporary(derivatives.temporaryBytes);
        return VERNON_STATUS_OK;
    }
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

class TapePullback final : public DevicePullbackExecution {
public:
    TapePullback(VernonRuntimeContext &context, std::shared_ptr<const Signature> signature,
                 std::shared_ptr<OwnedPipeline> forward, std::shared_ptr<OwnedPipeline> backward,
                 std::shared_ptr<const BindingSpecPlan> forwardBindingSpecs,
                 std::shared_ptr<const BindingSpecPlan> backwardBindingSpecs, VernonLaunchSize grid,
                 DeviceValues retainedDevices, HostValues retainedHosts, size_t staticTapeBytesHint,
                 std::shared_ptr<std::atomic<size_t>> learnedTapeStride, PlanningPolicy planningPolicy,
                 bool requiresTapeStatus)
        : context_(context), signature_(std::move(signature)), forward_(std::move(forward)),
          backward_(std::move(backward)), forwardBindingSpecs_(std::move(forwardBindingSpecs)),
          backwardBindingSpecs_(std::move(backwardBindingSpecs)), grid_(grid),
          retainedDevices_(std::move(retainedDevices)), retainedHosts_(std::move(retainedHosts)),
          staticTapeBytesHint_(staticTapeBytesHint), learnedTapeStride_(std::move(learnedTapeStride)),
          planningPolicy_(planningPolicy), requiresTapeStatus_(requiresTapeStatus) {
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
        if (!invocationExtent(grid_, (**backward_).workgroupSize, extent))
            return fail(context_, "GPU pullback invocation extent overflows");
        if (!derivatives.prepare(context_, *signature_, cotangents, gradients, *backwardBindingSpecs_, extent,
                                 3 * sizeof(uint32_t), temporaryLimit, derivativeError))
            return fail(context_, std::move(derivativeError), derivatives.failureStatus);
        return applyPrepared(derivatives, derivativeError, temporaryLimit, false);
    }

    VernonStatus applyDevice(const VernonAdDeviceValueSet *cotangents, VernonAdDeviceValueSet &gradients,
                             const PullbackApplyOptions &options,
                             execution::detail::RhiCommandPlanSink *sink) override {
        VernonLaunchSize extent{};
        if (!invocationExtent(grid_, (**backward_).workgroupSize, extent))
            return fail(context_, "GPU device pullback invocation extent overflows");
        uint64_t retryReadbacks = 0;
        for (bool retry = sink != nullptr;; retry = false) {
            control_ = {};
            size_t temporaryLimit = 0;
            std::shared_ptr<AutodiffMemoryReservation> applyReservation =
                reserveApplyMemory(context_, options.maximumTemporaryBytes, temporaryLimit);
            if (!applyReservation) {
                if (retry) {
                    if (!flushForMemoryRetry(context_, *sink))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    continue;
                }
                return fail(context_, "GPU bounded replay cannot reserve device apply-time memory budget");
            }
            PreparedDerivativeValues derivatives;
            std::string derivativeError;
            if (!derivatives.prepareDevice(context_, *signature_, cotangents, gradients, *backwardBindingSpecs_, extent,
                                           3 * sizeof(uint32_t), temporaryLimit, derivativeError)) {
                if (retry && isMemoryBudgetFailure(derivativeError)) {
                    if (!flushForMemoryRetry(context_, *sink))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    continue;
                }
                return fail(context_, std::move(derivativeError), derivatives.failureStatus);
            }
            size_t actualTemporaryBytes = 0;
            const VernonStatus status =
                applyPrepared(derivatives, derivativeError, temporaryLimit, true, sink, &actualTemporaryBytes);
            if (status != VERNON_STATUS_OK) {
                if (retry && isMemoryBudgetFailure(invocationDiagnostic(context_))) {
                    retryReadbacks = control_.readbacks;
                    if (!flushForMemoryRetry(context_, *sink))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    continue;
                }
                return status;
            }
            if (retryReadbacks > std::numeric_limits<uint64_t>::max() - control_.readbacks)
                return fail(context_, "GPU bounded-replay readback telemetry overflows", VERNON_STATUS_INTERNAL_ERROR);
            control_.readbacks += retryReadbacks;
            if (!applyReservation->shrink(actualTemporaryBytes))
                return fail(context_, "GPU bounded-replay reservation accounting failed", VERNON_STATUS_INTERNAL_ERROR);
            if (sink) {
                sink->retain(std::make_shared<PreparedDerivativeValues>(std::move(derivatives)));
                sink->retain(std::move(applyReservation));
            }
            return VERNON_STATUS_OK;
        }
    }

    PullbackMemoryUsage memoryUsage() const override { return pullbackMemoryUsage(memory_); }
    PullbackControlPlaneUsage controlPlaneUsage() const override { return control_; }

private:
    VernonStatus applyPrepared(PreparedDerivativeValues &derivatives, std::string &derivativeError,
                               size_t temporaryLimit, bool deviceGradients,
                               execution::detail::RhiCommandPlanSink *sink = nullptr,
                               size_t *actualTemporaryBytes = nullptr) {
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
                {retained.buffer.handle(), inserted->second.buffer.handle(), 0, 0, retained.buffer.size()});
        }
        BindingPlan forwardBindings;
        BindingPlan backwardBindings;
        std::string bindingError;
        if (!materializeBindingPlan((**forward_).bindingProjection, *forwardBindingSpecs_, retainedDevices_,
                                    retainedHosts_, working, derivatives.devices, forwardBindings, bindingError) ||
            !materializeBindingPlan((**backward_).bindingProjection, *backwardBindingSpecs_, retainedDevices_,
                                    retainedHosts_, working, derivatives.devices, backwardBindings, bindingError))
            return fail(context_, std::move(bindingError), VERNON_STATUS_INTERNAL_ERROR);

        size_t workgroupVolume = 1;
        const VernonLaunchSize forwardWorkgroup = (**forward_).workgroupSize;
        for (uint32_t extent : {forwardWorkgroup.x, forwardWorkgroup.y, forwardWorkgroup.z})
            if (!checkedMultiply(workgroupVolume, static_cast<size_t>(extent), workgroupVolume))
                return fail(context_, "GPU bounded-replay workgroup volume overflows", VERNON_STATUS_INTERNAL_ERROR);
        constexpr size_t statusBytes = sizeof(BatchSummary);
        size_t tapeStride = std::max(staticTapeBytesHint_, learnedTapeStride_->load(std::memory_order_relaxed));
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
                             statusBytes, batchBudget, requiresTapeStatus_))
            return fail(context_, "GPU bounded-replay temporary memory exceeds the apply-time budget");
        size_t batchCapacity = batchBudget.capacity;
        size_t tapeBytes = batchBudget.tapeBytes;
        size_t segmentBytes = batchBudget.segmentBytes;
        size_t statusBufferBytes = batchBudget.statusBytes;
        size_t peakTemporaryBytes = batchBudget.memory.peakTemporaryBytes;

        if (!findTape((**forward_).bindingProjection) || !findTape((**backward_).bindingProjection) ||
            !findReplaySegment((**forward_).bindingProjection) || !findReplaySegment((**backward_).bindingProjection))
            return fail(context_, "GPU Tape profile has no bounded-replay Tape and segment resources",
                        VERNON_STATUS_INTERNAL_ERROR);

        DeviceBuffer tape(context_, tapeBytes);
        DeviceBuffer segment(context_, segmentBytes);
        DeviceBuffer replayStatus(context_, statusBufferBytes);
        const uint32_t launchData[3]{grid_.x, grid_.y, grid_.z};
        DeviceBuffer launch(context_, sizeof(launchData));
        if (!tape.valid() || !segment.valid() || !replayStatus.valid() || !launch.valid())
            return fail(context_, "cannot allocate GPU bounded-replay resources", VERNON_STATUS_INTERNAL_ERROR);
        control_.temporaryAllocationBytes += tapeBytes + segmentBytes + statusBufferBytes + sizeof(launchData);

        std::string failedParameter;
        bool pristineWorking = true;
        std::vector<DeviceBufferCopy> pendingCopies = std::move(initialCopies);
        ReverseBatchScheduler scheduler(groupCount, batchCapacity);
        if (!scheduler.valid())
            return fail(context_, "GPU replay scheduler has zero batch capacity", VERNON_STATUS_INTERNAL_ERROR);
        std::vector<DeviceBufferCopy> publicationCopies;
        if (deviceGradients && (!derivatives.devicePublicationCopies(publicationCopies, derivativeError) ||
                                injectFailure(FailureBoundary::Publication)))
            return fail(context_,
                        derivativeError.empty() ? "cannot publish GPU replay device gradients"
                                                : std::move(derivativeError),
                        VERNON_STATUS_INTERNAL_ERROR);
        const bool canChainStaticReplay = vernon::rhi::deviceCommandCapabilities(context_.rhiDevice) &
                                          vernon::rhi::BackendCommandExplicitComputeDependencies;
        if (!requiresTapeStatus_ && canChainStaticReplay) {
            ReverseBatchScheduler staticScheduler(groupCount, batchCapacity);
            execution::detail::RhiCommandExecutionPlan staticPlan;
            bool firstBatch = true;
            while (!staticScheduler.empty()) {
                const BatchRange batch = staticScheduler.current();
                std::vector<Segment> metadata(batch.count);
                for (size_t index = 0; index < batch.count; ++index)
                    if (!initializeBatchSegment(grid_, (**forward_).workgroupSize, batch.begin + index, index,
                                                tapeStride, workgroupVolume, tapeBytes, metadata[index]))
                        return fail(context_, "cannot build static GPU replay segment metadata",
                                    VERNON_STATUS_INTERNAL_ERROR);
                if (!firstBatch && !planReplayRestoreCopies(**forward_, working, retainedDevices_, pendingCopies))
                    return fail(context_, "cannot restore static GPU replay primal shadow",
                                VERNON_STATUS_INTERNAL_ERROR);
                const BatchSummary emptySummary{};
                std::vector<DeviceBufferUpload> uploads{
                    {segment.handle(), 0, metadata.data(), metadata.size() * sizeof(Segment)},
                    {replayStatus.handle(), 0, &emptySummary, sizeof(emptySummary)}};
                if (firstBatch)
                    uploads.push_back({launch.handle(), 0, launchData, sizeof(launchData)});
                std::vector<VernonProgramArgument> arguments;
                ReplayArgumentViews views;
                if (!appendReplayArguments(context_, forwardBindings, tape, tapeBytes, segment, segmentBytes,
                                           replayStatus, statusBufferBytes, launch, sizeof(launchData), views,
                                           arguments, failedParameter))
                    return fail(context_, "cannot bind static GPU replay forward resource '" + failedParameter + "'",
                                VERNON_STATUS_INTERNAL_ERROR);
                execution::detail::RhiCommandExecutionPlan forwardPlan;
                if (const VernonStatus status =
                        buildPipelineCommandPlan(context_, pendingCopies, uploads, **forward_, arguments,
                                                 {static_cast<uint32_t>(batch.count), 1, 1}, {},
                                                 execution::detail::CommandNodeKind::Replay, forwardPlan);
                    status != VERNON_STATUS_OK)
                    return status;
                std::string compositionError;
                if (!execution::detail::appendRhiCommandExecutionPlan(staticPlan, std::move(forwardPlan), true,
                                                                      compositionError))
                    return fail(context_, std::move(compositionError), VERNON_STATUS_INTERNAL_ERROR);
                arguments.clear();
                views = {};
                if (!appendReplayArguments(context_, backwardBindings, tape, tapeBytes, segment, segmentBytes,
                                           replayStatus, statusBufferBytes, launch, sizeof(launchData), views,
                                           arguments, failedParameter))
                    return fail(context_, "cannot bind static GPU replay backward resource '" + failedParameter + "'",
                                VERNON_STATUS_INTERNAL_ERROR);
                execution::detail::RhiCommandExecutionPlan backwardPlan;
                const std::vector<DeviceBufferCopy> &copiesAfter =
                    deviceGradients && batch.begin == 0 ? publicationCopies : std::vector<DeviceBufferCopy>{};
                if (const VernonStatus status = buildPipelineCommandPlan(
                        context_, {}, {}, **backward_, arguments, {static_cast<uint32_t>(batch.count), 1, 1},
                        copiesAfter, execution::detail::CommandNodeKind::Derivative, backwardPlan);
                    status != VERNON_STATUS_OK)
                    return status;
                if (!execution::detail::appendRhiCommandExecutionPlan(staticPlan, std::move(backwardPlan), true,
                                                                      compositionError))
                    return fail(context_, std::move(compositionError), VERNON_STATUS_INTERNAL_ERROR);
                pendingCopies.clear();
                firstBatch = false;
                staticScheduler.commit();
            }
            const VernonStatus staticStatus = executeCommandPlanAndWait(context_, staticPlan, &control_, sink);
            if (staticStatus != VERNON_STATUS_OK)
                return staticStatus;
            if (sink) {
                sink->retain(std::make_shared<DeviceValues>(std::move(working)));
                sink->retain(std::make_shared<DeviceBuffer>(std::move(tape)));
                sink->retain(std::make_shared<DeviceBuffer>(std::move(segment)));
                sink->retain(std::make_shared<DeviceBuffer>(std::move(replayStatus)));
                sink->retain(std::make_shared<DeviceBuffer>(std::move(launch)));
            } else if (!deviceGradients) {
                if (!derivatives.stageGradients(*signature_, derivativeError))
                    return fail(context_, std::move(derivativeError), VERNON_STATUS_INTERNAL_ERROR);
                control_.readbacks += signature_->gradients.size();
                if (!derivatives.publishGradients())
                    return fail(context_, "cannot publish static GPU pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
            }
            if (actualTemporaryBytes)
                *actualTemporaryBytes = peakTemporaryBytes;
            memory_.observeTemporary(peakTemporaryBytes);
            return VERNON_STATUS_OK;
        }
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
                    if (!initializeBatchSegment(grid_, (**forward_).workgroupSize, batchBegin + index, index,
                                                tapeStride, workgroupVolume, tapeBytes, metadata[index]))
                        return fail(context_, "cannot initialize GPU replay segment", VERNON_STATUS_INTERNAL_ERROR);
                const BatchSummary emptySummary{};
                const std::vector<DeviceBufferUpload> uploads{
                    {segment.handle(), 0, metadata.data(), metadata.size() * sizeof(Segment)},
                    {replayStatus.handle(), 0, &emptySummary, sizeof(emptySummary)},
                    {launch.handle(), 0, launchData, sizeof(launchData)}};
                std::vector<VernonProgramArgument> arguments;
                ReplayArgumentViews forwardViews;
                if (!appendReplayArguments(context_, forwardBindings, tape, tapeBytes, segment, segmentBytes,
                                           replayStatus, statusBufferBytes, launch, sizeof(launchData), forwardViews,
                                           arguments, failedParameter))
                    return fail(context_, "cannot bind GPU replay forward resource '" + failedParameter + "'",
                                VERNON_STATUS_INTERNAL_ERROR);
                BatchSummary summary{};
                struct StatusContext {
                    const DeviceBuffer &buffer;
                    BatchSummary &summary;
                } statusContext{replayStatus, summary};
                const auto readStatus = [](void *opaque) {
                    auto &state = *static_cast<StatusContext *>(opaque);
                    return state.buffer.download(0, &state.summary, sizeof(state.summary))
                               ? VERNON_RHI_STATUS_OK
                               : VERNON_RHI_STATUS_INTERNAL_ERROR;
                };
                VernonStatus status = executePipelineStatusCommandDagAndWait(
                    context_, pendingCopies, **forward_, arguments, {static_cast<uint32_t>(batchGroups), 1, 1}, uploads,
                    replayStatus.handle(), 0, sizeof(summary), readStatus, &statusContext,
                    execution::detail::CommandNodeKind::Replay, &control_, sink);
                if (status != VERNON_STATUS_OK)
                    return status;
                pendingCopies.clear();
                size_t batchLanes = 0;
                if (!checkedMultiply(batchGroups, workgroupVolume, batchLanes))
                    return fail(context_, "GPU replay batch lane count overflows", VERNON_STATUS_INTERNAL_ERROR);
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
                size_t learnedStride = learnedTapeStride_->load(std::memory_order_relaxed);
                while (learnedStride < requiredStride &&
                       !learnedTapeStride_->compare_exchange_weak(learnedStride, requiredStride,
                                                                  std::memory_order_relaxed)) {
                }
                size_t requiredGroupBytes = 0;
                BatchBudget replacementBudget;
                if (!checkedMultiply(requiredStride, workgroupVolume, requiredGroupBytes) ||
                    !planBatchBudget(planningPolicy_, derivatives.temporaryBytes, temporaryLimit, groupCount,
                                     requiredGroupBytes, statusBytes, replacementBudget, requiresTapeStatus_))
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

            std::vector<VernonProgramArgument> arguments;
            ReplayArgumentViews backwardViews;
            if (!appendReplayArguments(context_, backwardBindings, tape, tapeBytes, segment, segmentBytes, replayStatus,
                                       statusBufferBytes, launch, sizeof(launchData), backwardViews, arguments,
                                       failedParameter))
                return fail(context_, "cannot bind GPU replay backward resource '" + failedParameter + "'",
                            VERNON_STATUS_INTERNAL_ERROR);
            const VernonLaunchSize backwardGrid{static_cast<uint32_t>(batchGroups), 1, 1};
            VernonStatus status =
                deviceGradients && batch.begin == 0
                    ? executePipelineCommandDagAndWait(context_, {}, {}, **backward_, arguments, backwardGrid,
                                                       publicationCopies,
                                                       execution::detail::CommandNodeKind::Derivative, &control_, sink)
                    : executePipelineCommandDagAndWait(**backward_, backwardGrid, arguments, {},
                                                       execution::detail::CommandNodeKind::Derivative, &control_, sink);
            if (status != VERNON_STATUS_OK)
                return status;
            scheduler.commit();
        }

        if (!deviceGradients) {
            if (!derivatives.stageGradients(*signature_, derivativeError))
                return fail(context_, std::move(derivativeError), VERNON_STATUS_INTERNAL_ERROR);
            control_.readbacks += signature_->gradients.size();
            if (!derivatives.publishGradients())
                return fail(context_, "cannot publish GPU pullback gradients", VERNON_STATUS_INTERNAL_ERROR);
        }
        if (sink) {
            sink->retain(std::make_shared<DeviceValues>(std::move(working)));
            sink->retain(std::make_shared<DeviceBuffer>(std::move(tape)));
            sink->retain(std::make_shared<DeviceBuffer>(std::move(segment)));
            sink->retain(std::make_shared<DeviceBuffer>(std::move(replayStatus)));
            sink->retain(std::make_shared<DeviceBuffer>(std::move(launch)));
        }
        if (actualTemporaryBytes)
            *actualTemporaryBytes = peakTemporaryBytes;
        memory_.observeTemporary(peakTemporaryBytes);
        return VERNON_STATUS_OK;
    }
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
    std::shared_ptr<std::atomic<size_t>> learnedTapeStride_;
    PlanningPolicy planningPolicy_{};
    bool requiresTapeStatus_{true};
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
    HostValues retainedHosts, size_t staticTapeBytesHint, std::shared_ptr<std::atomic<size_t>> learnedTapeStride,
    PlanningPolicy planningPolicy, bool requiresTapeStatus) {
    auto pullback = std::make_unique<TapePullback>(
        context, std::move(signature), std::move(forward), std::move(backward), std::move(forwardBindingSpecs),
        std::move(backwardBindingSpecs), grid, std::move(retainedDevices), std::move(retainedHosts),
        staticTapeBytesHint, std::move(learnedTapeStride), planningPolicy, requiresTapeStatus);
    return pullback->valid() ? std::move(pullback) : nullptr;
}

} // namespace vernon::runtime::ad::gpu
