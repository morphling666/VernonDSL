#include "runtime_cpu_replay.h"

#include "host_effect_transaction.h"
#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"
#include "runtime_autodiff_memory_usage.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
#include "host_tape_test_hooks.h"
#endif

namespace vernon::runtime::ad::cpu {

class CpuTapeRangeExecutor {
public:
    struct SegmentScratch {
        std::vector<uint8_t> argumentFrames;
        std::vector<const void *> laneArguments;
        std::shared_ptr<HostStaticTapeBatch> recyclableTape;
        std::shared_ptr<HostTapeDispatchBudget> tapeBudget;
    };

    CpuTapeRangeExecutor(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                         std::shared_ptr<const CpuResidualPlan> plan, std::shared_ptr<HostTapeMemoryPolicy> policy)
        : context_(context), program_(std::move(program)), plan_(std::move(plan)), policy_(std::move(policy)) {}

    std::shared_ptr<HostStaticTapeBatch> allocate(size_t laneCount,
                                                  const std::shared_ptr<HostTapeDispatchBudget> &budget) const {
        return HostStaticTapeBatch::create(laneCount, plan_->staticTapeStride, policy_->invocationLimit(), policy_,
                                           budget);
    }

    VernonStatus tryAllocateWholeDispatch(size_t invocationCount, std::shared_ptr<HostStaticTapeBatch> &tape,
                                          std::shared_ptr<HostTapeDispatchBudget> &budget, bool &admitted) const {
        admitted = false;
        size_t constructionBytes = 0;
        if (!hostStaticTapeBatchPureStaticBytes(invocationCount, plan_->staticTapeStride, constructionBytes))
            return fail(context_, "CPU whole-dispatch Tape size overflows");
        if (constructionBytes > policy_->contextLimit())
            return VERNON_STATUS_OK;
        budget = HostTapeDispatchBudget::reserve(policy_, constructionBytes);
        if (!budget)
            return VERNON_STATUS_OK;
        tape = allocate(invocationCount, budget);
        if (!tape)
            return fail(context_, "CPU whole-dispatch Tape allocation disagrees with its admitted static plan",
                        VERNON_STATUS_INTERNAL_ERROR);
        admitted = true;
        return VERNON_STATUS_OK;
    }

    template <typename ArgumentsAt, typename TapeIndexAt>
    VernonStatus executeRange(VernonCpuRangeV1 &range, HostStaticTapeBatch &tape, ArgumentsAt &&argumentsAt,
                              TapeIndexAt &&tapeIndexAt, std::string &diagnostic) const {
        const HostProfileLayout &layout = program_->forwardLayout;
        for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
            const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
            uint8_t *arguments = argumentsAt(localLinear, coordinates);
            const size_t tapeIndex = tapeIndexAt(localLinear, coordinates);
            for (const HostArgument &argument : layout.arguments) {
                if (argument.builtin.empty())
                    continue;
                if (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN) {
                    VernonAdTapeAllocator *descriptor = tape.descriptor(tapeIndex);
                    if (!descriptor) {
                        diagnostic = "CPU Tape dispatch lane is unavailable";
                        return VERNON_STATUS_INTERNAL_ERROR;
                    }
                    std::memcpy(arguments + argument.offset, &descriptor, sizeof(VernonAdTapeAllocator *));
                } else if (!writeInvocationBuiltin(argument, coordinates, arguments)) {
                    diagnostic = "CPU Tape dispatch has an unsupported builtin ABI";
                    return VERNON_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
        const VernonStatus status = program_->forward->entry(&invocation);
        if (status != VERNON_STATUS_OK || range.outcome == VERNON_CPU_RANGE_YIELDED_V1)
            return status;
        if (range.outcome != VERNON_CPU_RANGE_COMPLETE_V1 ||
            range.completed_lanes != range.lane_end - range.lane_begin) {
            diagnostic = "CPU Tape forward lanes completed at different barrier phases";
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
            const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
            const size_t tapeIndex = tapeIndexAt(localLinear, coordinates);
            VernonAdTapeAllocator *descriptor = tape.descriptor(tapeIndex);
            if (!descriptor) {
                diagnostic = "CPU Tape dispatch has no allocator descriptor";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            if (descriptor->status != VERNON_AD_TAPE_ALLOCATOR_OK) {
                diagnostic = std::string("autodiff tape allocator ") + allocatorFailure(descriptor->status) +
                             " after requiring " + std::to_string(descriptor->required_bytes) + " bytes under the " +
                             std::to_string(policy_->contextLimit()) + "-byte context limit";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            if (!tape.rootRegion(tapeIndex)) {
                diagnostic = "CPU Tape dispatch did not seal its Tape";
                return VERNON_STATUS_INTERNAL_ERROR;
            }
        }
        return VERNON_STATUS_OK;
    }

    VernonStatus finish(HostStaticTapeBatch &tape, HostTapeDispatchBudget &budget, std::string_view scope,
                        bool retainConstructionStorage = false) const {
        if (!tape.compact(retainConstructionStorage))
            return fail(context_, "CPU " + std::string(scope) + " Tape could not be compacted");
        budget.commit();
        return VERNON_STATUS_OK;
    }

    VernonStatus replayPreparedGroup(VernonLaunchSize computeGrid, size_t groupLinear,
                                     const std::vector<uint8_t> &baseArguments,
                                     std::shared_ptr<HostStaticTapeBatch> &tapeBatch, SegmentScratch &scratch,
                                     const PullbackApplyOptions &options) const {
        const HostProfileLayout &layout = program_->forwardLayout;
        size_t workgroupVolume = 1;
        for (uint32_t dimension : layout.workgroup) {
            if (workgroupVolume > SIZE_MAX / dimension)
                return fail(context_, "CPU bounded replay workgroup size overflows");
            workgroupVolume *= dimension;
        }
        if (scratch.recyclableTape &&
            scratch.recyclableTape->constructionBytes() <= options.maximumReusableConstructionBytes &&
            scratch.recyclableTape->resetRecyclableConstruction()) {
            tapeBatch = std::move(scratch.recyclableTape);
        } else {
            scratch.recyclableTape.reset();
            scratch.tapeBudget.reset();
            size_t maximumTapeBytes = 0;
            if (!checkedAddBytes(maximumTapeBytes, workgroupVolume, policy_->invocationLimit()))
                return fail(context_, "CPU bounded replay segment Tape upper bound overflows",
                            VERNON_STATUS_INTERNAL_ERROR);
            const size_t applyLimit =
                std::min({maximumTapeBytes, policy_->contextLimit(), options.maximumTemporaryBytes});
            scratch.tapeBudget = HostTapeDispatchBudget::reserve(policy_, applyLimit);
            if (!scratch.tapeBudget)
                return fail(context_, "CPU bounded replay cannot reserve one complete-workgroup Tape segment",
                            VERNON_STATUS_INTERNAL_ERROR);
            tapeBatch = allocate(workgroupVolume, scratch.tapeBudget);
            if (!tapeBatch)
                return fail(context_, "CPU bounded replay workgroup Tape exceeds the context budget",
                            VERNON_STATUS_INTERNAL_ERROR);
        }
        size_t argumentFrameBytes = 0;
        if (!checkedAddBytes(argumentFrameBytes, workgroupVolume, layout.argumentsSize))
            return fail(context_, "CPU bounded replay argument frames overflow");
        try {
            scratch.argumentFrames.resize(argumentFrameBytes);
            scratch.laneArguments.assign(workgroupVolume, nullptr);
        } catch (const std::bad_alloc &) {
            return fail(context_, "cannot allocate CPU bounded replay segment scratch", VERNON_STATUS_INTERNAL_ERROR);
        } catch (const std::length_error &) {
            return fail(context_, "CPU bounded replay segment scratch exceeds container limits",
                        VERNON_STATUS_INTERNAL_ERROR);
        }
        std::string invocationFailure;
        const CpuRangeCallback execute = [&](VernonCpuRangeV1 &range) {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                if (!scratch.laneArguments[localLinear]) {
                    uint8_t *arguments = scratch.argumentFrames.data() + localLinear * layout.argumentsSize;
                    std::memcpy(arguments, baseArguments.data(), layout.argumentsSize);
                    scratch.laneArguments[localLinear] = arguments;
                }
            }
            range.arguments = scratch.laneArguments[range.lane_begin];
            range.arguments_size = layout.argumentsSize;
            range.lane_arguments = scratch.laneArguments.data();
            range.lane_table_count = scratch.laneArguments.size();
            return executeRange(
                range, *tapeBatch,
                [&](size_t localLinear, const CpuLaneCoordinates &) {
                    return scratch.argumentFrames.data() + localLinear * layout.argumentsSize;
                },
                [](size_t localLinear, const CpuLaneCoordinates &) { return localLinear; }, invocationFailure);
        };
        CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context_);
        const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
        const VernonStatus status = scheduler.dispatchGroupInline(grid, layout.workgroup, groupLinear, execute);
        if (status != VERNON_STATUS_OK)
            return fail(context_, invocationFailure.empty() ? scheduler.lastDiagnostic() : invocationFailure, status);
        const bool retainConstruction = tapeBatch->constructionBytes() <= options.maximumReusableConstructionBytes;
        return finish(*tapeBatch, *scratch.tapeBudget, "bounded replay segment", retainConstruction);
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::shared_ptr<const CpuResidualPlan> plan_;
    std::shared_ptr<HostTapeMemoryPolicy> policy_;
};

class SegmentReplayCpuPullback final : public PullbackExecution {
public:
    SegmentReplayCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                             std::shared_ptr<const CpuResidualPlan> plan,
                             std::shared_ptr<HostTapeMemoryPolicy> tapePolicy, Signature signature,
                             VernonLaunchSize computeGrid, OwnedAdValueSet retainedInputs)
        : context_(context), program_(std::move(program)), plan_(std::move(plan)), tapePolicy_(std::move(tapePolicy)),
          signature_(std::move(signature)), computeGrid_(computeGrid), retainedInputs_(std::move(retainedInputs)) {}

    PullbackMemoryUsage memoryUsage() const override {
        MemoryAccounting memory;
        memory.retainedAllocationBytes = retainedInputs_.retainedBytes();
        memory.peakTemporaryBytes = peakTemporaryBytes_.load(std::memory_order_relaxed);
        return pullbackMemoryUsage(memory);
    }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &options) override {
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> stagedGradients;
        if (VernonStatus status =
                prepareGradientDestinations(context_, signature_, gradients, destinations, stagedGradients);
            status != VERNON_STATUS_OK)
            return status;
        OwnedAdValueSet replayInputs(retainedInputs_.set);
        std::vector<uint8_t> arguments(program_->forwardLayout.argumentsSize);
        std::vector<StorageRange> storageRanges;
        std::vector<StagedTensorView> tensorViews;
        HostEffectTransaction transaction(0);
        if (VernonStatus status = stageForwardInputs(context_, program_->forwardLayout, replayInputs.set, arguments,
                                                     transaction, storageRanges, tensorViews);
            status != VERNON_STATUS_OK)
            return status;
        RuntimeTensorShapes tensorShapes;
        for (const StagedTensorView &view : tensorViews) {
            const std::string owner = tensorOwnerName(*view.argument);
            tensorShapes.emplace(owner, view.shape);
        }
        RetainedPrimalLeaves retainedPrimals;
        if (VernonStatus status =
                retainRequiredPrimalLeaves(context_, program_->backwardLayout, replayInputs.set, retainedPrimals);
            status != VERNON_STATUS_OK)
            return status;
        size_t groupCount = 1;
        for (uint32_t dimension : {computeGrid_.x, computeGrid_.y, computeGrid_.z}) {
            if (groupCount > SIZE_MAX / dimension)
                return fail(context_, "CPU bounded replay group count overflows");
            groupCount *= dimension;
        }
        CpuTapeRangeExecutor executor(context_, program_, plan_, tapePolicy_);
        CpuTapeRangeExecutor::SegmentScratch scratch;
        bool replayShadowIsPristine = true;
        for (size_t group = groupCount; group-- > 0;) {
            if (!replayShadowIsPristine)
                restoreReplayReadWriteShadows(tensorViews);
            replayShadowIsPristine = false;
            std::shared_ptr<HostStaticTapeBatch> tapeBatch;
            if (VernonStatus status =
                    executor.replayPreparedGroup(computeGrid_, group, arguments, tapeBatch, scratch, options);
                status != VERNON_STATUS_OK)
                return status;
            size_t observed = peakTemporaryBytes_.load(std::memory_order_relaxed);
            const size_t allocated = scratch.tapeBudget->usage().peakBytes;
            while (observed < allocated &&
                   !peakTemporaryBytes_.compare_exchange_weak(observed, allocated, std::memory_order_relaxed)) {
            }
            RetainedPrimalTensorViewRefs retainedTensorViews;
            for (StagedTensorView &view : tensorViews) {
                const std::string owner = tensorOwnerName(*view.argument);
                if (program_->requiredPrimalTensorOwners.find(owner) != program_->requiredPrimalTensorOwners.end())
                    retainedTensorViews.emplace(owner, RetainedPrimalTensorViewRef{&view.shape, view.packed.data()});
            }
            if (VernonStatus status = applyCpuBackwardSegment(context_, program_, signature_, tensorShapes,
                                                              retainedPrimals, retainedTensorViews, computeGrid_,
                                                              tapeBatch, group, cotangents, stagedGradients);
                status != VERNON_STATUS_OK)
                return status;
            std::shared_ptr<HostStaticTapeBatch> completedTape = std::move(tapeBatch);
            if (completedTape && completedTape->constructionBytes() <= options.maximumReusableConstructionBytes &&
                completedTape->markConstructionRecyclable())
                scratch.recyclableTape = std::move(completedTape);
        }
        commitGradientDestinations(destinations, stagedGradients);
        return VERNON_STATUS_OK;
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::shared_ptr<const CpuResidualPlan> plan_;
    std::shared_ptr<HostTapeMemoryPolicy> tapePolicy_;
    Signature signature_;
    VernonLaunchSize computeGrid_{};
    OwnedAdValueSet retainedInputs_;
    std::atomic<size_t> peakTemporaryBytes_{};
};

class TapedStructuredCpuExecutable final : public Executable {
public:
    TapedStructuredCpuExecutable(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                                 std::shared_ptr<const CpuResidualPlan> plan)
        : context_(context), program_(std::move(program)), plan_(std::move(plan)),
          tapePolicy_(context.autodiffMemoryPolicy) {}

    const Signature &signature() const override { return program_->signature; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize computeGrid,
                         const VernonAdValueSet &inputs, VernonAdValueSet *outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        if (target.externalEncoder() || !outputs)
            return fail(context_, "CPU autodiff does not accept a GPU command encoder");
        const HostProfileLayout &forwardLayout = program_->forwardLayout;
        VernonLaunchSize extent{};
        size_t invocationCount = 0;
        if (!invocationExtent(computeGrid,
                              {forwardLayout.workgroup[0], forwardLayout.workgroup[1], forwardLayout.workgroup[2]},
                              extent) ||
            !carrierCount(extent, invocationCount))
            return fail(context_, "CPU autodiff launch size overflows");
        if (plan_->permitsWholeDispatchRetention) {
            CpuTapeRangeExecutor executor(context_, program_, plan_, tapePolicy_);
            std::shared_ptr<HostStaticTapeBatch> tapeBatch;
            std::shared_ptr<HostTapeDispatchBudget> budget;
            bool admitted = false;
            if (VernonStatus status = executor.tryAllocateWholeDispatch(invocationCount, tapeBatch, budget, admitted);
                status != VERNON_STATUS_OK)
                return status;
            if (admitted)
                return forwardWithRetainedTape(computeGrid, inputs, *outputs, pullback, std::move(tapeBatch),
                                               std::move(budget));
        }
        OwnedAdValueSet retainedInputs(inputs);
        return runCpuForward(
            context_, program_->primalLayout, program_->signature, computeGrid, inputs, *outputs,
            program_->requiredPrimalTensorOwners, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                    uint8_t *arguments = frames[localLinear].arguments;
                    for (const HostArgument &argument : program_->primalLayout.arguments)
                        if (!argument.builtin.empty() && !writeInvocationBuiltin(argument, coordinates, arguments)) {
                            diagnostic = "CPU bounded replay primal has an unsupported builtin ABI";
                            return VERNON_STATUS_INVALID_ARGUMENT;
                        }
                }
                const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
                return program_->primal->entry(&invocation);
            },
            [&](size_t, Signature runtimeSignature, RuntimeTensorShapes, RetainedPrimalTensorViews,
                std::unique_ptr<PullbackExecution> &pending) {
                pending = std::make_unique<SegmentReplayCpuPullback>(context_, program_, plan_, tapePolicy_,
                                                                     std::move(runtimeSignature), computeGrid,
                                                                     std::move(retainedInputs));
                return VERNON_STATUS_OK;
            },
            std::nullopt, true);
    }

private:
    VernonStatus forwardWithRetainedTape(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs,
                                         VernonAdValueSet &outputs, std::unique_ptr<PullbackExecution> &pullback,
                                         std::shared_ptr<HostStaticTapeBatch> tapeBatch,
                                         std::shared_ptr<HostTapeDispatchBudget> dispatchBudget) {
        CpuTapeRangeExecutor executor(context_, program_, plan_, tapePolicy_);
        RetainedPrimalLeaves retainedPrimals;
        if (VernonStatus status =
                retainRequiredPrimalLeaves(context_, program_->backwardLayout, inputs, retainedPrimals);
            status != VERNON_STATUS_OK)
            return status;
        return runCpuForward(
            context_, program_->forwardLayout, program_->signature, computeGrid, inputs, outputs,
            program_->requiredPrimalTensorOwners, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                return executor.executeRange(
                    range, *tapeBatch,
                    [&](size_t, const CpuLaneCoordinates &coordinates) {
                        return frames[coordinates.linearIndex].arguments;
                    },
                    [](size_t, const CpuLaneCoordinates &coordinates) { return coordinates.linearIndex; }, diagnostic);
            },
            [&](size_t, Signature runtimeSignature, RuntimeTensorShapes tensorShapes,
                RetainedPrimalTensorViews retainedTensorViews, std::unique_ptr<PullbackExecution> &pending) {
                if (VernonStatus status = executor.finish(*tapeBatch, *dispatchBudget, "whole-dispatch");
                    status != VERNON_STATUS_OK)
                    return status;
                pending = createRetainedTapeCpuPullback(context_, program_, std::move(runtimeSignature), computeGrid,
                                                        std::move(tapeBatch), std::move(tensorShapes),
                                                        std::move(retainedPrimals), std::move(retainedTensorViews));
                return VERNON_STATUS_OK;
            });
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::shared_ptr<const CpuResidualPlan> plan_;
    std::shared_ptr<HostTapeMemoryPolicy> tapePolicy_;
};

std::shared_ptr<Executable> createTapedStructuredCpuExecutable(VernonRuntimeContext &context,
                                                               std::shared_ptr<const CpuAutodiffProgram> program,
                                                               std::shared_ptr<const CpuResidualPlan> plan) {
    return std::make_shared<TapedStructuredCpuExecutable>(context, std::move(program), std::move(plan));
}

} // namespace vernon::runtime::ad::cpu
