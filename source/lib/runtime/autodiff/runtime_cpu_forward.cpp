#include "runtime_cpu_forward.h"

#include "host_effect_transaction.h"
#include "host_tape_allocator.h"
#include "runtime/backend_cpu.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_state.h"
#include "runtime/tensor_bridge.h"

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

VernonStatus runCpuForward(VernonRuntimeContext &context, const HostProfileLayout &layout, const Signature &signature,
                           VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                           const std::unordered_set<std::string> &requiredPrimalTensorOwners,
                           std::unique_ptr<PullbackExecution> &pullback, CpuForwardInvoke invoke,
                           CpuForwardFinish finish, std::optional<size_t> groupLinear, bool compactAllGroups) {
    const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
    if (!validateDispatchContract(layout.dispatchContract, grid, layout.workgroup, invocationDiagnostic(context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonLaunchSize extent{};
    size_t invocationCount = 0;
    if (!invocationExtent(computeGrid, {layout.workgroup[0], layout.workgroup[1], layout.workgroup[2]}, extent) ||
        !carrierCount(extent, invocationCount))
        return fail(context, "native CPU autodiff launch size overflows");
    if (inputs.value_count != signature.inputs.size() || outputs.value_count != 0 || !layout.results.empty())
        return fail(context, "autodiff forward values do not match the host profile");
    std::vector<uint8_t> arguments(layout.argumentsSize);
    std::vector<StorageRange> storageRanges;
    std::vector<StagedTensorView> tensorViews;
    HostEffectTransaction transaction(0);
    if (VernonStatus status = stageForwardInputs(context, layout, inputs, arguments, transaction, storageRanges,
                                                 tensorViews, groupLinear.has_value());
        status != VERNON_STATUS_OK)
        return status;
    Signature runtimeSignature = signature;
    if (!materializeRuntimeSignature(runtimeSignature, inputs))
        return fail(context, "cannot materialize native CPU autodiff runtime signature");
    RuntimeTensorShapes tensorShapes;
    RetainedPrimalTensorViews retainedTensorViews;
    tensorShapes.reserve(tensorViews.size());
    for (const StagedTensorView &view : tensorViews) {
        const std::string owner = tensorOwnerName(*view.argument);
        const auto [retained, inserted] = tensorShapes.emplace(owner, view.shape);
        if (!inserted && retained->second != view.shape)
            return fail(context, "native CPU autodiff retained incompatible TensorView shapes for one owner");
        if (requiredPrimalTensorOwners.find(owner) != requiredPrimalTensorOwners.end())
            retainedTensorViews.emplace(owner, RetainedPrimalTensorView{view.shape, view.packed});
    }
    CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context);
    std::mutex invocationFailureMutex;
    std::string invocationFailure;
    const auto recordInvocationFailure = [&](std::string diagnostic) {
        std::lock_guard lock(invocationFailureMutex);
        if (invocationFailure.empty())
            invocationFailure = std::move(diagnostic);
    };
    std::vector<ForwardInvocationFrame> invocationFrames;
    std::vector<uint8_t> argumentFrames;
    std::vector<uint8_t> resultFrames;
    std::vector<const void *> laneArguments;
    std::vector<void *> laneResults;
    size_t frameCount = invocationCount;
    try {
        size_t workgroupVolume = 1;
        for (uint32_t dimension : layout.workgroup) {
            if (workgroupVolume > SIZE_MAX / dimension)
                return fail(context, "CPU autodiff workgroup size overflows");
            workgroupVolume *= dimension;
        }
        frameCount = groupLinear || compactAllGroups ? workgroupVolume : invocationCount;
        size_t argumentFrameBytes = 0;
        size_t resultFrameBytes = 0;
        if (!checkedAddBytes(argumentFrameBytes, frameCount, layout.argumentsSize) ||
            !checkedAddBytes(resultFrameBytes, frameCount, layout.resultsSize))
            return fail(context, "CPU autodiff lane frame size overflows");
        invocationFrames.resize(frameCount);
        argumentFrames.resize(argumentFrameBytes);
        resultFrames.resize(resultFrameBytes);
        laneArguments.resize(frameCount);
        laneResults.resize(frameCount);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate persistent CPU autodiff lane frames");
    }
    const CpuRangeCallback executeRange = [&](VernonCpuRangeV1 &range) {
        try {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                const size_t frameIndex = groupLinear || compactAllGroups ? localLinear : invocationIndex;
                ForwardInvocationFrame &frame = invocationFrames[frameIndex];
                if (!frame.arguments && layout.argumentsSize) {
                    frame.arguments = argumentFrames.data() + frameIndex * layout.argumentsSize;
                    std::memcpy(frame.arguments, arguments.data(), layout.argumentsSize);
                }
                if (!frame.results && layout.resultsSize)
                    frame.results = resultFrames.data() + frameIndex * layout.resultsSize;
                const size_t tableIndex = groupLinear || compactAllGroups ? localLinear : invocationIndex;
                laneArguments[tableIndex] = frame.arguments;
                laneResults[tableIndex] = frame.results;
            }
        } catch (const std::bad_alloc &) {
            recordInvocationFailure("cannot allocate active CPU autodiff lane frames");
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        const size_t firstInvocation = cpuRangeCoordinates(range, range.lane_begin).linearIndex;
        const size_t firstTableIndex = groupLinear || compactAllGroups ? range.lane_begin : firstInvocation;
        range.arguments = laneArguments[firstTableIndex];
        range.arguments_size = layout.argumentsSize;
        range.results = laneResults[firstInvocation];
        range.results_size = layout.resultsSize;
        range.lane_arguments = laneArguments.data();
        range.lane_results = laneResults.data();
        range.lane_table_count = laneArguments.size();
        std::string diagnostic;
        VernonStatus status = invoke(range, invocationFrames, diagnostic);
        if (status != VERNON_STATUS_OK && !diagnostic.empty())
            recordInvocationFailure(std::move(diagnostic));
        if (status == VERNON_STATUS_OK && range.outcome == VERNON_CPU_RANGE_COMPLETE_V1) {
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                const size_t frameIndex = groupLinear || compactAllGroups ? localLinear : invocationIndex;
                const size_t tableIndex = frameIndex;
                laneArguments[tableIndex] = nullptr;
                laneResults[tableIndex] = nullptr;
                invocationFrames[frameIndex] = {};
            }
        }
        return status;
    };
    const VernonStatus dispatchStatus =
        groupLinear        ? scheduler.dispatchGroupInline(grid, layout.workgroup, *groupLinear, executeRange)
        : compactAllGroups ? scheduler.dispatchInline(grid, layout.workgroup, executeRange)
        : invocationCount <= kCpuAdInlineInvocationLimit
            ? scheduler.dispatchInline(grid, layout.workgroup, executeRange)
            : scheduler.dispatch(grid, layout.workgroup, executeRange);
    if (dispatchStatus != VERNON_STATUS_OK)
        return fail(context,
                    invocationFailure.empty()
                        ? (scheduler.lastDiagnostic().empty() ? "CPU autodiff forward workgroup execution failed"
                                                              : scheduler.lastDiagnostic())
                        : invocationFailure,
                    dispatchStatus);
    std::unique_ptr<PullbackExecution> pendingPullback;
    if (VernonStatus status = finish(invocationCount, runtimeSignature, std::move(tensorShapes),
                                     std::move(retainedTensorViews), pendingPullback);
        status != VERNON_STATUS_OK)
        return status;
    if (!pendingPullback)
        return fail(context, "native CPU autodiff did not create a pullback", VERNON_STATUS_INTERNAL_ERROR);
    if (VernonStatus status = flushStagedTensorViews(context, tensorViews); status != VERNON_STATUS_OK)
        return status;
    if (!transaction.commit(nullptr))
        return fail(context, "autodiff effect transaction was already committed");
    pullback = std::move(pendingPullback);
    return VERNON_STATUS_OK;
}

std::unordered_set<std::string> requiredPrimalTensorOwners(const HostProfileLayout &backwardLayout) {
    std::unordered_set<std::string> owners;
    for (const HostArgument &argument : backwardLayout.arguments)
        if (isPrimalSource(argument) && argument.tensorView)
            owners.insert(argument.name.substr(7));
    return owners;
}

VernonStatus retainRequiredPrimalLeaves(VernonRuntimeContext &context, const HostProfileLayout &backwardLayout,
                                        const VernonAdValueSet &inputs, RetainedPrimalLeaves &retainedPrimals) {
    for (const HostArgument &argument : backwardLayout.arguments) {
        if (!isPrimalSource(argument) || argument.tensorView)
            continue;
        for (const HostFrameLeaf &leaf : argument.leaves) {
            std::string path = leaf.value.path;
            if (path.rfind("primal.", 0) != 0)
                return fail(context, "CPU primal reflection path is invalid");
            path.erase(0, 7);
            const VernonAdValue *value = findValue(inputs, path);
            if (!value || value->size != leaf.value.byteSize || value->dtype != leaf.value.dtype)
                return fail(context, "CPU required primal does not match the forward inputs");
            retainedPrimals[path] = std::vector<uint8_t>(static_cast<const uint8_t *>(value->data),
                                                         static_cast<const uint8_t *>(value->data) + value->size);
        }
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime::ad::cpu
