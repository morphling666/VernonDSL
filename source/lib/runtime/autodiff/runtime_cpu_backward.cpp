#include "runtime_cpu_pullback.h"

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

struct CpuBackwardOwnedState {
    Signature signature;
    RuntimeTensorShapes tensorShapes;
    RetainedPrimalLeaves retainedPrimals;
    RetainedPrimalTensorViews retainedTensorViews;
    RetainedPrimalTensorViewRefs retainedTensorViewRefs;

    CpuBackwardOwnedState(Signature runtimeSignature, RuntimeTensorShapes shapes, RetainedPrimalLeaves primals,
                          RetainedPrimalTensorViews tensorViews)
        : signature(std::move(runtimeSignature)), tensorShapes(std::move(shapes)), retainedPrimals(std::move(primals)),
          retainedTensorViews(std::move(tensorViews)) {
        retainedTensorViewRefs.reserve(retainedTensorViews.size());
        for (auto &[owner, view] : retainedTensorViews)
            retainedTensorViewRefs.emplace(owner, RetainedPrimalTensorViewRef{&view.shape, view.packed.data()});
    }
};

struct CpuBackwardBorrowedState {
    std::reference_wrapper<const Signature> signature;
    std::reference_wrapper<const RuntimeTensorShapes> tensorShapes;
    std::reference_wrapper<const RetainedPrimalLeaves> retainedPrimals;
    std::reference_wrapper<const RetainedPrimalTensorViewRefs> retainedTensorViews;
};

class CpuBackwardExecutor {
public:
    CpuBackwardExecutor(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                        Signature signature, VernonLaunchSize computeGrid,
                        std::shared_ptr<HostStaticTapeBatch> tapeBatch, RuntimeTensorShapes tensorShapes,
                        RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews,
                        std::optional<size_t> groupLinear = std::nullopt)
        : context_(context), program_(std::move(program)),
          state_(std::in_place_type<CpuBackwardOwnedState>, std::move(signature), std::move(tensorShapes),
                 std::move(retainedPrimals), std::move(retainedTensorViews)),
          computeGrid_(computeGrid), tapeBatch_(std::move(tapeBatch)), groupLinear_(groupLinear) {}

    CpuBackwardExecutor(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                        CpuBackwardBorrowedState state, VernonLaunchSize computeGrid,
                        std::shared_ptr<HostStaticTapeBatch> tapeBatch, size_t groupLinear)
        : context_(context), program_(std::move(program)), state_(std::move(state)), computeGrid_(computeGrid),
          tapeBatch_(std::move(tapeBatch)), groupLinear_(groupLinear) {}

    PullbackMemoryUsage memoryUsage() const {
        MemoryAccounting memory;
        if (tapeBatch_) {
            memory.logicalPayloadBytes = tapeBatch_->logicalBytes();
            memory.residentBytes = tapeBatch_->residentBytes();
            memory.allocatedBytes = tapeBatch_->allocatedBytes();
        }
        memory.retainedAllocationBytes = memory.allocatedBytes;
        const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_);
        if (!owned)
            return pullbackMemoryUsage(memory);
        for (const auto &retained : owned->retainedPrimals)
            if (!checkedAddBytes(memory.retainedAllocationBytes, retained.second.capacity(), sizeof(uint8_t))) {
                memory.retainedAllocationBytes = std::numeric_limits<size_t>::max();
                return pullbackMemoryUsage(memory);
            }
        for (const auto &retained : owned->retainedTensorViews)
            if (!checkedAddBytes(memory.retainedAllocationBytes, retained.second.shape.capacity(), sizeof(uint64_t)) ||
                !checkedAddBytes(memory.retainedAllocationBytes, retained.second.packed.capacity(), sizeof(uint8_t))) {
                memory.retainedAllocationBytes = std::numeric_limits<size_t>::max();
                return pullbackMemoryUsage(memory);
            }
        return pullbackMemoryUsage(memory);
    }

    VernonStatus applyInto(const VernonAdValueSet *cotangents, std::vector<std::vector<uint8_t>> &stagedGradients) {
        if (!groupLinear_ || externalStagedGradients_)
            return fail(context_, "CPU segmented pullback accumulation is unavailable", VERNON_STATUS_INTERNAL_ERROR);
        externalStagedGradients_ = &stagedGradients;
        VernonAdValueSet unused{sizeof(VernonAdValueSet), nullptr, 0, {}};
        const VernonStatus status = apply(cotangents, unused);
        externalStagedGradients_ = nullptr;
        return status;
    }

    std::shared_ptr<HostStaticTapeBatch> releaseTapeBatch() { return std::move(tapeBatch_); }

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) {
        const Signature *const signature_ = &runtimeSignature();
        const RuntimeTensorShapes *const tensorShapes_ = &runtimeTensorShapes();
        const RetainedPrimalLeaves *const retainedPrimals_ = &retainedPrimalLeaves();
        const RetainedPrimalTensorViewRefs *const retainedTensorViews_ = &retainedPrimalTensorViews();
        const uint32_t grid[3]{computeGrid_.x, computeGrid_.y, computeGrid_.z};
        const HostProfileLayout &layout = program_->backwardLayout;
        if (!validateDispatchContract(layout.dispatchContract, grid, layout.workgroup, invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        VernonLaunchSize extent{};
        if (!invocationExtent(computeGrid_, {layout.workgroup[0], layout.workgroup[1], layout.workgroup[2]}, extent))
            return fail(context_, "CPU pullback invocation extent overflows");
        if (signature_->cotangents.empty())
            return fail(context_, "CPU pullback has no cotangent leaves");
        if ((cotangents && cotangents->value_count != signature_->cotangents.size()) ||
            (!cotangents && signature_->cotangents.size() != 1))
            return fail(context_, "CPU pullback requires exactly the reflected output cotangent leaves");
        std::unordered_map<std::string_view, size_t> gradientIndices;
        gradientIndices.reserve(signature_->gradients.size());
        for (size_t index = 0; index < signature_->gradients.size(); ++index)
            gradientIndices.emplace(signature_->gradients[index].path, index);
        std::vector<std::string> gradientOwnership(signature_->gradients.size());
        std::vector<bool> storageGradients(signature_->gradients.size());
        for (size_t gradientIndex : program_->resultGradientIndices) {
            if (gradientIndex >= gradientOwnership.size())
                return fail(context_, "CPU pullback result gradient index is invalid");
            gradientOwnership[gradientIndex] = "invocation_private";
        }
        for (const HostArgument &argument : layout.arguments)
            if (argument.tensorView && !isShapeSource(argument) && !isPrimalSource(argument))
                if (const auto gradient = gradientIndices.find(argument.name); gradient != gradientIndices.end()) {
                    gradientOwnership[gradient->second] = argument.accumulationOwnership;
                    storageGradients[gradient->second] = true;
                }
        if (std::any_of(gradientOwnership.begin(), gradientOwnership.end(),
                        [](const std::string &ownership) { return ownership.empty(); }))
            return fail(context_, "CPU pullback gradient has no reflected ownership");
        size_t invocationCount = 0;
        if (!carrierCount(extent, invocationCount))
            return fail(context_, "CPU pullback dispatch size overflows");
        size_t workgroupVolume = 1;
        for (uint32_t dimension : layout.workgroup) {
            if (workgroupVolume > SIZE_MAX / dimension)
                return fail(context_, "CPU pullback workgroup size overflows");
            workgroupVolume *= dimension;
        }
        const bool segmented = groupLinear_.has_value();
        const size_t frameCount = segmented ? workgroupVolume : invocationCount;
        const bool inlineExecution = segmented || invocationCount <= kCpuAdInlineInvocationLimit;
        std::vector<bool> sharedCotangents(signature_->cotangents.size());
        if (cotangents) {
            for (size_t index = 0; index < signature_->cotangents.size(); ++index) {
                const ValueAbi &logical = signature_->cotangents[index];
                ValueAbi carried = logical;
                if (!materializeCarrierValue(carried, extent))
                    return fail(context_, "CPU pullback cotangent size overflows");
                const VernonAdValue *value = findValue(*cotangents, logical.path);
                if (!value)
                    return fail(context_, "output cotangent does not match backward reflection");
                if (valueMatches(*value, logical))
                    sharedCotangents[index] = true;
                else if (!valueMatches(*value, carried))
                    return fail(context_, "output cotangent does not match backward reflection");
            }
        }
        size_t requiredGradientBytes = 0;
        for (size_t index = 0; index < signature_->cotangents.size(); ++index)
            if (!checkedAddBytes(requiredGradientBytes, sharedCotangents[index] ? 1 : invocationCount,
                                 signature_->cotangents[index].byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
        for (size_t gradientIndex = 0; gradientIndex < signature_->gradients.size(); ++gradientIndex) {
            const size_t byteSize = signature_->gradients[gradientIndex].byteSize;
            if (!checkedAddBytes(requiredGradientBytes, 1, byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
            const size_t carrierCount =
                storageGradients[gradientIndex] || gradientOwnership[gradientIndex] == "none" ? 0
                : gradientOwnership[gradientIndex] == "invocation_private" ? (inlineExecution ? 1 : frameCount)
                                                                           : 0;
            if (!storageGradients[gradientIndex] && gradientOwnership[gradientIndex] != "none" &&
                gradientOwnership[gradientIndex] != "invocation_private")
                return fail(context_, "CPU pullback non-Storage gradient requires invocation-private ownership");
            if (!checkedAddBytes(requiredGradientBytes, carrierCount, byteSize))
                return fail(context_, "CPU pullback dispatch memory size overflows");
        }
        if (requiredGradientBytes > kCpuAdGradientDispatchLimit)
            return fail(context_, "CPU pullback requires " + std::to_string(requiredGradientBytes) +
                                      " gradient bytes, exceeding the " + std::to_string(kCpuAdGradientDispatchLimit) +
                                      "-byte dispatch limit");
        std::vector<const HostArgument *> cotangentArguments;
        for (const HostArgument &argument : layout.arguments)
            if (argument.builtin.empty() && !isShapeSource(argument) && !isPrimalSource(argument) &&
                gradientIndices.find(argument.name) == gradientIndices.end())
                cotangentArguments.push_back(&argument);
        if (cotangentArguments.size() != signature_->cotangents.size())
            return fail(context_, "CPU pullback has an invalid cotangent ABI");
        std::vector<std::vector<uint8_t>> cotangentBytes;
        std::vector<const uint8_t *> directCotangents(signature_->cotangents.size());
        for (size_t index = 0; index < signature_->cotangents.size(); ++index) {
            const HostArgument &argument = *cotangentArguments[index];
            const size_t leafCount = argument.tensorView ? argument.tensorView->leaves.size() : argument.leaves.size();
            if (leafCount != 1 || argument.name != signature_->cotangents[index].path)
                return fail(context_, "CPU pullback cotangent reflection is inconsistent");
            ValueAbi abi = signature_->cotangents[index];
            if (!materializeCarrierValue(abi, extent))
                return fail(context_, "CPU pullback cotangent size overflows");
            const size_t byteSize = sharedCotangents[index] ? signature_->cotangents[index].byteSize : abi.byteSize;
            if (segmented && cotangents && !sharedCotangents[index]) {
                const VernonAdValue *value = findValue(*cotangents, abi.path);
                if (!value || value->size != byteSize)
                    return fail(context_, "output cotangent does not match backward reflection");
                directCotangents[index] = static_cast<const uint8_t *>(value->data);
                cotangentBytes.emplace_back();
                continue;
            }
            std::vector<uint8_t> bytes(byteSize);
            if (cotangents) {
                const VernonAdValue *value = findValue(*cotangents, abi.path);
                if (!value || value->size != byteSize)
                    return fail(context_, "output cotangent does not match backward reflection");
                std::memcpy(bytes.data(), value->data, byteSize);
            } else {
                if (signature_->cotangents.size() != 1 ||
                    !makeCotangentBytes(nullptr, abi, bytes, invocationDiagnostic(context_)))
                    return VERNON_STATUS_INVALID_ARGUMENT;
            }
            cotangentBytes.push_back(std::move(bytes));
        }
        std::vector<VernonAdValue *> destinations;
        std::vector<std::vector<uint8_t>> localStagedGradients;
        std::vector<std::vector<uint8_t>> &stagedGradients =
            externalStagedGradients_ ? *externalStagedGradients_ : localStagedGradients;
        if (!externalStagedGradients_)
            if (VernonStatus status =
                    prepareGradientDestinations(context_, *signature_, gradients, destinations, stagedGradients);
                status != VERNON_STATUS_OK)
                return status;
        if (stagedGradients.size() != signature_->gradients.size())
            return fail(context_, "CPU segmented pullback gradient staging is inconsistent",
                        VERNON_STATUS_INTERNAL_ERROR);
        size_t argumentFrameBytes = 0;
        size_t resultFrameBytes = 0;
        if (!checkedAddBytes(argumentFrameBytes, frameCount, layout.argumentsSize) ||
            !checkedAddBytes(resultFrameBytes, frameCount, layout.resultsSize))
            return fail(context_, "CPU pullback invocation frame size overflows");
        std::vector<uint8_t> packedArguments(argumentFrameBytes);
        std::vector<uint8_t> packedResults(resultFrameBytes);
        std::vector<const void *> laneArguments(frameCount);
        std::vector<void *> laneResults(frameCount);
        std::vector<size_t> privateGradientOffsets(signature_->gradients.size(), std::numeric_limits<size_t>::max());
        size_t privateGradientStride = 0;
        for (size_t gradientIndex = 0; gradientIndex < signature_->gradients.size(); ++gradientIndex) {
            if (storageGradients[gradientIndex] || gradientOwnership[gradientIndex] != "invocation_private")
                continue;
            privateGradientOffsets[gradientIndex] = privateGradientStride;
            if (!checkedAddBytes(privateGradientStride, 1, signature_->gradients[gradientIndex].byteSize))
                return fail(context_, "CPU pullback private gradient layout overflows");
        }
        size_t privateGradientBytes = 0;
        if (!checkedAddBytes(privateGradientBytes, inlineExecution ? 1 : frameCount, privateGradientStride))
            return fail(context_, "CPU pullback private gradient storage overflows");
        std::vector<uint8_t> privateGradients(privateGradientBytes);
        uint8_t shapeSourceSentinel{};
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
        HostTapeTraversalMetrics *traversalMetrics = currentHostTapeTraversalMetrics();
#endif
        std::mutex dispatchFailureMutex;
        std::string dispatchFailure;
        const auto failDispatch = [&](std::string message, VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
            std::lock_guard lock(dispatchFailureMutex);
            if (dispatchFailure.empty())
                dispatchFailure = std::move(message);
            return status;
        };
        CpuWorkgroupScheduler &scheduler = cpuWorkgroupScheduler(context_);
        const CpuRangeCallback executeRange = [&](VernonCpuRangeV1 &range) {
            std::vector<HostStaticTapeBatch::Reader> staticReaders;
            if (tapeBatch_) {
                staticReaders.resize(range.lane_end - range.lane_begin);
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const size_t tapeIndex =
                        segmented ? localLinear : cpuRangeCoordinates(range, localLinear).linearIndex;
                    if (!tapeBatch_->initializeReader(tapeIndex, staticReaders[localLinear - range.lane_begin]))
                        return failDispatch("CPU pullback tape reader could not be initialized");
                }
            }
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                auto prepareLane = [&]() -> VernonStatus {
                    const size_t invocationIndex = coordinates.linearIndex;
                    const size_t frameIndex = segmented ? localLinear : invocationIndex;
                    uint8_t *arguments = packedArguments.data() + frameIndex * layout.argumentsSize;
                    uint8_t *results = packedResults.data() + frameIndex * layout.resultsSize;
                    const size_t tableIndex = segmented ? localLinear : invocationIndex;
                    laneArguments[tableIndex] = arguments;
                    laneResults[tableIndex] = results;
                    for (const HostArgument &argument : layout.arguments) {
                        if (isPrimalSource(argument)) {
                            if (argument.tensorView) {
                                std::string owner = argument.name.substr(7);
                                const auto retained = retainedTensorViews_->find(owner);
                                if (retained == retainedTensorViews_->end() || !retained->second.shape ||
                                    !writeTensorViewDescriptor(argument, *retained->second.shape,
                                                               retained->second.packed, arguments))
                                    return failDispatch("CPU pullback required TensorView primal was not retained");
                                continue;
                            }
                            for (const HostFrameLeaf &leaf : argument.leaves) {
                                std::string path = leaf.value.path;
                                if (path.rfind("primal.", 0) != 0)
                                    return failDispatch("CPU pullback primal reflection path is invalid");
                                path.erase(0, 7);
                                const auto retained = retainedPrimals_->find(path);
                                if (retained == retainedPrimals_->end() ||
                                    retained->second.size() != leaf.value.byteSize)
                                    return failDispatch("CPU pullback required primal was not retained");
                                std::memcpy(arguments + leaf.frameOffset, retained->second.data(),
                                            retained->second.size());
                            }
                            continue;
                        }
                        if (!argument.tensorView)
                            continue;
                        uint8_t *data = &shapeSourceSentinel;
                        const std::vector<uint64_t> *shape = nullptr;
                        std::vector<uint64_t> descriptorShape;
                        if (isShapeSource(argument)) {
                            const auto retained = tensorShapes_->find(tensorOwnerName(argument));
                            if (retained == tensorShapes_->end())
                                return failDispatch("CPU pullback TensorView argument has no retained forward shape");
                            shape = &retained->second;
                        } else {
                            const auto gradient = gradientIndices.find(argument.name);
                            if (gradient == gradientIndices.end())
                                continue;
                            const size_t gradientIndex = gradient->second;
                            if (!tensorViewDescriptorShape(argument, signature_->gradients[gradientIndex].logicalShape,
                                                           descriptorShape))
                                return failDispatch("CPU pullback gradient descriptor shape is inconsistent");
                            shape = &descriptorShape;
                            data = gradientOwnership[gradientIndex] == "none" ? &shapeSourceSentinel
                                                                              : stagedGradients[gradientIndex].data();
                        }
                        if (!shape || !writeTensorViewDescriptor(argument, *shape, data, arguments))
                            return failDispatch("CPU pullback TensorView descriptor overflows");
                    }
                    if (tapeBatch_) {
                        if (!layout.tapeAllocatorOffset || !layout.tapeRootRegionOffset)
                            return failDispatch("CPU pullback dynamic tape ABI is incomplete");
                        HostStaticTapeBatch::Reader &reader = staticReaders[localLinear - range.lane_begin];
                        VernonAdTapeAllocator *descriptor = reader.descriptor();
                        const VernonAdRegionHandle root = reader.rootRegion();
                        if (!descriptor)
                            return failDispatch("CPU pullback tape view is unavailable");
                        if (!root)
                            return failDispatch("CPU pullback dynamic tape has no root region");
                        std::memcpy(arguments + *layout.tapeAllocatorOffset, &descriptor,
                                    sizeof(VernonAdTapeAllocator *));
                        std::memcpy(arguments + *layout.tapeRootRegionOffset, &root, sizeof(root));
                    } else if (layout.tapeAllocatorOffset || layout.tapeRootRegionOffset) {
                        return failDispatch("CPU pullback no-Tape ABI contains Tape arguments");
                    }
                    for (size_t cotangentIndex = 0; cotangentIndex < cotangentArguments.size(); ++cotangentIndex) {
                        const HostArgument &argument = *cotangentArguments[cotangentIndex];
                        const uint8_t *source =
                            directCotangents[cotangentIndex]
                                ? directCotangents[cotangentIndex] +
                                      invocationIndex * signature_->cotangents[cotangentIndex].byteSize
                                : cotangentBytes[cotangentIndex].data() +
                                      (sharedCotangents[cotangentIndex]
                                           ? 0
                                           : invocationIndex * signature_->cotangents[cotangentIndex].byteSize);
                        if (argument.tensorView) {
                            std::vector<uint64_t> descriptorShape;
                            if (!tensorViewDescriptorShape(
                                    argument, signature_->cotangents[cotangentIndex].logicalShape, descriptorShape) ||
                                !writeTensorViewDescriptor(argument, descriptorShape, const_cast<uint8_t *>(source),
                                                           arguments))
                                return failDispatch("CPU pullback cotangent descriptor is inconsistent");
                        } else {
                            const HostFrameLeaf &leaf = argument.leaves.front();
                            std::memcpy(arguments + leaf.frameOffset, source,
                                        signature_->cotangents[cotangentIndex].byteSize);
                        }
                    }
                    for (const HostArgument &argument : layout.arguments) {
                        if (argument.builtin.empty() || argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN ||
                            argument.builtin == "ad_tape_root_region")
                            continue;
                        if (!writeInvocationBuiltin(argument, coordinates, arguments))
                            return failDispatch("native CPU pullback has an unsupported builtin ABI");
                    }
                    return VERNON_STATUS_OK;
                };
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
                const VernonStatus prepareStatus = withHostTapeTraversalMetrics(traversalMetrics, prepareLane);
#else
                const VernonStatus prepareStatus = prepareLane();
#endif
                if (prepareStatus != VERNON_STATUS_OK)
                    return prepareStatus;
            }
            const size_t firstInvocation = cpuRangeCoordinates(range, range.lane_begin).linearIndex;
            const size_t firstTableIndex = segmented ? range.lane_begin : firstInvocation;
            range.arguments = laneArguments[firstTableIndex];
            range.arguments_size = layout.argumentsSize;
            range.results = laneResults[firstInvocation];
            range.results_size = layout.resultsSize;
            range.lane_arguments = laneArguments.data();
            range.lane_results = laneResults.data();
            range.lane_table_count = laneArguments.size();
            const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
            auto invokeBackward = [&] { return program_->backward->entry(&invocation); };
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
            const VernonStatus status = withHostTapeTraversalMetrics(traversalMetrics, invokeBackward);
#else
            const VernonStatus status = invokeBackward();
#endif
            if (status != VERNON_STATUS_OK)
                return failDispatch("autodiff backward profile invocation failed", status);
            if (range.outcome == VERNON_CPU_RANGE_YIELDED_V1)
                return VERNON_STATUS_OK;
            if (range.outcome != VERNON_CPU_RANGE_COMPLETE_V1 ||
                range.completed_lanes != range.lane_end - range.lane_begin)
                return failDispatch("autodiff backward lanes completed at different barrier phases");
            for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                const size_t invocationIndex = cpuRangeCoordinates(range, localLinear).linearIndex;
                const size_t frameIndex = segmented ? localLinear : invocationIndex;
                uint8_t *results = packedResults.data() + frameIndex * layout.resultsSize;
                uint8_t *lanePrivateGradients =
                    privateGradients.data() + (inlineExecution ? 0 : frameIndex * privateGradientStride);
                if (VernonStatus accumulateStatus =
                        accumulateBackwardResults(layout, *signature_, results, program_->resultGradientIndices,
                                                  lanePrivateGradients, privateGradientOffsets);
                    accumulateStatus != VERNON_STATUS_OK)
                    return failDispatch("CPU pullback result gradient ABI is inconsistent", accumulateStatus);
                const size_t tableIndex = segmented ? localLinear : invocationIndex;
                laneArguments[tableIndex] = nullptr;
                laneResults[tableIndex] = nullptr;
            }
            return VERNON_STATUS_OK;
        };
        const VernonStatus dispatchStatus =
            segmented         ? scheduler.dispatchGroupInline(grid, layout.workgroup, *groupLinear_, executeRange)
            : inlineExecution ? scheduler.dispatchInline(grid, layout.workgroup, executeRange)
                              : scheduler.dispatch(grid, layout.workgroup, executeRange);
        if (dispatchStatus != VERNON_STATUS_OK) {
            const std::string diagnostic = !dispatchFailure.empty() ? dispatchFailure
                                           : scheduler.lastDiagnostic().empty()
                                               ? "CPU pullback workgroup execution failed"
                                               : scheduler.lastDiagnostic();
            return fail(context_, diagnostic, dispatchStatus);
        }
        for (size_t gradientIndex = 0; gradientIndex < signature_->gradients.size(); ++gradientIndex) {
            if (gradientOwnership[gradientIndex] == "none" || storageGradients[gradientIndex]) {
                continue;
            } else {
                const size_t contributionCount = inlineExecution ? 1 : frameCount;
                for (size_t contribution = 0; contribution < contributionCount; ++contribution)
                    if (VernonStatus status =
                            accumulateGradientBytes(context_, signature_->gradients[gradientIndex],
                                                    privateGradients.data() + contribution * privateGradientStride +
                                                        privateGradientOffsets[gradientIndex],
                                                    stagedGradients[gradientIndex]);
                        status != VERNON_STATUS_OK)
                        return status;
            }
        }
        if (!externalStagedGradients_)
            commitGradientDestinations(destinations, stagedGradients);
        return VERNON_STATUS_OK;
    }

private:
    const Signature &runtimeSignature() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->signature;
        return std::get<CpuBackwardBorrowedState>(state_).signature.get();
    }

    const RuntimeTensorShapes &runtimeTensorShapes() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->tensorShapes;
        return std::get<CpuBackwardBorrowedState>(state_).tensorShapes.get();
    }

    const RetainedPrimalLeaves &retainedPrimalLeaves() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->retainedPrimals;
        return std::get<CpuBackwardBorrowedState>(state_).retainedPrimals.get();
    }

    const RetainedPrimalTensorViewRefs &retainedPrimalTensorViews() const {
        if (const auto *owned = std::get_if<CpuBackwardOwnedState>(&state_))
            return owned->retainedTensorViewRefs;
        return std::get<CpuBackwardBorrowedState>(state_).retainedTensorViews.get();
    }

    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
    std::variant<CpuBackwardOwnedState, CpuBackwardBorrowedState> state_;
    VernonLaunchSize computeGrid_{};
    std::shared_ptr<HostStaticTapeBatch> tapeBatch_;
    std::optional<size_t> groupLinear_;
    std::vector<std::vector<uint8_t>> *externalStagedGradients_{};
};

class NoTapeCpuPullback final : public PullbackExecution {
public:
    NoTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                      Signature signature, VernonLaunchSize computeGrid, RuntimeTensorShapes tensorShapes,
                      RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews)
        : executor_(context, std::move(program), std::move(signature), computeGrid, nullptr, std::move(tensorShapes),
                    std::move(retainedPrimals), std::move(retainedTensorViews)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &) override {
        return executor_.apply(cotangents, gradients);
    }

    PullbackMemoryUsage memoryUsage() const override { return executor_.memoryUsage(); }

private:
    CpuBackwardExecutor executor_;
};

class RetainedTapeCpuPullback final : public PullbackExecution {
public:
    RetainedTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                            Signature signature, VernonLaunchSize computeGrid,
                            std::shared_ptr<HostStaticTapeBatch> tapeBatch, RuntimeTensorShapes tensorShapes,
                            RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews)
        : executor_(context, std::move(program), std::move(signature), computeGrid, std::move(tapeBatch),
                    std::move(tensorShapes), std::move(retainedPrimals), std::move(retainedTensorViews)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                       const PullbackApplyOptions &) override {
        return executor_.apply(cotangents, gradients);
    }

    PullbackMemoryUsage memoryUsage() const override { return executor_.memoryUsage(); }

private:
    CpuBackwardExecutor executor_;
};

class NoTapeStructuredCpuExecutable final : public Executable {
public:
    NoTapeStructuredCpuExecutable(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program)
        : context_(context), program_(std::move(program)) {}

    const Signature &signature() const override { return program_->signature; }

    VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize computeGrid,
                         const VernonAdValueSet &inputs, VernonAdValueSet *outputs,
                         std::unique_ptr<PullbackExecution> &pullback) override {
        if (target.externalEncoder() || !outputs)
            return fail(context_, "CPU autodiff does not accept a GPU command encoder");
        const HostProfileLayout &forwardLayout = program_->forwardLayout;
        const HostProfileLayout &backwardLayout = program_->backwardLayout;
        const uint32_t grid[3]{computeGrid.x, computeGrid.y, computeGrid.z};
        if (!validateDispatchContract(backwardLayout.dispatchContract, grid, backwardLayout.workgroup,
                                      invocationDiagnostic(context_)))
            return VERNON_STATUS_INVALID_ARGUMENT;
        RetainedPrimalLeaves retainedPrimals;
        if (VernonStatus status = retainRequiredPrimalLeaves(context_, backwardLayout, inputs, retainedPrimals);
            status != VERNON_STATUS_OK)
            return status;
        return runCpuForward(
            context_, forwardLayout, program_->signature, computeGrid, inputs, *outputs,
            program_->requiredPrimalTensorOwners, pullback,
            [&](VernonCpuRangeV1 &range, auto &frames, std::string &diagnostic) {
                for (size_t localLinear = range.lane_begin; localLinear < range.lane_end; ++localLinear) {
                    const CpuLaneCoordinates coordinates = cpuRangeCoordinates(range, localLinear);
                    uint8_t *arguments = frames[coordinates.linearIndex].arguments;
                    for (const HostArgument &argument : forwardLayout.arguments)
                        if (!argument.builtin.empty() && !writeInvocationBuiltin(argument, coordinates, arguments)) {
                            diagnostic = "native CPU no-Tape autodiff has an unsupported builtin ABI";
                            return VERNON_STATUS_INVALID_ARGUMENT;
                        }
                }
                const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
                return program_->forward->entry(&invocation);
            },
            [&](size_t, Signature runtimeSignature, RuntimeTensorShapes tensorShapes,
                RetainedPrimalTensorViews retainedTensorViews, std::unique_ptr<PullbackExecution> &pending) {
                pending = createNoTapeCpuPullback(context_, program_, std::move(runtimeSignature), computeGrid,
                                                  std::move(tensorShapes), std::move(retainedPrimals),
                                                  std::move(retainedTensorViews));
                return VERNON_STATUS_OK;
            });
    }

private:
    VernonRuntimeContext &context_;
    std::shared_ptr<const CpuAutodiffProgram> program_;
};

std::unique_ptr<PullbackExecution>
createNoTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                        Signature signature, VernonLaunchSize computeGrid, RuntimeTensorShapes tensorShapes,
                        RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews) {
    return std::make_unique<NoTapeCpuPullback>(context, std::move(program), std::move(signature), computeGrid,
                                               std::move(tensorShapes), std::move(retainedPrimals),
                                               std::move(retainedTensorViews));
}

std::unique_ptr<PullbackExecution>
createRetainedTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                              Signature signature, VernonLaunchSize computeGrid,
                              std::shared_ptr<HostStaticTapeBatch> tapeBatch, RuntimeTensorShapes tensorShapes,
                              RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews) {
    return std::make_unique<RetainedTapeCpuPullback>(context, std::move(program), std::move(signature), computeGrid,
                                                     std::move(tapeBatch), std::move(tensorShapes),
                                                     std::move(retainedPrimals), std::move(retainedTensorViews));
}

VernonStatus applyCpuBackwardSegment(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                                     const Signature &signature, const RuntimeTensorShapes &tensorShapes,
                                     const RetainedPrimalLeaves &retainedPrimals,
                                     const RetainedPrimalTensorViewRefs &retainedTensorViews,
                                     VernonLaunchSize computeGrid, std::shared_ptr<HostStaticTapeBatch> &tapeBatch,
                                     size_t groupLinear, const VernonAdValueSet *cotangents,
                                     std::vector<std::vector<uint8_t>> &stagedGradients) {
    CpuBackwardExecutor segment(context, std::move(program),
                                CpuBackwardBorrowedState{signature, tensorShapes, retainedPrimals, retainedTensorViews},
                                computeGrid, std::move(tapeBatch), groupLinear);
    const VernonStatus status = segment.applyInto(cotangents, stagedGradients);
    tapeBatch = segment.releaseTapeBatch();
    return status;
}

std::shared_ptr<Executable> createNoTapeStructuredCpuExecutable(VernonRuntimeContext &context,
                                                                std::shared_ptr<const CpuAutodiffProgram> program) {
    return std::make_shared<NoTapeStructuredCpuExecutable>(context, std::move(program));
}

} // namespace vernon::runtime::ad::cpu
