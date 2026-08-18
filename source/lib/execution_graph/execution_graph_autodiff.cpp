#include "VernonExecutionGraph.h"

#include "autodiff/autodiff_memory_accounting.h"
#include "execution_graph_internal.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vernon::execution {
namespace {

bool writes(AccessMode access) { return access != AccessMode::Read; }

struct DerivativeEndpointHash {
    size_t operator()(const DerivativeEndpointKey &endpoint) const {
        return (static_cast<size_t>(endpoint.kind) << 32u) ^ endpoint.id;
    }
};

using EndpointValues =
    std::unordered_map<DerivativeEndpointKey, std::shared_ptr<GraphAutodiffValue>, DerivativeEndpointHash>;
using GraphMemoryAccounting = vernon::autodiff::MemoryAccounting<uint64_t>;

bool checkedAdd(uint64_t left, uint64_t right, uint64_t &result) {
    if (right > std::numeric_limits<uint64_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

class BackwardCommandProgram final : public detail::RhiCommandPlanSink {
public:
    BackwardCommandProgram(VernonRhiDevice device, uint64_t &submissions, uint64_t &waits,
                           uint64_t &deviceWaitNanoseconds, std::string &error)
        : device_(device), submissions_(submissions), waits_(waits), deviceWaitNanoseconds_(deviceWaitNanoseconds),
          error_(error) {}
    ~BackwardCommandProgram() override { finish(false, {}); }

    VernonRhiStatus append(detail::RhiCommandExecutionPlan plan) override {
        std::string error;
        if (!detail::appendRhiCommandExecutionPlan(plan_, std::move(plan), true, error)) {
            error_ = std::move(error);
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        return VERNON_RHI_STATUS_OK;
    }

    void retain(std::shared_ptr<void> context) override {
        if (context)
            retainedContexts_.push_back(std::move(context));
    }

    void onCompletion(std::shared_ptr<detail::RhiCommandCompletion> completion) override {
        if (completion)
            completions_.push_back(std::move(completion));
    }

    VernonRhiStatus flush() override {
        if (plan_.commands.nodes.empty()) {
            return finish(true, {}) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        struct DiagnosticEncoder {
            detail::RhiCommandNodeEncoder original;
            VernonRhiDevice device;
            std::string *error;
            uint32_t index;
            detail::CommandNodeKind kind;
        };
        const auto encode = [](void *opaque, VernonRhiCommandEncoder encoder) {
            auto &context = *static_cast<DiagnosticEncoder *>(opaque);
            const VernonRhiStatus status = context.original.encode(context.original.context, encoder);
            if (status != VERNON_RHI_STATUS_OK && context.error->empty())
                *context.error = "graph pullback command node " + std::to_string(context.index) + " (kind " +
                                 std::to_string(static_cast<uint32_t>(context.kind)) + ") failed to encode";
            if (status != VERNON_RHI_STATUS_OK) {
                const VernonStringView diagnostic = vernonRhiDeviceGetLastError(context.device);
                if (diagnostic.data && diagnostic.size) {
                    *context.error += ": ";
                    context.error->append(diagnostic.data, diagnostic.size);
                }
            }
            return status;
        };
        const auto complete = [](void *opaque) {
            auto &context = *static_cast<DiagnosticEncoder *>(opaque);
            const VernonRhiStatus status = context.original.complete(context.original.context);
            if (status != VERNON_RHI_STATUS_OK && context.error->empty())
                *context.error = "graph pullback status node " + std::to_string(context.index) + " failed";
            return status;
        };
        for (uint32_t index = 0; index < plan_.encoders.size(); ++index) {
            auto context = std::make_shared<DiagnosticEncoder>(
                DiagnosticEncoder{plan_.encoders[index], device_, &error_, index, plan_.commands.nodes[index].kind});
            plan_.encoders[index] = context->original.encode
                                        ? detail::RhiCommandNodeEncoder{encode, context.get(), nullptr}
                                        : detail::RhiCommandNodeEncoder{nullptr, context.get(), complete};
            plan_.retainedContexts.push_back(std::move(context));
        }
        std::string validationError;
        if (!detail::validateRhiCommandExecutionPlan(plan_, validationError)) {
            error_ = std::move(validationError);
            plan_ = {};
            finish(false, {});
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        detail::RhiCommandDagExecutionStats stats;
        const VernonRhiStatus status =
            detail::executeRhiCommandPlanAndWait(device_, VERNON_RHI_QUEUE_COMPUTE, plan_, &stats);
        plan_ = {};
        const bool completionSucceeded = finish(status == VERNON_RHI_STATUS_OK, stats);
        if (!checkedAdd(submissions_, stats.submissions, submissions_) || !checkedAdd(waits_, stats.waits, waits_) ||
            !checkedAdd(deviceWaitNanoseconds_, stats.deviceWaitNanoseconds, deviceWaitNanoseconds_)) {
            error_ = "graph pullback command-program telemetry overflows";
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        if (status != VERNON_RHI_STATUS_OK && error_.empty()) {
            error_ = "graph pullback command program failed (RHI status " +
                     std::to_string(static_cast<uint32_t>(status)) + ")";
            const VernonStringView diagnostic = vernonRhiDeviceGetLastError(device_);
            if (diagnostic.data && diagnostic.size) {
                error_ += ": ";
                error_.append(diagnostic.data, diagnostic.size);
            }
        }
        return status == VERNON_RHI_STATUS_OK && !completionSucceeded ? VERNON_RHI_STATUS_INTERNAL_ERROR : status;
    }

private:
    bool finish(bool succeeded, const detail::RhiCommandDagExecutionStats &stats) noexcept {
        bool completed = true;
        bool callbackSucceeded = succeeded;
        const auto runPhase = [&](bool validationPhase) {
            for (const auto &completion : completions_) {
                if (completion->validationPhase() != validationPhase)
                    continue;
                try {
                    completion->complete(callbackSucceeded, stats);
                } catch (const std::exception &exception) {
                    completed = false;
                    callbackSucceeded = false;
                    if (error_.empty())
                        error_ = exception.what();
                } catch (...) {
                    completed = false;
                    callbackSucceeded = false;
                    if (error_.empty())
                        error_ = "graph pullback completion callback failed";
                }
            }
        };
        runPhase(true);
        runPhase(false);
        completions_.clear();
        retainedContexts_.clear();
        return completed;
    }

    VernonRhiDevice device_{};
    detail::RhiCommandExecutionPlan plan_;
    std::vector<std::shared_ptr<void>> retainedContexts_;
    std::vector<std::shared_ptr<detail::RhiCommandCompletion>> completions_;
    uint64_t &submissions_;
    uint64_t &waits_;
    uint64_t &deviceWaitNanoseconds_;
    std::string &error_;
};

detail::CommandResourceAccess commandBufferAccess(VernonRhiBuffer buffer, uint64_t offset, uint64_t byteSize,
                                                  AccessMode access) {
    return detail::rhiBufferAccess(buffer, offset, byteSize, access,
                                   access == AccessMode::Read ? VERNON_RHI_STATE_TRANSFER_SOURCE
                                                              : VERNON_RHI_STATE_TRANSFER_DESTINATION);
}

template <typename Encode>
bool submitDeviceCommands(VernonRhiDevice device, Encode &&encode, std::vector<detail::CommandResourceAccess> accesses,
                          std::vector<detail::RhiCommandResourceBinding> bindings,
                          std::vector<detail::CommandResourceAccess> initialAccesses,
                          std::vector<detail::CommandResourceAccess> finalAccesses, const char *failure,
                          std::string &error, detail::RhiCommandPlanSink *sink = nullptr) {
    using Encoder = std::remove_reference_t<Encode>;
    const auto invoke = [](void *context, VernonRhiCommandEncoder encoder) {
        return (*static_cast<Encoder *>(context))(encoder);
    };
    detail::RhiCommandExecutionPlan plan;
    auto retained = std::make_shared<Encoder>(std::forward<Encode>(encode));
    detail::CommandNode transfer;
    transfer.kind = detail::CommandNodeKind::Checkpoint;
    transfer.queue = detail::CommandQueueClass::Transfer;
    transfer.accesses = std::move(accesses);
    plan.commands.nodes.push_back(std::move(transfer));
    plan.encoders.push_back({invoke, retained.get()});
    plan.retainedContexts.push_back(retained);
    if (!finalAccesses.empty()) {
        detail::CommandNode finalState;
        finalState.kind = detail::CommandNodeKind::Checkpoint;
        finalState.queue = detail::CommandQueueClass::Transfer;
        finalState.predecessors = {0};
        finalState.accesses = std::move(finalAccesses);
        plan.commands.nodes.push_back(std::move(finalState));
        plan.encoders.push_back({[](void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }, nullptr});
    }
    plan.bindings = std::move(bindings);
    plan.initialAccesses = std::move(initialAccesses);
    const VernonRhiStatus status = sink ? sink->append(std::move(plan))
                                        : detail::executeRhiCommandPlanAndWait(device, VERNON_RHI_QUEUE_COMPUTE, plan);
    if (status == VERNON_RHI_STATUS_OK)
        return true;
    error = failure;
    const VernonStringView diagnostic = vernonRhiDeviceGetLastError(device);
    if (diagnostic.data && diagnostic.size) {
        error += ": ";
        error.append(diagnostic.data, diagnostic.size);
    }
    return false;
}

struct ResourceSnapshot {
    struct Range {
        uint64_t resourceOffset{};
        uint64_t storageOffset{};
        uint64_t byteSize{};
    };

    ResourceSnapshot() = default;
    ~ResourceSnapshot() { reset(); }
    ResourceSnapshot(const ResourceSnapshot &) = delete;
    ResourceSnapshot &operator=(const ResourceSnapshot &) = delete;
    ResourceSnapshot(ResourceSnapshot &&other) noexcept
        : resource(std::move(other.resource)), bytes(std::exchange(other.bytes, nullptr)),
          device(
              std::exchange(other.device, VernonRhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0})),
          sourceBuffer(std::exchange(other.sourceBuffer,
                                     VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0})),
          storageBuffer(std::exchange(other.storageBuffer,
                                      VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0})),
          byteSize(std::exchange(other.byteSize, 0)),
          alignment(std::exchange(other.alignment, alignof(std::max_align_t))), ranges(std::move(other.ranges)) {}
    ResourceSnapshot &operator=(ResourceSnapshot &&other) noexcept {
        if (this != &other) {
            reset();
            resource = std::move(other.resource);
            bytes = std::exchange(other.bytes, nullptr);
            device =
                std::exchange(other.device, VernonRhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
            sourceBuffer = std::exchange(other.sourceBuffer,
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
            storageBuffer = std::exchange(other.storageBuffer,
                                          VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
            byteSize = std::exchange(other.byteSize, 0);
            alignment = std::exchange(other.alignment, alignof(std::max_align_t));
            ranges = std::move(other.ranges);
        }
        return *this;
    }

    std::shared_ptr<GraphCheckpointResource> resource;
    std::byte *bytes{};
    VernonRhiDevice device{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBuffer sourceBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBuffer storageBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    uint64_t byteSize{};
    size_t alignment{alignof(std::max_align_t)};
    std::vector<Range> ranges;

    bool restore(std::string &error, detail::RhiCommandPlanSink *sink = nullptr) const {
        if (storageBuffer.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
            return submitDeviceCommands(
                device,
                [&](VernonRhiCommandEncoder encoder) {
                    for (const Range &range : ranges)
                        if (vernonRhiCommandEncoderCopyBuffer(device, encoder, storageBuffer, range.storageOffset,
                                                              sourceBuffer, range.resourceOffset,
                                                              range.byteSize) != VERNON_RHI_STATUS_OK)
                            return VERNON_RHI_STATUS_INTERNAL_ERROR;
                    return VERNON_RHI_STATUS_OK;
                },
                [&] {
                    std::vector<detail::CommandResourceAccess> accesses;
                    accesses.reserve(ranges.size() * 2);
                    for (const Range &range : ranges) {
                        accesses.push_back(
                            commandBufferAccess(storageBuffer, range.storageOffset, range.byteSize, AccessMode::Read));
                        accesses.push_back(
                            commandBufferAccess(sourceBuffer, range.resourceOffset, range.byteSize, AccessMode::Write));
                    }
                    return accesses;
                }(),
                {{rhi::encodeResourceKey(storageBuffer), ResourceKind::Buffer, storageBuffer, {}},
                 {rhi::encodeResourceKey(sourceBuffer), ResourceKind::Buffer, sourceBuffer, {}}},
                {detail::rhiBufferAccess(storageBuffer, 0, byteSize, AccessMode::Write,
                                         VERNON_RHI_STATE_TRANSFER_DESTINATION),
                 detail::rhiBufferAccess(sourceBuffer, 0, resource->byteSize(), AccessMode::ReadWrite,
                                         VERNON_RHI_STATE_SHADER_WRITE, VERNON_RHI_STAGE_COMPUTE)},
                {detail::rhiBufferAccess(storageBuffer, 0, byteSize, AccessMode::Write,
                                         VERNON_RHI_STATE_TRANSFER_DESTINATION),
                 detail::rhiBufferAccess(sourceBuffer, 0, resource->byteSize(), AccessMode::Read,
                                         VERNON_RHI_STATE_COMMON)},
                "device-local graph state restoration failed", error, sink);
        }
        std::vector<GraphByteRange> resourceRanges;
        resourceRanges.reserve(ranges.size());
        for (const Range &range : ranges)
            resourceRanges.push_back({range.resourceOffset, range.byteSize});
        return resource->copyRangesFrom(resourceRanges, bytes, error);
    }

private:
    void reset() {
        if (device.index != VERNON_RHI_INVALID_HANDLE_INDEX && storageBuffer.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            (void)vernonRhiDeviceDestroyBuffer(device, storageBuffer);
        if (bytes)
            ::operator delete(bytes, std::align_val_t(alignment));
        device = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        sourceBuffer = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        storageBuffer = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        bytes = nullptr;
        byteSize = 0;
    }
};

bool captureResource(const std::shared_ptr<GraphCheckpointResource> &resource, ResourceSnapshot &snapshot,
                     std::string &error, const std::vector<GraphByteRange> *declaredRanges = nullptr) {
    if (!resource) {
        error = "graph checkpoint resource has no native byte view";
        return false;
    }
    snapshot.resource = resource;
    const uint64_t resourceByteSize = resource->byteSize();
    const uint64_t requestedAlignment = resource->alignment();
    if (resourceByteSize > std::numeric_limits<size_t>::max() ||
        requestedAlignment > std::numeric_limits<size_t>::max() || !requestedAlignment ||
        (requestedAlignment & (requestedAlignment - 1))) {
        error = "graph checkpoint resource has invalid size or alignment";
        return false;
    }
    uint64_t storageOffset = 0;
    const std::vector<GraphByteRange> whole{{0, resourceByteSize}};
    const std::vector<GraphByteRange> &ranges = declaredRanges ? *declaredRanges : whole;
    for (const GraphByteRange &range : ranges) {
        if (!range.byteSize || range.offset > resourceByteSize || range.byteSize > resourceByteSize - range.offset ||
            !checkedAdd(storageOffset, range.byteSize, snapshot.byteSize)) {
            error = "graph checkpoint resource has an invalid write footprint";
            return false;
        }
        snapshot.ranges.push_back({range.offset, storageOffset, range.byteSize});
        storageOffset = snapshot.byteSize;
    }
    snapshot.alignment = std::max<size_t>(requestedAlignment, alignof(std::max_align_t));
    try {
        if (snapshot.byteSize)
            snapshot.bytes =
                static_cast<std::byte *>(::operator new(snapshot.byteSize, std::align_val_t(snapshot.alignment)));
    } catch (const std::bad_alloc &) {
        error = "cannot allocate graph checkpoint storage";
        return false;
    }
    std::vector<GraphByteRange> resourceRanges;
    resourceRanges.reserve(snapshot.ranges.size());
    for (const ResourceSnapshot::Range &range : snapshot.ranges)
        resourceRanges.push_back({range.resourceOffset, range.byteSize});
    return resource->copyRangesTo(resourceRanges, snapshot.bytes, error);
}

bool captureDeviceResource(VernonRhiDevice device, VernonRhiBuffer buffer,
                           const std::shared_ptr<GraphCheckpointResource> &resource, ResourceSnapshot &snapshot,
                           std::string &error, const std::vector<GraphByteRange> *declaredRanges = nullptr) {
    if (!resource) {
        error = "graph checkpoint resource has no device snapshot metadata";
        return false;
    }
    const uint64_t resourceByteSize = resource->byteSize();
    const uint64_t requestedAlignment = resource->alignment();
    if (!resourceByteSize || !requestedAlignment || (requestedAlignment & (requestedAlignment - 1))) {
        error = "graph checkpoint resource has invalid size or alignment";
        return false;
    }
    const std::vector<GraphByteRange> whole{{0, resourceByteSize}};
    const std::vector<GraphByteRange> &ranges = declaredRanges ? *declaredRanges : whole;
    uint64_t storageOffset = 0;
    for (const GraphByteRange &range : ranges) {
        if (!range.byteSize || range.offset > resourceByteSize || range.byteSize > resourceByteSize - range.offset ||
            !checkedAdd(storageOffset, range.byteSize, snapshot.byteSize)) {
            error = "graph checkpoint resource has an invalid write footprint";
            return false;
        }
        snapshot.ranges.push_back({range.offset, storageOffset, range.byteSize});
        storageOffset = snapshot.byteSize;
    }
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = snapshot.byteSize;
    descriptor.alignment = requestedAlignment;
    descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    if (vernonRhiDeviceCreateBuffer(device, &descriptor, &snapshot.storageBuffer) != VERNON_RHI_STATUS_OK) {
        error = "cannot allocate device-local graph state snapshot";
        return false;
    }
    snapshot.device = device;
    snapshot.sourceBuffer = buffer;
    snapshot.resource = resource;
    return submitDeviceCommands(
        device,
        [&](VernonRhiCommandEncoder encoder) {
            for (const ResourceSnapshot::Range &range : snapshot.ranges)
                if (vernonRhiCommandEncoderCopyBuffer(device, encoder, buffer, range.resourceOffset,
                                                      snapshot.storageBuffer, range.storageOffset,
                                                      range.byteSize) != VERNON_RHI_STATUS_OK)
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
            return VERNON_RHI_STATUS_OK;
        },
        [&] {
            std::vector<detail::CommandResourceAccess> accesses;
            accesses.reserve(snapshot.ranges.size() * 2);
            for (const ResourceSnapshot::Range &range : snapshot.ranges) {
                accesses.push_back(commandBufferAccess(buffer, range.resourceOffset, range.byteSize, AccessMode::Read));
                accesses.push_back(commandBufferAccess(snapshot.storageBuffer, range.storageOffset, range.byteSize,
                                                       AccessMode::Write));
            }
            return accesses;
        }(),
        {{rhi::encodeResourceKey(buffer), ResourceKind::Buffer, buffer, {}},
         {rhi::encodeResourceKey(snapshot.storageBuffer), ResourceKind::Buffer, snapshot.storageBuffer, {}}},
        {detail::rhiBufferAccess(buffer, 0, resourceByteSize, AccessMode::ReadWrite, VERNON_RHI_STATE_SHADER_WRITE,
                                 VERNON_RHI_STAGE_COMPUTE),
         detail::rhiBufferAccess(snapshot.storageBuffer, 0, snapshot.byteSize, AccessMode::Read,
                                 VERNON_RHI_STATE_COMMON)},
        {detail::rhiBufferAccess(buffer, 0, resourceByteSize, AccessMode::Read, VERNON_RHI_STATE_COMMON),
         detail::rhiBufferAccess(snapshot.storageBuffer, 0, snapshot.byteSize, AccessMode::Write,
                                 VERNON_RHI_STATE_TRANSFER_DESTINATION)},
        "device-local graph state capture failed", error);
}

class CheckpointStorage {
public:
    ~CheckpointStorage() { reset(); }
    CheckpointStorage() = default;
    CheckpointStorage(const CheckpointStorage &) = delete;
    CheckpointStorage &operator=(const CheckpointStorage &) = delete;

    bool allocate(detail::ExecutionProvider provider, VernonRhiDevice device, uint64_t byteSize,
                  uint64_t requestedAlignment, std::string &error) {
        reset();
        if (byteSize > std::numeric_limits<size_t>::max() || requestedAlignment > std::numeric_limits<size_t>::max()) {
            error = "graph checkpoint storage exceeds host address space";
            return false;
        }
        if (!requestedAlignment || (requestedAlignment & (requestedAlignment - 1))) {
            error = "graph checkpoint storage has invalid alignment";
            return false;
        }
        size_ = static_cast<size_t>(byteSize);
        alignment_ = std::max<size_t>(static_cast<size_t>(requestedAlignment), alignof(std::max_align_t));
        if (!size_)
            return true;
        if (provider == detail::ExecutionProvider::Rhi) {
            VernonRhiBufferDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.size = byteSize;
            descriptor.alignment = requestedAlignment;
            descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
            descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
            if (vernonRhiDeviceCreateBuffer(device, &descriptor, &buffer_) != VERNON_RHI_STATUS_OK) {
                size_ = 0;
                error = "cannot allocate device-local graph checkpoint storage";
                return false;
            }
            device_ = device;
            return true;
        }
        try {
            data_ = static_cast<std::byte *>(::operator new(size_, std::align_val_t(alignment_)));
        } catch (const std::bad_alloc &) {
            size_ = 0;
            error = "cannot allocate graph checkpoint storage";
            return false;
        }
        return true;
    }

    std::byte *data() { return data_; }
    const std::byte *data() const { return data_; }
    uint64_t size() const { return size_; }
    VernonRhiBuffer buffer() const { return buffer_; }

private:
    void reset() {
        if (device_.index != VERNON_RHI_INVALID_HANDLE_INDEX && buffer_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            (void)vernonRhiDeviceDestroyBuffer(device_, buffer_);
        if (data_)
            ::operator delete(data_, std::align_val_t(alignment_));
        device_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        buffer_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        data_ = nullptr;
        size_ = 0;
        alignment_ = alignof(std::max_align_t);
    }

    VernonRhiDevice device_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBuffer buffer_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::byte *data_{};
    size_t size_{};
    size_t alignment_{alignof(std::max_align_t)};
};

struct CheckpointEntry {
    uint64_t offset{};
    uint64_t byteSize{};
    AutodiffResourceVersion version;
    std::shared_ptr<GraphCheckpointResource> resource;
    bool captured{};
    bool released{};
};

} // namespace

class GraphPullback::Impl {
public:
    Impl(std::shared_ptr<CompiledExecutionGraph::State> retainedPlan,
         std::shared_ptr<const ExecutionBindings> retainedBindings, ExecutionSubmission forwardSubmission)
        : plan(std::move(retainedPlan)), bindings(std::move(retainedBindings)), forward(std::move(forwardSubmission)) {}

    bool initializeCheckpointStorage(std::string &error) {
        if (!plan->hasAutodiffCheckpointPlan)
            return true;
        const AutodiffDagCheckpointPlan &checkpointPlan = plan->autodiffCheckpointPlan;
        if (checkpointPlan.passVersions.size() != plan->schedule.size()) {
            error = "compiled graph has no resource-version schedule";
            return false;
        }
        currentResourceEpochs.assign(plan->resourceRecords.size(), 0);
        uint64_t maximumAlignment = alignof(std::max_align_t);
        checkpointEntries.clear();
        checkpointEntries.reserve(checkpointPlan.checkpointResources.size());
        checkpointResourcesByProducer.assign(plan->schedule.size(), {});
        for (uint32_t resourceIndex = 0; resourceIndex < checkpointPlan.checkpointResources.size(); ++resourceIndex) {
            const AutodiffCheckpointResource &resource = checkpointPlan.checkpointResources[resourceIndex];
            uint64_t end = 0;
            if (resource.producer >= checkpointResourcesByProducer.size() || !resource.version.epoch ||
                !resource.alignment || (resource.alignment & (resource.alignment - 1)) ||
                resource.version.resource >= plan->resourceRecords.size() ||
                !checkedAdd(resource.offset, resource.byteSize, end) ||
                end > checkpointPlan.persistentCheckpointBytes) {
                error = "compiled graph checkpoint layout is invalid";
                return false;
            }
            const auto &checkpoint = plan->resourceRecords[resource.version.resource].checkpoint;
            if (!checkpoint || checkpoint->byteSize() != resource.byteSize ||
                checkpoint->alignment() != resource.alignment) {
                error = "compiled graph checkpoint resource does not match its storage layout";
                return false;
            }
            maximumAlignment = std::max(maximumAlignment, resource.alignment);
            checkpointEntries.push_back({resource.offset, resource.byteSize, resource.version, checkpoint});
            checkpointResourcesByProducer[resource.producer].push_back(resourceIndex);
        }
        if (!checkpointStorage.allocate(plan->provider, plan->device, checkpointPlan.persistentCheckpointBytes,
                                        maximumAlignment, error))
            return false;
        observedPeakRuntimeManagedBytes = checkpointStorage.size();
        return true;
    }

    uint64_t retainedAllocationBytesUnlocked() const {
        uint64_t result = 0;
        for (const auto &tape : tapes) {
            uint64_t total = 0;
            if (!checkedAdd(result, tape.second->retainedAllocationBytes(), total))
                return std::numeric_limits<uint64_t>::max();
            result = total;
        }
        return result;
    }

    bool memoryAccounting(uint64_t additionalRetainedBytes, uint64_t temporaryBytes,
                          GraphMemoryAccounting &accounting) const {
        accounting = {};
        if (!checkedAdd(retainedAllocationBytesUnlocked(), additionalRetainedBytes,
                        accounting.retainedAllocationBytes) ||
            !checkedAdd(checkpointStorage.size(), initialStateBytes, accounting.checkpointBytes))
            return false;
        accounting.peakTemporaryBytes = temporaryBytes;
        uint64_t budgeted = 0;
        if (!accounting.budgetedBytes(budgeted))
            return false;
        accounting.residentBytes = budgeted;
        accounting.allocatedBytes = budgeted;
        return true;
    }

    void observeManagedBytes(uint64_t temporaryBytes = 0) {
        GraphMemoryAccounting accounting;
        uint64_t managed = std::numeric_limits<uint64_t>::max();
        if (memoryAccounting(0, temporaryBytes, accounting))
            (void)accounting.budgetedBytes(managed);
        observedPeakRuntimeManagedBytes = std::max(observedPeakRuntimeManagedBytes, managed);
    }

    bool observeTapeAllocation(uint64_t additionalTapeBytes, uint64_t budgetTemporaryBytes,
                               uint64_t observedTemporaryBytes, std::string &error) {
        GraphMemoryAccounting accounting;
        uint64_t managed = 0;
        if (!memoryAccounting(additionalTapeBytes, budgetTemporaryBytes, accounting) ||
            !accounting.budgetedBytes(managed)) {
            error = "graph Runtime-managed memory accounting overflows";
            return false;
        }
        if (plan->hasAutodiffCheckpointPlan && managed > plan->autodiffCheckpointPlan.memoryBudget) {
            error = "graph autodiff runtime allocation exceeds the compiled checkpoint memory budget";
            return false;
        }
        accounting.peakTemporaryBytes = observedTemporaryBytes;
        if (!accounting.budgetedBytes(managed))
            managed = std::numeric_limits<uint64_t>::max();
        observedPeakRuntimeManagedBytes = std::max(observedPeakRuntimeManagedBytes, managed);
        return true;
    }

    void observeForwardPeak(uint64_t peakTapeBytes, uint64_t observedTemporaryBytes) {
        GraphMemoryAccounting accounting;
        uint64_t managed = std::numeric_limits<uint64_t>::max();
        if (memoryAccounting(peakTapeBytes, observedTemporaryBytes, accounting))
            (void)accounting.budgetedBytes(managed);
        observedPeakRuntimeManagedBytes = std::max(observedPeakRuntimeManagedBytes, managed);
    }

    bool preflightTapeAllocation(uint64_t peakTapeBytes, uint64_t temporaryBytes, std::string &error) const {
        GraphMemoryAccounting accounting;
        uint64_t managed = 0;
        if (!memoryAccounting(peakTapeBytes, temporaryBytes, accounting) || !accounting.budgetedBytes(managed)) {
            error = "graph Runtime-managed memory accounting overflows";
            return false;
        }
        if (plan->hasAutodiffCheckpointPlan && managed > plan->autodiffCheckpointPlan.memoryBudget) {
            error = "graph autodiff forward allocation would exceed the compiled checkpoint memory budget";
            return false;
        }
        return true;
    }

    void eraseTapes(uint32_t begin, uint32_t end) {
        for (auto tape = tapes.begin(); tape != tapes.end();)
            if (tape->first >= begin && tape->first < end)
                tape = tapes.erase(tape);
            else
                ++tape;
    }

    bool captureCheckpoint(uint32_t resourceIndex, VernonRhiCommandEncoder encoder, std::string &error) {
        if (resourceIndex >= checkpointEntries.size()) {
            error = "compiled graph checkpoint resource index is invalid";
            return false;
        }
        CheckpointEntry &entry = checkpointEntries[resourceIndex];
        if (entry.version.resource >= currentResourceEpochs.size() ||
            currentResourceEpochs[entry.version.resource] != entry.version.epoch) {
            error = "graph checkpoint capture does not match the planned resource version";
            return false;
        }
        if (plan->provider == detail::ExecutionProvider::Rhi) {
            const auto &record = plan->resourceRecords[entry.version.resource];
            if (record.resource.kind != ResourceKind::Buffer ||
                vernonRhiCommandEncoderCopyBuffer(plan->device, encoder, record.buffer, 0, checkpointStorage.buffer(),
                                                  entry.offset, entry.byteSize) != VERNON_RHI_STATUS_OK) {
                error = "cannot encode device-local graph checkpoint capture";
                return false;
            }
            checkpointStorageInitialized = true;
        } else {
            void *destination = entry.byteSize ? checkpointStorage.data() + entry.offset : nullptr;
            if (!entry.resource->copyTo(destination, entry.byteSize, error))
                return false;
        }
        entry.captured = true;
        return true;
    }

    bool validatePassInputs(uint32_t offset, std::string &error) const {
        if (!plan->hasAutodiffCheckpointPlan)
            return true;
        if (offset >= plan->autodiffCheckpointPlan.passVersions.size()) {
            error = "graph autodiff pass has no resource-version state";
            return false;
        }
        for (const AutodiffResourceVersion &version : plan->autodiffCheckpointPlan.passVersions[offset].inputs)
            if (version.resource >= currentResourceEpochs.size() ||
                currentResourceEpochs[version.resource] != version.epoch) {
                error = "graph replay input does not match the required resource version";
                return false;
            }
        return true;
    }

    bool publishPassOutputs(uint32_t offset, std::string &error) {
        if (!plan->hasAutodiffCheckpointPlan)
            return true;
        if (offset >= plan->autodiffCheckpointPlan.passVersions.size()) {
            error = "graph autodiff pass has no resource-version state";
            return false;
        }
        for (const AutodiffResourceVersion &version : plan->autodiffCheckpointPlan.passVersions[offset].outputs) {
            if (!version.epoch || version.resource >= currentResourceEpochs.size()) {
                error = "graph autodiff pass produced an invalid resource version";
                return false;
            }
            currentResourceEpochs[version.resource] = version.epoch;
        }
        return true;
    }

    bool executeRange(uint32_t begin, uint32_t end, bool retainTapes, bool captureCheckpoints, std::string &error,
                      bool collectMetrics = false, uint64_t temporaryBytes = 0, bool temporaryBytesAllocated = false,
                      detail::RhiCommandPlanSink *sink = nullptr) {
        struct Context {
            Impl &pullback;
            bool retainTapes;
            bool captureCheckpoints;
            bool collectMetrics;
            uint64_t temporaryBytes;
            bool temporaryBytesAllocated;
            std::string &error;
            std::unordered_map<uint32_t, std::unique_ptr<PassPullback>> plannedTapes;
        };
        auto retainedContext = std::make_shared<Context>(Context{*this,
                                                                 retainTapes,
                                                                 captureCheckpoints,
                                                                 collectMetrics,
                                                                 temporaryBytes,
                                                                 temporaryBytesAllocated,
                                                                 error,
                                                                 {}});
        Context &context = *retainedContext;
        const auto execute = [](void *opaque, uint32_t offset, ComputePass &compute, ComputeEncoder &encoder,
                                const ExecutionResources &resources) {
            auto &context = *static_cast<Context *>(opaque);
            Impl &pullback = context.pullback;
            if (!pullback.validatePassInputs(offset, context.error))
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            if (DifferentiablePass *differentiable = compute.differentiable()) {
                if (!pullback.preflightTapeAllocation(differentiable->estimatedForwardPeakBytes(),
                                                      context.temporaryBytes, context.error))
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                std::unique_ptr<PassPullback> tape;
                bool succeeded = true;
                const auto planned = context.plannedTapes.find(offset);
                if (planned != context.plannedTapes.end()) {
                    tape = std::move(planned->second);
                    context.plannedTapes.erase(planned);
                } else {
                    try {
                        succeeded = differentiable->forward(&encoder, resources, nullptr, tape, context.error);
                    } catch (const std::exception &exception) {
                        context.error = "differentiable pass '" + compute.name() +
                                        "' forward threw an exception: " + exception.what();
                        return VERNON_RHI_STATUS_INTERNAL_ERROR;
                    } catch (...) {
                        context.error =
                            "differentiable pass '" + compute.name() + "' forward threw an unknown exception";
                        return VERNON_RHI_STATUS_INTERNAL_ERROR;
                    }
                }
                if (!succeeded || !tape) {
                    if (context.error.empty())
                        context.error = "differentiable pass '" + compute.name() + "' produced no forward pullback";
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                }
                pullback.observeForwardPeak(differentiable->estimatedForwardPeakBytes(),
                                            context.temporaryBytesAllocated ? context.temporaryBytes : 0);
                if (context.collectMetrics) {
                    uint64_t activeOperations = 0;
                    uint64_t recomputationCost = 0;
                    if (!checkedAdd(pullback.totalActiveOperationCount, tape->activeOperationCount(),
                                    activeOperations) ||
                        !checkedAdd(pullback.totalBaseRecomputationCost, tape->recomputationCost(),
                                    recomputationCost)) {
                        context.error = "graph pullback telemetry overflows";
                        return VERNON_RHI_STATUS_INTERNAL_ERROR;
                    }
                    pullback.totalActiveOperationCount = activeOperations;
                    pullback.totalBaseRecomputationCost = recomputationCost;
                    GraphAutodiffPassTelemetry telemetry;
                    telemetry.scheduleOffset = offset;
                    telemetry.passName = compute.name();
                    telemetry.residualSourceKind = tape->residualSourceKind();
                    telemetry.controlHistoryKind = tape->controlHistoryKind();
                    telemetry.estimatedTapeBytes = tape->estimatedTapeBytes();
                    telemetry.logicalResidualBytes = tape->logicalResidualBytes();
                    telemetry.residentTapeBytes = tape->residentTapeBytes();
                    telemetry.allocatedTapeBytes = tape->allocatedTapeBytes();
                    telemetry.retainedAllocationBytes = tape->retainedAllocationBytes();
                    telemetry.peakTemporaryTapeBytes = tape->peakTemporaryTapeBytes();
                    telemetry.activeOperationCount = tape->activeOperationCount();
                    telemetry.recomputationCost = tape->recomputationCost();
                    pullback.passTelemetryValues.push_back(std::move(telemetry));
                }
                uint64_t budgetTemporaryBytes = context.temporaryBytes;
                uint64_t observedTemporaryBytes = context.temporaryBytesAllocated ? context.temporaryBytes : 0;
                if (!checkedAdd(budgetTemporaryBytes, tape->peakTemporaryTapeBytes(), budgetTemporaryBytes) ||
                    !checkedAdd(observedTemporaryBytes, tape->peakTemporaryTapeBytes(), observedTemporaryBytes)) {
                    context.error = "graph pullback temporary Tape telemetry overflows";
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                }
                if (!pullback.observeTapeAllocation(tape->retainedAllocationBytes(), budgetTemporaryBytes,
                                                    observedTemporaryBytes, context.error))
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                if (context.retainTapes)
                    pullback.tapes[offset] = std::move(tape);
            } else {
                const VernonRhiStatus status = compute.execute(encoder, resources);
                if (status != VERNON_RHI_STATUS_OK) {
                    context.error = "execution pass '" + compute.name() + "' failed during graph VJP forward execution";
                    return status;
                }
                if (context.collectMetrics) {
                    GraphAutodiffPassTelemetry telemetry;
                    telemetry.scheduleOffset = offset;
                    telemetry.passName = compute.name();
                    telemetry.residualSourceKind = "none";
                    telemetry.controlHistoryKind = "none";
                    telemetry.controlHistoryBytes = uint64_t{0};
                    pullback.passTelemetryValues.push_back(std::move(telemetry));
                }
            }
            if (!pullback.publishPassOutputs(offset, context.error))
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            if (!context.captureCheckpoints || !pullback.plan->hasAutodiffCheckpointPlan)
                return VERNON_RHI_STATUS_OK;
            if (pullback.plan->provider == detail::ExecutionProvider::Cpu)
                for (uint32_t resourceIndex : pullback.checkpointResourcesByProducer[offset])
                    if (!pullback.captureCheckpoint(
                            resourceIndex,
                            VernonRhiCommandEncoder{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0},
                            context.error))
                        return VERNON_RHI_STATUS_INTERNAL_ERROR;
            return VERNON_RHI_STATUS_OK;
        };
        const auto planForward = [](void *opaque, uint32_t offset, ComputePass &compute,
                                    const ExecutionResources &resources, detail::RhiCommandExecutionPlan &commands) {
            auto &context = *static_cast<Context *>(opaque);
            DifferentiablePass *differentiable = compute.differentiable();
            if (!differentiable)
                return VERNON_RHI_STATUS_UNSUPPORTED;
            if (!context.pullback.preflightTapeAllocation(differentiable->estimatedForwardPeakBytes(),
                                                          context.temporaryBytes, context.error))
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            std::unique_ptr<PassPullback> tape;
            if (!differentiable->forward(nullptr, resources, &commands, tape, context.error) || !tape)
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            if (commands.commands.nodes.empty()) {
                context.error = "differentiable pass '" + compute.name() + "' produced no forward commands";
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            }
            context.plannedTapes[offset] = std::move(tape);
            return VERNON_RHI_STATUS_OK;
        };
        const auto appendCheckpoints = [](void *opaque, uint32_t offset, uint32_t predecessor,
                                          detail::RhiCommandExecutionPlan &commands) {
            auto &context = *static_cast<Context *>(opaque);
            Impl &pullback = context.pullback;
            if (!context.captureCheckpoints || !pullback.plan->hasAutodiffCheckpointPlan)
                return VERNON_RHI_STATUS_OK;
            struct CheckpointContext {
                Impl &pullback;
                uint32_t resourceIndex;
                std::string &error;
            };
            const auto encode = [](void *opaque, VernonRhiCommandEncoder encoder) {
                auto &checkpoint = *static_cast<CheckpointContext *>(opaque);
                return checkpoint.pullback.captureCheckpoint(checkpoint.resourceIndex, encoder, checkpoint.error)
                           ? VERNON_RHI_STATUS_OK
                           : VERNON_RHI_STATUS_INTERNAL_ERROR;
            };
            for (uint32_t resourceIndex : pullback.checkpointResourcesByProducer[offset]) {
                if (resourceIndex >= pullback.checkpointEntries.size())
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                const CheckpointEntry &entry = pullback.checkpointEntries[resourceIndex];
                if (entry.version.resource >= pullback.plan->resourceRecords.size())
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                const auto &record = pullback.plan->resourceRecords[entry.version.resource];
                detail::CommandNode checkpoint;
                checkpoint.kind = detail::CommandNodeKind::Checkpoint;
                checkpoint.queue = detail::CommandQueueClass::Transfer;
                checkpoint.predecessors = {predecessor};
                checkpoint.accesses = {commandBufferAccess(record.buffer, 0, entry.byteSize, AccessMode::Read),
                                       commandBufferAccess(pullback.checkpointStorage.buffer(), entry.offset,
                                                           entry.byteSize, AccessMode::Write)};
                commands.commands.nodes.push_back(std::move(checkpoint));
                auto retained =
                    std::make_shared<CheckpointContext>(CheckpointContext{pullback, resourceIndex, context.error});
                commands.encoders.push_back({encode, retained.get()});
                commands.retainedContexts.push_back(std::move(retained));
                detail::appendRhiBufferBinding(commands.bindings, record.buffer);
                detail::appendRhiBufferBinding(commands.bindings, pullback.checkpointStorage.buffer());
                predecessor = static_cast<uint32_t>(commands.commands.nodes.size() - 1);
            }
            return VERNON_RHI_STATUS_OK;
        };
        VernonRhiStatus status = VERNON_RHI_STATUS_OK;
        if (plan->provider == detail::ExecutionProvider::Cpu) {
            status = detail::executeCpuScheduleRange(plan->device, plan->resources, plan->passes, plan->schedule,
                                                     bindings, begin, end, execute, &context);
        } else {
            detail::RhiCommandExecutionPlan commands;
            status = detail::submitRhiComputeCommandRange(
                plan->device, plan->resources, plan->resourceRecords, plan->passes, plan->schedule, plan->scopes,
                plan->commandDag, bindings, begin, end, execute, &context, nullptr, appendCheckpoints, &context,
                planForward, &context, sink ? &commands : nullptr);
            if (status == VERNON_RHI_STATUS_OK && sink) {
                status = sink->append(std::move(commands));
                if (status == VERNON_RHI_STATUS_OK)
                    sink->retain(std::move(retainedContext));
            }
        }
        if (status == VERNON_RHI_STATUS_UNSUPPORTED && error.empty())
            error = "native graph VJP does not support render passes";
        else if (status != VERNON_RHI_STATUS_OK && error.empty()) {
            error = "native graph VJP schedule execution failed (RHI status " +
                    std::to_string(static_cast<uint32_t>(status)) + ")";
            const VernonStringView diagnostic = vernonRhiDeviceGetLastError(plan->device);
            if (diagnostic.data && diagnostic.size) {
                error += ": ";
                error.append(diagnostic.data, diagnostic.size);
            }
        }
        return status == VERNON_RHI_STATUS_OK;
    }

    bool captureState(const std::vector<uint32_t> &resourceIds, std::vector<ResourceSnapshot> &output, uint64_t &bytes,
                      std::string &error,
                      const std::vector<std::vector<GraphByteRange>> *declaredRanges = nullptr) const {
        if (declaredRanges && declaredRanges->size() != resourceIds.size()) {
            error = "graph restoration footprint table is inconsistent";
            return false;
        }
        bytes = 0;
        output.clear();
        output.reserve(resourceIds.size());
        for (size_t index = 0; index < resourceIds.size(); ++index) {
            const uint32_t resourceId = resourceIds[index];
            ResourceSnapshot snapshot;
            const std::vector<GraphByteRange> *ranges = declaredRanges ? &(*declaredRanges)[index] : nullptr;
            const auto &record = plan->resourceRecords[resourceId];
            const bool captured =
                plan->provider == detail::ExecutionProvider::Rhi
                    ? record.resource.kind == ResourceKind::Buffer &&
                          captureDeviceResource(plan->device, record.buffer, record.checkpoint, snapshot, error, ranges)
                    : captureResource(record.checkpoint, snapshot, error, ranges);
            if (!captured) {
                if (error.empty())
                    error = "graph state snapshot requires a checkpointable buffer";
                return false;
            }
            uint64_t total = 0;
            if (!checkedAdd(bytes, snapshot.byteSize, total)) {
                error = "graph final-state snapshot size overflows";
                return false;
            }
            bytes = total;
            output.push_back(std::move(snapshot));
        }
        return true;
    }

    bool captureInitialState(std::string &error) {
        if (!plan->autodiffCheckpointPlan.initialStateBytes)
            return true;
        if (!captureState(plan->autodiffInitialResources, initialState, initialStateBytes, error,
                          &plan->autodiffInitialRanges))
            return false;
        if (initialStateBytes != plan->autodiffCheckpointPlan.initialStateBytes) {
            error = "compiled graph initial-state checkpoint size does not match runtime resources";
            return false;
        }
        return true;
    }

    bool restoreInitialState(std::string &error, detail::RhiCommandPlanSink *sink = nullptr) {
        for (size_t index = 0; index < initialState.size(); ++index) {
            const ResourceSnapshot &snapshot = initialState[index];
            if (!snapshot.restore(error, sink))
                return false;
            if (index >= plan->autodiffInitialResources.size() ||
                plan->autodiffInitialResources[index] >= currentResourceEpochs.size()) {
                error = "graph initial-state resource version is invalid";
                return false;
            }
            currentResourceEpochs[plan->autodiffInitialResources[index]] = 0;
        }
        return true;
    }

    bool restoreCheckpoint(uint32_t cutIndex, std::string &error, detail::RhiCommandPlanSink *sink = nullptr) {
        if (cutIndex >= plan->autodiffCheckpointPlan.cuts.size()) {
            error = "checkpoint replay segment has an invalid starting cut";
            return false;
        }
        const auto &checkpointResources = plan->autodiffCheckpointPlan.cuts[cutIndex].checkpointResources;
        for (uint32_t resourceIndex : checkpointResources) {
            if (resourceIndex >= checkpointEntries.size()) {
                error = "checkpoint replay resource index is invalid";
                return false;
            }
            const CheckpointEntry &entry = checkpointEntries[resourceIndex];
            if (!entry.captured) {
                error = "checkpoint replay resource was not captured during forward execution";
                return false;
            }
            if (entry.released) {
                error = "checkpoint replay attempted to read a released resource";
                return false;
            }
            if (entry.version.resource >= currentResourceEpochs.size()) {
                error = "checkpoint replay resource version is invalid";
                return false;
            }
            if (plan->provider == detail::ExecutionProvider::Rhi &&
                plan->resourceRecords[entry.version.resource].resource.kind != ResourceKind::Buffer) {
                error = "device-local graph checkpoint restore requires a buffer resource";
                return false;
            }
        }
        if (plan->provider == detail::ExecutionProvider::Rhi) {
            const bool restored = submitDeviceCommands(
                plan->device,
                [&](VernonRhiCommandEncoder native) {
                    for (uint32_t resourceIndex : checkpointResources) {
                        const CheckpointEntry &entry = checkpointEntries[resourceIndex];
                        const auto &record = plan->resourceRecords[entry.version.resource];
                        if (vernonRhiCommandEncoderCopyBuffer(plan->device, native, checkpointStorage.buffer(),
                                                              entry.offset, record.buffer, 0,
                                                              entry.byteSize) != VERNON_RHI_STATUS_OK)
                            return VERNON_RHI_STATUS_INTERNAL_ERROR;
                    }
                    return VERNON_RHI_STATUS_OK;
                },
                [&] {
                    std::vector<detail::CommandResourceAccess> accesses;
                    accesses.reserve(checkpointResources.size() * 2);
                    for (uint32_t resourceIndex : checkpointResources) {
                        const CheckpointEntry &entry = checkpointEntries[resourceIndex];
                        const auto &record = plan->resourceRecords[entry.version.resource];
                        accesses.push_back(commandBufferAccess(checkpointStorage.buffer(), entry.offset, entry.byteSize,
                                                               AccessMode::Read));
                        accesses.push_back(commandBufferAccess(record.buffer, 0, entry.byteSize, AccessMode::Write));
                    }
                    return accesses;
                }(),
                [&] {
                    std::vector<detail::RhiCommandResourceBinding> bindings;
                    detail::appendRhiBufferBinding(bindings, checkpointStorage.buffer());
                    for (uint32_t resourceIndex : checkpointResources)
                        detail::appendRhiBufferBinding(
                            bindings, plan->resourceRecords[checkpointEntries[resourceIndex].version.resource].buffer);
                    return bindings;
                }(),
                [&] {
                    std::vector<detail::CommandResourceAccess> accesses;
                    accesses.push_back(detail::rhiBufferAccess(
                        checkpointStorage.buffer(), 0, checkpointStorage.size(),
                        checkpointStorageInSourceState ? AccessMode::Read : AccessMode::Write,
                        checkpointStorageInSourceState ? VERNON_RHI_STATE_TRANSFER_SOURCE
                                                       : VERNON_RHI_STATE_TRANSFER_DESTINATION));
                    for (uint32_t resourceIndex : checkpointResources) {
                        const CheckpointEntry &entry = checkpointEntries[resourceIndex];
                        accesses.push_back(detail::rhiBufferAccess(
                            plan->resourceRecords[entry.version.resource].buffer, 0, entry.byteSize,
                            AccessMode::ReadWrite, VERNON_RHI_STATE_SHADER_WRITE, VERNON_RHI_STAGE_COMPUTE));
                    }
                    return accesses;
                }(),
                [&] {
                    std::vector<detail::CommandResourceAccess> accesses;
                    accesses.push_back(detail::rhiBufferAccess(checkpointStorage.buffer(), 0, checkpointStorage.size(),
                                                               AccessMode::Read, VERNON_RHI_STATE_TRANSFER_SOURCE));
                    for (uint32_t resourceIndex : checkpointResources) {
                        const CheckpointEntry &entry = checkpointEntries[resourceIndex];
                        accesses.push_back(detail::rhiBufferAccess(
                            plan->resourceRecords[entry.version.resource].buffer, 0, entry.byteSize,
                            AccessMode::ReadWrite, VERNON_RHI_STATE_SHADER_WRITE, VERNON_RHI_STAGE_COMPUTE));
                    }
                    return accesses;
                }(),
                "device-local graph checkpoint restore failed", error, sink);
            if (!restored)
                return false;
            checkpointStorageInSourceState = true;
        } else {
            for (uint32_t resourceIndex : checkpointResources) {
                const CheckpointEntry &entry = checkpointEntries[resourceIndex];
                const void *source = entry.byteSize ? checkpointStorage.data() + entry.offset : nullptr;
                if (!entry.resource->copyFrom(source, entry.byteSize, error))
                    return false;
            }
        }
        for (uint32_t resourceIndex : checkpointResources) {
            const CheckpointEntry &entry = checkpointEntries[resourceIndex];
            currentResourceEpochs[entry.version.resource] = entry.version.epoch;
        }
        return true;
    }

    bool releaseCheckpoints(const std::vector<uint32_t> &resourceIndices, std::string &error) {
        for (uint32_t resourceIndex : resourceIndices) {
            if (resourceIndex >= checkpointEntries.size()) {
                error = "checkpoint replay release index is invalid";
                return false;
            }
            CheckpointEntry &entry = checkpointEntries[resourceIndex];
            if (!entry.captured || entry.released) {
                error = "checkpoint replay release order is invalid";
                return false;
            }
            // The immutable bytes remain resident for reusable pullbacks. This
            // marker validates one backward application's replay lifetime.
            entry.released = true;
        }
        return true;
    }

    void resetCheckpointLiveness() {
        for (CheckpointEntry &entry : checkpointEntries)
            entry.released = false;
    }

    void discardForwardProgress() {
        eraseTapes(0, static_cast<uint32_t>(plan->schedule.size()));
        for (CheckpointEntry &entry : checkpointEntries) {
            entry.captured = false;
            entry.released = false;
        }
    }

    bool addValue(EndpointValues &values, const DerivativeEndpointKey &endpoint,
                  std::shared_ptr<GraphAutodiffValue> contribution, std::string &error) const {
        const auto found = values.find(endpoint);
        if (found == values.end()) {
            values.emplace(endpoint, std::move(contribution));
            return true;
        }
        auto sum = found->second->add(*contribution, error);
        if (!sum)
            return false;
        found->second = std::move(sum);
        return true;
    }

    bool observeBackwardValues(const EndpointValues &accumulated, const EndpointValues *localValues,
                               const NamedGraphAutodiffValues *namedValues, uint64_t temporaryBytes,
                               bool includeAccumulationTemporary, std::string &error) {
        uint64_t valueBytes = 0;
        std::unordered_set<uintptr_t> identities;
        const auto add = [&](const std::shared_ptr<GraphAutodiffValue> &value) {
            if (!value || !identities.insert(value->logicalIdentity()).second)
                return true;
            return checkedAdd(valueBytes, value->allocationBytes(), valueBytes);
        };
        for (const auto &value : accumulated)
            if (!add(value.second)) {
                error = "graph backward value memory accounting overflows";
                return false;
            }
        if (localValues)
            for (const auto &value : *localValues)
                if (!add(value.second)) {
                    error = "graph backward value memory accounting overflows";
                    return false;
                }
        if (namedValues)
            for (const auto &value : *namedValues)
                if (!add(value.second)) {
                    error = "graph backward value memory accounting overflows";
                    return false;
                }
        if (includeAccumulationTemporary && !checkedAdd(valueBytes, valueBytes, valueBytes)) {
            error = "graph backward value memory accounting overflows";
            return false;
        }
        uint64_t totalTemporaryBytes = 0;
        GraphMemoryAccounting accounting;
        uint64_t managed = 0;
        if (!checkedAdd(temporaryBytes, valueBytes, totalTemporaryBytes) ||
            !memoryAccounting(0, totalTemporaryBytes, accounting) || !accounting.budgetedBytes(managed)) {
            error = "graph backward value memory accounting overflows";
            return false;
        }
        if (plan->hasAutodiffCheckpointPlan && managed > plan->autodiffCheckpointPlan.memoryBudget) {
            error = "graph autodiff backward values exceed the compiled checkpoint memory budget";
            return false;
        }
        observedPeakRuntimeManagedBytes = std::max(observedPeakRuntimeManagedBytes, managed);
        return true;
    }

    bool pullbackApplyOptions(uint64_t temporaryBytes, PassPullbackApplyOptions &options, std::string &error) const {
        if (!plan->hasAutodiffCheckpointPlan) {
            options = {};
            return true;
        }
        uint64_t totalTemporaryBytes = 0;
        GraphMemoryAccounting accounting;
        uint64_t managed = 0;
        if (!checkedAdd(temporaryBytes, plan->autodiffCheckpointPlan.backwardValueBytes, totalTemporaryBytes) ||
            !memoryAccounting(0, totalTemporaryBytes, accounting) || !accounting.budgetedBytes(managed)) {
            error = "graph pullback apply-time budget accounting overflows";
            return false;
        }
        if (managed > plan->autodiffCheckpointPlan.memoryBudget) {
            error = "graph pullback has no remaining compiled checkpoint budget";
            return false;
        }
        const uint64_t remaining = plan->autodiffCheckpointPlan.memoryBudget - managed;
        options.maximumTemporaryBytes = remaining;
        options.maximumReusableConstructionBytes = remaining;
        return true;
    }

    bool applySegment(uint32_t begin, uint32_t end, EndpointValues &accumulated, uint64_t temporaryBytes,
                      std::string &error, detail::RhiCommandPlanSink *sink = nullptr) {
        std::vector<VernonRhiBuffer> buffers(
            plan->resources.size(), VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        ExecutionResources resources(plan->resources, buffers, bindings);
        for (uint32_t offset = end; offset-- > begin;) {
            const uint32_t passIndex = plan->schedule[offset];
            auto *compute = dynamic_cast<ComputePass *>(plan->passes[passIndex].get());
            DifferentiablePass *differentiable = compute ? compute->differentiable() : nullptr;
            if (!differentiable) {
                for (const ResourceUse &use : plan->passes[passIndex]->uses())
                    if (writes(use.access) &&
                        accumulated.find({DerivativeEndpointKind::Resource, use.resource.id}) != accumulated.end()) {
                        error = "non-differentiable pass '" + plan->passes[passIndex]->name() +
                                "' lies on the graph VJP path";
                        return false;
                    }
                continue;
            }
            bool active = false;
            for (const PassDerivativeMapping &mapping : differentiable->cotangentMappings())
                active |= accumulated.find(mapping.endpoint) != accumulated.end();
            if (!active)
                continue;
            NamedGraphAutodiffValues localCotangents;
            EndpointValues localByEndpoint;
            for (const PassDerivativeMapping &mapping : differentiable->cotangentMappings()) {
                auto local = localByEndpoint.find(mapping.endpoint);
                if (local == localByEndpoint.end()) {
                    std::shared_ptr<GraphAutodiffValue> value;
                    const auto found = accumulated.find(mapping.endpoint);
                    if (found != accumulated.end()) {
                        value = std::move(found->second);
                        accumulated.erase(found);
                    } else if (!differentiable->zeroCotangent(mapping.path, resources, value, error)) {
                        return false;
                    }
                    local = localByEndpoint.emplace(mapping.endpoint, std::move(value)).first;
                }
                localCotangents.emplace_back(mapping.path, local->second);
            }
            const auto tape = tapes.find(offset);
            if (tape == tapes.end()) {
                error = "differentiable pass '" + compute->name() + "' produced no retained forward pullback";
                return false;
            }
            NamedGraphAutodiffValues localGradients;
            PassPullbackApplyOptions applyOptions;
            if (!pullbackApplyOptions(temporaryBytes, applyOptions, error))
                return false;
            if (sink)
                for (const auto &cotangent : localCotangents)
                    sink->retain(cotangent.second);
            if (!tape->second->apply(localCotangents, localGradients, applyOptions, sink, error))
                return false;
            if ((!sink &&
                 (!checkedAdd(submissionCount, tape->second->submissionCount(), submissionCount) ||
                  !checkedAdd(waitCount, tape->second->waitCount(), waitCount) ||
                  !checkedAdd(deviceWaitNanoseconds, tape->second->deviceWaitNanoseconds(), deviceWaitNanoseconds))) ||
                !checkedAdd(readbackCount, tape->second->readbackCount(), readbackCount) ||
                !checkedAdd(atomicPublicationCount, tape->second->atomicPublicationCount(), atomicPublicationCount) ||
                !checkedAdd(temporaryAllocationTrafficBytes, tape->second->temporaryAllocationTrafficBytes(),
                            temporaryAllocationTrafficBytes)) {
                error = "graph pullback control-plane telemetry overflows";
                return false;
            }
            const uint64_t peakTemporaryTapeBytes = tape->second->peakTemporaryTapeBytes();
            if (!observeTapeAllocation(0, peakTemporaryTapeBytes, peakTemporaryTapeBytes, error))
                return false;
            const auto telemetry =
                std::find_if(passTelemetryValues.begin(), passTelemetryValues.end(),
                             [&](const GraphAutodiffPassTelemetry &value) { return value.scheduleOffset == offset; });
            if (telemetry != passTelemetryValues.end())
                telemetry->peakTemporaryTapeBytes = std::max(telemetry->peakTemporaryTapeBytes, peakTemporaryTapeBytes);
            if (!observeBackwardValues(accumulated, &localByEndpoint, &localGradients, temporaryBytes, true, error))
                return false;
            std::unordered_map<std::string, std::shared_ptr<GraphAutodiffValue>> gradientsByPath;
            for (auto &gradient : localGradients)
                gradientsByPath.emplace(gradient.first, std::move(gradient.second));
            EndpointValues passContributions;
            std::unordered_map<DerivativeEndpointKey, std::unordered_set<uintptr_t>, DerivativeEndpointHash>
                contributionIdentities;
            for (const PassDerivativeMapping &mapping : differentiable->gradientMappings()) {
                const auto found = gradientsByPath.find(mapping.path);
                if (found == gradientsByPath.end() || !found->second) {
                    error = "pass pullback omitted gradient path '" + mapping.path + "'";
                    return false;
                }
                if (!contributionIdentities[mapping.endpoint].insert(found->second->logicalIdentity()).second)
                    continue;
                if (!addValue(passContributions, mapping.endpoint, found->second, error))
                    return false;
            }
            for (auto &contribution : passContributions)
                if (!addValue(accumulated, contribution.first, std::move(contribution.second), error))
                    return false;
            if (!observeBackwardValues(accumulated, nullptr, nullptr, temporaryBytes, false, error))
                return false;
        }
        return true;
    }

    bool implicitObjective(EndpointValues &accumulated, std::string &error) const {
        if (plan->objectives.size() != 1) {
            error = "implicit graph cotangent requires exactly one objective";
            return false;
        }
        const NamedDerivativeEndpoint &objective = plan->objectives.front();
        std::vector<VernonRhiBuffer> buffers(
            plan->resources.size(), VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        ExecutionResources resources(plan->resources, buffers, bindings);
        for (uint32_t offset = static_cast<uint32_t>(plan->schedule.size()); offset-- > 0;) {
            auto *compute = dynamic_cast<ComputePass *>(plan->passes[plan->schedule[offset]].get());
            DifferentiablePass *differentiable = compute ? compute->differentiable() : nullptr;
            if (!differentiable)
                continue;
            for (const PassDerivativeMapping &mapping : differentiable->cotangentMappings())
                if (mapping.endpoint == objective.endpoint) {
                    std::shared_ptr<GraphAutodiffValue> value;
                    if (!differentiable->implicitCotangent(mapping.path, resources, value, error))
                        return false;
                    accumulated.emplace(objective.endpoint, std::move(value));
                    return true;
                }
        }
        error = "graph objective is not produced by a differentiable pass";
        return false;
    }

    std::shared_ptr<CompiledExecutionGraph::State> plan;
    std::shared_ptr<const ExecutionBindings> bindings;
    ExecutionSubmission forward;
    std::unordered_map<uint32_t, std::unique_ptr<PassPullback>> tapes;
    CheckpointStorage checkpointStorage;
    std::vector<CheckpointEntry> checkpointEntries;
    std::vector<std::vector<uint32_t>> checkpointResourcesByProducer;
    std::vector<uint32_t> currentResourceEpochs;
    std::vector<ResourceSnapshot> initialState;
    uint64_t initialStateBytes{};
    uint64_t observedPeakRuntimeManagedBytes{};
    uint64_t totalActiveOperationCount{};
    uint64_t totalBaseRecomputationCost{};
    uint64_t submissionCount{};
    uint64_t waitCount{};
    uint64_t readbackCount{};
    uint64_t atomicPublicationCount{};
    uint64_t temporaryAllocationTrafficBytes{};
    uint64_t deviceWaitNanoseconds{};
    std::vector<GraphAutodiffPassTelemetry> passTelemetryValues;
    bool retainedTapesAvailable{true};
    bool checkpointStorageInitialized{};
    bool checkpointStorageInSourceState{};
    mutable std::mutex mutex;
};

std::shared_ptr<GraphPullback> CompiledExecutionGraph::vjp(std::shared_ptr<const ExecutionBindings> bindings,
                                                           std::string &error) const {
    error.clear();
    if (state_->differentiableInputs.empty() || state_->objectives.empty()) {
        error = state_->differentiableInputs.empty() ? "compiled graph has no differentiable inputs"
                                                     : "compiled graph has no objectives";
        return {};
    }
    if (!bindings) {
        if (!state_->parameterNames.empty()) {
            error = "compiled execution graph requires parameter bindings";
            return {};
        }
    } else if (bindings->graphIdentity_ != state_->graphIdentity ||
               bindings->values_.size() != state_->parameterNames.size()) {
        error = "execution bindings do not belong to this compiled graph";
        return {};
    }
    std::unique_ptr<GraphPullback::Impl> impl;
    std::vector<ResourceSnapshot> transactionState;
    const auto rollback = [&] {
        if (impl)
            impl->discardForwardProgress();
        for (const ResourceSnapshot &snapshot : transactionState) {
            try {
                std::string restoreError;
                if (!snapshot.restore(restoreError)) {
                    if (restoreError.empty())
                        restoreError = "graph VJP forward rollback failed";
                    if (error.empty())
                        error = std::move(restoreError);
                    else
                        error += "; rollback failed: " + restoreError;
                }
            } catch (const std::exception &exception) {
                if (error.empty())
                    error = std::string("graph VJP forward rollback threw an exception: ") + exception.what();
                else
                    error += std::string("; rollback threw an exception: ") + exception.what();
            } catch (...) {
                if (error.empty())
                    error = "graph VJP forward rollback threw an unknown exception";
                else
                    error += "; rollback threw an unknown exception";
            }
        }
    };
    try {
        auto forwardImpl = std::make_unique<ExecutionSubmission::Impl>(state_, bindings);
        forwardImpl->status = VERNON_RHI_STATUS_OK;
        forwardImpl->state = ExecutionSubmission::State::Succeeded;
        impl = std::make_unique<GraphPullback::Impl>(state_, std::move(bindings),
                                                     ExecutionSubmission(std::move(forwardImpl)));
        if (!impl->initializeCheckpointStorage(error) || !impl->captureInitialState(error)) {
            rollback();
            return {};
        }
        uint64_t transactionBytes = 0;
        if (!impl->captureState(state_->autodiffTransactionResources, transactionState, transactionBytes, error,
                                &state_->autodiffTransactionRanges)) {
            rollback();
            return {};
        }
        uint32_t retainBegin = 0;
        if (state_->hasAutodiffCheckpointPlan) {
            if (state_->autodiffCheckpointPlan.replaySegments.empty()) {
                error = "compiled graph checkpoint plan has no replay segments";
                rollback();
                return {};
            }
            if (transactionBytes != state_->autodiffCheckpointPlan.transactionBytes) {
                error = "graph VJP transaction storage does not match its compiled checkpoint plan";
                rollback();
                return {};
            }
            retainBegin = state_->autodiffCheckpointPlan.replaySegments.back().beginStep;
        }
        if (!impl->executeRange(0, retainBegin, false, state_->hasAutodiffCheckpointPlan, error, true, transactionBytes,
                                true) ||
            !impl->executeRange(retainBegin, static_cast<uint32_t>(state_->schedule.size()), true,
                                state_->hasAutodiffCheckpointPlan, error, true, transactionBytes, true)) {
            rollback();
            return {};
        }
        impl->observeManagedBytes(transactionBytes);
        auto result = std::shared_ptr<GraphPullback>(new GraphPullback(std::move(impl)));
        transactionState.clear();
        return result;
    } catch (const std::exception &exception) {
        error = std::string("graph VJP forward threw an exception: ") + exception.what();
    } catch (...) {
        error = "graph VJP forward threw an unknown exception";
    }
    rollback();
    return {};
}

GraphPullback::GraphPullback(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GraphPullback::~GraphPullback() = default;

ExecutionSubmission &GraphPullback::forwardSubmission() { return impl_->forward; }
const ExecutionSubmission &GraphPullback::forwardSubmission() const { return impl_->forward; }

std::shared_ptr<GraphBackwardSubmission> GraphPullback::submit(const NamedGraphAutodiffValues &cotangents,
                                                               bool implicit) {
    std::lock_guard lock(impl_->mutex);
    auto submission = std::make_shared<GraphBackwardSubmission>();
    std::vector<ResourceSnapshot> finalState;
    try {
        if (!impl_->retainedTapesAvailable) {
            submission->error_ = "graph pullback is no longer reusable after failed checkpoint replay";
            submission->state_ = GraphBackwardSubmission::State::Failed;
            return submission;
        }
        impl_->submissionCount = 0;
        impl_->waitCount = 0;
        impl_->readbackCount = 0;
        impl_->atomicPublicationCount = 0;
        impl_->temporaryAllocationTrafficBytes = 0;
        impl_->deviceWaitNanoseconds = 0;
        EndpointValues accumulated;
        if (implicit) {
            if (!impl_->implicitObjective(accumulated, submission->error_)) {
                submission->state_ = GraphBackwardSubmission::State::Failed;
                return submission;
            }
        } else {
            if (cotangents.size() != impl_->plan->objectives.size()) {
                submission->error_ = "graph pullback requires exactly one cotangent per objective";
                submission->state_ = GraphBackwardSubmission::State::Failed;
                return submission;
            }
            std::unordered_map<std::string, std::shared_ptr<GraphAutodiffValue>> supplied;
            for (const auto &value : cotangents)
                supplied.emplace(value.first, value.second);
            for (const NamedDerivativeEndpoint &objective : impl_->plan->objectives) {
                const auto found = supplied.find(objective.name);
                if (found == supplied.end()) {
                    submission->error_ = "graph pullback requires exactly one cotangent per objective";
                    submission->state_ = GraphBackwardSubmission::State::Failed;
                    return submission;
                }
                accumulated.emplace(objective.endpoint, found->second);
            }
        }

        uint64_t finalStateBytes = 0;
        if (impl_->plan->hasAutodiffCheckpointPlan && impl_->plan->autodiffCheckpointPlan.replaySegments.size() > 1) {
            impl_->resetCheckpointLiveness();
            if (!impl_->captureState(impl_->plan->autodiffRestorationResources, finalState, finalStateBytes,
                                     submission->error_, &impl_->plan->autodiffRestorationRanges)) {
                submission->state_ = GraphBackwardSubmission::State::Failed;
                return submission;
            }
            if (finalStateBytes != impl_->plan->autodiffCheckpointPlan.restorationBytes) {
                submission->error_ = "graph final-state storage does not match its compiled checkpoint plan";
                submission->state_ = GraphBackwardSubmission::State::Failed;
                return submission;
            }
            impl_->observeManagedBytes(finalStateBytes);
        }
        bool succeeded = true;
        std::unique_ptr<BackwardCommandProgram> commandProgram;
        if (impl_->plan->provider != detail::ExecutionProvider::Cpu)
            commandProgram =
                std::make_unique<BackwardCommandProgram>(impl_->plan->device, impl_->submissionCount, impl_->waitCount,
                                                         impl_->deviceWaitNanoseconds, submission->error_);
        const auto flushCommands = [&] { return !commandProgram || commandProgram->flush() == VERNON_RHI_STATUS_OK; };
        if (!impl_->plan->hasAutodiffCheckpointPlan) {
            succeeded = impl_->applySegment(0, static_cast<uint32_t>(impl_->plan->schedule.size()), accumulated, 0,
                                            submission->error_, commandProgram.get());
        } else {
            const auto &segments = impl_->plan->autodiffCheckpointPlan.replaySegments;
            for (uint32_t segmentIndex = static_cast<uint32_t>(segments.size()); succeeded && segmentIndex-- > 0;) {
                const AutodiffReplaySegment &segment = segments[segmentIndex];
                if (segmentIndex + 1 != segments.size()) {
                    if (!flushCommands() || !impl_->restoreInitialState(submission->error_, commandProgram.get()) ||
                        (segment.beginStep && !impl_->restoreCheckpoint(segment.checkpointIndex, submission->error_,
                                                                        commandProgram.get()))) {
                        succeeded = false;
                        break;
                    }
                    impl_->eraseTapes(segment.beginStep, segment.endStep);
                    succeeded = impl_->executeRange(segment.beginStep, segment.endStep, true, false, submission->error_,
                                                    false, finalStateBytes, true, commandProgram.get());
                    if (succeeded)
                        succeeded = flushCommands();
                    impl_->observeManagedBytes(finalStateBytes);
                }
                if (succeeded)
                    succeeded = impl_->applySegment(segment.beginStep, segment.endStep, accumulated, finalStateBytes,
                                                    submission->error_, commandProgram.get());
                if (succeeded)
                    succeeded = impl_->releaseCheckpoints(segment.releaseCheckpointResources, submission->error_);
                if (segments.size() > 1) {
                    impl_->eraseTapes(segment.beginStep, segment.endStep);
                    impl_->retainedTapesAvailable = false;
                }
            }
        }
        if (succeeded && commandProgram)
            for (auto &[endpoint, value] : accumulated) {
                (void)endpoint;
                if (!value->materialize(commandProgram.get(), submission->error_)) {
                    succeeded = false;
                    break;
                }
            }
        if (succeeded)
            succeeded = flushCommands();
        impl_->resetCheckpointLiveness();
        if (impl_->plan->hasAutodiffCheckpointPlan && !impl_->retainedTapesAvailable) {
            const auto &segments = impl_->plan->autodiffCheckpointPlan.replaySegments;
            if (segments.size() > 1) {
                const AutodiffReplaySegment &retainedSegment = segments.back();
                std::string reconstructionError;
                if (!impl_->restoreInitialState(reconstructionError, commandProgram.get()) ||
                    !impl_->restoreCheckpoint(retainedSegment.checkpointIndex, reconstructionError,
                                              commandProgram.get()) ||
                    !impl_->executeRange(retainedSegment.beginStep, retainedSegment.endStep, true, false,
                                         reconstructionError, false, finalStateBytes, true, commandProgram.get()) ||
                    !flushCommands()) {
                    if (succeeded) {
                        succeeded = false;
                        submission->error_ = std::move(reconstructionError);
                    }
                } else {
                    impl_->observeManagedBytes(finalStateBytes);
                    impl_->retainedTapesAvailable = true;
                }
            }
        }
        for (const ResourceSnapshot &snapshot : finalState) {
            std::string restoreError;
            bool restored = false;
            try {
                restored = snapshot.restore(restoreError);
            } catch (const std::exception &exception) {
                restoreError = std::string("graph final-state restoration threw an exception: ") + exception.what();
            } catch (...) {
                restoreError = "graph final-state restoration threw an unknown exception";
            }
            if (!restored) {
                if (restoreError.empty())
                    restoreError = "graph final-state restoration failed";
                impl_->retainedTapesAvailable = false;
                if (succeeded) {
                    succeeded = false;
                    submission->error_ = std::move(restoreError);
                } else
                    submission->error_ += "; final-state restoration failed: " + restoreError;
            }
        }
        if (succeeded) {
            NamedGraphAutodiffValues gradients;
            gradients.reserve(impl_->plan->differentiableInputs.size());
            for (const NamedDerivativeEndpoint &input : impl_->plan->differentiableInputs) {
                const auto found = accumulated.find(input.endpoint);
                if (found == accumulated.end()) {
                    succeeded = false;
                    submission->error_ =
                        "differentiable input '" + input.name + "' is disconnected from every objective";
                    break;
                }
                gradients.emplace_back(input.name, found->second);
            }
            if (succeeded)
                submission->gradients_ = std::move(gradients);
        }
        submission->state_ =
            succeeded ? GraphBackwardSubmission::State::Succeeded : GraphBackwardSubmission::State::Failed;
        return submission;
    } catch (const std::exception &exception) {
        impl_->retainedTapesAvailable = false;
        impl_->resetCheckpointLiveness();
        submission->gradients_.clear();
        submission->error_ = std::string("graph pullback application threw an exception: ") + exception.what();
    } catch (...) {
        impl_->retainedTapesAvailable = false;
        impl_->resetCheckpointLiveness();
        submission->gradients_.clear();
        submission->error_ = "graph pullback application threw an unknown exception";
    }
    for (const ResourceSnapshot &snapshot : finalState) {
        try {
            std::string restoreError;
            if (!snapshot.restore(restoreError)) {
                if (restoreError.empty())
                    restoreError = "graph final-state restoration failed";
                submission->error_ += "; final-state restoration failed: " + restoreError;
            }
        } catch (const std::exception &exception) {
            submission->error_ += std::string("; final-state restoration threw an exception: ") + exception.what();
        } catch (...) {
            submission->error_ += "; final-state restoration threw an unknown exception";
        }
    }
    submission->state_ = GraphBackwardSubmission::State::Failed;
    return submission;
}

uint64_t GraphPullback::estimatedTapeBytes() const {
    std::lock_guard lock(impl_->mutex);
    uint64_t result = 0;
    for (const auto &tape : impl_->tapes)
        if (!checkedAdd(result, tape.second->estimatedTapeBytes(), result))
            return std::numeric_limits<uint64_t>::max();
    return result;
}

uint64_t GraphPullback::logicalResidualBytes() const {
    std::lock_guard lock(impl_->mutex);
    uint64_t result = 0;
    for (const auto &tape : impl_->tapes)
        if (!checkedAdd(result, tape.second->logicalResidualBytes(), result))
            return std::numeric_limits<uint64_t>::max();
    return result;
}

uint64_t GraphPullback::residentTapeBytes() const {
    std::lock_guard lock(impl_->mutex);
    uint64_t result = 0;
    for (const auto &tape : impl_->tapes)
        if (!checkedAdd(result, tape.second->residentTapeBytes(), result))
            return std::numeric_limits<uint64_t>::max();
    return result;
}

uint64_t GraphPullback::allocatedTapeBytes() const {
    std::lock_guard lock(impl_->mutex);
    uint64_t result = 0;
    for (const auto &tape : impl_->tapes)
        if (!checkedAdd(result, tape.second->allocatedTapeBytes(), result))
            return std::numeric_limits<uint64_t>::max();
    return result;
}

uint64_t GraphPullback::retainedAllocationBytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->retainedAllocationBytesUnlocked();
}

uint64_t GraphPullback::checkpointBytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->checkpointStorage.size();
}

uint64_t GraphPullback::peakRuntimeManagedBytes() const {
    std::lock_guard lock(impl_->mutex);
    return std::max(impl_->observedPeakRuntimeManagedBytes, impl_->retainedAllocationBytesUnlocked());
}

uint64_t GraphPullback::submissionCount() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->submissionCount;
}

uint64_t GraphPullback::waitCount() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->waitCount;
}

uint64_t GraphPullback::readbackCount() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->readbackCount;
}

uint64_t GraphPullback::atomicPublicationCount() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->atomicPublicationCount;
}

uint64_t GraphPullback::temporaryAllocationTrafficBytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->temporaryAllocationTrafficBytes;
}

uint64_t GraphPullback::deviceWaitNanoseconds() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->deviceWaitNanoseconds;
}

uint64_t GraphPullback::tapeContextLimitBytes() const {
    std::lock_guard lock(impl_->mutex);
    uint64_t limit = 0;
    for (const auto &tape : impl_->tapes) {
        const uint64_t candidate = tape.second->tapeContextLimitBytes();
        if (candidate)
            limit = limit ? std::min(limit, candidate) : candidate;
    }
    return limit;
}

double GraphPullback::recomputationFactor() const {
    std::lock_guard lock(impl_->mutex);
    const uint64_t replayCost =
        impl_->plan->hasAutodiffCheckpointPlan ? impl_->plan->autodiffCheckpointPlan.replayCost : 0;
    return 1.0 + (static_cast<double>(impl_->totalBaseRecomputationCost) + static_cast<double>(replayCost)) /
                     std::max<uint64_t>(impl_->totalActiveOperationCount, 1);
}

std::vector<GraphAutodiffPassTelemetry> GraphPullback::passTelemetry() const {
    std::lock_guard lock(impl_->mutex);
    std::vector<GraphAutodiffPassTelemetry> result = impl_->passTelemetryValues;
    std::sort(result.begin(), result.end(),
              [](const GraphAutodiffPassTelemetry &left, const GraphAutodiffPassTelemetry &right) {
                  return left.scheduleOffset < right.scheduleOffset;
              });
    if (impl_->plan->hasAutodiffCheckpointPlan) {
        const AutodiffDagCheckpointPlan &checkpointPlan = impl_->plan->autodiffCheckpointPlan;
        std::vector<const AutodiffCheckpointResource *> resources;
        resources.reserve(checkpointPlan.checkpointResources.size());
        for (const AutodiffCheckpointResource &resource : checkpointPlan.checkpointResources)
            resources.push_back(&resource);
        std::sort(resources.begin(), resources.end(),
                  [](const AutodiffCheckpointResource *left, const AutodiffCheckpointResource *right) {
                      return left->offset < right->offset;
                  });
        for (size_t index = 0; index < resources.size(); ++index) {
            const AutodiffCheckpointResource &resource = *resources[index];
            const uint64_t end =
                index + 1 < resources.size() ? resources[index + 1]->offset : checkpointPlan.persistentCheckpointBytes;
            const uint64_t chargedBytes = end >= resource.offset ? end - resource.offset : 0;
            auto telemetry = std::find_if(result.begin(), result.end(), [&](const GraphAutodiffPassTelemetry &item) {
                return item.scheduleOffset == resource.producer;
            });
            if (telemetry != result.end() &&
                !checkedAdd(telemetry->checkpointBytes, chargedBytes, telemetry->checkpointBytes))
                telemetry->checkpointBytes = std::numeric_limits<uint64_t>::max();
        }
    }
    return result;
}

GraphBackwardSubmission::State GraphBackwardSubmission::state() const { return state_; }

bool GraphBackwardSubmission::wait(std::string &error) {
    error = error_;
    return state_ == State::Succeeded;
}

const NamedGraphAutodiffValues &GraphBackwardSubmission::gradients() const { return gradients_; }

} // namespace vernon::execution
