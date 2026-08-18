#include "VernonRHI.h"

#include "logical_resource_record.h"
#include "rhi_internal.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct EncoderSlot {
    struct Cleanup {
        void *context{};
        uint64_t object{};
        void (*function)(void *, uint64_t){};
    };
    struct RetainedResource {
        vernon::rhi::ResourceKind kind{};
        uint64_t key{};

        bool operator==(const RetainedResource &other) const { return kind == other.kind && key == other.key; }
    };
    struct RetainedResourceHash {
        size_t operator()(const RetainedResource &resource) const {
            return std::hash<uint64_t>{}(resource.key) ^
                   (std::hash<uint32_t>{}(static_cast<uint32_t>(resource.kind)) << 1);
        }
    };
    struct ActionIdentity {
        void *context{};
        void (*function)(void *, uint64_t){};

        bool operator==(const ActionIdentity &other) const {
            return context == other.context && function == other.function;
        }
    };
    struct ActionIdentityHash {
        size_t operator()(const ActionIdentity &action) const {
            return std::hash<void *>{}(action.context) ^
                   (std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(action.function)) << 1);
        }
    };
    struct ColorOperation {
        uint32_t location{};
        VernonRhiLoadOperation load{VERNON_RHI_LOAD_PRESERVE};
        VernonRhiStoreOperation store{VERNON_RHI_STORE_PRESERVE};
        std::array<float, 4> clear{};
    };
    struct DepthOperation {
        VernonRhiLoadOperation depthLoad{VERNON_RHI_LOAD_PRESERVE};
        VernonRhiStoreOperation depthStore{VERNON_RHI_STORE_PRESERVE};
        VernonRhiLoadOperation stencilLoad{VERNON_RHI_LOAD_DISCARD};
        VernonRhiStoreOperation stencilStore{VERNON_RHI_STORE_DISCARD};
        float clearDepth{1.0f};
        uint32_t clearStencil{};
        bool present{};
    };

    std::mutex mutex;
    VernonRhiDevice device{};
    VernonRhiBackend backend{VERNON_RHI_BACKEND_CUDA};
    uint64_t native{};
    VernonRhiCommandEncoderStats stats{};
    int32_t renderX{};
    int32_t renderY{};
    uint32_t renderWidth{};
    uint32_t renderHeight{};
    uint32_t renderLayers{};
    std::array<uint64_t, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colorTargets{};
    std::array<uint64_t, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colorResources{};
    uint64_t depthTarget{};
    uint64_t depthResource{};
    uint32_t generation{1};
    uint32_t backendRendering{};
    uint64_t backendRenderingObject{};
    bool alive{true};
    bool initializing{true};
    bool busy{};
    bool rendering{};
    bool hasRenderingDescriptor{};
    bool finished{};
    bool failed{};
    bool exclusiveDeviceSlot{};
    std::vector<ColorOperation> colorOperations;
    DepthOperation depthOperation;
    std::vector<Cleanup> cleanups;
    std::vector<Cleanup> rollbacks;
    std::unordered_set<ActionIdentity, ActionIdentityHash> cleanupIdentities;
    std::unordered_set<ActionIdentity, ActionIdentityHash> rollbackIdentities;
    std::unordered_set<RetainedResource, RetainedResourceHash> retainedResources;
    std::unordered_set<RetainedResource, RetainedResourceHash> pendingWriteResources;
    uint64_t admittedUploadBytes{};
    bool unknownPendingWrites{};
};

struct CompletionSlot {
    std::mutex mutex;
    std::condition_variable condition;
    VernonRhiDevice device{};
    VernonRhiBackend backend{VERNON_RHI_BACKEND_CUDA};
    uint64_t native{};
    VernonRhiCommandEncoderStats stats{};
    VernonRhiCompletionState state{VERNON_RHI_COMPLETION_PENDING};
    VernonRhiStatus result{VERNON_RHI_STATUS_OK};
    bool externalSignalRequired{};
    bool backendRetirementRequired{};
    bool observing{};
    bool finalized{};
    bool admissionHeld{};
    std::vector<EncoderSlot::Cleanup> cleanups;
    std::unordered_set<EncoderSlot::RetainedResource, EncoderSlot::RetainedResourceHash> retainedResources;
    uint64_t admittedUploadBytes{};
};

std::mutex registryMutex;
std::vector<std::shared_ptr<EncoderSlot>> encoderSlots;
std::vector<uint32_t> encoderGenerations;
std::vector<uint32_t> freeEncoderIndices;
std::unordered_map<uint64_t, size_t> activeEncoderCounts;
std::vector<std::shared_ptr<CompletionSlot>> completionSlots;
std::vector<uint32_t> completionGenerations;
std::vector<uint32_t> freeCompletionIndices;
std::unordered_map<uint64_t, size_t> inFlightSubmissionCounts;
std::unordered_map<uint64_t, uint64_t> admittedUploadBytes;
std::unordered_map<uint64_t, VernonRhiCommandLimits> commandLimits;
constexpr VernonRhiCommandLimits defaultCommandLimits{
    sizeof(VernonRhiCommandLimits), 8, 8, 64, 64u * 1024u * 1024u, {0, 0, 0}};

template <typename Handle> bool validHandle(Handle handle) {
    return handle.index != VERNON_RHI_INVALID_HANDLE_INDEX && handle.generation != 0;
}

template <typename Handle> uint64_t logicalResourceKey(Handle handle) {
    return validHandle(handle) ? vernon::rhi::encodeResourceKey(handle) : 0;
}

bool sameDevice(VernonRhiDevice left, VernonRhiDevice right) {
    return left.index == right.index && left.generation == right.generation;
}

uint64_t deviceKey(VernonRhiDevice device) { return vernon::rhi::encodeResourceKey(device); }

VernonRhiCommandLimits commandLimitsLocked(VernonRhiDevice device) {
    const auto found = commandLimits.find(deviceKey(device));
    return found == commandLimits.end() ? defaultCommandLimits : found->second;
}

VernonRhiStatus reserveSubmission(VernonRhiDevice device) {
    std::lock_guard<std::mutex> guard(registryMutex);
    const uint64_t key = deviceKey(device);
    const auto found = inFlightSubmissionCounts.find(key);
    if (found != inFlightSubmissionCounts.end() &&
        found->second >= commandLimitsLocked(device).max_in_flight_submissions)
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    try {
        ++inFlightSubmissionCounts[key];
    } catch (const std::bad_alloc &) {
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_OK;
}

void releaseSubmission(VernonRhiDevice device) {
    std::lock_guard<std::mutex> guard(registryMutex);
    const auto found = inFlightSubmissionCounts.find(deviceKey(device));
    if (found != inFlightSubmissionCounts.end() && found->second && --found->second == 0)
        inFlightSubmissionCounts.erase(found);
}

VernonRhiStatus reserveUploadBytes(const std::shared_ptr<EncoderSlot> &slot, uint64_t size) {
    std::lock_guard<std::mutex> slotGuard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> registryGuard(registryMutex);
    const uint64_t key = deviceKey(slot->device);
    const auto found = admittedUploadBytes.find(key);
    const uint64_t current = found == admittedUploadBytes.end() ? 0 : found->second;
    const uint64_t maximum = commandLimitsLocked(slot->device).max_upload_bytes;
    if (size > maximum || current > maximum - size)
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    try {
        admittedUploadBytes[key] = current + size;
    } catch (const std::bad_alloc &) {
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    slot->admittedUploadBytes += size;
    return VERNON_RHI_STATUS_OK;
}

void releaseUploadBytes(VernonRhiDevice device, uint64_t size) {
    if (!size)
        return;
    std::lock_guard<std::mutex> guard(registryMutex);
    const auto found = admittedUploadBytes.find(deviceKey(device));
    if (found == admittedUploadBytes.end())
        return;
    if (found->second <= size)
        admittedUploadBytes.erase(found);
    else
        found->second -= size;
}

void releaseEncoderUploadBytes(const std::shared_ptr<EncoderSlot> &slot, uint64_t size) {
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (size > slot->admittedUploadBytes)
            size = slot->admittedUploadBytes;
        slot->admittedUploadBytes -= size;
    }
    releaseUploadBytes(slot->device, size);
}

std::shared_ptr<EncoderSlot> lookup(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    std::lock_guard<std::mutex> guard(registryMutex);
    if (encoder.index >= encoderSlots.size() || encoder.index >= encoderGenerations.size() ||
        encoderGenerations[encoder.index] != encoder.generation)
        return {};
    const auto &slot = encoderSlots[encoder.index];
    return slot && sameDevice(slot->device, device) ? slot : std::shared_ptr<EncoderSlot>{};
}

std::shared_ptr<EncoderSlot> lookupKey(VernonRhiDevice device, uint64_t key) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return {};
    return lookup(device, {static_cast<uint32_t>(encodedIndex - 1), generation});
}

std::shared_ptr<CompletionSlot> lookup(VernonRhiDevice device, VernonRhiCompletion completion) {
    std::lock_guard<std::mutex> guard(registryMutex);
    if (completion.index >= completionSlots.size() || completion.index >= completionGenerations.size() ||
        completionGenerations[completion.index] != completion.generation)
        return {};
    const auto &slot = completionSlots[completion.index];
    return slot && sameDevice(slot->device, device) ? slot : std::shared_ptr<CompletionSlot>{};
}

bool registerCompletion(const std::shared_ptr<CompletionSlot> &slot, VernonRhiCompletion &completion) {
    std::lock_guard<std::mutex> guard(registryMutex);
    size_t liveCompletions = 0;
    for (const auto &candidate : completionSlots)
        if (candidate && sameDevice(candidate->device, slot->device))
            ++liveCompletions;
    if (liveCompletions >= commandLimitsLocked(slot->device).max_live_completions)
        return false;
    uint32_t index{};
    if (freeCompletionIndices.empty()) {
        index = static_cast<uint32_t>(completionSlots.size());
        try {
            completionSlots.push_back(slot);
            completionGenerations.push_back(1);
        } catch (...) {
            if (completionSlots.size() > index)
                completionSlots.pop_back();
            throw;
        }
    } else {
        index = freeCompletionIndices.back();
        freeCompletionIndices.pop_back();
        completionSlots[index] = slot;
    }
    completion = {index, completionGenerations[index]};
    return true;
}

void unregisterCompletion(uint32_t index, const std::shared_ptr<CompletionSlot> &slot) {
    std::lock_guard<std::mutex> guard(registryMutex);
    if (index >= completionSlots.size() || completionSlots[index] != slot)
        return;
    completionSlots[index].reset();
    uint32_t &generation = completionGenerations[index];
    if (++generation == 0)
        generation = 1;
    freeCompletionIndices.push_back(index);
}

VernonRhiStatus observeCompletion(const std::shared_ptr<CompletionSlot> &slot) {
    std::vector<EncoderSlot::Cleanup> cleanups;
    std::unordered_set<EncoderSlot::RetainedResource, EncoderSlot::RetainedResourceHash> resources;
    bool retireBackend{};
    bool backendCompleted{true};
    bool releaseAdmission{};
    uint64_t uploadBytes{};
    {
        std::unique_lock lock(slot->mutex);
        slot->condition.wait(lock, [&] {
            return !slot->observing && (!slot->externalSignalRequired || slot->state != VERNON_RHI_COMPLETION_PENDING);
        });
        if (slot->finalized)
            return slot->result;
        slot->observing = true;
        retireBackend = slot->backendRetirementRequired;
    }
    if (retireBackend)
        backendCompleted = vernon::rhi::completeCommandRecording(slot->device, slot->native);
    {
        std::lock_guard lock(slot->mutex);
        if (!backendCompleted) {
            slot->state = VERNON_RHI_COMPLETION_FAILED;
            slot->result = VERNON_RHI_STATUS_INTERNAL_ERROR;
        } else if (slot->state == VERNON_RHI_COMPLETION_PENDING) {
            slot->state = VERNON_RHI_COMPLETION_SUCCEEDED;
        }
        cleanups = std::move(slot->cleanups);
        resources = std::move(slot->retainedResources);
        uploadBytes = std::exchange(slot->admittedUploadBytes, 0);
        slot->backendRetirementRequired = false;
        slot->finalized = true;
        slot->observing = false;
        releaseAdmission = std::exchange(slot->admissionHeld, false);
    }
    slot->condition.notify_all();
    for (auto cleanup = cleanups.rbegin(); cleanup != cleanups.rend(); ++cleanup)
        cleanup->function(cleanup->context, cleanup->object);
    for (const auto &resource : resources)
        vernon::rhi::releaseResource(slot->device, resource.kind, resource.key);
    releaseUploadBytes(slot->device, uploadBytes);
    if (releaseAdmission)
        releaseSubmission(slot->device);
    return slot->result;
}

void unregisterEncoder(uint32_t index, const std::shared_ptr<EncoderSlot> &slot) {
    std::lock_guard<std::mutex> guard(registryMutex);
    if (index >= encoderSlots.size() || encoderSlots[index] != slot)
        return;
    encoderSlots[index].reset();
    uint32_t &generation = encoderGenerations[index];
    if (++generation == 0)
        generation = 1;
    freeEncoderIndices.push_back(index);
    const auto active = activeEncoderCounts.find(deviceKey(slot->device));
    if (active != activeEncoderCounts.end() && active->second && --active->second == 0)
        activeEncoderCounts.erase(active);
}

bool validLoad(VernonRhiLoadOperation operation) {
    return operation >= VERNON_RHI_LOAD_CLEAR && operation <= VERNON_RHI_LOAD_DISCARD;
}

bool validStore(VernonRhiStoreOperation operation) {
    return operation >= VERNON_RHI_STORE_PRESERVE && operation <= VERNON_RHI_STORE_DISCARD;
}

bool validRendering(const VernonRhiRenderingDescriptor &descriptor) {
    if (!descriptor.width || !descriptor.height || !descriptor.layers ||
        descriptor.color_attachment_count > VERNON_RHI_MAX_COLOR_ATTACHMENTS ||
        (descriptor.color_attachment_count && !descriptor.color_attachments))
        return false;
    for (size_t index = 0; index < descriptor.color_attachment_count; ++index) {
        const auto &attachment = descriptor.color_attachments[index];
        if (!validHandle(attachment.view) || attachment.location >= 8 || !validLoad(attachment.load_operation) ||
            !validStore(attachment.store_operation))
            return false;
        for (size_t previous = 0; previous < index; ++previous)
            if (descriptor.color_attachments[previous].location == attachment.location)
                return false;
    }
    if (descriptor.depth_stencil_attachment) {
        const auto &attachment = *descriptor.depth_stencil_attachment;
        if (!validHandle(attachment.view) || !validLoad(attachment.depth_load_operation) ||
            !validStore(attachment.depth_store_operation) || !validLoad(attachment.stencil_load_operation) ||
            !validStore(attachment.stencil_store_operation) || attachment.clear_depth < 0.0f ||
            attachment.clear_depth > 1.0f)
            return false;
    }
    return descriptor.color_attachment_count != 0 || descriptor.depth_stencil_attachment;
}

} // namespace

bool vernon::rhi::deviceHasActiveCommandEncoder(VernonRhiDevice device) {
    std::lock_guard<std::mutex> guard(registryMutex);
    const auto found = activeEncoderCounts.find(deviceKey(device));
    return found != activeEncoderCounts.end() && found->second != 0;
}

extern "C" VernonRhiStatus vernonRhiDeviceSetCommandLimits(VernonRhiDevice device,
                                                           const VernonRhiCommandLimits *limits) {
    if (!limits || limits->struct_size < sizeof(*limits) || !limits->max_active_recordings ||
        !limits->max_in_flight_submissions || !limits->max_live_completions || !limits->max_upload_bytes ||
        !vernon::rhi::deviceExists(device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(registryMutex);
    const uint64_t key = deviceKey(device);
    const auto active = activeEncoderCounts.find(key);
    const auto submitted = inFlightSubmissionCounts.find(key);
    const auto uploads = admittedUploadBytes.find(key);
    if ((active != activeEncoderCounts.end() && active->second) ||
        (submitted != inFlightSubmissionCounts.end() && submitted->second) ||
        (uploads != admittedUploadBytes.end() && uploads->second))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    for (const auto &completion : completionSlots)
        if (completion && sameDevice(completion->device, device))
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    try {
        VernonRhiCommandLimits normalized = *limits;
        normalized.struct_size = sizeof(normalized);
        std::fill(std::begin(normalized.reserved), std::end(normalized.reserved), uint64_t{0});
        commandLimits[key] = normalized;
    } catch (const std::bad_alloc &) {
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceGetCommandLimits(VernonRhiDevice device, VernonRhiCommandLimits *output) {
    if (!output || output->struct_size < sizeof(*output) || !vernon::rhi::deviceExists(device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(registryMutex);
    *output = commandLimitsLocked(device);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateCommandEncoder(VernonRhiDevice device,
                                                               const VernonRhiCommandEncoderDescriptor *descriptor,
                                                               VernonRhiCommandEncoder *output) {
    if (output)
        *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !vernon::rhi::deviceExists(device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;

    std::shared_ptr<EncoderSlot> slot;
    uint32_t index{};
    try {
        slot = std::make_shared<EncoderSlot>();
        slot->device = device;
        slot->exclusiveDeviceSlot =
            !(vernon::rhi::deviceCommandCapabilities(device) & vernon::rhi::BackendCommandIndependentRecording);
        std::lock_guard<std::mutex> guard(registryMutex);
        const auto active = activeEncoderCounts.find(deviceKey(device));
        if (active != activeEncoderCounts.end() && active->second) {
            if (slot->exclusiveDeviceSlot)
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            if (active->second >= commandLimitsLocked(device).max_active_recordings)
                return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
        }
        bool appended = false;
        if (freeEncoderIndices.empty()) {
            index = static_cast<uint32_t>(encoderSlots.size());
            encoderSlots.emplace_back();
            try {
                encoderGenerations.push_back(1);
            } catch (...) {
                encoderSlots.pop_back();
                throw;
            }
            encoderSlots[index] = slot;
            appended = true;
        } else {
            index = freeEncoderIndices.back();
            freeEncoderIndices.pop_back();
            encoderSlots[index] = slot;
        }
        slot->generation = encoderGenerations[index];
        try {
            ++activeEncoderCounts[deviceKey(device)];
        } catch (...) {
            encoderSlots[index].reset();
            if (appended) {
                encoderSlots.pop_back();
                encoderGenerations.pop_back();
            } else {
                freeEncoderIndices.push_back(index);
            }
            throw;
        }
    } catch (const std::bad_alloc &) {
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

    uint64_t native{};
    VernonRhiBackend backend{};
    if (!vernon::rhi::beginCommandRecording(device, native, backend)) {
        {
            std::lock_guard<std::mutex> guard(slot->mutex);
            slot->alive = false;
        }
        unregisterEncoder(index, slot);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->native = native;
        slot->backend = backend;
        slot->initializing = false;
    }
    *output = {index, slot->generation};
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyCommandEncoder(VernonRhiDevice device,
                                                                VernonRhiCommandEncoder encoder) {
    auto slot = lookup(device, encoder);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;

    VernonRhiDevice recordedDevice{};
    uint64_t native{};
    std::vector<EncoderSlot::Cleanup> cleanups;
    std::vector<EncoderSlot::Cleanup> rollbacks;
    std::unordered_set<EncoderSlot::RetainedResource, EncoderSlot::RetainedResourceHash> resources;
    uint64_t uploadBytes{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || slot->rendering)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->alive = false;
        recordedDevice = slot->device;
        native = slot->native;
        cleanups = std::move(slot->cleanups);
        rollbacks = std::move(slot->rollbacks);
        resources = std::move(slot->retainedResources);
        uploadBytes = std::exchange(slot->admittedUploadBytes, 0);
    }
    vernon::rhi::abandonCommandRecording(recordedDevice, native);
    for (auto rollback = rollbacks.rbegin(); rollback != rollbacks.rend(); ++rollback)
        rollback->function(rollback->context, rollback->object);
    for (auto cleanup = cleanups.rbegin(); cleanup != cleanups.rend(); ++cleanup)
        cleanup->function(cleanup->context, cleanup->object);
    for (const auto &resource : resources)
        vernon::rhi::releaseResource(recordedDevice, resource.kind, resource.key);
    releaseUploadBytes(recordedDevice, uploadBytes);
    unregisterEncoder(encoder.index, slot);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderBarrier(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                          const VernonRhiBarrier *barriers, size_t barrierCount) {
    auto slot = lookup(device, encoder);
    if (!slot || (barrierCount && !barriers))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    constexpr uint32_t allStages = VERNON_RHI_STAGE_COMPUTE | VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT;
    constexpr uint32_t allAccess =
        VERNON_RHI_ACCESS_TRANSFER_READ | VERNON_RHI_ACCESS_TRANSFER_WRITE | VERNON_RHI_ACCESS_SHADER_READ |
        VERNON_RHI_ACCESS_SHADER_WRITE | VERNON_RHI_ACCESS_COLOR_READ | VERNON_RHI_ACCESS_COLOR_WRITE |
        VERNON_RHI_ACCESS_DEPTH_STENCIL_READ | VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE | VERNON_RHI_ACCESS_VERTEX_READ |
        VERNON_RHI_ACCESS_INDEX_READ | VERNON_RHI_ACCESS_INDIRECT_READ | VERNON_RHI_ACCESS_HOST_READ |
        VERNON_RHI_ACCESS_HOST_WRITE;
    constexpr uint32_t allAspects =
        VERNON_RHI_IMAGE_ASPECT_COLOR | VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL;
    for (size_t index = 0; index < barrierCount; ++index)
        if (barriers[index].struct_size < sizeof(VernonRhiBarrier) ||
            barriers[index].old_state > VERNON_RHI_STATE_PRESENT ||
            barriers[index].new_state > VERNON_RHI_STATE_PRESENT || barriers[index].is_image > 1 ||
            (barriers[index].source_stage_mask & ~allStages) || (barriers[index].destination_stage_mask & ~allStages) ||
            (barriers[index].source_access & ~allAccess) || (barriers[index].destination_access & ~allAccess) ||
            (barriers[index].is_image &&
             (!barriers[index].image_subresources.mip_level_count ||
              !barriers[index].image_subresources.array_layer_count || !barriers[index].image_subresources.aspects ||
              (barriers[index].image_subresources.aspects & ~allAspects))))
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t key = vernon::rhi::encodeResourceKey(encoder);
    for (size_t index = 0; index < barrierCount; ++index) {
        const vernon::rhi::ResourceKind kind =
            barriers[index].is_image ? vernon::rhi::ResourceKind::Image : vernon::rhi::ResourceKind::Buffer;
        const uint64_t resource = barriers[index].is_image ? logicalResourceKey(barriers[index].image)
                                                           : logicalResourceKey(barriers[index].buffer);
        if (!vernon::rhi::retainCommandResource(device, key, kind, resource))
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    uint64_t native{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->busy = true;
        native = slot->native;
    }
    const VernonRhiStatus recordStatus =
        barrierCount ? vernon::rhi::recordBarriers(device, key, native, barriers, barrierCount) : VERNON_RHI_STATUS_OK;
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        if (recordStatus == VERNON_RHI_STATUS_OK) {
            slot->stats.barrier_count += static_cast<uint32_t>(barrierCount);
            for (size_t index = 0; index < barrierCount; ++index) {
                const vernon::rhi::ResourceKind kind =
                    barriers[index].is_image ? vernon::rhi::ResourceKind::Image : vernon::rhi::ResourceKind::Buffer;
                const uint64_t resource = barriers[index].is_image ? logicalResourceKey(barriers[index].image)
                                                                   : logicalResourceKey(barriers[index].buffer);
                slot->pendingWriteResources.erase({kind, resource});
            }
        } else
            slot->failed = true;
    }
    return recordStatus;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderCopyBuffer(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                             VernonRhiBuffer source, uint64_t sourceOffset,
                                                             VernonRhiBuffer destination, uint64_t destinationOffset,
                                                             uint64_t size) {
    auto slot = lookup(device, encoder);
    if (!slot || !validHandle(source) || !validHandle(destination) || !size)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t key = vernon::rhi::encodeResourceKey(encoder);
    const uint64_t sourceResource = logicalResourceKey(source);
    const uint64_t destinationResource = logicalResourceKey(destination);
    if (!vernon::rhi::retainCommandResource(device, key, vernon::rhi::ResourceKind::Buffer, sourceResource) ||
        !vernon::rhi::retainCommandResource(device, key, vernon::rhi::ResourceKind::Buffer, destinationResource))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    uint64_t native{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->busy = true;
        native = slot->native;
    }
    const VernonRhiStatus recordStatus =
        vernon::rhi::recordBufferCopy(device, native, source, sourceOffset, destination, destinationOffset, size);
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        if (recordStatus == VERNON_RHI_STATUS_OK)
            slot->pendingWriteResources.insert({vernon::rhi::ResourceKind::Buffer, destinationResource});
        else
            slot->failed = true;
    }
    return recordStatus;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderUploadBuffer(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                               VernonRhiBuffer destination, uint64_t destinationOffset,
                                                               const void *source, uint64_t size) {
    if (!source || !size)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto slot = lookup(device, encoder);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VernonRhiStatus status = reserveUploadBytes(slot, size);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.alignment = 4;
    descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE;
    descriptor.memory_class = VERNON_RHI_MEMORY_UPLOAD;
    VernonRhiBuffer staging{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    status = vernonRhiDeviceCreateBuffer(device, &descriptor, &staging);
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiDeviceUploadBuffer(device, staging, 0, source, size);
    bool recorded = false;
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCommandEncoderCopyBuffer(device, encoder, staging, 0, destination, destinationOffset, size);
    recorded = status == VERNON_RHI_STATUS_OK;
    if (staging.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
        const VernonRhiStatus destroyStatus = vernonRhiDeviceDestroyBuffer(device, staging);
        if (status == VERNON_RHI_STATUS_OK)
            status = destroyStatus;
    }
    if (!recorded)
        releaseEncoderUploadBytes(slot, size);
    return status;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderBeginRendering(VernonRhiDevice device,
                                                                 VernonRhiCommandEncoder encoder,
                                                                 const VernonRhiRenderingDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !validRendering(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto slot = lookup(device, encoder);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    try {
        slot->colorOperations.resize(descriptor->color_attachment_count);
    } catch (const std::bad_alloc &) {
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        const auto &source = descriptor->color_attachments[index];
        auto &target = slot->colorOperations[index];
        target.location = source.location;
        target.load = source.load_operation;
        target.store = source.store_operation;
        std::copy(std::begin(source.clear_color), std::end(source.clear_color), target.clear.begin());
    }
    slot->depthOperation = {};
    if (descriptor->depth_stencil_attachment) {
        const auto &source = *descriptor->depth_stencil_attachment;
        slot->depthOperation = {source.depth_load_operation,
                                source.depth_store_operation,
                                source.stencil_load_operation,
                                source.stencil_store_operation,
                                source.clear_depth,
                                source.clear_stencil,
                                true};
    }
    slot->rendering = true;
    slot->hasRenderingDescriptor = true;
    slot->renderX = descriptor->offset_x;
    slot->renderY = descriptor->offset_y;
    slot->renderWidth = descriptor->width;
    slot->renderHeight = descriptor->height;
    slot->renderLayers = descriptor->layers;
    ++slot->stats.rendering_scope_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderEndRendering(VernonRhiDevice device,
                                                               VernonRhiCommandEncoder encoder) {
    auto slot = lookup(device, encoder);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->busy || !slot->rendering || slot->finished)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    uint32_t colorDiscardMask = 0;
    for (size_t index = 0; index < slot->colorOperations.size(); ++index)
        if (slot->colorOperations[index].store == VERNON_RHI_STORE_DISCARD)
            colorDiscardMask |= uint32_t{1} << slot->colorOperations[index].location;
    const uint32_t depthStencilDiscard =
        (slot->depthOperation.present && slot->depthOperation.depthStore == VERNON_RHI_STORE_DISCARD
             ? VERNON_RHI_ATTACHMENT_DEPTH
             : 0) |
        (slot->depthOperation.present && slot->depthOperation.stencilStore == VERNON_RHI_STORE_DISCARD
             ? VERNON_RHI_ATTACHMENT_STENCIL
             : 0);
    if (slot->backendRendering &&
        !vernon::rhi::endCommandRendering(device, slot->native, slot->backend, slot->backendRendering, colorDiscardMask,
                                          depthStencilDiscard, slot->colorResources.data(), slot->colorResources.size(),
                                          slot->depthResource, slot->backendRenderingObject))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot->backendRendering = 0;
    slot->backendRenderingObject = 0;
    slot->rendering = false;
    slot->hasRenderingDescriptor = false;
    slot->colorOperations.clear();
    slot->depthOperation = {};
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderClearColorAttachment(VernonRhiDevice device,
                                                                       VernonRhiCommandEncoder encoder,
                                                                       uint32_t location, const float clearColor[4]) {
    auto slot = lookup(device, encoder);
    if (!slot || !clearColor)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished || location >= 8)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!slot->backendRendering) {
        const auto operation = std::find_if(slot->colorOperations.begin(), slot->colorOperations.end(),
                                            [location](const auto &value) { return value.location == location; });
        if (operation == slot->colorOperations.end())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        operation->load = VERNON_RHI_LOAD_CLEAR;
        std::copy(clearColor, clearColor + 4, operation->clear.begin());
        ++slot->stats.clear_count;
        return VERNON_RHI_STATUS_OK;
    }
    const VernonRhiStatus status = vernon::rhi::clearCommandColor(
        device, slot->native, slot->backend, slot->backendRendering, slot->renderX, slot->renderY, slot->renderWidth,
        slot->renderHeight, slot->renderLayers, slot->colorTargets[location], location, clearColor);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    ++slot->stats.clear_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderClearDepthStencilAttachment(VernonRhiDevice device,
                                                                              VernonRhiCommandEncoder encoder,
                                                                              float clearDepth, uint32_t clearStencil,
                                                                              uint32_t aspects) {
    auto slot = lookup(device, encoder);
    if (!slot || clearDepth < 0.0f || clearDepth > 1.0f || !aspects ||
        (aspects & ~(VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL)))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!slot->backendRendering) {
        if (!slot->depthOperation.present)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        if (aspects & VERNON_RHI_ATTACHMENT_DEPTH) {
            slot->depthOperation.depthLoad = VERNON_RHI_LOAD_CLEAR;
            slot->depthOperation.clearDepth = clearDepth;
        }
        if (aspects & VERNON_RHI_ATTACHMENT_STENCIL) {
            slot->depthOperation.stencilLoad = VERNON_RHI_LOAD_CLEAR;
            slot->depthOperation.clearStencil = clearStencil;
        }
        ++slot->stats.clear_count;
        return VERNON_RHI_STATUS_OK;
    }
    const VernonRhiStatus status = vernon::rhi::clearCommandDepthStencil(
        device, slot->native, slot->backend, slot->backendRendering, slot->renderX, slot->renderY, slot->renderWidth,
        slot->renderHeight, slot->renderLayers, slot->depthTarget, clearDepth, clearStencil, aspects);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    ++slot->stats.clear_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderFinish(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    auto slot = lookup(device, encoder);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->finished = true;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceSubmit(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                 VernonRhiCompletion *output) {
    if (output)
        *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    auto slot = lookup(device, encoder);
    if (!slot || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiStatus admissionStatus = reserveSubmission(device);
    if (admissionStatus != VERNON_RHI_STATUS_OK)
        return admissionStatus;
    std::shared_ptr<CompletionSlot> completion;
    VernonRhiCompletion handle{};
    try {
        completion = std::make_shared<CompletionSlot>();
        completion->device = device;
        if (!registerCompletion(completion, handle)) {
            releaseSubmission(device);
            return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
        }
    } catch (const std::bad_alloc &) {
        releaseSubmission(device);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    uint64_t native{};
    bool computeWrites{};
    bool completed{};
    bool externalCompletion{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || !slot->finished || slot->failed) {
            unregisterCompletion(handle.index, completion);
            releaseSubmission(device);
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        slot->busy = true;
        native = slot->native;
        computeWrites = slot->unknownPendingWrites || !slot->pendingWriteResources.empty();
    }
    const bool submitted =
        vernon::rhi::submitCommandRecording(device, native, computeWrites, completed, externalCompletion);
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        if (submitted) {
            slot->alive = false;
            slot->rollbacks.clear();
            ++slot->stats.submission_count;
            completion->backend = slot->backend;
            completion->native = slot->native;
            completion->stats = slot->stats;
            completion->state = completed ? VERNON_RHI_COMPLETION_SUCCEEDED : VERNON_RHI_COMPLETION_PENDING;
            completion->externalSignalRequired = externalCompletion;
            completion->backendRetirementRequired = !completed;
            completion->admissionHeld = true;
            completion->cleanups = std::move(slot->cleanups);
            completion->retainedResources = std::move(slot->retainedResources);
            completion->admittedUploadBytes = std::exchange(slot->admittedUploadBytes, 0);
        } else {
            slot->failed = true;
        }
    }
    if (!submitted) {
        unregisterCompletion(handle.index, completion);
        releaseSubmission(device);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    unregisterEncoder(encoder.index, slot);
    *output = handle;
    if (completed)
        (void)observeCompletion(completion);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCompletionGetState(VernonRhiDevice device, VernonRhiCompletion completion,
                                                       VernonRhiCompletionState *output) {
    auto slot = lookup(device, completion);
    if (!slot || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    bool shouldPoll{};
    bool shouldFinalize{};
    uint64_t native{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        shouldPoll = slot->state == VERNON_RHI_COMPLETION_PENDING && !slot->externalSignalRequired &&
                     slot->backendRetirementRequired;
        shouldFinalize = slot->state != VERNON_RHI_COMPLETION_PENDING && !slot->finalized;
        native = slot->native;
    }
    if (shouldPoll) {
        bool completed{};
        bool succeeded{};
        if (vernon::rhi::pollCommandRecording(device, native, completed, succeeded) && completed) {
            std::lock_guard<std::mutex> guard(slot->mutex);
            if (slot->state == VERNON_RHI_COMPLETION_PENDING) {
                slot->state = succeeded ? VERNON_RHI_COMPLETION_SUCCEEDED : VERNON_RHI_COMPLETION_FAILED;
                slot->result = succeeded ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
            }
            shouldFinalize = true;
        }
    }
    if (shouldFinalize)
        (void)observeCompletion(slot);
    std::lock_guard<std::mutex> guard(slot->mutex);
    *output = slot->state;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCompletionWait(VernonRhiDevice device, VernonRhiCompletion completion) {
    auto slot = lookup(device, completion);
    return slot ? observeCompletion(slot) : VERNON_RHI_STATUS_INVALID_ARGUMENT;
}

extern "C" VernonRhiStatus vernonRhiCompletionSignal(VernonRhiDevice device, VernonRhiCompletion completion,
                                                     VernonRhiStatus result) {
    auto slot = lookup(device, completion);
    if (!slot || static_cast<uint32_t>(result) > static_cast<uint32_t>(VERNON_RHI_STATUS_RESOURCE_EXHAUSTED))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->externalSignalRequired || slot->state != VERNON_RHI_COMPLETION_PENDING)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->result = result;
        slot->state = result == VERNON_RHI_STATUS_OK ? VERNON_RHI_COMPLETION_SUCCEEDED : VERNON_RHI_COMPLETION_FAILED;
    }
    slot->condition.notify_all();
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCompletionGetCommandStats(VernonRhiDevice device, VernonRhiCompletion completion,
                                                              VernonRhiCommandEncoderStats *output) {
    auto slot = lookup(device, completion);
    if (!slot || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(slot->mutex);
    *output = slot->stats;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyCompletion(VernonRhiDevice device, VernonRhiCompletion completion) {
    auto slot = lookup(device, completion);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiStatus status = observeCompletion(slot);
    unregisterCompletion(completion.index, slot);
    return status;
}

void vernon::rhi::drainDeviceCompletions(VernonRhiDevice device) {
    std::vector<std::pair<uint32_t, std::shared_ptr<CompletionSlot>>> pending;
    {
        std::lock_guard<std::mutex> guard(registryMutex);
        for (uint32_t index = 0; index < completionSlots.size(); ++index)
            if (completionSlots[index] && sameDevice(completionSlots[index]->device, device))
                pending.emplace_back(index, completionSlots[index]);
    }
    for (const auto &[index, completion] : pending) {
        (void)observeCompletion(completion);
        unregisterCompletion(index, completion);
    }
}

void vernon::rhi::forgetDeviceCommandLimits(VernonRhiDevice device) {
    std::lock_guard<std::mutex> guard(registryMutex);
    const uint64_t key = deviceKey(device);
    commandLimits.erase(key);
    inFlightSubmissionCounts.erase(key);
    admittedUploadBytes.erase(key);
}

uint64_t vernon::rhi::commandEncoderKey(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    auto slot = lookup(device, encoder);
    if (!slot)
        return 0;
    std::lock_guard<std::mutex> guard(slot->mutex);
    return slot->alive && !slot->initializing ? vernon::rhi::encodeResourceKey(encoder) : 0;
}

uint64_t vernon::rhi::commandEncoderNative(VernonRhiDevice device, uint64_t key, VernonRhiBackend backend) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return 0;
    std::lock_guard<std::mutex> guard(slot->mutex);
    return slot->alive && !slot->initializing && !slot->busy && slot->backend == backend && !slot->finished
               ? slot->native
               : 0;
}

bool vernon::rhi::commandEncoderRendering(VernonRhiDevice device, uint64_t key) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    return slot->alive && slot->rendering && !slot->finished;
}

bool vernon::rhi::commandEncoderHasRenderingDescriptor(VernonRhiDevice device, uint64_t key) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    return slot->alive && slot->rendering && slot->hasRenderingDescriptor && !slot->finished;
}

bool vernon::rhi::commandColorOperations(VernonRhiDevice device, uint64_t key, size_t index,
                                         VernonRhiLoadOperation &load, VernonRhiStoreOperation &store, float clear[4]) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || !slot->rendering || !slot->hasRenderingDescriptor || index >= slot->colorOperations.size())
        return false;
    const auto &operations = slot->colorOperations[index];
    load = operations.load;
    store = operations.store;
    std::copy(operations.clear.begin(), operations.clear.end(), clear);
    return true;
}

bool vernon::rhi::commandDepthOperations(VernonRhiDevice device, uint64_t key, VernonRhiLoadOperation &depthLoad,
                                         VernonRhiStoreOperation &depthStore, VernonRhiLoadOperation &stencilLoad,
                                         VernonRhiStoreOperation &stencilStore, float &clearDepth,
                                         uint32_t &clearStencil) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || !slot->rendering || !slot->hasRenderingDescriptor || !slot->depthOperation.present)
        return false;
    depthLoad = slot->depthOperation.depthLoad;
    depthStore = slot->depthOperation.depthStore;
    stencilLoad = slot->depthOperation.stencilLoad;
    stencilStore = slot->depthOperation.stencilStore;
    clearDepth = slot->depthOperation.clearDepth;
    clearStencil = slot->depthOperation.clearStencil;
    return true;
}

int vernon::rhi::claimCommandRendering(VernonRhiDevice device, uint64_t key, uint32_t backendKind) {
    if (!backendKind)
        return -1;
    auto slot = lookupKey(device, key);
    if (!slot)
        return -1;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished)
        return -1;
    if (slot->backendRendering)
        return slot->backendRendering == backendKind ? 0 : -1;
    slot->backendRendering = backendKind;
    return 1;
}

uint64_t vernon::rhi::commandRenderingObject(VernonRhiDevice device, uint64_t key, uint64_t candidate) {
    if (!candidate)
        return 0;
    auto slot = lookupKey(device, key);
    if (!slot)
        return 0;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished)
        return 0;
    if (!slot->backendRenderingObject)
        slot->backendRenderingObject = candidate;
    return slot->backendRenderingObject;
}

bool vernon::rhi::beginProviderRendering(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    auto slot = lookup(device, encoder);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished)
        return false;
    slot->rendering = true;
    slot->hasRenderingDescriptor = false;
    slot->colorOperations.clear();
    slot->depthOperation = {};
    ++slot->stats.rendering_scope_count;
    return true;
}

bool vernon::rhi::recordProviderCommand(VernonRhiDevice device, uint64_t key, bool draw) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || slot->finished || (draw != slot->rendering))
        return false;
    if (draw)
        ++slot->stats.draw_count;
    else
        ++slot->stats.dispatch_count;
    return true;
}

bool vernon::rhi::recordCommandWriteResource(VernonRhiDevice device, uint64_t key, ResourceKind kind,
                                             uint64_t resourceKey) {
    auto slot = lookupKey(device, key);
    if (!slot || !resourceKey)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || slot->finished)
        return false;
    try {
        slot->pendingWriteResources.insert({kind, resourceKey});
    } catch (const std::bad_alloc &) {
        slot->unknownPendingWrites = true;
    }
    return true;
}

bool vernon::rhi::retainCommandResource(VernonRhiDevice device, uint64_t key, ResourceKind kind, uint64_t resourceKey) {
    auto slot = lookupKey(device, key);
    if (!slot || !resourceKey)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || slot->finished)
        return false;
    const EncoderSlot::RetainedResource identity{kind, resourceKey};
    if (slot->retainedResources.find(identity) != slot->retainedResources.end())
        return true;
    try {
        slot->retainedResources.insert(identity);
    } catch (const std::bad_alloc &) {
        return false;
    }
    if (!retainResource(device, kind, resourceKey)) {
        slot->retainedResources.erase(identity);
        return false;
    }
    return true;
}

bool vernon::rhi::deferCommandCleanup(VernonRhiDevice device, uint64_t key, void *context, uint64_t object,
                                      void (*cleanup)(void *, uint64_t)) {
    if (!cleanup)
        return false;
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->finished)
        return false;
    if (object == 0) {
        const EncoderSlot::ActionIdentity identity{context, cleanup};
        if (slot->cleanupIdentities.find(identity) != slot->cleanupIdentities.end()) {
            cleanup(context, 0);
            return true;
        }
        try {
            slot->cleanupIdentities.insert(identity);
        } catch (const std::bad_alloc &) {
            return false;
        }
    }
    try {
        slot->cleanups.push_back({context, object, cleanup});
    } catch (const std::bad_alloc &) {
        if (object == 0)
            slot->cleanupIdentities.erase({context, cleanup});
        return false;
    }
    return true;
}

bool vernon::rhi::deferCommandRollback(VernonRhiDevice device, uint64_t key, void *context, uint64_t object,
                                       void (*rollback)(void *, uint64_t)) {
    if (!context || !rollback)
        return false;
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->finished)
        return false;
    const EncoderSlot::ActionIdentity identity{context, rollback};
    if (slot->rollbackIdentities.find(identity) != slot->rollbackIdentities.end())
        return true;
    try {
        slot->rollbackIdentities.insert(identity);
        slot->rollbacks.push_back({context, object, rollback});
    } catch (const std::bad_alloc &) {
        slot->rollbackIdentities.erase(identity);
        return false;
    }
    return true;
}

bool vernon::rhi::setCommandRenderingTargets(VernonRhiDevice device, uint64_t key, const uint64_t *colors,
                                             const uint64_t *resources, size_t colorCount, uint64_t depth,
                                             uint64_t depthResource) {
    if (colorCount > VERNON_RHI_MAX_COLOR_ATTACHMENTS || (colorCount && (!colors || !resources)))
        return false;
    auto slot = lookupKey(device, key);
    if (!slot)
        return false;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished)
        return false;
    if (colorCount) {
        std::copy(colors, colors + colorCount, slot->colorTargets.begin());
        std::copy(resources, resources + colorCount, slot->colorResources.begin());
    }
    slot->depthTarget = depth;
    slot->depthResource = depthResource;
    return true;
}
