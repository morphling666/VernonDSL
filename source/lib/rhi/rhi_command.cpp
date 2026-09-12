#include "VernonRHI.h"

#include "image_data_layout.h"
#include "rhi_internal.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using EncoderLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::CommandEncoderResourceTag>;
using CompletionLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::CompletionResourceTag>;
using EncoderHandle = vernon::rhi::ResourceHandle<vernon::rhi::CommandEncoderResourceTag>;
using CompletionHandle = vernon::rhi::ResourceHandle<vernon::rhi::CompletionResourceTag>;

struct EncoderSlot {
    explicit EncoderSlot(EncoderLifecycle value) noexcept : lifecycle(std::move(value)) {}

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

    EncoderLifecycle lifecycle;
    vernon::rhi::CommandDeviceStateControl *commandState{};
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
    uint32_t backendRendering{};
    uint64_t backendRenderingObject{};
    bool alive{true};
    bool initializing{true};
    bool busy{};
    bool rendering{};
    bool hasRenderingDescriptor{};
    bool finished{};
    bool failed{};
    bool nativeOwned{};
    bool activeCounted{};
    bool exclusiveDeviceSlot{};
    std::vector<ColorOperation> colorOperations;
    DepthOperation depthOperation;
    std::vector<Cleanup> cleanups;
    std::vector<Cleanup> rollbacks;
    std::unordered_set<ActionIdentity, ActionIdentityHash> cleanupIdentities;
    std::unordered_set<ActionIdentity, ActionIdentityHash> rollbackIdentities;
    std::unordered_set<RetainedResource, RetainedResourceHash> retainedResourceIdentities;
    std::vector<vernon::rhi::RetainedRhiResourceLease> retainedResourceLeases;
    std::unordered_set<RetainedResource, RetainedResourceHash> pendingWriteResources;
    uint64_t admittedUploadBytes{};
    bool unknownPendingWrites{};
};

struct CompletionSlot {
    explicit CompletionSlot(CompletionLifecycle value) noexcept : lifecycle(std::move(value)) {}

    CompletionLifecycle lifecycle;
    vernon::rhi::CommandDeviceStateControl *commandState{};
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
    bool initializing{};
    bool observing{};
    bool finalized{};
    bool admissionHeld{};
    bool liveCounted{};
    std::vector<EncoderSlot::Cleanup> cleanups;
    std::vector<vernon::rhi::RetainedRhiResourceLease> retainedResourceLeases;
    uint64_t admittedUploadBytes{};
};

constexpr VernonRhiCommandLimits defaultCommandLimits{
    sizeof(VernonRhiCommandLimits), 8, 8, 64, 64u * 1024u * 1024u, {0, 0, 0}};

} // namespace

static vernon::Result<void, vernon::RhiError> retainCommandResourceImpl(VernonRhiDevice device, uint64_t key,
                                                                        vernon::rhi::ResourceKind kind,
                                                                        uint64_t resourceKey,
                                                                        bool expectedBusy) noexcept;

namespace vernon::rhi {

class CommandDeviceStateControl final : public CheckedIntrusiveControl<CommandDeviceStateControl> {
public:
    explicit CommandDeviceStateControl(OwnerRef value) noexcept : owner(std::move(value)) {}
    ~CommandDeviceStateControl() noexcept {
        if (owner.childCount() || activeEncoderCount || inFlightSubmissionCount || liveCompletionCount ||
            admittedUploadBytes)
            resultContractViolation();
        for (size_t index = 0; index < encoderSlots.size(); ++index)
            if (encoderSlots.get(index)->lifecycle.snapshot().occupied)
                resultContractViolation();
        for (size_t index = 0; index < completionSlots.size(); ++index)
            if (completionSlots.get(index)->lifecycle.snapshot().occupied)
                resultContractViolation();
    }

    OwnerRef owner;
    std::mutex encoderCreationMutex;
    std::mutex completionCreationMutex;
    std::mutex quotaMutex;
    StableResourceSlotContainer<EncoderSlot> encoderSlots;
    StableResourceSlotContainer<CompletionSlot> completionSlots;
    VernonRhiCommandLimits limits{defaultCommandLimits};
    uint32_t activeEncoderCount{};
    uint32_t inFlightSubmissionCount{};
    uint32_t liveCompletionCount{};
    uint64_t admittedUploadBytes{};
};

CommandDeviceStateRef::CommandDeviceStateRef() noexcept = default;

CommandDeviceStateRef::CommandDeviceStateRef(CheckedIntrusiveRef<CommandDeviceStateControl> control) noexcept {
    control_.emplace(std::move(control));
}

CommandDeviceStateRef::CommandDeviceStateRef(CommandDeviceStateRef &&other) noexcept
    : control_(std::move(other.control_)) {}

CommandDeviceStateRef &CommandDeviceStateRef::operator=(CommandDeviceStateRef &&other) noexcept {
    control_ = std::move(other.control_);
    return *this;
}

CommandDeviceStateRef::~CommandDeviceStateRef() noexcept = default;

CommandDeviceStateRef::operator bool() const noexcept { return static_cast<bool>(control_); }

Result<CommandDeviceStateRef, RhiError> CommandDeviceStateRef::retain() const noexcept {
    if (!control_)
        return Result<CommandDeviceStateRef, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"retain_command_state", 0, 0}})};
    auto retained = control_.value().retain();
    if (retained.isErr())
        return Result<CommandDeviceStateRef, RhiError>{err(toRhiError(std::move(retained).error()))};
    return Result<CommandDeviceStateRef, RhiError>{ok(CommandDeviceStateRef{std::move(retained).value()})};
}

Result<ChildLease, RhiError> CommandDeviceStateRef::retainDeviceLease() const noexcept {
    if (!control_)
        return Result<ChildLease, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"retain_device_lease", 0, 0}})};
    auto reservation = control_.value()->owner.reserveChild();
    if (reservation.isErr())
        return Result<ChildLease, RhiError>{err(toRhiError(std::move(reservation).error()))};
    auto committed = std::move(reservation).value().commit();
    if (committed.isErr())
        return Result<ChildLease, RhiError>{err(toRhiError(std::move(committed).error()))};
    return Result<ChildLease, RhiError>{ok(std::move(committed).value())};
}

Result<CommandDeviceStateRef, RhiError> createCommandDeviceState(OwnerRef owner) noexcept {
    auto *control = new (std::nothrow) CommandDeviceStateControl(std::move(owner));
    if (!control)
        return Result<CommandDeviceStateRef, RhiError>{
            err(RhiError{RhiErrorCode::ResourceExhausted, {"allocate_command_state", 0, 0}})};
    return Result<CommandDeviceStateRef, RhiError>{
        ok(CommandDeviceStateRef{CheckedIntrusiveRef<CommandDeviceStateControl>::adopt(control)})};
}

CommandDeviceStateControl &commandDeviceState(CommandDeviceStateRef &state) noexcept {
    return state.control_.value().value();
}

} // namespace vernon::rhi

namespace {

template <typename Handle> bool validHandle(Handle handle) {
    return handle.index != VERNON_RHI_INVALID_HANDLE_INDEX && handle.generation != 0;
}

template <typename Handle> uint64_t logicalResourceKey(Handle handle) {
    return validHandle(handle)
               ? (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1)
               : 0;
}

bool sameDevice(VernonRhiDevice left, VernonRhiDevice right) {
    return left.index == right.index && left.generation == right.generation;
}

VernonRhiStatus reserveSubmission(vernon::rhi::CommandDeviceStateControl &state) {
    std::lock_guard<std::mutex> guard(state.quotaMutex);
    if (state.inFlightSubmissionCount >= state.limits.max_in_flight_submissions)
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    ++state.inFlightSubmissionCount;
    return VERNON_RHI_STATUS_OK;
}

void releaseSubmission(vernon::rhi::CommandDeviceStateControl &state) {
    std::lock_guard<std::mutex> guard(state.quotaMutex);
    if (!state.inFlightSubmissionCount)
        vernon::resultContractViolation();
    --state.inFlightSubmissionCount;
}

template <typename Slot> struct PinnedCommandSlot {
    PinnedCommandSlot() noexcept = default;
    PinnedCommandSlot(vernon::rhi::CommandDeviceStateRef stateValue, Slot *slotValue,
                      vernon::OperationPin pinValue) noexcept
        : state(std::move(stateValue)), slot(slotValue) {
        pin.emplace(std::move(pinValue));
    }

    vernon::rhi::CommandDeviceStateRef state;
    Slot *slot{};
    vernon::Option<vernon::OperationPin> pin;

    explicit operator bool() const noexcept { return slot != nullptr; }
    Slot *operator->() const noexcept { return slot; }
};

VernonRhiStatus reserveUploadBytes(PinnedCommandSlot<EncoderSlot> &pinned, uint64_t size) {
    EncoderSlot *slot = pinned.slot;
    auto &state = vernon::rhi::commandDeviceState(pinned.state);
    std::lock_guard<std::mutex> registryGuard(state.quotaMutex);
    std::lock_guard<std::mutex> slotGuard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t current = state.admittedUploadBytes;
    const uint64_t maximum = state.limits.max_upload_bytes;
    if (size > maximum || current > maximum - size)
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    state.admittedUploadBytes = current + size;
    slot->admittedUploadBytes += size;
    return VERNON_RHI_STATUS_OK;
}

void releaseUploadBytes(vernon::rhi::CommandDeviceStateControl &state, uint64_t size) {
    if (!size)
        return;
    std::lock_guard<std::mutex> guard(state.quotaMutex);
    if (size > state.admittedUploadBytes)
        vernon::resultContractViolation();
    state.admittedUploadBytes -= size;
}

void releaseEncoderUploadBytes(PinnedCommandSlot<EncoderSlot> &pinned, uint64_t size) {
    EncoderSlot *slot = pinned.slot;
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (size > slot->admittedUploadBytes)
            size = slot->admittedUploadBytes;
        slot->admittedUploadBytes -= size;
    }
    releaseUploadBytes(vernon::rhi::commandDeviceState(pinned.state), size);
}

PinnedCommandSlot<EncoderSlot> lookup(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    auto retained = vernon::rhi::commandState(device);
    if (retained.isErr())
        return {};
    auto stateRef = std::move(retained).value();
    auto &state = vernon::rhi::commandDeviceState(stateRef);
    EncoderSlot *slot = state.encoderSlots.get(encoder.index);
    if (!slot)
        return {};
    auto pin = slot->lifecycle.pin({encoder.index, encoder.generation});
    if (pin.isErr())
        return {};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!sameDevice(slot->device, device))
            return {};
    }
    return {std::move(stateRef), slot, std::move(pin).value()};
}

PinnedCommandSlot<EncoderSlot> lookupKey(VernonRhiDevice device, uint64_t key) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return {};
    return lookup(device, {static_cast<uint32_t>(encodedIndex - 1), generation});
}

PinnedCommandSlot<CompletionSlot> lookup(VernonRhiDevice device, VernonRhiCompletion completion) {
    auto retained = vernon::rhi::commandState(device);
    if (retained.isErr())
        return {};
    auto stateRef = std::move(retained).value();
    auto &state = vernon::rhi::commandDeviceState(stateRef);
    CompletionSlot *slot = state.completionSlots.get(completion.index);
    if (!slot)
        return {};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (slot->initializing || !sameDevice(slot->device, device))
            return {};
    }
    auto pin = slot->lifecycle.pin({completion.index, completion.generation});
    if (pin.isErr())
        return {};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (slot->initializing || !sameDevice(slot->device, device))
            return {};
    }
    return {std::move(stateRef), slot, std::move(pin).value()};
}

vernon::Result<CompletionSlot *, vernon::RhiError>
reserveCompletionSlot(vernon::rhi::CommandDeviceStateControl &state) noexcept {
    {
        std::lock_guard<std::mutex> guard(state.quotaMutex);
        if (state.liveCompletionCount >= state.limits.max_live_completions)
            return vernon::Result<CompletionSlot *, vernon::RhiError>{vernon::err(vernon::RhiError{
                vernon::RhiErrorCode::ResourceExhausted, {"reserve_completion", state.liveCompletionCount, 0}})};
    }
    const size_t slotCount = state.completionSlots.size();
    for (size_t index = 0; index < slotCount; ++index) {
        CompletionSlot *slot = state.completionSlots.get(index);
        if (!slot->lifecycle.snapshot().occupied)
            return vernon::Result<CompletionSlot *, vernon::RhiError>{vernon::ok(slot)};
    }
    if (slotCount >= UINT32_MAX)
        return vernon::Result<CompletionSlot *, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::ResourceExhausted, {"reserve_completion", slotCount, 0}})};
    auto lifecycle = CompletionLifecycle::create(static_cast<uint32_t>(slotCount));
    if (lifecycle.isErr())
        return vernon::Result<CompletionSlot *, vernon::RhiError>{vernon::err(std::move(lifecycle).error())};
    return state.completionSlots.emplace("reserve_completion", std::move(lifecycle).value());
}

vernon::Result<void, vernon::RhiError>
releaseLeases(std::vector<vernon::rhi::RetainedRhiResourceLease> &leases) noexcept {
    for (auto lease = leases.rbegin(); lease != leases.rend(); ++lease) {
        if (!lease->active())
            continue;
        auto released = lease->release();
        if (released.isErr())
            return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(released).error())};
    }
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> rollbackRetainedResources(VernonRhiDevice device, EncoderSlot &slot,
                                                                 size_t checkpoint) noexcept {
    for (;;) {
        std::unique_lock<std::mutex> guard(slot.mutex);
        if (checkpoint > slot.retainedResourceLeases.size())
            return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
                vernon::RhiErrorCode::LifecycleFailure, {"rollback_command_resources", checkpoint, 0}})};
        if (slot.retainedResourceLeases.size() == checkpoint)
            return vernon::Result<void, vernon::RhiError>{vernon::ok()};
        auto &storedLease = slot.retainedResourceLeases.back();
        bool foundIdentity = false;
        for (auto identity = slot.retainedResourceIdentities.begin(); identity != slot.retainedResourceIdentities.end();
             ++identity)
            if (storedLease.authorizes(device.index, device.generation, static_cast<uint32_t>(identity->kind),
                                       identity->key)) {
                foundIdentity = true;
                auto identityNode = slot.retainedResourceIdentities.extract(identity);
                auto lease = std::move(storedLease);
                slot.retainedResourceLeases.pop_back();
                guard.unlock();
                auto released = lease.release();
                if (released.isErr()) {
                    guard.lock();
                    slot.retainedResourceIdentities.insert(std::move(identityNode));
                    slot.retainedResourceLeases.push_back(std::move(lease));
                    return released;
                }
                break;
            }
        if (!foundIdentity)
            return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
                vernon::RhiErrorCode::LifecycleFailure, {"rollback_command_resource_identity", checkpoint, 0}})};
    }
}

vernon::Result<void, vernon::RhiError> teardownEncoder(void *context, EncoderHandle) noexcept {
    auto &slot = *static_cast<EncoderSlot *>(context);
    VernonRhiDevice device{};
    uint64_t native{};
    bool abandonNative{};
    bool activeCounted{};
    uint64_t uploadBytes{};
    std::vector<EncoderSlot::Cleanup> cleanups;
    std::vector<EncoderSlot::Cleanup> rollbacks;
    std::vector<vernon::rhi::RetainedRhiResourceLease> resources;
    vernon::rhi::CommandDeviceStateControl *state{};
    {
        std::lock_guard<std::mutex> guard(slot.mutex);
        if (slot.initializing || slot.busy || slot.rendering)
            return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
                vernon::RhiErrorCode::LifecycleFailure, {"teardown_command_encoder_busy", slot.native, 0}})};
        state = slot.commandState;
        if (!state)
            return vernon::Result<void, vernon::RhiError>{vernon::err(
                vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure, {"teardown_command_encoder_state", 0, 0}})};
        slot.failed = true;
        device = slot.device;
        native = slot.native;
        abandonNative = std::exchange(slot.nativeOwned, false);
        activeCounted = slot.activeCounted;
        uploadBytes = std::exchange(slot.admittedUploadBytes, 0);
        cleanups = std::move(slot.cleanups);
        rollbacks = std::move(slot.rollbacks);
        resources = std::move(slot.retainedResourceLeases);
        slot.cleanupIdentities.clear();
        slot.rollbackIdentities.clear();
        slot.retainedResourceIdentities.clear();
    }
    if (abandonNative)
        vernon::rhi::abandonCommandRecording(device, native);
    for (auto rollback = rollbacks.rbegin(); rollback != rollbacks.rend(); ++rollback)
        rollback->function(rollback->context, rollback->object);
    for (auto cleanup = cleanups.rbegin(); cleanup != cleanups.rend(); ++cleanup)
        cleanup->function(cleanup->context, cleanup->object);
    auto released = releaseLeases(resources);
    releaseUploadBytes(*state, uploadBytes);
    if (released.isErr()) {
        std::lock_guard<std::mutex> guard(slot.mutex);
        slot.retainedResourceLeases = std::move(resources);
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(released).error())};
    }
    resources.clear();
    if (activeCounted) {
        std::lock_guard<std::mutex> guard(state->quotaMutex);
        if (!state->activeEncoderCount)
            vernon::resultContractViolation();
        --state->activeEncoderCount;
    }
    {
        std::lock_guard<std::mutex> guard(slot.mutex);
        slot.activeCounted = false;
        slot.alive = false;
        slot.device = {};
        slot.commandState = nullptr;
    }
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownCompletion(void *context, CompletionHandle) noexcept {
    auto &slot = *static_cast<CompletionSlot *>(context);
    vernon::rhi::CommandDeviceStateControl *state{};
    bool liveCounted{};
    {
        std::lock_guard<std::mutex> guard(slot.mutex);
        state = slot.commandState;
        if (!state)
            return vernon::Result<void, vernon::RhiError>{vernon::err(
                vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure, {"teardown_completion_state", 0, 0}})};
        if (slot.observing || slot.backendRetirementRequired || slot.admissionHeld || slot.admittedUploadBytes ||
            !slot.cleanups.empty())
            return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
                vernon::RhiErrorCode::LifecycleFailure, {"teardown_completion_not_finalized", slot.native, 0}})};
        for (const auto &lease : slot.retainedResourceLeases)
            if (lease.active())
                return vernon::Result<void, vernon::RhiError>{
                    vernon::err(vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure,
                                                 {"teardown_completion_retained_resource", slot.native, 0}})};
        liveCounted = slot.liveCounted;
        slot.liveCounted = false;
        slot.device = {};
        slot.commandState = nullptr;
        slot.cleanups.clear();
        slot.retainedResourceLeases.clear();
    }
    if (liveCounted) {
        std::lock_guard<std::mutex> guard(state->quotaMutex);
        if (!state->liveCompletionCount)
            vernon::resultContractViolation();
        --state->liveCompletionCount;
    }
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

VernonRhiStatus observeCompletion(PinnedCommandSlot<CompletionSlot> &pinned) {
    CompletionSlot *slot = pinned.slot;
    std::vector<EncoderSlot::Cleanup> cleanups;
    std::vector<vernon::rhi::RetainedRhiResourceLease> resources;
    bool retireBackend{};
    bool backendCompleted{true};
    bool releaseAdmission{};
    uint64_t uploadBytes{};
    {
        std::unique_lock lock(slot->mutex);
        if (slot->initializing)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
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
        resources = std::move(slot->retainedResourceLeases);
        uploadBytes = std::exchange(slot->admittedUploadBytes, 0);
        slot->backendRetirementRequired = false;
        releaseAdmission = std::exchange(slot->admissionHeld, false);
    }
    for (auto cleanup = cleanups.rbegin(); cleanup != cleanups.rend(); ++cleanup)
        cleanup->function(cleanup->context, cleanup->object);
    auto released = releaseLeases(resources);
    auto &state = vernon::rhi::commandDeviceState(pinned.state);
    releaseUploadBytes(state, uploadBytes);
    if (releaseAdmission)
        releaseSubmission(state);
    VernonRhiStatus result{};
    {
        std::lock_guard lock(slot->mutex);
        if (released.isErr()) {
            slot->state = VERNON_RHI_COMPLETION_FAILED;
            slot->result = vernon::toVernonRhiStatus(released.error());
            slot->retainedResourceLeases = std::move(resources);
        } else {
            resources.clear();
            slot->retainedResourceLeases.clear();
            slot->finalized = true;
        }
        slot->observing = false;
        result = slot->result;
    }
    slot->condition.notify_all();
    return result;
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
    auto retained = commandState(device);
    if (retained.isErr())
        return false;
    auto state = std::move(retained).value();
    auto &local = commandDeviceState(state);
    std::lock_guard<std::mutex> guard(local.quotaMutex);
    return local.activeEncoderCount != 0;
}

extern "C" VernonRhiStatus vernonRhiDeviceSetCommandLimits(VernonRhiDevice device,
                                                           const VernonRhiCommandLimits *limits) {
    if (!limits || limits->struct_size < sizeof(*limits) || !limits->max_active_recordings ||
        !limits->max_in_flight_submissions || !limits->max_live_completions || !limits->max_upload_bytes)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto retained = vernon::rhi::commandState(device);
    if (retained.isErr())
        return vernon::toVernonRhiStatus(retained.error());
    auto stateRef = std::move(retained).value();
    auto &state = vernon::rhi::commandDeviceState(stateRef);
    std::scoped_lock creationGuards(state.encoderCreationMutex, state.completionCreationMutex);
    std::lock_guard<std::mutex> guard(state.quotaMutex);
    if (state.activeEncoderCount || state.inFlightSubmissionCount || state.admittedUploadBytes ||
        state.liveCompletionCount)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VernonRhiCommandLimits normalized = *limits;
    normalized.struct_size = sizeof(normalized);
    std::fill(std::begin(normalized.reserved), std::end(normalized.reserved), uint64_t{0});
    state.limits = normalized;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceGetCommandLimits(VernonRhiDevice device, VernonRhiCommandLimits *output) {
    if (!output || output->struct_size < sizeof(*output))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto retained = vernon::rhi::commandState(device);
    if (retained.isErr())
        return vernon::toVernonRhiStatus(retained.error());
    auto stateRef = std::move(retained).value();
    auto &state = vernon::rhi::commandDeviceState(stateRef);
    std::lock_guard<std::mutex> guard(state.quotaMutex);
    *output = state.limits;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceCreateCommandEncoder(VernonRhiDevice device,
                                                               const VernonRhiCommandEncoderDescriptor *descriptor,
                                                               VernonRhiCommandEncoder *output) {
    if (output)
        *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto retained = vernon::rhi::commandState(device);
    if (retained.isErr())
        return vernon::toVernonRhiStatus(retained.error());
    auto stateRef = std::move(retained).value();
    auto &state = vernon::rhi::commandDeviceState(stateRef);
    std::lock_guard<std::mutex> creationGuard(state.encoderCreationMutex);
    auto creation = vernon::rhi::ResourceCreationReservation::create(state.owner);
    if (creation.isErr())
        return vernon::toVernonRhiStatus(creation.error());
    const bool exclusive =
        !(vernon::rhi::deviceCommandCapabilities(device) & vernon::rhi::BackendCommandIndependentRecording);
    EncoderSlot *slot{};
    {
        std::lock_guard<std::mutex> guard(state.quotaMutex);
        if (state.activeEncoderCount && exclusive)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        if (state.activeEncoderCount >= state.limits.max_active_recordings)
            return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    }
    const size_t slotCount = state.encoderSlots.size();
    for (size_t index = 0; index < slotCount; ++index) {
        EncoderSlot *candidate = state.encoderSlots.get(index);
        if (!candidate->lifecycle.snapshot().occupied) {
            slot = candidate;
            break;
        }
    }
    if (!slot) {
        if (slotCount >= UINT32_MAX)
            return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
        auto lifecycle = EncoderLifecycle::create(static_cast<uint32_t>(slotCount));
        if (lifecycle.isErr())
            return vernon::toVernonRhiStatus(lifecycle.error());
        auto appended = state.encoderSlots.emplace("reserve_command_encoder", std::move(lifecycle).value());
        if (appended.isErr())
            return vernon::toVernonRhiStatus(appended.error());
        slot = appended.value();
    }
    {
        std::lock_guard<std::mutex> slotGuard(slot->mutex);
        slot->commandState = &state;
        slot->device = device;
        slot->backend = VERNON_RHI_BACKEND_CUDA;
        slot->native = 0;
        slot->stats = {};
        slot->renderX = slot->renderY = 0;
        slot->renderWidth = slot->renderHeight = slot->renderLayers = 0;
        slot->colorTargets = {};
        slot->colorResources = {};
        slot->depthTarget = slot->depthResource = 0;
        slot->backendRendering = 0;
        slot->backendRenderingObject = 0;
        slot->alive = true;
        slot->initializing = true;
        slot->busy = false;
        slot->rendering = false;
        slot->hasRenderingDescriptor = false;
        slot->finished = false;
        slot->failed = false;
        slot->nativeOwned = false;
        slot->activeCounted = false;
        slot->exclusiveDeviceSlot = exclusive;
        slot->colorOperations.clear();
        slot->depthOperation = {};
        slot->cleanups.clear();
        slot->rollbacks.clear();
        slot->cleanupIdentities.clear();
        slot->rollbackIdentities.clear();
        slot->retainedResourceIdentities.clear();
        slot->retainedResourceLeases.clear();
        slot->pendingWriteResources.clear();
        slot->admittedUploadBytes = 0;
        slot->unknownPendingWrites = false;
    }

    uint64_t native{};
    VernonRhiBackend backend{};
    if (!vernon::rhi::beginCommandRecording(device, native, backend)) {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->alive = false;
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->native = native;
        slot->backend = backend;
        slot->nativeOwned = true;
        slot->initializing = false;
    }
    auto published = slot->lifecycle.publish(std::move(creation).value(), slot, teardownEncoder);
    if (published.isErr()) {
        vernon::rhi::abandonCommandRecording(device, native);
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->alive = false;
        return vernon::toVernonRhiStatus(published.error());
    }
    {
        std::lock_guard<std::mutex> guard(state.quotaMutex);
        ++state.activeEncoderCount;
    }
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->activeCounted = true;
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyCommandEncoder(VernonRhiDevice device,
                                                                VernonRhiCommandEncoder encoder) {
    auto retained = vernon::rhi::commandState(device);
    if (retained.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto stateRef = std::move(retained).value();
    auto &state = vernon::rhi::commandDeviceState(stateRef);
    EncoderSlot *slot = state.encoderSlots.get(encoder.index);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto destroyed = slot->lifecycle.destroyPublic({encoder.index, encoder.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : vernon::toVernonRhiStatus(destroyed.error());
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
    const uint64_t key = logicalResourceKey(encoder);
    uint64_t native{};
    size_t resourceCheckpoint{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->busy = true;
        native = slot->native;
        resourceCheckpoint = slot->retainedResourceLeases.size();
    }
    for (size_t index = 0; index < barrierCount; ++index) {
        const vernon::rhi::ResourceKind kind =
            barriers[index].is_image ? vernon::rhi::ResourceKind::Image : vernon::rhi::ResourceKind::Buffer;
        const uint64_t resource = barriers[index].is_image ? logicalResourceKey(barriers[index].image)
                                                           : logicalResourceKey(barriers[index].buffer);
        auto retained = retainCommandResourceImpl(device, key, kind, resource, true);
        if (retained.isErr()) {
            auto rolledBack = rollbackRetainedResources(device, *slot.slot, resourceCheckpoint);
            {
                std::lock_guard<std::mutex> guard(slot->mutex);
                slot->busy = false;
            }
            if (rolledBack.isErr())
                return vernon::toVernonRhiStatus(rolledBack.error());
            return vernon::toVernonRhiStatus(retained.error());
        }
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
    const uint64_t key = logicalResourceKey(encoder);
    const uint64_t sourceResource = logicalResourceKey(source);
    const uint64_t destinationResource = logicalResourceKey(destination);
    uint64_t native{};
    size_t resourceCheckpoint{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->busy = true;
        native = slot->native;
        resourceCheckpoint = slot->retainedResourceLeases.size();
    }
    auto retainedSource =
        retainCommandResourceImpl(device, key, vernon::rhi::ResourceKind::Buffer, sourceResource, true);
    if (retainedSource.isErr()) {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        return vernon::toVernonRhiStatus(retainedSource.error());
    }
    auto retainedDestination =
        retainCommandResourceImpl(device, key, vernon::rhi::ResourceKind::Buffer, destinationResource, true);
    if (retainedDestination.isErr()) {
        auto rolledBack = rollbackRetainedResources(device, *slot.slot, resourceCheckpoint);
        {
            std::lock_guard<std::mutex> guard(slot->mutex);
            slot->busy = false;
        }
        if (rolledBack.isErr())
            return vernon::toVernonRhiStatus(rolledBack.error());
        return vernon::toVernonRhiStatus(retainedDestination.error());
    }
    const VernonRhiStatus recordStatus =
        vernon::rhi::recordBufferCopy(device, native, source, sourceOffset, destination, destinationOffset, size);
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        if (recordStatus == VERNON_RHI_STATUS_OK) {
            try {
                slot->pendingWriteResources.insert({vernon::rhi::ResourceKind::Buffer, destinationResource});
            } catch (const std::bad_alloc &) {
                slot->unknownPendingWrites = true;
            }
        } else
            slot->failed = true;
    }
    return recordStatus;
}

namespace {

bool imageCopyBoxValid(const VernonRhiImageDescriptor &image, uint32_t mip, uint32_t layer, uint32_t x, uint32_t y,
                       uint32_t z, uint32_t width, uint32_t height, uint32_t depth) {
    if (mip >= image.mip_levels || layer >= image.array_layers || !width || !height || !depth)
        return false;
    const uint32_t mipWidth = vernon::rhi::imageMipExtent(image.width, mip);
    const uint32_t mipHeight = vernon::rhi::imageMipExtent(image.height, mip);
    const uint32_t mipDepth =
        image.dimension == VERNON_RHI_IMAGE_3D ? vernon::rhi::imageMipExtent(image.depth, mip) : 1;
    return x < mipWidth && width <= mipWidth - x && y < mipHeight && height <= mipHeight - y && z < mipDepth &&
           depth <= mipDepth - z && (image.dimension == VERNON_RHI_IMAGE_3D || (z == 0 && depth == 1));
}

bool intervalsOverlap(uint32_t firstOffset, uint32_t firstSize, uint32_t secondOffset, uint32_t secondSize) {
    return uint64_t{firstOffset} < uint64_t{secondOffset} + secondSize &&
           uint64_t{secondOffset} < uint64_t{firstOffset} + firstSize;
}

bool imageCopyBoxesOverlap(const VernonRhiImageCopyRegion &first, bool firstSource,
                           const VernonRhiImageCopyRegion &second, bool secondSource) {
    const uint32_t firstMip = firstSource ? first.source_mip_level : first.destination_mip_level;
    const uint32_t firstLayer = firstSource ? first.source_array_layer : first.destination_array_layer;
    const uint32_t firstX = firstSource ? first.source_x : first.destination_x;
    const uint32_t firstY = firstSource ? first.source_y : first.destination_y;
    const uint32_t firstZ = firstSource ? first.source_z : first.destination_z;
    const uint32_t secondMip = secondSource ? second.source_mip_level : second.destination_mip_level;
    const uint32_t secondLayer = secondSource ? second.source_array_layer : second.destination_array_layer;
    const uint32_t secondX = secondSource ? second.source_x : second.destination_x;
    const uint32_t secondY = secondSource ? second.source_y : second.destination_y;
    const uint32_t secondZ = secondSource ? second.source_z : second.destination_z;
    return firstMip == secondMip && firstLayer == secondLayer && (first.aspects & second.aspects) &&
           intervalsOverlap(firstX, first.width, secondX, second.width) &&
           intervalsOverlap(firstY, first.height, secondY, second.height) &&
           intervalsOverlap(firstZ, first.depth, secondZ, second.depth);
}

bool validateImageCopies(VernonRhiDevice device, VernonRhiImage source, VernonRhiImage destination,
                         const VernonRhiImageCopyRegion *regions, size_t regionCount) {
    VernonRhiImageDescriptor sourceDescriptor{};
    VernonRhiImageDescriptor destinationDescriptor{};
    const uint64_t sourceResource = vernon::rhi::imageResource(device, source);
    const uint64_t destinationResource = vernon::rhi::imageResource(device, destination);
    if (!sourceResource || !destinationResource ||
        !vernon::rhi::describeImageResource(device, sourceResource, sourceDescriptor) ||
        !vernon::rhi::describeImageResource(device, destinationResource, destinationDescriptor) ||
        sourceDescriptor.format != destinationDescriptor.format ||
        sourceDescriptor.dimension != destinationDescriptor.dimension ||
        sourceDescriptor.sample_count != destinationDescriptor.sample_count ||
        !(sourceDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(destinationDescriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) || sourceResource == destinationResource)
        return false;
    const uint32_t availableAspects = vernon::rhi::imageFormatAspects(sourceDescriptor.format);
    for (size_t index = 0; index < regionCount; ++index) {
        const VernonRhiImageCopyRegion &region = regions[index];
        if (region.struct_size < sizeof(region) || !region.aspects || (region.aspects & ~availableAspects) ||
            std::any_of(
                std::begin(region.reserved), std::end(region.reserved), [](uint32_t value) { return value != 0; }) ||
            !imageCopyBoxValid(sourceDescriptor, region.source_mip_level, region.source_array_layer, region.source_x,
                               region.source_y, region.source_z, region.width, region.height, region.depth) ||
            !imageCopyBoxValid(destinationDescriptor, region.destination_mip_level, region.destination_array_layer,
                               region.destination_x, region.destination_y, region.destination_z, region.width,
                               region.height, region.depth))
            return false;
        for (size_t previous = 0; previous < index; ++previous)
            if (imageCopyBoxesOverlap(regions[previous], false, region, false))
                return false;
    }
    return true;
}

} // namespace

extern "C" VernonRhiStatus vernonRhiCommandEncoderCopyImage(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                            VernonRhiImage source, VernonRhiImage destination,
                                                            const VernonRhiImageCopyRegion *regions,
                                                            size_t regionCount) {
    auto slot = lookup(device, encoder);
    if (!slot || !validHandle(source) || !validHandle(destination) || !regionCount || !regions ||
        !validateImageCopies(device, source, destination, regions, regionCount))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t key = logicalResourceKey(encoder);
    const uint64_t sourceResource = logicalResourceKey(source);
    const uint64_t destinationResource = logicalResourceKey(destination);
    uint64_t native{};
    size_t resourceCheckpoint{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || slot->rendering || slot->finished || slot->failed)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        slot->busy = true;
        native = slot->native;
        resourceCheckpoint = slot->retainedResourceLeases.size();
    }
    auto retainedSource =
        retainCommandResourceImpl(device, key, vernon::rhi::ResourceKind::Image, sourceResource, true);
    if (retainedSource.isErr()) {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        return vernon::toVernonRhiStatus(retainedSource.error());
    }
    auto retainedDestination =
        retainCommandResourceImpl(device, key, vernon::rhi::ResourceKind::Image, destinationResource, true);
    if (retainedDestination.isErr()) {
        auto rolledBack = rollbackRetainedResources(device, *slot.slot, resourceCheckpoint);
        {
            std::lock_guard<std::mutex> guard(slot->mutex);
            slot->busy = false;
        }
        if (rolledBack.isErr())
            return vernon::toVernonRhiStatus(rolledBack.error());
        return vernon::toVernonRhiStatus(retainedDestination.error());
    }
    const VernonRhiStatus recordStatus =
        vernon::rhi::recordImageCopy(device, key, native, source, destination, regions, regionCount);
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        if (recordStatus == VERNON_RHI_STATUS_OK) {
            try {
                slot->pendingWriteResources.insert({vernon::rhi::ResourceKind::Image, destinationResource});
            } catch (const std::bad_alloc &) {
                slot->unknownPendingWrites = true;
            }
        } else
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
        device, slot->native, slot->backend, slot->backendRendering, slot->backendRenderingObject, slot->renderX,
        slot->renderY, slot->renderWidth, slot->renderHeight, slot->renderLayers, slot->colorTargets[location],
        location, clearColor);
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
        device, slot->native, slot->backend, slot->backendRendering, slot->backendRenderingObject, slot->renderX,
        slot->renderY, slot->renderWidth, slot->renderHeight, slot->renderLayers, slot->depthTarget, clearDepth,
        clearStencil, aspects);
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
    auto &state = vernon::rhi::commandDeviceState(slot.state);
    const VernonRhiStatus admissionStatus = reserveSubmission(state);
    if (admissionStatus != VERNON_RHI_STATUS_OK)
        return admissionStatus;
    auto creation = vernon::rhi::ResourceCreationReservation::create(state.owner);
    if (creation.isErr()) {
        releaseSubmission(state);
        return vernon::toVernonRhiStatus(creation.error());
    }
    std::unique_lock<std::mutex> completionCreationGuard(state.completionCreationMutex);
    auto reserved = reserveCompletionSlot(state);
    if (reserved.isErr()) {
        releaseSubmission(state);
        return vernon::toVernonRhiStatus(reserved.error());
    }
    CompletionSlot *completion = reserved.value();
    {
        std::lock_guard<std::mutex> guard(completion->mutex);
        completion->commandState = &state;
        completion->device = device;
        completion->backend = VERNON_RHI_BACKEND_CUDA;
        completion->native = 0;
        completion->stats = {};
        completion->state = VERNON_RHI_COMPLETION_PENDING;
        completion->result = VERNON_RHI_STATUS_OK;
        completion->externalSignalRequired = false;
        completion->backendRetirementRequired = false;
        completion->initializing = true;
        completion->observing = false;
        completion->finalized = false;
        completion->admissionHeld = false;
        completion->liveCounted = false;
        completion->cleanups.clear();
        completion->retainedResourceLeases.clear();
        completion->admittedUploadBytes = 0;
    }
    auto published = completion->lifecycle.publish(std::move(creation).value(), completion, teardownCompletion);
    if (published.isErr()) {
        releaseSubmission(state);
        return vernon::toVernonRhiStatus(published.error());
    }
    const VernonRhiCompletion handle{published.value().index, published.value().generation};
    {
        std::lock_guard<std::mutex> guard(state.quotaMutex);
        ++state.liveCompletionCount;
    }
    {
        std::lock_guard<std::mutex> guard(completion->mutex);
        completion->liveCounted = true;
    }
    completionCreationGuard.unlock();
    uint64_t native{};
    bool computeWrites{};
    bool completed{};
    bool externalCompletion{};
    bool invalidEncoder{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->initializing || slot->busy || !slot->finished || slot->failed) {
            invalidEncoder = true;
        } else {
            slot->busy = true;
            native = slot->native;
            computeWrites = slot->unknownPendingWrites || !slot->pendingWriteResources.empty();
        }
    }
    if (invalidEncoder) {
        (void)completion->lifecycle.destroyPublic({handle.index, handle.generation});
        releaseSubmission(state);
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    if (slot.pin.value().release().isErr())
        vernon::resultContractViolation();
    auto destroyAttempt = slot.slot->lifecycle.prepareDestroyPublic({encoder.index, encoder.generation});
    if (destroyAttempt.isErr()) {
        {
            std::lock_guard<std::mutex> guard(slot->mutex);
            slot->busy = false;
        }
        (void)completion->lifecycle.destroyPublic({handle.index, handle.generation});
        releaseSubmission(state);
        return vernon::toVernonRhiStatus(destroyAttempt.error());
    }
    const bool submitted =
        vernon::rhi::submitCommandRecording(device, native, computeWrites, completed, externalCompletion);
    if (!submitted) {
        {
            std::lock_guard<std::mutex> guard(slot->mutex);
            slot->busy = false;
        }
        auto rolledBack = destroyAttempt.value().rollback();
        if (rolledBack.isErr())
            vernon::resultContractViolation();
        (void)completion->lifecycle.destroyPublic({handle.index, handle.generation});
        releaseSubmission(state);
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    VernonRhiBackend submittedBackend{};
    VernonRhiCommandEncoderStats submittedStats{};
    std::vector<EncoderSlot::Cleanup> submittedCleanups;
    std::vector<vernon::rhi::RetainedRhiResourceLease> submittedResources;
    uint64_t submittedUploadBytes{};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        slot->busy = false;
        slot->alive = false;
        slot->nativeOwned = false;
        slot->rollbacks.clear();
        ++slot->stats.submission_count;
        submittedBackend = slot->backend;
        submittedStats = slot->stats;
        submittedCleanups = std::move(slot->cleanups);
        submittedResources = std::move(slot->retainedResourceLeases);
        slot->retainedResourceIdentities.clear();
        submittedUploadBytes = std::exchange(slot->admittedUploadBytes, 0);
    }
    {
        std::lock_guard<std::mutex> guard(completion->mutex);
        completion->backend = submittedBackend;
        completion->native = native;
        completion->stats = submittedStats;
        completion->state = completed ? VERNON_RHI_COMPLETION_SUCCEEDED : VERNON_RHI_COMPLETION_PENDING;
        completion->externalSignalRequired = externalCompletion;
        completion->backendRetirementRequired = !completed;
        completion->admissionHeld = true;
        completion->cleanups = std::move(submittedCleanups);
        completion->retainedResourceLeases = std::move(submittedResources);
        completion->admittedUploadBytes = submittedUploadBytes;
    }
    auto encoderDestroyed = destroyAttempt.value().commit();
    if (encoderDestroyed.isErr())
        vernon::resultContractViolation();
    *output = handle;
    {
        std::lock_guard<std::mutex> guard(completion->mutex);
        completion->initializing = false;
    }
    if (completed) {
        auto pinnedCompletion = lookup(device, handle);
        if (!pinnedCompletion)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        (void)observeCompletion(pinnedCompletion);
    }
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
        if (slot->initializing)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
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
        if (slot->initializing || !slot->externalSignalRequired || slot->state != VERNON_RHI_COMPLETION_PENDING)
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
    if (slot->initializing)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = slot->stats;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyCompletion(VernonRhiDevice device, VernonRhiCompletion completion) {
    auto slot = lookup(device, completion);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VernonRhiStatus status = observeCompletion(slot);
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->finalized)
            return status;
    }
    CompletionSlot *raw = slot.slot;
    if (slot.pin.value().release().isErr())
        vernon::resultContractViolation();
    auto destroyed = raw->lifecycle.destroyPublic({completion.index, completion.generation});
    if (destroyed.isErr() && status == VERNON_RHI_STATUS_OK)
        status = vernon::toVernonRhiStatus(destroyed.error());
    return status;
}

uint64_t vernon::rhi::commandEncoderKey(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    auto slot = lookup(device, encoder);
    if (!slot)
        return 0;
    std::lock_guard<std::mutex> guard(slot->mutex);
    return slot->alive && !slot->initializing ? logicalResourceKey(encoder) : 0;
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
    if (!slot->alive || !slot->rendering)
        return false;
    if (!slot->hasRenderingDescriptor)
        return true;
    if (index >= slot->colorOperations.size())
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
    if (!slot->alive || !slot->rendering)
        return false;
    if (!slot->hasRenderingDescriptor || !slot->depthOperation.present)
        return true;
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

void vernon::rhi::rollbackCommandRenderingClaim(VernonRhiDevice device, uint64_t key, uint32_t backendKind) noexcept {
    auto slot = lookupKey(device, key);
    if (!slot)
        return;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (slot->backendRendering == backendKind && !slot->backendRenderingObject)
        slot->backendRendering = 0;
}

uint64_t vernon::rhi::commandRenderingObject(VernonRhiDevice device, uint64_t key, uint64_t candidate) {
    auto slot = lookupKey(device, key);
    if (!slot)
        return 0;
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished)
        return 0;
    if (candidate && !slot->backendRenderingObject)
        slot->backendRenderingObject = candidate;
    return slot->backendRenderingObject;
}

vernon::Result<void, vernon::RhiError>
vernon::rhi::installCommandRenderingObject(VernonRhiDevice device, uint64_t key, uint64_t candidate,
                                           const uint64_t *colors, const uint64_t *resources, size_t colorCount,
                                           uint64_t depth, uint64_t depthResource) noexcept {
    if (!candidate || colorCount > VERNON_RHI_MAX_COLOR_ATTACHMENTS || (colorCount && (!colors || !resources)))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"install_command_rendering_object", key, 0}})};
    auto slot = lookupKey(device, key);
    if (!slot)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"install_command_rendering_object", key, 0}})};
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->busy || !slot->rendering || slot->finished || slot->backendRenderingObject)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::LifecycleFailure, {"install_command_rendering_object", key, 0}})};
    if (colorCount) {
        std::copy(colors, colors + colorCount, slot->colorTargets.begin());
        std::copy(resources, resources + colorCount, slot->colorResources.begin());
    }
    slot->depthTarget = depth;
    slot->depthResource = depthResource;
    slot->backendRenderingObject = candidate;
    return Result<void, RhiError>{ok()};
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

static vernon::Result<void, vernon::RhiError> retainCommandResourceImpl(VernonRhiDevice device, uint64_t key,
                                                                        vernon::rhi::ResourceKind kind,
                                                                        uint64_t resourceKey,
                                                                        bool expectedBusy) noexcept {
    using namespace vernon;
    using namespace vernon::rhi;
    auto slot = lookupKey(device, key);
    if (!slot || !resourceKey)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"retain_command_resource", resourceKey, 0}})};
    const EncoderSlot::RetainedResource identity{kind, resourceKey};
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->finished || slot->busy != expectedBusy)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"retain_command_resource", resourceKey, 0}})};
        if (slot->retainedResourceIdentities.find(identity) != slot->retainedResourceIdentities.end())
            return Result<void, RhiError>{ok()};
    }
    auto retained = retainResource(device, kind, resourceKey);
    if (retained.isErr())
        return Result<void, RhiError>{err(std::move(retained).error())};
    auto lease = std::move(retained).value();
    bool releaseLease = false;
    RhiError commitError{};
    bool commitFailed = false;
    {
        std::lock_guard<std::mutex> guard(slot->mutex);
        if (!slot->alive || slot->finished || slot->busy != expectedBusy) {
            releaseLease = true;
            commitFailed = true;
            commitError = {RhiErrorCode::LifecycleFailure, {"retain_command_resource_commit", resourceKey, 0}};
        } else if (slot->retainedResourceIdentities.find(identity) != slot->retainedResourceIdentities.end()) {
            releaseLease = true;
        } else {
            try {
                slot->retainedResourceIdentities.insert(identity);
                slot->retainedResourceLeases.push_back(std::move(lease));
            } catch (const std::bad_alloc &) {
                slot->retainedResourceIdentities.erase(identity);
                releaseLease = true;
                commitFailed = true;
                commitError = {RhiErrorCode::ResourceExhausted, {"retain_command_resource_commit", resourceKey, 0}};
            }
        }
    }
    if (releaseLease) {
        auto released = lease.release();
        if (released.isErr())
            return Result<void, RhiError>{err(std::move(released).error())};
    }
    if (commitFailed)
        return Result<void, RhiError>{err(commitError)};
    return Result<void, RhiError>{ok()};
}

vernon::Result<void, vernon::RhiError> vernon::rhi::retainCommandResource(VernonRhiDevice device, uint64_t key,
                                                                          ResourceKind kind,
                                                                          uint64_t resourceKey) noexcept {
    return retainCommandResourceImpl(device, key, kind, resourceKey, false);
}

vernon::Result<uint64_t, vernon::RhiError> vernon::rhi::resolveCommandResource(VernonRhiDevice device, uint64_t key,
                                                                               ResourceKind kind,
                                                                               uint64_t resourceKey) noexcept {
    auto slot = lookupKey(device, key);
    if (!slot || !resourceKey)
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_command_resource", resourceKey, 0}})};
    std::unique_lock<std::mutex> guard(slot->mutex);
    if (!slot->alive || slot->initializing || slot->finished)
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::LifecycleFailure, {"resolve_command_resource", resourceKey, 0}})};
    for (auto &lease : slot->retainedResourceLeases) {
        if (!lease.authorizes(device.index, device.generation, static_cast<uint32_t>(kind), resourceKey))
            continue;
        auto pinned = lease.pin();
        if (pinned.isErr())
            return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
        guard.unlock();
        return resolvePinnedResource(device, kind, resourceKey);
    }
    return Result<uint64_t, RhiError>{
        err(RhiError{RhiErrorCode::LifecycleFailure, {"resolve_command_resource", resourceKey, 0}})};
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
