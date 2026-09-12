#include "rhi/device_registry.h"
#include "rhi/rhi_lifecycle.h"

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

std::atomic<unsigned> failNothrowAllocationCountdown{};

void *operator new(std::size_t size) {
    if (void *memory = std::malloc(size))
        return memory;
    std::abort();
}

void *operator new[](std::size_t size) { return ::operator new(size); }

void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
    unsigned current = failNothrowAllocationCountdown.load(std::memory_order_acquire);
    while (current != 0) {
        if (failNothrowAllocationCountdown.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel,
                                                                 std::memory_order_acquire)) {
            if (current == 1)
                return nullptr;
            break;
        }
    }
    return std::malloc(size);
}

void *operator new[](std::size_t size, const std::nothrow_t &tag) noexcept { return ::operator new(size, tag); }

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete(void *memory, const std::nothrow_t &) noexcept { std::free(memory); }
void operator delete[](void *memory, const std::nothrow_t &) noexcept { std::free(memory); }

namespace {

struct InjectedFailurePolicy {
    static void failNext(vernon::rhi::LifecycleAllocationPoint point) noexcept {
        next.store(static_cast<unsigned>(point) + 1, std::memory_order_release);
    }

    [[nodiscard]] static bool fail(vernon::rhi::LifecycleAllocationPoint point) noexcept {
        unsigned expected = static_cast<unsigned>(point) + 1;
        return next.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
    }

    static void snapshotCreated() noexcept { liveSnapshots.fetch_add(1, std::memory_order_relaxed); }
    static void snapshotDestroyed() noexcept { liveSnapshots.fetch_sub(1, std::memory_order_relaxed); }
    static void afterSnapshotLoad() noexcept {
        if (!pauseAfterSnapshotLoad.load(std::memory_order_acquire))
            return;
        snapshotLoaded.store(true, std::memory_order_release);
        while (!resumeSnapshotReader.load(std::memory_order_acquire)) {
        }
    }

    inline static std::atomic<unsigned> next{};
    inline static std::atomic<unsigned> liveSnapshots{};
    inline static std::atomic<bool> pauseAfterSnapshotLoad{};
    inline static std::atomic<bool> snapshotLoaded{};
    inline static std::atomic<bool> resumeSnapshotReader{};
};

struct TestDevice {
    explicit TestDevice(std::atomic<unsigned> &destructions) noexcept : destructions(&destructions) {}
    ~TestDevice() noexcept { destructions->fetch_add(1, std::memory_order_relaxed); }

    [[nodiscard]] vernon::Result<void, vernon::RhiError> initialize(bool fail = false) noexcept {
        if (fail)
            return vernon::Result<void, vernon::RhiError>{
                vernon::err(vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"initialize_device", 0, 0}})};
        initialized = true;
        nativeAlive = true;
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    }

    std::atomic<unsigned> *destructions;
    std::mutex mutex;
    unsigned teardownCalls{};
    bool failTeardown{};
    bool initialized{};
    bool nativeAlive{};
};

vernon::Result<void, vernon::RhiError> teardownDevice(TestDevice &device) noexcept {
    ++device.teardownCalls;
    if (device.failTeardown) {
        device.failTeardown = false;
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"teardown_device", 0, 0}})};
    }
    device.nativeAlive = false;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

struct NativeResource {
    vernon::OwnerRef *owner{};
    unsigned teardownCalls{};
    bool failTeardown{};
    bool nativeAlive{};
    bool childNativeAlive{};
    bool reservationPrecededAllocation{};
    bool childHeldDuringTeardown{};
};

struct BufferTag {};
using BufferHandle = vernon::rhi::ResourceHandle<BufferTag>;

vernon::Result<void, vernon::RhiError>
teardownTypedBuffer(void *context, vernon::rhi::ResourceHandle<vernon::rhi::BufferResourceTag>) noexcept {
    *static_cast<bool *>(context) = false;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

struct BackendDeviceContext {
    std::mutex mutex;
    std::vector<NativeResource> resources;
};

vernon::Result<void, vernon::RhiError> teardownResource(void *context, BufferHandle handle) noexcept {
    auto &device = *static_cast<BackendDeviceContext *>(context);
    if (handle.index >= device.resources.size())
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::InvalidArgument, {"teardown_resource", handle.index, handle.generation}})};
    auto &native = device.resources[handle.index];
    ++native.teardownCalls;
    native.childHeldDuringTeardown = native.owner->childCount() == 1;
    if (native.failTeardown) {
        native.failTeardown = false;
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"teardown_resource", 0, 0}})};
    }
    native.nativeAlive = false;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownCascadingResource(void *context, BufferHandle handle) noexcept {
    auto &device = *static_cast<BackendDeviceContext *>(context);
    auto &native = device.resources[handle.index];
    ++native.teardownCalls;
    if (native.failTeardown) {
        native.failTeardown = false;
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"teardown_parent", 0, 0}})};
    }
    native.childNativeAlive = false;
    native.nativeAlive = false;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

NativeResource &prepareNativeResource(BackendDeviceContext &device, std::uint32_t index, vernon::OwnerRef &owner) {
    if (device.resources.size() <= index)
        device.resources.resize(static_cast<std::size_t>(index) + 1);
    NativeResource &native = device.resources[index];
    native.owner = &owner;
    native.nativeAlive = true;
    return native;
}

using TestRegistry = vernon::rhi::DeviceRegistry<TestDevice, 1, InjectedFailurePolicy>;
using TestRegistryThree = vernon::rhi::DeviceRegistry<TestDevice, 3, InjectedFailurePolicy>;
using TestResourceSlot = vernon::rhi::ResourceLifecycleSlot<BufferTag, InjectedFailurePolicy>;
using TestRetainedLease = vernon::rhi::RetainedResourceLease<BufferTag>;
using TestReleaseAttempt = vernon::rhi::RetainedResourceReleaseAttempt<BufferTag>;

static_assert(!std::is_copy_constructible_v<TestRegistry>);
static_assert(!std::is_move_constructible_v<TestDevice>);
static_assert(!std::is_move_constructible_v<BackendDeviceContext>);
static_assert(!std::is_copy_constructible_v<vernon::rhi::DevicePublishReservation<TestDevice>>);
static_assert(std::is_nothrow_move_constructible_v<vernon::rhi::DevicePublishReservation<TestDevice>>);
static_assert(!std::is_copy_constructible_v<vernon::rhi::ResourceCreationReservation>);
static_assert(std::is_nothrow_move_constructible_v<vernon::rhi::ResourceCreationReservation>);
static_assert(!std::is_copy_constructible_v<TestRetainedLease>);
static_assert(std::is_nothrow_move_constructible_v<TestRetainedLease>);
static_assert(!std::is_copy_constructible_v<TestReleaseAttempt>);
static_assert(std::is_nothrow_move_constructible_v<TestReleaseAttempt>);
static_assert(!std::is_same_v<TestRetainedLease, vernon::OperationPin>);
static_assert(std::is_nothrow_move_constructible_v<TestResourceSlot>);

template <typename Registry>
vernon::rhi::DeviceRegistryHandle publishInitializedDevice(Registry &registry, std::atomic<unsigned> &destructions) {
    auto reservation = registry.reserve(destructions);
    EXPECT_TRUE(reservation.isOk());
    EXPECT_TRUE(reservation.value().device().initialize().isOk());
    auto published = registry.publish(std::move(reservation).value());
    EXPECT_TRUE(published.isOk());
    return published.value();
}

vernon::OwnerRef createDeviceOwner(std::uint64_t maximumChildren = 16) {
    auto created = vernon::OwnerControlBlock::create(maximumChildren);
    EXPECT_TRUE(created.isOk());
    return std::move(created).value();
}

TestResourceSlot createResourceSlot(std::uint32_t index, std::uint32_t maximumRetained = UINT32_MAX,
                                    std::uint64_t maximumControlReferences = UINT64_MAX) {
    auto created = TestResourceSlot::create(index, maximumRetained, maximumControlReferences);
    EXPECT_TRUE(created.isOk());
    return std::move(created).value();
}

vernon::rhi::ResourceCreationReservation reserveResource(vernon::OwnerRef &owner) {
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner);
    EXPECT_TRUE(reservation.isOk());
    return std::move(reservation).value();
}

class Gate {
public:
    void open() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            open_ = true;
        }
        condition_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> guard(mutex_);
        condition_.wait(guard, [this] { return open_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool open_{};
};

TEST(RhiDeviceRegistryCoreTest, LookupRemoveAndRepublishLinearizeGeneration) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    auto reservation = registry.reserve(destructions);
    ASSERT_TRUE(reservation.isOk());
    ASSERT_TRUE(reservation.value().device().initialize().isOk());
    auto first = registry.publish(std::move(reservation).value());
    ASSERT_TRUE(first.isOk());
    {
        auto lookup = registry.lookup(first.value());
        ASSERT_TRUE(lookup.isOk());
        EXPECT_TRUE(lookup.value().device().nativeAlive);
    }

    ASSERT_TRUE(registry.remove(first.value(), teardownDevice).isOk());
    EXPECT_TRUE(registry.lookup(first.value()).isErr());
    const auto second = publishInitializedDevice(registry, destructions);
    EXPECT_EQ(second.index, first.value().index);
    EXPECT_NE(second.generation, first.value().generation);
    EXPECT_TRUE(registry.remove(second, teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, ConcurrentLookupPinPreventsNativeTeardown) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    const auto published = publishInitializedDevice(registry, destructions);

    Gate anchored;
    Gate release;
    std::thread lookupThread([&] {
        auto anchor = registry.lookup(published);
        ASSERT_TRUE(anchor.isOk());
        anchored.open();
        release.wait();
        EXPECT_TRUE(anchor.value().device().nativeAlive);
    });

    anchored.wait();
    auto rejected = registry.remove(published, teardownDevice);
    ASSERT_TRUE(rejected.isErr());
    EXPECT_EQ(rejected.error().code, vernon::RhiErrorCode::LifecycleFailure);
    release.open();
    lookupThread.join();
    EXPECT_TRUE(registry.remove(published, teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, LookupRemoveRaceNeverRetainsFreedEntry) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    for (unsigned iteration = 0; iteration != 256; ++iteration) {
        const auto published = publishInitializedDevice(registry, destructions);
        std::atomic<bool> start{};
        std::atomic<bool> lookupSucceeded{};
        std::thread reader([&] {
            while (!start.load(std::memory_order_acquire)) {
            }
            auto anchor = registry.lookup(published);
            if (anchor.isOk()) {
                EXPECT_TRUE(anchor.value().device().nativeAlive);
                lookupSucceeded.store(true, std::memory_order_release);
            }
        });

        start.store(true, std::memory_order_release);
        auto removed = registry.remove(published, teardownDevice);
        reader.join();
        if (removed.isErr()) {
            EXPECT_TRUE(lookupSucceeded.load(std::memory_order_acquire));
            ASSERT_TRUE(registry.remove(published, teardownDevice).isOk());
        }
    }
    EXPECT_EQ(destructions.load(std::memory_order_relaxed), 256u);
}

TEST(RhiDeviceRegistryCoreTest, OldSnapshotRemainsAliveUntilAdmittedReaderQuiesces) {
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 0u);
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    const auto first = publishInitializedDevice(registry, destructions);
    InjectedFailurePolicy::snapshotLoaded.store(false, std::memory_order_relaxed);
    InjectedFailurePolicy::resumeSnapshotReader.store(false, std::memory_order_relaxed);
    InjectedFailurePolicy::pauseAfterSnapshotLoad.store(true, std::memory_order_release);
    std::atomic<bool> lookupRejected{};
    std::thread reader([&] {
        auto lookup = registry.lookup(first);
        lookupRejected.store(lookup.isErr(), std::memory_order_release);
    });
    while (!InjectedFailurePolicy::snapshotLoaded.load(std::memory_order_acquire)) {
    }

    auto removed = registry.remove(first, teardownDevice);
    EXPECT_TRUE(removed.isOk());
    if (removed.isOk())
        EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 2u);
    InjectedFailurePolicy::resumeSnapshotReader.store(true, std::memory_order_release);
    reader.join();
    InjectedFailurePolicy::pauseAfterSnapshotLoad.store(false, std::memory_order_release);
    EXPECT_TRUE(lookupRejected.load(std::memory_order_acquire));

    const auto second = publishInitializedDevice(registry, destructions);
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 1u);
    EXPECT_TRUE(registry.remove(second, teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, RemoveTeardownFailureRestoresPublishedOpenEntry) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    const auto published = publishInitializedDevice(registry, destructions);
    {
        auto anchor = registry.lookup(published);
        ASSERT_TRUE(anchor.isOk());
        anchor.value().device().failTeardown = true;
    }

    auto failed = registry.remove(published, teardownDevice);
    ASSERT_TRUE(failed.isErr());
    EXPECT_EQ(failed.error().code, vernon::RhiErrorCode::BackendFailure);
    {
        auto stillPublished = registry.lookup(published);
        ASSERT_TRUE(stillPublished.isOk());
        EXPECT_TRUE(stillPublished.value().device().nativeAlive);
        EXPECT_EQ(stillPublished.value().device().teardownCalls, 1u);
    }
    EXPECT_TRUE(registry.remove(published, teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, EntryAllocationFailureLeavesStableSlotEmpty) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    InjectedFailurePolicy::failNext(vernon::rhi::LifecycleAllocationPoint::DeviceEntry);
    auto failed = registry.reserve(destructions);
    ASSERT_TRUE(failed.isErr());

    const auto published = publishInitializedDevice(registry, destructions);
    EXPECT_EQ(published.generation, 1u);
    EXPECT_TRUE(registry.remove(published, teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, SnapshotAllocationFailureDoesNotPublishReservation) {
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 0u);
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    auto reservation = registry.reserve(destructions);
    ASSERT_TRUE(reservation.isOk());
    ASSERT_TRUE(reservation.value().device().initialize().isOk());
    InjectedFailurePolicy::failNext(vernon::rhi::LifecycleAllocationPoint::RegistrySnapshot);

    auto failed = registry.publish(std::move(reservation).value());
    ASSERT_TRUE(failed.isErr());
    EXPECT_EQ(failed.error().code, vernon::RhiErrorCode::ResourceExhausted);
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(destructions.load(std::memory_order_relaxed), 1u);
    EXPECT_TRUE(registry.lookup({0, 1}).isErr());
}

TEST(RhiDeviceRegistryCoreTest, CloneRetainFailureRollsBackEarlierEntryRetains) {
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 0u);
    std::atomic<unsigned> destructions{};
    TestRegistryThree registry(2);
    const auto first = publishInitializedDevice(registry, destructions);
    const auto second = publishInitializedDevice(registry, destructions);
    {
        auto saturatedSecond = registry.lookup(second);
        ASSERT_TRUE(saturatedSecond.isOk());

        auto third = registry.reserve(destructions);
        ASSERT_TRUE(third.isOk());
        ASSERT_TRUE(third.value().device().initialize().isOk());
        auto failed = registry.publish(std::move(third).value());
        ASSERT_TRUE(failed.isErr());
        EXPECT_EQ(failed.error().code, vernon::RhiErrorCode::ResourceExhausted);

        auto firstStillRetainable = registry.lookup(first);
        ASSERT_TRUE(firstStillRetainable.isOk());
    }
    EXPECT_TRUE(registry.remove(first, teardownDevice).isOk());
    EXPECT_TRUE(registry.remove(second, teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, QuiescentWritersReclaimAllRetiredSnapshots) {
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 0u);
    std::atomic<unsigned> destructions{};
    {
        TestRegistry registry;
        for (unsigned iteration = 0; iteration != 64; ++iteration) {
            const auto handle = publishInitializedDevice(registry, destructions);
            EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 1u);
            ASSERT_TRUE(registry.remove(handle, teardownDevice).isOk());
            EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 1u);
        }
    }
    EXPECT_EQ(InjectedFailurePolicy::liveSnapshots.load(std::memory_order_relaxed), 0u);
}

TEST(RhiDeviceRegistryCoreTest, EntryRetainSaturationIsFallible) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry(1);
    auto reservation = registry.reserve(destructions);
    ASSERT_TRUE(reservation.isOk());
    ASSERT_TRUE(reservation.value().device().initialize().isOk());
    auto published = registry.publish(std::move(reservation).value());
    ASSERT_TRUE(published.isOk());
    auto saturated = registry.lookup(published.value());
    ASSERT_TRUE(saturated.isErr());
    EXPECT_EQ(saturated.error().code, vernon::RhiErrorCode::ResourceExhausted);
    EXPECT_TRUE(registry.remove(published.value(), teardownDevice).isOk());
}

TEST(RhiLifecycleCoreTest, RealNothrowAllocationFailuresAreReported) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    failNothrowAllocationCountdown.store(3, std::memory_order_release);
    auto entryFailure = registry.reserve(destructions);
    ASSERT_TRUE(entryFailure.isErr());
    EXPECT_EQ(entryFailure.error().code, vernon::RhiErrorCode::ResourceExhausted);

    auto reservation = registry.reserve(destructions);
    ASSERT_TRUE(reservation.isOk());
    ASSERT_TRUE(reservation.value().device().initialize().isOk());
    failNothrowAllocationCountdown.store(1, std::memory_order_release);
    auto snapshotFailure = registry.publish(std::move(reservation).value());
    ASSERT_TRUE(snapshotFailure.isErr());
    EXPECT_EQ(snapshotFailure.error().code, vernon::RhiErrorCode::ResourceExhausted);

    failNothrowAllocationCountdown.store(1, std::memory_order_release);
    auto controlFailure = TestResourceSlot::create(1);
    ASSERT_TRUE(controlFailure.isErr());
    EXPECT_EQ(controlFailure.error().code, vernon::RhiErrorCode::ResourceExhausted);
}

TEST(RhiDeviceRegistryCoreTest, FailedInitializationNeverPublishesAndDestroysReservation) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    {
        auto reservation = registry.reserve(destructions);
        ASSERT_TRUE(reservation.isOk());
        auto failed = reservation.value().device().initialize(true);
        ASSERT_TRUE(failed.isErr());
        EXPECT_TRUE(registry.lookup({0, 1}).isErr());
    }
    EXPECT_EQ(destructions.load(std::memory_order_relaxed), 1u);
}

TEST(RhiDeviceRegistryCoreTest, CapacityFailureDestroysInitializedReservation) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    const auto first = publishInitializedDevice(registry, destructions);
    auto second = registry.reserve(destructions);
    ASSERT_TRUE(second.isOk());
    ASSERT_TRUE(second.value().device().initialize().isOk());

    auto full = registry.publish(std::move(second).value());
    ASSERT_TRUE(full.isErr());
    EXPECT_EQ(full.error().code, vernon::RhiErrorCode::ResourceExhausted);
    EXPECT_EQ(destructions.load(std::memory_order_relaxed), 1u);
    EXPECT_TRUE(registry.remove(first, teardownDevice).isOk());
    EXPECT_EQ(destructions.load(std::memory_order_relaxed), 2u);
}

TEST(RhiDeviceRegistryCoreTest, ConcurrentLookupCannotObservePublishReservation) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    auto reservation = registry.reserve(destructions);
    ASSERT_TRUE(reservation.isOk());
    ASSERT_TRUE(reservation.value().device().initialize().isOk());

    std::atomic<bool> invisible{};
    std::thread reader([&] { invisible.store(registry.lookup({0, 1}).isErr(), std::memory_order_release); });
    reader.join();
    EXPECT_TRUE(invisible.load(std::memory_order_acquire));

    auto published = registry.publish(std::move(reservation).value());
    ASSERT_TRUE(published.isOk());
    EXPECT_TRUE(registry.lookup(published.value()).isOk());
    EXPECT_TRUE(registry.remove(published.value(), teardownDevice).isOk());
}

TEST(RhiDeviceRegistryCoreTest, CommittedResourceChildPreventsDeviceTeardown) {
    std::atomic<unsigned> destructions{};
    TestRegistry registry;
    const auto device = publishInitializedDevice(registry, destructions);
    vernon::OwnerRef owner = [&] {
        auto anchor = registry.lookup(device);
        EXPECT_TRUE(anchor.isOk());
        auto retainedOwner = anchor.value().retainOwner();
        EXPECT_TRUE(retainedOwner.isOk());
        return std::move(retainedOwner).value();
    }();

    TestResourceSlot slot = createResourceSlot(2);
    BackendDeviceContext resourceDevice;
    NativeResource &native = prepareNativeResource(resourceDevice, 2, owner);
    auto resource = slot.publish(reserveResource(owner), &resourceDevice, teardownResource);
    ASSERT_TRUE(resource.isOk());
    auto rejected = registry.remove(device, teardownDevice);
    ASSERT_TRUE(rejected.isErr());
    EXPECT_EQ(rejected.error().code, vernon::RhiErrorCode::LifecycleFailure);

    ASSERT_TRUE(slot.destroyPublic(resource.value()).isOk());
    EXPECT_TRUE(registry.remove(device, teardownDevice).isOk());
}

TEST(RhiResourceLifecycleCoreTest, ReservationPrecedesNativeActionAndPublishCommitsIt) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(3);
    auto reservation = reserveResource(owner);
    ASSERT_EQ(owner.childCount(), 1u);

    BackendDeviceContext device;
    NativeResource &native = prepareNativeResource(device, 3, owner);
    native.reservationPrecededAllocation = owner.childCount() == 1;
    auto published = slot.publish(std::move(reservation), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    EXPECT_TRUE(native.reservationPrecededAllocation);
    EXPECT_EQ(owner.childCount(), 1u);

    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());
    EXPECT_TRUE(native.childHeldDuringTeardown);
    EXPECT_FALSE(native.nativeAlive);
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(RhiResourceLifecycleCoreTest, MovedReservationRollsBackExactlyOnceBeforeNativeAllocation) {
    vernon::OwnerRef owner = createDeviceOwner();
    {
        auto reservation = reserveResource(owner);
        auto moved = std::move(reservation);
        EXPECT_EQ(owner.childCount(), 1u);
        (void)moved;
    }
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(RhiResourceLifecycleCoreTest, ControlAllocationFailureIsFallible) {
    InjectedFailurePolicy::failNext(vernon::rhi::LifecycleAllocationPoint::ResourceControl);
    auto failed = TestResourceSlot::create(33);
    ASSERT_TRUE(failed.isErr());
    EXPECT_EQ(failed.error().code, vernon::RhiErrorCode::ResourceExhausted);
}

TEST(RhiResourceLifecycleCoreTest, FailedPublishRollsBackPreallocatedReservation) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(4);
    auto reservation = reserveResource(owner);
    BackendDeviceContext device;
    NativeResource &native = prepareNativeResource(device, 4, owner);

    InjectedFailurePolicy::failNext(vernon::rhi::LifecycleAllocationPoint::ResourcePublish);
    auto failed = slot.publish(std::move(reservation), &device, teardownResource);
    ASSERT_TRUE(failed.isErr());
    EXPECT_EQ(owner.childCount(), 0u);
    EXPECT_FALSE(slot.snapshot().occupied);
    EXPECT_TRUE(native.nativeAlive);
}

TEST(RhiResourceLifecycleCoreTest, PublicDestroyTeardownFailureRollsBackAliveState) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(5);
    BackendDeviceContext device;
    NativeResource &native = prepareNativeResource(device, 5, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    native.failTeardown = true;

    auto failed = slot.destroyPublic(published.value());
    ASSERT_TRUE(failed.isErr());
    auto rolledBack = slot.snapshot();
    EXPECT_TRUE(rolledBack.occupied);
    EXPECT_TRUE(rolledBack.publicAlive);
    EXPECT_EQ(rolledBack.generation, published.value().generation);
    EXPECT_EQ(owner.childCount(), 1u);

    EXPECT_TRUE(slot.destroyPublic(published.value()).isOk());
    EXPECT_TRUE(native.childHeldDuringTeardown);
    EXPECT_FALSE(slot.snapshot().occupied);
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(RhiResourceLifecycleCoreTest, ActivePinRejectsPreparedPublicDestroyWithoutMutation) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(42);
    BackendDeviceContext device;
    prepareNativeResource(device, 42, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto pin = slot.pin(published.value());
    ASSERT_TRUE(pin.isOk());

    auto prepared = slot.prepareDestroyPublic(published.value());
    ASSERT_TRUE(prepared.isErr());
    const auto snapshot = slot.snapshot();
    EXPECT_TRUE(snapshot.occupied);
    EXPECT_TRUE(snapshot.publicAlive);
    EXPECT_EQ(owner.childCount(), 1u);

    ASSERT_TRUE(pin.value().release().isOk());
    EXPECT_TRUE(slot.destroyPublic(published.value()).isOk());
}

TEST(RhiResourceLifecycleCoreTest, PreparedPublicDestroyRejectsPinsAndRollbackRestoresAdmission) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(43);
    BackendDeviceContext device;
    prepareNativeResource(device, 43, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto prepared = slot.prepareDestroyPublic(published.value());
    ASSERT_TRUE(prepared.isOk());

    auto rejected = slot.pin(published.value());
    EXPECT_TRUE(rejected.isErr());
    EXPECT_TRUE(prepared.value().rollback().isOk());
    auto restored = slot.pin(published.value());
    ASSERT_TRUE(restored.isOk());
    ASSERT_TRUE(restored.value().release().isOk());
    EXPECT_TRUE(slot.destroyPublic(published.value()).isOk());
}

TEST(RhiResourceLifecycleCoreTest, FinalRetainedReleaseTeardownFailureIsRetryable) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(6);
    BackendDeviceContext device;
    NativeResource &native = prepareNativeResource(device, 6, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());
    EXPECT_EQ(native.teardownCalls, 0u);
    native.failTeardown = true;

    auto failed = retained.value().release();
    ASSERT_TRUE(failed.isErr());
    EXPECT_TRUE(retained.value().active());
    EXPECT_EQ(slot.snapshot().retainedLeaseCount, 1u);
    EXPECT_TRUE(slot.snapshot().occupied);
    EXPECT_EQ(owner.childCount(), 1u);

    EXPECT_TRUE(retained.value().release().isOk());
    EXPECT_FALSE(retained.value().active());
    EXPECT_FALSE(slot.snapshot().occupied);
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(RhiResourceLifecycleCoreTest, RetainedLeasePinsAfterPublicDestroyWhilePublicPinIsStale) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(35);
    BackendDeviceContext device;
    prepareNativeResource(device, 35, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());

    auto retainedPin = retained.value().pin();
    ASSERT_TRUE(retainedPin.isOk());
    auto publicPin = slot.pin(published.value());
    ASSERT_TRUE(publicPin.isErr());
    ASSERT_TRUE(retainedPin.value().release().isOk());
    EXPECT_TRUE(retained.value().release().isOk());
}

TEST(RhiResourceLifecycleCoreTest, TypeErasedRetainedLeasePinsAndMatchesTypedHandle) {
    using TypedSlot = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::BufferResourceTag, InjectedFailurePolicy>;
    vernon::OwnerRef owner = createDeviceOwner();
    auto created = TypedSlot::create(40);
    ASSERT_TRUE(created.isOk());
    TypedSlot slot = std::move(created).value();
    bool nativeAlive = true;
    auto published = slot.publish(reserveResource(owner), &nativeAlive, teardownTypedBuffer);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    vernon::rhi::RetainedRhiResourceLease erased{std::move(retained).value()};
    EXPECT_TRUE(erased.holds(published.value()));
    EXPECT_EQ(erased.variantIndex(), 0u);
    const uint64_t key = vernon::rhi::encodeResourceKey(published.value());
    ASSERT_TRUE(erased.bindOwner(3, 7, 1, key));
    EXPECT_TRUE(erased.authorizes(3, 7, 1, key));
    EXPECT_FALSE(erased.authorizes(4, 7, 1, key));
    EXPECT_FALSE(erased.authorizes(3, 8, 1, key));
    EXPECT_FALSE(erased.authorizes(3, 7, 2, key));
    EXPECT_FALSE(erased.authorizes(3, 7, 1, key + 1));
    EXPECT_FALSE(erased.bindOwner(3, 7, 1, key));
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());

    auto pin = erased.pin();
    ASSERT_TRUE(pin.isOk());
    ASSERT_TRUE(pin.value().release().isOk());
    ASSERT_TRUE(erased.release().isOk());
    EXPECT_FALSE(nativeAlive);
}

TEST(RhiResourceLifecycleCoreTest, ActiveRetainedPinRejectsFinalReleasePrepareWithoutMutation) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(36);
    BackendDeviceContext device;
    prepareNativeResource(device, 36, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());
    auto pin = retained.value().pin();
    ASSERT_TRUE(pin.isOk());

    auto rejected = retained.value().prepareRelease();
    ASSERT_TRUE(rejected.isErr());
    EXPECT_TRUE(retained.value().active());
    EXPECT_EQ(slot.snapshot().retainedLeaseCount, 1u);
    EXPECT_TRUE(slot.snapshot().occupied);

    ASSERT_TRUE(pin.value().release().isOk());
    EXPECT_TRUE(retained.value().release().isOk());
}

TEST(RhiResourceLifecycleCoreTest, PreparedFinalReleaseRejectsPinsAndRollbackRestoresAdmission) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(37);
    BackendDeviceContext device;
    prepareNativeResource(device, 37, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());

    auto prepared = retained.value().prepareRelease();
    ASSERT_TRUE(prepared.isOk());
    EXPECT_TRUE(retained.value().pin().isErr());
    ASSERT_TRUE(prepared.value().rollback().isOk());
    auto restoredPin = retained.value().pin();
    ASSERT_TRUE(restoredPin.isOk());
    ASSERT_TRUE(restoredPin.value().release().isOk());
    EXPECT_TRUE(retained.value().release().isOk());
}

TEST(RhiResourceLifecycleCoreTest, FailedReleaseCommitRollsBackAndCanBePreparedAgain) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(38);
    BackendDeviceContext device;
    NativeResource &native = prepareNativeResource(device, 38, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());
    native.failTeardown = true;

    {
        auto prepared = retained.value().prepareRelease();
        ASSERT_TRUE(prepared.isOk());
        auto failed = prepared.value().commit();
        ASSERT_TRUE(failed.isErr());
        EXPECT_TRUE(prepared.value().active());
        EXPECT_TRUE(retained.value().active());
    }

    EXPECT_EQ(slot.snapshot().retainedLeaseCount, 1u);
    EXPECT_TRUE(slot.snapshot().occupied);
    auto restoredPin = retained.value().pin();
    ASSERT_TRUE(restoredPin.isOk());
    ASSERT_TRUE(restoredPin.value().release().isOk());
    auto retry = retained.value().prepareRelease();
    ASSERT_TRUE(retry.isOk());
    EXPECT_TRUE(retry.value().commit().isOk());
    EXPECT_FALSE(retained.value().active());
}

TEST(RhiResourceLifecycleCoreTest, ParentReleasePreparePrecedesCascadingNativeTeardown) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(39);
    BackendDeviceContext device;
    NativeResource &native = prepareNativeResource(device, 39, owner);
    native.childNativeAlive = true;
    auto published = slot.publish(reserveResource(owner), &device, teardownCascadingResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    ASSERT_TRUE(slot.destroyPublic(published.value()).isOk());

    auto pin = retained.value().pin();
    ASSERT_TRUE(pin.isOk());
    auto rejected = retained.value().prepareRelease();
    ASSERT_TRUE(rejected.isErr());
    EXPECT_TRUE(native.childNativeAlive);
    EXPECT_EQ(native.teardownCalls, 0u);
    ASSERT_TRUE(pin.value().release().isOk());

    auto prepared = retained.value().prepareRelease();
    ASSERT_TRUE(prepared.isOk());
    EXPECT_TRUE(native.childNativeAlive);
    ASSERT_TRUE(prepared.value().commit().isOk());
    EXPECT_FALSE(native.childNativeAlive);
    EXPECT_FALSE(native.nativeAlive);
}

TEST(RhiResourceLifecycleCoreTest, RetainedLeaseAnchorsControlAcrossContainerChanges) {
    vernon::OwnerRef owner = createDeviceOwner();
    BackendDeviceContext device;
    prepareNativeResource(device, 7, owner);
    std::vector<TestResourceSlot> slots;
    slots.push_back(createResourceSlot(7));
    auto published = slots.front().publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());
    auto retained = slots.front().retain(published.value());
    ASSERT_TRUE(retained.isOk());

    const NativeResource *oldAddress = &device.resources[7];
    while (&device.resources[7] == oldAddress)
        device.resources.push_back(NativeResource{&owner});
    ASSERT_TRUE(slots.front().destroyPublic(published.value()).isOk());

    for (std::uint32_t index = 8; index != 32; ++index)
        slots.push_back(createResourceSlot(index));
    slots.clear();

    EXPECT_TRUE(retained.value().release().isOk());
    EXPECT_FALSE(device.resources[7].nativeAlive);
    EXPECT_EQ(device.resources[7].teardownCalls, 1u);
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(RhiResourceLifecycleCoreTest, ControlRetainSaturationIsFallibleWithoutChangingCount) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(34, UINT32_MAX, 1);
    BackendDeviceContext device;
    prepareNativeResource(device, 34, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());

    auto saturated = slot.retain(published.value());
    ASSERT_TRUE(saturated.isErr());
    EXPECT_EQ(saturated.error().code, vernon::RhiErrorCode::ResourceExhausted);
    EXPECT_EQ(slot.snapshot().retainedLeaseCount, 0u);
    EXPECT_TRUE(slot.destroyPublic(published.value()).isOk());
}

TEST(RhiResourceLifecycleCoreTest, OperationPinAndRetainedCountFailClosed) {
    vernon::OwnerRef owner = createDeviceOwner();
    TestResourceSlot slot = createResourceSlot(32, 1);
    BackendDeviceContext device;
    prepareNativeResource(device, 32, owner);
    auto published = slot.publish(reserveResource(owner), &device, teardownResource);
    ASSERT_TRUE(published.isOk());

    auto pin = slot.pin(published.value());
    ASSERT_TRUE(pin.isOk());
    auto rejected = slot.destroyPublic(published.value());
    ASSERT_TRUE(rejected.isErr());
    EXPECT_TRUE(slot.snapshot().publicAlive);
    ASSERT_TRUE(pin.value().release().isOk());

    auto retained = slot.retain(published.value());
    ASSERT_TRUE(retained.isOk());
    auto saturated = slot.retain(published.value());
    ASSERT_TRUE(saturated.isErr());
    EXPECT_EQ(saturated.error().code, vernon::RhiErrorCode::ResourceExhausted);
    ASSERT_TRUE(retained.value().release().isOk());
    auto duplicate = retained.value().release();
    ASSERT_TRUE(duplicate.isErr());
    EXPECT_EQ(duplicate.error().code, vernon::RhiErrorCode::LifecycleFailure);
    EXPECT_TRUE(slot.destroyPublic(published.value()).isOk());
}

} // namespace
