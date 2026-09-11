#include "VernonLifecycle.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace {

vernon::OwnerRef createOwner(std::uint64_t maximumChildren = 16, std::uint64_t maximumReferences = 64) {
    auto created = vernon::OwnerControlBlock::create(maximumChildren, maximumReferences);
    EXPECT_TRUE(created.isOk());
    return std::move(created).value();
}

vernon::OperationRef createOperations(std::uint64_t maximumPins = 16, std::uint64_t maximumReferences = 64) {
    auto created = vernon::OperationControlBlock::create(maximumPins, maximumReferences);
    EXPECT_TRUE(created.isOk());
    return std::move(created).value();
}

vernon::ChildLease createChildThatOutlivesCreator() {
    vernon::OwnerRef owner = createOwner();
    auto reservation = owner.reserveChild();
    EXPECT_TRUE(reservation.isOk());
    auto lease = std::move(reservation).value().commit();
    EXPECT_TRUE(lease.isOk());
    return std::move(lease).value();
}

class Gate {
public:
    void open() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            open_ = true;
        }
        condition_.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return open_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool open_{};
};

struct LifecycleHookBarrier {
    vernon::detail::LifecycleTestEvent event;
    std::string_view operation;
    Gate reached;
    Gate resume;
    std::atomic<bool> consumed{false};
};

std::atomic<LifecycleHookBarrier *> activeLifecycleHookBarrier{};

void lifecycleTestHook(vernon::detail::LifecycleTestEvent event, const char *operation) noexcept {
    LifecycleHookBarrier *barrier = activeLifecycleHookBarrier.load(std::memory_order_acquire);
    if (!barrier || barrier->event != event || barrier->operation != operation ||
        barrier->consumed.exchange(true, std::memory_order_acq_rel))
        return;
    barrier->reached.open();
    barrier->resume.wait();
}

class ScopedLifecycleHook {
public:
    explicit ScopedLifecycleHook(LifecycleHookBarrier &barrier) noexcept {
        activeLifecycleHookBarrier.store(&barrier, std::memory_order_release);
        vernon::detail::setLifecycleTestHook(lifecycleTestHook);
    }
    ScopedLifecycleHook(const ScopedLifecycleHook &) = delete;
    ScopedLifecycleHook &operator=(const ScopedLifecycleHook &) = delete;
    ~ScopedLifecycleHook() noexcept {
        vernon::detail::setLifecycleTestHook(nullptr);
        activeLifecycleHookBarrier.store(nullptr, std::memory_order_release);
    }
};

TEST(LifecycleControlTest, ReservationRollbackAndCommittedLeaseHaveOneOwnerCount) {
    vernon::OwnerRef owner = createOwner();
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(owner.childCount(), 0u);

    {
        auto reservation = owner.reserveChild();
        ASSERT_TRUE(reservation.isOk());
        EXPECT_EQ(owner.childCount(), 1u);
    }
    EXPECT_EQ(owner.childCount(), 0u);

    auto reservation = owner.reserveChild();
    ASSERT_TRUE(reservation.isOk());
    auto lease = std::move(reservation).value().commit();
    ASSERT_TRUE(lease.isOk());
    EXPECT_EQ(owner.childCount(), 1u);
    EXPECT_TRUE(lease.value().release());
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(LifecycleControlTest, CloseFailureAndTeardownFailureRestoreOpen) {
    vernon::OwnerRef owner = createOwner();
    auto reservation = owner.reserveChild();
    ASSERT_TRUE(reservation.isOk());
    auto lease = std::move(reservation).value().commit();
    ASSERT_TRUE(lease.isOk());

    auto rejected = owner.beginClose();
    ASSERT_TRUE(rejected.isErr());
    EXPECT_EQ(rejected.error().code, vernon::LifecycleErrorCode::LiveChildren);
    EXPECT_EQ(rejected.error().context.value, 1u);
    EXPECT_EQ(rejected.error().context.detail, static_cast<std::uint32_t>(vernon::LifecycleState::Closing));
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);

    ASSERT_TRUE(lease.value().release());
    EXPECT_EQ(rejected.error().context.value, 1u);

    {
        auto close = owner.beginClose();
        ASSERT_TRUE(close.isOk());
        EXPECT_EQ(owner.state(), vernon::LifecycleState::Closing);
        EXPECT_EQ(close.value().snapshot().state, vernon::LifecycleState::Closing);
        EXPECT_EQ(close.value().snapshot().leaseCount, 0u);
    }
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);

    auto failedTeardown = owner.beginClose();
    ASSERT_TRUE(failedTeardown.isOk());
    EXPECT_TRUE(failedTeardown.value().rollback());
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);
}

TEST(LifecycleControlTest, ClosedOwnerCannotReopenOrAdmitChildren) {
    vernon::OwnerRef owner = createOwner();
    auto close = owner.beginClose();
    ASSERT_TRUE(close.isOk());
    ASSERT_TRUE(close.value().commit());
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Closed);

    auto reservation = owner.reserveChild();
    ASSERT_TRUE(reservation.isErr());
    EXPECT_EQ(reservation.error().code, vernon::LifecycleErrorCode::AlreadyClosed);
    auto repeated = owner.beginClose();
    ASSERT_TRUE(repeated.isErr());
    EXPECT_EQ(repeated.error().code, vernon::LifecycleErrorCode::AlreadyClosed);
}

TEST(LifecycleControlTest, CountersSaturateWithoutWrapping) {
    vernon::OwnerRef owner = createOwner(1);
    auto first = owner.reserveChild();
    ASSERT_TRUE(first.isOk());
    auto saturated = owner.reserveChild();
    ASSERT_TRUE(saturated.isErr());
    EXPECT_EQ(saturated.error().code, vernon::LifecycleErrorCode::AdmissionSaturated);
    EXPECT_EQ(owner.childCount(), 1u);
    EXPECT_TRUE(first.value().rollback());
    EXPECT_EQ(owner.childCount(), 0u);

    vernon::OwnerRef referenceLimited = createOwner(1, 1);
    auto retained = referenceLimited.retain();
    ASSERT_TRUE(retained.isErr());
    EXPECT_EQ(retained.error().code, vernon::LifecycleErrorCode::AdmissionSaturated);

    vernon::OperationRef operationLimited = createOperations(1);
    auto pin = operationLimited.tryPin();
    ASSERT_TRUE(pin.isOk());
    auto saturatedPin = operationLimited.tryPin();
    ASSERT_TRUE(saturatedPin.isErr());
    EXPECT_EQ(saturatedPin.error().code, vernon::LifecycleErrorCode::PinRejected);
    EXPECT_EQ(operationLimited.pinCount(), 1u);

    vernon::OperationRef operationReferenceLimited = createOperations(1, 1);
    auto retainedOperation = operationReferenceLimited.retain();
    ASSERT_TRUE(retainedOperation.isErr());
    EXPECT_EQ(retainedOperation.error().code, vernon::LifecycleErrorCode::AdmissionSaturated);
    auto pinWithoutReferenceCapacity = operationReferenceLimited.tryPin();
    ASSERT_TRUE(pinWithoutReferenceCapacity.isErr());
    EXPECT_EQ(pinWithoutReferenceCapacity.error().code, vernon::LifecycleErrorCode::AdmissionSaturated);
    auto destructionWithoutReferenceCapacity = operationReferenceLimited.beginDestroy();
    ASSERT_TRUE(destructionWithoutReferenceCapacity.isErr());
    EXPECT_EQ(destructionWithoutReferenceCapacity.error().code, vernon::LifecycleErrorCode::AdmissionSaturated);
    EXPECT_EQ(operationReferenceLimited.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(operationReferenceLimited.pinCount(), 0u);
}

TEST(LifecycleControlTest, InvalidCounterConfigurationIsRejectedWithoutNormalization) {
    auto noReferences = vernon::OwnerControlBlock::create(1, 0);
    ASSERT_TRUE(noReferences.isErr());
    EXPECT_EQ(noReferences.error().code, vernon::LifecycleErrorCode::InvalidConfiguration);

    auto oversizedChildren = vernon::OwnerControlBlock::create(vernon::OwnerControlBlock::maximumChildCount + 1, 1);
    ASSERT_TRUE(oversizedChildren.isErr());
    EXPECT_EQ(oversizedChildren.error().code, vernon::LifecycleErrorCode::InvalidConfiguration);

    auto noOperationReferences = vernon::OperationControlBlock::create(1, 0);
    ASSERT_TRUE(noOperationReferences.isErr());
    EXPECT_EQ(noOperationReferences.error().code, vernon::LifecycleErrorCode::InvalidConfiguration);
    auto oversizedPins = vernon::OperationControlBlock::create(vernon::OperationControlBlock::maximumPinCount + 1, 1);
    ASSERT_TRUE(oversizedPins.isErr());
    EXPECT_EQ(oversizedPins.error().code, vernon::LifecycleErrorCode::InvalidConfiguration);
}

TEST(LifecycleControlTest, StaleReservationLeaseAndPinReleasesFailClosed) {
    vernon::OwnerRef owner = createOwner();
    auto reservation = owner.reserveChild();
    ASSERT_TRUE(reservation.isOk());
    ASSERT_TRUE(reservation.value().rollback());
    auto repeatedRollback = reservation.value().rollback();
    ASSERT_TRUE(repeatedRollback.isErr());
    EXPECT_EQ(repeatedRollback.error().code, vernon::LifecycleErrorCode::StaleAdmission);

    auto committedReservation = owner.reserveChild();
    ASSERT_TRUE(committedReservation.isOk());
    auto lease = committedReservation.value().commit();
    ASSERT_TRUE(lease.isOk());
    ASSERT_TRUE(lease.value().release());
    auto repeatedRelease = lease.value().release();
    ASSERT_TRUE(repeatedRelease.isErr());
    EXPECT_EQ(repeatedRelease.error().code, vernon::LifecycleErrorCode::StaleAdmission);

    vernon::OperationRef operations = createOperations();
    auto pin = operations.tryPin();
    ASSERT_TRUE(pin.isOk());
    ASSERT_TRUE(pin.value().release());
    auto repeatedPinRelease = pin.value().release();
    ASSERT_TRUE(repeatedPinRelease.isErr());
    EXPECT_EQ(repeatedPinRelease.error().code, vernon::LifecycleErrorCode::StaleAdmission);
}

TEST(LifecycleControlTest, ChildLeaseRetainsParentControlBlock) {
    vernon::ChildLease lease = createChildThatOutlivesCreator();
    auto owner = lease.retainOwner();
    ASSERT_TRUE(owner.isOk());
    EXPECT_EQ(owner.value().childCount(), 1u);
    ASSERT_TRUE(lease.release());

    auto close = owner.value().beginClose();
    ASSERT_TRUE(close.isOk());
    EXPECT_TRUE(close.value().commit());
}

TEST(LifecycleControlTest, CompletedGuardsReleaseReferencesImmediately) {
    vernon::OwnerRef owner = createOwner(1, 2);
    std::vector<vernon::ChildReservation> reservations;
    std::vector<vernon::ChildLease> leases;
    std::vector<vernon::CloseAttempt> closes;
    reservations.reserve(16);
    leases.reserve(16);
    closes.reserve(16);

    for (std::size_t iteration = 0; iteration < 16; ++iteration) {
        auto reservation = owner.reserveChild();
        ASSERT_TRUE(reservation.isOk());
        ASSERT_TRUE(reservation.value().rollback());
        reservations.push_back(std::move(reservation).value());

        auto committed = owner.reserveChild();
        ASSERT_TRUE(committed.isOk());
        auto lease = committed.value().commit();
        ASSERT_TRUE(lease.isOk());
        ASSERT_TRUE(lease.value().release());
        leases.push_back(std::move(lease).value());

        auto close = owner.beginClose();
        ASSERT_TRUE(close.isOk());
        ASSERT_TRUE(close.value().rollback());
        closes.push_back(std::move(close).value());
    }
}

TEST(LifecycleControlTest, PinsAndDestructionAttemptsAnchorOperationControl) {
    vernon::OperationRef operations = createOperations();
    auto pin = operations.tryPin();
    ASSERT_TRUE(pin.isOk());
    operations.reset();
    EXPECT_TRUE(pin.value().release());

    operations = createOperations();
    auto destruction = operations.beginDestroy();
    ASSERT_TRUE(destruction.isOk());
    operations.reset();
    EXPECT_TRUE(destruction.value().rollback());
}

TEST(LifecycleControlTest, CompletedOperationGuardsReleaseReferencesImmediately) {
    vernon::OperationRef operations = createOperations(1, 2);
    std::vector<vernon::OperationPin> pins;
    std::vector<vernon::DestructionAttempt> destructionAttempts;
    pins.reserve(16);
    destructionAttempts.reserve(16);

    for (std::size_t iteration = 0; iteration < 16; ++iteration) {
        auto pin = operations.tryPin();
        ASSERT_TRUE(pin.isOk());
        ASSERT_TRUE(pin.value().release());
        pins.push_back(std::move(pin).value());

        auto destruction = operations.beginDestroy();
        ASSERT_TRUE(destruction.isOk());
        ASSERT_TRUE(destruction.value().rollback());
        destructionAttempts.push_back(std::move(destruction).value());
    }
}

TEST(LifecycleControlTest, OperationPinsRejectConcurrentDestruction) {
    vernon::OperationRef operations = createOperations();
    auto pin = operations.tryPin();
    ASSERT_TRUE(pin.isOk());
    EXPECT_EQ(operations.pinCount(), 1u);

    auto rejected = operations.beginDestroy();
    ASSERT_TRUE(rejected.isErr());
    EXPECT_EQ(rejected.error().code, vernon::LifecycleErrorCode::PinRejected);
    EXPECT_EQ(rejected.error().context.value, 1u);
    EXPECT_EQ(operations.state(), vernon::LifecycleState::Open);

    ASSERT_TRUE(pin.value().release());
    EXPECT_EQ(rejected.error().context.value, 1u);

    {
        auto destruction = operations.beginDestroy();
        ASSERT_TRUE(destruction.isOk());
        EXPECT_EQ(operations.state(), vernon::LifecycleState::Closing);
        auto stalePin = operations.tryPin();
        ASSERT_TRUE(stalePin.isErr());
        EXPECT_EQ(stalePin.error().code, vernon::LifecycleErrorCode::NotOpen);
    }
    EXPECT_EQ(operations.state(), vernon::LifecycleState::Open);

    auto destruction = operations.beginDestroy();
    ASSERT_TRUE(destruction.isOk());
    ASSERT_TRUE(destruction.value().commit());
    EXPECT_EQ(operations.state(), vernon::LifecycleState::Closed);
    auto closedPin = operations.tryPin();
    ASSERT_TRUE(closedPin.isErr());
    EXPECT_EQ(closedPin.error().code, vernon::LifecycleErrorCode::AlreadyClosed);
}

TEST(LifecycleControlTest, AdmissionWinsBeforeCloseDeterministically) {
    vernon::OwnerRef owner = createOwner();
    LifecycleHookBarrier barrier{vernon::detail::LifecycleTestEvent::AfterAcquireLinearized, "reserve_child"};
    ScopedLifecycleHook hook{barrier};
    std::atomic<std::size_t> nativeMutations{0};
    bool admissionSucceeded = false;
    bool rollbackSucceeded = false;

    std::thread admission([&] {
        auto reservation = owner.reserveChild();
        admissionSucceeded = reservation.isOk();
        if (admissionSucceeded) {
            nativeMutations.fetch_add(1, std::memory_order_relaxed);
            rollbackSucceeded = reservation.value().rollback().isOk();
        }
    });

    barrier.reached.wait();
    EXPECT_EQ(nativeMutations.load(std::memory_order_relaxed), 0u);
    auto close = owner.beginClose();
    EXPECT_EQ(nativeMutations.load(std::memory_order_relaxed), 0u);
    barrier.resume.open();
    admission.join();

    ASSERT_TRUE(admissionSucceeded);
    ASSERT_TRUE(rollbackSucceeded);
    ASSERT_TRUE(close.isErr());
    EXPECT_EQ(close.error().code, vernon::LifecycleErrorCode::LiveChildren);
    EXPECT_EQ(nativeMutations.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(LifecycleControlTest, ClosingWinsBeforeAdmissionDeterministically) {
    vernon::OwnerRef owner = createOwner();
    LifecycleHookBarrier barrier{vernon::detail::LifecycleTestEvent::AfterClosingLinearized, "begin_owner_close"};
    ScopedLifecycleHook hook{barrier};
    std::atomic<std::size_t> nativeMutations{0};
    bool closeSucceeded = false;
    bool rollbackSucceeded = false;

    std::thread closing([&] {
        auto close = owner.beginClose();
        closeSucceeded = close.isOk();
        if (closeSucceeded)
            rollbackSucceeded = close.value().rollback().isOk();
    });

    barrier.reached.wait();
    auto reservation = owner.reserveChild();
    const bool admissionRejected = reservation.isErr();
    const vernon::LifecycleErrorCode admissionError =
        admissionRejected ? reservation.error().code : vernon::LifecycleErrorCode::TeardownFailed;
    if (reservation.isOk()) {
        nativeMutations.fetch_add(1, std::memory_order_relaxed);
        (void)reservation.value().rollback();
    }
    barrier.resume.open();
    closing.join();

    EXPECT_TRUE(closeSucceeded);
    EXPECT_TRUE(rollbackSucceeded);
    EXPECT_TRUE(admissionRejected);
    EXPECT_EQ(admissionError, vernon::LifecycleErrorCode::NotOpen);
    EXPECT_EQ(nativeMutations.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(owner.childCount(), 0u);
}

TEST(LifecycleControlTest, PinWinsBeforeDestroyDeterministically) {
    vernon::OperationRef operations = createOperations();
    LifecycleHookBarrier barrier{vernon::detail::LifecycleTestEvent::AfterAcquireLinearized, "pin_child_operation"};
    ScopedLifecycleHook hook{barrier};
    std::atomic<std::size_t> nativeOperations{0};
    bool pinSucceeded = false;
    bool releaseSucceeded = false;

    std::thread pinning([&] {
        auto pin = operations.tryPin();
        pinSucceeded = pin.isOk();
        if (pinSucceeded) {
            nativeOperations.fetch_add(1, std::memory_order_relaxed);
            releaseSucceeded = pin.value().release().isOk();
        }
    });

    barrier.reached.wait();
    EXPECT_EQ(nativeOperations.load(std::memory_order_relaxed), 0u);
    auto destruction = operations.beginDestroy();
    EXPECT_EQ(nativeOperations.load(std::memory_order_relaxed), 0u);
    barrier.resume.open();
    pinning.join();

    EXPECT_TRUE(pinSucceeded);
    EXPECT_TRUE(releaseSucceeded);
    ASSERT_TRUE(destruction.isErr());
    EXPECT_EQ(destruction.error().code, vernon::LifecycleErrorCode::PinRejected);
    EXPECT_EQ(nativeOperations.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(operations.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(operations.pinCount(), 0u);
}

TEST(LifecycleControlTest, DestroyWinsBeforePinDeterministically) {
    vernon::OperationRef operations = createOperations();
    LifecycleHookBarrier barrier{vernon::detail::LifecycleTestEvent::AfterClosingLinearized, "begin_child_destroy"};
    ScopedLifecycleHook hook{barrier};
    std::atomic<std::size_t> nativeOperations{0};
    bool destructionSucceeded = false;
    bool rollbackSucceeded = false;

    std::thread destroying([&] {
        auto destruction = operations.beginDestroy();
        destructionSucceeded = destruction.isOk();
        if (destructionSucceeded)
            rollbackSucceeded = destruction.value().rollback().isOk();
    });

    barrier.reached.wait();
    auto pin = operations.tryPin();
    const bool pinRejected = pin.isErr();
    const vernon::LifecycleErrorCode pinError =
        pinRejected ? pin.error().code : vernon::LifecycleErrorCode::TeardownFailed;
    if (pin.isOk()) {
        nativeOperations.fetch_add(1, std::memory_order_relaxed);
        (void)pin.value().release();
    }
    barrier.resume.open();
    destroying.join();

    EXPECT_TRUE(destructionSucceeded);
    EXPECT_TRUE(rollbackSucceeded);
    EXPECT_TRUE(pinRejected);
    EXPECT_EQ(pinError, vernon::LifecycleErrorCode::NotOpen);
    EXPECT_EQ(nativeOperations.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(operations.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(operations.pinCount(), 0u);
}

TEST(LifecycleControlTest, AdmissionCloseAndPinDestroyRemainConsistentUnderStress) {
    constexpr std::size_t workerCount = 4;
    constexpr std::size_t iterationCount = 5000;
    vernon::OwnerRef owner = createOwner(64, 64);
    auto anchorReservation = owner.reserveChild();
    ASSERT_TRUE(anchorReservation.isOk());
    auto anchor = anchorReservation.value().commit();
    ASSERT_TRUE(anchor.isOk());

    vernon::OperationRef operations = createOperations(64, 64);
    std::atomic<bool> start{false};
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(workerCount * 2 + 2);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
            }
            for (std::size_t iteration = 0; iteration < iterationCount; ++iteration) {
                auto reservation = owner.reserveChild();
                if (reservation.isOk()) {
                    if (reservation.value().rollback().isErr())
                        failures.fetch_add(1, std::memory_order_relaxed);
                } else if (reservation.error().code != vernon::LifecycleErrorCode::NotOpen) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
        workers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
            }
            for (std::size_t iteration = 0; iteration < iterationCount; ++iteration) {
                auto pin = operations.tryPin();
                if (pin.isOk()) {
                    if (pin.value().release().isErr())
                        failures.fetch_add(1, std::memory_order_relaxed);
                } else if (pin.error().code != vernon::LifecycleErrorCode::NotOpen) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    workers.emplace_back([&] {
        while (!start.load(std::memory_order_acquire)) {
        }
        for (std::size_t iteration = 0; iteration < iterationCount; ++iteration) {
            auto close = owner.beginClose();
            if (close.isOk() || close.error().code != vernon::LifecycleErrorCode::LiveChildren)
                failures.fetch_add(1, std::memory_order_relaxed);
        }
    });
    workers.emplace_back([&] {
        while (!start.load(std::memory_order_acquire)) {
        }
        for (std::size_t iteration = 0; iteration < iterationCount; ++iteration) {
            auto destruction = operations.beginDestroy();
            if (destruction.isOk()) {
                if (destruction.value().rollback().isErr())
                    failures.fetch_add(1, std::memory_order_relaxed);
            } else if (destruction.error().code != vernon::LifecycleErrorCode::PinRejected &&
                       destruction.error().code != vernon::LifecycleErrorCode::NotOpen) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers)
        worker.join();

    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(owner.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(owner.childCount(), 1u);
    EXPECT_EQ(operations.state(), vernon::LifecycleState::Open);
    EXPECT_EQ(operations.pinCount(), 0u);
    EXPECT_TRUE(anchor.value().release());
    EXPECT_EQ(owner.childCount(), 0u);
}

} // namespace
