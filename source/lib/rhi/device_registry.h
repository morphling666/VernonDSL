#ifndef VERNON_RHI_DEVICE_REGISTRY_H
#define VERNON_RHI_DEVICE_REGISTRY_H

#include "VernonError.hpp"
#include "VernonLifecycle.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

namespace vernon::rhi {

enum class LifecycleAllocationPoint : std::uint8_t {
    DeviceEntry,
    RegistrySnapshot,
    ResourceControl,
    ResourcePublish,
};

struct DefaultLifecycleFailurePolicy {
    [[nodiscard]] static constexpr bool fail(LifecycleAllocationPoint) noexcept { return false; }
    static constexpr void snapshotCreated() noexcept {}
    static constexpr void snapshotDestroyed() noexcept {}
    static constexpr void afterSnapshotLoad() noexcept {}
};

struct DeviceRegistryHandle {
    std::uint32_t index{};
    std::uint32_t generation{};
};

namespace detail {

[[nodiscard]] constexpr std::uint32_t nextRhiGeneration(std::uint32_t generation) noexcept {
    ++generation;
    return generation == 0 ? 1 : generation;
}

[[nodiscard]] constexpr RhiError invalidRhiHandle(const char *operation, std::uint64_t value = 0) noexcept {
    return {RhiErrorCode::InvalidArgument, {operation, value, 0}};
}

[[nodiscard]] constexpr RhiError exhaustedRhiOperation(const char *operation, std::uint64_t value = 0) noexcept {
    return {RhiErrorCode::ResourceExhausted, {operation, value, 0}};
}

} // namespace detail

template <typename Device>
class DeviceRegistryEntry final : public CheckedIntrusiveControl<DeviceRegistryEntry<Device>> {
public:
    template <typename... Args>
    DeviceRegistryEntry(std::uint64_t maximumReferences, OwnerRef deviceOwner, OperationRef deviceOperations,
                        Args &&...arguments) noexcept
        : CheckedIntrusiveControl<DeviceRegistryEntry>(maximumReferences), device(std::forward<Args>(arguments)...),
          owner(std::move(deviceOwner)), operations(std::move(deviceOperations)) {}

    Device device;
    OwnerRef owner;
    OperationRef operations;
};

template <typename Device> class [[nodiscard]] DevicePublishReservation {
    using Entry = DeviceRegistryEntry<Device>;

public:
    DevicePublishReservation(const DevicePublishReservation &) = delete;
    DevicePublishReservation &operator=(const DevicePublishReservation &) = delete;

    DevicePublishReservation(DevicePublishReservation &&) noexcept = default;
    DevicePublishReservation &operator=(DevicePublishReservation &&) = delete;

    [[nodiscard]] Device &device() const noexcept { return entry_->device; }
    [[nodiscard]] Result<OwnerRef, RhiError> retainOwner() const noexcept {
        auto retained = entry_->owner.retain();
        if (retained.isErr())
            return Result<OwnerRef, RhiError>{err(toRhiError(std::move(retained).error()))};
        return Result<OwnerRef, RhiError>{ok(std::move(retained).value())};
    }

private:
    DevicePublishReservation(CheckedIntrusiveRef<Entry> entry, const void *registryIdentity) noexcept
        : entry_(std::move(entry)), registryIdentity_(registryIdentity) {}

    CheckedIntrusiveRef<Entry> entry_;
    const void *registryIdentity_{};

    template <typename, std::size_t, typename> friend class DeviceRegistry;
};

template <typename Device> class [[nodiscard]] DeviceRegistryAnchor {
    using Entry = DeviceRegistryEntry<Device>;

public:
    DeviceRegistryAnchor(const DeviceRegistryAnchor &) = delete;
    DeviceRegistryAnchor &operator=(const DeviceRegistryAnchor &) = delete;

    DeviceRegistryAnchor(DeviceRegistryAnchor &&) noexcept = default;
    DeviceRegistryAnchor &operator=(DeviceRegistryAnchor &&) noexcept = default;

    [[nodiscard]] Device &device() const noexcept { return entry_->device; }

    [[nodiscard]] Result<OwnerRef, RhiError> retainOwner() const noexcept {
        auto retained = entry_->owner.retain();
        if (retained.isErr())
            return Result<OwnerRef, RhiError>{err(toRhiError(std::move(retained).error()))};
        return Result<OwnerRef, RhiError>{ok(std::move(retained).value())};
    }

private:
    DeviceRegistryAnchor(CheckedIntrusiveRef<Entry> entry, OperationPin pin) noexcept
        : entry_(std::move(entry)), pin_(std::move(pin)) {}

    CheckedIntrusiveRef<Entry> entry_;
    OperationPin pin_;

    template <typename, std::size_t, typename> friend class DeviceRegistry;
};

template <typename Device, std::size_t Capacity, typename FailurePolicy = DefaultLifecycleFailurePolicy>
class DeviceRegistry {
    static_assert(Capacity > 0, "DeviceRegistry requires non-zero capacity");

    using Entry = DeviceRegistryEntry<Device>;
    using Anchor = DeviceRegistryAnchor<Device>;

    struct SnapshotSlot {
        Option<CheckedIntrusiveRef<Entry>> entry;
        std::uint32_t generation{1};
    };

    struct Snapshot {
        std::array<SnapshotSlot, Capacity> slots{};
        Snapshot *nextRetired{};

        ~Snapshot() noexcept { FailurePolicy::snapshotDestroyed(); }
    };

    class ReaderGuard {
    public:
        explicit ReaderGuard(const DeviceRegistry &registry) noexcept : registry_(&registry) {
            registry_->activeReaders_.fetch_add(1, std::memory_order_seq_cst);
        }
        ReaderGuard(const ReaderGuard &) = delete;
        ReaderGuard &operator=(const ReaderGuard &) = delete;
        ~ReaderGuard() noexcept { registry_->activeReaders_.fetch_sub(1, std::memory_order_seq_cst); }

    private:
        const DeviceRegistry *registry_;
    };

public:
    using Handle = DeviceRegistryHandle;
    using Teardown = Result<void, RhiError> (*)(Device &) noexcept;
    using Reservation = DevicePublishReservation<Device>;

    explicit DeviceRegistry(
        std::uint64_t maximumEntryReferences = CheckedIntrusiveControl<Entry>::maximumReferenceCount) noexcept
        : maximumEntryReferences_(maximumEntryReferences) {
        if (maximumEntryReferences == 0)
            resultContractViolation();
    }

    DeviceRegistry(const DeviceRegistry &) = delete;
    DeviceRegistry &operator=(const DeviceRegistry &) = delete;

    ~DeviceRegistry() noexcept {
        if (activeReaders_.load(std::memory_order_seq_cst) != 0)
            resultContractViolation();
        Snapshot *current = current_.exchange(nullptr, std::memory_order_seq_cst);
        if (current)
            for (const SnapshotSlot &slot : current->slots)
                if (slot.entry)
                    resultContractViolation();
        delete current;
        reclaimRetired();
    }

    template <typename... Args> [[nodiscard]] Result<Reservation, RhiError> reserve(Args &&...arguments) noexcept {
        static_assert(std::is_nothrow_constructible_v<Device, Args &&...>,
                      "DeviceRegistry::reserve requires non-throwing device construction");
        auto owner = OwnerControlBlock::create();
        if (owner.isErr())
            return Result<Reservation, RhiError>{err(toRhiError(std::move(owner).error()))};
        auto operations = OperationControlBlock::create();
        if (operations.isErr())
            return Result<Reservation, RhiError>{err(toRhiError(std::move(operations).error()))};
        if (FailurePolicy::fail(LifecycleAllocationPoint::DeviceEntry))
            return Result<Reservation, RhiError>{err(detail::exhaustedRhiOperation("allocate_device_entry"))};

        Entry *entry = new (std::nothrow) Entry(maximumEntryReferences_, std::move(owner).value(),
                                                std::move(operations).value(), std::forward<Args>(arguments)...);
        if (!entry)
            return Result<Reservation, RhiError>{err(detail::exhaustedRhiOperation("allocate_device_entry"))};
        return Result<Reservation, RhiError>{ok(Reservation{CheckedIntrusiveRef<Entry>::adopt(entry), this})};
    }

    [[nodiscard]] Result<Handle, RhiError> publish(Reservation reservation) noexcept {
        if (reservation.registryIdentity_ != this)
            return Result<Handle, RhiError>{err(detail::invalidRhiHandle("publish_device_registry"))};
        std::lock_guard<std::mutex> guard(writeMutex_);
        Snapshot *current = current_.load(std::memory_order_acquire);
        std::size_t index = 0;
        while (index != Capacity && current && current->slots[index].entry)
            ++index;
        if (index == Capacity)
            return Result<Handle, RhiError>{err(detail::exhaustedRhiOperation("publish_device", Capacity))};

        auto cloned = cloneSnapshot(current, false, 0);
        if (cloned.isErr())
            return Result<Handle, RhiError>{err(std::move(cloned).error())};
        Snapshot *next = cloned.value();
        next->slots[index].entry.emplace(std::move(reservation.entry_));
        const Handle handle{static_cast<std::uint32_t>(index), next->slots[index].generation};
        publishSnapshot(next, current);
        return Result<Handle, RhiError>{ok(handle)};
    }

    // Reader admission is sequenced before loading current_. A writer first
    // exchanges current_, then reclaims only after a seq_cst zero-reader
    // observation. Therefore a pre-exchange reader prevents reclamation, while
    // a post-exchange reader can only load the new immutable snapshot.
    [[nodiscard]] Result<Anchor, RhiError> lookup(Handle handle) const noexcept {
        if (handle.index >= Capacity || handle.generation == 0)
            return Result<Anchor, RhiError>{err(detail::invalidRhiHandle("lookup_device", handle.generation))};
        ReaderGuard reader{*this};
        Snapshot *snapshot = current_.load(std::memory_order_seq_cst);
        FailurePolicy::afterSnapshotLoad();
        if (!snapshot)
            return Result<Anchor, RhiError>{err(detail::invalidRhiHandle("lookup_device", handle.generation))};
        const SnapshotSlot &slot = snapshot->slots[handle.index];
        if (!slot.entry || slot.generation != handle.generation)
            return Result<Anchor, RhiError>{err(detail::invalidRhiHandle("lookup_device", handle.generation))};

        auto retained = slot.entry.value().retain();
        if (retained.isErr())
            return Result<Anchor, RhiError>{err(toRhiError(std::move(retained).error()))};
        auto pin = retained.value()->operations.tryPin();
        if (pin.isErr())
            return Result<Anchor, RhiError>{err(toRhiError(std::move(pin).error()))};
        return Result<Anchor, RhiError>{ok(Anchor{std::move(retained).value(), std::move(pin).value()})};
    }

    // The callback runs while the cold-write mutex is held and must not call
    // this registry. It may acquire the backend device-local native lock.
    [[nodiscard]] Result<void, RhiError> remove(Handle handle, Teardown teardown) noexcept {
        if (handle.index >= Capacity || handle.generation == 0 || !teardown)
            return Result<void, RhiError>{err(detail::invalidRhiHandle("remove_device", handle.generation))};

        std::lock_guard<std::mutex> guard(writeMutex_);
        Snapshot *current = current_.load(std::memory_order_acquire);
        if (!current)
            return Result<void, RhiError>{err(detail::invalidRhiHandle("remove_device", handle.generation))};
        const SnapshotSlot &slot = current->slots[handle.index];
        if (!slot.entry || slot.generation != handle.generation)
            return Result<void, RhiError>{err(detail::invalidRhiHandle("remove_device", handle.generation))};

        auto cloned = cloneSnapshot(current, true, handle.index);
        if (cloned.isErr())
            return Result<void, RhiError>{err(std::move(cloned).error())};
        Snapshot *next = cloned.value();
        Entry &entry = slot.entry.value().value();
        auto close = entry.owner.beginClose();
        if (close.isErr()) {
            delete next;
            return Result<void, RhiError>{err(toRhiError(std::move(close).error()))};
        }
        auto destruction = entry.operations.beginDestroy();
        if (destruction.isErr()) {
            delete next;
            return Result<void, RhiError>{err(toRhiError(std::move(destruction).error()))};
        }
        auto tornDown = teardown(entry.device);
        if (tornDown.isErr()) {
            delete next;
            return tornDown;
        }
        if (destruction.value().commit().isErr() || close.value().commit().isErr())
            resultContractViolation();
        publishSnapshot(next, current);
        return Result<void, RhiError>{ok()};
    }

private:
    [[nodiscard]] static Result<Snapshot *, RhiError> cloneSnapshot(const Snapshot *source, bool omitEntry,
                                                                    std::size_t omittedIndex) noexcept {
        if (FailurePolicy::fail(LifecycleAllocationPoint::RegistrySnapshot))
            return Result<Snapshot *, RhiError>{err(detail::exhaustedRhiOperation("allocate_registry_snapshot"))};
        Snapshot *clone = new (std::nothrow) Snapshot();
        if (!clone)
            return Result<Snapshot *, RhiError>{err(detail::exhaustedRhiOperation("allocate_registry_snapshot"))};
        FailurePolicy::snapshotCreated();
        if (!source)
            return Result<Snapshot *, RhiError>{ok(clone)};

        for (std::size_t index = 0; index != Capacity; ++index) {
            clone->slots[index].generation = source->slots[index].generation;
            if (omitEntry && index == omittedIndex) {
                clone->slots[index].generation = detail::nextRhiGeneration(source->slots[index].generation);
                continue;
            }
            if (!source->slots[index].entry)
                continue;
            auto retained = source->slots[index].entry.value().retain();
            if (retained.isErr()) {
                delete clone;
                return Result<Snapshot *, RhiError>{err(toRhiError(std::move(retained).error()))};
            }
            clone->slots[index].entry.emplace(std::move(retained).value());
        }
        return Result<Snapshot *, RhiError>{ok(clone)};
    }

    void publishSnapshot(Snapshot *next, Snapshot *expectedCurrent) noexcept {
        Snapshot *old = current_.exchange(next, std::memory_order_seq_cst);
        if (old != expectedCurrent)
            resultContractViolation();
        if (old) {
            old->nextRetired = retired_;
            retired_ = old;
        }
        reclaimRetired();
    }

    void reclaimRetired() noexcept {
        if (activeReaders_.load(std::memory_order_seq_cst) != 0)
            return;
        Snapshot *snapshot = std::exchange(retired_, nullptr);
        while (snapshot) {
            Snapshot *next = snapshot->nextRetired;
            delete snapshot;
            snapshot = next;
        }
    }

    mutable std::mutex writeMutex_;
    mutable std::atomic<Snapshot *> current_{};
    mutable std::atomic<std::uint64_t> activeReaders_{};
    Snapshot *retired_{};
    const std::uint64_t maximumEntryReferences_;
};

} // namespace vernon::rhi

#endif
