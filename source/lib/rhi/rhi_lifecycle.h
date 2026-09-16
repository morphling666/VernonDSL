#ifndef VERNON_RHI_LIFECYCLE_H
#define VERNON_RHI_LIFECYCLE_H

#include "device_registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>
#include <variant>

namespace vernon::rhi {

template <typename Handle> constexpr uint64_t encodeResourceKey(Handle handle) noexcept {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

// Append-only storage for device-local resource slots. A device's existing
// writer lock serializes emplace(), while readers use acquire-published page
// links and size without taking that lock. Pages and their constructed slots
// never move, so pointers returned by get()/emplace() remain stable.
template <typename T, std::size_t SlotsPerPage = 16> class StableResourceSlotContainer {
    static_assert(SlotsPerPage != 0);

    struct Page {
        Page() noexcept : next(nullptr) {}

        [[nodiscard]] void *address(std::size_t index) noexcept { return storage + index * sizeof(T); }

        [[nodiscard]] T *slot(std::size_t index) noexcept {
            return std::launder(reinterpret_cast<T *>(address(index)));
        }

        [[nodiscard]] const T *slot(std::size_t index) const noexcept {
            return std::launder(reinterpret_cast<const T *>(storage + index * sizeof(T)));
        }

        std::atomic<Page *> next;
        alignas(T) std::byte storage[sizeof(T) * SlotsPerPage];
    };

public:
    using value_type = T;

    StableResourceSlotContainer() noexcept = default;
    StableResourceSlotContainer(const StableResourceSlotContainer &) = delete;
    StableResourceSlotContainer &operator=(const StableResourceSlotContainer &) = delete;
    StableResourceSlotContainer(StableResourceSlotContainer &&) = delete;
    StableResourceSlotContainer &operator=(StableResourceSlotContainer &&) = delete;

    ~StableResourceSlotContainer() noexcept {
        std::size_t remaining = size_.load(std::memory_order_relaxed);
        Page *page = head_.load(std::memory_order_relaxed);
        while (page) {
            const std::size_t constructed = remaining < SlotsPerPage ? remaining : SlotsPerPage;
            for (std::size_t index = 0; index < constructed; ++index)
                page->slot(index)->~T();
            remaining -= constructed;
            Page *next = page->next.load(std::memory_order_relaxed);
            delete page;
            page = next;
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_.load(std::memory_order_acquire); }

    [[nodiscard]] T *get(std::size_t index) noexcept {
        if (index >= size_.load(std::memory_order_acquire))
            return nullptr;
        return locate(index);
    }

    [[nodiscard]] const T *get(std::size_t index) const noexcept {
        if (index >= size_.load(std::memory_order_acquire))
            return nullptr;
        return locate(index);
    }

    [[nodiscard]] T &operator[](std::size_t index) noexcept { return *get(index); }
    [[nodiscard]] const T &operator[](std::size_t index) const noexcept { return *get(index); }

    template <typename... Args>
    [[nodiscard]] Result<T *, RhiError> emplace(const char *operation, Args &&...args) noexcept {
        static_assert(std::is_nothrow_constructible_v<T, Args...>,
                      "stable resource slots must be nothrow constructible");

        const std::size_t index = size_.load(std::memory_order_relaxed);
        if (index >= (std::numeric_limits<std::uint32_t>::max)())
            return Result<T *, RhiError>{err(RhiError{RhiErrorCode::ResourceExhausted, {operation, index, 0}})};

        const std::size_t offset = index % SlotsPerPage;
        Page *page = tail_;
        Page *newPage = nullptr;
        if (!page || offset == 0) {
            newPage = new (std::nothrow) Page;
            if (!newPage)
                return Result<T *, RhiError>{err(RhiError{RhiErrorCode::ResourceExhausted, {operation, index, 0}})};
            page = newPage;
        }

        T *value = new (page->address(offset)) T(std::forward<Args>(args)...);
        if (newPage) {
            if (tail_)
                tail_->next.store(newPage, std::memory_order_release);
            else
                head_.store(newPage, std::memory_order_release);
            tail_ = newPage;
        }
        size_.store(index + 1, std::memory_order_release);
        return Result<T *, RhiError>{ok(value)};
    }

private:
    [[nodiscard]] T *locate(std::size_t index) noexcept {
        Page *page = head_.load(std::memory_order_acquire);
        for (std::size_t pageIndex = index / SlotsPerPage; pageIndex != 0; --pageIndex)
            page = page->next.load(std::memory_order_acquire);
        return page->slot(index % SlotsPerPage);
    }

    [[nodiscard]] const T *locate(std::size_t index) const noexcept {
        const Page *page = head_.load(std::memory_order_acquire);
        for (std::size_t pageIndex = index / SlotsPerPage; pageIndex != 0; --pageIndex)
            page = page->next.load(std::memory_order_acquire);
        return page->slot(index % SlotsPerPage);
    }

    std::atomic<Page *> head_{};
    Page *tail_{};
    std::atomic<std::size_t> size_{};
};

struct BufferResourceTag {};
struct ImageResourceTag {};
struct ImageViewResourceTag {};
struct SamplerResourceTag {};
struct NativeDescriptorRangeResourceTag {};
struct CommandEncoderResourceTag {};
struct CompletionResourceTag {};

template <typename ResourceTag> struct ResourceHandle {
    std::uint32_t index{};
    std::uint32_t generation{};
};

class [[nodiscard]] ResourceCreationReservation {
public:
    ResourceCreationReservation(const ResourceCreationReservation &) = delete;
    ResourceCreationReservation &operator=(const ResourceCreationReservation &) = delete;

    ResourceCreationReservation(ResourceCreationReservation &&other) noexcept
        : child_(std::move(other.child_)), operations_(std::move(other.operations_)) {}
    ResourceCreationReservation &operator=(ResourceCreationReservation &&) = delete;

    [[nodiscard]] static Result<ResourceCreationReservation, RhiError> create(const OwnerRef &deviceOwner) noexcept {
        auto child = deviceOwner.reserveChild();
        if (child.isErr())
            return Result<ResourceCreationReservation, RhiError>{err(toRhiError(std::move(child).error()))};
        auto operations = OperationControlBlock::create();
        if (operations.isErr())
            return Result<ResourceCreationReservation, RhiError>{err(toRhiError(std::move(operations).error()))};
        return Result<ResourceCreationReservation, RhiError>{
            ok(ResourceCreationReservation{std::move(child).value(), std::move(operations).value()})};
    }

private:
    ResourceCreationReservation(ChildReservation child, OperationRef operations) noexcept
        : child_(std::move(child)), operations_(std::move(operations)) {}

    [[nodiscard]] Result<ChildLease, RhiError> commitChild() noexcept {
        auto committed = child_.commit();
        if (committed.isErr())
            return Result<ChildLease, RhiError>{err(toRhiError(std::move(committed).error()))};
        return Result<ChildLease, RhiError>{ok(std::move(committed).value())};
    }

    [[nodiscard]] OperationRef takeOperations() noexcept { return std::move(operations_); }

    ChildReservation child_;
    OperationRef operations_;

    template <typename, typename> friend class ResourceLifecycleSlot;
};

template <typename ResourceTag> class ResourceLifecycleControl;
template <typename ResourceTag> class RetainedResourceReleaseAttempt;
template <typename ResourceTag> class PublicResourceDestroyAttempt;

template <typename ResourceTag> class [[nodiscard]] RetainedResourceLease {
public:
    using Handle = ResourceHandle<ResourceTag>;
    using ReleaseAttempt = RetainedResourceReleaseAttempt<ResourceTag>;

    RetainedResourceLease(const RetainedResourceLease &) = delete;
    RetainedResourceLease &operator=(const RetainedResourceLease &) = delete;

    RetainedResourceLease(RetainedResourceLease &&other) noexcept
        : control_(std::move(other.control_)), handle_(other.handle_) {}
    RetainedResourceLease &operator=(RetainedResourceLease &&) = delete;

    ~RetainedResourceLease() noexcept {
        if (control_) {
            auto released = release();
            if (released.isErr())
                resultContractViolation();
        }
    }

    [[nodiscard]] Result<OperationPin, RhiError> pin() noexcept;
    [[nodiscard]] Result<ReleaseAttempt, RhiError> prepareRelease() noexcept;
    [[nodiscard]] Result<void, RhiError> release() noexcept;

    [[nodiscard]] bool active() const noexcept { return static_cast<bool>(control_); }
    [[nodiscard]] Handle handle() const noexcept { return handle_; }

private:
    RetainedResourceLease(CheckedIntrusiveRef<ResourceLifecycleControl<ResourceTag>> control, Handle handle) noexcept
        : control_(std::move(control)), handle_(handle) {}

    CheckedIntrusiveRef<ResourceLifecycleControl<ResourceTag>> control_;
    Handle handle_{};

    template <typename, typename> friend class ResourceLifecycleSlot;
    friend class ResourceLifecycleControl<ResourceTag>;
    friend class RetainedResourceReleaseAttempt<ResourceTag>;
};

class [[nodiscard]] RetainedRhiResourceLease {
public:
    using BufferLease = RetainedResourceLease<BufferResourceTag>;
    using ImageLease = RetainedResourceLease<ImageResourceTag>;
    using ImageViewLease = RetainedResourceLease<ImageViewResourceTag>;
    using SamplerLease = RetainedResourceLease<SamplerResourceTag>;

    RetainedRhiResourceLease(const RetainedRhiResourceLease &) = delete;
    RetainedRhiResourceLease &operator=(const RetainedRhiResourceLease &) = delete;

    RetainedRhiResourceLease(RetainedRhiResourceLease &&) noexcept = default;
    RetainedRhiResourceLease &operator=(RetainedRhiResourceLease &&) = delete;
    ~RetainedRhiResourceLease() noexcept = default;

    explicit RetainedRhiResourceLease(BufferLease lease) noexcept : lease_(std::move(lease)) {}
    explicit RetainedRhiResourceLease(ImageLease lease) noexcept : lease_(std::move(lease)) {}
    explicit RetainedRhiResourceLease(ImageViewLease lease) noexcept : lease_(std::move(lease)) {}
    explicit RetainedRhiResourceLease(SamplerLease lease) noexcept : lease_(std::move(lease)) {}

    [[nodiscard]] Result<void, RhiError> release() noexcept {
        return std::visit([](auto &lease) noexcept { return lease.release(); }, lease_);
    }

    [[nodiscard]] Result<OperationPin, RhiError> pin() noexcept {
        return std::visit([](auto &lease) noexcept { return lease.pin(); }, lease_);
    }

    [[nodiscard]] bool active() const noexcept {
        return std::visit([](const auto &lease) noexcept { return lease.active(); }, lease_);
    }

    template <typename ResourceTag> [[nodiscard]] bool holds(ResourceHandle<ResourceTag> handle) const noexcept {
        using Lease = RetainedResourceLease<ResourceTag>;
        if (!std::holds_alternative<Lease>(lease_))
            return false;
        const auto held = std::get<Lease>(lease_).handle();
        return held.index == handle.index && held.generation == handle.generation;
    }

    [[nodiscard]] std::size_t variantIndex() const noexcept { return lease_.index(); }

    [[nodiscard]] bool bindOwner(std::uint32_t deviceIndex, std::uint32_t deviceGeneration, std::uint32_t kind,
                                 std::uint64_t key) noexcept {
        if (ownerBound_ || !active() || key != identity())
            return false;
        deviceIndex_ = deviceIndex;
        deviceGeneration_ = deviceGeneration;
        kind_ = kind;
        key_ = key;
        ownerBound_ = true;
        return true;
    }

    [[nodiscard]] bool authorizes(std::uint32_t deviceIndex, std::uint32_t deviceGeneration, std::uint32_t kind,
                                  std::uint64_t key) const noexcept {
        return ownerBound_ && active() && deviceIndex_ == deviceIndex && deviceGeneration_ == deviceGeneration &&
               kind_ == kind && key_ == key && key == identity();
    }

private:
    [[nodiscard]] std::uint64_t identity() const noexcept {
        return std::visit([](const auto &lease) noexcept { return encodeResourceKey(lease.handle()); }, lease_);
    }

    std::variant<BufferLease, ImageLease, ImageViewLease, SamplerLease> lease_;
    std::uint32_t deviceIndex_{};
    std::uint32_t deviceGeneration_{};
    std::uint32_t kind_{};
    std::uint64_t key_{};
    bool ownerBound_{};
};

template <typename ResourceTag>
class ResourceLifecycleControl final : public CheckedIntrusiveControl<ResourceLifecycleControl<ResourceTag>> {
public:
    using Handle = ResourceHandle<ResourceTag>;
    using Teardown = Result<void, RhiError> (*)(void *, Handle) noexcept;

    struct Snapshot {
        std::uint32_t retainedLeaseCount{};
        std::uint32_t generation{};
        bool publicAlive{};
        bool occupied{};
    };

    ResourceLifecycleControl(std::uint32_t index, std::uint32_t maximumRetainedLeases,
                             std::uint64_t maximumReferences) noexcept
        : CheckedIntrusiveControl<ResourceLifecycleControl>(maximumReferences), index_(index),
          maximumRetainedLeases_(maximumRetainedLeases) {}

    ResourceLifecycleControl(const ResourceLifecycleControl &) = delete;
    ResourceLifecycleControl &operator=(const ResourceLifecycleControl &) = delete;

    ~ResourceLifecycleControl() noexcept {
        if (occupied_)
            resultContractViolation();
    }

private:
    [[nodiscard]] Result<RetainedResourceLease<ResourceTag>, RhiError>
    retain(Handle handle, CheckedIntrusiveRef<ResourceLifecycleControl> anchor) noexcept {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!isPublic(handle))
            return Result<RetainedResourceLease<ResourceTag>, RhiError>{err(staleResource("retain_resource", handle))};
        if (retainedLeaseCount_ >= maximumRetainedLeases_) {
            return Result<RetainedResourceLease<ResourceTag>, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"retain_resource", retainedLeaseCount_, index_}})};
        }
        ++retainedLeaseCount_;
        return Result<RetainedResourceLease<ResourceTag>, RhiError>{
            ok(RetainedResourceLease<ResourceTag>{std::move(anchor), handle})};
    }

    [[nodiscard]] Result<OperationPin, RhiError> pinPublic(Handle handle) noexcept {
        if (releasePrepared_.load(std::memory_order_acquire) || publicDestroyPrepared_.load(std::memory_order_acquire))
            return Result<OperationPin, RhiError>{err(releaseInProgress("pin_resource"))};
        std::lock_guard<std::mutex> guard(mutex_);
        if (releasePrepared_.load(std::memory_order_relaxed) || publicDestroyPrepared_.load(std::memory_order_relaxed))
            return Result<OperationPin, RhiError>{err(releaseInProgress("pin_resource"))};
        if (!isPublic(handle))
            return Result<OperationPin, RhiError>{err(staleResource("pin_resource", handle))};
        return pinOperation();
    }

    [[nodiscard]] Result<OperationPin, RhiError> pinRetained(Handle handle) noexcept {
        if (releasePrepared_.load(std::memory_order_acquire))
            return Result<OperationPin, RhiError>{err(releaseInProgress("pin_retained_resource"))};
        std::lock_guard<std::mutex> guard(mutex_);
        if (releasePrepared_.load(std::memory_order_relaxed))
            return Result<OperationPin, RhiError>{err(releaseInProgress("pin_retained_resource"))};
        if (!isRetained(handle))
            return Result<OperationPin, RhiError>{err(staleResource("pin_retained_resource", handle))};
        return pinOperation();
    }

    [[nodiscard]] Result<OperationPin, RhiError> pinOperation() noexcept {
        auto pinned = operations_.value().tryPin();
        if (pinned.isErr())
            return Result<OperationPin, RhiError>{err(toRhiError(std::move(pinned).error()))};
        return Result<OperationPin, RhiError>{ok(std::move(pinned).value())};
    }

    [[nodiscard]] Result<PublicResourceDestroyAttempt<ResourceTag>, RhiError>
    prepareDestroyPublic(Handle handle, Option<CheckedIntrusiveRef<ResourceLifecycleControl>> anchor) noexcept;

    [[nodiscard]] Snapshot snapshot() const noexcept {
        std::lock_guard<std::mutex> guard(mutex_);
        return {retainedLeaseCount_, generation_, publicAlive_, occupied_};
    }

private:
    [[nodiscard]] bool isPublic(Handle handle) const noexcept {
        return handle.index == index_ && handle.generation == generation_ && occupied_ && publicAlive_;
    }

    [[nodiscard]] bool isRetained(Handle handle) const noexcept {
        return handle.index == index_ && handle.generation == generation_ && occupied_;
    }

    [[nodiscard]] RhiError staleResource(const char *operation, Handle handle) const noexcept {
        return {RhiErrorCode::InvalidArgument, {operation, handle.generation, index_}};
    }

    [[nodiscard]] RhiError releaseInProgress(const char *operation) const noexcept {
        return {RhiErrorCode::LifecycleFailure, {operation, generation_, index_}};
    }

    [[nodiscard]] Result<RetainedResourceReleaseAttempt<ResourceTag>, RhiError>
    prepareRelease(RetainedResourceLease<ResourceTag> &lease, Handle handle) noexcept;

    void recycleAfterTeardown() noexcept {
        retainedLeaseCount_ = 0;
        publicAlive_ = false;
        operations_.reset();
        auto released = childLease_.value().release();
        if (released.isErr())
            resultContractViolation();
        childLease_.reset();
        occupied_ = false;
        deviceContext_ = nullptr;
        teardown_ = nullptr;
        generation_ = detail::nextRhiGeneration(generation_);
    }

    mutable std::mutex mutex_;
    std::uint32_t index_{};
    std::uint32_t retainedLeaseCount_{};
    std::uint32_t generation_{1};
    std::uint32_t maximumRetainedLeases_;
    bool publicAlive_{};
    bool occupied_{};
    std::atomic<bool> releasePrepared_{};
    std::atomic<bool> publicDestroyPrepared_{};
    void *deviceContext_{};
    Teardown teardown_{};
    Option<ChildLease> childLease_;
    Option<OperationRef> operations_;

    template <typename, typename> friend class ResourceLifecycleSlot;
    friend class RetainedResourceLease<ResourceTag>;
    friend class RetainedResourceReleaseAttempt<ResourceTag>;
    friend class PublicResourceDestroyAttempt<ResourceTag>;
};

template <typename ResourceTag> class [[nodiscard]] RetainedResourceReleaseAttempt {
public:
    using Handle = ResourceHandle<ResourceTag>;
    using Lease = RetainedResourceLease<ResourceTag>;
    using Control = ResourceLifecycleControl<ResourceTag>;

    RetainedResourceReleaseAttempt(const RetainedResourceReleaseAttempt &) = delete;
    RetainedResourceReleaseAttempt &operator=(const RetainedResourceReleaseAttempt &) = delete;

    RetainedResourceReleaseAttempt(RetainedResourceReleaseAttempt &&other) noexcept
        : lease_(std::exchange(other.lease_, nullptr)), lock_(std::move(other.lock_)),
          destruction_(std::move(other.destruction_)), active_(std::exchange(other.active_, false)) {}
    RetainedResourceReleaseAttempt &operator=(RetainedResourceReleaseAttempt &&) = delete;
    ~RetainedResourceReleaseAttempt() noexcept {
        if (active_) {
            auto rolledBack = rollback();
            if (rolledBack.isErr())
                resultContractViolation();
        }
    }

    [[nodiscard]] Result<void, RhiError> commit() noexcept {
        if (!active_ || !lease_ || !lock_.owns_lock())
            return inactiveAttempt("commit_retained_resource_release");

        Control &control = lease_->control_.value();
        if (destruction_) {
            auto tornDown = control.teardown_(control.deviceContext_, Handle{control.index_, control.generation_});
            if (tornDown.isErr())
                return tornDown;
            auto committed = destruction_.value().commit();
            if (committed.isErr())
                return Result<void, RhiError>{err(toRhiError(std::move(committed).error()))};
            destruction_.reset();
            control.recycleAfterTeardown();
        } else {
            --control.retainedLeaseCount_;
        }

        active_ = false;
        control.releasePrepared_.store(false, std::memory_order_release);
        lock_.unlock();
        lease_->control_.reset();
        lease_ = nullptr;
        return Result<void, RhiError>{ok()};
    }

    [[nodiscard]] Result<void, RhiError> rollback() noexcept {
        if (!active_ || !lease_ || !lock_.owns_lock())
            return inactiveAttempt("rollback_retained_resource_release");
        if (destruction_) {
            auto rolledBack = destruction_.value().rollback();
            if (rolledBack.isErr())
                return Result<void, RhiError>{err(toRhiError(std::move(rolledBack).error()))};
            destruction_.reset();
        }
        lease_->control_->releasePrepared_.store(false, std::memory_order_release);
        active_ = false;
        lock_.unlock();
        lease_ = nullptr;
        return Result<void, RhiError>{ok()};
    }

    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    RetainedResourceReleaseAttempt(Lease &lease, std::unique_lock<std::mutex> lock,
                                   Option<DestructionAttempt> destruction) noexcept
        : lease_(&lease), lock_(std::move(lock)), destruction_(std::move(destruction)), active_(true) {}

    [[nodiscard]] static Result<void, RhiError> inactiveAttempt(const char *operation) noexcept {
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::LifecycleFailure, {operation, 0, 0}})};
    }

    Lease *lease_{};
    // destruction_ must be destroyed before lock_ so implicit rollback cannot
    // race another lifecycle transition on this control.
    std::unique_lock<std::mutex> lock_;
    Option<DestructionAttempt> destruction_;
    bool active_{};

    friend class ResourceLifecycleControl<ResourceTag>;
};

template <typename ResourceTag> class [[nodiscard]] PublicResourceDestroyAttempt {
public:
    using Handle = ResourceHandle<ResourceTag>;
    using Control = ResourceLifecycleControl<ResourceTag>;

    PublicResourceDestroyAttempt(const PublicResourceDestroyAttempt &) = delete;
    PublicResourceDestroyAttempt &operator=(const PublicResourceDestroyAttempt &) = delete;

    PublicResourceDestroyAttempt(PublicResourceDestroyAttempt &&other) noexcept
        : control_(std::exchange(other.control_, nullptr)), anchor_(std::move(other.anchor_)),
          lock_(std::move(other.lock_)), destruction_(std::move(other.destruction_)), hideOnly_(other.hideOnly_),
          active_(std::exchange(other.active_, false)) {}
    PublicResourceDestroyAttempt &operator=(PublicResourceDestroyAttempt &&) = delete;
    ~PublicResourceDestroyAttempt() noexcept {
        if (active_) {
            auto rolledBack = rollback();
            if (rolledBack.isErr())
                resultContractViolation();
        }
    }

    [[nodiscard]] Result<void, RhiError> commit() noexcept {
        if (!active_ || !control_ || !lock_.owns_lock())
            return inactiveAttempt("commit_public_resource_destroy");
        Control &control = *control_;
        if (hideOnly_) {
            control.publicAlive_ = false;
        } else {
            auto tornDown = control.teardown_(control.deviceContext_, Handle{control.index_, control.generation_});
            if (tornDown.isErr())
                return tornDown;
            auto committed = destruction_.value().commit();
            if (committed.isErr())
                return Result<void, RhiError>{err(toRhiError(std::move(committed).error()))};
            destruction_.reset();
            control.recycleAfterTeardown();
        }
        control.publicDestroyPrepared_.store(false, std::memory_order_release);
        active_ = false;
        lock_.unlock();
        anchor_.reset();
        control_ = nullptr;
        return Result<void, RhiError>{ok()};
    }

    [[nodiscard]] Result<void, RhiError> rollback() noexcept {
        if (!active_ || !control_ || !lock_.owns_lock())
            return inactiveAttempt("rollback_public_resource_destroy");
        Control &control = *control_;
        if (destruction_) {
            auto rolledBack = destruction_.value().rollback();
            if (rolledBack.isErr())
                return Result<void, RhiError>{err(toRhiError(std::move(rolledBack).error()))};
            destruction_.reset();
        }
        control.publicDestroyPrepared_.store(false, std::memory_order_release);
        active_ = false;
        lock_.unlock();
        anchor_.reset();
        control_ = nullptr;
        return Result<void, RhiError>{ok()};
    }

    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    PublicResourceDestroyAttempt(Control *control, Option<CheckedIntrusiveRef<Control>> anchor,
                                 std::unique_lock<std::mutex> lock, Option<DestructionAttempt> destruction,
                                 bool hideOnly) noexcept
        : control_(control), anchor_(std::move(anchor)), lock_(std::move(lock)), destruction_(std::move(destruction)),
          hideOnly_(hideOnly), active_(true) {}

    [[nodiscard]] static Result<void, RhiError> inactiveAttempt(const char *operation) noexcept {
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::LifecycleFailure, {operation, 0, 0}})};
    }

    Control *control_{};
    Option<CheckedIntrusiveRef<Control>> anchor_;
    std::unique_lock<std::mutex> lock_;
    Option<DestructionAttempt> destruction_;
    bool hideOnly_{};
    bool active_{};

    friend class ResourceLifecycleControl<ResourceTag>;
};

template <typename ResourceTag> Result<OperationPin, RhiError> RetainedResourceLease<ResourceTag>::pin() noexcept {
    if (!control_)
        return Result<OperationPin, RhiError>{
            err(RhiError{RhiErrorCode::LifecycleFailure, {"pin_retained_resource", handle_.generation, 0}})};
    return control_->pinRetained(handle_);
}

template <typename ResourceTag>
Result<typename RetainedResourceLease<ResourceTag>::ReleaseAttempt, RhiError>
RetainedResourceLease<ResourceTag>::prepareRelease() noexcept {
    if (!control_)
        return Result<ReleaseAttempt, RhiError>{err(
            RhiError{RhiErrorCode::LifecycleFailure, {"prepare_retained_resource_release", handle_.generation, 0}})};
    return control_->prepareRelease(*this, handle_);
}

template <typename ResourceTag> Result<void, RhiError> RetainedResourceLease<ResourceTag>::release() noexcept {
    auto prepared = prepareRelease();
    if (prepared.isErr())
        return Result<void, RhiError>{err(std::move(prepared).error())};
    return prepared.value().commit();
}

template <typename ResourceTag>
Result<PublicResourceDestroyAttempt<ResourceTag>, RhiError> ResourceLifecycleControl<ResourceTag>::prepareDestroyPublic(
    Handle handle, Option<CheckedIntrusiveRef<ResourceLifecycleControl>> anchor) noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!isPublic(handle))
        return Result<PublicResourceDestroyAttempt<ResourceTag>, RhiError>{
            err(staleResource("prepare_destroy_resource", handle))};

    Option<DestructionAttempt> destruction;
    const bool hideOnly = retainedLeaseCount_ != 0;
    if (!hideOnly) {
        auto begun = operations_.value().beginDestroy();
        if (begun.isErr())
            return Result<PublicResourceDestroyAttempt<ResourceTag>, RhiError>{
                err(toRhiError(std::move(begun).error()))};
        destruction.emplace(std::move(begun).value());
    }
    publicDestroyPrepared_.store(true, std::memory_order_release);
    return Result<PublicResourceDestroyAttempt<ResourceTag>, RhiError>{ok(PublicResourceDestroyAttempt<ResourceTag>{
        this, std::move(anchor), std::move(lock), std::move(destruction), hideOnly})};
}

template <typename ResourceTag>
Result<RetainedResourceReleaseAttempt<ResourceTag>, RhiError>
ResourceLifecycleControl<ResourceTag>::prepareRelease(RetainedResourceLease<ResourceTag> &lease,
                                                      Handle handle) noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!lease.control_ || &lease.control_.value() != this || !isRetained(handle) ||
        handle.index != lease.handle_.index || handle.generation != lease.handle_.generation) {
        return Result<RetainedResourceReleaseAttempt<ResourceTag>, RhiError>{
            err(staleResource("prepare_retained_resource_release", handle))};
    }
    if (retainedLeaseCount_ == 0) {
        return Result<RetainedResourceReleaseAttempt<ResourceTag>, RhiError>{
            err(RhiError{RhiErrorCode::LifecycleFailure, {"prepare_retained_resource_release", 0, index_}})};
    }

    Option<DestructionAttempt> destruction;
    if (!publicAlive_ && retainedLeaseCount_ == 1) {
        auto begun = operations_.value().beginDestroy();
        if (begun.isErr())
            return Result<RetainedResourceReleaseAttempt<ResourceTag>, RhiError>{
                err(toRhiError(std::move(begun).error()))};
        destruction.emplace(std::move(begun).value());
    }
    releasePrepared_.store(true, std::memory_order_release);
    return Result<RetainedResourceReleaseAttempt<ResourceTag>, RhiError>{
        ok(RetainedResourceReleaseAttempt<ResourceTag>{lease, std::move(lock), std::move(destruction)})};
}

template <typename ResourceTag, typename FailurePolicy = DefaultLifecycleFailurePolicy> class ResourceLifecycleSlot {
public:
    using Handle = ResourceHandle<ResourceTag>;
    using Control = ResourceLifecycleControl<ResourceTag>;
    using RetainedLease = RetainedResourceLease<ResourceTag>;
    using DestroyAttempt = PublicResourceDestroyAttempt<ResourceTag>;
    using Snapshot = typename Control::Snapshot;
    using Teardown = typename Control::Teardown;

    ResourceLifecycleSlot(const ResourceLifecycleSlot &) = delete;
    ResourceLifecycleSlot &operator=(const ResourceLifecycleSlot &) = delete;
    ResourceLifecycleSlot(ResourceLifecycleSlot &&) noexcept = default;
    ResourceLifecycleSlot &operator=(ResourceLifecycleSlot &&) noexcept = default;

    [[nodiscard]] static Result<ResourceLifecycleSlot, RhiError>
    create(std::uint32_t index, std::uint32_t maximumRetainedLeases = (std::numeric_limits<std::uint32_t>::max)(),
           std::uint64_t maximumControlReferences = CheckedIntrusiveControl<Control>::maximumReferenceCount) noexcept {
        if (maximumControlReferences == 0)
            return Result<ResourceLifecycleSlot, RhiError>{err(
                RhiError{RhiErrorCode::InvalidArgument, {"create_resource_control", maximumControlReferences, index}})};
        if (FailurePolicy::fail(LifecycleAllocationPoint::ResourceControl))
            return Result<ResourceLifecycleSlot, RhiError>{
                err(detail::exhaustedRhiOperation("allocate_resource_control"))};
        Control *raw = new (std::nothrow) Control(index, maximumRetainedLeases, maximumControlReferences);
        if (!raw)
            return Result<ResourceLifecycleSlot, RhiError>{
                err(detail::exhaustedRhiOperation("allocate_resource_control"))};
        CheckedIntrusiveRef<Control> control = CheckedIntrusiveRef<Control>::adopt(raw);
        return Result<ResourceLifecycleSlot, RhiError>{ok(ResourceLifecycleSlot{std::move(control)})};
    }

    // deviceContext must be the stable Device address owned by a published
    // DeviceRegistry entry. The callback locates native records by typed
    // handle; passing an address into a movable resource container is invalid.
    [[nodiscard]] Result<Handle, RhiError> publish(ResourceCreationReservation reservation, void *deviceContext,
                                                   Teardown teardown) noexcept {
        if (!teardown)
            return Result<Handle, RhiError>{err(detail::invalidRhiHandle("publish_resource"))};
        std::lock_guard<std::mutex> guard(control_->mutex_);
        if (control_->occupied_)
            return Result<Handle, RhiError>{err(RhiError{
                RhiErrorCode::InvalidArgument, {"publish_resource", control_->generation_, control_->index_}})};
        if (FailurePolicy::fail(LifecycleAllocationPoint::ResourcePublish))
            return Result<Handle, RhiError>{err(RhiError{
                RhiErrorCode::ResourceExhausted, {"publish_resource", control_->generation_, control_->index_}})};

        auto committed = reservation.commitChild();
        if (committed.isErr())
            return Result<Handle, RhiError>{err(std::move(committed).error())};
        control_->childLease_.emplace(std::move(committed).value());
        control_->operations_.emplace(reservation.takeOperations());
        control_->retainedLeaseCount_ = 0;
        control_->occupied_ = true;
        control_->publicAlive_ = true;
        control_->deviceContext_ = deviceContext;
        control_->teardown_ = teardown;
        return Result<Handle, RhiError>{ok(Handle{control_->index_, control_->generation_})};
    }

    [[nodiscard]] Result<RetainedLease, RhiError> retain(Handle handle) noexcept {
        auto anchor = control_.retain();
        if (anchor.isErr())
            return Result<RetainedLease, RhiError>{err(toRhiError(std::move(anchor).error()))};
        return control_->retain(handle, std::move(anchor).value());
    }

    [[nodiscard]] Result<OperationPin, RhiError> pin(Handle handle) noexcept { return control_->pinPublic(handle); }

    [[nodiscard]] Result<DestroyAttempt, RhiError> prepareDestroyPublic(Handle handle) noexcept {
        auto anchor = control_.retain();
        if (anchor.isErr())
            return Result<DestroyAttempt, RhiError>{err(toRhiError(std::move(anchor).error()))};
        Option<CheckedIntrusiveRef<Control>> retained;
        retained.emplace(std::move(anchor).value());
        return control_->prepareDestroyPublic(handle, std::move(retained));
    }

    [[nodiscard]] Result<void, RhiError> destroyPublic(Handle handle) noexcept {
        auto prepared = control_->prepareDestroyPublic(handle, {});
        if (prepared.isErr())
            return Result<void, RhiError>{err(std::move(prepared).error())};
        return prepared.value().commit();
    }

    [[nodiscard]] Snapshot snapshot() const noexcept { return control_->snapshot(); }

private:
    explicit ResourceLifecycleSlot(CheckedIntrusiveRef<Control> control) noexcept : control_(std::move(control)) {}

    CheckedIntrusiveRef<Control> control_;
};

} // namespace vernon::rhi

#endif
