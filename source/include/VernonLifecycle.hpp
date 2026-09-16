#ifndef VERNON_LIFECYCLE_HPP
#define VERNON_LIFECYCLE_HPP

#include "VernonError.hpp"
#include "VernonResult.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace vernon {

enum class LifecycleState : std::uint8_t {
    Open = 0,
    Closing = 1,
    Closed = 2,
};

struct LifecycleSnapshot {
    LifecycleState state{LifecycleState::Open};
    std::uint64_t leaseCount{};
};

static_assert(std::is_trivially_copyable_v<LifecycleSnapshot>);

namespace detail {

struct NoLifecycleHooks {
    static constexpr void beforeAcquire(const char *) noexcept {}
    static constexpr void afterAcquire(const char *) noexcept {}
    static constexpr void beforeClosing(const char *) noexcept {}
    static constexpr void afterClosing(const char *) noexcept {}
};

template <typename HookPolicy = NoLifecycleHooks> class AtomicLifecycleCounter {
public:
    static constexpr std::uint64_t maximumLeaseCount = (std::uint64_t{1} << 62) - 1;

    explicit AtomicLifecycleCounter(std::uint64_t maximum = maximumLeaseCount) noexcept : maximum_(maximum) {
        if (maximum > maximumLeaseCount)
            resultContractViolation();
    }

    AtomicLifecycleCounter(const AtomicLifecycleCounter &) = delete;
    AtomicLifecycleCounter &operator=(const AtomicLifecycleCounter &) = delete;

    [[nodiscard]] Result<void, LifecycleError> acquire(const char *operation,
                                                       LifecycleErrorCode saturationCode) noexcept {
        std::uint64_t current = word_.load(std::memory_order_acquire);
        for (;;) {
            const LifecycleState currentState = decodeState(current);
            if (currentState != LifecycleState::Open)
                return Result<void, LifecycleError>{err(stateError(currentState, operation, decodeCount(current)))};
            const std::uint64_t currentCount = decodeCount(current);
            if (currentCount >= maximum_)
                return Result<void, LifecycleError>{
                    err(LifecycleError{saturationCode, {operation, currentCount, encodeDetail(currentState)}})};
            const std::uint64_t desired = encode(LifecycleState::Open, currentCount + 1);
            HookPolicy::beforeAcquire(operation);
            if (word_.compare_exchange_weak(current, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
                HookPolicy::afterAcquire(operation);
                return Result<void, LifecycleError>{ok()};
            }
        }
    }

    [[nodiscard]] Result<void, LifecycleError> release(const char *operation) noexcept {
        std::uint64_t current = word_.load(std::memory_order_acquire);
        for (;;) {
            const std::uint64_t currentCount = decodeCount(current);
            if (currentCount == 0)
                return Result<void, LifecycleError>{err(LifecycleError{
                    LifecycleErrorCode::CounterUnderflow, {operation, 0, encodeDetail(decodeState(current))}})};
            const std::uint64_t desired = encode(decodeState(current), currentCount - 1);
            if (word_.compare_exchange_weak(current, desired, std::memory_order_acq_rel, std::memory_order_acquire))
                return Result<void, LifecycleError>{ok()};
        }
    }

    [[nodiscard]] Result<LifecycleSnapshot, LifecycleError> beginClosing(const char *operation) noexcept {
        std::uint64_t current = word_.load(std::memory_order_acquire);
        for (;;) {
            const LifecycleState currentState = decodeState(current);
            if (currentState != LifecycleState::Open)
                return Result<LifecycleSnapshot, LifecycleError>{
                    err(stateError(currentState, operation, decodeCount(current)))};
            const std::uint64_t desired = encode(LifecycleState::Closing, decodeCount(current));
            HookPolicy::beforeClosing(operation);
            if (word_.compare_exchange_weak(current, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
                HookPolicy::afterClosing(operation);
                return Result<LifecycleSnapshot, LifecycleError>{
                    ok(LifecycleSnapshot{LifecycleState::Closing, decodeCount(current)})};
            }
        }
    }

    [[nodiscard]] Result<void, LifecycleError> restoreOpen(const char *operation) noexcept {
        return transitionFromClosing(LifecycleState::Open, false, operation);
    }

    [[nodiscard]] Result<void, LifecycleError> finishClosed(const char *operation) noexcept {
        return transitionFromClosing(LifecycleState::Closed, true, operation);
    }

    [[nodiscard]] LifecycleState state() const noexcept { return decodeState(word_.load(std::memory_order_acquire)); }

    [[nodiscard]] std::uint64_t leaseCount() const noexcept {
        return decodeCount(word_.load(std::memory_order_acquire));
    }

private:
    static constexpr std::uint64_t stateShift = 62;
    static constexpr std::uint64_t countMask = maximumLeaseCount;

    static constexpr std::uint64_t encode(LifecycleState state, std::uint64_t count) noexcept {
        return (static_cast<std::uint64_t>(state) << stateShift) | count;
    }

    static constexpr LifecycleState decodeState(std::uint64_t word) noexcept {
        return static_cast<LifecycleState>(word >> stateShift);
    }

    static constexpr std::uint64_t decodeCount(std::uint64_t word) noexcept { return word & countMask; }

    static constexpr std::uint32_t encodeDetail(LifecycleState state) noexcept {
        return static_cast<std::uint32_t>(state);
    }

    static constexpr LifecycleError stateError(LifecycleState state, const char *operation,
                                               std::uint64_t count) noexcept {
        return {state == LifecycleState::Closed ? LifecycleErrorCode::AlreadyClosed : LifecycleErrorCode::NotOpen,
                {operation, count, encodeDetail(state)}};
    }

    [[nodiscard]] Result<void, LifecycleError> transitionFromClosing(LifecycleState destination, bool requireNoLeases,
                                                                     const char *operation) noexcept {
        std::uint64_t current = word_.load(std::memory_order_acquire);
        for (;;) {
            const LifecycleState currentState = decodeState(current);
            const std::uint64_t currentCount = decodeCount(current);
            if (currentState != LifecycleState::Closing)
                return Result<void, LifecycleError>{err(stateError(currentState, operation, currentCount))};
            if (requireNoLeases && currentCount != 0)
                return Result<void, LifecycleError>{err(LifecycleError{
                    LifecycleErrorCode::LiveChildren, {operation, currentCount, encodeDetail(currentState)}})};
            const std::uint64_t desired = encode(destination, currentCount);
            if (word_.compare_exchange_weak(current, desired, std::memory_order_acq_rel, std::memory_order_acquire))
                return Result<void, LifecycleError>{ok()};
        }
    }

    std::atomic<std::uint64_t> word_{encode(LifecycleState::Open, 0)};
    const std::uint64_t maximum_;
};

inline void requireLifecycleSuccess(Result<void, LifecycleError> result) noexcept {
    if (result.isErr())
        resultContractViolation();
}

} // namespace detail

template <typename T> class CheckedIntrusiveRef;

template <typename T> class CheckedIntrusiveControl {
public:
    static constexpr std::uint64_t maximumReferenceCount = std::numeric_limits<std::uint64_t>::max();

    CheckedIntrusiveControl(const CheckedIntrusiveControl &) = delete;
    CheckedIntrusiveControl &operator=(const CheckedIntrusiveControl &) = delete;

protected:
    explicit CheckedIntrusiveControl(std::uint64_t maximumReferences = maximumReferenceCount) noexcept
        : maximumReferences_(maximumReferences) {
        if (maximumReferences == 0)
            resultContractViolation();
    }
    ~CheckedIntrusiveControl() = default;

private:
    [[nodiscard]] Result<void, LifecycleError> retainReference() noexcept {
        auto retained = checkedAtomicRetain(references_, maximumReferences_);
        if (retained.isErr())
            return Result<void, LifecycleError>{
                err(LifecycleError{LifecycleErrorCode::AdmissionSaturated,
                                   {"retain_intrusive_reference", references_.load(std::memory_order_acquire), 0}})};
        return Result<void, LifecycleError>{ok()};
    }

    void releaseReference() noexcept {
        std::uint64_t current = references_.load(std::memory_order_acquire);
        for (;;) {
            if (current == 0)
                resultContractViolation();
            if (references_.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
                if (current == 1)
                    delete static_cast<T *>(this);
                return;
            }
        }
    }

    std::atomic<std::uint64_t> references_{1};
    const std::uint64_t maximumReferences_;

    friend class CheckedIntrusiveRef<T>;
};

template <typename T> class [[nodiscard]] CheckedIntrusiveRef {
public:
    CheckedIntrusiveRef(const CheckedIntrusiveRef &) = delete;
    CheckedIntrusiveRef &operator=(const CheckedIntrusiveRef &) = delete;

    CheckedIntrusiveRef(CheckedIntrusiveRef &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    CheckedIntrusiveRef &operator=(CheckedIntrusiveRef &&other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    ~CheckedIntrusiveRef() noexcept { reset(); }

    [[nodiscard]] static CheckedIntrusiveRef adopt(T *value) noexcept {
        if (!value)
            resultContractViolation();
        return CheckedIntrusiveRef{value};
    }

    [[nodiscard]] static Result<CheckedIntrusiveRef, LifecycleError> retain(T *value) noexcept {
        if (!value)
            return Result<CheckedIntrusiveRef, LifecycleError>{
                err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"retain_intrusive_reference", 0, 0}})};
        auto retained = value->retainReference();
        if (retained.isErr())
            return Result<CheckedIntrusiveRef, LifecycleError>{err(std::move(retained).error())};
        return Result<CheckedIntrusiveRef, LifecycleError>{ok(CheckedIntrusiveRef{value})};
    }

    [[nodiscard]] Result<CheckedIntrusiveRef, LifecycleError> retain() const noexcept { return retain(&value()); }

    [[nodiscard]] T &value() const noexcept {
        if (!value_)
            resultContractViolation();
        return *value_;
    }

    [[nodiscard]] T *operator->() const noexcept { return &value(); }
    [[nodiscard]] explicit operator bool() const noexcept { return value_ != nullptr; }

    void reset() noexcept {
        if (value_) {
            T *value = std::exchange(value_, nullptr);
            value->releaseReference();
        }
    }

private:
    explicit CheckedIntrusiveRef(T *value) noexcept : value_(value) {}

    T *value_{};
};

class OwnerControlBlock;
class OwnerRef;
class ChildReservation;
class ChildLease;
class CloseAttempt;

class [[nodiscard]] OwnerRef {
public:
    OwnerRef(const OwnerRef &) = delete;
    OwnerRef &operator=(const OwnerRef &) = delete;

    OwnerRef(OwnerRef &&other) noexcept : control_(std::exchange(other.control_, nullptr)) {}
    OwnerRef &operator=(OwnerRef &&other) noexcept;
    ~OwnerRef() noexcept;

    [[nodiscard]] Result<OwnerRef, LifecycleError> retain() const noexcept;
    [[nodiscard]] Result<ChildReservation, LifecycleError> reserveChild() const noexcept;
    [[nodiscard]] Result<CloseAttempt, LifecycleError> beginClose() const noexcept;
    [[nodiscard]] LifecycleState state() const noexcept;
    [[nodiscard]] std::uint64_t childCount() const noexcept;
    void reset() noexcept;

private:
    struct AdoptReference {};

    explicit OwnerRef(OwnerControlBlock *control, AdoptReference) noexcept : control_(control) {}
    [[nodiscard]] OwnerControlBlock &control() const noexcept;

    OwnerControlBlock *control_{};

    friend class OwnerControlBlock;
    friend class ChildReservation;
    friend class ChildLease;
    friend class CloseAttempt;
};

class OwnerControlBlock {
public:
    static constexpr std::uint64_t maximumChildCount = detail::AtomicLifecycleCounter<>::maximumLeaseCount;
    static constexpr std::uint64_t maximumReferenceCount = std::numeric_limits<std::uint64_t>::max();

    OwnerControlBlock(const OwnerControlBlock &) = delete;
    OwnerControlBlock &operator=(const OwnerControlBlock &) = delete;

    [[nodiscard]] static Result<OwnerRef, LifecycleError>
    create(std::uint64_t maximumChildren = maximumChildCount,
           std::uint64_t maximumReferences = maximumReferenceCount) noexcept;

private:
    OwnerControlBlock(std::uint64_t maximumChildren, std::uint64_t maximumReferences) noexcept
        : children_(maximumChildren), maximumReferences_(maximumReferences) {}
    ~OwnerControlBlock() = default;

    [[nodiscard]] Result<OwnerRef, LifecycleError> retainReference() noexcept;
    void releaseReference() noexcept;
    [[nodiscard]] Result<ChildReservation, LifecycleError> reserveChild(OwnerRef owner) noexcept;
    [[nodiscard]] Result<CloseAttempt, LifecycleError> beginClose(OwnerRef owner) noexcept;

    detail::AtomicLifecycleCounter<> children_;
    std::atomic<std::uint64_t> references_{1};
    const std::uint64_t maximumReferences_;

    friend class OwnerRef;
    friend class ChildReservation;
    friend class ChildLease;
    friend class CloseAttempt;
};

class [[nodiscard]] ChildReservation {
public:
    ChildReservation(const ChildReservation &) = delete;
    ChildReservation &operator=(const ChildReservation &) = delete;

    ChildReservation(ChildReservation &&other) noexcept
        : owner_(std::move(other.owner_)), active_(std::exchange(other.active_, false)) {}
    ChildReservation &operator=(ChildReservation &&) = delete;
    ~ChildReservation() noexcept;

    [[nodiscard]] Result<ChildLease, LifecycleError> commit() noexcept;
    [[nodiscard]] Result<void, LifecycleError> rollback() noexcept;
    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    explicit ChildReservation(OwnerRef owner) noexcept : owner_(std::move(owner)), active_(true) {}

    OwnerRef owner_;
    bool active_{};

    friend class OwnerControlBlock;
};

class [[nodiscard]] ChildLease {
public:
    ChildLease(const ChildLease &) = delete;
    ChildLease &operator=(const ChildLease &) = delete;

    ChildLease(ChildLease &&other) noexcept
        : owner_(std::move(other.owner_)), active_(std::exchange(other.active_, false)) {}
    ChildLease &operator=(ChildLease &&) = delete;
    ~ChildLease() noexcept;

    [[nodiscard]] Result<void, LifecycleError> release() noexcept;
    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] Result<OwnerRef, LifecycleError> retainOwner() const noexcept;

private:
    explicit ChildLease(OwnerRef owner) noexcept : owner_(std::move(owner)), active_(true) {}

    OwnerRef owner_;
    bool active_{};

    friend class ChildReservation;
};

class [[nodiscard]] CloseAttempt {
public:
    CloseAttempt(const CloseAttempt &) = delete;
    CloseAttempt &operator=(const CloseAttempt &) = delete;

    CloseAttempt(CloseAttempt &&other) noexcept
        : owner_(std::move(other.owner_)), snapshot_(other.snapshot_), active_(std::exchange(other.active_, false)) {}
    CloseAttempt &operator=(CloseAttempt &&) = delete;
    ~CloseAttempt() noexcept;

    [[nodiscard]] Result<void, LifecycleError> commit() noexcept;
    [[nodiscard]] Result<void, LifecycleError> rollback() noexcept;
    [[nodiscard]] LifecycleSnapshot snapshot() const noexcept { return snapshot_; }
    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    CloseAttempt(OwnerRef owner, LifecycleSnapshot snapshot) noexcept
        : owner_(std::move(owner)), snapshot_(snapshot), active_(true) {}

    OwnerRef owner_;
    LifecycleSnapshot snapshot_{};
    bool active_{};

    friend class OwnerControlBlock;
};

inline OwnerRef &OwnerRef::operator=(OwnerRef &&other) noexcept {
    if (this != &other) {
        reset();
        control_ = std::exchange(other.control_, nullptr);
    }
    return *this;
}

inline OwnerRef::~OwnerRef() noexcept { reset(); }

inline void OwnerRef::reset() noexcept {
    if (control_) {
        OwnerControlBlock *control = std::exchange(control_, nullptr);
        control->releaseReference();
    }
}

inline OwnerControlBlock &OwnerRef::control() const noexcept {
    if (!control_)
        resultContractViolation();
    return *control_;
}

inline Result<OwnerRef, LifecycleError> OwnerRef::retain() const noexcept { return control().retainReference(); }

inline Result<ChildReservation, LifecycleError> OwnerRef::reserveChild() const noexcept {
    auto retained = retain();
    if (retained.isErr())
        return Result<ChildReservation, LifecycleError>{err(std::move(retained).error())};
    return control().reserveChild(std::move(retained).value());
}

inline Result<CloseAttempt, LifecycleError> OwnerRef::beginClose() const noexcept {
    auto retained = retain();
    if (retained.isErr())
        return Result<CloseAttempt, LifecycleError>{err(std::move(retained).error())};
    return control().beginClose(std::move(retained).value());
}

inline LifecycleState OwnerRef::state() const noexcept { return control().children_.state(); }

inline std::uint64_t OwnerRef::childCount() const noexcept { return control().children_.leaseCount(); }

inline Result<OwnerRef, LifecycleError> OwnerControlBlock::create(std::uint64_t maximumChildren,
                                                                  std::uint64_t maximumReferences) noexcept {
    if (maximumChildren > detail::AtomicLifecycleCounter<>::maximumLeaseCount || maximumReferences == 0)
        return Result<OwnerRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::InvalidConfiguration,
                               {"create_owner_control", maximumChildren, maximumReferences == 0 ? 1u : 2u}})};
    OwnerControlBlock *control = new (std::nothrow) OwnerControlBlock(maximumChildren, maximumReferences);
    if (!control)
        return Result<OwnerRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::AllocationFailed, {"create_owner_control", 0, 0}})};
    return Result<OwnerRef, LifecycleError>{ok(OwnerRef{control, OwnerRef::AdoptReference{}})};
}

inline Result<OwnerRef, LifecycleError> OwnerControlBlock::retainReference() noexcept {
    auto retained = checkedAtomicRetain(references_, maximumReferences_);
    if (retained.isErr())
        return Result<OwnerRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::AdmissionSaturated,
                               {"retain_owner_control", references_.load(std::memory_order_acquire), 0}})};
    return Result<OwnerRef, LifecycleError>{ok(OwnerRef{this, OwnerRef::AdoptReference{}})};
}

inline void OwnerControlBlock::releaseReference() noexcept {
    std::uint64_t current = references_.load(std::memory_order_acquire);
    for (;;) {
        if (current == 0)
            resultContractViolation();
        if (references_.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            if (current == 1)
                delete this;
            return;
        }
    }
}

inline Result<ChildReservation, LifecycleError> OwnerControlBlock::reserveChild(OwnerRef owner) noexcept {
    auto acquired = children_.acquire("reserve_child", LifecycleErrorCode::AdmissionSaturated);
    if (acquired.isErr())
        return Result<ChildReservation, LifecycleError>{err(std::move(acquired).error())};
    return Result<ChildReservation, LifecycleError>{ok(ChildReservation{std::move(owner)})};
}

inline Result<CloseAttempt, LifecycleError> OwnerControlBlock::beginClose(OwnerRef owner) noexcept {
    auto transition = children_.beginClosing("begin_owner_close");
    if (transition.isErr())
        return Result<CloseAttempt, LifecycleError>{err(std::move(transition).error())};
    const LifecycleSnapshot snapshot = transition.value();
    if (snapshot.leaseCount != 0) {
        detail::requireLifecycleSuccess(children_.restoreOpen("reject_owner_close"));
        return Result<CloseAttempt, LifecycleError>{err(
            LifecycleError{LifecycleErrorCode::LiveChildren,
                           {"begin_owner_close", snapshot.leaseCount, static_cast<std::uint32_t>(snapshot.state)}})};
    }
    return Result<CloseAttempt, LifecycleError>{ok(CloseAttempt{std::move(owner), snapshot})};
}

inline ChildReservation::~ChildReservation() noexcept {
    if (active_)
        detail::requireLifecycleSuccess(owner_.control().children_.release("rollback_child_reservation"));
}

inline Result<ChildLease, LifecycleError> ChildReservation::commit() noexcept {
    if (!active_)
        return Result<ChildLease, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"commit_child_reservation", 0, 0}})};
    active_ = false;
    return Result<ChildLease, LifecycleError>{ok(ChildLease{std::move(owner_)})};
}

inline Result<void, LifecycleError> ChildReservation::rollback() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"rollback_child_reservation", 0, 0}})};
    auto released = owner_.control().children_.release("rollback_child_reservation");
    if (released.isErr())
        return released;
    active_ = false;
    owner_.reset();
    return Result<void, LifecycleError>{ok()};
}

inline ChildLease::~ChildLease() noexcept {
    if (active_)
        detail::requireLifecycleSuccess(owner_.control().children_.release("release_child_lease"));
}

inline Result<void, LifecycleError> ChildLease::release() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"release_child_lease", 0, 0}})};
    auto released = owner_.control().children_.release("release_child_lease");
    if (released.isErr())
        return released;
    active_ = false;
    owner_.reset();
    return Result<void, LifecycleError>{ok()};
}

inline Result<OwnerRef, LifecycleError> ChildLease::retainOwner() const noexcept {
    if (!active_)
        return Result<OwnerRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"retain_child_owner", 0, 0}})};
    return owner_.retain();
}

inline CloseAttempt::~CloseAttempt() noexcept {
    if (active_)
        detail::requireLifecycleSuccess(owner_.control().children_.restoreOpen("rollback_owner_close"));
}

inline Result<void, LifecycleError> CloseAttempt::commit() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"commit_owner_close", 0, 0}})};
    auto closed = owner_.control().children_.finishClosed("commit_owner_close");
    if (closed.isErr())
        return closed;
    active_ = false;
    owner_.reset();
    return Result<void, LifecycleError>{ok()};
}

inline Result<void, LifecycleError> CloseAttempt::rollback() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"rollback_owner_close", 0, 0}})};
    auto restored = owner_.control().children_.restoreOpen("rollback_owner_close");
    if (restored.isErr())
        return restored;
    active_ = false;
    owner_.reset();
    return Result<void, LifecycleError>{ok()};
}

class OperationControlBlock;
class OperationRef;
class OperationPin;
class DestructionAttempt;

class [[nodiscard]] OperationRef {
public:
    OperationRef(const OperationRef &) = delete;
    OperationRef &operator=(const OperationRef &) = delete;

    OperationRef(OperationRef &&other) noexcept : control_(std::exchange(other.control_, nullptr)) {}
    OperationRef &operator=(OperationRef &&other) noexcept;
    ~OperationRef() noexcept;

    [[nodiscard]] Result<OperationRef, LifecycleError> retain() const noexcept;
    [[nodiscard]] Result<OperationPin, LifecycleError> tryPin() const noexcept;
    [[nodiscard]] Result<DestructionAttempt, LifecycleError> beginDestroy() const noexcept;
    [[nodiscard]] LifecycleState state() const noexcept;
    [[nodiscard]] std::uint64_t pinCount() const noexcept;
    void reset() noexcept;

private:
    struct AdoptReference {};

    explicit OperationRef(OperationControlBlock *control, AdoptReference) noexcept : control_(control) {}
    [[nodiscard]] OperationControlBlock &control() const noexcept;

    OperationControlBlock *control_{};

    friend class OperationControlBlock;
    friend class OperationPin;
    friend class DestructionAttempt;
};

class OperationControlBlock {
public:
    static constexpr std::uint64_t maximumPinCount = detail::AtomicLifecycleCounter<>::maximumLeaseCount;
    static constexpr std::uint64_t maximumReferenceCount = std::numeric_limits<std::uint64_t>::max();

    OperationControlBlock(const OperationControlBlock &) = delete;
    OperationControlBlock &operator=(const OperationControlBlock &) = delete;

    [[nodiscard]] static Result<OperationRef, LifecycleError>
    create(std::uint64_t maximumPins = maximumPinCount,
           std::uint64_t maximumReferences = maximumReferenceCount) noexcept;

private:
    OperationControlBlock(std::uint64_t maximumPins, std::uint64_t maximumReferences) noexcept
        : pins_(maximumPins), maximumReferences_(maximumReferences) {}
    ~OperationControlBlock() noexcept {
        if (pins_.leaseCount() != 0)
            resultContractViolation();
    }

    [[nodiscard]] Result<OperationRef, LifecycleError> retainReference() noexcept;
    void releaseReference() noexcept;
    [[nodiscard]] Result<OperationPin, LifecycleError> tryPin(OperationRef control) noexcept;
    [[nodiscard]] Result<DestructionAttempt, LifecycleError> beginDestroy(OperationRef control) noexcept;

    detail::AtomicLifecycleCounter<> pins_;
    std::atomic<std::uint64_t> references_{1};
    const std::uint64_t maximumReferences_;

    friend class OperationRef;
    friend class OperationPin;
    friend class DestructionAttempt;
};

class [[nodiscard]] OperationPin {
public:
    OperationPin(const OperationPin &) = delete;
    OperationPin &operator=(const OperationPin &) = delete;

    OperationPin(OperationPin &&other) noexcept
        : control_(std::move(other.control_)), active_(std::exchange(other.active_, false)) {}
    OperationPin &operator=(OperationPin &&) = delete;
    ~OperationPin() noexcept;

    [[nodiscard]] Result<void, LifecycleError> release() noexcept;
    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    explicit OperationPin(OperationRef control) noexcept : control_(std::move(control)), active_(true) {}

    OperationRef control_;
    bool active_{};

    friend class OperationControlBlock;
};

class [[nodiscard]] DestructionAttempt {
public:
    DestructionAttempt(const DestructionAttempt &) = delete;
    DestructionAttempt &operator=(const DestructionAttempt &) = delete;

    DestructionAttempt(DestructionAttempt &&other) noexcept
        : control_(std::move(other.control_)), snapshot_(other.snapshot_),
          active_(std::exchange(other.active_, false)) {}
    DestructionAttempt &operator=(DestructionAttempt &&) = delete;
    ~DestructionAttempt() noexcept;

    [[nodiscard]] Result<void, LifecycleError> commit() noexcept;
    [[nodiscard]] Result<void, LifecycleError> rollback() noexcept;
    [[nodiscard]] LifecycleSnapshot snapshot() const noexcept { return snapshot_; }
    [[nodiscard]] bool active() const noexcept { return active_; }

private:
    DestructionAttempt(OperationRef control, LifecycleSnapshot snapshot) noexcept
        : control_(std::move(control)), snapshot_(snapshot), active_(true) {}

    OperationRef control_;
    LifecycleSnapshot snapshot_{};
    bool active_{};

    friend class OperationControlBlock;
};

inline OperationRef &OperationRef::operator=(OperationRef &&other) noexcept {
    if (this != &other) {
        reset();
        control_ = std::exchange(other.control_, nullptr);
    }
    return *this;
}

inline OperationRef::~OperationRef() noexcept { reset(); }

inline void OperationRef::reset() noexcept {
    if (control_) {
        OperationControlBlock *control = std::exchange(control_, nullptr);
        control->releaseReference();
    }
}

inline OperationControlBlock &OperationRef::control() const noexcept {
    if (!control_)
        resultContractViolation();
    return *control_;
}

inline Result<OperationRef, LifecycleError> OperationRef::retain() const noexcept {
    return control().retainReference();
}

inline Result<OperationPin, LifecycleError> OperationRef::tryPin() const noexcept {
    auto retained = retain();
    if (retained.isErr())
        return Result<OperationPin, LifecycleError>{err(std::move(retained).error())};
    return control().tryPin(std::move(retained).value());
}

inline Result<DestructionAttempt, LifecycleError> OperationRef::beginDestroy() const noexcept {
    auto retained = retain();
    if (retained.isErr())
        return Result<DestructionAttempt, LifecycleError>{err(std::move(retained).error())};
    return control().beginDestroy(std::move(retained).value());
}

inline LifecycleState OperationRef::state() const noexcept { return control().pins_.state(); }

inline std::uint64_t OperationRef::pinCount() const noexcept { return control().pins_.leaseCount(); }

inline Result<OperationRef, LifecycleError> OperationControlBlock::create(std::uint64_t maximumPins,
                                                                          std::uint64_t maximumReferences) noexcept {
    if (maximumPins > maximumPinCount || maximumReferences == 0)
        return Result<OperationRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::InvalidConfiguration,
                               {"create_operation_control", maximumPins, maximumReferences == 0 ? 1u : 2u}})};
    OperationControlBlock *control = new (std::nothrow) OperationControlBlock(maximumPins, maximumReferences);
    if (!control)
        return Result<OperationRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::AllocationFailed, {"create_operation_control", 0, 0}})};
    return Result<OperationRef, LifecycleError>{ok(OperationRef{control, OperationRef::AdoptReference{}})};
}

inline Result<OperationRef, LifecycleError> OperationControlBlock::retainReference() noexcept {
    auto retained = checkedAtomicRetain(references_, maximumReferences_);
    if (retained.isErr())
        return Result<OperationRef, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::AdmissionSaturated,
                               {"retain_operation_control", references_.load(std::memory_order_acquire), 0}})};
    return Result<OperationRef, LifecycleError>{ok(OperationRef{this, OperationRef::AdoptReference{}})};
}

inline void OperationControlBlock::releaseReference() noexcept {
    std::uint64_t current = references_.load(std::memory_order_acquire);
    for (;;) {
        if (current == 0)
            resultContractViolation();
        if (references_.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            if (current == 1)
                delete this;
            return;
        }
    }
}

inline Result<OperationPin, LifecycleError> OperationControlBlock::tryPin(OperationRef control) noexcept {
    auto acquired = pins_.acquire("pin_child_operation", LifecycleErrorCode::PinRejected);
    if (acquired.isErr())
        return Result<OperationPin, LifecycleError>{err(std::move(acquired).error())};
    return Result<OperationPin, LifecycleError>{ok(OperationPin{std::move(control)})};
}

inline Result<DestructionAttempt, LifecycleError> OperationControlBlock::beginDestroy(OperationRef control) noexcept {
    auto transition = pins_.beginClosing("begin_child_destroy");
    if (transition.isErr())
        return Result<DestructionAttempt, LifecycleError>{err(std::move(transition).error())};
    const LifecycleSnapshot snapshot = transition.value();
    if (snapshot.leaseCount != 0) {
        detail::requireLifecycleSuccess(pins_.restoreOpen("reject_child_destroy"));
        return Result<DestructionAttempt, LifecycleError>{err(
            LifecycleError{LifecycleErrorCode::PinRejected,
                           {"begin_child_destroy", snapshot.leaseCount, static_cast<std::uint32_t>(snapshot.state)}})};
    }
    return Result<DestructionAttempt, LifecycleError>{ok(DestructionAttempt{std::move(control), snapshot})};
}

inline OperationPin::~OperationPin() noexcept {
    if (active_)
        detail::requireLifecycleSuccess(control_.control().pins_.release("release_operation_pin"));
}

inline Result<void, LifecycleError> OperationPin::release() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"release_operation_pin", 0, 0}})};
    auto released = control_.control().pins_.release("release_operation_pin");
    if (released.isErr())
        return released;
    active_ = false;
    control_.reset();
    return Result<void, LifecycleError>{ok()};
}

inline DestructionAttempt::~DestructionAttempt() noexcept {
    if (active_)
        detail::requireLifecycleSuccess(control_.control().pins_.restoreOpen("rollback_child_destroy"));
}

inline Result<void, LifecycleError> DestructionAttempt::commit() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"commit_child_destroy", 0, 0}})};
    auto closed = control_.control().pins_.finishClosed("commit_child_destroy");
    if (closed.isErr())
        return closed;
    active_ = false;
    control_.reset();
    return Result<void, LifecycleError>{ok()};
}

inline Result<void, LifecycleError> DestructionAttempt::rollback() noexcept {
    if (!active_)
        return Result<void, LifecycleError>{
            err(LifecycleError{LifecycleErrorCode::StaleAdmission, {"rollback_child_destroy", 0, 0}})};
    auto restored = control_.control().pins_.restoreOpen("rollback_child_destroy");
    if (restored.isErr())
        return restored;
    active_ = false;
    control_.reset();
    return Result<void, LifecycleError>{ok()};
}

} // namespace vernon

#endif
