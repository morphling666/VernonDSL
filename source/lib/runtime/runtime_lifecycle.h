#ifndef VERNON_RUNTIME_RUNTIME_LIFECYCLE_H
#define VERNON_RUNTIME_RUNTIME_LIFECYCLE_H

#include "VernonError.hpp"
#include "VernonLifecycle.hpp"

#include <utility>

struct VernonRuntimeContext;

namespace vernon::runtime {

// A reservation is acquired before constructing a Runtime child and is only
// published after the child is complete. Destruction closes the operation
// control before the child releases its parent lease.
class [[nodiscard]] RuntimeChildLifecycle {
public:
    RuntimeChildLifecycle(const RuntimeChildLifecycle &) = delete;
    RuntimeChildLifecycle &operator=(const RuntimeChildLifecycle &) = delete;

    RuntimeChildLifecycle(RuntimeChildLifecycle &&other) noexcept
        : reservation_(std::move(other.reservation_)), operations_(std::move(other.operations_)),
          lease_(other.lease_.take()), published_(std::exchange(other.published_, false)) {}
    RuntimeChildLifecycle &operator=(RuntimeChildLifecycle &&) = delete;
    ~RuntimeChildLifecycle() noexcept {
        if (published_ && operations_.state() != LifecycleState::Closed)
            resultContractViolation();
    }

    [[nodiscard]] static Result<RuntimeChildLifecycle, RuntimeError> reserve(const OwnerRef &owner) noexcept {
        auto child = owner.reserveChild();
        if (child.isErr())
            return Result<RuntimeChildLifecycle, RuntimeError>{err(toRuntimeError(std::move(child).error()))};
        auto operations = OperationControlBlock::create();
        if (operations.isErr())
            return Result<RuntimeChildLifecycle, RuntimeError>{err(toRuntimeError(std::move(operations).error()))};
        return Result<RuntimeChildLifecycle, RuntimeError>{
            ok(RuntimeChildLifecycle{std::move(child).value(), std::move(operations).value()})};
    }

    [[nodiscard]] Result<void, RuntimeError> publish() noexcept {
        if (published_)
            return Result<void, RuntimeError>{
                err(RuntimeError{RuntimeErrorCode::LifecycleFailure, {"publish_runtime_child", 0, 0}})};
        auto committed = reservation_.commit();
        if (committed.isErr())
            return Result<void, RuntimeError>{err(toRuntimeError(std::move(committed).error()))};
        lease_.emplace(std::move(committed).value());
        published_ = true;
        return Result<void, RuntimeError>{ok()};
    }

    [[nodiscard]] Result<OperationPin, RuntimeError> pin() const noexcept {
        if (!published_)
            return Result<OperationPin, RuntimeError>{
                err(RuntimeError{RuntimeErrorCode::LifecycleFailure, {"pin_unpublished_runtime_child", 0, 0}})};
        auto pinned = operations_.tryPin();
        if (pinned.isErr())
            return Result<OperationPin, RuntimeError>{err(toRuntimeError(std::move(pinned).error()))};
        return Result<OperationPin, RuntimeError>{ok(std::move(pinned).value())};
    }

    [[nodiscard]] Result<DestructionAttempt, RuntimeError> beginDestroy() const noexcept {
        if (!published_)
            return Result<DestructionAttempt, RuntimeError>{
                err(RuntimeError{RuntimeErrorCode::LifecycleFailure, {"destroy_unpublished_runtime_child", 0, 0}})};
        auto attempt = operations_.beginDestroy();
        if (attempt.isErr())
            return Result<DestructionAttempt, RuntimeError>{err(toRuntimeError(std::move(attempt).error()))};
        return Result<DestructionAttempt, RuntimeError>{ok(std::move(attempt).value())};
    }

    [[nodiscard]] bool published() const noexcept { return published_; }

private:
    RuntimeChildLifecycle(ChildReservation reservation, OperationRef operations) noexcept
        : reservation_(std::move(reservation)), operations_(std::move(operations)) {}

    ChildReservation reservation_;
    OperationRef operations_;
    Option<ChildLease> lease_;
    bool published_{};
};

} // namespace vernon::runtime

#endif
