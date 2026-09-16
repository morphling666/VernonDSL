#include "VernonError.hpp"
#include "VernonLifecycle.hpp"
#include "VernonResult.hpp"

#include <type_traits>
#include <utility>

namespace {

using TestResult = vernon::Result<unsigned, vernon::RuntimeError>;

TestResult increment(unsigned value) noexcept { return TestResult{vernon::ok(value + 1)}; }

static_assert(std::is_nothrow_move_constructible_v<TestResult>);
static_assert(noexcept(vernon::checkedAdd(1u, 2u)));
static_assert(noexcept(vernon::checkedAtomicRetain(std::declval<std::atomic<unsigned> &>())));
static_assert(noexcept(vernon::tryMakeUnique<unsigned>()));
static_assert(noexcept(std::declval<TestResult>().andThen(increment)));
static_assert(noexcept(vernon::renderEmergencyDiagnostic(std::declval<vernon::RuntimeError>(), std::declval<char *>(),
                                                         std::declval<std::size_t>())));
static_assert(std::is_nothrow_move_constructible_v<vernon::OwnerRef>);
static_assert(std::is_nothrow_destructible_v<vernon::ChildReservation>);
static_assert(std::is_nothrow_destructible_v<vernon::ChildLease>);
static_assert(std::is_nothrow_destructible_v<vernon::CloseAttempt>);
static_assert(std::is_nothrow_move_constructible_v<vernon::OperationRef>);
static_assert(std::is_nothrow_destructible_v<vernon::OperationPin>);
static_assert(std::is_nothrow_destructible_v<vernon::DestructionAttempt>);

[[maybe_unused]] TestResult composeWithoutExceptions(unsigned value) noexcept {
    TestResult result{vernon::ok(value)};
    return std::move(result).andThen(increment);
}

} // namespace
