#include "VernonError.hpp"
#include "VernonLifecycle.hpp"
#include "VernonResult.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace {

std::atomic<std::size_t> allocations{};
std::atomic<bool> failNothrowAllocation{};

struct Tracked {
    explicit Tracked(int &destructions) noexcept : destructions(&destructions) {}
    Tracked(Tracked &&other) noexcept : destructions(other.destructions) { other.destructions = nullptr; }
    Tracked &operator=(Tracked &&) = delete;
    ~Tracked() {
        if (destructions)
            ++*destructions;
    }

    int *destructions;
};

struct NoThrowValue {
    int value{};
};

enum class TestError : unsigned char {
    Failed,
};

static_assert(!std::is_copy_constructible_v<vernon::Option<std::unique_ptr<int>>>);
static_assert(std::is_nothrow_move_constructible_v<vernon::Option<std::unique_ptr<int>>>);
static_assert(std::is_nothrow_move_constructible_v<vernon::Result<std::unique_ptr<int>, TestError>>);
static_assert(std::is_nothrow_destructible_v<vernon::Option<Tracked>>);
static_assert(std::is_nothrow_destructible_v<vernon::Result<Tracked, TestError>>);
static_assert(noexcept(vernon::checkedAdd(1u, 2u)));
static_assert(noexcept(vernon::checkedMultiply(2u, 3u)));
static_assert(noexcept(vernon::tryMakeUnique<NoThrowValue>()));

void accessFailedResultValue() {
    vernon::Result<int, TestError> result{vernon::err(TestError::Failed)};
    (void)result.value();
}

void accessSuccessfulResultError() {
    vernon::Result<int, TestError> result{vernon::ok(1)};
    (void)result.error();
}

} // namespace

void *operator new(std::size_t size) {
    if (void *memory = std::malloc(size)) {
        allocations.fetch_add(1, std::memory_order_relaxed);
        return memory;
    }
    throw std::bad_alloc{};
}

void *operator new[](std::size_t size) { return ::operator new(size); }

void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
    if (failNothrowAllocation.exchange(false, std::memory_order_relaxed))
        return nullptr;
    void *memory = std::malloc(size);
    if (memory)
        allocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
}

void *operator new[](std::size_t size, const std::nothrow_t &tag) noexcept { return ::operator new(size, tag); }

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete(void *memory, const std::nothrow_t &) noexcept { std::free(memory); }
void operator delete[](void *memory, const std::nothrow_t &) noexcept { std::free(memory); }

std::size_t vernonTestAllocationCount() noexcept { return allocations.load(std::memory_order_relaxed); }

TEST(FallibleValueTest, OptionSupportsMoveOnlyValuesCompositionAndTake) {
    vernon::Option<std::unique_ptr<int>> value{vernon::some(std::make_unique<int>(7))};
    auto mapped = std::move(value).map([](std::unique_ptr<int> input) noexcept { return *input + 5; });
    ASSERT_TRUE(mapped.hasValue());
    EXPECT_EQ(mapped.value(), 12);

    auto chained =
        std::move(mapped).andThen([](int input) noexcept { return vernon::Option<int>{vernon::some(input * 2)}; });
    ASSERT_TRUE(chained);
    EXPECT_EQ(chained.value(), 24);

    auto taken = chained.take();
    EXPECT_FALSE(chained);
    ASSERT_TRUE(taken);
    EXPECT_EQ(taken.value(), 24);

    auto recovered = vernon::Option<int>{}.orElse([]() noexcept { return vernon::Option<int>{vernon::some(9)}; });
    ASSERT_TRUE(recovered);
    EXPECT_EQ(recovered.value(), 9);
}

TEST(FallibleValueTest, ResultSupportsMoveOnlyValuesAndComposition) {
    vernon::Result<std::unique_ptr<int>, TestError> value{vernon::ok(std::make_unique<int>(3))};
    auto mapped = std::move(value).map([](std::unique_ptr<int> input) noexcept { return *input + 4; });
    ASSERT_TRUE(mapped.isOk());
    EXPECT_EQ(mapped.value(), 7);

    auto chained = std::move(mapped).andThen(
        [](int input) noexcept { return vernon::Result<int, TestError>{vernon::ok(input * 2)}; });
    ASSERT_TRUE(chained);
    EXPECT_EQ(chained.value(), 14);

    vernon::Result<void, TestError> failure{vernon::err(TestError::Failed)};
    auto recovered =
        std::move(failure).orElse([](TestError) noexcept { return vernon::Result<void, TestError>{vernon::ok()}; });
    EXPECT_TRUE(recovered);

    bool consumed = false;
    vernon::Result<int, TestError> consumable{vernon::ok(11)};
    auto voidResult = std::move(consumable).map([&consumed](int input) noexcept { consumed = input == 11; });
    EXPECT_TRUE(voidResult);
    EXPECT_TRUE(consumed);

    vernon::Result<void, TestError> emptySuccess{vernon::ok()};
    auto mappedVoid = std::move(emptySuccess).map([]() noexcept {});
    EXPECT_TRUE(mappedVoid);
}

TEST(FallibleValueTest, ActiveAlternativesAreDestroyedExactlyOnce) {
    int optionDestructions = 0;
    int resultDestructions = 0;
    {
        vernon::Option<Tracked> option{vernon::some(Tracked{optionDestructions})};
        vernon::Result<Tracked, TestError> result{vernon::ok(Tracked{resultDestructions})};
        EXPECT_TRUE(option);
        EXPECT_TRUE(result);
    }
    EXPECT_EQ(optionDestructions, 1);
    EXPECT_EQ(resultDestructions, 1);
}

TEST(FallibleValueDeathTest, InactiveAlternativeAccessTerminates) {
    EXPECT_DEATH_IF_SUPPORTED((void)vernon::Option<int>{}.value(), "");
    EXPECT_DEATH_IF_SUPPORTED(accessFailedResultValue(), "");
    EXPECT_DEATH_IF_SUPPORTED(accessSuccessfulResultError(), "");
}

TEST(FallibleValueTest, CheckedArithmeticAndAtomicCountersFailClosed) {
    auto sum = vernon::checkedAdd(std::numeric_limits<std::uint32_t>::max(), std::uint32_t{1});
    ASSERT_TRUE(sum.isErr());
    EXPECT_EQ(sum.error(), vernon::ArithmeticError::Overflow);

    auto product = vernon::checkedMultiply(std::numeric_limits<std::int64_t>::min(), std::int64_t{-1});
    ASSERT_TRUE(product.isErr());
    EXPECT_EQ(product.error(), vernon::ArithmeticError::Overflow);

    std::atomic<std::uint32_t> counter{2};
    EXPECT_TRUE(vernon::checkedAtomicRetain(counter, std::uint32_t{3}));
    auto saturated = vernon::checkedAtomicRetain(counter, std::uint32_t{3});
    EXPECT_TRUE(saturated.isErr());
    EXPECT_EQ(counter.load(), 3u);
    EXPECT_TRUE(vernon::checkedAtomicRelease(counter));

    counter.store(0);
    auto underflow = vernon::checkedAtomicRelease(counter);
    EXPECT_TRUE(underflow.isErr());
    EXPECT_EQ(counter.load(), 0u);
}

TEST(FallibleValueTest, FallibleAllocationReportsExhaustion) {
    failNothrowAllocation.store(true, std::memory_order_relaxed);
    auto object = vernon::tryMakeUnique<NoThrowValue>();
    ASSERT_TRUE(object.isErr());
    EXPECT_EQ(object.error(), vernon::AllocationError::Exhausted);

    auto empty = vernon::tryAllocateBytes(0);
    ASSERT_TRUE(empty.isErr());
    EXPECT_EQ(empty.error(), vernon::AllocationError::InvalidSize);

    failNothrowAllocation.store(true, std::memory_order_relaxed);
    auto owner = vernon::OwnerControlBlock::create();
    ASSERT_TRUE(owner.isErr());
    EXPECT_EQ(owner.error().code, vernon::LifecycleErrorCode::AllocationFailed);

    failNothrowAllocation.store(true, std::memory_order_relaxed);
    auto operations = vernon::OperationControlBlock::create();
    ASSERT_TRUE(operations.isErr());
    EXPECT_EQ(operations.error().code, vernon::LifecycleErrorCode::AllocationFailed);
}

TEST(FallibleValueTest, SuccessfulPrimitiveOperationsDoNotAllocate) {
    const std::size_t before = allocations.load(std::memory_order_relaxed);
    vernon::Result<int, TestError> result{vernon::ok(4)};
    auto mapped = std::move(result).map([](int value) noexcept { return value + 1; });
    vernon::Option<int> option{vernon::some(mapped.value())};
    auto taken = option.take();
    EXPECT_EQ(taken.value(), 5);
    EXPECT_EQ(allocations.load(std::memory_order_relaxed), before);
}

TEST(FallibleValueTest, ErrorsConvertExplicitlyAtNativeBoundaries) {
    constexpr vernon::LifecycleError saturated{vernon::LifecycleErrorCode::AdmissionSaturated, {"retain", 42, 3}};
    static_assert(vernon::toVernonRhiStatus(saturated) == VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);
    static_assert(vernon::toVernonStatus(saturated) == VERNON_STATUS_INTERNAL_ERROR);

    constexpr vernon::RhiError unsupported{vernon::RhiErrorCode::Unsupported, {"create_device", 0, 0}};
    static_assert(vernon::toVernonRhiStatus(unsupported) == VERNON_RHI_STATUS_UNSUPPORTED);
    static_assert(vernon::toVernonStatus(unsupported) == VERNON_STATUS_UNSUPPORTED_TARGET);

    constexpr vernon::ProviderError exhausted{vernon::ProviderErrorCode::ResourceExhausted,
                                              {"create_binding_set", 4096, 0}};
    static_assert(vernon::toVernonRhiStatus(exhausted) == VERNON_RHI_STATUS_RESOURCE_EXHAUSTED);
    static_assert(vernon::toVernonStatus(exhausted) == VERNON_STATUS_INTERNAL_ERROR);
}

TEST(FallibleValueTest, EmergencyDiagnosticIsBoundedAndNullTerminated) {
    constexpr vernon::RuntimeError error{vernon::RuntimeErrorCode::RhiFailure,
                                         {"synchronize_device_with_a_deliberately_long_static_name", 19, 2}};
    char output[32];
    const std::size_t required = vernon::renderEmergencyDiagnostic(error, output, sizeof(output));
    EXPECT_GT(required, sizeof(output) - 1);
    EXPECT_EQ(output[sizeof(output) - 1], '\0');
    EXPECT_STREQ(output, "domain=3 code=6 operation=synch");
}
