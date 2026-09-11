#include "VernonLifecycle.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

std::size_t vernonTestAllocationCount() noexcept;

namespace {

vernon::OwnerRef createOwner() {
    auto created = vernon::OwnerControlBlock::create();
    EXPECT_TRUE(created.isOk());
    return std::move(created).value();
}

vernon::OperationRef createOperations() {
    auto created = vernon::OperationControlBlock::create();
    EXPECT_TRUE(created.isOk());
    return std::move(created).value();
}

TEST(LifecyclePrimitiveBenchmark, AdmissionPinAndReleaseHaveNoAllocationOrGlobalLookup) {
    constexpr std::size_t iterationCount = 50000;
    vernon::OwnerRef owner = createOwner();
    vernon::OperationRef operations = createOperations();
    std::vector<std::uint64_t> admissionSamples(iterationCount);
    std::vector<std::uint64_t> childReleaseSamples(iterationCount);
    std::vector<std::uint64_t> pinSamples(iterationCount);
    std::vector<std::uint64_t> pinReleaseSamples(iterationCount);
    const std::size_t allocationsBefore = vernonTestAllocationCount();

    for (std::size_t iteration = 0; iteration < iterationCount; ++iteration) {
        const auto admissionBegin = std::chrono::steady_clock::now();
        auto reservation = owner.reserveChild();
        const auto admissionEnd = std::chrono::steady_clock::now();
        ASSERT_TRUE(reservation.isOk());
        auto lease = reservation.value().commit();
        ASSERT_TRUE(lease.isOk());

        const auto childReleaseBegin = std::chrono::steady_clock::now();
        ASSERT_TRUE(lease.value().release());
        const auto childReleaseEnd = std::chrono::steady_clock::now();

        const auto pinBegin = std::chrono::steady_clock::now();
        auto pin = operations.tryPin();
        const auto pinEnd = std::chrono::steady_clock::now();
        ASSERT_TRUE(pin.isOk());

        const auto pinReleaseBegin = std::chrono::steady_clock::now();
        ASSERT_TRUE(pin.value().release());
        const auto pinReleaseEnd = std::chrono::steady_clock::now();

        admissionSamples[iteration] = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(admissionEnd - admissionBegin).count());
        childReleaseSamples[iteration] = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(childReleaseEnd - childReleaseBegin).count());
        pinSamples[iteration] =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(pinEnd - pinBegin).count());
        pinReleaseSamples[iteration] = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(pinReleaseEnd - pinReleaseBegin).count());
    }

    const std::size_t allocationCount = vernonTestAllocationCount() - allocationsBefore;
    EXPECT_EQ(allocationCount, 0u);
    EXPECT_EQ(owner.childCount(), 0u);
    EXPECT_EQ(operations.pinCount(), 0u);

    const auto record = [&](const char *name, std::vector<std::uint64_t> &samples) {
        std::sort(samples.begin(), samples.end());
        const std::string prefix{name};
        RecordProperty((prefix + "_median_ns").c_str(), samples[iterationCount / 2]);
        RecordProperty((prefix + "_p95_ns").c_str(), samples[iterationCount * 95 / 100]);
        RecordProperty((prefix + "_p99_ns").c_str(), samples[iterationCount * 99 / 100]);
        RecordProperty((prefix + "_maximum_ns").c_str(), samples.back());
    };
    record("admission", admissionSamples);
    record("child_release", childReleaseSamples);
    record("operation_pin", pinSamples);
    record("pin_release", pinReleaseSamples);
    RecordProperty("allocation_count", allocationCount);
}

} // namespace
