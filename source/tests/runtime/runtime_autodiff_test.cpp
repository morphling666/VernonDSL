#include "runtime/autodiff/host_effect_transaction.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>

namespace {

TEST(RuntimeAutodiffTapeAllocator, HasFrozenVersionedAbi) {
    EXPECT_EQ(VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION, 2u);
    EXPECT_EQ(sizeof(VernonAdTapeAllocatorStatus), 4u);
    EXPECT_EQ(sizeof(VernonAdRegionHandle), 8u);
    EXPECT_EQ(sizeof(VernonAdRecordHandle), 8u);
    EXPECT_EQ(sizeof(VernonAdTapeAllocator), sizeof(void *) == 8 ? 128u : 68u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, abi_version), sizeof(void *) == 8 ? 8u : 4u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, status), sizeof(void *) == 8 ? 12u : 8u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, user_data), sizeof(void *) == 8 ? 16u : 12u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, capacity_bytes), sizeof(void *) == 8 ? 24u : 16u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, required_bytes), sizeof(void *) == 8 ? 32u : 20u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, reset), sizeof(void *) == 8 ? 40u : 24u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, begin_region), sizeof(void *) == 8 ? 48u : 28u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, reserve_record), sizeof(void *) == 8 ? 56u : 32u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, write_leaf), sizeof(void *) == 8 ? 64u : 36u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, set_child), sizeof(void *) == 8 ? 72u : 40u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, end_region), sizeof(void *) == 8 ? 80u : 44u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, seal), sizeof(void *) == 8 ? 88u : 48u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_leaf), sizeof(void *) == 8 ? 96u : 52u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_child), sizeof(void *) == 8 ? 104u : 56u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_executed_count), sizeof(void *) == 8 ? 112u : 60u);
    EXPECT_EQ(offsetof(VernonAdTapeAllocator, read_exit_kind), sizeof(void *) == 8 ? 120u : 64u);

    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator incompatible = storage.descriptor();
    --incompatible.struct_size;
    EXPECT_EQ(incompatible.reset(&incompatible), VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
    incompatible = storage.descriptor();
    ++incompatible.abi_version;
    EXPECT_EQ(incompatible.reset(&incompatible), VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
}

TEST(RuntimeAutodiffEffectTransaction, CommitsOrDiscardsExactlyOnce) {
    float storage = 3.0f;
    float output = -1.0f;
    vernon::runtime::ad::HostEffectTransaction transaction(sizeof(output));
    auto *stagedStorage = reinterpret_cast<float *>(transaction.stageStorage(&storage, sizeof(storage), true));
    auto *stagedOutput = reinterpret_cast<float *>(transaction.stagedOutput());
    ASSERT_NE(stagedStorage, nullptr);
    ASSERT_NE(stagedOutput, nullptr);
    *stagedStorage = 4.0f;
    *stagedOutput = 9.0f;
    EXPECT_TRUE(transaction.commit(&output));
    EXPECT_FLOAT_EQ(storage, 4.0f);
    EXPECT_FLOAT_EQ(output, 9.0f);
    EXPECT_EQ(transaction.stageStorage(&storage, sizeof(storage), true), nullptr);
    EXPECT_FALSE(transaction.commit(&output));

    vernon::runtime::ad::HostEffectTransaction discarded(sizeof(output));
    ASSERT_NE(discarded.stageStorage(&storage, sizeof(storage), true), nullptr);
    EXPECT_TRUE(discarded.discard());
    EXPECT_EQ(discarded.stagedOutput(), nullptr);
    EXPECT_FALSE(discarded.commit(&output));
}

TEST(RuntimeAutodiffTapeAllocator, PreservesNestedRecordsInSealedSnapshot) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator &allocator = storage.descriptor();
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRegionHandle child = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    VernonAdRecordHandle childRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.reserve_record(&allocator, root, sizeof(uint32_t), alignof(uint32_t), 1, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t rootValue = 0x12345678u;
    const uint32_t childValue = 0xabcdef01u;
    ASSERT_EQ(allocator.write_leaf(&allocator, rootRecord, 0, &rootValue, sizeof(rootValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, root, &child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.set_child(&allocator, rootRecord, 0, child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.reserve_record(&allocator, child, sizeof(uint32_t), alignof(uint32_t), 0, &childRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.write_leaf(&allocator, childRecord, 0, &childValue, sizeof(childValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.end_region(&allocator, child, 1, 2), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.end_region(&allocator, root, 1, 3), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);

    std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot = storage.takeSnapshot();
    ASSERT_NE(snapshot, nullptr);
    VernonAdTapeAllocator *reader = snapshot->descriptor();
    uint32_t read = 0;
    ASSERT_EQ(reader->read_leaf(reader, root, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, rootValue);
    VernonAdRegionHandle readChild = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(reader->read_child(reader, root, 0, 0, &readChild), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(readChild, child);
    ASSERT_EQ(reader->read_leaf(reader, child, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, childValue);
    size_t count = 0;
    uint32_t exitKind = 0;
    ASSERT_EQ(reader->read_executed_count(reader, root, &count), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(reader->read_exit_kind(reader, root, &exitKind), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(exitKind, 3u);
}

TEST(RuntimeAutodiffTapeAllocator, ReportsCapacityAndArithmeticFailures) {
    vernon::runtime::ad::HostDynamicTape bounded(8);
    VernonAdTapeAllocator &allocator = bounded.descriptor();
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.reserve_record(&allocator, region, 5, 4, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.required_bytes, 5u);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 8, 8, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    EXPECT_EQ(allocator.required_bytes, 16u);

    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, std::numeric_limits<size_t>::max(), 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 1, 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
}

TEST(RuntimeAutodiffTapeAllocator, WritesReservedOffsetsAfterStorageGrowth) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator &allocator = storage.descriptor();
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, root, sizeof(uint32_t), alignof(uint32_t), 0, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    VernonAdRecordHandle growthRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.reserve_record(&allocator, root, 4096, 8, 0, &growthRecord), VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t value = 0x12345678u;
    ASSERT_EQ(allocator.write_leaf(&allocator, rootRecord, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.end_region(&allocator, root, 2, 7), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot = storage.takeSnapshot();
    ASSERT_NE(snapshot, nullptr);
    uint32_t read = 0;
    VernonAdTapeAllocator *reader = snapshot->descriptor();
    ASSERT_EQ(reader->read_leaf(reader, root, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, value);
}

TEST(RuntimeAutodiffTapeAllocator, RetainsSharedMemoryPolicyChargeUntilSnapshotRelease) {
    auto policy = std::make_shared<vernon::runtime::ad::HostTapeMemoryPolicy>(16, 16);
    std::shared_ptr<const vernon::runtime::ad::HostTapeSnapshot> snapshot;
    {
        vernon::runtime::ad::HostDynamicTape first(std::numeric_limits<size_t>::max(), policy);
        VernonAdTapeAllocator &allocator = first.descriptor();
        VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator.reserve_record(&allocator, region, 16, 1, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator.end_region(&allocator, region, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator.seal(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        snapshot = first.takeSnapshot();
        ASSERT_NE(snapshot, nullptr);
    }
    vernon::runtime::ad::HostDynamicTape second(std::numeric_limits<size_t>::max(), policy);
    VernonAdTapeAllocator &allocator = second.descriptor();
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 1, 1, 0, &record),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    snapshot.reset();
    ASSERT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator.begin_region(&allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reserve_record(&allocator, region, 1, 1, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiffTapeAllocator, DispatchBudgetAllowsUnevenInvocationUsage) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>(16, 16);
    std::shared_ptr<HostTapeDispatchBudget> budget = HostTapeDispatchBudget::reserve(policy, 16);
    ASSERT_NE(budget, nullptr);
    ASSERT_EQ(budget->capacity(), 16u);

    std::shared_ptr<const HostTapeSnapshot> largeSnapshot;
    std::shared_ptr<const HostTapeSnapshot> smallSnapshot;
    {
        HostDynamicTape large(16, policy, budget);
        VernonAdTapeAllocator &largeAllocator = large.descriptor();
        VernonAdRegionHandle largeRegion = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle largeRecord = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(largeAllocator.begin_region(&largeAllocator, VERNON_AD_INVALID_REGION_HANDLE, &largeRegion),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(largeAllocator.reserve_record(&largeAllocator, largeRegion, 12, 1, 0, &largeRecord),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(largeAllocator.end_region(&largeAllocator, largeRegion, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(largeAllocator.seal(&largeAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        largeSnapshot = large.takeSnapshot();
        ASSERT_NE(largeSnapshot, nullptr);

        HostDynamicTape small(16, policy, budget);
        VernonAdTapeAllocator &smallAllocator = small.descriptor();
        VernonAdRegionHandle smallRegion = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle smallRecord = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(smallAllocator.begin_region(&smallAllocator, VERNON_AD_INVALID_REGION_HANDLE, &smallRegion),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(smallAllocator.reserve_record(&smallAllocator, smallRegion, 4, 1, 0, &smallRecord),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(smallAllocator.end_region(&smallAllocator, smallRegion, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(smallAllocator.seal(&smallAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        smallSnapshot = small.takeSnapshot();
        ASSERT_NE(smallSnapshot, nullptr);
    }

    EXPECT_EQ(budget->usedBytes(), 16u);
    budget->commit();
    HostDynamicTape blocked(16, policy);
    VernonAdTapeAllocator &blockedAllocator = blocked.descriptor();
    VernonAdRegionHandle blockedRegion = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle blockedRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(blockedAllocator.begin_region(&blockedAllocator, VERNON_AD_INVALID_REGION_HANDLE, &blockedRegion),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(blockedAllocator.reserve_record(&blockedAllocator, blockedRegion, 1, 1, 0, &blockedRecord),
              VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);

    largeSnapshot.reset();
    smallSnapshot.reset();
    budget.reset();
    ASSERT_EQ(blockedAllocator.reset(&blockedAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(blockedAllocator.begin_region(&blockedAllocator, VERNON_AD_INVALID_REGION_HANDLE, &blockedRegion),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(blockedAllocator.reserve_record(&blockedAllocator, blockedRegion, 1, 1, 0, &blockedRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiffTapeAllocator, DispatchReservationsAtomicallyPartitionContextBudget) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>(16, 16);
    std::shared_ptr<HostTapeDispatchBudget> first = HostTapeDispatchBudget::reserve(policy, 10);
    std::shared_ptr<HostTapeDispatchBudget> second = HostTapeDispatchBudget::reserve(policy, 10);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->capacity(), 10u);
    EXPECT_EQ(second->capacity(), 6u);
    EXPECT_EQ(HostTapeDispatchBudget::reserve(policy, 1), nullptr);

    first->commit();
    std::shared_ptr<HostTapeDispatchBudget> replacement = HostTapeDispatchBudget::reserve(policy, 10);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->capacity(), 10u);
}

TEST(RuntimeAutodiffTapeAllocator, RejectsCopiedDescriptorWithoutTrustingUserData) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator copied = storage.descriptor();
    copied.user_data = reinterpret_cast<void *>(uintptr_t{1});
    EXPECT_EQ(copied.reset(&copied), VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    EXPECT_EQ(copied.status, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);

    VernonAdTapeAllocator &allocator = storage.descriptor();
    allocator.user_data = reinterpret_cast<void *>(uintptr_t{1});
    EXPECT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiffTapeAllocator, AllowsSequentialThreadMigration) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator &allocator = storage.descriptor();
    VernonAdTapeAllocatorStatus status = VERNON_AD_TAPE_ALLOCATOR_OK;
    std::thread other([&] { status = allocator.reset(&allocator); });
    other.join();
    EXPECT_EQ(status, VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(allocator.reset(&allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiffTapeAllocator, SerializesConcurrentMigratedCalls) {
    vernon::runtime::ad::HostDynamicTape storage;
    VernonAdTapeAllocator &allocator = storage.descriptor();
    std::array<VernonAdTapeAllocatorStatus, 8> statuses{};
    std::array<std::thread, 8> callers;

    for (size_t index = 0; index < callers.size(); ++index) {
        callers[index] = std::thread([&, index] {
            statuses[index] = VERNON_AD_TAPE_ALLOCATOR_OK;
            for (size_t iteration = 0; iteration < 1000; ++iteration)
                if ((statuses[index] = allocator.reset(&allocator)) != VERNON_AD_TAPE_ALLOCATOR_OK)
                    return;
        });
    }
    for (std::thread &caller : callers)
        caller.join();
    for (VernonAdTapeAllocatorStatus status : statuses)
        EXPECT_EQ(status, VERNON_AD_TAPE_ALLOCATOR_OK);
}

TEST(RuntimeAutodiff, CpuTapePolicyIsLazyForOrdinaryRuntimeContexts) {
    VernonRuntimeContext context;
    EXPECT_EQ(context.cpuTapePolicy, nullptr);
}

TEST(RuntimeAutodiff, ReplacesStaleInvocationDiagnosticAtPublicBoundary) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_CPU;
    vernon::runtime::invocationDiagnostic(context) = "stale autodiff diagnostic";
    constexpr char invalidBundle[] = "{";
    EXPECT_EQ(vernonRuntimeLoadPipelineBundleWithOptions(&context, invalidBundle, sizeof(invalidBundle) - 1, nullptr),
              nullptr);
    const VernonStringView error = vernonRuntimeGetLastError(&context);
    const std::string message(error.data, error.size);
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message, "stale autodiff diagnostic");
}

} // namespace
