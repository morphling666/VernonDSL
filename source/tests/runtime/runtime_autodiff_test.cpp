#include "runtime/autodiff/host_effect_transaction.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/runtime_autodiff_policy.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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

    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    auto budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    auto storage = HostStaticTapeBatch::create(1, sizeof(uint32_t), 1024, policy, budget);
    ASSERT_NE(storage, nullptr);
    VernonAdTapeAllocator incompatible = *storage->descriptor(0);
    --incompatible.struct_size;
    EXPECT_EQ(incompatible.reset(&incompatible), VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
    incompatible = *storage->descriptor(0);
    ++incompatible.abi_version;
    EXPECT_EQ(incompatible.reset(&incompatible), VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI);
}

TEST(RuntimeAutodiffTapeAllocator, HasFrozenImmutablePageLayout) {
    using namespace vernon::runtime::ad;
    EXPECT_EQ(kHostTapePageLayoutVersion, 1u);
    EXPECT_EQ(kHostTapeSnapshotRegionResidentBytes, 4 * sizeof(size_t));
    EXPECT_EQ(kHostTapeSnapshotRecordResidentBytes, 4 * sizeof(size_t));
    EXPECT_EQ(offsetof(detail::HostTapeSnapshotRegion, exitKind), 3 * sizeof(size_t));
    EXPECT_EQ(offsetof(detail::HostTapeSnapshotRecord, childCount), 3 * sizeof(size_t));
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

TEST(RuntimeAutodiffTapeAllocator, PreservesNestedRecordsInCompactedDynamicBatch) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    auto budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    HostDynamicTapeBatch storage(1, 4096, policy, budget);
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRegionHandle child = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    VernonAdRecordHandle childRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    VernonAdRecordHandle trailingRootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(storage.beginRegion(0, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.reserveRecord(0, root, sizeof(uint32_t), alignof(uint32_t), 1, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t rootValue = 0x12345678u;
    const uint32_t childValue = 0xabcdef01u;
    const uint32_t trailingRootValue = 0x87654321u;
    ASSERT_EQ(storage.writeLeaf(0, rootRecord, 0, &rootValue, sizeof(rootValue)), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.beginRegion(0, root, &child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.setChild(0, rootRecord, 0, child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.reserveRecord(0, child, sizeof(uint32_t), alignof(uint32_t), 0, &childRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.writeLeaf(0, childRecord, 0, &childValue, sizeof(childValue)), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.endRegion(0, child, 1, 2), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.reserveRecord(0, root, sizeof(uint32_t), alignof(uint32_t), 0, &trailingRootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.writeLeaf(0, trailingRootRecord, 0, &trailingRootValue, sizeof(trailingRootValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.endRegion(0, root, 1, 3), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.seal(0), VERNON_AD_TAPE_ALLOCATOR_OK);
    const size_t constructionBytes = storage.residentBytes();
    ASSERT_TRUE(storage.compact());
    EXPECT_EQ(policy->usage().currentBytes, storage.residentBytes());
    EXPECT_GT(policy->usage().peakBytes, std::max(constructionBytes, storage.residentBytes()));

    root = storage.rootRegion(0);
    uint32_t read = 0;
    ASSERT_EQ(storage.readLeaf(0, root, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, rootValue);
    VernonAdRegionHandle readChild = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(storage.readChild(0, root, 0, 0, &readChild), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.readLeaf(0, readChild, 0, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, childValue);
    ASSERT_EQ(storage.readLeaf(0, root, 1, 0, &read, sizeof(read)), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(read, trailingRootValue);
    size_t count = 0;
    uint32_t exitKind = 0;
    ASSERT_EQ(storage.readCount(0, root, &count), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.readExit(0, root, &exitKind), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(exitKind, 3u);
}

TEST(RuntimeAutodiffTapeAllocator, FailedChunkMaterializationDoesNotConsumeRegionHandle) {
    using namespace vernon::runtime::ad;
    constexpr size_t capacity = 2u * 1024u * 1024u;
    auto policy = std::make_shared<HostTapeMemoryPolicy>(capacity, capacity);
    auto budget = HostTapeDispatchBudget::reserve(policy, capacity);
    ASSERT_NE(budget, nullptr);
    HostDynamicTapeBatch storage(1, capacity, policy, budget);
    const size_t remaining = capacity - budget->usedBytes();
    ASSERT_GT(remaining, 0u);
    ASSERT_TRUE(budget->reserveTransientBytes(remaining));
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    EXPECT_EQ(storage.beginRegion(0, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    budget->releaseTransientBytes(remaining);
    ASSERT_EQ(storage.reset(0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(storage.beginRegion(0, VERNON_AD_INVALID_REGION_HANDLE, &region), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(region, 1u);
}

TEST(RuntimeAutodiffTapeAllocator, UsesOneCompactStaticBatchWithoutPerLaneTapes) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    std::shared_ptr<HostTapeDispatchBudget> budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    std::shared_ptr<HostStaticTapeBatch> batch = HostStaticTapeBatch::create(3, 16, 1024, policy, budget);
    ASSERT_NE(batch, nullptr);
    size_t pureStaticBytes = 0;
    ASSERT_TRUE(hostStaticTapeBatchPureStaticBytes(3, 16, pureStaticBytes));
    EXPECT_EQ(batch->residentBytes(), pureStaticBytes);
    EXPECT_EQ(batch->residentBytes(), batch->allocatedBytes());
    EXPECT_EQ(policy->usage().currentBytes, batch->residentBytes());
    EXPECT_EQ(policy->usage().peakBytes, pureStaticBytes);

    for (size_t lane = 0; lane < batch->size(); ++lane) {
        VernonAdTapeAllocator *allocator = batch->descriptor(lane);
        ASSERT_NE(allocator, nullptr);
        VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(allocator->reset(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator->reserve_record(allocator, region, sizeof(uint32_t), alignof(uint32_t), 0, &record),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        const uint32_t value = static_cast<uint32_t>(lane + 11);
        ASSERT_EQ(allocator->write_leaf(allocator, record, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator->end_region(allocator, region, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(allocator->seal(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(batch->rootRegion(lane), region);
        uint32_t restored = 0;
        ASSERT_EQ(allocator->read_leaf(allocator, region, 0, 0, &restored, sizeof(restored)),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(restored, value);
    }
    EXPECT_EQ(batch->logicalBytes(), 3 * sizeof(uint32_t));
    EXPECT_GT(batch->residentBytes(), batch->logicalBytes());
    EXPECT_EQ(batch->residentBytes(), batch->allocatedBytes());
    const size_t constructionBytes = batch->residentBytes();
    ASSERT_TRUE(batch->compact());
    EXPECT_TRUE(batch->isCompacted());
    EXPECT_EQ(batch->size(), 3u);
    EXPECT_EQ(batch->descriptor(0), nullptr);
    EXPECT_LT(batch->residentBytes(), constructionBytes);
    EXPECT_EQ(batch->residentBytes(), 3u * 16u);
    EXPECT_EQ(policy->usage().currentBytes, batch->residentBytes());
    HostStaticTapeBatch::Reader reader;
    ASSERT_TRUE(batch->initializeReader(2, reader));
    uint32_t restored = 0;
    ASSERT_EQ(
        reader.descriptor()->read_leaf(reader.descriptor(), reader.rootRegion(), 0, 0, &restored, sizeof(restored)),
        VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(restored, 13u);

    batch.reset();
    EXPECT_EQ(policy->usage().currentBytes, 0u);
}

TEST(RuntimeAutodiffTapeAllocator, ReusesStaticReplayConstructionStorageAcrossSegments) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    auto budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    auto batch = HostStaticTapeBatch::create(2, 16, 1024, policy, budget);
    ASSERT_NE(batch, nullptr);
    const size_t constructionBytes = batch->allocatedBytes();

    auto recordSegment = [&](uint32_t base) {
        for (size_t lane = 0; lane < batch->size(); ++lane) {
            VernonAdTapeAllocator *allocator = batch->descriptor(lane);
            ASSERT_NE(allocator, nullptr);
            VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
            VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
            ASSERT_EQ(allocator->reset(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
                      VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(allocator->reserve_record(allocator, region, sizeof(uint32_t), alignof(uint32_t), 0, &record),
                      VERNON_AD_TAPE_ALLOCATOR_OK);
            const uint32_t value = base + static_cast<uint32_t>(lane);
            ASSERT_EQ(allocator->write_leaf(allocator, record, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(allocator->end_region(allocator, region, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(allocator->seal(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
        }
        ASSERT_TRUE(batch->compact(true));
        budget->commit();
    };

    recordSegment(10);
    EXPECT_EQ(batch->allocatedBytes(), constructionBytes);
    EXPECT_EQ(policy->usage().currentBytes, constructionBytes);
    HostStaticTapeBatch::Reader firstReader;
    ASSERT_TRUE(batch->initializeReader(1, firstReader));
    uint32_t restored = 0;
    ASSERT_EQ(firstReader.descriptor()->read_leaf(firstReader.descriptor(), firstReader.rootRegion(), 0, 0, &restored,
                                                  sizeof(restored)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(restored, 11u);

    ASSERT_TRUE(batch->markConstructionRecyclable());
    ASSERT_TRUE(batch->resetRecyclableConstruction());
    EXPECT_EQ(batch->allocatedBytes(), constructionBytes);
    recordSegment(20);
    HostStaticTapeBatch::Reader secondReader;
    ASSERT_TRUE(batch->initializeReader(1, secondReader));
    ASSERT_EQ(secondReader.descriptor()->read_leaf(secondReader.descriptor(), secondReader.rootRegion(), 0, 0,
                                                   &restored, sizeof(restored)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(restored, 21u);
    EXPECT_EQ(policy->usage().peakBytes, constructionBytes);
}

TEST(RuntimeAutodiffTapeAllocator, ResetsFailedCompactLaneWithoutLeakingItsBatchCharge) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    std::shared_ptr<HostTapeDispatchBudget> budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    std::shared_ptr<HostStaticTapeBatch> batch = HostStaticTapeBatch::create(1, 8, 1024, policy, budget);
    ASSERT_NE(batch, nullptr);
    const size_t allocated = batch->allocatedBytes();
    VernonAdTapeAllocator *allocator = batch->descriptor(0);
    ASSERT_NE(allocator, nullptr);
    VernonAdRegionHandle region = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator->reset(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &region),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->reserve_record(allocator, region, 8, 4, 0, &record), VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t value = 17;
    EXPECT_EQ(allocator->write_leaf(allocator, record, 6, &value, sizeof(value)),
              VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    EXPECT_EQ(allocator->reset(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(batch->logicalBytes(), 0u);
    EXPECT_EQ(batch->allocatedBytes(), allocated);
    EXPECT_EQ(policy->usage().currentBytes, allocated);
    batch.reset();
    EXPECT_EQ(policy->usage().currentBytes, 0u);
}

TEST(RuntimeAutodiffTapeAllocator, PromotesDynamicLaneIntoSharedImmutableBatch) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    std::shared_ptr<HostTapeDispatchBudget> budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    std::shared_ptr<HostStaticTapeBatch> batch = HostStaticTapeBatch::create(2, 4, 1024, policy, budget);
    ASSERT_NE(batch, nullptr);
    VernonAdTapeAllocator *allocator = batch->descriptor(0);
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRegionHandle child = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    VernonAdRecordHandle childRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator->reset(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->reserve_record(allocator, root, sizeof(uint32_t), alignof(uint32_t), 1, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t rootValue = 23;
    const uint32_t childValue = 29;
    ASSERT_EQ(allocator->write_leaf(allocator, rootRecord, 0, &rootValue, sizeof(rootValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->begin_region(allocator, root, &child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->set_child(allocator, rootRecord, 0, child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->reserve_record(allocator, child, sizeof(uint32_t), alignof(uint32_t), 0, &childRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->write_leaf(allocator, childRecord, 0, &childValue, sizeof(childValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->end_region(allocator, child, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->end_region(allocator, root, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->seal(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);

    VernonAdTapeAllocator *staticAllocator = batch->descriptor(1);
    VernonAdRegionHandle staticRoot = VERNON_AD_INVALID_REGION_HANDLE;
    VernonAdRecordHandle staticRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(staticAllocator->reset(staticAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->begin_region(staticAllocator, VERNON_AD_INVALID_REGION_HANDLE, &staticRoot),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->reserve_record(staticAllocator, staticRoot, sizeof(uint32_t), alignof(uint32_t), 0,
                                              &staticRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    const uint32_t staticValue = 37;
    ASSERT_EQ(staticAllocator->write_leaf(staticAllocator, staticRecord, 0, &staticValue, sizeof(staticValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->end_region(staticAllocator, staticRoot, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->seal(staticAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);

    EXPECT_EQ(batch->logicalBytes(), 3 * sizeof(uint32_t));
    const size_t constructionBytes = batch->constructionBytes();
    ASSERT_TRUE(batch->compact(true));
    budget->commit();
    HostStaticTapeBatch::Reader reader;
    ASSERT_TRUE(batch->initializeReader(0, reader));
    EXPECT_NE(reader.rootRegion(), VERNON_AD_INVALID_REGION_HANDLE);
    VernonAdRegionHandle compactChild = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(reader.descriptor()->read_child(reader.descriptor(), reader.rootRegion(), 0, 0, &compactChild),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    uint32_t restored = 0;
    ASSERT_EQ(reader.descriptor()->read_leaf(reader.descriptor(), compactChild, 0, 0, &restored, sizeof(restored)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(restored, childValue);
    HostStaticTapeBatch::Reader staticReader;
    ASSERT_TRUE(batch->initializeReader(1, staticReader));
    ASSERT_EQ(staticReader.descriptor()->read_leaf(staticReader.descriptor(), staticReader.rootRegion(), 0, 0,
                                                   &restored, sizeof(restored)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(restored, staticValue);
    ASSERT_TRUE(batch->markConstructionRecyclable());
    ASSERT_TRUE(batch->resetRecyclableConstruction());
    EXPECT_EQ(batch->constructionState(), HostStaticTapeBatch::ConstructionState::Constructing);
    EXPECT_GE(batch->constructionBytes(), constructionBytes);
    EXPECT_NE(batch->descriptor(0), nullptr);

    allocator = batch->descriptor(0);
    root = child = VERNON_AD_INVALID_REGION_HANDLE;
    rootRecord = childRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(allocator->reset(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->begin_region(allocator, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->reserve_record(allocator, root, sizeof(uint32_t), alignof(uint32_t), 1, &rootRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->begin_region(allocator, root, &child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->set_child(allocator, rootRecord, 0, child), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->reserve_record(allocator, child, sizeof(uint32_t), alignof(uint32_t), 0, &childRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->write_leaf(allocator, childRecord, 0, &rootValue, sizeof(rootValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->end_region(allocator, child, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->end_region(allocator, root, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(allocator->seal(allocator), VERNON_AD_TAPE_ALLOCATOR_OK);

    staticAllocator = batch->descriptor(1);
    staticRoot = VERNON_AD_INVALID_REGION_HANDLE;
    staticRecord = VERNON_AD_INVALID_RECORD_HANDLE;
    ASSERT_EQ(staticAllocator->reset(staticAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->begin_region(staticAllocator, VERNON_AD_INVALID_REGION_HANDLE, &staticRoot),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->reserve_record(staticAllocator, staticRoot, sizeof(uint32_t), alignof(uint32_t), 0,
                                              &staticRecord),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->write_leaf(staticAllocator, staticRecord, 0, &staticValue, sizeof(staticValue)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->end_region(staticAllocator, staticRoot, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(staticAllocator->seal(staticAllocator), VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_TRUE(batch->compact(true));
    budget->commit();
    HostStaticTapeBatch::Reader recycledReader;
    ASSERT_TRUE(batch->initializeReader(0, recycledReader));
    compactChild = VERNON_AD_INVALID_REGION_HANDLE;
    ASSERT_EQ(recycledReader.descriptor()->read_child(recycledReader.descriptor(), recycledReader.rootRegion(), 0, 0,
                                                      &compactChild),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    ASSERT_EQ(recycledReader.descriptor()->read_leaf(recycledReader.descriptor(), compactChild, 0, 0, &restored,
                                                     sizeof(restored)),
              VERNON_AD_TAPE_ALLOCATOR_OK);
    EXPECT_EQ(restored, rootValue);
}

TEST(RuntimeAutodiffTapeAllocator, SupportsConcurrentDynamicLaneWritersAndReaders) {
    using namespace vernon::runtime::ad;
    constexpr size_t laneCount = 32;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    auto budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    HostDynamicTapeBatch batch(laneCount, 4096, policy, budget);
    std::vector<std::thread> writers;
    for (size_t lane = 0; lane < laneCount; ++lane) {
        writers.emplace_back([&, lane] {
            VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
            VernonAdRegionHandle child = VERNON_AD_INVALID_REGION_HANDLE;
            VernonAdRecordHandle rootRecord = VERNON_AD_INVALID_RECORD_HANDLE;
            VernonAdRecordHandle childRecord = VERNON_AD_INVALID_RECORD_HANDLE;
            ASSERT_EQ(batch.beginRegion(lane, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.reserveRecord(lane, root, sizeof(uint32_t), alignof(uint32_t), 1, &rootRecord),
                      VERNON_AD_TAPE_ALLOCATOR_OK);
            const uint32_t rootValue = static_cast<uint32_t>(lane + 100);
            ASSERT_EQ(batch.writeLeaf(lane, rootRecord, 0, &rootValue, sizeof(rootValue)), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.beginRegion(lane, root, &child), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.setChild(lane, rootRecord, 0, child), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.reserveRecord(lane, child, sizeof(uint32_t), alignof(uint32_t), 0, &childRecord),
                      VERNON_AD_TAPE_ALLOCATOR_OK);
            const uint32_t childValue = static_cast<uint32_t>(lane + 200);
            ASSERT_EQ(batch.writeLeaf(lane, childRecord, 0, &childValue, sizeof(childValue)),
                      VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.endRegion(lane, child, lane + 1, 2), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.endRegion(lane, root, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.seal(lane), VERNON_AD_TAPE_ALLOCATOR_OK);
        });
    }
    for (std::thread &writer : writers)
        writer.join();

    ASSERT_TRUE(batch.compact());
    EXPECT_EQ(batch.logicalBytes(), laneCount * 2 * sizeof(uint32_t));
    EXPECT_EQ(policy->usage().currentBytes, batch.residentBytes());
    for (size_t lane = 0; lane < laneCount; ++lane) {
        const VernonAdRegionHandle root = batch.rootRegion(lane);
        VernonAdRegionHandle child = VERNON_AD_INVALID_REGION_HANDLE;
        ASSERT_EQ(batch.readChild(lane, root, 0, 0, &child), VERNON_AD_TAPE_ALLOCATOR_OK);
        uint32_t restored = 0;
        ASSERT_EQ(batch.readLeaf(lane, child, 0, 0, &restored, sizeof(restored)), VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(restored, lane + 200);
        size_t count = 0;
        uint32_t exitKind = 0;
        ASSERT_EQ(batch.readCount(lane, child, &count), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(batch.readExit(lane, child, &exitKind), VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(count, lane + 1);
        EXPECT_EQ(exitKind, 2u);
    }
}

TEST(RuntimeAutodiffTapeAllocator, AppendsAcrossConcurrentArenaChunks) {
    using namespace vernon::runtime::ad;
    constexpr size_t laneCount = 16;
    constexpr size_t recordsPerLane = 64;
    constexpr size_t payloadSize = 128;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    auto budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    HostDynamicTapeBatch batch(laneCount, 1u << 20, policy, budget);
    std::vector<std::thread> writers;
    for (size_t lane = 0; lane < laneCount; ++lane) {
        writers.emplace_back([&, lane] {
            VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
            ASSERT_EQ(batch.beginRegion(lane, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
            std::array<std::byte, payloadSize> payload{};
            for (size_t ordinal = 0; ordinal < recordsPerLane; ++ordinal) {
                VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
                ASSERT_EQ(batch.reserveRecord(lane, root, payload.size(), alignof(uint64_t), 0, &record),
                          VERNON_AD_TAPE_ALLOCATOR_OK);
                const uint64_t marker = lane * recordsPerLane + ordinal;
                std::memcpy(payload.data(), &marker, sizeof(marker));
                ASSERT_EQ(batch.writeLeaf(lane, record, 0, payload.data(), payload.size()),
                          VERNON_AD_TAPE_ALLOCATOR_OK);
            }
            ASSERT_EQ(batch.endRegion(lane, root, recordsPerLane, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
            ASSERT_EQ(batch.seal(lane), VERNON_AD_TAPE_ALLOCATOR_OK);
        });
    }
    for (std::thread &writer : writers)
        writer.join();

    ASSERT_TRUE(batch.compact());
    EXPECT_EQ(batch.logicalBytes(), laneCount * recordsPerLane * payloadSize);
    for (size_t lane = 0; lane < laneCount; ++lane) {
        uint64_t marker = 0;
        ASSERT_EQ(batch.readLeaf(lane, batch.rootRegion(lane), recordsPerLane - 1, 0, &marker, sizeof(marker)),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(marker, lane * recordsPerLane + recordsPerLane - 1);
    }
}

TEST(RuntimeAutodiffTapeAllocator, DynamicBatchRollbackInvalidatesStaleHandlesAndRetainsArenaCapacity) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>();
    auto budget = HostTapeDispatchBudget::reserve(policy, policy->contextLimit());
    ASSERT_NE(budget, nullptr);
    {
        HostDynamicTapeBatch batch(1, 1024, policy, budget);
        VernonAdRegionHandle staleRegion = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle staleRecord = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(batch.beginRegion(0, VERNON_AD_INVALID_REGION_HANDLE, &staleRegion), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(batch.reserveRecord(0, staleRegion, sizeof(uint32_t), alignof(uint32_t), 0, &staleRecord),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_GT(policy->usage().currentBytes, 0u);
        const size_t retainedArenaBytes = batch.residentBytes();
        ASSERT_EQ(batch.reset(0), VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(policy->usage().currentBytes, retainedArenaBytes);
        const uint32_t staleValue = 7;
        EXPECT_EQ(batch.writeLeaf(0, staleRecord, 0, &staleValue, sizeof(staleValue)),
                  VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
        ASSERT_EQ(batch.reset(0), VERNON_AD_TAPE_ALLOCATOR_OK);

        VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
        VernonAdRecordHandle record = VERNON_AD_INVALID_RECORD_HANDLE;
        ASSERT_EQ(batch.beginRegion(0, VERNON_AD_INVALID_REGION_HANDLE, &root), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(batch.reserveRecord(0, root, sizeof(uint32_t), alignof(uint32_t), 0, &record),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        const uint32_t value = 31;
        ASSERT_EQ(batch.writeLeaf(0, record, 0, &value, sizeof(value)), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(batch.endRegion(0, root, 1, 0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_EQ(batch.seal(0), VERNON_AD_TAPE_ALLOCATOR_OK);
        ASSERT_TRUE(batch.compact());
        size_t staleCount = 0;
        EXPECT_EQ(batch.readCount(0, staleRegion, &staleCount), VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
        uint32_t restored = 0;
        ASSERT_EQ(batch.readLeaf(0, batch.rootRegion(0), 0, 0, &restored, sizeof(restored)),
                  VERNON_AD_TAPE_ALLOCATOR_OK);
        EXPECT_EQ(restored, value);
        EXPECT_EQ(policy->usage().currentBytes, batch.allocatedBytes());
    }
    EXPECT_EQ(policy->usage().currentBytes, 0u);
}

TEST(RuntimeAutodiffTapeAllocator, DispatchReservationsDoNotPessimisticallyChargeContextBudget) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<HostTapeMemoryPolicy>(16, 16);
    std::shared_ptr<HostTapeDispatchBudget> first = HostTapeDispatchBudget::reserve(policy, 10);
    std::shared_ptr<HostTapeDispatchBudget> second = HostTapeDispatchBudget::reserve(policy, 10);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->capacity(), 10u);
    EXPECT_EQ(second->capacity(), 10u);
    EXPECT_NE(HostTapeDispatchBudget::reserve(policy, 1), nullptr);

    first->commit();
    std::shared_ptr<HostTapeDispatchBudget> replacement = HostTapeDispatchBudget::reserve(policy, 10);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->capacity(), 10u);
}

TEST(RuntimeAutodiffTapeAllocator, BackendNeutralReservationsChargeAndReleaseContextBudget) {
    using namespace vernon::runtime::ad;
    auto policy = std::make_shared<AutodiffMemoryPolicy>(64, 64);
    std::shared_ptr<AutodiffMemoryReservation> retained = AutodiffMemoryReservation::reserve(policy, 24);
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(retained->bytes(), 24u);
    EXPECT_EQ(policy->usage().currentBytes, 24u);
    EXPECT_FALSE(retained->shrink(25));
    EXPECT_TRUE(retained->shrink(16));
    EXPECT_EQ(retained->bytes(), 16u);
    EXPECT_EQ(policy->usage().currentBytes, 16u);
    EXPECT_EQ(AutodiffMemoryReservation::reserve(policy, 49), nullptr);
    {
        std::shared_ptr<AutodiffMemoryReservation> temporary = AutodiffMemoryReservation::reserve(policy, 48);
        ASSERT_NE(temporary, nullptr);
        EXPECT_EQ(policy->usage().currentBytes, 64u);
    }
    EXPECT_EQ(policy->usage().currentBytes, 16u);
    retained.reset();
    EXPECT_EQ(policy->usage().currentBytes, 0u);
    EXPECT_EQ(policy->usage().peakBytes, 64u);
}

TEST(RuntimeAutodiff, ParsesBackendNeutralPlanningPolicies) {
    using namespace vernon::runtime::ad;
    PlanningPolicy policy{};
    EXPECT_TRUE(parsePlanningPolicy("min_memory", policy));
    EXPECT_EQ(policy, PlanningPolicy::MinMemory);
    EXPECT_TRUE(parsePlanningPolicy("balanced", policy));
    EXPECT_EQ(policy, PlanningPolicy::Balanced);
    EXPECT_TRUE(parsePlanningPolicy("min_runtime", policy));
    EXPECT_EQ(policy, PlanningPolicy::MinRuntime);
    EXPECT_FALSE(parsePlanningPolicy("heuristic", policy));
}

TEST(RuntimeAutodiff, AutodiffMemoryPolicyIsLazyForOrdinaryRuntimeContexts) {
    VernonRuntimeContext context;
    EXPECT_EQ(context.autodiffMemoryPolicy, nullptr);
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
