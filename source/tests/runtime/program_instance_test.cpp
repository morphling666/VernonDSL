#include "runtime/program_instance.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>

namespace {

using vernon::runtime::program::ProgramInstance;

TEST(ProgramInstance, CommitsAndReusesPreparedBindings) {
    int executable = 0;
    ProgramInstance instance(&executable);
    const auto payload = std::make_shared<int>(42);

    auto firstResult = instance.beginInvocation();
    ASSERT_TRUE(firstResult.isOk());
    auto first = std::move(firstResult).value();
    auto firstMatch = first->matches(3, "value-v1");
    ASSERT_TRUE(firstMatch.isOk());
    EXPECT_FALSE(firstMatch.value());
    EXPECT_TRUE(first->stage(3, "value-v1", payload, 16, 1).isOk());
    auto firstCommit = first->commit();
    ASSERT_TRUE(firstCommit.isOk());
    const auto firstSnapshot = std::move(firstCommit).value();
    ASSERT_NE(firstSnapshot->find(3), nullptr);
    EXPECT_EQ(std::static_pointer_cast<int>(*firstSnapshot->find(3)), payload);

    auto secondResult = instance.beginInvocation();
    ASSERT_TRUE(secondResult.isOk());
    auto second = std::move(secondResult).value();
    auto reused = second->matches(3, "value-v1");
    ASSERT_TRUE(reused.isOk());
    EXPECT_TRUE(reused.value());
    EXPECT_TRUE(second->observeReuses(1).isOk());
    auto secondCommit = second->commit();
    ASSERT_TRUE(secondCommit.isOk());
    EXPECT_EQ(std::move(secondCommit).value().get(), firstSnapshot.get());

    const auto telemetry = instance.telemetry();
    EXPECT_EQ(telemetry.prepareCount, 1u);
    EXPECT_EQ(telemetry.reuseCount, 1u);
    EXPECT_EQ(telemetry.uploadBytes, 16u);
    EXPECT_EQ(telemetry.uploadRanges, 1u);
}

TEST(ProgramInstance, RollbackDoesNotPublishStagedBindings) {
    int executable = 0;
    ProgramInstance instance(&executable);

    auto failedResult = instance.beginInvocation();
    ASSERT_TRUE(failedResult.isOk());
    auto failed = std::move(failedResult).value();
    EXPECT_TRUE(failed->stage(1, "failed", std::make_shared<int>(1), 8, 1).isOk());
    EXPECT_TRUE(failed->freeze().isOk());
    failed->rollback();

    auto nextResult = instance.beginInvocation();
    ASSERT_TRUE(nextResult.isOk());
    auto next = std::move(nextResult).value();
    auto matched = next->matches(1, "failed");
    ASSERT_TRUE(matched.isOk());
    EXPECT_FALSE(matched.value());
    EXPECT_TRUE(next->commit().isOk());
    EXPECT_EQ(instance.telemetry().rollbackCount, 1u);
    EXPECT_EQ(instance.telemetry().prepareCount, 0u);
}

TEST(ProgramInstance, CommitPublishesTheFrozenInvocationSnapshotWithoutRebuildingIt) {
    int executable = 0;
    ProgramInstance instance(&executable);
    auto invocationResult = instance.beginInvocation();
    ASSERT_TRUE(invocationResult.isOk());
    auto invocation = std::move(invocationResult).value();
    EXPECT_TRUE(invocation->stage(1, "frozen", std::make_shared<int>(7), 0, 0).isOk());

    auto frozenResult = invocation->freeze();
    ASSERT_TRUE(frozenResult.isOk());
    const auto frozen = std::move(frozenResult).value();
    auto committedResult = invocation->commit();
    ASSERT_TRUE(committedResult.isOk());
    const auto committed = std::move(committedResult).value();

    EXPECT_EQ(frozen.get(), committed.get());
    ASSERT_NE(committed->find(1), nullptr);
    EXPECT_EQ(*std::static_pointer_cast<int>(*committed->find(1)), 7);
}

TEST(ProgramInstance, ConcurrentInvocationsRetainIndependentSnapshots) {
    int executable = 0;
    ProgramInstance instance(&executable);
    auto firstResult = instance.beginInvocation();
    auto secondResult = instance.beginInvocation();
    ASSERT_TRUE(firstResult.isOk());
    ASSERT_TRUE(secondResult.isOk());
    auto first = std::move(firstResult).value();
    auto second = std::move(secondResult).value();
    const auto firstPayload = std::make_shared<int>(1);
    const auto secondPayload = std::make_shared<int>(2);

    EXPECT_TRUE(first->stage(0, "first", firstPayload, 0, 0).isOk());
    EXPECT_TRUE(second->stage(0, "second", secondPayload, 0, 0).isOk());
    std::atomic<uint32_t> ready{};
    std::atomic<bool> start{};
    bool firstCommitted = false;
    bool secondCommitted = false;
    std::shared_ptr<const vernon::runtime::program::InvocationSnapshot> firstSnapshot;
    std::shared_ptr<const vernon::runtime::program::InvocationSnapshot> secondSnapshot;
    std::thread firstThread([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        auto result = first->commit();
        firstCommitted = result.isOk();
        if (firstCommitted)
            firstSnapshot = std::move(result).value();
    });
    std::thread secondThread([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        auto result = second->commit();
        secondCommitted = result.isOk();
        if (secondCommitted)
            secondSnapshot = std::move(result).value();
    });
    while (ready.load(std::memory_order_acquire) != 2)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();
    ASSERT_TRUE(firstCommitted);
    ASSERT_TRUE(secondCommitted);

    EXPECT_EQ(std::static_pointer_cast<int>(*firstSnapshot->find(0)), firstPayload);
    EXPECT_EQ(std::static_pointer_cast<int>(*secondSnapshot->find(0)), secondPayload);
}

TEST(ProgramInstance, ConcurrentDisjointCommitsMergeIntoPersistentState) {
    int executable = 0;
    ProgramInstance instance(&executable);
    auto firstResult = instance.beginInvocation();
    auto secondResult = instance.beginInvocation();
    ASSERT_TRUE(firstResult.isOk());
    ASSERT_TRUE(secondResult.isOk());
    auto first = std::move(firstResult).value();
    auto second = std::move(secondResult).value();

    EXPECT_TRUE(first->stage(1, "first", std::make_shared<int>(1), 0, 0).isOk());
    EXPECT_TRUE(second->stage(2, "second", std::make_shared<int>(2), 0, 0).isOk());
    std::atomic<uint32_t> ready{};
    std::atomic<bool> start{};
    bool firstCommitted = false;
    bool secondCommitted = false;
    std::thread firstThread([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        firstCommitted = first->commit().isOk();
    });
    std::thread secondThread([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        secondCommitted = second->commit().isOk();
    });
    while (ready.load(std::memory_order_acquire) != 2)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    firstThread.join();
    secondThread.join();
    ASSERT_TRUE(firstCommitted);
    ASSERT_TRUE(secondCommitted);

    auto mergedResult = instance.beginInvocation();
    ASSERT_TRUE(mergedResult.isOk());
    auto merged = std::move(mergedResult).value();
    auto firstMatch = merged->matches(1, "first");
    auto secondMatch = merged->matches(2, "second");
    ASSERT_TRUE(firstMatch.isOk());
    ASSERT_TRUE(secondMatch.isOk());
    EXPECT_TRUE(firstMatch.value());
    EXPECT_TRUE(secondMatch.value());
    EXPECT_TRUE(merged->commit().isOk());
    EXPECT_EQ(instance.telemetry().prepareCount, 2u);
}

TEST(ProgramInstance, ClearInvalidatesOutstandingTransactions) {
    int executable = 0;
    ProgramInstance instance(&executable);
    auto staleResult = instance.beginInvocation();
    ASSERT_TRUE(staleResult.isOk());
    auto stale = std::move(staleResult).value();
    instance.clear();
    auto currentResult = instance.beginInvocation();
    ASSERT_TRUE(currentResult.isOk());
    auto current = std::move(currentResult).value();
    EXPECT_TRUE(current->commit().isOk());

    auto staleCommit = stale->commit();
    ASSERT_TRUE(staleCommit.isErr());
    EXPECT_EQ(staleCommit.error(), vernon::runtime::program::BindingError::StaleInstance);
}

} // namespace
