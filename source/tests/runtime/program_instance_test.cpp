#include "runtime/program_instance.h"

#include <gtest/gtest.h>

#include <memory>

namespace {

using vernon::runtime::program::ProgramInstance;

TEST(ProgramInstance, CommitsAndReusesPreparedBindings) {
    int executable = 0;
    ProgramInstance instance(&executable);
    const auto payload = std::make_shared<int>(42);

    auto first = instance.beginInvocation();
    EXPECT_EQ(first->find(3, "value-v1"), nullptr);
    first->stage(3, "value-v1", payload, 16, 1);
    const auto firstSnapshot = first->commit();
    ASSERT_NE(firstSnapshot->find(3), nullptr);
    EXPECT_EQ(std::static_pointer_cast<int>(*firstSnapshot->find(3)), payload);

    auto second = instance.beginInvocation();
    const std::shared_ptr<void> *reused = second->find(3, "value-v1");
    ASSERT_NE(reused, nullptr);
    EXPECT_EQ(std::static_pointer_cast<int>(*reused), payload);
    second->commit();

    const auto telemetry = instance.telemetry();
    EXPECT_EQ(telemetry.prepareCount, 1u);
    EXPECT_EQ(telemetry.reuseCount, 1u);
    EXPECT_EQ(telemetry.uploadBytes, 16u);
    EXPECT_EQ(telemetry.uploadRanges, 1u);
}

TEST(ProgramInstance, RollbackDoesNotPublishStagedBindings) {
    int executable = 0;
    ProgramInstance instance(&executable);

    auto failed = instance.beginInvocation();
    failed->stage(1, "failed", std::make_shared<int>(1), 8, 1);
    failed->rollback();

    auto next = instance.beginInvocation();
    EXPECT_EQ(next->find(1, "failed"), nullptr);
    next->commit();
    EXPECT_EQ(instance.telemetry().rollbackCount, 1u);
    EXPECT_EQ(instance.telemetry().prepareCount, 0u);
}

TEST(ProgramInstance, ConcurrentInvocationsRetainIndependentSnapshots) {
    int executable = 0;
    ProgramInstance instance(&executable);
    auto first = instance.beginInvocation();
    auto second = instance.beginInvocation();
    const auto firstPayload = std::make_shared<int>(1);
    const auto secondPayload = std::make_shared<int>(2);

    first->stage(0, "first", firstPayload, 0, 0);
    second->stage(0, "second", secondPayload, 0, 0);
    const auto firstSnapshot = first->commit();
    const auto secondSnapshot = second->commit();

    EXPECT_EQ(std::static_pointer_cast<int>(*firstSnapshot->find(0)), firstPayload);
    EXPECT_EQ(std::static_pointer_cast<int>(*secondSnapshot->find(0)), secondPayload);
}

TEST(ProgramInstance, ClearInvalidatesOutstandingTransactions) {
    int executable = 0;
    ProgramInstance instance(&executable);
    auto stale = instance.beginInvocation();
    instance.clear();
    auto current = instance.beginInvocation();
    current->commit();

    EXPECT_THROW(stale->commit(), std::runtime_error);
}

} // namespace
