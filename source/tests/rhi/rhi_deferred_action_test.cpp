#include "rhi/rhi_deferred_action.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

struct CallbackState {
    std::vector<std::uint64_t> calls;
    std::vector<std::uint64_t> failures;
};

void recordCallback(void *context, std::uint64_t object) {
    auto &state = *static_cast<CallbackState *>(context);
    state.calls.push_back(object);
    for (std::uint64_t failure : state.failures)
        if (failure == object)
            throw std::runtime_error("injected deferred callback failure");
}

struct RetainedResource {
    std::vector<std::uint64_t> *calls{};
    std::uint64_t object{};
    bool releaseFails{};
    bool held{true};

    [[nodiscard]] bool active() const noexcept { return held; }

    [[nodiscard]] vernon::Result<void, vernon::RhiError> release() noexcept {
        calls->push_back(object);
        if (releaseFails)
            return vernon::Result<void, vernon::RhiError>{vernon::err(
                vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure, {"release_test_resource", object, 0}})};
        held = false;
        return vernon::Result<void, vernon::RhiError>{vernon::ok()};
    }
};

TEST(RhiDeferredActionTest, ThrowBecomesStableRhiErrorWithoutUnwinding) {
    CallbackState state{{}, {7}};
    std::vector<vernon::rhi::DeferredAction> actions{{&state, 7, recordCallback}};

    static_assert(noexcept(vernon::rhi::drainDeferredActionsReverse(actions, "strict_cleanup")));
    EXPECT_NO_THROW({
        auto result = vernon::rhi::drainDeferredActionsReverse(actions, "strict_cleanup");
        ASSERT_TRUE(result.isErr());
        EXPECT_EQ(result.error().code, vernon::RhiErrorCode::BackendFailure);
        EXPECT_STREQ(result.error().context.operation, "strict_cleanup");
        EXPECT_EQ(result.error().context.value, 7u);
        EXPECT_EQ(result.error().context.detail, 0u);
    });
    ASSERT_EQ(actions.size(), 1u);
    EXPECT_EQ(actions.front().object, 7u);
    state.failures.clear();
    EXPECT_TRUE(vernon::rhi::drainDeferredActionsReverse(actions, "strict_cleanup").isOk());
    EXPECT_TRUE(actions.empty());
    EXPECT_EQ(state.calls, (std::vector<std::uint64_t>{7, 7}));
}

TEST(RhiDeferredActionTest, MultipleCleanupsContinueInReverseOrderAfterThrows) {
    CallbackState state{{}, {3, 2}};
    std::vector<vernon::rhi::DeferredAction> actions{
        {&state, 1, recordCallback}, {&state, 2, recordCallback}, {&state, 3, recordCallback}};

    auto result = vernon::rhi::drainDeferredActionsReverse(actions, "completion_cleanup_callback");

    ASSERT_TRUE(result.isErr());
    EXPECT_EQ(state.calls, (std::vector<std::uint64_t>{3, 2, 1}));
    EXPECT_EQ(result.error().code, vernon::RhiErrorCode::BackendFailure);
    EXPECT_STREQ(result.error().context.operation, "completion_cleanup_callback");
    EXPECT_EQ(result.error().context.value, 3u);
    EXPECT_EQ(result.error().context.detail, 0u);
    ASSERT_EQ(actions.size(), 2u);
    EXPECT_EQ(actions[0].object, 2u);
    EXPECT_EQ(actions[1].object, 3u);
    state.failures.clear();
    EXPECT_TRUE(vernon::rhi::drainDeferredActionsReverse(actions, "completion_cleanup_callback").isOk());
    EXPECT_TRUE(actions.empty());
    EXPECT_EQ(state.calls, (std::vector<std::uint64_t>{3, 2, 1, 3, 2}));
}

TEST(RhiDeferredActionTest, RollbackFailurePrecedesCleanupFailureAndBothSequencesDrain) {
    CallbackState state{{}, {4, 2}};
    std::vector<vernon::rhi::DeferredAction> rollbacks{{&state, 1, recordCallback}, {&state, 2, recordCallback}};
    std::vector<vernon::rhi::DeferredAction> cleanups{{&state, 3, recordCallback}, {&state, 4, recordCallback}};

    auto result = vernon::rhi::drainDeferredRollbackAndCleanup(rollbacks, cleanups);
    ASSERT_TRUE(result.isErr());
    EXPECT_EQ(state.calls, (std::vector<std::uint64_t>{2, 1, 4, 3}));
    EXPECT_EQ(result.error().code, vernon::RhiErrorCode::BackendFailure);
    EXPECT_STREQ(result.error().context.operation, "command_rollback_callback");
    EXPECT_EQ(result.error().context.value, 2u);
    EXPECT_EQ(result.error().context.detail, 0u);
    ASSERT_EQ(rollbacks.size(), 1u);
    EXPECT_EQ(rollbacks.front().object, 2u);
    ASSERT_EQ(cleanups.size(), 1u);
    EXPECT_EQ(cleanups.front().object, 4u);
    state.failures.clear();
    EXPECT_TRUE(vernon::rhi::drainDeferredRollbackAndCleanup(rollbacks, cleanups).isOk());
    EXPECT_TRUE(rollbacks.empty());
    EXPECT_TRUE(cleanups.empty());
    EXPECT_EQ(state.calls, (std::vector<std::uint64_t>{2, 1, 4, 3, 2, 4}));
}

TEST(RhiDeferredActionTest, RetainedResourceFailuresDoNotStopLaterReleasesAndRemainRetryable) {
    std::vector<std::uint64_t> calls;
    calls.reserve(6);
    std::vector<RetainedResource> resources{
        {&calls, 1, false}, {&calls, 2, true}, {&calls, 3, false}, {&calls, 4, true}};

    auto result = vernon::rhi::releaseRetainedResourcesReverse(resources);

    ASSERT_TRUE(result.isErr());
    EXPECT_EQ(calls, (std::vector<std::uint64_t>{4, 3, 2, 1}));
    EXPECT_EQ(result.error().code, vernon::RhiErrorCode::LifecycleFailure);
    EXPECT_STREQ(result.error().context.operation, "release_test_resource");
    EXPECT_EQ(result.error().context.value, 4u);
    EXPECT_EQ(result.error().context.detail, 0u);
    EXPECT_FALSE(resources[0].active());
    EXPECT_TRUE(resources[1].active());
    EXPECT_FALSE(resources[2].active());
    EXPECT_TRUE(resources[3].active());

    resources[1].releaseFails = false;
    resources[3].releaseFails = false;
    ASSERT_TRUE(vernon::rhi::releaseRetainedResourcesReverse(resources).isOk());
    EXPECT_EQ(calls, (std::vector<std::uint64_t>{4, 3, 2, 1, 4, 2}));
}

} // namespace
