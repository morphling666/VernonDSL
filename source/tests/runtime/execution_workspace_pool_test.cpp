#include "runtime/autodiff/execution_workspace_pool.h"

#include <gtest/gtest.h>

#include <memory>

namespace {

using vernon::runtime::ad::CanonicalProgramExecution;
using vernon::runtime::ad::CanonicalProgramExecutionWorkspace;
using vernon::runtime::ad::ExecutionWorkspacePool;

class CountingExecution final : public CanonicalProgramExecution {
public:
    const vernon::runtime::ad::Signature &signature() const override { return signature_; }
    std::unique_ptr<CanonicalProgramExecutionWorkspace> createWorkspace() const override {
        ++created;
        return std::make_unique<CanonicalProgramExecutionWorkspace>();
    }
    VernonStatus forward(const vernon::runtime::ad::ForwardExecutionTarget &,
                         std::unique_ptr<vernon::runtime::ad::PullbackExecution> &) override {
        return VERNON_STATUS_INTERNAL_ERROR;
    }

    mutable int created{};

private:
    vernon::runtime::ad::Signature signature_;
};

TEST(ExecutionWorkspacePool, LeasedWorkspaceReturnsOnlyAfterItsLastHolder) {
    const auto execution = std::make_shared<CountingExecution>();
    const auto pool = std::make_shared<ExecutionWorkspacePool>(execution);
    EXPECT_EQ(execution->created, 1);

    std::shared_ptr<CanonicalProgramExecutionWorkspace> invocation = pool->acquire();
    ASSERT_NE(invocation, nullptr);
    CanonicalProgramExecutionWorkspace *const first = invocation.get();
    std::shared_ptr<const void> submission = invocation;
    invocation.reset();

    const std::shared_ptr<CanonicalProgramExecutionWorkspace> concurrent = pool->acquire();
    ASSERT_NE(concurrent, nullptr);
    EXPECT_NE(concurrent.get(), first);
    EXPECT_EQ(execution->created, 2);

    submission.reset();
    const std::shared_ptr<CanonicalProgramExecutionWorkspace> reused = pool->acquire();
    EXPECT_EQ(reused.get(), first);
    EXPECT_EQ(execution->created, 2);
}

TEST(ExecutionWorkspacePool, LeasesOutliveThePoolOwner) {
    const auto execution = std::make_shared<CountingExecution>();
    auto pool = std::make_shared<ExecutionWorkspacePool>(execution);
    std::shared_ptr<CanonicalProgramExecutionWorkspace> lease = pool->acquire();
    ASSERT_NE(lease, nullptr);
    const std::weak_ptr<ExecutionWorkspacePool> observed = pool;
    pool.reset();
    EXPECT_FALSE(observed.expired());
    lease.reset();
    EXPECT_TRUE(observed.expired());
}

TEST(ExecutionWorkspacePool, PoolWithoutAnExecutionLeasesNothing) {
    const auto pool = std::make_shared<ExecutionWorkspacePool>(nullptr);
    EXPECT_EQ(pool->acquire(), nullptr);
}

} // namespace
