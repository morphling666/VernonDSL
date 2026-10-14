#ifndef VERNON_RUNTIME_AUTODIFF_EXECUTION_WORKSPACE_POOL_H
#define VERNON_RUNTIME_AUTODIFF_EXECUTION_WORKSPACE_POOL_H

#include "runtime_autodiff_internal.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace vernon::runtime::ad {

// The execution workspaces of one Program instance; the pool must be owned through std::shared_ptr. A leased
// workspace returns to the pool only when its invocation and the submission that reads its storage have both
// released it, because preparing it for another invocation frees that storage.
class ExecutionWorkspacePool : public std::enable_shared_from_this<ExecutionWorkspacePool> {
public:
    explicit ExecutionWorkspacePool(std::shared_ptr<CanonicalProgramExecution> execution)
        : execution_(std::move(execution)) {
        if (execution_)
            if (std::unique_ptr<CanonicalProgramExecutionWorkspace> workspace = execution_->createWorkspace())
                idle_.push_back(std::move(workspace));
    }

    std::shared_ptr<CanonicalProgramExecutionWorkspace> acquire() noexcept {
        try {
            std::unique_ptr<CanonicalProgramExecutionWorkspace> workspace;
            {
                std::lock_guard lock(mutex_);
                if (!idle_.empty()) {
                    workspace = std::move(idle_.back());
                    idle_.pop_back();
                }
            }
            if (!workspace && execution_)
                workspace = execution_->createWorkspace();
            if (!workspace)
                return {};
            return std::shared_ptr<CanonicalProgramExecutionWorkspace>(
                workspace.release(),
                [pool = shared_from_this()](CanonicalProgramExecutionWorkspace *released) noexcept {
                    pool->release(std::unique_ptr<CanonicalProgramExecutionWorkspace>(released));
                });
        } catch (...) {
            return {};
        }
    }

private:
    void release(std::unique_ptr<CanonicalProgramExecutionWorkspace> workspace) noexcept {
        try {
            std::lock_guard lock(mutex_);
            idle_.push_back(std::move(workspace));
        } catch (...) {
        }
    }

    const std::shared_ptr<CanonicalProgramExecution> execution_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<CanonicalProgramExecutionWorkspace>> idle_;
};

} // namespace vernon::runtime::ad

#endif
