#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_FORWARD_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_FORWARD_H

#include "VernonResult.hpp"
#include "VernonRuntime.h"
#include "invocation_outcome.h"

#include <map>
#include <memory>

namespace vernon::runtime {
struct ProgramInvocationContext;
namespace ad {
class PullbackExecution;
}
namespace program {
struct InvocationSnapshot;
}
} // namespace vernon::runtime

namespace vernon::runtime::program_execution {

enum class PullbackTransferError {
    LifecycleUnavailable,
    AllocationFailure,
};

VernonStatus
forwardProgramInvocation(VernonProgramExecutable &pipeline, const VernonProgramArgument *arguments,
                         size_t argumentCount, VernonPullback *&pullback, InvocationMutationOutcome &outcome,
                         const ProgramInvocationContext *programContext = nullptr,
                         std::map<VernonProgramNodeId, std::unique_ptr<ad::PullbackExecution>> *nodePullbacks = nullptr,
                         bool retainPullback = true);
void attachProgramSnapshot(VernonPullback &pullback, std::shared_ptr<const program::InvocationSnapshot> snapshot);
vernon::Result<std::unique_ptr<VernonPullback>, PullbackTransferError>
makeRetainedProgramPullback(VernonProgramExecutable &pipeline, std::unique_ptr<ad::PullbackExecution> &execution,
                            const std::shared_ptr<const program::InvocationSnapshot> &snapshot) noexcept;

} // namespace vernon::runtime::program_execution

#endif
