#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_FORWARD_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_FORWARD_H

#include "VernonRuntime.h"

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

VernonStatus
forwardProgramInvocation(VernonProgramExecutable &pipeline, const VernonProgramArgument *arguments,
                         size_t argumentCount, VernonPullback *&pullback,
                         const ProgramInvocationContext *programContext = nullptr,
                         std::map<VernonProgramNodeId, std::unique_ptr<ad::PullbackExecution>> *nodePullbacks = nullptr,
                         bool retainPullback = true);
void attachProgramSnapshot(VernonPullback &pullback, std::shared_ptr<const program::InvocationSnapshot> snapshot);
VernonPullback *makeRetainedProgramPullback(VernonProgramExecutable &pipeline,
                                            std::unique_ptr<ad::PullbackExecution> execution,
                                            std::shared_ptr<const program::InvocationSnapshot> snapshot);

} // namespace vernon::runtime::program_execution

#endif
