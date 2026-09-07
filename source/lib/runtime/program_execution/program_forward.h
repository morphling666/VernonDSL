#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_FORWARD_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PROGRAM_FORWARD_H

#include "VernonRuntime.h"

#include <memory>

namespace vernon::runtime {
struct ProgramInvocationContext;
namespace program {
struct InvocationSnapshot;
}
} // namespace vernon::runtime

namespace vernon::runtime::program_execution {

VernonStatus forwardProgramInvocation(VernonProgramExecutable &pipeline,
                                      const VernonStageInvocationDescriptor &invocation, VernonPullback *&pullback,
                                      const ProgramInvocationContext *programContext = nullptr);
void attachProgramSnapshot(VernonPullback &pullback, std::shared_ptr<const program::InvocationSnapshot> snapshot);

} // namespace vernon::runtime::program_execution

#endif
