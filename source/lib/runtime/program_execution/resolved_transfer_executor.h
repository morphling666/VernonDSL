#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_RESOLVED_TRANSFER_EXECUTOR_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_RESOLVED_TRANSFER_EXECUTOR_H

#include "materialized_node_frame.h"

#include <vector>

namespace vernon::execution::detail {
struct RhiCommandExecutionPlan;
}

namespace vernon::runtime::program_execution {

class SubmissionRecord;

class ResolvedTransferExecutor {
public:
    ResolvedTransferExecutor(VernonRuntimeContext &context, ProgramInvocationState &state)
        : context_(context), state_(state) {}

    bool prepareGraph(program::GraphDirection graph, std::string &error);
    // The transfer fragments that must run before `consumer`, in order; none when it needs no transfer.
    VernonStatus planBeforeConsumer(program::NodeKey consumer, const std::vector<DeviceBufferCopy> &physicalCopies,
                                    std::vector<vernon::execution::detail::RhiCommandExecutionPlan> &fragments);
    bool restoreForRetry(program::GraphDirection graph, SubmissionRecord &submission, std::string &error);
    bool readbackBoundaryValues(const std::vector<char> &values, std::string &error) const;

private:
    VernonRuntimeContext &context_;
    ProgramInvocationState &state_;
    std::vector<DeviceBufferCopy> initialCopies_;
    program::GraphDirection preparedGraph_{program::GraphDirection::Forward};
    bool prepared_{};
    bool initialCopiesPending_{};
};

} // namespace vernon::runtime::program_execution

#endif
