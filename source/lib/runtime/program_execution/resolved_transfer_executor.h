#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_RESOLVED_TRANSFER_EXECUTOR_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_RESOLVED_TRANSFER_EXECUTOR_H

#include "materialized_node_frame.h"

namespace vernon::execution::detail {
struct RhiCommandExecutionPlan;
}

namespace vernon::runtime::program_execution {

class ResolvedTransferExecutor {
public:
    ResolvedTransferExecutor(VernonRuntimeContext &context, ProgramInvocationState &state)
        : context_(context), state_(state) {}

    bool prepareGraph(program::GraphDirection graph, std::string &error);
    VernonStatus appendBeforeConsumer(program::NodeKey consumer, const std::vector<DeviceBufferCopy> &physicalCopies,
                                      vernon::execution::detail::RhiCommandExecutionPlan &commands, std::string &error);
    bool restoreForRetry(program::GraphDirection graph, std::string &error);
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
