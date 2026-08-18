#include "operator/operator_execution.h"

#include "operator/operator_lowering.h"

#include <utility>

namespace vernon::ops {

bool buildOperatorExecutionPlan(const OperatorDag &operators,
                                const std::vector<execution::detail::RhiCommandNodeEncoder> &operatorEncoders,
                                std::vector<execution::detail::RhiCommandResourceBinding> bindings,
                                OperatorExecutionPlan &plan, std::string &error) {
    OperatorExecutionPlan result;
    if (!lowerOperatorDagToCommandDag(operators, result.commands, error))
        return false;
    if (operatorEncoders.size() != result.commands.nodes.size()) {
        error = "operator executable mapping does not match lowered commands";
        return false;
    }
    result.encoders = operatorEncoders;
    result.bindings = std::move(bindings);
    if (!execution::detail::validateRhiCommandExecutionPlan(result, error))
        return false;
    plan = std::move(result);
    return true;
}

VernonRhiStatus executeOperatorExecutionPlanAndWait(VernonRhiDevice device, const OperatorExecutionPlan &plan) {
    return execution::detail::executeRhiCommandPlanAndWait(device, VERNON_RHI_QUEUE_COMPUTE, plan);
}

} // namespace vernon::ops
