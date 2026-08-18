#ifndef VERNON_OPERATOR_EXECUTION_H
#define VERNON_OPERATOR_EXECUTION_H

#include "execution_graph/execution_graph_internal.h"
#include "operator/operator_model.h"

#include <string>
#include <vector>

namespace vernon::ops {

using OperatorExecutionPlan = execution::detail::RhiCommandExecutionPlan;

bool buildOperatorExecutionPlan(const OperatorDag &operators,
                                const std::vector<execution::detail::RhiCommandNodeEncoder> &operatorEncoders,
                                std::vector<execution::detail::RhiCommandResourceBinding> bindings,
                                OperatorExecutionPlan &plan, std::string &error);

VernonRhiStatus executeOperatorExecutionPlanAndWait(VernonRhiDevice device, const OperatorExecutionPlan &plan);

} // namespace vernon::ops

#endif
