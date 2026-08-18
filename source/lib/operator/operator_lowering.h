#ifndef VERNON_OPERATOR_LOWERING_H
#define VERNON_OPERATOR_LOWERING_H

#include "execution_graph/execution_command_model.h"
#include "operator/operator_model.h"

#include <string>
namespace vernon::ops {

bool lowerOperatorDagToCommandDag(const OperatorDag &operators, execution::detail::CommandDag &commands,
                                  std::string &error);

} // namespace vernon::ops

#endif
