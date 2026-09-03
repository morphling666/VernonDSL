#include "compiler_program_serializer.h"

#include <utility>

namespace vernon::compiler {

llvm::json::Object serializeCanonicalProgram(CanonicalProgramSerializationPlan plan) {
    llvm::json::Object program{
        {"stages", std::move(plan.stages)},
        {"parameters", llvm::json::Array()},
        {"storages", std::move(plan.storages)},
        {"values", std::move(plan.values)},
        {"shape_symbols", llvm::json::Array()},
        {"shape_constraints", llvm::json::Array()},
        {"alias_preconditions", llvm::json::Array()},
        {"graphs", std::move(plan.graphs)},
        {"abi", std::move(plan.abi)},
    };
    if (plan.residualContract)
        program["residual_contract"] = std::move(*plan.residualContract);
    return program;
}

} // namespace vernon::compiler
