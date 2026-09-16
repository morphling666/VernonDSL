#ifndef VERNON_COMPILER_PROGRAM_DERIVATIVE_H
#define VERNON_COMPILER_PROGRAM_DERIVATIVE_H

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace vernon::compiler {

enum class ProgramBoundaryRole {
    Input,
    Output,
    Cotangent,
    Gradient,
};

struct ProgramBoundaryIdentity {
    int64_t slot{};
    std::string path;
    ProgramBoundaryRole role{ProgramBoundaryRole::Input};
};

using ProgramValuePathComponent = std::variant<std::string, uint32_t>;

struct ProgramDerivativeProjectionPlan {
    ProgramBoundaryIdentity derivative;
    ProgramBoundaryIdentity primal;
    std::vector<ProgramValuePathComponent> valuePath;
};

bool planProgramDerivativeProjections(const std::vector<ProgramBoundaryIdentity> &boundaries,
                                      std::vector<ProgramDerivativeProjectionPlan> &projections, std::string &error);
llvm::json::Array
serializeProgramDerivativeProjections(const std::vector<ProgramDerivativeProjectionPlan> &projections);

} // namespace vernon::compiler

#endif
