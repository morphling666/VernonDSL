#ifndef VERNON_COMPILER_PROGRAM_TAPE_H
#define VERNON_COMPILER_PROGRAM_TAPE_H

#include "VernonProgramPlanTypes.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <vector>

namespace vernon::compiler {

using ProgramTapeCarrier = vernon::program_plan::TapeCarrier;

struct ProgramTapePlan {
    int64_t value{};
    bool forwardProducer{};
    bool backwardConsumer{};
    std::vector<ProgramTapeCarrier> requiredCarriers;
    std::vector<ProgramTapeCarrier> optionalCarriers;
};

std::vector<ProgramTapePlan> planProgramTapes(const llvm::json::Array &values, const llvm::json::Array &graphs);
llvm::json::Array serializeProgramTapePlans(const std::vector<ProgramTapePlan> &plans);

} // namespace vernon::compiler

#endif
