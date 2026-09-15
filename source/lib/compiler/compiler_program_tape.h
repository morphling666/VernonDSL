#ifndef VERNON_COMPILER_PROGRAM_TAPE_H
#define VERNON_COMPILER_PROGRAM_TAPE_H

#include "VernonProgramPlanTypes.h"
#include "compiler_program_graph.h"
#include "compiler_program_storage.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vernon::compiler {

using ProgramTapeCarrier = vernon::program_plan::TapeCarrier;

struct ProgramTapePlan {
    int64_t value{};
    bool forwardProducer{};
    bool backwardConsumer{};
    uint64_t minimumTapeStrideBytes{};
    std::vector<ProgramTapeCarrier> requiredCarriers;
    std::vector<ProgramTapeCarrier> optionalCarriers;
};

bool planProgramTapes(const llvm::json::Array &values, const llvm::json::Array &graphs,
                      std::vector<ProgramTapePlan> &plans, std::string &error);
bool applyProgramTapeSizingContracts(llvm::json::Array &values, const std::vector<CanonicalProgramGraph> &graphs,
                                     const ProgramResourceIndex &resources, std::string &error);
llvm::json::Array serializeProgramTapePlans(const std::vector<ProgramTapePlan> &plans);

} // namespace vernon::compiler

#endif
