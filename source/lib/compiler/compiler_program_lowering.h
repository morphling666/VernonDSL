#ifndef VERNON_COMPILER_PROGRAM_LOWERING_H
#define VERNON_COMPILER_PROGRAM_LOWERING_H

#include "compiler_program_graph.h"
#include "compiler_program_stage.h"
#include "compiler_program_storage.h"

#include "llvm/Support/JSON.h"

#include <map>
#include <string>
#include <vector>

namespace vernon::compiler {

struct CanonicalProgramLoweringPlan {
    llvm::json::Object stages;
    llvm::json::Object stageContracts;
    llvm::json::Object targetImplementations;
    std::map<std::string, llvm::json::Array> nodesByGraph;
};

bool lowerCanonicalProgramStages(const llvm::json::Array &rawValues, const std::vector<CanonicalProgramGraph> &graphs,
                                 ProgramResourceIndex &resourceIndex, llvm::json::Array &storages,
                                 CanonicalProgramLoweringPlan &plan, std::string &error);

} // namespace vernon::compiler

#endif
