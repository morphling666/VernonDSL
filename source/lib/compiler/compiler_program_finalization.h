#ifndef VERNON_COMPILER_PROGRAM_FINALIZATION_H
#define VERNON_COMPILER_PROGRAM_FINALIZATION_H

#include "compiler_program_types.h"

#include <string>
#include <vector>

namespace vernon::compiler {

bool buildCanonicalComputeProgram(const llvm::json::Object &execution,
                                  const std::vector<CanonicalComputeStage> &compiledStages, llvm::json::Object &program,
                                  llvm::json::Object &stageContracts, llvm::json::Object &targetImplementations,
                                  std::string &error);

} // namespace vernon::compiler

#endif
