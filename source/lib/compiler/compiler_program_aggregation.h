#ifndef VERNON_COMPILER_PROGRAM_AGGREGATION_H
#define VERNON_COMPILER_PROGRAM_AGGREGATION_H

#include "compiler_program_types.h"

#include <string>

namespace vernon::compiler {

bool appendCompiledProgramModule(CanonicalProgramStage &stage, const std::string &requestId,
                                 const std::string &implementationStageId, ProgramStageOperation operation,
                                 const llvm::json::Object &reflection, const llvm::json::Object &entry,
                                 std::string &error);

bool finalizeProgramStageAggregation(const CanonicalProgramStage &stage, std::string &error);

} // namespace vernon::compiler

#endif
