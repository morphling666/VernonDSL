#ifndef VERNON_COMPILER_PROGRAM_TARGET_AGGREGATION_H
#define VERNON_COMPILER_PROGRAM_TARGET_AGGREGATION_H

#include "compiler_program_types.h"

#include <string>

namespace vernon::compiler {

bool appendProgramTargetModuleMetadata(CanonicalProgramStage &stage, const llvm::json::Object &reflection,
                                       std::string &error);

} // namespace vernon::compiler

#endif
