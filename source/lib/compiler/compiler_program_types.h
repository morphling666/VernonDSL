#ifndef VERNON_COMPILER_PROGRAM_TYPES_H
#define VERNON_COMPILER_PROGRAM_TYPES_H

#include "llvm/Support/JSON.h"

#include <string>

namespace vernon::compiler {

struct CanonicalComputeStage {
    std::string requestId;
    std::string implementationStageId;
    llvm::json::Object compiledReflection;
    llvm::json::Object compiledEntry;
};

} // namespace vernon::compiler

#endif
