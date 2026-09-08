#ifndef VERNON_COMPILER_PROGRAM_TYPES_H
#define VERNON_COMPILER_PROGRAM_TYPES_H

#include "llvm/Support/JSON.h"

#include <map>
#include <optional>
#include <string>

namespace vernon::compiler {

enum class ProgramStageOperation {
    Compute,
    Graphics,
};

struct CompiledProgramModule {
    std::string entryName;
    llvm::json::Object entry;
};

struct ProgramTargetImplementation {
    std::string target;
    llvm::json::Object metadata;
};

struct CanonicalProgramStage {
    std::string requestId;
    std::string implementationStageId;
    ProgramStageOperation operation{ProgramStageOperation::Compute};
    std::map<std::string, CompiledProgramModule> modules;
    llvm::json::Object portableReflection;
    ProgramTargetImplementation targetImplementation;
    std::string targetIdentity;
    std::optional<int64_t> pipelineVersion;
};

} // namespace vernon::compiler

#endif
