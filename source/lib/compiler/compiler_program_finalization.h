#ifndef VERNON_COMPILER_PROGRAM_FINALIZATION_H
#define VERNON_COMPILER_PROGRAM_FINALIZATION_H

#include "llvm/Support/JSON.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace vernon::compiler {

struct CanonicalComputeStage {
    std::string requestId;
    std::string implementationStageId;
    llvm::json::Object compiledReflection;
    llvm::json::Object compiledEntry;
};

bool normalizeProgramImplementationAbi(llvm::json::Object &execution, llvm::json::Object &request,
                                       const llvm::json::Object &compiledEntry, std::string &error);

std::optional<size_t> resolveProgramValueLeafIndex(const llvm::json::Object &layout, llvm::StringRef source,
                                                   llvm::StringRef parameter);

bool buildCanonicalComputeProgram(const llvm::json::Object &execution,
                                  const std::vector<CanonicalComputeStage> &compiledStages, llvm::json::Object &program,
                                  llvm::json::Object &stageContracts, llvm::json::Object &targetImplementations,
                                  std::string &error);

} // namespace vernon::compiler

#endif
