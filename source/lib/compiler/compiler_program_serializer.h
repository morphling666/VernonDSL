#ifndef VERNON_COMPILER_PROGRAM_SERIALIZER_H
#define VERNON_COMPILER_PROGRAM_SERIALIZER_H

#include "llvm/Support/JSON.h"

#include <optional>

namespace vernon::compiler {

struct CanonicalProgramSerializationPlan {
    llvm::json::Object stages;
    llvm::json::Array storages;
    llvm::json::Array values;
    llvm::json::Array graphs;
    llvm::json::Object abi;
    std::optional<llvm::json::Object> residualContract;
};

llvm::json::Object serializeCanonicalProgram(CanonicalProgramSerializationPlan plan);

} // namespace vernon::compiler

#endif
