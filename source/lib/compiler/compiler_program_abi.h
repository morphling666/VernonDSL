#ifndef VERNON_COMPILER_PROGRAM_ABI_H
#define VERNON_COMPILER_PROGRAM_ABI_H

#include "llvm/Support/JSON.h"

#include <string>

namespace vernon::compiler {

// Builds the compiler-owned public Program ABI. This phase is intentionally
// separate from graph linking and serialization so boundary identity has one
// implementation and the runtime only consumes its serialized result.
bool buildCanonicalProgramAbi(const llvm::json::Object &signature, const llvm::json::Array &values,
                              const llvm::json::Array &storages, const llvm::json::Array &graphs,
                              llvm::json::Object &abi, std::string &error);

} // namespace vernon::compiler

#endif
