#ifndef VERNON_COMPILER_PROGRAM_CAPTURE_H
#define VERNON_COMPILER_PROGRAM_CAPTURE_H

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <set>

namespace vernon::compiler {

void collectProgramBackwardCaptures(const llvm::json::Object &execution, std::set<int64_t> &captures);

void rebuildProgramDependencies(llvm::json::Object &execution, llvm::StringRef graphName);

void rebuildProgramCaptures(llvm::json::Object &execution);

} // namespace vernon::compiler

#endif
