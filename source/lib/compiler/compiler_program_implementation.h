#ifndef VERNON_COMPILER_PROGRAM_IMPLEMENTATION_H
#define VERNON_COMPILER_PROGRAM_IMPLEMENTATION_H

#include "llvm/Support/JSON.h"

#include <string>

namespace vernon::compiler {

bool normalizeProgramImplementationAbi(llvm::json::Object &execution, llvm::json::Object &request,
                                       const llvm::json::Object &compiledEntry, std::string &error);

} // namespace vernon::compiler

#endif
