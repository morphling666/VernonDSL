#ifndef VERNON_COMPILER_PROGRAM_BUILTIN_H
#define VERNON_COMPILER_PROGRAM_BUILTIN_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace vernon::compiler {

bool buildProgramBuiltinMlir(llvm::StringRef operation, llvm::StringRef elementType, uint32_t rank,
                             llvm::ArrayRef<std::string> leafDtypes, std::string &entry, std::string &mlir,
                             std::string &error);

} // namespace vernon::compiler

#endif
