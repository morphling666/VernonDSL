#ifndef VERNON_COMPILER_PROGRAM_STAGE_H
#define VERNON_COMPILER_PROGRAM_STAGE_H

#include "compiler_program_storage.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace vernon::compiler {

bool programResourceAccessSatisfies(llvm::StringRef physical, llvm::StringRef logical);
bool programAbiShapesCompatible(const llvm::json::Array *logical, const llvm::json::Array *physical);
std::optional<std::string> reflectedProgramResourceAccess(const llvm::json::Object &row);
const llvm::json::Object *programEndpointLayout(const llvm::json::Object &row, bool resource);
llvm::json::Object programValueCarrier(llvm::StringRef tag, int64_t slot, const llvm::json::Object &layout);
llvm::json::Object compiledProgramEndpointAbi(const llvm::json::Object &row, llvm::StringRef module, int64_t index);

bool indexProgramNodeBindings(const llvm::json::Array &rawBindings, std::map<std::string, int64_t> &boundValues,
                              std::string &error);

std::optional<int64_t> ensureProgramResourceAccess(int64_t valueId, bool attachment,
                                                   std::map<int64_t, ProgramLogicalResource> &resources,
                                                   std::map<int64_t, int64_t> &accessByValue,
                                                   llvm::json::Array &accesses);

bool compatibleProgramBindingShape(llvm::StringRef role, llvm::StringRef carrier, const llvm::json::Array *logical,
                                   const llvm::json::Array *physical);

std::optional<size_t> resolveProgramValueLeafIndex(const llvm::json::Object &layout, llvm::StringRef source,
                                                   llvm::StringRef parameter);

} // namespace vernon::compiler

#endif
