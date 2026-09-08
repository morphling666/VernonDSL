#ifndef VERNON_COMPILER_JSON_H
#define VERNON_COMPILER_JSON_H

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>

namespace vernon::compiler {

llvm::json::Array copyJsonArray(const llvm::json::Array &source);
llvm::json::Object copyJsonObject(const llvm::json::Object &source);

const llvm::json::Object *findJsonObjectByIntegerId(const llvm::json::Array &objects, int64_t id);

std::string canonicalJsonSha256(const llvm::json::Value &value);

} // namespace vernon::compiler

#endif
