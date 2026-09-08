#ifndef VERNON_COMPILER_PROGRAM_ASSEMBLY_H
#define VERNON_COMPILER_PROGRAM_ASSEMBLY_H

#include "compiler_program_graph.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace vernon::compiler {

bool assembleCanonicalProgram(const llvm::json::Object &rawSignature,
                              const std::vector<CanonicalProgramGraph> &selectedGraphs, llvm::json::Object stages,
                              llvm::json::Array storages, llvm::json::Array values,
                              std::map<std::string, llvm::json::Array> canonicalNodesByGraph,
                              const std::map<int64_t, int64_t> &storageByValue, const std::set<int64_t> &capturedValues,
                              llvm::json::Object &program, std::string &error);

} // namespace vernon::compiler

#endif
