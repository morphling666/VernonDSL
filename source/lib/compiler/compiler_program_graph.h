#ifndef VERNON_COMPILER_PROGRAM_GRAPH_H
#define VERNON_COMPILER_PROGRAM_GRAPH_H

#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vernon::compiler {

struct CanonicalProgramGraph {
    const llvm::json::Object *graph{};
    std::vector<const llvm::json::Object *> nodes;
    std::vector<int64_t> arguments;
    std::vector<int64_t> results;
    std::string name;
    std::string direction;
};

struct CanonicalProgramLinkPlan {
    std::vector<CanonicalProgramGraph> graphs;
    size_t nodeCount{};
};

enum class DuplicateProgramValuePolicy {
    Reject,
    CollapseOperandUses,
};

// Selects and validates the forward/backward graph set before any storage,
// target-binding, or serialization phase runs.
bool selectCanonicalProgramGraphs(const llvm::json::Array &rawGraphs, std::vector<CanonicalProgramGraph> &graphs,
                                  std::string &error);
bool linkCanonicalProgram(const llvm::json::Array &rawGraphs, size_t compiledStageCount, CanonicalProgramLinkPlan &plan,
                          std::string &error);

bool canonicalProgramValueIds(const llvm::json::Array &ids, DuplicateProgramValuePolicy duplicatePolicy,
                              llvm::json::Array &result, std::string &error);

} // namespace vernon::compiler

#endif
