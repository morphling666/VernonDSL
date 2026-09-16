#include "compiler_program_graph.h"

#include <optional>
#include <set>
#include <utility>

namespace vernon::compiler {

bool selectCanonicalProgramGraphs(const llvm::json::Array &rawGraphs, std::vector<CanonicalProgramGraph> &graphs,
                                  std::string &error) {
    const llvm::json::Object *forward = nullptr;
    const llvm::json::Object *backward = nullptr;
    for (const llvm::json::Value &value : rawGraphs) {
        const llvm::json::Object *graph = value.getAsObject();
        const std::optional<llvm::StringRef> direction = graph ? graph->getString("direction") : std::nullopt;
        if (!graph || !direction) {
            error = "canonical compute finalization requires named forward/backward graphs";
            return false;
        }
        if (*direction == "forward") {
            if (forward) {
                error = "canonical compute finalization requires exactly one forward graph";
                return false;
            }
            forward = graph;
        } else if (*direction == "backward") {
            if (backward) {
                error = "canonical compute finalization requires at most one backward graph";
                return false;
            }
            backward = graph;
        } else {
            error = "canonical compute finalization received an unknown graph direction";
            return false;
        }
    }
    if (!forward) {
        error = "canonical compute finalization requires exactly one forward graph and signature";
        return false;
    }

    graphs.clear();
    const auto append = [&](const llvm::json::Object *graph) {
        CanonicalProgramGraph view;
        view.graph = graph;
        view.direction = graph->getString("direction")->str();
        view.name = graph->getString("name").value_or(view.direction).str();
        const llvm::json::Array *nodes = graph->getArray("nodes");
        const llvm::json::Array *arguments = graph->getArray("arguments");
        const llvm::json::Array *results = graph->getArray("results");
        if (view.name.empty() || !nodes || nodes->empty() || !arguments || !results) {
            error = "canonical compute finalization requires one non-empty " + view.direction + " compute graph";
            return false;
        }
        for (const llvm::json::Value &nodeValue : *nodes) {
            const llvm::json::Object *node = nodeValue.getAsObject();
            if (!node) {
                error = "canonical compute graph contains an invalid node";
                return false;
            }
            view.nodes.push_back(node);
        }
        const auto appendIds = [&](const llvm::json::Array &source, std::vector<int64_t> &target) {
            for (const llvm::json::Value &value : source) {
                const std::optional<int64_t> id = value.getAsInteger();
                if (!id) {
                    error = "canonical compute graph contains an invalid boundary Value ID";
                    return false;
                }
                target.push_back(*id);
            }
            return true;
        };
        if (!appendIds(*arguments, view.arguments) || !appendIds(*results, view.results))
            return false;
        graphs.push_back(std::move(view));
        return true;
    };
    return append(forward) && (!backward || append(backward));
}

bool linkCanonicalProgram(const llvm::json::Array &rawGraphs, size_t compiledStageCount, CanonicalProgramLinkPlan &plan,
                          std::string &error) {
    plan = {};
    if (!selectCanonicalProgramGraphs(rawGraphs, plan.graphs, error))
        return false;
    for (const CanonicalProgramGraph &graph : plan.graphs)
        plan.nodeCount += graph.nodes.size();
    if (plan.nodeCount != compiledStageCount) {
        error = "compiled stages do not exactly cover canonical compute nodes";
        return false;
    }
    return true;
}

bool canonicalProgramValueIds(const llvm::json::Array &ids, DuplicateProgramValuePolicy duplicatePolicy,
                              llvm::json::Array &canonical, std::string &error) {
    std::set<int64_t> unique;
    for (const llvm::json::Value &value : ids) {
        const std::optional<int64_t> id = value.getAsInteger();
        if (!id) {
            error = "canonical compute node has invalid operand or result ids";
            return false;
        }
        if (!unique.insert(*id).second && duplicatePolicy == DuplicateProgramValuePolicy::Reject) {
            error = "canonical compute node has duplicate result ids";
            return false;
        }
    }
    canonical.clear();
    canonical.reserve(unique.size());
    for (int64_t id : unique)
        canonical.emplace_back(id);
    return true;
}

} // namespace vernon::compiler
