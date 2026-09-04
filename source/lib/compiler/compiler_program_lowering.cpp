#include "compiler_program_lowering.h"

#include "compiler_json.h"
#include "compiler_program_compute.h"
#include "compiler_program_graphics.h"

#include <utility>

namespace vernon::compiler {

bool lowerCanonicalProgramStages(const llvm::json::Array &rawValues, const std::vector<CanonicalProgramGraph> &graphs,
                                 ProgramResourceIndex &resourceIndex, llvm::json::Array &storages,
                                 CanonicalProgramLoweringPlan &plan, std::string &error) {
    auto &compiledByRequest = resourceIndex.compiledByRequest;
    auto &resourcesByStage = resourceIndex.resourcesByStage;
    for (const CanonicalProgramGraph &graph : graphs) {
        llvm::json::Array &canonicalNodes = plan.nodesByGraph[graph.name];
        for (const llvm::json::Object *node : graph.nodes) {
            const std::string logicalStage = node->getString("stage")->str();
            const CanonicalProgramStage &compiled = *compiledByRequest[logicalStage];
            const llvm::StringRef requestId = compiled.requestId;
            const llvm::json::Object &compiledReflection = compiled.portableReflection;
            const llvm::json::Array *nodeOperands = node->getArray("operands");
            const llvm::json::Array *nodeResults = node->getArray("results");
            const llvm::json::Array *rawBindings = node->getArray("bindings");
            const llvm::json::Array *workgroups = node->getArray("grid");
            std::map<int64_t, ProgramLogicalResource> &resources = resourcesByStage[compiled.requestId];
            std::map<std::string, int64_t> boundValues;
            if (!indexProgramNodeBindings(*rawBindings, boundValues, error))
                return false;

            if (node->getString("kind") == "render") {
                ProgramGraphicsInterfacePlan graphicsPlan;
                if (!buildCanonicalGraphicsInterfaces(compiledReflection, rawValues, storages, boundValues, resources,
                                                      graphicsPlan, error))
                    return false;
                if (!applyProgramStorageUsageRequirements(storages, graphicsPlan.storageUsageRequirements, error))
                    return false;
                llvm::json::Array attachmentConstraints;
                llvm::json::Object operation;
                if (!buildCanonicalGraphicsOperation(*node, rawValues, storages, boundValues, resources,
                                                     graphicsPlan.fragmentOutputs.json(), graphicsPlan.vertexCountValue,
                                                     graphicsPlan.accessByValue, graphicsPlan.accesses.json(),
                                                     attachmentConstraints, operation, error))
                    return false;
                llvm::json::Array requiredFeatures;
                if (const llvm::json::Array *rawFeatures = compiledReflection.getArray("required_features"))
                    requiredFeatures = copyJsonArray(*rawFeatures);
                llvm::json::Object stageContract;
                if (!buildCanonicalGraphicsStageContract(*node, std::move(requiredFeatures), graphicsPlan,
                                                         std::move(attachmentConstraints), stageContract, error))
                    return false;
                const std::string contractHash = canonicalJsonSha256(llvm::json::Value(copyJsonObject(stageContract)));
                plan.stageContracts[requestId] = std::move(stageContract);
                if (!graphicsPlan.implementationEndpoints.empty() || !compiled.targetImplementation.metadata.empty()) {
                    llvm::json::Object implementation{
                        {"target", compiled.targetImplementation.target},
                        {"metadata", copyJsonObject(compiled.targetImplementation.metadata)},
                    };
                    implementation["endpoints"] = graphicsPlan.implementationEndpoints.take();
                    plan.targetImplementations[requestId] = std::move(implementation);
                }
                plan.stages[requestId] = llvm::json::Object{{"operation", "graphics"}, {"contract_hash", contractHash}};

                llvm::json::Array operands;
                llvm::json::Array results;
                if (!canonicalProgramValueIds(*nodeOperands, DuplicateProgramValuePolicy::CollapseOperandUses, operands,
                                              error) ||
                    !canonicalProgramValueIds(*nodeResults, DuplicateProgramValuePolicy::Reject, results, error))
                    return false;
                canonicalNodes.emplace_back(llvm::json::Object{{"id", node->getInteger("id").value_or(0)},
                                                               {"name", node->getString("name").value_or("").str()},
                                                               {"stage", requestId.str()},
                                                               {"operands", std::move(operands)},
                                                               {"results", std::move(results)},
                                                               {"bindings", graphicsPlan.endpointBindings.take()},
                                                               {"accesses", graphicsPlan.accesses.take()},
                                                               {"operation", std::move(operation)}});
                continue;
            }

            const auto computeModule = compiled.modules.find("compute");
            if (computeModule == compiled.modules.end()) {
                error = "canonical compute stage has no compute module";
                return false;
            }
            const llvm::json::Object &compiledEntry = computeModule->second.entry;
            int64_t nextSlot = 0;
            if (!prepareCanonicalComputeInterface(compiledEntry, rawValues, boundValues, nextSlot, error))
                return false;
            ProgramComputeEndpointPlan computePlan;
            if (!buildCanonicalComputeEndpoints(compiledEntry, rawValues, boundValues, resources, nextSlot, computePlan,
                                                error))
                return false;
            llvm::json::Object stageContract;
            if (!buildCanonicalComputeStageContract(compiledEntry, compiledReflection, computePlan.endpoints.take(),
                                                    stageContract, error))
                return false;
            const std::string contractHash = canonicalJsonSha256(llvm::json::Value(copyJsonObject(stageContract)));
            plan.stageContracts[requestId] = std::move(stageContract);
            if (!computePlan.implementationEndpoints.empty() || !compiled.targetImplementation.metadata.empty()) {
                llvm::json::Object implementation{
                    {"target", compiled.targetImplementation.target},
                    {"metadata", copyJsonObject(compiled.targetImplementation.metadata)},
                };
                implementation["endpoints"] = computePlan.implementationEndpoints.take();
                plan.targetImplementations[requestId] = std::move(implementation);
            }
            plan.stages[requestId] = llvm::json::Object{{"operation", "compute"}, {"contract_hash", contractHash}};
            llvm::json::Array operands;
            llvm::json::Array results;
            if (!canonicalProgramValueIds(*nodeOperands, DuplicateProgramValuePolicy::CollapseOperandUses, operands,
                                          error) ||
                !canonicalProgramValueIds(*nodeResults, DuplicateProgramValuePolicy::Reject, results, error))
                return false;
            canonicalNodes.emplace_back(llvm::json::Object{
                {"id", node->getInteger("id").value_or(0)},
                {"name", node->getString("name").value_or("").str()},
                {"stage", requestId.str()},
                {"operands", std::move(operands)},
                {"results", std::move(results)},
                {"bindings", computePlan.endpointBindings.take()},
                {"accesses", computePlan.accesses.take()},
                {"operation", llvm::json::Object{{"tag", "compute"}, {"workgroups", copyJsonArray(*workgroups)}}}});
        }
    }
    return true;
}

} // namespace vernon::compiler
