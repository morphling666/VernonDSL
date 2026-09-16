#include "compiler_program_tape.h"
#include "compiler_program_storage.h"

#include <algorithm>
#include <optional>

namespace vernon::compiler {
namespace {

llvm::json::Array serializeCarriers(const std::vector<ProgramTapeCarrier> &carriers) {
    llvm::json::Array result;
    for (ProgramTapeCarrier carrier : carriers)
        result.emplace_back(std::string(program_plan::tapeCarrierPlanName(carrier)));
    return result;
}

} // namespace

bool applyProgramTapeSizingContracts(llvm::json::Array &values, const std::vector<CanonicalProgramGraph> &graphs,
                                     const ProgramResourceIndex &resources, std::string &error) {
    for (const CanonicalProgramGraph &graph : graphs) {
        if (graph.direction != "forward")
            continue;
        for (const llvm::json::Object *node : graph.nodes) {
            const std::optional<llvm::StringRef> stageName = node->getString("stage");
            const llvm::json::Array *results = node->getArray("results");
            const auto compiled =
                stageName ? resources.compiledByRequest.find(stageName->str()) : resources.compiledByRequest.end();
            if (compiled == resources.compiledByRequest.end() || !results) {
                error = "forward Program node has no compiled stage or result contract";
                return false;
            }
            for (const llvm::json::Value &result : *results) {
                const std::optional<int64_t> valueId = result.getAsInteger();
                if (!valueId || *valueId < 0 || static_cast<size_t>(*valueId) >= values.size()) {
                    error = "forward Program node has an invalid result Value";
                    return false;
                }
                llvm::json::Object *value = values[static_cast<size_t>(*valueId)].getAsObject();
                if (!value || !isProgramAdTapeType(value->getString("type").value_or("")))
                    continue;
                const std::optional<uint64_t> minimumStride = compiled->second->minimumTapeStrideBytes;
                if (!minimumStride || !*minimumStride) {
                    error = "compiled autodiff tape producer has no minimum stride contract";
                    return false;
                }
                (*value)["minimum_tape_stride_bytes"] = static_cast<int64_t>(*minimumStride);
            }
        }
    }
    for (const llvm::json::Value &row : values) {
        const llvm::json::Object *value = row.getAsObject();
        if (value && isProgramAdTapeType(value->getString("type").value_or("")) &&
            !value->getInteger("minimum_tape_stride_bytes")) {
            error = "Program autodiff tape Value has no compiled producer sizing contract";
            return false;
        }
    }
    return true;
}

bool planProgramTapes(const llvm::json::Array &values, const llvm::json::Array &graphs,
                      std::vector<ProgramTapePlan> &plans, std::string &error) {
    const bool hasBackward = std::any_of(graphs.begin(), graphs.end(), [](const llvm::json::Value &graphRow) {
        const llvm::json::Object *graph = graphRow.getAsObject();
        return graph && graph->getString("direction") == "backward";
    });
    plans.clear();
    for (const llvm::json::Value &valueRow : values) {
        const llvm::json::Object *value = valueRow.getAsObject();
        const std::optional<int64_t> valueId = value ? value->getInteger("id") : std::nullopt;
        const std::optional<llvm::StringRef> type = value ? value->getString("type") : std::nullopt;
        if (!valueId || !type || !isProgramAdTapeType(*type))
            continue;
        const std::optional<int64_t> minimumStride = value->getInteger("minimum_tape_stride_bytes");
        if (!minimumStride || *minimumStride <= 0) {
            error = "autodiff tape Value requires a positive minimum_tape_stride_bytes contract";
            return false;
        }
        const llvm::json::Object *origin = value->getObject("origin");
        const bool producer =
            origin && origin->getString("tag") == "node_result" && origin->getString("graph") == "forward";
        plans.push_back({*valueId,
                         producer,
                         hasBackward,
                         static_cast<uint64_t>(*minimumStride),
                         {ProgramTapeCarrier::TapeData, ProgramTapeCarrier::ReplaySegment},
                         {ProgramTapeCarrier::LaunchMetadata, ProgramTapeCarrier::ReplayStatus}});
    }
    return true;
}

llvm::json::Array serializeProgramTapePlans(const std::vector<ProgramTapePlan> &plans) {
    llvm::json::Array result;
    for (const ProgramTapePlan &plan : plans)
        result.emplace_back(llvm::json::Object{
            {"value", plan.value},
            {"forward_producer", plan.forwardProducer},
            {"backward_consumer", plan.backwardConsumer},
            {"minimum_tape_stride_bytes", static_cast<int64_t>(plan.minimumTapeStrideBytes)},
            {"required_carriers", serializeCarriers(plan.requiredCarriers)},
            {"optional_carriers", serializeCarriers(plan.optionalCarriers)},
        });
    return result;
}

} // namespace vernon::compiler
