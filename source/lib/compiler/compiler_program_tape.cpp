#include "compiler_program_tape.h"

#include <algorithm>
#include <optional>

namespace vernon::compiler {
namespace {

bool isAdTapeType(llvm::StringRef type) { return type == "!vernon.ad_tape" || type.starts_with("!vernon.ad_tape<"); }

llvm::json::Array serializeCarriers(const std::vector<ProgramTapeCarrier> &carriers) {
    llvm::json::Array result;
    for (ProgramTapeCarrier carrier : carriers)
        result.emplace_back(std::string(program_plan::tapeCarrierPlanName(carrier)));
    return result;
}

} // namespace

std::vector<ProgramTapePlan> planProgramTapes(const llvm::json::Array &values, const llvm::json::Array &graphs) {
    const bool hasBackward = std::any_of(graphs.begin(), graphs.end(), [](const llvm::json::Value &graphRow) {
        const llvm::json::Object *graph = graphRow.getAsObject();
        return graph && graph->getString("direction") == "backward";
    });
    std::vector<ProgramTapePlan> plans;
    for (const llvm::json::Value &valueRow : values) {
        const llvm::json::Object *value = valueRow.getAsObject();
        const std::optional<int64_t> valueId = value ? value->getInteger("id") : std::nullopt;
        const std::optional<llvm::StringRef> type = value ? value->getString("type") : std::nullopt;
        if (!valueId || !type || !isAdTapeType(*type))
            continue;
        const llvm::json::Object *origin = value->getObject("origin");
        const bool producer =
            origin && origin->getString("tag") == "node_result" && origin->getString("graph") == "forward";
        plans.push_back({*valueId,
                         producer,
                         hasBackward,
                         {ProgramTapeCarrier::TapeData, ProgramTapeCarrier::ReplaySegment},
                         {ProgramTapeCarrier::LaunchMetadata, ProgramTapeCarrier::ReplayStatus}});
    }
    return plans;
}

llvm::json::Array serializeProgramTapePlans(const std::vector<ProgramTapePlan> &plans) {
    llvm::json::Array result;
    for (const ProgramTapePlan &plan : plans)
        result.emplace_back(llvm::json::Object{
            {"value", plan.value},
            {"forward_producer", plan.forwardProducer},
            {"backward_consumer", plan.backwardConsumer},
            {"required_carriers", serializeCarriers(plan.requiredCarriers)},
            {"optional_carriers", serializeCarriers(plan.optionalCarriers)},
        });
    return result;
}

} // namespace vernon::compiler
