#include "compiler_program_finalization.h"
#include "compiler_program_assembly.h"
#include "compiler_program_capture.h"
#include "compiler_program_graph.h"
#include "compiler_program_lowering.h"
#include "compiler_program_storage.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vernon::compiler {

bool finalizeCanonicalProgram(const llvm::json::Object &execution,
                              const std::vector<CanonicalProgramStage> &compiledStages, llvm::json::Object &program,
                              llvm::json::Object &stageContracts, llvm::json::Object &targetImplementations,
                              std::string &error) {
    const llvm::json::Array *rawValues = execution.getArray("values");
    const llvm::json::Array *rawGraphs = execution.getArray("graphs");
    const llvm::json::Object *rawSignature = execution.getObject("signature");
    CanonicalProgramLinkPlan link;
    if (!rawValues || !rawGraphs || !rawSignature ||
        !linkCanonicalProgram(*rawGraphs, compiledStages.size(), link, error)) {
        if (error.empty())
            error = "canonical compute finalization requires exactly one forward graph and signature";
        return false;
    }
    const std::vector<CanonicalProgramGraph> &selectedGraphs = link.graphs;

    ProgramResourceIndex resourceIndex;
    if (!indexProgramResources(*rawValues, selectedGraphs, compiledStages, resourceIndex, error))
        return false;
    std::set<int64_t> capturedValues;
    collectProgramBackwardCaptures(execution, capturedValues);
    ProgramStoragePlan storagePlan;
    if (!materializeProgramStoragePlan(*rawValues, selectedGraphs, resourceIndex, capturedValues, storagePlan, error))
        return false;

    llvm::json::Array &storages = storagePlan.storages;
    llvm::json::Array &values = storagePlan.values;
    auto &storageByValue = storagePlan.storageByValue;

    CanonicalProgramLoweringPlan lowering;
    if (!lowerCanonicalProgramStages(*rawValues, selectedGraphs, resourceIndex, storages, lowering, error))
        return false;
    stageContracts = std::move(lowering.stageContracts);
    targetImplementations = std::move(lowering.targetImplementations);
    return assembleCanonicalProgram(*rawSignature, selectedGraphs, std::move(lowering.stages), std::move(storages),
                                    std::move(values), std::move(lowering.nodesByGraph), storageByValue, capturedValues,
                                    program, error);
}

} // namespace vernon::compiler
