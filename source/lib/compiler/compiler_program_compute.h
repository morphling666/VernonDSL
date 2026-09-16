#ifndef VERNON_COMPILER_PROGRAM_COMPUTE_H
#define VERNON_COMPILER_PROGRAM_COMPUTE_H

#include "VernonProgramPlanTypes.h"
#include "compiler_program_plan_records.h"
#include "compiler_program_stage.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace vernon::compiler {

struct ProgramLogicalResource;

struct ProgramComputeEndpointPlan {
    ProgramEndpointRecords endpoints;
    ProgramEndpointBindingRecords endpointBindings;
    ProgramImplementationEndpointRecords implementationEndpoints;
    ProgramAccessRecords accesses;
    std::optional<llvm::json::Object> semanticMetadataCarrier;
    std::optional<llvm::json::Object> physicalMetadataCarrier;
};

bool isProgramTapeCarrierRole(llvm::StringRef role);

bool isProgramKernelHiddenBuiltin(llvm::StringRef builtin);

bool isProgramKernelTapeBuiltin(llvm::StringRef builtin);

bool prepareCanonicalComputeInterface(const llvm::json::Object &compiledEntry, const llvm::json::Array &rawValues,
                                      const ProgramNodeBindingIndex &boundValues, int64_t &nextPortableSlot,
                                      std::string &error);

bool buildCanonicalComputeEndpoints(const llvm::json::Object &compiledEntry, const llvm::json::Array &rawValues,
                                    const ProgramNodeBindingIndex &boundValues,
                                    std::map<int64_t, ProgramLogicalResource> &resources, int64_t nextPortableSlot,
                                    ProgramComputeEndpointPlan &plan, std::string &error);

bool validateCanonicalComputePortableSlots(const llvm::json::Array &endpoints, std::string &error);

bool buildCanonicalComputeStageContract(const llvm::json::Object &compiledEntry,
                                        const llvm::json::Object &compiledReflection, llvm::json::Array endpoints,
                                        std::optional<llvm::json::Object> metadataCarrier,
                                        llvm::json::Object &stageContract, std::string &error);

} // namespace vernon::compiler

#endif
