#ifndef VERNON_COMPILER_PROGRAM_GRAPHICS_H
#define VERNON_COMPILER_PROGRAM_GRAPHICS_H

#include "compiler_program_plan_records.h"
#include "compiler_program_storage.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <string>

namespace vernon::compiler {

struct ProgramGraphicsInterfacePlan {
    ProgramEndpointRecords endpoints;
    ProgramEndpointBindingRecords endpointBindings;
    ProgramImplementationEndpointRecords implementationEndpoints;
    ProgramAccessRecords accesses;
    ProgramVertexInputRecords vertexInputs;
    ProgramVertexOutputRecords vertexOutputs;
    ProgramFragmentInputRecords fragmentInputs;
    ProgramFragmentOutputRecords fragmentOutputs;
    std::map<int64_t, int64_t> accessByValue;
};

bool buildCanonicalGraphicsInterfaces(const llvm::json::Object &compiledReflection, const llvm::json::Array &rawValues,
                                      const std::map<std::string, int64_t> &boundValues,
                                      std::map<int64_t, ProgramLogicalResource> &resources,
                                      ProgramGraphicsInterfacePlan &plan, std::string &error);

bool buildCanonicalGraphicsOperation(
    const llvm::json::Object &node, const llvm::json::Array &rawValues, const llvm::json::Array &storages,
    const std::map<std::string, int64_t> &boundValues, std::map<int64_t, ProgramLogicalResource> &resources,
    const llvm::json::Array &fragmentOutputs, std::map<int64_t, int64_t> &accessByValue, llvm::json::Array &accesses,
    llvm::json::Array &attachmentConstraints, llvm::json::Object &operation, std::string &error);

} // namespace vernon::compiler

#endif
