#ifndef VERNON_COMPILER_PROGRAM_PLAN_RECORDS_H
#define VERNON_COMPILER_PROGRAM_PLAN_RECORDS_H

#include "llvm/Support/JSON.h"

#include <utility>

namespace vernon::compiler {

template <typename Tag> class ProgramPlanRecords {
public:
    template <typename Value> void emplace_back(Value &&value) { records_.emplace_back(std::forward<Value>(value)); }

    bool empty() const { return records_.empty(); }
    llvm::json::Array &json() { return records_; }
    const llvm::json::Array &json() const { return records_; }
    llvm::json::Array take() { return std::move(records_); }

private:
    llvm::json::Array records_;
};

struct ProgramEndpointRecordTag;
struct ProgramEndpointBindingRecordTag;
struct ProgramImplementationEndpointRecordTag;
struct ProgramAccessRecordTag;
struct ProgramVertexInputRecordTag;
struct ProgramVertexOutputRecordTag;
struct ProgramFragmentInputRecordTag;
struct ProgramFragmentOutputRecordTag;

using ProgramEndpointRecords = ProgramPlanRecords<ProgramEndpointRecordTag>;
using ProgramEndpointBindingRecords = ProgramPlanRecords<ProgramEndpointBindingRecordTag>;
using ProgramImplementationEndpointRecords = ProgramPlanRecords<ProgramImplementationEndpointRecordTag>;
using ProgramAccessRecords = ProgramPlanRecords<ProgramAccessRecordTag>;
using ProgramVertexInputRecords = ProgramPlanRecords<ProgramVertexInputRecordTag>;
using ProgramVertexOutputRecords = ProgramPlanRecords<ProgramVertexOutputRecordTag>;
using ProgramFragmentInputRecords = ProgramPlanRecords<ProgramFragmentInputRecordTag>;
using ProgramFragmentOutputRecords = ProgramPlanRecords<ProgramFragmentOutputRecordTag>;

} // namespace vernon::compiler

#endif
