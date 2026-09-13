#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_BOUNDARY_BINDER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_BOUNDARY_BINDER_H

#include "VernonRuntime.h"
#include "runtime/program_execution/program_invocation_state.h"
#include "runtime/program_execution/publication_transaction.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime::program {
struct Program;
struct ResolvedExecutionPlan;
} // namespace vernon::runtime::program

namespace vernon::runtime::ad {

struct ProgramBoundaryBindingRequest {
    const VernonProgramArgument *arguments{};
    size_t argumentCount{};
    const std::vector<std::pair<uint32_t, uint32_t>> &valueBySlot;
    program_execution::PublicationTransaction *publication{};
};

struct ProgramBoundaryBindingScratch {
    struct TensorOwner {
        program::ProgramOwnerId owner;
        const VernonTensorView *tensor{};
    };

    void reset(const program::Program &program);
    size_t ownerIndex(program::ProgramOwnerId owner) const;

    std::vector<const VernonProgramArgument *> bySlot;
    std::vector<std::optional<VernonProgramArgument>> externalValues;
    std::vector<std::optional<VernonProgramArgument>> ownerBindings;
    std::vector<TensorOwner> tensorOwners;
    std::vector<size_t> stagedOwners;
    std::vector<char> inPlaceOwners;
    std::vector<char> liveValues;
    std::vector<char> liveStorages;
    size_t valueCount{};
};

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const program::ResolvedExecutionPlan &plan, const ProgramBoundaryBindingRequest &request,
                           ProgramBoundaryBindingScratch &scratch,
                           std::map<uint32_t, program_execution::ProgramStorageBacking> &backings,
                           std::vector<char> &live, std::string &error);

} // namespace vernon::runtime::ad

#endif
