#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_BOUNDARY_BINDER_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_BOUNDARY_BINDER_H

#include "VernonRuntime.h"
#include "runtime/autodiff/program_publication.h"

#include <cstdint>
#include <map>
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

struct ProgramStorageState {
    uint32_t owner{UINT32_MAX};
    size_t bytes{};
    bool sized{};
    std::optional<VernonProgramArgument> external;
    std::optional<VernonProgramArgument> initial;
};

struct ProgramBoundaryBindingRequest {
    const VernonStageInvocationDescriptor &invocation;
    const std::vector<std::pair<uint32_t, uint32_t>> &valueBySlot;
    std::vector<PendingProgramPublication> *publications{};
};

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const program::ResolvedExecutionPlan &plan, const ProgramBoundaryBindingRequest &request,
                           std::map<uint32_t, VernonProgramArgument> &externalValues,
                           std::map<uint32_t, ProgramStorageState> &backings, std::vector<char> &live,
                           std::string &error);

} // namespace vernon::runtime::ad

#endif
