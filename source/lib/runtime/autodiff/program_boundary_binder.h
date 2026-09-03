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
}

namespace vernon::runtime::ad {

struct ProgramStorageBacking {
    uint32_t owner{UINT32_MAX};
    size_t bytes{};
    bool sized{};
    std::optional<VernonPipelineArgument> external;
};

struct ProgramBoundaryBindingRequest {
    const VernonPipelineInvocation &invocation;
    const std::vector<std::pair<uint32_t, uint32_t>> &valueBySlot;
    std::vector<PendingProgramPublication> *publications{};
};

bool bindProgramBoundaries(VernonRuntimeContext &context, const program::Program &execution,
                           const ProgramBoundaryBindingRequest &request,
                           std::map<uint32_t, VernonPipelineArgument> &externalValues,
                           std::map<uint32_t, ProgramStorageBacking> &backings, std::vector<char> &live,
                           std::string &error);

} // namespace vernon::runtime::ad

#endif
