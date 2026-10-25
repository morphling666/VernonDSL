#ifndef VERNON_RUNTIME_PROGRAM_INVOCATION_CONTEXT_H
#define VERNON_RUNTIME_PROGRAM_INVOCATION_CONTEXT_H

#include <cstdint>
#include <optional>
#include <string>

namespace vernon::runtime::program {
class InvocationSnapshot;
}
namespace vernon::runtime::ad {
class CanonicalProgramExecutionWorkspace;
}

namespace vernon::runtime {

struct RecordedGraphicsPasses;
class InstanceGroups;
class StoredGraphicsScopes;

struct ProgramInvocationContext {
    const program::InvocationSnapshot &bindings;
    const std::optional<uint64_t> &checkpointMemoryBudget;
    const std::string &checkpointPolicy;
    ad::CanonicalProgramExecutionWorkspace *executionWorkspace{};
    // Validated recordings by graphics Node; the executor also writes their telemetry here.
    RecordedGraphicsPasses *recordedGraphics{};
    // The instance's persistent groups, which its bindings in `bindings` decide.
    InstanceGroups *groups{};
    // The instance's stored rendering scopes, which its invocations replay while their inputs are unchanged.
    StoredGraphicsScopes *storedGraphics{};
};

} // namespace vernon::runtime

#endif
