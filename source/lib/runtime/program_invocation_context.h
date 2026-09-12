#ifndef VERNON_RUNTIME_PROGRAM_INVOCATION_CONTEXT_H
#define VERNON_RUNTIME_PROGRAM_INVOCATION_CONTEXT_H

#include <cstdint>
#include <optional>
#include <string>

namespace vernon::runtime::program {
class InvocationSnapshot;
}

namespace vernon::runtime {

struct ProgramInvocationContext {
    const program::InvocationSnapshot &bindings;
    const std::optional<uint64_t> &checkpointMemoryBudget;
    const std::string &checkpointPolicy;
};

} // namespace vernon::runtime

#endif
