#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_SPEC_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_SPEC_H

#include "VernonRuntime.h"
#include "runtime/program_execution/program_invocation_state.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace vernon::runtime::program_execution {
class PublicationTransaction;
}

namespace vernon::runtime::ad {

class HostStaticTapeBatch;
struct CanonicalBoundaryBindings {
    const VernonProgramArgument *arguments{};
    size_t argumentCount{};
    const std::vector<std::pair<uint32_t, uint32_t>> &valueBySlot;
    program_execution::PublicationTransaction &publication;
};

struct ForwardInvocationSpec {
    const std::vector<char> &requiredValues;
    CanonicalBoundaryBindings bindings;
};

struct PullbackInvocationSpec {
    const std::vector<char> &requiredValues;
    CanonicalBoundaryBindings bindings;
    const std::vector<std::vector<uint8_t>> &captures;
    const std::vector<std::vector<uint64_t>> &captureShapes;
    const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures{};
    const std::vector<program_execution::CanonicalValueSnapshot> *retainedSnapshots{};
};

} // namespace vernon::runtime::ad

#endif
