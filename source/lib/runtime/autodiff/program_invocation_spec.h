#ifndef VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_SPEC_H
#define VERNON_RUNTIME_AUTODIFF_PROGRAM_INVOCATION_SPEC_H

#include "VernonRuntime.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace vernon::runtime::ad {

class HostStaticTapeBatch;
struct PendingProgramPublication;
struct ValueAbi;

struct ProgramLeafBinding {
    uint32_t value{UINT32_MAX};
    size_t byteOffset{};
    size_t elementStride{};
    size_t leafElementBytes{};
    size_t elementCount{};
};

struct ProgramLeafFrameSource {
    const VernonAdValueSet *values{};
    const std::vector<ValueAbi> *signature{};
    const std::vector<ProgramLeafBinding> *bindings{};
};

struct CanonicalForwardBindings {
    const VernonPipelineInvocation &invocation;
    const std::vector<std::pair<uint32_t, uint32_t>> &valueBySlot;
    std::vector<PendingProgramPublication> &publications;
};

struct HostForwardBindings {
    ProgramLeafFrameSource inputs;
    ProgramLeafFrameSource outputs;
};

struct ForwardInvocationSpec {
    const std::vector<char> &requiredValues;
    std::variant<CanonicalForwardBindings, HostForwardBindings> bindings;
};

struct PullbackInvocationSpec {
    const std::vector<char> &requiredValues;
    ProgramLeafFrameSource cotangents;
    ProgramLeafFrameSource gradients;
    const std::vector<std::vector<uint8_t>> &captures;
    const std::vector<std::vector<uint64_t>> &captureShapes;
    const std::vector<std::shared_ptr<HostStaticTapeBatch>> *tapeCaptures{};
};

} // namespace vernon::runtime::ad

#endif
