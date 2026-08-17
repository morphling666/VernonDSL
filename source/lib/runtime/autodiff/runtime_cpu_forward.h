#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_FORWARD_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_FORWARD_H

#include "runtime_cpu_preparation.h"

#include <functional>

namespace vernon::runtime::ad::cpu {

struct ForwardInvocationFrame {
    uint8_t *arguments{};
    uint8_t *results{};
};

using CpuForwardInvoke =
    std::function<VernonStatus(VernonCpuRangeV1 &, std::vector<ForwardInvocationFrame> &, std::string &)>;
using CpuForwardFinish = std::function<VernonStatus(size_t, Signature, RuntimeTensorShapes, RetainedPrimalTensorViews,
                                                    std::unique_ptr<PullbackExecution> &)>;

VernonStatus runCpuForward(VernonRuntimeContext &context, const HostProfileLayout &layout, const Signature &signature,
                           VernonLaunchSize computeGrid, const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                           const std::unordered_set<std::string> &requiredPrimalTensorOwners,
                           std::unique_ptr<PullbackExecution> &pullback, CpuForwardInvoke invoke,
                           CpuForwardFinish finish, std::optional<size_t> groupLinear = std::nullopt,
                           bool compactAllGroups = false);
std::unordered_set<std::string> requiredPrimalTensorOwners(const HostProfileLayout &backwardLayout);
VernonStatus retainRequiredPrimalLeaves(VernonRuntimeContext &context, const HostProfileLayout &backwardLayout,
                                        const VernonAdValueSet &inputs, RetainedPrimalLeaves &retainedPrimals);

} // namespace vernon::runtime::ad::cpu

#endif
