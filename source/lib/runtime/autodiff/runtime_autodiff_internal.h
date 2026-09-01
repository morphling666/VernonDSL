#ifndef VERNON_RUNTIME_RUNTIME_AUTODIFF_INTERNAL_H
#define VERNON_RUNTIME_RUNTIME_AUTODIFF_INTERNAL_H

#include "VernonRuntime.h"
#include "runtime/autodiff/runtime_autodiff_policy.h"
#include "runtime/autodiff/runtime_direct_autodiff.h"
#include "runtime/autodiff/runtime_forward_plan.h"
#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::execution::detail {
class RhiCommandPlanSink;
struct RhiCommandExecutionPlan;
} // namespace vernon::execution::detail

namespace vernon::runtime {
class ContextLease;
}

namespace vernon::runtime::ad {

struct ValueAbi {
    std::string path;
    VernonDataType dtype{};
    size_t byteSize{};
    size_t alignment{1};
    std::vector<uint64_t> logicalShape;
};

inline std::string canonicalValueLeafPath(const std::string &root, const ValueLeaf &leaf) {
    std::string result = root;
    for (const ValuePathComponent &component : leaf.path) {
        if (!result.empty())
            result.push_back('.');
        result += component.field ? *component.field : std::to_string(component.index);
    }
    return result;
}

inline bool sameValueAbi(const ValueAbi &left, const ValueAbi &right) {
    return left.dtype == right.dtype && left.byteSize == right.byteSize && left.alignment == right.alignment &&
           left.logicalShape == right.logicalShape;
}

struct Signature {
    std::vector<ValueAbi> inputs;
    std::vector<ValueAbi> outputs;
    std::vector<ValueAbi> tape;
    std::vector<ValueAbi> cotangents;
    std::vector<ValueAbi> gradients;
    bool storageObjectives{};
};

struct PullbackMemoryUsage {
    size_t logicalResidualBytes{};
    size_t residentBytes{};
    size_t allocatedBytes{};
    size_t retainedAllocationBytes{};
    size_t peakTemporaryBytes{};
};

struct PullbackControlPlaneUsage {
    uint64_t submissions{};
    uint64_t waits{};
    uint64_t readbacks{};
    uint64_t atomicPublications{};
    uint64_t temporaryAllocationBytes{};
    uint64_t deviceWaitNanoseconds{};
};

struct PullbackApplyOptions {
    size_t maximumTemporaryBytes{std::numeric_limits<size_t>::max()};
    size_t maximumReusableConstructionBytes{};
};

struct ForwardExecutionTarget {
    VernonRuntimeProviderObject encoder{};
    const VernonPipelineInvocation *invocation{};
    execution::detail::RhiCommandExecutionPlan *commandPlan{};

    bool externalEncoder() const { return encoder.value != 0; }
    bool deferredCommandPlan() const { return commandPlan != nullptr; }
    bool encodedInvocation() const { return externalEncoder() || deferredCommandPlan(); }
};

class PullbackExecution {
public:
    virtual ~PullbackExecution() = default;
    virtual VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients,
                               const PullbackApplyOptions &options) = 0;
    virtual PullbackMemoryUsage memoryUsage() const = 0;
    virtual PullbackControlPlaneUsage controlPlaneUsage() const { return {}; }
    virtual AutodiffPullbackCheckpointPlan checkpointPlan() const { return {}; }
    virtual std::vector<AutodiffPullbackPassTelemetry> passTelemetry() const { return {}; }
    virtual uint64_t peakRuntimeManagedBytes() const;
};

class DevicePullbackExecution : public PullbackExecution {
public:
    virtual VernonStatus applyDevice(const VernonAdDeviceValueSet *cotangents, VernonAdDeviceValueSet &gradients,
                                     const PullbackApplyOptions &options,
                                     execution::detail::RhiCommandPlanSink *sink = nullptr) = 0;
};

class Executable {
public:
    virtual ~Executable() = default;
    virtual const Signature &signature() const = 0;
    virtual VernonStatus forward(const ForwardExecutionTarget &target, VernonLaunchSize computeGrid,
                                 const VernonAdValueSet &inputs, VernonAdValueSet *outputs,
                                 std::unique_ptr<PullbackExecution> &pullback) = 0;
};

size_t dtypeSize(VernonDataType dtype);
bool materializeDerivativeValueAbi(ValueAbi &derivative, const std::vector<ValueAbi> &sources);
bool appendParameterValueAbi(const Parameter &parameter, const std::string &rootPath, std::vector<ValueAbi> &values,
                             std::string &error);
bool validLaunchSize(VernonLaunchSize grid);
bool invocationExtent(VernonLaunchSize grid, VernonLaunchSize workgroup, VernonLaunchSize &extent);
bool carrierCount(VernonLaunchSize extent, size_t &count);
bool materializeCarrierValue(ValueAbi &abi, VernonLaunchSize extent);
bool validSet(const VernonAdValueSet *set, bool required);
VernonAdValue *findValue(VernonAdValueSet &set, const std::string &path);
const VernonAdValue *findValue(const VernonAdValueSet &set, const std::string &path);
bool valueMatches(const VernonAdValue &value, const ValueAbi &abi);
bool derivativeAbiMatches(const ValueAbi &primal, const ValueAbi &derivative);
bool makeCotangentBytes(const VernonAdValueSet *cotangents, const ValueAbi &abi, std::vector<uint8_t> &bytes,
                        std::string &error);
bool validateDerivativeGroupsAgainstSignature(VernonRuntimeContext &context,
                                              const std::vector<AutodiffDerivativeGroup> &groups,
                                              const Signature &signature);

bool createCpuExecutable(VernonRuntimeContext &context, const Stage &primal, const Stage &forward,
                         const Stage &backward, const std::vector<std::string> &gradientPaths,
                         uint64_t staticTapeBytesHint, const std::string &residualStorage,
                         const std::string &selectedPolicy, bool wholeDispatchRetentionPermitted,
                         std::shared_ptr<Executable> &executable);
bool createCpuEntryExecutable(VernonRuntimeContext &context, VernonCpuEntryPoint primalEntry,
                              VernonStringView primalReflection, VernonStringView primalName,
                              VernonCpuEntryPoint forwardEntry, VernonStringView forwardReflection,
                              VernonStringView forwardName, VernonCpuEntryPoint backwardEntry,
                              VernonStringView backwardReflection, VernonStringView backwardName,
                              const std::vector<std::string> &gradientPaths, uint64_t staticTapeBytesHint,
                              const std::string &residualStorage, const std::string &selectedPolicy,
                              bool wholeDispatchRetentionPermitted, std::shared_ptr<Executable> &executable);
bool createGpuExecutable(VernonRuntimeContext &context, const Stage &primal, const Stage &forward,
                         const Stage &backward, const std::vector<std::string> &gradientPaths,
                         uint64_t staticTapeBytesHint, const std::string &residualStorage,
                         const std::string &selectedPolicy, std::shared_ptr<Executable> &executable);
bool resolvePipelineAutodiff(VernonPipelineBundle &bundle, const AutodiffProfile &profile,
                             VernonLoadedPipeline &pipeline);
bool resolveProgramAutodiff(VernonLoadedPipeline &pipeline,
                            const std::vector<AutodiffDerivativeGroup> &derivativeGroups);

} // namespace vernon::runtime::ad

struct VernonPullback {
    std::shared_ptr<vernon::runtime::ContextLease> contextLease;
    std::unique_ptr<vernon::runtime::ad::PullbackExecution> execution;
};

#endif
