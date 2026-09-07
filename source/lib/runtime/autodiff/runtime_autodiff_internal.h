#ifndef VERNON_RUNTIME_RUNTIME_AUTODIFF_INTERNAL_H
#define VERNON_RUNTIME_RUNTIME_AUTODIFF_INTERNAL_H

#include "VernonRuntime.h"
#include "runtime/autodiff/runtime_autodiff_telemetry.h"
#include "runtime/program_execution/execution_control_plane.h"
#include "runtime/stage_artifact.h"
#include "runtime/stage_binding_plan.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime {
class ContextLease;
struct ProgramInvocationContext;
} // namespace vernon::runtime
namespace vernon::runtime::program {
class InvocationSnapshot;
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

struct PullbackApplyOptions {
    size_t maximumTemporaryBytes{std::numeric_limits<size_t>::max()};
};

struct ForwardExecutionTarget {
    const VernonProgramArgument *arguments{};
    size_t argumentCount{};
    const ProgramInvocationContext *programContext{};
};

class PullbackExecution {
public:
    virtual ~PullbackExecution() = default;
    virtual VernonStatus apply(const VernonProgramArgument *arguments, size_t argumentCount,
                               const PullbackApplyOptions &options) = 0;
    virtual PullbackMemoryUsage memoryUsage() const = 0;
    virtual program_execution::ExecutionControlPlaneUsage controlPlaneUsage() const { return {}; }
    virtual AutodiffPullbackCheckpointPlan checkpointPlan() const { return {}; }
    virtual std::vector<AutodiffPullbackPassTelemetry> passTelemetry() const { return {}; }
    virtual uint64_t peakRuntimeManagedBytes() const;
};

class CanonicalProgramExecution {
public:
    virtual ~CanonicalProgramExecution() = default;
    virtual const Signature &signature() const = 0;
    virtual VernonStatus forward(const ForwardExecutionTarget &target,
                                 std::unique_ptr<PullbackExecution> &pullback) = 0;
};

size_t dtypeSize(VernonDataType dtype);
bool materializeDerivativeValueAbi(ValueAbi &derivative, const std::vector<ValueAbi> &sources);
bool appendValueLayoutAbi(const ValueLayout &layout, const std::vector<uint64_t> &shape, const std::string &rootPath,
                          std::vector<ValueAbi> &values, std::string &error);
bool appendParameterValueAbi(const Parameter &parameter, const std::string &rootPath, std::vector<ValueAbi> &values,
                             std::string &error);
bool validLaunchSize(VernonLaunchSize grid);
bool invocationExtent(VernonLaunchSize grid, VernonLaunchSize workgroup, VernonLaunchSize &extent);
bool carrierCount(VernonLaunchSize extent, size_t &count);
bool materializeCarrierValue(ValueAbi &abi, VernonLaunchSize extent);
bool derivativeAbiMatches(const ValueAbi &primal, const ValueAbi &derivative);
bool validateDerivativeGroupsAgainstSignature(VernonRuntimeContext &context,
                                              const std::vector<AutodiffDerivativeGroup> &groups,
                                              const Signature &signature);

bool resolveProgramAutodiff(VernonProgramExecutable &pipeline,
                            const std::vector<AutodiffDerivativeGroup> &derivativeGroups);

} // namespace vernon::runtime::ad

struct VernonPullback {
    std::shared_ptr<vernon::runtime::ContextLease> contextLease;
    std::unique_ptr<vernon::runtime::ad::PullbackExecution> execution;
    std::shared_ptr<const vernon::runtime::program::InvocationSnapshot> programSnapshot;
};

#endif
