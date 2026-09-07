#include "runtime_autodiff_internal.h"

#include "runtime/pipeline_metadata.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"
#include "runtime_autodiff_telemetry.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

CanonicalProgramAutodiffState *canonicalProgramAutodiff(VernonProgramExecutable *pipeline) {
    return pipeline ? &pipeline->programAutodiff : nullptr;
}

const CanonicalProgramAutodiffState *canonicalProgramAutodiff(const VernonProgramExecutable *pipeline) {
    return pipeline ? &pipeline->programAutodiff : nullptr;
}

} // namespace

namespace vernon::runtime::ad {

size_t dtypeSize(VernonDataType dtype) {
    switch (dtype) {
    case VERNON_DATA_BOOL:
    case VERNON_DATA_U8:
        return 1;
    case VERNON_DATA_F16:
        return 2;
    case VERNON_DATA_I32:
    case VERNON_DATA_U32:
    case VERNON_DATA_F32:
        return sizeof(float);
    case VERNON_DATA_F64:
        return sizeof(double);
    default:
        return 0;
    }
}

bool materializeDerivativeValueAbi(ValueAbi &derivative, const std::vector<ValueAbi> &sources) {
    const auto source = std::find_if(sources.begin(), sources.end(),
                                     [&](const ValueAbi &value) { return value.path == derivative.path; });
    if (source == sources.end())
        return false;
    const size_t sourceScalar = dtypeSize(source->dtype);
    const size_t derivativeScalar = dtypeSize(derivative.dtype);
    if (!sourceScalar || !derivativeScalar || source->byteSize % sourceScalar ||
        (source->byteSize / sourceScalar &&
         derivativeScalar > std::numeric_limits<size_t>::max() / (source->byteSize / sourceScalar)))
        return false;
    derivative.byteSize = (source->byteSize / sourceScalar) * derivativeScalar;
    derivative.logicalShape = source->logicalShape;
    return true;
}

bool appendValueLayoutAbi(const ValueLayout &layout, const std::vector<uint64_t> &shape, const std::string &rootPath,
                          std::vector<ValueAbi> &values, std::string &error) {
    if (layout.leaves.empty())
        return error = "autodiff parameter has no canonical Value layout", false;
    size_t elementCount = 1;
    bool dynamicShape = false;
    for (uint64_t extent : shape) {
        if (!extent) {
            dynamicShape = true;
            continue;
        }
        if (elementCount > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
            return error = "autodiff parameter shape overflows", false;
        elementCount *= static_cast<size_t>(extent);
    }
    for (const ValueLeaf &leaf : layout.leaves) {
        const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
        size_t bytes = dtype ? dtypeSize(*dtype) : 0;
        if (!bytes || !leaf.scalarCount ||
            bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(leaf.scalarCount))
            return error = "autodiff parameter leaf ABI is invalid", false;
        bytes *= static_cast<size_t>(leaf.scalarCount);
        if (!dynamicShape) {
            if (elementCount && bytes > std::numeric_limits<size_t>::max() / elementCount)
                return error = "autodiff parameter leaf byte size overflows", false;
            bytes *= elementCount;
        }
        std::string path = canonicalValueLeafPath(rootPath, leaf);
        std::vector<uint64_t> logicalShape = shape;
        logicalShape.insert(logicalShape.end(), leaf.shape.begin(), leaf.shape.end());
        ValueAbi candidate{std::move(path), *dtype, bytes, std::max<size_t>(layout.alignment, 1),
                           std::move(logicalShape)};
        const auto existing = std::find_if(values.begin(), values.end(),
                                           [&](const ValueAbi &value) { return value.path == candidate.path; });
        if (existing != values.end()) {
            if (!sameValueAbi(*existing, candidate))
                return error = "autodiff parameters define conflicting ABI values for '" + candidate.path + "'", false;
        } else {
            values.push_back(std::move(candidate));
        }
    }
    return true;
}

bool appendParameterValueAbi(const Parameter &parameter, const std::string &rootPath, std::vector<ValueAbi> &values,
                             std::string &error) {
    const ValueLayout &layout = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
    return appendValueLayoutAbi(layout, parameter.shape, rootPath, values, error);
}

bool validLaunchSize(VernonLaunchSize grid) { return grid.x != 0 && grid.y != 0 && grid.z != 0; }

bool invocationExtent(VernonLaunchSize grid, VernonLaunchSize workgroup, VernonLaunchSize &extent) {
    if (!validLaunchSize(grid) || !validLaunchSize(workgroup) ||
        grid.x > std::numeric_limits<uint32_t>::max() / workgroup.x ||
        grid.y > std::numeric_limits<uint32_t>::max() / workgroup.y ||
        grid.z > std::numeric_limits<uint32_t>::max() / workgroup.z)
        return false;
    extent = {grid.x * workgroup.x, grid.y * workgroup.y, grid.z * workgroup.z};
    return true;
}

bool carrierCount(VernonLaunchSize extent, size_t &count) {
    if (!validLaunchSize(extent) || extent.x > SIZE_MAX / extent.y ||
        static_cast<size_t>(extent.x) * extent.y > SIZE_MAX / extent.z)
        return false;
    count = static_cast<size_t>(extent.x) * extent.y * extent.z;
    return true;
}

bool materializeCarrierValue(ValueAbi &abi, VernonLaunchSize extent) {
    size_t count = 0;
    if (!carrierCount(extent, count) || (count && abi.byteSize > SIZE_MAX / count))
        return false;
    abi.byteSize *= count;
    if (count > 1)
        abi.logicalShape.insert(abi.logicalShape.begin(), {extent.z, extent.y, extent.x});
    return true;
}

bool derivativeAbiMatches(const ValueAbi &primal, const ValueAbi &derivative) {
    if (primal.dtype != VERNON_DATA_F16 && primal.dtype != VERNON_DATA_F32 && primal.dtype != VERNON_DATA_F64)
        return false;
    const VernonDataType expected = primal.dtype == VERNON_DATA_F64 ? VERNON_DATA_F64 : VERNON_DATA_F32;
    if (derivative.dtype != expected || derivative.logicalShape != primal.logicalShape)
        return false;
    const size_t primalScalarSize = dtypeSize(primal.dtype);
    const size_t derivativeScalarSize = dtypeSize(derivative.dtype);
    return primalScalarSize && derivativeScalarSize && primal.byteSize % primalScalarSize == 0 &&
           derivative.byteSize == primal.byteSize / primalScalarSize * derivativeScalarSize;
}

bool validateDerivativeGroupsAgainstSignature(VernonRuntimeContext &context,
                                              const std::vector<AutodiffDerivativeGroup> &groups,
                                              const Signature &signature) {
    const auto matches = [&](AutodiffDerivativeRole role, const std::vector<ValueAbi> &values) {
        std::vector<std::string> expected = autodiffDerivativeLeafPaths(groups, role);
        if (expected.size() != values.size())
            return false;
        std::vector<std::string> actual;
        actual.reserve(values.size());
        for (const ValueAbi &value : values)
            actual.push_back(value.path);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        return expected == actual;
    };
    if (!matches(AutodiffDerivativeRole::Gradient, signature.gradients)) {
        invocationDiagnostic(context) = "autodiff gradient groups do not match the executable signature";
        return false;
    }
    if (!matches(AutodiffDerivativeRole::Cotangent, signature.cotangents)) {
        invocationDiagnostic(context) = "autodiff cotangent groups do not match the executable signature";
        return false;
    }
    return true;
}

} // namespace vernon::runtime::ad

namespace vernon::runtime::program_execution {

VernonStatus forwardProgramInvocation(VernonProgramExecutable &pipeline, const VernonProgramArgument *arguments,
                                      size_t argumentCount, VernonPullback *&pullback,
                                      const ProgramInvocationContext *programContext) {
    pullback = nullptr;
    const auto *autodiff = canonicalProgramAutodiff(&pipeline);
    auto *executable = autodiff ? autodiff->canonicalExecution.get() : nullptr;
    if (!executable || (argumentCount && !arguments)) {
        invocationDiagnostic(*pipeline.context) = "invalid canonical Program invocation";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    std::unique_ptr<ad::PullbackExecution> execution;
    const ad::ForwardExecutionTarget target{arguments, argumentCount, programContext};
    const VernonStatus status = executable->forward(target, execution);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!execution)
        return VERNON_STATUS_OK;
    auto result = std::make_unique<VernonPullback>();
    result->contextLease = vernon::runtime::acquireContextLease(*pipeline.context);
    result->execution = std::move(execution);
    pullback = result.release();
    return VERNON_STATUS_OK;
}

void attachProgramSnapshot(VernonPullback &pullback, std::shared_ptr<const program::InvocationSnapshot> snapshot) {
    pullback.programSnapshot = std::move(snapshot);
}

} // namespace vernon::runtime::program_execution

namespace {

VernonStatus fail(VernonRuntimeContext *context, std::string_view message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) noexcept {
    try {
        if (context)
            vernon::runtime::invocationDiagnostic(*context) = message;
    } catch (...) {
    }
    return status;
}

} // namespace

namespace {

const vernon::runtime::program::Program *programExecution(const VernonProgramExecutable *pipeline) {
    if (!pipeline)
        return nullptr;
    const vernon::runtime::program::Program &program = pipeline->executionPlan->resolvedProgram->program;
    return vernon::runtime::program::findGraph(program, "backward") ? &program : nullptr;
}

} // namespace

vernon::runtime::AutodiffPullbackMemoryUsage
vernon::runtime::autodiffPullbackMemoryUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const ad::PullbackMemoryUsage usage = pullback->execution->memoryUsage();
    return {usage.logicalResidualBytes, usage.residentBytes, usage.allocatedBytes, usage.retainedAllocationBytes,
            usage.peakTemporaryBytes};
}

vernon::runtime::AutodiffPullbackControlPlaneUsage
vernon::runtime::autodiffPullbackControlPlaneUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const program_execution::ExecutionControlPlaneUsage usage = pullback->execution->controlPlaneUsage();
    return {usage.submissions,
            usage.waits,
            usage.readbacks,
            usage.atomicPublications,
            usage.temporaryAllocationBytes,
            usage.deviceWaitNanoseconds};
}

uint64_t vernon::runtime::ad::PullbackExecution::peakRuntimeManagedBytes() const {
    const PullbackMemoryUsage usage = memoryUsage();
    return std::max(std::max(usage.retainedAllocationBytes, usage.allocatedBytes), usage.peakTemporaryBytes);
}

vernon::runtime::AutodiffPullbackCheckpointPlan
vernon::runtime::autodiffPullbackCheckpointPlan(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    return pullback->execution->checkpointPlan();
}

std::vector<vernon::runtime::AutodiffPullbackPassTelemetry>
vernon::runtime::autodiffPullbackPassTelemetry(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    return pullback->execution->passTelemetry();
}

uint64_t vernon::runtime::autodiffPullbackPeakRuntimeManagedBytes(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return 0;
    return pullback->execution->peakRuntimeManagedBytes();
}

void vernon::runtime::autodiffSetProgramCheckpointPlan(VernonProgramExecutable *pipeline, const uint64_t *memoryBudget,
                                                       std::string_view policy) {
    if (!pipeline)
        return;
    if (memoryBudget)
        pipeline->programAutodiff.checkpointMemoryBudget = *memoryBudget;
    else
        pipeline->programAutodiff.checkpointMemoryBudget.reset();
    pipeline->programAutodiff.checkpointPolicy = std::string(policy);
}

size_t vernon::runtime::autodiffHostTapeContextLimit(const VernonRuntimeContext *context) {
    return context ? ad::autodiffMemoryContextLimit(context->autodiffMemoryPolicy) : 0;
}

VernonRhiDevice vernon::runtime::autodiffRhiDevice(const VernonRuntimeContext *context) {
    return context ? context->rhiDevice : VernonRhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

extern "C" {

uint8_t vernonRuntimeProgramExecutableHasProgramAutodiff(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    return programExecution(pipeline) ? 1 : 0;
}

size_t vernonRuntimeProgramExecutableGetAdDerivativeGroupCount(const VernonProgramExecutable *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *autodiff = canonicalProgramAutodiff(pipeline);
    return autodiff ? autodiff->derivativeGroups.size() : 0;
}

VernonStatus vernonRuntimeProgramExecutableGetAdDerivativeGroupByIndex(const VernonProgramExecutable *pipeline,
                                                                       size_t groupIndex,
                                                                       VernonAdDerivativeGroupView *view) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *autodiff = canonicalProgramAutodiff(pipeline);
    if (!autodiff || !view || view->struct_size < sizeof(*view) || groupIndex >= autodiff->derivativeGroups.size())
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff derivative group query");
    const vernon::runtime::AutodiffDerivativeGroup &group = autodiff->derivativeGroups[groupIndex];
    *view = {sizeof(*view),
             group.role == vernon::runtime::AutodiffDerivativeRole::Gradient ? VERNON_AD_DERIVATIVE_GRADIENT
                                                                             : VERNON_AD_DERIVATIVE_COTANGENT,
             {group.declaredPath.data(), group.declaredPath.size()},
             group.leafPaths.size(),
             {}};
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeProgramExecutableGetAdDerivativeGroupLeaf(const VernonProgramExecutable *pipeline,
                                                                    size_t groupIndex, size_t leafIndex,
                                                                    VernonStringView *leafPath) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *autodiff = canonicalProgramAutodiff(pipeline);
    if (!autodiff || !leafPath || groupIndex >= autodiff->derivativeGroups.size()) {
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff derivative group leaf query");
    }
    const vernon::runtime::AutodiffDerivativeGroup &group = autodiff->derivativeGroups[groupIndex];
    if (leafIndex >= group.leafPaths.size())
        return fail(pipeline->context, "invalid autodiff derivative group leaf query");
    const std::string &path = group.leafPaths[leafIndex];
    *leafPath = {path.data(), path.size()};
    return VERNON_STATUS_OK;
}

VernonStatus vernonProgramPullbackApplyWithOptions(VernonPullback *pullback, const VernonProgramArgument *arguments,
                                                   size_t argumentCount, const VernonPullbackApplyOptions *options) {
    VernonRuntimeContext *context = pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr;
    try {
        using namespace vernon::runtime::ad;
        vernon::runtime::RuntimeDiagnosticScope diagnostic(context);
        if (!pullback || !pullback->execution || (argumentCount && !arguments) || !options ||
            options->struct_size != sizeof(VernonPullbackApplyOptions) ||
            options->abi_version != VERNON_PULLBACK_APPLY_OPTIONS_VERSION ||
            std::any_of(std::begin(options->reserved), std::end(options->reserved),
                        [](uint32_t value) { return value != 0; }))
            return fail(context, "invalid pullback invocation");
        const auto boundedSize = [](uint64_t value) {
            return value > std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max()
                                                              : static_cast<size_t>(value);
        };
        const PullbackApplyOptions runtimeOptions{boundedSize(options->maximum_temporary_bytes)};
        return pullback->execution->apply(arguments, argumentCount, runtimeOptions);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate pullback state", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "pullback allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        return fail(context, "unexpected pullback failure", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus vernonProgramPullbackApply(VernonPullback *pullback, const VernonProgramArgument *arguments,
                                        size_t argumentCount) {
    const VernonPullbackApplyOptions options{sizeof(VernonPullbackApplyOptions),
                                             VERNON_PULLBACK_APPLY_OPTIONS_VERSION,
                                             std::numeric_limits<uint64_t>::max(),
                                             {}};
    return vernonProgramPullbackApplyWithOptions(pullback, arguments, argumentCount, &options);
}

void vernonProgramPullbackDestroy(VernonPullback *pullback) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(
        pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr);
    delete pullback;
}

} // extern "C"
