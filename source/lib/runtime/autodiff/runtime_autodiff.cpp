#include "runtime_autodiff_internal.h"

#include "host_tape_allocator.h"
#include "runtime/runtime_state.h"
#include "runtime_direct_autodiff.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

bool validSet(const VernonAdValueSet *set, bool required) {
    return (!required && !set) ||
           (set && set->struct_size >= sizeof(VernonAdValueSet) && (!set->value_count || set->values));
}

VernonAdValue *findValue(VernonAdValueSet &set, const std::string &path) {
    for (size_t index = 0; index < set.value_count; ++index) {
        VernonAdValue &value = set.values[index];
        if (value.struct_size >= sizeof(VernonAdValue) && value.path.data && value.path.size == path.size() &&
            std::memcmp(value.path.data, path.data(), path.size()) == 0)
            return &value;
    }
    return nullptr;
}

const VernonAdValue *findValue(const VernonAdValueSet &set, const std::string &path) {
    for (size_t index = 0; index < set.value_count; ++index) {
        const VernonAdValue &value = set.values[index];
        if (value.struct_size >= sizeof(VernonAdValue) && value.path.data && value.path.size == path.size() &&
            std::memcmp(value.path.data, path.data(), path.size()) == 0)
            return &value;
    }
    return nullptr;
}

bool valueMatches(const VernonAdValue &value, const ValueAbi &abi) {
    if (value.dtype != abi.dtype || !value.data || value.size != abi.byteSize ||
        value.rank != abi.logicalShape.size() || (value.rank && !value.shape))
        return false;
    return value.rank == 0 || std::equal(abi.logicalShape.begin(), abi.logicalShape.end(), value.shape);
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

bool makeCotangentBytes(const VernonAdValueSet *cotangents, const ValueAbi &abi, std::vector<uint8_t> &bytes,
                        std::string &error) {
    bytes.resize(abi.byteSize);
    if (cotangents) {
        if (cotangents->value_count != 1) {
            error = "pullback requires one output cotangent";
            return false;
        }
        const VernonAdValue *cotangent = findValue(*cotangents, abi.path);
        if (!cotangent || !valueMatches(*cotangent, abi)) {
            error = "output cotangent does not match backward reflection";
            return false;
        }
        std::memcpy(bytes.data(), cotangent->data, abi.byteSize);
        return true;
    }
    if (abi.byteSize != dtypeSize(abi.dtype)) {
        error = "a non-scalar output requires an explicit cotangent";
        return false;
    }
    if (abi.dtype == VERNON_DATA_F32) {
        const float one = 1.0f;
        std::memcpy(bytes.data(), &one, sizeof(one));
        return true;
    }
    if (abi.dtype == VERNON_DATA_F64) {
        const double one = 1.0;
        std::memcpy(bytes.data(), &one, sizeof(one));
        return true;
    }
    error = "implicit cotangents require an f32 or f64 scalar output";
    return false;
}

bool validateDerivativeGroupsAgainstSignature(VernonRuntimeContext &context,
                                              const std::vector<AutodiffDerivativeGroup> &groups,
                                              const Signature &signature) {
    const auto matches = [&](AutodiffDerivativeRole role, const std::vector<ValueAbi> &values) {
        const std::vector<std::string> expected = autodiffDerivativeLeafPaths(groups, role);
        if (expected.size() != values.size())
            return false;
        for (size_t index = 0; index < values.size(); ++index)
            if (expected[index] != values[index].path)
                return false;
        return true;
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

bool resolvePipelineAutodiff(VernonPipelineBundle &bundle, const AutodiffProfile &profiles,
                             VernonLoadedPipeline &pipeline) {
    if (!bundle.context || !bundle.autodiff)
        return false;
    const Stage &primal = bundle.stages.at(profiles.primal);
    const Stage &forward = bundle.stages.at(profiles.forwardWithTape);
    const Stage &backward = bundle.stages.at(profiles.backward);
    const std::vector<std::string> gradientPaths =
        autodiffDerivativeLeafPaths(bundle.autodiff->derivativeGroups, AutodiffDerivativeRole::Gradient);
    std::shared_ptr<Executable> executable;
    if (bundle.context->backend != VERNON_RUNTIME_CPU) {
        invocationDiagnostic(*bundle.context) = "GPU autodiff is unsupported; runtime only supports CPU autodiff";
        return false;
    }
    const bool resolved = createCpuExecutable(
        *bundle.context, primal, forward, backward, gradientPaths, profiles.staticTapeBytesHint,
        profiles.residualStorage, profiles.selectedPolicy, profiles.wholeDispatchRetentionPermitted, executable);
    if (!resolved)
        return false;
    if (!validateDerivativeGroupsAgainstSignature(*bundle.context, bundle.autodiff->derivativeGroups,
                                                  executable->signature()))
        return false;
    pipeline.autodiff = VernonLoadedAutodiff{std::move(executable), bundle.autodiff->derivativeGroups};
    return true;
}

} // namespace vernon::runtime::ad

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

const vernon::runtime::ad::Executable *autodiffExecutable(const VernonLoadedPipeline *pipeline) {
    return pipeline && pipeline->autodiff ? pipeline->autodiff->executable.get() : nullptr;
}

} // namespace

namespace {

bool copyMetadata(const std::vector<vernon::runtime::ad::ValueAbi> &values, size_t index,
                  VernonAdValueMetadataView *metadata) {
    if (!metadata || metadata->struct_size < sizeof(*metadata) || index >= values.size())
        return false;
    const vernon::runtime::ad::ValueAbi &value = values[index];
    *metadata = {sizeof(*metadata),
                 {value.path.data(), value.path.size()},
                 value.dtype,
                 static_cast<uint32_t>(value.logicalShape.size()),
                 value.logicalShape.empty() ? nullptr : value.logicalShape.data(),
                 {}};
    return true;
}

} // namespace

bool vernon::runtime::hasAutodiffStorageObjectives(const VernonLoadedPipeline *pipeline) {
    const ad::Executable *executable = autodiffExecutable(pipeline);
    return executable && executable->signature().storageObjectives;
}

VernonLaunchSize vernon::runtime::autodiffWorkgroupSize(const VernonLoadedPipeline *pipeline) {
    return pipeline ? pipeline->workgroupSize : VernonLaunchSize{};
}

std::vector<vernon::runtime::AutodiffWriteFootprint>
vernon::runtime::autodiffWriteFootprints(const VernonLoadedPipeline *pipeline) {
    std::vector<AutodiffWriteFootprint> result;
    if (!pipeline)
        return result;
    result.reserve(pipeline->writeFootprints.size());
    for (const TensorViewWriteFootprint &footprint : pipeline->writeFootprints)
        result.push_back({footprint.owner, footprint.wholeView, footprint.indices});
    return result;
}

std::vector<vernon::runtime::AutodiffWriteFootprint>
vernon::runtime::autodiffReadFootprints(const VernonLoadedPipeline *pipeline) {
    std::vector<AutodiffWriteFootprint> result;
    if (!pipeline)
        return result;
    result.reserve(pipeline->readFootprints.size());
    for (const TensorViewWriteFootprint &footprint : pipeline->readFootprints)
        result.push_back({footprint.owner, footprint.wholeView, footprint.indices});
    return result;
}

vernon::runtime::AutodiffPullbackMemoryUsage
vernon::runtime::autodiffPullbackMemoryUsage(const VernonPullback *pullback) {
    if (!pullback || !pullback->execution)
        return {};
    const ad::PullbackMemoryUsage usage = pullback->execution->memoryUsage();
    return {usage.logicalResidualBytes, usage.residentBytes, usage.allocatedBytes, usage.retainedAllocationBytes,
            usage.peakTemporaryBytes};
}

size_t vernon::runtime::autodiffHostTapeContextLimit(const VernonRuntimeContext *context) {
    return context && context->cpuTapePolicy ? context->cpuTapePolicy->contextLimit() : 0;
}

extern "C" {

size_t vernonRuntimeLoadedPipelineGetAdOutputCount(const VernonLoadedPipeline *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().outputs.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdOutputByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                           VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().outputs, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff output query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeLoadedPipelineGetAdCotangentCount(const VernonLoadedPipeline *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().cotangents.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdCotangentByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                              VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().cotangents, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff cotangent query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeLoadedPipelineGetAdGradientCount(const VernonLoadedPipeline *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().gradients.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdGradientByIndex(const VernonLoadedPipeline *pipeline, size_t index,
                                                             VernonAdValueMetadataView *metadata) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !copyMetadata(executable->signature().gradients, index, metadata))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff gradient query");
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeLoadedPipelineGetAdDerivativeGroupCount(const VernonLoadedPipeline *pipeline) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    return pipeline && pipeline->autodiff ? pipeline->autodiff->derivativeGroups.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdDerivativeGroupByIndex(const VernonLoadedPipeline *pipeline,
                                                                    size_t groupIndex,
                                                                    VernonAdDerivativeGroupView *view) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !pipeline->autodiff || !view || view->struct_size < sizeof(*view) ||
        groupIndex >= pipeline->autodiff->derivativeGroups.size())
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff derivative group query");
    const vernon::runtime::AutodiffDerivativeGroup &group = pipeline->autodiff->derivativeGroups[groupIndex];
    *view = {sizeof(*view),
             group.role == vernon::runtime::AutodiffDerivativeRole::Gradient ? VERNON_AD_DERIVATIVE_GRADIENT
                                                                             : VERNON_AD_DERIVATIVE_COTANGENT,
             {group.declaredPath.data(), group.declaredPath.size()},
             group.leafPaths.size(),
             {}};
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdDerivativeGroupLeaf(const VernonLoadedPipeline *pipeline,
                                                                 size_t groupIndex, size_t leafIndex,
                                                                 VernonStringView *leafPath) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
    if (!pipeline || !pipeline->autodiff || !leafPath || groupIndex >= pipeline->autodiff->derivativeGroups.size()) {
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff derivative group leaf query");
    }
    const vernon::runtime::AutodiffDerivativeGroup &group = pipeline->autodiff->derivativeGroups[groupIndex];
    if (leafIndex >= group.leafPaths.size())
        return fail(pipeline->context, "invalid autodiff derivative group leaf query");
    const std::string &path = group.leafPaths[leafIndex];
    *leafPath = {path.data(), path.size()};
    return VERNON_STATUS_OK;
}

VernonStatus vernonAdPipelineForward(VernonLoadedPipeline *pipeline, VernonLaunchSize computeGrid,
                                     const VernonAdValueSet *inputs, VernonAdValueSet *outputs,
                                     VernonPullback **pullback) {
    try {
        using namespace vernon::runtime::ad;
        vernon::runtime::RuntimeDiagnosticScope diagnostic(pipeline ? pipeline->context : nullptr);
        if (pullback)
            *pullback = nullptr;
        auto *executable = pipeline && pipeline->autodiff ? pipeline->autodiff->executable.get() : nullptr;
        if (!pipeline || !pullback || !executable || !validLaunchSize(computeGrid) || !validSet(inputs, true) ||
            !validSet(outputs, true))
            return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff forward invocation");
        auto result = std::make_unique<VernonPullback>();
        result->contextLease = vernon::runtime::acquireContextLease(*pipeline->context);
        std::unique_ptr<PullbackExecution> execution;
        VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT;
        if (auto *host = dynamic_cast<HostExecutable *>(executable))
            status = host->forward(computeGrid, *inputs, *outputs, execution);
        if (status != VERNON_STATUS_OK)
            return status;
        if (!execution)
            return fail(pipeline->context, "autodiff forward produced no pullback", VERNON_STATUS_INTERNAL_ERROR);
        result->execution = std::move(execution);
        *pullback = result.release();
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(pipeline ? pipeline->context : nullptr, "cannot allocate autodiff forward state",
                    VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(pipeline ? pipeline->context : nullptr, "autodiff forward allocation is too large",
                    VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        return fail(pipeline ? pipeline->context : nullptr, "unexpected autodiff forward failure",
                    VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus vernonPullbackApplyWithOptions(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                            VernonAdValueSet *gradients, const VernonPullbackApplyOptions *options) {
    VernonRuntimeContext *context = pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr;
    try {
        using namespace vernon::runtime::ad;
        vernon::runtime::RuntimeDiagnosticScope diagnostic(context);
        if (!pullback || !pullback->execution || !validSet(cotangents, false) || !validSet(gradients, true) ||
            !options || options->struct_size != sizeof(VernonPullbackApplyOptions) ||
            options->abi_version != VERNON_PULLBACK_APPLY_OPTIONS_VERSION ||
            std::any_of(std::begin(options->reserved), std::end(options->reserved),
                        [](uint32_t value) { return value != 0; }))
            return fail(context, "invalid pullback invocation");
        const auto boundedSize = [](uint64_t value) {
            return value > std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max()
                                                              : static_cast<size_t>(value);
        };
        const PullbackApplyOptions runtimeOptions{boundedSize(options->maximum_temporary_bytes),
                                                  boundedSize(options->maximum_reusable_construction_bytes)};
        return pullback->execution->apply(cotangents, *gradients, runtimeOptions);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate pullback state", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "pullback allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        return fail(context, "unexpected pullback failure", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus vernonPullbackApply(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                 VernonAdValueSet *gradients) {
    const VernonPullbackApplyOptions options{sizeof(VernonPullbackApplyOptions),
                                             VERNON_PULLBACK_APPLY_OPTIONS_VERSION,
                                             std::numeric_limits<uint64_t>::max(),
                                             0,
                                             {}};
    return vernonPullbackApplyWithOptions(pullback, cotangents, gradients, &options);
}

void vernonPullbackDestroy(VernonPullback *pullback) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(
        pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr);
    delete pullback;
}

} // extern "C"
