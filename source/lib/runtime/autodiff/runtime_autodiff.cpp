#include "runtime_autodiff_internal.h"

#include "VernonAutodiffGraph.h"
#include "runtime/runtime_state.h"
#include "runtime_direct_autodiff.h"

#include <algorithm>
#include <cstring>
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

bool carrierCount(VernonLaunchSize grid, size_t &count) {
    if (!validLaunchSize(grid) || grid.x > SIZE_MAX / grid.y ||
        static_cast<size_t>(grid.x) * grid.y > SIZE_MAX / grid.z)
        return false;
    count = static_cast<size_t>(grid.x) * grid.y * grid.z;
    return true;
}

bool materializeCarrierValue(ValueAbi &abi, VernonLaunchSize grid) {
    size_t count = 0;
    if (!carrierCount(grid, count) || (count && abi.byteSize > SIZE_MAX / count))
        return false;
    abi.byteSize *= count;
    if (count > 1)
        abi.logicalShape.insert(abi.logicalShape.begin(), {grid.z, grid.y, grid.x});
    return true;
}

bool resourceByteSize(const ResourceAbi &abi, VernonLaunchSize grid, size_t &size) {
    size = abi.value.byteSize;
    if (!abi.runtimeCarrier)
        return true;
    size_t count = 0;
    if (!carrierCount(grid, count) || (count && size > SIZE_MAX / count))
        return false;
    size *= count;
    return true;
}

VernonRhiBufferDescriptor gpuBufferDescriptor(const ValueAbi &abi) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = abi.byteSize;
    descriptor.alignment = abi.alignment;
    descriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    return descriptor;
}

VernonRhiStatus createGraphBuffer(execution::ExecutionGraph &graph, const ValueAbi &value,
                                  execution::GraphBuffer &buffer, bool exported) {
    return graph.createBuffer(gpuBufferDescriptor(value), buffer, exported);
}

bool uploadGpuBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, const void *data,
                     size_t size, const char *label) {
    if (!data || !size ||
        vernonRhiDeviceUploadBuffer(context.rhiDevice, buffer, offset, data, size) != VERNON_RHI_STATUS_OK) {
        invocationDiagnostic(context) = std::string("failed to upload ") + label;
        return false;
    }
    return true;
}

bool downloadGpuBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, void *data, size_t size,
                       const char *label) {
    if (!data || !size ||
        vernonRhiDeviceDownloadBuffer(context.rhiDevice, buffer, offset, data, size) != VERNON_RHI_STATUS_OK) {
        invocationDiagnostic(context) = std::string("failed to download ") + label;
        return false;
    }
    return true;
}

bool clearGpuBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, size_t size,
                    const char *label) {
    constexpr size_t chunkSize = 64 * 1024;
    const std::vector<uint8_t> zero(std::min(size, chunkSize));
    for (size_t written = 0; written < size; written += zero.size()) {
        const size_t count = std::min(zero.size(), size - written);
        if (!uploadGpuBuffer(context, buffer, offset + written, zero.data(), count, label))
            return false;
    }
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

bool createImmediateGpuGraph(VernonLoadedPipeline &pipeline) {
    if (!pipeline.autodiff || !std::dynamic_pointer_cast<GpuGraphExecutable>(pipeline.autodiff->executable))
        return false;
    AutodiffGraph builder(pipeline.context);
    AutodiffGraphNode node;
    VernonStatus status = builder.addNode("immediate", &pipeline, node);
    if (status != VERNON_STATUS_OK)
        return false;
    const Signature &signature = pipeline.autodiff->executable->signature();
    for (const ValueAbi &input : signature.inputs) {
        const bool differentiable = std::any_of(signature.gradients.begin(), signature.gradients.end(),
                                                [&](const ValueAbi &gradient) { return gradient.path == input.path; });
        status = builder.declareInput(node, input.path, input.path, differentiable ? input.path : std::string());
        if (status != VERNON_STATUS_OK)
            return false;
    }
    if (builder.setOutput(node) != VERNON_STATUS_OK)
        return false;
    std::unique_ptr<CompiledAutodiffGraph> compiled;
    if (builder.compile(compiled) != VERNON_STATUS_OK)
        return false;
    pipeline.autodiff->immediateGraph = std::shared_ptr<CompiledAutodiffGraph>(std::move(compiled));
    return true;
}

bool resolvePipelineAutodiff(VernonPipelineBundle &bundle, const AutodiffVariant &profiles,
                             VernonLoadedPipeline &pipeline) {
    if (!bundle.context || !bundle.autodiff)
        return false;
    const Stage &forward = bundle.stages.at(profiles.forwardWithTape);
    const Stage &backward = bundle.stages.at(profiles.backward);
    const std::vector<std::string> gradientPaths =
        autodiffDerivativeLeafPaths(bundle.autodiff->derivativeGroups, AutodiffDerivativeRole::Gradient);
    std::shared_ptr<Executable> executable;
    bool resolved = false;
    if (bundle.context->backend == VERNON_RUNTIME_CPU) {
        if (bundle.autodiff->protocol == "dynamic_v2")
            resolved = createCpuExecutable(*bundle.context, forward, backward, gradientPaths, executable);
        else
            invocationDiagnostic(*bundle.context) = "CPU autodiff profile uses an unsupported protocol";
    } else {
        resolved = createGpuExecutable(bundle, profiles.forwardWithTape, profiles.backward, gradientPaths,
                                       profiles.launch, executable);
    }
    if (!resolved)
        return false;
    if (!validateDerivativeGroupsAgainstSignature(*bundle.context, bundle.autodiff->derivativeGroups,
                                                  executable->signature()))
        return false;
    pipeline.autodiff = VernonLoadedAutodiff{std::move(executable), {}, bundle.autodiff->derivativeGroups};
    return bundle.context->backend == VERNON_RUNTIME_CPU || createImmediateGpuGraph(pipeline);
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

class GraphPullbackExecution final : public vernon::runtime::ad::PullbackExecution {
public:
    explicit GraphPullbackExecution(std::unique_ptr<vernon::runtime::AutodiffGraphPullback> pullback)
        : pullback_(std::move(pullback)) {}

    VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) override {
        return pullback_->apply(cotangents, gradients);
    }

private:
    std::unique_ptr<vernon::runtime::AutodiffGraphPullback> pullback_;
};

VernonStatus forwardGpuGraph(VernonLoadedPipeline &pipeline, VernonLaunchSize computeGrid,
                             const VernonAdValueSet &inputs, VernonAdValueSet &outputs,
                             std::unique_ptr<vernon::runtime::ad::PullbackExecution> &execution) {
    using namespace vernon::runtime;
    using namespace vernon::runtime::ad;
    std::shared_ptr<CompiledAutodiffGraph> compiled = pipeline.autodiff->immediateGraph;
    if (!compiled)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonAdValueSet mutableInputs = inputs;
    std::unique_ptr<AutodiffGraphPullback> pullback;
    const VernonStatus status = compiled->forward(computeGrid, mutableInputs, outputs, pullback);
    if (status != VERNON_STATUS_OK)
        return status;
    execution = std::make_unique<GraphPullbackExecution>(std::move(pullback));
    return VERNON_STATUS_OK;
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
        if (dynamic_cast<GpuGraphExecutable *>(executable))
            status = forwardGpuGraph(*pipeline, computeGrid, *inputs, *outputs, execution);
        else if (auto *host = dynamic_cast<HostExecutable *>(executable))
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

VernonStatus vernonPullbackApply(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                 VernonAdValueSet *gradients) {
    VernonRuntimeContext *context = pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr;
    try {
        using namespace vernon::runtime::ad;
        vernon::runtime::RuntimeDiagnosticScope diagnostic(context);
        if (!pullback || !pullback->execution || !validSet(cotangents, false) || !validSet(gradients, true))
            return fail(context, "invalid pullback invocation");
        return pullback->execution->apply(cotangents, *gradients);
    } catch (const std::bad_alloc &) {
        return fail(context, "cannot allocate pullback state", VERNON_STATUS_INTERNAL_ERROR);
    } catch (const std::length_error &) {
        return fail(context, "pullback allocation is too large", VERNON_STATUS_INTERNAL_ERROR);
    } catch (...) {
        return fail(context, "unexpected pullback failure", VERNON_STATUS_INTERNAL_ERROR);
    }
}

void vernonPullbackDestroy(VernonPullback *pullback) {
    vernon::runtime::RuntimeDiagnosticScope diagnostic(
        pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr);
    delete pullback;
}

} // extern "C"
