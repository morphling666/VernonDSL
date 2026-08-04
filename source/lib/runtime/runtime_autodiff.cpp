#include "runtime_autodiff_internal.h"

#include "VernonAutodiffGraph.h"
#include "runtime_state.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
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
    return value.dtype == abi.dtype && value.data && value.size == abi.byteSize;
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
    if (!abi.logicalShape.empty() || abi.byteSize != dtypeSize(abi.dtype)) {
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

} // namespace vernon::runtime::ad

namespace {

VernonStatus fail(VernonRuntimeContext *context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    if (context)
        vernon::runtime::invocationDiagnostic(*context) = std::move(message);
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

extern "C" {

VernonStatus vernonRuntimeLoadedPipelineGetAdOutputDataType(const VernonLoadedPipeline *pipeline,
                                                            VernonDataType *dtype) {
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !dtype)
        return fail(pipeline ? pipeline->context : nullptr, "pipeline has no autodiff output");
    *dtype = executable->signature().output.dtype;
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdOutputRank(const VernonLoadedPipeline *pipeline, size_t *rank) {
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !rank)
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff output rank query");
    *rank = executable->signature().output.logicalShape.size();
    return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdOutputDimension(const VernonLoadedPipeline *pipeline, size_t index,
                                                             uint64_t *extent) {
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !extent)
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff output dimension query");
    const auto &shape = executable->signature().output.logicalShape;
    if (index >= shape.size())
        return fail(pipeline->context, "autodiff output dimension is out of range");
    *extent = shape[index];
    return VERNON_STATUS_OK;
}

size_t vernonRuntimeLoadedPipelineGetAdGradientCount(const VernonLoadedPipeline *pipeline) {
    const auto *executable = autodiffExecutable(pipeline);
    return executable ? executable->signature().gradients.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetAdGradient(const VernonLoadedPipeline *pipeline, size_t index,
                                                      VernonStringView *path, VernonDataType *dtype) {
    const auto *executable = autodiffExecutable(pipeline);
    if (!executable || !path || !dtype || index >= executable->signature().gradients.size())
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff gradient query");
    const auto &gradient = executable->signature().gradients[index];
    *path = {gradient.path.data(), gradient.path.size()};
    *dtype = gradient.dtype;
    return VERNON_STATUS_OK;
}

VernonStatus vernonAdPipelineForward(VernonLoadedPipeline *pipeline, VernonLaunchSize computeGrid,
                                     const VernonAdValueSet *inputs, VernonAdValueSet *outputs,
                                     VernonPullback **pullback) {
    using namespace vernon::runtime::ad;
    if (pipeline && pipeline->context)
        vernon::runtime::clearInvocationDiagnostic(*pipeline->context);
    if (pullback)
        *pullback = nullptr;
    auto *executable = pipeline && pipeline->autodiff ? pipeline->autodiff->executable.get() : nullptr;
    if (!pipeline || !pullback || !executable || !validLaunchSize(computeGrid) || !validSet(inputs, true) ||
        !validSet(outputs, true))
        return fail(pipeline ? pipeline->context : nullptr, "invalid autodiff forward invocation");
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
    auto result = std::make_unique<VernonPullback>();
    result->contextLease = vernon::runtime::acquireContextLease(*pipeline->context);
    result->execution = std::move(execution);
    *pullback = result.release();
    return VERNON_STATUS_OK;
}

VernonStatus vernonPullbackApply(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                 VernonAdValueSet *gradients) {
    using namespace vernon::runtime::ad;
    if (pullback && pullback->contextLease)
        vernon::runtime::clearInvocationDiagnostic(pullback->contextLease->get());
    if (!pullback || !pullback->execution || !validSet(cotangents, false) || !validSet(gradients, true))
        return fail(pullback && pullback->contextLease ? &pullback->contextLease->get() : nullptr,
                    "invalid pullback invocation");
    return pullback->execution->apply(cotangents, *gradients);
}

void vernonPullbackDestroy(VernonPullback *pullback) { delete pullback; }

} // extern "C"
