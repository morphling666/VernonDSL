#ifndef VERNON_RUNTIME_RUNTIME_AUTODIFF_INTERNAL_H
#define VERNON_RUNTIME_RUNTIME_AUTODIFF_INTERNAL_H

#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_manifest.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct VernonRuntimeContext;

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

inline bool sameValueAbi(const ValueAbi &left, const ValueAbi &right) {
    return left.dtype == right.dtype && left.byteSize == right.byteSize && left.alignment == right.alignment &&
           left.logicalShape == right.logicalShape;
}

struct Signature {
    std::vector<ValueAbi> inputs;
    ValueAbi output;
    std::vector<ValueAbi> tape;
    ValueAbi cotangent;
    std::vector<ValueAbi> gradients;
};

class PullbackExecution {
public:
    virtual ~PullbackExecution() = default;
    virtual VernonStatus apply(const VernonAdValueSet *cotangents, VernonAdValueSet &gradients) = 0;
};

class Executable {
public:
    virtual ~Executable() = default;
    virtual const Signature &signature() const = 0;
};

class HostExecutable : public Executable {
public:
    virtual VernonStatus forward(VernonLaunchSize computeGrid, const VernonAdValueSet &inputs,
                                 VernonAdValueSet &outputs, std::unique_ptr<PullbackExecution> &pullback) = 0;
};

enum class GpuResourceRole { Input, Storage, Output, Tape, Cotangent, Gradient };

struct ResourceAbi {
    ValueAbi value;
    std::vector<uint64_t> physicalShape;
    uint64_t byteSize{};
    VernonValueAccess access{};
    GpuResourceRole role{};
    bool runtimeCarrier{};
};

struct GpuBufferBinding {
    VernonRhiBuffer buffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    uint64_t size{};
};

class GpuGraphExecutable : public Executable {
public:
    virtual VernonRuntimeContext &context() const = 0;
    virtual VernonRhiDevice device() const = 0;
    virtual const std::vector<ResourceAbi> &forwardResources() const = 0;
    virtual const std::vector<ResourceAbi> &backwardResources() const = 0;
    virtual VernonStatus encodeForward(VernonRuntimeProviderObject encoder, VernonLaunchSize computeGrid,
                                       const std::vector<GpuBufferBinding> &bindings) = 0;
    virtual VernonStatus encodeBackward(VernonRuntimeProviderObject encoder, VernonLaunchSize computeGrid,
                                        const std::vector<GpuBufferBinding> &bindings) = 0;
};

size_t dtypeSize(VernonDataType dtype);
bool validLaunchSize(VernonLaunchSize grid);
bool carrierCount(VernonLaunchSize grid, size_t &count);
bool materializeCarrierValue(ValueAbi &abi, VernonLaunchSize grid);
bool resourceByteSize(const ResourceAbi &abi, VernonLaunchSize grid, size_t &size);
VernonRhiBufferDescriptor gpuBufferDescriptor(const ValueAbi &abi);
VernonRhiStatus createGraphBuffer(execution::ExecutionGraph &graph, const ValueAbi &value,
                                  execution::GraphBuffer &buffer, bool exported = false);
bool uploadGpuBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, const void *data,
                     size_t size, const char *label);
bool downloadGpuBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, void *data, size_t size,
                       const char *label);
bool clearGpuBuffer(VernonRuntimeContext &context, VernonRhiBuffer buffer, uint64_t offset, size_t size,
                    const char *label);
bool validSet(const VernonAdValueSet *set, bool required);
VernonAdValue *findValue(VernonAdValueSet &set, const std::string &path);
const VernonAdValue *findValue(const VernonAdValueSet &set, const std::string &path);
bool valueMatches(const VernonAdValue &value, const ValueAbi &abi);
bool derivativeAbiMatches(const ValueAbi &primal, const ValueAbi &derivative);
bool makeCotangentBytes(const VernonAdValueSet *cotangents, const ValueAbi &abi, std::vector<uint8_t> &bytes,
                        std::string &error);

bool createCpuExecutable(VernonRuntimeContext &context, const Stage &forward, const Stage &backward,
                         const std::vector<std::string> &gradientPaths, std::shared_ptr<Executable> &executable);
bool createGpuExecutable(VernonPipelineBundle &bundle, const std::string &forwardId, const std::string &backwardId,
                         const std::vector<std::string> &gradientPaths, const AutodiffLaunchPlan &launch,
                         std::shared_ptr<Executable> &executable);
bool createImmediateGpuGraph(VernonLoadedPipeline &pipeline);

} // namespace vernon::runtime::ad

struct VernonPullback {
    std::shared_ptr<vernon::runtime::ContextLease> contextLease;
    std::unique_ptr<vernon::runtime::ad::PullbackExecution> execution;
};

#endif
