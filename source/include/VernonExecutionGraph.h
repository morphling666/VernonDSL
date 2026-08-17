#ifndef VERNON_EXECUTION_GRAPH_H
#define VERNON_EXECUTION_GRAPH_H

#include "VernonRHI.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::execution {

class ExecutionGraph;
class CompiledExecutionGraph;
class ExecutionSubmission;
class ExecutionResources;
class GraphPullback;
class GraphBackwardSubmission;
namespace detail {
struct ExecutionGraphTestAccess;
}

enum class ResourceKind : uint8_t { Buffer, Image };
enum class AccessMode : uint8_t { Read, Write, ReadWrite };

struct GraphResource {
    uint32_t id{UINT32_MAX};
    ResourceKind kind{ResourceKind::Buffer};
    uint64_t graphIdentity{};
};

struct GraphBuffer : GraphResource {
    VernonRhiBuffer handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
};

struct GraphImage : GraphResource {
    VernonRhiImage handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiImageView view{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiFormat format{VERNON_RHI_FORMAT_UNDEFINED};
    uint32_t width{};
    uint32_t height{};
    uint32_t layers{1};
    uint32_t samples{1};
    VernonRhiImageSubresourceRange subresources{0, UINT32_MAX, 0, UINT32_MAX, VERNON_RHI_IMAGE_ASPECT_COLOR};
};

enum PassFlagBits : uint32_t { PassNone = 0, PassNeverCull = 1u << 0, PassNoMerge = 1u << 1, PassSideEffect = 1u << 2 };

enum class ImageUseRole : uint8_t { None, Sampled, Storage, ColorAttachment, DepthAttachment, Transfer };

struct ResourceUse {
    GraphResource resource;
    AccessMode access{AccessMode::Read};
    VernonRhiResourceState state{VERNON_RHI_STATE_COMMON};
    uint32_t stageMask{};
    ImageUseRole imageRole{ImageUseRole::None};
    VernonRhiImageSubresourceRange imageSubresources{};
    std::optional<GraphImage> image;
};

struct ColorAttachmentUse {
    uint32_t location{};
    GraphImage image;
    VernonRhiLoadOperation load{VERNON_RHI_LOAD_PRESERVE};
    VernonRhiStoreOperation store{VERNON_RHI_STORE_PRESERVE};
    float clear[4]{};
};

struct DepthStencilAttachmentUse {
    GraphImage image;
    VernonRhiLoadOperation depthLoad{VERNON_RHI_LOAD_PRESERVE};
    VernonRhiStoreOperation depthStore{VERNON_RHI_STORE_PRESERVE};
    float clearDepth{1.0f};
    VernonRhiLoadOperation stencilLoad{VERNON_RHI_LOAD_DISCARD};
    VernonRhiStoreOperation stencilStore{VERNON_RHI_STORE_DISCARD};
    uint32_t clearStencil{};
    bool readOnlyDepth{};
    bool readOnlyStencil{};
};

class GraphicsEncoder {
public:
    GraphicsEncoder(VernonRhiDevice device, VernonRhiCommandEncoder encoder) : device_(device), encoder_(encoder) {}
    VernonRhiDevice device() const { return device_; }
    VernonRhiCommandEncoder native() const { return encoder_; }

private:
    VernonRhiDevice device_{};
    VernonRhiCommandEncoder encoder_;
};

class ComputeEncoder {
public:
    ComputeEncoder(VernonRhiDevice device, VernonRhiCommandEncoder encoder) : device_(device), encoder_(encoder) {}
    VernonRhiDevice device() const { return device_; }
    VernonRhiCommandEncoder native() const { return encoder_; }

private:
    VernonRhiDevice device_;
    VernonRhiCommandEncoder encoder_;
};

struct ExecutionParameter {
    uint32_t id{UINT32_MAX};
    uint64_t graphIdentity{};
};

class ExecutionBindingValue {
public:
    virtual ~ExecutionBindingValue() = default;
};

struct ExecutionBinding {
    ExecutionParameter parameter;
    std::shared_ptr<const ExecutionBindingValue> value;
};

enum class DerivativeEndpointKind : uint8_t { Resource, Parameter };

struct DerivativeEndpointKey {
    DerivativeEndpointKind kind{DerivativeEndpointKind::Resource};
    uint32_t id{UINT32_MAX};

    bool operator==(const DerivativeEndpointKey &other) const { return kind == other.kind && id == other.id; }
};

struct NamedDerivativeEndpoint {
    std::string name;
    DerivativeEndpointKey endpoint;
};

struct PassDerivativeMapping {
    std::string path;
    DerivativeEndpointKey endpoint;
};

struct PassPrimalResourceMapping {
    std::string path;
    uint32_t resource{};
};

struct GraphByteRange {
    uint64_t offset{};
    uint64_t byteSize{};
};

struct PassWriteFootprint {
    uint32_t resource{};
    std::vector<GraphByteRange> ranges;
};

class GraphAutodiffValue {
public:
    virtual ~GraphAutodiffValue() = default;
    virtual uintptr_t logicalIdentity() const = 0;
    virtual uint64_t allocationBytes() const = 0;
    virtual std::shared_ptr<GraphAutodiffValue> add(const GraphAutodiffValue &other, std::string &error) const = 0;
};

using NamedGraphAutodiffValues = std::vector<std::pair<std::string, std::shared_ptr<GraphAutodiffValue>>>;

struct PassPullbackApplyOptions {
    uint64_t maximumTemporaryBytes{std::numeric_limits<uint64_t>::max()};
    uint64_t maximumReusableConstructionBytes{};
};

class PassPullback {
public:
    virtual ~PassPullback() = default;
    virtual bool apply(const NamedGraphAutodiffValues &cotangents, NamedGraphAutodiffValues &gradients,
                       std::string &error) = 0;
    virtual bool applyWithOptions(const NamedGraphAutodiffValues &cotangents, NamedGraphAutodiffValues &gradients,
                                  const PassPullbackApplyOptions &, std::string &error) {
        return apply(cotangents, gradients, error);
    }
    virtual uint64_t estimatedTapeBytes() const = 0;
    virtual uint64_t logicalResidualBytes() const = 0;
    virtual uint64_t residentTapeBytes() const = 0;
    virtual uint64_t allocatedTapeBytes() const = 0;
    virtual uint64_t retainedAllocationBytes() const { return allocatedTapeBytes(); }
    virtual uint64_t peakTemporaryTapeBytes() const { return 0; }
    virtual uint64_t submissionCount() const { return 0; }
    virtual uint64_t waitCount() const { return 0; }
    virtual uint64_t readbackCount() const { return 0; }
    virtual uint64_t atomicPublicationCount() const { return 0; }
    virtual uint64_t temporaryAllocationTrafficBytes() const { return 0; }
    virtual uint64_t deviceWaitNanoseconds() const { return 0; }
    virtual uint64_t activeOperationCount() const = 0;
    virtual uint64_t recomputationCost() const = 0;
    virtual uint64_t tapeContextLimitBytes() const = 0;
    virtual std::string residualSourceKind() const { return "capture"; }
    virtual std::string controlHistoryKind() const { return "unknown"; }
};

class DifferentiablePass {
public:
    virtual ~DifferentiablePass() = default;
    virtual const std::vector<PassDerivativeMapping> &gradientMappings() const = 0;
    virtual const std::vector<PassDerivativeMapping> &cotangentMappings() const = 0;
    virtual const std::vector<PassPrimalResourceMapping> &requiredPrimalResources() const {
        static const std::vector<PassPrimalResourceMapping> empty;
        return empty;
    }
    virtual const std::vector<PassWriteFootprint> &writeFootprints() const {
        static const std::vector<PassWriteFootprint> empty;
        return empty;
    }
    virtual const std::vector<PassWriteFootprint> &readFootprints() const {
        static const std::vector<PassWriteFootprint> empty;
        return empty;
    }
    virtual bool forward(ComputeEncoder &encoder, const ExecutionResources &resources,
                         std::unique_ptr<PassPullback> &pullback, std::string &error) = 0;
    virtual bool zeroCotangent(const std::string &path, const ExecutionResources &resources,
                               std::shared_ptr<GraphAutodiffValue> &value, std::string &error) = 0;
    virtual bool implicitCotangent(const std::string &path, const ExecutionResources &resources,
                                   std::shared_ptr<GraphAutodiffValue> &value, std::string &error) = 0;
    virtual uint64_t estimatedResidualBytes() const { return 0; }
    virtual uint64_t estimatedRetainedAllocationBytes() const = 0;
    virtual uint64_t estimatedForwardPeakBytes() const { return estimatedRetainedAllocationBytes(); }
    virtual uint64_t replayCost() const { return 0; }
    virtual uint64_t resourceReloadCost() const { return 0; }
    virtual uint64_t recomputationCost() const { return 0; }
    virtual bool deterministicReductionLegal() const { return true; }
    virtual bool hasCheckpointPlanningMetadata() const { return false; }
    virtual bool supportsReplay() const { return false; }
};

class GraphCheckpointResource {
public:
    virtual ~GraphCheckpointResource() = default;
    virtual uint64_t byteSize() const = 0;
    virtual uint64_t alignment() const { return 1; }
    virtual bool copyTo(void *destination, uint64_t byteSize, std::string &error) const = 0;
    virtual bool copyFrom(const void *source, uint64_t byteSize, std::string &error) = 0;
    virtual bool copyRangeTo(uint64_t offset, void *destination, uint64_t byteSize, std::string &error) const;
    virtual bool copyRangeFrom(uint64_t offset, const void *source, uint64_t byteSize, std::string &error);
    virtual bool copyRangesTo(const std::vector<GraphByteRange> &ranges, void *packedDestination,
                              std::string &error) const;
    virtual bool copyRangesFrom(const std::vector<GraphByteRange> &ranges, const void *packedSource,
                                std::string &error);
};

class ExecutionBindings {
public:
    const std::shared_ptr<const ExecutionBindingValue> &at(ExecutionParameter parameter) const;
    size_t size() const { return values_.size(); }

private:
    friend class CompiledExecutionGraph;
    friend class GraphPullback;
    friend class ExecutionBindingsBuilder;
    friend class ExecutionResources;
    ExecutionBindings(uint64_t graphIdentity, std::vector<std::shared_ptr<const ExecutionBindingValue>> values)
        : graphIdentity_(graphIdentity), values_(std::move(values)) {}

    uint64_t graphIdentity_{};
    std::vector<std::shared_ptr<const ExecutionBindingValue>> values_;
};

class ExecutionBindingsBuilder {
public:
    void set(ExecutionParameter parameter, std::shared_ptr<const ExecutionBindingValue> value);
    std::shared_ptr<const ExecutionBindings> snapshot() const;

private:
    friend class CompiledExecutionGraph;
    ExecutionBindingsBuilder(uint64_t graphIdentity, size_t parameterCount);

    uint64_t graphIdentity_{};
    std::vector<std::shared_ptr<const ExecutionBindingValue>> values_;
    mutable std::shared_ptr<const ExecutionBindings> cached_;
};

class ExecutionResources {
public:
    ExecutionResources(const std::vector<GraphResource> &resources, const std::vector<VernonRhiBuffer> &buffers,
                       std::shared_ptr<const ExecutionBindings> bindings)
        : resources_(resources), buffers_(buffers), bindings_(std::move(bindings)) {}
    const std::vector<GraphResource> &all() const { return resources_; }
    bool buffer(GraphBuffer resource, VernonRhiBuffer &output) const;
    bool hasBindings() const { return static_cast<bool>(bindings_); }
    const std::shared_ptr<const ExecutionBindingValue> &binding(ExecutionParameter parameter) const;

private:
    friend class CompiledExecutionGraph;
    const std::vector<GraphResource> &resources_;
    const std::vector<VernonRhiBuffer> &buffers_;
    std::shared_ptr<const ExecutionBindings> bindings_;
};

class ExecutionPass {
public:
    explicit ExecutionPass(std::string name);
    virtual ~ExecutionPass() = default;
    ExecutionPass(const ExecutionPass &) = delete;
    ExecutionPass &operator=(const ExecutionPass &) = delete;

    const std::string &name() const { return name_; }
    void dependsOn(ExecutionPass &dependency);
    void setFlags(uint32_t flags);
    uint32_t flags() const { return flags_; }
    const std::vector<ResourceUse> &uses() const { return uses_; }

    virtual void declare() = 0;

protected:
    void ensureMutable() const;
    void read(GraphResource resource, VernonRhiResourceState state = VERNON_RHI_STATE_SHADER_READ,
              uint32_t stageMask = 0);
    void read(GraphImage resource, VernonRhiResourceState state = VERNON_RHI_STATE_SHADER_READ, uint32_t stageMask = 0);
    void write(GraphResource resource, VernonRhiResourceState state = VERNON_RHI_STATE_SHADER_WRITE,
               uint32_t stageMask = 0);
    void write(GraphImage resource, VernonRhiResourceState state = VERNON_RHI_STATE_SHADER_WRITE,
               uint32_t stageMask = 0);
    void readWrite(GraphResource resource, VernonRhiResourceState state = VERNON_RHI_STATE_SHADER_WRITE,
                   uint32_t stageMask = 0);
    void readWrite(GraphImage resource, VernonRhiResourceState state = VERNON_RHI_STATE_SHADER_WRITE,
                   uint32_t stageMask = 0);

private:
    friend class ExecutionGraph;
    void resetDeclaration();
    void finishDeclaration();

    std::string name_;
    uint32_t flags_{};
    uint32_t configuredFlags_{};
    bool declaring_{};
    ExecutionGraph *owner_{};
    bool frozen_{};
    std::vector<ExecutionPass *> dependencies_;
    std::vector<ResourceUse> uses_;
};

class RenderPass : public ExecutionPass {
public:
    using ExecutionPass::ExecutionPass;
    virtual VernonRhiStatus execute(GraphicsEncoder &encoder, const ExecutionResources &resources) = 0;
    const std::vector<ColorAttachmentUse> &colors() const { return colors_; }
    const DepthStencilAttachmentUse *depthAttachment() const { return depth_.get(); }
    const uint32_t *renderAreaData() const { return renderArea_; }

protected:
    void color(uint32_t location, const ColorAttachmentUse &attachment);
    void depth(const DepthStencilAttachmentUse &attachment);
    void renderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

private:
    friend class ExecutionGraph;
    std::vector<ColorAttachmentUse> colors_;
    std::unique_ptr<DepthStencilAttachmentUse> depth_;
    uint32_t renderArea_[4]{};
};

class ComputePass : public ExecutionPass {
public:
    using ExecutionPass::ExecutionPass;
    virtual VernonRhiStatus execute(ComputeEncoder &encoder, const ExecutionResources &resources) = 0;
    virtual DifferentiablePass *differentiable() { return nullptr; }
};

struct CompiledScope {
    bool rendering{};
    std::vector<uint32_t> passIndices;
    std::vector<VernonRhiBarrier> barriers;
};

struct AutodiffReplaySegment {
    uint32_t beginStep{};
    uint32_t endStep{};
    uint32_t checkpointIndex{std::numeric_limits<uint32_t>::max()};
    uint64_t logicalResidualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t peakBytes{};
    uint64_t replayCost{};
    std::vector<uint32_t> releaseCheckpointResources;
};

struct AutodiffResourceVersion {
    uint32_t resource{};
    uint32_t epoch{};
};

enum class AutodiffVersionSource : uint8_t {
    RetainedOwner,
    InitialState,
    Checkpoint,
    Replay,
};

struct AutodiffRequiredResourceVersion {
    uint32_t consumer{};
    AutodiffResourceVersion version;
    uint32_t producer{std::numeric_limits<uint32_t>::max()};
    AutodiffVersionSource source{AutodiffVersionSource::RetainedOwner};
    std::string path;
};

struct AutodiffPassVersionState {
    std::vector<AutodiffResourceVersion> inputs;
    std::vector<AutodiffResourceVersion> outputs;
};

struct AutodiffCheckpointResource {
    uint32_t producer{};
    AutodiffResourceVersion version;
    uint64_t offset{};
    uint64_t byteSize{};
    uint64_t alignment{1};
    uint32_t firstCut{std::numeric_limits<uint32_t>::max()};
    uint32_t lastCut{};
};

struct AutodiffLivenessCut {
    uint32_t scheduleOffset{};
    std::vector<uint32_t> checkpointResources;
};

struct AutodiffDagCheckpointPlan {
    std::vector<AutodiffCheckpointResource> checkpointResources;
    std::vector<AutodiffLivenessCut> cuts;
    std::vector<AutodiffReplaySegment> replaySegments;
    std::vector<AutodiffPassVersionState> passVersions;
    std::vector<AutodiffRequiredResourceVersion> requiredVersions;
    uint64_t persistentCheckpointBytes{};
    uint64_t initialStateBytes{};
    uint64_t restorationBytes{};
    uint64_t transactionBytes{};
    uint64_t logicalResidualBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t maximumForwardPeakBytes{};
    uint64_t backwardValueBytes{};
    uint64_t memoryBudget{};
    uint64_t peakBytes{};
    uint64_t replayCost{};
    uint64_t captureStoreBytes{};
    uint64_t backwardLoadBytes{};
    uint64_t checkpointCopyBytes{};
    uint64_t resourceReloadCost{};
    uint64_t recomputationCost{};
    uint64_t weightedRuntimeCost{};
    bool deterministicReductionLegal{true};
    std::string selectedPolicy{"balanced"};
};

namespace detail {
enum class ExecutionProvider : uint8_t { Cpu, Rhi };

struct ExecutionResourceRecord {
    GraphResource resource;
    bool exported{};
    bool graphOwned{};
    uint64_t resourceKey{};
    std::vector<uint64_t> imageViewKeys;
    VernonRhiBuffer buffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiImage image{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::shared_ptr<GraphCheckpointResource> checkpoint;
};
} // namespace detail

class DeviceExecutionSession {
public:
    explicit DeviceExecutionSession(VernonRhiDevice device);
    ~DeviceExecutionSession();
    DeviceExecutionSession(DeviceExecutionSession &&) noexcept;
    DeviceExecutionSession &operator=(DeviceExecutionSession &&) noexcept;
    DeviceExecutionSession(const DeviceExecutionSession &) = delete;
    DeviceExecutionSession &operator=(const DeviceExecutionSession &) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class ExecutionGraph {
public:
    ExecutionGraph();
    explicit ExecutionGraph(VernonRhiDevice device);
    ~ExecutionGraph();
    ExecutionGraph(const ExecutionGraph &) = delete;
    ExecutionGraph &operator=(const ExecutionGraph &) = delete;

    template <typename Pass, typename... Arguments> Pass &emplacePass(Arguments &&...arguments) {
        if (compiled_)
            throw std::logic_error("cannot mutate a compiled execution graph builder");
        auto pass = std::make_unique<Pass>(std::forward<Arguments>(arguments)...);
        Pass &reference = *pass;
        passes_.push_back(std::move(pass));
        reference.owner_ = this;
        dirty_ = true;
        return reference;
    }

    VernonRhiStatus createBuffer(const VernonRhiBufferDescriptor &descriptor, GraphBuffer &output,
                                 bool exported = false);
    GraphBuffer importHostBuffer(uint64_t identity, bool exported = false,
                                 std::shared_ptr<GraphCheckpointResource> checkpoint = {});
    GraphBuffer importBuffer(VernonRhiBuffer buffer, bool exported = false,
                             std::shared_ptr<GraphCheckpointResource> checkpoint = {});
    GraphImage importImage(VernonRhiImage image, VernonRhiImageView view, bool exported = false);
    ExecutionParameter parameter(std::string name);
    void setAutodiffEndpoints(std::vector<NamedDerivativeEndpoint> differentiableInputs,
                              std::vector<NamedDerivativeEndpoint> objectives);
    void planAutodiffCheckpoints(uint64_t memoryBudget);
    std::shared_ptr<CompiledExecutionGraph> compile(std::string &error);
    bool validate(std::string &error);

private:
    friend class ExecutionPass;
    friend struct detail::ExecutionGraphTestAccess;

    bool buildPlan(std::string &error);
    bool validateDeclarations(std::string &error) const;

    detail::ExecutionProvider provider_{detail::ExecutionProvider::Cpu};
    VernonRhiDevice device_;
    uint64_t graphIdentity_{};
    std::vector<std::unique_ptr<ExecutionPass>> passes_;
    std::vector<GraphResource> resources_;
    std::vector<detail::ExecutionResourceRecord> resourceRecords_;
    std::unordered_map<uint64_t, uint32_t> importedBuffers_;
    std::unordered_map<uint64_t, uint32_t> importedHostBuffers_;
    std::unordered_map<uint64_t, uint32_t> importedImages_;
    std::vector<std::string> parameterNames_;
    std::unordered_map<std::string, uint32_t> parameterIds_;
    std::vector<uint32_t> schedule_;
    std::vector<CompiledScope> scopes_;
    std::vector<uint32_t> autodiffInitialResources_;
    std::vector<std::vector<GraphByteRange>> autodiffInitialRanges_;
    std::vector<uint32_t> autodiffTransactionResources_;
    std::vector<std::vector<GraphByteRange>> autodiffTransactionRanges_;
    std::vector<uint32_t> autodiffRestorationResources_;
    std::vector<std::vector<GraphByteRange>> autodiffRestorationRanges_;
    AutodiffDagCheckpointPlan autodiffCheckpointPlan_;
    uint64_t autodiffMemoryBudget_{};
    bool hasAutodiffSchedule_{};
    std::vector<NamedDerivativeEndpoint> differentiableInputs_;
    std::vector<NamedDerivativeEndpoint> objectives_;
    bool dirty_{true};
    bool compiled_{};
};

class ExecutionSubmission {
public:
    enum class State : uint8_t { Pending, Succeeded, Failed };

    ExecutionSubmission();
    ~ExecutionSubmission();
    ExecutionSubmission(ExecutionSubmission &&) noexcept;
    ExecutionSubmission &operator=(ExecutionSubmission &&) noexcept;
    ExecutionSubmission(const ExecutionSubmission &) = delete;
    ExecutionSubmission &operator=(const ExecutionSubmission &) = delete;

    State state() const;
    VernonRhiStatus wait();
    VernonRhiStatus signal(VernonRhiStatus result);
    VernonRhiStatus status() const;
    const VernonRhiCommandEncoderStats &commandStats() const;

private:
    friend class CompiledExecutionGraph;
    class Impl;
    explicit ExecutionSubmission(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class CompiledExecutionGraph {
public:
    ~CompiledExecutionGraph();
    CompiledExecutionGraph(const CompiledExecutionGraph &) = delete;
    CompiledExecutionGraph &operator=(const CompiledExecutionGraph &) = delete;

    ExecutionBindingsBuilder createBindings(const std::vector<ExecutionBinding> &initial) const;
    ExecutionSubmission submit(std::shared_ptr<const ExecutionBindings> bindings = {}) const;
    std::shared_ptr<GraphPullback> vjp(std::shared_ptr<const ExecutionBindings> bindings, std::string &error) const;
    const std::vector<uint32_t> &schedule() const;
    const std::vector<CompiledScope> &scopes() const;
    const AutodiffDagCheckpointPlan *autodiffCheckpointPlan() const;

private:
    friend class ExecutionGraph;
    friend class ExecutionSubmission;
    friend class GraphPullback;
    struct State;
    explicit CompiledExecutionGraph(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
};

class GraphBackwardSubmission {
public:
    enum class State : uint8_t { Pending, Succeeded, Failed };

    State state() const;
    bool wait(std::string &error);
    const NamedGraphAutodiffValues &gradients() const;

private:
    friend class GraphPullback;
    State state_{State::Pending};
    std::string error_;
    NamedGraphAutodiffValues gradients_;
};

struct GraphAutodiffPassTelemetry {
    uint32_t scheduleOffset{};
    std::string passName;
    std::string residualSourceKind{"capture"};
    std::string controlHistoryKind{"unknown"};
    uint64_t estimatedTapeBytes{};
    uint64_t logicalResidualBytes{};
    uint64_t residentTapeBytes{};
    uint64_t allocatedTapeBytes{};
    uint64_t retainedAllocationBytes{};
    uint64_t peakTemporaryTapeBytes{};
    uint64_t checkpointBytes{};
    uint64_t activeOperationCount{};
    uint64_t recomputationCost{};
    std::optional<uint64_t> controlHistoryBytes;
};

class GraphPullback {
public:
    ~GraphPullback();
    GraphPullback(const GraphPullback &) = delete;
    GraphPullback &operator=(const GraphPullback &) = delete;
    ExecutionSubmission &forwardSubmission();
    const ExecutionSubmission &forwardSubmission() const;
    std::shared_ptr<GraphBackwardSubmission> submit(const NamedGraphAutodiffValues &cotangents, bool implicit);
    uint64_t estimatedTapeBytes() const;
    uint64_t logicalResidualBytes() const;
    uint64_t residentTapeBytes() const;
    uint64_t allocatedTapeBytes() const;
    uint64_t retainedAllocationBytes() const;
    uint64_t checkpointBytes() const;
    uint64_t peakRuntimeManagedBytes() const;
    uint64_t submissionCount() const;
    uint64_t waitCount() const;
    uint64_t readbackCount() const;
    uint64_t atomicPublicationCount() const;
    uint64_t temporaryAllocationTrafficBytes() const;
    uint64_t deviceWaitNanoseconds() const;
    uint64_t tapeContextLimitBytes() const;
    double recomputationFactor() const;
    std::vector<GraphAutodiffPassTelemetry> passTelemetry() const;

private:
    friend class CompiledExecutionGraph;
    class Impl;
    explicit GraphPullback(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace vernon::execution

#endif
