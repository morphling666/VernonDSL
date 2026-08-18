#include "VernonExecutionGraph.h"
#include "execution_graph/execution_graph_checkpoint_planner_internal.h"
#include "execution_graph/execution_graph_internal.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <future>

namespace vernon::execution::detail {

struct ExecutionGraphTestAccess {
    static GraphBuffer importBuffer(ExecutionGraph &graph, VernonRhiBuffer buffer, bool exported = false) {
        const uint64_t key = (static_cast<uint64_t>(buffer.generation) << 32) | (uint64_t{buffer.index} + 1);
        if (const auto found = graph.importedBuffers_.find(key); found != graph.importedBuffers_.end()) {
            graph.resourceRecords_[found->second].exported |= exported;
            GraphBuffer result;
            result.id = found->second;
            result.kind = ResourceKind::Buffer;
            result.graphIdentity = graph.graphIdentity_;
            result.handle = buffer;
            return result;
        }
        GraphBuffer result;
        result.id = static_cast<uint32_t>(graph.resources_.size());
        result.kind = ResourceKind::Buffer;
        result.graphIdentity = graph.graphIdentity_;
        result.handle = buffer;
        graph.resources_.push_back(result);
        graph.resourceRecords_.push_back({result, exported});
        graph.resourceRecords_.back().buffer = buffer;
        graph.resourceRecords_.back().resourceKey = key;
        graph.importedBuffers_.emplace(key, result.id);
        graph.dirty_ = true;
        return result;
    }

    static GraphImage importImage(ExecutionGraph &graph, VernonRhiImage image, VernonRhiImageView view,
                                  VernonRhiFormat format, uint32_t width, uint32_t height, uint32_t layers = 1,
                                  uint32_t samples = 1, bool exported = false,
                                  const VernonRhiImageSubresourceRange *subresources = nullptr) {
        const uint32_t aspects =
            subresources ? subresources->aspects
                         : static_cast<uint32_t>(format == VERNON_RHI_FORMAT_D32_FLOAT ? VERNON_RHI_IMAGE_ASPECT_DEPTH
                                                 : format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT
                                                     ? VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL
                                                     : VERNON_RHI_IMAGE_ASPECT_COLOR);
        const uint32_t baseMip = subresources ? subresources->base_mip_level : 0;
        const uint32_t mipCount =
            subresources && subresources->mip_level_count != UINT32_MAX ? subresources->mip_level_count : 1;
        const uint32_t baseLayer = subresources ? subresources->base_array_layer : 0;
        const uint32_t layerCount =
            subresources && subresources->array_layer_count != UINT32_MAX ? subresources->array_layer_count : layers;
        GraphImage result;
        result.kind = ResourceKind::Image;
        result.graphIdentity = graph.graphIdentity_;
        result.handle = image;
        result.view = view;
        result.format = format;
        result.width = width;
        result.height = height;
        result.layers = layerCount;
        result.samples = samples;
        result.subresources = {baseMip, mipCount, baseLayer, layerCount, aspects};
        const uint64_t key = (static_cast<uint64_t>(image.generation) << 32) | (uint64_t{image.index} + 1);
        if (const auto found = graph.importedImages_.find(key); found != graph.importedImages_.end()) {
            graph.resourceRecords_[found->second].exported |= exported;
            result.id = found->second;
            return result;
        }
        result.id = static_cast<uint32_t>(graph.resources_.size());
        graph.resources_.push_back(result);
        graph.resourceRecords_.push_back({result, exported});
        graph.resourceRecords_.back().image = image;
        graph.resourceRecords_.back().resourceKey = key;
        graph.importedImages_.emplace(key, result.id);
        graph.dirty_ = true;
        return result;
    }
};

} // namespace vernon::execution::detail

namespace {

using namespace vernon::execution;

GraphBuffer importBufferForTesting(ExecutionGraph &graph, VernonRhiBuffer buffer, bool exported = false) {
    return detail::ExecutionGraphTestAccess::importBuffer(graph, buffer, exported);
}

GraphImage importImageForTesting(ExecutionGraph &graph, VernonRhiImage image, VernonRhiImageView view,
                                 VernonRhiFormat format, uint32_t width, uint32_t height, uint32_t layers = 1,
                                 uint32_t samples = 1, bool exported = false,
                                 const VernonRhiImageSubresourceRange *subresources = nullptr) {
    return detail::ExecutionGraphTestAccess::importImage(graph, image, view, format, width, height, layers, samples,
                                                         exported, subresources);
}

class TestComputePass final : public ComputePass {
public:
    TestComputePass(std::string name, GraphResource readResource, GraphResource writeResource)
        : ComputePass(std::move(name)), readResource_(readResource), writeResource_(writeResource) {}

    void declare() override {
        if (readResource_.id != UINT32_MAX)
            read(readResource_);
        if (writeResource_.id != UINT32_MAX)
            write(writeResource_);
    }

    void addReadForTesting(GraphResource resource) { read(resource); }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphResource readResource_;
    GraphResource writeResource_;
};

class CpuComputePass final : public ComputePass {
public:
    CpuComputePass(std::string name, GraphResource resource, AccessMode access, std::vector<std::string> &events,
                   VernonRhiStatus status = VERNON_RHI_STATUS_OK)
        : ComputePass(std::move(name)), resource_(resource), access_(access), events_(events), status_(status) {}

    void declare() override {
        if (access_ == AccessMode::Read)
            read(resource_);
        else if (access_ == AccessMode::Write)
            write(resource_);
        else
            readWrite(resource_);
        setFlags(PassSideEffect);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override {
        events_.push_back(name());
        return status_;
    }

private:
    GraphResource resource_;
    AccessMode access_;
    std::vector<std::string> &events_;
    VernonRhiStatus status_;
};

struct IntegerBindingValue final : ExecutionBindingValue {
    explicit IntegerBindingValue(int value) : value(value) {}
    int value;
};

struct ScalarAutodiffValue final : GraphAutodiffValue {
    explicit ScalarAutodiffValue(double value) : value(value) {}

    uintptr_t logicalIdentity() const override { return reinterpret_cast<uintptr_t>(this); }
    uint64_t allocationBytes() const override { return sizeof(value); }

    std::shared_ptr<GraphAutodiffValue> add(const GraphAutodiffValue &other, std::string &error) const override {
        const auto *scalar = dynamic_cast<const ScalarAutodiffValue *>(&other);
        if (!scalar) {
            error = "incompatible test autodiff value";
            return {};
        }
        return std::make_shared<ScalarAutodiffValue>(value + scalar->value);
    }
    bool materialize(detail::RhiCommandPlanSink *sink, std::string &error) override {
        (void)sink;
        (void)error;
        return true;
    }

    double value;
};

class ByteCheckpointResource final : public GraphCheckpointResource {
public:
    explicit ByteCheckpointResource(uint64_t byteSize, uint64_t expectedAlignment = 1)
        : bytes_(byteSize), expectedAlignment_(expectedAlignment) {}

    uint64_t byteSize() const override { return bytes_.size(); }
    uint64_t alignment() const override { return expectedAlignment_; }
    void expectAlignment(uint64_t alignment) { expectedAlignment_ = alignment; }
    void failRestoreWithoutError(bool value = true) { failRestoreWithoutError_ = value; }
    uint32_t hostCopyToCount() const { return hostCopyToCount_; }
    uint32_t hostCopyFromCount() const { return hostCopyFromCount_; }
    float floatValue() const {
        float value = 0;
        std::memcpy(&value, bytes_.data(), sizeof(value));
        return value;
    }
    void setFloatValue(float value) { std::memcpy(bytes_.data(), &value, sizeof(value)); }

    bool copyTo(void *destination, uint64_t byteSize, std::string &error) const override {
        ++hostCopyToCount_;
        if (byteSize != bytes_.size() || reinterpret_cast<uintptr_t>(destination) % expectedAlignment_ != 0) {
            error = "test checkpoint copy size mismatch";
            return false;
        }
        std::memcpy(destination, bytes_.data(), bytes_.size());
        return true;
    }

    bool copyFrom(const void *source, uint64_t byteSize, std::string &error) override {
        ++hostCopyFromCount_;
        if (failRestoreWithoutError_)
            return false;
        if (byteSize != bytes_.size() || reinterpret_cast<uintptr_t>(source) % expectedAlignment_ != 0) {
            error = "test checkpoint restore size mismatch";
            return false;
        }
        std::memcpy(bytes_.data(), source, bytes_.size());
        return true;
    }

    bool copyRangeTo(uint64_t offset, void *destination, uint64_t byteSize, std::string &error) const override {
        ++hostCopyToCount_;
        if (offset > bytes_.size() || byteSize > bytes_.size() - offset) {
            error = "test checkpoint range exceeds resource";
            return false;
        }
        std::memcpy(destination, bytes_.data() + offset, byteSize);
        return true;
    }

    bool copyRangeFrom(uint64_t offset, const void *source, uint64_t byteSize, std::string &error) override {
        ++hostCopyFromCount_;
        if (failRestoreWithoutError_)
            return false;
        if (offset > bytes_.size() || byteSize > bytes_.size() - offset) {
            error = "test checkpoint range exceeds resource";
            return false;
        }
        std::memcpy(bytes_.data() + offset, source, byteSize);
        return true;
    }

private:
    std::vector<uint8_t> bytes_;
    uint64_t expectedAlignment_;
    bool failRestoreWithoutError_{};
    mutable uint32_t hostCopyToCount_{};
    uint32_t hostCopyFromCount_{};
};

GraphBuffer importCheckpointBuffer(ExecutionGraph &graph, uint64_t identity, bool exported = false) {
    return graph.importHostBuffer(identity, exported,
                                  std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float)));
}

class ScalarPassPullback final : public PassPullback {
public:
    ScalarPassPullback(std::vector<PassDerivativeMapping> gradients, double scale, bool fail, uint64_t tapeBytes = 8,
                       std::shared_ptr<std::atomic<uint32_t>> remainingFailures = {})
        : gradients_(std::move(gradients)), scale_(scale), fail_(fail), tapeStorage_(tapeBytes),
          remainingFailures_(std::move(remainingFailures)) {}

    bool apply(const NamedGraphAutodiffValues &cotangents, NamedGraphAutodiffValues &gradients,
               const PassPullbackApplyOptions &, detail::RhiCommandPlanSink *sink, std::string &error) override {
        uint32_t remaining = remainingFailures_ ? remainingFailures_->load() : 0;
        const bool oneShotFailure =
            remainingFailures_ && remaining &&
            remainingFailures_->compare_exchange_strong(remaining, remaining - 1, std::memory_order_relaxed);
        if (fail_ || oneShotFailure) {
            error = "test backward failure";
            return false;
        }
        double total = 0.0;
        for (const auto &cotangent : cotangents) {
            const auto scalar = std::dynamic_pointer_cast<ScalarAutodiffValue>(cotangent.second);
            if (!scalar) {
                error = "incompatible test cotangent";
                return false;
            }
            total += scalar->value;
        }
        for (const PassDerivativeMapping &gradient : gradients_)
            gradients.emplace_back(gradient.path, std::make_shared<ScalarAutodiffValue>(total * scale_));
        if (sink) {
            detail::RhiCommandExecutionPlan plan;
            detail::CommandNode node;
            node.kind = detail::CommandNodeKind::Derivative;
            node.queue = detail::CommandQueueClass::Compute;
            plan.commands.nodes.push_back(std::move(node));
            plan.encoders.push_back({[](void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }, nullptr});
            if (sink->append(std::move(plan)) != VERNON_RHI_STATUS_OK) {
                error = "cannot append scalar test pullback plan";
                return false;
            }
        }
        return true;
    }

    uint64_t estimatedTapeBytes() const override { return tapeStorage_.size(); }
    uint64_t logicalResidualBytes() const override { return 4; }
    uint64_t residentTapeBytes() const override { return tapeStorage_.size(); }
    uint64_t allocatedTapeBytes() const override { return tapeStorage_.size(); }
    uint64_t activeOperationCount() const override { return 1; }
    uint64_t recomputationCost() const override { return 0; }
    uint64_t tapeContextLimitBytes() const override { return 0; }

private:
    std::vector<PassDerivativeMapping> gradients_;
    double scale_;
    bool fail_;
    std::vector<std::byte> tapeStorage_;
    std::shared_ptr<std::atomic<uint32_t>> remainingFailures_;
};

class ScalarDifferentiablePass final : public ComputePass, public DifferentiablePass {
public:
    ScalarDifferentiablePass(std::string name, GraphResource input, GraphResource output, double scale,
                             bool failForward = false, bool failBackward = false,
                             std::shared_ptr<std::atomic<uint32_t>> forwardCount = {},
                             uint32_t failForwardAfter = UINT32_MAX, uint64_t estimatedResidualBytes = 8,
                             uint64_t actualTapeBytes = 0,
                             std::shared_ptr<std::atomic<uint32_t>> remainingBackwardFailures = {},
                             bool supportsReplay = true, bool throwForward = false)
        : ComputePass(std::move(name)), input_(input), output_(output), scale_(scale), failForward_(failForward),
          failBackward_(failBackward), forwardCount_(std::move(forwardCount)), failForwardAfter_(failForwardAfter),
          estimatedResidualBytes_(estimatedResidualBytes),
          actualTapeBytes_(actualTapeBytes ? actualTapeBytes : estimatedResidualBytes),
          remainingBackwardFailures_(std::move(remainingBackwardFailures)), supportsReplay_(supportsReplay),
          throwForward_(throwForward), gradients_({{"input", {DerivativeEndpointKind::Resource, input.id}}}),
          cotangents_({{"output", {DerivativeEndpointKind::Resource, output.id}}}) {}

    void declare() override {
        read(input_);
        write(output_);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }
    DifferentiablePass *differentiable() override { return this; }
    const std::vector<PassDerivativeMapping> &gradientMappings() const override { return gradients_; }
    const std::vector<PassDerivativeMapping> &cotangentMappings() const override { return cotangents_; }
    const std::vector<PassPrimalResourceMapping> &requiredPrimalResources() const override { return requiredPrimals_; }
    const std::vector<PassWriteFootprint> &writeFootprints() const override { return writeFootprints_; }
    const std::vector<PassWriteFootprint> &readFootprints() const override { return readFootprints_; }
    void setWriteFootprint(std::vector<GraphByteRange> ranges) { writeFootprints_ = {{output_.id, std::move(ranges)}}; }
    void setReadFootprint(std::vector<GraphByteRange> ranges) { readFootprints_ = {{input_.id, std::move(ranges)}}; }
    void setWriteFootprints(std::vector<std::vector<GraphByteRange>> declarations) {
        writeFootprints_.clear();
        for (std::vector<GraphByteRange> &ranges : declarations)
            writeFootprints_.push_back({output_.id, std::move(ranges)});
    }
    uint64_t estimatedResidualBytes() const override { return estimatedResidualBytes_; }
    uint64_t estimatedRetainedAllocationBytes() const override { return estimatedResidualBytes_; }
    uint64_t replayCost() const override { return 1; }
    bool hasCheckpointPlanningMetadata() const override { return true; }
    bool supportsReplay() const override { return supportsReplay_; }

    bool forward(ComputeEncoder *encoder, const ExecutionResources &, detail::RhiCommandExecutionPlan *plan,
                 std::unique_ptr<PassPullback> &pullback, std::string &error) override {
        if (throwForward_)
            throw std::runtime_error("test forward exception");
        const bool countedFailure = forwardCount_ && forwardCount_->fetch_add(1) >= failForwardAfter_;
        if (failForward_ || countedFailure) {
            error = "test forward failure";
            return false;
        }
        pullback = std::make_unique<ScalarPassPullback>(gradients_, scale_, failBackward_, actualTapeBytes_,
                                                        remainingBackwardFailures_);
        if (plan) {
            if (encoder) {
                error = "scalar test pass received both an encoder and command plan";
                return false;
            }
            detail::CommandNode node;
            node.kind = detail::CommandNodeKind::Derivative;
            node.queue = detail::CommandQueueClass::Compute;
            plan->commands.nodes.push_back(std::move(node));
            plan->encoders.push_back({[](void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }, nullptr});
        } else if (!encoder) {
            error = "scalar test pass requires an encoder or command plan";
            return false;
        }
        return true;
    }

    bool zeroCotangent(const std::string &, const ExecutionResources &, std::shared_ptr<GraphAutodiffValue> &value,
                       std::string &) override {
        value = std::make_shared<ScalarAutodiffValue>(0.0);
        return true;
    }

    bool implicitCotangent(const std::string &, const ExecutionResources &, std::shared_ptr<GraphAutodiffValue> &value,
                           std::string &) override {
        value = std::make_shared<ScalarAutodiffValue>(1.0);
        return true;
    }

private:
    GraphResource input_;
    GraphResource output_;
    double scale_;
    bool failForward_;
    bool failBackward_;
    std::shared_ptr<std::atomic<uint32_t>> forwardCount_;
    uint32_t failForwardAfter_;
    uint64_t estimatedResidualBytes_;
    uint64_t actualTapeBytes_;
    std::shared_ptr<std::atomic<uint32_t>> remainingBackwardFailures_;
    bool supportsReplay_;
    bool throwForward_;
    std::vector<PassDerivativeMapping> gradients_;
    std::vector<PassDerivativeMapping> cotangents_;
    std::vector<PassPrimalResourceMapping> requiredPrimals_{{"primal.input", input_.id}};
    std::vector<PassWriteFootprint> readFootprints_;
    std::vector<PassWriteFootprint> writeFootprints_;
};

class AliasedGradientPullback final : public PassPullback {
public:
    bool apply(const NamedGraphAutodiffValues &cotangents, NamedGraphAutodiffValues &gradients,
               const PassPullbackApplyOptions &, detail::RhiCommandPlanSink *sink, std::string &error) override {
        if (sink) {
            error = "aliased test pullback does not support planned apply";
            return false;
        }
        if (cotangents.size() != 1) {
            error = "aliased test pullback requires one cotangent";
            return false;
        }
        const auto cotangent = std::dynamic_pointer_cast<ScalarAutodiffValue>(cotangents.front().second);
        if (!cotangent) {
            error = "incompatible aliased test cotangent";
            return false;
        }
        auto shared = std::make_shared<ScalarAutodiffValue>(2.0 * cotangent->value);
        gradients.emplace_back("left_alias", shared);
        gradients.emplace_back("right_alias", std::move(shared));
        return true;
    }

    uint64_t estimatedTapeBytes() const override { return 0; }
    uint64_t logicalResidualBytes() const override { return 0; }
    uint64_t residentTapeBytes() const override { return 0; }
    uint64_t allocatedTapeBytes() const override { return 0; }
    uint64_t activeOperationCount() const override { return 1; }
    uint64_t recomputationCost() const override { return 0; }
    uint64_t tapeContextLimitBytes() const override { return 0; }
};

class AliasedGradientPass final : public ComputePass, public DifferentiablePass {
public:
    AliasedGradientPass(GraphResource input, GraphResource output)
        : ComputePass("aliased-gradient"), input_(input), output_(output),
          gradients_({{"left_alias", {DerivativeEndpointKind::Resource, input.id}},
                      {"right_alias", {DerivativeEndpointKind::Resource, input.id}}}),
          cotangents_({{"output", {DerivativeEndpointKind::Resource, output.id}}}) {}

    void declare() override {
        read(input_);
        write(output_);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }
    DifferentiablePass *differentiable() override { return this; }
    const std::vector<PassDerivativeMapping> &gradientMappings() const override { return gradients_; }
    const std::vector<PassDerivativeMapping> &cotangentMappings() const override { return cotangents_; }
    uint64_t estimatedResidualBytes() const override { return 0; }
    uint64_t estimatedRetainedAllocationBytes() const override { return 0; }
    uint64_t replayCost() const override { return 1; }
    bool hasCheckpointPlanningMetadata() const override { return true; }
    bool supportsReplay() const override { return true; }

    bool forward(ComputeEncoder *encoder, const ExecutionResources &, detail::RhiCommandExecutionPlan *plan,
                 std::unique_ptr<PassPullback> &pullback, std::string &error) override {
        if (!encoder || plan) {
            error = "aliased test pass does not support planned forward";
            return false;
        }
        pullback = std::make_unique<AliasedGradientPullback>();
        return true;
    }

    bool zeroCotangent(const std::string &, const ExecutionResources &, std::shared_ptr<GraphAutodiffValue> &value,
                       std::string &) override {
        value = std::make_shared<ScalarAutodiffValue>(0.0);
        return true;
    }

    bool implicitCotangent(const std::string &, const ExecutionResources &, std::shared_ptr<GraphAutodiffValue> &value,
                           std::string &) override {
        value = std::make_shared<ScalarAutodiffValue>(1.0);
        return true;
    }

private:
    GraphResource input_;
    GraphResource output_;
    std::vector<PassDerivativeMapping> gradients_;
    std::vector<PassDerivativeMapping> cotangents_;
};

class ResourceSquarePass final : public ComputePass, public DifferentiablePass {
public:
    ResourceSquarePass(std::string name, GraphResource input, GraphResource output,
                       std::shared_ptr<ByteCheckpointResource> inputState,
                       std::shared_ptr<ByteCheckpointResource> outputState)
        : ComputePass(std::move(name)), input_(input), output_(output), inputState_(std::move(inputState)),
          outputState_(std::move(outputState)), gradients_({{"input", {DerivativeEndpointKind::Resource, input.id}}}),
          cotangents_({{"output", {DerivativeEndpointKind::Resource, output.id}}}) {}

    void declare() override {
        read(input_);
        write(output_);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override {
        const float value = inputState_->floatValue();
        outputState_->setFloatValue(value * value);
        return VERNON_RHI_STATUS_OK;
    }
    DifferentiablePass *differentiable() override { return this; }
    const std::vector<PassDerivativeMapping> &gradientMappings() const override { return gradients_; }
    const std::vector<PassDerivativeMapping> &cotangentMappings() const override { return cotangents_; }
    uint64_t estimatedResidualBytes() const override { return 100; }
    uint64_t estimatedRetainedAllocationBytes() const override { return 100; }
    uint64_t replayCost() const override { return 1; }
    bool hasCheckpointPlanningMetadata() const override { return true; }
    bool supportsReplay() const override { return true; }

    bool forward(ComputeEncoder *encoder, const ExecutionResources &resources, detail::RhiCommandExecutionPlan *plan,
                 std::unique_ptr<PassPullback> &pullback, std::string &error) override {
        if (!encoder || plan) {
            error = "resource-square test pass does not support planned forward";
            return false;
        }
        const float value = inputState_->floatValue();
        execute(*encoder, resources);
        pullback = std::make_unique<ScalarPassPullback>(gradients_, 2.0 * value, false, estimatedResidualBytes());
        return true;
    }

    bool zeroCotangent(const std::string &, const ExecutionResources &, std::shared_ptr<GraphAutodiffValue> &value,
                       std::string &) override {
        value = std::make_shared<ScalarAutodiffValue>(0.0);
        return true;
    }

    bool implicitCotangent(const std::string &, const ExecutionResources &, std::shared_ptr<GraphAutodiffValue> &value,
                           std::string &) override {
        value = std::make_shared<ScalarAutodiffValue>(1.0);
        return true;
    }

private:
    GraphResource input_;
    GraphResource output_;
    std::shared_ptr<ByteCheckpointResource> inputState_;
    std::shared_ptr<ByteCheckpointResource> outputState_;
    std::vector<PassDerivativeMapping> gradients_;
    std::vector<PassDerivativeMapping> cotangents_;
};

class ParameterComputePass final : public ComputePass {
public:
    ParameterComputePass(ExecutionParameter parameter, std::vector<std::shared_ptr<const IntegerBindingValue>> &values)
        : ComputePass("parameter"), parameter_(parameter), values_(values) {}

    void declare() override { setFlags(PassSideEffect); }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &resources) override {
        values_.push_back(std::dynamic_pointer_cast<const IntegerBindingValue>(resources.binding(parameter_)));
        return values_.back() ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    ExecutionParameter parameter_;
    std::vector<std::shared_ptr<const IntegerBindingValue>> &values_;
};

class TestRenderPass final : public RenderPass {
public:
    TestRenderPass(std::string name, GraphImage target, VernonRhiLoadOperation load = VERNON_RHI_LOAD_PRESERVE,
                   VernonRhiStoreOperation store = VERNON_RHI_STORE_PRESERVE)
        : RenderPass(std::move(name)), target_(target), load_(load), store_(store) {}

    void declare() override {
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = load_;
        attachment.store = store_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    VernonRhiLoadOperation load_;
    VernonRhiStoreOperation store_;
};

class TestRenderReadPass final : public RenderPass {
public:
    TestRenderReadPass(std::string name, GraphImage target, GraphResource input)
        : RenderPass(std::move(name)), target_(target), input_(input) {}

    void declare() override {
        read(input_, VERNON_RHI_STATE_SHADER_READ, VERNON_RHI_STAGE_FRAGMENT);
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    GraphResource input_;
};

class TestRenderResourcePass final : public RenderPass {
public:
    TestRenderResourcePass(std::string name, GraphImage target, GraphResource resource, AccessMode access)
        : RenderPass(std::move(name)), target_(target), resource_(resource), access_(access) {}

    void declare() override {
        if (access_ == AccessMode::Read)
            read(resource_);
        else
            write(resource_);
        ColorAttachmentUse attachment{};
        attachment.image = target_;
        color(0, attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    GraphResource resource_;
    AccessMode access_;
};

class TestMultiRenderPass final : public RenderPass {
public:
    TestMultiRenderPass(std::string name, GraphImage first, GraphImage second)
        : RenderPass(std::move(name)), first_(first), second_(second) {}

    void declare() override {
        ColorAttachmentUse first{};
        first.image = first_;
        color(0, first);
        ColorAttachmentUse second{};
        second.image = second_;
        color(1, second);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage first_;
    GraphImage second_;
};

class TestDepthPass final : public RenderPass {
public:
    TestDepthPass(std::string name, GraphImage target, VernonRhiLoadOperation load, bool readOnlyDepth,
                  VernonRhiStoreOperation depthStore = VERNON_RHI_STORE_PRESERVE,
                  VernonRhiLoadOperation stencilLoad = VERNON_RHI_LOAD_DISCARD,
                  VernonRhiStoreOperation stencilStore = VERNON_RHI_STORE_DISCARD, bool readOnlyStencil = false)
        : RenderPass(std::move(name)), target_(target), load_(load), depthStore_(depthStore), stencilLoad_(stencilLoad),
          stencilStore_(stencilStore), readOnlyDepth_(readOnlyDepth), readOnlyStencil_(readOnlyStencil) {}

    void declare() override {
        DepthStencilAttachmentUse attachment{};
        attachment.image = target_;
        attachment.depthLoad = load_;
        attachment.depthStore = depthStore_;
        attachment.stencilLoad = stencilLoad_;
        attachment.stencilStore = stencilStore_;
        attachment.readOnlyDepth = readOnlyDepth_;
        attachment.readOnlyStencil = readOnlyStencil_;
        depth(attachment);
    }

    VernonRhiStatus execute(GraphicsEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage target_;
    VernonRhiLoadOperation load_;
    VernonRhiStoreOperation depthStore_;
    VernonRhiLoadOperation stencilLoad_;
    VernonRhiStoreOperation stencilStore_;
    bool readOnlyDepth_;
    bool readOnlyStencil_;
};

class TestImageSubresourcePass final : public ComputePass {
public:
    TestImageSubresourcePass(std::string name, GraphImage image, AccessMode access)
        : ComputePass(std::move(name)), image_(image), access_(access) {}

    void declare() override {
        if (access_ == AccessMode::Read)
            read(image_);
        else
            write(image_);
        setFlags(PassSideEffect);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    GraphImage image_;
    AccessMode access_;
};

class ConditionalFlagPass final : public ComputePass {
public:
    ConditionalFlagPass(std::string name, bool &sideEffect) : ComputePass(std::move(name)), sideEffect_(sideEffect) {}

    void declare() override {
        if (sideEffect_)
            setFlags(PassSideEffect);
    }

    VernonRhiStatus execute(ComputeEncoder &, const ExecutionResources &) override { return VERNON_RHI_STATUS_OK; }

private:
    bool &sideEffect_;
};

GraphResource none() { return {UINT32_MAX, ResourceKind::Buffer}; }

TEST(ExecutionGraph, InfersHazardsAndHonorsExplicitDependencies) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer source = importBufferForTesting(graph, {0, 1});
    const GraphBuffer intermediate = importBufferForTesting(graph, {1, 1});
    const GraphBuffer output = importBufferForTesting(graph, {2, 1}, true);
    auto &produce = graph.emplacePass<TestComputePass>("produce", source, intermediate);
    auto &consume = graph.emplacePass<TestComputePass>("consume", intermediate, output);
    consume.dependsOn(produce);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0, 1}));
    ASSERT_EQ(plan->scopes().size(), 2u);
    EXPECT_FALSE(plan->scopes()[0].rendering);
}

TEST(ExecutionGraph, CpuProviderExecutesCompiledRawWarAndWawSchedules) {
    const auto executeHazard = [](AccessMode firstAccess, AccessMode secondAccess) {
        ExecutionGraph graph;
        const GraphBuffer shared = graph.importHostBuffer(1, true);
        std::vector<std::string> events;
        graph.emplacePass<CpuComputePass>("first", shared, firstAccess, events);
        graph.emplacePass<CpuComputePass>("second", shared, secondAccess, events);

        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0, 1}));
        auto submission = plan->submit();
        EXPECT_EQ(submission.wait(), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(events, (std::vector<std::string>{"first", "second"}));
    };

    executeHazard(AccessMode::Write, AccessMode::Read);
    executeHazard(AccessMode::Read, AccessMode::Write);
    executeHazard(AccessMode::Write, AccessMode::Write);
}

TEST(ExecutionGraph, CpuProviderPropagatesFailureAndStopsSchedule) {
    ExecutionGraph graph;
    const GraphBuffer shared = graph.importHostBuffer(1, true);
    std::vector<std::string> events;
    graph.emplacePass<CpuComputePass>("first", shared, AccessMode::Write, events);
    graph.emplacePass<CpuComputePass>("failure", shared, AccessMode::ReadWrite, events,
                                      VERNON_RHI_STATUS_INTERNAL_ERROR);
    graph.emplacePass<CpuComputePass>("not-run", shared, AccessMode::Read, events);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto submission = plan->submit();
    EXPECT_EQ(submission.wait(), VERNON_RHI_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(events, (std::vector<std::string>{"first", "failure"}));
}

TEST(ExecutionGraph, CompiledPlanOwnsPassesAndSupportsRepeatedSubmissions) {
    std::vector<std::string> events;
    std::shared_ptr<CompiledExecutionGraph> plan;
    {
        ExecutionGraph graph;
        const GraphBuffer shared = graph.importHostBuffer(1, true);
        graph.emplacePass<CpuComputePass>("run", shared, AccessMode::Write, events);
        std::string error;
        plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_THROW(graph.emplacePass<CpuComputePass>("late", shared, AccessMode::Read, events), std::logic_error);
    }

    auto first = plan->submit();
    EXPECT_EQ(first.wait(), VERNON_RHI_STATUS_OK);
    auto second = plan->submit();
    EXPECT_EQ(second.wait(), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(events, (std::vector<std::string>{"run", "run"}));
}

TEST(ExecutionGraph, FreezesRetainedPassReferencesAfterCompilation) {
    ExecutionGraph graph;
    const GraphBuffer output = graph.importHostBuffer(1, true);
    auto &pass = graph.emplacePass<TestComputePass>("run", none(), output);
    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    EXPECT_THROW(pass.setFlags(PassNeverCull), std::logic_error);
    EXPECT_THROW(pass.dependsOn(pass), std::logic_error);
    EXPECT_THROW(pass.addReadForTesting(output), std::logic_error);
}

TEST(ExecutionGraph, SubmissionRetainsCompiledPlan) {
    std::vector<std::string> events;
    ExecutionGraph graph;
    const GraphBuffer shared = graph.importHostBuffer(1, true);
    graph.emplacePass<CpuComputePass>("run", shared, AccessMode::Write, events);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;

    auto submission = plan->submit();
    plan.reset();
    EXPECT_EQ(submission.state(), ExecutionSubmission::State::Succeeded);
    EXPECT_EQ(submission.wait(), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(events, (std::vector<std::string>{"run"}));
}

TEST(ExecutionGraph, FusesCompatibleRenderPassesAndSplitsCompute) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 64, 64, 1, 1, true);
    const GraphBuffer buffer = importBufferForTesting(graph, {0, 1}, true);
    graph.emplacePass<TestRenderPass>("first", color, VERNON_RHI_LOAD_CLEAR);
    graph.emplacePass<TestRenderPass>("second", color);
    graph.emplacePass<TestComputePass>("compute", none(), buffer);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    EXPECT_TRUE(plan->scopes()[0].rendering);
    EXPECT_EQ(plan->scopes()[0].passIndices, (std::vector<uint32_t>{0, 1}));
    EXPECT_FALSE(plan->scopes()[1].rendering);
}

TEST(ExecutionGraph, RejectsDependencyCycles) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    auto &first = graph.emplacePass<TestComputePass>("first", none(), none());
    auto &second = graph.emplacePass<TestComputePass>("second", none(), none());
    first.setFlags(PassSideEffect);
    second.setFlags(PassSideEffect);
    first.dependsOn(second);
    second.dependsOn(first);

    std::string error;
    EXPECT_FALSE(graph.compile(error));
    EXPECT_NE(error.find("cycle"), std::string::npos);
}

TEST(ExecutionGraph, CullsTransientPassesWithoutLiveConsumers) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer transient = importBufferForTesting(graph, {0, 1});
    graph.emplacePass<TestComputePass>("dead", none(), transient);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_TRUE(plan->schedule().empty());
    EXPECT_TRUE(plan->scopes().empty());
}

TEST(ExecutionGraph, RebuildClearsDeclarationDerivedFlags) {
    ExecutionGraph graph;
    bool sideEffect = true;
    graph.emplacePass<ConditionalFlagPass>("conditional", sideEffect);
    std::string error;
    ASSERT_TRUE(graph.validate(error)) << error;
    sideEffect = false;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    EXPECT_TRUE(compiled->schedule().empty());
}

TEST(ExecutionGraph, KeepsUnexportedAutodiffObjectiveProducerLive) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer objective = importCheckpointBuffer(graph, 2);
    graph.emplacePass<ScalarDifferentiablePass>("objective", input, objective, 2.0);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});

    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    EXPECT_EQ(compiled->schedule(), (std::vector<uint32_t>{0}));
}

TEST(ExecutionGraph, DeduplicatesImportsAndPromotesExportedResources) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer first = importBufferForTesting(graph, {4, 9});
    const GraphBuffer second = importBufferForTesting(graph, {4, 9}, true);
    EXPECT_EQ(first.id, second.id);
    EXPECT_EQ(first.graphIdentity, second.graphIdentity);

    const GraphImage firstView = importImageForTesting(graph, {7, 3}, {10, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16);
    const GraphImage secondView = importImageForTesting(graph, {7, 3}, {11, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16);
    EXPECT_EQ(firstView.id, secondView.id);
    EXPECT_NE(firstView.view.index, secondView.view.index);

    graph.emplacePass<TestComputePass>("write", none(), second);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0}));
}

TEST(ExecutionGraph, AliasedImportsPreserveRawWarAndWawHazards) {
    const auto expectBothPassesLive = [](AccessMode firstAccess, AccessMode secondAccess) {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphBuffer firstAlias = importBufferForTesting(graph, {4, 9}, secondAccess != AccessMode::Read);
        const GraphBuffer secondAlias = importBufferForTesting(graph, {4, 9});
        const GraphBuffer output = importBufferForTesting(graph, {5, 1}, secondAccess == AccessMode::Read);
        if (firstAccess == AccessMode::Read)
            graph.emplacePass<TestComputePass>("first", firstAlias, output);
        else
            graph.emplacePass<TestComputePass>("first", none(), firstAlias);
        if (secondAccess == AccessMode::Read)
            graph.emplacePass<TestComputePass>("second", secondAlias, output);
        else
            graph.emplacePass<TestComputePass>("second", none(), secondAlias);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_EQ(plan->schedule(), (std::vector<uint32_t>{0, 1}));
    };
    expectBothPassesLive(AccessMode::Write, AccessMode::Read);
    expectBothPassesLive(AccessMode::Read, AccessMode::Write);
    expectBothPassesLive(AccessMode::Write, AccessMode::Write);
}

TEST(ExecutionGraph, RejectsResourceFromAnotherGraphWithMatchingNumericId) {
    ExecutionGraph first({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    ExecutionGraph second({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer foreign = importBufferForTesting(first, {0, 1});
    const GraphBuffer output = importBufferForTesting(second, {1, 1}, true);
    second.emplacePass<TestComputePass>("foreign", foreign, output);

    std::string error;
    EXPECT_FALSE(second.validate(error));
    EXPECT_NE(error.find("another execution graph"), std::string::npos);
    error.clear();
    EXPECT_FALSE(second.compile(error));
    EXPECT_NE(error.find("another execution graph"), std::string::npos);
    EXPECT_NE(error.find("foreign"), std::string::npos);
}

TEST(ExecutionGraph, DerivesBarrierStageAccessAndStateFromUses) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphBuffer intermediate = importBufferForTesting(graph, {0, 1});
    const GraphImage color =
        importImageForTesting(graph, {1, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    graph.emplacePass<TestComputePass>("produce", none(), intermediate);
    graph.emplacePass<TestRenderReadPass>("consume", color, intermediate);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    ASSERT_EQ(plan->scopes()[1].barriers.size(), 1u);
    const VernonRhiBarrier &barrier = plan->scopes()[1].barriers.front();
    EXPECT_EQ(barrier.source_stage_mask, VERNON_RHI_STAGE_COMPUTE);
    EXPECT_EQ(barrier.destination_stage_mask, VERNON_RHI_STAGE_FRAGMENT);
    EXPECT_EQ(barrier.source_access, VERNON_RHI_ACCESS_SHADER_WRITE);
    EXPECT_EQ(barrier.destination_access, VERNON_RHI_ACCESS_SHADER_READ);
    EXPECT_EQ(barrier.old_state, VERNON_RHI_STATE_SHADER_WRITE);
    EXPECT_EQ(barrier.new_state, VERNON_RHI_STATE_SHADER_READ);
}

TEST(ExecutionGraph, TracksImageHazardsByParentAndSubresourceRange) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const VernonRhiImageSubresourceRange mip0{0, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_COLOR};
    const VernonRhiImageSubresourceRange mip1{1, 1, 0, 1, VERNON_RHI_IMAGE_ASPECT_COLOR};
    const GraphImage first =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, false, &mip0);
    const GraphImage second =
        importImageForTesting(graph, {0, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 1, 1, false, &mip1);
    ASSERT_EQ(first.id, second.id);
    graph.emplacePass<TestImageSubresourcePass>("write-mip-0", first, AccessMode::Write);
    graph.emplacePass<TestImageSubresourcePass>("read-mip-1", second, AccessMode::Read);
    graph.emplacePass<TestImageSubresourcePass>("read-mip-0", first, AccessMode::Read);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 3u);
    EXPECT_TRUE(plan->scopes()[1].barriers.empty());
    ASSERT_EQ(plan->scopes()[2].barriers.size(), 1u);
    const VernonRhiBarrier &barrier = plan->scopes()[2].barriers.front();
    EXPECT_EQ(barrier.image_subresources.base_mip_level, 0u);
    EXPECT_EQ(barrier.image_subresources.mip_level_count, 1u);
    EXPECT_EQ(barrier.image_subresources.base_array_layer, 0u);
    EXPECT_EQ(barrier.image_subresources.array_layer_count, 1u);
    EXPECT_EQ(barrier.image_subresources.aspects, VERNON_RHI_IMAGE_ASPECT_COLOR);
    EXPECT_EQ(barrier.old_state, VERNON_RHI_STATE_SHADER_WRITE);
    EXPECT_EQ(barrier.new_state, VERNON_RHI_STATE_SHADER_READ);
    EXPECT_EQ(barrier.source_access, VERNON_RHI_ACCESS_SHADER_WRITE);
    EXPECT_EQ(barrier.destination_access, VERNON_RHI_ACCESS_SHADER_READ);
}

TEST(ExecutionGraph, PreservesComputeImageViewMetadataAndRejectsSlicedImages) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const VernonRhiImageSubresourceRange range{2, 1, 3, 2, VERNON_RHI_IMAGE_ASPECT_COLOR};
        const GraphImage image =
            importImageForTesting(graph, {4, 2}, {7, 3}, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 2, 1, true, &range);
        auto &pass = graph.emplacePass<TestImageSubresourcePass>("image", image, AccessMode::Read);
        std::string error;
        ASSERT_TRUE(graph.validate(error)) << error;
        ASSERT_EQ(pass.uses().size(), 1u);
        ASSERT_TRUE(pass.uses().front().image.has_value());
        EXPECT_EQ(pass.uses().front().image->view.index, image.view.index);
        EXPECT_EQ(pass.uses().front().image->subresources.base_mip_level, 2u);
        EXPECT_EQ(pass.uses().front().image->subresources.base_array_layer, 3u);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage image =
            importImageForTesting(graph, {4, 2}, {7, 3}, VERNON_RHI_FORMAT_RGBA8_UNORM, 8, 8, 1, 1, true);
        graph.emplacePass<TestComputePass>("sliced-image", none(), image);
        std::string error;
        EXPECT_FALSE(graph.validate(error));
    }
}

TEST(ExecutionGraph, RejectsInvalidAttachmentFormatAndExtent) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("depth-as-color", depth);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("depth-as-color"), std::string::npos);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage first =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
        const GraphImage second =
            importImageForTesting(graph, {1, 1}, {1, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 32, 16, 1, 1, true);
        graph.emplacePass<TestMultiRenderPass>("mismatched", first, second);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("incompatible extent"), std::string::npos);
    }
}

TEST(ExecutionGraph, RejectsClearOnReadOnlyDepthAndSplitsReadOnlyChanges) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("read-only-clear", depth, VERNON_RHI_LOAD_CLEAR, true);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("read-only-clear"), std::string::npos);
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depth =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("write", depth, VERNON_RHI_LOAD_PRESERVE, false);
        auto &read = graph.emplacePass<TestDepthPass>("read", depth, VERNON_RHI_LOAD_PRESERVE, true);
        read.setFlags(PassSideEffect);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        EXPECT_EQ(plan->scopes().size(), 2u);
    }
}

TEST(ExecutionGraph, TracksWritableStencilAndRejectsReadOnlyStencilMutation) {
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depthStencil =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("stencil-write", depthStencil, VERNON_RHI_LOAD_PRESERVE, true,
                                         VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_PRESERVE,
                                         false);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        ASSERT_EQ(plan->schedule(), (std::vector<uint32_t>{0}));
    }
    {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage depthStencil =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1, 1, true);
        graph.emplacePass<TestDepthPass>("read-only-stencil-clear", depthStencil, VERNON_RHI_LOAD_PRESERVE, false,
                                         VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_CLEAR, VERNON_RHI_STORE_PRESERVE,
                                         true);
        std::string error;
        EXPECT_FALSE(graph.compile(error));
        EXPECT_NE(error.find("read-only-stencil-clear"), std::string::npos);
    }
}

TEST(ExecutionGraph, SplitsScopesWhenIntermediateDiscardCannotBeRepresented) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    graph.emplacePass<TestRenderPass>("discard-output", color, VERNON_RHI_LOAD_CLEAR, VERNON_RHI_STORE_DISCARD);
    graph.emplacePass<TestRenderPass>("preserve-input", color, VERNON_RHI_LOAD_PRESERVE);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    EXPECT_TRUE(plan->scopes()[0].rendering);
    EXPECT_TRUE(plan->scopes()[1].rendering);
}

TEST(ExecutionGraph, FusesAnIntermediateClearButSplitsAnIntermediateDiscardLoad) {
    const auto compileScopes = [](VernonRhiLoadOperation secondLoad) {
        ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        const GraphImage color =
            importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
        graph.emplacePass<TestRenderPass>("first", color);
        graph.emplacePass<TestRenderPass>("second", color, secondLoad);
        std::string error;
        auto plan = graph.compile(error);
        EXPECT_TRUE(plan) << error;
        return plan ? plan->scopes().size() : 0;
    };
    EXPECT_EQ(compileScopes(VERNON_RHI_LOAD_CLEAR), 1u);
    EXPECT_EQ(compileScopes(VERNON_RHI_LOAD_DISCARD), 2u);
}

TEST(ExecutionGraph, SplitsRenderScopesAcrossDiscardedStencilBoundary) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage depthStencil =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1, 1, true);
    graph.emplacePass<TestDepthPass>("discard-stencil", depthStencil, VERNON_RHI_LOAD_PRESERVE, false,
                                     VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_DISCARD);
    graph.emplacePass<TestDepthPass>("preserve-stencil", depthStencil, VERNON_RHI_LOAD_PRESERVE, false,
                                     VERNON_RHI_STORE_PRESERVE, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_PRESERVE);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->scopes().size(), 2u);
}

TEST(ExecutionGraph, SplitsRenderScopesForNonAttachmentHazards) {
    ExecutionGraph graph({static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    const GraphImage color =
        importImageForTesting(graph, {0, 1}, {0, 1}, VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 16, 1, 1, true);
    const GraphBuffer buffer = importBufferForTesting(graph, {1, 1});
    graph.emplacePass<TestRenderResourcePass>("write", color, buffer, AccessMode::Write);
    graph.emplacePass<TestRenderResourcePass>("read", color, buffer, AccessMode::Read);

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    ASSERT_EQ(plan->scopes().size(), 2u);
    ASSERT_EQ(plan->scopes()[1].barriers.size(), 2u);
    EXPECT_EQ(plan->scopes()[1].barriers[0].source_stage_mask, 0u);
    EXPECT_EQ(plan->scopes()[1].barriers[0].source_access,
              VERNON_RHI_ACCESS_COLOR_READ | VERNON_RHI_ACCESS_COLOR_WRITE);
}

TEST(ExecutionGraph, ValidatesParameterSchemasAndInitialBindings) {
    ExecutionGraph graph;
    const ExecutionParameter parameter = graph.parameter("value");
    EXPECT_THROW(graph.parameter("value"), std::invalid_argument);
    std::vector<std::shared_ptr<const IntegerBindingValue>> observed;
    graph.emplacePass<ParameterComputePass>(parameter, observed);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;

    EXPECT_THROW(plan->createBindings({}), std::invalid_argument);
    const auto value = std::make_shared<IntegerBindingValue>(1);
    EXPECT_THROW(plan->createBindings({{parameter, value}, {parameter, value}}), std::invalid_argument);

    ExecutionGraph foreignGraph;
    const ExecutionParameter foreign = foreignGraph.parameter("foreign");
    EXPECT_THROW(plan->createBindings({{foreign, value}}), std::invalid_argument);
}

TEST(ExecutionGraph, SnapshotsSparseParameterUpdatesAndRetainsSubmissionValues) {
    ExecutionGraph graph;
    const ExecutionParameter dynamic = graph.parameter("dynamic");
    const ExecutionParameter constant = graph.parameter("constant");
    std::vector<std::shared_ptr<const IntegerBindingValue>> observed;
    graph.emplacePass<ParameterComputePass>(dynamic, observed);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;

    const auto firstDynamic = std::make_shared<IntegerBindingValue>(1);
    const auto constantValue = std::make_shared<IntegerBindingValue>(7);
    auto bindings = plan->createBindings({{dynamic, firstDynamic}, {constant, constantValue}});
    const auto firstSnapshot = bindings.snapshot();
    EXPECT_EQ(firstSnapshot, bindings.snapshot());
    EXPECT_EQ(firstSnapshot->at(constant), constantValue);

    ExecutionSubmission first = plan->submit(firstSnapshot);
    const auto secondDynamic = std::make_shared<IntegerBindingValue>(2);
    bindings.set(dynamic, secondDynamic);
    const auto secondSnapshot = bindings.snapshot();
    EXPECT_NE(firstSnapshot, secondSnapshot);
    EXPECT_EQ(firstSnapshot->at(dynamic), firstDynamic);
    EXPECT_EQ(secondSnapshot->at(dynamic), secondDynamic);
    EXPECT_EQ(secondSnapshot->at(constant), constantValue);
    ExecutionSubmission second = plan->submit(secondSnapshot);

    ASSERT_EQ(observed.size(), 2u);
    EXPECT_EQ(observed[0]->value, 1);
    EXPECT_EQ(observed[1]->value, 2);
    plan.reset();
    EXPECT_EQ(first.wait(), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(second.wait(), VERNON_RHI_STATUS_OK);
}

TEST(ExecutionGraphAutodiff, ComposesNativePullbacksAndSupportsReuseAfterPlanRelease) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer intermediate = importCheckpointBuffer(graph, 2);
    const GraphBuffer objective = importCheckpointBuffer(graph, 3, true);
    graph.emplacePass<ScalarDifferentiablePass>("first", input, intermediate, 2.0);
    graph.emplacePass<ScalarDifferentiablePass>("second", intermediate, objective, 5.0);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    EXPECT_EQ(pullback->forwardSubmission().state(), ExecutionSubmission::State::Succeeded);
    EXPECT_EQ(pullback->estimatedTapeBytes(), 16u);
    EXPECT_EQ(pullback->logicalResidualBytes(), 8u);
    EXPECT_EQ(pullback->residentTapeBytes(), 16u);
    EXPECT_EQ(pullback->allocatedTapeBytes(), 16u);
    EXPECT_EQ(pullback->checkpointBytes(), 0u);
    EXPECT_EQ(pullback->peakRuntimeManagedBytes(), 24u);
    const std::vector<GraphAutodiffPassTelemetry> telemetry = pullback->passTelemetry();
    ASSERT_EQ(telemetry.size(), 2u);
    EXPECT_EQ(telemetry[0].passName, "first");
    EXPECT_EQ(telemetry[1].passName, "second");
    EXPECT_EQ(telemetry[0].scheduleOffset, 0u);
    EXPECT_EQ(telemetry[1].scheduleOffset, 1u);
    for (const GraphAutodiffPassTelemetry &pass : telemetry) {
        EXPECT_EQ(pass.residualSourceKind, "capture");
        EXPECT_EQ(pass.estimatedTapeBytes, 8u);
        EXPECT_EQ(pass.logicalResidualBytes, 4u);
        EXPECT_EQ(pass.residentTapeBytes, 8u);
        EXPECT_EQ(pass.allocatedTapeBytes, 8u);
        EXPECT_EQ(pass.checkpointBytes, 0u);
        EXPECT_EQ(pass.activeOperationCount, 1u);
        EXPECT_EQ(pass.recomputationCost, 0u);
        EXPECT_FALSE(pass.controlHistoryBytes);
    }
    plan.reset();

    auto explicitSubmission = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(3.0)}}, false);
    ASSERT_TRUE(explicitSubmission->wait(error)) << error;
    ASSERT_EQ(explicitSubmission->gradients().size(), 1u);
    const auto explicitGradient =
        std::dynamic_pointer_cast<ScalarAutodiffValue>(explicitSubmission->gradients().front().second);
    ASSERT_TRUE(explicitGradient);
    EXPECT_DOUBLE_EQ(explicitGradient->value, 30.0);

    auto implicitSubmission = pullback->submit({}, true);
    ASSERT_TRUE(implicitSubmission->wait(error)) << error;
    const auto implicitGradient =
        std::dynamic_pointer_cast<ScalarAutodiffValue>(implicitSubmission->gradients().front().second);
    ASSERT_TRUE(implicitGradient);
    EXPECT_DOUBLE_EQ(implicitGradient->value, 10.0);
}

TEST(ExecutionGraphAutodiff, SerializesConcurrentApplicationsWithoutSharingGradients) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer intermediate = importCheckpointBuffer(graph, 2);
    const GraphBuffer objective = importCheckpointBuffer(graph, 3, true);
    graph.emplacePass<ScalarDifferentiablePass>("first", input, intermediate, 2.0);
    graph.emplacePass<ScalarDifferentiablePass>("second", intermediate, objective, 5.0);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;

    std::vector<std::future<double>> applications;
    for (uint32_t index = 1; index <= 8; ++index)
        applications.push_back(std::async(std::launch::async, [pullback, index] {
            auto submission = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(index)}}, false);
            std::string applyError;
            if (!submission->wait(applyError))
                throw std::runtime_error(applyError);
            const auto gradient =
                std::dynamic_pointer_cast<ScalarAutodiffValue>(submission->gradients().front().second);
            if (!gradient)
                throw std::runtime_error("concurrent graph VJP returned an incompatible gradient");
            return gradient->value;
        }));
    for (uint32_t index = 0; index < applications.size(); ++index)
        EXPECT_DOUBLE_EQ(applications[index].get(), static_cast<double>((index + 1) * 10));
}

TEST(ExecutionGraphAutodiff, AccumulatesBranchedObjectivesByEndpointIdentity) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer leftObjective = importCheckpointBuffer(graph, 2, true);
    const GraphBuffer rightObjective = importCheckpointBuffer(graph, 3, true);
    graph.emplacePass<ScalarDifferentiablePass>("left", input, leftObjective, 2.0);
    graph.emplacePass<ScalarDifferentiablePass>("right", input, rightObjective, 3.0);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"left", {DerivativeEndpointKind::Resource, leftObjective.id}},
                                {"right", {DerivativeEndpointKind::Resource, rightObjective.id}}});
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    auto submission = pullback->submit(
        {{"left", std::make_shared<ScalarAutodiffValue>(5.0)}, {"right", std::make_shared<ScalarAutodiffValue>(7.0)}},
        false);
    ASSERT_TRUE(submission->wait(error)) << error;
    const auto gradient = std::dynamic_pointer_cast<ScalarAutodiffValue>(submission->gradients().front().second);
    ASSERT_TRUE(gradient);
    EXPECT_DOUBLE_EQ(gradient->value, 31.0);
}

TEST(ExecutionGraphAutodiff, DeduplicatesAliasedContributionsAndPublishesGradientsTransactionally) {
    const auto compileAliasedGraph = [](bool includeDisconnected, std::string &error) {
        ExecutionGraph graph;
        const GraphBuffer input = importCheckpointBuffer(graph, 1);
        const GraphBuffer objective = importCheckpointBuffer(graph, 2, true);
        graph.emplacePass<AliasedGradientPass>(input, objective);
        std::vector<NamedDerivativeEndpoint> inputs{{"input", {DerivativeEndpointKind::Resource, input.id}}};
        if (includeDisconnected) {
            const GraphBuffer disconnected = graph.importHostBuffer(3);
            inputs.push_back({"disconnected", {DerivativeEndpointKind::Resource, disconnected.id}});
        }
        graph.setAutodiffEndpoints(std::move(inputs),
                                   {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
        return graph.compile(error);
    };

    std::string error;
    auto compiled = compileAliasedGraph(false, error);
    ASSERT_TRUE(compiled) << error;
    auto pullback = compiled->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    auto succeeded = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(3.0)}}, false);
    ASSERT_TRUE(succeeded->wait(error)) << error;
    ASSERT_EQ(succeeded->gradients().size(), 1u);
    const auto gradient = std::dynamic_pointer_cast<ScalarAutodiffValue>(succeeded->gradients().front().second);
    ASSERT_TRUE(gradient);
    EXPECT_DOUBLE_EQ(gradient->value, 6.0);

    compiled = compileAliasedGraph(true, error);
    ASSERT_TRUE(compiled) << error;
    pullback = compiled->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    auto failed = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(3.0)}}, false);
    EXPECT_FALSE(failed->wait(error));
    EXPECT_NE(error.find("disconnected"), std::string::npos);
    EXPECT_TRUE(failed->gradients().empty());
}

TEST(ExecutionGraphAutodiff, ReturnsMultipleDeclaredInputGradients) {
    ExecutionGraph graph;
    const GraphBuffer leftInput = importCheckpointBuffer(graph, 1);
    const GraphBuffer rightInput = importCheckpointBuffer(graph, 2);
    const GraphBuffer leftObjective = importCheckpointBuffer(graph, 3, true);
    const GraphBuffer rightObjective = importCheckpointBuffer(graph, 4, true);
    graph.emplacePass<ScalarDifferentiablePass>("left", leftInput, leftObjective, 2.0);
    graph.emplacePass<ScalarDifferentiablePass>("right", rightInput, rightObjective, 3.0);
    graph.setAutodiffEndpoints({{"left_input", {DerivativeEndpointKind::Resource, leftInput.id}},
                                {"right_input", {DerivativeEndpointKind::Resource, rightInput.id}}},
                               {{"left", {DerivativeEndpointKind::Resource, leftObjective.id}},
                                {"right", {DerivativeEndpointKind::Resource, rightObjective.id}}});
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    auto submission = pullback->submit(
        {{"left", std::make_shared<ScalarAutodiffValue>(5.0)}, {"right", std::make_shared<ScalarAutodiffValue>(7.0)}},
        false);
    ASSERT_TRUE(submission->wait(error)) << error;
    ASSERT_EQ(submission->gradients().size(), 2u);
    const auto left = std::dynamic_pointer_cast<ScalarAutodiffValue>(submission->gradients()[0].second);
    const auto right = std::dynamic_pointer_cast<ScalarAutodiffValue>(submission->gradients()[1].second);
    ASSERT_TRUE(left);
    ASSERT_TRUE(right);
    EXPECT_DOUBLE_EQ(left->value, 10.0);
    EXPECT_DOUBLE_EQ(right->value, 21.0);
}

TEST(ExecutionGraphAutodiff, PublishesNoPartialGradientsForDisconnectedInput) {
    ExecutionGraph graph;
    const GraphBuffer connected = importCheckpointBuffer(graph, 1);
    const GraphBuffer disconnected = graph.importHostBuffer(2);
    const GraphBuffer objective = importCheckpointBuffer(graph, 3, true);
    graph.emplacePass<ScalarDifferentiablePass>("connected", connected, objective, 2.0);
    graph.setAutodiffEndpoints({{"connected", {DerivativeEndpointKind::Resource, connected.id}},
                                {"disconnected", {DerivativeEndpointKind::Resource, disconnected.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    auto pullback = compiled->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    auto backward = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    EXPECT_FALSE(backward->wait(error));
    EXPECT_TRUE(backward->gradients().empty());
    EXPECT_NE(error.find("disconnected"), std::string::npos);
}

TEST(ExecutionGraphAutodiff, CompilationRejectsDuplicateLogicalEndpoints) {
    ExecutionGraph duplicateInputs;
    const GraphBuffer input = duplicateInputs.importHostBuffer(1);
    const GraphBuffer objective = duplicateInputs.importHostBuffer(2, true);
    duplicateInputs.emplacePass<ScalarDifferentiablePass>("step", input, objective, 2.0);
    duplicateInputs.setAutodiffEndpoints({{"first", {DerivativeEndpointKind::Resource, input.id}},
                                          {"second", {DerivativeEndpointKind::Resource, input.id}}},
                                         {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
    std::string error;
    EXPECT_FALSE(duplicateInputs.compile(error));
    EXPECT_NE(error.find("invalid differentiable input endpoint"), std::string::npos);

    ExecutionGraph duplicateObjectives;
    const GraphBuffer secondInput = duplicateObjectives.importHostBuffer(3);
    const GraphBuffer secondObjective = duplicateObjectives.importHostBuffer(4, true);
    duplicateObjectives.emplacePass<ScalarDifferentiablePass>("step", secondInput, secondObjective, 2.0);
    duplicateObjectives.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, secondInput.id}}},
                                             {{"first", {DerivativeEndpointKind::Resource, secondObjective.id}},
                                              {"second", {DerivativeEndpointKind::Resource, secondObjective.id}}});
    error.clear();
    EXPECT_FALSE(duplicateObjectives.compile(error));
    EXPECT_NE(error.find("invalid objective endpoint"), std::string::npos);
}

TEST(ExecutionGraphAutodiff, CompilationRejectsNonDifferentiableWriteOnActiveReversePath) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer objective = importCheckpointBuffer(graph, 2, true);
    std::vector<std::string> events;
    graph.emplacePass<CpuComputePass>("non-differentiable", objective, AccessMode::Write, events);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});

    std::string error;
    auto plan = graph.compile(error);
    EXPECT_FALSE(plan);
    EXPECT_NE(error.find("non-differentiable pass"), std::string::npos);
}

TEST(ExecutionGraphAutodiff, RollsBackPartialForwardPullbacksOnFailure) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer intermediate = importCheckpointBuffer(graph, 2);
    const GraphBuffer objective = importCheckpointBuffer(graph, 3, true);
    graph.emplacePass<ScalarDifferentiablePass>("first", input, intermediate, 2.0);
    graph.emplacePass<ScalarDifferentiablePass>("failure", intermediate, objective, 5.0, true);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});

    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_FALSE(plan->vjp({}, error));
    EXPECT_EQ(error, "test forward failure");
}

TEST(ExecutionGraphAutodiff, RestoresWritableResourcesAfterForwardFailureOrException) {
    const auto run = [](bool throwForward) {
        ExecutionGraph graph;
        auto inputState = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
        auto intermediateState = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
        auto objectiveState = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
        inputState->setFloatValue(3.0f);
        intermediateState->setFloatValue(-7.0f);
        objectiveState->setFloatValue(-11.0f);
        const GraphBuffer input = graph.importHostBuffer(1, false, inputState);
        const GraphBuffer intermediate = graph.importHostBuffer(2, false, intermediateState);
        const GraphBuffer objective = graph.importHostBuffer(3, true, objectiveState);
        graph.emplacePass<ResourceSquarePass>("mutate", input, intermediate, inputState, intermediateState);
        graph.emplacePass<ScalarDifferentiablePass>("failure", intermediate, objective, 1.0, !throwForward, false,
                                                    nullptr, UINT32_MAX, 8, 0, nullptr, true, throwForward);
        graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                                   {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
        std::string error;
        auto compiled = graph.compile(error);
        EXPECT_TRUE(compiled) << error;
        if (!compiled)
            return;
        EXPECT_FALSE(compiled->vjp({}, error));
        if (throwForward)
            EXPECT_NE(error.find("test forward exception"), std::string::npos);
        else
            EXPECT_EQ(error, "test forward failure");
        EXPECT_FLOAT_EQ(intermediateState->floatValue(), -7.0f);
        EXPECT_FLOAT_EQ(objectiveState->floatValue(), -11.0f);
    };
    run(false);
    run(true);
}

TEST(ExecutionGraphAutodiff, ReportsRollbackFailureWhenCheckpointResourceProvidesNoError) {
    ExecutionGraph graph;
    auto inputState = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
    auto intermediateState = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
    auto objectiveState = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
    inputState->setFloatValue(3.0f);
    intermediateState->setFloatValue(-7.0f);
    objectiveState->setFloatValue(-11.0f);
    const GraphBuffer input = graph.importHostBuffer(1, false, inputState);
    const GraphBuffer intermediate = graph.importHostBuffer(2, false, intermediateState);
    const GraphBuffer objective = graph.importHostBuffer(3, true, objectiveState);
    graph.emplacePass<ResourceSquarePass>("mutate", input, intermediate, inputState, intermediateState);
    graph.emplacePass<ScalarDifferentiablePass>("failure", intermediate, objective, 1.0, true);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    intermediateState->failRestoreWithoutError();

    EXPECT_FALSE(compiled->vjp({}, error));
    EXPECT_EQ(error, "test forward failure; rollback failed: graph VJP forward rollback failed");
}

TEST(ExecutionGraphAutodiff, PublishesNoGradientsWhenBackwardFails) {
    ExecutionGraph graph;
    const GraphBuffer input = importCheckpointBuffer(graph, 1);
    const GraphBuffer objective = importCheckpointBuffer(graph, 2, true);
    graph.emplacePass<ScalarDifferentiablePass>("failure", input, objective, 2.0, false, true);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, objective.id}}});
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;

    auto submission = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    EXPECT_FALSE(submission->wait(error));
    EXPECT_EQ(error, "test backward failure");
    EXPECT_TRUE(submission->gradients().empty());
    EXPECT_EQ(submission->state(), GraphBackwardSubmission::State::Failed);
}

TEST(ExecutionGraphAutodiff, RestoresStateAndPublishesNoGradientsWhenReplayFails) {
    ExecutionGraph graph;
    std::vector<GraphBuffer> resources;
    std::vector<std::shared_ptr<ByteCheckpointResource>> resourceStates;
    for (uint64_t identity = 1; identity <= 5; ++identity) {
        auto state = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
        resources.push_back(graph.importHostBuffer(identity, identity == 5, state));
        resourceStates.push_back(std::move(state));
    }
    const auto forwardCount = std::make_shared<std::atomic<uint32_t>>(0);
    graph.emplacePass<ScalarDifferentiablePass>("replay-failure", resources[0], resources[1], 2.0, false, false,
                                                forwardCount, 1, 100);
    for (uint32_t index = 1; index < 4; ++index)
        graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                    resources[index + 1], 2.0, false, false, nullptr, UINT32_MAX, 100);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
    graph.planAutodiffCheckpoints(250);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    EXPECT_GT(pullback->checkpointBytes(), 0u);
    EXPECT_GE(pullback->peakRuntimeManagedBytes(), pullback->allocatedTapeBytes() + pullback->checkpointBytes());
    resourceStates.back()->failRestoreWithoutError();

    auto submission = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    EXPECT_FALSE(submission->wait(error));
    EXPECT_EQ(error, "test forward failure; final-state restoration failed: graph final-state restoration failed");
    EXPECT_TRUE(submission->gradients().empty());
    EXPECT_EQ(submission->state(), GraphBackwardSubmission::State::Failed);

    auto retry = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    EXPECT_FALSE(retry->wait(error));
    EXPECT_EQ(error, "graph pullback is no longer reusable after failed checkpoint replay");
}

TEST(ExecutionGraphAutodiff, ReconstructsRetainedTapesAfterRecoverableBackwardFailure) {
    ExecutionGraph graph;
    std::vector<GraphBuffer> resources;
    for (uint64_t identity = 1; identity <= 5; ++identity)
        resources.push_back(graph.importHostBuffer(
            identity, identity == 5, std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float))));
    const auto remainingFailures = std::make_shared<std::atomic<uint32_t>>(1);
    graph.emplacePass<ScalarDifferentiablePass>("one-shot-failure", resources[0], resources[1], 2.0, false, false,
                                                nullptr, UINT32_MAX, 100, 0, remainingFailures);
    for (uint32_t index = 1; index < 4; ++index)
        graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                    resources[index + 1], 2.0, false, false, nullptr, UINT32_MAX, 100);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
    graph.planAutodiffCheckpoints(250);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    ASSERT_GT(pullback->checkpointBytes(), 0u);

    auto failed = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    EXPECT_FALSE(failed->wait(error));
    EXPECT_EQ(error, "test backward failure");
    auto retry = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    ASSERT_TRUE(retry->wait(error)) << error;
    const auto gradient = std::dynamic_pointer_cast<ScalarAutodiffValue>(retry->gradients().front().second);
    ASSERT_TRUE(gradient);
    EXPECT_DOUBLE_EQ(gradient->value, 16.0);
}

TEST(ExecutionGraphAutodiff, RejectsRuntimeTapeAllocationAbovePlannedBudget) {
    ExecutionGraph graph;
    const GraphBuffer input = graph.importHostBuffer(1, false, std::make_shared<ByteCheckpointResource>(sizeof(float)));
    const GraphBuffer output = graph.importHostBuffer(2, true, std::make_shared<ByteCheckpointResource>(sizeof(float)));
    graph.emplacePass<ScalarDifferentiablePass>("underestimated", input, output, 2.0, false, false, nullptr, UINT32_MAX,
                                                1, 64);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, output.id}}});
    graph.planAutodiffCheckpoints(17);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    EXPECT_FALSE(plan->vjp({}, error));
    EXPECT_EQ(error, "graph autodiff runtime allocation exceeds the compiled checkpoint memory budget");
}

TEST(ExecutionGraphAutodiff, RejectsCheckpointReplayForNonReplayablePass) {
    ExecutionGraph graph;
    const GraphBuffer input = graph.importHostBuffer(1, false, std::make_shared<ByteCheckpointResource>(sizeof(float)));
    const GraphBuffer intermediate =
        graph.importHostBuffer(2, false, std::make_shared<ByteCheckpointResource>(sizeof(float)));
    const GraphBuffer output = graph.importHostBuffer(3, true, std::make_shared<ByteCheckpointResource>(sizeof(float)));
    auto &sideEffectful = graph.emplacePass<ScalarDifferentiablePass>("side-effectful", input, intermediate, 2.0, false,
                                                                      false, nullptr, UINT32_MAX, 100);
    sideEffectful.setFlags(PassSideEffect);
    graph.emplacePass<ScalarDifferentiablePass>("pure", intermediate, output, 2.0, false, false, nullptr, UINT32_MAX,
                                                100);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, input.id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, output.id}}});
    graph.planAutodiffCheckpoints(150);
    std::string error;
    EXPECT_FALSE(graph.compile(error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule requires replaying a non-replayable node");
}

TEST(ExecutionGraphAutodiff, ReplaysFromOriginalInputAndRestoresCallerState) {
    ExecutionGraph graph;
    std::vector<GraphBuffer> resources;
    std::vector<std::shared_ptr<ByteCheckpointResource>> states;
    for (uint64_t identity = 1; identity <= 4; ++identity) {
        auto state = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
        states.push_back(state);
        resources.push_back(graph.importHostBuffer(identity, identity == 4, state));
    }
    states.front()->setFloatValue(2.0f);
    for (uint32_t index = 0; index < 3; ++index)
        graph.emplacePass<ResourceSquarePass>("square" + std::to_string(index), resources[index], resources[index + 1],
                                              states[index], states[index + 1]);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
    graph.planAutodiffCheckpoints(200);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    const AutodiffDagCheckpointPlan *checkpointPlan = plan->autodiffCheckpointPlan();
    ASSERT_NE(checkpointPlan, nullptr);
    EXPECT_EQ(checkpointPlan->initialStateBytes, sizeof(float));
    EXPECT_GT(checkpointPlan->cuts.size(), 0u);
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    EXPECT_FLOAT_EQ(states.back()->floatValue(), 256.0f);

    states.front()->setFloatValue(10.0f);
    states.back()->setFloatValue(999.0f);
    auto submission = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
    ASSERT_TRUE(submission->wait(error)) << error;
    const auto gradient = std::dynamic_pointer_cast<ScalarAutodiffValue>(submission->gradients().front().second);
    ASSERT_TRUE(gradient);
    EXPECT_DOUBLE_EQ(gradient->value, 1024.0);
    EXPECT_FLOAT_EQ(states.front()->floatValue(), 10.0f);
    EXPECT_FLOAT_EQ(states.back()->floatValue(), 999.0f);
}

TEST(ExecutionGraphAutodiff, RestoresInitialInputsFirstReadAfterCheckpointCut) {
    ExecutionGraph graph;
    std::vector<std::shared_ptr<ByteCheckpointResource>> states;
    std::vector<GraphBuffer> resources;
    for (uint64_t identity = 1; identity <= 5; ++identity) {
        auto state = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
        states.push_back(state);
        resources.push_back(graph.importHostBuffer(identity, false, state));
    }
    states[0]->setFloatValue(2.0f);
    states[3]->setFloatValue(2.0f);
    graph.emplacePass<ResourceSquarePass>("early-square-0", resources[0], resources[1], states[0], states[1]);
    graph.emplacePass<ResourceSquarePass>("early-square-1", resources[1], resources[2], states[1], states[2]);
    graph.emplacePass<ResourceSquarePass>("late-square", resources[3], resources[4], states[3], states[4]);
    graph.setAutodiffEndpoints({{"early_input", {DerivativeEndpointKind::Resource, resources[0].id}},
                                {"late_input", {DerivativeEndpointKind::Resource, resources[3].id}}},
                               {{"early_objective", {DerivativeEndpointKind::Resource, resources[2].id}},
                                {"late_objective", {DerivativeEndpointKind::Resource, resources[4].id}}});
    graph.planAutodiffCheckpoints(230);
    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    ASSERT_FALSE(compiled->autodiffCheckpointPlan()->cuts.empty());
    auto pullback = compiled->vjp({}, error);
    ASSERT_TRUE(pullback) << error;

    states[3]->setFloatValue(10.0f);
    for (uint32_t application = 0; application < 2; ++application) {
        auto backward = pullback->submit({{"early_objective", std::make_shared<ScalarAutodiffValue>(0.0)},
                                          {"late_objective", std::make_shared<ScalarAutodiffValue>(1.0)}},
                                         false);
        ASSERT_TRUE(backward->wait(error)) << error;
        const auto lateGradient = std::dynamic_pointer_cast<ScalarAutodiffValue>(backward->gradients()[1].second);
        ASSERT_TRUE(lateGradient);
        EXPECT_DOUBLE_EQ(lateGradient->value, 4.0);
        EXPECT_FLOAT_EQ(states[3]->floatValue(), 10.0f);
    }
}

TEST(ExecutionGraphAutodiff, UsesAlignedCheckpointStorageAcrossReusableApplications) {
    ExecutionGraph graph;
    std::vector<GraphBuffer> resources;
    std::vector<std::shared_ptr<ByteCheckpointResource>> checkpointResources;
    for (uint64_t identity = 1; identity <= 5; ++identity) {
        auto checkpoint = std::make_shared<ByteCheckpointResource>(sizeof(float), 64);
        checkpointResources.push_back(checkpoint);
        resources.push_back(graph.importHostBuffer(identity, identity == 5, std::move(checkpoint)));
    }
    for (uint32_t index = 0; index < 4; ++index)
        graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                    resources[index + 1], 2.0, false, false, nullptr, UINT32_MAX, 100);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
    graph.planAutodiffCheckpoints(268);
    std::string error;
    auto plan = graph.compile(error);
    ASSERT_TRUE(plan) << error;
    const AutodiffDagCheckpointPlan *checkpointPlan = plan->autodiffCheckpointPlan();
    ASSERT_NE(checkpointPlan, nullptr);
    EXPECT_EQ(checkpointPlan->restorationBytes, 5 * sizeof(float));
    EXPECT_EQ(checkpointPlan->peakBytes, 268u);
    for (const AutodiffCheckpointResource &checkpoint : checkpointPlan->checkpointResources) {
        ASSERT_LT(checkpoint.producer + 1, checkpointResources.size());
        checkpointResources[checkpoint.producer + 1]->expectAlignment(checkpoint.alignment);
    }
    auto pullback = plan->vjp({}, error);
    ASSERT_TRUE(pullback) << error;
    EXPECT_EQ(pullback->checkpointBytes(), checkpointPlan->persistentCheckpointBytes);
    uint64_t attributedCheckpointBytes = 0;
    for (const GraphAutodiffPassTelemetry &pass : pullback->passTelemetry())
        attributedCheckpointBytes += pass.checkpointBytes;
    EXPECT_EQ(attributedCheckpointBytes, pullback->checkpointBytes());
    const uint64_t retainedTapeBytes = pullback->allocatedTapeBytes();

    for (uint32_t application = 0; application < 2; ++application) {
        auto submission = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
        ASSERT_TRUE(submission->wait(error)) << error;
        const auto gradient = std::dynamic_pointer_cast<ScalarAutodiffValue>(submission->gradients().front().second);
        ASSERT_TRUE(gradient);
        EXPECT_DOUBLE_EQ(gradient->value, 16.0);
        EXPECT_EQ(pullback->allocatedTapeBytes(), retainedTapeBytes);
    }
    EXPECT_GE(pullback->peakRuntimeManagedBytes(), pullback->allocatedTapeBytes() + pullback->checkpointBytes());
    EXPECT_GT(pullback->recomputationFactor(), 1.0);
}

TEST(ExecutionGraphAutodiff, MetalSnapshotsAndCheckpointsRemainDeviceLocal) {
    VernonRhiOwnedDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
    deviceDescriptor.backend = VERNON_RHI_BACKEND_METAL;
    VernonRhiDevice device = vernonRhiCreateDevice(&deviceDescriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        GTEST_SKIP() << "Metal device is unavailable";

    std::vector<VernonRhiBuffer> buffers;
    std::vector<std::shared_ptr<ByteCheckpointResource>> states;
    std::string error;
    {
        ExecutionGraph graph(device);
        std::vector<GraphBuffer> resources;
        VernonRhiBufferDescriptor bufferDescriptor{};
        bufferDescriptor.struct_size = sizeof(bufferDescriptor);
        bufferDescriptor.size = sizeof(float);
        bufferDescriptor.alignment = alignof(float);
        bufferDescriptor.usage =
            VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
        bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
        for (uint32_t index = 0; index < 5; ++index) {
            VernonRhiBuffer buffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
            ASSERT_EQ(vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer), VERNON_RHI_STATUS_OK);
            buffers.push_back(buffer);
            auto state = std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float));
            states.push_back(state);
            resources.push_back(graph.importBuffer(buffer, index == 4, std::move(state)));
        }
        for (uint32_t index = 0; index < 4; ++index)
            graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                        resources[index + 1], 2.0, false, false, nullptr, UINT32_MAX,
                                                        100);
        graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                                   {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
        graph.planAutodiffCheckpoints(250);
        auto compiled = graph.compile(error);
        ASSERT_TRUE(compiled) << error;
        auto pullback = compiled->vjp({}, error);
        ASSERT_TRUE(pullback) << error;
        ASSERT_GT(pullback->checkpointBytes(), 0u);
        auto backward = pullback->submit({{"objective", std::make_shared<ScalarAutodiffValue>(1.0)}}, false);
        ASSERT_TRUE(backward->wait(error)) << error;
        for (const auto &state : states) {
            EXPECT_EQ(state->hostCopyToCount(), 0u);
            EXPECT_EQ(state->hostCopyFromCount(), 0u);
        }
    }
    for (VernonRhiBuffer buffer : buffers)
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(device, buffer), VERNON_RHI_STATUS_OK);
    vernonRhiDestroyDevice(device);
}

TEST(ExecutionGraphAutodiff, PlansDeclaredDirtyRangesAndKeepsUnknownFootprintsConservative) {
    auto compilePlan = [](bool declareRanges) {
        ExecutionGraph graph;
        std::vector<GraphBuffer> resources;
        for (uint64_t identity = 1; identity <= 5; ++identity)
            resources.push_back(
                graph.importHostBuffer(identity, false, std::make_shared<ByteCheckpointResource>(16, alignof(float))));
        for (uint32_t index = 0; index < 4; ++index) {
            auto &pass = graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                                     resources[index + 1], 2.0, false, false, nullptr,
                                                                     UINT32_MAX, 100);
            if (declareRanges) {
                pass.setReadFootprint({{4, sizeof(float)}});
                pass.setWriteFootprint({{4, sizeof(float)}});
            }
        }
        graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                                   {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
        graph.planAutodiffCheckpoints(declareRanges ? 390 : 450);
        std::string error;
        auto compiled = graph.compile(error);
        EXPECT_TRUE(compiled) << error;
        return compiled;
    };

    auto ranged = compilePlan(true);
    ASSERT_TRUE(ranged);
    ASSERT_FALSE(ranged->autodiffCheckpointPlan()->cuts.empty());
    EXPECT_EQ(ranged->autodiffCheckpointPlan()->restorationBytes, 5u * sizeof(float));
    EXPECT_EQ(ranged->autodiffCheckpointPlan()->transactionBytes, 4u * sizeof(float));

    auto conservative = compilePlan(false);
    ASSERT_TRUE(conservative);
    ASSERT_FALSE(conservative->autodiffCheckpointPlan()->cuts.empty());
    EXPECT_EQ(conservative->autodiffCheckpointPlan()->restorationBytes, 5u * 16u);
    EXPECT_EQ(conservative->autodiffCheckpointPlan()->transactionBytes, 4u * 16u);
}

TEST(ExecutionGraphAutodiff, DuplicateFootprintsUnionAndConservativeDeclarationWins) {
    auto restorationBytes = [](std::vector<std::vector<GraphByteRange>> declarations) {
        const bool conservative =
            std::any_of(declarations.begin(), declarations.end(), [](const auto &ranges) { return ranges.empty(); });
        ExecutionGraph graph;
        std::vector<GraphBuffer> resources;
        for (uint64_t identity = 1; identity <= 5; ++identity)
            resources.push_back(
                graph.importHostBuffer(identity, false, std::make_shared<ByteCheckpointResource>(16, alignof(float))));
        for (uint32_t index = 0; index < 4; ++index) {
            auto &pass = graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                                     resources[index + 1], 2.0, false, false, nullptr,
                                                                     UINT32_MAX, 100);
            pass.setWriteFootprints(declarations);
        }
        graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                                   {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
        graph.planAutodiffCheckpoints(conservative ? 450 : 400);
        std::string error;
        auto compiled = graph.compile(error);
        EXPECT_TRUE(compiled) << error;
        return compiled ? compiled->autodiffCheckpointPlan()->restorationBytes : 0;
    };

    EXPECT_EQ(restorationBytes({{{0, sizeof(float)}}, {{8, sizeof(float)}}}), 16u + 8u * sizeof(float));
    EXPECT_EQ(restorationBytes({{{0, sizeof(float)}}, {}}), 5u * 16u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, StoresImmutablePlanInCompiledGraph) {
    ExecutionGraph graph;
    std::vector<GraphBuffer> resources;
    for (uint64_t identity = 1; identity <= 5; ++identity)
        resources.push_back(graph.importHostBuffer(
            identity, identity == 5, std::make_shared<ByteCheckpointResource>(sizeof(float), alignof(float))));
    for (uint32_t index = 0; index < 4; ++index)
        graph.emplacePass<ScalarDifferentiablePass>("step" + std::to_string(index), resources[index],
                                                    resources[index + 1], 2.0, false, false, nullptr, UINT32_MAX, 100);
    graph.setAutodiffEndpoints({{"input", {DerivativeEndpointKind::Resource, resources.front().id}}},
                               {{"objective", {DerivativeEndpointKind::Resource, resources.back().id}}});
    graph.planAutodiffCheckpoints(250);
    std::string error;
    auto compiled = graph.compile(error);
    ASSERT_TRUE(compiled) << error;
    const AutodiffDagCheckpointPlan *plan = compiled->autodiffCheckpointPlan();
    ASSERT_NE(plan, nullptr);
    EXPECT_FALSE(plan->cuts.empty());
    EXPECT_LE(plan->peakBytes, 250u);
    EXPECT_THROW(graph.planAutodiffCheckpoints(0), std::logic_error);
}

struct TestDagNode {
    std::vector<uint32_t> predecessors;
    uint64_t checkpointBytes;
    uint64_t residualBytes;
    uint64_t replayCost;
    bool replayable;
    uint64_t checkpointAlignment{1};
};

std::vector<detail::AutodiffDagNode> valueDag(std::initializer_list<TestDagNode> nodes) {
    std::vector<detail::AutodiffDagNode> result;
    result.reserve(nodes.size());
    for (const TestDagNode &source : nodes) {
        detail::AutodiffDagNode node;
        node.predecessors = source.predecessors;
        node.residualBytes = source.residualBytes;
        node.retainedAllocationBytes = source.residualBytes;
        node.forwardPeakBytes = source.residualBytes;
        node.replayCost = source.replayCost;
        node.replayable = source.replayable;
        if (source.checkpointBytes)
            node.outputs.push_back({{static_cast<uint32_t>(result.size()), 1},
                                    source.checkpointBytes,
                                    source.checkpointAlignment,
                                    true,
                                    {}});
        result.push_back(std::move(node));
    }
    for (uint32_t consumer = 0; consumer < result.size(); ++consumer)
        for (uint32_t producer : result[consumer].predecessors)
            if (producer < result.size() && !result[producer].outputs.empty() &&
                std::find(result[producer].outputs[0].consumers.begin(), result[producer].outputs[0].consumers.end(),
                          consumer) == result[producer].outputs[0].consumers.end())
                result[producer].outputs[0].consumers.push_back(consumer);
    return result;
}

TEST(ExecutionGraphAutodiffCheckpointTest, PlansDagCutsFromLiveResources) {
    std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 4, 10, 1, true, 4},
        {{0}, 3, 10, 2, true},
        {{0}, 5, 10, 3, true},
        {{1, 2}, 0, 10, 4, true},
    });
    nodes[0].resourceReloadCost = 2;
    nodes[1].recomputationCost = 3;
    nodes[0].requiredVersions.push_back({"primal.initial", {7, 0}, UINT32_MAX});
    nodes[3].requiredVersions.push_back({"primal.join", nodes[0].outputs[0].version, 0});
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 27, plan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    EXPECT_FALSE(plan.cuts.empty());
    for (const AutodiffLivenessCut &cut : plan.cuts)
        for (uint32_t resource : cut.checkpointResources)
            EXPECT_LT(resource, plan.checkpointResources.size());
    EXPECT_LE(plan.peakBytes, 27u);
    EXPECT_EQ(plan.selectedPolicy, "min_runtime");
    EXPECT_EQ(plan.captureStoreBytes, 80u);
    EXPECT_EQ(plan.backwardLoadBytes, 40u);
    EXPECT_EQ(plan.checkpointCopyBytes, 2 * plan.persistentCheckpointBytes);
    EXPECT_EQ(plan.resourceReloadCost, 2u);
    EXPECT_EQ(plan.recomputationCost, 3u);
    EXPECT_EQ(plan.logicalResidualBytes, 40u);
    EXPECT_EQ(plan.retainedAllocationBytes, 40u);
    EXPECT_EQ(plan.maximumForwardPeakBytes, 10u);
    ASSERT_EQ(plan.passVersions.size(), nodes.size());
    ASSERT_EQ(plan.requiredVersions.size(), 2u);
    EXPECT_EQ(plan.requiredVersions[0].source, AutodiffVersionSource::RetainedOwner);
    EXPECT_EQ(plan.requiredVersions[1].source, AutodiffVersionSource::Checkpoint);
    EXPECT_EQ(plan.requiredVersions[1].version.resource, nodes[0].outputs[0].version.resource);
    EXPECT_EQ(plan.requiredVersions[1].version.epoch, nodes[0].outputs[0].version.epoch);
    EXPECT_EQ(plan.weightedRuntimeCost, plan.captureStoreBytes + plan.backwardLoadBytes + plan.checkpointCopyBytes +
                                            4 * plan.resourceReloadCost + 8 * plan.recomputationCost +
                                            16 * plan.replayCost);
}

TEST(ExecutionGraphAutodiffCheckpointTest, RejectsReplayThatViolatesDeterministicReductionConstraints) {
    std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 4, 10, 1, true, 4},
        {{0}, 3, 10, 2, true},
        {{0}, 5, 10, 3, true},
        {{1, 2}, 0, 10, 4, true},
    });
    nodes[1].deterministicReductionLegal = false;
    AutodiffDagCheckpointPlan plan;
    std::string error;
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(nodes, 27, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule violates deterministic reduction constraints");
}

TEST(ExecutionGraphAutodiffCheckpointTest, SupportsInternalMemoryAndRuntimePolicies) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 1, 10, 1, true},
        {{0}, 1, 10, 1, true},
        {{1}, 0, 10, 1, true},
    });
    AutodiffDagCheckpointPlan runtimePlan;
    AutodiffDagCheckpointPlan memoryPlan;
    AutodiffDagCheckpointPlan balancedPlan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 64, runtimePlan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 64, memoryPlan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinMemory))
        << error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 64, balancedPlan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::Balanced))
        << error;
    EXPECT_EQ(runtimePlan.selectedPolicy, "min_runtime");
    EXPECT_TRUE(runtimePlan.cuts.empty());
    EXPECT_EQ(memoryPlan.selectedPolicy, "min_memory");
    EXPECT_LT(memoryPlan.peakBytes, runtimePlan.peakBytes);
    EXPECT_GT(memoryPlan.cuts.size(), runtimePlan.cuts.size());
    EXPECT_EQ(balancedPlan.selectedPolicy, "balanced");
    const auto score = [](const AutodiffDagCheckpointPlan &plan) { return plan.peakBytes + plan.weightedRuntimeCost; };
    EXPECT_LE(score(balancedPlan), score(runtimePlan));
    EXPECT_LE(score(balancedPlan), score(memoryPlan));
}

TEST(ExecutionGraphAutodiffCheckpointTest, DoesNotCheckpointSchedulingOnlyPredecessors) {
    std::vector<detail::AutodiffDagNode> nodes =
        valueDag({{{}, 10, 100, 1, true}, {{0}, 10, 100, 1, true}, {{1}, 0, 100, 1, true}});
    nodes[0].outputs[0].consumers.clear();
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 200, plan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    ASSERT_EQ(plan.cuts.size(), 1u);
    EXPECT_EQ(plan.cuts[0].scheduleOffset, 1u);
    EXPECT_TRUE(plan.checkpointResources.empty());
}

TEST(ExecutionGraphAutodiffCheckpointTest, CheckpointsOnlyLiveOutputs) {
    std::vector<detail::AutodiffDagNode> nodes =
        valueDag({{{}, 1, 100, 1, true}, {{0}, 0, 100, 1, true}, {{1}, 0, 100, 1, true}});
    nodes[0].outputs.push_back({{99, 1}, 1000, 8, true, {}});
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 101, plan, error)) << error;
    ASSERT_EQ(plan.checkpointResources.size(), 1u);
    EXPECT_EQ(plan.checkpointResources[0].version.resource, 0u);
    EXPECT_EQ(plan.persistentCheckpointBytes, 1u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, NoCutPlanAllocatesOnlyTransactionSnapshot) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({{{}, 4, 16, 1, true}, {{0}, 4, 16, 1, true}});
    auto noCheckpointNodes = nodes;
    for (auto &node : noCheckpointNodes)
        for (auto &output : node.outputs)
            output.checkpointable = false;
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(
        detail::planDagAutodiffCheckpoints(noCheckpointNodes, 160, plan, error, 64, false, 128, true, 128, true))
        << error;
    EXPECT_TRUE(plan.cuts.empty());
    EXPECT_EQ(plan.initialStateBytes, 0u);
    EXPECT_EQ(plan.restorationBytes, 0u);
    EXPECT_EQ(plan.transactionBytes, 128u);
    EXPECT_EQ(plan.peakBytes, 160u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, TracksSharedCheckpointCutLifetimes) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 2, 10, 1, true},
        {{}, 1, 10, 1, true},
        {{1}, 1, 10, 1, true},
        {{2}, 1, 10, 1, true},
        {{0, 3}, 0, 10, 1, true},
    });
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 24, plan, error, 0, true, 0, true, 0, true, 0,
                                                   detail::AutodiffCheckpointPolicy::MinRuntime))
        << error;
    ASSERT_EQ(plan.cuts.size(), 2u);
    ASSERT_FALSE(plan.checkpointResources.empty());
    EXPECT_TRUE(
        std::any_of(plan.checkpointResources.begin(), plan.checkpointResources.end(),
                    [](const AutodiffCheckpointResource &resource) { return resource.firstCut < resource.lastCut; }));
    for (const AutodiffCheckpointResource &resource : plan.checkpointResources)
        EXPECT_LE(resource.firstCut, resource.lastCut);
    ASSERT_EQ(plan.replaySegments.size(), 3u);
    for (const AutodiffReplaySegment &segment : plan.replaySegments)
        for (uint32_t resource : segment.releaseCheckpointResources)
            EXPECT_LT(resource, plan.checkpointResources.size());
    EXPECT_LE(plan.peakBytes, 24u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, SplitsTiedDagPeakSegmentsTogether) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 2, 10, 1, true},
        {{0}, 2, 10, 1, true},
        {{1}, 2, 10, 1, true},
        {{2}, 0, 10, 1, true},
    });
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 16, plan, error)) << error;
    EXPECT_EQ(plan.cuts.size(), 3u);
    EXPECT_EQ(plan.replaySegments.size(), 4u);
    EXPECT_EQ(plan.persistentCheckpointBytes, 6u);
    EXPECT_EQ(plan.peakBytes, 16u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, AcceptsTemporaryPeakIncreaseNeededForFeasibleCuts) {
    std::vector<detail::AutodiffDagNode> nodes(3);
    for (auto &node : nodes) {
        node.residualBytes = 10;
        node.retainedAllocationBytes = 10;
        node.forwardPeakBytes = 10;
        node.replayCost = 1;
        node.replayable = true;
    }
    nodes[0].outputs.push_back({{0, 1}, 15, 1, true, {2}});
    nodes[2].predecessors.push_back(0);
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 25, plan, error)) << error;
    EXPECT_EQ(plan.cuts.size(), 2u);
    EXPECT_EQ(plan.peakBytes, 25u);
    EXPECT_EQ(plan.persistentCheckpointBytes, 15u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, ReplacesEarlierCutsToFindFeasibleFrontier) {
    const std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 7, 15, 1, true},
        {{0}, 19, 18, 1, true},
        {{1}, 10, 13, 1, true},
        {{2}, 0, 20, 1, true},
    });
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 48, plan, error)) << error;
    ASSERT_EQ(plan.cuts.size(), 2u);
    EXPECT_EQ(plan.cuts[0].scheduleOffset, 1u);
    EXPECT_EQ(plan.cuts[1].scheduleOffset, 3u);
    EXPECT_EQ(plan.peakBytes, 48u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, ChargesPhysicalRetainedAllocationInsteadOfLogicalPayload) {
    std::vector<detail::AutodiffDagNode> nodes = valueDag({
        {{}, 0, 1, 1, true},
        {{0}, 0, 1, 1, true},
    });
    nodes[0].retainedAllocationBytes = 24;
    nodes[0].forwardPeakBytes = 24;
    nodes[1].retainedAllocationBytes = 24;
    nodes[1].forwardPeakBytes = 24;
    AutodiffDagCheckpointPlan plan;
    std::string error;
    ASSERT_TRUE(detail::planDagAutodiffCheckpoints(nodes, 48, plan, error)) << error;
    EXPECT_EQ(plan.peakBytes, 48u);
    EXPECT_EQ(plan.replaySegments.front().retainedAllocationBytes, 48u);
    EXPECT_EQ(plan.replaySegments.front().logicalResidualBytes, 2u);
    EXPECT_EQ(plan.logicalResidualBytes, 2u);
    EXPECT_EQ(plan.retainedAllocationBytes, 48u);
}

TEST(ExecutionGraphAutodiffCheckpointTest, RejectsInvalidOrUnbudgetableDags) {
    AutodiffDagCheckpointPlan plan;
    std::string error;
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(valueDag({{{1}, 1, 1, 1, true}}), 1, plan, error));
    EXPECT_EQ(error, "autodiff DAG nodes are not in topological order");
    EXPECT_FALSE(
        detail::planDagAutodiffCheckpoints(valueDag({{{}, 1, 1, 1, true}, {{0, 0}, 1, 1, 1, true}}), 2, plan, error));
    EXPECT_EQ(error, "autodiff DAG node contains a duplicate predecessor");
    auto invalidVersion = valueDag({{{}, 1, 1, 1, true}, {{0}, 0, 1, 1, true}});
    invalidVersion[1].requiredVersions.push_back({"primal.missing", {999, 1}, 0});
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(invalidVersion, 2, plan, error));
    EXPECT_EQ(error, "autodiff DAG contains an invalid required resource version");
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(valueDag({{{}, 1, 1, 1, false}}), 0, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule cannot satisfy the memory budget");

    const auto nonReplayableTail = valueDag({{{}, 1, 10, 1, true}, {{0}, 1, 10, 1, true}, {{1}, 0, 1, 1, false}});
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(nonReplayableTail, 12, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule requires replaying a non-replayable node");

    const std::vector<detail::AutodiffDagNode> diamond = valueDag({
        {{}, 4, 10, 1, true},
        {{0}, 3, 10, 1, true},
        {{0}, 5, 10, 1, true},
        {{1, 2}, 0, 10, 1, true},
    });
    EXPECT_FALSE(detail::planDagAutodiffCheckpoints(diamond, 21, plan, error));
    EXPECT_EQ(error, "autodiff DAG checkpoint schedule cannot satisfy the memory budget");
}

} // namespace
