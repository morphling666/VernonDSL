#ifndef VERNON_EXECUTION_GRAPH_H
#define VERNON_EXECUTION_GRAPH_H

#include "VernonRHI.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::execution {

class ExecutionGraph;
class CompiledExecutionGraph;
class ExecutionSubmission;
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

class ExecutionBindings {
public:
    const std::shared_ptr<const ExecutionBindingValue> &at(ExecutionParameter parameter) const;
    size_t size() const { return values_.size(); }

private:
    friend class CompiledExecutionGraph;
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
    const std::vector<GraphResource> &all() const { return resources_; }
    bool buffer(GraphBuffer resource, VernonRhiBuffer &output) const;
    bool hasBindings() const { return static_cast<bool>(bindings_); }
    const std::shared_ptr<const ExecutionBindingValue> &binding(ExecutionParameter parameter) const;

private:
    friend class CompiledExecutionGraph;
    ExecutionResources(const std::vector<GraphResource> &resources, const std::vector<VernonRhiBuffer> &buffers,
                       std::shared_ptr<const ExecutionBindings> bindings)
        : resources_(resources), buffers_(buffers), bindings_(std::move(bindings)) {}

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

    std::string name_;
    uint32_t flags_{};
    ExecutionGraph *owner_{};
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
};

struct CompiledScope {
    bool rendering{};
    std::vector<uint32_t> passIndices;
    std::vector<VernonRhiBarrier> barriers;
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
    GraphBuffer importHostBuffer(uint64_t identity, bool exported = false);
    GraphBuffer importBuffer(VernonRhiBuffer buffer, bool exported = false);
    GraphImage importImage(VernonRhiImage image, VernonRhiImageView view, bool exported = false);
    ExecutionParameter parameter(std::string name);
    std::shared_ptr<CompiledExecutionGraph> compile(std::string &error);
    bool validate(std::string &error) const;

private:
    friend class ExecutionPass;
    friend struct detail::ExecutionGraphTestAccess;

    bool buildPlan(std::string &error);

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
    const std::vector<uint32_t> &schedule() const;
    const std::vector<CompiledScope> &scopes() const;

private:
    friend class ExecutionGraph;
    friend class ExecutionSubmission;
    struct State;
    explicit CompiledExecutionGraph(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
};

} // namespace vernon::execution

#endif
