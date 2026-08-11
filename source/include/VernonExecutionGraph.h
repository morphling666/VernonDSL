#ifndef VERNON_EXECUTION_GRAPH_H
#define VERNON_EXECUTION_GRAPH_H

#include "VernonRHI.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::execution {

class ExecutionGraph;
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

class ExecutionResources {
public:
    const std::vector<GraphResource> &all() const { return resources_; }
    bool buffer(GraphBuffer resource, VernonRhiBuffer &output) const;

private:
    friend class ExecutionGraph;
    ExecutionResources(const std::vector<GraphResource> &resources, const std::vector<VernonRhiBuffer> &buffers)
        : resources_(resources), buffers_(buffers) {}

    const std::vector<GraphResource> &resources_;
    const std::vector<VernonRhiBuffer> &buffers_;
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
    bool compile(std::string &error);
    bool validate(std::string &error) const;
    VernonRhiStatus execute();

    const std::vector<uint32_t> &schedule() const { return schedule_; }
    const std::vector<CompiledScope> &scopes() const { return scopes_; }
    const VernonRhiCommandEncoderStats &lastStats() const { return lastStats_; }

private:
    friend class ExecutionPass;
    friend struct detail::ExecutionGraphTestAccess;
    struct ResourceRecord {
        GraphResource resource;
        bool exported{};
        bool graphOwned{};
        uint64_t resourceKey{};
        std::vector<uint64_t> imageViewKeys;
        VernonRhiBuffer buffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        VernonRhiImage image{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    };

    enum class Provider : uint8_t { Cpu, Rhi };

    VernonRhiStatus executeCpu();
    VernonRhiStatus executeRhi();

    Provider provider_{Provider::Cpu};
    VernonRhiDevice device_;
    uint64_t graphIdentity_{};
    std::vector<std::unique_ptr<ExecutionPass>> passes_;
    std::vector<GraphResource> resources_;
    std::vector<ResourceRecord> resourceRecords_;
    std::unordered_map<uint64_t, uint32_t> importedBuffers_;
    std::unordered_map<uint64_t, uint32_t> importedHostBuffers_;
    std::unordered_map<uint64_t, uint32_t> importedImages_;
    std::vector<uint32_t> schedule_;
    std::vector<CompiledScope> scopes_;
    VernonRhiCommandEncoderStats lastStats_{};
    bool dirty_{true};
};

} // namespace vernon::execution

#endif
