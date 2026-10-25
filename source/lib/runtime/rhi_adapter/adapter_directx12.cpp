#include "adapter_common.h"
#include "adapter_directx12_test_hooks.h"
#include "rhi/rhi_internal.h"
#include "runtime/vertex_attribute_capabilities.h"

#if defined(VERNON_HAS_DIRECTX12_RHI)

#include "rhi/directx12_backend.h"
#include "rhi/image_descriptor_validation.h"
#include "runtime/interface_names.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

constexpr size_t kMaxGroups = VERNON_RUNTIME_PROVIDER_MAX_GROUPS;
constexpr size_t kMaxVertexBindings = VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS;
constexpr uint32_t kNone = UINT32_MAX;
constexpr uint32_t kRootDwordLimit = 64;

struct DirectX12DepthStencilSnapshot {
    uint32_t depthEnable{};
    uint32_t depthWriteMask{};
    uint32_t depthFunction{};
    uint32_t stencilEnable{};
    uint32_t stencilReadMask{};
    uint32_t stencilWriteMask{};
    uint32_t frontStencilFunction{};
    uint32_t frontStencilPassOperation{};
    uint32_t backStencilFunction{};
    uint32_t backStencilPassOperation{};
};

struct DirectX12AdapterState {
    rhi::directx12::DeviceState *device{};
    UINT resourceIncrement{};
    UINT samplerIncrement{};
    std::atomic<uint64_t> nextSerial{};
    DirectX12DepthStencilSnapshot lastDepthStencilState{};
};

DirectX12AdapterState &directX12State(VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_DIRECTX12);
    assert(adapter.backend.state);
    return *static_cast<DirectX12AdapterState *>(adapter.backend.state);
}

const DirectX12AdapterState &directX12State(const VernonRuntimeRhiAdapter &adapter) {
    assert(adapter.rhiBackend == VERNON_RHI_BACKEND_DIRECTX12);
    assert(adapter.backend.state);
    return *static_cast<const DirectX12AdapterState *>(adapter.backend.state);
}

rhi::directx12::DeviceState &directX12Device(VernonRuntimeRhiAdapter &adapter) {
    return *directX12State(adapter).device;
}

const rhi::directx12::DeviceState &directX12Device(const VernonRuntimeRhiAdapter &adapter) {
    return *directX12State(adapter).device;
}

void destroyBackend(void *state) noexcept { delete static_cast<DirectX12AdapterState *>(state); }
RhiAdapterResult<void> synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        if (static_cast<DirectX12AdapterState *>(state)->device->synchronize(error))
            return RhiAdapterResult<void>{vernon::ok()};
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "directx12_synchronize");
    } catch (...) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "directx12_synchronize");
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const DirectX12AdapterState *>(state)->device);
}
const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

template <typename Object> void releaseObject(Object *&object) {
    if (object)
        object->Release();
    object = nullptr;
}

struct PreparedShader {
    uint32_t stage{};
    std::vector<uint8_t> artifact;
};

// Retained by the groups and pipeline layouts made from it, since commands that used those may still be recording.
struct GroupLayout : RhiRetainedObject {
    struct Entry {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        // The entry's descriptor offset in its table; an entry repeating an earlier entry's register and range type
        // is the same value read by another stage, so it shares that descriptor and does not write it.
        std::optional<D3D12_DESCRIPTOR_RANGE_TYPE> range;
        uint32_t descriptor{kNone};
        bool writesDescriptor{};
        // The offset of a constant-buffer or raw-SRV inline value in the group's inline block.
        uint32_t inlineOffset{kNone};
        // Graphics inline values travel as root constants, which the pipeline layout places.
        uint32_t rootOffset{kNone};
    };
    std::vector<Entry> entries;
    // Register spaces are zero: the pipeline layout gives each group index its own space.
    std::vector<D3D12_DESCRIPTOR_RANGE> resourceRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE> samplerRanges;
    uint32_t inlineSize{};
    uint32_t rootDataSize{};
};

struct RootConstantMember {
    uint32_t group{};
    uint32_t dataOffset{};
    uint32_t offset{};
    uint32_t size{};
};

// One stage's root constants, spanning every group whose inline values that stage reads.
struct RootConstantBlock {
    uint32_t parameter{};
    uint32_t valueCount{};
    uint32_t groupMask{};
    std::vector<RootConstantMember> members;
};

struct PreparedLayout : RhiRetainedObject {
    std::array<GroupLayout *, kMaxGroups> groups{};
    uint32_t groupCount{};
    // Bit g is set when the group layout at index g has entries; an index without entries is never bound.
    uint32_t groupMask{};
    // Root parameter index of each group index's tables, or kNone.
    std::array<uint32_t, kMaxGroups> resourceTables{};
    std::array<uint32_t, kMaxGroups> samplerTables{};
    std::vector<RootConstantBlock> rootBlocks;
    uint32_t rootGroupMask{};
    std::vector<VernonRuntimeProviderVertexBinding> vertexBindings;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
    ID3D12RootSignature *rootSignature{};

    ~PreparedLayout() override {
        releaseObject(rootSignature);
        for (uint32_t group = 0; group < groupCount; ++group)
            releaseRetainedObject(groups[group]);
    }
};

struct PreparedPipeline : RhiRetainedObject {
    VernonRuntimeRhiAdapter *adapter{};
    PreparedLayout *layout{};
    ID3D12PipelineState *pipeline{};
    // By vertex binding, from the graphics variant; Core prepares variants with a stride for every binding.
    std::vector<uint32_t> vertexStrides;
    bool graphics{};

    ~PreparedPipeline() override {
        adapter->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
        releaseObject(pipeline);
        releaseRetainedObject(layout);
    }
};

// Shader-visible descriptor tables written for one command encoder; the group reuses them while that encoder records.
struct DescriptorTables {
    uint64_t encoder{};
    ID3D12DescriptorHeap *resourceHeap{};
    ID3D12DescriptorHeap *samplerHeap{};
    D3D12_GPU_DESCRIPTOR_HANDLE resourceGpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE samplerGpu{};
};

struct DirectX12Group : RhiGroup {
    struct Value {
        // A Buffer, Image or Sampler; null for the default sampler and for values without a resource.
        void *native{};
        uint64_t offset{};
        uint64_t size{};
    };
    struct BufferUse {
        rhi::directx12::Buffer *buffer{};
        D3D12_RESOURCE_STATES state{};
    };
    struct ImageUse {
        rhi::directx12::Image *image{};
        D3D12_RESOURCE_STATES state{};
    };
    rhi::directx12::DeviceState *device{};
    GroupLayout *layout{};
    uint64_t serial{};
    std::vector<Value> values;
    // Views read the inline values at `inlineOffset` in `inlineResource`: upload memory that stays in GENERIC_READ (a
    // persistent block of the device, or the transient memory of a transient group's encoder), or the updatable block
    // of an updatable group, whose writes leave it readable by shaders.
    ID3D12Resource *inlineResource{};
    uint64_t inlineOffset{};
    vernon::rhi::PersistentBlock updatable{};
    std::vector<uint8_t> updatableBytes;
    std::vector<uint8_t> rootData;
    std::vector<BufferUse> buffers;
    std::vector<ImageUse> images;
    std::mutex mutex;
    DescriptorTables tables;

    ~DirectX12Group() override { releaseRetainedObject(layout); }
};

// Setting a root signature resets every root argument; the serials record which group's arguments are set.
struct BindPointState {
    const PreparedLayout *layout{};
    std::array<uint64_t, kMaxGroups> tables{};
    std::array<uint64_t, kMaxGroups> constants{};
};

// Applied state holds group serials rather than pointers, because addresses are reused.
struct DirectX12Commands : RhiCommands {
    ID3D12GraphicsCommandList *list{};
    PreparedPipeline *pipeline{};
    std::array<DirectX12Group *, kMaxGroups> groups{};
    std::array<DescriptorTables, kMaxGroups> tables{};
    BindPointState graphicsState;
    BindPointState computeState;
    // The heaps the bound groups' tables share, and the heaps set on the command list; null when unknown.
    ID3D12DescriptorHeap *resourceHeap{};
    ID3D12DescriptorHeap *samplerHeap{};
    ID3D12DescriptorHeap *setResourceHeap{};
    ID3D12DescriptorHeap *setSamplerHeap{};
    bool scopeOpened{};
    // Advances whenever resource states may have changed, so each group index's transitions are reapplied.
    uint64_t epoch{1};
    std::array<uint64_t, kMaxGroups> transitionedGroups{};
    std::array<uint64_t, kMaxGroups> transitionedEpochs{};
    struct VertexBuffer {
        rhi::directx12::Buffer *buffer{};
        uint64_t offset{};
        uint64_t size{};
    };
    std::array<VertexBuffer, kMaxVertexBindings> vertexBuffers{};
    // The epoch in which each binding's buffer last became a vertex buffer.
    std::array<uint64_t, kMaxVertexBindings> vertexEpochs{};
    // Bit b is set when the list holds the view of binding b's buffer, with stride vertexViewStrides[b].
    uint32_t vertexViews{};
    std::array<uint32_t, kMaxVertexBindings> vertexViewStrides{};
    bool viewportValid{};
    bool scissorValid{};
    bool stencilValid{};
    std::array<uint32_t, 4> viewport{};
    std::array<uint32_t, 4> scissor{};
    uint32_t stencilReference{};
    D3D_PRIMITIVE_TOPOLOGY topology{D3D_PRIMITIVE_TOPOLOGY_UNDEFINED};
    uint64_t indexBufferOffset{UINT64_MAX};
    uint64_t indexBufferEpoch{};
    rhi::directx12::Buffer *indexBuffer{};
};

void restoreBufferState(void *context, uint64_t state) {
    static_cast<rhi::directx12::Buffer *>(context)->state = static_cast<D3D12_RESOURCE_STATES>(state);
}

bool computeEntry(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
}

bool supportedEntry(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    constexpr uint32_t graphicsStages = VERNON_RUNTIME_PROVIDER_STAGE_VERTEX | VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
    // Descriptor tables are visible to every stage, so a descriptor may serve both shader stages; inline values
    // become per-stage root constants and must name exactly one stage.
    const bool graphicsDescriptorStages = entry.stage_mask != 0 && (entry.stage_mask & ~graphicsStages) == 0;
    const bool compute =
        ((entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
          entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && entry.binding != UINT32_MAX) ||
         entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER || entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
         entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
         ((entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) &&
          entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE)) &&
        computeEntry(entry) && entry.element_size != 0;
    const bool graphicsResource =
        (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER || entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
         entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) &&
        entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE && graphicsDescriptorStages &&
        entry.binding != UINT32_MAX;
    const bool graphicsInline = entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                                entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                (entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                 entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                entry.element_size != 0 && entry.element_size % sizeof(uint32_t) == 0 &&
                                entry.element_alignment >= sizeof(uint32_t) &&
                                (entry.element_alignment & (entry.element_alignment - 1)) == 0;
    const bool graphicsUniformBuffer = (entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                        entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) &&
                                       entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                       graphicsDescriptorStages && entry.binding != UINT32_MAX &&
                                       entry.element_size != 0;
    return (compute || graphicsResource || graphicsInline || graphicsUniformBuffer) && entry.array_count == 1;
}

// The compiler declares read storage buffers NonWritable, which HLSL makes t-register ByteAddressBuffers, so they are
// raw SRVs and only written storage buffers are raw UAVs. Graphics inline values become root constants, while compute
// inline values are read storage buffers.
std::optional<D3D12_DESCRIPTOR_RANGE_TYPE> descriptorRangeType(const VernonRuntimeProviderBindingLayoutEntry &entry) {
    switch (entry.kind) {
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
        return writesBinding(entry) ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        if (computeEntry(entry))
            return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

const D3D12_SAMPLER_DESC &samplerDescriptor(const DirectX12Group::Value &value) {
    static constexpr D3D12_SAMPLER_DESC defaultSampler{D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                                                       D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                       D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                       D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                       0,
                                                       1,
                                                       D3D12_COMPARISON_FUNC_ALWAYS,
                                                       {},
                                                       0,
                                                       D3D12_FLOAT32_MAX};
    const auto *sampler = static_cast<const rhi::directx12::Sampler *>(value.native);
    return sampler ? sampler->descriptor : defaultSampler;
}

// TEXTURE2D and TEXTURECUBE views cannot start past layer 0, so such selections use the one-element array forms,
// which shaders declaring the non-array resource type read identically.
D3D12_SHADER_RESOURCE_VIEW_DESC sampledImageView(const rhi::directx12::Image &image) {
    const rhi::directx12::ImageSubresourceRange range = rhi::directx12::imageSubresourceRange(image);
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = image.format == DXGI_FORMAT_D32_FLOAT ? DXGI_FORMAT_R32_FLOAT : image.format;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    const bool firstLayerSelected = range.firstLayer != 0;
    switch (image.dimension) {
    case VERNON_RHI_IMAGE_3D:
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        view.Texture3D = {range.firstMip, range.mipCount, 0.0f};
        return view;
    case VERNON_RHI_IMAGE_CUBE:
    case VERNON_RHI_IMAGE_CUBE_ARRAY:
        if (image.dimension == VERNON_RHI_IMAGE_CUBE && !firstLayerSelected) {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            view.TextureCube = {range.firstMip, range.mipCount, 0.0f};
            return view;
        }
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        view.TextureCubeArray = {range.firstMip, range.mipCount, range.firstLayer,
                                 range.layerCount / vernon::rhi::cubeFaceCount, 0.0f};
        return view;
    case VERNON_RHI_IMAGE_2D:
        if (!firstLayerSelected) {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Texture2D = {range.firstMip, range.mipCount, 0, 0.0f};
            return view;
        }
        [[fallthrough]];
    case VERNON_RHI_IMAGE_2D_ARRAY:
        break;
    }
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    view.Texture2DArray = {range.firstMip, range.mipCount, range.firstLayer, range.layerCount, 0, 0.0f};
    return view;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC storageImageView(const rhi::directx12::Image &image) {
    const rhi::directx12::ImageSubresourceRange range = rhi::directx12::imageSubresourceRange(image);
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = image.format;
    if (image.dimension == VERNON_RHI_IMAGE_3D) {
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        view.Texture3D = {range.firstMip, 0, UINT_MAX};
    } else if (image.dimension == VERNON_RHI_IMAGE_2D && range.firstLayer == 0) {
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        view.Texture2D = {range.firstMip, 0};
    } else {
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray = {range.firstMip, range.firstLayer, range.layerCount, 0};
    }
    return view;
}

UINT rawBufferWords(uint64_t size) { return static_cast<UINT>(std::max<uint64_t>(1, (size + 3) / 4)); }

D3D12_UNORDERED_ACCESS_VIEW_DESC rawBufferUav(uint64_t offset, uint64_t size) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.Buffer.FirstElement = offset / 4;
    view.Buffer.NumElements = rawBufferWords(size);
    view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    return view;
}

D3D12_SHADER_RESOURCE_VIEW_DESC rawBufferSrv(uint64_t offset, uint64_t size) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.FirstElement = offset / 4;
    view.Buffer.NumElements = rawBufferWords(size);
    view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    return view;
}

RhiAdapterResult<void> transitionBuffer(VernonRuntimeRhiAdapter &adapter, const DirectX12Commands &commands,
                                        rhi::directx12::Buffer &buffer, D3D12_RESOURCE_STATES target) {
    if (buffer.state == target)
        return RhiAdapterResult<void>{vernon::ok()};
    if (auto deferred = deferCommandRollback(adapter, commands.encoder, &buffer, static_cast<uint64_t>(buffer.state),
                                             restoreBufferState);
        !deferred)
        return deferred;
    rhi::directx12::transition(commands.list, buffer.resource, buffer.state, target);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> transitionImage(VernonRuntimeRhiAdapter &adapter, const DirectX12Commands &commands,
                                       rhi::directx12::Image &image, D3D12_RESOURCE_STATES target) {
    auto transitioned =
        rhi::directx12::transitionCommandImage(adapter.rhiDevice, commands.encoder.value, commands.list, image, target);
    if (transitioned.isErr())
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(transitioned).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

D3D12_BLEND blendFactor(uint32_t value) {
    constexpr D3D12_BLEND values[]{D3D12_BLEND_ZERO,          D3D12_BLEND_ONE,           D3D12_BLEND_SRC_COLOR,
                                   D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_DEST_COLOR,    D3D12_BLEND_INV_DEST_COLOR,
                                   D3D12_BLEND_SRC_ALPHA,     D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA,
                                   D3D12_BLEND_INV_DEST_ALPHA};
    return value < std::size(values) ? values[value] : D3D12_BLEND_ZERO;
}

D3D12_BLEND_OP blendOperation(uint32_t value) {
    constexpr D3D12_BLEND_OP values[]{D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT, D3D12_BLEND_OP_REV_SUBTRACT,
                                      D3D12_BLEND_OP_MIN, D3D12_BLEND_OP_MAX};
    return value < std::size(values) ? values[value] : D3D12_BLEND_OP_ADD;
}

D3D12_COMPARISON_FUNC compareOperation(uint32_t value) {
    constexpr D3D12_COMPARISON_FUNC values[]{D3D12_COMPARISON_FUNC_NEVER,         D3D12_COMPARISON_FUNC_LESS,
                                             D3D12_COMPARISON_FUNC_EQUAL,         D3D12_COMPARISON_FUNC_LESS_EQUAL,
                                             D3D12_COMPARISON_FUNC_GREATER,       D3D12_COMPARISON_FUNC_NOT_EQUAL,
                                             D3D12_COMPARISON_FUNC_GREATER_EQUAL, D3D12_COMPARISON_FUNC_ALWAYS};
    return value < std::size(values) ? values[value] : D3D12_COMPARISON_FUNC_NEVER;
}

D3D12_STENCIL_OP stencilOperation(uint32_t value) {
    constexpr D3D12_STENCIL_OP values[]{D3D12_STENCIL_OP_KEEP,     D3D12_STENCIL_OP_ZERO,     D3D12_STENCIL_OP_REPLACE,
                                        D3D12_STENCIL_OP_INCR_SAT, D3D12_STENCIL_OP_DECR_SAT, D3D12_STENCIL_OP_INVERT,
                                        D3D12_STENCIL_OP_INCR,     D3D12_STENCIL_OP_DECR};
    return value < std::size(values) ? values[value] : D3D12_STENCIL_OP_KEEP;
}

D3D12_CULL_MODE cullMode(uint32_t value) {
    constexpr D3D12_CULL_MODE values[]{D3D12_CULL_MODE_NONE, D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_BACK};
    return value < std::size(values) ? values[value] : D3D12_CULL_MODE_NONE;
}

DXGI_FORMAT vertexFormat(uint32_t dtype, uint32_t components) {
    static constexpr DXGI_FORMAT formats[][4] = {
        {DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32A32_SINT},
        {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32A32_UINT},
        {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R16G16B16A16_FLOAT},
        {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT},
    };
    const int row = dtype == VERNON_RUNTIME_PROVIDER_I32   ? 0
                    : dtype == VERNON_RUNTIME_PROVIDER_U32 ? 1
                    : dtype == VERNON_RUNTIME_PROVIDER_F16 ? 2
                    : dtype == VERNON_RUNTIME_PROVIDER_F32 ? 3
                                                           : -1;
    return row < 0 || components == 0 || components > 4 ? DXGI_FORMAT_UNKNOWN : formats[row][components - 1];
}

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    const auto &device = directX12Device(adapter);
    return {0x4433443132u, reinterpret_cast<uintptr_t>(device.device), static_cast<uint64_t>(device.shaderModel)};
}

RhiAdapterResult<void> prepareShaderResult(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size < 4 || descriptor->stage == 0 || std::memcmp(descriptor->data, "DXBC", 4) != 0 ||
        descriptor->format.size != 4 || std::memcmp(descriptor->format.data, "dxil", 4) != 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_adapter_received_an_invalid_dxil_shader_descriptor");
    try {
        auto shader = std::make_unique<PreparedShader>();
        shader->stage = descriptor->stage;
        const auto *bytes = static_cast<const uint8_t *>(descriptor->data);
        shader->artifact.assign(bytes, bytes + descriptor->size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "d3d12_shader_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> prepareGroupLayoutImplResult(const VernonRuntimeProviderGroupLayoutDescriptor &descriptor,
                                                    VernonRuntimeProviderObject *output) {
    auto layout = std::make_unique<GroupLayout>();
    layout->entries.reserve(descriptor.entry_count);
    for (size_t index = 0; index < descriptor.entry_count; ++index) {
        const auto &source = descriptor.entries[index];
        if (!supportedEntry(source))
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "d3d12_group_layout_contains_an_unsupported_binding",
                                   static_cast<uint32_t>(source.kind), source.array_count);
        GroupLayout::Entry entry;
        entry.layout = source;
        entry.range = descriptorRangeType(source);
        if (!entry.range) {
            entry.rootOffset = layout->rootDataSize;
            layout->rootDataSize += source.element_size;
        } else {
            auto &ranges =
                *entry.range == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER ? layout->samplerRanges : layout->resourceRanges;
            const auto existing = std::find_if(ranges.begin(), ranges.end(), [&](const D3D12_DESCRIPTOR_RANGE &range) {
                return range.RangeType == *entry.range && range.BaseShaderRegister == source.binding;
            });
            entry.writesDescriptor = existing == ranges.end();
            entry.descriptor = entry.writesDescriptor ? static_cast<uint32_t>(ranges.size())
                                                      : existing->OffsetInDescriptorsFromTableStart;
            if (entry.writesDescriptor) {
                ranges.push_back({*entry.range, 1, source.binding, 0, entry.descriptor});
                if (packedUniformBytes(source.kind, source.interface_kind)) {
                    // Constant buffer views start at, and span, multiples of 256 bytes.
                    entry.inlineOffset = layout->inlineSize;
                    layout->inlineSize = (entry.inlineOffset + source.element_size + 255u) & ~255u;
                }
            }
        }
        layout->entries.push_back(entry);
    }
    *output = toHandle(layout.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareGroupLayoutResult(void *, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                                VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->entry_count != 0 && !descriptor->entries))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_adapter_received_an_invalid_group_layout");
    try {
        return prepareGroupLayoutImplResult(*descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "d3d12_group_layout_preparation_ran_out_of_memory");
    }
}

// Places each stage's graphics inline values of every group in one root-constant block: shader block members follow
// entry argument order, at offsets aligned to each value's alignment.
RhiAdapterResult<void> placeRootConstants(PreparedLayout &layout, std::vector<D3D12_ROOT_PARAMETER> &parameters) {
    struct Candidate {
        uint32_t group{};
        const GroupLayout::Entry *entry{};
    };
    constexpr uint32_t shaderStages[]{VERNON_RUNTIME_PROVIDER_STAGE_VERTEX, VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT};
    for (uint32_t stage : shaderStages) {
        std::vector<Candidate> candidates;
        for (uint32_t group = 0; group < layout.groupCount; ++group)
            for (const GroupLayout::Entry &entry : layout.groups[group]->entries)
                if (entry.rootOffset != kNone && entry.layout.stage_mask == stage)
                    candidates.push_back({group, &entry});
        if (candidates.empty())
            continue;
        std::sort(candidates.begin(), candidates.end(), [](const Candidate &left, const Candidate &right) {
            return left.entry->layout.argument_index < right.entry->layout.argument_index;
        });
        constexpr uint32_t rootBytes = kRootDwordLimit * sizeof(uint32_t);
        RootConstantBlock block;
        block.parameter = static_cast<uint32_t>(parameters.size());
        uint32_t size = 0;
        for (const Candidate &candidate : candidates) {
            const auto &entry = candidate.entry->layout;
            const uint32_t offset = entry.element_alignment > rootBytes
                                        ? UINT32_MAX
                                        : (size + entry.element_alignment - 1) & ~(entry.element_alignment - 1);
            if (offset > rootBytes || entry.element_size > rootBytes - offset)
                return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                       "d3d12_graphics_root_signature_exceeds_the_64_dword_limit");
            block.members.push_back({candidate.group, candidate.entry->rootOffset, offset, entry.element_size});
            block.groupMask |= 1u << candidate.group;
            size = offset + entry.element_size;
        }
        block.valueCount = size / sizeof(uint32_t);
        auto &parameter = parameters.emplace_back();
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameter.Constants.ShaderRegister = ::vernon::runtime::kDirectXRootConstantRegister;
        parameter.Constants.RegisterSpace = ::vernon::runtime::kDirectXRootConstantRegisterSpace;
        parameter.Constants.Num32BitValues = block.valueCount;
        parameter.ShaderVisibility = stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ? D3D12_SHADER_VISIBILITY_VERTEX
                                                                                   : D3D12_SHADER_VISIBILITY_PIXEL;
        layout.rootGroupMask |= block.groupMask;
        layout.rootBlocks.push_back(std::move(block));
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutImplResult(VernonRuntimeRhiAdapter &adapter,
                                               const VernonRuntimeProviderPipelineLayoutDescriptor &descriptor,
                                               VernonRuntimeProviderObject *output) {
    auto layout = std::make_unique<PreparedLayout>();
    layout->resourceTables.fill(kNone);
    layout->samplerTables.fill(kNone);
    size_t rangeCount = 0;
    for (size_t group = 0; group < descriptor.group_layout_count; ++group) {
        auto *groupLayout = fromHandle<GroupLayout>(descriptor.group_layouts[group]);
        if (!groupLayout)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "d3d12_pipeline_layout_group_layout_is_invalid", group);
        groupLayout->references.fetch_add(1, std::memory_order_relaxed);
        layout->groups[group] = groupLayout;
        layout->groupCount = static_cast<uint32_t>(group + 1);
        rangeCount += groupLayout->resourceRanges.size() + groupLayout->samplerRanges.size();
        if (!groupLayout->entries.empty())
            layout->groupMask |= 1u << group;
    }
    // Root parameters point into `ranges`, which therefore never reallocates.
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    ranges.reserve(rangeCount);
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    uint32_t rootDwords = 0;
    for (uint32_t group = 0; group < layout->groupCount; ++group) {
        const GroupLayout &groupLayout = *layout->groups[group];
        for (const auto *tableRanges : {&groupLayout.resourceRanges, &groupLayout.samplerRanges}) {
            if (tableRanges->empty())
                continue;
            const size_t first = ranges.size();
            for (D3D12_DESCRIPTOR_RANGE range : *tableRanges) {
                range.RegisterSpace = group + ::vernon::runtime::kDirectXDescriptorSetRegisterSpaceOffset;
                ranges.push_back(range);
            }
            (tableRanges == &groupLayout.resourceRanges ? layout->resourceTables : layout->samplerTables)[group] =
                static_cast<uint32_t>(parameters.size());
            auto &parameter = parameters.emplace_back();
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.DescriptorTable = {static_cast<UINT>(tableRanges->size()), ranges.data() + first};
            parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            ++rootDwords;
        }
    }
    if (auto placed = placeRootConstants(*layout, parameters); !placed)
        return placed;
    for (const RootConstantBlock &block : layout->rootBlocks)
        rootDwords += block.valueCount;
    if (rootDwords > kRootDwordLimit)
        return providerFailure(vernon::ProviderErrorCode::Unsupported,
                               "d3d12_graphics_root_signature_exceeds_the_64_dword_limit", rootDwords);
    layout->vertexBindings.assign(descriptor.vertex_bindings,
                                  descriptor.vertex_bindings + descriptor.vertex_binding_count);
    for (size_t index = 0; index < descriptor.vertex_attribute_count; ++index) {
        const auto &attribute = descriptor.vertex_attributes[index];
        std::string capabilityDiagnostic;
        if (!validateVertexAttributeCapability(VertexAttributeBackend::DirectX12, attribute,
                                               D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, false, capabilityDiagnostic))
            return providerFailure(vernon::ProviderErrorCode::Unsupported, "d3d12_validate_vertex_attribute_capability",
                                   attribute.location, attribute.binding);
        layout->vertexAttributes.push_back(attribute);
    }
    const D3D12_ROOT_SIGNATURE_DESC root{
        static_cast<UINT>(parameters.size()), parameters.empty() ? nullptr : parameters.data(), 0, nullptr,
        layout->vertexAttributes.empty() ? D3D12_ROOT_SIGNATURE_FLAG_NONE
                                         : D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    HRESULT result = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    releaseObject(errors);
    if (FAILED(result)) {
        releaseObject(serialized);
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "D3D12SerializeRootSignature",
                               static_cast<uint64_t>(result));
    }
    result = directX12Device(adapter).device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&layout->rootSignature));
    releaseObject(serialized);
    if (FAILED(result))
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "ID3D12Device::CreateRootSignature",
                               static_cast<uint64_t>(result));
    *output = toHandle(layout.release());
    adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->group_layout_count > kMaxGroups ||
        (descriptor->group_layout_count != 0 && !descriptor->group_layouts) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_adapter_received_an_invalid_pipeline_layout");
    if (descriptor->push_constant_size != 0)
        return providerFailure(vernon::ProviderErrorCode::Unsupported, "d3d12_dispatch_push_constants_are_unsupported");
    try {
        return prepareLayoutImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "d3d12_layout_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> prepareGraphicsPipelineImplResult(VernonRuntimeRhiAdapter &adapter,
                                                         const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                                         PreparedPipeline &pipeline) {
    const PreparedLayout &layout = *pipeline.layout;
    if (descriptor.color_format_count > 8 || descriptor.sample_count != 1 ||
        (descriptor.depth_stencil_format && descriptor.depth_stencil_format != DXGI_FORMAT_D32_FLOAT &&
         descriptor.depth_stencil_format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_graphics_pipeline_uses_an_unsupported_attachment_configuration");
    PreparedShader *vertex = nullptr;
    PreparedShader *fragment = nullptr;
    for (size_t index = 0; index < descriptor.shader_count; ++index) {
        auto *shader = fromHandle<PreparedShader>(descriptor.shaders[index]);
        if (!shader)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "d3d12_graphics_pipeline_contains_an_invalid_shader");
        if (shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
            vertex = shader;
        else if (shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
            fragment = shader;
    }
    if (!vertex || !fragment)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_graphics_pipeline_requires_vertex_and_fragment_shaders");
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements;
    for (const VernonRuntimeProviderVertexAttribute &attribute : layout.vertexAttributes) {
        const auto binding = std::find_if(layout.vertexBindings.begin(), layout.vertexBindings.end(),
                                          [&](const VernonRuntimeProviderVertexBinding &candidate) {
                                              return candidate.binding == attribute.binding;
                                          });
        const DXGI_FORMAT format = vertexFormat(attribute.dtype, attribute.component_count);
        if (binding == layout.vertexBindings.end() || format == DXGI_FORMAT_UNKNOWN)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "d3d12_vertex_attribute_format_is_unsupported");
        inputElements.push_back({"TEXCOORD", attribute.location, format, attribute.binding, attribute.relative_offset,
                                 binding->divisor ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                                  : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                                 binding->divisor});
    }
    pipeline.vertexStrides.assign(descriptor.vertex_strides,
                                  descriptor.vertex_strides + descriptor.vertex_stride_count);
    pipeline.vertexStrides.resize(kMaxVertexBindings);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC native{};
    native.pRootSignature = layout.rootSignature;
    native.VS = {vertex->artifact.data(), vertex->artifact.size()};
    native.PS = {fragment->artifact.data(), fragment->artifact.size()};
    if (descriptor.color_blend_count != descriptor.color_format_count ||
        (descriptor.color_blend_count && !descriptor.color_blends))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_graphics_pipeline_blend_state_does_not_match_its_attachments");
    native.BlendState.AlphaToCoverageEnable = FALSE;
    native.BlendState.IndependentBlendEnable = descriptor.color_format_count > 1;
    for (size_t index = 0; index < descriptor.color_format_count; ++index) {
        const auto &source = descriptor.color_blends[index];
        if (source.blend_enabled > 1 || source.source_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            source.destination_color_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            source.source_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            source.destination_alpha_factor > VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA ||
            source.color_operation > VERNON_RHI_BLEND_MAXIMUM || source.alpha_operation > VERNON_RHI_BLEND_MAXIMUM ||
            (source.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "d3d12_graphics_pipeline_contains_an_invalid_blend_state");
        auto &target = native.BlendState.RenderTarget[index];
        target.BlendEnable = source.blend_enabled;
        target.SrcBlend = blendFactor(source.source_color_factor);
        target.DestBlend = blendFactor(source.destination_color_factor);
        target.BlendOp = blendOperation(source.color_operation);
        target.SrcBlendAlpha = blendFactor(source.source_alpha_factor);
        target.DestBlendAlpha = blendFactor(source.destination_alpha_factor);
        target.BlendOpAlpha = blendOperation(source.alpha_operation);
        target.RenderTargetWriteMask = static_cast<UINT8>(source.write_mask);
    }
    native.SampleMask = UINT_MAX;
    const auto &rasterization = descriptor.rasterization;
    if (rasterization.cull_mode > VERNON_RHI_CULL_BACK || rasterization.front_face > VERNON_RHI_FRONT_FACE_CLOCKWISE ||
        rasterization.depth_clamp || rasterization.depth_bias_enabled > 1 ||
        !std::isfinite(rasterization.depth_bias_constant) || !std::isfinite(rasterization.depth_bias_slope) ||
        static_cast<double>(rasterization.depth_bias_constant) < (std::numeric_limits<INT>::min)() ||
        static_cast<double>(rasterization.depth_bias_constant) > (std::numeric_limits<INT>::max)())
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_graphics_pipeline_contains_an_unsupported_rasterization_state");
    native.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    native.RasterizerState.CullMode = cullMode(rasterization.cull_mode);
    native.RasterizerState.FrontCounterClockwise = rasterization.front_face == VERNON_RHI_FRONT_FACE_COUNTER_CLOCKWISE;
    native.RasterizerState.DepthBias =
        rasterization.depth_bias_enabled ? static_cast<INT>(rasterization.depth_bias_constant) : 0;
    native.RasterizerState.SlopeScaledDepthBias =
        rasterization.depth_bias_enabled ? rasterization.depth_bias_slope : 0.0f;
    native.RasterizerState.DepthClipEnable = TRUE;
    const auto &depthStencil = descriptor.depth_stencil;
    const bool hasDepth = descriptor.depth_stencil_format != DXGI_FORMAT_UNKNOWN;
    const bool hasStencil = descriptor.depth_stencil_format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    if (depthStencil.depth_test > 1 || depthStencil.depth_write > 1 || depthStencil.stencil_test > 1 ||
        depthStencil.depth_compare > VERNON_RHI_COMPARE_ALWAYS ||
        (!hasDepth && (depthStencil.depth_test || depthStencil.depth_write)) ||
        (!hasStencil && depthStencil.stencil_test) ||
        depthStencil.front.stencil_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
        depthStencil.front.depth_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
        depthStencil.front.pass > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
        depthStencil.front.compare > VERNON_RHI_COMPARE_ALWAYS ||
        depthStencil.back.stencil_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
        depthStencil.back.depth_fail > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
        depthStencil.back.pass > VERNON_RHI_STENCIL_DECREMENT_WRAP ||
        depthStencil.back.compare > VERNON_RHI_COMPARE_ALWAYS || depthStencil.stencil_read_mask > UINT8_MAX ||
        depthStencil.stencil_write_mask > UINT8_MAX)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_graphics_pipeline_contains_an_invalid_depth_stencil_state");
    native.DepthStencilState.DepthEnable = depthStencil.depth_test;
    native.DepthStencilState.DepthWriteMask =
        depthStencil.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    native.DepthStencilState.DepthFunc =
        compareOperation(depthStencil.depth_test ? depthStencil.depth_compare : VERNON_RHI_COMPARE_ALWAYS);
    native.DepthStencilState.StencilEnable = depthStencil.stencil_test;
    native.DepthStencilState.StencilReadMask = static_cast<UINT8>(depthStencil.stencil_read_mask);
    native.DepthStencilState.StencilWriteMask = static_cast<UINT8>(depthStencil.stencil_write_mask);
    const auto stencilFace = [](const VernonStencilFaceState &source) {
        return D3D12_DEPTH_STENCILOP_DESC{stencilOperation(source.stencil_fail), stencilOperation(source.depth_fail),
                                          stencilOperation(source.pass), compareOperation(source.compare)};
    };
    native.DepthStencilState.FrontFace = stencilFace(depthStencil.front);
    native.DepthStencilState.BackFace = stencilFace(depthStencil.back);
    native.InputLayout = {inputElements.data(), static_cast<UINT>(inputElements.size())};
    native.PrimitiveTopologyType = descriptor.topology == 1   ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
                                   : descriptor.topology == 2 ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
                                                              : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    native.NumRenderTargets = static_cast<UINT>(descriptor.color_format_count);
    for (size_t index = 0; index < descriptor.color_format_count; ++index)
        native.RTVFormats[index] = static_cast<DXGI_FORMAT>(descriptor.color_formats[index]);
    native.DSVFormat = static_cast<DXGI_FORMAT>(descriptor.depth_stencil_format);
    native.SampleDesc.Count = std::max(1u, descriptor.sample_count);
    const HRESULT result =
        directX12Device(adapter).device->CreateGraphicsPipelineState(&native, IID_PPV_ARGS(&pipeline.pipeline));
    if (FAILED(result))
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "ID3D12Device::CreateGraphicsPipelineState",
                               static_cast<uint64_t>(result));
    directX12State(adapter).lastDepthStencilState = {
        static_cast<uint32_t>(native.DepthStencilState.DepthEnable),
        static_cast<uint32_t>(native.DepthStencilState.DepthWriteMask),
        static_cast<uint32_t>(native.DepthStencilState.DepthFunc),
        static_cast<uint32_t>(native.DepthStencilState.StencilEnable),
        static_cast<uint32_t>(native.DepthStencilState.StencilReadMask),
        static_cast<uint32_t>(native.DepthStencilState.StencilWriteMask),
        static_cast<uint32_t>(native.DepthStencilState.FrontFace.StencilFunc),
        static_cast<uint32_t>(native.DepthStencilState.FrontFace.StencilPassOp),
        static_cast<uint32_t>(native.DepthStencilState.BackFace.StencilFunc),
        static_cast<uint32_t>(native.DepthStencilState.BackFace.StencilPassOp),
    };
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareComputePipelineImplResult(VernonRuntimeRhiAdapter &adapter,
                                                        const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                                        PreparedPipeline &pipeline) {
    auto *shader = descriptor.shader_count == 1 ? fromHandle<PreparedShader>(descriptor.shaders[0]) : nullptr;
    if (!shader || shader->stage != VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_compute_pipeline_requires_one_compute_shader");
    D3D12_COMPUTE_PIPELINE_STATE_DESC native{};
    native.pRootSignature = pipeline.layout->rootSignature;
    native.CS = {shader->artifact.data(), shader->artifact.size()};
    const HRESULT result =
        directX12Device(adapter).device->CreateComputePipelineState(&native, IID_PPV_ARGS(&pipeline.pipeline));
    if (FAILED(result))
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "ID3D12Device::CreateComputePipelineState",
                               static_cast<uint64_t>(result));
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout || !descriptor->shaders ||
        descriptor->shader_count == 0)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_adapter_received_an_invalid_pipeline_descriptor");
    try {
        auto pipeline = std::make_unique<PreparedPipeline>();
        pipeline->adapter = &adapter;
        adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
        layout->references.fetch_add(1, std::memory_order_relaxed);
        pipeline->layout = layout;
        pipeline->graphics = descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
        auto prepared = pipeline->graphics ? prepareGraphicsPipelineImplResult(adapter, *descriptor, *pipeline)
                                           : prepareComputePipelineImplResult(adapter, *descriptor, *pipeline);
        if (!prepared)
            return prepared;
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure,
                               "d3d12_pipeline_preparation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> validateGroupValue(const GroupLayout::Entry &entry,
                                          const VernonRuntimeProviderBindingValue &value) {
    const auto &layout = entry.layout;
    if (value.kind != layout.kind)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_group_value_kind_is_invalid",
                               layout.slot);
    if (packedUniformBytes(layout.kind, layout.interface_kind)) {
        if (!value.payload.inline_value.data || value.payload.inline_value.size != layout.element_size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "d3d12_inline_or_uniform_buffer_binding_size_is_invalid", layout.slot);
        return RhiAdapterResult<void>{vernon::ok()};
    }
    if ((value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
        return layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER
                   ? RhiAdapterResult<void>{vernon::ok()}
                   : providerFailure(vernon::ProviderErrorCode::Unsupported,
                                     "d3d12_default_resource_is_supported_only_for_samplers", layout.slot);
    const auto *resource = providerBindingResource(value);
    const bool image =
        layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE || layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
    const bool sampler = layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER;
    const uint64_t expectedKind = sampler ? kRhiSamplerResource : kRhiBufferResource;
    if (!resource || !resource->resource.value ||
        (image ? !isRhiImageReference(*resource) : (resource->identity & kRhiResourceKindMask) != expectedKind) ||
        (!image && !sampler && resource->offset >= resource->size))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_resource_binding_is_invalid",
                               layout.slot);
    if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
        layout.element_size > resource->size - resource->offset)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_storage_binding_is_too_small",
                               layout.slot);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupImplResult(VernonRuntimeRhiAdapter &adapter,
                                             const VernonRuntimeProviderGroupDescriptor &descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto *layout = fromHandle<GroupLayout>(descriptor.group_layout);
    if (!layout || descriptor.value_count != layout->entries.size() ||
        (descriptor.value_count != 0 && !descriptor.values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_adapter_received_an_invalid_group_descriptor");
    const VernonRuntimeProviderBindingValue *values = descriptor.values;
    for (size_t index = 0; index < layout->entries.size(); ++index)
        if (auto valid = validateGroupValue(layout->entries[index], values[index]); !valid)
            return valid;
    auto group = std::make_unique<DirectX12Group>();
    group->device = &directX12Device(adapter);
    layout->references.fetch_add(1, std::memory_order_relaxed);
    group->layout = layout;
    group->values.resize(layout->entries.size());
    group->rootData.resize(layout->rootDataSize);
    std::vector<uint8_t> inlineData(layout->inlineSize);
    for (size_t index = 0; index < layout->entries.size(); ++index) {
        const GroupLayout::Entry &entry = layout->entries[index];
        const VernonRuntimeProviderBindingValue &value = values[index];
        if (entry.rootOffset != kNone) {
            std::memcpy(group->rootData.data() + entry.rootOffset, value.payload.inline_value.data,
                        value.payload.inline_value.size);
            continue;
        }
        if (packedUniformBytes(entry.layout.kind, entry.layout.interface_kind)) {
            if (entry.inlineOffset != kNone)
                std::memcpy(inlineData.data() + entry.inlineOffset, value.payload.inline_value.data,
                            value.payload.inline_value.size);
            continue;
        }
        if ((value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0)
            continue;
        const VernonRuntimeProviderResourceReference &resource = *providerBindingResource(value);
        auto acquired = acquireGroupResource(adapter, *group, resource);
        if (!acquired)
            return RhiAdapterResult<void>{vernon::err(std::move(acquired).error())};
        void *native = reinterpret_cast<void *>(static_cast<uintptr_t>(acquired.value()));
        if (!native)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_resource_binding_is_stale",
                                   entry.layout.slot);
        group->values[index] = {native, resource.offset, resource.size};
        if (writesBinding(entry.layout))
            group->writes.push_back(resource);
        switch (entry.layout.kind) {
        case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
            if (entry.writesDescriptor)
                group->buffers.push_back(
                    {static_cast<rhi::directx12::Buffer *>(native), *entry.range == D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                                                                        ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                                        : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE});
            break;
        case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
            if (entry.writesDescriptor)
                group->images.push_back({static_cast<rhi::directx12::Image *>(native),
                                         entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE
                                             ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                             : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE});
            break;
        default:
            break;
        }
    }
    if (!inlineData.empty()) {
        if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT) {
            auto written = writeCommandTransient(adapter, descriptor.command_encoder, inlineData.data(),
                                                 inlineData.size(), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (!written)
                return RhiAdapterResult<void>{vernon::err(std::move(written).error())};
            group->inlineResource = reinterpret_cast<ID3D12Resource *>(static_cast<uintptr_t>(written.value().buffer));
            group->inlineOffset = written.value().offset;
        } else if (descriptor.lifetime == VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE) {
            auto placed =
                placeUpdatable(adapter, *group, inlineData.size(), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (!placed)
                return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
            group->updatable = placed.value();
            group->inlineResource = reinterpret_cast<ID3D12Resource *>(static_cast<uintptr_t>(placed.value().buffer));
            group->inlineOffset = placed.value().offset;
            group->updatableBytes = std::move(inlineData);
        } else {
            auto placed = placePersistent(adapter, *group, inlineData.data(), inlineData.size(),
                                          D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
            if (!placed)
                return RhiAdapterResult<void>{vernon::err(std::move(placed).error())};
            group->inlineResource = reinterpret_cast<ID3D12Resource *>(static_cast<uintptr_t>(placed.value().buffer));
            group->inlineOffset = placed.value().offset;
        }
    }
    group->serial = directX12State(adapter).nextSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    *output = toHandle(group.release());
    adapter.groupCreations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> createGroupResult(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                                         VernonRuntimeProviderObject *output) {
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_adapter_received_an_invalid_group_descriptor");
    try {
        return createGroupImplResult(*static_cast<VernonRuntimeRhiAdapter *>(data), *descriptor, output);
    } catch (const std::bad_alloc &) {
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "d3d12_group_creation_ran_out_of_memory");
    }
}

RhiAdapterResult<void> updateGroupResult(void *data, VernonRuntimeProviderObject commandsHandle,
                                         VernonRuntimeProviderObject groupHandle,
                                         const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<DirectX12Commands>(commandsHandle);
    auto *group = fromHandle<DirectX12Group>(groupHandle);
    if (!commands || !group || commands->rendering || valueCount != group->layout->entries.size() ||
        (valueCount && !values))
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_group_update_is_invalid");
    bool rooted = false;
    for (size_t index = 0; index < valueCount; ++index) {
        const GroupLayout::Entry &entry = group->layout->entries[index];
        if (!packedUniformBytes(entry.layout.kind, entry.layout.interface_kind))
            continue;
        const auto &bytes = values[index].payload.inline_value;
        if (!bytes.data || bytes.size != entry.layout.element_size)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_inline_binding_is_invalid",
                                   entry.layout.slot);
        if (entry.rootOffset != kNone) {
            std::memcpy(group->rootData.data() + entry.rootOffset, bytes.data, bytes.size);
            rooted = true;
        } else if (entry.inlineOffset != kNone) {
            std::memcpy(group->updatableBytes.data() + entry.inlineOffset, bytes.data, bytes.size);
        }
    }
    if (!group->updatableBytes.empty())
        if (auto written = writeUpdatable(adapter, commands->encoder, group->updatable, group->updatableBytes.data(),
                                          group->updatableBytes.size());
            !written)
            return written;
    // Commands set a group's root constants again only for a serial they have not set.
    if (rooted)
        group->serial = directX12State(adapter).nextSerial.fetch_add(1, std::memory_order_relaxed) + 1;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> beginCommandsResult(void *data, VernonRuntimeProviderObject encoder,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!output)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_begin_commands_output_is_null");
    std::unique_ptr<DirectX12Commands> commands(new (std::nothrow) DirectX12Commands());
    if (!commands)
        return providerFailure(vernon::ProviderErrorCode::ResourceExhausted, "d3d12_commands_allocation");
    if (auto begun = beginRhiCommands(adapter, encoder, *commands); !begun)
        return begun;
    commands->list = reinterpret_cast<ID3D12GraphicsCommandList *>(static_cast<uintptr_t>(commands->native));
    *output = toHandle(commands.release());
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endCommandsResult(void *, VernonRuntimeProviderObject handle) {
    std::unique_ptr<DirectX12Commands> commands(fromHandle<DirectX12Commands>(handle));
    return commands
               ? RhiAdapterResult<void>{vernon::ok()}
               : providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_end_commands_handle_is_invalid");
}

// The RHI's barriers and copies change resource states, and its mipmap generation and clears set descriptor heaps, a
// root signature and a pipeline state.
RhiAdapterResult<void> resumeCommandsResult(void *, VernonRuntimeProviderObject handle) {
    auto *commands = fromHandle<DirectX12Commands>(handle);
    if (!commands || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_resume_commands_is_invalid");
    commands->graphicsState = {};
    commands->computeState = {};
    commands->setResourceHeap = nullptr;
    commands->setSamplerHeap = nullptr;
    ++commands->epoch;
    if (commands->pipeline)
        commands->list->SetPipelineState(commands->pipeline->pipeline);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> writeGroupTables(VernonRuntimeRhiAdapter &adapter, const DirectX12Commands &commands,
                                        DirectX12Group &group) {
    const GroupLayout &layout = *group.layout;
    const DirectX12AdapterState &state = directX12State(adapter);
    ID3D12Device *device = group.device->device;
    DescriptorTables tables{commands.encoder.value};
    D3D12_CPU_DESCRIPTOR_HANDLE resourceCpu{};
    D3D12_CPU_DESCRIPTOR_HANDLE samplerCpu{};
    std::string backendDiagnostic;
    const auto resourceCount = static_cast<uint32_t>(layout.resourceRanges.size());
    const auto samplerCount = static_cast<uint32_t>(layout.samplerRanges.size());
    if (resourceCount &&
        !group.device->acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, resourceCount,
                                          tables.resourceHeap, resourceCpu, tables.resourceGpu, backendDiagnostic))
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "d3d12_acquire_resource_descriptors",
                               resourceCount);
    if (samplerCount &&
        !group.device->acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, samplerCount, tables.samplerHeap,
                                          samplerCpu, tables.samplerGpu, backendDiagnostic))
        return providerFailure(vernon::ProviderErrorCode::BackendFailure, "d3d12_acquire_sampler_descriptors",
                               samplerCount);
    for (size_t index = 0; index < layout.entries.size(); ++index) {
        const GroupLayout::Entry &entry = layout.entries[index];
        if (!entry.writesDescriptor)
            continue;
        const DirectX12Group::Value &value = group.values[index];
        if (*entry.range == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER) {
            device->CreateSampler(&samplerDescriptor(value),
                                  {samplerCpu.ptr + static_cast<SIZE_T>(entry.descriptor) * state.samplerIncrement});
            continue;
        }
        const D3D12_CPU_DESCRIPTOR_HANDLE target{resourceCpu.ptr +
                                                 static_cast<SIZE_T>(entry.descriptor) * state.resourceIncrement};
        if (entry.inlineOffset != kNone && entry.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            const D3D12_CONSTANT_BUFFER_VIEW_DESC view{group.inlineResource->GetGPUVirtualAddress() +
                                                           group.inlineOffset + entry.inlineOffset,
                                                       (entry.layout.element_size + 255u) & ~255u};
            device->CreateConstantBufferView(&view, target);
        } else if (entry.inlineOffset != kNone) {
            const D3D12_SHADER_RESOURCE_VIEW_DESC view =
                rawBufferSrv(group.inlineOffset + entry.inlineOffset, entry.layout.element_size);
            device->CreateShaderResourceView(group.inlineResource, &view, target);
        } else if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            ID3D12Resource *resource = static_cast<rhi::directx12::Buffer *>(value.native)->resource;
            if (*entry.range == D3D12_DESCRIPTOR_RANGE_TYPE_UAV) {
                const D3D12_UNORDERED_ACCESS_VIEW_DESC view = rawBufferUav(value.offset, value.size);
                device->CreateUnorderedAccessView(resource, nullptr, &view, target);
            } else {
                const D3D12_SHADER_RESOURCE_VIEW_DESC view = rawBufferSrv(value.offset, value.size);
                device->CreateShaderResourceView(resource, &view, target);
            }
        } else if (entry.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
            const auto &image = *static_cast<rhi::directx12::Image *>(value.native);
            const D3D12_UNORDERED_ACCESS_VIEW_DESC view = storageImageView(image);
            device->CreateUnorderedAccessView(image.resource, nullptr, &view, target);
        } else {
            const auto &image = *static_cast<rhi::directx12::Image *>(value.native);
            const D3D12_SHADER_RESOURCE_VIEW_DESC view = sampledImageView(image);
            device->CreateShaderResourceView(image.resource, &view, target);
        }
    }
    group.tables = tables;
    return RhiAdapterResult<void>{vernon::ok()};
}

// Makes `target` the heap the bound groups' tables share; returns whether that moved it from another heap.
bool adoptHeap(ID3D12DescriptorHeap *&target, ID3D12DescriptorHeap *heap) {
    if (!heap || target == heap)
        return false;
    const bool moved = target != nullptr;
    target = heap;
    return moved;
}

/* A command reads its groups' descriptors through one resource heap and one sampler heap, but a group keeps the tables
 * it wrote earlier in this encoder, and the descriptor ring moves to a new heap when one fills. A group whose tables
 * are missing or lie in another heap is rewritten into the ring's current heap; when that moves the shared heap, the
 * groups checked before it are checked again. The ring's new heap is larger than the old, so this settles. */
RhiAdapterResult<void> prepareGroups(VernonRuntimeRhiAdapter &adapter, DirectX12Commands &commands) {
    constexpr uint32_t kSettlePasses = 4;
    const PreparedLayout &layout = *commands.pipeline->layout;
    for (uint32_t index = 0; index < layout.groupCount; ++index) {
        if ((layout.groupMask & (1u << index)) == 0)
            continue;
        DirectX12Group *group = commands.groups[index];
        if (!group)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_command_group_is_unbound", index);
    }
    for (uint32_t pass = 0; pass < kSettlePasses; ++pass) {
        bool moved = false;
        for (uint32_t index = 0; index < layout.groupCount; ++index) {
            if ((layout.groupMask & (1u << index)) == 0)
                continue;
            DirectX12Group &group = *commands.groups[index];
            const bool resources = !group.layout->resourceRanges.empty();
            const bool samplers = !group.layout->samplerRanges.empty();
            if (!resources && !samplers)
                continue;
            std::lock_guard<std::mutex> guard(group.mutex);
            const DescriptorTables &tables = group.tables;
            const bool current =
                tables.encoder == commands.encoder.value &&
                (!resources || !commands.resourceHeap || tables.resourceHeap == commands.resourceHeap) &&
                (!samplers || !commands.samplerHeap || tables.samplerHeap == commands.samplerHeap);
            if (!current)
                if (auto written = writeGroupTables(adapter, commands, group); !written)
                    return written;
            moved |= adoptHeap(commands.resourceHeap, tables.resourceHeap);
            moved |= adoptHeap(commands.samplerHeap, tables.samplerHeap);
            commands.tables[index] = tables;
        }
        if (!moved)
            break;
        if (pass + 1 == kSettlePasses)
            return providerFailure(vernon::ProviderErrorCode::BackendFailure, "d3d12_descriptor_heaps_did_not_settle");
    }
    if (commands.setResourceHeap == commands.resourceHeap && commands.setSamplerHeap == commands.samplerHeap)
        return RhiAdapterResult<void>{vernon::ok()};
    std::array<ID3D12DescriptorHeap *, 2> heaps{};
    UINT heapCount = 0;
    for (ID3D12DescriptorHeap *heap : {commands.resourceHeap, commands.samplerHeap})
        if (heap)
            heaps[heapCount++] = heap;
    commands.list->SetDescriptorHeaps(heapCount, heaps.data());
    commands.setResourceHeap = commands.resourceHeap;
    commands.setSamplerHeap = commands.samplerHeap;
    // Tables set before name descriptors in the heaps just replaced.
    commands.graphicsState.tables.fill(0);
    commands.computeState.tables.fill(0);
    return RhiAdapterResult<void>{vernon::ok()};
}

// Sets the root signature, the descriptor tables and the root constants of the bound groups that the native state
// lacks.
void flushBindPoint(DirectX12Commands &commands, BindPointState &state, bool graphics) {
    const PreparedLayout &layout = *commands.pipeline->layout;
    ID3D12GraphicsCommandList *list = commands.list;
    if (state.layout != &layout) {
        if (graphics)
            list->SetGraphicsRootSignature(layout.rootSignature);
        else
            list->SetComputeRootSignature(layout.rootSignature);
        state.layout = &layout;
        state.tables.fill(0);
        state.constants.fill(0);
    }
    const auto setTable = [&](uint32_t parameter, D3D12_GPU_DESCRIPTOR_HANDLE table) {
        if (parameter == kNone)
            return;
        if (graphics)
            list->SetGraphicsRootDescriptorTable(parameter, table);
        else
            list->SetComputeRootDescriptorTable(parameter, table);
    };
    uint32_t changedConstants = 0;
    for (uint32_t index = 0; index < layout.groupCount; ++index) {
        if ((layout.groupMask & (1u << index)) == 0)
            continue;
        const uint64_t serial = commands.groups[index]->serial;
        if (state.tables[index] != serial) {
            setTable(layout.resourceTables[index], commands.tables[index].resourceGpu);
            setTable(layout.samplerTables[index], commands.tables[index].samplerGpu);
            state.tables[index] = serial;
        }
        if ((layout.rootGroupMask & (1u << index)) != 0 && state.constants[index] != serial) {
            changedConstants |= 1u << index;
            state.constants[index] = serial;
        }
    }
    for (const RootConstantBlock &block : layout.rootBlocks) {
        if ((block.groupMask & changedConstants) == 0)
            continue;
        std::array<uint32_t, kRootDwordLimit> values{};
        for (const RootConstantMember &member : block.members)
            std::memcpy(reinterpret_cast<uint8_t *>(values.data()) + member.offset,
                        commands.groups[member.group]->rootData.data() + member.dataOffset, member.size);
        list->SetGraphicsRoot32BitConstants(block.parameter, block.valueCount, values.data(), 0);
    }
}

RhiAdapterResult<void> transitionGroups(VernonRuntimeRhiAdapter &adapter, DirectX12Commands &commands) {
    for (uint32_t index = 0; index < commands.pipeline->layout->groupCount; ++index) {
        if ((commands.pipeline->layout->groupMask & (1u << index)) == 0)
            continue;
        const DirectX12Group &group = *commands.groups[index];
        if (commands.transitionedGroups[index] == group.serial && commands.transitionedEpochs[index] == commands.epoch)
            continue;
        for (const DirectX12Group::BufferUse &use : group.buffers)
            if (auto transitioned = transitionBuffer(adapter, commands, *use.buffer, use.state); !transitioned)
                return transitioned;
        for (const DirectX12Group::ImageUse &use : group.images)
            if (auto transitioned = transitionImage(adapter, commands, *use.image, use.state); !transitioned)
                return transitioned;
        commands.transitionedGroups[index] = group.serial;
        commands.transitionedEpochs[index] = commands.epoch;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

// Sets the views of the bound pipeline's vertex bindings that the list lacks; a view carries the variant's stride.
RhiAdapterResult<void> applyVertexBuffers(VernonRuntimeRhiAdapter &adapter, DirectX12Commands &commands) {
    const PreparedPipeline &pipeline = *commands.pipeline;
    for (const VernonRuntimeProviderVertexBinding &vertexBinding : pipeline.layout->vertexBindings) {
        const uint32_t binding = vertexBinding.binding;
        const DirectX12Commands::VertexBuffer &buffer = commands.vertexBuffers[binding];
        if (commands.vertexEpochs[binding] != commands.epoch) {
            if (auto transitioned = transitionBuffer(adapter, commands, *buffer.buffer,
                                                     D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                !transitioned)
                return transitioned;
            commands.vertexEpochs[binding] = commands.epoch;
        }
        const uint32_t stride = pipeline.vertexStrides[binding];
        if ((commands.vertexViews & (1u << binding)) != 0 && commands.vertexViewStrides[binding] == stride)
            continue;
        const D3D12_VERTEX_BUFFER_VIEW view{buffer.buffer->resource->GetGPUVirtualAddress() + buffer.offset,
                                            static_cast<UINT>(buffer.size), stride};
        commands.list->IASetVertexBuffers(binding, 1, &view);
        commands.vertexViews |= 1u << binding;
        commands.vertexViewStrides[binding] = stride;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> bindIndexBuffer(VernonRuntimeRhiAdapter &adapter, DirectX12Commands &commands,
                                       rhi::directx12::Buffer *buffer,
                                       const VernonRuntimeProviderResourceReference &reference) {
    if (!buffer || !buffer->resource || reference.size == 0 || reference.size > UINT32_MAX)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                               "d3d12_draw_contains_an_invalid_index_buffer");
    if (commands.indexBuffer != buffer) {
        commands.indexBuffer = buffer;
        commands.indexBufferOffset = UINT64_MAX;
        commands.indexBufferEpoch = 0;
    }
    if (commands.indexBufferEpoch != commands.epoch) {
        if (auto transitioned = transitionBuffer(adapter, commands, *buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
            !transitioned)
            return transitioned;
        commands.indexBufferEpoch = commands.epoch;
    }
    if (commands.indexBufferOffset != reference.offset) {
        const D3D12_INDEX_BUFFER_VIEW view{buffer->resource->GetGPUVirtualAddress() + reference.offset,
                                           static_cast<UINT>(reference.size), DXGI_FORMAT_R32_UINT};
        commands.list->IASetIndexBuffer(&view);
        commands.indexBufferOffset = reference.offset;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

void setDynamicState(DirectX12Commands &commands, uint32_t topologyIndex,
                     const VernonRuntimeProviderDrawPayload &draw) {
    ID3D12GraphicsCommandList *list = commands.list;
    const std::array<uint32_t, 4> viewport{draw.viewport[0], draw.viewport[1], draw.viewport[2], draw.viewport[3]};
    if (!commands.viewportValid || commands.viewport != viewport) {
        const D3D12_VIEWPORT native{static_cast<float>(viewport[0]),
                                    static_cast<float>(viewport[1]),
                                    static_cast<float>(viewport[2]),
                                    static_cast<float>(viewport[3]),
                                    0,
                                    1};
        list->RSSetViewports(1, &native);
        commands.viewport = viewport;
        commands.viewportValid = true;
    }
    const std::array<uint32_t, 4> scissor{draw.scissor[0], draw.scissor[1], draw.scissor[2], draw.scissor[3]};
    if (!commands.scissorValid || commands.scissor != scissor) {
        const D3D12_RECT native{static_cast<LONG>(scissor[0]), static_cast<LONG>(scissor[1]),
                                static_cast<LONG>(scissor[0] + scissor[2]), static_cast<LONG>(scissor[1] + scissor[3])};
        list->RSSetScissorRects(1, &native);
        commands.scissor = scissor;
        commands.scissorValid = true;
    }
    if (!commands.stencilValid || commands.stencilReference != draw.stencil_reference) {
        list->OMSetStencilRef(draw.stencil_reference);
        commands.stencilReference = draw.stencil_reference;
        commands.stencilValid = true;
    }
    const D3D_PRIMITIVE_TOPOLOGY topology = topologyIndex == 1   ? D3D_PRIMITIVE_TOPOLOGY_LINELIST
                                            : topologyIndex == 2 ? D3D_PRIMITIVE_TOPOLOGY_POINTLIST
                                                                 : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    if (commands.topology != topology) {
        list->IASetPrimitiveTopology(topology);
        commands.topology = topology;
    }
}

struct DirectX12PacketDecoder {
    VernonRuntimeRhiAdapter &adapter;
    DirectX12Commands &commands;
    const RhiPacketBatch &batch;

    RhiAdapterResult<void> beginRendering() {
        commands.rendering = true;
        commands.scopeOpened = false;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    // A scope no draw opened clears its attachments as it ends, which may set descriptor heaps.
    RhiAdapterResult<void> endRendering() {
        commands.rendering = false;
        commands.setResourceHeap = nullptr;
        commands.setSamplerHeap = nullptr;
        ++commands.epoch;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> bindPipeline(RhiRetainedObject &object) {
        auto &pipeline = static_cast<PreparedPipeline &>(object);
        if (!pipeline.pipeline)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_bind_pipeline_is_invalid");
        const PreparedLayout &layout = *pipeline.layout;
        for (uint32_t group = 0; group < kMaxGroups; ++group)
            if (commands.groups[group] &&
                (group >= layout.groupCount || commands.groups[group]->layout != layout.groups[group]))
                commands.groups[group] = nullptr;
        commands.list->SetPipelineState(pipeline.pipeline);
        commands.pipeline = &pipeline;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> bindGroup(uint32_t index, RhiGroup &object) {
        auto &group = static_cast<DirectX12Group &>(object);
        if (!commands.pipeline || index >= commands.pipeline->layout->groupCount ||
            group.layout != commands.pipeline->layout->groups[index])
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_bind_group_is_invalid", index);
        commands.groups[index] = &group;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> bindVertexBuffers(uint32_t first, const uint32_t *resources, size_t count) {
        for (size_t index = 0; index < count; ++index) {
            const VernonRuntimeProviderResourceReference &reference = batch.resources[resources[index]];
            const size_t binding = first + index;
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(batch.natives[resources[index]]);
            if (!buffer || !buffer->resource || reference.size - reference.offset > UINT32_MAX)
                return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_vertex_buffer_is_invalid",
                                       binding);
            commands.vertexBuffers[binding] = {buffer, reference.offset, reference.size - reference.offset};
            commands.vertexEpochs[binding] = 0;
            commands.vertexViews &= ~(1u << binding);
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> dispatch(const uint32_t *groupCount) {
        if (!commands.pipeline || commands.pipeline->graphics || commands.rendering)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "d3d12_adapter_received_an_invalid_dispatch");
        if (auto prepared = prepareGroups(adapter, commands); !prepared)
            return prepared;
        flushBindPoint(commands, commands.computeState, false);
        if (auto transitioned = transitionGroups(adapter, commands); !transitioned)
            return transitioned;
        commands.list->Dispatch(groupCount[0], groupCount[1], groupCount[2]);
        // The dispatch may write what a later command reads in another state.
        ++commands.epoch;
        return RhiAdapterResult<void>{vernon::ok()};
    }

    RhiAdapterResult<void> draw(uint32_t topology, uint32_t indexResource, const VernonRuntimeProviderDrawPayload &draw,
                                const vernon::rhi::OpenedRendering &) {
        if (!commands.pipeline || !commands.pipeline->graphics || !commands.rendering)
            return providerFailure(vernon::ProviderErrorCode::InvalidArgument,
                                   "d3d12_adapter_received_an_invalid_draw_descriptor");
        // SV_InstanceID and SV_VertexID exclude the draw offsets, so nonzero offsets would change `instance_index`
        // and `vertex_index`.
        if (draw.first_instance != 0 || draw.base_vertex != 0)
            return providerFailure(vernon::ProviderErrorCode::Unsupported,
                                   "d3d12_draw_base_instance_or_base_vertex_is_unsupported");
        // Opening the native scope clears its attachments, which may set descriptor heaps, and transitions its images.
        if (!commands.scopeOpened) {
            commands.scopeOpened = true;
            commands.setResourceHeap = nullptr;
            commands.setSamplerHeap = nullptr;
            ++commands.epoch;
        }
        if (auto prepared = prepareGroups(adapter, commands); !prepared)
            return prepared;
        flushBindPoint(commands, commands.graphicsState, true);
        if (auto transitioned = transitionGroups(adapter, commands); !transitioned)
            return transitioned;
        if (auto applied = applyVertexBuffers(adapter, commands); !applied)
            return applied;
        setDynamicState(commands, topology, draw);
        if (indexResource == UINT32_MAX) {
            commands.list->DrawInstanced(draw.vertex_count, draw.instance_count, 0, 0);
            return RhiAdapterResult<void>{vernon::ok()};
        }
        if (auto bound = bindIndexBuffer(adapter, commands,
                                         reinterpret_cast<rhi::directx12::Buffer *>(batch.natives[indexResource]),
                                         batch.resources[indexResource]);
            !bound)
            return bound;
        commands.list->DrawIndexedInstanced(draw.vertex_count, draw.instance_count, draw.first_index, 0, 0);
        return RhiAdapterResult<void>{vernon::ok()};
    }
};

RhiRetainedObject *packetPipeline(VernonRuntimeProviderObject handle) noexcept {
    return fromHandle<PreparedPipeline>(handle);
}
RhiGroup *packetGroup(VernonRuntimeProviderObject handle) noexcept { return fromHandle<DirectX12Group>(handle); }

// A batch starts with nothing bound; the native binding state carries over, since the encoder keeps every object an
// earlier batch bound alive.
RhiAdapterResult<void> encodePacketsResult(void *data, VernonRuntimeProviderObject handle,
                                           VernonRuntimeProviderObject batchHandle, const uint32_t *payload,
                                           size_t payloadWords) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *commands = fromHandle<DirectX12Commands>(handle);
    auto *batch = fromHandle<RhiPacketBatch>(batchHandle);
    if (!commands || !batch || commands->rendering)
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_encode_packets_is_invalid");
    commands->pipeline = nullptr;
    commands->groups.fill(nullptr);
    DirectX12PacketDecoder decoder{adapter, *commands, *batch};
    return decodeRhiPackets(adapter, commands->encoder, *batch, payload, payloadWords, decoder);
}

void destroyShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedShader>(handle); }
void destroyGroupLayout(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<GroupLayout>(handle));
}
void destroyLayout(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedLayout>(handle));
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<PreparedPipeline>(handle));
}
void destroyGroup(void *, VernonRuntimeProviderObject handle) {
    releaseRetainedObject(fromHandle<DirectX12Group>(handle));
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "D3D12 shader preparation failed");
}
VernonStatus prepareGroupLayout(void *data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareGroupLayoutResult(data, descriptor, output),
                          "D3D12 group layout preparation failed");
}
VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareLayoutResult(data, descriptor, output), "D3D12 layout preparation failed");
}
VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, preparePipelineResult(data, descriptor, output),
                          "D3D12 pipeline preparation failed");
}
VernonStatus createGroup(void *data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                         VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createGroupResult(data, descriptor, output), "D3D12 group creation failed");
}
VernonStatus beginCommands(void *data, VernonRuntimeProviderObject encoder, VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, beginCommandsResult(data, encoder, output), "D3D12 commands could not begin");
}
VernonStatus endCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, endCommandsResult(data, commands), "D3D12 commands failed");
}
VernonStatus resumeCommands(void *data, VernonRuntimeProviderObject commands) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, resumeCommandsResult(data, commands), "D3D12 commands could not resume");
}
VernonStatus encodePackets(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject batch,
                           const uint32_t *payload, size_t payloadWords) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodePacketsResult(data, commands, batch, payload, payloadWords),
                          "D3D12 packet encoding failed");
}
VernonStatus updateGroup(void *data, VernonRuntimeProviderObject commands, VernonRuntimeProviderObject group,
                         const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, updateGroupResult(data, commands, group, values, valueCount),
                          "D3D12 group update failed");
}
VernonStatus createPacketBatch(void *data, const VernonRuntimeProviderPacketBatchDescriptor *descriptor,
                               VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createRhiPacketBatch(adapter, descriptor, {packetPipeline, packetGroup}, output),
                          "D3D12 packet batch creation failed");
}

} // namespace

void initializeDirectX12Provider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_group_layout = prepareGroupLayout;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_group = createGroup;
    adapter.provider.begin_commands = beginCommands;
    adapter.provider.end_commands = endCommands;
    adapter.provider.resume_commands = resumeCommands;
    adapter.provider.create_packet_batch = createPacketBatch;
    adapter.provider.encode_packets = encodePackets;
    adapter.provider.update_group = updateGroup;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_group_layout = destroyGroupLayout;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_group = destroyGroup;
    adapter.provider.destroy_packet_batch = destroyRhiPacketBatch;
}

} // namespace vernon::runtime::rhi_adapter

namespace vernon::runtime {

DirectX12AdapterDepthStencilStats
getDirectX12AdapterDepthStencilStats(const VernonRuntimeRhiAdapter &adapter) noexcept {
    const auto &state = rhi_adapter::directX12State(adapter).lastDepthStencilState;
    return {state.depthEnable,          state.depthWriteMask,
            state.depthFunction,        state.stencilEnable,
            state.stencilReadMask,      state.stencilWriteMask,
            state.frontStencilFunction, state.frontStencilPassOperation,
            state.backStencilFunction,  state.backStencilPassOperation};
}

VernonRuntimeRhiAdapter *createDirectX12RhiAdapter(VernonRhiDevice device, VernonRhiBackend backend) {
    if (backend != VERNON_RHI_BACKEND_DIRECTX12)
        return nullptr;
    auto resolvedDeviceState = rhi::deviceState(device, backend);
    if (resolvedDeviceState.isErr())
        return nullptr;
    auto *deviceState = static_cast<rhi::directx12::DeviceState *>(resolvedDeviceState.value());
    auto state =
        std::unique_ptr<rhi_adapter::DirectX12AdapterState>(new (std::nothrow) rhi_adapter::DirectX12AdapterState());
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!state || !adapter)
        return nullptr;
    state->device = deviceState;
    state->resourceIncrement =
        deviceState->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    state->samplerIncrement = deviceState->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    [[maybe_unused]] auto *adoptedState = state.release();
    rhi_adapter::initializeDirectX12Provider(*adapter);
    return adapter.release();
}

uint32_t getDirectX12BlendFactorMapping(uint32_t value) {
    return static_cast<uint32_t>(rhi_adapter::blendFactor(value));
}
uint32_t getDirectX12BlendOperationMapping(uint32_t value) {
    return static_cast<uint32_t>(rhi_adapter::blendOperation(value));
}
uint32_t getDirectX12CompareOperationMapping(uint32_t value) {
    return static_cast<uint32_t>(rhi_adapter::compareOperation(value));
}
uint32_t getDirectX12StencilOperationMapping(uint32_t value) {
    return static_cast<uint32_t>(rhi_adapter::stencilOperation(value));
}
uint32_t getDirectX12CullModeMapping(uint32_t value) { return static_cast<uint32_t>(rhi_adapter::cullMode(value)); }

} // namespace vernon::runtime

#endif
