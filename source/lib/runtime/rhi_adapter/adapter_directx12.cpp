#include "adapter_common.h"
#include "adapter_directx12_test_hooks.h"
#include "rhi/rhi_internal.h"
#include "runtime/runtime_test_hooks.h"
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
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

// D3D12 reserves register spaces 0xFFFFFFF0 and above.
constexpr uint32_t kDescriptorSetLimit = 0xFFFFFFF0U - ::vernon::runtime::kDirectXDescriptorSetRegisterSpaceOffset;

D3D12_DESCRIPTOR_RANGE descriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE type,
                                       const VernonRuntimeProviderBindingLayoutEntry &entry) {
    return {type, 1, entry.binding, entry.set + ::vernon::runtime::kDirectXDescriptorSetRegisterSpaceOffset,
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
}

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
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure, {"directx12_synchronize", 0, 0}})};
    } catch (...) {
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure, {"directx12_synchronize", 0, 0}})};
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const DirectX12AdapterState *>(state)->device);
}
const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity};

struct PreparedShader {
    uint32_t stage{};
    std::vector<uint8_t> artifact;
};
struct PreparedLayout {
    std::vector<VernonRuntimeProviderBindingLayoutEntry> entries;
    std::vector<VernonRuntimeProviderVertexAttribute> vertexAttributes;
};
struct PreparedPipeline {
    struct InlineRootMember {
        uint32_t slot{};
        uint32_t destinationOffset{};
        uint32_t size{};
    };
    struct InlineRootBlock {
        uint32_t rootParameter{};
        uint32_t valueCount{};
        std::vector<InlineRootMember> members;
    };
    std::atomic<uint32_t> references{1};
    VernonRuntimeRhiAdapter *owner{};
    rhi::directx12::DeviceState *device{};
    ID3D12RootSignature *rootSignature{};
    ID3D12PipelineState *pipeline{};
    bool graphics{};
    bool hasResourceTable{};
    bool hasSamplerTable{};
    std::vector<InlineRootBlock> inlineRootBlocks;

    ~PreparedPipeline() {
        if (pipeline)
            pipeline->Release();
        if (rootSignature)
            rootSignature->Release();
        if (owner)
            owner->livePreparedPipelines.fetch_sub(1, std::memory_order_relaxed);
    }
};
// Shader-visible descriptor tables written for one command encoder; a binding set reused within that encoder keeps
// them instead of acquiring and writing new ones.
struct DescriptorTables {
    uint64_t encoder{};
    ID3D12DescriptorHeap *resourceHeap{};
    ID3D12DescriptorHeap *samplerHeap{};
    D3D12_GPU_DESCRIPTOR_HANDLE resourceGpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE samplerGpu{};
};
struct PreparedBindingSet {
    std::atomic<uint32_t> references{1};
    struct Slot {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        ID3D12Resource *resource{};
        void *opaqueResource{};
        VernonRuntimeProviderResourceReference resourceReference{};
        rhi::directx12::Buffer inlineResource;
        uint64_t offset{};
        uint64_t size{};
        uint32_t stride{};
        uint32_t flags{};
        std::vector<uint8_t> inlineStorage;
    };
    rhi::directx12::DeviceState *device{};
    std::vector<Slot> slots;
    std::unordered_map<uint32_t, size_t> slotIndices;
    uint32_t resourceDescriptorCount{};
    uint32_t samplerDescriptorCount{};
    // Draws and dispatches cache separately because one binding set may be bound by both within one encoder.
    DescriptorTables drawDescriptors;
    DescriptorTables dispatchDescriptors;
    std::mutex mutex;
};

void releaseCommandBindings(void *context, uint64_t);
void releaseCommandPipeline(void *context, uint64_t);

void restoreBufferState(void *context, uint64_t state) {
    static_cast<rhi::directx12::Buffer *>(context)->state = static_cast<D3D12_RESOURCE_STATES>(state);
}

// Graphics inline values become root constants, while compute inline values are raw UAV buffers.
std::optional<D3D12_DESCRIPTOR_RANGE_TYPE> descriptorRangeType(const VernonRuntimeProviderBindingLayoutEntry &entry,
                                                               bool compute) {
    switch (entry.kind) {
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
        return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
        if (compute)
            return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

// Each table lists its descriptors in layout entry order, which is the order binding encoders write them in.
RhiAdapterResult<void> appendDescriptorTables(const std::vector<VernonRuntimeProviderBindingLayoutEntry> &entries,
                                              bool compute, std::vector<D3D12_DESCRIPTOR_RANGE> &resourceRanges,
                                              std::vector<D3D12_DESCRIPTOR_RANGE> &samplerRanges,
                                              std::vector<D3D12_ROOT_PARAMETER> &parameters,
                                              PreparedPipeline &pipeline) {
    for (const auto &entry : entries) {
        const std::optional<D3D12_DESCRIPTOR_RANGE_TYPE> type = descriptorRangeType(entry, compute);
        if (!type)
            continue;
        if (entry.set >= kDescriptorSetLimit)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument,
                {"d3d12_descriptor_set_exceeds_the_register_space_range", entry.set, kDescriptorSetLimit}})};
        (*type == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER ? samplerRanges : resourceRanges)
            .push_back(descriptorRange(*type, entry));
    }
    for (const auto *ranges : {&resourceRanges, &samplerRanges}) {
        if (ranges->empty())
            continue;
        auto &parameter = parameters.emplace_back();
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {static_cast<UINT>(ranges->size()), ranges->data()};
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    pipeline.hasResourceTable = !resourceRanges.empty();
    pipeline.hasSamplerTable = !samplerRanges.empty();
    return RhiAdapterResult<void>{vernon::ok()};
}

struct AcquiredDescriptorTables {
    DescriptorTables tables;
    // Write cursors into freshly acquired tables; unused when the cached tables are reused.
    D3D12_CPU_DESCRIPTOR_HANDLE resourceCpu{};
    D3D12_CPU_DESCRIPTOR_HANDLE samplerCpu{};
    bool reused{};
};

RhiAdapterResult<AcquiredDescriptorTables> acquireDescriptorTables(rhi::directx12::DeviceState &device,
                                                                   const PreparedBindingSet *bindings,
                                                                   const DescriptorTables *cached, uint64_t encoder,
                                                                   std::string &diagnostic) {
    AcquiredDescriptorTables acquired;
    if (cached && cached->encoder == encoder) {
        acquired.tables = *cached;
        acquired.reused = true;
        return RhiAdapterResult<AcquiredDescriptorTables>{vernon::ok(acquired)};
    }
    acquired.tables.encoder = encoder;
    const uint32_t resourceCount = bindings ? bindings->resourceDescriptorCount : 0;
    const uint32_t samplerCount = bindings ? bindings->samplerDescriptorCount : 0;
    if (resourceCount && !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, resourceCount,
                                                    acquired.tables.resourceHeap, acquired.resourceCpu,
                                                    acquired.tables.resourceGpu, diagnostic))
        return RhiAdapterResult<AcquiredDescriptorTables>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_acquire_resource_descriptors", resourceCount, 0}})};
    if (samplerCount &&
        !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, samplerCount, acquired.tables.samplerHeap,
                                   acquired.samplerCpu, acquired.tables.samplerGpu, diagnostic))
        return RhiAdapterResult<AcquiredDescriptorTables>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_acquire_sampler_descriptors", samplerCount, 0}})};
    return RhiAdapterResult<AcquiredDescriptorTables>{vernon::ok(acquired)};
}

// Root parameter order matches appendDescriptorTables: the resource table, then the sampler table.
void bindDescriptorTables(ID3D12GraphicsCommandList *commands, const PreparedPipeline &pipeline,
                          const DescriptorTables &tables) {
    std::array<ID3D12DescriptorHeap *, 2> heaps{};
    UINT heapCount = 0;
    for (ID3D12DescriptorHeap *heap : {tables.resourceHeap, tables.samplerHeap})
        if (heap)
            heaps[heapCount++] = heap;
    if (heapCount)
        commands->SetDescriptorHeaps(heapCount, heaps.data());
    if (pipeline.graphics)
        commands->SetGraphicsRootSignature(pipeline.rootSignature);
    else
        commands->SetComputeRootSignature(pipeline.rootSignature);
    UINT rootIndex = 0;
    const auto bindTable = [&](D3D12_GPU_DESCRIPTOR_HANDLE table) {
        if (pipeline.graphics)
            commands->SetGraphicsRootDescriptorTable(rootIndex++, table);
        else
            commands->SetComputeRootDescriptorTable(rootIndex++, table);
    };
    if (pipeline.hasResourceTable)
        bindTable(tables.resourceGpu);
    if (pipeline.hasSamplerTable)
        bindTable(tables.samplerGpu);
}

const D3D12_SAMPLER_DESC &samplerDescriptor(const PreparedBindingSet::Slot &slot) {
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
    const auto *sampler = static_cast<const rhi::directx12::Sampler *>(slot.opaqueResource);
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

RhiAdapterResult<void> transition(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                  ID3D12GraphicsCommandList *commands, rhi::directx12::Buffer &buffer,
                                  D3D12_RESOURCE_STATES target) {
    auto deferred =
        deferCommandRollback(adapter, encoder, &buffer, static_cast<uint64_t>(buffer.state), restoreBufferState);
    if (!deferred)
        return deferred;
    rhi::directx12::transition(commands, buffer.resource, buffer.state, target);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> transition(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                  ID3D12GraphicsCommandList *commands, rhi::directx12::Image &image,
                                  D3D12_RESOURCE_STATES target) {
    auto transitioned =
        rhi::directx12::transitionCommandImage(adapter.rhiDevice, encoder.value, commands, image, target);
    if (transitioned.isErr())
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(transitioned).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

struct DescriptorWriter {
    VernonRuntimeRhiAdapter &adapter;
    VernonRuntimeProviderObject commandEncoder;
    ID3D12GraphicsCommandList *commands;
    rhi::directx12::DeviceState &device;
    // A binding set reused within one command encoder keeps its descriptors, but its resources still need transitions.
    bool reuse;
    D3D12_CPU_DESCRIPTOR_HANDLE resourceCpu;
    D3D12_CPU_DESCRIPTOR_HANDLE samplerCpu;
};

/* Draws and dispatches bind resource buffers, images, and samplers alike: each is transitioned and then, unless
 * reused, written in layout order. Returns nullopt for slot kinds whose binding differs between the two. */
std::optional<RhiAdapterResult<void>> writeResourceDescriptor(DescriptorWriter &writer,
                                                              const PreparedBindingSet::Slot &slot) {
    const auto failure = [&](vernon::ProviderErrorCode code, const char *reason) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{code, {reason, slot.layout.slot, 0}})};
    };
    const auto advance = [&](D3D12_CPU_DESCRIPTOR_HANDLE &handle, D3D12_DESCRIPTOR_HEAP_TYPE type) {
        handle.ptr += writer.device.device->GetDescriptorHandleIncrementSize(type);
    };
    switch (slot.layout.kind) {
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER: {
        auto *buffer = static_cast<rhi::directx12::Buffer *>(slot.opaqueResource);
        if (!buffer || !buffer->resource)
            return failure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_binding_has_an_invalid_storage_buffer");
        if (!transition(writer.adapter, writer.commandEncoder, writer.commands, *buffer,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
            return failure(vernon::ProviderErrorCode::BackendFailure, "d3d12_could_not_track_storage_buffer_state");
        if (!writer.reuse) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            view.Format = DXGI_FORMAT_R32_TYPELESS;
            view.Buffer.FirstElement = slot.offset / 4;
            view.Buffer.NumElements = static_cast<UINT>(std::max<uint64_t>(1, (slot.size + 3) / 4));
            view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            writer.device.device->CreateUnorderedAccessView(buffer->resource, nullptr, &view, writer.resourceCpu);
            advance(writer.resourceCpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE: {
        const bool storage = slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
        auto *image = static_cast<rhi::directx12::Image *>(slot.opaqueResource);
        if (!image || !image->resource)
            return failure(vernon::ProviderErrorCode::InvalidArgument, "d3d12_binding_has_an_invalid_image");
        if (!transition(writer.adapter, writer.commandEncoder, writer.commands, *image,
                        storage ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE))
            return failure(vernon::ProviderErrorCode::BackendFailure, "d3d12_could_not_track_image_state");
        if (!writer.reuse) {
            if (storage) {
                const D3D12_UNORDERED_ACCESS_VIEW_DESC view = storageImageView(*image);
                writer.device.device->CreateUnorderedAccessView(image->resource, nullptr, &view, writer.resourceCpu);
            } else {
                const D3D12_SHADER_RESOURCE_VIEW_DESC view = sampledImageView(*image);
                writer.device.device->CreateShaderResourceView(image->resource, &view, writer.resourceCpu);
            }
            advance(writer.resourceCpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        return RhiAdapterResult<void>{vernon::ok()};
    }
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        if (!writer.reuse) {
            writer.device.device->CreateSampler(&samplerDescriptor(slot), writer.samplerCpu);
            advance(writer.samplerCpu, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        }
        return RhiAdapterResult<void>{vernon::ok()};
    default:
        return std::nullopt;
    }
}

/* Packed uniform bytes live in a slot-owned default-heap buffer, so a fresh descriptor table stages and copies them
 * before writing a constant-buffer view (uniform buffers) or a raw UAV (every other packed kind). */
RhiAdapterResult<void> writeInlineBufferDescriptor(DescriptorWriter &writer, PreparedBindingSet::Slot &slot,
                                                   std::string &backendDiagnostic) {
    if (writer.reuse)
        return RhiAdapterResult<void>{vernon::ok()};
    const size_t size = slot.inlineStorage.size();
    ID3D12Resource *upload = nullptr;
    size_t uploadOffset = 0;
    uint8_t *mapped = nullptr;
    if (!writer.device.acquireStaging(true, size, 256, upload, uploadOffset, mapped, backendDiagnostic))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                              {"d3d12_acquire_inline_binding_staging", size, slot.layout.slot}})};
    std::memcpy(mapped, slot.inlineStorage.data(), size);
    const bool constantBuffer = slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
    const auto transitionInline = [&](D3D12_RESOURCE_STATES target) -> RhiAdapterResult<void> {
        if (!transition(writer.adapter, writer.commandEncoder, writer.commands, slot.inlineResource, target))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure,
                {"d3d12_could_not_track_inline_binding_state", static_cast<uint64_t>(target), slot.layout.slot}})};
        return RhiAdapterResult<void>{vernon::ok()};
    };
    if (auto copyDestination = transitionInline(D3D12_RESOURCE_STATE_COPY_DEST); !copyDestination)
        return copyDestination;
    writer.commands->CopyBufferRegion(slot.inlineResource.resource, 0, upload, uploadOffset, size);
    if (auto bound = transitionInline(constantBuffer ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                                                     : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        !bound)
        return bound;
    if (constantBuffer) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC view{};
        view.BufferLocation = slot.inlineResource.resource->GetGPUVirtualAddress();
        view.SizeInBytes = (static_cast<UINT>(size) + 255u) & ~255u;
        writer.device.device->CreateConstantBufferView(&view, writer.resourceCpu);
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.Buffer.NumElements = static_cast<UINT>(std::max<size_t>(1, (size + 3) / 4));
        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        writer.device.device->CreateUnorderedAccessView(slot.inlineResource.resource, nullptr, &view,
                                                        writer.resourceCpu);
    }
    writer.resourceCpu.ptr +=
        writer.device.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_command_pipeline_cleanup_registration_failed", 0, 0}})};
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                  {"d3d12_command_bindings_cleanup_registration_failed", 0, 0}})};
        }
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

template <typename Object> void releaseObject(Object *&object) {
    if (object)
        object->Release();
    object = nullptr;
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
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"d3d12_adapter_received_an_invalid_dxil_shader_descriptor", 0, 0}})};
    try {
        auto shader = std::make_unique<PreparedShader>();
        shader->stage = descriptor->stage;
        const auto *bytes = static_cast<const uint8_t *>(descriptor->data);
        shader->artifact.assign(bytes, bytes + descriptor->size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_shader_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> prepareLayoutResult(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"d3d12_adapter_received_an_invalid_pipeline_layout", 0, 0}})};
    try {
        auto layout = std::make_unique<PreparedLayout>();
        layout->entries.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
        constexpr uint32_t graphicsStages =
            VERNON_RUNTIME_PROVIDER_STAGE_VERTEX | VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
        for (const auto &entry : layout->entries) {
            // Descriptor tables are visible to every stage, so a descriptor may serve both shader stages; inline
            // values become per-stage root constants and must name exactly one stage.
            const bool graphicsDescriptorStages = entry.stage_mask != 0 && (entry.stage_mask & ~graphicsStages) == 0;
            const bool compute =
                ((entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                  entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && entry.binding != UINT32_MAX) ||
                 entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                 entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                 entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                 ((entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                   entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) &&
                  entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE)) &&
                entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE && entry.element_size != 0;
            const bool graphicsResource = (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                           entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                           entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) &&
                                          entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                          graphicsDescriptorStages && entry.binding != UINT32_MAX;
            const bool graphicsInline = entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                                        entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                        (entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                         entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                        entry.element_size != 0 && entry.element_size % sizeof(uint32_t) == 0;
            const bool graphicsUniformBuffer = (entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                                entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) &&
                                               entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                               graphicsDescriptorStages && entry.binding != UINT32_MAX &&
                                               entry.element_size != 0;
            const bool vertex = entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                                entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT &&
                                entry.binding < D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT && entry.element_size != 0;
            if ((!compute && !graphicsResource && !graphicsInline && !graphicsUniformBuffer && !vertex) ||
                entry.array_count != 1)
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::Unsupported,
                                                      {"d3d12_layout_contains_an_unsupported_binding",
                                                       static_cast<uint32_t>(entry.kind), entry.array_count}})};
        }
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            const bool bindingExists =
                std::any_of(layout->entries.begin(), layout->entries.end(), [&](const auto &entry) {
                    return entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.binding == attribute.binding;
                });
            std::string capabilityDiagnostic;
            if (!bindingExists)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::Unsupported,
                    {"d3d12_validate_vertex_attribute_binding", attribute.binding, attribute.location}})};
            if (!validateVertexAttributeCapability(VertexAttributeBackend::DirectX12, attribute,
                                                   D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, false,
                                                   capabilityDiagnostic))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::Unsupported,
                    {"d3d12_validate_vertex_attribute_capability", attribute.location, attribute.binding}})};
            layout->vertexAttributes.push_back(attribute);
        }
        *output = toHandle(layout.release());
        adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_layout_preparation_ran_out_of_memory", 0, 0}})};
    }
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

RhiAdapterResult<void> prepareGraphicsPipelineImplResult(VernonRuntimeRhiAdapter &adapter,
                                                         const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                                         PreparedLayout &layout, VernonRuntimeProviderObject *output) {
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                              {"d3d12_graphics_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    pipeline->owner = &adapter;
    adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
    pipeline->device = &directX12Device(adapter);
    pipeline->graphics = true;
    if (descriptor.color_format_count > 8 || descriptor.sample_count != 1 ||
        (descriptor.depth_stencil_format && descriptor.depth_stencil_format != DXGI_FORMAT_D32_FLOAT &&
         descriptor.depth_stencil_format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"d3d12_graphics_pipeline_uses_an_unsupported_attachment_configuration", 0, 0}})};
    PreparedShader *vertex = nullptr;
    PreparedShader *fragment = nullptr;
    for (size_t index = 0; index < descriptor.shader_count; ++index) {
        auto *shader = fromHandle<PreparedShader>(descriptor.shaders[index]);
        if (!shader)
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"d3d12_graphics_pipeline_contains_an_invalid_shader", 0, 0}})};
        if (shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
            vertex = shader;
        else if (shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
            fragment = shader;
    }
    if (!vertex || !fragment)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"d3d12_graphics_pipeline_requires_vertex_and_fragment_shaders", 0, 0}})};

    std::vector<D3D12_DESCRIPTOR_RANGE> resourceRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE> samplerRanges;
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    parameters.reserve(2 + layout.entries.size());
    auto tables = appendDescriptorTables(layout.entries, false, resourceRanges, samplerRanges, parameters, *pipeline);
    if (!tables)
        return tables;
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements;
    for (const VernonRuntimeProviderVertexAttribute &attribute : layout.vertexAttributes) {
        const auto binding = std::find_if(layout.entries.begin(), layout.entries.end(), [&](const auto &entry) {
            return entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.binding == attribute.binding;
        });
        const DXGI_FORMAT format = vertexFormat(attribute.dtype, attribute.component_count);
        if (binding == layout.entries.end() || format == DXGI_FORMAT_UNKNOWN)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::Unsupported, {"d3d12_vertex_attribute_format_is_unsupported", 0, 0}})};
        inputElements.push_back({"TEXCOORD", attribute.location, format, attribute.binding, attribute.relative_offset,
                                 binding->divisor ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                                  : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                                 binding->divisor});
    }
    constexpr uint32_t shaderStages[]{VERNON_RUNTIME_PROVIDER_STAGE_VERTEX, VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT};
    for (uint32_t stage : shaderStages) {
        const uint32_t rootParameter = static_cast<uint32_t>(parameters.size());
        uint32_t valueSize = 0;
        PreparedPipeline::InlineRootBlock rootBlock;
        rootBlock.rootParameter = rootParameter;
        std::vector<const VernonRuntimeProviderBindingLayoutEntry *> inlineEntries;
        for (const auto &entry : layout.entries)
            if (entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM && entry.stage_mask == stage)
                inlineEntries.push_back(&entry);
        std::sort(inlineEntries.begin(), inlineEntries.end(), [](const auto *left, const auto *right) {
            // Push-constant block members follow entry argument order, while provider slots are name-sorted.
            return left->argument_index < right->argument_index;
        });
        for (const auto *entry : inlineEntries) {
            const uint32_t alignment = entry->element_alignment;
            if (!entry->element_size || entry->element_size % sizeof(uint32_t) != 0 || alignment < sizeof(uint32_t) ||
                (alignment & (alignment - 1)) != 0)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"d3d12_inline_uniform_layout_is_invalid", 0, 0}})};
            if (valueSize > UINT32_MAX - (alignment - 1))
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"d3d12_root_constant_layout_size_overflow", 0, 0}})};
            valueSize = (valueSize + alignment - 1) & ~(alignment - 1);
            rootBlock.members.push_back({entry->slot, valueSize / sizeof(uint32_t), entry->element_size});
            if (entry->element_size > UINT32_MAX - valueSize)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"d3d12_root_constant_layout_size_overflow", 0, 0}})};
            valueSize += entry->element_size;
        }
        if (valueSize != 0) {
            auto &parameter = parameters.emplace_back();
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameter.Constants.ShaderRegister = ::vernon::runtime::kDirectXRootConstantRegister;
            parameter.Constants.RegisterSpace = ::vernon::runtime::kDirectXRootConstantRegisterSpace;
            parameter.Constants.Num32BitValues = (valueSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
            parameter.ShaderVisibility = stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ? D3D12_SHADER_VISIBILITY_VERTEX
                                                                                       : D3D12_SHADER_VISIBILITY_PIXEL;
            rootBlock.valueCount = parameter.Constants.Num32BitValues;
            pipeline->inlineRootBlocks.push_back(std::move(rootBlock));
        }
    }
    uint32_t rootDwords =
        static_cast<uint32_t>(pipeline->hasResourceTable) + static_cast<uint32_t>(pipeline->hasSamplerTable);
    for (const D3D12_ROOT_PARAMETER &parameter : parameters)
        if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS)
            rootDwords += parameter.Constants.Num32BitValues;
    if (rootDwords > 64)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::Unsupported,
                                              {"d3d12_graphics_root_signature_exceeds_the_64_dword_limit", 0, 0}})};
    const D3D12_ROOT_SIGNATURE_DESC root{static_cast<UINT>(parameters.size()), parameters.data(), 0, nullptr,
                                         D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    HRESULT result = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(result)) {
        releaseObject(errors);
        releaseObject(serialized);
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                              {"D3D12SerializeRootSignature", static_cast<uint64_t>(result), 0}})};
    }
    releaseObject(errors);
    result = pipeline->device->device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&pipeline->rootSignature));
    releaseObject(serialized);
    if (FAILED(result))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                  {"ID3D12Device::CreateRootSignature", static_cast<uint64_t>(result), 0}})};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC native{};
    native.pRootSignature = pipeline->rootSignature;
    native.VS = {vertex->artifact.data(), vertex->artifact.size()};
    native.PS = {fragment->artifact.data(), fragment->artifact.size()};
    if (descriptor.color_blend_count != descriptor.color_format_count ||
        (descriptor.color_blend_count && !descriptor.color_blends))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"d3d12_graphics_pipeline_blend_state_does_not_match_its_attachments", 0, 0}})};
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
            return RhiAdapterResult<void>{
                vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                  {"d3d12_graphics_pipeline_contains_an_invalid_blend_state", 0, 0}})};
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
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"d3d12_graphics_pipeline_contains_an_unsupported_rasterization_state", 0, 0}})};
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
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"d3d12_graphics_pipeline_contains_an_invalid_depth_stencil_state", 0, 0}})};
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
    result = pipeline->device->device->CreateGraphicsPipelineState(&native, IID_PPV_ARGS(&pipeline->pipeline));
    if (FAILED(result))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                  {"ID3D12Device::CreateGraphicsPipelineState", static_cast<uint64_t>(result), 0}})};
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
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> prepareGraphicsPipelineResult(VernonRuntimeRhiAdapter &adapter,
                                                     const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                                     PreparedLayout &layout, VernonRuntimeProviderObject *output) {
    try {
        return prepareGraphicsPipelineImplResult(adapter, descriptor, layout, output);
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                              {"d3d12_graphics_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> preparePipelineResult(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (descriptor && descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE && output && layout &&
        descriptor->shader_count == 2 && descriptor->shaders)
        return prepareGraphicsPipelineResult(adapter, *descriptor, *layout, output);
    auto *shader = descriptor && descriptor->shader_count == 1 && descriptor->shaders
                       ? fromHandle<PreparedShader>(descriptor->shaders[0])
                       : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || !layout || !shader)
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"d3d12_adapter_received_an_invalid_compute_pipeline_descriptor", 0, 0}})};
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    pipeline->owner = &adapter;
    adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
    pipeline->device = &directX12Device(adapter);
    std::vector<D3D12_DESCRIPTOR_RANGE> resourceRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE> samplerRanges;
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    try {
        parameters.reserve(2);
        auto tables =
            appendDescriptorTables(layout->entries, true, resourceRanges, samplerRanges, parameters, *pipeline);
        if (!tables)
            return tables;
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_pipeline_preparation_ran_out_of_memory", 0, 0}})};
    }
    D3D12_ROOT_SIGNATURE_DESC root{};
    root.NumParameters = static_cast<UINT>(parameters.size());
    root.pParameters = parameters.empty() ? nullptr : parameters.data();
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    HRESULT result = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(result)) {
        releaseObject(errors);
        releaseObject(serialized);
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                              {"D3D12SerializeRootSignature", static_cast<uint64_t>(result), 0}})};
    }
    releaseObject(errors);
    result = pipeline->device->device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&pipeline->rootSignature));
    releaseObject(serialized);
    if (FAILED(result))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                  {"ID3D12Device::CreateRootSignature", static_cast<uint64_t>(result), 0}})};
    D3D12_COMPUTE_PIPELINE_STATE_DESC native{};
    native.pRootSignature = pipeline->rootSignature;
    native.CS = {shader->artifact.data(), shader->artifact.size()};
    result = pipeline->device->device->CreateComputePipelineState(&native, IID_PPV_ARGS(&pipeline->pipeline));
    if (FAILED(result))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                  {"ID3D12Device::CreateComputePipelineState", static_cast<uint64_t>(result), 0}})};
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> initializeBindingsResult(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"d3d12_binding_values_do_not_match_the_prepared_layout", 0, 0}})};
    std::vector<size_t> valueIndices(bindings.slots.size());
    std::vector<uint8_t> seenSlots(bindings.slots.size());
    std::vector<uint64_t> resolvedValues(bindings.slots.size());
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || seenSlots[found->second])
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"d3d12_binding_slot_is_invalid_or_duplicated", 0, 0}})};
        seenSlots[found->second] = 1;
        valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[valueIndices[index]];
        if (value->kind != slot.layout.kind)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"d3d12_binding_slot_or_kind_is_invalid", 0, 0}})};
        if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
            if (!value->payload.inline_value.data || value->payload.inline_value.size != slot.inlineStorage.size())
                return RhiAdapterResult<void>{vernon::err(
                    vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                          {"d3d12_inline_or_uniform_buffer_binding_size_is_invalid", 0, 0}})};
            resolvedValues[index] = 0;
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto &resource = value->payload.buffer.resource;
            auto resolved = resolveRhiResource(adapter, resource);
            if (!resolved)
                return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
            const uint64_t native = std::move(resolved).value();
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(native);
            if ((resource.identity & kRhiResourceKindMask) != kRhiBufferResource || !buffer || !buffer->resource ||
                resource.offset > resource.size || slot.layout.element_size > resource.size - resource.offset)
                return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                    vernon::ProviderErrorCode::InvalidArgument, {"d3d12_storage_binding_is_invalid", 0, 0}})};
            resolvedValues[index] = native;
        } else {
            const bool defaultSampler = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                        (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            if (defaultSampler)
                resolvedValues[index] = 0;
            else {
                const auto *resource = providerBindingResource(*value);
                const bool image = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                   slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
                const uint64_t expectedKind =
                    slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? kRhiSamplerResource : kRhiBufferResource;
                if (!resource)
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                          {"d3d12_graphics_resource_binding_is_invalid", 0, 0}})};
                auto resolved = resolveRhiResource(adapter, *resource);
                if (!resolved)
                    return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
                const uint64_t native = std::move(resolved).value();
                if ((image ? !isRhiImageReference(*resource)
                           : (resource->identity & kRhiResourceKindMask) != expectedKind) ||
                    (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && value->payload.buffer.stride == 0))
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                          {"d3d12_graphics_resource_binding_is_invalid", 0, 0}})};
                resolvedValues[index] = native;
            }
        }
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto &value = values[valueIndices[index]];
        const bool defaultSampler = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                    (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
        const auto *bindingResource = providerBindingResource(value);
        const VernonRuntimeProviderResourceReference resource =
            defaultSampler || !bindingResource ? VernonRuntimeProviderResourceReference{} : *bindingResource;
        slot.flags = value.flags;
        slot.resourceReference = {};
        if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
            std::memcpy(slot.inlineStorage.data(), value.payload.inline_value.data, value.payload.inline_value.size);
            slot.resource = slot.inlineResource.resource;
            slot.offset = 0;
            slot.size = slot.inlineStorage.size();
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(resolvedValues[index]);
            slot.resource = buffer->resource;
            slot.opaqueResource = buffer;
            slot.resourceReference = value.payload.buffer.resource;
            slot.offset = value.payload.buffer.resource.offset;
            slot.size = value.payload.buffer.resource.size - value.payload.buffer.resource.offset;
        } else {
            slot.opaqueResource = reinterpret_cast<void *>(resolvedValues[index]);
            if (!slot.opaqueResource) {
                slot.offset = 0;
                slot.size = 0;
                slot.stride = 0;
                continue;
            }
            slot.resourceReference = resource;
            slot.offset = resource.offset;
            slot.size = resource.size;
            slot.stride = value.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER ? value.payload.buffer.stride : 0;
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
                slot.resource = static_cast<rhi::directx12::Buffer *>(slot.opaqueResource)->resource;
        }
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> retainBindingResources(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              const PreparedBindingSet *bindings) {
    if (!bindings)
        return RhiAdapterResult<void>{vernon::ok()};
    for (const auto &slot : bindings->slots) {
        if (!slot.resourceReference.resource.value)
            continue;
        auto retained = retainCommandResource(adapter, encoder, slot.resourceReference);
        if (!retained)
            return retained;
    }
    return RhiAdapterResult<void>{vernon::ok()};
}

void destroyBindingSetImpl(PreparedBindingSet &bindings) {
    for (auto &slot : bindings.slots)
        bindings.device->destroyBuffer(slot.inlineResource);
}

RhiAdapterResult<void> createBindingsResult(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                                            VernonRuntimeProviderObject *output) {
    std::string backendDiagnostic;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"d3d12_adapter_received_an_invalid_binding_set_descriptor", 0, 0}})};
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        bindings->device = &directX12Device(adapter);
        bindings->slots.resize(layout->entries.size());
        bindings->slotIndices.reserve(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            auto &slot = bindings->slots[index];
            slot.layout = layout->entries[index];
            if (!bindings->slotIndices.emplace(slot.layout.slot, index).second) {
                destroyBindingSetImpl(*bindings);
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"d3d12_binding_layout_contains_duplicate_slots", 0, 0}})};
            }
            if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                slot.inlineStorage.resize(slot.layout.element_size);
                const bool needsInlineResource = slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                                 slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                                 slot.layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                if (needsInlineResource) {
                    const size_t resourceSize =
                        slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                            ? (static_cast<size_t>(slot.layout.element_size) + 255u) & ~size_t{255u}
                            : slot.layout.element_size;
                    if (!bindings->device->createBuffer(slot.inlineResource, resourceSize,
                                                        slot.layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER,
                                                        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                                        backendDiagnostic)) {
                        destroyBindingSetImpl(*bindings);
                        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                            vernon::ProviderErrorCode::BackendFailure,
                            {"d3d12_create_inline_binding_buffer", resourceSize, slot.layout.slot}})};
                    }
                }
            }
            const std::optional<D3D12_DESCRIPTOR_RANGE_TYPE> range =
                descriptorRangeType(slot.layout, slot.layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE);
            bindings->resourceDescriptorCount += range && *range != D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            bindings->samplerDescriptorCount += range == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        }
        auto status = initializeBindingsResult(adapter, *bindings, descriptor->values, descriptor->value_count);
        if (!status) {
            destroyBindingSetImpl(*bindings);
            return status;
        }
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return RhiAdapterResult<void>{vernon::ok()};
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_binding_preparation_ran_out_of_memory", 0, 0}})};
    }
}

RhiAdapterResult<void> encodeDispatchResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    std::string backendDiagnostic;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline ||
        (bindings == nullptr && descriptor->bindings.value != 0))
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                              {"d3d12_adapter_received_an_invalid_dispatch_descriptor", 0, 0}})};
    auto &device = *pipeline->device;
    auto nativeCommand = nativeCommandEncoder(adapter, commandEncoder);
    if (!nativeCommand)
        return RhiAdapterResult<void>{vernon::err(std::move(nativeCommand).error())};
    auto *commands =
        reinterpret_cast<ID3D12GraphicsCommandList *>(static_cast<uintptr_t>(std::move(nativeCommand).value()));
    auto rendering = commandEncoderRendering(adapter, commandEncoder);
    if (!rendering)
        return RhiAdapterResult<void>{vernon::err(std::move(rendering).error())};
    if (std::move(rendering).value())
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"d3d12_dispatch_command_encoder_is_invalid", 0, 0}})};
    commands->SetPipelineState(pipeline->pipeline);
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_dispatch_could_not_retain_provider_objects", 0, 0}})};
    std::unique_lock<std::mutex> bindingGuard;
    if (bindings)
        bindingGuard = std::unique_lock<std::mutex>(bindings->mutex);
    auto acquiredTables = acquireDescriptorTables(device, bindings, bindings ? &bindings->dispatchDescriptors : nullptr,
                                                  commandEncoder.value, backendDiagnostic);
    if (!acquiredTables)
        return RhiAdapterResult<void>{vernon::err(std::move(acquiredTables).error())};
    const AcquiredDescriptorTables acquired = std::move(acquiredTables).value();
    if (bindings) {
        if (!retainBindingResources(adapter, commandEncoder, bindings))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"d3d12_dispatch_could_not_retain_its_resources", 0, 0}})};
        DescriptorWriter writer{adapter,         commandEncoder,       commands,           device,
                                acquired.reused, acquired.resourceCpu, acquired.samplerCpu};
        for (auto &slot : bindings->slots) {
            if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                if (auto written = writeInlineBufferDescriptor(writer, slot, backendDiagnostic); !written)
                    return written;
            } else if (std::optional<RhiAdapterResult<void>> written = writeResourceDescriptor(writer, slot)) {
                if (!*written)
                    return std::move(*written);
            } else {
                return RhiAdapterResult<void>{
                    vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                      {"d3d12_dispatch_contains_an_unsupported_binding",
                                                       slot.layout.slot, static_cast<uint32_t>(slot.layout.kind)}})};
            }
        }
        bindings->dispatchDescriptors = acquired.tables;
    }
    bindDescriptorTables(commands, *pipeline, acquired.tables);
    commands->Dispatch(descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"d3d12_dispatch_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> encodeDrawResult(void *data, VernonRuntimeProviderObject commandEncoder,
                                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    std::string backendDiagnostic;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    const uint64_t invalidDescriptor =
        (!validCommonDrawDescriptor(descriptor) ? uint64_t{1} : 0) | (!pipeline ? uint64_t{2} : 0) |
        (pipeline && !pipeline->graphics ? uint64_t{4} : 0) | (pipeline && !pipeline->pipeline ? uint64_t{8} : 0) |
        (descriptor && descriptor->bindings.value != 0 && !bindings ? uint64_t{16} : 0);
    if (invalidDescriptor)
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"d3d12_adapter_received_an_invalid_draw_descriptor", invalidDescriptor, 0}})};
    auto &device = *pipeline->device;
    auto nativeCommand = nativeCommandEncoder(adapter, commandEncoder);
    if (!nativeCommand)
        return RhiAdapterResult<void>{vernon::err(std::move(nativeCommand).error())};
    auto *commands =
        reinterpret_cast<ID3D12GraphicsCommandList *>(static_cast<uintptr_t>(std::move(nativeCommand).value()));
    auto opened = openCommandRendering(adapter, commandEncoder);
    if (!opened)
        return RhiAdapterResult<void>{vernon::err(std::move(opened).error())};
    commands->SetPipelineState(pipeline->pipeline);
    commands->OMSetStencilRef(descriptor->stencil_reference);
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::BackendFailure, {"d3d12_draw_could_not_retain_provider_objects", 0, 0}})};
    std::unique_lock<std::mutex> bindingGuard;
    if (bindings)
        bindingGuard = std::unique_lock<std::mutex>(bindings->mutex);
    auto acquiredTables = acquireDescriptorTables(device, bindings, bindings ? &bindings->drawDescriptors : nullptr,
                                                  commandEncoder.value, backendDiagnostic);
    if (!acquiredTables)
        return RhiAdapterResult<void>{vernon::err(std::move(acquiredTables).error())};
    const AcquiredDescriptorTables acquired = std::move(acquiredTables).value();
    bindDescriptorTables(commands, *pipeline, acquired.tables);
    if (bindings) {
        if (!retainBindingResources(adapter, commandEncoder, bindings))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"d3d12_draw_could_not_retain_its_bindings", 0, 0}})};
        std::array<uint32_t, 64> rootValues{};
        for (const auto &block : pipeline->inlineRootBlocks) {
            std::fill_n(rootValues.begin(), block.valueCount, uint32_t{0});
            for (const auto &member : block.members) {
                const auto found = bindings->slotIndices.find(member.slot);
                if (found == bindings->slotIndices.end() || bindings->slots[found->second].inlineStorage.empty())
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                          {"d3d12_draw_contains_an_invalid_inline_uniform", 0, 0}})};
                const auto &slot = bindings->slots[found->second];
                std::memcpy(reinterpret_cast<uint8_t *>(rootValues.data()) +
                                member.destinationOffset * sizeof(uint32_t),
                            slot.inlineStorage.data(), member.size);
            }
            commands->SetGraphicsRoot32BitConstants(block.rootParameter, block.valueCount, rootValues.data(), 0);
        }
        DescriptorWriter writer{adapter,         commandEncoder,       commands,           device,
                                acquired.reused, acquired.resourceCpu, acquired.samplerCpu};
        for (auto &slot : bindings->slots) {
            // Graphics inline values travel as root constants above and own no descriptor.
            if (slot.layout.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
                if (auto written = writeInlineBufferDescriptor(writer, slot, backendDiagnostic); !written)
                    return written;
            } else if (std::optional<RhiAdapterResult<void>> written = writeResourceDescriptor(writer, slot)) {
                if (!*written)
                    return std::move(*written);
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                auto *buffer = static_cast<rhi::directx12::Buffer *>(slot.opaqueResource);
                if (!buffer || !buffer->resource)
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                          {"d3d12_draw_contains_an_invalid_vertex_buffer", 0, 0}})};
                if (!transition(adapter, commandEncoder, commands, *buffer,
                                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER))
                    return RhiAdapterResult<void>{
                        vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::BackendFailure,
                                                          {"d3d12_draw_could_not_track_vertex_buffer_state", 0, 0}})};
                const D3D12_VERTEX_BUFFER_VIEW view{slot.resource->GetGPUVirtualAddress() + slot.offset,
                                                    static_cast<UINT>(slot.size), slot.stride};
                commands->IASetVertexBuffers(slot.layout.binding, 1, &view);
            }
        }
        bindings->drawDescriptors = acquired.tables;
    }
    const D3D12_VIEWPORT viewport{static_cast<float>(descriptor->viewport[0]),
                                  static_cast<float>(descriptor->viewport[1]),
                                  static_cast<float>(descriptor->viewport[2]),
                                  static_cast<float>(descriptor->viewport[3]),
                                  0,
                                  1};
    const D3D12_RECT scissor{static_cast<LONG>(descriptor->scissor[0]), static_cast<LONG>(descriptor->scissor[1]),
                             static_cast<LONG>(descriptor->scissor[0] + descriptor->scissor[2]),
                             static_cast<LONG>(descriptor->scissor[1] + descriptor->scissor[3])};
    commands->RSSetViewports(1, &viewport);
    commands->RSSetScissorRects(1, &scissor);
    const D3D_PRIMITIVE_TOPOLOGY topology = descriptor->topology == 1   ? D3D_PRIMITIVE_TOPOLOGY_LINELIST
                                            : descriptor->topology == 2 ? D3D_PRIMITIVE_TOPOLOGY_POINTLIST
                                                                        : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    commands->IASetPrimitiveTopology(topology);
    // SV_InstanceID and SV_VertexID exclude the draw offsets, so nonzero offsets would change `instance_index` and
    // `vertex_index`.
    if (descriptor->first_instance != 0 || descriptor->base_vertex != 0)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::Unsupported, {"d3d12_draw_base_instance_or_base_vertex_is_unsupported", 0, 0}})};
    if (descriptor->index_count != 0) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"d3d12_draw_could_not_retain_its_index_buffer", 0, 0}})};
        auto resolved = resolveCommandRhiResource(adapter, commandEncoder, descriptor->index_buffer);
        if (!resolved)
            return RhiAdapterResult<void>{vernon::err(std::move(resolved).error())};
        auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(static_cast<uintptr_t>(std::move(resolved).value()));
        if (!buffer || !buffer->resource)
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::InvalidArgument, {"d3d12_draw_contains_an_invalid_index_buffer", 0, 0}})};
        if (!transition(adapter, commandEncoder, commands, *buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER))
            return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
                vernon::ProviderErrorCode::BackendFailure, {"d3d12_draw_could_not_track_index_buffer_state", 0, 0}})};
        const D3D12_INDEX_BUFFER_VIEW indexView{buffer->resource->GetGPUVirtualAddress() +
                                                    descriptor->index_buffer.offset,
                                                static_cast<UINT>(descriptor->index_buffer.size), DXGI_FORMAT_R32_UINT};
        commands->IASetIndexBuffer(&indexView);
        commands->DrawIndexedInstanced(descriptor->index_count, descriptor->instance_count, 0, descriptor->base_vertex,
                                       descriptor->first_instance);
    } else {
        commands->DrawInstanced(descriptor->vertex_count, descriptor->instance_count, 0, descriptor->first_instance);
    }
    if (!recordProviderCommand(adapter, commandEncoder, true))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"d3d12_draw_command_encoder_state_changed", 0, 0}})};
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> destroyShaderResult(void *, VernonRuntimeProviderObject handle) {
    delete fromHandle<PreparedShader>(handle);
    return RhiAdapterResult<void>{vernon::ok()};
}
RhiAdapterResult<void> destroyLayoutResult(void *, VernonRuntimeProviderObject handle) {
    delete fromHandle<PreparedLayout>(handle);
    return RhiAdapterResult<void>{vernon::ok()};
}
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete pipeline;
}
RhiAdapterResult<void> destroyPipelineResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (!bindings || bindings->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    destroyBindingSetImpl(*bindings);
    delete bindings;
}
RhiAdapterResult<void> destroyBindingsResult(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
    return RhiAdapterResult<void>{vernon::ok()};
}

void destroyShader(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyShaderResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "D3D12 shader destruction failed");
}
void destroyLayout(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyLayoutResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "D3D12 layout destruction failed");
}
void destroyPipeline(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyPipelineResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "D3D12 pipeline destruction failed");
}
void destroyBindings(void *data, VernonRuntimeProviderObject handle) {
    auto result = destroyBindingsResult(data, handle);
    if (!result)
        recordProviderError(*static_cast<VernonRuntimeRhiAdapter *>(data), std::move(result).error(),
                            "D3D12 binding-set destruction failed");
}

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, prepareShaderResult(data, descriptor, output), "D3D12 shader preparation failed");
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
VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, createBindingsResult(data, descriptor, output), "D3D12 binding-set creation failed");
}
VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject encoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDispatchResult(data, encoder, descriptor), "D3D12 dispatch encoding failed");
}
VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject encoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, encodeDrawResult(data, encoder, descriptor), "D3D12 draw encoding failed");
}

} // namespace

void initializeDirectX12Provider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PROGRAM_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainRhiResourceCallback;
    adapter.provider.release_resource = releaseRhiResourceCallback;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_binding_set = createBindings;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
    adapter.provider.begin_rendering = beginRenderingCallback;
    adapter.provider.end_rendering = endRenderingCallback;
    adapter.provider.destroy_shader = destroyShader;
    adapter.provider.destroy_pipeline_layout = destroyLayout;
    adapter.provider.destroy_pipeline = destroyPipeline;
    adapter.provider.destroy_binding_set = destroyBindings;
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
