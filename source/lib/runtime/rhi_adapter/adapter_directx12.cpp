#include "adapter_common.h"
#include "adapter_directx12_test_hooks.h"
#include "rhi/rhi_internal.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime/vertex_attribute_capabilities.h"

#if defined(VERNON_HAS_DIRECTX12_RHI)

#include "rhi/directx12_backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

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
VernonStatus synchronizeBackend(void *state, std::string &error) noexcept {
    try {
        return static_cast<DirectX12AdapterState *>(state)->device->synchronize(error) ? VERNON_STATUS_OK
                                                                                       : VERNON_STATUS_INTERNAL_ERROR;
    } catch (...) {
        setBackendError(error, "DirectX 12 synchronization threw an exception");
        return VERNON_STATUS_INTERNAL_ERROR;
    }
}
uint64_t resourceIdentity(const void *state) noexcept {
    return reinterpret_cast<uintptr_t>(static_cast<const DirectX12AdapterState *>(state)->device);
}
void invalidateBackend(void *) noexcept {}

const RhiAdapterBackendOps backendOps{destroyBackend, synchronizeBackend, resourceIdentity, invalidateBackend};

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
    std::vector<size_t> valueIndices;
    std::vector<uint8_t> seenSlots;
    std::vector<uint64_t> resolvedValues;
    uint32_t resourceDescriptorCount{};
    uint32_t samplerDescriptorCount{};
    uint64_t descriptorContentRevision{1};
    uint64_t descriptorEncoder{};
    uint64_t descriptorRevision{};
    ID3D12DescriptorHeap *resourceHeap{};
    ID3D12DescriptorHeap *samplerHeap{};
    D3D12_GPU_DESCRIPTOR_HANDLE resourceGpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE samplerGpu{};
    uint64_t computeDescriptorEncoder{};
    uint64_t computeDescriptorRevision{};
    ID3D12DescriptorHeap *computeResourceHeap{};
    D3D12_GPU_DESCRIPTOR_HANDLE computeResourceGpu{};
    std::mutex mutex;
};

void releaseCommandBindings(void *context, uint64_t);
void releaseCommandPipeline(void *context, uint64_t);

void restoreBufferState(void *context, uint64_t state) {
    static_cast<rhi::directx12::Buffer *>(context)->state = static_cast<D3D12_RESOURCE_STATES>(state);
}

void restoreImageState(void *context, uint64_t state) {
    static_cast<rhi::directx12::Image *>(context)->state = static_cast<D3D12_RESOURCE_STATES>(state);
}

bool transition(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                ID3D12GraphicsCommandList *commands, rhi::directx12::Buffer &buffer, D3D12_RESOURCE_STATES target) {
    if (!deferCommandRollback(adapter, encoder, &buffer, static_cast<uint64_t>(buffer.state), restoreBufferState))
        return false;
    rhi::directx12::transition(commands, buffer.resource, buffer.state, target);
    return true;
}

bool transition(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                ID3D12GraphicsCommandList *commands, rhi::directx12::Image &image, D3D12_RESOURCE_STATES target) {
    if (!deferCommandRollback(adapter, encoder, &image, static_cast<uint64_t>(image.state), restoreImageState))
        return false;
    rhi::directx12::transition(commands, image.resource, image.state, target);
    return true;
}

bool retainCommandObjects(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                          PreparedPipeline &pipeline, PreparedBindingSet *bindings) {
    pipeline.references.fetch_add(1, std::memory_order_relaxed);
    if (!deferCommandCleanup(adapter, encoder, &pipeline, 0, releaseCommandPipeline)) {
        releaseCommandPipeline(&pipeline, 0);
        return false;
    }
    if (bindings) {
        bindings->references.fetch_add(1, std::memory_order_relaxed);
        if (!deferCommandCleanup(adapter, encoder, bindings, 0, releaseCommandBindings)) {
            releaseCommandBindings(bindings, 0);
            return false;
        }
    }
    return true;
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

VernonStatus prepareShader(void *data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !descriptor->data ||
        descriptor->size < 4 || descriptor->stage == 0 || std::memcmp(descriptor->data, "DXBC", 4) != 0 ||
        descriptor->format.size != 4 || std::memcmp(descriptor->format.data, "dxil", 4) != 0)
        return fail(adapter, "D3D12 adapter received an invalid DXIL shader descriptor");
    try {
        auto shader = std::make_unique<PreparedShader>();
        shader->stage = descriptor->stage;
        const auto *bytes = static_cast<const uint8_t *>(descriptor->data);
        shader->artifact.assign(bytes, bytes + descriptor->size);
        *output = toHandle(shader.release());
        adapter.shaderPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "D3D12 shader preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus prepareLayout(void *data, const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                           VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        (descriptor->binding_count != 0 && !descriptor->bindings) ||
        (descriptor->vertex_attribute_count != 0 && !descriptor->vertex_attributes))
        return fail(adapter, "D3D12 adapter received an invalid pipeline layout");
    try {
        auto layout = std::make_unique<PreparedLayout>();
        layout->entries.assign(descriptor->bindings, descriptor->bindings + descriptor->binding_count);
        for (const auto &entry : layout->entries) {
            const bool compute = (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                  entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                                  entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) &&
                                 entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE && entry.element_size != 0;
            const bool graphicsResource = (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
                                           entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                           entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) &&
                                          entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE &&
                                          (entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                           entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                          entry.binding != UINT32_MAX;
            const bool graphicsInline = entry.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                                        entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                        (entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                         entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                        entry.element_size != 0 && entry.element_size % sizeof(uint32_t) == 0;
            const bool graphicsUniformBuffer = entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER &&
                                               entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                               (entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                                entry.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT) &&
                                               entry.binding != UINT32_MAX && entry.element_size != 0;
            const bool vertex = entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
                                entry.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT &&
                                entry.binding < D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT && entry.element_size != 0;
            if ((!compute && !graphicsResource && !graphicsInline && !graphicsUniformBuffer && !vertex) ||
                entry.array_count != 1)
                return fail(adapter, "D3D12 layout contains an unsupported binding", VERNON_STATUS_UNSUPPORTED_TARGET);
        }
        for (size_t index = 0; index < descriptor->vertex_attribute_count; ++index) {
            const auto &attribute = descriptor->vertex_attributes[index];
            const bool bindingExists =
                std::any_of(layout->entries.begin(), layout->entries.end(), [&](const auto &entry) {
                    return entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.binding == attribute.binding;
                });
            std::string capabilityDiagnostic;
            if (!bindingExists)
                return fail(adapter, "D3D12 vertex attribute references an unknown binding",
                            VERNON_STATUS_UNSUPPORTED_TARGET);
            if (!validateVertexAttributeCapability(VertexAttributeBackend::DirectX12, attribute,
                                                   D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, false,
                                                   capabilityDiagnostic))
                return fail(adapter, std::move(capabilityDiagnostic), VERNON_STATUS_UNSUPPORTED_TARGET);
            layout->vertexAttributes.push_back(attribute);
        }
        *output = toHandle(layout.release());
        adapter.layoutPreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "D3D12 layout preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
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

VernonStatus prepareGraphicsPipelineImpl(VernonRuntimeRhiAdapter &adapter,
                                         const VernonRuntimeProviderPipelineDescriptor &descriptor,
                                         PreparedLayout &layout, VernonRuntimeProviderObject *output) {
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return fail(adapter, "D3D12 graphics pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    pipeline->owner = &adapter;
    adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
    pipeline->device = &directX12Device(adapter);
    pipeline->graphics = true;
    if (descriptor.color_format_count == 0) {
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    }
    if (descriptor.color_format_count > 8 || descriptor.sample_count != 1 ||
        (descriptor.depth_stencil_format && descriptor.depth_stencil_format != DXGI_FORMAT_D32_FLOAT &&
         descriptor.depth_stencil_format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT))
        return fail(adapter, "D3D12 graphics pipeline uses an unsupported attachment configuration");
    PreparedShader *vertex = nullptr;
    PreparedShader *fragment = nullptr;
    for (size_t index = 0; index < descriptor.shader_count; ++index) {
        auto *shader = fromHandle<PreparedShader>(descriptor.shaders[index]);
        if (!shader)
            return fail(adapter, "D3D12 graphics pipeline contains an invalid shader");
        if (shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX)
            vertex = shader;
        else if (shader->stage == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT)
            fragment = shader;
    }
    if (!vertex || !fragment)
        return fail(adapter, "D3D12 graphics pipeline requires vertex and fragment shaders");

    std::vector<D3D12_DESCRIPTOR_RANGE> resourceRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE> samplerRanges;
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements;
    for (const auto &entry : layout.entries) {
        if (entry.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER)
            resourceRanges.push_back(
                {D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, entry.binding, entry.set, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND});
        else if (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER)
            resourceRanges.push_back(
                {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, entry.binding, entry.set, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND});
        else if (entry.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
            resourceRanges.push_back(
                {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, entry.binding, entry.set, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND});
        else if (entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE)
            resourceRanges.push_back(
                {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, entry.binding, entry.set, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND});
        else if (entry.kind == VERNON_RUNTIME_PROVIDER_SAMPLER)
            samplerRanges.push_back({D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, entry.binding, entry.set,
                                     D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND});
    }
    for (const VernonRuntimeProviderVertexAttribute &attribute : layout.vertexAttributes) {
        const auto binding = std::find_if(layout.entries.begin(), layout.entries.end(), [&](const auto &entry) {
            return entry.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && entry.binding == attribute.binding;
        });
        const DXGI_FORMAT format = vertexFormat(attribute.dtype, attribute.component_count);
        if (binding == layout.entries.end() || format == DXGI_FORMAT_UNKNOWN)
            return fail(adapter, "D3D12 vertex attribute format is unsupported", VERNON_STATUS_UNSUPPORTED_TARGET);
        inputElements.push_back({"TEXCOORD", attribute.location, format, attribute.binding, attribute.relative_offset,
                                 binding->divisor ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                                  : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                                 binding->divisor});
    }
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    parameters.reserve(2 + layout.entries.size());
    if (!resourceRanges.empty()) {
        pipeline->hasResourceTable = true;
        auto &parameter = parameters.emplace_back();
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {static_cast<UINT>(resourceRanges.size()), resourceRanges.data()};
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    if (!samplerRanges.empty()) {
        pipeline->hasSamplerTable = true;
        auto &parameter = parameters.emplace_back();
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {static_cast<UINT>(samplerRanges.size()), samplerRanges.data()};
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
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
            // HLSL constant-buffer members follow entry argument order, while provider slots are name-sorted.
            return left->argument_index < right->argument_index;
        });
        for (const auto *entry : inlineEntries) {
            if (!entry->element_size || entry->element_size % sizeof(uint32_t) != 0)
                return fail(adapter, "D3D12 inline uniform size is invalid");
            const bool registerAggregate = entry->element_size > 16;
            if (registerAggregate || valueSize % 16 + entry->element_size > 16) {
                if (valueSize > UINT32_MAX - 15)
                    return fail(adapter, "D3D12 root-constant layout size overflow");
                valueSize = (valueSize + 15) & ~uint32_t{15};
            }
            rootBlock.members.push_back({entry->slot, valueSize / sizeof(uint32_t), entry->element_size});
            if (entry->element_size > UINT32_MAX - valueSize)
                return fail(adapter, "D3D12 root-constant layout size overflow");
            valueSize += entry->element_size;
        }
        if (valueSize != 0) {
            auto &parameter = parameters.emplace_back();
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameter.Constants.ShaderRegister = 0;
            parameter.Constants.RegisterSpace = 0;
            parameter.Constants.Num32BitValues = (valueSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
            parameter.ShaderVisibility = stage == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ? D3D12_SHADER_VISIBILITY_VERTEX
                                                                                       : D3D12_SHADER_VISIBILITY_PIXEL;
            rootBlock.valueCount = parameter.Constants.Num32BitValues;
            pipeline->inlineRootBlocks.push_back(std::move(rootBlock));
        }
    }
    uint32_t rootDwords =
        static_cast<uint32_t>(!resourceRanges.empty()) + static_cast<uint32_t>(!samplerRanges.empty());
    for (const D3D12_ROOT_PARAMETER &parameter : parameters)
        if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS)
            rootDwords += parameter.Constants.Num32BitValues;
    if (rootDwords > 64)
        return fail(adapter, "D3D12 graphics root signature exceeds the 64-DWORD limit",
                    VERNON_STATUS_UNSUPPORTED_TARGET);
    const D3D12_ROOT_SIGNATURE_DESC root{static_cast<UINT>(parameters.size()), parameters.data(), 0, nullptr,
                                         D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    HRESULT result = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(result)) {
        const std::string message =
            errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize())
                   : "D3D12 graphics root-signature serialization failed";
        releaseObject(errors);
        releaseObject(serialized);
        return fail(adapter, message, VERNON_STATUS_INTERNAL_ERROR);
    }
    releaseObject(errors);
    result = pipeline->device->device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&pipeline->rootSignature));
    releaseObject(serialized);
    if (FAILED(result))
        return fail(adapter, "ID3D12Device::CreateRootSignature failed", VERNON_STATUS_INTERNAL_ERROR);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC native{};
    native.pRootSignature = pipeline->rootSignature;
    native.VS = {vertex->artifact.data(), vertex->artifact.size()};
    native.PS = {fragment->artifact.data(), fragment->artifact.size()};
    if (descriptor.color_blend_count != descriptor.color_format_count ||
        (descriptor.color_blend_count && !descriptor.color_blends))
        return fail(adapter, "D3D12 graphics pipeline blend state does not match its attachments");
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
            return fail(adapter, "D3D12 graphics pipeline contains an invalid blend state");
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
        return fail(adapter, "D3D12 graphics pipeline contains an unsupported rasterization state");
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
        return fail(adapter, "D3D12 graphics pipeline contains an invalid depth/stencil state");
    native.DepthStencilState.DepthEnable = depthStencil.depth_test;
    native.DepthStencilState.DepthWriteMask =
        depthStencil.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    native.DepthStencilState.DepthFunc =
        compareOperation(depthStencil.depth_test ? depthStencil.depth_compare : VERNON_RHI_COMPARE_ALWAYS);
    native.DepthStencilState.StencilEnable = depthStencil.stencil_test;
    native.DepthStencilState.StencilReadMask = static_cast<UINT8>(depthStencil.stencil_read_mask);
    native.DepthStencilState.StencilWriteMask = static_cast<UINT8>(depthStencil.stencil_write_mask);
    const auto stencilFace = [](const VernonRuntimeProviderStencilFaceState &source) {
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
        return fail(adapter, "ID3D12Device::CreateGraphicsPipelineState failed", VERNON_STATUS_INTERNAL_ERROR);
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
    return VERNON_STATUS_OK;
}

VernonStatus prepareGraphicsPipeline(VernonRuntimeRhiAdapter &adapter,
                                     const VernonRuntimeProviderPipelineDescriptor &descriptor, PreparedLayout &layout,
                                     VernonRuntimeProviderObject *output) {
    try {
        return prepareGraphicsPipelineImpl(adapter, descriptor, layout, output);
    } catch (const std::bad_alloc &) {
        return fail(adapter, "D3D12 graphics pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus preparePipeline(void *data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                             VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (descriptor && descriptor->kind == VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE && output && layout &&
        descriptor->shader_count == 2 && descriptor->shaders)
        return prepareGraphicsPipeline(adapter, *descriptor, *layout, output);
    auto *shader = descriptor && descriptor->shader_count == 1 && descriptor->shaders
                       ? fromHandle<PreparedShader>(descriptor->shaders[0])
                       : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->kind != VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE || !layout || !shader)
        return fail(adapter, "D3D12 adapter received an invalid compute pipeline descriptor");
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    try {
        ranges.reserve(layout->entries.size());
        for (const auto &entry : layout->entries)
            ranges.push_back(
                {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, entry.binding, entry.set, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND});
    } catch (const std::bad_alloc &) {
        return fail(adapter, "D3D12 pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable = {static_cast<UINT>(ranges.size()), ranges.data()};
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC root{};
    root.NumParameters = ranges.empty() ? 0u : 1u;
    root.pParameters = ranges.empty() ? nullptr : &parameter;
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    HRESULT result = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(result)) {
        const std::string message =
            errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize())
                   : "D3D12 root-signature serialization failed";
        releaseObject(errors);
        releaseObject(serialized);
        return fail(adapter, message, VERNON_STATUS_INTERNAL_ERROR);
    }
    releaseObject(errors);
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline) {
        releaseObject(serialized);
        return fail(adapter, "D3D12 pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
    pipeline->owner = &adapter;
    adapter.livePreparedPipelines.fetch_add(1, std::memory_order_relaxed);
    pipeline->device = &directX12Device(adapter);
    result = pipeline->device->device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&pipeline->rootSignature));
    releaseObject(serialized);
    if (FAILED(result))
        return fail(adapter, "ID3D12Device::CreateRootSignature failed", VERNON_STATUS_INTERNAL_ERROR);
    D3D12_COMPUTE_PIPELINE_STATE_DESC native{};
    native.pRootSignature = pipeline->rootSignature;
    native.CS = {shader->artifact.data(), shader->artifact.size()};
    result = pipeline->device->device->CreateComputePipelineState(&native, IID_PPV_ARGS(&pipeline->pipeline));
    if (FAILED(result))
        return fail(adapter, "ID3D12Device::CreateComputePipelineState failed", VERNON_STATUS_INTERNAL_ERROR);
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus retainResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return retainRhiResource(adapter, resource) ? VERNON_STATUS_OK
                                                : fail(adapter, "D3D12 adapter received a stale resource");
}

void releaseResource(void *data, VernonRuntimeProviderResourceReference resource) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    releaseRhiResource(adapter, resource);
}

VernonStatus updateBindingsImpl(VernonRuntimeRhiAdapter &adapter, PreparedBindingSet &bindings,
                                const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    if (valueCount != bindings.slots.size() || (valueCount != 0 && !values))
        return fail(adapter, "D3D12 binding values do not match the prepared layout");
    std::fill(bindings.seenSlots.begin(), bindings.seenSlots.end(), uint8_t{0});
    for (size_t index = 0; index < valueCount; ++index) {
        const auto found = bindings.slotIndices.find(values[index].slot);
        if (found == bindings.slotIndices.end() || bindings.seenSlots[found->second])
            return fail(adapter, "D3D12 binding slot is invalid or duplicated");
        bindings.seenSlots[found->second] = 1;
        bindings.valueIndices[found->second] = index;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto *value = &values[bindings.valueIndices[index]];
        if (value->kind != slot.layout.kind)
            return fail(adapter, "D3D12 binding slot or kind is invalid");
        if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
            if (!value->payload.inline_value.data || value->payload.inline_value.size != slot.inlineStorage.size())
                return fail(adapter, "D3D12 inline or uniform-buffer binding size is invalid");
            bindings.resolvedValues[index] = 0;
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto &resource = value->payload.buffer.resource;
            const uint64_t native = resolveRhiResource(adapter, resource);
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(native);
            if ((resource.identity & kRhiResourceKindMask) != kRhiBufferResource || !buffer || !buffer->resource ||
                resource.offset > resource.size || slot.layout.element_size > resource.size - resource.offset)
                return fail(adapter, "D3D12 storage binding is invalid");
            bindings.resolvedValues[index] = native;
        } else {
            const bool defaultSampler = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                        (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            if (defaultSampler)
                bindings.resolvedValues[index] = 0;
            else {
                const auto *resource = providerBindingResource(*value);
                const bool image = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                   slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
                const uint64_t expectedKind =
                    slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER ? kRhiSamplerResource : kRhiBufferResource;
                const uint64_t native = resource ? resolveRhiResource(adapter, *resource) : 0;
                if (!resource ||
                    (image ? !isRhiImageReference(*resource)
                           : (resource->identity & kRhiResourceKindMask) != expectedKind) ||
                    native == 0 ||
                    (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && value->payload.buffer.stride == 0))
                    return fail(adapter, "D3D12 graphics resource binding is invalid");
                bindings.resolvedValues[index] = native;
            }
        }
    }
    const auto sameResource = [](const VernonRuntimeProviderResourceReference &left,
                                 const VernonRuntimeProviderResourceReference &right) {
        return left.identity == right.identity && left.resource.value == right.resource.value &&
               left.offset == right.offset && left.size == right.size;
    };
    bool descriptorsChanged = false;
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        const auto &slot = bindings.slots[index];
        const auto &value = values[bindings.valueIndices[index]];
        const bool defaultSampler = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                    (value.flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
        const auto *bindingResource = providerBindingResource(value);
        const VernonRuntimeProviderResourceReference resource =
            defaultSampler || !bindingResource ? VernonRuntimeProviderResourceReference{} : *bindingResource;
        const bool slotChanged =
            slot.flags != value.flags || !sameResource(slot.resourceReference, resource) ||
            (value.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && slot.stride != value.payload.buffer.stride) ||
            (packedUniformBytes(value.kind, slot.layout.interface_kind) && value.payload.inline_value.data &&
             std::memcmp(slot.inlineStorage.data(), value.payload.inline_value.data, value.payload.inline_value.size));
        const bool graphicsRootConstant = slot.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
                                          slot.layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM &&
                                          (slot.layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_VERTEX ||
                                           slot.layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT);
        descriptorsChanged |=
            slotChanged && !graphicsRootConstant && slot.layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
    }
    for (size_t index = 0; index < bindings.slots.size(); ++index) {
        auto &slot = bindings.slots[index];
        const auto &value = values[bindings.valueIndices[index]];
        slot.flags = value.flags;
        slot.resourceReference = {};
        if (packedUniformBytes(slot.layout.kind, slot.layout.interface_kind)) {
            std::memcpy(slot.inlineStorage.data(), value.payload.inline_value.data, value.payload.inline_value.size);
            slot.resource = slot.inlineResource.resource;
            slot.offset = 0;
            slot.size = slot.inlineStorage.size();
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(bindings.resolvedValues[index]);
            slot.resource = buffer->resource;
            slot.opaqueResource = buffer;
            slot.resourceReference = value.payload.buffer.resource;
            slot.offset = value.payload.buffer.resource.offset;
            slot.size = value.payload.buffer.resource.size - value.payload.buffer.resource.offset;
        } else {
            slot.opaqueResource = reinterpret_cast<void *>(bindings.resolvedValues[index]);
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
    if (descriptorsChanged) {
        if (++bindings.descriptorContentRevision == 0)
            bindings.descriptorContentRevision = 1;
    }
    return VERNON_STATUS_OK;
}

bool retainBindingResources(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                            const PreparedBindingSet *bindings) {
    if (!bindings)
        return true;
    for (const auto &slot : bindings->slots)
        if (slot.resourceReference.resource.value && !retainCommandResource(adapter, encoder, slot.resourceReference))
            return false;
    return true;
}

void destroyBindingSetImpl(PreparedBindingSet &bindings) {
    for (auto &slot : bindings.slots)
        bindings.device->destroyBuffer(slot.inlineResource);
}

VernonStatus createBindings(void *data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                            VernonRuntimeProviderObject *output) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *layout = descriptor ? fromHandle<PreparedLayout>(descriptor->layout) : nullptr;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !layout)
        return fail(adapter, "D3D12 adapter received an invalid binding-set descriptor");
    try {
        auto bindings = std::make_unique<PreparedBindingSet>();
        bindings->device = &directX12Device(adapter);
        bindings->slots.resize(layout->entries.size());
        bindings->slotIndices.reserve(layout->entries.size());
        bindings->valueIndices.resize(layout->entries.size());
        bindings->seenSlots.resize(layout->entries.size());
        bindings->resolvedValues.resize(layout->entries.size());
        for (size_t index = 0; index < layout->entries.size(); ++index) {
            auto &slot = bindings->slots[index];
            slot.layout = layout->entries[index];
            if (!bindings->slotIndices.emplace(slot.layout.slot, index).second) {
                destroyBindingSetImpl(*bindings);
                return fail(adapter, "D3D12 binding layout contains duplicate slots");
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
                                                        adapter.error)) {
                        destroyBindingSetImpl(*bindings);
                        return VERNON_STATUS_INTERNAL_ERROR;
                    }
                }
            }
            bindings->resourceDescriptorCount += slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                                                 slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                                                 slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                                 slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
            bindings->samplerDescriptorCount += slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER;
        }
        const VernonStatus status = updateBindingsImpl(adapter, *bindings, descriptor->values, descriptor->value_count);
        if (status != VERNON_STATUS_OK) {
            destroyBindingSetImpl(*bindings);
            return status;
        }
        *output = toHandle(bindings.release());
        adapter.bindingCreations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(adapter, "D3D12 binding preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    }
}

VernonStatus updateBindings(void *data, VernonRuntimeProviderObject handle,
                            const VernonRuntimeProviderBindingValue *values, size_t valueCount) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *bindings = fromHandle<PreparedBindingSet>(handle);
    if (!bindings)
        return fail(adapter, "D3D12 adapter received an invalid binding set");
    std::lock_guard<std::mutex> guard(bindings->mutex);
    return updateBindingsImpl(adapter, *bindings, values, valueCount);
}

VernonStatus encodeDispatch(void *data, VernonRuntimeProviderObject commandEncoder,
                            const VernonRuntimeProviderDispatchDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline ||
        (bindings == nullptr && descriptor->bindings.value != 0))
        return fail(adapter, "D3D12 adapter received an invalid dispatch descriptor");
    auto &device = *pipeline->device;
    ID3D12DescriptorHeap *heap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(nativeCommandEncoder(adapter, commandEncoder));
    if (!commands || commandEncoderRendering(adapter, commandEncoder))
        return fail(adapter, "D3D12 dispatch command encoder is invalid");
    commands->SetPipelineState(pipeline->pipeline);
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "D3D12 dispatch could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    std::unique_lock<std::mutex> bindingGuard;
    if (bindings)
        bindingGuard = std::unique_lock<std::mutex>(bindings->mutex);
    const size_t slotCount = bindings ? bindings->slots.size() : 0;
    const bool reuseDescriptors = bindings && bindings->computeDescriptorEncoder == commandEncoder.value &&
                                  bindings->computeDescriptorRevision == bindings->descriptorContentRevision;
    if (reuseDescriptors) {
        heap = bindings->computeResourceHeap;
        gpu = bindings->computeResourceGpu;
    } else if (slotCount && !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true,
                                                       static_cast<uint32_t>(slotCount), heap, cpu, gpu, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    const UINT increment = device.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (bindings) {
        if (!retainBindingResources(adapter, commandEncoder, bindings))
            return fail(adapter, "D3D12 dispatch could not retain its resources", VERNON_STATUS_INTERNAL_ERROR);
        if (!reuseDescriptors) {
            for (auto &slot : bindings->slots) {
                if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
                    ID3D12Resource *upload = nullptr;
                    size_t uploadOffset = 0;
                    uint8_t *mapped = nullptr;
                    if (!device.acquireStaging(true, slot.inlineStorage.size(), 256, upload, uploadOffset, mapped,
                                               adapter.error))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    std::memcpy(mapped, slot.inlineStorage.data(), slot.inlineStorage.size());
                    if (!transition(adapter, commandEncoder, commands, slot.inlineResource,
                                    D3D12_RESOURCE_STATE_COPY_DEST))
                        return fail(adapter, "D3D12 dispatch could not track inline resource state",
                                    VERNON_STATUS_INTERNAL_ERROR);
                    commands->CopyBufferRegion(slot.inlineResource.resource, 0, upload, uploadOffset,
                                               slot.inlineStorage.size());
                    if (!transition(adapter, commandEncoder, commands, slot.inlineResource,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
                        return fail(adapter, "D3D12 dispatch could not track inline resource state",
                                    VERNON_STATUS_INTERNAL_ERROR);
                }
                D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                view.Format = DXGI_FORMAT_R32_TYPELESS;
                view.Buffer.FirstElement = slot.offset / 4;
                view.Buffer.NumElements = static_cast<UINT>(std::max<uint64_t>(1, (slot.size + 3) / 4));
                view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                device.device->CreateUnorderedAccessView(slot.resource, nullptr, &view, cpu);
                cpu.ptr += increment;
            }
            bindings->computeDescriptorEncoder = commandEncoder.value;
            bindings->computeDescriptorRevision = bindings->descriptorContentRevision;
            bindings->computeResourceHeap = heap;
            bindings->computeResourceGpu = gpu;
        }
    }
    commands->SetComputeRootSignature(pipeline->rootSignature);
    if (heap) {
        commands->SetDescriptorHeaps(1, &heap);
        commands->SetComputeRootDescriptorTable(0, gpu);
    }
    commands->Dispatch(descriptor->group_count[0], descriptor->group_count[1], descriptor->group_count[2]);
    if (!recordProviderCommand(adapter, commandEncoder, false))
        return fail(adapter, "D3D12 dispatch command encoder state changed");
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

VernonStatus encodeDraw(void *data, VernonRuntimeProviderObject commandEncoder,
                        const VernonRuntimeProviderDrawDescriptor *descriptor) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto *pipeline = descriptor ? fromHandle<PreparedPipeline>(descriptor->pipeline) : nullptr;
    auto *bindings = descriptor ? fromHandle<PreparedBindingSet>(descriptor->bindings) : nullptr;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !pipeline || !pipeline->graphics ||
        !pipeline->pipeline || (descriptor->bindings.value != 0 && !bindings) ||
        descriptor->color_attachment_count == 0 ||
        descriptor->color_attachment_count > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS)
        return fail(adapter, "D3D12 adapter received an invalid draw descriptor");
    auto &device = *pipeline->device;
    ID3D12DescriptorHeap *rtvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    D3D12_GPU_DESCRIPTOR_HANDLE ignoredGpu{};
    ID3D12DescriptorHeap *dsvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
    ID3D12DescriptorHeap *resourceHeap = nullptr;
    ID3D12DescriptorHeap *samplerHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE resourceCpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE resourceGpu{};
    D3D12_CPU_DESCRIPTOR_HANDLE samplerCpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE samplerGpu{};
    auto *commands = reinterpret_cast<ID3D12GraphicsCommandList *>(nativeCommandEncoder(adapter, commandEncoder));
    const int claim = claimCommandRendering(adapter, commandEncoder, vernon::rhi::CommandRenderingStateless);
    if (!commands || claim < 0)
        return fail(adapter, "D3D12 draw command encoder is invalid");
    commands->SetPipelineState(pipeline->pipeline);
    commands->OMSetStencilRef(descriptor->stencil_reference);
    adapter.lastStencilReference.store(descriptor->stencil_reference, std::memory_order_relaxed);
    adapter.lastDrawIndexed.store(descriptor->index_count != 0, std::memory_order_relaxed);
    const bool firstDraw = claim != 0;
    const bool standaloneRendering = !commandEncoderHasRenderingDescriptor(adapter, commandEncoder);
    if (!retainCommandObjects(adapter, commandEncoder, *pipeline, bindings))
        return fail(adapter, "D3D12 draw could not retain provider objects", VERNON_STATUS_INTERNAL_ERROR);
    std::unique_lock<std::mutex> bindingGuard;
    if (bindings)
        bindingGuard = std::unique_lock<std::mutex>(bindings->mutex);
    const uint32_t resourceCount = bindings ? bindings->resourceDescriptorCount : 0;
    const uint32_t samplerCount = bindings ? bindings->samplerDescriptorCount : 0;
    const bool reuseDescriptors = bindings && bindings->descriptorEncoder == commandEncoder.value &&
                                  bindings->descriptorRevision == bindings->descriptorContentRevision;
    if (reuseDescriptors) {
        resourceHeap = bindings->resourceHeap;
        samplerHeap = bindings->samplerHeap;
        resourceGpu = bindings->resourceGpu;
        samplerGpu = bindings->samplerGpu;
    } else {
        if (resourceCount && !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true, resourceCount,
                                                        resourceHeap, resourceCpu, resourceGpu, adapter.error))
            return VERNON_STATUS_INTERNAL_ERROR;
        if (samplerCount && !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, true, samplerCount,
                                                       samplerHeap, samplerCpu, samplerGpu, adapter.error))
            return VERNON_STATUS_INTERNAL_ERROR;
    }
    std::array<VernonRhiLoadOperation, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorLoads{};
    std::array<std::array<float, 4>, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorClears{};
    for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
        colorLoads[index] = static_cast<VernonRhiLoadOperation>(descriptor->color_attachments[index].load_operation);
        std::copy(std::begin(descriptor->color_attachments[index].clear_color),
                  std::end(descriptor->color_attachments[index].clear_color), colorClears[index].begin());
        VernonRhiStoreOperation ignoredStore =
            static_cast<VernonRhiStoreOperation>(descriptor->color_attachments[index].store_operation);
        (void)commandColorOperations(adapter, commandEncoder, index, colorLoads[index], ignoredStore,
                                     colorClears[index].data());
    }
    VernonRhiLoadOperation depthLoad = static_cast<VernonRhiLoadOperation>(descriptor->depth_load_operation);
    VernonRhiStoreOperation depthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation stencilLoad = static_cast<VernonRhiLoadOperation>(descriptor->stencil_load_operation);
    VernonRhiStoreOperation stencilStore = static_cast<VernonRhiStoreOperation>(descriptor->stencil_store_operation);
    float clearDepth = descriptor->clear_depth;
    uint32_t clearStencil = descriptor->clear_stencil;
    (void)commandDepthOperations(adapter, commandEncoder, depthLoad, depthStore, stencilLoad, stencilStore, clearDepth,
                                 clearStencil);
    if (firstDraw && !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false,
                                                static_cast<uint32_t>(descriptor->color_attachment_count), rtvHeap, rtv,
                                                ignoredGpu, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    if (firstDraw && descriptor->depth_stencil_view.resource.value &&
        !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false, 1, dsvHeap, dsv, ignoredGpu, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> renderTargets{};
    std::array<uint64_t, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> renderTargetResources{};
    const bool hasDepth = descriptor->depth_stencil_view.resource.value != 0;
    const UINT rtvIncrement = device.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    if (firstDraw) {
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            const auto &attachment = descriptor->color_attachments[index];
            if (!retainCommandResource(adapter, commandEncoder, attachment.view))
                return fail(adapter, "D3D12 draw could not retain its color attachment", VERNON_STATUS_INTERNAL_ERROR);
            if (!isRhiImageReference(attachment.view))
                return fail(adapter, "D3D12 draw contains an invalid color attachment");
            auto *image = reinterpret_cast<rhi::directx12::Image *>(resolveRhiResource(adapter, attachment.view));
            if (!image || !image->resource)
                return fail(adapter, "D3D12 draw contains an empty color attachment");
            renderTargetResources[attachment.location] = reinterpret_cast<uintptr_t>(image->resource);
            if (!transition(adapter, commandEncoder, commands, *image, D3D12_RESOURCE_STATE_RENDER_TARGET))
                return fail(adapter, "D3D12 draw could not track color attachment state", VERNON_STATUS_INTERNAL_ERROR);
            D3D12_RENDER_TARGET_VIEW_DESC view{};
            view.Format = image->format;
            view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            renderTargets[index] = rtv;
            device.device->CreateRenderTargetView(image->resource, &view, rtv);
            rtv.ptr += rtvIncrement;
        }
        if (hasDepth) {
            if (!retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_view))
                return fail(adapter, "D3D12 draw could not retain its depth attachment", VERNON_STATUS_INTERNAL_ERROR);
            if (!isRhiImageReference(descriptor->depth_stencil_view))
                return fail(adapter, "D3D12 draw contains an invalid depth attachment");
            auto *image =
                reinterpret_cast<rhi::directx12::Image *>(resolveRhiResource(adapter, descriptor->depth_stencil_view));
            if (!image || !image->resource ||
                (image->format != DXGI_FORMAT_D32_FLOAT && image->format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT))
                return fail(adapter, "D3D12 draw contains an invalid depth attachment");
            if (image->format != DXGI_FORMAT_D32_FLOAT_S8X24_UINT &&
                (stencilLoad != VERNON_RHI_LOAD_DISCARD || stencilStore != VERNON_RHI_STORE_DISCARD || clearStencil))
                return fail(adapter, "D3D12 draw requests stencil operations for a depth-only attachment");
            if (!transition(adapter, commandEncoder, commands, *image, D3D12_RESOURCE_STATE_DEPTH_WRITE))
                return fail(adapter, "D3D12 draw could not track depth attachment state", VERNON_STATUS_INTERNAL_ERROR);
            D3D12_DEPTH_STENCIL_VIEW_DESC view{};
            view.Format = image->format;
            view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            device.device->CreateDepthStencilView(image->resource, &view, dsv);
        }
        std::array<uint64_t, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS> colorTargets{};
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            colorTargets[descriptor->color_attachments[index].location] = renderTargets[index].ptr;
        auto *depthImage =
            hasDepth
                ? reinterpret_cast<rhi::directx12::Image *>(resolveRhiResource(adapter, descriptor->depth_stencil_view))
                : nullptr;
        if (!setCommandRenderingTargets(adapter, commandEncoder, colorTargets.data(), renderTargetResources.data(),
                                        colorTargets.size(), hasDepth ? dsv.ptr : 0,
                                        depthImage ? reinterpret_cast<uintptr_t>(depthImage->resource) : 0))
            return fail(adapter, "D3D12 command render-target registration failed", VERNON_STATUS_INTERNAL_ERROR);
    }
    std::array<ID3D12DescriptorHeap *, 2> heaps{};
    UINT heapCount = 0;
    if (resourceHeap)
        heaps[heapCount++] = resourceHeap;
    if (samplerHeap)
        heaps[heapCount++] = samplerHeap;
    if (heapCount)
        commands->SetDescriptorHeaps(heapCount, heaps.data());
    commands->SetGraphicsRootSignature(pipeline->rootSignature);
    UINT rootIndex = 0;
    if (pipeline->hasResourceTable)
        commands->SetGraphicsRootDescriptorTable(rootIndex++, resourceGpu);
    if (pipeline->hasSamplerTable)
        commands->SetGraphicsRootDescriptorTable(rootIndex, samplerGpu);
    if (bindings) {
        if (!retainBindingResources(adapter, commandEncoder, bindings))
            return fail(adapter, "D3D12 draw could not retain its bindings", VERNON_STATUS_INTERNAL_ERROR);
        std::array<uint32_t, 64> rootValues{};
        for (const auto &block : pipeline->inlineRootBlocks) {
            std::fill_n(rootValues.begin(), block.valueCount, uint32_t{0});
            for (const auto &member : block.members) {
                const auto found = bindings->slotIndices.find(member.slot);
                if (found == bindings->slotIndices.end() || bindings->slots[found->second].inlineStorage.empty())
                    return fail(adapter, "D3D12 draw contains an invalid inline uniform");
                const auto &slot = bindings->slots[found->second];
                std::memcpy(reinterpret_cast<uint8_t *>(rootValues.data()) +
                                member.destinationOffset * sizeof(uint32_t),
                            slot.inlineStorage.data(), member.size);
            }
            commands->SetGraphicsRoot32BitConstants(block.rootParameter, block.valueCount, rootValues.data(), 0);
        }
        const UINT resourceIncrement =
            device.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT samplerIncrement =
            device.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        for (auto &slot : bindings->slots) {
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
                if (!reuseDescriptors) {
                    ID3D12Resource *upload = nullptr;
                    size_t uploadOffset = 0;
                    uint8_t *mapped = nullptr;
                    if (!device.acquireStaging(true, slot.inlineStorage.size(), 256, upload, uploadOffset, mapped,
                                               adapter.error))
                        return VERNON_STATUS_INTERNAL_ERROR;
                    std::memcpy(mapped, slot.inlineStorage.data(), slot.inlineStorage.size());
                    if (!transition(adapter, commandEncoder, commands, slot.inlineResource,
                                    D3D12_RESOURCE_STATE_COPY_DEST))
                        return fail(adapter, "D3D12 draw could not track inline resource state",
                                    VERNON_STATUS_INTERNAL_ERROR);
                    commands->CopyBufferRegion(slot.inlineResource.resource, 0, upload, uploadOffset,
                                               slot.inlineStorage.size());
                    if (!transition(adapter, commandEncoder, commands, slot.inlineResource,
                                    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER))
                        return fail(adapter, "D3D12 draw could not track inline resource state",
                                    VERNON_STATUS_INTERNAL_ERROR);
                    D3D12_CONSTANT_BUFFER_VIEW_DESC view{};
                    view.BufferLocation = slot.inlineResource.resource->GetGPUVirtualAddress();
                    view.SizeInBytes = (static_cast<UINT>(slot.inlineStorage.size()) + 255u) & ~255u;
                    device.device->CreateConstantBufferView(&view, resourceCpu);
                    resourceCpu.ptr += resourceIncrement;
                }
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
                if (slot.layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
                    if (!reuseDescriptors) {
                        ID3D12Resource *upload = nullptr;
                        size_t uploadOffset = 0;
                        uint8_t *mapped = nullptr;
                        if (!device.acquireStaging(true, slot.inlineStorage.size(), 256, upload, uploadOffset, mapped,
                                                   adapter.error))
                            return VERNON_STATUS_INTERNAL_ERROR;
                        std::memcpy(mapped, slot.inlineStorage.data(), slot.inlineStorage.size());
                        if (!transition(adapter, commandEncoder, commands, slot.inlineResource,
                                        D3D12_RESOURCE_STATE_COPY_DEST))
                            return fail(adapter, "D3D12 draw could not track inline resource state",
                                        VERNON_STATUS_INTERNAL_ERROR);
                        commands->CopyBufferRegion(slot.inlineResource.resource, 0, upload, uploadOffset,
                                                   slot.inlineStorage.size());
                        if (!transition(adapter, commandEncoder, commands, slot.inlineResource,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
                            return fail(adapter, "D3D12 draw could not track inline resource state",
                                        VERNON_STATUS_INTERNAL_ERROR);
                        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                        view.Format = DXGI_FORMAT_R32_TYPELESS;
                        view.Buffer.FirstElement = 0;
                        view.Buffer.NumElements =
                            static_cast<UINT>(std::max<uint64_t>(1, (slot.inlineStorage.size() + 3) / 4));
                        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                        device.device->CreateUnorderedAccessView(slot.inlineResource.resource, nullptr, &view,
                                                                 resourceCpu);
                        resourceCpu.ptr += resourceIncrement;
                    }
                } else {
                    auto *buffer = static_cast<rhi::directx12::Buffer *>(slot.opaqueResource);
                    if (!buffer || !buffer->resource)
                        return fail(adapter, "D3D12 draw contains an invalid storage buffer");
                    if (!transition(adapter, commandEncoder, commands, *buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
                        return fail(adapter, "D3D12 draw could not track storage buffer state",
                                    VERNON_STATUS_INTERNAL_ERROR);
                    if (!reuseDescriptors) {
                        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                        view.Format = DXGI_FORMAT_R32_TYPELESS;
                        view.Buffer.FirstElement = slot.offset / 4;
                        view.Buffer.NumElements = static_cast<UINT>(std::max<uint64_t>(1, (slot.size + 3) / 4));
                        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                        device.device->CreateUnorderedAccessView(buffer->resource, nullptr, &view, resourceCpu);
                        resourceCpu.ptr += resourceIncrement;
                    }
                }
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE) {
                auto *image = static_cast<rhi::directx12::Image *>(slot.opaqueResource);
                if (!image || !image->resource ||
                    !transition(adapter, commandEncoder, commands, *image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS))
                    return fail(adapter, "D3D12 dispatch could not track storage image state",
                                VERNON_STATUS_INTERNAL_ERROR);
                if (!reuseDescriptors) {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
                    view.Format = image->format;
                    const bool selected = image->view.struct_size >= sizeof(VernonRhiImageViewDescriptor);
                    const uint32_t mip = selected ? image->view.base_mip_level : 0;
                    if (image->dimension == VERNON_RHI_IMAGE_3D) {
                        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                        view.Texture3D.MipSlice = mip;
                        view.Texture3D.WSize = UINT_MAX;
                    } else if (selected && image->view.array_layer_count > 1) {
                        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                        view.Texture2DArray.MipSlice = mip;
                        view.Texture2DArray.FirstArraySlice = image->view.base_array_layer;
                        view.Texture2DArray.ArraySize = image->view.array_layer_count;
                    } else {
                        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                        view.Texture2D.MipSlice = mip;
                    }
                    device.device->CreateUnorderedAccessView(image->resource, nullptr, &view, resourceCpu);
                    resourceCpu.ptr += resourceIncrement;
                }
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
                auto *image = static_cast<rhi::directx12::Image *>(slot.opaqueResource);
                if (!transition(adapter, commandEncoder, commands, *image, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE))
                    return fail(adapter, "D3D12 draw could not track sampled image state",
                                VERNON_STATUS_INTERNAL_ERROR);
                if (!reuseDescriptors) {
                    const D3D12_RESOURCE_DESC native = image->resource->GetDesc();
                    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
                    view.Format = image->format == DXGI_FORMAT_D32_FLOAT ? DXGI_FORMAT_R32_FLOAT : image->format;
                    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    const bool selected = image->view.struct_size >= sizeof(VernonRhiImageViewDescriptor);
                    const uint32_t firstMip = selected ? image->view.base_mip_level : 0;
                    const uint32_t mipCount = selected ? image->view.mip_level_count : native.MipLevels;
                    if (image->dimension == VERNON_RHI_IMAGE_CUBE) {
                        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                        view.TextureCube.MostDetailedMip = firstMip;
                        view.TextureCube.MipLevels = mipCount;
                    } else if (image->dimension == VERNON_RHI_IMAGE_3D) {
                        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                        view.Texture3D.MostDetailedMip = firstMip;
                        view.Texture3D.MipLevels = mipCount;
                    } else if (selected && image->view.array_layer_count > 1) {
                        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                        view.Texture2DArray.MostDetailedMip = firstMip;
                        view.Texture2DArray.MipLevels = mipCount;
                        view.Texture2DArray.FirstArraySlice = image->view.base_array_layer;
                        view.Texture2DArray.ArraySize = image->view.array_layer_count;
                    } else {
                        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                        view.Texture2D.MostDetailedMip = firstMip;
                        view.Texture2D.MipLevels = mipCount;
                    }
                    device.device->CreateShaderResourceView(image->resource, &view, resourceCpu);
                    resourceCpu.ptr += resourceIncrement;
                }
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) {
                const auto *sampler = static_cast<const rhi::directx12::Sampler *>(slot.opaqueResource);
                const D3D12_SAMPLER_DESC defaultSampler{D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                                                        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                        0,
                                                        1,
                                                        D3D12_COMPARISON_FUNC_ALWAYS,
                                                        {},
                                                        0,
                                                        D3D12_FLOAT32_MAX};
                if (!reuseDescriptors) {
                    device.device->CreateSampler(sampler ? &sampler->descriptor : &defaultSampler, samplerCpu);
                    samplerCpu.ptr += samplerIncrement;
                }
            } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
                auto *buffer = static_cast<rhi::directx12::Buffer *>(slot.opaqueResource);
                if (!buffer || !buffer->resource)
                    return fail(adapter, "D3D12 draw contains an invalid vertex buffer");
                if (!transition(adapter, commandEncoder, commands, *buffer,
                                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER))
                    return fail(adapter, "D3D12 draw could not track vertex buffer state",
                                VERNON_STATUS_INTERNAL_ERROR);
                const D3D12_VERTEX_BUFFER_VIEW view{slot.resource->GetGPUVirtualAddress() + slot.offset,
                                                    static_cast<UINT>(slot.size), slot.stride};
                commands->IASetVertexBuffers(slot.layout.binding, 1, &view);
            }
        }
        if (!reuseDescriptors) {
            bindings->descriptorEncoder = commandEncoder.value;
            bindings->descriptorRevision = bindings->descriptorContentRevision;
            bindings->resourceHeap = resourceHeap;
            bindings->samplerHeap = samplerHeap;
            bindings->resourceGpu = resourceGpu;
            bindings->samplerGpu = samplerGpu;
        }
    }
    if (firstDraw) {
        commands->OMSetRenderTargets(static_cast<UINT>(descriptor->color_attachment_count), renderTargets.data(), FALSE,
                                     hasDepth ? &dsv : nullptr);
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            if (colorLoads[index] == VERNON_RHI_LOAD_CLEAR)
                commands->ClearRenderTargetView(renderTargets[index], colorClears[index].data(), 0, nullptr);
        if (hasDepth && (depthLoad == VERNON_RHI_LOAD_CLEAR || stencilLoad == VERNON_RHI_LOAD_CLEAR)) {
            UINT flags = 0;
            if (depthLoad == VERNON_RHI_LOAD_CLEAR)
                flags |= D3D12_CLEAR_FLAG_DEPTH;
            if (stencilLoad == VERNON_RHI_LOAD_CLEAR)
                flags |= D3D12_CLEAR_FLAG_STENCIL;
            commands->ClearDepthStencilView(dsv, static_cast<D3D12_CLEAR_FLAGS>(flags), clearDepth,
                                            static_cast<UINT8>(clearStencil), 0, nullptr);
        }
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
    if (descriptor->index_count != 0) {
        if (!retainCommandResource(adapter, commandEncoder, descriptor->index_buffer))
            return fail(adapter, "D3D12 draw could not retain its index buffer", VERNON_STATUS_INTERNAL_ERROR);
        auto *buffer =
            reinterpret_cast<rhi::directx12::Buffer *>(resolveRhiResource(adapter, descriptor->index_buffer));
        if (!buffer || !buffer->resource)
            return fail(adapter, "D3D12 draw contains an invalid index buffer");
        if (!transition(adapter, commandEncoder, commands, *buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER))
            return fail(adapter, "D3D12 draw could not track index buffer state", VERNON_STATUS_INTERNAL_ERROR);
        const D3D12_INDEX_BUFFER_VIEW indexView{buffer->resource->GetGPUVirtualAddress() +
                                                    descriptor->index_buffer.offset,
                                                static_cast<UINT>(descriptor->index_buffer.size), DXGI_FORMAT_R32_UINT};
        commands->IASetIndexBuffer(&indexView);
        commands->DrawIndexedInstanced(descriptor->index_count, descriptor->instance_count, 0, descriptor->first_vertex,
                                       descriptor->first_instance);
    } else {
        commands->DrawInstanced(descriptor->vertex_count, descriptor->instance_count, descriptor->first_vertex,
                                descriptor->first_instance);
    }
    if (standaloneRendering) {
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            if (descriptor->color_attachments[index].store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD) {
                auto *image = reinterpret_cast<rhi::directx12::Image *>(
                    resolveRhiResource(adapter, descriptor->color_attachments[index].view));
                commands->DiscardResource(image->resource, nullptr);
            }
        if (hasDepth && descriptor->depth_store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD &&
            descriptor->stencil_store_operation == VERNON_RUNTIME_PROVIDER_STORE_DISCARD) {
            auto *image =
                reinterpret_cast<rhi::directx12::Image *>(resolveRhiResource(adapter, descriptor->depth_stencil_view));
            commands->DiscardResource(image->resource, nullptr);
        }
    }
    if (!recordProviderCommand(adapter, commandEncoder, true))
        return fail(adapter, "D3D12 draw command encoder state changed");
    adapter.dispatches.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
}

void destroyShader(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedShader>(handle); }
void destroyLayout(void *, VernonRuntimeProviderObject handle) { delete fromHandle<PreparedLayout>(handle); }
void releaseCommandPipeline(void *context, uint64_t) {
    auto *pipeline = static_cast<PreparedPipeline *>(context);
    if (!pipeline || pipeline->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    delete pipeline;
}
void destroyPipeline(void *, VernonRuntimeProviderObject handle) {
    releaseCommandPipeline(fromHandle<PreparedPipeline>(handle), 0);
}
void releaseCommandBindings(void *context, uint64_t) {
    auto *bindings = static_cast<PreparedBindingSet *>(context);
    if (!bindings || bindings->references.fetch_sub(1, std::memory_order_acq_rel) != 1)
        return;
    destroyBindingSetImpl(*bindings);
    delete bindings;
}
void destroyBindings(void *, VernonRuntimeProviderObject handle) {
    releaseCommandBindings(fromHandle<PreparedBindingSet>(handle), 0);
}

} // namespace

void initializeDirectX12Provider(VernonRuntimeRhiAdapter &adapter) {
    adapter.provider.struct_size = sizeof(adapter.provider);
    adapter.provider.abi_version = VERNON_PIPELINE_VERSION;
    adapter.provider.user_data = &adapter;
    adapter.provider.get_capabilities = getCapabilities;
    adapter.provider.get_device_identity = getDeviceIdentity;
    adapter.provider.prepare_shader = prepareShader;
    adapter.provider.prepare_pipeline_layout = prepareLayout;
    adapter.provider.prepare_pipeline = preparePipeline;
    adapter.provider.retain_resource = retainResource;
    adapter.provider.release_resource = releaseResource;
    adapter.provider.describe_image = describeProviderImageCallback;
    adapter.provider.create_binding_set = createBindings;
    adapter.provider.update_binding_set = updateBindings;
    adapter.provider.encode_dispatch = encodeDispatch;
    adapter.provider.encode_draw = encodeDraw;
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
    auto *deviceState = static_cast<rhi::directx12::DeviceState *>(rhi::deviceState(device, backend));
    if (!deviceState)
        return nullptr;
    auto state =
        std::unique_ptr<rhi_adapter::DirectX12AdapterState>(new (std::nothrow) rhi_adapter::DirectX12AdapterState());
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!state || !adapter)
        return nullptr;
    state->device = deviceState;
    adapter->rhiBackend = backend;
    if (!adapter->backend.adopt(state.get(), &rhi_adapter::backendOps))
        return nullptr;
    (void)state.release();
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
