#include "../vertex_attribute_capabilities.h"
#include "adapter_common.h"

#if defined(VERNON_HAS_DIRECTX12_RHI)

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::rhi_adapter {
namespace {

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
    rhi::directx12::DeviceState *device{};
    ID3D12RootSignature *rootSignature{};
    ID3D12PipelineState *pipeline{};
    bool graphics{};
    bool hasResourceTable{};
    bool hasSamplerTable{};
    std::vector<InlineRootBlock> inlineRootBlocks;
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

uint32_t getCapabilities(void *) {
    return VERNON_RUNTIME_PROVIDER_COMPUTE | VERNON_RUNTIME_PROVIDER_GRAPHICS | VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP;
}

VernonRuntimeProviderDeviceIdentity getDeviceIdentity(void *data) {
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return {0x4433443132u, reinterpret_cast<uintptr_t>(adapter.directX12Device->device),
            static_cast<uint64_t>(adapter.directX12Device->shaderModel)};
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

VernonStatus prepareGraphicsPipeline(VernonRuntimeRhiAdapter &adapter,
                                     const VernonRuntimeProviderPipelineDescriptor &descriptor, PreparedLayout &layout,
                                     VernonRuntimeProviderObject *output) {
    auto pipeline = std::unique_ptr<PreparedPipeline>(new (std::nothrow) PreparedPipeline());
    if (!pipeline)
        return fail(adapter, "D3D12 graphics pipeline preparation ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
    pipeline->device = adapter.directX12Device;
    pipeline->graphics = true;
    if (descriptor.color_format_count == 0) {
        *output = toHandle(pipeline.release());
        adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
        return VERNON_STATUS_OK;
    }
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
            const bool registerAggregate = entry->vector_count > 1 || entry->element_size > 16;
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
    native.BlendState.AlphaToCoverageEnable = FALSE;
    native.BlendState.IndependentBlendEnable = FALSE;
    for (auto &target : native.BlendState.RenderTarget)
        target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    native.SampleMask = UINT_MAX;
    native.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    native.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    native.RasterizerState.DepthClipEnable = TRUE;
    native.DepthStencilState.DepthEnable = descriptor.depth_stencil_format ? TRUE : FALSE;
    native.DepthStencilState.DepthWriteMask =
        descriptor.depth_stencil_format ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    native.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    native.DepthStencilState.StencilEnable = FALSE;
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
    if (FAILED(result)) {
        releaseObject(pipeline->rootSignature);
        return fail(adapter, "ID3D12Device::CreateGraphicsPipelineState failed", VERNON_STATUS_INTERNAL_ERROR);
    }
    *output = toHandle(pipeline.release());
    adapter.pipelinePreparations.fetch_add(1, std::memory_order_relaxed);
    return VERNON_STATUS_OK;
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
    pipeline->device = adapter.directX12Device;
    result = pipeline->device->device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&pipeline->rootSignature));
    releaseObject(serialized);
    if (FAILED(result))
        return fail(adapter, "ID3D12Device::CreateRootSignature failed", VERNON_STATUS_INTERNAL_ERROR);
    D3D12_COMPUTE_PIPELINE_STATE_DESC native{};
    native.pRootSignature = pipeline->rootSignature;
    native.CS = {shader->artifact.data(), shader->artifact.size()};
    result = pipeline->device->device->CreateComputePipelineState(&native, IID_PPV_ARGS(&pipeline->pipeline));
    if (FAILED(result)) {
        releaseObject(pipeline->rootSignature);
        return fail(adapter, "ID3D12Device::CreateComputePipelineState failed", VERNON_STATUS_INTERNAL_ERROR);
    }
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
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            if (!value->inline_data || value->inline_size != slot.inlineStorage.size())
                return fail(adapter, "D3D12 inline or uniform-buffer binding size is invalid");
            bindings.resolvedValues[index] = 0;
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const uint64_t native = resolveRhiResource(adapter, value->resource);
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(native);
            if ((value->resource.identity & kDirectX12ResourceKindMask) != kDirectX12BufferResource || !buffer ||
                !buffer->resource || value->resource.offset > value->resource.size ||
                slot.layout.element_size > value->resource.size - value->resource.offset)
                return fail(adapter, "D3D12 storage binding is invalid");
            bindings.resolvedValues[index] = native;
        } else {
            const bool defaultSampler = slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER &&
                                        (value->flags & VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE) != 0;
            if (defaultSampler)
                bindings.resolvedValues[index] = 0;
            else {
                const uint64_t expectedKind =
                    slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ? kDirectX12ImageResource
                    : slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER     ? kDirectX12SamplerResource
                                                                              : kDirectX12BufferResource;
                const uint64_t native = resolveRhiResource(adapter, value->resource);
                if ((value->resource.identity & kDirectX12ResourceKindMask) != expectedKind || native == 0 ||
                    (slot.layout.kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER && value->stride == 0))
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
        const VernonRuntimeProviderResourceReference resource =
            defaultSampler ? VernonRuntimeProviderResourceReference{} : value.resource;
        const bool slotChanged =
            slot.flags != value.flags || !sameResource(slot.resourceReference, resource) ||
            slot.stride != value.stride ||
            (value.inline_data && std::memcmp(slot.inlineStorage.data(), value.inline_data, value.inline_size));
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
        if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
            slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
            std::memcpy(slot.inlineStorage.data(), value.inline_data, value.inline_size);
            slot.resource = slot.inlineResource.resource;
            slot.offset = 0;
            slot.size = slot.inlineStorage.size();
        } else if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            auto *buffer = reinterpret_cast<rhi::directx12::Buffer *>(bindings.resolvedValues[index]);
            slot.resource = buffer->resource;
            slot.opaqueResource = buffer;
            slot.resourceReference = value.resource;
            slot.offset = value.resource.offset;
            slot.size = value.resource.size - value.resource.offset;
        } else {
            slot.opaqueResource = reinterpret_cast<void *>(bindings.resolvedValues[index]);
            if (!slot.opaqueResource) {
                slot.offset = 0;
                slot.size = 0;
                slot.stride = 0;
                continue;
            }
            slot.resourceReference = value.resource;
            slot.offset = value.resource.offset;
            slot.size = value.resource.size;
            slot.stride = value.stride;
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
        bindings->device = adapter.directX12Device;
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
            if (slot.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
                slot.inlineStorage.resize(slot.layout.element_size);
                const bool needsInlineResource = slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                                                 slot.layout.stage_mask == VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                if (needsInlineResource) {
                    const size_t resourceSize =
                        slot.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                            ? (static_cast<size_t>(slot.layout.element_size) + 255u) & ~size_t{255u}
                            : slot.layout.element_size;
                    if (!bindings->device->createBuffer(
                            slot.inlineResource, resourceSize, slot.layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE,
                            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, adapter.error)) {
                        destroyBindingSetImpl(*bindings);
                        return VERNON_STATUS_INTERNAL_ERROR;
                    }
                }
            }
            bindings->resourceDescriptorCount += slot.layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
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
        descriptor->color_attachment_count == 0 || descriptor->color_attachment_count > 8)
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
    std::array<VernonRhiLoadOperation, 8> colorLoads{};
    std::array<std::array<float, 4>, 8> colorClears{};
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
    VernonRhiStoreOperation ignoredDepthStore = static_cast<VernonRhiStoreOperation>(descriptor->depth_store_operation);
    VernonRhiLoadOperation ignoredStencilLoad = VERNON_RHI_LOAD_DISCARD;
    VernonRhiStoreOperation ignoredStencilStore = VERNON_RHI_STORE_DISCARD;
    float clearDepth = descriptor->clear_depth;
    uint32_t ignoredClearStencil = 0;
    (void)commandDepthOperations(adapter, commandEncoder, depthLoad, ignoredDepthStore, ignoredStencilLoad,
                                 ignoredStencilStore, clearDepth, ignoredClearStencil);
    if (firstDraw && !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, false,
                                                static_cast<uint32_t>(descriptor->color_attachment_count), rtvHeap, rtv,
                                                ignoredGpu, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    if (firstDraw && descriptor->depth_stencil_attachment.resource.value &&
        !device.acquireDescriptors(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, false, 1, dsvHeap, dsv, ignoredGpu, adapter.error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 8> renderTargets{};
    std::array<uint64_t, 8> renderTargetResources{};
    const bool hasDepth = descriptor->depth_stencil_attachment.resource.value != 0;
    const UINT rtvIncrement = device.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    if (firstDraw) {
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index) {
            const auto &attachment = descriptor->color_attachments[index];
            if (!retainCommandResource(adapter, commandEncoder, attachment.image))
                return fail(adapter, "D3D12 draw could not retain its color attachment", VERNON_STATUS_INTERNAL_ERROR);
            if ((attachment.image.identity & kDirectX12ResourceKindMask) != kDirectX12ImageResource)
                return fail(adapter, "D3D12 draw contains an invalid color attachment");
            auto *image = reinterpret_cast<rhi::directx12::Image *>(resolveRhiResource(adapter, attachment.image));
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
            if (!retainCommandResource(adapter, commandEncoder, descriptor->depth_stencil_attachment))
                return fail(adapter, "D3D12 draw could not retain its depth attachment", VERNON_STATUS_INTERNAL_ERROR);
            if ((descriptor->depth_stencil_attachment.identity & kDirectX12ResourceKindMask) != kDirectX12ImageResource)
                return fail(adapter, "D3D12 draw contains an invalid depth attachment");
            auto *image = reinterpret_cast<rhi::directx12::Image *>(
                resolveRhiResource(adapter, descriptor->depth_stencil_attachment));
            if (!image || !image->resource || image->format != DXGI_FORMAT_D32_FLOAT)
                return fail(adapter, "D3D12 draw contains an invalid depth attachment");
            if (!transition(adapter, commandEncoder, commands, *image, D3D12_RESOURCE_STATE_DEPTH_WRITE))
                return fail(adapter, "D3D12 draw could not track depth attachment state", VERNON_STATUS_INTERNAL_ERROR);
            D3D12_DEPTH_STENCIL_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_D32_FLOAT;
            view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            device.device->CreateDepthStencilView(image->resource, &view, dsv);
        }
        std::array<uint64_t, 8> colorTargets{};
        for (size_t index = 0; index < descriptor->color_attachment_count; ++index)
            colorTargets[descriptor->color_attachments[index].location] = renderTargets[index].ptr;
        auto *depthImage = hasDepth ? reinterpret_cast<rhi::directx12::Image *>(
                                          resolveRhiResource(adapter, descriptor->depth_stencil_attachment))
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
                    if (image->dimension == VERNON_RHI_IMAGE_CUBE) {
                        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                        view.TextureCube.MipLevels = native.MipLevels;
                    } else {
                        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                        view.Texture2D.MipLevels = native.MipLevels;
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
        if (hasDepth && depthLoad == VERNON_RHI_LOAD_CLEAR)
            commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, clearDepth, 0, 0, nullptr);
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
            if (descriptor->color_attachments[index].store_operation == VERNON_RHI_STORE_DISCARD) {
                auto *image = reinterpret_cast<rhi::directx12::Image *>(
                    resolveRhiResource(adapter, descriptor->color_attachments[index].image));
                commands->DiscardResource(image->resource, nullptr);
            }
        if (hasDepth && descriptor->depth_store_operation == VERNON_RHI_STORE_DISCARD) {
            auto *image = reinterpret_cast<rhi::directx12::Image *>(
                resolveRhiResource(adapter, descriptor->depth_stencil_attachment));
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
    releaseObject(pipeline->pipeline);
    releaseObject(pipeline->rootSignature);
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

#endif
