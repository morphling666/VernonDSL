#include "graphics_directx12_encoder.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)

#include "backend_directx12.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <vector>

namespace vernon::runtime {
namespace {

template <typename T> void release(T *&value) {
    if (value)
        value->Release();
    value = nullptr;
}

VernonStatus fail(std::string &error, std::string message, VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    error = std::move(message);
    return status;
}

bool check(HRESULT result, const char *operation, std::string &error) {
    if (SUCCEEDED(result))
        return true;
    error = std::string(operation) + " failed with HRESULT " + std::to_string(static_cast<uint32_t>(result));
    return false;
}

D3D12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CreationNodeMask = 1;
    result.VisibleNodeMask = 1;
    return result;
}

ID3D12Resource *createUpload(DirectX12ContextState &context, const std::vector<uint8_t> &data, std::string &error) {
    const size_t size = std::max<size_t>(256, (data.size() + 255) & ~size_t{255});
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES heap = heapProperties(D3D12_HEAP_TYPE_UPLOAD);
    ID3D12Resource *resource = nullptr;
    if (!check(context.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                       IID_PPV_ARGS(&resource)),
               "ID3D12Device::CreateCommittedResource", error))
        return nullptr;
    uint8_t *mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    if (!check(resource->Map(0, &noRead, reinterpret_cast<void **>(&mapped)), "ID3D12Resource::Map", error)) {
        release(resource);
        return nullptr;
    }
    std::memset(mapped, 0, size);
    if (!data.empty())
        std::memcpy(mapped, data.data(), data.size());
    resource->Unmap(0, nullptr);
    return resource;
}

std::optional<DXGI_FORMAT> textureFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case VERNON_TEXTURE_R8_UNORM:
        return DXGI_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;
    case VERNON_TEXTURE_R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case VERNON_TEXTURE_RG8_UNORM:
        return DXGI_FORMAT_R8G8_UNORM;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return DXGI_FORMAT_R11G11B10_FLOAT;
    default:
        return std::nullopt;
    }
}

DXGI_FORMAT vertexFormat(uint32_t components) {
    switch (components) {
    case 1:
        return DXGI_FORMAT_R32_FLOAT;
    case 2:
        return DXGI_FORMAT_R32G32_FLOAT;
    case 3:
        return DXGI_FORMAT_R32G32B32_FLOAT;
    case 4:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE topologyType(VernonPrimitiveTopology topology) {
    if (topology == VERNON_TOPOLOGY_LINE_LIST)
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    if (topology == VERNON_TOPOLOGY_POINT_LIST)
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

D3D_PRIMITIVE_TOPOLOGY topology(VernonPrimitiveTopology value) {
    if (value == VERNON_TOPOLOGY_LINE_LIST)
        return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    if (value == VERNON_TOPOLOGY_POINT_LIST)
        return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
}

void transition(ID3D12GraphicsCommandList *commands, ID3D12Resource *resource, D3D12_RESOURCE_STATES &current,
                D3D12_RESOURCE_STATES target) {
    if (current == target)
        return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = current;
    barrier.Transition.StateAfter = target;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commands->ResourceBarrier(1, &barrier);
    current = target;
}

D3D12_TEXTURE_ADDRESS_MODE addressMode(VernonSamplerWrapMode mode) {
    if (mode == VERNON_SAMPLER_MIRRORED_REPEAT)
        return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    if (mode == VERNON_SAMPLER_CLAMP_TO_EDGE)
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    if (mode == VERNON_SAMPLER_CLAMP_TO_BORDER)
        return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
}

D3D12_FILTER samplerFilter(const VernonSamplerDescriptor &descriptor) {
    return descriptor.min_filter == VERNON_SAMPLER_LINEAR || descriptor.mag_filter == VERNON_SAMPLER_LINEAR ||
                   descriptor.mip_filter == VERNON_SAMPLER_LINEAR
               ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
               : D3D12_FILTER_MIN_MAG_MIP_POINT;
}

std::vector<uint8_t> uniformBlock(const char *stage, const Variant &variant, const PlannedGraphicsInvocation &plan,
                                  std::string &error) {
    struct Uniform {
        uint32_t index;
        const VernonTensorView *tensor;
        std::array<float, 2> resolution;
        bool generated;
    };
    std::vector<Uniform> uniforms;
    for (const Parameter &parameter : variant.parameters) {
        const auto found = plan.arguments.find(parameter.slot);
        if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR)
            continue;
        for (const ParameterUse &use : parameter.uses)
            if (use.stage == stage && use.interfaceKind == "uniform" && use.binding == UINT32_MAX)
                uniforms.push_back({use.index, &found->second->tensor, {}, false});
    }
    for (const Parameter &parameter : variant.internalParameters)
        if (parameter.source == "system_value")
            for (const ParameterUse &use : parameter.uses)
                if (use.stage == stage && use.interfaceKind == "uniform" && use.binding == UINT32_MAX)
                    uniforms.push_back({use.index, nullptr, plan.resolution, true});
    std::sort(uniforms.begin(), uniforms.end(),
              [](const Uniform &left, const Uniform &right) { return left.index < right.index; });
    std::vector<uint8_t> result;
    for (const Uniform &uniform : uniforms) {
        std::vector<uint8_t> bytes;
        if (uniform.generated) {
            bytes.resize(sizeof(uniform.resolution));
            std::memcpy(bytes.data(), uniform.resolution.data(), bytes.size());
        } else {
            auto packed = packTensorRowMajor(*uniform.tensor);
            if (!packed) {
                error = "D3D12 graphics uniform cannot be packed";
                return {};
            }
            bytes = std::move(*packed);
        }
        const size_t lane = result.size() % 16;
        if (lane + bytes.size() > 16)
            result.resize((result.size() + 15) & ~size_t{15});
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    return result;
}

} // namespace

VernonStatus encodeAndSubmitDirectX12Graphics(VernonRuntimeContext &context, DirectX12PipelineState &state,
                                              const Variant &variant, const VernonPipelineInvocation &invocation,
                                              const PlannedGraphicsInvocation &plan, std::string &error) {
    if (state.vertexDxil.empty() || state.fragmentDxil.empty())
        return fail(error, "D3D12 graphics pipeline has no DXIL stages", VERNON_STATUS_INTERNAL_ERROR);
    DirectX12ContextState &dx = directX12State(context);
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements;
    std::vector<D3D12_VERTEX_BUFFER_VIEW> vertexViews;
    inputElements.reserve(plan.vertexInputs.size());
    vertexViews.reserve(plan.vertexInputs.size());
    for (uint32_t index = 0; index < plan.vertexInputs.size(); ++index) {
        const PlannedVertexInput &input = plan.vertexInputs[index];
        if (input.components > 4)
            return fail(error, "D3D12 matrix vertex inputs are not implemented", VERNON_STATUS_UNSUPPORTED_TARGET);
        const DXGI_FORMAT format = vertexFormat(input.components);
        if (format == DXGI_FORMAT_UNKNOWN)
            return fail(error, "D3D12 vertex input format is unsupported", VERNON_STATUS_UNSUPPORTED_TARGET);
        inputElements.push_back({"TEXCOORD", input.use->location, format, index, 0,
                                 input.instanced ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                                 : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                                 input.instanced ? 1u : 0u});
        const VernonTensorView &tensor = *input.tensor;
        vertexViews.push_back(
            {directX12BufferState(*tensor.buffer).resource->GetGPUVirtualAddress() + tensor.byte_offset,
             static_cast<UINT>(tensor.byte_size), static_cast<UINT>(tensor.byte_strides[0])});
    }

    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    std::vector<D3D12_ROOT_PARAMETER> rootParameters;
    const std::vector<uint8_t> vertexUniforms = uniformBlock("vertex", variant, plan, error);
    if (!error.empty())
        return VERNON_STATUS_INVALID_ARGUMENT;
    const std::vector<uint8_t> fragmentUniforms = uniformBlock("fragment", variant, plan, error);
    if (!error.empty())
        return VERNON_STATUS_INVALID_ARGUMENT;
    const bool hasVertexUniforms = !vertexUniforms.empty();
    const bool hasFragmentUniforms = !fragmentUniforms.empty();
    if (hasVertexUniforms) {
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameter.Descriptor.ShaderRegister = 0;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        rootParameters.push_back(parameter);
    }
    if (hasFragmentUniforms) {
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameter.Descriptor.ShaderRegister = 0;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameters.push_back(parameter);
    }
    ranges.reserve(plan.sampledResources.size() * 2);
    rootParameters.reserve(rootParameters.size() + plan.sampledResources.size() * 2);
    for (const auto &[binding, resource] : plan.sampledResources) {
        D3D12_DESCRIPTOR_RANGE textureRange{};
        textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        textureRange.NumDescriptors = 1;
        textureRange.BaseShaderRegister = binding.second;
        textureRange.RegisterSpace = binding.first;
        textureRange.OffsetInDescriptorsFromTableStart = 0;
        ranges.push_back(textureRange);
        D3D12_ROOT_PARAMETER textureParameter{};
        textureParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        textureParameter.DescriptorTable.NumDescriptorRanges = 1;
        textureParameter.DescriptorTable.pDescriptorRanges = &ranges.back();
        textureParameter.ShaderVisibility = resource.stages == PLANNED_STAGE_VERTEX     ? D3D12_SHADER_VISIBILITY_VERTEX
                                            : resource.stages == PLANNED_STAGE_FRAGMENT ? D3D12_SHADER_VISIBILITY_PIXEL
                                                                                        : D3D12_SHADER_VISIBILITY_ALL;
        rootParameters.push_back(textureParameter);
        D3D12_DESCRIPTOR_RANGE samplerRange{};
        samplerRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        samplerRange.NumDescriptors = 1;
        // SPIRV-Cross combines sampled-image pairs and assigns the sampler the
        // sampled texture's register even when SPIR-V reflection retains the
        // sampler's original standalone binding.
        samplerRange.BaseShaderRegister = binding.second;
        samplerRange.RegisterSpace = binding.first;
        samplerRange.OffsetInDescriptorsFromTableStart = 0;
        ranges.push_back(samplerRange);
        D3D12_ROOT_PARAMETER samplerParameter = textureParameter;
        samplerParameter.DescriptorTable.pDescriptorRanges = &ranges.back();
        rootParameters.push_back(samplerParameter);
    }
    // range pointers must be rebound after vector growth.
    size_t rangeIndex = 0;
    const size_t descriptorRootStart = (hasVertexUniforms ? 1 : 0) + (hasFragmentUniforms ? 1 : 0);
    for (size_t index = descriptorRootStart; index < rootParameters.size(); ++index)
        rootParameters[index].DescriptorTable.pDescriptorRanges = &ranges[rangeIndex++];

    D3D12_ROOT_SIGNATURE_DESC rootDescription{};
    rootDescription.NumParameters = static_cast<UINT>(rootParameters.size());
    rootDescription.pParameters = rootParameters.data();
    rootDescription.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob *serialized = nullptr;
    ID3DBlob *rootErrors = nullptr;
    if (!check(D3D12SerializeRootSignature(&rootDescription, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &rootErrors),
               "D3D12SerializeRootSignature", error)) {
        if (rootErrors)
            error.assign(static_cast<const char *>(rootErrors->GetBufferPointer()), rootErrors->GetBufferSize());
        release(rootErrors);
        release(serialized);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    release(rootErrors);
    ID3D12RootSignature *rootSignature = nullptr;
    if (!check(dx.device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&rootSignature)),
               "ID3D12Device::CreateRootSignature", error)) {
        release(serialized);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    release(serialized);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = rootSignature;
    pipeline.VS = {state.vertexDxil.data(), state.vertexDxil.size()};
    pipeline.PS = {state.fragmentDxil.data(), state.fragmentDxil.size()};
    pipeline.BlendState.AlphaToCoverageEnable = FALSE;
    pipeline.BlendState.IndependentBlendEnable = FALSE;
    for (D3D12_RENDER_TARGET_BLEND_DESC &target : pipeline.BlendState.RenderTarget) {
        target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        target.SrcBlend = D3D12_BLEND_ONE;
        target.DestBlend = D3D12_BLEND_ZERO;
        target.BlendOp = D3D12_BLEND_OP_ADD;
        target.SrcBlendAlpha = D3D12_BLEND_ONE;
        target.DestBlendAlpha = D3D12_BLEND_ZERO;
        target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        target.LogicOp = D3D12_LOGIC_OP_NOOP;
    }
    pipeline.SampleMask = UINT_MAX;
    pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pipeline.RasterizerState.DepthClipEnable = TRUE;
    pipeline.DepthStencilState.DepthEnable = FALSE;
    pipeline.DepthStencilState.StencilEnable = FALSE;
    pipeline.InputLayout = {inputElements.data(), static_cast<UINT>(inputElements.size())};
    pipeline.PrimitiveTopologyType = topologyType(invocation.topology);
    pipeline.NumRenderTargets = static_cast<UINT>(plan.maximumAttachmentLocation + 1);
    for (const VernonColorAttachment *attachment : plan.attachments) {
        const auto format = textureFormat(attachment->texture->format);
        if (!format) {
            release(rootSignature);
            return fail(error, "D3D12 render-target format is unsupported", VERNON_STATUS_UNSUPPORTED_TARGET);
        }
        pipeline.RTVFormats[attachment->location] = *format;
    }
    pipeline.SampleDesc.Count = 1;
    std::string pipelineKey;
    const auto appendKey = [&](const auto &value) {
        pipelineKey.append(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    appendKey(pipeline.PrimitiveTopologyType);
    appendKey(pipeline.NumRenderTargets);
    for (UINT index = 0; index < pipeline.NumRenderTargets; ++index)
        appendKey(pipeline.RTVFormats[index]);
    for (const D3D12_INPUT_ELEMENT_DESC &element : inputElements) {
        appendKey(element.SemanticIndex);
        appendKey(element.Format);
        appendKey(element.InputSlot);
        appendKey(element.AlignedByteOffset);
        appendKey(element.InputSlotClass);
        appendKey(element.InstanceDataStepRate);
    }
    for (const auto &[binding, resource] : plan.sampledResources) {
        appendKey(binding);
        appendKey(resource.stages);
    }
    appendKey(hasVertexUniforms);
    appendKey(hasFragmentUniforms);
    ID3D12PipelineState *pipelineState = nullptr;
    const auto cached = state.graphicsPipelines.find(pipelineKey);
    if (cached != state.graphicsPipelines.end()) {
        pipelineState = cached->second;
        pipelineState->AddRef();
    } else {
        if (!check(dx.device->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(&pipelineState)),
                   "ID3D12Device::CreateGraphicsPipelineState", error)) {
            release(rootSignature);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        pipelineState->AddRef();
        state.graphicsPipelines.emplace(std::move(pipelineKey), pipelineState);
        ++state.graphicsPipelineCreations;
    }

    ID3D12DescriptorHeap *rtvHeap = nullptr;
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescription{};
    rtvDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDescription.NumDescriptors = pipeline.NumRenderTargets;
    if (!check(dx.device->CreateDescriptorHeap(&rtvDescription, IID_PPV_ARGS(&rtvHeap)),
               "ID3D12Device::CreateDescriptorHeap", error)) {
        release(pipelineState);
        release(rootSignature);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    const UINT rtvIncrement = dx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvStart = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (const VernonColorAttachment *attachment : plan.attachments) {
        D3D12_CPU_DESCRIPTOR_HANDLE handle{rtvStart.ptr + attachment->location * rtvIncrement};
        dx.device->CreateRenderTargetView(directX12TextureState(*attachment->texture).resource, nullptr, handle);
    }

    ID3D12DescriptorHeap *resourceHeap = nullptr;
    ID3D12DescriptorHeap *samplerHeap = nullptr;
    if (!plan.sampledResources.empty()) {
        D3D12_DESCRIPTOR_HEAP_DESC description{};
        description.NumDescriptors = static_cast<UINT>(plan.sampledResources.size());
        description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        if (!check(dx.device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&resourceHeap)),
                   "ID3D12Device::CreateDescriptorHeap", error)) {
            release(rtvHeap);
            release(pipelineState);
            release(rootSignature);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        if (!check(dx.device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&samplerHeap)),
                   "ID3D12Device::CreateDescriptorHeap", error)) {
            release(resourceHeap);
            release(rtvHeap);
            release(pipelineState);
            release(rootSignature);
            return VERNON_STATUS_INTERNAL_ERROR;
        }
    }
    const UINT resourceIncrement = dx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT samplerIncrement = dx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    size_t descriptorIndex = 0;
    for (const auto &[binding, sampled] : plan.sampledResources) {
        VernonDeviceTexture &texture = *sampled.texture;
        DirectX12TextureState &textureState = directX12TextureState(texture);
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = textureState.format;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (texture.dimension == VERNON_TEXTURE_CUBE) {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            view.TextureCube.MipLevels = texture.mipLevels;
        } else if (texture.dimension == VERNON_TEXTURE_3D) {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            view.Texture3D.MipLevels = texture.mipLevels;
        } else {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Texture2D.MipLevels = texture.mipLevels;
        }
        D3D12_CPU_DESCRIPTOR_HANDLE resourceHandle = resourceHeap->GetCPUDescriptorHandleForHeapStart();
        resourceHandle.ptr += descriptorIndex * resourceIncrement;
        dx.device->CreateShaderResourceView(textureState.resource, &view, resourceHandle);
        const VernonSamplerDescriptor defaultSampler{
            sizeof(VernonSamplerDescriptor), VERNON_SAMPLER_REPEAT, VERNON_SAMPLER_REPEAT, VERNON_SAMPLER_REPEAT,
            VERNON_SAMPLER_LINEAR,           VERNON_SAMPLER_LINEAR, VERNON_SAMPLER_LINEAR, {0, 0, 0, 0}};
        const VernonSamplerDescriptor &source = sampled.sampler ? sampled.sampler->descriptor : defaultSampler;
        D3D12_SAMPLER_DESC sampler{};
        sampler.Filter = samplerFilter(source);
        sampler.AddressU = addressMode(source.wrap_u);
        sampler.AddressV = addressMode(source.wrap_v);
        sampler.AddressW = addressMode(source.wrap_w);
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        D3D12_CPU_DESCRIPTOR_HANDLE samplerHandle = samplerHeap->GetCPUDescriptorHandleForHeapStart();
        samplerHandle.ptr += descriptorIndex * samplerIncrement;
        dx.device->CreateSampler(&sampler, samplerHandle);
        ++descriptorIndex;
    }

    ID3D12Resource *vertexConstants = hasVertexUniforms ? createUpload(dx, vertexUniforms, error) : nullptr;
    ID3D12Resource *fragmentConstants = hasFragmentUniforms ? createUpload(dx, fragmentUniforms, error) : nullptr;
    const auto cleanup = [&]() {
        release(vertexConstants);
        release(fragmentConstants);
        release(samplerHeap);
        release(resourceHeap);
        release(rtvHeap);
        release(pipelineState);
        release(rootSignature);
    };
    if ((hasVertexUniforms && !vertexConstants) || (hasFragmentUniforms && !fragmentConstants)) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    if (!check(dx.allocator->Reset(), "ID3D12CommandAllocator::Reset", error) ||
        !check(dx.commands->Reset(dx.allocator, pipelineState), "ID3D12GraphicsCommandList::Reset", error)) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    for (const VernonColorAttachment *attachment : plan.attachments) {
        DirectX12TextureState &texture = directX12TextureState(*attachment->texture);
        transition(dx.commands, texture.resource, texture.state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }
    for (const auto &[binding, sampled] : plan.sampledResources) {
        DirectX12TextureState &texture = directX12TextureState(*sampled.texture);
        transition(dx.commands, texture.resource, texture.state,
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    dx.commands->SetGraphicsRootSignature(rootSignature);
    UINT rootIndex = 0;
    if (vertexConstants)
        dx.commands->SetGraphicsRootConstantBufferView(rootIndex++, vertexConstants->GetGPUVirtualAddress());
    if (fragmentConstants)
        dx.commands->SetGraphicsRootConstantBufferView(rootIndex++, fragmentConstants->GetGPUVirtualAddress());
    if (resourceHeap) {
        ID3D12DescriptorHeap *heaps[] = {resourceHeap, samplerHeap};
        dx.commands->SetDescriptorHeaps(2, heaps);
        D3D12_GPU_DESCRIPTOR_HANDLE resourceGpu = resourceHeap->GetGPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE samplerGpu = samplerHeap->GetGPUDescriptorHandleForHeapStart();
        for (size_t index = 0; index < plan.sampledResources.size(); ++index) {
            dx.commands->SetGraphicsRootDescriptorTable(rootIndex++, resourceGpu);
            dx.commands->SetGraphicsRootDescriptorTable(rootIndex++, samplerGpu);
            resourceGpu.ptr += resourceIncrement;
            samplerGpu.ptr += samplerIncrement;
        }
    }
    dx.commands->IASetPrimitiveTopology(topology(invocation.topology));
    if (!vertexViews.empty())
        dx.commands->IASetVertexBuffers(0, static_cast<UINT>(vertexViews.size()), vertexViews.data());
    D3D12_INDEX_BUFFER_VIEW indexView{};
    if (plan.indexBinding) {
        indexView.BufferLocation = directX12BufferState(*plan.indexBinding->buffer).resource->GetGPUVirtualAddress() +
                                   plan.indexBinding->offset;
        indexView.SizeInBytes = plan.indexBinding->index_count * sizeof(uint32_t);
        indexView.Format = DXGI_FORMAT_R32_UINT;
        dx.commands->IASetIndexBuffer(&indexView);
    }
    const D3D12_VIEWPORT viewport{
        static_cast<float>(invocation.viewport[0]),
        static_cast<float>(invocation.viewport[1]),
        static_cast<float>(invocation.viewport[2] ? invocation.viewport[2] : plan.attachmentWidth),
        static_cast<float>(invocation.viewport[3] ? invocation.viewport[3] : plan.attachmentHeight),
        0.0f,
        1.0f};
    const D3D12_RECT scissor{
        static_cast<LONG>(invocation.scissor[0]), static_cast<LONG>(invocation.scissor[1]),
        static_cast<LONG>(invocation.scissor[0] +
                          (invocation.scissor[2] ? invocation.scissor[2] : plan.attachmentWidth)),
        static_cast<LONG>(invocation.scissor[1] +
                          (invocation.scissor[3] ? invocation.scissor[3] : plan.attachmentHeight))};
    dx.commands->RSSetViewports(1, &viewport);
    dx.commands->RSSetScissorRects(1, &scissor);
    std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> renderTargets;
    renderTargets.reserve(pipeline.NumRenderTargets);
    for (UINT location = 0; location < pipeline.NumRenderTargets; ++location)
        renderTargets.push_back({rtvStart.ptr + location * rtvIncrement});
    dx.commands->OMSetRenderTargets(static_cast<UINT>(renderTargets.size()), renderTargets.data(), FALSE, nullptr);
    if (plan.indexBinding)
        dx.commands->DrawIndexedInstanced(plan.indexBinding->index_count, plan.instanceCount, 0, 0, 0);
    else
        dx.commands->DrawInstanced(plan.vertexCount, plan.instanceCount, 0, 0);
    for (const VernonColorAttachment *attachment : plan.attachments) {
        DirectX12TextureState &texture = directX12TextureState(*attachment->texture);
        transition(dx.commands, texture.resource, texture.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    if (!check(dx.commands->Close(), "ID3D12GraphicsCommandList::Close", error)) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    ID3D12CommandList *lists[] = {dx.commands};
    dx.queue->ExecuteCommandLists(1, lists);
    const VernonStatus sync = synchronizeDirectX12(context);
    cleanup();
    return sync;
}

} // namespace vernon::runtime

#else

namespace vernon::runtime {
VernonStatus encodeAndSubmitDirectX12Graphics(VernonRuntimeContext &, DirectX12PipelineState &, const Variant &,
                                              const VernonPipelineInvocation &, const PlannedGraphicsInvocation &,
                                              std::string &) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}
} // namespace vernon::runtime

#endif
