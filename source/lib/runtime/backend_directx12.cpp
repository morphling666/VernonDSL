#include "backend_directx12.h"
#include "runtime_test_hooks.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

namespace vernon::runtime {
namespace {

std::atomic<bool> forceWarpForTests{false};

template <typename T> void release(T *&value) {
    if (value)
        value->Release();
    value = nullptr;
}

bool failed(VernonRuntimeContext &context, HRESULT result, const char *operation) {
    if (SUCCEEDED(result))
        return false;
    context.error = std::string(operation) + " failed with HRESULT 0x";
    constexpr char digits[] = "0123456789abcdef";
    const uint32_t code = static_cast<uint32_t>(result);
    for (int shift = 28; shift >= 0; shift -= 4)
        context.error.push_back(digits[(code >> shift) & 0xf]);
    if (result == DXGI_ERROR_DEVICE_REMOVED && directX12State(context).device) {
        const HRESULT reason = directX12State(context).device->GetDeviceRemovedReason();
        context.error += " (device removed reason 0x";
        const uint32_t reasonCode = static_cast<uint32_t>(reason);
        for (int shift = 28; shift >= 0; shift -= 4)
            context.error.push_back(digits[(reasonCode >> shift) & 0xf]);
        context.error += ")";
    }
    return true;
}

D3D12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES result{};
    result.Type = type;
    result.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    result.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    result.CreationNodeMask = 1;
    result.VisibleNodeMask = 1;
    return result;
}

D3D12_RESOURCE_DESC bufferDescription(size_t size) {
    D3D12_RESOURCE_DESC result{};
    result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    result.Width = std::max<size_t>(4, (size + 3) & ~size_t{3});
    result.Height = 1;
    result.DepthOrArraySize = 1;
    result.MipLevels = 1;
    result.Format = DXGI_FORMAT_UNKNOWN;
    result.SampleDesc.Count = 1;
    result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    result.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return result;
}

bool createBufferResource(VernonRuntimeContext &context, size_t size, D3D12_HEAP_TYPE heapType,
                          D3D12_RESOURCE_STATES initialState, ID3D12Resource **resource, bool unorderedAccess) {
    D3D12_RESOURCE_DESC description = bufferDescription(size);
    if (!unorderedAccess)
        description.Flags = D3D12_RESOURCE_FLAG_NONE;
    const D3D12_HEAP_PROPERTIES heap = heapProperties(heapType);
    return !failed(context,
                   directX12State(context).device->CreateCommittedResource(
                       &heap, D3D12_HEAP_FLAG_NONE, &description, initialState, nullptr, IID_PPV_ARGS(resource)),
                   "ID3D12Device::CreateCommittedResource");
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

bool beginCommands(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    if (failed(context, state.allocator->Reset(), "ID3D12CommandAllocator::Reset") ||
        failed(context, state.commands->Reset(state.allocator, nullptr), "ID3D12GraphicsCommandList::Reset"))
        return false;
    return true;
}

bool submitCommands(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    if (failed(context, state.commands->Close(), "ID3D12GraphicsCommandList::Close"))
        return false;
    ID3D12CommandList *lists[] = {state.commands};
    state.queue->ExecuteCommandLists(1, lists);
    return synchronizeDirectX12(context) == VERNON_STATUS_OK;
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

bool createStaging(VernonRuntimeContext &context, size_t size, bool upload, ID3D12Resource **resource) {
    return createBufferResource(context, size, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK,
                                upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST, resource,
                                false);
}

} // namespace

bool probeDirectX12(std::string &diagnostic) {
    IDXGIFactory6 *factory = nullptr;
    const HRESULT result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(result)) {
        diagnostic = "CreateDXGIFactory1 failed";
        return false;
    }
    bool available = false;
    for (uint32_t index = 0; !available; ++index) {
        IDXGIAdapter1 *adapter = nullptr;
        if (factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) ==
            DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 description{};
        adapter->GetDesc1(&description);
        available = !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                    SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr));
        release(adapter);
    }
    release(factory);
    if (!available)
        diagnostic = "no D3D12-capable adapter was found";
    return available;
}

bool initializeDirectX12Context(VernonRuntimeContext &context, uint32_t deviceIndex) {
    auto *state = new DirectX12ContextState();
    installRuntimeBackendState(context, state);
    const auto cleanup = [&] {
        destroyDirectX12Context(context);
        destroyRuntimeBackendState(context);
        return false;
    };
    if (failed(context, CreateDXGIFactory1(IID_PPV_ARGS(&state->factory)), "CreateDXGIFactory1"))
        return cleanup();
    if (forceWarpForTests.load()) {
        if (deviceIndex != 0 || failed(context, state->factory->EnumWarpAdapter(IID_PPV_ARGS(&state->adapter)),
                                       "IDXGIFactory::EnumWarpAdapter"))
            return cleanup();
    }
    for (uint32_t index = 0; !state->adapter; ++index) {
        IDXGIAdapter1 *candidate = nullptr;
        if (state->factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                       IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 description{};
        candidate->GetDesc1(&description);
        if (!(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
            if (deviceIndex == 0) {
                state->adapter = candidate;
                break;
            }
            --deviceIndex;
        }
        release(candidate);
    }
    if (!state->adapter) {
        context.error = "requested D3D12 hardware adapter was not found";
        return cleanup();
    }
    static constexpr D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1,
                                                   D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1,
                                                   D3D_FEATURE_LEVEL_11_0};
    for (D3D_FEATURE_LEVEL level : levels)
        if (SUCCEEDED(D3D12CreateDevice(state->adapter, level, IID_PPV_ARGS(&state->device)))) {
            state->featureLevel = level;
            break;
        }
    if (!state->device) {
        context.error = "D3D12CreateDevice failed for the selected adapter";
        return cleanup();
    }
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_8};
    HRESULT shaderModelResult =
        state->device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
    while (shaderModelResult == E_INVALIDARG && shaderModel.HighestShaderModel > D3D_SHADER_MODEL_6_0) {
        shaderModel.HighestShaderModel =
            static_cast<D3D_SHADER_MODEL>(static_cast<uint32_t>(shaderModel.HighestShaderModel) - 1);
        shaderModelResult =
            state->device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
    }
    if (FAILED(shaderModelResult))
        shaderModel.HighestShaderModel = D3D_SHADER_MODEL_5_1;
    state->shaderModel = shaderModel.HighestShaderModel;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignature{D3D_ROOT_SIGNATURE_VERSION_1_1};
    if (FAILED(state->device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSignature, sizeof(rootSignature))))
        rootSignature.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    state->rootSignatureVersion = rootSignature.HighestVersion;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (SUCCEEDED(state->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
        state->resourceBindingTier = options.ResourceBindingTier;
    D3D12_COMMAND_QUEUE_DESC queueDescription{};
    queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (failed(context, state->device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&state->queue)),
               "ID3D12Device::CreateCommandQueue") ||
        failed(context,
               state->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&state->allocator)),
               "ID3D12Device::CreateCommandAllocator") ||
        failed(context,
               state->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, state->allocator, nullptr,
                                                IID_PPV_ARGS(&state->commands)),
               "ID3D12Device::CreateCommandList") ||
        failed(context, state->commands->Close(), "ID3D12GraphicsCommandList::Close") ||
        failed(context, state->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&state->fence)),
               "ID3D12Device::CreateFence"))
        return cleanup();
    state->fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!state->fenceEvent) {
        context.error = "CreateEventW failed for the D3D12 fence";
        return cleanup();
    }
    return true;
}

void setDirectX12WarpForTests(bool enabled) { forceWarpForTests.store(enabled); }

void destroyDirectX12Context(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    if (state.queue && state.fence)
        synchronizeDirectX12(context);
    if (state.fenceEvent)
        CloseHandle(state.fenceEvent);
    state.fenceEvent = nullptr;
    release(state.fence);
    release(state.commands);
    release(state.allocator);
    release(state.queue);
    release(state.device);
    release(state.adapter);
    release(state.factory);
}

VernonStatus synchronizeDirectX12(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    const uint64_t value = ++state.fenceValue;
    if (failed(context, state.queue->Signal(state.fence, value), "ID3D12CommandQueue::Signal"))
        return VERNON_STATUS_INTERNAL_ERROR;
    if (state.fence->GetCompletedValue() < value) {
        if (failed(context, state.fence->SetEventOnCompletion(value, state.fenceEvent),
                   "ID3D12Fence::SetEventOnCompletion"))
            return VERNON_STATUS_INTERNAL_ERROR;
        WaitForSingleObject(state.fenceEvent, INFINITE);
    }
    return VERNON_STATUS_OK;
}

bool createDirectX12Buffer(VernonDeviceBuffer &buffer) {
    auto *state = new DirectX12BufferState();
    installRuntimeBackendState(buffer, state);
    if (!createBufferResource(*buffer.context, buffer.size, D3D12_HEAP_TYPE_DEFAULT, state->state, &state->resource,
                              true)) {
        destroyRuntimeBackendState(buffer);
        return false;
    }
    return true;
}

void destroyDirectX12Buffer(VernonDeviceBuffer &buffer) {
    synchronizeDirectX12(*buffer.context);
    release(directX12BufferState(buffer).resource);
}

VernonStatus copyToDirectX12Buffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
    if (!source || offset > buffer.size || size > buffer.size - offset)
        return VERNON_STATUS_INVALID_ARGUMENT;
    ID3D12Resource *upload = nullptr;
    if (!createStaging(*buffer.context, size, true, &upload))
        return VERNON_STATUS_INTERNAL_ERROR;
    void *mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    if (failed(*buffer.context, upload->Map(0, &noRead, &mapped), "ID3D12Resource::Map")) {
        release(upload);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(mapped, source, size);
    upload->Unmap(0, nullptr);
    DirectX12BufferState &state = directX12BufferState(buffer);
    if (!beginCommands(*buffer.context)) {
        release(upload);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    transition(directX12State(*buffer.context).commands, state.resource, state.state, D3D12_RESOURCE_STATE_COPY_DEST);
    directX12State(*buffer.context).commands->CopyBufferRegion(state.resource, offset, upload, 0, size);
    transition(directX12State(*buffer.context).commands, state.resource, state.state,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const bool submitted = submitCommands(*buffer.context);
    release(upload);
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
}

VernonStatus copyFromDirectX12Buffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
    if (!destination || offset > buffer.size || size > buffer.size - offset)
        return VERNON_STATUS_INVALID_ARGUMENT;
    ID3D12Resource *readback = nullptr;
    if (!createStaging(*buffer.context, size, false, &readback))
        return VERNON_STATUS_INTERNAL_ERROR;
    auto &mutableBuffer = const_cast<VernonDeviceBuffer &>(buffer);
    DirectX12BufferState &state = directX12BufferState(mutableBuffer);
    if (!beginCommands(*buffer.context)) {
        release(readback);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    transition(directX12State(*buffer.context).commands, state.resource, state.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    directX12State(*buffer.context).commands->CopyBufferRegion(readback, 0, state.resource, offset, size);
    transition(directX12State(*buffer.context).commands, state.resource, state.state,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!submitCommands(*buffer.context)) {
        release(readback);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    void *mapped = nullptr;
    const D3D12_RANGE readRange{0, size};
    if (failed(*buffer.context, readback->Map(0, &readRange, &mapped), "ID3D12Resource::Map")) {
        release(readback);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(destination, mapped, size);
    readback->Unmap(0, nullptr);
    release(readback);
    return VERNON_STATUS_OK;
}

bool createDirectX12Texture(VernonDeviceTexture &texture) {
    const auto format = textureFormat(texture.format);
    if (!format) {
        texture.context->error = "D3D12 texture format is unsupported";
        return false;
    }
    auto *state = new DirectX12TextureState();
    state->format = *format;
    installRuntimeBackendState(texture, state);
    D3D12_RESOURCE_DESC description{};
    description.Dimension = texture.dimension == VERNON_TEXTURE_3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                                                   : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = texture.width;
    description.Height = texture.height;
    description.DepthOrArraySize = static_cast<UINT16>(
        texture.dimension == VERNON_TEXTURE_CUBE ? 6 : (texture.dimension == VERNON_TEXTURE_3D ? texture.depth : 1));
    description.MipLevels = static_cast<UINT16>(texture.mipLevels);
    description.Format = *format;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    description.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_HEAP_PROPERTIES heap = heapProperties(D3D12_HEAP_TYPE_DEFAULT);
    if (failed(*texture.context,
               directX12State(*texture.context)
                   .device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description, state->state, nullptr,
                                                    IID_PPV_ARGS(&state->resource)),
               "ID3D12Device::CreateCommittedResource")) {
        destroyRuntimeBackendState(texture);
        return false;
    }
    return true;
}

void destroyDirectX12Texture(VernonDeviceTexture &texture) {
    synchronizeDirectX12(*texture.context);
    release(directX12TextureState(texture).resource);
}

VernonStatus copyToDirectX12Texture(VernonDeviceTexture &texture, const void *source, size_t size) {
    if (!source || texture.mipLevels != 1)
        return VERNON_STATUS_INVALID_ARGUMENT;
    DirectX12ContextState &context = directX12State(*texture.context);
    DirectX12TextureState &state = directX12TextureState(texture);
    const D3D12_RESOURCE_DESC description = state.resource->GetDesc();
    const UINT subresourceCount = texture.dimension == VERNON_TEXTURE_CUBE ? 6u : 1u;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(subresourceCount);
    std::vector<UINT> rows(subresourceCount);
    std::vector<UINT64> rowBytes(subresourceCount);
    UINT64 required = 0;
    context.device->GetCopyableFootprints(&description, 0, subresourceCount, 0, footprints.data(), rows.data(),
                                          rowBytes.data(), &required);
    size_t tightSize = 0;
    for (UINT subresource = 0; subresource < subresourceCount; ++subresource)
        tightSize +=
            static_cast<size_t>(rowBytes[subresource]) * rows[subresource] * footprints[subresource].Footprint.Depth;
    if (size != tightSize)
        return VERNON_STATUS_INVALID_ARGUMENT;
    ID3D12Resource *upload = nullptr;
    if (!createStaging(*texture.context, static_cast<size_t>(required), true, &upload))
        return VERNON_STATUS_INTERNAL_ERROR;
    uint8_t *mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    upload->Map(0, &noRead, reinterpret_cast<void **>(&mapped));
    size_t sourceOffset = 0;
    for (UINT subresource = 0; subresource < subresourceCount; ++subresource) {
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint = footprints[subresource];
        for (UINT depth = 0; depth < footprint.Footprint.Depth; ++depth)
            for (UINT row = 0; row < rows[subresource]; ++row) {
                std::memcpy(mapped + footprint.Offset + depth * footprint.Footprint.RowPitch * rows[subresource] +
                                row * footprint.Footprint.RowPitch,
                            static_cast<const uint8_t *>(source) + sourceOffset,
                            static_cast<size_t>(rowBytes[subresource]));
                sourceOffset += static_cast<size_t>(rowBytes[subresource]);
            }
    }
    upload->Unmap(0, nullptr);
    beginCommands(*texture.context);
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_COPY_DEST);
    for (UINT subresource = 0; subresource < subresourceCount; ++subresource) {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = state.resource;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = subresource;
        D3D12_TEXTURE_COPY_LOCATION staging{};
        staging.pResource = upload;
        staging.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        staging.PlacedFootprint = footprints[subresource];
        context.commands->CopyTextureRegion(&destination, 0, 0, 0, &staging, nullptr);
    }
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    const bool submitted = submitCommands(*texture.context);
    release(upload);
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
}

VernonStatus copyFromDirectX12Texture(const VernonDeviceTexture &texture, void *destination, size_t size) {
    if (!destination || texture.dimension != VERNON_TEXTURE_2D || texture.mipLevels != 1)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto &mutableTexture = const_cast<VernonDeviceTexture &>(texture);
    DirectX12ContextState &context = directX12State(*texture.context);
    DirectX12TextureState &state = directX12TextureState(mutableTexture);
    const D3D12_RESOURCE_DESC description = state.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowBytes = 0;
    UINT64 required = 0;
    context.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows, &rowBytes, &required);
    if (size != static_cast<size_t>(rowBytes) * rows)
        return VERNON_STATUS_INVALID_ARGUMENT;
    ID3D12Resource *readback = nullptr;
    if (!createStaging(*texture.context, static_cast<size_t>(required), false, &readback))
        return VERNON_STATUS_INTERNAL_ERROR;
    beginCommands(*texture.context);
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION destinationLocation{};
    destinationLocation.pResource = readback;
    destinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destinationLocation.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION sourceLocation{};
    sourceLocation.pResource = state.resource;
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    context.commands->CopyTextureRegion(&destinationLocation, 0, 0, 0, &sourceLocation, nullptr);
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    if (!submitCommands(*texture.context)) {
        release(readback);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    uint8_t *mapped = nullptr;
    const D3D12_RANGE readRange{0, static_cast<SIZE_T>(required)};
    readback->Map(0, &readRange, reinterpret_cast<void **>(&mapped));
    for (UINT row = 0; row < rows; ++row)
        std::memcpy(static_cast<uint8_t *>(destination) + row * rowBytes, mapped + row * footprint.Footprint.RowPitch,
                    static_cast<size_t>(rowBytes));
    readback->Unmap(0, nullptr);
    release(readback);
    return VERNON_STATUS_OK;
}

bool createDirectX12Sampler(VernonDeviceSampler &sampler) {
    installRuntimeBackendState(sampler, new DirectX12SamplerState());
    return true;
}

void destroyDirectX12Sampler(VernonDeviceSampler &) {}

bool loadDirectX12Kernel(VernonRuntimeContext &context, const void *artifact, size_t artifactSize,
                         const char *reflection, size_t reflectionSize, const char *entry, size_t entrySize,
                         DirectX12KernelState &state, ReflectedEntry &metadata) {
    if (!artifact || artifactSize < 4 || std::memcmp(artifact, "DXBC", 4) != 0) {
        context.error = "DirectX artifact is not a DXIL container";
        return false;
    }
    const nlohmann::json parsed = nlohmann::json::parse(reflection, reflection + reflectionSize, nullptr, false);
    if (parsed.is_discarded() || !parseReflection(parsed, std::string(entry, entrySize), metadata, context.error))
        return false;
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    uint32_t fallbackBinding = 0;
    for (ReflectedArgument &argument : metadata.arguments) {
        if (argument.kind == "builtin")
            continue;
        if (argument.kind == "tensor" && !argument.storageLeaves.empty()) {
            for (const ReflectedStorageLeaf &leaf : argument.storageLeaves) {
                D3D12_DESCRIPTOR_RANGE range{};
                range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                range.NumDescriptors = 1;
                range.BaseShaderRegister = leaf.binding;
                range.RegisterSpace = argument.descriptorSet;
                range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                ranges.push_back(range);
            }
        } else {
            if (argument.binding == UINT32_MAX)
                argument.binding = fallbackBinding;
            D3D12_DESCRIPTOR_RANGE range{};
            range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            range.NumDescriptors = 1;
            range.BaseShaderRegister = argument.binding;
            range.RegisterSpace = argument.descriptorSet;
            range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            ranges.push_back(range);
        }
        fallbackBinding += static_cast<uint32_t>(std::max(argument.storageLeaves.size(), size_t{1}));
    }
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(ranges.size());
    parameter.DescriptorTable.pDescriptorRanges = ranges.data();
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC root{};
    root.NumParameters = ranges.empty() ? 0u : 1u;
    root.pParameters = ranges.empty() ? nullptr : &parameter;
    ID3DBlob *serialized = nullptr;
    ID3DBlob *errors = nullptr;
    HRESULT result = D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(result)) {
        context.error =
            errors ? std::string(static_cast<const char *>(errors->GetBufferPointer()), errors->GetBufferSize())
                   : "D3D12SerializeRootSignature failed";
        release(errors);
        release(serialized);
        return false;
    }
    release(errors);
    if (failed(context,
               directX12State(context).device->CreateRootSignature(
                   0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&state.rootSignature)),
               "ID3D12Device::CreateRootSignature")) {
        release(serialized);
        return false;
    }
    release(serialized);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = state.rootSignature;
    pipeline.CS = {artifact, artifactSize};
    if (failed(context,
               directX12State(context).device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&state.pipeline)),
               "ID3D12Device::CreateComputePipelineState")) {
        destroyDirectX12Kernel(context, state);
        return false;
    }
    state.descriptorCount = static_cast<uint32_t>(ranges.size());
    return true;
}

void destroyDirectX12Kernel(VernonRuntimeContext &context, DirectX12KernelState &state) {
    synchronizeDirectX12(context);
    release(state.pipeline);
    release(state.rootSignature);
    state.descriptorCount = 0;
}

VernonStatus launchDirectX12Kernel(VernonRuntimeContext &context, const DirectX12KernelState &state,
                                   const ReflectedEntry &metadata, VernonLaunchSize globalSize,
                                   const VernonLaunchArgument *arguments, size_t argumentCount) {
    size_t expectedArguments = 0;
    for (const ReflectedArgument &argument : metadata.arguments)
        if (argument.kind != "builtin")
            ++expectedArguments;
    if (argumentCount != expectedArguments || (!arguments && argumentCount)) {
        context.error = "D3D12 compute argument count does not match reflection";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    DirectX12ContextState &contextState = directX12State(context);
    ID3D12DescriptorHeap *heap = nullptr;
    std::vector<ID3D12Resource *> scalarResources;
    if (state.descriptorCount) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
        heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDescription.NumDescriptors = state.descriptorCount;
        heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (failed(context, contextState.device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&heap)),
                   "ID3D12Device::CreateDescriptorHeap"))
            return VERNON_STATUS_INTERNAL_ERROR;
    }
    const UINT increment =
        contextState.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        heap ? heap->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE{};
    size_t reflectedIndex = 0;
    for (size_t argumentIndex = 0; argumentIndex < argumentCount; ++argumentIndex) {
        while (metadata.arguments[reflectedIndex].kind == "builtin")
            ++reflectedIndex;
        const ReflectedArgument &reflected = metadata.arguments[reflectedIndex++];
        const VernonLaunchArgument &argument = arguments[argumentIndex];
        ID3D12Resource *resource = nullptr;
        size_t resourceSize = 0;
        if (argument.kind == VERNON_LAUNCH_TENSOR && argument.buffer && argument.buffer->context == &context) {
            resource = directX12BufferState(*argument.buffer).resource;
            resourceSize = argument.buffer->size;
        } else if (argument.kind == VERNON_LAUNCH_SCALAR && argument.scalar_data && argument.scalar_size) {
            VernonDeviceBuffer temporary{};
            temporary.context = &context;
            temporary.size = argument.scalar_size;
            temporary.alignment = 4;
            if (!createDirectX12Buffer(temporary) ||
                copyToDirectX12Buffer(temporary, 0, argument.scalar_data, argument.scalar_size) != VERNON_STATUS_OK) {
                if (temporary.backendState) {
                    resource = directX12BufferState(temporary).resource;
                    directX12BufferState(temporary).resource = nullptr;
                    release(resource);
                    destroyRuntimeBackendState(temporary);
                }
                release(heap);
                for (ID3D12Resource *value : scalarResources)
                    release(value);
                return VERNON_STATUS_INTERNAL_ERROR;
            }
            resource = directX12BufferState(temporary).resource;
            directX12BufferState(temporary).resource = nullptr;
            destroyRuntimeBackendState(temporary);
            scalarResources.push_back(resource);
            resourceSize = argument.scalar_size;
        } else {
            context.error = "D3D12 compute argument has an invalid kind or resource";
            release(heap);
            for (ID3D12Resource *value : scalarResources)
                release(value);
            return VERNON_STATUS_INVALID_ARGUMENT;
        }
        const size_t descriptors = std::max(reflected.storageLeaves.size(), size_t{1});
        for (size_t leaf = 0; leaf < descriptors; ++leaf) {
            const size_t byteOffset = reflected.storageLeaves.empty() ? 0 : reflected.storageLeaves[leaf].byteOffset;
            const size_t byteSize = byteOffset < resourceSize ? resourceSize - byteOffset : 0;
            if (!byteSize) {
                context.error = "D3D12 compute reflection points outside its argument buffer";
                release(heap);
                for (ID3D12Resource *value : scalarResources)
                    release(value);
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            view.Format = DXGI_FORMAT_R32_TYPELESS;
            view.Buffer.FirstElement = byteOffset / 4;
            view.Buffer.NumElements = static_cast<UINT>(std::max<size_t>(1, (byteSize + 3) / 4));
            view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            contextState.device->CreateUnorderedAccessView(resource, nullptr, &view, handle);
            handle.ptr += increment;
        }
    }
    if (!beginCommands(context)) {
        release(heap);
        for (ID3D12Resource *value : scalarResources)
            release(value);
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    contextState.commands->SetComputeRootSignature(state.rootSignature);
    contextState.commands->SetPipelineState(state.pipeline);
    if (heap) {
        contextState.commands->SetDescriptorHeaps(1, &heap);
        contextState.commands->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    }
    const UINT groupsX = (globalSize.x + metadata.workgroup[0] - 1) / metadata.workgroup[0];
    const UINT groupsY = (globalSize.y + metadata.workgroup[1] - 1) / metadata.workgroup[1];
    const UINT groupsZ = (globalSize.z + metadata.workgroup[2] - 1) / metadata.workgroup[2];
    contextState.commands->Dispatch(groupsX, groupsY, groupsZ);
    const bool submitted = submitCommands(context);
    release(heap);
    for (ID3D12Resource *value : scalarResources)
        release(value);
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
}

} // namespace vernon::runtime

#else

namespace vernon::runtime {
bool probeDirectX12(std::string &diagnostic) {
    diagnostic = "VernonRuntime was built without DirectX 12 support";
    return false;
}
bool initializeDirectX12Context(VernonRuntimeContext &, uint32_t) { return false; }
void destroyDirectX12Context(VernonRuntimeContext &) {}
VernonStatus synchronizeDirectX12(VernonRuntimeContext &) { return VERNON_STATUS_UNSUPPORTED_TARGET; }
bool createDirectX12Buffer(VernonDeviceBuffer &) { return false; }
void destroyDirectX12Buffer(VernonDeviceBuffer &) {}
VernonStatus copyToDirectX12Buffer(VernonDeviceBuffer &, size_t, const void *, size_t) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}
VernonStatus copyFromDirectX12Buffer(const VernonDeviceBuffer &, size_t, void *, size_t) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}
bool createDirectX12Texture(VernonDeviceTexture &) { return false; }
void destroyDirectX12Texture(VernonDeviceTexture &) {}
VernonStatus copyToDirectX12Texture(VernonDeviceTexture &, const void *, size_t) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}
VernonStatus copyFromDirectX12Texture(const VernonDeviceTexture &, void *, size_t) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}
bool createDirectX12Sampler(VernonDeviceSampler &) { return false; }
void destroyDirectX12Sampler(VernonDeviceSampler &) {}
bool loadDirectX12Kernel(VernonRuntimeContext &, const void *, size_t, const char *, size_t, const char *, size_t,
                         DirectX12KernelState &, ReflectedEntry &) {
    return false;
}
void destroyDirectX12Kernel(VernonRuntimeContext &, DirectX12KernelState &) {}
VernonStatus launchDirectX12Kernel(VernonRuntimeContext &, const DirectX12KernelState &, const ReflectedEntry &,
                                   VernonLaunchSize, const VernonLaunchArgument *, size_t) {
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}
} // namespace vernon::runtime

#endif
