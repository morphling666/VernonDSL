#include "backend_directx12.h"
#include "rhi_adapter/adapter_internal.h"
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
using rhi::directx12::transition;

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

bool beginCommands(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    return state.beginCommands(context.error);
}

bool submitCommands(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    return state.submitCommands(context.error);
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

D3D12_TEXTURE_ADDRESS_MODE samplerAddressMode(VernonSamplerWrapMode mode) {
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
    if (!state->initialize(deviceIndex, forceWarpForTests.load(), context.error)) {
        delete state;
        return false;
    }
    state->adapter = createBorrowedDirectX12RhiAdapter(*state);
    if (!state->adapter) {
        context.error = "failed to create D3D12 Runtime RHI adapter";
        delete state;
        return false;
    }
    installRuntimeBackendState(context, state);
    return true;
}

void setDirectX12WarpForTests(bool enabled) { forceWarpForTests.store(enabled); }

void destroyDirectX12Context(VernonRuntimeContext &context) {
    DirectX12ContextState &state = directX12State(context);
    vernonRuntimeRhiAdapterDestroy(state.adapter);
    state.adapter = nullptr;
    state.shutdown();
}

VernonStatus synchronizeDirectX12(VernonRuntimeContext &context) {
    return directX12State(context).synchronize(context.error) ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
}

bool createDirectX12Buffer(VernonDeviceBuffer &buffer) {
    auto *state = new DirectX12BufferState();
    if (!directX12State(*buffer.context)
             .createBuffer(*state, buffer.size, true, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                           buffer.context->error)) {
        delete state;
        return false;
    }
    installRuntimeBackendState(buffer, state);
    return true;
}

void destroyDirectX12Buffer(VernonDeviceBuffer &buffer) {
    synchronizeDirectX12(*buffer.context);
    directX12State(*buffer.context).destroyBuffer(directX12BufferState(buffer));
}

VernonStatus copyToDirectX12Buffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
    if (!source || offset > buffer.size || size > buffer.size - offset)
        return VERNON_STATUS_INVALID_ARGUMENT;
    DirectX12ContextState &context = directX12State(*buffer.context);
    ID3D12Resource *upload = nullptr;
    size_t uploadOffset = 0;
    uint8_t *mapped = nullptr;
    if (!context.acquireStaging(true, size, 256, upload, uploadOffset, mapped, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(mapped, source, size);
    DirectX12BufferState &state = directX12BufferState(buffer);
    if (!beginCommands(*buffer.context))
        return VERNON_STATUS_INTERNAL_ERROR;
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_COPY_DEST);
    context.commands->CopyBufferRegion(state.resource, offset, upload, uploadOffset, size);
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const bool submitted = submitCommands(*buffer.context);
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
}

VernonStatus copyFromDirectX12Buffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
    if (!destination || offset > buffer.size || size > buffer.size - offset)
        return VERNON_STATUS_INVALID_ARGUMENT;
    DirectX12ContextState &context = directX12State(*buffer.context);
    ID3D12Resource *readback = nullptr;
    size_t readbackOffset = 0;
    uint8_t *mapped = nullptr;
    if (!context.acquireStaging(false, size, 256, readback, readbackOffset, mapped, buffer.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    auto &mutableBuffer = const_cast<VernonDeviceBuffer &>(buffer);
    DirectX12BufferState &state = directX12BufferState(mutableBuffer);
    if (!beginCommands(*buffer.context))
        return VERNON_STATUS_INTERNAL_ERROR;
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    context.commands->CopyBufferRegion(readback, readbackOffset, state.resource, offset, size);
    transition(context.commands, state.resource, state.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!submitCommands(*buffer.context))
        return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(destination, mapped, size);
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
    if (!directX12State(*texture.context)
             .createImage(*state, description, *format, D3D12_RESOURCE_STATE_COMMON, texture.context->error)) {
        destroyRuntimeBackendState(texture);
        return false;
    }
    return true;
}

void destroyDirectX12Texture(VernonDeviceTexture &texture) {
    synchronizeDirectX12(*texture.context);
    directX12State(*texture.context).destroyImage(directX12TextureState(texture));
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
    size_t uploadOffset = 0;
    uint8_t *mapped = nullptr;
    if (!context.acquireStaging(true, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, upload,
                                uploadOffset, mapped, texture.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    context.device->GetCopyableFootprints(&description, 0, subresourceCount, uploadOffset, footprints.data(),
                                          rows.data(), rowBytes.data(), nullptr);
    size_t sourceOffset = 0;
    for (UINT subresource = 0; subresource < subresourceCount; ++subresource) {
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT &footprint = footprints[subresource];
        for (UINT depth = 0; depth < footprint.Footprint.Depth; ++depth)
            for (UINT row = 0; row < rows[subresource]; ++row) {
                std::memcpy(
                    mapped + (footprint.Offset - uploadOffset) +
                        depth * footprint.Footprint.RowPitch * rows[subresource] + row * footprint.Footprint.RowPitch,
                    static_cast<const uint8_t *>(source) + sourceOffset, static_cast<size_t>(rowBytes[subresource]));
                sourceOffset += static_cast<size_t>(rowBytes[subresource]);
            }
    }
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
    size_t readbackOffset = 0;
    uint8_t *mapped = nullptr;
    if (!context.acquireStaging(false, static_cast<size_t>(required), D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, readback,
                                readbackOffset, mapped, texture.context->error))
        return VERNON_STATUS_INTERNAL_ERROR;
    context.device->GetCopyableFootprints(&description, 0, 1, readbackOffset, &footprint, &rows, &rowBytes, nullptr);
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
    if (!submitCommands(*texture.context))
        return VERNON_STATUS_INTERNAL_ERROR;
    for (UINT row = 0; row < rows; ++row)
        std::memcpy(static_cast<uint8_t *>(destination) + row * rowBytes,
                    mapped + (footprint.Offset - readbackOffset) + row * footprint.Footprint.RowPitch,
                    static_cast<size_t>(rowBytes));
    return VERNON_STATUS_OK;
}

bool createDirectX12Sampler(VernonDeviceSampler &sampler) {
    auto *state = new DirectX12SamplerState();
    state->descriptor.Filter = samplerFilter(sampler.descriptor);
    state->descriptor.AddressU = samplerAddressMode(sampler.descriptor.wrap_u);
    state->descriptor.AddressV = samplerAddressMode(sampler.descriptor.wrap_v);
    state->descriptor.AddressW = samplerAddressMode(sampler.descriptor.wrap_w);
    state->descriptor.MaxAnisotropy = 1;
    state->descriptor.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    state->descriptor.MinLOD = 0;
    state->descriptor.MaxLOD = D3D12_FLOAT32_MAX;
    installRuntimeBackendState(sampler, state);
    return true;
}

void destroyDirectX12Sampler(VernonDeviceSampler &) {}

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
} // namespace vernon::runtime

#endif
