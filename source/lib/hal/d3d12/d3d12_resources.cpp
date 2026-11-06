#include "hal/d3d12/d3d12_resources.h"

#include "VernonFormats.hpp"
#include "hal/d3d12/d3d12_state.h"
#include "hal/native_buffers.h"
#include "hal/native_images.h"
#include "hal/resources.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace vernon::hal {
namespace {

using d3d12::DeviceState;

Serials<d3d12::Buffer> &serials() {
    static Serials<d3d12::Buffer> instance;
    return instance;
}

template <typename T = void> Done<T> internal(const char *what, const std::string &detail) {
    return fail<T>(VERNON_HAL_INTERNAL_ERROR, detail.empty() ? std::string(what) : std::string(what) + ": " + detail);
}

uint32_t mipExtent(uint32_t size, uint32_t level) { return (std::max)(size >> level, 1u); }

size_t alignUp(size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

// The format an image's resource is created in: the typeless format of its family where views reinterpret it, so
// that an 8-bit RGBA or BGRA image has UNORM and sRGB views and a sampled depth image has a color view.
DXGI_FORMAT resourceFormat(DXGI_FORMAT format, uint32_t usage) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:
        return (usage & VERNON_IMAGE_SAMPLED) ? DXGI_FORMAT_R16_TYPELESS : format;
    case DXGI_FORMAT_D32_FLOAT:
        return (usage & VERNON_IMAGE_SAMPLED) ? DXGI_FORMAT_R32_TYPELESS : format;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return (usage & VERNON_IMAGE_SAMPLED) ? DXGI_FORMAT_R32G8X24_TYPELESS : format;
    default:
        return format;
    }
}

// The color format a shader reads a depth format's depth through.
DXGI_FORMAT depthColorFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_D16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_D32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    default:
        return format;
    }
}

// UAVs have no sRGB formats, so an sRGB image is stored through its UNORM view.
DXGI_FORMAT linearFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    default:
        return format;
    }
}

D3D12_RESOURCE_BARRIER transitionBarrier(ID3D12Resource *resource, uint32_t subresource, D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = subresource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

// Moves `subresources` of `image`, which tracks their states, to `target`. Runs under the device's lock.
void transition(ID3D12GraphicsCommandList *commands, d3d12::Image &image, const std::vector<uint32_t> &subresources,
                D3D12_RESOURCE_STATES target) {
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (const uint32_t index : subresources) {
        D3D12_RESOURCE_STATES &state = image.states[index];
        if (state == target)
            continue;
        barriers.push_back(transitionBarrier(image.resource, index, state, target));
        state = target;
    }
    if (!barriers.empty())
        commands->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
}

std::vector<uint32_t> allSubresources(const d3d12::Image &image) {
    std::vector<uint32_t> result(image.states.size());
    for (size_t index = 0; index < result.size(); ++index)
        result[index] = static_cast<uint32_t>(index);
    return result;
}

std::vector<uint32_t> levelSubresources(const d3d12::Image &image, uint32_t mip) {
    std::vector<uint32_t> result;
    for (uint32_t plane = 0; plane < image.planes; ++plane)
        for (uint32_t layer = 0; layer < image.layers; ++layer)
            result.push_back(image.subresource(mip, layer, plane));
    return result;
}

// One plane of an image transfer, staged in rows of `RowPitch` bytes. A depth-stencil image copies whole
// subresources, so its box is the whole mip level and the region lies inside it.
struct PlaneCopy {
    size_t transfer{};
    uint32_t plane{};
    uint32_t subresource{};
    uint32_t texelBytes{};
    // Within the copied box.
    uint32_t regionX{};
    uint32_t regionY{};
    D3D12_BOX box{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
};

// Lays out the plane copies of `regions` from offset 0 of a staging allocation; false with `count` zero when a
// region's data layout does not match the image, or with `unsupported` when D3D12 cannot copy it.
bool planCopies(ID3D12Device *device, const VernonImageDescriptor &descriptor, const d3d12::Image &image,
                const VernonHalImageRegion *const *regions, size_t count, bool writes, std::vector<PlaneCopy> &copies,
                size_t &total, bool &unsupported) {
    unsupported = false;
    total = 0;
    const D3D12_RESOURCE_DESC resource = image.resource->GetDesc();
    const bool depth = formats::hasDepth(descriptor.format);
    for (size_t index = 0; index < count; ++index) {
        const VernonHalImageRegion &region = *regions[index];
        const auto bytes = regionBytes(descriptor, region);
        if (!bytes)
            return false;
        const uint32_t width = mipExtent(descriptor.extent.width, region.mip_level);
        const uint32_t height = mipExtent(descriptor.extent.height, region.mip_level);
        if (depth && writes &&
            (region.offset_x || region.offset_y || region.width != width || region.height != height)) {
            unsupported = true;
            return false;
        }
        const size_t texels = size_t{region.width} * region.height * region.depth;
        uint32_t planes[2]{};
        uint32_t planeCount = 0;
        if (region.aspect & (VERNON_IMAGE_ASPECT_COLOR | VERNON_IMAGE_ASPECT_DEPTH))
            planes[planeCount++] = 0;
        if (region.aspect & VERNON_IMAGE_ASPECT_STENCIL)
            planes[planeCount++] = 1;
        for (uint32_t which = 0; which < planeCount; ++which) {
            PlaneCopy copy;
            copy.transfer = index;
            copy.plane = planes[which];
            copy.subresource = image.subresource(region.mip_level, region.array_layer, copy.plane);
            copy.texelBytes = !formats::hasStencil(descriptor.format) ? static_cast<uint32_t>(*bytes / texels)
                              : copy.plane == 0                       ? 4u
                                                                      : 1u;
            if (depth) {
                copy.regionX = region.offset_x;
                copy.regionY = region.offset_y;
                copy.box = {0, 0, 0, width, height, 1};
            } else {
                copy.box = {region.offset_x,
                            region.offset_y,
                            region.offset_z,
                            region.offset_x + region.width,
                            region.offset_y + region.height,
                            region.offset_z + region.depth};
            }
            device->GetCopyableFootprints(&resource, copy.subresource, 1, 0, &copy.layout, nullptr, nullptr, nullptr);
            D3D12_SUBRESOURCE_FOOTPRINT &footprint = copy.layout.Footprint;
            footprint.Width = copy.box.right - copy.box.left;
            footprint.Height = copy.box.bottom - copy.box.top;
            footprint.Depth = copy.box.back - copy.box.front;
            footprint.RowPitch = static_cast<UINT>(
                alignUp(size_t{footprint.Width} * copy.texelBytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
            total = alignUp(total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            copy.layout.Offset = total;
            total += size_t{footprint.RowPitch} * footprint.Height * footprint.Depth;
            copies.push_back(copy);
        }
    }
    return true;
}

// Copies the region's texels of one plane between the host data, whose texels pack depth and stencil into
// kPackedDepthStencilBytes when the region transfers both, and the staged rows at `staged`.
void moveTexels(const PlaneCopy &copy, const VernonHalImageRegion &region, uint8_t *staged, uint8_t *host,
                bool toStaging) {
    const bool packed = region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
    const size_t hostStride = packed ? kPackedDepthStencilBytes : copy.texelBytes;
    const size_t hostOffset = packed && copy.plane == 1 ? sizeof(float) : 0;
    const size_t pitch = copy.layout.Footprint.RowPitch;
    const size_t slice = pitch * copy.layout.Footprint.Height;
    for (uint32_t z = 0; z < region.depth; ++z)
        for (uint32_t y = 0; y < region.height; ++y) {
            uint8_t *row = staged + z * slice + (y + copy.regionY) * pitch + size_t{copy.regionX} * copy.texelBytes;
            uint8_t *hostRow = host + ((size_t{z} * region.height + y) * region.width) * hostStride + hostOffset;
            if (hostStride == copy.texelBytes) {
                const size_t bytes = size_t{region.width} * copy.texelBytes;
                toStaging ? std::memcpy(row, hostRow, bytes) : std::memcpy(hostRow, row, bytes);
                continue;
            }
            for (uint32_t x = 0; x < region.width; ++x) {
                uint8_t *texel = row + size_t{x} * copy.texelBytes;
                uint8_t *hostTexel = hostRow + x * hostStride;
                toStaging ? std::memcpy(texel, hostTexel, copy.texelBytes)
                          : std::memcpy(hostTexel, texel, copy.texelBytes);
            }
        }
}

Result<void, std::string> writeBuffer(DeviceState &state, d3d12::Buffer &buffer, const VernonHalBufferWrite *writes,
                                      size_t count) {
    using Written = Result<void, std::string>;
    if (buffer.memory == VERNON_HAL_MEMORY_UPLOAD) {
        for (size_t index = 0; index < count; ++index)
            std::memcpy(buffer.mapped + writes[index].offset, writes[index].source,
                        static_cast<size_t>(writes[index].size));
        return Written{ok()};
    }
    std::vector<size_t> offsets(count);
    size_t total = 0;
    for (size_t index = 0; index < count; ++index) {
        offsets[index] = alignUp(total, 16);
        total = offsets[index] + static_cast<size_t>(writes[index].size);
    }
    if (!total)
        return Written{ok()};
    std::lock_guard<std::mutex> guard(state.mutex);
    std::string error;
    d3d12::CommandFrame *frame = nullptr;
    ID3D12Resource *staging = nullptr;
    size_t stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.uploadCommands(frame, error) ||
        !state.acquireStaging(*frame, true, total, 16, staging, stagingOffset, mapped, error))
        return Written{err(std::move(error))};
    // Copies move bytes, and buffers promote from COMMON to COPY_DEST on their own.
    for (size_t index = 0; index < count; ++index) {
        std::memcpy(mapped + offsets[index], writes[index].source, static_cast<size_t>(writes[index].size));
        frame->commands->CopyBufferRegion(buffer.resource, writes[index].offset, staging,
                                          stagingOffset + offsets[index], writes[index].size);
    }
    return Written{ok()};
}

Result<void, std::string> readBuffer(DeviceState &state, d3d12::Buffer &buffer, const VernonHalBufferRead *reads,
                                     size_t count) {
    using Read = Result<void, std::string>;
    std::lock_guard<std::mutex> guard(state.mutex);
    std::string error;
    // A readback buffer is never a copy source; once the queue drains, its mapping holds what was written to it.
    if (buffer.memory == VERNON_HAL_MEMORY_READBACK) {
        if (!state.synchronize(error))
            return Read{err(std::move(error))};
        for (size_t index = 0; index < count; ++index)
            std::memcpy(reads[index].destination, buffer.mapped + reads[index].offset,
                        static_cast<size_t>(reads[index].size));
        return Read{ok()};
    }
    std::vector<size_t> offsets(count);
    size_t total = 0;
    for (size_t index = 0; index < count; ++index) {
        offsets[index] = alignUp(total, 16);
        total = offsets[index] + static_cast<size_t>(reads[index].size);
    }
    if (!total)
        return Read{ok()};
    d3d12::CommandFrame *frame = nullptr;
    if (!state.beginCommands(frame, error))
        return Read{err(std::move(error))};
    ID3D12Resource *staging = nullptr;
    size_t stagingOffset = 0;
    uint8_t *mapped = nullptr;
    if (!state.acquireStaging(*frame, false, total, 16, staging, stagingOffset, mapped, error)) {
        state.abandonCommands(frame);
        return Read{err(std::move(error))};
    }
    for (size_t index = 0; index < count; ++index)
        frame->commands->CopyBufferRegion(staging, stagingOffset + offsets[index], buffer.resource, reads[index].offset,
                                          reads[index].size);
    uint64_t value = 0;
    if (!state.submitCommands(frame, value, error) || !state.waitForValue(value, error))
        return Read{err(std::move(error))};
    for (size_t index = 0; index < count; ++index)
        std::memcpy(reads[index].destination, mapped + offsets[index], static_cast<size_t>(reads[index].size));
    return Read{ok()};
}

// One D3D12 buffer, named by its serial.
class NativeBuffer {
public:
    NativeBuffer(std::shared_ptr<DeviceState> state, d3d12::Buffer buffer, uint64_t size)
        : state_(std::move(state)), identity_(nativeBufferIdentity(state_.get())), buffer_(buffer), size_(size) {
        serial_ = serials().add(&buffer_);
    }
    NativeBuffer(const NativeBuffer &) = delete;
    NativeBuffer &operator=(const NativeBuffer &) = delete;
    ~NativeBuffer() {
        serials().remove(serial_);
        std::lock_guard<std::mutex> guard(state_->mutex);
        state_->destroyBuffer(buffer_);
    }

    d3d12::Buffer &buffer() { return buffer_; }
    uint64_t size() const { return size_; }
    VernonRuntimeProviderResourceReference reference(uint64_t offset, uint64_t size) const {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = identity_;
        reference.resource.value = serial_;
        reference.offset = offset;
        reference.size = size;
        return reference;
    }

private:
    std::shared_ptr<DeviceState> state_;
    uint64_t identity_;
    d3d12::Buffer buffer_;
    uint64_t size_;
    uint64_t serial_{};
};

struct D3D12HalBuffer final : VernonHalBuffer {
    D3D12HalBuffer(std::shared_ptr<DeviceState> state, d3d12::Buffer buffer, uint64_t size)
        : native(std::move(state), buffer, size) {}

    NativeBuffer native;
};

struct D3D12Image final : VernonHalImage {
    explicit D3D12Image(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    D3D12Image(const D3D12Image &) = delete;
    D3D12Image &operator=(const D3D12Image &) = delete;
    ~D3D12Image() override {
        forgetUndefined(&serial, 1);
        std::lock_guard<std::mutex> guard(state->mutex);
        state->releaseAfterSubmittedWork(native.resource);
    }

    std::shared_ptr<DeviceState> state;
    // Tracks the states of every subresource, for the image and all its views.
    d3d12::Image native;
    uint64_t serial{nextImageSerial()};
};

struct D3D12View final : VernonHalImageView {
    explicit D3D12View(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    D3D12View(const D3D12View &) = delete;
    D3D12View &operator=(const D3D12View &) = delete;
    // Recordings copy the descriptors when they are made, so the CPU descriptors go at once.
    ~D3D12View() override {
        if (serial)
            nativeViews().remove(serial);
        std::lock_guard<std::mutex> guard(state->mutex);
        state->resourceDescriptors.free(binding.srv);
        state->resourceDescriptors.free(binding.uav);
        state->resourceDescriptors.free(binding.clear);
        state->renderTargets.free(binding.rtv);
        state->depthStencils.free(binding.dsv);
    }

    std::shared_ptr<DeviceState> state;
    d3d12::View binding;
    NativeView entry;
    uint64_t serial{};
};

struct D3D12Sampler final : VernonHalSampler {
    explicit D3D12Sampler(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    D3D12Sampler(const D3D12Sampler &) = delete;
    D3D12Sampler &operator=(const D3D12Sampler &) = delete;
    ~D3D12Sampler() override {
        if (serial)
            nativeSamplers().remove(serial);
        std::lock_guard<std::mutex> guard(state->mutex);
        state->samplerDescriptors.free(sampler.cpu);
    }

    std::shared_ptr<DeviceState> state;
    d3d12::Sampler sampler;
    uint64_t serial{};
};

// A window's flip-model swapchain at the configured extent, which DXGI stretches to the window, and the ring of
// images that passes render to and presents copy into its back buffer. The images are the surface's own, so a frame
// acquires one without waiting for the swapchain.
struct D3D12Surface final : VernonHalSurface {
    explicit D3D12Surface(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    D3D12Surface(const D3D12Surface &) = delete;
    D3D12Surface &operator=(const D3D12Surface &) = delete;
    ~D3D12Surface() override {
        std::lock_guard<std::mutex> guard(state->mutex);
        std::string ignored;
        (void)state->synchronize(ignored);
        d3d12::release(swapchain);
    }

    std::shared_ptr<DeviceState> state;
    HWND window{};
    VernonHalSurfaceConfiguration configuration{};
    // Index-aligned; each view covers its image.
    std::vector<std::unique_ptr<D3D12Image>> images;
    std::vector<std::unique_ptr<D3D12View>> views;
    size_t next{};
    IDXGISwapChain3 *swapchain{};
    uint32_t width{};
    uint32_t height{};
    UINT flags{};
};

// DXGI presents only from a window station with a visible desktop, which services and SSH sessions lack.
bool interactiveDesktop() {
    USEROBJECTFLAGS flags{};
    HWINSTA station = GetProcessWindowStation();
    return station && GetUserObjectInformationW(station, UOI_FLAGS, &flags, sizeof(flags), nullptr) &&
           (flags.dwFlags & WSF_VISIBLE);
}

bool clientExtent(HWND window, uint32_t &width, uint32_t &height) {
    RECT rect{};
    if (!GetClientRect(window, &rect))
        return false;
    width = static_cast<uint32_t>((std::max)(rect.right - rect.left, LONG{0}));
    height = static_cast<uint32_t>((std::max)(rect.bottom - rect.top, LONG{0}));
    return true;
}

// Creates the swapchain, or resizes its buffers to the configured extent. Runs under the device's lock; the queue
// drains first, since resizing needs the old back buffers idle.
bool rebuildSwapchain(D3D12Surface &surface, std::string &error) {
    DeviceState &state = *surface.state;
    const uint32_t width = surface.configuration.width;
    const uint32_t height = surface.configuration.height;
    if (surface.swapchain && surface.width == width && surface.height == height)
        return true;
    if (!state.synchronize(error))
        return false;
    if (surface.swapchain) {
        if (!d3d12::check(surface.swapchain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, surface.flags),
                          "IDXGISwapChain::ResizeBuffers", error))
            return false;
    } else {
        DXGI_SWAP_CHAIN_DESC1 description{};
        description.Width = width;
        description.Height = height;
        description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.SampleDesc.Count = 1;
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = std::clamp(surface.configuration.image_count, 2u, 16u);
        description.Scaling = DXGI_SCALING_STRETCH;
        description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        description.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        description.Flags = state.tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
        IDXGISwapChain1 *created = nullptr;
        if (!d3d12::check(state.factory->CreateSwapChainForHwnd(state.queue, surface.window, &description, nullptr,
                                                                nullptr, &created),
                          "IDXGIFactory2::CreateSwapChainForHwnd", error))
            return false;
        const HRESULT queried = created->QueryInterface(IID_PPV_ARGS(&surface.swapchain));
        d3d12::release(created);
        if (!d3d12::check(queried, "IDXGISwapChain1::QueryInterface", error))
            return false;
        surface.flags = description.Flags;
        (void)state.factory->MakeWindowAssociation(surface.window, DXGI_MWA_NO_ALT_ENTER);
    }
    surface.width = width;
    surface.height = height;
    return true;
}

class D3D12DeviceBuffers final : public DeviceBuffers {
public:
    explicit D3D12DeviceBuffers(std::shared_ptr<DeviceState> state)
        : state_(state), pool_(std::make_shared<BufferPool<NativeBuffer>>([state](uint64_t size) {
              using Created = Result<std::unique_ptr<NativeBuffer>, std::string>;
              d3d12::Buffer native{};
              std::string error;
              {
                  std::lock_guard<std::mutex> guard(state->mutex);
                  if (!state->createBuffer(native, size, VERNON_HAL_BUFFER_STORAGE, VERNON_HAL_MEMORY_DEVICE, error))
                      return Created{err("cannot create a D3D12 buffer: " + error)};
              }
              return Created{ok(std::make_unique<NativeBuffer>(state, native, size))};
          })) {}

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) override { return pool_->allocate(size); }

    Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) override {
        NativeBuffer &native = static_cast<BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the upload overruns the buffer"))};
        const VernonHalBufferWrite write{0, bytes, size};
        return writeBuffer(*state_, native.buffer(), &write, 1);
    }

    Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) override {
        NativeBuffer &native = static_cast<const BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the download overruns the buffer"))};
        const VernonHalBufferRead read{0, bytes, size};
        return readBuffer(*state_, native.buffer(), &read, 1);
    }

private:
    std::shared_ptr<DeviceState> state_;
    std::shared_ptr<BufferPool<NativeBuffer>> pool_;
};

class D3D12Resources final : public Resources {
public:
    explicit D3D12Resources(std::shared_ptr<DeviceState> state) : owner_(std::move(state)), state_(*owner_) {}

    Done<uint32_t> formatCapabilities(VernonTextureFormat format) override {
        const formats::Format *entry = formats::find(format);
        if (!entry)
            return fail<uint32_t>(VERNON_HAL_INVALID_ARGUMENT, "the format is unknown");
        const DXGI_FORMAT native = d3d12::dxgiFormat(format);
        if (native == DXGI_FORMAT_UNKNOWN)
            return Done<uint32_t>{ok(0u)};
        const D3D12_FEATURE_DATA_FORMAT_SUPPORT support = query(native);
        // Depth images are sampled through a color view of their typeless resource, which decides filtering.
        const D3D12_FEATURE_DATA_FORMAT_SUPPORT sampled = query(depthColorFormat(native));
        const auto has1 = [&](D3D12_FORMAT_SUPPORT1 bits) { return (support.Support1 & bits) == bits; };
        const auto has2 = [&](D3D12_FORMAT_SUPPORT2 bits) { return (support.Support2 & bits) == bits; };
        if (!has1(D3D12_FORMAT_SUPPORT1_TEXTURE2D))
            return Done<uint32_t>{ok(0u)};
        uint32_t result = VERNON_HAL_FORMAT_COPYABLE;
        if (has1(formats::hasDepth(format) ? D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL : D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
            result |= VERNON_HAL_FORMAT_RENDERABLE;
        if (has1(D3D12_FORMAT_SUPPORT1_BLENDABLE))
            result |= VERNON_HAL_FORMAT_BLENDABLE;
        if (sampled.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)
            result |= VERNON_HAL_FORMAT_FILTERABLE;
        if (has1(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
            has2(static_cast<D3D12_FORMAT_SUPPORT2>(D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD |
                                                    D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)))
            result |= VERNON_HAL_FORMAT_STORAGE;
        if (has1(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET))
            result |= VERNON_HAL_FORMAT_MULTISAMPLE;
        if (has1(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE))
            result |= VERNON_HAL_FORMAT_RESOLVABLE;
        return Done<uint32_t>{ok(result & entry->baseline)};
    }

    Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalBuffer>>;
        d3d12::Buffer native{};
        std::string error;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            if (!state_.createBuffer(native, descriptor.size, descriptor.usage, descriptor.memory, error))
                return internal<std::unique_ptr<VernonHalBuffer>>("cannot create the buffer", error);
        }
        auto buffer =
            std::unique_ptr<D3D12HalBuffer>(new (std::nothrow) D3D12HalBuffer(owner_, native, descriptor.size));
        if (!buffer) {
            std::lock_guard<std::mutex> guard(state_.mutex);
            state_.destroyBuffer(native);
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        }
        return Created{ok(std::unique_ptr<VernonHalBuffer>(std::move(buffer)))};
    }

    Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) override {
        auto written = writeBuffer(state_, static_cast<D3D12HalBuffer &>(buffer).native.buffer(), writes, count);
        if (written.isErr())
            return internal("cannot write the buffer", written.error());
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) override {
        auto read = readBuffer(state_, static_cast<D3D12HalBuffer &>(buffer).native.buffer(), reads, count);
        if (read.isErr())
            return internal("cannot read the buffer", read.error());
        return Done<void>{ok()};
    }

    Done<void *> address(const VernonHalBuffer &) override {
        return fail<void *>(VERNON_HAL_UNSUPPORTED, "only CPU and CUDA buffers have an address");
    }

    Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &descriptor) override {
        auto made = makeImage(descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImage>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImage>>{ok(std::unique_ptr<VernonHalImage>(std::move(made).value()))};
    }

    Done<void> write(VernonHalImage &image, const VernonHalImageWrite *writes, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        auto &target = static_cast<D3D12Image &>(image);
        d3d12::Image &native = target.native;
        std::vector<const VernonHalImageRegion *> regions(count);
        for (size_t index = 0; index < count; ++index)
            regions[index] = &writes[index].region;
        std::vector<PlaneCopy> copies;
        size_t total = 0;
        bool unsupported = false;
        if (!planCopies(state_.device, image.descriptor, native, regions.data(), count, true, copies, total,
                        unsupported))
            return unsupported ? fail(VERNON_HAL_UNSUPPORTED, "DirectX 12 writes whole depth-stencil mip levels")
                               : fail(VERNON_HAL_INVALID_ARGUMENT, "a write's data layout does not match the image");
        std::lock_guard<std::mutex> guard(state_.mutex);
        std::string error;
        d3d12::CommandFrame *frame = nullptr;
        ID3D12Resource *staging = nullptr;
        size_t stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!state_.uploadCommands(frame, error) ||
            !state_.acquireStaging(*frame, true, total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging, stagingOffset,
                                   mapped, error))
            return internal("cannot stage the image's write", error);
        std::vector<uint32_t> written;
        for (const PlaneCopy &copy : copies) {
            moveTexels(copy, writes[copy.transfer].region, mapped + copy.layout.Offset,
                       const_cast<uint8_t *>(static_cast<const uint8_t *>(writes[copy.transfer].source)), true);
            written.push_back(copy.subresource);
        }
        transition(frame->commands, native, written, D3D12_RESOURCE_STATE_COPY_DEST);
        for (PlaneCopy copy : copies) {
            copy.layout.Offset += stagingOffset;
            D3D12_TEXTURE_COPY_LOCATION source{staging, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            source.PlacedFootprint = copy.layout;
            D3D12_TEXTURE_COPY_LOCATION destination{native.resource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            destination.SubresourceIndex = copy.subresource;
            frame->commands->CopyTextureRegion(&destination, copy.box.left, copy.box.top, copy.box.front, &source,
                                               nullptr);
        }
        forgetUndefined(&target.serial, 1);
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalImage &image, const VernonHalImageRead *reads, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        auto &source = static_cast<D3D12Image &>(image);
        if (anyUndefined(&source.serial, 1))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "the image's contents are undefined");
        d3d12::Image &native = source.native;
        std::vector<const VernonHalImageRegion *> regions(count);
        for (size_t index = 0; index < count; ++index) {
            const auto bytes = regionBytes(image.descriptor, reads[index].region);
            if (!bytes || *bytes != reads[index].size)
                return fail(VERNON_HAL_INVALID_ARGUMENT, "a read's size or data layout does not match the image");
            regions[index] = &reads[index].region;
        }
        std::vector<PlaneCopy> copies;
        size_t total = 0;
        bool unsupported = false;
        if (!planCopies(state_.device, image.descriptor, native, regions.data(), count, false, copies, total,
                        unsupported))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "a read's data layout does not match the image");
        std::lock_guard<std::mutex> guard(state_.mutex);
        std::string error;
        d3d12::CommandFrame *frame = nullptr;
        if (!state_.beginCommands(frame, error))
            return internal("cannot read the image", error);
        ID3D12Resource *staging = nullptr;
        size_t stagingOffset = 0;
        uint8_t *mapped = nullptr;
        if (!state_.acquireStaging(*frame, false, total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, staging, stagingOffset,
                                   mapped, error)) {
            state_.abandonCommands(frame);
            return internal("cannot stage the image's read", error);
        }
        std::vector<uint32_t> sources;
        for (const PlaneCopy &copy : copies)
            sources.push_back(copy.subresource);
        transition(frame->commands, native, sources, D3D12_RESOURCE_STATE_COPY_SOURCE);
        const bool depth = formats::hasDepth(image.descriptor.format);
        for (PlaneCopy copy : copies) {
            copy.layout.Offset += stagingOffset;
            D3D12_TEXTURE_COPY_LOCATION destination{staging, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            destination.PlacedFootprint = copy.layout;
            D3D12_TEXTURE_COPY_LOCATION from{native.resource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            from.SubresourceIndex = copy.subresource;
            frame->commands->CopyTextureRegion(&destination, 0, 0, 0, &from, depth ? nullptr : &copy.box);
        }
        // The host waits for this read only, not for later submissions.
        uint64_t value = 0;
        if (!state_.submitCommands(frame, value, error) || !state_.waitForValue(value, error))
            return internal("cannot read the image", error);
        for (size_t index = 0; index < count; ++index)
            if (reads[index].region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL))
                std::memset(reads[index].destination, 0, reads[index].size);
        for (const PlaneCopy &copy : copies)
            moveTexels(copy, reads[copy.transfer].region, mapped + copy.layout.Offset,
                       static_cast<uint8_t *>(reads[copy.transfer].destination), false);
        return Done<void>{ok()};
    }

    Done<void> generateMipmaps(VernonHalImage &image) override {
        const VernonImageDescriptor &descriptor = image.descriptor;
        if (descriptor.mip_level_count < 2 || formats::hasDepth(descriptor.format) ||
            !(descriptor.usage & VERNON_IMAGE_TRANSFER_SOURCE) ||
            !(descriptor.usage & VERNON_IMAGE_TRANSFER_DESTINATION))
            return fail(VERNON_HAL_INVALID_ARGUMENT,
                        "mipmaps need a color image with several mips that transfers in both directions");
        d3d12::Image &native = static_cast<D3D12Image &>(image).native;
        if (!(native.flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || !state_.mipmapRoot)
            return fail(VERNON_HAL_UNSUPPORTED, "the device cannot filter the image's mips");
        const bool volume = descriptor.dimension == VERNON_TEXTURE_3D;
        const DXGI_FORMAT shaderFormat = d3d12::dxgiFormat(descriptor.format);
        const DXGI_FORMAT storageFormat = linearFormat(shaderFormat);
        const bool srgb = storageFormat != shaderFormat;
        const uint32_t mips = descriptor.mip_level_count;
        const uint32_t layers = native.layers;
        std::lock_guard<std::mutex> guard(state_.mutex);
        std::string error;
        d3d12::CommandFrame *frame = nullptr;
        if (!state_.beginCommands(frame, error))
            return internal("cannot generate the image's mipmaps", error);
        ID3D12DescriptorHeap *heap = nullptr;
        if (!state_.frameHeap(*frame, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2 * (mips - 1), heap, error)) {
            state_.abandonCommands(frame);
            return internal("cannot generate the image's mipmaps", error);
        }
        const UINT increment = state_.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
        ID3D12GraphicsCommandList *commands = frame->commands;
        transition(commands, native, allSubresources(native), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        commands->SetDescriptorHeaps(1, &heap);
        commands->SetComputeRootSignature(state_.mipmapRoot);
        commands->SetPipelineState(volume ? state_.mipmap3d : state_.mipmap2d);
        for (uint32_t level = 1; level < mips; ++level) {
            transition(commands, native, levelSubresources(native, level), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const SIZE_T slot = SIZE_T{2} * (level - 1) * increment;
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = shaderFormat;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = storageFormat;
            if (volume) {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                srv.Texture3D.MostDetailedMip = level - 1;
                srv.Texture3D.MipLevels = 1;
                uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                uav.Texture3D.MipSlice = level;
                uav.Texture3D.WSize = mipExtent(descriptor.extent.depth, level);
            } else {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                srv.Texture2DArray.MostDetailedMip = level - 1;
                srv.Texture2DArray.MipLevels = 1;
                srv.Texture2DArray.ArraySize = layers;
                uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                uav.Texture2DArray.MipSlice = level;
                uav.Texture2DArray.ArraySize = layers;
            }
            state_.device->CreateShaderResourceView(native.resource, &srv, {cpu.ptr + slot});
            state_.device->CreateUnorderedAccessView(native.resource, nullptr, &uav, {cpu.ptr + slot + increment});
            commands->SetComputeRootDescriptorTable(0, {gpu.ptr + slot});
            const uint32_t sourceWidth = mipExtent(descriptor.extent.width, level - 1);
            const uint32_t sourceHeight = mipExtent(descriptor.extent.height, level - 1);
            const uint32_t sourceDepth = volume ? mipExtent(descriptor.extent.depth, level - 1) : layers;
            const uint32_t constants[4] = {sourceWidth, sourceHeight, sourceDepth, srgb ? 1u : 0u};
            commands->SetComputeRoot32BitConstants(1, 4, constants, 0);
            const uint32_t group = volume ? 4 : 8;
            commands->Dispatch((mipExtent(sourceWidth, 1) + group - 1) / group,
                               (mipExtent(sourceHeight, 1) + group - 1) / group,
                               volume ? (mipExtent(sourceDepth, 1) + 3) / 4 : layers);
            // The transition orders the level's writes before the next level reads them.
            transition(commands, native, levelSubresources(native, level),
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        uint64_t value = 0;
        if (!state_.submitCommands(frame, value, error))
            return internal("cannot submit the image's mipmaps", error);
        return Done<void>{ok()};
    }

    Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &image,
                                                         const VernonImageViewDescriptor &descriptor) override {
        auto made = makeView(static_cast<D3D12Image &>(image), descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImageView>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImageView>>{
            ok(std::unique_ptr<VernonHalImageView>(std::move(made).value()))};
    }

    Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSampler>>;
        if (descriptor.min_filter > VERNON_HAL_FILTER_LINEAR || descriptor.mag_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.mip_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.address_u > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_v > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_w > VERNON_HAL_ADDRESS_MIRRORED_REPEAT || !(descriptor.max_anisotropy <= 16.0f))
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_INVALID_ARGUMENT,
                                                           "the sampler descriptor is invalid");
        constexpr D3D12_TEXTURE_ADDRESS_MODE kAddress[] = {
            D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
        const auto filter = [](VernonHalFilter value) {
            return value == VERNON_HAL_FILTER_LINEAR ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT;
        };
        auto sampler = std::unique_ptr<D3D12Sampler>(new (std::nothrow) D3D12Sampler(owner_));
        if (!sampler)
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a sampler");
        D3D12_SAMPLER_DESC &native = sampler->sampler.descriptor;
        if (descriptor.max_anisotropy > 1.0f) {
            native.Filter = D3D12_ENCODE_ANISOTROPIC_FILTER(D3D12_FILTER_REDUCTION_TYPE_STANDARD);
            native.MaxAnisotropy = static_cast<UINT>(std::ceil(descriptor.max_anisotropy));
        } else {
            native.Filter =
                D3D12_ENCODE_BASIC_FILTER(filter(descriptor.min_filter), filter(descriptor.mag_filter),
                                          filter(descriptor.mip_filter), D3D12_FILTER_REDUCTION_TYPE_STANDARD);
            native.MaxAnisotropy = 1;
        }
        native.AddressU = kAddress[descriptor.address_u];
        native.AddressV = kAddress[descriptor.address_v];
        native.AddressW = kAddress[descriptor.address_w];
        native.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        native.MaxLOD = D3D12_FLOAT32_MAX;
        std::string error;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            sampler->sampler.cpu = state_.samplerDescriptors.allocate(error);
        }
        if (!sampler->sampler.cpu.ptr)
            return internal<std::unique_ptr<VernonHalSampler>>("cannot create the sampler", error);
        state_.device->CreateSampler(&native, sampler->sampler.cpu);
        sampler->serial = nativeSamplers().add(&sampler->sampler);
        return Created{ok(std::unique_ptr<VernonHalSampler>(std::move(sampler)))};
    }

    Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSurface>>;
        const auto window = static_cast<HWND>(descriptor.native.win32_window);
        if (!window || !IsWindow(window))
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_INVALID_ARGUMENT, "the window is invalid");
        if (!interactiveDesktop())
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_UNSUPPORTED,
                                                           "the process has no visible desktop to present to");
        auto surface = std::unique_ptr<D3D12Surface>(new (std::nothrow) D3D12Surface(owner_));
        if (!surface)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a surface");
        surface->window = window;
        return Created{ok(std::unique_ptr<VernonHalSurface>(std::move(surface)))};
    }

    Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &surface) override {
        VernonHalSurfaceCapabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        if (!clientExtent(static_cast<D3D12Surface &>(surface).window, capabilities.current_width,
                          capabilities.current_height))
            return fail<VernonHalSurfaceCapabilities>(VERNON_HAL_INTERNAL_ERROR, "cannot query the window");
        capabilities.formats[0] = VERNON_TEXTURE_RGBA8_UNORM;
        capabilities.format_count = 1;
        // A flip-model swapchain presenting without a sync interval shows the latest frame at the next vertical
        // blank, and tears only where the device allows it.
        capabilities.present_modes = (1u << VERNON_HAL_PRESENT_FIFO) | (1u << VERNON_HAL_PRESENT_MAILBOX);
        if (state_.tearing)
            capabilities.present_modes |= 1u << VERNON_HAL_PRESENT_IMMEDIATE;
        // Without sampled or storage usage, an acquired image is read only by a preserving render scope or a read,
        // which a submission rejects while its contents are undefined.
        capabilities.usage = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE;
        capabilities.min_image_count = 1;
        return Done<VernonHalSurfaceCapabilities>{ok(capabilities)};
    }

    // The new images replace the old ones only once all of them exist.
    Done<void> configure(VernonHalSurface &surface, const VernonHalSurfaceConfiguration &configuration) override {
        auto &target = static_cast<D3D12Surface &>(surface);
        VernonImageDescriptor descriptor{VERNON_TEXTURE_2D,
                                         {configuration.width, configuration.height, 1},
                                         configuration.format,
                                         1,
                                         1,
                                         1,
                                         configuration.usage | VERNON_IMAGE_TRANSFER_SOURCE};
        const VernonImageViewDescriptor viewDescriptor{
            VERNON_TEXTURE_2D, configuration.format, {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR}};
        std::vector<std::unique_ptr<D3D12Image>> images;
        std::vector<std::unique_ptr<D3D12View>> views;
        for (uint32_t index = 0; index < configuration.image_count; ++index) {
            auto image = makeImage(descriptor);
            if (image.isErr())
                return Done<void>{err(std::move(image).error())};
            std::unique_ptr<D3D12Image> made = std::move(image).value();
            made->device = surface.device;
            made->descriptor = descriptor;
            made->surface = &surface;
            auto view = makeView(*made, viewDescriptor);
            if (view.isErr())
                return Done<void>{err(std::move(view).error())};
            std::unique_ptr<D3D12View> madeView = std::move(view).value();
            madeView->image = made.get();
            madeView->descriptor = viewDescriptor;
            images.push_back(std::move(made));
            views.push_back(std::move(madeView));
        }
        // The old views go before their images.
        std::vector<std::unique_ptr<D3D12View>> oldViews = std::exchange(target.views, std::move(views));
        std::vector<std::unique_ptr<D3D12Image>> oldImages = std::exchange(target.images, std::move(images));
        target.configuration = configuration;
        target.next = 0;
        std::string error;
        bool rebuilt = false;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            rebuilt = rebuildSwapchain(target, error);
        }
        oldViews.clear();
        oldImages.clear();
        if (!rebuilt)
            return internal("cannot configure the surface", error);
        return Done<void>{ok()};
    }

    // A window whose client area is no longer the configured extent reports out of date, so the host configures
    // again.
    Done<VernonHalSurfaceImage> acquire(VernonHalSurface &surface) override {
        auto &source = static_cast<D3D12Surface &>(surface);
        if (source.images.empty())
            return fail<VernonHalSurfaceImage>(VERNON_HAL_INVALID_ARGUMENT, "the surface is not configured");
        uint32_t width = 0;
        uint32_t height = 0;
        if (!clientExtent(source.window, width, height))
            return fail<VernonHalSurfaceImage>(VERNON_HAL_INTERNAL_ERROR, "cannot query the window");
        if (width != source.configuration.width || height != source.configuration.height)
            return fail<VernonHalSurfaceImage>(VERNON_HAL_SURFACE_OUT_OF_DATE, "the surface is out of date");
        D3D12Image &image = *source.images[source.next];
        markUndefined(image.serial);
        return Done<VernonHalSurfaceImage>{ok(VernonHalSurfaceImage{&image, source.views[source.next].get(), 0})};
    }

    Done<void> present(VernonHalSurface &surface) override {
        auto &target = static_cast<D3D12Surface &>(surface);
        D3D12Image &presented = *target.images[target.next];
        target.next = (target.next + 1) % target.images.size();
        std::lock_guard<std::mutex> guard(state_.mutex);
        std::string error;
        if (!rebuildSwapchain(target, error))
            return internal("cannot present the surface", error);
        ID3D12Resource *back = nullptr;
        if (!d3d12::check(
                target.swapchain->GetBuffer(target.swapchain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back)),
                "IDXGISwapChain::GetBuffer", error))
            return internal("cannot present the surface", error);
        d3d12::CommandFrame *frame = nullptr;
        if (!state_.beginCommands(frame, error)) {
            d3d12::release(back);
            return internal("cannot present the surface", error);
        }
        ID3D12GraphicsCommandList *commands = frame->commands;
        transition(commands, presented.native, allSubresources(presented.native), D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_RESOURCE_BARRIER barrier =
            transitionBarrier(back, 0, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->ResourceBarrier(1, &barrier);
        commands->CopyResource(back, presented.native.resource);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        commands->ResourceBarrier(1, &barrier);
        // The swapchain keeps its buffer alive.
        d3d12::release(back);
        uint64_t value = 0;
        if (!state_.submitCommands(frame, value, error))
            return internal("cannot present the surface", error);
        UINT interval = 1;
        UINT flags = 0;
        if (target.configuration.present_mode != VERNON_HAL_PRESENT_FIFO)
            interval = 0;
        if (target.configuration.present_mode == VERNON_HAL_PRESENT_IMMEDIATE && state_.tearing)
            flags = DXGI_PRESENT_ALLOW_TEARING;
        const HRESULT shown = target.swapchain->Present(interval, flags);
        if (shown == DXGI_ERROR_DEVICE_REMOVED || shown == DXGI_ERROR_DEVICE_RESET) {
            state_.lost = true;
            return internal("cannot present the surface", "the D3D12 device was removed");
        }
        if (!d3d12::check(shown, "IDXGISwapChain::Present", error))
            return internal("cannot present the surface", error);
        return Done<void>{ok()};
    }

    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalBuffer &buffer,
                                                           uint64_t offset, uint64_t size) override {
        return Done<VernonRuntimeProviderResourceReference>{
            ok(static_cast<const D3D12HalBuffer &>(buffer).native.reference(offset, size))};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalImageView &view) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeViewReference);
        reference.resource.value = static_cast<const D3D12View &>(view).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalSampler &sampler) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeSamplerReference);
        reference.resource.value = static_cast<const D3D12Sampler &>(sampler).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }

private:
    D3D12_FEATURE_DATA_FORMAT_SUPPORT query(DXGI_FORMAT format) const {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        if (FAILED(state_.device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))))
            return D3D12_FEATURE_DATA_FORMAT_SUPPORT{format};
        return support;
    }

    // Whether the device stores to `format` through a typed UAV.
    bool typedStore(DXGI_FORMAT format) const {
        const D3D12_FEATURE_DATA_FORMAT_SUPPORT support = query(format);
        return (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
               (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
    }

    Done<std::unique_ptr<D3D12Image>> makeImage(const VernonImageDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<D3D12Image>>;
        if (!validImage(descriptor))
            return fail<std::unique_ptr<D3D12Image>>(VERNON_HAL_INVALID_ARGUMENT, "the image descriptor is invalid");
        const DXGI_FORMAT format = d3d12::dxgiFormat(descriptor.format);
        if (format == DXGI_FORMAT_UNKNOWN)
            return fail<std::unique_ptr<D3D12Image>>(VERNON_HAL_UNSUPPORTED, "DirectX 12 cannot create the image");
        const uint32_t usage = descriptor.usage;
        const bool depth = formats::hasDepth(descriptor.format);
        const bool volume = descriptor.dimension == VERNON_TEXTURE_3D;
        const D3D12_FEATURE_DATA_FORMAT_SUPPORT support = query(format);
        uint32_t required = volume ? D3D12_FORMAT_SUPPORT1_TEXTURE3D : D3D12_FORMAT_SUPPORT1_TEXTURE2D;
        if (descriptor.dimension == VERNON_TEXTURE_CUBE || descriptor.dimension == VERNON_TEXTURE_CUBE_ARRAY)
            required |= D3D12_FORMAT_SUPPORT1_TEXTURECUBE;
        if ((usage & VERNON_IMAGE_SAMPLED) && !depth)
            required |= D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
        if (usage & VERNON_IMAGE_COLOR_ATTACHMENT)
            required |= D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
        if (usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT)
            required |= D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL;
        if ((support.Support1 & required) != required || ((usage & VERNON_IMAGE_STORAGE) && !typedStore(format)))
            return fail<std::unique_ptr<D3D12Image>>(VERNON_HAL_UNSUPPORTED,
                                                     "the image's format does not support its usage");
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
        if (usage & VERNON_IMAGE_COLOR_ATTACHMENT)
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT)
            flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        // Mips are filtered by a compute shader, and integer color attachments are cleared, through UAVs.
        const DXGI_FORMAT storage = linearFormat(format);
        const bool mipmapped = descriptor.mip_level_count > 1 && !formats::isInteger(descriptor.format);
        const bool integerTarget = (usage & VERNON_IMAGE_COLOR_ATTACHMENT) && formats::isInteger(descriptor.format);
        if (!depth && ((usage & VERNON_IMAGE_STORAGE) || ((mipmapped || integerTarget) && typedStore(storage))))
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_RESOURCE_DESC resource{};
        resource.Dimension = volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resource.Width = descriptor.extent.width;
        resource.Height = descriptor.extent.height;
        resource.DepthOrArraySize =
            static_cast<UINT16>(volume ? descriptor.extent.depth : descriptor.array_layer_count);
        resource.MipLevels = static_cast<UINT16>(descriptor.mip_level_count);
        resource.Format = resourceFormat(format, usage);
        resource.SampleDesc.Count = 1;
        resource.Flags = flags;
        auto image = std::unique_ptr<D3D12Image>(new (std::nothrow) D3D12Image(owner_));
        if (!image)
            return fail<std::unique_ptr<D3D12Image>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate an image");
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        std::string error;
        if (!d3d12::check(state_.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resource,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&image->native.resource)),
                          "ID3D12Device::CreateCommittedResource", error))
            return internal<std::unique_ptr<D3D12Image>>("cannot create the image", error);
        d3d12::Image &native = image->native;
        native.format = resource.Format;
        native.vernonFormat = descriptor.format;
        native.dimension = descriptor.dimension;
        native.mipLevels = descriptor.mip_level_count;
        native.layers = volume ? 1 : descriptor.array_layer_count;
        native.planes = formats::hasStencil(descriptor.format) ? 2 : 1;
        native.flags = flags;
        native.states.assign(size_t{native.mipLevels} * native.layers * native.planes, D3D12_RESOURCE_STATE_COMMON);
        return Made{ok(std::move(image))};
    }

    Done<std::unique_ptr<D3D12View>> makeView(D3D12Image &image, const VernonImageViewDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<D3D12View>>;
        if (!validView(image.descriptor, descriptor))
            return fail<std::unique_ptr<D3D12View>>(VERNON_HAL_INVALID_ARGUMENT, "the view descriptor is invalid");
        const DXGI_FORMAT format = d3d12::dxgiFormat(descriptor.format);
        if (format == DXGI_FORMAT_UNKNOWN)
            return fail<std::unique_ptr<D3D12View>>(VERNON_HAL_UNSUPPORTED, "DirectX 12 cannot view the format");
        auto view = std::unique_ptr<D3D12View>(new (std::nothrow) D3D12View(owner_));
        if (!view)
            return fail<std::unique_ptr<D3D12View>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a view");
        const VernonImageSubresourceRange &range = descriptor.subresources;
        d3d12::View &binding = view->binding;
        binding.owner = &image.native;
        binding.format = format;
        binding.dimension = descriptor.dimension;
        binding.baseMip = range.base_mip_level;
        binding.mips = range.mip_level_count;
        binding.baseLayer = range.base_array_layer;
        binding.layers = range.array_layer_count;
        binding.aspects = range.aspects;
        const uint32_t usage = image.descriptor.usage;
        const bool integerClear = (usage & VERNON_IMAGE_COLOR_ATTACHMENT) && formats::isInteger(descriptor.format) &&
                                  (image.native.flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        std::string error;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            const auto allocate = [&](d3d12::DescriptorAllocator &allocator, D3D12_CPU_DESCRIPTOR_HANDLE &handle) {
                handle = allocator.allocate(error);
                return handle.ptr != 0;
            };
            if (((usage & VERNON_IMAGE_SAMPLED) && !allocate(state_.resourceDescriptors, binding.srv)) ||
                ((usage & VERNON_IMAGE_STORAGE) && !allocate(state_.resourceDescriptors, binding.uav)) ||
                (integerClear && !allocate(state_.resourceDescriptors, binding.clear)) ||
                ((usage & VERNON_IMAGE_COLOR_ATTACHMENT) && !allocate(state_.renderTargets, binding.rtv)) ||
                ((usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT) && !allocate(state_.depthStencils, binding.dsv)))
                return internal<std::unique_ptr<D3D12View>>("cannot create the view", error);
        }
        ID3D12Resource *resource = image.native.resource;
        if (binding.srv.ptr) {
            const D3D12_SHADER_RESOURCE_VIEW_DESC srv = shaderView(binding);
            state_.device->CreateShaderResourceView(resource, &srv, binding.srv);
        }
        if (binding.uav.ptr) {
            const D3D12_UNORDERED_ACCESS_VIEW_DESC uav = storageView(binding);
            state_.device->CreateUnorderedAccessView(resource, nullptr, &uav, binding.uav);
        }
        if (binding.clear.ptr) {
            const D3D12_UNORDERED_ACCESS_VIEW_DESC uav = storageView(binding);
            state_.device->CreateUnorderedAccessView(resource, nullptr, &uav, binding.clear);
        }
        if (binding.rtv.ptr) {
            const D3D12_RENDER_TARGET_VIEW_DESC rtv = renderTargetView(binding);
            state_.device->CreateRenderTargetView(resource, &rtv, binding.rtv);
        }
        if (binding.dsv.ptr) {
            const D3D12_DEPTH_STENCIL_VIEW_DESC dsv = depthStencilView(binding);
            state_.device->CreateDepthStencilView(resource, &dsv, binding.dsv);
        }
        view->entry = {image.descriptor, descriptor, image.serial, &binding};
        view->serial = nativeViews().add(&view->entry);
        return Made{ok(std::move(view))};
    }

    // HLSL declares a 2D texture without layers, so a 2D view is a single-layer array only past layer 0, and a cube
    // view a cube array only past the first cube.
    static D3D12_SHADER_RESOURCE_VIEW_DESC shaderView(const d3d12::View &view) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = view.format;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        uint32_t plane = 0;
        if (view.format == DXGI_FORMAT_D16_UNORM || view.format == DXGI_FORMAT_D32_FLOAT) {
            srv.Format = depthColorFormat(view.format);
        } else if (view.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
            if (view.aspects == VERNON_IMAGE_ASPECT_STENCIL) {
                // The stencil plane reads in green; shaders read it in red.
                srv.Format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
                srv.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
                    D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1,
                    D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
                    D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
                plane = 1;
            } else {
                srv.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            }
        }
        switch (view.dimension) {
        case VERNON_TEXTURE_3D:
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv.Texture3D = {view.baseMip, view.mips, 0.0f};
            return srv;
        case VERNON_TEXTURE_CUBE:
            if (view.baseLayer == 0) {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                srv.TextureCube = {view.baseMip, view.mips, 0.0f};
                return srv;
            }
            [[fallthrough]];
        case VERNON_TEXTURE_CUBE_ARRAY:
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            srv.TextureCubeArray = {view.baseMip, view.mips, view.baseLayer, view.layers / 6, 0.0f};
            return srv;
        case VERNON_TEXTURE_2D:
            if (view.baseLayer == 0) {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Texture2D = {view.baseMip, view.mips, plane, 0.0f};
                return srv;
            }
            [[fallthrough]];
        case VERNON_TEXTURE_2D_ARRAY:
            break;
        }
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv.Texture2DArray = {view.baseMip, view.mips, view.baseLayer, view.layers, plane, 0.0f};
        return srv;
    }

    static D3D12_UNORDERED_ACCESS_VIEW_DESC storageView(const d3d12::View &view) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = view.format;
        if (view.dimension == VERNON_TEXTURE_3D) {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uav.Texture3D = {view.baseMip, 0, UINT(-1)};
        } else if (view.dimension == VERNON_TEXTURE_2D && view.baseLayer == 0) {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Texture2D = {view.baseMip, 0};
        } else {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uav.Texture2DArray = {view.baseMip, view.baseLayer, view.layers, 0};
        }
        return uav;
    }

    static D3D12_RENDER_TARGET_VIEW_DESC renderTargetView(const d3d12::View &view) {
        D3D12_RENDER_TARGET_VIEW_DESC rtv{};
        rtv.Format = view.format;
        if (view.dimension == VERNON_TEXTURE_3D) {
            rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
            rtv.Texture3D = {view.baseMip, 0, UINT(-1)};
        } else if (view.owner->layers == 1) {
            rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            rtv.Texture2D = {view.baseMip, 0};
        } else {
            rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            rtv.Texture2DArray = {view.baseMip, view.baseLayer, view.layers, 0};
        }
        return rtv;
    }

    static D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilView(const d3d12::View &view) {
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
        dsv.Format = view.format;
        if (view.owner->layers == 1) {
            dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            dsv.Texture2D = {view.baseMip};
        } else {
            dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            dsv.Texture2DArray = {view.baseMip, view.baseLayer, view.layers};
        }
        return dsv;
    }

    std::shared_ptr<DeviceState> owner_;
    DeviceState &state_;
};

} // namespace

d3d12::Buffer *directX12Buffer(uint64_t serial) { return serials().find(serial); }

std::unique_ptr<Resources> createDirectX12Resources(std::shared_ptr<d3d12::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<Resources>(new (std::nothrow) D3D12Resources(std::move(state)));
}

std::unique_ptr<DeviceBuffers> createDirectX12DeviceBuffers(std::shared_ptr<d3d12::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<DeviceBuffers>(new (std::nothrow) D3D12DeviceBuffers(std::move(state)));
}

} // namespace vernon::hal
