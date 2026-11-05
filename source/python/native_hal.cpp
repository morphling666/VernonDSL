#include "native_hal.h"

#include "native_lifecycle_test_hooks.h"

#if defined(VERNON_HAS_GLFW_CONTEXT_OWNER)
#include "glfw_context_owner.h"
#endif

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::string halError() {
    const VernonStringView view = vernonHalGetLastError();
    return view.data ? std::string(view.data, view.size) : std::string();
}

namespace {

[[noreturn]] void throwHal(const char *message) { throw std::runtime_error(std::string(message) + ": " + halError()); }

bool opengl(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

VernonHalDevice *createDevice(VernonRuntimeBackend backend, uint32_t deviceIndex,
                              const VernonOpenGLContextCallbacks *callbacks) {
    VernonHalDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = backend;
    descriptor.device_index = deviceIndex;
    descriptor.opengl = callbacks;
    VernonHalDevice *device = nullptr;
    if (vernonHalCreateDevice(&descriptor, &device) != VERNON_HAL_OK)
        throwHal("cannot create the Vernon HAL device");
    return device;
}

#if defined(VERNON_HAS_GLFW_CONTEXT_OWNER)
class OwnedGlfwContext final : public HalContextOwner {
public:
    OwnedGlfwContext(vernon::host::GlfwContextApi api, uint16_t apiMajor, uint16_t apiMinor)
        : context(api, apiMajor, apiMinor) {}

    ~OwnedGlfwContext() override { vernon::python::testing::noteOwnedContextDestroyed(); }

    VernonOpenGLContextCallbacks callbacks() noexcept { return context.callbacks(); }

private:
    vernon::host::GlfwContextOwner context;
};
#endif

} // namespace

HalDeviceState::HalDeviceState(VernonRuntimeBackend backend, uint32_t deviceIndex) : backend(backend) {
    if (!opengl(backend)) {
        device = createDevice(backend, deviceIndex, nullptr);
        return;
    }
#if defined(VERNON_HAS_GLFW_CONTEXT_OWNER)
    const bool embedded = backend == VERNON_RUNTIME_OPENGL_ES;
    auto owner = std::make_shared<OwnedGlfwContext>(embedded ? vernon::host::GlfwContextApi::OpenGLES
                                                             : vernon::host::GlfwContextApi::OpenGL,
                                                    embedded ? 3 : 4, embedded ? 1 : 3);
    const VernonOpenGLContextCallbacks callbacks = owner->callbacks();
    contextOwner = std::move(owner);
    device = createDevice(backend, deviceIndex, &callbacks);
#else
    throw std::runtime_error("owned OpenGL context support is unavailable");
#endif
}

HalDeviceState::HalDeviceState(VernonRuntimeBackend backend, const VernonOpenGLContextCallbacks &callbacks,
                               std::shared_ptr<HalContextOwner> contextOwner)
    : backend(backend), contextOwner(std::move(contextOwner)) {
    device = createDevice(backend, 0, &callbacks);
}

HalDeviceState::~HalDeviceState() {
    // Every Runtime, resource and schedule holds this state, so the device has none left and its destruction, which
    // also destroys its runtime, cannot fail.
    if (vernonHalDestroyDevice(device) != VERNON_HAL_OK)
        return;
    vernon::python::testing::noteRuntimeDestroyed();
    vernon::python::testing::noteDeviceDestroyed();
}

HalBuffer::HalBuffer(std::shared_ptr<HalDeviceState> host, size_t size) : host(std::move(host)), size(size) {
    VernonHalBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.usage = VERNON_HAL_BUFFER_UNIFORM | VERNON_HAL_BUFFER_STORAGE | VERNON_HAL_BUFFER_VERTEX |
                       VERNON_HAL_BUFFER_INDEX | VERNON_HAL_BUFFER_INDIRECT;
    descriptor.memory = VERNON_HAL_MEMORY_DEVICE;
    if (vernonHalCreateBuffer(this->host->device, &descriptor, &handle) != VERNON_HAL_OK)
        throwHal("cannot create the HAL buffer");
}

HalBuffer::~HalBuffer() { vernonHalDestroyBuffer(handle); }

void HalBuffer::upload(const nb::bytes &data, size_t offset) {
    if (offset > size || data.size() > size - offset)
        throw std::invalid_argument("HAL buffer upload is outside the buffer");
    const VernonHalBufferWrite write{offset, data.c_str(), data.size()};
    if (vernonHalBufferWrite(handle, &write, 1) != VERNON_HAL_OK)
        throwHal("HAL buffer upload failed");
}

void HalBuffer::uploadRanges(const nb::list &ranges) {
    std::vector<nb::bytes> sources;
    std::vector<VernonHalBufferWrite> writes;
    sources.reserve(nb::len(ranges));
    writes.reserve(nb::len(ranges));
    for (nb::handle item : ranges) {
        nb::tuple range = nb::cast<nb::tuple>(item);
        if (nb::len(range) != 2)
            throw std::invalid_argument("HAL buffer upload range must contain an offset and bytes");
        const size_t offset = nb::cast<size_t>(range[0]);
        sources.push_back(nb::cast<nb::bytes>(range[1]));
        if (offset > size || sources.back().size() > size - offset)
            throw std::invalid_argument("HAL buffer upload range is outside the buffer");
        writes.push_back({offset, sources.back().c_str(), sources.back().size()});
    }
    if (!writes.empty() && vernonHalBufferWrite(handle, writes.data(), writes.size()) != VERNON_HAL_OK)
        throwHal("HAL buffer range upload failed");
}

nb::bytes HalBuffer::download() const {
    std::string data(size, '\0');
    const VernonHalBufferRead read{0, data.data(), data.size()};
    if (size && vernonHalBufferRead(handle, &read, 1) != VERNON_HAL_OK)
        throwHal("HAL buffer download failed");
    return nb::bytes(data.data(), data.size());
}

nb::list HalBuffer::downloadRanges(const nb::list &ranges) const {
    std::vector<size_t> offsets;
    std::vector<std::string> downloads;
    offsets.reserve(nb::len(ranges));
    downloads.reserve(nb::len(ranges));
    for (nb::handle item : ranges) {
        nb::tuple range = nb::cast<nb::tuple>(item);
        if (range.size() != 2)
            throw std::invalid_argument("HAL buffer download range must contain begin and end");
        const size_t begin = nb::cast<size_t>(range[0]);
        const size_t end = nb::cast<size_t>(range[1]);
        if (begin > end || end > size)
            throw std::invalid_argument("HAL buffer download range is outside the buffer");
        offsets.push_back(begin);
        downloads.emplace_back(end - begin, '\0');
    }
    nb::list result;
    if (downloads.empty())
        return result;
    std::vector<VernonHalBufferRead> reads;
    reads.reserve(downloads.size());
    for (size_t index = 0; index < downloads.size(); ++index)
        reads.push_back({offsets[index], downloads[index].data(), downloads[index].size()});
    if (vernonHalBufferRead(handle, reads.data(), reads.size()) != VERNON_HAL_OK)
        throwHal("HAL buffer range download failed");
    for (size_t index = 0; index < downloads.size(); ++index)
        result.append(nb::make_tuple(offsets[index], nb::bytes(downloads[index].data(), downloads[index].size())));
    return result;
}

void *HalBuffer::address() {
    if (!address_ && vernonHalBufferGetAddress(handle, &address_) != VERNON_HAL_OK)
        address_ = nullptr;
    return address_;
}

HalImage::HalImage(std::shared_ptr<HalDeviceState> host, uint32_t width, uint32_t height, uint32_t depth,
                   VernonTextureFormat format, VernonTextureDimension dimension, uint32_t mipLevels, uint32_t usage,
                   uint32_t arrayLayers)
    : host(std::move(host)), width(width), height(height), depth(depth), format(format), dimension(dimension),
      mipLevels(mipLevels), usage(usage), layers(arrayLayers                        ? arrayLayers
                                                 : dimension == VERNON_TEXTURE_CUBE ? 6u
                                                                                    : 1u) {
    if (!width || !height || !depth || !mipLevels || !usage)
        throw std::invalid_argument("HAL image extent, mip count, and usage must be non-zero");
    if (dimension != VERNON_TEXTURE_3D && depth != 1)
        throw std::invalid_argument("only a 3D HAL image may have depth greater than one");
    if ((dimension == VERNON_TEXTURE_CUBE || dimension == VERNON_TEXTURE_CUBE_ARRAY) && width != height)
        throw std::invalid_argument("HAL cube image faces must be square");
    if ((dimension == VERNON_TEXTURE_2D || dimension == VERNON_TEXTURE_3D) && layers != 1)
        throw std::invalid_argument("only layered HAL images may have more than one array layer");
    if ((dimension == VERNON_TEXTURE_CUBE && layers != 6) || (dimension == VERNON_TEXTURE_CUBE_ARRAY && layers % 6))
        throw std::invalid_argument("HAL cube image layers must be six faces per cube");
    if (!vernon::formats::find(format))
        throw std::invalid_argument("unsupported HAL image format");
    VernonImageDescriptor descriptor{};
    descriptor.dimension = dimension;
    descriptor.extent = {width, height, depth};
    descriptor.format = format;
    descriptor.mip_level_count = mipLevels;
    descriptor.array_layer_count = layers;
    descriptor.sample_count = 1;
    descriptor.usage = usage;
    if (vernonHalCreateImage(this->host->device, &descriptor, &handle) != VERNON_HAL_OK)
        throwHal("cannot create the HAL image");
}

HalImage::~HalImage() { (void)vernonHalDestroyImage(handle); }

void HalImage::upload(const nb::bytes &data, uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                      uint32_t uploadWidth, uint32_t uploadHeight, uint32_t uploadDepth, uint32_t baseArrayLayer,
                      uint32_t arrayLayerCount, uint32_t aspects) {
    if (!(usage & VERNON_IMAGE_TRANSFER_DESTINATION))
        throw std::runtime_error("image format does not support upload");
    const Region region = checkedRegion({mipLevel, offsetX, offsetY, offsetZ, uploadWidth, uploadHeight, uploadDepth,
                                         baseArrayLayer, arrayLayerCount, aspects},
                                        "upload");
    const Layout layout = dataLayout(format, region.aspects);
    const size_t layerSize = checkedByteSize(region.width, region.height, region.depth, layout.pixelSize);
    if (data.size() != layerSize * region.arrayLayerCount)
        throw std::runtime_error("HAL image upload size does not match its extent");
    std::vector<VernonHalImageWrite> writes(region.arrayLayerCount);
    for (uint32_t layer = 0; layer < region.arrayLayerCount; ++layer)
        writes[layer] = {layerRegion(region, layer, layout), data.c_str() + layerSize * layer};
    if (vernonHalImageWrite(handle, writes.data(), writes.size()) != VERNON_HAL_OK)
        throwHal("HAL image upload failed");
}

void HalImage::uploadRegions(const nb::list &regions) {
    if (!(usage & VERNON_IMAGE_TRANSFER_DESTINATION))
        throw std::runtime_error("image format does not support upload");
    std::vector<nb::bytes> sources;
    std::vector<VernonHalImageWrite> writes;
    sources.reserve(nb::len(regions));
    for (nb::handle item : regions) {
        nb::dict request = nb::cast<nb::dict>(item);
        const Region region = parseRegion(request);
        sources.push_back(nb::cast<nb::bytes>(request["data"]));
        const Layout layout = dataLayout(format, region.aspects);
        const size_t layerSize = checkedByteSize(region.width, region.height, region.depth, layout.pixelSize);
        if (region.arrayLayerCount > (std::numeric_limits<size_t>::max)() / layerSize ||
            sources.back().size() != layerSize * region.arrayLayerCount)
            throw std::invalid_argument("HAL image upload size does not match its region");
        for (uint32_t layer = 0; layer < region.arrayLayerCount; ++layer)
            writes.push_back({layerRegion(region, layer, layout), sources.back().c_str() + layerSize * layer});
    }
    if (!writes.empty() && vernonHalImageWrite(handle, writes.data(), writes.size()) != VERNON_HAL_OK)
        throwHal("HAL image batch upload failed");
}

nb::bytes HalImage::download(uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                             uint32_t downloadWidth, uint32_t downloadHeight, uint32_t downloadDepth,
                             uint32_t baseArrayLayer, uint32_t arrayLayerCount, uint32_t aspects) const {
    if (!(usage & VERNON_IMAGE_TRANSFER_SOURCE))
        throw std::runtime_error("image format does not support download");
    const Region region = checkedRegion({mipLevel, offsetX, offsetY, offsetZ, downloadWidth, downloadHeight,
                                         downloadDepth, baseArrayLayer, arrayLayerCount, aspects},
                                        "download");
    const Layout layout = dataLayout(format, region.aspects);
    const size_t layerSize = checkedByteSize(region.width, region.height, region.depth, layout.pixelSize);
    std::string data(layerSize * region.arrayLayerCount, '\0');
    std::vector<VernonHalImageRead> reads(region.arrayLayerCount);
    for (uint32_t layer = 0; layer < region.arrayLayerCount; ++layer)
        reads[layer] = {layerRegion(region, layer, layout), data.data() + layer * layerSize, layerSize};
    if (vernonHalImageRead(handle, reads.data(), reads.size()) != VERNON_HAL_OK)
        throwHal("HAL image download failed");
    return nb::bytes(data.data(), data.size());
}

nb::list HalImage::downloadRegions(const nb::list &regions) const {
    if (!(usage & VERNON_IMAGE_TRANSFER_SOURCE))
        throw std::runtime_error("image format does not support download");
    std::vector<std::string> destinations;
    std::vector<VernonHalImageRead> reads;
    destinations.reserve(nb::len(regions));
    for (nb::handle item : regions) {
        const Region region = parseRegion(nb::cast<nb::dict>(item));
        const Layout layout = dataLayout(format, region.aspects);
        const size_t layerSize = checkedByteSize(region.width, region.height, region.depth, layout.pixelSize);
        if (region.arrayLayerCount > (std::numeric_limits<size_t>::max)() / layerSize)
            throw std::overflow_error("HAL image batch download size overflows");
        destinations.emplace_back(layerSize * region.arrayLayerCount, '\0');
        for (uint32_t layer = 0; layer < region.arrayLayerCount; ++layer)
            reads.push_back(
                {layerRegion(region, layer, layout), destinations.back().data() + layerSize * layer, layerSize});
    }
    if (!reads.empty() && vernonHalImageRead(handle, reads.data(), reads.size()) != VERNON_HAL_OK)
        throwHal("HAL image batch download failed");
    nb::list result;
    for (const std::string &destination : destinations)
        result.append(nb::bytes(destination.data(), destination.size()));
    return result;
}

void HalImage::generateMipmaps() {
    if (mipLevels < 2)
        throw std::runtime_error("image has no mip chain to generate");
    if (vernonHalImageGenerateMipmaps(handle) != VERNON_HAL_OK)
        throwHal("HAL image mipmap generation failed");
}

uint32_t HalImage::mipExtent(uint32_t extent, uint32_t level) {
    const uint32_t value = extent >> level;
    return value ? value : 1;
}

HalImage::Layout HalImage::dataLayout(VernonTextureFormat format, uint32_t aspects) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    if (!entry)
        throw std::invalid_argument("unsupported HAL image data layout");
    if (vernon::formats::hasDepth(format) && vernon::formats::hasStencil(format)) {
        if (aspects == VERNON_IMAGE_ASPECT_DEPTH)
            return {sizeof(float), VERNON_HAL_IMAGE_DATA_DEPTH, VERNON_HAL_IMAGE_DATA_FLOAT32};
        if (aspects == VERNON_IMAGE_ASPECT_STENCIL)
            return {sizeof(uint8_t), VERNON_HAL_IMAGE_DATA_STENCIL, VERNON_HAL_IMAGE_DATA_UINT8};
        if (aspects != (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL))
            throw std::invalid_argument("unsupported depth/stencil transfer aspects");
    }
    return {entry->pixelBytes, static_cast<VernonHalImageDataFormat>(entry->dataFormat),
            static_cast<VernonHalImageDataType>(entry->dataType)};
}

size_t HalImage::checkedByteSize(uint32_t width, uint32_t height, uint32_t depth, size_t pixelSize) {
    size_t size = width;
    for (const size_t extent : {static_cast<size_t>(height), static_cast<size_t>(depth), pixelSize}) {
        if (extent && size > std::numeric_limits<size_t>::max() / extent)
            throw std::overflow_error("HAL image byte size overflows");
        size *= extent;
    }
    return size;
}

VernonHalImageRegion HalImage::layerRegion(const Region &region, uint32_t layer, const Layout &layout) {
    VernonHalImageRegion result{};
    result.mip_level = region.mipLevel;
    result.array_layer = region.baseArrayLayer + layer;
    result.aspect = region.aspects;
    result.offset_x = region.offsetX;
    result.offset_y = region.offsetY;
    result.offset_z = region.offsetZ;
    result.width = region.width;
    result.height = region.height;
    result.depth = region.depth;
    result.data_format = layout.format;
    result.data_type = layout.type;
    return result;
}

// Zero extents, layer counts and aspects select the rest of the mip level, the remaining layers and the format's
// aspects.
HalImage::Region HalImage::checkedRegion(Region region, const char *operation) const {
    if (region.mipLevel >= mipLevels)
        throw std::invalid_argument(std::string("HAL image ") + operation + " mip level is out of range");
    const uint32_t mipWidth = mipExtent(width, region.mipLevel);
    const uint32_t mipHeight = mipExtent(height, region.mipLevel);
    const uint32_t mipDepth = dimension == VERNON_TEXTURE_3D ? mipExtent(depth, region.mipLevel) : depth;
    if (region.offsetX >= mipWidth || region.offsetY >= mipHeight || region.offsetZ >= mipDepth ||
        region.baseArrayLayer >= layers)
        throw std::invalid_argument(std::string("HAL image ") + operation + " region is out of range");
    region.width = region.width ? region.width : mipWidth - region.offsetX;
    region.height = region.height ? region.height : mipHeight - region.offsetY;
    region.depth = region.depth ? region.depth : mipDepth - region.offsetZ;
    region.arrayLayerCount = region.arrayLayerCount ? region.arrayLayerCount : layers - region.baseArrayLayer;
    if (region.width > mipWidth - region.offsetX || region.height > mipHeight - region.offsetY ||
        region.depth > mipDepth - region.offsetZ || region.arrayLayerCount > layers - region.baseArrayLayer)
        throw std::invalid_argument(std::string("HAL image ") + operation + " region is out of range");
    if (!region.aspects)
        region.aspects = vernon::formats::aspects(format);
    return region;
}

HalImage::Region HalImage::parseRegion(const nb::dict &region) const {
    Region parsed{
        nb::cast<uint32_t>(region["mip_level"]),
        nb::cast<uint32_t>(region["offset_x"]),
        nb::cast<uint32_t>(region["offset_y"]),
        nb::cast<uint32_t>(region["offset_z"]),
        nb::cast<uint32_t>(region["width"]),
        nb::cast<uint32_t>(region["height"]),
        nb::cast<uint32_t>(region["depth"]),
        nb::cast<uint32_t>(region["base_array_layer"]),
        nb::cast<uint32_t>(region["array_layer_count"]),
        nb::cast<uint32_t>(region["aspects"]),
    };
    if (parsed.mipLevel >= mipLevels || !parsed.width || !parsed.height || !parsed.depth || !parsed.arrayLayerCount)
        throw std::invalid_argument("HAL image transfer region has an invalid extent");
    const uint32_t mipWidth = mipExtent(width, parsed.mipLevel);
    const uint32_t mipHeight = mipExtent(height, parsed.mipLevel);
    const uint32_t mipDepth = dimension == VERNON_TEXTURE_3D ? mipExtent(depth, parsed.mipLevel) : depth;
    if (parsed.offsetX >= mipWidth || parsed.offsetY >= mipHeight || parsed.offsetZ >= mipDepth ||
        parsed.width > mipWidth - parsed.offsetX || parsed.height > mipHeight - parsed.offsetY ||
        parsed.depth > mipDepth - parsed.offsetZ || parsed.baseArrayLayer >= layers ||
        parsed.arrayLayerCount > layers - parsed.baseArrayLayer ||
        (dimension != VERNON_TEXTURE_3D && (parsed.offsetZ != 0 || parsed.depth != 1)))
        throw std::invalid_argument("HAL image transfer region is out of range");
    (void)dataLayout(format, parsed.aspects);
    return parsed;
}

HalImageView::HalImageView(HalImage *image, VernonTextureFormat format, VernonTextureDimension dimension,
                           uint32_t baseMipLevel, uint32_t mipLevelCount, uint32_t baseArrayLayer,
                           uint32_t arrayLayerCount, uint32_t aspects)
    : image(image), host(image ? image->host : nullptr), format(format), dimension(dimension),
      baseMipLevel(baseMipLevel), mipLevelCount(mipLevelCount), baseArrayLayer(baseArrayLayer),
      arrayLayerCount(arrayLayerCount), aspects(aspects) {
    if (!image || !mipLevelCount || !arrayLayerCount || !aspects)
        throw std::invalid_argument("HAL image view requires an image and non-empty subresources");
    if (!vernon::formats::find(format))
        throw std::invalid_argument("unsupported HAL image view format");
    VernonImageViewDescriptor descriptor{};
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.subresources = {baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount, aspects};
    if (vernonHalCreateImageView(image->handle, &descriptor, &handle) != VERNON_HAL_OK)
        throw std::invalid_argument("cannot create the HAL image view: " + halError());
    width = std::max(image->width >> baseMipLevel, 1u);
    height = std::max(image->height >> baseMipLevel, 1u);
    layers = dimension == VERNON_TEXTURE_3D ? 1u : arrayLayerCount;
}

HalImageView::~HalImageView() { (void)vernonHalDestroyImageView(handle); }

HalSampler::HalSampler(std::shared_ptr<HalDeviceState> host, VernonHalAddressMode address) : host(std::move(host)) {
    if (address > VERNON_HAL_ADDRESS_MIRRORED_REPEAT)
        throw std::invalid_argument("unsupported sampler address mode");
    VernonHalSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.min_filter = VERNON_HAL_FILTER_LINEAR;
    descriptor.mag_filter = VERNON_HAL_FILTER_LINEAR;
    descriptor.mip_filter = VERNON_HAL_FILTER_LINEAR;
    descriptor.address_u = address;
    descriptor.address_v = address;
    descriptor.address_w = address;
    descriptor.max_anisotropy = 1.0f;
    if (vernonHalCreateSampler(this->host->device, &descriptor, &handle) != VERNON_HAL_OK)
        throwHal("cannot create the HAL sampler");
}

HalSampler::~HalSampler() { vernonHalDestroySampler(handle); }

HalDevice::HalDevice(VernonRuntimeBackend backend, uint32_t deviceIndex)
    : state(std::make_shared<HalDeviceState>(backend, deviceIndex)) {}

HalDevice::HalDevice(std::shared_ptr<HalDeviceState> state) : state(std::move(state)) {}

std::unique_ptr<HalDevice> HalDevice::createExternalOpenGL(VernonRuntimeBackend backend, uintptr_t userData,
                                                           uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                           uint16_t apiMajor, uint16_t apiMinor) {
    if (!opengl(backend))
        throw std::invalid_argument("an external OpenGL device requires an OpenGL backend");
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.user_data = reinterpret_cast<void *>(userData);
    callbacks.make_current = reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
    callbacks.get_proc_address = reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
    callbacks.api_version_major = apiMajor;
    callbacks.api_version_minor = apiMinor;
    return std::make_unique<HalDevice>(std::make_shared<HalDeviceState>(backend, callbacks));
}

std::unique_ptr<HalDevice> HalDevice::createOwnedOpenGL(VernonRuntimeBackend backend, uint16_t apiMajor,
                                                        uint16_t apiMinor) {
#if defined(VERNON_HAS_GLFW_CONTEXT_OWNER)
    if (!opengl(backend))
        throw std::invalid_argument("an owned OpenGL device requires an OpenGL backend");
    const auto api = backend == VERNON_RUNTIME_OPENGL_ES ? vernon::host::GlfwContextApi::OpenGLES
                                                         : vernon::host::GlfwContextApi::OpenGL;
    auto owner = std::make_shared<OwnedGlfwContext>(api, apiMajor, apiMinor);
    return std::make_unique<HalDevice>(std::make_shared<HalDeviceState>(backend, owner->callbacks(), owner));
#else
    (void)backend;
    (void)apiMajor;
    (void)apiMinor;
    throw std::runtime_error("owned OpenGL context support is unavailable");
#endif
}

std::unique_ptr<HalBuffer> HalDevice::createBuffer(size_t size) { return std::make_unique<HalBuffer>(state, size); }

std::unique_ptr<HalImage> HalDevice::createImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                                 VernonTextureDimension dimension, uint32_t depth, uint32_t mipLevels,
                                                 uint32_t usage, uint32_t arrayLayers) {
    if (dimension == VERNON_TEXTURE_3D && vernon::formats::hasDepth(format))
        throw std::invalid_argument("3D depth textures are unsupported");
    if (!usage) {
        usage = VERNON_IMAGE_TRANSFER_SOURCE | VERNON_IMAGE_TRANSFER_DESTINATION | VERNON_IMAGE_SAMPLED;
        if (dimension != VERNON_TEXTURE_3D)
            usage |= vernon::formats::hasDepth(format) ? VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT
                                                       : VERNON_IMAGE_COLOR_ATTACHMENT;
    }
    return std::make_unique<HalImage>(state, width, height, depth, format, dimension, mipLevels, usage, arrayLayers);
}

std::unique_ptr<HalImage> HalDevice::createAttachmentImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                                           uint32_t usage) {
    constexpr uint32_t attachmentUsages = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT;
    if (!usage || (usage & ~attachmentUsages))
        throw std::invalid_argument("attachment image usage must contain only attachment roles");
    const bool depthFormat = vernon::formats::hasDepth(format);
    const bool depthUsage = (usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0;
    if (depthFormat != depthUsage || (depthUsage && (usage & VERNON_IMAGE_COLOR_ATTACHMENT)))
        throw std::invalid_argument("attachment image format does not match its usage");
    return std::make_unique<HalImage>(state, width, height, 1, format, VERNON_TEXTURE_2D, 1, usage, 1);
}

std::unique_ptr<HalSampler> HalDevice::createSampler(VernonHalAddressMode address) {
    return std::make_unique<HalSampler>(state, address);
}
