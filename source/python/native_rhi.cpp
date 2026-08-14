#include "native_rhi.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string stringView(VernonStringView view) { return view.data ? std::string(view.data, view.size) : std::string(); }

VernonRhiFormat rhiFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return VERNON_RHI_FORMAT_RGBA8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return VERNON_RHI_FORMAT_RGBA8_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return VERNON_RHI_FORMAT_RGBA16_FLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return VERNON_RHI_FORMAT_RGBA32_FLOAT;
    case VERNON_TEXTURE_R8_UNORM:
        return VERNON_RHI_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_R16_FLOAT:
        return VERNON_RHI_FORMAT_R16_FLOAT;
    case VERNON_TEXTURE_R32_FLOAT:
        return VERNON_RHI_FORMAT_R32_FLOAT;
    case VERNON_TEXTURE_RG8_UNORM:
        return VERNON_RHI_FORMAT_RG8_UNORM;
    case VERNON_TEXTURE_RGB8_UNORM:
        return VERNON_RHI_FORMAT_RGB8_UNORM;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return VERNON_RHI_FORMAT_R11G11B10_FLOAT;
    case VERNON_TEXTURE_D32_FLOAT:
        return VERNON_RHI_FORMAT_D32_FLOAT;
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        return VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    }
    throw std::invalid_argument("unsupported attachment image format");
}

VernonRhiImageDimension rhiDimension(VernonTextureDimension dimension) {
    switch (dimension) {
    case VERNON_TEXTURE_2D:
        return VERNON_RHI_IMAGE_2D;
    case VERNON_TEXTURE_3D:
        return VERNON_RHI_IMAGE_3D;
    case VERNON_TEXTURE_CUBE:
        return VERNON_RHI_IMAGE_CUBE;
    }
    throw std::invalid_argument("unsupported texture dimension");
}

} // namespace

RhiHostState::RhiHostState(VernonRhiBackend backend, uint32_t deviceIndex) : backend(backend) {
    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = backend;
    descriptor.device_index = deviceIndex;
    device = vernonRhiCreateDevice(&descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        throw std::runtime_error("cannot create Vernon RHI device");
}

RhiHostState::RhiHostState(VernonRhiBackend backend, const VernonOpenGLContextCallbacks &callbacks) : backend(backend) {
    VernonRhiOwnedDeviceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.backend = backend;
    descriptor.opengl_callbacks = &callbacks;
    device = vernonRhiCreateDevice(&descriptor);
    if (device.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        throw std::runtime_error("cannot create external OpenGL RHI device");
}

RhiHostState::~RhiHostState() { vernonRhiDestroyDevice(device); }

RhiBuffer::RhiBuffer(std::shared_ptr<RhiHostState> host, size_t size) : host(std::move(host)), size(size) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                       VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX |
                       VERNON_RHI_BUFFER_INDEX;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    if (vernonRhiDeviceCreateBuffer(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("cannot create Vernon RHI buffer");
}

RhiBuffer::~RhiBuffer() { vernonRhiDeviceDestroyBuffer(host->device, handle); }

void RhiBuffer::upload(const nb::bytes &data, size_t offset) {
    if (offset > size || data.size() > size - offset ||
        vernonRhiDeviceUploadBuffer(host->device, handle, offset, data.c_str(), data.size()) != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("RHI buffer upload failed");
}

void RhiBuffer::uploadRanges(const nb::list &ranges) {
    std::vector<std::pair<size_t, nb::bytes>> uploads;
    std::vector<VernonRhiBufferUploadRange> nativeRanges;
    uploads.reserve(nb::len(ranges));
    nativeRanges.reserve(nb::len(ranges));
    for (nb::handle item : ranges) {
        nb::tuple range = nb::cast<nb::tuple>(item);
        if (nb::len(range) != 2)
            throw std::invalid_argument("RHI buffer upload range must contain an offset and bytes");
        const size_t offset = nb::cast<size_t>(range[0]);
        nb::bytes data = nb::cast<nb::bytes>(range[1]);
        if (offset > size || data.size() > size - offset)
            throw std::invalid_argument("RHI buffer upload range is outside the buffer");
        uploads.emplace_back(offset, std::move(data));
    }
    for (const auto &[offset, data] : uploads)
        nativeRanges.push_back({offset, data.c_str(), data.size()});
    if (vernonRhiDeviceUploadBufferRanges(host->device, handle, nativeRanges.data(), nativeRanges.size()) !=
        VERNON_RHI_STATUS_OK)
        throw std::runtime_error("RHI buffer range upload failed");
}

nb::bytes RhiBuffer::download() const {
    std::string data(size, '\0');
    if (vernonRhiDeviceDownloadBuffer(host->device, handle, 0, data.data(), data.size()) != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("RHI buffer download failed");
    return nb::bytes(data.data(), data.size());
}

RhiImage::RhiImage(std::shared_ptr<RhiHostState> host, uint32_t width, uint32_t height, uint32_t depth,
                   VernonTextureFormat format, VernonTextureDimension dimension, uint32_t mipLevels, uint32_t usage)
    : host(std::move(host)), width(width), height(height), depth(depth), format(format), dimension(dimension),
      mipLevels(mipLevels), usage(usage), layers(dimension == VERNON_TEXTURE_CUBE ? 6u : 1u) {
    if (!width || !height || !depth || !mipLevels || !usage)
        throw std::invalid_argument("RHI image extent, mip count, and usage must be non-zero");
    if (dimension != VERNON_TEXTURE_3D && depth != 1)
        throw std::invalid_argument("only a 3D RHI image may have depth greater than one");
    if (dimension == VERNON_TEXTURE_CUBE && width != height)
        throw std::invalid_argument("RHI cube image faces must be square");
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = rhiDimension(dimension);
    descriptor.format = rhiFormat(format);
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.mip_levels = mipLevels;
    descriptor.array_layers = layers;
    descriptor.sample_count = 1;
    descriptor.usage = usage;
    if (vernonRhiDeviceCreateImage(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("cannot create Vernon RHI image");
}

RhiImage::~RhiImage() { vernonRhiDeviceDestroyImage(host->device, handle); }

void RhiImage::upload(const nb::bytes &data, uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                      uint32_t uploadWidth, uint32_t uploadHeight, uint32_t uploadDepth) {
    if (!(usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION))
        throw std::runtime_error("image format does not support upload");
    if (mipLevel >= mipLevels)
        throw std::invalid_argument("RHI image upload mip level is out of range");
    const uint32_t mipWidth = mipExtent(width, mipLevel);
    const uint32_t mipHeight = mipExtent(height, mipLevel);
    const uint32_t mipDepth = dimension == VERNON_TEXTURE_3D ? mipExtent(depth, mipLevel) : depth;
    uploadWidth = uploadWidth ? uploadWidth : mipWidth - offsetX;
    uploadHeight = uploadHeight ? uploadHeight : mipHeight - offsetY;
    uploadDepth = uploadDepth ? uploadDepth : mipDepth - offsetZ;
    if (offsetX >= mipWidth || offsetY >= mipHeight || offsetZ >= mipDepth || uploadWidth > mipWidth - offsetX ||
        uploadHeight > mipHeight - offsetY || uploadDepth > mipDepth - offsetZ)
        throw std::invalid_argument("RHI image upload region is out of range");
    const Layout layout = dataLayout(format);
    const size_t layerSize = checkedByteSize(uploadWidth, uploadHeight, uploadDepth, layout.pixelSize);
    if (data.size() != layerSize * layers)
        throw std::runtime_error("RHI image upload size does not match its extent");
    std::vector<VernonRhiImageUploadDescriptor> descriptors(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        VernonRhiImageUploadDescriptor &descriptor = descriptors[layer];
        descriptor.struct_size = sizeof(descriptor);
        descriptor.mip_level = mipLevel;
        descriptor.array_layer = layer;
        descriptor.offset_x = offsetX;
        descriptor.offset_y = offsetY;
        descriptor.offset_z = offsetZ;
        descriptor.width = uploadWidth;
        descriptor.height = uploadHeight;
        descriptor.depth = uploadDepth;
        descriptor.source_format = layout.format;
        descriptor.source_type = layout.type;
        descriptor.data = data.c_str() + layerSize * layer;
    }
    if (vernonRhiDeviceUploadImage(host->device, handle, descriptors.data(), descriptors.size()) !=
        VERNON_RHI_STATUS_OK)
        throw std::runtime_error("RHI image upload failed: " + stringView(vernonRhiDeviceGetLastError(host->device)));
}

nb::bytes RhiImage::download(uint32_t mipLevel, uint32_t offsetX, uint32_t offsetY, uint32_t offsetZ,
                             uint32_t downloadWidth, uint32_t downloadHeight, uint32_t downloadDepth) const {
    if (!(usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE))
        throw std::runtime_error("image format does not support download");
    if (mipLevel >= mipLevels)
        throw std::invalid_argument("RHI image download mip level is out of range");
    const uint32_t mipWidth = mipExtent(width, mipLevel);
    const uint32_t mipHeight = mipExtent(height, mipLevel);
    const uint32_t mipDepth = dimension == VERNON_TEXTURE_3D ? mipExtent(depth, mipLevel) : depth;
    downloadWidth = downloadWidth ? downloadWidth : mipWidth - offsetX;
    downloadHeight = downloadHeight ? downloadHeight : mipHeight - offsetY;
    downloadDepth = downloadDepth ? downloadDepth : mipDepth - offsetZ;
    if (offsetX >= mipWidth || offsetY >= mipHeight || offsetZ >= mipDepth || downloadWidth > mipWidth - offsetX ||
        downloadHeight > mipHeight - offsetY || downloadDepth > mipDepth - offsetZ)
        throw std::invalid_argument("RHI image download region is out of range");
    const Layout layout = dataLayout(format);
    const size_t layerSize = checkedByteSize(downloadWidth, downloadHeight, downloadDepth, layout.pixelSize);
    std::string data(layerSize * layers, '\0');
    for (uint32_t layer = 0; layer < layers; ++layer) {
        VernonRhiImageDownloadDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.mip_level = mipLevel;
        descriptor.array_layer = layer;
        descriptor.offset_x = offsetX;
        descriptor.offset_y = offsetY;
        descriptor.offset_z = offsetZ;
        descriptor.width = downloadWidth;
        descriptor.height = downloadHeight;
        descriptor.depth = downloadDepth;
        descriptor.destination_format = layout.format;
        descriptor.destination_type = layout.type;
        if (vernonRhiDeviceDownloadImage(host->device, handle, &descriptor, data.data() + layer * layerSize,
                                         layerSize) != VERNON_RHI_STATUS_OK)
            throw std::runtime_error("RHI image download failed: " +
                                     stringView(vernonRhiDeviceGetLastError(host->device)));
    }
    return nb::bytes(data.data(), data.size());
}

void RhiImage::generateMipmaps() {
    if (mipLevels < 2)
        throw std::runtime_error("image has no mip chain to generate");
    if (vernonRhiDeviceGenerateImageMipmaps(host->device, handle) != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("RHI image mipmap generation failed: " +
                                 stringView(vernonRhiDeviceGetLastError(host->device)));
}

uint32_t RhiImage::mipExtent(uint32_t extent, uint32_t level) {
    const uint32_t value = extent >> level;
    return value ? value : 1;
}

RhiImage::Layout RhiImage::dataLayout(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_R8_UNORM:
        return {1, VERNON_RHI_IMAGE_DATA_RED, VERNON_RHI_IMAGE_DATA_UINT8};
    case VERNON_TEXTURE_RG8_UNORM:
        return {2, VERNON_RHI_IMAGE_DATA_RG, VERNON_RHI_IMAGE_DATA_UINT8};
    case VERNON_TEXTURE_RGB8_UNORM:
        return {3, VERNON_RHI_IMAGE_DATA_RGB, VERNON_RHI_IMAGE_DATA_UINT8};
    case VERNON_TEXTURE_RGBA8_UNORM:
    case VERNON_TEXTURE_RGBA8_SRGB:
        return {4, VERNON_RHI_IMAGE_DATA_RGBA, VERNON_RHI_IMAGE_DATA_UINT8};
    case VERNON_TEXTURE_R16_FLOAT:
        return {2, VERNON_RHI_IMAGE_DATA_RED, VERNON_RHI_IMAGE_DATA_FLOAT16};
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return {8, VERNON_RHI_IMAGE_DATA_RGBA, VERNON_RHI_IMAGE_DATA_FLOAT16};
    case VERNON_TEXTURE_R32_FLOAT:
        return {4, VERNON_RHI_IMAGE_DATA_RED, VERNON_RHI_IMAGE_DATA_FLOAT32};
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return {16, VERNON_RHI_IMAGE_DATA_RGBA, VERNON_RHI_IMAGE_DATA_FLOAT32};
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return {4, VERNON_RHI_IMAGE_DATA_RGB, VERNON_RHI_IMAGE_DATA_UINT32};
    case VERNON_TEXTURE_D32_FLOAT:
        return {4, VERNON_RHI_IMAGE_DATA_DEPTH, VERNON_RHI_IMAGE_DATA_FLOAT32};
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        return {8, VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL, VERNON_RHI_IMAGE_DATA_FLOAT32};
    }
    throw std::invalid_argument("unsupported RHI image data layout");
}

size_t RhiImage::checkedByteSize(uint32_t width, uint32_t height, uint32_t depth, size_t pixelSize) {
    size_t size = width;
    for (const size_t extent : {static_cast<size_t>(height), static_cast<size_t>(depth), pixelSize}) {
        if (extent && size > std::numeric_limits<size_t>::max() / extent)
            throw std::overflow_error("RHI image byte size overflows");
        size *= extent;
    }
    return size;
}

RhiImageView::RhiImageView(RhiImage *image, VernonTextureFormat format, VernonTextureDimension dimension,
                           uint32_t baseMipLevel, uint32_t mipLevelCount, uint32_t baseArrayLayer,
                           uint32_t arrayLayerCount, uint32_t aspects)
    : image(image), host(image ? image->host : nullptr), format(format), dimension(dimension),
      baseMipLevel(baseMipLevel), mipLevelCount(mipLevelCount), baseArrayLayer(baseArrayLayer),
      arrayLayerCount(arrayLayerCount), aspects(aspects) {
    if (!image || !mipLevelCount || !arrayLayerCount || !aspects)
        throw std::invalid_argument("RHI image view requires an image and non-empty subresources");
    VernonRhiImageViewDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.image = image->handle;
    descriptor.dimension = rhiDimension(dimension);
    descriptor.format = rhiFormat(format);
    descriptor.base_mip_level = baseMipLevel;
    descriptor.mip_level_count = mipLevelCount;
    descriptor.base_array_layer = baseArrayLayer;
    descriptor.array_layer_count = arrayLayerCount;
    descriptor.aspects = aspects;
    if (vernonRhiDeviceCreateImageView(host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
        throw std::invalid_argument("cannot create Vernon RHI image view: " +
                                    stringView(vernonRhiDeviceGetLastError(host->device)));
    width = std::max(image->width >> baseMipLevel, 1u);
    height = std::max(image->height >> baseMipLevel, 1u);
    layers = dimension == VERNON_TEXTURE_3D ? 1u : arrayLayerCount;
}

RhiImageView::~RhiImageView() { vernonRhiDeviceDestroyImageView(host->device, handle); }

RhiSampler::RhiSampler(std::shared_ptr<RhiHostState> host, VernonRhiSamplerAddressMode address)
    : host(std::move(host)) {
    if (address > VERNON_RHI_ADDRESS_MIRRORED_REPEAT)
        throw std::invalid_argument("unsupported sampler address mode");
    VernonRhiSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.min_filter = VERNON_RHI_FILTER_LINEAR;
    descriptor.mag_filter = VERNON_RHI_FILTER_LINEAR;
    descriptor.mip_filter = VERNON_RHI_FILTER_LINEAR;
    descriptor.address_u = address;
    descriptor.address_v = address;
    descriptor.address_w = address;
    descriptor.max_anisotropy = 1.0f;
    if (vernonRhiDeviceCreateSampler(this->host->device, &descriptor, &handle) != VERNON_RHI_STATUS_OK)
        throw std::runtime_error("cannot create Vernon RHI sampler");
}

RhiSampler::~RhiSampler() { vernonRhiDeviceDestroySampler(host->device, handle); }

RhiHost::RhiHost(VernonRhiBackend backend, uint32_t deviceIndex)
    : state(std::make_shared<RhiHostState>(backend, deviceIndex)) {}

RhiHost::RhiHost(std::shared_ptr<RhiHostState> state) : state(std::move(state)) {}

std::unique_ptr<RhiHost> RhiHost::createExternalOpenGL(VernonRhiBackend backend, uintptr_t userData,
                                                       uintptr_t makeCurrent, uintptr_t getProcAddress,
                                                       uint16_t apiMajor, uint16_t apiMinor) {
    if (backend != VERNON_RHI_BACKEND_OPENGL && backend != VERNON_RHI_BACKEND_OPENGL_ES)
        throw std::invalid_argument("external OpenGL host requires an OpenGL backend");
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.user_data = reinterpret_cast<void *>(userData);
    callbacks.make_current = reinterpret_cast<VernonOpenGLMakeCurrentFn>(makeCurrent);
    callbacks.get_proc_address = reinterpret_cast<VernonOpenGLGetProcAddressFn>(getProcAddress);
    callbacks.api_version_major = apiMajor;
    callbacks.api_version_minor = apiMinor;
    return std::make_unique<RhiHost>(std::make_shared<RhiHostState>(backend, callbacks));
}

std::unique_ptr<RhiBuffer> RhiHost::createBuffer(size_t size) { return std::make_unique<RhiBuffer>(state, size); }

std::unique_ptr<RhiImage> RhiHost::createImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                               VernonTextureDimension dimension, uint32_t depth, uint32_t mipLevels,
                                               uint32_t usage) {
    if (dimension == VERNON_TEXTURE_3D &&
        (format == VERNON_TEXTURE_D32_FLOAT || format == VERNON_TEXTURE_D32_FLOAT_S8_UINT))
        throw std::invalid_argument("3D depth textures are unsupported");
    if (!usage) {
        usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION | VERNON_RHI_IMAGE_SAMPLED;
        if (dimension != VERNON_TEXTURE_3D)
            usage |= format == VERNON_TEXTURE_D32_FLOAT || format == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                         ? VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT
                         : VERNON_RHI_IMAGE_COLOR_ATTACHMENT;
    }
    return std::make_unique<RhiImage>(state, width, height, depth, format, dimension, mipLevels, usage);
}

std::unique_ptr<RhiImage> RhiHost::createAttachmentImage(uint32_t width, uint32_t height, VernonTextureFormat format,
                                                         uint32_t usage) {
    constexpr uint32_t attachmentUsages = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT;
    if (!usage || (usage & ~attachmentUsages))
        throw std::invalid_argument("attachment image usage must contain only attachment roles");
    const bool depthFormat = format == VERNON_TEXTURE_D32_FLOAT || format == VERNON_TEXTURE_D32_FLOAT_S8_UINT;
    const bool depthUsage = (usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0;
    if (depthFormat != depthUsage || (depthUsage && (usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)))
        throw std::invalid_argument("attachment image format does not match its usage");
    return std::make_unique<RhiImage>(state, width, height, 1, format, VERNON_TEXTURE_2D, 1, usage);
}

std::unique_ptr<RhiSampler> RhiHost::createSampler(VernonRhiSamplerAddressMode address) {
    return std::make_unique<RhiSampler>(state, address);
}
