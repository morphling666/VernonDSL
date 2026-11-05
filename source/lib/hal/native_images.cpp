#include "hal/native_images.h"

#include "VernonFormats.hpp"
#include "runtime/graphics_scope_materializer.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>

namespace vernon::hal {
namespace {

constexpr uint32_t kCubeFaces = 6;

std::mutex undefinedMutex;
std::vector<uint64_t> undefinedImages;
// Whether `undefinedImages` may hold any image, read without the lock on the submission path.
std::atomic<bool> anyUndefinedImage{false};

bool cube(VernonTextureDimension dimension) {
    return dimension == VERNON_TEXTURE_CUBE || dimension == VERNON_TEXTURE_CUBE_ARRAY;
}

bool validShape(const VernonImageDescriptor &image) {
    const auto &extent = image.extent;
    if (!extent.width || !extent.height || !extent.depth || !image.array_layer_count)
        return false;
    switch (image.dimension) {
    case VERNON_TEXTURE_2D:
        return extent.depth == 1 && image.array_layer_count == 1;
    case VERNON_TEXTURE_2D_ARRAY:
        return extent.depth == 1;
    case VERNON_TEXTURE_3D:
        return image.array_layer_count == 1;
    case VERNON_TEXTURE_CUBE:
        return extent.width == extent.height && extent.depth == 1 && image.array_layer_count == kCubeFaces;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return extent.width == extent.height && extent.depth == 1 && image.array_layer_count % kCubeFaces == 0;
    }
    return false;
}

uint32_t maximumMipLevels(const VernonImageDescriptor &image) {
    uint32_t extent = std::max(image.extent.width, image.extent.height);
    if (image.dimension == VERNON_TEXTURE_3D)
        extent = std::max(extent, image.extent.depth);
    uint32_t levels = 1;
    for (; extent > 1; extent >>= 1)
        ++levels;
    return levels;
}

// A view reinterprets layers, never the 3D/layered split: 2D and 2D_ARRAY views select layers of any non-3D image,
// and cube views select whole cubes, which start at a multiple of 6 layers.
bool compatibleDimension(VernonTextureDimension image, VernonTextureDimension view, uint32_t baseLayer,
                         uint32_t layerCount) {
    switch (view) {
    case VERNON_TEXTURE_2D:
        return image != VERNON_TEXTURE_3D && layerCount == 1;
    case VERNON_TEXTURE_2D_ARRAY:
        return image != VERNON_TEXTURE_3D;
    case VERNON_TEXTURE_3D:
        return image == VERNON_TEXTURE_3D && baseLayer == 0 && layerCount == 1;
    case VERNON_TEXTURE_CUBE:
        return cube(image) && baseLayer % kCubeFaces == 0 && layerCount == kCubeFaces;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return cube(image) && baseLayer % kCubeFaces == 0 && layerCount % kCubeFaces == 0;
    }
    return false;
}

// The host bytes of one texel of `aspect`: a packed depth-stencil texel, one aspect of a depth-stencil format, or a
// whole texel of any other; zero when the data layout does not match.
size_t texelBytes(VernonTextureFormat format, uint32_t aspect, VernonHalImageDataFormat dataFormat,
                  VernonHalImageDataType dataType) {
    const formats::Format *entry = formats::find(format);
    if (!entry)
        return 0;
    const bool whole = static_cast<int>(entry->dataFormat) == static_cast<int>(dataFormat) &&
                       static_cast<int>(entry->dataType) == static_cast<int>(dataType);
    if (whole)
        return aspect == entry->aspects() ? entry->pixelBytes : 0;
    if (!formats::hasDepth(format) || !formats::hasStencil(format))
        return 0;
    if (aspect == VERNON_IMAGE_ASPECT_DEPTH && dataFormat == VERNON_HAL_IMAGE_DATA_DEPTH &&
        dataType == VERNON_HAL_IMAGE_DATA_FLOAT32)
        return sizeof(float);
    if (aspect == VERNON_IMAGE_ASPECT_STENCIL && dataFormat == VERNON_HAL_IMAGE_DATA_STENCIL &&
        dataType == VERNON_HAL_IMAGE_DATA_UINT8)
        return 1;
    return 0;
}

const NativeView *liveView(const VernonRuntimeProviderResourceReference &reference, uint64_t identity) {
    return reference.identity == identity ? nativeViews().find(reference.resource.value) : nullptr;
}

} // namespace

Serials<NativeView> &nativeViews() {
    static Serials<NativeView> instance;
    return instance;
}

Serials<void> &nativeSamplers() {
    static Serials<void> instance;
    return instance;
}

uint64_t nextImageSerial() {
    static std::atomic<uint64_t> next{1};
    return next++;
}

bool describeNativeView(const VernonRuntimeProviderResourceReference &reference,
                        VernonRuntimeProviderImageDescription &description) {
    if ((reference.identity & kReferenceKindMask) != kNativeViewReference ||
        description.struct_size < sizeof(description))
        return false;
    const NativeView *view = nativeViews().find(reference.resource.value);
    if (!view)
        return false;
    description.image = view->image;
    description.view = view->view;
    description.parent_identity = view->image_serial;
    description.resource_kind = VERNON_RUNTIME_PROVIDER_IMAGE_VIEW;
    return true;
}

void markUndefined(uint64_t image) {
    std::lock_guard<std::mutex> guard(undefinedMutex);
    if (std::find(undefinedImages.begin(), undefinedImages.end(), image) == undefinedImages.end())
        undefinedImages.push_back(image);
    anyUndefinedImage = true;
}

void forgetUndefined(const uint64_t *images, size_t count) {
    if (!count || !anyUndefinedImage)
        return;
    std::lock_guard<std::mutex> guard(undefinedMutex);
    for (size_t index = 0; index < count; ++index)
        undefinedImages.erase(std::remove(undefinedImages.begin(), undefinedImages.end(), images[index]),
                              undefinedImages.end());
    anyUndefinedImage = !undefinedImages.empty();
}

bool anyUndefined(const uint64_t *images, size_t count) {
    if (!count || !anyUndefinedImage)
        return false;
    std::lock_guard<std::mutex> guard(undefinedMutex);
    for (size_t index = 0; index < count; ++index)
        if (std::find(undefinedImages.begin(), undefinedImages.end(), images[index]) != undefinedImages.end())
            return true;
    return false;
}

bool validImage(const VernonImageDescriptor &image) {
    constexpr uint32_t usages = VERNON_IMAGE_SAMPLED | VERNON_IMAGE_STORAGE | VERNON_IMAGE_COLOR_ATTACHMENT |
                                VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE |
                                VERNON_IMAGE_TRANSFER_DESTINATION;
    if (!formats::find(image.format) || !validShape(image))
        return false;
    const bool depth = formats::hasDepth(image.format);
    return image.mip_level_count != 0 && image.mip_level_count <= maximumMipLevels(image) && image.sample_count == 1 &&
           (image.usage & ~usages) == 0 && !(depth && (image.usage & VERNON_IMAGE_COLOR_ATTACHMENT)) &&
           !(!depth && (image.usage & VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT));
}

bool validView(const VernonImageDescriptor &image, const VernonImageViewDescriptor &view) {
    const VernonImageSubresourceRange &range = view.subresources;
    return formats::find(view.format) && formats::viewCompatible(image.format, view.format) && range.aspects != 0 &&
           (range.aspects & ~formats::aspects(image.format)) == 0 && range.mip_level_count != 0 &&
           range.array_layer_count != 0 && range.base_mip_level < image.mip_level_count &&
           range.mip_level_count <= image.mip_level_count - range.base_mip_level &&
           range.base_array_layer < image.array_layer_count &&
           range.array_layer_count <= image.array_layer_count - range.base_array_layer &&
           compatibleDimension(image.dimension, view.dimension, range.base_array_layer, range.array_layer_count);
}

std::optional<size_t> regionBytes(const VernonImageDescriptor &image, const VernonHalImageRegion &region) {
    const bool packed = region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
    const size_t texel = texelBytes(image.format, region.aspect, region.data_format, region.data_type);
    if (!texel || (image.dimension != VERNON_TEXTURE_3D && (region.offset_z != 0 || region.depth != 1)))
        return std::nullopt;
    const size_t texels = size_t{region.width} * region.height * region.depth;
    if (texels > (std::numeric_limits<size_t>::max)() / kPackedDepthStencilBytes)
        return std::nullopt;
    return texels * (packed ? kPackedDepthStencilBytes : texel);
}

Result<ScopeImages, std::string> checkRenderScope(const runtime::GraphicsRenderScope &scope, uint64_t identity) {
    using Checked = Result<ScopeImages, std::string>;
    const auto failure = [](std::string message) { return Checked{err(std::move(message))}; };
    ScopeImages images;
    const auto attachment = [&](const VernonRuntimeProviderResourceReference &reference, uint32_t usage,
                                bool preserved) -> const NativeView * {
        const NativeView *view = liveView(reference, identity);
        if (!view || !(view->image.usage & usage) || view->view.subresources.mip_level_count != 1 ||
            view->view.subresources.array_layer_count != 1 ||
            view->view.subresources.aspects != formats::aspects(view->view.format))
            return nullptr;
        const uint32_t mip = view->view.subresources.base_mip_level;
        if (scope.width > std::max(view->image.extent.width >> mip, 1u) ||
            scope.height > std::max(view->image.extent.height >> mip, 1u))
            return nullptr;
        if (preserved)
            images.readImages.push_back(view->image_serial);
        images.attachmentImages.push_back(view->image_serial);
        return view;
    };
    for (const VernonRuntimeProviderColorAttachment &color : scope.colors) {
        const NativeView *view = attachment(color.view, VERNON_IMAGE_COLOR_ATTACHMENT,
                                            color.load_operation == VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE);
        if (!view)
            return failure("a color attachment is destroyed or cannot be rendered to");
        images.colors.push_back(view);
    }
    if (scope.depth) {
        const VernonRuntimeProviderDepthStencilAttachment &depth = *scope.depth;
        images.depth = attachment(depth.view, VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                                  depth.depth_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ||
                                      depth.stencil_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE);
        if (!images.depth)
            return failure("the depth attachment is destroyed or cannot be rendered to");
        images.stencil = formats::hasStencil(images.depth->view.format);
        if (!images.stencil && depth.stencil_load_operation == VERNON_RUNTIME_PROVIDER_LOAD_CLEAR)
            return failure("the depth attachment clears a stencil it does not have");
    }
    for (const VernonRuntimeProviderResourceReference &reference : scope.reads.images) {
        const NativeView *view = liveView(reference, identity);
        if (!view || !(view->image.usage & VERNON_IMAGE_SAMPLED))
            return failure("a sampled image is destroyed or cannot be sampled");
        images.reads.push_back(view);
        images.readImages.push_back(view->image_serial);
    }
    return Checked{ok(std::move(images))};
}

} // namespace vernon::hal
