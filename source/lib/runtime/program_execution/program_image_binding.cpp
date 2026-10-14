#include "program_image_binding.h"

#include "VernonFormats.hpp"
#include "rhi/image_descriptor_validation.h"
#include "rhi/rhi_internal.h"
#include "runtime/provider_image_description.h"
#include "runtime/runtime_state.h"
#include "runtime/stage_binding_plan.h"

#include <algorithm>
#include <array>
#include <optional>

namespace vernon::runtime::program_execution {
namespace {

bool decodeImage(uint64_t key, VernonRhiImage &image) {
    const uint32_t encodedIndex = static_cast<uint32_t>(key);
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    image = {encodedIndex - 1, generation};
    return true;
}

std::optional<VernonRhiImageDimension> dimension(const std::string &value) {
    const std::optional<VernonTextureDimension> parsed = artifactTextureDimension(value);
    return parsed ? std::optional<VernonRhiImageDimension>(static_cast<VernonRhiImageDimension>(*parsed))
                  : std::nullopt;
}

std::optional<VernonRhiFormat> format(std::string_view value) {
    const vernon::formats::Format *entry = vernon::formats::find(value);
    return entry ? std::optional<VernonRhiFormat>(entry->rhi) : std::nullopt;
}

uint32_t aspects(const std::vector<std::string> &values) {
    uint32_t result = 0;
    for (const std::string &value : values)
        result |= value == "color"     ? VERNON_RHI_IMAGE_ASPECT_COLOR
                  : value == "depth"   ? VERNON_RHI_IMAGE_ASPECT_DEPTH
                  : value == "stencil" ? VERNON_RHI_IMAGE_ASPECT_STENCIL
                                       : 0;
    return result;
}

uint32_t usage(const std::vector<std::string> &values) {
    uint32_t result = 0;
    for (const std::string &value : values)
        result |= value == "sampled"                    ? VERNON_RHI_IMAGE_SAMPLED
                  : value == "storage"                  ? VERNON_RHI_IMAGE_STORAGE
                  : value == "color_attachment"         ? VERNON_RHI_IMAGE_COLOR_ATTACHMENT
                  : value == "depth_stencil_attachment" ? VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT
                  : value == "transfer_source"          ? VERNON_RHI_IMAGE_TRANSFER_SOURCE
                  : value == "transfer_destination"     ? VERNON_RHI_IMAGE_TRANSFER_DESTINATION
                                                        : 0;
    return result;
}

} // namespace

bool resolveBorrowedProgramImage(VernonRuntimeContext &context, const program::Storage &storage,
                                 VernonRuntimeProviderResourceReference reference, BoundProgramImage &resolved,
                                 std::string &error) {
    if (storage.ownership != program::StorageOwnership::Borrowed ||
        storage.descriptorKind != program::StorageDescriptorKind::Image || !storage.image.extent.empty())
        return error = "Program borrowed image Storage has an invalid canonical descriptor", false;
    uint64_t parentKey = 0;
    if (!reference.identity || !reference.resource.value ||
        !rhi::describeImageViewResource(context.rhiDevice, reference.resource.value, resolved.view, resolved.image,
                                        parentKey) ||
        !decodeImage(parentKey, resolved.parent))
        return error = "Program borrowed image is not a valid current invocation image view", false;
    const std::optional<VernonRhiImageDimension> expectedDimension = dimension(storage.image.dimension);
    const std::optional<VernonRhiFormat> expectedFormat = format(storage.image.format);
    const uint32_t expectedAspects = aspects(storage.image.aspects);
    const uint32_t expectedUsage = usage(storage.image.usage);
    const bool genericSampledConstraint =
        storage.image.format == "unknown" &&
        !(expectedUsage & (VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT));
    // Shaders read through the view, so a 2D view of one array layer satisfies a 2D binding.
    if (!expectedDimension || !rhi::knownImageDimension(resolved.view.dimension) ||
        (!genericSampledConstraint && resolved.view.dimension != *expectedDimension))
        return error = "Program borrowed image dimension does not match its Storage descriptor", false;
    if ((!genericSampledConstraint && (!expectedFormat || resolved.image.format != *expectedFormat)) ||
        (genericSampledConstraint && resolved.image.format == VERNON_RHI_FORMAT_UNDEFINED))
        return error = "Program borrowed image format does not match its Storage descriptor", false;
    if ((!genericSampledConstraint && (resolved.view.aspects & expectedAspects) != expectedAspects) ||
        (genericSampledConstraint && !resolved.view.aspects))
        return error = "Program borrowed image aspects do not match its Storage descriptor", false;
    if (resolved.image.sample_count != storage.image.sampleCount)
        return error = "Program borrowed image sample count does not match its Storage descriptor", false;
    if (resolved.view.mip_level_count < storage.image.mipLevels ||
        resolved.view.array_layer_count < storage.image.arrayLayers)
        return error = "Program borrowed image mip or layer range does not match its Storage descriptor", false;
    if ((resolved.image.usage & expectedUsage) != expectedUsage)
        return error = "Program borrowed image usage does not match its Storage descriptor", false;
    if (!resolved.image.width || !resolved.image.height || !resolved.image.depth)
        return error = "Program borrowed image has an invalid current invocation extent", false;
    return true;
}

bool materializeOwnedProgramImageDescriptor(const program::Storage &storage, const std::array<uint32_t, 3> &extent,
                                            VernonRhiImageDescriptor &descriptor, std::string &error) {
    if (storage.ownership != program::StorageOwnership::Owned ||
        storage.descriptorKind != program::StorageDescriptorKind::Image || storage.image.extent.size() != 3 ||
        !extent[0] || !extent[1] || !extent[2])
        return error = "owned Program image Storage has an invalid extent authority", false;
    const std::optional<VernonRhiImageDimension> imageDimension = dimension(storage.image.dimension);
    const std::optional<VernonRhiFormat> imageFormat = format(storage.image.format);
    const uint32_t imageUsage = usage(storage.image.usage);
    if (!imageDimension || !imageFormat || !imageUsage)
        return error = "owned Program image Storage has an invalid creation descriptor", false;
    descriptor = {};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = *imageDimension;
    descriptor.format = *imageFormat;
    descriptor.width = extent[0];
    descriptor.height = extent[1];
    descriptor.depth = extent[2];
    descriptor.mip_levels = storage.image.mipLevels;
    descriptor.array_layers = storage.image.arrayLayers;
    descriptor.sample_count = storage.image.sampleCount;
    descriptor.usage = imageUsage;
    if (!rhi::validImageShape(descriptor))
        return error = "owned Program image extent does not match its dimension", false;
    return true;
}

} // namespace vernon::runtime::program_execution
