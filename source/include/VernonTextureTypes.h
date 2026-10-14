#ifndef VERNON_TEXTURE_TYPES_H
#define VERNON_TEXTURE_TYPES_H

#include <stdint.h>

typedef enum VernonTextureFormat {
#define VERNON_FORMAT(name, NAME, rhi, texture, ...) VERNON_TEXTURE_##NAME = texture,
#include "VernonFormats.def"
} VernonTextureFormat;

/* Values and layer rules match VernonRhiImageDimension: a cube-array layer is cube * 6 + face. */
typedef enum VernonTextureDimension {
    VERNON_TEXTURE_2D = 0,
    VERNON_TEXTURE_3D = 1,
    VERNON_TEXTURE_CUBE = 2,
    VERNON_TEXTURE_2D_ARRAY = 3,
    VERNON_TEXTURE_CUBE_ARRAY = 4
} VernonTextureDimension;

typedef enum VernonImageUsageBits {
    VERNON_IMAGE_SAMPLED = 1u << 0,
    VERNON_IMAGE_STORAGE = 1u << 1,
    VERNON_IMAGE_COLOR_ATTACHMENT = 1u << 2,
    VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT = 1u << 3,
    VERNON_IMAGE_TRANSFER_SOURCE = 1u << 4,
    VERNON_IMAGE_TRANSFER_DESTINATION = 1u << 5
} VernonImageUsageBits;

typedef enum VernonImageAspectBits {
    VERNON_IMAGE_ASPECT_COLOR = 1u << 0,
    VERNON_IMAGE_ASPECT_DEPTH = 1u << 1,
    VERNON_IMAGE_ASPECT_STENCIL = 1u << 2
} VernonImageAspectBits;

typedef enum VernonImageBindingRole {
    VERNON_IMAGE_BINDING_SAMPLED = 0,
    VERNON_IMAGE_BINDING_STORAGE = 1,
    VERNON_IMAGE_BINDING_COLOR_ATTACHMENT = 2,
    VERNON_IMAGE_BINDING_DEPTH_STENCIL_ATTACHMENT = 3,
    VERNON_IMAGE_BINDING_TRANSFER = 4
} VernonImageBindingRole;

typedef enum VernonImageSampleResultClass {
    VERNON_IMAGE_SAMPLE_FLOAT = 0,
    VERNON_IMAGE_SAMPLE_SIGNED_INTEGER = 1,
    VERNON_IMAGE_SAMPLE_UNSIGNED_INTEGER = 2
} VernonImageSampleResultClass;

typedef struct VernonImageExtent {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
} VernonImageExtent;

typedef struct VernonImageDescriptor {
    VernonTextureDimension dimension;
    VernonImageExtent extent;
    VernonTextureFormat format;
    uint32_t mip_level_count;
    uint32_t array_layer_count;
    uint32_t sample_count;
    uint32_t usage;
} VernonImageDescriptor;

typedef struct VernonImageSubresourceRange {
    uint32_t base_mip_level;
    uint32_t mip_level_count;
    uint32_t base_array_layer;
    uint32_t array_layer_count;
    uint32_t aspects;
} VernonImageSubresourceRange;

typedef struct VernonImageViewDescriptor {
    VernonTextureDimension dimension;
    VernonTextureFormat format;
    VernonImageSubresourceRange subresources;
} VernonImageViewDescriptor;

#endif
