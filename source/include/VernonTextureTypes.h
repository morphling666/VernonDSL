#ifndef VERNON_TEXTURE_TYPES_H
#define VERNON_TEXTURE_TYPES_H

#include <stdint.h>

typedef enum VernonTextureFormat {
#define VERNON_FORMAT(name, NAME, texture, ...) VERNON_TEXTURE_##NAME = texture,
#include "VernonFormats.def"
} VernonTextureFormat;

/* A cube-array layer is cube * 6 + face. */
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

/* The words of the format table's baseline column; a device narrows a format's baseline. */
typedef enum VernonHalFormatCapabilityBits {
    VERNON_HAL_FORMAT_RENDERABLE = 1u << 0,
    VERNON_HAL_FORMAT_BLENDABLE = 1u << 1,
    VERNON_HAL_FORMAT_FILTERABLE = 1u << 2,
    VERNON_HAL_FORMAT_STORAGE = 1u << 3,
    VERNON_HAL_FORMAT_RESOLVABLE = 1u << 4,
    VERNON_HAL_FORMAT_MULTISAMPLE = 1u << 5,
    VERNON_HAL_FORMAT_COPYABLE = 1u << 6
} VernonHalFormatCapabilityBits;

/* The host layout of image transfer data, tightly packed rows of texels; the format table's data_format and data_type
 * columns name these. */
typedef enum VernonHalImageDataFormat {
    VERNON_HAL_IMAGE_DATA_RED = 0,
    VERNON_HAL_IMAGE_DATA_RG = 1,
    VERNON_HAL_IMAGE_DATA_RGB = 2,
    VERNON_HAL_IMAGE_DATA_BGR = 3,
    VERNON_HAL_IMAGE_DATA_RGBA = 4,
    VERNON_HAL_IMAGE_DATA_BGRA = 5,
    VERNON_HAL_IMAGE_DATA_DEPTH = 6,
    VERNON_HAL_IMAGE_DATA_DEPTH_STENCIL = 7,
    VERNON_HAL_IMAGE_DATA_STENCIL = 8
} VernonHalImageDataFormat;

typedef enum VernonHalImageDataType {
    VERNON_HAL_IMAGE_DATA_UINT8 = 0,
    VERNON_HAL_IMAGE_DATA_FLOAT16 = 1,
    VERNON_HAL_IMAGE_DATA_FLOAT32 = 2,
    VERNON_HAL_IMAGE_DATA_UINT32 = 3,
    VERNON_HAL_IMAGE_DATA_INT8 = 4,
    VERNON_HAL_IMAGE_DATA_UINT16 = 5,
    VERNON_HAL_IMAGE_DATA_INT16 = 6,
    VERNON_HAL_IMAGE_DATA_INT32 = 7
} VernonHalImageDataType;

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
    VERNON_IMAGE_SAMPLE_UNSIGNED_INTEGER = 2,
    /* The image is not sampled, so no sample result class constrains its view format. */
    VERNON_IMAGE_SAMPLE_NONE = 3
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
