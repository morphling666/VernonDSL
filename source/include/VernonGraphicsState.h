#ifndef VERNON_GRAPHICS_STATE_H
#define VERNON_GRAPHICS_STATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VernonRhiCompareOperation {
    VERNON_RHI_COMPARE_NEVER = 0,
    VERNON_RHI_COMPARE_LESS = 1,
    VERNON_RHI_COMPARE_EQUAL = 2,
    VERNON_RHI_COMPARE_LESS_EQUAL = 3,
    VERNON_RHI_COMPARE_GREATER = 4,
    VERNON_RHI_COMPARE_NOT_EQUAL = 5,
    VERNON_RHI_COMPARE_GREATER_EQUAL = 6,
    VERNON_RHI_COMPARE_ALWAYS = 7
} VernonRhiCompareOperation;

typedef enum VernonRhiStencilOperation {
    VERNON_RHI_STENCIL_KEEP = 0,
    VERNON_RHI_STENCIL_ZERO = 1,
    VERNON_RHI_STENCIL_REPLACE = 2,
    VERNON_RHI_STENCIL_INCREMENT_CLAMP = 3,
    VERNON_RHI_STENCIL_DECREMENT_CLAMP = 4,
    VERNON_RHI_STENCIL_INVERT = 5,
    VERNON_RHI_STENCIL_INCREMENT_WRAP = 6,
    VERNON_RHI_STENCIL_DECREMENT_WRAP = 7
} VernonRhiStencilOperation;

typedef enum VernonRhiCullMode {
    VERNON_RHI_CULL_NONE = 0,
    VERNON_RHI_CULL_FRONT = 1,
    VERNON_RHI_CULL_BACK = 2
} VernonRhiCullMode;

typedef enum VernonRhiFrontFace {
    VERNON_RHI_FRONT_FACE_COUNTER_CLOCKWISE = 0,
    VERNON_RHI_FRONT_FACE_CLOCKWISE = 1
} VernonRhiFrontFace;

typedef enum VernonRhiBlendFactor {
    VERNON_RHI_BLEND_ZERO = 0,
    VERNON_RHI_BLEND_ONE = 1,
    VERNON_RHI_BLEND_SOURCE_COLOR = 2,
    VERNON_RHI_BLEND_ONE_MINUS_SOURCE_COLOR = 3,
    VERNON_RHI_BLEND_DESTINATION_COLOR = 4,
    VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_COLOR = 5,
    VERNON_RHI_BLEND_SOURCE_ALPHA = 6,
    VERNON_RHI_BLEND_ONE_MINUS_SOURCE_ALPHA = 7,
    VERNON_RHI_BLEND_DESTINATION_ALPHA = 8,
    VERNON_RHI_BLEND_ONE_MINUS_DESTINATION_ALPHA = 9
} VernonRhiBlendFactor;

typedef enum VernonRhiBlendOperation {
    VERNON_RHI_BLEND_ADD = 0,
    VERNON_RHI_BLEND_SUBTRACT = 1,
    VERNON_RHI_BLEND_REVERSE_SUBTRACT = 2,
    VERNON_RHI_BLEND_MINIMUM = 3,
    VERNON_RHI_BLEND_MAXIMUM = 4
} VernonRhiBlendOperation;

typedef enum VernonRhiColorWriteBits {
    VERNON_RHI_COLOR_WRITE_RED = 1u << 0,
    VERNON_RHI_COLOR_WRITE_GREEN = 1u << 1,
    VERNON_RHI_COLOR_WRITE_BLUE = 1u << 2,
    VERNON_RHI_COLOR_WRITE_ALPHA = 1u << 3,
    VERNON_RHI_COLOR_WRITE_ALL = (1u << 4) - 1
} VernonRhiColorWriteBits;

typedef struct VernonRasterizationState {
    uint32_t cull_mode;
    uint32_t front_face;
    uint32_t depth_clamp;
    uint32_t depth_bias_enabled;
    float depth_bias_constant;
    float depth_bias_slope;
} VernonRasterizationState;

typedef struct VernonStencilFaceState {
    uint32_t stencil_fail;
    uint32_t depth_fail;
    uint32_t pass;
    uint32_t compare;
} VernonStencilFaceState;

typedef struct VernonDepthStencilState {
    uint32_t depth_test;
    uint32_t depth_write;
    uint32_t depth_compare;
    uint32_t stencil_test;
    VernonStencilFaceState front;
    VernonStencilFaceState back;
    uint32_t stencil_read_mask;
    uint32_t stencil_write_mask;
} VernonDepthStencilState;

typedef struct VernonColorBlendState {
    uint32_t blend_enabled;
    uint32_t source_color_factor;
    uint32_t destination_color_factor;
    uint32_t color_operation;
    uint32_t source_alpha_factor;
    uint32_t destination_alpha_factor;
    uint32_t alpha_operation;
    uint32_t write_mask;
} VernonColorBlendState;

/* Source-compatible names for the RHI and provider surfaces. */
typedef VernonRasterizationState VernonRhiRasterizationState;
typedef VernonStencilFaceState VernonRhiStencilFaceState;
typedef VernonDepthStencilState VernonRhiDepthStencilState;
typedef VernonColorBlendState VernonRhiColorBlendState;
typedef VernonRasterizationState VernonRuntimeProviderRasterizationState;
typedef VernonStencilFaceState VernonRuntimeProviderStencilFaceState;
typedef VernonDepthStencilState VernonRuntimeProviderDepthStencilState;
typedef VernonColorBlendState VernonRuntimeProviderColorBlendState;

#ifdef __cplusplus
}
#endif

#endif
