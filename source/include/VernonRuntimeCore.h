#ifndef VERNON_RUNTIME_CORE_H
#define VERNON_RUNTIME_CORE_H

#include "VernonRuntimeProvider.h"

#if defined(VERNON_RUNTIME_CORE_STATIC)
#define VERNON_RUNTIME_CORE_CAPI
#elif defined(_WIN32) && defined(VERNON_RUNTIME_CORE_BUILD)
#define VERNON_RUNTIME_CORE_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_RUNTIME_CORE_CAPI __declspec(dllimport)
#else
#define VERNON_RUNTIME_CORE_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonRuntimeCorePipeline VernonRuntimeCorePipeline;
typedef struct VernonRuntimeCoreBindings VernonRuntimeCoreBindings;
typedef struct VernonRuntimeCoreGraphicsVariant VernonRuntimeCoreGraphicsVariant;

typedef struct VernonRuntimeCoreGraphicsCompatibility {
    uint32_t struct_size;
    uint32_t topology;
    const uint32_t *color_formats;
    size_t color_format_count;
    uint32_t depth_stencil_format;
    uint32_t sample_count;
    const uint32_t *vertex_strides;
    size_t vertex_stride_count;
    VernonRasterizationState rasterization;
    VernonDepthStencilState depth_stencil;
    const VernonColorBlendState *color_blends;
    size_t color_blend_count;
    uint32_t reserved[4];
} VernonRuntimeCoreGraphicsCompatibility;

typedef struct VernonRuntimeCoreDrawInvocation {
    uint32_t struct_size;
    VernonRuntimeProviderObject command_encoder;
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
    const VernonRuntimeProviderColorAttachment *color_attachments;
    size_t color_attachment_count;
    VernonRuntimeProviderResourceReference depth_stencil_view;
    VernonRuntimeProviderLoadOperation depth_load_operation;
    VernonRuntimeProviderStoreOperation depth_store_operation;
    float clear_depth;
    uint32_t viewport[4];
    uint32_t scissor[4];
    uint32_t topology;
    VernonRuntimeProviderResourceReference index_buffer;
    uint32_t index_count;
    uint32_t index_type;
    VernonRuntimeProviderLoadOperation stencil_load_operation;
    VernonRuntimeProviderStoreOperation stencil_store_operation;
    uint32_t clear_stencil;
    uint32_t stencil_reference;
    uint32_t render_area[4];
} VernonRuntimeCoreDrawInvocation;

typedef struct VernonRuntimeCorePipelineDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderPipelineKind kind;
    uint32_t required_capabilities;
    const VernonRuntimeProviderShaderDescriptor *shaders;
    size_t shader_count;
    const VernonRuntimeProviderBindingLayoutEntry *bindings;
    size_t binding_count;
    const VernonRuntimeProviderVertexAttribute *vertex_attributes;
    size_t vertex_attribute_count;
    uint32_t push_constant_size;
    uint32_t topology;
    const uint32_t *color_formats;
    size_t color_format_count;
    uint32_t depth_stencil_format;
    uint32_t sample_count;
    uint32_t workgroup_size[3];
    VernonRasterizationState rasterization;
    VernonDepthStencilState depth_stencil;
    const VernonColorBlendState *color_blends;
    size_t color_blend_count;
    uint32_t reserved[4];
} VernonRuntimeCorePipelineDescriptor;

VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePreparePipeline(
    const VernonRuntimeDeviceProvider *provider, const VernonRuntimeCorePipelineDescriptor *descriptor,
    VernonRuntimeCorePipeline **pipeline);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCorePipelineDestroy(VernonRuntimeCorePipeline *pipeline);
VERNON_RUNTIME_CORE_CAPI VernonRuntimeProviderDeviceIdentity
vernonRuntimeCorePipelineGetDeviceIdentity(const VernonRuntimeCorePipeline *pipeline);

VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreCreateBindings(VernonRuntimeCorePipeline *pipeline,
                                                                      const VernonRuntimeProviderBindingValue *values,
                                                                      size_t value_count,
                                                                      VernonRuntimeCoreBindings **bindings);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreUpdateBindings(VernonRuntimeCoreBindings *bindings,
                                                                      const VernonRuntimeProviderBindingValue *values,
                                                                      size_t value_count);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCoreBindingsDestroy(VernonRuntimeCoreBindings *bindings);

VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePrepareGraphicsVariant(
    VernonRuntimeCorePipeline *pipeline, const VernonRuntimeCoreGraphicsCompatibility *compatibility,
    VernonRuntimeCoreGraphicsVariant **variant);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCoreGraphicsVariantDestroy(VernonRuntimeCoreGraphicsVariant *variant);

VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEncodeDispatch(const VernonRuntimeCorePipeline *pipeline,
                                                                      const VernonRuntimeCoreBindings *bindings,
                                                                      VernonRuntimeProviderObject command_encoder,
                                                                      const uint32_t group_count[3],
                                                                      const void *push_constants,
                                                                      size_t push_constant_size);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEncodeDraw(const VernonRuntimeCorePipeline *pipeline,
                                                                  const VernonRuntimeCoreBindings *bindings,
                                                                  VernonRuntimeProviderObject command_encoder,
                                                                  uint32_t vertex_count, uint32_t instance_count,
                                                                  uint32_t first_vertex, uint32_t first_instance);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEncodeDrawInvocation(
    const VernonRuntimeCorePipeline *pipeline, const VernonRuntimeCoreBindings *bindings,
    const VernonRuntimeCoreDrawInvocation *invocation);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(
    const VernonRuntimeCoreGraphicsVariant *variant, const VernonRuntimeCoreBindings *bindings,
    const VernonRuntimeCoreDrawInvocation *invocation);

#ifdef __cplusplus
}
#endif

#endif
