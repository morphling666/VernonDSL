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

typedef struct VernonRuntimeCoreGroupLayout VernonRuntimeCoreGroupLayout;
typedef struct VernonRuntimeCorePipeline VernonRuntimeCorePipeline;
typedef struct VernonRuntimeCoreGroup VernonRuntimeCoreGroup;
typedef struct VernonRuntimeCoreGraphicsVariant VernonRuntimeCoreGraphicsVariant;
typedef struct VernonRuntimeCoreCommands VernonRuntimeCoreCommands;

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
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_instance;
    /* Indexed draws only: added to each index before vertex fetch. */
    int32_t base_vertex;
    uint32_t viewport[4];
    uint32_t scissor[4];
    uint32_t topology;
    VernonRuntimeProviderResourceReference index_buffer;
    uint32_t index_count;
    uint32_t index_type;
    uint32_t stencil_reference;
} VernonRuntimeCoreDrawInvocation;

typedef struct VernonRuntimeCorePipelineDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderPipelineKind kind;
    uint32_t required_capabilities;
    const VernonRuntimeProviderShaderDescriptor *shaders;
    size_t shader_count;
    /* Group layout g binds at group index g; an index without bindings takes a group layout without entries. */
    VernonRuntimeCoreGroupLayout *const *group_layouts;
    size_t group_layout_count;
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

/*
 * Equal entries prepare the same group layout object, which is what makes two
 * pipelines compatible at a group index. Entry `slot` is the entry's index;
 * `set` is ignored. Group layouts are reference counted: pipelines and groups
 * keep theirs alive.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePrepareGroupLayout(
    const VernonRuntimeDeviceProvider *provider, const VernonRuntimeProviderBindingLayoutEntry *entries,
    size_t entry_count, VernonRuntimeCoreGroupLayout **group_layout);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCoreGroupLayoutDestroy(VernonRuntimeCoreGroupLayout *group_layout);

VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePreparePipeline(
    const VernonRuntimeDeviceProvider *provider, const VernonRuntimeCorePipelineDescriptor *descriptor,
    VernonRuntimeCorePipeline **pipeline);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCorePipelineDestroy(VernonRuntimeCorePipeline *pipeline);
VERNON_RUNTIME_CORE_CAPI VernonRuntimeProviderDeviceIdentity
vernonRuntimeCorePipelineGetDeviceIdentity(const VernonRuntimeCorePipeline *pipeline);

/*
 * An immutable group of values for one group layout, index-aligned with its
 * entries. A transient group is bound only by commands of `command_encoder`.
 * Destroying a group that pending commands bound keeps it alive until they
 * complete.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreCreateGroup(VernonRuntimeCoreGroupLayout *group_layout,
                                                                   const VernonRuntimeProviderBindingValue *values,
                                                                   size_t value_count,
                                                                   VernonRuntimeProviderGroupLifetime lifetime,
                                                                   VernonRuntimeProviderObject command_encoder,
                                                                   VernonRuntimeCoreGroup **group);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCoreGroupDestroy(VernonRuntimeCoreGroup *group);

VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePrepareGraphicsVariant(
    VernonRuntimeCorePipeline *pipeline, const VernonRuntimeCoreGraphicsCompatibility *compatibility,
    VernonRuntimeCoreGraphicsVariant **variant);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCoreGraphicsVariantDestroy(VernonRuntimeCoreGraphicsVariant *variant);

/*
 * The command state of one command encoder; see VernonRuntimeDeviceProvider.
 * A bind of the pipeline or group already bound is free, and binding a
 * pipeline keeps the groups bound at the indices whose group layouts it
 * shares. A dispatch or draw requires a group at every index of the bound
 * pipeline whose group layout has entries.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreBeginCommands(const VernonRuntimeDeviceProvider *provider,
                                                                     VernonRuntimeProviderObject command_encoder,
                                                                     VernonRuntimeCoreCommands **commands);
/* Ends and destroys the command state, even when it reports a failure. */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEndCommands(VernonRuntimeCoreCommands *commands);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreBindPipeline(VernonRuntimeCoreCommands *commands,
                                                                    VernonRuntimeCorePipeline *pipeline);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreBindGraphicsVariant(VernonRuntimeCoreCommands *commands,
                                                                           VernonRuntimeCoreGraphicsVariant *variant);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreBindGroup(VernonRuntimeCoreCommands *commands, uint32_t index,
                                                                 VernonRuntimeCoreGroup *group);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreDispatch(VernonRuntimeCoreCommands *commands,
                                                                const uint32_t group_count[3],
                                                                const void *push_constants, size_t push_constant_size);
/*
 * Draws are encoded between these calls and render into the rendering
 * descriptor's attachments, which Core validates against each other and the
 * render area.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreBeginRendering(
    VernonRuntimeCoreCommands *commands, const VernonRuntimeProviderRenderingDescriptor *descriptor);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEndRendering(VernonRuntimeCoreCommands *commands);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreDraw(VernonRuntimeCoreCommands *commands,
                                                            const VernonRuntimeCoreDrawInvocation *invocation);

#ifdef __cplusplus
}
#endif

#endif
