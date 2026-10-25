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
typedef struct VernonRuntimeCorePackets VernonRuntimeCorePackets;
typedef struct VernonRuntimeCorePacketBatch VernonRuntimeCorePacketBatch;

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
    /* The whole 32-bit index buffer; the draw reads index_count indices from first_index on. */
    VernonRuntimeProviderResourceReference index_buffer;
    uint32_t first_index;
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
    const VernonRuntimeProviderVertexBinding *vertex_bindings;
    size_t vertex_binding_count;
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
 * A group of values for one group layout, index-aligned with its entries. A
 * transient group is bound only by commands of `command_encoder`. Only an
 * updatable group changes: it needs a provider with update_group, and its
 * inline values are undefined until vernonRuntimeCoreUpdateGroup writes them.
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
 * A packet recording for one provider; see stored_plans.md section 11. Each
 * command is validated when it is recorded. A bind of the pipeline, group or
 * vertex buffer already bound records nothing, and binding a pipeline keeps
 * the groups bound at the indices whose group layouts it shares and every
 * bound vertex buffer. A dispatch or draw requires a group at every index of
 * the bound pipeline whose group layout has entries, and a draw a vertex
 * buffer at every vertex binding and an open rendering scope. The recording
 * keeps the pipelines and groups it names alive until it is reset.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsCreate(const VernonRuntimeDeviceProvider *provider,
                                                                     VernonRuntimeCorePackets **packets);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCorePacketsDestroy(VernonRuntimeCorePackets *packets);
/* Forgets the recorded commands and releases what they named, keeping the recording's storage. */
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCorePacketsReset(VernonRuntimeCorePackets *packets);
VERNON_RUNTIME_CORE_CAPI int vernonRuntimeCorePacketsIsEmpty(const VernonRuntimeCorePackets *packets);
/* The payload words the recorded commands wrote, which a batch of them is encoded with unless they are patched. */
VERNON_RUNTIME_CORE_CAPI const uint32_t *vernonRuntimeCorePacketsGetPayload(const VernonRuntimeCorePackets *packets,
                                                                            size_t *word_count);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsBindPipeline(VernonRuntimeCorePackets *packets,
                                                                           VernonRuntimeCorePipeline *pipeline);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsBindGraphicsVariant(
    VernonRuntimeCorePackets *packets, VernonRuntimeCoreGraphicsVariant *variant);
/*
 * A transient group makes the recording encodable only on the group's command
 * encoder.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsBindGroup(VernonRuntimeCorePackets *packets,
                                                                        uint32_t index, VernonRuntimeCoreGroup *group);
/* Binds buffers[i] at vertex binding first_binding + i. */
VERNON_RUNTIME_CORE_CAPI VernonStatus
vernonRuntimeCorePacketsBindVertexBuffers(VernonRuntimeCorePackets *packets, uint32_t first_binding,
                                          const VernonRuntimeProviderResourceReference *buffers, size_t buffer_count);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsDispatch(VernonRuntimeCorePackets *packets,
                                                                       const uint32_t group_count[3]);
/*
 * Draws are recorded between these calls and render into the rendering
 * descriptor's attachments, which Core validates against each other and the
 * render area.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsBeginRendering(
    VernonRuntimeCorePackets *packets, const VernonRuntimeProviderRenderingDescriptor *descriptor);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsEndRendering(VernonRuntimeCorePackets *packets);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsDraw(VernonRuntimeCorePackets *packets,
                                                                   const VernonRuntimeCoreDrawInvocation *invocation);
/*
 * The payload words a draw of `invocation` writes, checked as PacketsDraw
 * checks them, which patch the copy of a recording's payload that a stored
 * batch is encoded with.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreDrawPayload(const VernonRuntimeCoreDrawInvocation *invocation,
                                                                   VernonRuntimeProviderDrawPayload *payload);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsPushDebugGroup(VernonRuntimeCorePackets *packets,
                                                                             VernonStringView label);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsPopDebugGroup(VernonRuntimeCorePackets *packets);
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketsInsertDebugMarker(VernonRuntimeCorePackets *packets,
                                                                                VernonStringView label);

/*
 * An immutable provider batch of a recording that ends outside a rendering
 * scope with balanced debug groups. The batch keeps what the recording named
 * alive; it does not depend on the recording.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCorePacketBatchCreate(const VernonRuntimeCorePackets *packets,
                                                                         VernonRuntimeCorePacketBatch **batch);
VERNON_RUNTIME_CORE_CAPI void vernonRuntimeCorePacketBatchDestroy(VernonRuntimeCorePacketBatch *batch);

/* The command state of one command encoder; see VernonRuntimeDeviceProvider. */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreBeginCommands(const VernonRuntimeDeviceProvider *provider,
                                                                     VernonRuntimeProviderObject command_encoder,
                                                                     VernonRuntimeCoreCommands **commands);
/* Ends and destroys the command state, even when it reports a failure. */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEndCommands(VernonRuntimeCoreCommands *commands);
/* Called after other commands were recorded on the native encoder since the last batch. */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreResumeCommands(VernonRuntimeCoreCommands *commands);
/*
 * Writes the inline values of an updatable group between batches: batches
 * encoded before the call read its previous values and batches encoded after
 * it, including those of commands recorded but not yet encoded, the new ones.
 * `values` are the group's values with new inline bytes.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreUpdateGroup(VernonRuntimeCoreCommands *commands,
                                                                   VernonRuntimeCoreGroup *group,
                                                                   const VernonRuntimeProviderBindingValue *values,
                                                                   size_t value_count);
/*
 * Encodes `batch` with `payload`, whose words are checked against the limits
 * of the commands that wrote them: non-zero vertex and index counts and grids,
 * index ranges inside the index buffer, the stencil reference and the clear
 * values. A draw of no instances draws nothing.
 */
VERNON_RUNTIME_CORE_CAPI VernonStatus vernonRuntimeCoreEncodePackets(VernonRuntimeCoreCommands *commands,
                                                                     const VernonRuntimeCorePacketBatch *batch,
                                                                     const uint32_t *payload,
                                                                     size_t payload_word_count);

#ifdef __cplusplus
}
#endif

#endif
