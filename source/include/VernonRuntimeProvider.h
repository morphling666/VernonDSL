#ifndef VERNON_RUNTIME_PROVIDER_H
#define VERNON_RUNTIME_PROVIDER_H

#include "VernonCommon.h"
#include "VernonGraphicsState.h"
#include "VernonTextureTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { VERNON_RUNTIME_PROVIDER_MAX_SHADER_STAGES = 8, VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS = 8 };

typedef enum VernonRuntimeProviderCapabilityBits {
    VERNON_RUNTIME_PROVIDER_COMPUTE = 1u << 0,
    VERNON_RUNTIME_PROVIDER_GRAPHICS = 1u << 1,
    VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP = 1u << 2
} VernonRuntimeProviderCapabilityBits;

enum { VERNON_RUNTIME_PROVIDER_MAX_GROUPS = 8, VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS = 16 };

typedef struct VernonRuntimeProviderObject {
    uint64_t value;
} VernonRuntimeProviderObject;

typedef struct VernonRuntimeProviderDeviceIdentity {
    uint64_t adapter_id;
    uint64_t device_id;
    uint64_t driver_version;
} VernonRuntimeProviderDeviceIdentity;

typedef enum VernonRuntimeProviderPipelineKind {
    VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE = 0,
    VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE = 1
} VernonRuntimeProviderPipelineKind;

typedef enum VernonRuntimeProviderShaderStageBits {
    VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE = 1u << 0,
    VERNON_RUNTIME_PROVIDER_STAGE_VERTEX = 1u << 1,
    VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT = 1u << 2
} VernonRuntimeProviderShaderStageBits;

typedef enum VernonRuntimeProviderBindingKind {
    VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER = 0,
    VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER = 1,
    VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE = 2,
    VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE = 3,
    VERNON_RUNTIME_PROVIDER_SAMPLER = 4,
    VERNON_RUNTIME_PROVIDER_INLINE_VALUE = 5
} VernonRuntimeProviderBindingKind;

typedef enum VernonRuntimeProviderBindingInterface {
    VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE = 0,
    VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM = 1
} VernonRuntimeProviderBindingInterface;

typedef enum VernonRuntimeProviderNumericType {
    VERNON_RUNTIME_PROVIDER_I32 = 1,
    VERNON_RUNTIME_PROVIDER_U32 = 2,
    VERNON_RUNTIME_PROVIDER_F16 = 3,
    VERNON_RUNTIME_PROVIDER_F32 = 4,
    VERNON_RUNTIME_PROVIDER_F64 = 5
} VernonRuntimeProviderNumericType;

typedef enum VernonRuntimeProviderBindingValueFlags {
    VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE = 1u << 0,
    VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE = 1u << 1,
    VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE = 1u << 2
} VernonRuntimeProviderBindingValueFlags;

typedef enum VernonRuntimeProviderLoadOperation {
    VERNON_RUNTIME_PROVIDER_LOAD_CLEAR = 0,
    VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE = 1,
    VERNON_RUNTIME_PROVIDER_LOAD_DISCARD = 2
} VernonRuntimeProviderLoadOperation;

typedef enum VernonRuntimeProviderStoreOperation {
    VERNON_RUNTIME_PROVIDER_STORE_PRESERVE = 0,
    VERNON_RUNTIME_PROVIDER_STORE_DISCARD = 1
} VernonRuntimeProviderStoreOperation;

typedef struct VernonRuntimeProviderShaderDescriptor {
    uint32_t struct_size;
    uint32_t stage;
    VernonStringView format;
    const void *data;
    size_t size;
    VernonStringView entry;
    VernonStringView artifact_hash;
    uint32_t reserved[4];
} VernonRuntimeProviderShaderDescriptor;

typedef struct VernonRuntimeProviderBindingLayoutEntry {
    uint32_t slot;
    uint32_t set;
    uint32_t binding;
    VernonRuntimeProviderBindingKind kind;
    uint32_t stage_mask;
    uint32_t access;
    uint32_t array_count;
    uint32_t argument_index;
    uint32_t element_size;
    VernonRuntimeProviderBindingInterface interface_kind;
    VernonStringView name;
    uint32_t element_count;
    /* OpenGL native-uniform column count; zero for buffered/non-OpenGL bindings. */
    uint32_t vector_count;
    uint32_t element_alignment;
    uint32_t numeric_type;
    VernonTextureDimension image_dimension;
    VernonTextureFormat storage_image_format;
    VernonImageSampleResultClass sample_result_class;
} VernonRuntimeProviderBindingLayoutEntry;

typedef struct VernonRuntimeProviderVertexAttribute {
    uint32_t binding;
    uint32_t location;
    uint32_t dtype;
    uint32_t component_count;
    uint32_t relative_offset;
} VernonRuntimeProviderVertexAttribute;

/*
 * One vertex buffer binding of a graphics pipeline. Vertex buffers belong to
 * no group: encode_bind_vertex_buffers binds them, and the graphics variant
 * gives each binding its stride.
 */
typedef struct VernonRuntimeProviderVertexBinding {
    uint32_t binding;
    /* 0 advances per vertex; n advances once every n instances. */
    uint32_t divisor;
} VernonRuntimeProviderVertexBinding;

/*
 * The bindings of one group. Entry `slot` is the entry's index; `set` is
 * zero, since a group layout serves any group index: index g of a pipeline
 * layout is descriptor set, root descriptor table or argument buffer g.
 */
typedef struct VernonRuntimeProviderGroupLayoutDescriptor {
    uint32_t struct_size;
    const VernonRuntimeProviderBindingLayoutEntry *entries;
    size_t entry_count;
    uint32_t reserved[4];
} VernonRuntimeProviderGroupLayoutDescriptor;

/*
 * Group layout g binds at group index g. Core prepares one group layout per
 * distinct descriptor, so two pipeline layouts are compatible at g exactly
 * when they hold the same group layout object there.
 */
typedef struct VernonRuntimeProviderPipelineLayoutDescriptor {
    uint32_t struct_size;
    const VernonRuntimeProviderObject *group_layouts;
    size_t group_layout_count;
    const VernonRuntimeProviderVertexAttribute *vertex_attributes;
    size_t vertex_attribute_count;
    const VernonRuntimeProviderVertexBinding *vertex_bindings;
    size_t vertex_binding_count;
    uint32_t push_constant_size;
    uint32_t reserved[4];
} VernonRuntimeProviderPipelineLayoutDescriptor;

typedef struct VernonRuntimeProviderPipelineDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderPipelineKind kind;
    uint32_t required_capabilities;
    const VernonRuntimeProviderObject *shaders;
    size_t shader_count;
    VernonRuntimeProviderObject layout;
    uint32_t topology;
    const uint32_t *color_formats;
    size_t color_format_count;
    uint32_t depth_stencil_format;
    uint32_t sample_count;
    uint32_t workgroup_size[3];
    const uint32_t *vertex_strides;
    size_t vertex_stride_count;
    VernonRasterizationState rasterization;
    VernonDepthStencilState depth_stencil;
    const VernonColorBlendState *color_blends;
    size_t color_blend_count;
    uint32_t reserved[4];
} VernonRuntimeProviderPipelineDescriptor;

typedef struct VernonRuntimeProviderResourceReference {
    uint64_t identity;
    VernonRuntimeProviderObject resource;
    uint64_t offset;
    /* Byte length of the referenced range, which starts at `offset`. */
    uint64_t size;
} VernonRuntimeProviderResourceReference;

typedef struct VernonRuntimeProviderInlineValue {
    const void *data;
    size_t size;
} VernonRuntimeProviderInlineValue;

typedef struct VernonRuntimeProviderBufferReference {
    VernonRuntimeProviderResourceReference resource;
    uint32_t stride;
} VernonRuntimeProviderBufferReference;

typedef struct VernonRuntimeProviderImageReference {
    VernonRuntimeProviderResourceReference view;
} VernonRuntimeProviderImageReference;

typedef struct VernonRuntimeProviderSamplerReference {
    VernonRuntimeProviderResourceReference resource;
} VernonRuntimeProviderSamplerReference;

typedef struct VernonRuntimeProviderBindingValue {
    uint32_t slot;
    VernonRuntimeProviderBindingKind kind;
    uint32_t flags;
    union {
        VernonRuntimeProviderInlineValue inline_value;
        VernonRuntimeProviderBufferReference buffer;
        VernonRuntimeProviderImageReference image;
        VernonRuntimeProviderSamplerReference sampler;
    } payload;
} VernonRuntimeProviderBindingValue;

typedef enum VernonRuntimeProviderImageResourceKind {
    VERNON_RUNTIME_PROVIDER_IMAGE_OWNER = 0,
    VERNON_RUNTIME_PROVIDER_IMAGE_VIEW = 1
} VernonRuntimeProviderImageResourceKind;

typedef struct VernonRuntimeProviderImageDescription {
    uint32_t struct_size;
    VernonImageDescriptor image;
    VernonImageViewDescriptor view;
    uint64_t parent_identity;
    VernonRuntimeProviderImageResourceKind resource_kind;
    uint32_t reserved[3];
} VernonRuntimeProviderImageDescription;

typedef enum VernonRuntimeProviderGroupLifetime {
    VERNON_RUNTIME_PROVIDER_GROUP_PERSISTENT = 0,
    /* Bound only by commands of `command_encoder`, which may own its memory. */
    VERNON_RUNTIME_PROVIDER_GROUP_TRANSIENT = 1,
    /* Persistent; its inline values are undefined until update_group writes them. Made without a command encoder. */
    VERNON_RUNTIME_PROVIDER_GROUP_UPDATABLE = 2
} VernonRuntimeProviderGroupLifetime;

/*
 * `values` are index-aligned with the group layout's entries. The group
 * copies inline values and keeps its resources alive until it is destroyed
 * and no command that bound it is pending.
 */
typedef struct VernonRuntimeProviderGroupDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderObject group_layout;
    const VernonRuntimeProviderBindingValue *values;
    size_t value_count;
    VernonRuntimeProviderGroupLifetime lifetime;
    VernonRuntimeProviderObject command_encoder;
    uint32_t reserved[4];
} VernonRuntimeProviderGroupDescriptor;

typedef struct VernonRuntimeProviderColorAttachment {
    uint32_t location;
    VernonRuntimeProviderResourceReference view;
    VernonRuntimeProviderLoadOperation load_operation;
    VernonRuntimeProviderStoreOperation store_operation;
    VernonClearColor clear_color;
} VernonRuntimeProviderColorAttachment;

typedef struct VernonRuntimeProviderDepthStencilAttachment {
    VernonRuntimeProviderResourceReference view;
    VernonRuntimeProviderLoadOperation depth_load_operation;
    VernonRuntimeProviderStoreOperation depth_store_operation;
    float clear_depth;
    VernonRuntimeProviderLoadOperation stencil_load_operation;
    VernonRuntimeProviderStoreOperation stencil_store_operation;
    uint32_t clear_stencil;
} VernonRuntimeProviderDepthStencilAttachment;

/*
 * The attachments that draws between begin_rendering and end_rendering render
 * into. Load operations apply when the scope begins and store operations when
 * it ends; render_area is {x, y, width, height}.
 *
 * sampled_images lists the images and image views the scope's draws sample;
 * they must not overlap an attachment. The provider makes them readable
 * before the scope opens, because backends such as Vulkan cannot change an
 * image's layout inside a render pass: a draw may fail when it samples an
 * image that is not listed and not already readable.
 */
typedef struct VernonRuntimeProviderRenderingDescriptor {
    uint32_t struct_size;
    const VernonRuntimeProviderColorAttachment *color_attachments;
    size_t color_attachment_count;
    const VernonRuntimeProviderDepthStencilAttachment *depth_stencil_attachment;
    uint32_t render_area[4];
    uint32_t layers;
    const VernonRuntimeProviderResourceReference *sampled_images;
    size_t sampled_image_count;
    uint32_t reserved[4];
} VernonRuntimeProviderRenderingDescriptor;

/*
 * A packet is a uint32_t header, the opcode in the low 16 bits and the
 * packet's word count including the header in the high 16 bits, followed by
 * its operands. Object operands index the batch's objects, resource operands
 * its resources and rendering operands its renderings; payload operands index
 * the payload words passed to encode_packets.
 */
typedef enum VernonRuntimeProviderPacketOpcode {
    /* rendering, payload of one VernonClearColor per color attachment, then the depth clear value as a float and
     * the stencil clear value */
    VERNON_RUNTIME_PROVIDER_PACKET_BEGIN_RENDERING = 1,
    VERNON_RUNTIME_PROVIDER_PACKET_END_RENDERING = 2,
    /* pipeline object */
    VERNON_RUNTIME_PROVIDER_PACKET_BIND_PIPELINE = 3,
    /* group index, group object */
    VERNON_RUNTIME_PROVIDER_PACKET_BIND_GROUP = 4,
    /* first binding, then one buffer resource per binding */
    VERNON_RUNTIME_PROVIDER_PACKET_BIND_VERTEX_BUFFERS = 5,
    /* topology, payload of a VernonRuntimeProviderDrawPayload */
    VERNON_RUNTIME_PROVIDER_PACKET_DRAW = 6,
    /* topology, resource of the 32-bit index buffer, payload of a VernonRuntimeProviderDrawPayload */
    VERNON_RUNTIME_PROVIDER_PACKET_DRAW_INDEXED = 7,
    /* payload of the three group counts */
    VERNON_RUNTIME_PROVIDER_PACKET_DISPATCH = 8,
    /* label byte count, then the label bytes in whole words */
    VERNON_RUNTIME_PROVIDER_PACKET_PUSH_DEBUG_GROUP = 9,
    VERNON_RUNTIME_PROVIDER_PACKET_POP_DEBUG_GROUP = 10,
    /* label byte count, then the label bytes in whole words */
    VERNON_RUNTIME_PROVIDER_PACKET_INSERT_DEBUG_MARKER = 11
} VernonRuntimeProviderPacketOpcode;

/* The payload words of a draw; every member is one word. */
typedef struct VernonRuntimeProviderDrawPayload {
    /* The index count of an indexed draw. */
    uint32_t vertex_count;
    /* A draw of no instances draws nothing; providers skip it. */
    uint32_t instance_count;
    uint32_t first_instance;
    /* Indexed draws only: added to each index before vertex fetch. */
    int32_t base_vertex;
    /* Indexed draws only: the first index read from the index buffer. */
    uint32_t first_index;
    uint32_t viewport[4];
    uint32_t scissor[4];
    uint32_t stencil_reference;
} VernonRuntimeProviderDrawPayload;

/*
 * A recorded command stream; see stored_plans.md section 11. The packets are
 * valid: each draw and dispatch has a compatible pipeline, a group at every
 * group index it requires and, for a draw, a vertex buffer at every vertex
 * binding and an open rendering scope. The batch starts with nothing bound and
 * ends outside a rendering scope with balanced debug groups.
 */
typedef struct VernonRuntimeProviderPacketBatchDescriptor {
    uint32_t struct_size;
    const uint32_t *packets;
    size_t packet_word_count;
    /* Pipelines and groups. */
    const VernonRuntimeProviderObject *objects;
    size_t object_count;
    /* Vertex and index buffers. */
    const VernonRuntimeProviderResourceReference *resources;
    size_t resource_count;
    /* Their clear values are ignored; BEGIN_RENDERING reads them from the payload. */
    const VernonRuntimeProviderRenderingDescriptor *renderings;
    size_t rendering_count;
    /* encode_packets receives at least this many payload words. */
    size_t payload_word_count;
    uint32_t reserved[4];
} VernonRuntimeProviderPacketBatchDescriptor;

/*
 * Commands are encoded between begin_commands and end_commands, which open
 * and close the provider's command state for one command encoder; an encoder
 * has at most one open command state. Commands are encoded as packet batches.
 * The owner of the commands may record other commands on the native encoder,
 * such as barriers, copies and debug groups, only between batches and only
 * before calling resume_commands, after which the provider reapplies the
 * native state those commands may have changed before its next batch. A
 * provider skips native calls whose state is already current and must not
 * assume that native state outside its commands, such as an OpenGL context
 * shared with other renderers, is unchanged between them.
 *
 * Preparation callbacks, create_group and create_packet_batch may allocate
 * and look objects up. A batch keeps the objects and resources it names alive
 * until it is destroyed and no command that encoded it is pending. The other
 * command callbacks are the invocation hot path and must not parse, perform
 * name lookup, create stable objects, or make unbounded heap allocations.
 * end_commands reports failures the provider detects only after encoding,
 * such as OpenGL errors.
 */
typedef struct VernonRuntimeDeviceProvider {
    uint32_t struct_size;
    uint32_t abi_version;
    void *user_data;

    uint32_t (*get_capabilities)(void *user_data);
    VernonRuntimeProviderDeviceIdentity (*get_device_identity)(void *user_data);

    VernonStatus (*prepare_shader)(void *user_data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                   VernonRuntimeProviderObject *shader);
    VernonStatus (*prepare_group_layout)(void *user_data, const VernonRuntimeProviderGroupLayoutDescriptor *descriptor,
                                         VernonRuntimeProviderObject *group_layout);
    VernonStatus (*prepare_pipeline_layout)(void *user_data,
                                            const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                            VernonRuntimeProviderObject *layout);
    VernonStatus (*prepare_pipeline)(void *user_data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                     VernonRuntimeProviderObject *pipeline);

    VernonStatus (*describe_image)(void *user_data, VernonRuntimeProviderResourceReference view,
                                   VernonRuntimeProviderImageDescription *description);
    VernonStatus (*create_group)(void *user_data, const VernonRuntimeProviderGroupDescriptor *descriptor,
                                 VernonRuntimeProviderObject *group);
    VernonStatus (*create_packet_batch)(void *user_data, const VernonRuntimeProviderPacketBatchDescriptor *descriptor,
                                        VernonRuntimeProviderObject *batch);

    VernonStatus (*begin_commands)(void *user_data, VernonRuntimeProviderObject command_encoder,
                                   VernonRuntimeProviderObject *commands);
    VernonStatus (*end_commands)(void *user_data, VernonRuntimeProviderObject commands);
    VernonStatus (*resume_commands)(void *user_data, VernonRuntimeProviderObject commands);
    VernonStatus (*encode_packets)(void *user_data, VernonRuntimeProviderObject commands,
                                   VernonRuntimeProviderObject batch, const uint32_t *payload,
                                   size_t payload_word_count);
    /*
     * Optional. Writes the inline values of an UPDATABLE group between batches of `commands`, outside rendering
     * scopes: batches encoded before the call read the previous values and batches encoded after it the new ones, on
     * every encoder. `values` are index-aligned with the group layout's entries and name the group's resources; only
     * inline values are read. The provider leaves `commands` ready for their next batch.
     */
    VernonStatus (*update_group)(void *user_data, VernonRuntimeProviderObject commands,
                                 VernonRuntimeProviderObject group, const VernonRuntimeProviderBindingValue *values,
                                 size_t value_count);

    void (*destroy_shader)(void *user_data, VernonRuntimeProviderObject shader);
    void (*destroy_group_layout)(void *user_data, VernonRuntimeProviderObject group_layout);
    void (*destroy_pipeline_layout)(void *user_data, VernonRuntimeProviderObject layout);
    void (*destroy_pipeline)(void *user_data, VernonRuntimeProviderObject pipeline);
    void (*destroy_group)(void *user_data, VernonRuntimeProviderObject group);
    void (*destroy_packet_batch)(void *user_data, VernonRuntimeProviderObject batch);
    uint32_t reserved[8];
} VernonRuntimeDeviceProvider;

#ifdef __cplusplus
}
#endif

#endif
