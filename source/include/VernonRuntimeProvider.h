#ifndef VERNON_RUNTIME_PROVIDER_H
#define VERNON_RUNTIME_PROVIDER_H

#include "VernonCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { VERNON_RUNTIME_PROVIDER_MAX_SHADER_STAGES = 8 };

typedef enum VernonRuntimeProviderCapabilityBits {
    VERNON_RUNTIME_PROVIDER_COMPUTE = 1u << 0,
    VERNON_RUNTIME_PROVIDER_GRAPHICS = 1u << 1,
    VERNON_RUNTIME_PROVIDER_NATIVE_INTEROP = 1u << 2
} VernonRuntimeProviderCapabilityBits;

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
    VERNON_RUNTIME_PROVIDER_INLINE_VALUE = 5,
    VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER = 6
} VernonRuntimeProviderBindingKind;

typedef enum VernonRuntimeProviderBindingInterface {
    VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE = 0,
    VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM = 1,
    VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT = 2
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
    VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE = 1u << 1
} VernonRuntimeProviderBindingValueFlags;

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
    uint32_t vector_count;
    uint32_t divisor;
    uint32_t element_alignment;
} VernonRuntimeProviderBindingLayoutEntry;

typedef struct VernonRuntimeProviderVertexAttribute {
    uint32_t binding;
    uint32_t location;
    uint32_t dtype;
    uint32_t component_count;
    uint32_t relative_offset;
} VernonRuntimeProviderVertexAttribute;

typedef struct VernonRuntimeProviderPipelineLayoutDescriptor {
    uint32_t struct_size;
    const VernonRuntimeProviderBindingLayoutEntry *bindings;
    size_t binding_count;
    const VernonRuntimeProviderVertexAttribute *vertex_attributes;
    size_t vertex_attribute_count;
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
    uint32_t reserved[4];
} VernonRuntimeProviderPipelineDescriptor;

typedef struct VernonRuntimeProviderResourceReference {
    uint64_t identity;
    VernonRuntimeProviderObject resource;
    uint64_t offset;
    uint64_t size;
} VernonRuntimeProviderResourceReference;

typedef struct VernonRuntimeProviderBindingValue {
    uint32_t slot;
    VernonRuntimeProviderBindingKind kind;
    VernonRuntimeProviderResourceReference resource;
    const void *inline_data;
    size_t inline_size;
    uint32_t flags;
    uint32_t stride;
} VernonRuntimeProviderBindingValue;

typedef struct VernonRuntimeProviderBindingSetDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderObject layout;
    const VernonRuntimeProviderBindingValue *values;
    size_t value_count;
    uint32_t reserved[4];
} VernonRuntimeProviderBindingSetDescriptor;

typedef struct VernonRuntimeProviderDispatchDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderObject pipeline;
    VernonRuntimeProviderObject bindings;
    uint32_t group_count[3];
    const void *push_constants;
    size_t push_constant_size;
    uint32_t reserved[4];
} VernonRuntimeProviderDispatchDescriptor;

typedef struct VernonRuntimeProviderColorAttachment {
    uint32_t location;
    VernonRuntimeProviderResourceReference image;
    uint32_t load_operation;
    uint32_t store_operation;
    float clear_color[4];
} VernonRuntimeProviderColorAttachment;

typedef struct VernonRuntimeProviderDrawDescriptor {
    uint32_t struct_size;
    VernonRuntimeProviderObject pipeline;
    VernonRuntimeProviderObject bindings;
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
    const VernonRuntimeProviderColorAttachment *color_attachments;
    size_t color_attachment_count;
    VernonRuntimeProviderResourceReference depth_stencil_attachment;
    uint32_t depth_load_operation;
    uint32_t depth_store_operation;
    float clear_depth;
    uint32_t viewport[4];
    uint32_t scissor[4];
    uint32_t topology;
    VernonRuntimeProviderResourceReference index_buffer;
    uint32_t index_count;
    uint32_t index_type;
    uint32_t reserved[4];
} VernonRuntimeProviderDrawDescriptor;

/*
 * Preparation callbacks may allocate. encode_dispatch and encode_draw are the
 * invocation hot path and must not parse, perform name lookup, create stable
 * objects, or make unbounded heap allocations.
 */
typedef struct VernonRuntimeDeviceProvider {
    uint32_t struct_size;
    uint32_t abi_version;
    void *user_data;

    uint32_t (*get_capabilities)(void *user_data);
    VernonRuntimeProviderDeviceIdentity (*get_device_identity)(void *user_data);

    VernonStatus (*prepare_shader)(void *user_data, const VernonRuntimeProviderShaderDescriptor *descriptor,
                                   VernonRuntimeProviderObject *shader);
    VernonStatus (*prepare_pipeline_layout)(void *user_data,
                                            const VernonRuntimeProviderPipelineLayoutDescriptor *descriptor,
                                            VernonRuntimeProviderObject *layout);
    VernonStatus (*prepare_pipeline)(void *user_data, const VernonRuntimeProviderPipelineDescriptor *descriptor,
                                     VernonRuntimeProviderObject *pipeline);

    VernonStatus (*retain_resource)(void *user_data, VernonRuntimeProviderResourceReference resource);
    void (*release_resource)(void *user_data, VernonRuntimeProviderResourceReference resource);
    VernonStatus (*create_binding_set)(void *user_data, const VernonRuntimeProviderBindingSetDescriptor *descriptor,
                                       VernonRuntimeProviderObject *bindings);
    VernonStatus (*update_binding_set)(void *user_data, VernonRuntimeProviderObject bindings,
                                       const VernonRuntimeProviderBindingValue *values, size_t value_count);

    VernonStatus (*encode_dispatch)(void *user_data, VernonRuntimeProviderObject command_encoder,
                                    const VernonRuntimeProviderDispatchDescriptor *descriptor);
    VernonStatus (*encode_draw)(void *user_data, VernonRuntimeProviderObject command_encoder,
                                const VernonRuntimeProviderDrawDescriptor *descriptor);

    void (*destroy_shader)(void *user_data, VernonRuntimeProviderObject shader);
    void (*destroy_pipeline_layout)(void *user_data, VernonRuntimeProviderObject layout);
    void (*destroy_pipeline)(void *user_data, VernonRuntimeProviderObject pipeline);
    void (*destroy_binding_set)(void *user_data, VernonRuntimeProviderObject bindings);
    uint32_t reserved[8];
} VernonRuntimeDeviceProvider;

#ifdef __cplusplus
}
#endif

#endif
