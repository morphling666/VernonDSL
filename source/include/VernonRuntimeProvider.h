#ifndef VERNON_RUNTIME_PROVIDER_H
#define VERNON_RUNTIME_PROVIDER_H

#include "VernonCommon.h"
#include "VernonGraphicsState.h"
#include "VernonTextureTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS = 8,
    VERNON_RUNTIME_PROVIDER_MAX_GROUPS = 8,
    VERNON_RUNTIME_PROVIDER_MAX_VERTEX_BINDINGS = 16
};

typedef struct VernonRuntimeProviderObject {
    uint64_t value;
} VernonRuntimeProviderObject;

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
    VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE = 1u << 1
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

/* One vertex buffer binding of a graphics pipeline; the graphics variant gives each binding its stride. */
typedef struct VernonRuntimeProviderVertexBinding {
    uint32_t binding;
    /* 0 advances per vertex; n advances once every n instances. */
    uint32_t divisor;
} VernonRuntimeProviderVertexBinding;

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
 * One indexed draw of a stream, in the layout of VkDrawIndexedIndirectCommand and OpenGL's
 * DrawElementsIndirectCommand.
 */
typedef struct VernonRuntimeProviderDrawRecord {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t base_vertex;
    uint32_t first_instance;
} VernonRuntimeProviderDrawRecord;

#ifdef __cplusplus
}
#endif

#endif
