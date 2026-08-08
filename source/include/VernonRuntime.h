#ifndef VERNON_C_RUNTIME_H
#define VERNON_C_RUNTIME_H

#include "VernonCommon.h"
#include "VernonOpenGLContext.h"
#include "VernonRHI.h"
#include "VernonRuntimeProvider.h"

#if defined(VERNON_RUNTIME_STATIC)
#define VERNON_RUNTIME_CAPI
#elif defined(_WIN32) && defined(VERNON_RUNTIME_BUILD)
#define VERNON_RUNTIME_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_RUNTIME_CAPI __declspec(dllimport)
#else
#define VERNON_RUNTIME_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonRuntimeContext VernonRuntimeContext;
typedef struct VernonPipelineBundle VernonPipelineBundle;
typedef struct VernonLoadedPipeline VernonLoadedPipeline;
typedef struct VernonPullback VernonPullback;

typedef enum VernonRuntimeBackend {
    VERNON_RUNTIME_CPU = 0,
    VERNON_RUNTIME_CUDA = 1,
    VERNON_RUNTIME_VULKAN = 2,
    VERNON_RUNTIME_OPENGL = 3,
    VERNON_RUNTIME_OPENGL_ES = 4,
    VERNON_RUNTIME_DIRECTX12 = 5,
    VERNON_RUNTIME_METAL = 6
} VernonRuntimeBackend;

typedef struct VernonRuntimeCapabilities {
    uint8_t available;
    uint8_t supports_compute;
    uint8_t supports_graphics;
    uint8_t supports_storage_buffers;
    uint16_t api_version_major;
    uint16_t api_version_minor;
    uint32_t graphics_draw_abi_version;
    VernonStringView diagnostic;
} VernonRuntimeCapabilities;

typedef struct VernonRuntimeCreateOptions {
    uint32_t struct_size;
    uint32_t device_index;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonRuntimeCreateOptions;

typedef struct VernonLaunchSize {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} VernonLaunchSize;

VERNON_RUNTIME_CAPI VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend);
VERNON_RUNTIME_CAPI VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                                         const VernonRuntimeCreateOptions *options);
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreateForRhiDevice(VernonRuntimeBackend backend,
                                                                          VernonRhiDevice device);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context);

typedef enum VernonTextureFormat {
    VERNON_TEXTURE_RGBA8_UNORM = 0,
    VERNON_TEXTURE_RGBA8_SRGB = 1,
    VERNON_TEXTURE_RGBA16_FLOAT = 2,
    VERNON_TEXTURE_RGBA32_FLOAT = 3,
    VERNON_TEXTURE_R8_UNORM = 4,
    VERNON_TEXTURE_R16_FLOAT = 5,
    VERNON_TEXTURE_R32_FLOAT = 6,
    VERNON_TEXTURE_RG8_UNORM = 7,
    VERNON_TEXTURE_RGB8_UNORM = 8,
    VERNON_TEXTURE_R11G11B10_FLOAT = 9,
    VERNON_TEXTURE_D32_FLOAT = 10,
    VERNON_TEXTURE_D32_FLOAT_S8_UINT = 11
} VernonTextureFormat;

typedef enum VernonTextureDimension {
    VERNON_TEXTURE_2D = 0,
    VERNON_TEXTURE_3D = 1,
    VERNON_TEXTURE_CUBE = 2
} VernonTextureDimension;

VERNON_RUNTIME_CAPI VernonLoadedPipeline *vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                                                                    size_t artifact_size, const char *reflection,
                                                                    size_t reflection_size, const char *entry,
                                                                    size_t entry_size);
VERNON_RUNTIME_CAPI VernonLoadedPipeline *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context,
                                                                    VernonCpuEntryPoint entry_point,
                                                                    const char *reflection, size_t reflection_size,
                                                                    const char *entry, size_t entry_size);
/*
 * Registers an AOT entry that was statically linked into the application.
 * Re-registering the same symbol and pointer is idempotent.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeRegisterStaticCpuEntry(VernonStringView symbol,
                                                                     VernonCpuEntryPoint entry_point);

typedef enum VernonIndexType { VERNON_INDEX_U32 = 0 } VernonIndexType;

typedef enum VernonPrimitiveTopology {
    VERNON_TOPOLOGY_TRIANGLE_LIST = 0,
    VERNON_TOPOLOGY_LINE_LIST = 1,
    VERNON_TOPOLOGY_POINT_LIST = 2
} VernonPrimitiveTopology;

typedef struct VernonIndexBinding {
    VernonIndexType type;
    size_t offset;
    uint32_t index_count;
    VernonRuntimeProviderResourceReference resource;
} VernonIndexBinding;

typedef struct VernonColorAttachment {
    uint32_t location;
    VernonRuntimeProviderResourceReference resource;
    uint32_t width;
    uint32_t height;
    VernonTextureFormat format;
    VernonRhiLoadOperation load_operation;
    VernonRhiStoreOperation store_operation;
    float clear_color[4];
} VernonColorAttachment;

typedef struct VernonDepthAttachment {
    VernonRuntimeProviderResourceReference resource;
    uint32_t width;
    uint32_t height;
    VernonTextureFormat format;
    VernonRhiLoadOperation load_operation;
    VernonRhiStoreOperation store_operation;
    float clear_depth;
    VernonRhiLoadOperation stencil_load_operation;
    VernonRhiStoreOperation stencil_store_operation;
    uint32_t clear_stencil;
} VernonDepthAttachment;

typedef struct VernonGraphicsState {
    uint32_t struct_size;
    VernonRasterizationState rasterization;
    VernonDepthStencilState depth_stencil;
    const VernonColorBlendState *color_blends;
    size_t color_blend_count;
    uint32_t reserved[4];
} VernonGraphicsState;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiBuffer(VernonRuntimeContext *context, VernonRhiBuffer buffer,
                                                                 uint64_t offset, uint64_t size,
                                                                 VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiImage(VernonRuntimeContext *context, VernonRhiImage image,
                                                                VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiSampler(VernonRuntimeContext *context,
                                                                  VernonRhiSampler sampler,
                                                                  VernonRuntimeProviderResourceReference *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeReferenceRhiCommandEncoder(VernonRuntimeContext *context,
                                                                         VernonRhiCommandEncoder encoder,
                                                                         VernonRuntimeProviderObject *output);

typedef struct VernonFeatureSetView {
    const char *const *names;
    size_t count;
} VernonFeatureSetView;

typedef enum VernonDataType {
    VERNON_DATA_BOOL = 0,
    VERNON_DATA_I32 = 1,
    VERNON_DATA_U32 = 2,
    VERNON_DATA_F16 = 3,
    VERNON_DATA_F32 = 4,
    VERNON_DATA_F64 = 5,
    VERNON_DATA_U8 = 6
} VernonDataType;

typedef enum VernonValueAccess {
    VERNON_ACCESS_READ = 0,
    VERNON_ACCESS_WRITE = 1,
    VERNON_ACCESS_READ_WRITE = 2
} VernonValueAccess;

typedef enum VernonTensorStorage { VERNON_TENSOR_HOST = 0, VERNON_TENSOR_RHI_RESOURCE = 1 } VernonTensorStorage;

typedef struct VernonValueLeafView {
    uint32_t dtype;
    uint32_t scalar_count;
    uint32_t byte_offset;
} VernonValueLeafView;

typedef enum VernonValuePathComponentKind {
    VERNON_VALUE_PATH_FIELD = 0,
    VERNON_VALUE_PATH_INDEX = 1
} VernonValuePathComponentKind;

typedef struct VernonValuePathComponentView {
    VernonValuePathComponentKind kind;
    VernonStringView field;
    uint64_t index;
} VernonValuePathComponentView;

typedef struct VernonPipelineValueLeafView {
    uint32_t struct_size;
    VernonValueLeafView value;
    const VernonValuePathComponentView *path;
    size_t path_count;
    const uint64_t *static_shape;
    uint32_t static_rank;
} VernonPipelineValueLeafView;

typedef struct VernonValueLayoutView {
    uint32_t struct_size;
    uint32_t byte_size;
    uint32_t alignment;
    VernonStringView layout_hash;
    const VernonValueLeafView *leaves;
    size_t leaf_count;
} VernonValueLayoutView;

VERNON_RUNTIME_CAPI VernonValueLayoutView vernonRuntimeGetScalarValueLayout(VernonDataType dtype);

typedef struct VernonTensorView {
    uint32_t struct_size;
    VernonTensorStorage storage;
    union {
        const void *host_data;
        VernonRuntimeProviderResourceReference resource;
    };
    VernonValueLayoutView element_layout;
    VernonValueAccess access;
    uint32_t rank;
    const uint64_t *shape;
    const int64_t *byte_strides;
    size_t byte_offset;
    size_t byte_size;
} VernonTensorView;

typedef struct VernonTextureView {
    VernonTextureFormat format;
    VernonValueAccess access;
    VernonTextureDimension dimension;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    /*
     * Optional sampling policy for compiler-generated implicit sampler
     * parameters. Explicit sampler parameters take precedence and ignore this
     * field. The sampler must belong to the pipeline runtime context.
     */
    VernonRuntimeProviderResourceReference resource;
    VernonRuntimeProviderResourceReference sampler_resource;
} VernonTextureView;

typedef enum VernonPipelineArgumentKind {
    VERNON_PIPELINE_TENSOR = 0,
    VERNON_PIPELINE_TEXTURE = 1,
    VERNON_PIPELINE_SAMPLER = 2
} VernonPipelineArgumentKind;

typedef struct VernonPipelineArgument {
    uint32_t slot;
    VernonPipelineArgumentKind kind;
    union {
        VernonTensorView tensor;
        VernonTextureView texture;
        VernonRuntimeProviderResourceReference resource;
    };
} VernonPipelineArgument;

typedef struct VernonPipelineInvocation {
    uint32_t struct_size;
    uint32_t abi_version;
    const VernonPipelineArgument *arguments;
    size_t argument_count;
    const VernonIndexBinding *index_binding;
    const VernonColorAttachment *color_attachments;
    size_t color_attachment_count;
    const VernonDepthAttachment *depth_attachment;
    VernonPrimitiveTopology topology;
    uint32_t vertex_count;
    uint32_t instance_count;
    VernonLaunchSize compute_grid;
    uint32_t viewport[4];
    uint32_t scissor[4];
    VernonRuntimeProviderObject command_encoder;
    const VernonGraphicsState *graphics_state;
    uint32_t stencil_reference;
} VernonPipelineInvocation;

typedef struct VernonPipelineParameterView {
    uint32_t slot;
    VernonStringView name;
    VernonPipelineArgumentKind kind;
    VernonValueLayoutView element_layout;
    VernonValueAccess access;
    uint32_t rank;
    const uint64_t *static_shape;
} VernonPipelineParameterView;

typedef struct VernonPipelineTextureConstraintView {
    uint32_t struct_size;
    VernonTextureDimension dimension;
    uint32_t has_format_constraint;
    VernonTextureFormat format;
    /* Reserved for future constraints; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonPipelineTextureConstraintView;

typedef struct VernonPipelineOutputView {
    VernonStringView name;
    VernonPipelineArgumentKind kind;
    VernonDataType dtype;
    VernonValueAccess access;
    uint32_t rank;
    const uint64_t *static_shape;
    uint32_t location;
} VernonPipelineOutputView;

typedef struct VernonAdValue {
    uint32_t struct_size;
    VernonStringView path;
    VernonDataType dtype;
    void *data;
    size_t size;
    /* Logical Value shape. Scalars use rank 0 and shape NULL. */
    uint32_t rank;
    /* Must remain valid for the duration of the API call that consumes this Value. */
    const uint64_t *shape;
} VernonAdValue;

typedef struct VernonAdValueSet {
    uint32_t struct_size;
    VernonAdValue *values;
    size_t value_count;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonAdValueSet;

typedef struct VernonAdValueMetadataView {
    uint32_t struct_size;
    VernonStringView path;
    VernonDataType dtype;
    uint32_t rank;
    const uint64_t *shape;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonAdValueMetadataView;

typedef enum VernonAdDerivativeRole {
    VERNON_AD_DERIVATIVE_GRADIENT = 0,
    VERNON_AD_DERIVATIVE_COTANGENT = 1
} VernonAdDerivativeRole;

typedef struct VernonAdDerivativeGroupView {
    uint32_t struct_size;
    VernonAdDerivativeRole role;
    VernonStringView declared_path;
    size_t leaf_count;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonAdDerivativeGroupView;

typedef struct VernonPipelineBundleLoadOptions {
    uint32_t struct_size;
    /*
     * UTF-8 directory containing the manifest. Required by external artifact
     * descriptors and CPU native-library sidecars.
     */
    const char *bundle_directory;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonPipelineBundleLoadOptions;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimePipelineBundleInspectTarget(const void *bundle, size_t bundle_size,
                                                                          VernonRuntimeBackend *target);
VERNON_RUNTIME_CAPI VernonPipelineBundle *
vernonRuntimeLoadPipelineBundleWithOptions(VernonRuntimeContext *context, const void *bundle, size_t bundle_size,
                                           const VernonPipelineBundleLoadOptions *options);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimePipelineBundleGetId(const VernonPipelineBundle *bundle);
VERNON_RUNTIME_CAPI void vernonRuntimePipelineBundleDestroy(VernonPipelineBundle *bundle);
VERNON_RUNTIME_CAPI VernonLoadedPipeline *vernonRuntimeResolvePipeline(VernonPipelineBundle *bundle,
                                                                       VernonFeatureSetView features);
VERNON_RUNTIME_CAPI void vernonRuntimeLoadedPipelineDestroy(VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetParameterCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetParameterByIndex(const VernonLoadedPipeline *pipeline,
                                                                                size_t index,
                                                                                VernonPipelineParameterView *parameter);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineFindParameter(const VernonLoadedPipeline *pipeline,
                                                                          VernonStringView name,
                                                                          VernonPipelineParameterView *parameter);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetParameterValueLeaf(const VernonLoadedPipeline *pipeline,
                                                                                  VernonStringView parameter_name,
                                                                                  size_t leaf_index,
                                                                                  VernonPipelineValueLeafView *leaf);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(
    const VernonLoadedPipeline *pipeline, size_t parameter_index, VernonPipelineTextureConstraintView *constraint);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeLoadedPipelineFindTextureConstraint(const VernonLoadedPipeline *pipeline, VernonStringView parameter_name,
                                                 VernonPipelineTextureConstraintView *constraint);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetOutputCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetOutputByIndex(const VernonLoadedPipeline *pipeline,
                                                                             size_t index,
                                                                             VernonPipelineOutputView *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineFindOutput(const VernonLoadedPipeline *pipeline,
                                                                       VernonStringView name,
                                                                       VernonPipelineOutputView *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimePipelineInvoke(VernonLoadedPipeline *pipeline,
                                                             const VernonPipelineInvocation *invocation);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimePipelineEncode(VernonRuntimeProviderObject encoder,
                                                             VernonLoadedPipeline *pipeline,
                                                             const VernonPipelineInvocation *invocation);
VERNON_RUNTIME_CAPI VernonStatus vernonAdPipelineForward(VernonLoadedPipeline *pipeline, VernonLaunchSize compute_grid,
                                                         const VernonAdValueSet *inputs, VernonAdValueSet *outputs,
                                                         VernonPullback **pullback);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetAdOutputCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetAdOutputByIndex(const VernonLoadedPipeline *pipeline,
                                                                               size_t index,
                                                                               VernonAdValueMetadataView *metadata);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetAdCotangentCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetAdCotangentByIndex(const VernonLoadedPipeline *pipeline,
                                                                                  size_t index,
                                                                                  VernonAdValueMetadataView *metadata);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetAdGradientCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetAdGradientByIndex(const VernonLoadedPipeline *pipeline,
                                                                                 size_t index,
                                                                                 VernonAdValueMetadataView *metadata);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetAdDerivativeGroupCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetAdDerivativeGroupByIndex(
    const VernonLoadedPipeline *pipeline, size_t group_index, VernonAdDerivativeGroupView *group);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetAdDerivativeGroupLeaf(
    const VernonLoadedPipeline *pipeline, size_t group_index, size_t leaf_index, VernonStringView *leaf_path);
VERNON_RUNTIME_CAPI VernonStatus vernonPullbackApply(VernonPullback *pullback, const VernonAdValueSet *cotangents,
                                                     VernonAdValueSet *gradients);
VERNON_RUNTIME_CAPI void vernonPullbackDestroy(VernonPullback *pullback);

#ifdef __cplusplus
}
#endif

#endif
