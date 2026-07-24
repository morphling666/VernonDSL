#ifndef VERNON_C_RUNTIME_H
#define VERNON_C_RUNTIME_H

#include "VernonCommon.h"

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
typedef struct VernonDeviceBuffer VernonDeviceBuffer;
typedef struct VernonDeviceTexture VernonDeviceTexture;
typedef struct VernonDeviceSampler VernonDeviceSampler;
typedef struct VernonLoadedKernel VernonLoadedKernel;
typedef struct VernonPipelineBundle VernonPipelineBundle;
typedef struct VernonLoadedPipeline VernonLoadedPipeline;

typedef enum VernonRuntimeBackend {
    VERNON_RUNTIME_CPU = 0,
    VERNON_RUNTIME_CUDA = 1,
    VERNON_RUNTIME_VULKAN = 2,
    VERNON_RUNTIME_OPENGL = 3,
    VERNON_RUNTIME_OPENGL_ES = 4
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
    /* Reserved for ABI compatibility; initialize both fields to zero. */
    uint16_t api_version_major;
    uint16_t api_version_minor;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonRuntimeCreateOptions;

typedef void (*VernonOpenGLMakeCurrentFn)(void *user_data);
typedef void *(*VernonOpenGLGetProcAddressFn)(void *user_data, const char *name);

typedef struct VernonExternalOpenGLContext {
    uint32_t struct_size;
    void *user_data;
    VernonOpenGLMakeCurrentFn make_current;
    VernonOpenGLGetProcAddressFn get_proc_address;
    uint16_t api_version_major;
    uint16_t api_version_minor;
    /* Reserved for future use; initialize all elements to zero. */
    uint32_t reserved[4];
} VernonExternalOpenGLContext;

typedef struct VernonLaunchSize {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} VernonLaunchSize;

typedef enum VernonLaunchArgumentKind { VERNON_LAUNCH_TENSOR = 0, VERNON_LAUNCH_SCALAR = 1 } VernonLaunchArgumentKind;

typedef struct VernonLaunchArgument {
    VernonLaunchArgumentKind kind;
    VernonDeviceBuffer *buffer;
    const void *scalar_data;
    size_t scalar_size;
} VernonLaunchArgument;

VERNON_RUNTIME_CAPI VernonRuntimeCapabilities vernonRuntimeGetCapabilities(VernonRuntimeBackend backend);
VERNON_RUNTIME_CAPI VernonRuntimeCapabilities vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context);
/* Compatibility convenience API; prefer vernonRuntimeCreateWithOptions. */
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreate(VernonRuntimeBackend backend, uint32_t device_index);
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                                                                         const VernonRuntimeCreateOptions *options);
/* Compatibility shorthand for the VERNON_RUNTIME_OPENGL backend. */
VERNON_RUNTIME_CAPI VernonRuntimeContext *
vernonRuntimeCreateExternalOpenGL(const VernonExternalOpenGLContext *external_context);
VERNON_RUNTIME_CAPI VernonRuntimeContext *
vernonRuntimeCreateExternalOpenGLForBackend(VernonRuntimeBackend backend,
                                            const VernonExternalOpenGLContext *external_context);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStringView vernonRuntimeGetLastError(const VernonRuntimeContext *context);

VERNON_RUNTIME_CAPI VernonDeviceBuffer *vernonRuntimeBufferAllocate(VernonRuntimeContext *context, size_t size,
                                                                    size_t alignment);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeBufferFree(VernonDeviceBuffer *buffer);
VERNON_RUNTIME_CAPI VernonDeviceBuffer *vernonRuntimeImportOpenGLBuffer(VernonRuntimeContext *context, uint32_t buffer,
                                                                        size_t size, size_t alignment);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeCopyFromHost(VernonDeviceBuffer *buffer, size_t offset,
                                                           const void *source, size_t size);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeCopyToHost(const VernonDeviceBuffer *buffer, size_t offset,
                                                         void *destination, size_t size);

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
    VERNON_TEXTURE_R11G11B10_FLOAT = 9
} VernonTextureFormat;

typedef enum VernonTextureDimension {
    VERNON_TEXTURE_2D = 0,
    VERNON_TEXTURE_3D = 1,
    VERNON_TEXTURE_CUBE = 2
} VernonTextureDimension;

typedef struct VernonTextureDescriptor {
    uint32_t struct_size;
    VernonTextureDimension dimension;
    VernonTextureFormat format;
    uint32_t width;
    uint32_t height;
    /* 3D depth; must be one for 2D and Cube sampled textures. */
    uint32_t depth;
    uint32_t mip_levels;
    uint32_t reserved[4];
} VernonTextureDescriptor;

typedef enum VernonSamplerWrapMode {
    VERNON_SAMPLER_REPEAT = 0,
    VERNON_SAMPLER_MIRRORED_REPEAT = 1,
    VERNON_SAMPLER_CLAMP_TO_EDGE = 2,
    VERNON_SAMPLER_CLAMP_TO_BORDER = 3
} VernonSamplerWrapMode;

typedef enum VernonSamplerFilter { VERNON_SAMPLER_NEAREST = 0, VERNON_SAMPLER_LINEAR = 1 } VernonSamplerFilter;

typedef struct VernonSamplerDescriptor {
    uint32_t struct_size;
    VernonSamplerWrapMode wrap_u;
    VernonSamplerWrapMode wrap_v;
    VernonSamplerWrapMode wrap_w;
    VernonSamplerFilter min_filter;
    VernonSamplerFilter mag_filter;
    VernonSamplerFilter mip_filter;
    uint32_t reserved[4];
} VernonSamplerDescriptor;

VERNON_RUNTIME_CAPI VernonDeviceTexture *vernonRuntimeTextureCreate(VernonRuntimeContext *context,
                                                                    const VernonTextureDescriptor *descriptor);
/* Compatibility convenience API for a one-mip 2D sampled texture. */
VERNON_RUNTIME_CAPI VernonDeviceTexture *vernonRuntimeTextureCreate2D(VernonRuntimeContext *context, uint32_t width,
                                                                      uint32_t height, VernonTextureFormat format);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeTextureFree(VernonDeviceTexture *texture);
VERNON_RUNTIME_CAPI VernonDeviceTexture *vernonRuntimeImportOpenGLTexture(VernonRuntimeContext *context,
                                                                          uint32_t texture,
                                                                          const VernonTextureDescriptor *descriptor);
/* Compatibility convenience API for importing a 2D OpenGL texture. */
VERNON_RUNTIME_CAPI VernonDeviceTexture *vernonRuntimeImportOpenGLTexture2D(VernonRuntimeContext *context,
                                                                            uint32_t texture, uint32_t width,
                                                                            uint32_t height,
                                                                            VernonTextureFormat format);
/*
 * Cube uploads contain six tightly packed faces in +X, -X, +Y, -Y, +Z, -Z
 * order. The current host upload path supports one-mip RGBA8 textures.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeTextureCopyFromHost(VernonDeviceTexture *texture, const void *source,
                                                                  size_t size);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeTextureCopyToHost(const VernonDeviceTexture *texture, void *destination,
                                                                size_t size);
VERNON_RUNTIME_CAPI VernonDeviceSampler *vernonRuntimeSamplerCreate(VernonRuntimeContext *context,
                                                                    const VernonSamplerDescriptor *descriptor);
VERNON_RUNTIME_CAPI VernonDeviceSampler *vernonRuntimeImportOpenGLSampler(VernonRuntimeContext *context,
                                                                          uint32_t sampler);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeSamplerFree(VernonDeviceSampler *sampler);

VERNON_RUNTIME_CAPI VernonLoadedKernel *vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                                                                  size_t artifact_size, const char *reflection,
                                                                  size_t reflection_size, const char *entry,
                                                                  size_t entry_size);
VERNON_RUNTIME_CAPI VernonLoadedKernel *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context,
                                                                  VernonCpuEntryPoint entry_point,
                                                                  const char *reflection, size_t reflection_size,
                                                                  const char *entry, size_t entry_size);
/*
 * Registers an AOT entry that was statically linked into the application.
 * Re-registering the same symbol and pointer is idempotent.
 */
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeRegisterStaticCpuEntry(VernonStringView symbol,
                                                                     VernonCpuEntryPoint entry_point);
VERNON_RUNTIME_CAPI VernonLoadedKernel *vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context,
                                                                       const char *directory);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeKernelUnload(VernonLoadedKernel *kernel);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLaunch(VernonLoadedKernel *kernel, VernonLaunchSize global_size,
                                                     const VernonLaunchArgument *arguments, size_t argument_count);

typedef enum VernonIndexType { VERNON_INDEX_U32 = 0 } VernonIndexType;

typedef enum VernonPrimitiveTopology {
    VERNON_TOPOLOGY_TRIANGLE_LIST = 0,
    VERNON_TOPOLOGY_LINE_LIST = 1,
    VERNON_TOPOLOGY_POINT_LIST = 2
} VernonPrimitiveTopology;

typedef struct VernonIndexBinding {
    VernonDeviceBuffer *buffer;
    VernonIndexType type;
    size_t offset;
    uint32_t index_count;
} VernonIndexBinding;

typedef struct VernonColorAttachment {
    uint32_t location;
    VernonDeviceTexture *texture;
} VernonColorAttachment;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeComputeToGraphicsBarrier(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context);

enum { VERNON_PIPELINE_INVOCATION_ABI_VERSION = 3 };

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
    VERNON_DATA_F64 = 5
} VernonDataType;

typedef enum VernonValueAccess {
    VERNON_ACCESS_READ = 0,
    VERNON_ACCESS_WRITE = 1,
    VERNON_ACCESS_READ_WRITE = 2
} VernonValueAccess;

typedef enum VernonTensorStorage { VERNON_TENSOR_HOST = 0, VERNON_TENSOR_DEVICE = 1 } VernonTensorStorage;

typedef struct VernonTensorView {
    uint32_t struct_size;
    VernonTensorStorage storage;
    union {
        const void *host_data;
        VernonDeviceBuffer *buffer;
    };
    VernonDataType dtype;
    VernonValueAccess access;
    uint32_t rank;
    const uint64_t *shape;
    const int64_t *byte_strides;
    size_t byte_offset;
    size_t byte_size;
} VernonTensorView;

typedef struct VernonTextureView {
    VernonDeviceTexture *texture;
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
    VernonDeviceSampler *sampler;
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
        VernonDeviceSampler *sampler;
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
    VernonPrimitiveTopology topology;
    uint32_t vertex_count;
    uint32_t instance_count;
    VernonLaunchSize compute_grid;
    uint32_t viewport[4];
    uint32_t scissor[4];
} VernonPipelineInvocation;

typedef struct VernonPipelineParameterView {
    uint32_t slot;
    VernonStringView name;
    VernonPipelineArgumentKind kind;
    VernonDataType dtype;
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

typedef enum VernonPipelineStepKind {
    VERNON_PIPELINE_DISPATCH = 0,
    VERNON_PIPELINE_BARRIER = 1,
    VERNON_PIPELINE_DRAW = 2
} VernonPipelineStepKind;

typedef struct VernonPipelineStepView {
    uint32_t struct_size;
    VernonPipelineStepKind kind;
    VernonStringView stage;
    VernonStringView vertex;
    VernonStringView fragment;
    VernonStringView source;
    VernonStringView destination;
    uint32_t has_grid;
    VernonLaunchSize grid;
} VernonPipelineStepView;

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
/* Compatibility convenience API; prefer the options-based loader. */
VERNON_RUNTIME_CAPI VernonPipelineBundle *vernonRuntimeLoadPipelineBundle(VernonRuntimeContext *context,
                                                                          const void *bundle, size_t bundle_size);
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
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetStepCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetStepByIndex(const VernonLoadedPipeline *pipeline,
                                                                           size_t index, VernonPipelineStepView *step);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimePipelineInvoke(VernonLoadedPipeline *pipeline,
                                                             const VernonPipelineInvocation *invocation);

#ifdef __cplusplus
}
#endif

#endif
