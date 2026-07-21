#ifndef VERNON_C_RUNTIME_H
#define VERNON_C_RUNTIME_H

#include "VernonCommon.h"

#if defined(_WIN32) && defined(VERNON_RUNTIME_BUILD)
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
  uint16_t api_version_major;
  uint16_t api_version_minor;
  uint32_t reserved[4];
} VernonRuntimeCreateOptions;

typedef void (*VernonOpenGLMakeCurrentFn)(void *user_data);
typedef void *(*VernonOpenGLGetProcAddressFn)(void *user_data,
                                              const char *name);

typedef struct VernonExternalOpenGLContext {
  uint32_t struct_size;
  void *user_data;
  VernonOpenGLMakeCurrentFn make_current;
  VernonOpenGLGetProcAddressFn get_proc_address;
  uint16_t api_version_major;
  uint16_t api_version_minor;
  uint32_t reserved[4];
} VernonExternalOpenGLContext;

typedef struct VernonLaunchSize {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} VernonLaunchSize;

typedef enum VernonLaunchArgumentKind {
  VERNON_LAUNCH_TENSOR = 0,
  VERNON_LAUNCH_SCALAR = 1
} VernonLaunchArgumentKind;

typedef struct VernonLaunchArgument {
  VernonLaunchArgumentKind kind;
  VernonDeviceBuffer *buffer;
  const void *scalar_data;
  size_t scalar_size;
} VernonLaunchArgument;

VERNON_RUNTIME_CAPI VernonRuntimeCapabilities
vernonRuntimeGetCapabilities(VernonRuntimeBackend backend);
VERNON_RUNTIME_CAPI VernonRuntimeCapabilities
vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonRuntimeContext *
vernonRuntimeCreate(VernonRuntimeBackend backend, uint32_t device_index);
VERNON_RUNTIME_CAPI VernonRuntimeContext *
vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                               const VernonRuntimeCreateOptions *options);
VERNON_RUNTIME_CAPI VernonRuntimeContext *vernonRuntimeCreateExternalOpenGL(
    const VernonExternalOpenGLContext *external_context);
VERNON_RUNTIME_CAPI VernonRuntimeContext *
vernonRuntimeCreateExternalOpenGLForBackend(
    VernonRuntimeBackend backend,
    const VernonExternalOpenGLContext *external_context);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeDestroy(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStringView
vernonRuntimeGetLastError(const VernonRuntimeContext *context);

VERNON_RUNTIME_CAPI VernonDeviceBuffer *
vernonRuntimeBufferAllocate(VernonRuntimeContext *context, size_t size,
                            size_t alignment);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeBufferFree(VernonDeviceBuffer *buffer);
VERNON_RUNTIME_CAPI VernonDeviceBuffer *
vernonRuntimeImportOpenGLBuffer(VernonRuntimeContext *context, uint32_t buffer,
                                size_t size, size_t alignment);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeCopyFromHost(
    VernonDeviceBuffer *buffer, size_t offset, const void *source, size_t size);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeCopyToHost(const VernonDeviceBuffer *buffer, size_t offset,
                        void *destination, size_t size);

typedef enum VernonTextureFormat {
  VERNON_TEXTURE_RGBA8_UNORM = 0
} VernonTextureFormat;

VERNON_RUNTIME_CAPI VernonDeviceTexture *
vernonRuntimeTextureCreate2D(VernonRuntimeContext *context, uint32_t width,
                             uint32_t height, VernonTextureFormat format);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeTextureFree(VernonDeviceTexture *texture);
VERNON_RUNTIME_CAPI VernonDeviceTexture *
vernonRuntimeImportOpenGLTexture2D(VernonRuntimeContext *context,
                                   uint32_t texture, uint32_t width,
                                   uint32_t height, VernonTextureFormat format);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeTextureCopyFromHost(
    VernonDeviceTexture *texture, const void *source, size_t size);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeTextureCopyToHost(
    const VernonDeviceTexture *texture, void *destination, size_t size);

VERNON_RUNTIME_CAPI VernonLoadedKernel *
vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                          size_t artifact_size, const char *reflection,
                          size_t reflection_size, const char *entry,
                          size_t entry_size);
VERNON_RUNTIME_CAPI VernonLoadedKernel *
vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context,
                          VernonCpuEntryPoint entry_point,
                          const char *reflection, size_t reflection_size,
                          const char *entry, size_t entry_size);
VERNON_RUNTIME_CAPI VernonLoadedKernel *
vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context,
                               const char *directory);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeKernelUnload(VernonLoadedKernel *kernel);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLaunch(
    VernonLoadedKernel *kernel, VernonLaunchSize global_size,
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

VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeComputeToGraphicsBarrier(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeSynchronize(VernonRuntimeContext *context);

enum { VERNON_PIPELINE_INVOCATION_ABI_VERSION = 1 };

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

typedef struct VernonTensorView {
  VernonDeviceBuffer *buffer;
  VernonDataType dtype;
  VernonValueAccess access;
  uint32_t rank;
  const uint64_t *shape;
  const uint64_t *byte_strides;
  size_t byte_offset;
} VernonTensorView;

typedef struct VernonTextureView {
  VernonDeviceTexture *texture;
  VernonTextureFormat format;
  VernonValueAccess access;
  uint32_t dimension;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
} VernonTextureView;

typedef struct VernonInlineValue {
  VernonDataType dtype;
  uint32_t rank;
  const uint64_t *shape;
  const void *data;
  size_t data_size;
} VernonInlineValue;

typedef enum VernonPipelineArgumentKind {
  VERNON_PIPELINE_TENSOR = 0,
  VERNON_PIPELINE_TEXTURE = 1,
  VERNON_PIPELINE_INLINE_VALUE = 2
} VernonPipelineArgumentKind;

typedef struct VernonPipelineArgument {
  uint32_t slot;
  VernonPipelineArgumentKind kind;
  union {
    VernonTensorView tensor;
    VernonTextureView texture;
    VernonInlineValue inline_value;
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

typedef struct VernonPipelineOutputView {
  VernonStringView name;
  VernonPipelineArgumentKind kind;
  VernonDataType dtype;
  VernonValueAccess access;
  uint32_t rank;
  const uint64_t *static_shape;
  uint32_t location;
} VernonPipelineOutputView;

typedef struct VernonPipelineBundleLoadOptions {
  uint32_t struct_size;
  const char *bundle_directory;
  uint32_t reserved[4];
} VernonPipelineBundleLoadOptions;

VERNON_RUNTIME_CAPI VernonStatus vernonRuntimePipelineBundleInspectTarget(
    const void *bundle, size_t bundle_size, VernonRuntimeBackend *target);
VERNON_RUNTIME_CAPI VernonPipelineBundle *
vernonRuntimeLoadPipelineBundle(VernonRuntimeContext *context,
                                const void *bundle, size_t bundle_size);
VERNON_RUNTIME_CAPI VernonPipelineBundle *
vernonRuntimeLoadPipelineBundleWithOptions(
    VernonRuntimeContext *context, const void *bundle, size_t bundle_size,
    const VernonPipelineBundleLoadOptions *options);
VERNON_RUNTIME_CAPI VernonPipelineBundle *
vernonRuntimeLoadPipelineBundleFromDirectory(VernonRuntimeContext *context,
                                             const char *directory);
VERNON_RUNTIME_CAPI VernonStringView
vernonRuntimePipelineBundleGetId(const VernonPipelineBundle *bundle);
VERNON_RUNTIME_CAPI void
vernonRuntimePipelineBundleDestroy(VernonPipelineBundle *bundle);
VERNON_RUNTIME_CAPI VernonLoadedPipeline *
vernonRuntimeResolvePipeline(VernonPipelineBundle *bundle,
                             VernonFeatureSetView features);
VERNON_RUNTIME_CAPI void
vernonRuntimeLoadedPipelineDestroy(VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI size_t vernonRuntimeLoadedPipelineGetParameterCount(
    const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetParameterByIndex(
    const VernonLoadedPipeline *pipeline, size_t index,
    VernonPipelineParameterView *parameter);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineFindParameter(
    const VernonLoadedPipeline *pipeline, VernonStringView name,
    VernonPipelineParameterView *parameter);
VERNON_RUNTIME_CAPI size_t
vernonRuntimeLoadedPipelineGetOutputCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineGetOutputByIndex(
    const VernonLoadedPipeline *pipeline, size_t index,
    VernonPipelineOutputView *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeLoadedPipelineFindOutput(
    const VernonLoadedPipeline *pipeline, VernonStringView name,
    VernonPipelineOutputView *output);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimePipelineInvoke(
    VernonLoadedPipeline *pipeline, const VernonPipelineInvocation *invocation);

#ifdef __cplusplus
}
#endif

#endif
