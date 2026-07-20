#ifndef VERNON_C_RUNTIME_H
#define VERNON_C_RUNTIME_H

#include "VernonCompiler.h"

#if defined(_WIN32) && defined(VERNON_DSL_RUNTIME_BUILD)
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
typedef struct VernonLoadedProgram VernonLoadedProgram;

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
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeDestroy(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStringView
vernonRuntimeGetLastError(const VernonRuntimeContext *context);

VERNON_RUNTIME_CAPI VernonDeviceBuffer *
vernonRuntimeBufferAllocate(VernonRuntimeContext *context, size_t size,
                            size_t alignment);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeBufferFree(VernonDeviceBuffer *buffer);
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

typedef struct VernonGraphicsStageArtifact {
  const void *data;
  size_t size;
} VernonGraphicsStageArtifact;

typedef struct VernonDrawBinding {
  uint32_t location;
  VernonDeviceBuffer *buffer;
  uint32_t component_count;
  uint32_t stride;
  size_t offset;
  uint32_t instance_divisor;
} VernonDrawBinding;

typedef struct VernonUniformBinding {
  const char *name;
  const float *values;
  uint32_t value_count;
} VernonUniformBinding;

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

typedef struct VernonDrawDescription {
  uint32_t struct_size;
  VernonDeviceTexture *target;
  const VernonDrawBinding *bindings;
  size_t binding_count;
  const VernonUniformBinding *uniforms;
  size_t uniform_count;
  uint32_t vertex_count;
  uint32_t instance_count;
  float clear_color[4];
  const VernonIndexBinding *index_binding;
  const VernonColorAttachment *color_attachments;
  size_t color_attachment_count;
  VernonPrimitiveTopology topology;
} VernonDrawDescription;

VERNON_RUNTIME_CAPI VernonLoadedProgram *
vernonRuntimeProgramLoadGraphics(VernonRuntimeContext *context,
                                 VernonGraphicsStageArtifact vertex,
                                 VernonGraphicsStageArtifact fragment);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeProgramUnload(VernonLoadedProgram *program);
VERNON_RUNTIME_CAPI VernonStatus vernonRuntimeDraw(
    VernonLoadedProgram *program, const VernonDrawDescription *description);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeComputeToGraphicsBarrier(VernonRuntimeContext *context);
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeSynchronize(VernonRuntimeContext *context);

#ifdef __cplusplus
}
#endif

#endif
