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
typedef struct VernonLoadedKernel VernonLoadedKernel;

typedef enum VernonRuntimeBackend {
  VERNON_RUNTIME_CPU = 0,
  VERNON_RUNTIME_CUDA = 1
} VernonRuntimeBackend;

typedef struct VernonRuntimeCapabilities {
  uint8_t available;
  uint8_t supports_compute;
  uint8_t reserved[2];
  VernonStringView diagnostic;
} VernonRuntimeCapabilities;

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
VERNON_RUNTIME_CAPI VernonRuntimeContext *
vernonRuntimeCreate(VernonRuntimeBackend backend, uint32_t device_index);
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
VERNON_RUNTIME_CAPI VernonStatus
vernonRuntimeSynchronize(VernonRuntimeContext *context);

#ifdef __cplusplus
}
#endif

#endif
