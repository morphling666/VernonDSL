#ifndef VERNON_C_COMPILER_H
#define VERNON_C_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(VERNON_DSL_COMPILER_BUILD)
#define VERNON_DSL_CAPI __declspec(dllexport)
#elif defined(_WIN32)
#define VERNON_DSL_CAPI __declspec(dllimport)
#else
#define VERNON_DSL_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VernonCompilerContext VernonCompilerContext;
typedef struct VernonCompileResult VernonCompileResult;

typedef struct VernonStringView {
  const char *data;
  size_t size;
} VernonStringView;

typedef enum VernonStatus {
  VERNON_STATUS_OK = 0,
  VERNON_STATUS_INVALID_ARGUMENT = 1,
  VERNON_STATUS_PARSE_ERROR = 2,
  VERNON_STATUS_VERIFICATION_ERROR = 3,
  VERNON_STATUS_UNSUPPORTED_TARGET = 4,
  VERNON_STATUS_INTERNAL_ERROR = 5
} VernonStatus;

typedef enum VernonTarget {
  VERNON_TARGET_CPU = 0,
  VERNON_TARGET_OPENGL = 1,
  VERNON_TARGET_OPENGL_ES = 2,
  VERNON_TARGET_VULKAN = 3,
  VERNON_TARGET_METAL = 4,
  VERNON_TARGET_DIRECTX = 5,
  VERNON_TARGET_CUDA = 6
} VernonTarget;

typedef struct VernonTargetCapabilities {
  uint8_t available;
  uint8_t supports_graphics;
  uint8_t supports_compute;
  uint8_t reserved;
} VernonTargetCapabilities;

typedef struct VernonCpuTextureCallbacks {
  void *user_data;
  void (*sample_2d)(void *user_data, uintptr_t texture, float u, float v,
                    float out_rgba[4]);
  void (*size_2d)(void *user_data, uintptr_t texture, int32_t level,
                  int32_t out_size[2]);
} VernonCpuTextureCallbacks;

typedef struct VernonCpuInvocation {
  // Argument and result layouts are described by the compile reflection.
  const void *arguments;
  size_t arguments_size;
  void *results;
  size_t results_size;
  const VernonCpuTextureCallbacks *textures;
} VernonCpuInvocation;

typedef VernonStatus (*VernonCpuEntryPoint)(
    const VernonCpuInvocation *invocation);

VERNON_DSL_CAPI VernonCompilerContext *vernonCompilerCreate(void);
VERNON_DSL_CAPI void vernonCompilerDestroy(VernonCompilerContext *context);

VERNON_DSL_CAPI VernonTargetCapabilities vernonCompilerGetTargetCapabilities(
    const VernonCompilerContext *context, VernonTarget target);

// Parses and verifies textual MLIR and returns deterministic reflection data.
VERNON_DSL_CAPI VernonCompileResult *
vernonCompilerValidateMlir(VernonCompilerContext *context, const char *source,
                           size_t source_size);

// Unimplemented targets return UNSUPPORTED_TARGET, never placeholder code.
VERNON_DSL_CAPI VernonCompileResult *
vernonCompilerCompileMlir(VernonCompilerContext *context, const char *source,
                          size_t source_size, VernonTarget target);

VERNON_DSL_CAPI void vernonCompileResultDestroy(VernonCompileResult *result);
VERNON_DSL_CAPI VernonStatus
vernonCompileResultGetStatus(const VernonCompileResult *result);
VERNON_DSL_CAPI VernonStringView
vernonCompileResultGetDiagnostics(const VernonCompileResult *result);
VERNON_DSL_CAPI VernonStringView
vernonCompileResultGetArtifact(const VernonCompileResult *result);
VERNON_DSL_CAPI size_t
vernonCompileResultGetArtifactCount(const VernonCompileResult *result);
VERNON_DSL_CAPI VernonStringView vernonCompileResultGetArtifactName(
    const VernonCompileResult *result, size_t index);
VERNON_DSL_CAPI VernonStringView vernonCompileResultGetArtifactData(
    const VernonCompileResult *result, size_t index);
VERNON_DSL_CAPI VernonStringView
vernonCompileResultGetReflection(const VernonCompileResult *result);
VERNON_DSL_CAPI VernonCpuEntryPoint
vernonCompileResultGetCpuEntry(const VernonCompileResult *result,
                               const char *entry_name, size_t entry_name_size);

#ifdef __cplusplus
}
#endif

#endif // VERNON_C_COMPILER_H
