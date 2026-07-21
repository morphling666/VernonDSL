#ifndef VERNON_C_COMPILER_H
#define VERNON_C_COMPILER_H

#include "VernonCommon.h"

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

typedef struct VernonCompileOptions {
  // Set to sizeof(VernonCompileOptions). This permits ABI-compatible extension.
  uint32_t struct_size;
  // Zero selects the target default. Used only by OpenGL and OpenGL ES.
  uint32_t glsl_version;
  uint32_t reserved[6];
  /*
   * CPU object target. Empty selects the compiler host triple. Examples:
   * "x86_64-pc-windows-msvc" and "arm64-apple-ios17.0".
   */
  VernonStringView cpu_target_triple;
  /* Empty selects the target's generic CPU. */
  VernonStringView cpu_name;
  /* Comma-separated LLVM target features, for example "+neon". */
  VernonStringView cpu_features;
} VernonCompileOptions;

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
VERNON_DSL_CAPI VernonCompileResult *vernonCompilerCompileMlirWithOptions(
    VernonCompilerContext *context, const char *source, size_t source_size,
    VernonTarget target, const VernonCompileOptions *options);
/*
 * Finalizes a host relocatable object into a temporary-loadable native
 * library. The result contains exactly one DLL/so/dylib artifact.
 */
VERNON_DSL_CAPI VernonCompileResult *
vernonCompilerLinkHostObject(VernonCompilerContext *context, const void *object,
                             size_t object_size);

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
