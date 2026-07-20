#include "VernonRuntime.h"

#include <cassert>
#include <cstring>

namespace {

#if defined(_WIN32)
#define GL_CALL __stdcall
#else
#define GL_CALL
#endif

using GlEnum = unsigned int;
using GlInt = int;
using GlSize = int;
using GlUint = unsigned int;

constexpr GlEnum kCompileStatus = 0x8B81;
constexpr GlEnum kLinkStatus = 0x8B82;
constexpr GlEnum kFramebufferComplete = 0x8CD5;

GlUint nextName = 1;
uint32_t drawCount = 0;

void makeCurrent(void *) {}
void GL_CALL placeholder() {}
GlUint GL_CALL createName(GlEnum) { return nextName++; }
GlUint GL_CALL createProgram() { return nextName++; }
void GL_CALL shaderSource(GlUint, GlSize, const char *const *, const GlInt *) {}
void GL_CALL getShaderiv(GlUint, GlEnum name, GlInt *value) {
  *value = name == kCompileStatus ? 1 : 0;
}
void GL_CALL getProgramiv(GlUint, GlEnum name, GlInt *value) {
  *value = name == kLinkStatus ? 1 : 0;
}
void GL_CALL genNames(GlSize count, GlUint *names) {
  while (count--)
    *names++ = nextName++;
}
GlEnum GL_CALL framebufferStatus(GlEnum) { return kFramebufferComplete; }
void GL_CALL drawArrays(GlEnum, GlInt, GlSize) { ++drawCount; }

void *getProcAddress(void *, const char *name) {
#define PROC(glName, function)                                                 \
  if (std::strcmp(name, glName) == 0)                                         \
  return reinterpret_cast<void *>(&function)
  PROC("glCreateShader", createName);
  PROC("glShaderSource", shaderSource);
  PROC("glGetShaderiv", getShaderiv);
  PROC("glCreateProgram", createProgram);
  PROC("glGetProgramiv", getProgramiv);
  PROC("glGenVertexArrays", genNames);
  PROC("glGenFramebuffers", genNames);
  PROC("glCheckFramebufferStatus", framebufferStatus);
  PROC("glDrawArrays", drawArrays);
#undef PROC
  return reinterpret_cast<void *>(&placeholder);
}

VernonRuntimeContext *create(VernonRuntimeBackend backend, uint16_t major,
                             uint16_t minor) {
  VernonExternalOpenGLContext external{};
  external.struct_size = sizeof(external);
  external.make_current = &makeCurrent;
  external.get_proc_address = &getProcAddress;
  external.api_version_major = major;
  external.api_version_minor = minor;
  return vernonRuntimeCreateExternalOpenGLForBackend(backend, &external);
}

} // namespace

int main() {
  VernonRuntimeCapabilities global =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_OPENGL_ES);
  assert(!global.available);
  assert(global.supports_graphics);

  VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
  assert(gl);
  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetContextCapabilities(gl);
  assert(capabilities.available && capabilities.supports_graphics);
  assert(capabilities.supports_compute && capabilities.supports_storage_buffers);
  constexpr char glBundle[] =
      R"({"pipeline_bundle_schema_version":1,"invocation_abi_version":1,"type":"vernon_pipeline_bundle","id":"external-gl","target":"opengl","features":[],"variants":[{"key":[],"parameters":[],"steps":[{"kind":"draw","vertex":"vs","fragment":"fs"}]}],"stage_artifacts":{"vs":{"stage":"vertex","entry":"main","source":"#version 330\nvoid main(){gl_Position=vec4(0.0);}"},"fs":{"stage":"fragment","entry":"main","source":"#version 330\nout vec4 color;void main(){color=vec4(1.0);}"}}})";
  VernonPipelineBundle *glLoaded =
      vernonRuntimeLoadPipelineBundle(gl, glBundle, std::strlen(glBundle));
  assert(glLoaded);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(glLoaded, {nullptr, 0});
  assert(pipeline);
  VernonDeviceTexture *target =
      vernonRuntimeImportOpenGLTexture2D(gl, 7, 16, 16,
                                         VERNON_TEXTURE_RGBA8_UNORM);
  assert(target);
  VernonColorAttachment attachment{0, target};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.vertex_count = 3;
  invocation.instance_count = 1;
  assert(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
  assert(drawCount == 1);
  assert(vernonRuntimeTextureFree(target) == VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(glLoaded);
  assert(vernonRuntimeDestroy(gl) == VERNON_STATUS_OK);

  VernonRuntimeContext *gles = create(VERNON_RUNTIME_OPENGL_ES, 3, 1);
  assert(gles);
  capabilities = vernonRuntimeGetContextCapabilities(gles);
  assert(capabilities.supports_compute);
  constexpr char bundle[] =
      R"({"pipeline_bundle_schema_version":1,"invocation_abi_version":1,"type":"vernon_pipeline_bundle","id":"gles-profile","target":"opengles","features":[],"variants":[{"key":[],"parameters":[],"steps":[{"kind":"draw","vertex":"vs","fragment":"fs"}]}],"stage_artifacts":{"vs":{"stage":"vertex","entry":"main","source":"#version 310 es\nvoid main(){}"},"fs":{"stage":"fragment","entry":"main","source":"#version 310 es\nvoid main(){}"}}})";
  VernonPipelineBundle *loaded =
      vernonRuntimeLoadPipelineBundle(gles, bundle, std::strlen(bundle));
  assert(loaded);
  vernonRuntimePipelineBundleDestroy(loaded);
  assert(vernonRuntimeDestroy(gles) == VERNON_STATUS_OK);
  return 0;
}

#undef GL_CALL
