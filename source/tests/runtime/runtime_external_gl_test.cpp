#include "VernonRuntime.h"
#include "runtime/content_hash.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

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
  if (std::strcmp(name, glName) == 0)                                          \
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

std::string schema2Bundle(const nlohmann::json &artifact) {
  nlohmann::json root = {
      {"schema_version", 2},
      {"type", "pipeline"},
      {"id", "schema2/gl"},
      {"target", "opengl"},
      {"invocation_abi_version", 1},
      {"features", nlohmann::json::array()},
      {"variants",
       nlohmann::json::array(
           {{{"key", nlohmann::json::array()},
             {"parameters", nlohmann::json::array()},
             {"steps", nlohmann::json::array({{{"kind", "draw"},
                                               {"vertex", "vs"},
                                               {"fragment", "fs"}}})}}})},
      {"stage_artifacts",
       {{"vs",
         {{"id", "vs"},
          {"stage", "vertex"},
          {"entry", "main"},
          {"target", "opengl"},
          {"artifact", artifact}}},
        {"fs",
         {{"id", "fs"},
          {"stage", "fragment"},
          {"entry", "main"},
          {"target", "opengl"},
          {"artifact", artifact}}}}}};
  const std::string canonical = root.dump(-1, ' ', false);
  root["content_hash"] =
      vernon::runtime::sha256Hex(canonical.data(), canonical.size());
  return root.dump(-1, ' ', false);
}

nlohmann::json inlineArtifact(const std::string &source) {
  return {{"format", "glsl"},
          {"storage", "inline"},
          {"encoding", "utf8"},
          {"data", source},
          {"size", source.size()},
          {"sha256", vernon::runtime::sha256Hex(source.data(), source.size())}};
}

} // namespace

TEST(RuntimeExternalGl, LoadsAssetsAndInvokesPipeline) {
  VernonRuntimeCapabilities global =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_OPENGL_ES);
  ASSERT_TRUE(!global.available);
  ASSERT_TRUE(global.supports_graphics);

  VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
  ASSERT_TRUE(gl);
  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetContextCapabilities(gl);
  ASSERT_TRUE(capabilities.available && capabilities.supports_graphics);
  ASSERT_TRUE(capabilities.supports_compute &&
              capabilities.supports_storage_buffers);
  constexpr char glBundle[] =
      R"({"pipeline_bundle_schema_version":1,"invocation_abi_version":1,"type":"vernon_pipeline_bundle","id":"external-gl","target":"opengl","features":[],"variants":[{"key":[],"parameters":[],"steps":[{"kind":"draw","vertex":"vs","fragment":"fs"}]}],"stage_artifacts":{"vs":{"stage":"vertex","entry":"main","source":"#version 330\nvoid main(){gl_Position=vec4(0.0);}"},"fs":{"stage":"fragment","entry":"main","source":"#version 330\nout vec4 color;void main(){color=vec4(1.0);}"}}})";
  VernonPipelineBundle *glLoaded =
      vernonRuntimeLoadPipelineBundle(gl, glBundle, std::strlen(glBundle));
  ASSERT_TRUE(glLoaded);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(glLoaded, {nullptr, 0});
  ASSERT_TRUE(pipeline);
  VernonDeviceTexture *target = vernonRuntimeImportOpenGLTexture2D(
      gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
  ASSERT_TRUE(target);
  VernonTextureDescriptor cubeDescriptor{};
  cubeDescriptor.struct_size = sizeof(cubeDescriptor);
  cubeDescriptor.dimension = VERNON_TEXTURE_CUBE;
  cubeDescriptor.format = VERNON_TEXTURE_RGBA16_FLOAT;
  cubeDescriptor.width = 32;
  cubeDescriptor.height = 32;
  cubeDescriptor.depth = 1;
  cubeDescriptor.mip_levels = 6;
  VernonDeviceTexture *cube =
      vernonRuntimeImportOpenGLTexture(gl, 8, &cubeDescriptor);
  ASSERT_TRUE(cube);
  VernonDeviceSampler *sampler = vernonRuntimeImportOpenGLSampler(gl, 9);
  ASSERT_TRUE(sampler);
  VernonColorAttachment attachment{0, target};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.vertex_count = 3;
  invocation.instance_count = 1;
  ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(drawCount == 1);
  ASSERT_TRUE(vernonRuntimeTextureFree(target) == VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeTextureFree(cube) == VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeSamplerFree(sampler) == VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(glLoaded);
  ASSERT_TRUE(vernonRuntimeDestroy(gl) == VERNON_STATUS_OK);

  gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
  ASSERT_TRUE(gl);
  const std::string source = "#version 330\nvoid main(){}";
  const std::string inlineBundle = schema2Bundle(inlineArtifact(source));
  VernonRuntimeBackend schema2Target = VERNON_RUNTIME_CPU;
  ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(
                  inlineBundle.data(), inlineBundle.size(), &schema2Target) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(schema2Target == VERNON_RUNTIME_OPENGL);
  glLoaded = vernonRuntimeLoadPipelineBundle(gl, inlineBundle.data(),
                                             inlineBundle.size());
  ASSERT_TRUE(glLoaded);
  vernonRuntimePipelineBundleDestroy(glLoaded);

  std::string corruptManifest = inlineBundle;
  const size_t contentHash = corruptManifest.find("\"content_hash\":\"");
  ASSERT_TRUE(contentHash != std::string::npos);
  corruptManifest[contentHash + std::strlen("\"content_hash\":\"")] ^= 1;
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundle(gl, corruptManifest.data(),
                                               corruptManifest.size()));

  nlohmann::json corruptInline = inlineArtifact(source);
  corruptInline["sha256"] = std::string(64, '0');
  const std::string corruptInlineBundle = schema2Bundle(corruptInline);
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundle(gl, corruptInlineBundle.data(),
                                               corruptInlineBundle.size()));
  nlohmann::json wrongFormat = inlineArtifact(source);
  wrongFormat["format"] = "ptx";
  const std::string wrongFormatBundle = schema2Bundle(wrongFormat);
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundle(gl, wrongFormatBundle.data(),
                                               wrongFormatBundle.size()));

  const std::filesystem::path assetRoot =
      std::filesystem::temp_directory_path() / "vernon_runtime_schema2_test";
  std::filesystem::remove_all(assetRoot);
  std::filesystem::create_directories(assetRoot / "artifacts");
  const std::filesystem::path artifactPath = assetRoot / "artifacts/stage.glsl";
  {
    std::ofstream output(artifactPath, std::ios::binary);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
  }
  const nlohmann::json externalArtifact = {
      {"format", "glsl"},
      {"storage", "external"},
      {"path", "artifacts/stage.glsl"},
      {"size", source.size()},
      {"sha256", vernon::runtime::sha256Hex(source.data(), source.size())}};
  const std::string externalBundle = schema2Bundle(externalArtifact);
  const std::string assetRootUtf8 = assetRoot.u8string();
  VernonPipelineBundleLoadOptions options{};
  options.struct_size = sizeof(options);
  options.bundle_directory = assetRootUtf8.c_str();
  glLoaded = vernonRuntimeLoadPipelineBundleWithOptions(
      gl, externalBundle.data(), externalBundle.size(), &options);
  ASSERT_TRUE(glLoaded);
  vernonRuntimePipelineBundleDestroy(glLoaded);

  nlohmann::json traversalArtifact = externalArtifact;
  traversalArtifact["path"] = "../stage.glsl";
  const std::string traversalBundle = schema2Bundle(traversalArtifact);
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(
      gl, traversalBundle.data(), traversalBundle.size(), &options));
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundle(gl, externalBundle.data(),
                                               externalBundle.size()));

  const std::filesystem::path outsidePath =
      assetRoot.parent_path() / "vernon_runtime_schema2_escape.glsl";
  {
    std::ofstream output(outsidePath, std::ios::binary);
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
  }
  std::error_code symlinkError;
  const std::filesystem::path symlinkPath = assetRoot / "artifacts/escape.glsl";
  std::filesystem::create_symlink(outsidePath, symlinkPath, symlinkError);
  if (!symlinkError) {
    nlohmann::json symlinkArtifact = externalArtifact;
    symlinkArtifact["path"] = "artifacts/escape.glsl";
    const std::string symlinkBundle = schema2Bundle(symlinkArtifact);
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(
        gl, symlinkBundle.data(), symlinkBundle.size(), &options));
  }
  std::filesystem::remove(outsidePath);

  {
    std::ofstream output(artifactPath, std::ios::binary | std::ios::trunc);
    output << "corrupt";
  }
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(
      gl, externalBundle.data(), externalBundle.size(), &options));
  std::filesystem::remove_all(assetRoot);
  ASSERT_TRUE(vernonRuntimeDestroy(gl) == VERNON_STATUS_OK);

  VernonRuntimeContext *gles = create(VERNON_RUNTIME_OPENGL_ES, 3, 1);
  ASSERT_TRUE(gles);
  capabilities = vernonRuntimeGetContextCapabilities(gles);
  ASSERT_TRUE(capabilities.supports_compute);
  constexpr char bundle[] =
      R"({"pipeline_bundle_schema_version":1,"invocation_abi_version":1,"type":"vernon_pipeline_bundle","id":"gles-profile","target":"opengles","features":[],"variants":[{"key":[],"parameters":[],"steps":[{"kind":"draw","vertex":"vs","fragment":"fs"}]}],"stage_artifacts":{"vs":{"stage":"vertex","entry":"main","source":"#version 310 es\nvoid main(){}"},"fs":{"stage":"fragment","entry":"main","source":"#version 310 es\nvoid main(){}"}}})";
  VernonPipelineBundle *loaded =
      vernonRuntimeLoadPipelineBundle(gles, bundle, std::strlen(bundle));
  ASSERT_TRUE(loaded);
  vernonRuntimePipelineBundleDestroy(loaded);
  ASSERT_TRUE(vernonRuntimeDestroy(gles) == VERNON_STATUS_OK);
}

#undef GL_CALL
