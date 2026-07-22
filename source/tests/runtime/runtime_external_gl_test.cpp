#include "VernonRuntime.h"
#include "runtime/content_hash.h"

#include <nlohmann/json.hpp>

#include <array>
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
using GlBoolean = unsigned char;
using GlInt = int;
using GlSize = int;
using GlUint = unsigned int;

constexpr GlEnum kCompileStatus = 0x8B81;
constexpr GlEnum kLinkStatus = 0x8B82;
constexpr GlEnum kFramebufferComplete = 0x8CD5;

GlUint nextName = 1;
uint32_t drawCount = 0;
GlBoolean matrixTranspose = 1;
std::array<float, 16> matrixUpload{};
std::array<float, 2> resolutionUpload{};
std::array<GlInt, 4> viewportUpload{};
GlUint boundSamplerUnit = UINT32_MAX;
GlUint boundSamplerName = UINT32_MAX;

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
GlInt GL_CALL getUniformLocation(GlUint, const char *) { return 0; }
void GL_CALL uniformMatrix4fv(GlInt, GlSize, GlBoolean transpose,
                              const float *data) {
  matrixTranspose = transpose;
  std::copy_n(data, matrixUpload.size(), matrixUpload.begin());
}
void GL_CALL uniform2fv(GlInt, GlSize, const float *data) {
  std::copy_n(data, resolutionUpload.size(), resolutionUpload.begin());
}
void GL_CALL bindSampler(GlUint unit, GlUint sampler) {
  boundSamplerUnit = unit;
  boundSamplerName = sampler;
}
void GL_CALL viewport(GlInt x, GlInt y, GlSize width, GlSize height) {
  viewportUpload = {x, y, width, height};
}

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
  PROC("glGetUniformLocation", getUniformLocation);
  PROC("glUniform2fv", uniform2fv);
  PROC("glUniformMatrix4fv", uniformMatrix4fv);
  PROC("glBindSampler", bindSampler);
  PROC("glViewport", viewport);
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

nlohmann::json inlineArtifact(const std::string &source);

std::string schema2Bundle(const nlohmann::json &artifact) {
  nlohmann::json root = {
      {"schema_version", 2},
      {"type", "pipeline"},
      {"id", "schema2/gl"},
      {"target", "opengl"},
      {"invocation_abi_version", 3},
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

std::string matrixBundle(const char *target) {
  const std::string source =
      "#version 330\nuniform mat4 transform;void main(){gl_Position=transform*"
      "vec4(0.0,0.0,0.0,1.0);}";
  nlohmann::json root =
      nlohmann::json::parse(schema2Bundle(inlineArtifact(source)));
  root["target"] = target;
  root["stage_artifacts"]["vs"]["target"] = target;
  root["stage_artifacts"]["fs"]["target"] = target;
  if (std::strcmp(target, "opengles") == 0) {
    root["stage_artifacts"]["vs"]["artifact"]["format"] = "gles";
    root["stage_artifacts"]["fs"]["artifact"]["format"] = "gles";
  }
  root["variants"][0]["parameters"] = nlohmann::json::array(
      {{{"slot", 0},
        {"name", "transform"},
        {"kind", "tensor"},
        {"dtype", "f32"},
        {"access", "read"},
        {"shape", nlohmann::json::array({4, 4})},
        {"uses", nlohmann::json::array(
                     {{{"stage", "vertex"},
                       {"interface", "uniform"},
                       {"uniform_name", "transform"},
                       {"dtype", "f32"},
                       {"shape", nlohmann::json::array({4, 4})}}})}}});
  root.erase("content_hash");
  const std::string canonical = root.dump(-1, ' ', false);
  root["content_hash"] =
      vernon::runtime::sha256Hex(canonical.data(), canonical.size());
  return root.dump(-1, ' ', false);
}

std::string internalValueBundle() {
  nlohmann::json root = nlohmann::json::parse(
      schema2Bundle(inlineArtifact("#version 330\nvoid main(){}")));
  root["variants"][0]["parameters"] = nlohmann::json::array(
      {{{"slot", 0},
        {"name", "image"},
        {"kind", "texture"},
        {"dtype", "f32"},
        {"access", "read"},
        {"dimension", "2d"},
        {"shape", nlohmann::json::array()},
        {"uses", nlohmann::json::array({{{"stage", "fragment"},
                                         {"interface", "resource"},
                                         {"index", 0},
                                         {"uniform_name", "image"},
                                         {"vernon.set", 0},
                                         {"vernon.binding", 3}}})}}});
  root["variants"][0]["internal_parameters"] = nlohmann::json::array(
      {{{"name", "__image_sampler"},
        {"kind", "sampler"},
        {"access", "read"},
        {"source", "implicit_sampler"},
        {"uses",
         nlohmann::json::array(
             {{{"stage", "fragment"},
               {"interface", "resource"},
               {"index", 1},
               {"sampled_texture_bindings",
                nlohmann::json::array({{{"set", 0}, {"binding", 3}}})}}})}},
       {{"name", "__resolution"},
        {"kind", "tensor"},
        {"dtype", "f32"},
        {"shape", nlohmann::json::array({2})},
        {"access", "read"},
        {"source", "system_value"},
        {"system_value", "resolution"},
        {"uses",
         nlohmann::json::array({{{"stage", "fragment"},
                                 {"interface", "uniform"},
                                 {"index", 2},
                                 {"dtype", "f32"},
                                 {"shape", nlohmann::json::array({2})},
                                 {"uniform_name", "__resolution"}}})}}});
  root.erase("content_hash");
  const std::string canonical = root.dump(-1, ' ', false);
  root["content_hash"] =
      vernon::runtime::sha256Hex(canonical.data(), canonical.size());
  return root.dump(-1, ' ', false);
}

std::string explicitSamplerBundle() {
  nlohmann::json root = nlohmann::json::parse(internalValueBundle());
  nlohmann::json &internal = root["variants"][0]["internal_parameters"];
  const auto sampler =
      std::find_if(internal.begin(), internal.end(), [](const auto &row) {
        return row.value("source", "") == "implicit_sampler";
      });
  nlohmann::json externalSampler = *sampler;
  externalSampler.erase("source");
  externalSampler["name"] = "explicit_sampler";
  externalSampler["slot"] = 1;
  root["variants"][0]["parameters"].push_back(std::move(externalSampler));
  internal.erase(sampler);
  root.erase("content_hash");
  const std::string canonical = root.dump(-1, ' ', false);
  root["content_hash"] =
      vernon::runtime::sha256Hex(canonical.data(), canonical.size());
  return root.dump(-1, ' ', false);
}

void expectMatrixUpload(VernonRuntimeBackend backend, const char *target,
                        uint16_t major, uint16_t minor,
                        const std::array<float, 16> &storage,
                        const std::array<uint64_t, 2> &strides,
                        GlBoolean expectedTranspose,
                        const std::array<float, 16> &expectedUpload) {
  matrixTranspose = expectedTranspose ? 0 : 1;
  matrixUpload.fill(0.0F);
  VernonRuntimeContext *gl = create(backend, major, minor);
  ASSERT_TRUE(gl);
  const std::string bundleData = matrixBundle(target);
  VernonPipelineBundle *bundle =
      vernonRuntimeLoadPipelineBundle(gl, bundleData.data(), bundleData.size());
  ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data,
                                     vernonRuntimeGetLastError(gl).size);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
  ASSERT_TRUE(pipeline);
  VernonDeviceTexture *renderTarget = vernonRuntimeImportOpenGLTexture2D(
      gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
  ASSERT_TRUE(renderTarget);

  constexpr std::array<uint64_t, 2> shape = {4, 4};
  VernonPipelineArgument argument{};
  argument.slot = 0;
  argument.kind = VERNON_PIPELINE_TENSOR;
  argument.tensor.struct_size = sizeof(VernonTensorView);
  argument.tensor.storage = VERNON_TENSOR_HOST;
  argument.tensor.host_data = storage.data();
  argument.tensor.dtype = VERNON_DATA_F32;
  argument.tensor.access = VERNON_ACCESS_READ;
  argument.tensor.rank = 2;
  argument.tensor.shape = shape.data();
  argument.tensor.byte_strides = strides.data();
  argument.tensor.byte_size = sizeof(storage);
  VernonColorAttachment attachment{0, renderTarget};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = &argument;
  invocation.argument_count = 1;
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.vertex_count = 3;
  invocation.instance_count = 1;
  ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation),
            VERNON_STATUS_OK);
  EXPECT_EQ(matrixTranspose, expectedTranspose);
  EXPECT_EQ(matrixUpload, expectedUpload);

  ASSERT_EQ(vernonRuntimeTextureFree(renderTarget), VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(bundle);
  ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
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

TEST(RuntimeExternalGl, UploadsColumnMajorMatricesWithoutCopying) {
  constexpr std::array<uint64_t, 2> strides = {sizeof(float),
                                               4 * sizeof(float)};
  constexpr std::array<float, 16> columnMajor = {
      1.0F, 5.0F, 9.0F,  13.0F, 2.0F, 6.0F, 10.0F, 14.0F,
      3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
  };
  expectMatrixUpload(VERNON_RUNTIME_OPENGL, "opengl", 4, 3, columnMajor,
                     strides, 0, columnMajor);
}

TEST(RuntimeExternalGl, UploadsRowMajorMatricesWithDesktopTranspose) {
  constexpr std::array<uint64_t, 2> strides = {4 * sizeof(float),
                                               sizeof(float)};
  constexpr std::array<float, 16> rowMajor = {
      1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
      9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
  };
  expectMatrixUpload(VERNON_RUNTIME_OPENGL, "opengl", 4, 3, rowMajor, strides,
                     1, rowMajor);
}

TEST(RuntimeExternalGl, PacksRowMajorMatricesForOpenGlEs) {
  constexpr std::array<uint64_t, 2> strides = {4 * sizeof(float),
                                               sizeof(float)};
  constexpr std::array<float, 16> rowMajor = {
      1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
      9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
  };
  constexpr std::array<float, 16> columnMajor = {
      1.0F, 5.0F, 9.0F,  13.0F, 2.0F, 6.0F, 10.0F, 14.0F,
      3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
  };
  expectMatrixUpload(VERNON_RUNTIME_OPENGL_ES, "opengles", 3, 1, rowMajor,
                     strides, 0, columnMajor);
}

TEST(RuntimeExternalGl, SuppliesImplicitSamplerAndEffectiveResolution) {
  resolutionUpload.fill(0.0F);
  viewportUpload.fill(0);
  boundSamplerUnit = UINT32_MAX;
  boundSamplerName = UINT32_MAX;
  VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
  ASSERT_TRUE(gl);
  const std::string bundleData = internalValueBundle();
  VernonPipelineBundle *bundle =
      vernonRuntimeLoadPipelineBundle(gl, bundleData.data(), bundleData.size());
  ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data,
                                     vernonRuntimeGetLastError(gl).size);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
  ASSERT_TRUE(pipeline);
  ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterCount(pipeline), 1u);

  VernonDeviceTexture *sampled = vernonRuntimeImportOpenGLTexture2D(
      gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
  VernonDeviceTexture *target = vernonRuntimeImportOpenGLTexture2D(
      gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
  VernonDeviceSampler *textureSampler =
      vernonRuntimeImportOpenGLSampler(gl, 19);
  ASSERT_TRUE(sampled && target && textureSampler);
  VernonPipelineArgument argument{};
  argument.slot = 0;
  argument.kind = VERNON_PIPELINE_TEXTURE;
  argument.texture = {sampled,
                      VERNON_TEXTURE_RGBA8_UNORM,
                      VERNON_ACCESS_READ,
                      VERNON_TEXTURE_2D,
                      4,
                      4,
                      1,
                      textureSampler};
  VernonColorAttachment attachment{0, target};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = &argument;
  invocation.argument_count = 1;
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.vertex_count = 3;
  invocation.instance_count = 1;
  invocation.viewport[0] = 2;
  invocation.viewport[1] = 3;
  invocation.viewport[2] = 7;
  invocation.viewport[3] = 9;
  ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation),
            VERNON_STATUS_OK);
  EXPECT_EQ(boundSamplerUnit, 3u);
  EXPECT_EQ(boundSamplerName, 19u);
  EXPECT_EQ(resolutionUpload, (std::array<float, 2>{7.0F, 9.0F}));
  EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{2, 3, 7, 9}));

  argument.texture.sampler = nullptr;
  std::fill(std::begin(invocation.viewport), std::end(invocation.viewport), 0);
  ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation),
            VERNON_STATUS_OK);
  EXPECT_EQ(resolutionUpload, (std::array<float, 2>{16.0F, 12.0F}));
  EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{0, 0, 16, 12}));
  EXPECT_EQ(boundSamplerName, 0u);

  VernonRuntimeContext *other = create(VERNON_RUNTIME_OPENGL, 4, 3);
  ASSERT_TRUE(other);
  VernonDeviceSampler *foreignSampler =
      vernonRuntimeImportOpenGLSampler(other, 20);
  ASSERT_TRUE(foreignSampler);
  argument.texture.sampler = foreignSampler;
  EXPECT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation),
            VERNON_STATUS_INVALID_ARGUMENT);
  ASSERT_EQ(vernonRuntimeSamplerFree(foreignSampler), VERNON_STATUS_OK);
  ASSERT_EQ(vernonRuntimeDestroy(other), VERNON_STATUS_OK);

  ASSERT_EQ(vernonRuntimeSamplerFree(textureSampler), VERNON_STATUS_OK);
  ASSERT_EQ(vernonRuntimeTextureFree(target), VERNON_STATUS_OK);
  ASSERT_EQ(vernonRuntimeTextureFree(sampled), VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(bundle);
  ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ExplicitSamplerOverridesTextureViewSampler) {
  boundSamplerName = UINT32_MAX;
  VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
  ASSERT_TRUE(gl);
  const std::string bundleData = explicitSamplerBundle();
  VernonPipelineBundle *bundle =
      vernonRuntimeLoadPipelineBundle(gl, bundleData.data(), bundleData.size());
  ASSERT_TRUE(bundle);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
  ASSERT_TRUE(pipeline);
  ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterCount(pipeline), 2u);

  VernonDeviceTexture *sampled = vernonRuntimeImportOpenGLTexture2D(
      gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
  VernonDeviceTexture *target = vernonRuntimeImportOpenGLTexture2D(
      gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
  VernonDeviceSampler *textureSampler =
      vernonRuntimeImportOpenGLSampler(gl, 20);
  VernonDeviceSampler *explicitSampler =
      vernonRuntimeImportOpenGLSampler(gl, 21);
  ASSERT_TRUE(sampled && target && textureSampler && explicitSampler);

  VernonPipelineArgument arguments[2]{};
  arguments[0].slot = 0;
  arguments[0].kind = VERNON_PIPELINE_TEXTURE;
  arguments[0].texture = {sampled,
                          VERNON_TEXTURE_RGBA8_UNORM,
                          VERNON_ACCESS_READ,
                          VERNON_TEXTURE_2D,
                          4,
                          4,
                          1,
                          textureSampler};
  arguments[1].slot = 1;
  arguments[1].kind = VERNON_PIPELINE_SAMPLER;
  arguments[1].sampler = explicitSampler;
  VernonColorAttachment attachment{0, target};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = arguments;
  invocation.argument_count = std::size(arguments);
  invocation.color_attachments = &attachment;
  invocation.color_attachment_count = 1;
  invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
  invocation.vertex_count = 3;
  invocation.instance_count = 1;
  ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation),
            VERNON_STATUS_OK);
  EXPECT_EQ(boundSamplerName, 21u);

  ASSERT_EQ(vernonRuntimeSamplerFree(explicitSampler), VERNON_STATUS_OK);
  ASSERT_EQ(vernonRuntimeSamplerFree(textureSampler), VERNON_STATUS_OK);
  ASSERT_EQ(vernonRuntimeTextureFree(target), VERNON_STATUS_OK);
  ASSERT_EQ(vernonRuntimeTextureFree(sampled), VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(bundle);
  ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, LoadsAssetsAndInvokesPipeline) {
  drawCount = 0;
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
  const std::string glBundle =
      schema2Bundle(inlineArtifact("#version 330\nvoid main(){}"));
  VernonPipelineBundle *glLoaded =
      vernonRuntimeLoadPipelineBundle(gl, glBundle.data(), glBundle.size());
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
  const std::string bundle = matrixBundle("opengles");
  VernonPipelineBundle *loaded =
      vernonRuntimeLoadPipelineBundle(gles, bundle.data(), bundle.size());
  ASSERT_TRUE(loaded);
  vernonRuntimePipelineBundleDestroy(loaded);
  ASSERT_TRUE(vernonRuntimeDestroy(gles) == VERNON_STATUS_OK);
}

#undef GL_CALL
