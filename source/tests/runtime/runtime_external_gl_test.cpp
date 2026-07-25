#include "VernonRuntime.h"
#include "runtime/content_hash.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

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
constexpr GlEnum kArrayBuffer = 0x8892;
constexpr GlEnum kElementArrayBuffer = 0x8893;

GlUint nextName = 1;
uint32_t drawCount = 0;
uint32_t dispatchCount = 0;
uint32_t storageBindingCount = 0;
uint32_t memoryBarrierCount = 0;
uint32_t clearCount = 0;
GlInt clearedDrawBuffer = -1;
std::array<float, 4> clearColor{};
GlBoolean matrixTranspose = 1;
std::array<float, 16> matrixUpload{};
std::array<float, 2> resolutionUpload{};
std::array<GlInt, 4> viewportUpload{};

VernonTextureDescriptor texture2DDescriptor(uint32_t width, uint32_t height, VernonTextureFormat format) {
    return {sizeof(VernonTextureDescriptor), VERNON_TEXTURE_2D, format, width, height, 1, 1, {0, 0, 0, 0}};
}

VernonDeviceTexture *createTexture2D(VernonRuntimeContext *runtime, uint32_t width, uint32_t height,
                                     VernonTextureFormat format) {
    const VernonTextureDescriptor descriptor = texture2DDescriptor(width, height, format);
    return vernonRuntimeTextureCreate(runtime, &descriptor);
}

VernonDeviceTexture *importOpenGLTexture2D(VernonRuntimeContext *runtime, uint32_t texture, uint32_t width,
                                           uint32_t height, VernonTextureFormat format) {
    const VernonTextureDescriptor descriptor = texture2DDescriptor(width, height, format);
    return vernonRuntimeImportOpenGLTexture(runtime, texture, &descriptor);
}
GlUint boundSamplerUnit = UINT32_MAX;
GlUint boundSamplerName = UINT32_MAX;
GlUint boundArrayBuffer = 0;
GlUint boundIndexBuffer = 0;
GlUint boundVertexArray = 0;
GlUint vertexAttributeArray = 0;
GlUint vertexAttributeLocation = UINT32_MAX;
GlUint capturedVertexAttributeDivisor = 0;
GlInt vertexAttributeComponents = 0;
GlSize vertexAttributeStride = 0;
uintptr_t vertexAttributeOffset = 0;
GlSize indexedDrawCount = 0;
uint32_t makeCurrentCount = 0;
std::vector<GlUint> deletedBuffers;
std::vector<GlUint> deletedTextures;
std::vector<GlUint> deletedSamplers;
std::vector<unsigned char> bufferStorage;
std::vector<unsigned char> textureStorage;

void makeCurrent(void *) { ++makeCurrentCount; }
void GL_CALL placeholder() {}
GlUint GL_CALL createName(GlEnum) { return nextName++; }
GlUint GL_CALL createProgram() { return nextName++; }
void GL_CALL shaderSource(GlUint, GlSize, const char *const *, const GlInt *) {}
void GL_CALL getShaderiv(GlUint, GlEnum name, GlInt *value) { *value = name == kCompileStatus ? 1 : 0; }
void GL_CALL getProgramiv(GlUint, GlEnum name, GlInt *value) { *value = name == kLinkStatus ? 1 : 0; }
void GL_CALL genNames(GlSize count, GlUint *names) {
    while (count--)
        *names++ = nextName++;
}
void GL_CALL deleteBufferNames(GlSize count, const GlUint *names) {
    deletedBuffers.insert(deletedBuffers.end(), names, names + count);
}
void GL_CALL deleteTextureNames(GlSize count, const GlUint *names) {
    deletedTextures.insert(deletedTextures.end(), names, names + count);
}
void GL_CALL deleteSamplerNames(GlSize count, const GlUint *names) {
    deletedSamplers.insert(deletedSamplers.end(), names, names + count);
}
void GL_CALL bufferData(GlEnum, std::intptr_t size, const void *, GlEnum) {
    bufferStorage.resize(static_cast<size_t>(size));
}
void GL_CALL bindBuffer(GlEnum target, GlUint name) {
    if (target == kArrayBuffer)
        boundArrayBuffer = name;
    else if (target == kElementArrayBuffer)
        boundIndexBuffer = name;
}
void GL_CALL bindVertexArray(GlUint name) { boundVertexArray = name; }
void GL_CALL bindBufferBase(GlEnum, GlUint, GlUint) { ++storageBindingCount; }
void GL_CALL dispatchCompute(GlUint, GlUint, GlUint) { ++dispatchCount; }
void GL_CALL memoryBarrier(unsigned) { ++memoryBarrierCount; }
void GL_CALL vertexAttribPointer(GlUint location, GlInt components, GlEnum, GlBoolean, GlSize stride,
                                 const void *offset) {
    vertexAttributeArray = boundVertexArray;
    vertexAttributeLocation = location;
    vertexAttributeComponents = components;
    vertexAttributeStride = stride;
    vertexAttributeOffset = reinterpret_cast<uintptr_t>(offset);
}
void GL_CALL vertexAttribDivisor(GlUint, GlUint divisor) { capturedVertexAttributeDivisor = divisor; }
void GL_CALL drawElementsInstanced(GlEnum, GlSize count, GlEnum, const void *, GlSize) {
    indexedDrawCount = count;
    ++drawCount;
}
void GL_CALL bufferSubData(GlEnum, std::intptr_t offset, std::intptr_t size, const void *source) {
    std::memcpy(bufferStorage.data() + offset, source, static_cast<size_t>(size));
}
void *GL_CALL mapBufferRange(GlEnum, std::intptr_t offset, std::intptr_t, unsigned) {
    return bufferStorage.data() + offset;
}
GlBoolean GL_CALL unmapBuffer(GlEnum) { return 1; }
void GL_CALL texImage2D(GlEnum, GlInt, GlInt, GlSize width, GlSize height, GlInt, GlEnum, GlEnum, const void *) {
    textureStorage.resize(static_cast<size_t>(width) * height * 4);
}
void GL_CALL texSubImage2D(GlEnum, GlInt, GlInt, GlInt, GlSize width, GlSize height, GlEnum, GlEnum,
                           const void *source) {
    const size_t size = static_cast<size_t>(width) * height * 4;
    textureStorage.assign(static_cast<const unsigned char *>(source),
                          static_cast<const unsigned char *>(source) + size);
}
void GL_CALL readPixels(GlInt, GlInt, GlSize width, GlSize height, GlEnum, GlEnum, void *destination) {
    std::memcpy(destination, textureStorage.data(), static_cast<size_t>(width) * height * 4);
}
GlEnum GL_CALL framebufferStatus(GlEnum) { return kFramebufferComplete; }
void GL_CALL drawArrays(GlEnum, GlInt, GlSize) { ++drawCount; }
void GL_CALL clearBufferfv(GlEnum, GlInt drawBuffer, const float *value) {
    ++clearCount;
    clearedDrawBuffer = drawBuffer;
    std::copy_n(value, clearColor.size(), clearColor.begin());
}
GlInt GL_CALL getUniformLocation(GlUint, const char *) { return 0; }
void GL_CALL uniformMatrix4fv(GlInt, GlSize, GlBoolean transpose, const float *data) {
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
void GL_CALL viewport(GlInt x, GlInt y, GlSize width, GlSize height) { viewportUpload = {x, y, width, height}; }

void *getProcAddress(void *, const char *name) {
#define PROC(glName, function)                                                                                         \
    if (std::strcmp(name, glName) == 0)                                                                                \
    return reinterpret_cast<void *>(&function)
    PROC("glCreateShader", createName);
    PROC("glShaderSource", shaderSource);
    PROC("glGetShaderiv", getShaderiv);
    PROC("glCreateProgram", createProgram);
    PROC("glGetProgramiv", getProgramiv);
    PROC("glGenVertexArrays", genNames);
    PROC("glBindVertexArray", bindVertexArray);
    PROC("glGenFramebuffers", genNames);
    PROC("glGenBuffers", genNames);
    PROC("glBindBuffer", bindBuffer);
    PROC("glBindBufferBase", bindBufferBase);
    PROC("glDeleteBuffers", deleteBufferNames);
    PROC("glBufferData", bufferData);
    PROC("glBufferSubData", bufferSubData);
    PROC("glMapBufferRange", mapBufferRange);
    PROC("glUnmapBuffer", unmapBuffer);
    PROC("glGenTextures", genNames);
    PROC("glDeleteTextures", deleteTextureNames);
    PROC("glTexImage2D", texImage2D);
    PROC("glTexSubImage2D", texSubImage2D);
    PROC("glReadPixels", readPixels);
    PROC("glGenSamplers", genNames);
    PROC("glDeleteSamplers", deleteSamplerNames);
    PROC("glCheckFramebufferStatus", framebufferStatus);
    PROC("glDrawArrays", drawArrays);
    PROC("glVertexAttribPointer", vertexAttribPointer);
    PROC("glVertexAttribDivisor", vertexAttribDivisor);
    PROC("glDrawElementsInstanced", drawElementsInstanced);
    PROC("glClearBufferfv", clearBufferfv);
    PROC("glGetUniformLocation", getUniformLocation);
    PROC("glUniform2fv", uniform2fv);
    PROC("glUniformMatrix4fv", uniformMatrix4fv);
    PROC("glBindSampler", bindSampler);
    PROC("glViewport", viewport);
    PROC("glDispatchCompute", dispatchCompute);
    PROC("glMemoryBarrier", memoryBarrier);
#undef PROC
    return reinterpret_cast<void *>(&placeholder);
}

VernonRuntimeContext *create(VernonRuntimeBackend backend, uint16_t major, uint16_t minor) {
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = &getProcAddress;
    callbacks.api_version_major = major;
    callbacks.api_version_minor = minor;
    return vernonRuntimeCreateOpenGLWithCallbacks(backend, &callbacks);
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
        {"runtime_requirements",
         {{"backend", "opengl"},
          {"features", nlohmann::json::array({"textures"})},
          {"glsl_version", 330},
          {"profile", "core"},
          {"api_version", nlohmann::json::array({3, 3})}}},
        {"variants", nlohmann::json::array({{{"key", nlohmann::json::array()},
                                             {"parameters", nlohmann::json::array()},
                                             {"program", {{"vertex", "vs"}, {"fragment", "fs"}}}}})},
        {"stage_artifacts",
         {{"vs", {{"id", "vs"}, {"stage", "vertex"}, {"entry", "main"}, {"target", "opengl"}, {"artifact", artifact}}},
          {"fs",
           {{"id", "fs"}, {"stage", "fragment"}, {"entry", "main"}, {"target", "opengl"}, {"artifact", artifact}}}}}};
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string matrixBundle(const char *target) {
    const std::string source = "#version 330\nuniform mat4 transform;void main(){gl_Position=transform*"
                               "vec4(0.0,0.0,0.0,1.0);}";
    nlohmann::json root = nlohmann::json::parse(schema2Bundle(inlineArtifact(source)));
    root["target"] = target;
    root["runtime_requirements"]["backend"] = target;
    root["stage_artifacts"]["vs"]["target"] = target;
    root["stage_artifacts"]["fs"]["target"] = target;
    if (std::strcmp(target, "opengles") == 0) {
        root["runtime_requirements"]["api_version"] = nlohmann::json::array({3, 1});
        root["runtime_requirements"]["glsl_version"] = 310;
        root["runtime_requirements"]["profile"] = "es";
        root["stage_artifacts"]["vs"]["artifact"]["format"] = "gles";
        root["stage_artifacts"]["fs"]["artifact"]["format"] = "gles";
    }
    root["variants"][0]["parameters"] =
        nlohmann::json::array({{{"slot", 0},
                                {"name", "transform"},
                                {"kind", "tensor"},
                                {"dtype", "f32"},
                                {"access", "read"},
                                {"shape", nlohmann::json::array({4, 4})},
                                {"uses", nlohmann::json::array({{{"stage", "vertex"},
                                                                 {"interface", "uniform"},
                                                                 {"uniform_name", "transform"},
                                                                 {"dtype", "f32"},
                                                                 {"shape", nlohmann::json::array({4, 4})}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string vertexInputBundle() {
    nlohmann::json root =
        nlohmann::json::parse(schema2Bundle(inlineArtifact("#version 330\nvoid main(){gl_Position=vec4(0.0);}")));
    root["variants"][0]["parameters"] =
        nlohmann::json::array({{{"slot", 0},
                                {"name", "position"},
                                {"kind", "tensor"},
                                {"dtype", "f32"},
                                {"access", "read"},
                                {"shape", nlohmann::json::array({3})},
                                {"uses", nlohmann::json::array({{{"stage", "vertex"},
                                                                 {"interface", "input"},
                                                                 {"vernon.location", 0},
                                                                 {"vernon.instance_divisor", 1},
                                                                 {"dtype", "f32"},
                                                                 {"shape", nlohmann::json::array({3})}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string internalValueBundle() {
    nlohmann::json root = nlohmann::json::parse(schema2Bundle(inlineArtifact("#version 330\nvoid main(){}")));
    root["variants"][0]["parameters"] =
        nlohmann::json::array({{{"slot", 0},
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
          {"uses", nlohmann::json::array(
                       {{{"stage", "fragment"},
                         {"interface", "resource"},
                         {"index", 1},
                         {"sampled_texture_bindings", nlohmann::json::array({{{"set", 0}, {"binding", 3}}})}}})}},
         {{"name", "__resolution"},
          {"kind", "tensor"},
          {"dtype", "f32"},
          {"shape", nlohmann::json::array({2})},
          {"access", "read"},
          {"source", "system_value"},
          {"system_value", "resolution"},
          {"uses", nlohmann::json::array({{{"stage", "fragment"},
                                           {"interface", "uniform"},
                                           {"index", 2},
                                           {"dtype", "f32"},
                                           {"shape", nlohmann::json::array({2})},
                                           {"uniform_name", "__resolution"}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string explicitSamplerBundle() {
    nlohmann::json root = nlohmann::json::parse(internalValueBundle());
    nlohmann::json &internal = root["variants"][0]["internal_parameters"];
    const auto sampler = std::find_if(internal.begin(), internal.end(),
                                      [](const auto &row) { return row.value("source", "") == "implicit_sampler"; });
    nlohmann::json externalSampler = *sampler;
    externalSampler.erase("source");
    externalSampler["name"] = "explicit_sampler";
    externalSampler["slot"] = 1;
    root["variants"][0]["parameters"].push_back(std::move(externalSampler));
    internal.erase(sampler);
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

void expectMatrixUpload(VernonRuntimeBackend backend, const char *target, uint16_t major, uint16_t minor,
                        const std::array<float, 16> &storage, const std::array<int64_t, 2> &strides,
                        GlBoolean expectedTranspose, const std::array<float, 16> &expectedUpload) {
    matrixTranspose = expectedTranspose ? 0 : 1;
    matrixUpload.fill(0.0F);
    clearCount = 0;
    clearedDrawBuffer = -1;
    clearColor.fill(1.0F);
    VernonRuntimeContext *gl = create(backend, major, minor);
    ASSERT_TRUE(gl);
    const std::string bundleData = matrixBundle(target);
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    VernonDeviceTexture *renderTarget = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
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
    const GlUint nextNameAfterPreparation = nextName;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(nextName, nextNameAfterPreparation);
    EXPECT_EQ(matrixTranspose, expectedTranspose);
    EXPECT_EQ(matrixUpload, expectedUpload);
    EXPECT_EQ(clearCount, 2U);
    EXPECT_EQ(clearedDrawBuffer, 0);
    EXPECT_EQ(clearColor, (std::array<float, 4>{}));

    ASSERT_EQ(vernonRuntimeTextureFree(renderTarget), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

nlohmann::json inlineArtifact(const std::string &source) {
    return {{"format", "glsl"},      {"storage", "inline"},
            {"encoding", "utf8"},    {"data", source},
            {"size", source.size()}, {"sha256", vernon::runtime::sha256Hex(source.data(), source.size())}};
}

} // namespace

TEST(RuntimeExternalGl, OwnsAllocatedResourcesButNotImportedNames) {
    deletedBuffers.clear();
    deletedTextures.clear();
    deletedSamplers.clear();
    makeCurrentCount = 0;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);

    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(gl, 16, 4);
    VernonDeviceBuffer *importedBuffer = vernonRuntimeImportOpenGLBuffer(gl, 9001, 16, 4);
    ASSERT_TRUE(buffer && importedBuffer);
    const std::array<unsigned char, 16> bytes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    ASSERT_EQ(vernonRuntimeCopyFromHost(buffer, 0, bytes.data(), bytes.size()), VERNON_STATUS_OK);
    std::array<unsigned char, 16> downloaded{};
    ASSERT_EQ(vernonRuntimeCopyToHost(buffer, 0, downloaded.data(), downloaded.size()), VERNON_STATUS_OK);
    EXPECT_EQ(downloaded, bytes);

    VernonDeviceTexture *texture = createTexture2D(gl, 2, 2, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceTexture *importedTexture = importOpenGLTexture2D(gl, 9002, 2, 2, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_TRUE(texture && importedTexture);
    ASSERT_EQ(vernonRuntimeTextureCopyFromHost(texture, bytes.data(), bytes.size()), VERNON_STATUS_OK);
    downloaded.fill(0);
    ASSERT_EQ(vernonRuntimeTextureCopyToHost(texture, downloaded.data(), downloaded.size()), VERNON_STATUS_OK);
    EXPECT_EQ(downloaded, bytes);

    VernonSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.wrap_u = VERNON_SAMPLER_REPEAT;
    descriptor.wrap_v = VERNON_SAMPLER_REPEAT;
    descriptor.wrap_w = VERNON_SAMPLER_REPEAT;
    descriptor.min_filter = VERNON_SAMPLER_LINEAR;
    descriptor.mag_filter = VERNON_SAMPLER_LINEAR;
    descriptor.mip_filter = VERNON_SAMPLER_LINEAR;
    VernonDeviceSampler *sampler = vernonRuntimeSamplerCreate(gl, &descriptor);
    VernonDeviceSampler *importedSampler = vernonRuntimeImportOpenGLSampler(gl, 9003);
    ASSERT_TRUE(sampler && importedSampler);

    ASSERT_EQ(vernonRuntimeBufferFree(importedBuffer), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureFree(importedTexture), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeSamplerFree(importedSampler), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeBufferFree(buffer), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureFree(texture), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeSamplerFree(sampler), VERNON_STATUS_OK);
    EXPECT_EQ(deletedBuffers.size(), 1u);
    EXPECT_EQ(deletedTextures.size(), 1u);
    EXPECT_EQ(deletedSamplers.size(), 1u);
    EXPECT_EQ(std::count(deletedBuffers.begin(), deletedBuffers.end(), 9001), 0);
    EXPECT_EQ(std::count(deletedTextures.begin(), deletedTextures.end(), 9002), 0);
    EXPECT_EQ(std::count(deletedSamplers.begin(), deletedSamplers.end(), 9003), 0);
    EXPECT_GT(makeCurrentCount, 0u);
    ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, InvokesDirectComputePipelineThroughRuntimeCoreProvider) {
    dispatchCount = 0;
    storageBindingCount = 0;
    memoryBarrierCount = 0;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(gl, 4 * sizeof(float), alignof(float));
    ASSERT_TRUE(buffer);

    static constexpr char source[] = "#version 430\nlayout(local_size_x=4) in;"
                                     "layout(std430,binding=0) buffer Output{float value[];} outputData;"
                                     "layout(std430,binding=1) readonly buffer Factor{float value;} factor;"
                                     "void main(){outputData.value[gl_LocalInvocationID.x]=factor.value;}";
    static constexpr char reflection[] = R"({
      "gpu_launch_abi_version": 1,
      "entries": [{
        "name": "main",
        "workgroup_size": [4, 1, 1],
        "cpu_arguments_size": 12,
        "arguments": [
          {"kind":"tensor","dtype":"f32","shape":[4],"alignment":4,"cpu_offset":0,"cpu_size":8,"binding":0},
          {"kind":"scalar","dtype":"f32","alignment":4,"cpu_offset":8,"cpu_size":4,"binding":1}
        ]
      }]
    })";
    VernonLoadedPipeline *pipeline =
        vernonRuntimeLoadArtifact(gl, source, sizeof(source) - 1, reflection, sizeof(reflection) - 1, "main", 4);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    const float factor = 2.0F;
    const uint64_t outputShape[]{4};
    const int64_t outputStride[]{sizeof(float)};
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_DEVICE;
    arguments[0].tensor.buffer = buffer;
    arguments[0].tensor.dtype = VERNON_DATA_F32;
    arguments[0].tensor.access = VERNON_ACCESS_READ_WRITE;
    arguments[0].tensor.rank = 1;
    arguments[0].tensor.shape = outputShape;
    arguments[0].tensor.byte_strides = outputStride;
    arguments[0].tensor.byte_size = 4 * sizeof(float);
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_HOST;
    arguments[1].tensor.host_data = &factor;
    arguments[1].tensor.dtype = VERNON_DATA_F32;
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};
    EXPECT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(dispatchCount, 1u);
    EXPECT_EQ(storageBindingCount, 2u);
    EXPECT_EQ(memoryBarrierCount, 1u);

    vernonRuntimeLoadedPipelineDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeBufferFree(buffer), VERNON_STATUS_OK);
    EXPECT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, UploadsColumnMajorMatricesWithoutCopying) {
    constexpr std::array<int64_t, 2> strides = {sizeof(float), 4 * sizeof(float)};
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL, "opengl", 4, 3, columnMajor, strides, 0, columnMajor);
}

TEST(RuntimeExternalGl, UploadsRowMajorMatricesWithDesktopTranspose) {
    constexpr std::array<int64_t, 2> strides = {4 * sizeof(float), sizeof(float)};
    constexpr std::array<float, 16> rowMajor = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL, "opengl", 4, 3, rowMajor, strides, 1, rowMajor);
}

TEST(RuntimeExternalGl, PacksRowMajorMatricesForOpenGlEs) {
    constexpr std::array<int64_t, 2> strides = {4 * sizeof(float), sizeof(float)};
    constexpr std::array<float, 16> rowMajor = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
    };
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL_ES, "opengles", 3, 1, rowMajor, strides, 0, columnMajor);
}

TEST(RuntimeExternalGl, RejectsNonCanonicalInternalParameterContracts) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    auto serialize = [](nlohmann::json root) {
        root.erase("content_hash");
        const std::string canonical = root.dump(-1, ' ', false);
        root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
        return root.dump(-1, ' ', false);
    };

    nlohmann::json legacy = nlohmann::json::parse(internalValueBundle());
    nlohmann::json &legacyUse = legacy["variants"][0]["internal_parameters"][0]["uses"][0];
    legacyUse.erase("sampled_texture_bindings");
    legacyUse["sampled_texture_set"] = 0;
    legacyUse["sampled_texture_binding"] = 3;
    std::string bundleData = serialize(std::move(legacy));
    EXPECT_FALSE(vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr));

    nlohmann::json ambiguous = nlohmann::json::parse(internalValueBundle());
    ambiguous["variants"][0]["internal_parameters"][0]["uses"][0]["sampled_texture_bindings"].push_back(
        {{"set", 0}, {"binding", 4}});
    bundleData = serialize(std::move(ambiguous));
    EXPECT_FALSE(vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr));

    nlohmann::json invalidResolution = nlohmann::json::parse(internalValueBundle());
    invalidResolution["variants"][0]["internal_parameters"][1]["uses"][0]["sampled_texture_bindings"] =
        nlohmann::json::array({{{"set", 0}, {"binding", 3}}});
    bundleData = serialize(std::move(invalidResolution));
    EXPECT_FALSE(vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr));

    ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
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
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterCount(pipeline), 1u);

    VernonDeviceTexture *sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceTexture *target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceSampler *textureSampler = vernonRuntimeImportOpenGLSampler(gl, 19);
    ASSERT_TRUE(sampled && target && textureSampler);
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TEXTURE;
    argument.texture = {sampled,       VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 4, 4, 1,
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
    const GlUint nextNameAfterPreparation = nextName;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(boundSamplerUnit, 3u);
    EXPECT_EQ(boundSamplerName, 19u);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{7.0F, 9.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{2, 3, 7, 9}));

    argument.texture.sampler = nullptr;
    std::fill(std::begin(invocation.viewport), std::end(invocation.viewport), 0);
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{16.0F, 12.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{0, 0, 16, 12}));
    EXPECT_EQ(boundSamplerName, 0u);
    EXPECT_EQ(nextName, nextNameAfterPreparation);

    VernonRuntimeContext *other = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(other);
    VernonDeviceSampler *foreignSampler = vernonRuntimeImportOpenGLSampler(other, 20);
    ASSERT_TRUE(foreignSampler);
    argument.texture.sampler = foreignSampler;
    EXPECT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
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
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterCount(pipeline), 2u);

    VernonDeviceTexture *sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceTexture *target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    VernonDeviceSampler *textureSampler = vernonRuntimeImportOpenGLSampler(gl, 20);
    VernonDeviceSampler *explicitSampler = vernonRuntimeImportOpenGLSampler(gl, 21);
    ASSERT_TRUE(sampled && target && textureSampler && explicitSampler);

    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TEXTURE;
    arguments[0].texture = {sampled,       VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 4, 4, 1,
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
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(boundSamplerName, 21u);

    ASSERT_EQ(vernonRuntimeSamplerFree(explicitSampler), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeSamplerFree(textureSampler), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureFree(target), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeTextureFree(sampled), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, BindsVertexAndIndexBuffersThroughRhi) {
    drawCount = 0;
    indexedDrawCount = 0;
    boundArrayBuffer = 0;
    boundIndexBuffer = 0;
    boundVertexArray = 0;
    vertexAttributeArray = 0;
    vertexAttributeLocation = UINT32_MAX;
    capturedVertexAttributeDivisor = 0;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string bundleData = vertexInputBundle();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

    VernonDeviceBuffer *vertices = vernonRuntimeImportOpenGLBuffer(gl, 30, 36, alignof(float));
    VernonDeviceBuffer *indices = vernonRuntimeImportOpenGLBuffer(gl, 31, 12, alignof(uint32_t));
    VernonDeviceTexture *target = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_TRUE(vertices && indices && target);
    constexpr uint64_t shape[2] = {3, 3};
    constexpr int64_t strides[2] = {3 * sizeof(float), sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_DEVICE;
    argument.tensor.buffer = vertices;
    argument.tensor.dtype = VERNON_DATA_F32;
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = 2;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 36;
    const VernonIndexBinding indexBinding{indices, VERNON_INDEX_U32, 0, 3};
    const VernonColorAttachment attachment{0, target};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.index_binding = &indexBinding;
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.vertex_count = 3;
    invocation.instance_count = 3;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    EXPECT_EQ(boundArrayBuffer, 30u);
    EXPECT_EQ(boundIndexBuffer, 31u);
    EXPECT_NE(vertexAttributeArray, 0u);
    EXPECT_EQ(vertexAttributeArray, boundVertexArray);
    EXPECT_EQ(vertexAttributeLocation, 0u);
    EXPECT_EQ(capturedVertexAttributeDivisor, 1u);
    EXPECT_EQ(vertexAttributeComponents, 3);
    EXPECT_EQ(vertexAttributeStride, 12);
    EXPECT_EQ(vertexAttributeOffset, 0u);
    EXPECT_EQ(indexedDrawCount, 3);
    EXPECT_EQ(drawCount, 1u);

    ASSERT_EQ(vernonRuntimeTextureFree(target), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeBufferFree(indices), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeBufferFree(vertices), VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(vernonRuntimeDestroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, LoadsAssetsAndInvokesPipeline) {
    drawCount = 0;
    VernonRuntimeCapabilities global = vernonRuntimeGetCapabilities(VERNON_RUNTIME_OPENGL_ES);
    ASSERT_TRUE(!global.available);
    ASSERT_TRUE(global.supports_graphics);

    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetContextCapabilities(gl);
    ASSERT_TRUE(capabilities.available && capabilities.supports_graphics);
    ASSERT_TRUE(capabilities.supports_compute && capabilities.supports_storage_buffers);
    const std::string glBundle = schema2Bundle(inlineArtifact("#version 330\nvoid main(){}"));
    VernonPipelineBundle *glLoaded =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, glBundle.data(), glBundle.size(), nullptr);
    ASSERT_TRUE(glLoaded);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(glLoaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    VernonDeviceTexture *target = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_TRUE(target);
    VernonTextureDescriptor cubeDescriptor{};
    cubeDescriptor.struct_size = sizeof(cubeDescriptor);
    cubeDescriptor.dimension = VERNON_TEXTURE_CUBE;
    cubeDescriptor.format = VERNON_TEXTURE_RGBA16_FLOAT;
    cubeDescriptor.width = 32;
    cubeDescriptor.height = 32;
    cubeDescriptor.depth = 1;
    cubeDescriptor.mip_levels = 6;
    VernonDeviceTexture *cube = vernonRuntimeImportOpenGLTexture(gl, 8, &cubeDescriptor);
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
    const GlUint nextNameAfterPreparation = nextName;
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(drawCount == 2);
    ASSERT_TRUE(nextName == nextNameAfterPreparation);
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
    ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(inlineBundle.data(), inlineBundle.size(), &schema2Target) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(schema2Target == VERNON_RUNTIME_OPENGL);
    glLoaded = vernonRuntimeLoadPipelineBundleWithOptions(gl, inlineBundle.data(), inlineBundle.size(), nullptr);
    ASSERT_TRUE(glLoaded);
    vernonRuntimePipelineBundleDestroy(glLoaded);

    std::string corruptManifest = inlineBundle;
    const size_t contentHash = corruptManifest.find("\"content_hash\":\"");
    ASSERT_TRUE(contentHash != std::string::npos);
    corruptManifest[contentHash + std::strlen("\"content_hash\":\"")] ^= 1;
    ASSERT_TRUE(
        !vernonRuntimeLoadPipelineBundleWithOptions(gl, corruptManifest.data(), corruptManifest.size(), nullptr));

    nlohmann::json corruptInline = inlineArtifact(source);
    corruptInline["sha256"] = std::string(64, '0');
    const std::string corruptInlineBundle = schema2Bundle(corruptInline);
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(gl, corruptInlineBundle.data(), corruptInlineBundle.size(),
                                                            nullptr));
    nlohmann::json wrongFormat = inlineArtifact(source);
    wrongFormat["format"] = "ptx";
    const std::string wrongFormatBundle = schema2Bundle(wrongFormat);
    ASSERT_TRUE(
        !vernonRuntimeLoadPipelineBundleWithOptions(gl, wrongFormatBundle.data(), wrongFormatBundle.size(), nullptr));

    const std::filesystem::path assetRoot = std::filesystem::temp_directory_path() / "vernon_runtime_schema2_test";
    std::filesystem::remove_all(assetRoot);
    std::filesystem::create_directories(assetRoot / "artifacts");
    const std::filesystem::path artifactPath = assetRoot / "artifacts/stage.glsl";
    {
        std::ofstream output(artifactPath, std::ios::binary);
        output.write(source.data(), static_cast<std::streamsize>(source.size()));
    }
    const nlohmann::json externalArtifact = {{"format", "glsl"},
                                             {"storage", "external"},
                                             {"path", "artifacts/stage.glsl"},
                                             {"size", source.size()},
                                             {"sha256", vernon::runtime::sha256Hex(source.data(), source.size())}};
    const std::string externalBundle = schema2Bundle(externalArtifact);
    const std::string assetRootUtf8 = assetRoot.u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = assetRootUtf8.c_str();
    glLoaded = vernonRuntimeLoadPipelineBundleWithOptions(gl, externalBundle.data(), externalBundle.size(), &options);
    ASSERT_TRUE(glLoaded);
    vernonRuntimePipelineBundleDestroy(glLoaded);

    nlohmann::json traversalArtifact = externalArtifact;
    traversalArtifact["path"] = "../stage.glsl";
    const std::string traversalBundle = schema2Bundle(traversalArtifact);
    ASSERT_TRUE(
        !vernonRuntimeLoadPipelineBundleWithOptions(gl, traversalBundle.data(), traversalBundle.size(), &options));
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(gl, externalBundle.data(), externalBundle.size(), nullptr));

    const std::filesystem::path outsidePath = assetRoot.parent_path() / "vernon_runtime_schema2_escape.glsl";
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
        ASSERT_TRUE(
            !vernonRuntimeLoadPipelineBundleWithOptions(gl, symlinkBundle.data(), symlinkBundle.size(), &options));
    }
    std::filesystem::remove(outsidePath);

    {
        std::ofstream output(artifactPath, std::ios::binary | std::ios::trunc);
        output << "corrupt";
    }
    ASSERT_TRUE(
        !vernonRuntimeLoadPipelineBundleWithOptions(gl, externalBundle.data(), externalBundle.size(), &options));
    std::filesystem::remove_all(assetRoot);
    ASSERT_TRUE(vernonRuntimeDestroy(gl) == VERNON_STATUS_OK);

    VernonRuntimeContext *gles = create(VERNON_RUNTIME_OPENGL_ES, 3, 1);
    ASSERT_TRUE(gles);
    capabilities = vernonRuntimeGetContextCapabilities(gles);
    ASSERT_TRUE(capabilities.supports_compute);
    const std::string bundle = matrixBundle("opengles");
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(gles, bundle.data(), bundle.size(), nullptr);
    ASSERT_TRUE(loaded);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(gles) == VERNON_STATUS_OK);
}

#undef GL_CALL
