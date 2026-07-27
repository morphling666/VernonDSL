#include "VernonRuntime.h"
#include "runtime/content_hash.h"
#include "runtime_rhi_test_utils.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
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
constexpr GlEnum kInt = 0x1404;
constexpr GlEnum kUnsignedInt = 0x1405;
constexpr GlEnum kFloat = 0x1406;
constexpr GlEnum kDouble = 0x140A;
constexpr GlEnum kHalfFloat = 0x140B;

GlUint nextName = 1;
uint32_t drawCount = 0;
uint32_t dispatchCount = 0;
uint32_t storageBindingCount = 0;
uint32_t memoryBarrierCount = 0;
uint32_t clearCount = 0;
uint32_t invalidateCount = 0;
uint32_t programBindCount = 0;
uint32_t framebufferBindCount = 0;
uint32_t viewportCount = 0;
uint32_t scissorCount = 0;
GlInt clearedDrawBuffer = -1;
std::array<float, 4> clearColor{};
GlBoolean matrixTranspose = 1;
std::array<float, 16> matrixUpload{};
std::array<float, 2> resolutionUpload{};
std::array<GlInt, 4> viewportUpload{};
std::array<GlInt, 4> scissorUpload{};

using vernon::tests::RhiImage;
using vernon::tests::RhiRuntime;

std::unordered_map<VernonRuntimeContext *, RhiRuntime> rhiRuntimes;

VernonRhiFormat rhiFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_R8_UNORM:
        return VERNON_RHI_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_RG8_UNORM:
        return VERNON_RHI_FORMAT_RG8_UNORM;
    case VERNON_TEXTURE_RGB8_UNORM:
        return VERNON_RHI_FORMAT_RGB8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return VERNON_RHI_FORMAT_RGBA8_SRGB;
    default:
        return VERNON_RHI_FORMAT_RGBA8_UNORM;
    }
}

RhiRuntime &rhiRuntime(VernonRuntimeContext *runtime) { return rhiRuntimes.at(runtime); }

RhiImage createTexture2D(VernonRuntimeContext *runtime, uint32_t width, uint32_t height, VernonTextureFormat format) {
    return vernon::tests::createImage(rhiRuntime(runtime), VERNON_RHI_IMAGE_2D, rhiFormat(format), width, height, 1,
                                      VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
}

RhiImage importOpenGLTexture2D(VernonRuntimeContext *runtime, uint32_t, uint32_t width, uint32_t height,
                               VernonTextureFormat format) {
    return createTexture2D(runtime, width, height, format);
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
GlEnum vertexAttributeType = 0;
char vertexAttributePointerKind = '\0';
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
void GL_CALL getIntegerv(GlEnum, GlInt *value) { *value = 16; }
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
void GL_CALL useProgram(GlUint) { ++programBindCount; }
void GL_CALL bindFramebuffer(GlEnum, GlUint) { ++framebufferBindCount; }
void GL_CALL bindBufferBase(GlEnum, GlUint, GlUint) { ++storageBindingCount; }
void GL_CALL dispatchCompute(GlUint, GlUint, GlUint) { ++dispatchCount; }
void GL_CALL memoryBarrier(unsigned) { ++memoryBarrierCount; }
void GL_CALL vertexAttribPointer(GlUint location, GlInt components, GlEnum type, GlBoolean, GlSize stride,
                                 const void *offset) {
    vertexAttributeArray = boundVertexArray;
    vertexAttributeLocation = location;
    vertexAttributeComponents = components;
    vertexAttributeType = type;
    vertexAttributePointerKind = 'F';
    vertexAttributeStride = stride;
    vertexAttributeOffset = reinterpret_cast<uintptr_t>(offset);
}
void GL_CALL vertexAttribIPointer(GlUint location, GlInt components, GlEnum type, GlSize stride, const void *offset) {
    vertexAttribPointer(location, components, type, 0, stride, offset);
    vertexAttributePointerKind = 'I';
}
void GL_CALL vertexAttribLPointer(GlUint location, GlInt components, GlEnum type, GlSize stride, const void *offset) {
    vertexAttribPointer(location, components, type, 0, stride, offset);
    vertexAttributePointerKind = 'L';
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
void GL_CALL invalidateFramebuffer(GlEnum, GlSize count, const GlEnum *) {
    invalidateCount += static_cast<uint32_t>(count);
}
void GL_CALL scissor(GlInt x, GlInt y, GlSize width, GlSize height) {
    ++scissorCount;
    scissorUpload = {x, y, width, height};
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
void GL_CALL viewport(GlInt x, GlInt y, GlSize width, GlSize height) {
    ++viewportCount;
    viewportUpload = {x, y, width, height};
}

void *getProcAddress(void *, const char *name) {
#define PROC(glName, function)                                                                                         \
    if (std::strcmp(name, glName) == 0)                                                                                \
    return reinterpret_cast<void *>(&function)
    PROC("glCreateShader", createName);
    PROC("glShaderSource", shaderSource);
    PROC("glGetShaderiv", getShaderiv);
    PROC("glCreateProgram", createProgram);
    PROC("glGetProgramiv", getProgramiv);
    PROC("glGetIntegerv", getIntegerv);
    PROC("glGenVertexArrays", genNames);
    PROC("glBindVertexArray", bindVertexArray);
    PROC("glUseProgram", useProgram);
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
    PROC("glBindFramebuffer", bindFramebuffer);
    PROC("glDrawArrays", drawArrays);
    PROC("glVertexAttribPointer", vertexAttribPointer);
    PROC("glVertexAttribIPointer", vertexAttribIPointer);
    PROC("glVertexAttribLPointer", vertexAttribLPointer);
    PROC("glVertexAttribDivisor", vertexAttribDivisor);
    PROC("glDrawElementsInstanced", drawElementsInstanced);
    PROC("glClearBufferfv", clearBufferfv);
    PROC("glInvalidateFramebuffer", invalidateFramebuffer);
    PROC("glGetUniformLocation", getUniformLocation);
    PROC("glUniform2fv", uniform2fv);
    PROC("glUniformMatrix4fv", uniformMatrix4fv);
    PROC("glBindSampler", bindSampler);
    PROC("glViewport", viewport);
    PROC("glScissor", scissor);
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
    RhiRuntime context = vernon::tests::createRhiRuntime(backend, &callbacks);
    VernonRuntimeContext *runtime = context.runtime;
    if (runtime)
        rhiRuntimes.emplace(runtime, context);
    return runtime;
}

VernonStatus destroy(VernonRuntimeContext *runtime) {
    auto found = rhiRuntimes.find(runtime);
    if (found == rhiRuntimes.end())
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus status = vernonRuntimeDestroy(runtime);
    if (status == VERNON_STATUS_OK) {
        vernonRhiDestroyDevice(found->second.device);
        rhiRuntimes.erase(found);
    }
    return status;
}

nlohmann::json inlineArtifact(const std::string &source);

nlohmann::json scalarElementLayout(const std::string &dtype) {
    const VernonDataType dataType = dtype == "i32"    ? VERNON_DATA_I32
                                    : dtype == "u32"  ? VERNON_DATA_U32
                                    : dtype == "f16"  ? VERNON_DATA_F16
                                    : dtype == "f64"  ? VERNON_DATA_F64
                                    : dtype == "bool" ? VERNON_DATA_BOOL
                                                      : VERNON_DATA_F32;
    const VernonValueLayoutView layout = vernonRuntimeGetScalarValueLayout(dataType);
    return {{"logical_type", dtype},
            {"byte_size", layout.byte_size},
            {"alignment", layout.alignment},
            {"layout_hash", std::string(layout.layout_hash.data, layout.layout_hash.size)},
            {"leaves",
             nlohmann::json::array(
                 {{{"path", nlohmann::json::array()}, {"dtype", dtype}, {"byte_offset", 0}, {"scalar_count", 1}}})}};
}

std::string pipelineBundle(const nlohmann::json &artifact) {
    nlohmann::json root = {
        {"schema_version", 4},
        {"type", "pipeline"},
        {"id", "pipeline/gl"},
        {"target", "opengl"},
        {"invocation_abi_version", VERNON_PIPELINE_INVOCATION_ABI_VERSION},
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
    nlohmann::json root = nlohmann::json::parse(pipelineBundle(inlineArtifact(source)));
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
                                {"element_layout", scalarElementLayout("f32")},
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

std::string vertexInputBundle(const char *dtype = "f32", uint32_t components = 3, uint32_t divisor = 1) {
    nlohmann::json root =
        nlohmann::json::parse(pipelineBundle(inlineArtifact("#version 330\nvoid main(){gl_Position=vec4(0.0);}")));
    root["variants"][0]["parameters"] = nlohmann::json::array(
        {{{"slot", 0},
          {"name", "position"},
          {"kind", "tensor"},
          {"element_layout", scalarElementLayout(dtype)},
          {"access", "read"},
          {"shape", nlohmann::json::array({components})},
          {"uses", nlohmann::json::array({{{"stage", "vertex"},
                                           {"interface", "input"},
                                           {"vernon.location", 0},
                                           {"vernon.instance_divisor", divisor},
                                           {"dtype", dtype},
                                           {"shape", nlohmann::json::array({components})},
                                           {"attribute_leaves", nlohmann::json::array({{{"location_offset", 0},
                                                                                        {"dtype", dtype},
                                                                                        {"component_count", components},
                                                                                        {"byte_offset", 0}}})}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string internalValueBundle() {
    nlohmann::json root = nlohmann::json::parse(pipelineBundle(inlineArtifact("#version 330\nvoid main(){}")));
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
          {"element_layout", scalarElementLayout("f32")},
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
    RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr std::array<uint64_t, 2> shape = {4, 4};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = storage.data();
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = 2;
    argument.tensor.shape = shape.data();
    argument.tensor.byte_strides = strides.data();
    argument.tensor.byte_size = sizeof(storage);
    VernonColorAttachment attachment{0, renderTarget.reference, 16, 16, VERNON_TEXTURE_RGBA8_UNORM};
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

    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

nlohmann::json inlineArtifact(const std::string &source) {
    return {{"format", "glsl"},      {"storage", "inline"},
            {"encoding", "utf8"},    {"data", source},
            {"size", source.size()}, {"sha256", vernon::runtime::sha256Hex(source.data(), source.size())}};
}

} // namespace

TEST(RuntimeExternalGl, InvokesDirectComputePipelineThroughRuntimeCoreProvider) {
    dispatchCount = 0;
    storageBindingCount = 0;
    memoryBarrierCount = 0;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    auto buffer =
        vernon::tests::createBuffer(rhiRuntime(gl), 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

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
          {"kind":"tensor","dtype":"f32","shape":[4],"element_layout":{"logical_type":"f32","byte_size":4,
           "alignment":4,"layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
           "alignment":4,"cpu_offset":0,"cpu_size":8,"binding":0},
          {"kind":"scalar","dtype":"f32","element_layout":{"logical_type":"f32","byte_size":4,
           "alignment":4,"layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
           "alignment":4,"cpu_offset":8,"cpu_size":4,"binding":1}
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
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = buffer.reference;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
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
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
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
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, buffer.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, CommandEncoderSuppressesRedundantDynamicState) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    VernonRhiDevice device = rhiRuntime(gl).device;
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &encoderDescriptor, &encoder), VERNON_RHI_STATUS_OK);

    VernonRhiColorAttachment color{};
    color.view = {0, 1};
    color.load_operation = VERNON_RHI_LOAD_CLEAR;
    color.store_operation = VERNON_RHI_STORE_PRESERVE;
    VernonRhiRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &color;
    rendering.color_attachment_count = 1;
    rendering.width = 16;
    rendering.height = 16;
    rendering.layers = 1;
    ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, encoder, &rendering), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderBindGraphicsPipeline(device, encoder, {0, 1}), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderBindGraphicsPipeline(device, encoder, {0, 1}), VERNON_RHI_STATUS_OK);
    const VernonRhiViewport viewport{0.0f, 0.0f, 16.0f, 16.0f, 0.0f, 1.0f};
    ASSERT_EQ(vernonRhiCommandEncoderSetViewport(device, encoder, &viewport), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderSetViewport(device, encoder, &viewport), VERNON_RHI_STATUS_OK);
    const float clearColor[4]{0.0f, 0.0f, 0.0f, 1.0f};
    ASSERT_EQ(vernonRhiCommandEncoderClearColorAttachment(device, encoder, 0, clearColor), VERNON_RHI_STATUS_OK);
    const VernonRhiDrawDescriptor draw{3, 1, 0, 0};
    ASSERT_EQ(vernonRhiCommandEncoderDraw(device, encoder, &draw), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);

    VernonRhiCommandEncoderStats stats{};
    ASSERT_EQ(vernonRhiCommandEncoderGetStats(device, encoder, &stats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(stats.rendering_scope_count, 1u);
    EXPECT_EQ(stats.graphics_pipeline_bind_count, 1u);
    EXPECT_EQ(stats.viewport_change_count, 1u);
    EXPECT_EQ(stats.clear_count, 1u);
    EXPECT_EQ(stats.draw_count, 1u);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceSubmit(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
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

    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
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

    RhiImage sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    auto textureSampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(textureSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TEXTURE;
    argument.texture = {VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ,      VERNON_TEXTURE_2D, 4, 4, 1,
                        sampled.reference,          textureSampler.reference};
    VernonColorAttachment attachment{0, target.reference, 16, 12, VERNON_TEXTURE_RGBA8_UNORM};
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
    EXPECT_NE(boundSamplerName, 0u);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{7.0F, 9.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{2, 3, 7, 9}));

    argument.texture.sampler_resource = {};
    std::fill(std::begin(invocation.viewport), std::end(invocation.viewport), 0);
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{16.0F, 12.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{0, 0, 16, 12}));
    EXPECT_EQ(boundSamplerName, 0u);
    EXPECT_EQ(nextName, nextNameAfterPreparation);

    VernonRuntimeContext *other = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(other);
    auto foreignSampler = vernon::tests::createSampler(rhiRuntime(other));
    ASSERT_NE(foreignSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    argument.texture.sampler_resource = foreignSampler.reference;
    EXPECT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(other).device, foreignSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(destroy(other), VERNON_STATUS_OK);

    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, textureSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
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

    RhiImage sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    auto textureSampler = vernon::tests::createSampler(rhiRuntime(gl));
    auto explicitSampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(textureSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(explicitSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TEXTURE;
    arguments[0].texture = {VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ,      VERNON_TEXTURE_2D, 4, 4, 1,
                            sampled.reference,          textureSampler.reference};
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_SAMPLER;
    arguments[1].resource = explicitSampler.reference;
    VernonColorAttachment attachment{0, target.reference, 16, 12, VERNON_TEXTURE_RGBA8_UNORM};
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
    EXPECT_NE(boundSamplerName, 0u);

    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, explicitSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, textureSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
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

    auto vertices = vernon::tests::createBuffer(rhiRuntime(gl), 36, alignof(float), VERNON_RHI_BUFFER_VERTEX);
    auto indices = vernon::tests::createBuffer(rhiRuntime(gl), 12, alignof(uint32_t), VERNON_RHI_BUFFER_INDEX);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(indices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[2] = {3, 3};
    constexpr int64_t strides[2] = {3 * sizeof(float), sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = vertices.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = 2;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 36;
    const VernonIndexBinding indexBinding{VERNON_INDEX_U32, 0, 3, indices.reference};
    const VernonColorAttachment attachment{0, target.reference, 16, 16, VERNON_TEXTURE_RGBA8_UNORM};
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
    void *vertexNative = nullptr;
    void *indexNative = nullptr;
    ASSERT_EQ(vernonRhiDeviceGetBufferNativeHandle(rhiRuntime(gl).device, vertices.handle, &vertexNative),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceGetBufferNativeHandle(rhiRuntime(gl).device, indices.handle, &indexNative),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(boundArrayBuffer, reinterpret_cast<uintptr_t>(vertexNative));
    EXPECT_EQ(boundIndexBuffer, reinterpret_cast<uintptr_t>(indexNative));
    EXPECT_NE(vertexAttributeArray, 0u);
    EXPECT_EQ(vertexAttributeArray, boundVertexArray);
    EXPECT_EQ(vertexAttributeLocation, 0u);
    EXPECT_EQ(capturedVertexAttributeDivisor, 1u);
    EXPECT_EQ(vertexAttributeComponents, 3);
    EXPECT_EQ(vertexAttributeStride, 12);
    EXPECT_EQ(vertexAttributeOffset, 0u);
    EXPECT_EQ(indexedDrawCount, 3);
    EXPECT_EQ(drawCount, 1u);

    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, indices.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, BindsAllFormalVertexNumericFormatsAndRejectsUnsupportedFormats) {
    struct Format {
        const char *name;
        VernonDataType dtype;
        uint32_t size;
        GlEnum nativeType;
        char pointerKind;
    };
    constexpr Format formats[] = {{"i32", VERNON_DATA_I32, 4, kInt, 'I'},
                                  {"u32", VERNON_DATA_U32, 4, kUnsignedInt, 'I'},
                                  {"f16", VERNON_DATA_F16, 2, kHalfFloat, 'F'},
                                  {"f32", VERNON_DATA_F32, 4, kFloat, 'F'},
                                  {"f64", VERNON_DATA_F64, 8, kDouble, 'L'}};
    for (const Format &format : formats) {
        SCOPED_TRACE(format.name);
        vertexAttributeType = 0;
        vertexAttributePointerKind = '\0';
        VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
        ASSERT_TRUE(gl);
        const std::string bundleData = vertexInputBundle(format.name);
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
        ASSERT_TRUE(bundle);
        VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
        ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

        const uint32_t stride = 3 * format.size;
        auto vertices = vernon::tests::createBuffer(rhiRuntime(gl), 3 * stride, format.size, VERNON_RHI_BUFFER_VERTEX);
        RhiImage target = importOpenGLTexture2D(gl, 8, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
        ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        const uint64_t shape[2] = {3, 3};
        const int64_t strides[2] = {stride, format.size};
        VernonPipelineArgument argument{};
        argument.kind = VERNON_PIPELINE_TENSOR;
        argument.tensor.struct_size = sizeof(VernonTensorView);
        argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        argument.tensor.resource = vertices.reference;
        argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(format.dtype);
        argument.tensor.access = VERNON_ACCESS_READ;
        argument.tensor.rank = 2;
        argument.tensor.shape = shape;
        argument.tensor.byte_strides = strides;
        argument.tensor.byte_size = 3 * stride;
        const VernonColorAttachment attachment{0, target.reference, 16, 16, VERNON_TEXTURE_RGBA8_UNORM};
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
        invocation.arguments = &argument;
        invocation.argument_count = 1;
        invocation.color_attachments = &attachment;
        invocation.color_attachment_count = 1;
        invocation.vertex_count = 3;
        invocation.instance_count = 3;
        ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
            << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
        EXPECT_EQ(vertexAttributeType, format.nativeType);
        EXPECT_EQ(vertexAttributePointerKind, format.pointerKind);
        EXPECT_EQ(vertexAttributeStride, static_cast<GlSize>(stride));

        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
        vernonRuntimeLoadedPipelineDestroy(pipeline);
        vernonRuntimePipelineBundleDestroy(bundle);
        ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
    }

    for (const char *dtype : {"bool", "u8"}) {
        SCOPED_TRACE(dtype);
        VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
        ASSERT_TRUE(gl);
        const std::string bundleData = vertexInputBundle(dtype);
        VernonPipelineBundle *bundle =
            vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
        ASSERT_TRUE(bundle);
        EXPECT_EQ(vernonRuntimeResolvePipeline(bundle, {nullptr, 0}), nullptr);
        const VernonStringView error = vernonRuntimeGetLastError(gl);
        EXPECT_NE(std::string_view(error.data, error.size).find("format capabilities"), std::string_view::npos);
        vernonRuntimePipelineBundleDestroy(bundle);
        ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
    }
}

TEST(RuntimeExternalGl, LoadsAssetsAndInvokesPipeline) {
    drawCount = 0;
    clearCount = 0;
    invalidateCount = 0;
    programBindCount = 0;
    framebufferBindCount = 0;
    viewportCount = 0;
    scissorCount = 0;
    VernonRuntimeCapabilities global = vernonRuntimeGetCapabilities(VERNON_RUNTIME_OPENGL_ES);
    ASSERT_TRUE(!global.available);
    ASSERT_TRUE(global.supports_graphics);

    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetContextCapabilities(gl);
    ASSERT_TRUE(capabilities.available && capabilities.supports_graphics);
    ASSERT_TRUE(capabilities.supports_compute && capabilities.supports_storage_buffers);
    const std::string glBundle = pipelineBundle(inlineArtifact("#version 330\nvoid main(){}"));
    VernonPipelineBundle *glLoaded =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, glBundle.data(), glBundle.size(), nullptr);
    ASSERT_TRUE(glLoaded);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(glLoaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    RhiImage cube = vernon::tests::createImage(rhiRuntime(gl), VERNON_RHI_IMAGE_CUBE, VERNON_RHI_FORMAT_RGBA16_FLOAT,
                                               32, 32, 1, VERNON_RHI_IMAGE_SAMPLED, 6);
    ASSERT_NE(cube.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    auto sampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonColorAttachment attachment{0, target.reference, 16, 16, VERNON_TEXTURE_RGBA8_UNORM};
    attachment.clear_color[0] = 0.25f;
    attachment.clear_color[1] = 0.5f;
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
    attachment.load_operation = VERNON_RHI_LOAD_PRESERVE;
    attachment.store_operation = VERNON_RHI_STORE_DISCARD;
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(drawCount == 2);
    ASSERT_EQ(clearCount, 1u);
    EXPECT_FLOAT_EQ(clearColor[0], 0.25f);
    EXPECT_FLOAT_EQ(clearColor[1], 0.5f);
    EXPECT_EQ(invalidateCount, 1u);
    EXPECT_EQ(scissorUpload, (std::array<GlInt, 4>{0, 0, 16, 16}));
    EXPECT_EQ(programBindCount, 1u);
    EXPECT_EQ(framebufferBindCount, 1u);
    EXPECT_EQ(viewportCount, 1u);
    EXPECT_EQ(scissorCount, 1u);
    ASSERT_TRUE(nextName == nextNameAfterPreparation);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, cube.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, sampler.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(glLoaded);
    ASSERT_TRUE(destroy(gl) == VERNON_STATUS_OK);

    gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string source = "#version 330\nvoid main(){}";
    const std::string inlineBundle = pipelineBundle(inlineArtifact(source));
    VernonRuntimeBackend inspectedTarget = VERNON_RUNTIME_CPU;
    ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(inlineBundle.data(), inlineBundle.size(), &inspectedTarget) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(inspectedTarget == VERNON_RUNTIME_OPENGL);
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
    const std::string corruptInlineBundle = pipelineBundle(corruptInline);
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(gl, corruptInlineBundle.data(), corruptInlineBundle.size(),
                                                            nullptr));
    nlohmann::json wrongFormat = inlineArtifact(source);
    wrongFormat["format"] = "ptx";
    const std::string wrongFormatBundle = pipelineBundle(wrongFormat);
    ASSERT_TRUE(
        !vernonRuntimeLoadPipelineBundleWithOptions(gl, wrongFormatBundle.data(), wrongFormatBundle.size(), nullptr));

    const std::filesystem::path assetRoot = std::filesystem::temp_directory_path() / "vernon_runtime_pipeline_test";
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
    const std::string externalBundle = pipelineBundle(externalArtifact);
    const std::string assetRootUtf8 = assetRoot.u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = assetRootUtf8.c_str();
    glLoaded = vernonRuntimeLoadPipelineBundleWithOptions(gl, externalBundle.data(), externalBundle.size(), &options);
    ASSERT_TRUE(glLoaded);
    vernonRuntimePipelineBundleDestroy(glLoaded);

    nlohmann::json traversalArtifact = externalArtifact;
    traversalArtifact["path"] = "../stage.glsl";
    const std::string traversalBundle = pipelineBundle(traversalArtifact);
    ASSERT_TRUE(
        !vernonRuntimeLoadPipelineBundleWithOptions(gl, traversalBundle.data(), traversalBundle.size(), &options));
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(gl, externalBundle.data(), externalBundle.size(), nullptr));

    const std::filesystem::path outsidePath = assetRoot.parent_path() / "vernon_runtime_pipeline_escape.glsl";
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
        const std::string symlinkBundle = pipelineBundle(symlinkArtifact);
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
    ASSERT_TRUE(destroy(gl) == VERNON_STATUS_OK);

    VernonRuntimeContext *gles = create(VERNON_RUNTIME_OPENGL_ES, 3, 1);
    ASSERT_TRUE(gles);
    capabilities = vernonRuntimeGetContextCapabilities(gles);
    ASSERT_TRUE(capabilities.supports_compute);
    const std::string bundle = matrixBundle("opengles");
    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(gles, bundle.data(), bundle.size(), nullptr);
    ASSERT_TRUE(loaded);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(destroy(gles) == VERNON_STATUS_OK);
}

#undef GL_CALL
