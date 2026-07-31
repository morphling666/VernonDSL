#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"
#include "runtime/content_hash.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime_rhi_test_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
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
uint32_t lastMemoryBarrierBits = 0;
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

class RuntimeGraphRenderPass final : public vernon::execution::RenderPass {
public:
    RuntimeGraphRenderPass(std::string name, vernon::execution::GraphImage target, VernonRuntimeContext *runtime,
                           VernonLoadedPipeline *pipeline, const VernonPipelineInvocation *invocation,
                           VernonRhiLoadOperation load, VernonRhiStoreOperation store = VERNON_RHI_STORE_PRESERVE)
        : RenderPass(std::move(name)), target_(target), runtime_(runtime), pipeline_(pipeline), invocation_(invocation),
          load_(load), store_(store) {}

    void declare() override {
        vernon::execution::ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = load_;
        attachment.store = store_;
        color(0, attachment);
        renderArea(0, 0, target_.width, target_.height);
    }

    VernonRhiStatus execute(vernon::execution::GraphicsEncoder &encoder,
                            const vernon::execution::ExecutionResources &) override {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime_, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        return vernonRuntimePipelineEncode(providerEncoder, pipeline_, invocation_) == VERNON_STATUS_OK
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    vernon::execution::GraphImage target_;
    VernonRuntimeContext *runtime_{};
    VernonLoadedPipeline *pipeline_{};
    const VernonPipelineInvocation *invocation_{};
    VernonRhiLoadOperation load_{};
    VernonRhiStoreOperation store_{};
};

class GraphImageWritePass final : public vernon::execution::ComputePass {
public:
    GraphImageWritePass(std::string name, vernon::execution::GraphImage image)
        : ComputePass(std::move(name)), image_(image) {}

    void declare() override { write(image_, VERNON_RHI_STATE_SHADER_WRITE); }

    VernonRhiStatus execute(vernon::execution::ComputeEncoder &,
                            const vernon::execution::ExecutionResources &) override {
        return VERNON_RHI_STATUS_OK;
    }

private:
    vernon::execution::GraphImage image_;
};

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
std::vector<GlEnum> shaderKinds;
std::vector<unsigned char> bufferStorage;
std::vector<unsigned char> textureStorage;
std::array<GlInt, 4> signedUniformUpload{};
std::array<GlUint, 4> unsignedUniformUpload{};

void makeCurrent(void *) { ++makeCurrentCount; }
void GL_CALL placeholder() {}
GlUint GL_CALL createName(GlEnum kind) {
    shaderKinds.push_back(kind);
    return nextName++;
}
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
void GL_CALL memoryBarrier(unsigned bits) {
    ++memoryBarrierCount;
    lastMemoryBarrierBits = bits;
}
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
void GL_CALL uniform1iv(GlInt, GlSize, const GlInt *data) { signedUniformUpload[0] = data[0]; }
void GL_CALL uniform2iv(GlInt, GlSize, const GlInt *data) { std::copy_n(data, 2, signedUniformUpload.begin()); }
void GL_CALL uniform3iv(GlInt, GlSize, const GlInt *data) { std::copy_n(data, 3, signedUniformUpload.begin()); }
void GL_CALL uniform4iv(GlInt, GlSize, const GlInt *data) { std::copy_n(data, 4, signedUniformUpload.begin()); }
void GL_CALL uniform1uiv(GlInt, GlSize, const GlUint *data) { unsignedUniformUpload[0] = data[0]; }
void GL_CALL uniform2uiv(GlInt, GlSize, const GlUint *data) { std::copy_n(data, 2, unsignedUniformUpload.begin()); }
void GL_CALL uniform3uiv(GlInt, GlSize, const GlUint *data) { std::copy_n(data, 3, unsignedUniformUpload.begin()); }
void GL_CALL uniform4uiv(GlInt, GlSize, const GlUint *data) { std::copy_n(data, 4, unsignedUniformUpload.begin()); }
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
    PROC("glUniform1iv", uniform1iv);
    PROC("glUniform2iv", uniform2iv);
    PROC("glUniform3iv", uniform3iv);
    PROC("glUniform4iv", uniform4iv);
    PROC("glUniform1uiv", uniform1uiv);
    PROC("glUniform2uiv", uniform2uiv);
    PROC("glUniform3uiv", uniform3uiv);
    PROC("glUniform4uiv", uniform4uiv);
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

nlohmann::json physicalValueLayout(uint64_t size, uint64_t alignment, std::initializer_list<uint64_t> byteStrides) {
    return {{"profile", "opengl_native_uniform"},
            {"transport", "native_uniform"},
            {"size", size},
            {"alignment", alignment},
            {"byte_strides", byteStrides}};
}

std::string pipelineBundle(const nlohmann::json &artifact) {
    nlohmann::json root = {
        {"pipeline_version", VERNON_PIPELINE_VERSION},
        {"type", "pipeline"},
        {"id", "pipeline/gl"},
        {"target", "opengl"},
        {"features", nlohmann::json::array()},
        {"runtime_requirements",
         {{"backend", "opengl"},
          {"features", nlohmann::json::array({"textures"})},
          {"glsl_version", 330},
          {"profile", "core"},
          {"api_version", nlohmann::json::array({3, 3})}}},
        {"variants", nlohmann::json::array({{{"key", nlohmann::json::array()},
                                             {"parameters", nlohmann::json::array()},
                                             {"outputs", nlohmann::json::array()},
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
    root["variants"][0]["parameters"] = nlohmann::json::array(
        {{{"slot", 0},
          {"name", "transform"},
          {"kind", "tensor"},
          {"type", "tensor<4x4xf32>"},
          {"element_layout", scalarElementLayout("f32")},
          {"access", "read"},
          {"shape", nlohmann::json::array({4, 4})},
          {"uses", nlohmann::json::array({{{"stage", "vertex"},
                                           {"interface", "uniform"},
                                           {"uniform_name", "transform"},
                                           {"dtype", "f32"},
                                           {"shape", nlohmann::json::array({4, 4})},
                                           {"physical_value_layout", physicalValueLayout(64, 16, {4, 16})}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string integerUniformBundle(const char *dtype, uint32_t components) {
    const std::string scalarType = std::strcmp(dtype, "u32") == 0 ? "uint" : "int";
    const std::string uniformType =
        components == 1 ? scalarType : (std::strcmp(dtype, "u32") == 0 ? "uvec" : "ivec") + std::to_string(components);
    const std::string value = components == 1 ? "budget" : "budget.x";
    const std::string source =
        "#version 330\nuniform " + uniformType + " budget;void main(){gl_Position=vec4(float(" + value + "));}";
    nlohmann::json root = nlohmann::json::parse(pipelineBundle(inlineArtifact(source)));
    const std::string shapeType =
        components == 1 ? dtype : std::string("tensor<") + std::to_string(components) + "x" + dtype + ">";
    const nlohmann::json shape = components == 1 ? nlohmann::json::array() : nlohmann::json::array({components});
    root["variants"][0]["parameters"] = nlohmann::json::array(
        {{{"slot", 0},
          {"name", "budget"},
          {"kind", "tensor"},
          {"type", shapeType},
          {"element_layout", scalarElementLayout(dtype)},
          {"access", "read"},
          {"shape", shape},
          {"uses",
           nlohmann::json::array({{{"stage", "vertex"},
                                   {"interface", "uniform"},
                                   {"uniform_name", "budget"},
                                   {"dtype", dtype},
                                   {"shape", shape},
                                   {"physical_value_layout",
                                    physicalValueLayout(components * sizeof(uint32_t), sizeof(uint32_t),
                                                        components == 1 ? std::initializer_list<uint64_t>{}
                                                                        : std::initializer_list<uint64_t>{4})}}})}}});
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
          {"type", std::string("tensor<") + std::to_string(components) + "x" + dtype + ">"},
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
                                {"type", "!vernon.texture<\"2d\", f32>"},
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
          {"type", "!vernon.sampler"},
          {"access", "read"},
          {"shape", nlohmann::json::array()},
          {"source", "implicit_sampler"},
          {"uses", nlohmann::json::array(
                       {{{"stage", "fragment"},
                         {"interface", "resource"},
                         {"index", 1},
                         {"sampled_texture_bindings", nlohmann::json::array({{{"set", 0}, {"binding", 3}}})}}})}},
         {{"name", "__resolution"},
          {"kind", "tensor"},
          {"type", "tensor<2xf32>"},
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
                                           {"uniform_name", "__resolution"},
                                           {"physical_value_layout", physicalValueLayout(8, 8, {4})}}})}}});
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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
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

void expectIntegerUniformUpload(const char *dtype, VernonDataType dataType, const void *data, uint32_t components) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string bundleData = integerUniformBundle(dtype, components);
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    const uint64_t shape = components;
    const int64_t stride = sizeof(uint32_t);
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = data;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(dataType);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = components == 1 ? 0 : 1;
    argument.tensor.shape = components == 1 ? nullptr : &shape;
    argument.tensor.byte_strides = components == 1 ? nullptr : &stride;
    argument.tensor.byte_size = components * sizeof(uint32_t);
    VernonColorAttachment attachment{0, renderTarget.reference, 16, 16, VERNON_TEXTURE_RGBA8_UNORM};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.vertex_count = 3;
    invocation.instance_count = 1;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);

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
    static constexpr char reflection[] = "{" VERNON_JSON_VERSION_FIELDS R"(,
      "entries": [{
        "name": "main",
        "workgroup_size": [4, 1, 1],
        "physical_layouts": {
          "vulkan_std430_storage_buffer": {
            "profile":"vulkan_std430_storage_buffer","packing":"resource_bindings"
          }
        },
        "arguments": [
          {"kind":"tensor","dtype":"f32","shape":[4],"element_layout":{"logical_type":"f32","byte_size":4,
           "alignment":4,"layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
           "physical_layouts":{"vulkan_std430_storage_buffer":{"profile":"vulkan_std430_storage_buffer",
           "kind":"descriptor_storage_leaves"}},
           "binding":0},
          {"kind":"scalar","dtype":"f32","element_layout":{"logical_type":"f32","byte_size":4,
           "alignment":4,"layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
           "physical_layouts":{"vulkan_std430_storage_buffer":{"profile":"vulkan_std430_storage_buffer",
           "size":4,"alignment":4,"byte_strides":[]}},
           "binding":1}
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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(rhiRuntime(gl).device, &encoderDescriptor, &encoder),
              VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerEncoder{};
    ASSERT_EQ(vernonRuntimeReferenceRhiCommandEncoder(gl, encoder, &providerEncoder), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK);
    VernonRhiCommandEncoderStats encoderStats{};
    ASSERT_EQ(vernonRhiCommandEncoderGetStats(rhiRuntime(gl).device, encoder, &encoderStats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(encoderStats.dispatch_count, 1u);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderGetStats(rhiRuntime(gl).device, encoder, &encoderStats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(encoderStats.submission_count, 1u);
    VernonRhiCommandEncoder nextEncoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(rhiRuntime(gl).device, &encoderDescriptor, &nextEncoder),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(rhiRuntime(gl).device, nextEncoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(rhiRuntime(gl).device, nextEncoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(rhiRuntime(gl).device, nextEncoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, buffer.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(rhiRuntime(gl).device, buffer.handle), 0u);
    auto replacement =
        vernon::tests::createBuffer(rhiRuntime(gl), 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(replacement.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    EXPECT_NE(replacement.handle.index, buffer.handle.index);
    for (size_t iteration = 0; iteration < 128; ++iteration)
        ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(dispatchCount, 129u);
    EXPECT_EQ(storageBindingCount, 258u);
    EXPECT_EQ(memoryBarrierCount, 129u);

    vernonRuntimeLoadedPipelineDestroy(pipeline);
    auto recycled =
        vernon::tests::createBuffer(rhiRuntime(gl), 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(recycled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    EXPECT_EQ(recycled.handle.index, buffer.handle.index);
    EXPECT_NE(recycled.handle.generation, buffer.handle.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, recycled.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, replacement.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, UploadsColumnMajorMatricesWithoutCopying) {
    constexpr std::array<int64_t, 2> strides = {sizeof(float), 4 * sizeof(float)};
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL, "opengl", 4, 3, columnMajor, strides, 0, columnMajor);
}

TEST(RuntimeExternalGl, UploadsSignedAndUnsignedIntegerUniforms) {
    signedUniformUpload.fill(0);
    unsignedUniformUpload.fill(0);
    constexpr GlInt signedValue = -17;
    expectIntegerUniformUpload("i32", VERNON_DATA_I32, &signedValue, 1);
    EXPECT_EQ(signedUniformUpload[0], signedValue);

    constexpr std::array<GlUint, 3> unsignedValue = {3, 5, 8};
    expectIntegerUniformUpload("u32", VERNON_DATA_U32, unsignedValue.data(), unsignedValue.size());
    EXPECT_EQ(unsignedUniformUpload[0], unsignedValue[0]);
    EXPECT_EQ(unsignedUniformUpload[1], unsignedValue[1]);
    EXPECT_EQ(unsignedUniformUpload[2], unsignedValue[2]);
}

#if defined(VERNON_OPENGL_RUNTIME_ACCEPTANCE_BUNDLE)
TEST(RuntimeExternalGl, InvokesGeneratedResolutionAndSignedUniformPipeline) {
    const std::filesystem::path manifestPath = VERNON_OPENGL_RUNTIME_ACCEPTANCE_BUNDLE;
    std::ifstream manifestInput(manifestPath, std::ios::binary);
    const std::string bundleData((std::istreambuf_iterator<char>(manifestInput)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bundleData.empty());
    const nlohmann::json manifest = nlohmann::json::parse(bundleData);
    const std::string fragmentId = manifest["variants"][0]["program"]["fragment"];
    const std::filesystem::path fragmentPath =
        manifestPath.parent_path() / manifest["stage_artifacts"][fragmentId]["artifact"]["path"].get<std::string>();
    std::ifstream fragmentInput(fragmentPath, std::ios::binary);
    const std::string fragmentSource((std::istreambuf_iterator<char>(fragmentInput)), std::istreambuf_iterator<char>());
    EXPECT_NE(fragmentSource.find("uniform int max_steps;"), std::string::npos);
    EXPECT_NE(fragmentSource.find("uniform vec2 _vernon_resolution;"), std::string::npos);

    resolutionUpload.fill(0.0F);
    signedUniformUpload.fill(0);
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), &options);
    ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

    constexpr std::array<float, 6> positions{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
    auto vertices = vernon::tests::createBuffer(rhiRuntime(gl), sizeof(positions), alignof(float),
                                                VERNON_RHI_BUFFER_VERTEX, positions.data());
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 37, 23, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr GlInt maxSteps = 19;
    constexpr std::array<uint64_t, 2> positionShape{3, 2};
    constexpr std::array<int64_t, 2> positionStrides{2 * sizeof(float), sizeof(float)};
    std::array<VernonPipelineArgument, 2> arguments{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_HOST;
    arguments[0].tensor.host_data = &maxSteps;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
    arguments[0].tensor.access = VERNON_ACCESS_READ;
    arguments[0].tensor.byte_size = sizeof(maxSteps);
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[1].tensor.resource = vertices.reference;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = positionShape.size();
    arguments[1].tensor.shape = positionShape.data();
    arguments[1].tensor.byte_strides = positionStrides.data();
    arguments[1].tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, renderTarget.reference, 37, 23, VERNON_TEXTURE_RGBA8_UNORM};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments.data();
    invocation.argument_count = arguments.size();
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.vertex_count = 3;
    invocation.instance_count = 1;
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{37.0F, 23.0F}));
    EXPECT_EQ(signedUniformUpload[0], maxSteps);

    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(bundle);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

TEST(RuntimeExternalGl, PacksRowMajorMatricesWithCanonicalStrides) {
    constexpr std::array<int64_t, 2> strides = {4 * sizeof(float), sizeof(float)};
    constexpr std::array<float, 16> rowMajor = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
    };
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL, "opengl", 4, 3, rowMajor, strides, 0, columnMajor);
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

    nlohmann::json nonzeroSet = nlohmann::json::parse(internalValueBundle());
    nonzeroSet["variants"][0]["parameters"][0]["uses"][0]["vernon.set"] = 1;
    nonzeroSet["variants"][0]["internal_parameters"][0]["uses"][0]["sampled_texture_bindings"][0]["set"] = 1;
    bundleData = serialize(std::move(nonzeroSet));
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle);
    EXPECT_FALSE(vernonRuntimeResolvePipeline(bundle, {nullptr, 0}));
    const VernonStringView error = vernonRuntimeGetLastError(gl);
    EXPECT_NE(std::string_view(error.data, error.size).find("descriptor set 0"), std::string_view::npos);
    vernonRuntimePipelineBundleDestroy(bundle);

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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
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
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(rhiRuntime(gl).device, &encoderDescriptor, &encoder),
              VERNON_RHI_STATUS_OK);
    VernonRhiColorAttachment nativeColor{};
    nativeColor.view = {target.handle.index, target.handle.generation};
    nativeColor.load_operation = VERNON_RHI_LOAD_CLEAR;
    nativeColor.store_operation = VERNON_RHI_STORE_PRESERVE;
    VernonRhiRenderingDescriptor rendering{};
    rendering.struct_size = sizeof(rendering);
    rendering.color_attachments = &nativeColor;
    rendering.color_attachment_count = 1;
    rendering.width = 16;
    rendering.height = 12;
    rendering.layers = 1;
    ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(rhiRuntime(gl).device, encoder, &rendering), VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderObject providerEncoder{};
    ASSERT_EQ(vernonRuntimeReferenceRhiCommandEncoder(gl, encoder, &providerEncoder), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(boundSamplerUnit, 3u);
    EXPECT_NE(boundSamplerName, 0u);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{7.0F, 9.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{2, 3, 7, 9}));
    EXPECT_EQ(nextName, nextNameAfterPreparation);

    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, textureSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    RhiImage replacementSampled = importOpenGLTexture2D(gl, 9, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    auto replacementSampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(replacementSampled.handle.index, sampled.handle.index);
    ASSERT_NE(replacementSampler.handle.index, textureSampler.handle.index);

    argument.texture.sampler_resource = {};
    std::fill(std::begin(invocation.viewport), std::end(invocation.viewport), 0);
    ASSERT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{16.0F, 12.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{0, 0, 16, 12}));
    EXPECT_EQ(boundSamplerName, 0u);

    VernonRuntimeContext *other = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(other);
    auto foreignSampler = vernon::tests::createSampler(rhiRuntime(other));
    ASSERT_NE(foreignSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    argument.texture.sampler_resource = foreignSampler.reference;
    EXPECT_EQ(vernonRuntimePipelineEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(other).device, foreignSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(destroy(other), VERNON_STATUS_OK);

    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCommandEncoderStats encoderStats{};
    ASSERT_EQ(vernonRhiCommandEncoderGetStats(rhiRuntime(gl).device, encoder, &encoderStats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(encoderStats.rendering_scope_count, 1u);
    EXPECT_EQ(encoderStats.draw_count, 2u);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    pipeline = nullptr;
    vernonRuntimePipelineBundleDestroy(bundle);
    bundle = nullptr;
    ASSERT_EQ(vernonRhiCommandEncoderFinish(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceSubmit(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderGetStats(rhiRuntime(gl).device, encoder, &encoderStats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(encoderStats.submission_count, 1u);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, replacementSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, replacementSampled.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    RhiImage recycledSampled = importOpenGLTexture2D(gl, 10, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    auto recycledSampler = vernon::tests::createSampler(rhiRuntime(gl));
    EXPECT_EQ(recycledSampled.handle.index, sampled.handle.index);
    EXPECT_NE(recycledSampled.handle.generation, sampled.handle.generation);
    EXPECT_EQ(recycledSampler.handle.index, textureSampler.handle.index);
    EXPECT_NE(recycledSampler.handle.generation, textureSampler.handle.generation);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, recycledSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, recycledSampled.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ExecutionGraphFusesDrawsAndSubmitsOnce) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string bundleData = internalValueBundle();
    VernonPipelineBundle *bundle =
        vernonRuntimeLoadPipelineBundleWithOptions(gl, bundleData.data(), bundleData.size(), nullptr);
    ASSERT_TRUE(bundle);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    RhiImage sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    auto sampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(sampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TEXTURE;
    argument.texture = {VERNON_TEXTURE_RGBA8_UNORM, VERNON_ACCESS_READ, VERNON_TEXTURE_2D, 4, 4, 1,
                        sampled.reference,          sampler.reference};
    VernonColorAttachment attachment{0, target.reference, 16, 12, VERNON_TEXTURE_RGBA8_UNORM};
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.vertex_count = 3;
    invocation.instance_count = 1;

    {
        invalidateCount = 0;
        lastMemoryBarrierBits = 0;
        framebufferBindCount = 0;
        const size_t commandsBeforeGraph = vernon::runtime::getRhiAdapterRecordedCommandCount(gl);
        vernon::execution::ExecutionGraph graph(rhiRuntime(gl).device);
        const auto graphTarget = graph.importImage(target.handle, {target.handle.index, target.handle.generation},
                                                   VERNON_RHI_FORMAT_RGBA8_UNORM, 16, 12, 1, 1, true);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceIsImageValid(rhiRuntime(gl).device, target.handle), 0u);
        RhiImage replacement = importOpenGLTexture2D(gl, 11, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
        EXPECT_NE(replacement.handle.index, target.handle.index);
        graph.emplacePass<GraphImageWritePass>("produce", graphTarget);
        graph.emplacePass<RuntimeGraphRenderPass>("first", graphTarget, gl, pipeline, &invocation,
                                                  VERNON_RHI_LOAD_CLEAR);
        graph.emplacePass<RuntimeGraphRenderPass>("second", graphTarget, gl, pipeline, &invocation,
                                                  VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_DISCARD);
        ASSERT_EQ(graph.execute(), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(graph.lastStats().rendering_scope_count, 1u);
        EXPECT_EQ(graph.lastStats().barrier_count, 1u);
        EXPECT_EQ(graph.lastStats().draw_count, 2u);
        EXPECT_EQ(graph.lastStats().submission_count, 1u);
        EXPECT_EQ(invalidateCount, 1u);
        EXPECT_NE(lastMemoryBarrierBits & 0x00000400u, 0u);
        EXPECT_EQ(framebufferBindCount, 1u);
        EXPECT_EQ(vernon::runtime::getRhiAdapterRecordedCommandCount(gl) - commandsBeforeGraph, 2u);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, replacement.handle), VERNON_RHI_STATUS_OK);
    }

    RhiImage recycled = importOpenGLTexture2D(gl, 12, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    EXPECT_EQ(recycled.handle.index, target.handle.index);
    EXPECT_NE(recycled.handle.generation, target.handle.generation);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, sampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, recycled.handle), VERNON_RHI_STATUS_OK);
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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
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
        invocation.abi_version = VERNON_PIPELINE_VERSION;
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
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.color_attachments = &attachment;
    invocation.color_attachment_count = 1;
    invocation.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
    invocation.vertex_count = 3;
    invocation.instance_count = 1;
    const GlUint nextNameAfterPreparation = nextName;
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    attachment.load_operation = VERNON_RHI_LOAD_PRESERVE;
    attachment.store_operation = VERNON_RHI_STORE_DISCARD;
    // The host may change OpenGL state between command encoders, so each invocation must restore its bindings.
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(drawCount == 2);
    ASSERT_EQ(clearCount, 1u);
    EXPECT_FLOAT_EQ(clearColor[0], 0.25f);
    EXPECT_FLOAT_EQ(clearColor[1], 0.5f);
    EXPECT_EQ(invalidateCount, 1u);
    EXPECT_EQ(scissorUpload, (std::array<GlInt, 4>{0, 0, 16, 16}));
    EXPECT_EQ(programBindCount, 2u);
    EXPECT_EQ(framebufferBindCount, 2u);
    EXPECT_EQ(viewportCount, 2u);
    EXPECT_EQ(scissorCount, 2u);
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
