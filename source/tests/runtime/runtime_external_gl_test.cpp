#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"
#include "direct_stage_execution_graph_pass.h"
#include "runtime/content_hash.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
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
constexpr GlEnum kBack = 0x0405;
constexpr GlEnum kClockwise = 0x0900;
constexpr GlEnum kLess = 0x0201;
constexpr GlEnum kSourceAlpha = 0x0302;
constexpr GlEnum kOneMinusSourceAlpha = 0x0303;
constexpr GlEnum kAdd = 0x8006;
constexpr GlEnum kColorAttachment0 = 0x8CE0;
constexpr GlEnum kStencilAttachment = 0x8D20;

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
uint32_t textureViewCount = 0;
uint32_t viewportCount = 0;
uint32_t scissorCount = 0;
uint32_t stencilFuncCount = 0;
GlInt stencilReference = 0;
GlUint stencilReadMask = 0;
GlBoolean depthWrite = 0;
GlEnum depthComparison = 0;
GlEnum cullMode = 0;
GlEnum frontWinding = 0;
std::array<float, 2> depthBias{};
std::array<GlEnum, 4> blendFactors{};
std::array<GlEnum, 2> blendOperations{};
std::array<GlBoolean, 4> colorWriteMask{};
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

VernonRhiImageViewDescriptor fullImageView(RhiImage image, VernonRhiFormat format) {
    VernonRhiImageViewDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.image = image.handle;
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = format;
    descriptor.mip_level_count = 1;
    descriptor.array_layer_count = 1;
    descriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    return descriptor;
}

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
std::vector<GlEnum> invalidatedAttachments;
std::vector<unsigned char> bufferStorage;
std::vector<unsigned char> textureStorage;
std::vector<GlInt> framebufferTextureLayers;
std::array<GlInt, 4> signedUniformUpload{};
std::array<GlUint, 4> unsignedUniformUpload{};

void makeCurrent(void *) { ++makeCurrentCount; }
void GL_CALL objectNoop(GlUint) {}
void GL_CALL noArgsNoop() {}
void GL_CALL objectPairNoop(GlUint, GlUint) {}
void GL_CALL infoLogNoop(GlUint, GlSize, GlSize *length, char *) {
    if (length)
        *length = 0;
}
void GL_CALL enumNoop(GlEnum) {}
void GL_CALL enumIntNoop(GlEnum, GlInt) {}
void GL_CALL deleteNamesNoop(GlSize, const GlUint *) {}
void GL_CALL framebufferTexture2DNoop(GlEnum, GlEnum, GlEnum, GlUint, GlInt) {}
void GL_CALL framebufferTextureLayer(GlEnum, GlEnum, GlUint, GlInt, GlInt layer) {
    framebufferTextureLayers.push_back(layer);
}
void GL_CALL drawBuffersNoop(GlSize, const GlEnum *) {}
void GL_CALL drawArraysInstanced(GlEnum, GlInt, GlSize, GlSize) { ++drawCount; }
void GL_CALL uniformFvNoop(GlInt, GlSize, const float *) {}
void GL_CALL uniformMatrixNoop(GlInt, GlSize, GlBoolean, const float *) {}
void GL_CALL uniform1iNoop(GlInt, GlInt) {}
void GL_CALL texImage3DNoop(GlEnum, GlInt, GlInt, GlSize, GlSize, GlSize, GlInt, GlEnum, GlEnum, const void *) {}
void GL_CALL texSubImage3DNoop(GlEnum, GlInt, GlInt, GlInt, GlInt, GlSize, GlSize, GlSize, GlEnum, GlEnum,
                               const void *) {}
void GL_CALL blendFuncSeparate(GlEnum sourceColor, GlEnum destinationColor, GlEnum sourceAlpha,
                               GlEnum destinationAlpha) {
    blendFactors = {sourceColor, destinationColor, sourceAlpha, destinationAlpha};
}
void GL_CALL blendEquationSeparate(GlEnum color, GlEnum alpha) { blendOperations = {color, alpha}; }
void GL_CALL colorMask(GlBoolean red, GlBoolean green, GlBoolean blue, GlBoolean alpha) {
    colorWriteMask = {red, green, blue, alpha};
}
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
void GL_CALL texImage2D(GlEnum, GlInt, GlInt, GlSize width, GlSize height, GlInt, GlEnum, GlEnum type, const void *) {
    textureStorage.resize(static_cast<size_t>(width) * height * (type == 0x8DAD ? 8 : 4));
    if (type == 0x8DAD)
        for (size_t offset = 0; offset < textureStorage.size(); offset += 8) {
            constexpr float depth = 0.25f;
            constexpr uint32_t stencil = 7;
            std::memcpy(textureStorage.data() + offset, &depth, sizeof(depth));
            std::memcpy(textureStorage.data() + offset + sizeof(depth), &stencil, sizeof(stencil));
        }
}
void GL_CALL texSubImage2D(GlEnum, GlInt, GlInt, GlInt, GlSize width, GlSize height, GlEnum, GlEnum,
                           const void *source) {
    const size_t size = static_cast<size_t>(width) * height * 4;
    textureStorage.assign(static_cast<const unsigned char *>(source),
                          static_cast<const unsigned char *>(source) + size);
}
void GL_CALL textureView(GlUint, GlEnum, GlUint, GlEnum, GlUint, GlUint, GlUint, GlUint) { ++textureViewCount; }
void GL_CALL readPixels(GlInt, GlInt, GlSize width, GlSize height, GlEnum, GlEnum type, void *destination) {
    std::memcpy(destination, textureStorage.data(), static_cast<size_t>(width) * height * (type == 0x8DAD ? 8 : 4));
}
GlEnum GL_CALL framebufferStatus(GlEnum) { return kFramebufferComplete; }
void GL_CALL drawArrays(GlEnum, GlInt, GlSize) { ++drawCount; }
void GL_CALL clearBufferfv(GlEnum, GlInt drawBuffer, const float *value) {
    ++clearCount;
    clearedDrawBuffer = drawBuffer;
    std::copy_n(value, clearColor.size(), clearColor.begin());
}
void GL_CALL clearBufferiv(GlEnum, GlInt, const GlInt *) {}
void GL_CALL clearBufferfi(GlEnum, GlInt, float, GlInt) {}
void GL_CALL readBuffer(GlEnum) {}
void GL_CALL depthMask(GlBoolean value) { depthWrite = value; }
void GL_CALL depthFunc(GlEnum value) { depthComparison = value; }
void GL_CALL cullFace(GlEnum value) { cullMode = value; }
void GL_CALL frontFace(GlEnum value) { frontWinding = value; }
void GL_CALL polygonOffset(float slope, float constant) { depthBias = {slope, constant}; }
void GL_CALL enablei(GlEnum, GlUint) {}
void GL_CALL disablei(GlEnum, GlUint) {}
void GL_CALL blendFuncSeparatei(GlUint, GlEnum, GlEnum, GlEnum, GlEnum) {}
void GL_CALL blendEquationSeparatei(GlUint, GlEnum, GlEnum) {}
void GL_CALL colorMaski(GlUint, GlBoolean, GlBoolean, GlBoolean, GlBoolean) {}
void GL_CALL stencilFuncSeparate(GlEnum, GlEnum, GlInt reference, GlUint mask) {
    ++stencilFuncCount;
    stencilReference = reference;
    stencilReadMask = mask;
}
void GL_CALL stencilOpSeparate(GlEnum, GlEnum, GlEnum, GlEnum) {}
void GL_CALL stencilMaskSeparate(GlEnum, GlUint) {}
void GL_CALL invalidateFramebuffer(GlEnum, GlSize count, const GlEnum *attachments) {
    invalidateCount += static_cast<uint32_t>(count);
    invalidatedAttachments.insert(invalidatedAttachments.end(), attachments, attachments + count);
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
    PROC("glCompileShader", objectNoop);
    PROC("glGetShaderiv", getShaderiv);
    PROC("glGetShaderInfoLog", infoLogNoop);
    PROC("glDeleteShader", objectNoop);
    PROC("glCreateProgram", createProgram);
    PROC("glAttachShader", objectPairNoop);
    PROC("glLinkProgram", objectNoop);
    PROC("glGetProgramiv", getProgramiv);
    PROC("glGetProgramInfoLog", infoLogNoop);
    PROC("glDeleteProgram", objectNoop);
    PROC("glGetIntegerv", getIntegerv);
    PROC("glGenVertexArrays", genNames);
    PROC("glDeleteVertexArrays", deleteNamesNoop);
    PROC("glBindVertexArray", bindVertexArray);
    PROC("glEnableVertexAttribArray", objectNoop);
    PROC("glUseProgram", useProgram);
    PROC("glGenFramebuffers", genNames);
    PROC("glDeleteFramebuffers", deleteNamesNoop);
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
    PROC("glBindTexture", objectPairNoop);
    PROC("glTextureView", textureView);
    PROC("glTexImage2D", texImage2D);
    PROC("glTexImage3D", texImage3DNoop);
    PROC("glTexSubImage2D", texSubImage2D);
    PROC("glTexSubImage3D", texSubImage3DNoop);
    PROC("glTexParameteri", enumIntNoop);
    PROC("glGenerateMipmap", enumNoop);
    PROC("glPixelStorei", enumIntNoop);
    PROC("glReadPixels", readPixels);
    PROC("glGenSamplers", genNames);
    PROC("glDeleteSamplers", deleteSamplerNames);
    PROC("glSamplerParameteri", enumIntNoop);
    PROC("glCheckFramebufferStatus", framebufferStatus);
    PROC("glBindFramebuffer", bindFramebuffer);
    PROC("glFramebufferTexture2D", framebufferTexture2DNoop);
    PROC("glFramebufferTextureLayer", framebufferTextureLayer);
    PROC("glDrawBuffers", drawBuffersNoop);
    PROC("glDrawArrays", drawArrays);
    PROC("glDrawArraysInstanced", drawArraysInstanced);
    PROC("glVertexAttribPointer", vertexAttribPointer);
    PROC("glVertexAttribIPointer", vertexAttribIPointer);
    PROC("glVertexAttribLPointer", vertexAttribLPointer);
    PROC("glVertexAttribDivisor", vertexAttribDivisor);
    PROC("glDrawElementsInstanced", drawElementsInstanced);
    PROC("glClearBufferfv", clearBufferfv);
    PROC("glClearBufferiv", clearBufferiv);
    PROC("glClearBufferfi", clearBufferfi);
    PROC("glReadBuffer", readBuffer);
    PROC("glDepthMask", depthMask);
    PROC("glDepthFunc", depthFunc);
    PROC("glEnable", enumNoop);
    PROC("glDisable", enumNoop);
    PROC("glCullFace", cullFace);
    PROC("glFrontFace", frontFace);
    PROC("glPolygonOffset", polygonOffset);
    PROC("glBlendFuncSeparate", blendFuncSeparate);
    PROC("glBlendEquationSeparate", blendEquationSeparate);
    PROC("glColorMask", colorMask);
    PROC("glEnablei", enablei);
    PROC("glDisablei", disablei);
    PROC("glBlendFuncSeparatei", blendFuncSeparatei);
    PROC("glBlendEquationSeparatei", blendEquationSeparatei);
    PROC("glColorMaski", colorMaski);
    PROC("glStencilFuncSeparate", stencilFuncSeparate);
    PROC("glStencilOpSeparate", stencilOpSeparate);
    PROC("glStencilMaskSeparate", stencilMaskSeparate);
    PROC("glInvalidateFramebuffer", invalidateFramebuffer);
    PROC("glGetUniformLocation", getUniformLocation);
    PROC("glUniform1fv", uniformFvNoop);
    PROC("glUniform2fv", uniform2fv);
    PROC("glUniform3fv", uniformFvNoop);
    PROC("glUniform4fv", uniformFvNoop);
    PROC("glUniform1iv", uniform1iv);
    PROC("glUniform2iv", uniform2iv);
    PROC("glUniform3iv", uniform3iv);
    PROC("glUniform4iv", uniform4iv);
    PROC("glUniform1uiv", uniform1uiv);
    PROC("glUniform2uiv", uniform2uiv);
    PROC("glUniform3uiv", uniform3uiv);
    PROC("glUniform4uiv", uniform4uiv);
    PROC("glUniformMatrix2fv", uniformMatrixNoop);
    PROC("glUniformMatrix2x3fv", uniformMatrixNoop);
    PROC("glUniformMatrix2x4fv", uniformMatrixNoop);
    PROC("glUniformMatrix3x2fv", uniformMatrixNoop);
    PROC("glUniformMatrix3fv", uniformMatrixNoop);
    PROC("glUniformMatrix3x4fv", uniformMatrixNoop);
    PROC("glUniformMatrix4x2fv", uniformMatrixNoop);
    PROC("glUniformMatrix4x3fv", uniformMatrixNoop);
    PROC("glUniformMatrix4fv", uniformMatrix4fv);
    PROC("glUniform1i", uniform1iNoop);
    PROC("glActiveTexture", enumNoop);
    PROC("glBindSampler", bindSampler);
    PROC("glFinish", noArgsNoop);
    PROC("glViewport", viewport);
    PROC("glScissor", scissor);
    PROC("glDispatchCompute", dispatchCompute);
    PROC("glMemoryBarrier", memoryBarrier);
#undef PROC
    return nullptr;
}

void *getProcAddressWithoutTextureView(void *userData, const char *name) {
    return std::strcmp(name, "glTextureView") == 0 ? nullptr : getProcAddress(userData, name);
}

VernonRuntimeContext *create(VernonRuntimeBackend backend, uint16_t major, uint16_t minor,
                             bool exposeTextureView = true) {
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = exposeTextureView ? &getProcAddress : &getProcAddressWithoutTextureView;
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

nlohmann::json nativeUniformPlan(uint64_t size, uint64_t alignment, std::initializer_list<uint64_t> byteStrides,
                                 const char *representation = "f32", std::vector<uint64_t> shape = {},
                                 const char *layoutHash = "test-layout") {
    const uint64_t scalarSize = std::min<uint64_t>(size, 4);
    nlohmann::json scalar = {{"kind", "scalar"},
                             {"representation", representation},
                             {"offset", 0},
                             {"size", scalarSize},
                             {"alignment", std::min<uint64_t>(alignment, 4)}};
    nlohmann::json root = scalar;
    if (!shape.empty() || byteStrides.size()) {
        if (shape.empty())
            shape.assign(byteStrides.size(), 1);
        if (byteStrides.size())
            root = {{"kind", "array"},
                    {"offset", 0},
                    {"size", size},
                    {"alignment", alignment},
                    {"shape", shape},
                    {"byte_strides", byteStrides},
                    {"children", nlohmann::json::array({std::move(scalar)})}};
    }
    return {{"kind", "native_uniform"},
            {"profile", "opengl_native_uniform"},
            {"canonical_layout_hash", layoutHash},
            {"root", std::move(root)}};
}

nlohmann::json vectorValueLayout(const char *dtype, uint32_t components) {
    const VernonDataType dataType = dtype == std::string("u32")   ? VERNON_DATA_U32
                                    : dtype == std::string("i32") ? VERNON_DATA_I32
                                                                  : VERNON_DATA_F32;
    const VernonValueLayoutView scalar = vernonRuntimeGetScalarValueLayout(dataType);
    const uint64_t byteSize = components * scalar.byte_size;
    const std::string spelling =
        components == 1 ? std::string(dtype) : std::string("tensor<") + std::to_string(components) + "x" + dtype + ">";
    const std::string hash =
        std::string(scalar.layout_hash.data, scalar.layout_hash.size) + "|vec" + std::to_string(components);
    nlohmann::json leaf = {
        {"path", nlohmann::json::array()}, {"dtype", dtype}, {"byte_offset", 0}, {"scalar_count", components}};
    if (components > 1)
        leaf["shape"] = nlohmann::json::array({components});
    return {{"logical_type", spelling},
            {"byte_size", byteSize},
            {"alignment", scalar.alignment},
            {"layout_hash", hash},
            {"leaves", nlohmann::json::array({std::move(leaf)})}};
}

std::string directGraphicsFixture(const nlohmann::json &artifact) {
    nlohmann::json root = {
        {"variants", nlohmann::json::array({{{"key", nlohmann::json::array()},
                                             {"parameters", nlohmann::json::array()},
                                             {"outputs", nlohmann::json::array()},
                                             {"program", {{"vertex", "vs"}, {"fragment", "fs"}}}}})},
        {"stages",
         {{"vs",
           {{"stage", "vertex"}, {"entry", "main"}, {"artifact", artifact}, {"reflection", nlohmann::json::object()}}},
          {"fs",
           {{"stage", "fragment"},
            {"entry", "main"},
            {"artifact", artifact},
            {"reflection", nlohmann::json::object()}}}}}};
    return root.dump(-1, ' ', false);
}

std::string matrixFixture(const char *target) {
    const std::string source = "#version 330\nuniform mat4 transform;void main(){gl_Position=transform*"
                               "vec4(0.0,0.0,0.0,1.0);}";
    nlohmann::json root = nlohmann::json::parse(directGraphicsFixture(inlineArtifact(source)));
    if (std::strcmp(target, "opengles") == 0) {
        root["stages"]["vs"]["artifact"]["format"] = "gles";
        root["stages"]["fs"]["artifact"]["format"] = "gles";
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
                                           {"transport", "native_uniform"},
                                           {"interface_plan", nativeUniformPlan(64, 16, {4, 16})}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string integerUniformFixture(const char *dtype, uint32_t components) {
    const std::string scalarType = std::strcmp(dtype, "u32") == 0 ? "uint" : "int";
    const std::string uniformType =
        components == 1 ? scalarType : (std::strcmp(dtype, "u32") == 0 ? "uvec" : "ivec") + std::to_string(components);
    const std::string value = components == 1 ? "budget" : "budget.x";
    const std::string source =
        "#version 330\nuniform " + uniformType + " budget;void main(){gl_Position=vec4(float(" + value + "));}";
    nlohmann::json root = nlohmann::json::parse(directGraphicsFixture(inlineArtifact(source)));
    const std::string shapeType =
        components == 1 ? dtype : std::string("tensor<") + std::to_string(components) + "x" + dtype + ">";
    const nlohmann::json shape = components == 1 ? nlohmann::json::array() : nlohmann::json::array({components});
    const nlohmann::json valueLayout = vectorValueLayout(dtype, components);
    const std::string layoutHash = valueLayout["layout_hash"].get<std::string>();
    root["variants"][0]["parameters"] = nlohmann::json::array(
        {{{"slot", 0},
          {"name", "budget"},
          {"kind", "tensor"},
          {"type", shapeType},
          {"element_layout", scalarElementLayout(dtype)},
          {"value_layout", valueLayout},
          {"access", "read"},
          {"shape", shape},
          {"uses", nlohmann::json::array(
                       {{{"stage", "vertex"},
                         {"interface", "uniform"},
                         {"uniform_name", "budget"},
                         {"dtype", dtype},
                         {"shape", shape},
                         {"transport", "native_uniform"},
                         {"interface_plan",
                          nativeUniformPlan(
                              components * sizeof(uint32_t), sizeof(uint32_t),
                              components == 1 ? std::initializer_list<uint64_t>{} : std::initializer_list<uint64_t>{4},
                              dtype, components == 1 ? std::vector<uint64_t>{} : std::vector<uint64_t>{components},
                              layoutHash.c_str())}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string vertexInputFixture(const char *dtype = "f32", uint32_t components = 3, uint32_t divisor = 1) {
    nlohmann::json root = nlohmann::json::parse(
        directGraphicsFixture(inlineArtifact("#version 330\nvoid main(){gl_Position=vec4(0.0);}")));
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

std::string internalValueFixture() {
    nlohmann::json root = nlohmann::json::parse(directGraphicsFixture(inlineArtifact("#version 330\nvoid main(){}")));
    root["variants"][0]["parameters"] =
        nlohmann::json::array({{{"slot", 0},
                                {"name", "image"},
                                {"kind", "image"},
                                {"type", "!vernon.texture<\"2d\", f32, \"unknown\", \"sampled\">"},
                                {"dtype", "f32"},
                                {"access", "read"},
                                {"dimension", "2d"},
                                {"binding_role", "sampled"},
                                {"sample_result_class", "float"},
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
                         {"sampled_image_bindings", nlohmann::json::array({{{"set", 0}, {"binding", 3}}})}}})}},
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
                                           {"transport", "native_uniform"},
                                           {"interface_plan", nativeUniformPlan(8, 8, {4})}}})}}});
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

std::string explicitSamplerFixture() {
    nlohmann::json root = nlohmann::json::parse(internalValueFixture());
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

std::vector<uint64_t> fixtureShape(const nlohmann::json &value) {
    return value.is_array() ? value.get<std::vector<uint64_t>>() : std::vector<uint64_t>{};
}

bool materializeFixtureUse(const nlohmann::json &source, vernon::runtime::ParameterUse &use) {
    use.stage = source.value("stage", "");
    use.interfaceKind = source.value("interface", "");
    use.uniformName = source.value("uniform_name", "");
    use.dtype = source.value("dtype", "");
    use.shape = fixtureShape(source.value("shape", nlohmann::json::array()));
    use.index = source.value("index", 0u);
    use.location = source.value("vernon.location", UINT32_MAX);
    use.divisor = source.value("vernon.instance_divisor", 0u);
    use.descriptorSet = source.value("vernon.set", 0u);
    use.binding = source.value("vernon.binding", UINT32_MAX);
    use.transport = source.value("transport", "");
    for (const nlohmann::json &binding : source.value("sampled_image_bindings", nlohmann::json::array()))
        use.sampledImageBindings.push_back({binding.value("set", 0u), binding.value("binding", UINT32_MAX)});
    for (const nlohmann::json &leaf : source.value("attribute_leaves", nlohmann::json::array()))
        use.attributeLeaves.push_back({leaf.value("location_offset", 0u), leaf.value("dtype", ""),
                                       leaf.value("component_count", 0u), leaf.value("byte_offset", 0u)});
    std::string error;
    if (source.contains("value_layout")) {
        use.valueLayout.emplace();
        if (!vernon::runtime::parseArtifactValueLayout(source["value_layout"], *use.valueLayout, error))
            return false;
    }
    if (source.contains("interface_plan")) {
        use.interfacePlan.emplace();
        if (!vernon::runtime::parseArtifactInterfacePlan(source["interface_plan"], *use.interfacePlan, error))
            return false;
    }
    if (source.contains("tensor_view_descriptor")) {
        const nlohmann::json &descriptor = source["tensor_view_descriptor"];
        use.tensorViewDescriptor = vernon::runtime::TensorViewDescriptorUse{
            descriptor.value("rank", 0u), descriptor.value("offset_binding", UINT32_MAX),
            descriptor.value("extent_bindings", std::vector<uint32_t>{}),
            descriptor.value("stride_bindings", std::vector<uint32_t>{})};
    }
    return true;
}

bool materializeFixtureParameter(const nlohmann::json &source, vernon::runtime::Parameter &parameter) {
    parameter.slot = source.value("slot", 0u);
    parameter.name = source.value("name", "");
    parameter.kind = source.value("kind", "");
    const std::string parameterSource = source.value("source", "");
    if (parameterSource.empty())
        parameter.source = vernon::runtime::StageParameterSource::Projected;
    else if (parameterSource == "direct")
        parameter.source = vernon::runtime::StageParameterSource::Direct;
    else if (parameterSource == "implicit_sampler")
        parameter.source = vernon::runtime::StageParameterSource::ImplicitSampler;
    else if (parameterSource == "system_value" && source.value("system_value", "") == "resolution")
        parameter.source = vernon::runtime::StageParameterSource::Resolution;
    else
        return false;
    parameter.access = source.value("access", "");
    parameter.addressSpace = source.value("address_space", "");
    parameter.dimension = source.value("dimension", "");
    parameter.bindingRole = source.value("binding_role", "");
    parameter.sampleResultClass = source.value("sample_result_class", "");
    parameter.exactStorageFormat = source.value("texture_format", "");
    parameter.shape = fixtureShape(source.value("shape", nlohmann::json::array()));
    std::string error;
    if (source.contains("value_layout")) {
        parameter.valueLayout.emplace();
        if (!vernon::runtime::parseArtifactValueLayout(source["value_layout"], *parameter.valueLayout, error))
            return false;
    }
    if (source.contains("element_layout") &&
        !vernon::runtime::parseArtifactValueLayout(source["element_layout"], parameter.elementLayout, error))
        return false;
    for (const nlohmann::json &sourceUse : source.value("uses", nlohmann::json::array())) {
        parameter.uses.emplace_back();
        if (!materializeFixtureUse(sourceUse, parameter.uses.back()))
            return false;
    }
    return true;
}

VernonStageExecutable *loadDirectGraphicsFixture(VernonRuntimeContext *context, const std::string &fixtureData) {
    const nlohmann::json fixture = nlohmann::json::parse(fixtureData, nullptr, false);
    if (!context || fixture.is_discarded() || !fixture.contains("variants") || fixture["variants"].size() != 1 ||
        !fixture.contains("stages"))
        return nullptr;
    const nlohmann::json &sourceVariant = fixture["variants"][0];
    vernon::runtime::StageBindingPlan variant;
    for (const auto &[stage, entry] : sourceVariant["program"].items())
        variant.artifactKeys.emplace(stage, entry.get<std::string>());
    variant.compute = variant.artifactKeys.count("compute") ? variant.artifactKeys.at("compute") : "";
    variant.vertex = variant.artifactKeys.count("vertex") ? variant.artifactKeys.at("vertex") : "";
    variant.fragment = variant.artifactKeys.count("fragment") ? variant.artifactKeys.at("fragment") : "";
    for (const nlohmann::json &source : sourceVariant.value("parameters", nlohmann::json::array())) {
        variant.parameters.emplace_back();
        if (!materializeFixtureParameter(source, variant.parameters.back()))
            return nullptr;
    }
    for (const nlohmann::json &source : sourceVariant.value("internal_parameters", nlohmann::json::array())) {
        variant.runtimeParameters.emplace_back();
        if (!materializeFixtureParameter(source, variant.runtimeParameters.back()))
            return nullptr;
    }
    for (const nlohmann::json &source : sourceVariant.value("outputs", nlohmann::json::array()))
        variant.outputs.push_back(
            {source.value("name", ""), source.value("kind", ""), source.value("dtype", ""), source.value("access", ""),
             fixtureShape(source.value("shape", nlohmann::json::array())), source.value("location", UINT32_MAX)});
    std::string error;
    if (!vernon::runtime::validateStageBindingPlan(variant, error))
        return nullptr;
    vernon::runtime::rebuildStageBindingLayoutViews(variant);

    vernon::runtime::BackendStageBuildInputs stages;
    stages.context = context;
    for (const auto &[id, source] : fixture["stages"].items()) {
        vernon::runtime::LoadedStageArtifact stage;
        stage.entry = source.value("entry", "");
        const nlohmann::json &artifact = source["artifact"];
        if (artifact.value("storage", "") != "inline" || !artifact["data"].is_string())
            return nullptr;
        stage.source = artifact["data"].get<std::string>();
        stages.artifacts.emplace(id, std::move(stage));
    }
    auto pipeline = std::make_unique<VernonStageExecutable>();
    pipeline->context = context;
    pipeline->bindingProjection = std::move(variant);
    if (!vernon::runtime::resolveBackendPipeline(stages, pipeline->bindingProjection, *pipeline))
        return nullptr;
    ++context->livePipelines;
    return pipeline.release();
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
    const std::string bundleData = matrixFixture(target);
    VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    const auto renderTargetView = vernon::tests::createImageView(rhiRuntime(gl), renderTarget, VERNON_RHI_IMAGE_2D,
                                                                 VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr std::array<uint64_t, 2> shape = {4, 4};
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = storage.data();
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = 2;
    argument.tensor.shape = shape.data();
    argument.tensor.byte_strides = strides.data();
    argument.tensor.byte_size = sizeof(storage);
    VernonColorAttachment attachment{0, renderTargetView.reference};
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3);
    graphics.bind(invocation);
    const GlUint nextNameAfterPreparation = nextName;
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(nextName, nextNameAfterPreparation);
    EXPECT_EQ(matrixTranspose, expectedTranspose);
    EXPECT_EQ(matrixUpload, expectedUpload);
    EXPECT_EQ(clearCount, 2U);
    EXPECT_EQ(clearedDrawBuffer, 0);
    EXPECT_EQ(clearColor, (std::array<float, 4>{}));

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, renderTargetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
    EXPECT_EQ(vernon::runtime::getRhiAdapterLivePreparedPipelineCount(gl), 0u);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

void expectIntegerUniformUpload(const char *dtype, VernonDataType dataType, const void *data, uint32_t components) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string bundleData = integerUniformFixture(dtype, components);
    VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    const auto renderTargetView = vernon::tests::createImageView(rhiRuntime(gl), renderTarget, VERNON_RHI_IMAGE_2D,
                                                                 VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    const uint64_t shape = components;
    const int64_t stride = sizeof(uint32_t);
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = data;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(dataType);
    argument.tensor.access = VERNON_ACCESS_READ;
    argument.tensor.rank = components == 1 ? 0 : 1;
    argument.tensor.shape = components == 1 ? nullptr : &shape;
    argument.tensor.byte_strides = components == 1 ? nullptr : &stride;
    argument.tensor.byte_size = components * sizeof(uint32_t);
    VernonColorAttachment attachment{0, renderTargetView.reference};
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3);
    graphics.bind(invocation);
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK);

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, renderTargetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

nlohmann::json inlineArtifact(const std::string &source) {
    return {{"format", "glsl"},      {"storage", "inline"},
            {"encoding", "utf8"},    {"data", source},
            {"size", source.size()}, {"sha256", vernon::runtime::sha256Hex(source.data(), source.size())}};
}

} // namespace

TEST(RuntimeExternalGl, CreatesAndReadsBackD32S8Image) {
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = &getProcAddress;
    callbacks.api_version_major = 4;
    callbacks.api_version_minor = 3;
    vernon::tests::RhiRuntime context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_OPENGL, &callbacks);
    ASSERT_NE(context.runtime, nullptr);

    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.width = 2;
    descriptor.height = 2;
    descriptor.depth = 1;
    descriptor.mip_levels = 1;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.format = VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    descriptor.usage = VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(context.device, &descriptor, &image), VERNON_RHI_STATUS_OK);
    std::array<uint8_t, 2 * 2 * 8> data{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = descriptor.width;
    download.height = descriptor.height;
    download.depth = descriptor.depth;
    download.destination_format = VERNON_RHI_IMAGE_DATA_DEPTH_STENCIL;
    download.destination_type = VERNON_RHI_IMAGE_DATA_FLOAT32;
    EXPECT_EQ(vernonRhiDeviceDownloadImage(context.device, image, &download, data.data(), data.size()),
              VERNON_RHI_STATUS_OK);
    for (size_t offset = 0; offset < data.size(); offset += 8) {
        float depth{};
        std::memcpy(&depth, data.data() + offset, sizeof(depth));
        EXPECT_FLOAT_EQ(depth, 0.25f);
        EXPECT_EQ(data[offset + sizeof(depth)], 7);
    }
    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeDestroy(context.runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

TEST(RuntimeExternalGl, ReadsEveryThreeDimensionalImageLayer) {
    framebufferTextureLayers.clear();
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = &getProcAddress;
    callbacks.api_version_major = 4;
    callbacks.api_version_minor = 3;
    vernon::tests::RhiRuntime context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_OPENGL, &callbacks);
    ASSERT_NE(context.runtime, nullptr);

    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_3D;
    descriptor.width = 2;
    descriptor.height = 2;
    descriptor.depth = 3;
    descriptor.mip_levels = 1;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    descriptor.usage =
        VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(context.device, &descriptor, &image), VERNON_RHI_STATUS_OK);

    std::array<uint8_t, 2 * 2 * 4> layer{};
    for (size_t index = 0; index < layer.size(); ++index)
        layer[index] = static_cast<uint8_t>(index);
    textureStorage.assign(layer.begin(), layer.end());
    std::array<uint8_t, 3 * 2 * 2 * 4> volume{};
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.width = descriptor.width;
    download.height = descriptor.height;
    download.depth = descriptor.depth;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    ASSERT_EQ(vernonRhiDeviceDownloadImage(context.device, image, &download, volume.data(), volume.size()),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(framebufferTextureLayers, (std::vector<GlInt>{0, 1, 2}));
    for (size_t offset = 0; offset < volume.size(); offset += layer.size())
        EXPECT_TRUE(std::equal(layer.begin(), layer.end(), volume.begin() + static_cast<ptrdiff_t>(offset)));

    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeDestroy(context.runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

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
        "dispatch_contract": {"unit_grid_axes": [], "requires_unit_workgroup": false},
        "physical_layouts": {
          "vulkan_std430_storage_buffer": {
            "profile":"vulkan_std430_storage_buffer","packing":"resource_bindings"
          }
        },
        "arguments": [
          {"kind":"tensor","dtype":"f32","access":"write","shape":[4],"element_layout":{"logical_type":"f32","byte_size":4,
           "alignment":4,"layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
           "physical_layouts":{"vulkan_std430_storage_buffer":{"profile":"vulkan_std430_storage_buffer",
           "kind":"resource_binding","resource_kind":"descriptor_storage_leaves"}},
           "binding":0},
          {"kind":"scalar","dtype":"f32","access":"read","value_layout":{"logical_type":"f32","byte_size":4,
           "alignment":4,"layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
           "physical_layouts":{"vulkan_std430_storage_buffer":{"profile":"vulkan_std430_storage_buffer",
           "kind":"byte_transport","canonical_layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
           "root":{"kind":"scalar","representation":"f32","offset":0,"size":4,"alignment":4}}},
           "binding":1}
        ]
      }]
    })";
    VernonStageExecutable *pipeline =
        vernonRuntimeLoadArtifact(gl, source, sizeof(source) - 1, reflection, sizeof(reflection) - 1, "main", 4);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    const float factor = 2.0F;
    const uint64_t outputShape[]{4};
    const int64_t outputStride[]{sizeof(float)};
    VernonProgramArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PROGRAM_TENSOR;
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
    arguments[1].kind = VERNON_PROGRAM_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_HOST;
    arguments[1].tensor.host_data = &factor;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
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
    ASSERT_EQ(vernonRuntimeStageEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    VernonRhiCommandEncoderStats encoderStats{};
    ASSERT_EQ(vernonRhiCommandEncoderFinish(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(rhiRuntime(gl).device, encoder, &encoderStats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(encoderStats.dispatch_count, 1u);
    EXPECT_EQ(encoderStats.submission_count, 1u);
    VernonRhiCommandEncoder nextEncoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(rhiRuntime(gl).device, &encoderDescriptor, &nextEncoder),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(rhiRuntime(gl).device, nextEncoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(rhiRuntime(gl).device, nextEncoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeStageEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, buffer.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsBufferValid(rhiRuntime(gl).device, buffer.handle), 0u);
    auto replacement =
        vernon::tests::createBuffer(rhiRuntime(gl), 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(replacement.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    EXPECT_NE(replacement.handle.index, buffer.handle.index);
    for (size_t iteration = 0; iteration < 128; ++iteration)
        ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(dispatchCount, 129u);
    EXPECT_EQ(storageBindingCount, 258u);
    EXPECT_EQ(memoryBarrierCount, 129u);

    vernonRuntimeStageExecutableDestroy(pipeline);
    auto recycled =
        vernon::tests::createBuffer(rhiRuntime(gl), 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(recycled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    EXPECT_EQ(recycled.handle.index, buffer.handle.index);
    EXPECT_NE(recycled.handle.generation, buffer.handle.generation);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, recycled.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, replacement.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, RejectsStorageImageMetadataThatDisagreesWithRhiResource) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    auto image = vernon::tests::createImage(rhiRuntime(gl), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, 1, 1, 1,
                                            VERNON_RHI_IMAGE_STORAGE);
    const auto imageView =
        vernon::tests::createImageView(rhiRuntime(gl), image, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(image.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    static constexpr char source[] = "#version 430\nlayout(local_size_x=1) in;"
                                     "layout(rgba32f,binding=0) uniform writeonly image2D outputImage;"
                                     "void main(){imageStore(outputImage,ivec2(0),vec4(1.0));}";
    static constexpr char reflection[] = "{" VERNON_JSON_VERSION_FIELDS R"(,
      "entries": [{
        "name": "main",
        "workgroup_size": [1, 1, 1],
        "dispatch_contract": {"unit_grid_axes": [], "requires_unit_workgroup": false},
        "physical_layouts": {
          "vulkan_std430_storage_buffer": {
            "profile":"vulkan_std430_storage_buffer","packing":"resource_bindings"
          }
        },
        "arguments": [{
          "kind":"image","resource_kind":"image","binding_role":"storage","access":"write",
          "dimension":"2d","exact_storage_format":"rgba32_float",
          "physical_layouts":{"vulkan_std430_storage_buffer":{
            "profile":"vulkan_std430_storage_buffer","kind":"resource_binding",
            "resource_kind":"image_reference"}},
          "vernon.binding":0
        }]
      }]
    })";
    VernonStageExecutable *pipeline =
        vernonRuntimeLoadArtifact(gl, source, sizeof(source) - 1, reflection, sizeof(reflection) - 1, "main", 4);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_IMAGE;
    argument.image.view = imageView.reference;
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {1, 1, 1};
    EXPECT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);

    vernonRuntimeStageExecutableDestroy(pipeline);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, imageView.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, image.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, RejectsComputeSampledTextureAtPipelinePreparation) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);

    static constexpr char source[] = "#version 430\nlayout(local_size_x=1) in;"
                                     "layout(binding=0) uniform sampler2D inputImage;"
                                     "void main(){vec4 value=texture(inputImage,vec2(0.5));}";
    static constexpr char reflection[] = "{" VERNON_JSON_VERSION_FIELDS R"(,
      "entries": [{
        "name": "main",
        "workgroup_size": [1, 1, 1],
        "dispatch_contract": {"unit_grid_axes": [], "requires_unit_workgroup": false},
        "physical_layouts": {
          "vulkan_std430_storage_buffer": {
            "profile":"vulkan_std430_storage_buffer","packing":"resource_bindings"
          }
        },
        "arguments": [{
          "kind":"image","resource_kind":"image","binding_role":"sampled",
          "sample_result_class":"float","access":"read","dimension":"2d",
          "physical_layouts":{"vulkan_std430_storage_buffer":{
            "profile":"vulkan_std430_storage_buffer","kind":"resource_binding",
            "resource_kind":"image_reference"}},
          "vernon.binding":0
        }]
      }]
    })";
    EXPECT_EQ(vernonRuntimeLoadArtifact(gl, source, sizeof(source) - 1, reflection, sizeof(reflection) - 1, "main", 4),
              nullptr);
    const VernonStringView error = vernonRuntimeGetLastError(gl);
    EXPECT_NE(std::string(error.data, error.size).find("compute pipelines do not support sampled textures"),
              std::string::npos);
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
    const auto &artifactSystem = manifest["variants"][0]["artifact_system"];
    const auto &artifact = artifactSystem["artifacts"].begin().value();
    const auto fragmentModule =
        std::find_if(artifact["modules"].begin(), artifact["modules"].end(),
                     [](const nlohmann::json &module) { return module.value("role", "") == "fragment"; });
    ASSERT_NE(fragmentModule, artifact["modules"].end());
    const std::string blob = (*fragmentModule)["blob"].get<std::string>();
    const std::filesystem::path shaderPath =
        manifestPath.parent_path() / manifest["blobs"][blob]["location"]["uri"].get<std::string>();
    std::ifstream shaderInput(shaderPath, std::ios::binary);
    const std::string shaderSource((std::istreambuf_iterator<char>(shaderInput)), std::istreambuf_iterator<char>());
    EXPECT_NE(shaderSource.find("uniform int max_steps;"), std::string::npos);
    EXPECT_NE(shaderSource.find("uniform vec2 _vernon_resolution;"), std::string::npos);

    resolutionUpload.fill(0.0F);
    signedUniformUpload.fill(0);
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string directory = manifestPath.parent_path().u8string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(gl, bundleData.data(), bundleData.size(), &options);
    ASSERT_TRUE(bundle) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

    constexpr std::array<float, 6> positions{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
    auto vertices = vernon::tests::createBuffer(rhiRuntime(gl), sizeof(positions), alignof(float),
                                                VERNON_RHI_BUFFER_VERTEX, positions.data());
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 37, 23, VERNON_TEXTURE_RGBA8_UNORM);
    const auto renderTargetView = vernon::tests::createImageView(rhiRuntime(gl), renderTarget, VERNON_RHI_IMAGE_2D,
                                                                 VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    constexpr GlInt maxSteps = 19;
    constexpr std::array<uint64_t, 2> positionShape{3, 2};
    constexpr std::array<int64_t, 2> positionStrides{2 * sizeof(float), sizeof(float)};
    VernonProgramParameterView maxStepsParameter{};
    VernonProgramParameterView positionParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"max_steps", 9}, &maxStepsParameter),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {"position", 8}, &positionParameter),
              VERNON_STATUS_OK);
    std::array<VernonProgramArgument, 2> arguments{};
    arguments[0].slot = maxStepsParameter.slot;
    arguments[0].kind = VERNON_PROGRAM_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_HOST;
    arguments[0].tensor.host_data = &maxSteps;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
    arguments[0].tensor.access = VERNON_ACCESS_READ;
    arguments[0].tensor.byte_size = sizeof(maxSteps);
    arguments[1].slot = positionParameter.slot;
    arguments[1].kind = VERNON_PROGRAM_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[1].tensor.resource = vertices.reference;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.rank = positionShape.size();
    arguments[1].tensor.shape = positionShape.data();
    arguments[1].tensor.byte_strides = positionStrides.data();
    arguments[1].tensor.byte_size = sizeof(positions);
    VernonColorAttachment attachment{0, renderTargetView.reference};
    const vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline, arguments.data(), arguments.size(), graphics),
              VERNON_STATUS_OK)
        << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{37.0F, 23.0F}));
    EXPECT_EQ(signedUniformUpload[0], maxSteps);

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, renderTargetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
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
    auto serialize = [](const nlohmann::json &root) { return root.dump(-1, ' ', false); };

    nlohmann::json legacy = nlohmann::json::parse(internalValueFixture());
    nlohmann::json &legacyUse = legacy["variants"][0]["internal_parameters"][0]["uses"][0];
    legacyUse.erase("sampled_image_bindings");
    legacyUse["sampled_texture_set"] = 0;
    legacyUse["sampled_texture_binding"] = 3;
    std::string bundleData = serialize(std::move(legacy));
    EXPECT_FALSE(loadDirectGraphicsFixture(gl, bundleData));

    nlohmann::json ambiguous = nlohmann::json::parse(internalValueFixture());
    ambiguous["variants"][0]["internal_parameters"][0]["uses"][0]["sampled_image_bindings"].push_back(
        {{"set", 0}, {"binding", 4}});
    bundleData = serialize(std::move(ambiguous));
    EXPECT_FALSE(loadDirectGraphicsFixture(gl, bundleData));

    nlohmann::json invalidResolution = nlohmann::json::parse(internalValueFixture());
    invalidResolution["variants"][0]["internal_parameters"][1]["uses"][0]["sampled_image_bindings"] =
        nlohmann::json::array({{{"set", 0}, {"binding", 3}}});
    bundleData = serialize(std::move(invalidResolution));
    EXPECT_FALSE(loadDirectGraphicsFixture(gl, bundleData));

    nlohmann::json nonzeroSet = nlohmann::json::parse(internalValueFixture());
    nonzeroSet["variants"][0]["parameters"][0]["uses"][0]["vernon.set"] = 1;
    nonzeroSet["variants"][0]["internal_parameters"][0]["uses"][0]["sampled_image_bindings"][0]["set"] = 1;
    bundleData = serialize(std::move(nonzeroSet));
    EXPECT_FALSE(loadDirectGraphicsFixture(gl, bundleData));
    const VernonStringView error = vernonRuntimeGetLastError(gl);
    EXPECT_NE(std::string_view(error.data, error.size).find("descriptor set 0"), std::string_view::npos);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, CreatesIdentityImageViewsWithoutTextureViewExtension) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 3, 3, false);
    ASSERT_TRUE(gl);
    RhiImage image = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    VernonRhiImageViewDescriptor descriptor = fullImageView(image, VERNON_RHI_FORMAT_RGBA8_UNORM);
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(rhiRuntime(gl).device, &descriptor, &view), VERNON_RHI_STATUS_OK);
    uint64_t parentNative = 0;
    uint64_t native = 0;
    ASSERT_EQ(vernonRhiDeviceGetImageNativeHandle(rhiRuntime(gl).device, image.handle, &parentNative),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, image.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsImageValid(rhiRuntime(gl).device, image.handle), 0u);
    EXPECT_EQ(vernonRhiDeviceGetImageViewNativeHandle(rhiRuntime(gl).device, view, &native), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(native, parentNative);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, RejectsAliasedImageViewsWithoutTextureViewExtension) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 3, 3, false);
    ASSERT_TRUE(gl);
    RhiImage image = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    VernonRhiImageViewDescriptor descriptor = fullImageView(image, VERNON_RHI_FORMAT_RGBA8_SRGB);
    VernonRhiImageView view{};
    EXPECT_EQ(vernonRhiDeviceCreateImageView(rhiRuntime(gl).device, &descriptor, &view), VERNON_RHI_STATUS_UNSUPPORTED);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, image.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, CreatesNativeAliasedImageViewsWithTextureViewExtension) {
    const uint32_t callsBefore = textureViewCount;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    RhiImage image = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    VernonRhiImageViewDescriptor descriptor = fullImageView(image, VERNON_RHI_FORMAT_RGBA8_SRGB);
    VernonRhiImageView view{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(rhiRuntime(gl).device, &descriptor, &view), VERNON_RHI_STATUS_OK);
    uint64_t parentNative = 0;
    uint64_t viewNative = 0;
    ASSERT_EQ(vernonRhiDeviceGetImageNativeHandle(rhiRuntime(gl).device, image.handle, &parentNative),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceGetImageViewNativeHandle(rhiRuntime(gl).device, view, &viewNative), VERNON_RHI_STATUS_OK);
    EXPECT_NE(viewNative, parentNative);
    EXPECT_EQ(textureViewCount, callsBefore + 1);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, view), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, image.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, SuppliesImplicitSamplerAndEffectiveResolution) {
    resolutionUpload.fill(0.0F);
    viewportUpload.fill(0);
    boundSamplerUnit = UINT32_MAX;
    boundSamplerName = UINT32_MAX;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string bundleData = internalValueFixture();
    VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    ASSERT_EQ(vernonRuntimeStageExecutableGetParameterCount(pipeline), 1u);

    RhiImage sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    VernonRhiImageViewDescriptor sampledViewDescriptor{};
    sampledViewDescriptor.struct_size = sizeof(sampledViewDescriptor);
    sampledViewDescriptor.image = sampled.handle;
    sampledViewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    sampledViewDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    sampledViewDescriptor.mip_level_count = 1;
    sampledViewDescriptor.array_layer_count = 1;
    sampledViewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    VernonRhiImageView sampledView{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(rhiRuntime(gl).device, &sampledViewDescriptor, &sampledView),
              VERNON_RHI_STATUS_OK);
    VernonRuntimeProviderResourceReference sampledViewReference{};
    ASSERT_EQ(vernonRuntimeReferenceRhiImageView(gl, sampledView, &sampledViewReference), VERNON_STATUS_OK);
    const auto targetView =
        vernon::tests::createImageView(rhiRuntime(gl), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    auto textureSampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(textureSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_IMAGE;
    argument.image = {sampledViewReference};
    VernonColorAttachment attachment{};
    attachment.location = 0;
    attachment.view = targetView.reference;
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3);
    graphics.dynamic.viewport[0] = 2;
    graphics.dynamic.viewport[1] = 3;
    graphics.dynamic.viewport[2] = 7;
    graphics.dynamic.viewport[3] = 9;
    graphics.bind(invocation);
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
    ASSERT_EQ(vernonRuntimeStageEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(boundSamplerUnit, 3u);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{7.0F, 9.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{2, 3, 7, 9}));
    EXPECT_EQ(nextName, nextNameAfterPreparation);

    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, textureSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, sampledView), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    RhiImage replacementSampled = importOpenGLTexture2D(gl, 9, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    auto replacementSampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(replacementSampled.handle.index, sampled.handle.index);
    ASSERT_EQ(replacementSampler.handle.index, textureSampler.handle.index);
    ASSERT_NE(replacementSampler.handle.generation, textureSampler.handle.generation);

    std::fill(std::begin(graphics.dynamic.viewport), std::end(graphics.dynamic.viewport), 0);
    ASSERT_EQ(vernonRuntimeStageEncode(providerEncoder, pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{16.0F, 12.0F}));
    EXPECT_EQ(viewportUpload, (std::array<GlInt, 4>{0, 0, 16, 12}));
    EXPECT_EQ(boundSamplerName, 0u);

    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    VernonRhiCommandEncoderStats encoderStats{};
    vernonRuntimeStageExecutableDestroy(pipeline);
    pipeline = nullptr;
    ASSERT_EQ(vernonRhiCommandEncoderFinish(rhiRuntime(gl).device, encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernon::tests::completeSubmission(rhiRuntime(gl).device, encoder, &encoderStats), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(encoderStats.rendering_scope_count, 1u);
    EXPECT_EQ(encoderStats.draw_count, 2u);
    EXPECT_EQ(encoderStats.submission_count, 1u);
    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, replacementSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, replacementSampled.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
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
    const std::string bundleData = internalValueFixture();
    VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    RhiImage sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    const auto sampledView =
        vernon::tests::createImageView(rhiRuntime(gl), sampled, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    auto sampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(sampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_IMAGE;
    argument.image = {sampledView.reference};
    VernonRhiImageViewDescriptor targetViewDescriptor = fullImageView(target, VERNON_RHI_FORMAT_RGBA8_UNORM);
    VernonRhiImageView targetView{};
    ASSERT_EQ(vernonRhiDeviceCreateImageView(rhiRuntime(gl).device, &targetViewDescriptor, &targetView),
              VERNON_RHI_STATUS_OK);
    VernonColorAttachment attachment{};
    attachment.location = 0;
    ASSERT_EQ(vernonRuntimeReferenceRhiImageView(gl, targetView, &attachment.view), VERNON_STATUS_OK);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3);
    graphics.bind(invocation);

    {
        invalidateCount = 0;
        lastMemoryBarrierBits = 0;
        framebufferBindCount = 0;
        const size_t commandsBeforeGraph = vernon::runtime::getRhiAdapterRecordedCommandCount(gl);
        vernon::execution::ExecutionGraph graph(rhiRuntime(gl).device);
        const auto graphTarget = graph.importImage(target.handle, targetView, true);
        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceIsImageValid(rhiRuntime(gl).device, target.handle), 0u);
        RhiImage replacement = importOpenGLTexture2D(gl, 11, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
        EXPECT_NE(replacement.handle.index, target.handle.index);
        graph.emplacePass<GraphImageWritePass>("produce", graphTarget);
        graph.emplacePass<vernon::tests::DirectStageGraphRenderPass>("first", graphTarget, gl, pipeline, &invocation,
                                                                     VERNON_RHI_LOAD_CLEAR);
        graph.emplacePass<vernon::tests::DirectStageGraphRenderPass>(
            "second", graphTarget, gl, pipeline, &invocation, VERNON_RHI_LOAD_PRESERVE, VERNON_RHI_STORE_DISCARD);
        std::string error;
        auto plan = graph.compile(error);
        ASSERT_TRUE(plan) << error;
        auto submission = plan->submit();
        ASSERT_EQ(submission.wait(), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(submission.commandStats().rendering_scope_count, 1u);
        EXPECT_EQ(submission.commandStats().barrier_count, 1u);
        EXPECT_EQ(submission.commandStats().draw_count, 2u);
        EXPECT_EQ(submission.commandStats().submission_count, 1u);
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
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, sampledView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, BindsExplicitSamplerSeparately) {
    boundSamplerName = UINT32_MAX;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const std::string bundleData = explicitSamplerFixture();
    VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
    ASSERT_EQ(vernonRuntimeStageExecutableGetParameterCount(pipeline), 2u);

    RhiImage sampled = importOpenGLTexture2D(gl, 8, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 12, VERNON_TEXTURE_RGBA8_UNORM);
    const auto sampledView =
        vernon::tests::createImageView(rhiRuntime(gl), sampled, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    const auto targetView =
        vernon::tests::createImageView(rhiRuntime(gl), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    auto explicitSampler = vernon::tests::createSampler(rhiRuntime(gl));
    ASSERT_NE(sampled.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(explicitSampler.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    VernonProgramArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PROGRAM_IMAGE;
    arguments[0].image = {sampledView.reference};
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PROGRAM_SAMPLER;
    arguments[1].resource = explicitSampler.reference;
    VernonColorAttachment attachment{0, targetView.reference};
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = std::size(arguments);
    vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3);
    graphics.bind(invocation);
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_NE(boundSamplerName, 0u);

    ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, explicitSampler.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, sampledView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
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
    const std::string bundleData = vertexInputFixture();
    VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

    auto vertices = vernon::tests::createBuffer(rhiRuntime(gl), 36, alignof(float), VERNON_RHI_BUFFER_VERTEX);
    auto indices = vernon::tests::createBuffer(rhiRuntime(gl), 12, alignof(uint32_t), VERNON_RHI_BUFFER_INDEX);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    const auto targetView =
        vernon::tests::createImageView(rhiRuntime(gl), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(indices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[2] = {3, 3};
    constexpr int64_t strides[2] = {3 * sizeof(float), sizeof(float)};
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
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
    const VernonColorAttachment attachment{0, targetView.reference};
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3, 3);
    graphics.draw.index_binding = &indexBinding;
    graphics.bind(invocation);
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK)
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

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, indices.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
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
        const std::string bundleData = vertexInputFixture(format.name);
        VernonStageExecutable *pipeline = loadDirectGraphicsFixture(gl, bundleData);
        ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);

        const uint32_t stride = 3 * format.size;
        auto vertices = vernon::tests::createBuffer(rhiRuntime(gl), 3 * stride, format.size, VERNON_RHI_BUFFER_VERTEX);
        RhiImage target = importOpenGLTexture2D(gl, 8, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
        const auto targetView =
            vernon::tests::createImageView(rhiRuntime(gl), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
        ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        const uint64_t shape[2] = {3, 3};
        const int64_t strides[2] = {stride, format.size};
        VernonProgramArgument argument{};
        argument.kind = VERNON_PROGRAM_TENSOR;
        argument.tensor.struct_size = sizeof(VernonTensorView);
        argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        argument.tensor.resource = vertices.reference;
        argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(format.dtype);
        argument.tensor.access = VERNON_ACCESS_READ;
        argument.tensor.rank = 2;
        argument.tensor.shape = shape;
        argument.tensor.byte_strides = strides;
        argument.tensor.byte_size = 3 * stride;
        const VernonColorAttachment attachment{0, targetView.reference};
        VernonStageInvocationDescriptor invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PROGRAM_VERSION;
        invocation.arguments = &argument;
        invocation.argument_count = 1;
        vernon::tests::GraphicsInvocationControls graphics(&attachment, 1, 3, 3);
        graphics.bind(invocation);
        ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK)
            << std::string(vernonRuntimeGetLastError(gl).data, vernonRuntimeGetLastError(gl).size);
        EXPECT_EQ(vertexAttributeType, format.nativeType);
        EXPECT_EQ(vertexAttributePointerKind, format.pointerKind);
        EXPECT_EQ(vertexAttributeStride, static_cast<GlSize>(stride));

        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
        vernonRuntimeStageExecutableDestroy(pipeline);
        ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
    }

    for (const char *dtype : {"bool", "u8"}) {
        SCOPED_TRACE(dtype);
        VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
        ASSERT_TRUE(gl);
        const std::string bundleData = vertexInputFixture(dtype);
        EXPECT_EQ(loadDirectGraphicsFixture(gl, bundleData), nullptr);
        const VernonStringView error = vernonRuntimeGetLastError(gl);
        EXPECT_NE(std::string_view(error.data, error.size).find("format capabilities"), std::string_view::npos);
        ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
    }
}

#undef GL_CALL
