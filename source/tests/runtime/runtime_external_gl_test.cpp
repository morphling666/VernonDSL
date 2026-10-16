#include "VernonRuntime.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/runtime_test_hooks.h"
#include "runtime_rhi_test_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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
constexpr GlEnum kMaxUniformBlockSize = 0x8A30;
constexpr GlEnum kNumExtensions = 0x821D;
constexpr GlEnum kActiveUniformBlocks = 0x8A36;
constexpr GlEnum kInt = 0x1404;
constexpr GlEnum kUnsignedInt = 0x1405;
constexpr GlEnum kFloat = 0x1406;
constexpr GlEnum kBack = 0x0405;
constexpr GlEnum kClockwise = 0x0900;
constexpr GlEnum kCounterClockwise = 0x0901;
constexpr GlEnum kLessEqual = 0x0203;
constexpr GlEnum kOne = 1;
constexpr GlEnum kSourceAlpha = 0x0302;
constexpr GlEnum kOneMinusSourceAlpha = 0x0303;
constexpr GlEnum kAdd = 0x8006;
constexpr GlEnum kSubtract = 0x800A;
constexpr GlEnum kColorAttachment0 = 0x8CE0;
constexpr GlEnum kStencilAttachment = 0x8D20;
constexpr GlEnum kAlreadySignaled = 0x911A;
constexpr GlEnum kTimeoutExpired = 0x911B;
constexpr GlEnum kMaxDebugGroupStackDepth = 0x826C;
constexpr GlEnum kDebugGroupStackDepth = 0x826D;
constexpr GlEnum kMaxDebugMessageLength = 0x9143;
constexpr GlEnum kTimestamp = 0x8E28;
constexpr GlEnum kQueryResult = 0x8866;
constexpr GlEnum kGpuDisjoint = 0x8FBB;

GlUint nextName = 1;
uint32_t drawCount = 0;
uint32_t dispatchCount = 0;
uint32_t storageBindingCount = 0;
uint32_t memoryBarrierCount = 0;
uint32_t lastMemoryBarrierBits = 0;
uint32_t createdSyncCount = 0;
uint32_t deletedSyncCount = 0;
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
std::vector<std::string> linkedUniformBlocks;
std::unordered_map<GlUint, GlUint> uniformBlockBindings;
std::vector<std::string> exposedExtensions;
constexpr GlInt kDefaultMaxDebugGroupDepth = 64;
constexpr GlInt kDefaultMaxDebugMessageLength = 256;
GlInt maxDebugGroupDepth = kDefaultMaxDebugGroupDepth;
GlInt maxDebugMessageLength = kDefaultMaxDebugMessageLength;
// The default debug group of the context counts toward the depth.
GlInt debugGroupDepth = 1;
// Debug label calls, timestamp queries and draws, in call order.
std::vector<std::string> orderedCalls;
// Each timestamp query advances the GPU clock by ten nanoseconds from an origin that is not zero.
constexpr uint64_t kGpuClockOrigin = 1'000'000;
uint64_t gpuClock = kGpuClockOrigin;
std::unordered_map<GlUint, uint64_t> queryTimestamps;
std::vector<GlUint> deletedQueries;
// Reading the flag clears it, like GL_GPU_DISJOINT_EXT.
GlInt gpuDisjoint = 0;
bool exposeTimerQueries = true;

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
GlEnum pendingError = 0;
GlEnum generatedMipmapError = 0;
uint32_t generatedMipmapCount = 0;

void makeCurrent(void *) { ++makeCurrentCount; }
void GL_CALL objectNoop(GlUint) {}
GlEnum GL_CALL getError() {
    const GlEnum result = pendingError;
    pendingError = 0;
    return result;
}
void GL_CALL generateMipmap(GlEnum) {
    ++generatedMipmapCount;
    pendingError = generatedMipmapError;
}
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
void GL_CALL blitFramebufferNoop(GlInt, GlInt, GlInt, GlInt, GlInt, GlInt, GlInt, GlInt, unsigned, GlEnum) {}
void GL_CALL drawBuffersNoop(GlSize, const GlEnum *) {}
void GL_CALL drawArraysInstanced(GlEnum, GlInt, GlSize, GlSize) {
    ++drawCount;
    orderedCalls.emplace_back("draw");
}
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
void GL_CALL getProgramiv(GlUint, GlEnum name, GlInt *value) {
    *value = name == kLinkStatus            ? 1
             : name == kActiveUniformBlocks ? static_cast<GlInt>(linkedUniformBlocks.size())
                                            : 0;
}
void GL_CALL getIntegerv(GlEnum name, GlInt *value) {
    switch (name) {
    case kMaxUniformBlockSize:
        *value = 16 * 1024;
        break;
    case kNumExtensions:
        *value = static_cast<GlInt>(exposedExtensions.size());
        break;
    case kMaxDebugGroupStackDepth:
        *value = maxDebugGroupDepth;
        break;
    case kDebugGroupStackDepth:
        *value = debugGroupDepth;
        break;
    case kMaxDebugMessageLength:
        *value = maxDebugMessageLength;
        break;
    case kGpuDisjoint:
        *value = std::exchange(gpuDisjoint, 0);
        break;
    default:
        *value = 16;
    }
}
const unsigned char *GL_CALL getStringi(GlEnum, GlUint index) {
    return index < exposedExtensions.size() ? reinterpret_cast<const unsigned char *>(exposedExtensions[index].c_str())
                                            : nullptr;
}
void GL_CALL pushDebugGroup(GlEnum, GlUint, GlSize length, const char *message) {
    ++debugGroupDepth;
    orderedCalls.push_back("push " + std::string(message, static_cast<size_t>(length)));
}
void GL_CALL popDebugGroup() {
    --debugGroupDepth;
    orderedCalls.emplace_back("pop");
}
void GL_CALL debugMessageInsert(GlEnum, GlEnum, GlUint, GlEnum, GlSize length, const char *message) {
    orderedCalls.push_back("marker " + std::string(message, static_cast<size_t>(length)));
}
void GL_CALL pushDebugGroupKhr(GlEnum source, GlUint id, GlSize length, const char *message) {
    orderedCalls.emplace_back("KHR");
    pushDebugGroup(source, id, length, message);
}
void GL_CALL genNames(GlSize count, GlUint *names) {
    while (count--)
        *names++ = nextName++;
}
void GL_CALL deleteQueries(GlSize count, const GlUint *names) {
    std::copy_if(names, names + count, std::back_inserter(deletedQueries), [](GlUint name) { return name != 0; });
}
void GL_CALL queryCounter(GlUint name, GlEnum target) {
    if (target != kTimestamp)
        return;
    gpuClock += 10;
    queryTimestamps[name] = gpuClock;
    orderedCalls.emplace_back("timestamp");
}
void GL_CALL getQueryObjectui64v(GlUint name, GlEnum parameter, uint64_t *value) {
    *value = parameter == kQueryResult ? queryTimestamps.at(name) : 0;
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
void GL_CALL bindBufferRange(GlEnum, GlUint, GlUint, std::intptr_t, std::intptr_t) { ++storageBindingCount; }
void GL_CALL bindImageTexture(GlUint, GlUint, GlInt, GlBoolean, GlInt, GlEnum, GlEnum) { ++storageBindingCount; }
void GL_CALL dispatchCompute(GlUint, GlUint, GlUint) { ++dispatchCount; }
void GL_CALL memoryBarrier(unsigned bits) {
    ++memoryBarrierCount;
    lastMemoryBarrierBits = bits;
}
void *GL_CALL fenceSync(GlEnum, unsigned) {
    ++createdSyncCount;
    return &createdSyncCount;
}
// While set, polls report fences unsignaled and only blocking waits complete them.
bool fencesPending = false;
GlEnum GL_CALL clientWaitSync(void *, unsigned, uint64_t timeout) {
    return fencesPending && !timeout ? kTimeoutExpired : kAlreadySignaled;
}
void GL_CALL deleteSync(void *) { ++deletedSyncCount; }
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
    orderedCalls.emplace_back("draw");
}
void GL_CALL bufferSubData(GlEnum, std::intptr_t offset, std::intptr_t size, const void *source) {
    std::memcpy(bufferStorage.data() + offset, source, static_cast<size_t>(size));
}
void GL_CALL copyImageSubData(GlUint, GlEnum, GlInt, GlInt, GlInt, GlInt, GlUint, GlEnum, GlInt, GlInt, GlInt, GlInt,
                              GlSize, GlSize, GlSize) {}
void *GL_CALL mapBufferRange(GlEnum, std::intptr_t offset, std::intptr_t, unsigned) {
    return bufferStorage.data() + offset;
}
GlBoolean GL_CALL unmapBuffer(GlEnum) { return 1; }
void initializeTextureStorage(GlSize width, GlSize height, bool depthStencil) {
    textureStorage.resize(static_cast<size_t>(width) * height * (depthStencil ? 8 : 4));
    if (depthStencil)
        for (size_t offset = 0; offset < textureStorage.size(); offset += 8) {
            constexpr float depth = 0.25f;
            constexpr uint32_t stencil = 7;
            std::memcpy(textureStorage.data() + offset, &depth, sizeof(depth));
            std::memcpy(textureStorage.data() + offset + sizeof(depth), &stencil, sizeof(stencil));
        }
}
void GL_CALL texImage2D(GlEnum, GlInt, GlInt, GlSize width, GlSize height, GlInt, GlEnum, GlEnum type, const void *) {
    initializeTextureStorage(width, height, type == 0x8DAD);
}
void GL_CALL texStorage2D(GlEnum, GlSize, GlEnum internalFormat, GlSize width, GlSize height) {
    initializeTextureStorage(width, height, internalFormat == 0x8CAD);
}
void GL_CALL texStorage3D(GlEnum, GlSize, GlEnum, GlSize width, GlSize height, GlSize depth) {
    textureStorage.resize(static_cast<size_t>(width) * height * depth * 4);
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
void GL_CALL drawArrays(GlEnum, GlInt, GlSize) {
    ++drawCount;
    orderedCalls.emplace_back("draw");
}
void GL_CALL clearBufferfv(GlEnum, GlInt drawBuffer, const float *value) {
    ++clearCount;
    clearedDrawBuffer = drawBuffer;
    std::copy_n(value, clearColor.size(), clearColor.begin());
}
void GL_CALL clearBufferiv(GlEnum, GlInt, const GlInt *) {}
void GL_CALL clearBufferuiv(GlEnum, GlInt, const GlUint *) {}
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
GlUint GL_CALL getUniformBlockIndex(GlUint, const char *name) {
    const auto found = std::find(linkedUniformBlocks.begin(), linkedUniformBlocks.end(), name);
    return found == linkedUniformBlocks.end() ? 0xFFFFFFFFU : static_cast<GlUint>(found - linkedUniformBlocks.begin());
}
void GL_CALL uniformBlockBinding(GlUint, GlUint block, GlUint binding) { uniformBlockBindings[block] = binding; }
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
    PROC("glGetError", getError);
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
    PROC("glGetStringi", getStringi);
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
    PROC("glBindBufferRange", bindBufferRange);
    PROC("glBindImageTexture", bindImageTexture);
    PROC("glDeleteBuffers", deleteBufferNames);
    PROC("glBufferData", bufferData);
    PROC("glBufferSubData", bufferSubData);
    PROC("glCopyImageSubData", copyImageSubData);
    PROC("glMapBufferRange", mapBufferRange);
    PROC("glUnmapBuffer", unmapBuffer);
    PROC("glGenTextures", genNames);
    PROC("glDeleteTextures", deleteTextureNames);
    PROC("glBindTexture", objectPairNoop);
    PROC("glTextureView", textureView);
    PROC("glTexImage2D", texImage2D);
    PROC("glTexImage3D", texImage3DNoop);
    PROC("glTexStorage2D", texStorage2D);
    PROC("glTexStorage3D", texStorage3D);
    PROC("glTexSubImage2D", texSubImage2D);
    PROC("glTexSubImage3D", texSubImage3DNoop);
    PROC("glTexParameteri", enumIntNoop);
    PROC("glGenerateMipmap", generateMipmap);
    PROC("glPixelStorei", enumIntNoop);
    PROC("glReadPixels", readPixels);
    PROC("glGenSamplers", genNames);
    PROC("glDeleteSamplers", deleteSamplerNames);
    PROC("glSamplerParameteri", enumIntNoop);
    PROC("glCheckFramebufferStatus", framebufferStatus);
    PROC("glBindFramebuffer", bindFramebuffer);
    PROC("glFramebufferTexture2D", framebufferTexture2DNoop);
    PROC("glFramebufferTextureLayer", framebufferTextureLayer);
    PROC("glBlitFramebuffer", blitFramebufferNoop);
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
    PROC("glClearBufferuiv", clearBufferuiv);
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
    PROC("glGetUniformBlockIndex", getUniformBlockIndex);
    PROC("glUniformBlockBinding", uniformBlockBinding);
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
    PROC("glFenceSync", fenceSync);
    PROC("glClientWaitSync", clientWaitSync);
    PROC("glDeleteSync", deleteSync);
    PROC("glPushDebugGroup", pushDebugGroup);
    PROC("glPopDebugGroup", popDebugGroup);
    PROC("glDebugMessageInsert", debugMessageInsert);
    PROC("glPushDebugGroupKHR", pushDebugGroupKhr);
    PROC("glPopDebugGroupKHR", popDebugGroup);
    PROC("glDebugMessageInsertKHR", debugMessageInsert);
    if (exposeTimerQueries) {
        PROC("glGenQueries", genNames);
        PROC("glDeleteQueries", deleteQueries);
        PROC("glQueryCounter", queryCounter);
        PROC("glQueryCounterEXT", queryCounter);
        PROC("glGetQueryObjectui64v", getQueryObjectui64v);
        PROC("glGetQueryObjectui64vEXT", getQueryObjectui64v);
    }
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

std::string_view lastError(VernonRuntimeContext *gl) {
    const VernonStringView error = vernonRuntimeGetLastError(gl);
    return {error.data, error.size};
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

void expectMatrixUpload(VernonRuntimeBackend backend, const char *manifest, uint16_t major, uint16_t minor,
                        const std::array<float, 16> &storage, const std::array<int64_t, 2> &strides,
                        GlBoolean expectedTranspose, const std::array<float, 16> &expectedUpload) {
    matrixTranspose = expectedTranspose ? 0 : 1;
    matrixUpload.fill(0.0F);
    clearCount = 0;
    clearedDrawBuffer = -1;
    clearColor.fill(1.0F);
    VernonRuntimeContext *gl = create(backend, major, minor);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, manifest);
        ASSERT_TRUE(pipeline) << lastError(gl);
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
        vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
        const GlUint nextNameAfterPreparation = nextName;
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), &argument, 1, graphics), VERNON_STATUS_OK)
            << lastError(gl);
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), &argument, 1, graphics), VERNON_STATUS_OK)
            << lastError(gl);
        EXPECT_EQ(nextName, nextNameAfterPreparation);
        EXPECT_EQ(matrixTranspose, expectedTranspose);
        EXPECT_EQ(matrixUpload, expectedUpload);
        EXPECT_EQ(clearCount, 2U);
        EXPECT_EQ(clearedDrawBuffer, 0);
        EXPECT_EQ(clearColor, (std::array<float, 4>{}));

        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, renderTargetView.handle),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    }
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

void expectIntegerUniformUpload(const char *manifest, const char *name, VernonDataType dataType, const void *data,
                                uint32_t components) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, manifest);
        ASSERT_TRUE(pipeline) << lastError(gl);
        RhiImage renderTarget = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
        const auto renderTargetView = vernon::tests::createImageView(rhiRuntime(gl), renderTarget, VERNON_RHI_IMAGE_2D,
                                                                     VERNON_RHI_FORMAT_RGBA8_UNORM);
        ASSERT_NE(renderTarget.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

        const uint64_t shape = components;
        const int64_t stride = sizeof(uint32_t);
        VernonProgramParameterView parameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline.get(), {name, std::strlen(name)}, &parameter),
                  VERNON_STATUS_OK);
        VernonProgramArgument argument{};
        argument.slot = parameter.slot;
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
        vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), &argument, 1, graphics), VERNON_STATUS_OK);

        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, renderTargetView.handle),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, renderTarget.handle), VERNON_RHI_STATUS_OK);
    }
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

// Records groups "a" and "b", a group "c" past a depth limit of three, and two markers with a five-byte message limit,
// then returns the debug calls that reached GL.
std::vector<std::string> recordDebugLabels(VernonRuntimeBackend backend, uint16_t major, uint16_t minor,
                                           std::vector<std::string> extensions = {}) {
    exposedExtensions = std::move(extensions);
    maxDebugGroupDepth = 3;
    maxDebugMessageLength = 5;
    debugGroupDepth = 1;
    orderedCalls.clear();
    VernonRuntimeContext *gl = create(backend, major, minor);
    exposedExtensions.clear();
    maxDebugGroupDepth = kDefaultMaxDebugGroupDepth;
    maxDebugMessageLength = kDefaultMaxDebugMessageLength;
    EXPECT_TRUE(gl);
    if (!gl)
        return {};
    const VernonRhiDevice device = rhiRuntime(gl).device;
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    const auto push = [&](std::string_view label) {
        EXPECT_EQ(vernonRhiCommandEncoderPushDebugGroup(device, encoder, {label.data(), label.size()}),
                  VERNON_RHI_STATUS_OK);
    };
    const auto marker = [&](std::string_view label) {
        EXPECT_EQ(vernonRhiCommandEncoderInsertDebugMarker(device, encoder, {label.data(), label.size()}),
                  VERNON_RHI_STATUS_OK);
    };
    EXPECT_EQ(vernonRhiCommandEncoderPopDebugGroup(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    push("a");
    push("b");
    push("c");
    marker("abc\xC3\xA9");
    marker("d\xC3\xA9");
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    for (int group = 0; group < 3; ++group)
        EXPECT_EQ(vernonRhiCommandEncoderPopDebugGroup(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
    EXPECT_EQ(debugGroupDepth, 1);
    return orderedCalls;
}

VernonRhiCommandEncoder createGraphicsEncoder(VernonRhiDevice device) {
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    return encoder;
}

// Submits the finished `encoder`, waits for it and reads its timers into `spans`.
VernonRhiStatus submitAndReadTimers(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                    std::vector<VernonRhiTimerSpan> &spans) {
    VernonRhiCompletion completion{};
    VernonRhiStatus status = vernonRhiDeviceSubmit(device, encoder, &completion);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    status = vernonRhiCompletionWait(device, completion);
    size_t count = 0;
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCompletionGetTimers(device, completion, nullptr, 0, &count);
    if (status == VERNON_RHI_STATUS_OK && count) {
        spans.resize(count);
        EXPECT_EQ(vernonRhiCompletionGetTimers(device, completion, spans.data(), count - 1, &count),
                  VERNON_RHI_STATUS_INVALID_ARGUMENT);
        status = vernonRhiCompletionGetTimers(device, completion, spans.data(), spans.size(), &count);
    }
    const VernonRhiStatus destroyStatus = vernonRhiDeviceDestroyCompletion(device, completion);
    return status == VERNON_RHI_STATUS_OK ? destroyStatus : status;
}

#if defined(VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE)
// Executes `executable` with every graphics Node drawing one triangle into the same target, after `recording` replaces
// the draws of the first Node, and returns the debug label calls, timestamp queries and draws of that Execute. With
// `submission`, the Execute's submission is handed out.
std::vector<std::string> executeModuleGraphics(VernonRuntimeContext *gl, VernonProgramExecutable *executable,
                                               const std::vector<VernonGraphicsCommand> &recording = {},
                                               const VernonProgramInvocationOptions *options = nullptr,
                                               VernonSubmission **submission = nullptr) {
    RhiRuntime &runtime = rhiRuntime(gl);
    constexpr std::array<float, 6> positions{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    constexpr std::array<uint64_t, 2> shape{3, 2};
    constexpr std::array<int64_t, 2> strides{2 * sizeof(float), sizeof(float)};
    const vernon::tests::RhiBuffer vertices = vernon::tests::createBuffer(runtime, sizeof(positions), alignof(float),
                                                                          VERNON_RHI_BUFFER_VERTEX, positions.data());
    RhiImage target = createTexture2D(gl, 8, 8, VERNON_TEXTURE_RGBA8_UNORM);
    const auto view =
        vernon::tests::createImageView(runtime, target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(executable);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    EXPECT_NE(invocation, nullptr);
    if (options)
        EXPECT_EQ(vernonRuntimeProgramInvocationSetOptions(invocation, options), VERNON_STATUS_OK);

    for (size_t index = 0; index < vernonRuntimeProgramExecutableGetParameterCount(executable); ++index) {
        VernonProgramParameterView parameter{};
        EXPECT_EQ(vernonRuntimeProgramExecutableGetParameterByIndex(executable, index, &parameter), VERNON_STATUS_OK);
        VernonProgramArgument argument{};
        argument.slot = parameter.slot;
        if (std::string_view(parameter.name.data, parameter.name.size) == "output") {
            argument.kind = VERNON_PROGRAM_IMAGE;
            argument.image = {view.reference};
        } else {
            argument.kind = VERNON_PROGRAM_TENSOR;
            argument.tensor.struct_size = sizeof(argument.tensor);
            argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
            argument.tensor.resource = vertices.reference;
            argument.tensor.element_layout = parameter.element_layout;
            argument.tensor.access = parameter.access;
            argument.tensor.rank = shape.size();
            argument.tensor.shape = shape.data();
            argument.tensor.byte_strides = strides.data();
            argument.tensor.byte_size = sizeof(positions);
        }
        const std::string text = "argument-" + std::to_string(index);
        const VernonProgramBindingToken token{sizeof(token), text.data(), text.size()};
        EXPECT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &argument, nullptr, 0, 0), VERNON_STATUS_OK);
    }

    const size_t nodeCount = vernonRuntimeProgramExecutableGetGraphicsNodeCount(executable);
    std::vector<VernonColorAttachment> attachments(nodeCount);
    std::vector<vernon::tests::CanonicalGraphicsControls> controls;
    controls.reserve(nodeCount);
    for (size_t node = 0; node < nodeCount; ++node) {
        attachments[node].view = view.reference;
        attachments[node].load_operation =
            node == 0 ? VERNON_RUNTIME_PROVIDER_LOAD_CLEAR : VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
        attachments[node].store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        vernon::tests::CanonicalGraphicsControls &graphics = controls.emplace_back(&attachments[node], 1, 3);
        graphics.renderPass.render_area[2] = graphics.renderPass.render_area[3] = 8;
        graphics.dynamic.viewport[2] = graphics.dynamic.viewport[3] = 8;
        graphics.dynamic.scissor[2] = graphics.dynamic.scissor[3] = 8;
        VernonProgramGraphicsControlsView slots{};
        EXPECT_EQ(vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(executable, node, &slots), VERNON_STATUS_OK);
        EXPECT_EQ(graphics.bind(invocation, slots), VERNON_STATUS_OK);
        if (node == 0 && !recording.empty())
            EXPECT_EQ(vernonRuntimeProgramInvocationRecordGraphics(invocation, slots.node, recording.data(),
                                                                   recording.size(), nullptr, 0),
                      VERNON_STATUS_OK);
    }

    orderedCalls.clear();
    EXPECT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK) << lastError(gl);
    std::vector<std::string> calls = std::move(orderedCalls);
    EXPECT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK);
    if (submission)
        EXPECT_EQ(vernonRuntimeProgramInvocationTakeSubmission(invocation, submission), VERNON_STATUS_OK);
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(runtime.device, view.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(runtime.device, target.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(runtime.device, vertices.handle), VERNON_RHI_STATUS_OK);
    return calls;
}

// Resolves the graph of Nodes "scene" and "overlay", both drawing `bundle` into one exported framebuffer, so that they
// fuse into one render scope.
VernonProgramExecutable *resolveSceneOverlayGraph(VernonRuntimeContext *gl, VernonProgramBundle *bundle,
                                                  std::array<VernonProgramNodeId, 2> &nodes) {
    VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(gl);
    std::array<VernonProgramNodeBindingToken, 2> outputs{};
    VernonProgramGraphStorage framebuffer{sizeof(framebuffer)};
    bool built = true;
    for (size_t node = 0; node < nodes.size() && built; ++node) {
        const std::string_view label = node == 0 ? "scene" : "overlay";
        outputs[node].struct_size = sizeof(outputs[node]);
        VernonProgramNodeBindingToken vertices{sizeof(vertices)};
        const std::string boundary = std::string(label) + "_vertices";
        built = vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, {label.data(), label.size()},
                                                    &nodes[node]) == VERNON_STATUS_OK &&
                vernonRuntimeProgramGraphFindBoundary(graph, nodes[node], VERNON_PROGRAM_BOUNDARY_OUTPUT, {"output", 6},
                                                      &outputs[node]) == VERNON_STATUS_OK &&
                vernonRuntimeProgramGraphFindBoundary(graph, nodes[node], VERNON_PROGRAM_BOUNDARY_INPUT,
                                                      {"vertices", 8}, &vertices) == VERNON_STATUS_OK &&
                vernonRuntimeProgramGraphExportBoundary(graph, &vertices, {boundary.data(), boundary.size()}) ==
                    VERNON_STATUS_OK;
    }
    built = built && vernonRuntimeProgramGraphCreateStorage(graph, &outputs[0], &framebuffer) == VERNON_STATUS_OK &&
            vernonRuntimeProgramGraphAppendStorage(graph, &framebuffer, &outputs[1]) == VERNON_STATUS_OK &&
            vernonRuntimeProgramGraphExportStorage(graph, &framebuffer, {"output", 6}) == VERNON_STATUS_OK;
    EXPECT_TRUE(built);
    VernonProgramExecutable *executable = built ? vernonRuntimeResolveProgramGraph(graph) : nullptr;
    vernonRuntimeProgramGraphDestroy(graph);
    return executable;
}

VernonProgramInvocationOptions invocationOptions(VernonCompletionMode mode, uint32_t timings) {
    return {sizeof(VernonProgramInvocationOptions), VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION, mode, timings, {}};
}

// Reads the scope timings of a completed submission, checking that a short capacity is rejected.
std::vector<VernonScopeTiming> readScopeTimings(VernonRuntimeContext *gl, VernonSubmission *submission) {
    size_t count = 0;
    EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, nullptr, 0, &count), VERNON_STATUS_OK);
    std::vector<VernonScopeTiming> scopes(count);
    if (count) {
        EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, scopes.data(), count - 1, &count),
                  VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(lastError(gl), "scope timing storage is smaller than the timed scopes");
        EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, scopes.data(), scopes.size(), &count), VERNON_STATUS_OK);
        EXPECT_EQ(count, scopes.size());
    }
    return scopes;
}

VernonGraphicsCommand graphicsCommand(VernonGraphicsCommandKind kind, std::string_view label = {}) {
    VernonGraphicsCommand command{};
    command.kind = kind;
    command.label = {label.data(), label.size()};
    return command;
}
#endif

} // namespace

TEST(RuntimeExternalGl, ScopesMipmapErrorsAndProbesFloatFormatSupport) {
    pendingError = 0;
    generatedMipmapError = 0;
    generatedMipmapCount = 0;
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = &getProcAddress;
    callbacks.api_version_major = 3;
    callbacks.api_version_minor = 1;
    vernon::tests::RhiRuntime context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_OPENGL_ES, &callbacks);
    ASSERT_NE(context.runtime, nullptr);

    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.width = 2;
    descriptor.height = 2;
    descriptor.depth = 1;
    descriptor.mip_levels = 2;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.format = VERNON_RHI_FORMAT_RGBA32_FLOAT;
    descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiImage image{};
    ASSERT_EQ(vernonRhiDeviceCreateImage(context.device, &descriptor, &image), VERNON_RHI_STATUS_OK);

    pendingError = 0x0500;
    EXPECT_NE(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(generatedMipmapCount, 0U);
    const VernonStringView dirtyContextError = vernonRhiDeviceGetLastError(context.device);
    EXPECT_NE(std::string(dirtyContextError.data, dirtyContextError.size).find("before entering Vernon"),
              std::string::npos);

    EXPECT_EQ(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(generatedMipmapCount, 1U);

    generatedMipmapError = 0x0502;
    EXPECT_EQ(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_UNSUPPORTED);
    EXPECT_EQ(generatedMipmapCount, 2U);
    generatedMipmapError = 0;

    EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, image), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRuntimeDestroy(context.runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}

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
    download.aspect = VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL;
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
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
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

#if defined(VERNON_OPENGL_STORAGE_IMAGE_BUNDLE)
TEST(RuntimeExternalGl, InvokesCanonicalComputeProgramThroughRuntimeCoreProvider) {
    dispatchCount = 0;
    storageBindingCount = 0;
    memoryBarrierCount = 0;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        const size_t bindingCreations = vernon::runtime::getRhiAdapterBindingCreationCount(gl);
        vernon::tests::OwnedProgramExecutable program(gl, VERNON_OPENGL_STORAGE_IMAGE_BUNDLE);
        ASSERT_TRUE(program) << lastError(gl);
        EXPECT_EQ(vernon::runtime::getRhiAdapterBindingCreationCount(gl), bindingCreations);
        auto image = vernon::tests::createImage(rhiRuntime(gl), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA32_FLOAT, 1,
                                                1, 1, VERNON_RHI_IMAGE_STORAGE);
        const auto imageView =
            vernon::tests::createImageView(rhiRuntime(gl), image, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA32_FLOAT);
        ASSERT_NE(image.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
        VernonProgramParameterView imageParameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(program.get(), {"image", 5}, &imageParameter),
                  VERNON_STATUS_OK);
        VernonProgramArgument argument{};
        argument.slot = imageParameter.slot;
        argument.kind = VERNON_PROGRAM_IMAGE;
        argument.image.view = imageView.reference;
        ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(program.get(), &argument, 1, {1, 1, 1}),
                  VERNON_STATUS_OK)
            << lastError(gl);
        EXPECT_EQ(dispatchCount, 1u);
        EXPECT_EQ(storageBindingCount, 1u);
        EXPECT_EQ(memoryBarrierCount, 3u);
        EXPECT_EQ(vernon::runtime::getRhiAdapterBindingCreationCount(gl), bindingCreations + 1);
        ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(program.get(), &argument, 1, {1, 1, 1}),
                  VERNON_STATUS_OK)
            << lastError(gl);
        EXPECT_EQ(dispatchCount, 2u);
        EXPECT_EQ(storageBindingCount, 2u);
        EXPECT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, imageView.handle), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, image.handle), VERNON_RHI_STATUS_OK);
    }
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_STORAGE_IMAGE_BUNDLE)
TEST(RuntimeExternalGl, RejectsStorageImageMetadataThatDisagreesWithRhiResource) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    auto image = vernon::tests::createImage(rhiRuntime(gl), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, 1, 1, 1,
                                            VERNON_RHI_IMAGE_STORAGE);
    const auto imageView =
        vernon::tests::createImageView(rhiRuntime(gl), image, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    ASSERT_NE(image.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);

    {
        vernon::tests::OwnedProgramExecutable program(gl, VERNON_OPENGL_STORAGE_IMAGE_BUNDLE);
        ASSERT_TRUE(program) << lastError(gl);
        VernonProgramParameterView imageParameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(program.get(), {"image", 5}, &imageParameter),
                  VERNON_STATUS_OK);
        VernonProgramArgument argument{};
        argument.slot = imageParameter.slot;
        argument.kind = VERNON_PROGRAM_IMAGE;
        argument.image.view = imageView.reference;
        EXPECT_EQ(vernon::tests::completeCanonicalComputeInvocation(program.get(), &argument, 1, {1, 1, 1}),
                  VERNON_STATUS_INVALID_ARGUMENT);
    }

    EXPECT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, imageView.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, image.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_MATRIX_UNIFORM_BUNDLE)
TEST(RuntimeExternalGl, UploadsColumnMajorMatricesWithoutCopying) {
    constexpr std::array<int64_t, 2> strides = {sizeof(float), 4 * sizeof(float)};
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL, VERNON_OPENGL_MATRIX_UNIFORM_BUNDLE, 4, 3, columnMajor, strides, 0,
                       columnMajor);
}
#endif

#if defined(VERNON_OPENGL_SIGNED_UNIFORM_BUNDLE) && defined(VERNON_OPENGL_UNSIGNED_UNIFORM_BUNDLE)
TEST(RuntimeExternalGl, UploadsSignedAndUnsignedIntegerUniforms) {
    signedUniformUpload.fill(0);
    unsignedUniformUpload.fill(0);
    constexpr GlInt signedValue = -17;
    expectIntegerUniformUpload(VERNON_OPENGL_SIGNED_UNIFORM_BUNDLE, "budget", VERNON_DATA_I32, &signedValue, 1);
    EXPECT_EQ(signedUniformUpload[0], signedValue);

    constexpr std::array<GlUint, 3> unsignedValue = {3, 5, 8};
    expectIntegerUniformUpload(VERNON_OPENGL_UNSIGNED_UNIFORM_BUNDLE, "budget", VERNON_DATA_U32, unsignedValue.data(),
                               unsignedValue.size());
    EXPECT_EQ(unsignedUniformUpload[0], unsignedValue[0]);
    EXPECT_EQ(unsignedUniformUpload[1], unsignedValue[1]);
    EXPECT_EQ(unsignedUniformUpload[2], unsignedValue[2]);
}
#endif

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
    ASSERT_TRUE(bundle) << lastError(gl);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_TRUE(pipeline) << lastError(gl);

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
        << lastError(gl);
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

#if defined(VERNON_OPENGL_MATRIX_UNIFORM_BUNDLE)
TEST(RuntimeExternalGl, PacksRowMajorMatricesWithCanonicalStrides) {
    constexpr std::array<int64_t, 2> strides = {4 * sizeof(float), sizeof(float)};
    constexpr std::array<float, 16> rowMajor = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
    };
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL, VERNON_OPENGL_MATRIX_UNIFORM_BUNDLE, 4, 3, rowMajor, strides, 0,
                       columnMajor);
}
#endif

#if defined(VERNON_OPENGLES_MATRIX_UNIFORM_BUNDLE)
TEST(RuntimeExternalGl, PacksRowMajorMatricesForOpenGlEs) {
    constexpr std::array<int64_t, 2> strides = {4 * sizeof(float), sizeof(float)};
    constexpr std::array<float, 16> rowMajor = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
    };
    constexpr std::array<float, 16> columnMajor = {
        1.0F, 5.0F, 9.0F, 13.0F, 2.0F, 6.0F, 10.0F, 14.0F, 3.0F, 7.0F, 11.0F, 15.0F, 4.0F, 8.0F, 12.0F, 16.0F,
    };
    expectMatrixUpload(VERNON_RUNTIME_OPENGL_ES, VERNON_OPENGLES_MATRIX_UNIFORM_BUNDLE, 3, 1, rowMajor, strides, 0,
                       columnMajor);
}
#endif

TEST(RuntimeExternalGl, ForwardsDebugLabelsWithinTheContextLimits) {
    const std::vector<std::string> core{"push a", "push b", "marker abc", "marker d\xC3\xA9", "pop", "pop"};
    EXPECT_EQ(recordDebugLabels(VERNON_RUNTIME_OPENGL, 4, 3), core);
    EXPECT_EQ(recordDebugLabels(VERNON_RUNTIME_OPENGL, 4, 1, {"GL_KHR_debug"}), core);
    EXPECT_EQ(recordDebugLabels(VERNON_RUNTIME_OPENGL_ES, 3, 2), core);
    EXPECT_EQ(
        recordDebugLabels(VERNON_RUNTIME_OPENGL_ES, 3, 1, {"GL_KHR_debug"}),
        (std::vector<std::string>{"KHR", "push a", "KHR", "push b", "marker abc", "marker d\xC3\xA9", "pop", "pop"}));
}

TEST(RuntimeExternalGl, RecordsNoDebugLabelsWithoutKhrDebug) {
    EXPECT_TRUE(recordDebugLabels(VERNON_RUNTIME_OPENGL, 4, 1).empty());
    EXPECT_TRUE(recordDebugLabels(VERNON_RUNTIME_OPENGL_ES, 3, 1).empty());
}

TEST(RuntimeExternalGl, ClosesTheDebugGroupsOfADestroyedEncoder) {
    debugGroupDepth = 1;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_NE(gl, nullptr);
    const VernonRhiDevice device = rhiRuntime(gl).device;
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder encoder{};
    ASSERT_EQ(vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder), VERNON_RHI_STATUS_OK);
    orderedCalls.clear();
    for (const std::string_view label : {"a", "b"})
        ASSERT_EQ(vernonRhiCommandEncoderPushDebugGroup(device, encoder, {label.data(), label.size()}),
                  VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(orderedCalls, (std::vector<std::string>{"push a", "push b", "pop", "pop"}));
    EXPECT_EQ(debugGroupDepth, 1);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ReportsNestedTimerSpansFromTheFirstTimer) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_NE(gl, nullptr);
    const VernonRhiDevice device = rhiRuntime(gl).device;
    const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
    orderedCalls.clear();
    deletedQueries.clear();
    std::array<uint32_t, 2> timers{};
    ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, &timers[0]), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, &timers[1]), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(timers, (std::array<uint32_t, 2>{0, 1}));
    ASSERT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, timers[1]), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, timers[0]), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    std::vector<VernonRhiTimerSpan> spans;
    ASSERT_EQ(submitAndReadTimers(device, encoder, spans), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(spans.size(), 2U);
    EXPECT_EQ(spans[0].begin_nanoseconds, 0U);
    EXPECT_EQ(spans[0].end_nanoseconds, 30U);
    EXPECT_EQ(spans[1].begin_nanoseconds, 10U);
    EXPECT_EQ(spans[1].end_nanoseconds, 20U);
    EXPECT_EQ(orderedCalls, std::vector<std::string>(4, "timestamp"));
    EXPECT_EQ(deletedQueries.size(), 4U);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, RejectsUnbalancedTimersAndReleasesAbandonedQueries) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_NE(gl, nullptr);
    const VernonRhiDevice device = rhiRuntime(gl).device;
    const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
    deletedQueries.clear();
    EXPECT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, 0), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, nullptr), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    std::array<uint32_t, 2> timers{};
    ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, &timers[0]), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, timers[0]), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, timers[0]), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, &timers[1]), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(deletedQueries.empty());
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(deletedQueries.size(), 3U);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ReportsTimersUnsupportedWithoutTimerQueries) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL_ES, 3, 1);
    ASSERT_NE(gl, nullptr);
    const VernonRhiDevice device = rhiRuntime(gl).device;
    const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
    orderedCalls.clear();
    uint32_t timer{};
    ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, &timer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, timer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    std::vector<VernonRhiTimerSpan> spans;
    EXPECT_EQ(submitAndReadTimers(device, encoder, spans), VERNON_RHI_STATUS_UNSUPPORTED);
    EXPECT_TRUE(orderedCalls.empty());
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, FailsOnlyTheTimersThatADisjointEventInterrupts) {
    exposedExtensions = {"GL_EXT_disjoint_timer_query"};
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL_ES, 3, 1);
    exposedExtensions.clear();
    ASSERT_NE(gl, nullptr);
    const VernonRhiDevice device = rhiRuntime(gl).device;
    for (const bool interrupted : {false, true}) {
        const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
        gpuDisjoint = 1;
        uint32_t timer{};
        ASSERT_EQ(vernonRhiCommandEncoderBeginTimer(device, encoder, &timer), VERNON_RHI_STATUS_OK);
        if (interrupted)
            gpuDisjoint = 1;
        ASSERT_EQ(vernonRhiCommandEncoderEndTimer(device, encoder, timer), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        std::vector<VernonRhiTimerSpan> spans;
        EXPECT_EQ(submitAndReadTimers(device, encoder, spans),
                  interrupted ? VERNON_RHI_STATUS_UNSUPPORTED : VERNON_RHI_STATUS_OK);
        EXPECT_EQ(spans.size(), interrupted ? 0U : 1U);
    }
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

#if defined(VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE)
TEST(RuntimeExternalGl, NestsFusedNodeDrawsInTheirLabelledGroups) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    VernonProgramBundle *bundle = vernon::tests::loadProgramBundle(gl, VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE);
    ASSERT_TRUE(bundle);
    std::array<VernonProgramNodeId, 2> nodes{};
    VernonProgramExecutable *executable = resolveSceneOverlayGraph(gl, bundle, nodes);
    ASSERT_TRUE(executable) << lastError(gl);

    VernonSubmission *submission = nullptr;
    EXPECT_EQ(executeModuleGraphics(gl, executable, {}, nullptr, &submission),
              (std::vector<std::string>{"push scene+overlay", "push scene", "draw", "pop", "push overlay", "draw",
                                        "pop", "pop"}));
    EXPECT_TRUE(readScopeTimings(gl, submission).empty());
    vernonSubmissionDestroy(submission);
    vernonRuntimeProgramExecutableDestroy(executable);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, TimesFusedNodesAsOneScope) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    VernonProgramBundle *bundle = vernon::tests::loadProgramBundle(gl, VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE);
    ASSERT_TRUE(bundle);
    std::array<VernonProgramNodeId, 2> nodes{};
    VernonProgramExecutable *executable = resolveSceneOverlayGraph(gl, bundle, nodes);
    ASSERT_TRUE(executable) << lastError(gl);

    const VernonProgramInvocationOptions options = invocationOptions(VERNON_COMPLETION_WAIT, 1);
    VernonSubmission *submission = nullptr;
    EXPECT_EQ(executeModuleGraphics(gl, executable, {}, &options, &submission),
              (std::vector<std::string>{"push scene+overlay", "timestamp", "push scene", "draw", "pop", "push overlay",
                                        "draw", "pop", "timestamp", "pop"}));
    const std::vector<VernonScopeTiming> scopes = readScopeTimings(gl, submission);
    ASSERT_EQ(scopes.size(), 1U);
    EXPECT_EQ(scopes[0].first_node, nodes[0]);
    EXPECT_EQ(scopes[0].node_count, 2U);
    EXPECT_EQ(scopes[0].device_submission, 0U);
    EXPECT_EQ(scopes[0].begin_nanoseconds, 0U);
    EXPECT_EQ(scopes[0].end_nanoseconds, 10U);
    vernonSubmissionDestroy(submission);
    vernonRuntimeProgramExecutableDestroy(executable);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ReportsScopeTimingsOnceAnAsynchronousSubmissionCompletes) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pass(gl, VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE);
        ASSERT_TRUE(pass);
        const VernonProgramInvocationOptions options = invocationOptions(VERNON_COMPLETION_ASYNCHRONOUS, 1);
        VernonSubmission *submission = nullptr;
        fencesPending = true;
        executeModuleGraphics(gl, pass.get(), {}, &options, &submission);
        size_t count = 1;
        EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, nullptr, 0, &count), VERNON_STATUS_NOT_READY);
        EXPECT_EQ(count, 0U);
        fencesPending = false;
        const std::vector<VernonScopeTiming> scopes = readScopeTimings(gl, submission);
        ASSERT_EQ(scopes.size(), 1U);
        EXPECT_EQ(scopes[0].node_count, 1U);
        EXPECT_EQ(scopes[0].begin_nanoseconds, 0U);
        EXPECT_EQ(scopes[0].end_nanoseconds, 10U);
        vernonSubmissionDestroy(submission);
    }
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ReportsNoScopeTimingsForAFailedSubmission) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pass(gl, VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE);
        ASSERT_TRUE(pass);
        const VernonProgramInvocationOptions options = invocationOptions(VERNON_COMPLETION_ASYNCHRONOUS, 1);
        VernonSubmission *submission = nullptr;
        fencesPending = true;
        executeModuleGraphics(gl, pass.get(), {}, &options, &submission);
        fencesPending = false;
        vernon::runtime::program_execution::setFailureInjectionForTesting(
            vernon::runtime::program_execution::FailureBoundary::Completion);
        size_t count = 1;
        EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, nullptr, 0, &count), VERNON_STATUS_INTERNAL_ERROR);
        vernon::runtime::program_execution::clearFailureInjectionForTesting();
        EXPECT_EQ(lastError(gl), "injected GPU command DAG wait failure");
        EXPECT_EQ(count, 0U);
        EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, nullptr, 0, &count), VERNON_STATUS_INTERNAL_ERROR);
        EXPECT_EQ(lastError(gl), "the submission failed, so its scopes were not timed");
        vernonSubmissionDestroy(submission);
    }
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, ReportsScopeTimingsUnsupportedWithoutTimerQueries) {
    exposeTimerQueries = false;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    exposeTimerQueries = true;
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pass(gl, VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE);
        ASSERT_TRUE(pass);
        const VernonProgramInvocationOptions options = invocationOptions(VERNON_COMPLETION_WAIT, 1);
        VernonSubmission *submission = nullptr;
        executeModuleGraphics(gl, pass.get(), {}, &options, &submission);
        size_t count = 1;
        EXPECT_EQ(vernonSubmissionGetScopeTimings(submission, nullptr, 0, &count), VERNON_STATUS_UNSUPPORTED_TARGET);
        EXPECT_EQ(lastError(gl), "the device cannot time the submission's scopes");
        EXPECT_EQ(count, 0U);
        vernonSubmissionDestroy(submission);
    }
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, KeepsRecordedDebugCommandsBetweenTheirDraws) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pass(gl, VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE);
        ASSERT_TRUE(pass);
        VernonGraphicsCommand draw{};
        draw.kind = VERNON_GRAPHICS_COMMAND_DRAW;
        draw.draw = {sizeof(VernonDrawCommand), nullptr, 3, 1, 0};
        VernonGraphicsCommand bindPipeline{};
        bindPipeline.kind = VERNON_GRAPHICS_COMMAND_BIND_PIPELINE;
        EXPECT_EQ(
            executeModuleGraphics(gl, pass.get(),
                                  {graphicsCommand(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, "triangles"), bindPipeline,
                                   graphicsCommand(VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, "a"), draw,
                                   graphicsCommand(VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, "b"), draw,
                                   graphicsCommand(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP),
                                   graphicsCommand(VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, "c")}),
            (std::vector<std::string>{"push ExternalVerticesGraphics.graphics", "push triangles", "marker a", "draw",
                                      "marker b", "draw", "pop", "marker c", "pop"}));
    }
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

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

#if defined(VERNON_OPENGL_EXPLICIT_SAMPLER_BUNDLE)
TEST(RuntimeExternalGl, BindsExplicitSamplerSeparately) {
    boundSamplerName = UINT32_MAX;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, VERNON_OPENGL_EXPLICIT_SAMPLER_BUNDLE);
        ASSERT_TRUE(pipeline) << lastError(gl);
        ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(pipeline.get()), 2u);

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
        VernonProgramParameterView imageParameter{};
        VernonProgramParameterView samplerParameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline.get(), {"image", 5}, &imageParameter),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline.get(), {"sampler", 7}, &samplerParameter),
                  VERNON_STATUS_OK);
        arguments[0].slot = imageParameter.slot;
        arguments[0].kind = VERNON_PROGRAM_IMAGE;
        arguments[0].image = {sampledView.reference};
        arguments[1].slot = samplerParameter.slot;
        arguments[1].kind = VERNON_PROGRAM_SAMPLER;
        arguments[1].resource = explicitSampler.reference;
        VernonColorAttachment attachment{0, targetView.reference};
        vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), arguments, std::size(arguments), graphics),
                  VERNON_STATUS_OK);
        EXPECT_NE(boundSamplerName, 0u);

        ASSERT_EQ(vernonRhiDeviceDestroySampler(rhiRuntime(gl).device, explicitSampler.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, sampledView.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, sampled.handle), VERNON_RHI_STATUS_OK);
    }
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_COUNTER_CLOCKWISE_WINDING_BUNDLE) && defined(VERNON_OPENGL_CLOCKWISE_WINDING_BUNDLE)
// GL vertex stages negate y, which mirrors window-space winding, so each declared front face selects the opposite one.
TEST(RuntimeExternalGl, MirrorsDeclaredFrontFaceForTheFlippedVertexY) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    RhiImage target = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const auto targetView =
        vernon::tests::createImageView(rhiRuntime(gl), target, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    const VernonColorAttachment attachment{0, targetView.reference};
    const vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    const std::array<std::pair<const char *, GlEnum>, 2> windings{{
        {VERNON_OPENGL_COUNTER_CLOCKWISE_WINDING_BUNDLE, kClockwise},
        {VERNON_OPENGL_CLOCKWISE_WINDING_BUNDLE, kCounterClockwise},
    }};
    for (const auto &[manifest, expectedWinding] : windings) {
        cullMode = 0;
        frontWinding = 0;
        vernon::tests::OwnedProgramExecutable pipeline(gl, manifest);
        ASSERT_TRUE(pipeline) << lastError(gl);
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), nullptr, 0, graphics), VERNON_STATUS_OK)
            << lastError(gl);
        EXPECT_EQ(cullMode, kBack) << manifest;
        EXPECT_EQ(frontWinding, expectedWinding) << manifest;
    }
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_FIXED_FUNCTION_STATE_BUNDLE)
TEST(RuntimeExternalGl, AppliesDeclaredDepthStencilAndBlendState) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    RhiImage color = importOpenGLTexture2D(gl, 7, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
    RhiImage depthStencil =
        vernon::tests::createImage(rhiRuntime(gl), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 16, 16, 1,
                                   VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT);
    ASSERT_NE(color.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_NE(depthStencil.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const auto colorView =
        vernon::tests::createImageView(rhiRuntime(gl), color, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    const auto depthStencilView = vernon::tests::createImageView(
        rhiRuntime(gl), depthStencil, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT, 1, 1,
        VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL);
    const VernonColorAttachment attachment{0, colorView.reference};
    VernonDepthAttachment depthAttachment{};
    depthAttachment.view = depthStencilView.reference;
    depthAttachment.clear_depth = 1.0F;
    vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3);
    graphics.renderPass.depth_attachment = &depthAttachment;
    graphics.dynamic.stencil_reference = 23;
    stencilFuncCount = 0;
    stencilReference = 0;
    stencilReadMask = 0;
    depthWrite = 0;
    depthComparison = 0;
    depthBias = {};
    blendFactors = {};
    blendOperations = {};
    colorWriteMask = {};
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, VERNON_OPENGL_FIXED_FUNCTION_STATE_BUNDLE);
        ASSERT_TRUE(pipeline) << lastError(gl);
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), nullptr, 0, graphics), VERNON_STATUS_OK)
            << lastError(gl);
    }
    EXPECT_EQ(stencilFuncCount, 2U);
    EXPECT_EQ(stencilReference, 23);
    EXPECT_EQ(stencilReadMask, 0x5AU);
    EXPECT_EQ(depthWrite, 1);
    EXPECT_EQ(depthComparison, kLessEqual);
    EXPECT_EQ(depthBias, (std::array<float, 2>{3.0F, 2.0F}));
    EXPECT_EQ(blendFactors, (std::array<GlEnum, 4>{kSourceAlpha, kOneMinusSourceAlpha, kOne, kOneMinusSourceAlpha}));
    EXPECT_EQ(blendOperations, (std::array<GlEnum, 2>{kSubtract, kAdd}));
    EXPECT_EQ(colorWriteMask, (std::array<GlBoolean, 4>{1, 1, 0, 0}));
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, depthStencilView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, colorView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, depthStencil.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, color.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE)
std::string stageShaderSource(const std::filesystem::path &manifestPath, const char *role) {
    std::ifstream manifestInput(manifestPath, std::ios::binary);
    const nlohmann::json manifest = nlohmann::json::parse(manifestInput);
    const auto &artifact = manifest["variants"][0]["artifact_system"]["artifacts"].begin().value();
    for (const nlohmann::json &module : artifact["modules"]) {
        if (module.value("role", "") != role)
            continue;
        const std::string blob = module["blob"].get<std::string>();
        std::ifstream shaderInput(manifestPath.parent_path() /
                                      manifest["blobs"][blob]["location"]["uri"].get<std::string>(),
                                  std::ios::binary);
        return {std::istreambuf_iterator<char>(shaderInput), std::istreambuf_iterator<char>()};
    }
    return {};
}

// The fragment's image and sampler own bindings 0 and 1, so the spilled `model` block must be moved to binding 2.
TEST(RuntimeExternalGl, PointsSpilledUniformBlocksAtTheirPlannedBinding) {
    const std::string vertex = stageShaderSource(VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE, "vertex");
    EXPECT_NE(vertex.find("uniform vernon_uniform_binding_2"), std::string::npos) << vertex;
    linkedUniformBlocks = {"vernon_uniform_binding_2"};
    uniformBlockBindings.clear();
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE);
        ASSERT_TRUE(pipeline) << lastError(gl);
        EXPECT_EQ(uniformBlockBindings, (std::unordered_map<GlUint, GlUint>{{0u, 2u}}));
    }
    linkedUniformBlocks.clear();
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}

TEST(RuntimeExternalGl, RejectsLinkedUniformBlocksWithoutAPlannedBinding) {
    linkedUniformBlocks = {"vernon_uniform_binding_2", "vernon_uniform_binding_9"};
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE);
        EXPECT_FALSE(pipeline);
    }
    linkedUniformBlocks.clear();
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_INDEXED_VERTEX_BUNDLE)
TEST(RuntimeExternalGl, BindsVertexAndIndexBuffersThroughRhi) {
    drawCount = 0;
    indexedDrawCount = 0;
    boundArrayBuffer = 0;
    boundIndexBuffer = 0;
    boundVertexArray = 0;
    vertexAttributeArray = 0;
    vertexAttributeLocation = UINT32_MAX;
    capturedVertexAttributeDivisor = 0;
    createdSyncCount = 0;
    deletedSyncCount = 0;
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    {
        vernon::tests::OwnedProgramExecutable pipeline(gl, VERNON_OPENGL_INDEXED_VERTEX_BUNDLE);
        ASSERT_TRUE(pipeline) << lastError(gl);

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
        VernonProgramParameterView positionParameter{};
        ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline.get(), {"position", 8}, &positionParameter),
                  VERNON_STATUS_OK);
        VernonProgramArgument argument{};
        argument.slot = positionParameter.slot;
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
        vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 0, 3);
        graphics.draw.index_binding = &indexBinding;
        ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), &argument, 1, graphics), VERNON_STATUS_OK)
            << lastError(gl);
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
        EXPECT_GT(createdSyncCount, 0u);
        EXPECT_EQ(deletedSyncCount, createdSyncCount);

        ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, indices.handle), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
    }
    ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_OPENGL_I32_VERTEX_BUNDLE) && defined(VERNON_OPENGL_U32_VERTEX_BUNDLE) &&                            \
    defined(VERNON_OPENGL_F32_VERTEX_BUNDLE)
TEST(RuntimeExternalGl, BindsCompilerSupportedVertexNumericFormatsFromCanonicalPrograms) {
    struct Format {
        const char *name;
        const char *manifest;
        VernonDataType dtype;
        uint32_t size;
        GlEnum nativeType;
        char pointerKind;
    };
    constexpr Format formats[] = {{"i32", VERNON_OPENGL_I32_VERTEX_BUNDLE, VERNON_DATA_I32, 4, kInt, 'I'},
                                  {"u32", VERNON_OPENGL_U32_VERTEX_BUNDLE, VERNON_DATA_U32, 4, kUnsignedInt, 'I'},
                                  {"f32", VERNON_OPENGL_F32_VERTEX_BUNDLE, VERNON_DATA_F32, 4, kFloat, 'F'}};
    for (const Format &format : formats) {
        SCOPED_TRACE(format.name);
        vertexAttributeType = 0;
        vertexAttributePointerKind = '\0';
        VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
        ASSERT_TRUE(gl);
        {
            vernon::tests::OwnedProgramExecutable pipeline(gl, format.manifest);
            ASSERT_TRUE(pipeline) << lastError(gl);
            const uint32_t stride = 3 * format.size;
            auto vertices =
                vernon::tests::createBuffer(rhiRuntime(gl), 3 * stride, format.size, VERNON_RHI_BUFFER_VERTEX);
            RhiImage target = importOpenGLTexture2D(gl, 8, 16, 16, VERNON_TEXTURE_RGBA8_UNORM);
            const auto targetView = vernon::tests::createImageView(rhiRuntime(gl), target, VERNON_RHI_IMAGE_2D,
                                                                   VERNON_RHI_FORMAT_RGBA8_UNORM);
            ASSERT_NE(vertices.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
            ASSERT_NE(target.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
            const uint64_t shape[2] = {3, 3};
            const int64_t strides[2] = {stride, format.size};
            VernonProgramParameterView positionParameter{};
            ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline.get(), {"position", 8}, &positionParameter),
                      VERNON_STATUS_OK);
            VernonProgramArgument argument{};
            argument.slot = positionParameter.slot;
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
            const vernon::tests::CanonicalGraphicsControls graphics(&attachment, 1, 3, 3);
            ASSERT_EQ(vernon::tests::completeCanonicalInvocation(pipeline.get(), &argument, 1, graphics),
                      VERNON_STATUS_OK)
                << lastError(gl);
            EXPECT_EQ(vertexAttributeType, format.nativeType);
            EXPECT_EQ(vertexAttributePointerKind, format.pointerKind);
            EXPECT_EQ(vertexAttributeStride, static_cast<GlSize>(stride));

            ASSERT_EQ(vernonRhiDeviceDestroyImageView(rhiRuntime(gl).device, targetView.handle), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiDeviceDestroyImage(rhiRuntime(gl).device, target.handle), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiDeviceDestroyBuffer(rhiRuntime(gl).device, vertices.handle), VERNON_RHI_STATUS_OK);
        }
        ASSERT_EQ(destroy(gl), VERNON_STATUS_OK);
    }
}
#endif

#undef GL_CALL
