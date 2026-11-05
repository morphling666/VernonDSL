#include "VernonRuntime.h"
#include "VernonSchedule.h"
#include "rhi/rhi_test_hooks.h"
#include "runtime_rhi_test_utils.h"
#include "schedule_test_support.h"

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
// The buffer bound at each indexed GL_UNIFORM_BUFFER binding point.
std::unordered_map<GlUint, GlUint> uniformBufferBindings;
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
std::vector<GlUint> framebufferTextures;
std::vector<GlUint> generatedFramebuffers;
std::vector<GlUint> deletedFramebuffers;
std::array<GlInt, 4> signedUniformUpload{};
std::array<GlUint, 4> unsignedUniformUpload{};
GlEnum pendingError = 0;
GlEnum generatedMipmapError = 0;
uint32_t generatedMipmapCount = 0;
GlEnum readPixelsError = 0;
uint32_t getErrorCount = 0;
constexpr GlEnum kInvalidEnum = 0x0500;
constexpr GlEnum kDebugOutput = 0x92E0;
constexpr GlEnum kDebugTypeError = 0x824C;
using GlDebugCallback = void(GL_CALL *)(GlEnum, GlEnum, GlUint, GlEnum, GlSize, const char *, const void *);
GlDebugCallback debugCallback{};
const void *debugCallbackUser{};
bool debugOutputEnabled = false;
// Records rror as GL does: glGetError keeps the first one, and enabled debug output reports every one.
void raiseError(GlEnum error, const std::string &message) {
    if (!pendingError)
        pendingError = error;
    if (debugOutputEnabled && debugCallback)
        debugCallback(0x8246, kDebugTypeError, error, 0x9146, static_cast<GlSize>(message.size()), message.c_str(),
                      debugCallbackUser);
}

void makeCurrent(void *) { ++makeCurrentCount; }
void GL_CALL objectNoop(GlUint) {}
GlEnum GL_CALL getError() {
    ++getErrorCount;
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
void GL_CALL enable(GlEnum capability) {
    if (capability == 0)
        raiseError(kInvalidEnum, "GL_INVALID_ENUM in glEnable(cap)");
    if (capability == kDebugOutput)
        debugOutputEnabled = true;
}
void GL_CALL disable(GlEnum capability) {
    if (capability == kDebugOutput)
        debugOutputEnabled = false;
}
void GL_CALL debugMessageCallback(GlDebugCallback callback, const void *user) {
    debugCallback = callback;
    debugCallbackUser = user;
}
void GL_CALL debugMessageControl(GlEnum, GlEnum, GlEnum, GlSize, const GlUint *, GlBoolean) {}
void GL_CALL enumIntNoop(GlEnum, GlInt) {}
void GL_CALL deleteNamesNoop(GlSize, const GlUint *) {}
void GL_CALL framebufferTexture2D(GlEnum, GlEnum, GlEnum, GlUint texture, GlInt) {
    if (texture)
        framebufferTextures.push_back(texture);
}
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
void GL_CALL genFramebuffers(GlSize count, GlUint *names) {
    genNames(count, names);
    generatedFramebuffers.insert(generatedFramebuffers.end(), names, names + count);
}
void GL_CALL deleteFramebuffers(GlSize count, const GlUint *names) {
    std::copy_if(names, names + count, std::back_inserter(deletedFramebuffers), [](GlUint name) { return name != 0; });
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
void GL_CALL bindBufferBase(GlEnum target, GlUint index, GlUint buffer) {
    ++storageBindingCount;
    if (target == 0x8A11)
        uniformBufferBindings[index] = buffer;
}
void GL_CALL bindBufferRange(GlEnum target, GlUint index, GlUint buffer, std::intptr_t, std::intptr_t) {
    bindBufferBase(target, index, buffer);
}
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
void GL_CALL copyBufferSubData(GlEnum, GlEnum, std::intptr_t, std::intptr_t, std::intptr_t) {}
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
    if (readPixelsError) {
        raiseError(readPixelsError, "GL_INVALID_OPERATION in glReadPixels");
        return;
    }
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
    PROC("glDisableVertexAttribArray", objectNoop);
    PROC("glUseProgram", useProgram);
    PROC("glGenFramebuffers", genFramebuffers);
    PROC("glDeleteFramebuffers", deleteFramebuffers);
    PROC("glGenBuffers", genNames);
    PROC("glBindBuffer", bindBuffer);
    PROC("glBindBufferBase", bindBufferBase);
    PROC("glBindBufferRange", bindBufferRange);
    PROC("glBindImageTexture", bindImageTexture);
    PROC("glDeleteBuffers", deleteBufferNames);
    PROC("glBufferData", bufferData);
    PROC("glBufferSubData", bufferSubData);
    PROC("glCopyBufferSubData", copyBufferSubData);
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
    PROC("glFramebufferTexture2D", framebufferTexture2D);
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
    PROC("glEnable", enable);
    PROC("glDisable", disable);
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
    PROC("glDebugMessageCallback", debugMessageCallback);
    PROC("glDebugMessageControl", debugMessageControl);
    PROC("glDebugMessageCallbackKHR", debugMessageCallback);
    PROC("glDebugMessageControlKHR", debugMessageControl);
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
                             bool exposeTextureView = true, uint32_t deviceFlags = 0) {
    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = exposeTextureView ? &getProcAddress : &getProcAddressWithoutTextureView;
    callbacks.api_version_major = major;
    callbacks.api_version_minor = minor;
    RhiRuntime context = vernon::tests::createRhiRuntime(backend, &callbacks,
                                                         vernon::tests::RhiTestDevicePreference::Hardware, deviceFlags);
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

using vernon::tests::DeviceTensor;
using vernon::tests::HalImage;
using vernon::tests::HalSampler;
using vernon::tests::HalTestDevice;
using vernon::tests::LinkedProgram;
using vernon::tests::PassControls;
using vernon::tests::ResolvedProgram;
using vernon::tests::ScheduleRecord;

// A VernonHal.h device on the fake driver, which reads the driver's state as it is when the device is created.
class FakeGlDevice {
public:
    explicit FakeGlDevice(VernonRuntimeBackend backend = VERNON_RUNTIME_OPENGL, uint16_t major = 4, uint16_t minor = 3,
                          uint32_t flags = 0) {
        VernonOpenGLContextCallbacks callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.make_current = &makeCurrent;
        callbacks.get_proc_address = &getProcAddress;
        callbacks.api_version_major = major;
        callbacks.api_version_minor = minor;
        EXPECT_TRUE(device_.adoptOpenGL(backend, callbacks, flags)) << vernon::tests::halError();
    }

    explicit operator bool() const { return device_.get() != nullptr; }
    const HalTestDevice &device() const { return device_; }
    VernonHalDevice *get() const { return device_.get(); }
    VernonRuntimeContext *runtime() const { return device_.runtime(); }

private:
    HalTestDevice device_;
};

// A fixture Program on the fake driver, linked as one schedule.
class FakeGlProgram {
public:
    FakeGlProgram(const FakeGlDevice &gl, const std::filesystem::path &manifest)
        : program_(gl.runtime(), manifest), linked_(gl.runtime(), program_.get()) {}

    explicit operator bool() const { return static_cast<bool>(linked_); }
    std::string error() const { return linked_.error(); }
    LinkedProgram &linked() { return linked_; }
    ScheduleRecord &record() { return linked_.record(); }
    VernonStatus call(uint64_t *point = nullptr) { return linked_.call(point); }

private:
    ResolvedProgram program_;
    LinkedProgram linked_;
};

// A color target of the fake driver, and the pass that clears it to zero and draws `vertexCount` vertices into it.
struct ColorPass {
    ColorPass(const FakeGlDevice &gl, uint32_t width, uint32_t height, uint32_t vertexCount)
        : target(gl.get(), VERNON_TEXTURE_RGBA8_UNORM, width, height, VERNON_IMAGE_COLOR_ATTACHMENT),
          controls(target.reference(), width) {
        controls.attachment.clear_color.f32[3] = 0.0F;
        controls.renderPass.render_area[3] = height;
        controls.dynamic.viewport[3] = height;
        controls.dynamic.scissor[3] = height;
        draw.vertex_count = vertexCount;
    }

    void writeTo(ScheduleRecord &record) const { controls.writeTo(record, 0, &draw); }

    HalImage target;
    PassControls controls;
    VernonDrawCommand draw{sizeof(VernonDrawCommand), nullptr, 0, 1, 0};
};

void expectMatrixUpload(VernonRuntimeBackend backend, const char *manifest, uint16_t major, uint16_t minor,
                        std::array<float, 16> storage, const std::array<int64_t, 2> &strides,
                        GlBoolean expectedTranspose, const std::array<float, 16> &expectedUpload) {
    matrixTranspose = expectedTranspose ? 0 : 1;
    matrixUpload.fill(0.0F);
    clearCount = 0;
    clearedDrawBuffer = -1;
    clearColor.fill(1.0F);
    const FakeGlDevice gl(backend, major, minor);
    ASSERT_TRUE(gl);
    FakeGlProgram program(gl, manifest);
    ASSERT_TRUE(program) << program.error();
    const ColorPass pass(gl, 16, 16, 3);
    constexpr std::array<uint64_t, 2> shape = {4, 4};
    program.record().bind("transform",
                          vernon::tests::hostView(storage.data(), sizeof(storage), 2, shape.data(), strides.data()));
    pass.writeTo(program.record());
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    const GlUint namesAfterFirstCall = nextName;
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    EXPECT_EQ(nextName, namesAfterFirstCall) << "a later call creates no GL object";
    EXPECT_EQ(matrixTranspose, expectedTranspose);
    EXPECT_EQ(matrixUpload, expectedUpload);
    EXPECT_EQ(clearCount, 2U);
    EXPECT_EQ(clearedDrawBuffer, 0);
    EXPECT_EQ(clearColor, (std::array<float, 4>{}));
}

void expectIntegerUniformUpload(const char *manifest, const char *name, const void *data, uint32_t components) {
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    FakeGlProgram program(gl, manifest);
    ASSERT_TRUE(program) << program.error();
    const ColorPass pass(gl, 16, 16, 3);
    std::array<uint32_t, 4> values{};
    std::memcpy(values.data(), data, components * sizeof(uint32_t));
    const uint64_t shape = components;
    const int64_t stride = sizeof(uint32_t);
    program.record().bind(name, vernon::tests::hostView(values.data(), components * sizeof(uint32_t),
                                                        components == 1 ? 0 : 1, components == 1 ? nullptr : &shape,
                                                        components == 1 ? nullptr : &stride));
    pass.writeTo(program.record());
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
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
    VernonRuntimeContext *gl = create(backend, major, minor, true, VERNON_RHI_DEVICE_LABELS);
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

TEST(RuntimeExternalGl, CachesOneFramebufferPerAttachmentSetUntilAnAttachedImageIsDestroyed) {
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3);
    ASSERT_TRUE(gl);
    const VernonRhiDevice device = rhiRuntime(gl).device;
    const RhiImage first = createTexture2D(gl, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    const RhiImage second = createTexture2D(gl, 4, 4, VERNON_TEXTURE_RGBA8_UNORM);
    const auto firstView =
        vernon::tests::createImageView(rhiRuntime(gl), first, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    const auto secondView =
        vernon::tests::createImageView(rhiRuntime(gl), second, VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM);
    const auto render = [&](VernonRhiImageView view) {
        const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
        VernonRhiColorAttachment color{};
        color.view = view;
        color.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
        color.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
        color.load_operation = VERNON_RHI_LOAD_CLEAR;
        color.store_operation = VERNON_RHI_STORE_PRESERVE;
        VernonRhiRenderingDescriptor rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.color_attachments = &color;
        rendering.color_attachment_count = 1;
        rendering.width = 4;
        rendering.height = 4;
        rendering.layers = 1;
        ASSERT_EQ(vernonRhiCommandEncoderBeginRendering(device, encoder, &rendering), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device, encoder), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
    };
    framebufferTextures.clear();
    generatedFramebuffers.clear();
    deletedFramebuffers.clear();

    render(firstView.handle);
    render(secondView.handle);
    render(firstView.handle);
    ASSERT_EQ(generatedFramebuffers.size(), 2U);
    EXPECT_EQ(framebufferTextures.size(), 2U) << "each attachment set is attached once";

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(device, firstView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(device, first.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(deletedFramebuffers, (std::vector<GlUint>{generatedFramebuffers[0]}));
    render(secondView.handle);
    EXPECT_EQ(generatedFramebuffers.size(), 2U);

    ASSERT_EQ(vernonRhiDeviceDestroyImageView(device, secondView.handle), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDestroyImage(device, second.handle), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(deletedFramebuffers, generatedFramebuffers);
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
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

// The module graphics Program linked on the fake driver, or with `sceneOverlay` the graph of its Nodes "scene" and
// "overlay". Every graphics node draws one triangle into one target, the first clearing it.
class ModuleGraphics {
public:
    explicit ModuleGraphics(const FakeGlDevice &gl, bool sceneOverlay = false)
        : program_(gl.runtime(), VERNON_OPENGL_MODULE_GRAPHICS_BUNDLE),
          graph_(sceneOverlay && program_ ? resolveSceneOverlayGraph(gl.runtime(), program_.bundle(), nodes_) : nullptr,
                 vernonRuntimeProgramExecutableDestroy),
          linked_(gl.runtime(), sceneOverlay ? graph_.get() : program_.get()),
          vertices_(gl.device(), kTriangle.data(), sizeof(kTriangle), {3, 2}, sizeof(float), {}, 0,
                    VERNON_HAL_BUFFER_VERTEX | VERNON_HAL_BUFFER_STORAGE),
          target_(gl.get(), VERNON_TEXTURE_RGBA8_UNORM, 8, 8, VERNON_IMAGE_COLOR_ATTACHMENT) {
        if (!linked_)
            return;
        for (const char *name : {"vertices", "scene_vertices", "overlay_vertices"})
            if (linked_.record().takes(name))
                linked_.record().bind(name, vertices_);
        for (size_t node = 0; linked_.record().passSlot(node) != SIZE_MAX; ++node) {
            controls_.push_back(std::make_unique<PassControls>(target_.reference(), 8));
            if (node)
                controls_.back()->attachment.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
            controls_.back()->writeTo(linked_.record(), node, &draw_);
        }
    }

    explicit operator bool() const { return static_cast<bool>(linked_); }
    std::string error() const { return linked_.error(); }
    VernonLinkedSchedule *get() const { return linked_.get(); }
    VernonProgramNodeId sceneNode() const { return nodes_[0]; }

    // Draws the first graphics node's triangles through `table` instead.
    void drawFirstNodeThrough(const VernonSchedulePassTable *table) {
        linked_.record().write(linked_.record().passSlot(0) + 4, table);
    }

    // Calls and waits, returning the debug label calls, timestamp queries and draws that reached the driver.
    std::vector<std::string> call(uint64_t *point = nullptr) {
        orderedCalls.clear();
        EXPECT_EQ(linked_.call(point), VERNON_STATUS_OK) << error();
        return std::move(orderedCalls);
    }
    VernonStatus submit(uint64_t &point) {
        return vernonScheduleCall(linked_.get(), linked_.record().data(), linked_.record().size(), &point);
    }

    // The scope timings of the completed call that signalled `point`, checking that a short capacity is rejected.
    std::vector<VernonScopeTiming> scopeTimings(uint64_t point) const {
        size_t count = 0;
        EXPECT_EQ(vernonScheduleGetScopeTimings(linked_.get(), point, nullptr, 0, &count), VERNON_STATUS_OK) << error();
        std::vector<VernonScopeTiming> scopes(count);
        if (count) {
            EXPECT_EQ(vernonScheduleGetScopeTimings(linked_.get(), point, scopes.data(), count - 1, &count),
                      VERNON_STATUS_INVALID_ARGUMENT);
            EXPECT_EQ(vernonScheduleGetScopeTimings(linked_.get(), point, scopes.data(), scopes.size(), &count),
                      VERNON_STATUS_OK);
            EXPECT_EQ(count, scopes.size());
        }
        return scopes;
    }

private:
    static constexpr std::array<float, 6> kTriangle{-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};

    ResolvedProgram program_;
    std::array<VernonProgramNodeId, 2> nodes_{};
    std::unique_ptr<VernonProgramExecutable, void (*)(VernonProgramExecutable *)> graph_;
    LinkedProgram linked_;
    DeviceTensor vertices_;
    HalImage target_;
    VernonDrawCommand draw_{sizeof(VernonDrawCommand), nullptr, 3, 1, 0};
    std::vector<std::unique_ptr<PassControls>> controls_;
};

VernonScheduleDebugCommand debugCommand(VernonGraphicsCommandKind kind, std::string_view label = {}) {
    return {static_cast<uint32_t>(kind), {label.data(), label.size()}};
}

// A pass table entry that draws the node's triangle with its own Stage.
VernonSchedulePassEntry triangleEntry(const VernonScheduleDebugCommand *commands = nullptr, size_t count = 0) {
    VernonSchedulePassEntry entry{};
    entry.struct_size = sizeof(entry);
    entry.draw = {sizeof(VernonDrawCommand), nullptr, 3, 1, 0};
    entry.debug_commands = commands;
    entry.debug_command_count = count;
    return entry;
}
#endif

} // namespace

namespace {

VernonRhiImage createMipmappedImage(VernonRhiDevice device) {
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
    EXPECT_EQ(vernonRhiDeviceCreateImage(device, &descriptor, &image), VERNON_RHI_STATUS_OK);
    return image;
}

std::string deviceError(VernonRhiDevice device) {
    const VernonStringView error = vernonRhiDeviceGetLastError(device);
    return {error.data, error.size};
}

VernonRhiStatus submitEmptyCommands(VernonRhiDevice device) {
    const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
    EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
    return vernon::tests::completeSubmission(device, encoder);
}

} // namespace

TEST(RuntimeExternalGl, ScopesMipmapErrorsAndProbesFloatFormatSupport) {
    for (const bool validation : {true, false}) {
        SCOPED_TRACE(validation ? "validation" : "no validation");
        pendingError = 0;
        generatedMipmapError = 0;
        generatedMipmapCount = 0;
        VernonOpenGLContextCallbacks callbacks{};
        callbacks.struct_size = sizeof(callbacks);
        callbacks.make_current = &makeCurrent;
        callbacks.get_proc_address = &getProcAddress;
        callbacks.api_version_major = 3;
        callbacks.api_version_minor = 1;
        vernon::tests::RhiRuntime context = vernon::tests::createRhiRuntime(
            VERNON_RUNTIME_OPENGL_ES, &callbacks, vernon::tests::RhiTestDevicePreference::Hardware,
            validation ? VERNON_RHI_DEVICE_VALIDATION : 0U);
        ASSERT_NE(context.runtime, nullptr);
        const VernonRhiImage image = createMipmappedImage(context.device);

        // An error left on the context fails the next operation in validation mode; otherwise the mipmap check
        // leaves it to the next completion.
        pendingError = 0x0500;
        if (validation) {
            EXPECT_NE(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_OK);
            EXPECT_EQ(generatedMipmapCount, 0U);
            EXPECT_NE(deviceError(context.device).find("before entering Vernon"), std::string::npos);
            EXPECT_EQ(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_OK);
        } else {
            EXPECT_EQ(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_OK);
            EXPECT_NE(submitEmptyCommands(context.device), VERNON_RHI_STATUS_OK);
            EXPECT_EQ(deviceError(context.device), "OpenGL error 1280");
        }
        EXPECT_EQ(generatedMipmapCount, 1U);

        generatedMipmapError = 0x0502;
        EXPECT_EQ(vernonRhiDeviceGenerateImageMipmaps(context.device, image), VERNON_RHI_STATUS_UNSUPPORTED);
        EXPECT_EQ(generatedMipmapCount, 2U);
        generatedMipmapError = 0;
        EXPECT_EQ(submitEmptyCommands(context.device), VERNON_RHI_STATUS_OK);

        EXPECT_EQ(vernonRhiDeviceDestroyImage(context.device, image), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRuntimeDestroy(context.runtime), VERNON_STATUS_OK);
        vernonRhiDestroyDevice(context.device);
    }
}

TEST(RuntimeExternalGl, FailsTheNextCompletionWithAnErrorRaisedOutsideValidation) {
    for (const bool ownsDebugOutput : {false, true}) {
        SCOPED_TRACE(ownsDebugOutput ? "debug output" : "glGetError");
        pendingError = 0;
        VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3, true,
                                          ownsDebugOutput ? VERNON_RHI_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT : 0U);
        ASSERT_NE(gl, nullptr);
        const VernonRhiDevice device = rhiRuntime(gl).device;
        EXPECT_EQ(debugOutputEnabled, ownsDebugOutput);
        const uint32_t errorChecks = getErrorCount;
        const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
        vernon::rhi::raiseOpenGLError(device);
        EXPECT_EQ(vernonRhiCommandEncoderFinish(device, encoder), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(getErrorCount, errorChecks) << "nothing reads errors before the completion";
        EXPECT_NE(vernon::tests::completeSubmission(device, encoder), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(deviceError(device),
                  ownsDebugOutput ? "OpenGL error 1280: GL_INVALID_ENUM in glEnable(cap)" : "OpenGL error 1280");
        if (ownsDebugOutput)
            EXPECT_EQ(getErrorCount, errorChecks);
        pendingError = 0;
        EXPECT_EQ(submitEmptyCommands(device), VERNON_RHI_STATUS_OK) << "the device stays usable";
        EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
        EXPECT_FALSE(debugOutputEnabled);
        EXPECT_EQ(debugCallback, nullptr);
    }
}

TEST(RuntimeExternalGl, FailsAFailedDownloadInEveryMode) {
    for (const uint32_t flags : {0U, static_cast<uint32_t>(VERNON_RHI_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT),
                                 static_cast<uint32_t>(VERNON_RHI_DEVICE_VALIDATION)}) {
        SCOPED_TRACE(flags);
        pendingError = 0;
        VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3, true, flags);
        ASSERT_NE(gl, nullptr);
        const VernonRhiDevice device = rhiRuntime(gl).device;
        const RhiImage image =
            vernon::tests::createImage(rhiRuntime(gl), VERNON_RHI_IMAGE_2D, VERNON_RHI_FORMAT_RGBA8_UNORM, 2, 2, 1,
                                       VERNON_RHI_IMAGE_TRANSFER_SOURCE);
        textureStorage.assign(16, 7);
        std::array<uint8_t, 16> data{};
        VernonRhiImageDownloadDescriptor download{};
        download.struct_size = sizeof(download);
        download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
        download.width = 2;
        download.height = 2;
        download.depth = 1;
        download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
        download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
        readPixelsError = 0x0502;
        EXPECT_NE(vernonRhiDeviceDownloadImage(device, image.handle, &download, data.data(), data.size()),
                  VERNON_RHI_STATUS_OK);
        EXPECT_NE(deviceError(device).find("download image"), std::string::npos) << deviceError(device);
        readPixelsError = 0;
        EXPECT_EQ(vernonRhiDeviceDownloadImage(device, image.handle, &download, data.data(), data.size()),
                  VERNON_RHI_STATUS_OK);
        EXPECT_EQ(data[0], 7);
        EXPECT_EQ(submitEmptyCommands(device), VERNON_RHI_STATUS_OK) << "the download reported its own error";
        EXPECT_EQ(vernonRhiDeviceDestroyImage(device, image.handle), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
    }
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
TEST(RuntimeExternalGl, CallsAComputeScheduleThatWritesAStorageImage) {
    dispatchCount = 0;
    storageBindingCount = 0;
    memoryBarrierCount = 0;
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    FakeGlProgram program(gl, VERNON_OPENGL_STORAGE_IMAGE_BUNDLE);
    ASSERT_TRUE(program) << program.error();
    const HalImage image(gl.get(), VERNON_TEXTURE_RGBA32_FLOAT, 1, 1, VERNON_IMAGE_STORAGE);
    ASSERT_TRUE(image);
    program.record().bindImage("image", image.reference());
    program.record().bindGrid(1);
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    EXPECT_EQ(dispatchCount, 1u);
    EXPECT_EQ(storageBindingCount, 1u);
    EXPECT_GT(memoryBarrierCount, 0u);
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    EXPECT_EQ(dispatchCount, 2u);
    EXPECT_EQ(storageBindingCount, 2u);
}

TEST(RuntimeExternalGl, RejectsAStorageImageOfAnotherFormat) {
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    FakeGlProgram program(gl, VERNON_OPENGL_STORAGE_IMAGE_BUNDLE);
    ASSERT_TRUE(program) << program.error();
    const HalImage image(gl.get(), VERNON_TEXTURE_RGBA8_UNORM, 1, 1, VERNON_IMAGE_STORAGE);
    ASSERT_TRUE(image);
    program.record().bindImage("image", image.reference());
    program.record().bindGrid(1);
    EXPECT_EQ(program.call(), VERNON_STATUS_INVALID_ARGUMENT);
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
    expectIntegerUniformUpload(VERNON_OPENGL_SIGNED_UNIFORM_BUNDLE, "budget", &signedValue, 1);
    EXPECT_EQ(signedUniformUpload[0], signedValue);

    constexpr std::array<GlUint, 3> unsignedValue = {3, 5, 8};
    expectIntegerUniformUpload(VERNON_OPENGL_UNSIGNED_UNIFORM_BUNDLE, "budget", unsignedValue.data(),
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
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    FakeGlProgram program(gl, manifestPath);
    ASSERT_TRUE(program) << program.error();
    constexpr std::array<float, 6> positions{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
    const DeviceTensor vertices(gl.device(), positions.data(), sizeof(positions), {3, 2}, sizeof(float), {}, 0,
                                VERNON_HAL_BUFFER_VERTEX);
    const ColorPass pass(gl, 37, 23, 3);
    constexpr GlInt maxSteps = 19;
    program.record().bindScalar("max_steps", maxSteps);
    program.record().bind("position", vertices);
    pass.writeTo(program.record());
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    EXPECT_EQ(resolutionUpload, (std::array<float, 2>{37.0F, 23.0F}));
    EXPECT_EQ(signedUniformUpload[0], maxSteps);
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
    VernonRuntimeContext *gl = create(VERNON_RUNTIME_OPENGL, 4, 3, true, VERNON_RHI_DEVICE_LABELS);
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
    deletedQueries.clear();
    // The second frame reuses the four queries the first one released.
    for (int frame = 0; frame < 2; ++frame) {
        const VernonRhiCommandEncoder encoder = createGraphicsEncoder(device);
        orderedCalls.clear();
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
        EXPECT_EQ(vernon::rhi::getOpenGLCallStats(device).queryCreations, 4U);
    }
    EXPECT_TRUE(deletedQueries.empty());
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
    EXPECT_EQ(deletedQueries.size(), 4U);
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
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device, encoder), VERNON_RHI_STATUS_OK);
    EXPECT_TRUE(deletedQueries.empty());
    EXPECT_EQ(destroy(gl), VERNON_STATUS_OK);
    EXPECT_EQ(deletedQueries.size(), 3U);
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
    const FakeGlDevice gl(VERNON_RUNTIME_OPENGL, 4, 3, VERNON_HAL_DEVICE_LABELS);
    ASSERT_TRUE(gl);
    ModuleGraphics graphics(gl, true);
    ASSERT_TRUE(graphics) << graphics.error();
    EXPECT_EQ(graphics.call(), (std::vector<std::string>{"push scene+overlay", "push scene", "draw", "pop",
                                                         "push overlay", "draw", "pop", "pop"}));
}

TEST(RuntimeExternalGl, TimesFusedNodesAsOneScope) {
    const FakeGlDevice gl(VERNON_RUNTIME_OPENGL, 4, 3, VERNON_HAL_DEVICE_LABELS);
    ASSERT_TRUE(gl);
    ModuleGraphics graphics(gl, true);
    ASSERT_TRUE(graphics) << graphics.error();
    ASSERT_EQ(vernonScheduleSetTimings(graphics.get(), 1), VERNON_STATUS_OK);
    uint64_t point = 0;
    EXPECT_EQ(graphics.call(&point),
              (std::vector<std::string>{"timestamp", "push scene+overlay", "push scene", "draw", "pop", "push overlay",
                                        "draw", "pop", "pop", "timestamp"}));
    const std::vector<VernonScopeTiming> scopes = graphics.scopeTimings(point);
    ASSERT_EQ(scopes.size(), 1U);
    EXPECT_EQ(scopes[0].first_node, graphics.sceneNode());
    EXPECT_EQ(scopes[0].node_count, 2U);
    EXPECT_EQ(scopes[0].begin_nanoseconds, 0U);
    EXPECT_EQ(scopes[0].end_nanoseconds, 10U);
}

TEST(RuntimeExternalGl, ReportsScopeTimingsOnceTheirCallIsWaitedFor) {
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    ModuleGraphics graphics(gl);
    ASSERT_TRUE(graphics) << graphics.error();
    ASSERT_EQ(vernonScheduleSetTimings(graphics.get(), 1), VERNON_STATUS_OK);
    uint64_t point = 0;
    fencesPending = true;
    ASSERT_EQ(graphics.submit(point), VERNON_STATUS_OK) << graphics.error();
    size_t count = 1;
    EXPECT_EQ(vernonScheduleGetScopeTimings(graphics.get(), point, nullptr, 0, &count), VERNON_STATUS_NOT_READY);
    EXPECT_EQ(count, 0U);
    fencesPending = false;
    ASSERT_EQ(vernonScheduleWait(gl.runtime(), point), VERNON_STATUS_OK) << graphics.error();
    const std::vector<VernonScopeTiming> scopes = graphics.scopeTimings(point);
    ASSERT_EQ(scopes.size(), 1U);
    EXPECT_EQ(scopes[0].node_count, 1U);
    EXPECT_EQ(scopes[0].begin_nanoseconds, 0U);
    EXPECT_EQ(scopes[0].end_nanoseconds, 10U);
}

TEST(RuntimeExternalGl, ReportsScopeTimingsUnsupportedWithoutTimerQueries) {
    exposeTimerQueries = false;
    const FakeGlDevice gl;
    exposeTimerQueries = true;
    ASSERT_TRUE(gl);
    ModuleGraphics graphics(gl);
    ASSERT_TRUE(graphics) << graphics.error();
    ASSERT_EQ(vernonScheduleSetTimings(graphics.get(), 1), VERNON_STATUS_OK);
    uint64_t point = 0;
    graphics.call(&point);
    size_t count = 1;
    EXPECT_EQ(vernonScheduleGetScopeTimings(graphics.get(), point, nullptr, 0, &count),
              VERNON_STATUS_UNSUPPORTED_TARGET);
    EXPECT_EQ(count, 0U);
}

TEST(RuntimeExternalGl, CallsWithoutDriverRoundTripsOutsideValidation) {
    const FakeGlDevice gl(VERNON_RUNTIME_OPENGL, 4, 3, VERNON_HAL_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT);
    ASSERT_TRUE(gl);
    ModuleGraphics graphics(gl);
    ASSERT_TRUE(graphics) << graphics.error();
    ASSERT_EQ(vernonScheduleSetTimings(graphics.get(), 1), VERNON_STATUS_OK);
    const VernonScheduleDebugCommand group[]{debugCommand(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, "triangles"),
                                             debugCommand(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP)};
    const VernonSchedulePassEntry entry = triangleEntry(&group[0], 1);
    VernonSchedulePassTable table{};
    table.struct_size = sizeof(table);
    table.entries = &entry;
    table.entry_count = 1;
    table.debug_commands = &group[1];
    table.debug_command_count = 1;
    graphics.drawFirstNodeThrough(&table);
    const auto frame = [&] {
        uint64_t point = 0;
        EXPECT_EQ(graphics.call(&point), (std::vector<std::string>{"timestamp", "draw", "timestamp"}));
        EXPECT_EQ(graphics.scopeTimings(point).size(), 1U);
    };
    frame();
    const uint32_t errorChecks = getErrorCount;
    const GlUint names = nextName;
    for (int steady = 0; steady < 3; ++steady)
        frame();
    EXPECT_EQ(getErrorCount, errorChecks);
    EXPECT_EQ(nextName, names) << "a steady call creates no GL object, timer queries included";
}

TEST(RuntimeExternalGl, KeepsPassTableDebugCommandsBetweenTheirDraws) {
    const FakeGlDevice gl(VERNON_RUNTIME_OPENGL, 4, 3, VERNON_HAL_DEVICE_LABELS);
    ASSERT_TRUE(gl);
    ModuleGraphics graphics(gl);
    ASSERT_TRUE(graphics) << graphics.error();
    const VernonScheduleDebugCommand commands[]{debugCommand(VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP, "triangles"),
                                                debugCommand(VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, "a"),
                                                debugCommand(VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, "b"),
                                                debugCommand(VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP),
                                                debugCommand(VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER, "c")};
    const VernonSchedulePassEntry entries[]{triangleEntry(&commands[0], 2), triangleEntry(&commands[2], 1)};
    VernonSchedulePassTable table{};
    table.struct_size = sizeof(table);
    table.entries = entries;
    table.entry_count = 2;
    table.debug_commands = &commands[3];
    table.debug_command_count = 2;
    graphics.drawFirstNodeThrough(&table);
    EXPECT_EQ(graphics.call(),
              (std::vector<std::string>{"push ExternalVerticesGraphics.graphics", "push triangles", "marker a", "draw",
                                        "marker b", "draw", "pop", "marker c", "pop"}));
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
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    FakeGlProgram program(gl, VERNON_OPENGL_EXPLICIT_SAMPLER_BUNDLE);
    ASSERT_TRUE(program) << program.error();
    const HalImage sampled(gl.get(), VERNON_TEXTURE_RGBA8_UNORM, 4, 4, VERNON_IMAGE_SAMPLED);
    const HalSampler sampler(gl.get(), VERNON_HAL_FILTER_NEAREST);
    ASSERT_TRUE(sampled);
    ASSERT_TRUE(sampler);
    const ColorPass pass(gl, 16, 12, 3);
    program.record().bindImage("image", sampled.reference());
    program.record().bindSampler("sampler", sampler.reference());
    pass.writeTo(program.record());
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    EXPECT_NE(boundSamplerName, 0u);
    EXPECT_NE(boundSamplerName, UINT32_MAX);
}
#endif

#if defined(VERNON_OPENGL_COUNTER_CLOCKWISE_WINDING_BUNDLE) && defined(VERNON_OPENGL_CLOCKWISE_WINDING_BUNDLE)
// GL vertex stages negate y, which mirrors window-space winding, so each declared front face selects the opposite one.
TEST(RuntimeExternalGl, MirrorsDeclaredFrontFaceForTheFlippedVertexY) {
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    const ColorPass pass(gl, 16, 16, 3);
    const std::array<std::pair<const char *, GlEnum>, 2> windings{{
        {VERNON_OPENGL_COUNTER_CLOCKWISE_WINDING_BUNDLE, kClockwise},
        {VERNON_OPENGL_CLOCKWISE_WINDING_BUNDLE, kCounterClockwise},
    }};
    for (const auto &[manifest, expectedWinding] : windings) {
        cullMode = 0;
        frontWinding = 0;
        FakeGlProgram program(gl, manifest);
        ASSERT_TRUE(program) << program.error();
        pass.writeTo(program.record());
        ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
        EXPECT_EQ(cullMode, kBack) << manifest;
        EXPECT_EQ(frontWinding, expectedWinding) << manifest;
    }
}
#endif

#if defined(VERNON_OPENGL_FIXED_FUNCTION_STATE_BUNDLE)
TEST(RuntimeExternalGl, AppliesDeclaredDepthStencilAndBlendState) {
    const FakeGlDevice gl;
    ASSERT_TRUE(gl);
    const HalImage depthStencil(gl.get(), VERNON_TEXTURE_D32_FLOAT_S8_UINT, 16, 16,
                                VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT,
                                VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
    ASSERT_TRUE(depthStencil);
    ColorPass pass(gl, 16, 16, 3);
    VernonDepthAttachment depthAttachment{};
    depthAttachment.view = depthStencil.reference();
    depthAttachment.clear_depth = 1.0F;
    pass.controls.renderPass.depth_attachment = &depthAttachment;
    pass.controls.dynamic.stencil_reference = 23;
    stencilFuncCount = 0;
    stencilReference = 0;
    stencilReadMask = 0;
    depthWrite = 0;
    depthComparison = 0;
    depthBias = {};
    blendFactors = {};
    blendOperations = {};
    colorWriteMask = {};
    FakeGlProgram program(gl, VERNON_OPENGL_FIXED_FUNCTION_STATE_BUNDLE);
    ASSERT_TRUE(program) << program.error();
    pass.writeTo(program.record());
    ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
    EXPECT_EQ(stencilFuncCount, 2U);
    EXPECT_EQ(stencilReference, 23);
    EXPECT_EQ(stencilReadMask, 0x5AU);
    EXPECT_EQ(depthWrite, 1);
    EXPECT_EQ(depthComparison, kLessEqual);
    EXPECT_EQ(depthBias, (std::array<float, 2>{3.0F, 2.0F}));
    EXPECT_EQ(blendFactors, (std::array<GlEnum, 4>{kSourceAlpha, kOneMinusSourceAlpha, kOne, kOneMinusSourceAlpha}));
    EXPECT_EQ(blendOperations, (std::array<GlEnum, 2>{kSubtract, kAdd}));
    EXPECT_EQ(colorWriteMask, (std::array<GlBoolean, 4>{1, 1, 0, 0}));
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

// The inputs of one call of the spilled-uniform bundle; the record's views point into this object, which must outlive
// every call.
struct SpilledUniformInputs {
    explicit SpilledUniformInputs(const FakeGlDevice &gl)
        : sampled(gl.get(), VERNON_TEXTURE_RGBA8_UNORM, 4, 4, VERNON_IMAGE_SAMPLED),
          sampler(gl.get(), VERNON_HAL_FILTER_NEAREST), pass(gl, 16, 16, 3) {
        for (size_t index = 0; index < 4; ++index)
            identity[index * 5] = 1.0F;
    }

    void writeTo(ScheduleRecord &record) {
        static constexpr std::array<uint64_t, 2> shape = {4, 4};
        static constexpr std::array<int64_t, 2> strides = {16, 4};
        for (const char *name : {"projection", "view", "model"})
            record.bind(name,
                        vernon::tests::hostView(identity.data(), sizeof(identity), 2, shape.data(), strides.data()));
        record.bindImage("image", sampled.reference());
        record.bindSampler("sampler", sampler.reference());
        pass.writeTo(record);
    }

    HalImage sampled;
    HalSampler sampler;
    ColorPass pass;
    std::array<float, 16> identity{};
};

// The fragment's image and sampler own bindings 0 and 1, so the spilled `model` block must be moved to binding 2.
TEST(RuntimeExternalGl, PointsSpilledUniformBlocksAtTheirPlannedBinding) {
    const std::string vertex = stageShaderSource(VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE, "vertex");
    EXPECT_NE(vertex.find("uniform vernon_uniform_binding_2"), std::string::npos) << vertex;
    linkedUniformBlocks = {"vernon_uniform_binding_2"};
    uniformBlockBindings.clear();
    {
        const FakeGlDevice gl;
        ASSERT_TRUE(gl);
        FakeGlProgram program(gl, VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE);
        ASSERT_TRUE(program) << program.error();
        SpilledUniformInputs inputs(gl);
        inputs.writeTo(program.record());
        ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
        EXPECT_EQ(uniformBlockBindings, (std::unordered_map<GlUint, GlUint>{{0u, 2u}}));
    }
    linkedUniformBlocks.clear();
}

TEST(RuntimeExternalGl, CallsReuseTheirUniformMemory) {
    linkedUniformBlocks = {"vernon_uniform_binding_2"};
    uniformBufferBindings.clear();
    {
        const FakeGlDevice gl;
        ASSERT_TRUE(gl);
        FakeGlProgram program(gl, VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE);
        ASSERT_TRUE(program) << program.error();
        SpilledUniformInputs inputs(gl);
        inputs.writeTo(program.record());

        constexpr size_t kArenas = 3;
        std::vector<GlUint> pages;
        for (size_t call = 0; call < kArenas; ++call) {
            ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
            pages.push_back(uniformBufferBindings[2]);
            EXPECT_NE(pages.back(), 0u);
        }
        const GlUint namesOnceWarm = nextName;
        for (size_t call = 0; call < 2 * kArenas; ++call) {
            ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
            EXPECT_EQ(uniformBufferBindings[2], pages[call % kArenas]);
        }
        EXPECT_EQ(nextName, namesOnceWarm);
    }
    linkedUniformBlocks.clear();
}

TEST(RuntimeExternalGl, RejectsLinkedUniformBlocksWithoutAPlannedBinding) {
    linkedUniformBlocks = {"vernon_uniform_binding_2", "vernon_uniform_binding_9"};
    {
        const FakeGlDevice gl;
        ASSERT_TRUE(gl);
        FakeGlProgram program(gl, VERNON_OPENGL_SPILLED_UNIFORM_BUNDLE);
        ASSERT_TRUE(program) << program.error();
        SpilledUniformInputs inputs(gl);
        inputs.writeTo(program.record());
        EXPECT_NE(program.call(), VERNON_STATUS_OK);
        EXPECT_NE(program.error().find("uniform block of the Stage has no planned binding"), std::string::npos)
            << program.error();
    }
    linkedUniformBlocks.clear();
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
    {
        const FakeGlDevice gl;
        ASSERT_TRUE(gl);
        FakeGlProgram program(gl, VERNON_OPENGL_INDEXED_VERTEX_BUNDLE);
        ASSERT_TRUE(program) << program.error();
        const GlUint vertexName = nextName;
        const DeviceTensor vertices(gl.device(), nullptr, 36, {3, 3}, sizeof(float), {}, 0, VERNON_HAL_BUFFER_VERTEX);
        const GlUint indexName = nextName;
        const vernon::tests::HalBuffer indices(gl.get(), 12, nullptr, VERNON_HAL_BUFFER_INDEX);
        ColorPass pass(gl, 16, 16, 0);
        pass.draw.instance_count = 3;
        const VernonIndexBinding indexBinding{VERNON_INDEX_U32, 0, 3, indices.reference(), 0};
        pass.draw.index_binding = &indexBinding;
        program.record().bind("position", vertices);
        pass.writeTo(program.record());
        ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
        EXPECT_EQ(boundArrayBuffer, vertexName);
        EXPECT_EQ(boundIndexBuffer, indexName);
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
    }
    EXPECT_EQ(deletedSyncCount, createdSyncCount) << "a destroyed device leaves no fence behind";
}
#endif

#if defined(VERNON_OPENGL_I32_VERTEX_BUNDLE) && defined(VERNON_OPENGL_U32_VERTEX_BUNDLE) &&                            \
    defined(VERNON_OPENGL_F32_VERTEX_BUNDLE)
TEST(RuntimeExternalGl, BindsCompilerSupportedVertexNumericFormatsFromCanonicalPrograms) {
    struct Format {
        const char *name;
        const char *manifest;
        uint32_t size;
        GlEnum nativeType;
        char pointerKind;
    };
    constexpr Format formats[] = {{"i32", VERNON_OPENGL_I32_VERTEX_BUNDLE, 4, kInt, 'I'},
                                  {"u32", VERNON_OPENGL_U32_VERTEX_BUNDLE, 4, kUnsignedInt, 'I'},
                                  {"f32", VERNON_OPENGL_F32_VERTEX_BUNDLE, 4, kFloat, 'F'}};
    for (const Format &format : formats) {
        SCOPED_TRACE(format.name);
        vertexAttributeType = 0;
        vertexAttributePointerKind = '\0';
        const FakeGlDevice gl;
        ASSERT_TRUE(gl);
        FakeGlProgram program(gl, format.manifest);
        ASSERT_TRUE(program) << program.error();
        const uint32_t stride = 3 * format.size;
        const DeviceTensor vertices(gl.device(), nullptr, 3 * stride, {3, 3}, format.size, {}, 0,
                                    VERNON_HAL_BUFFER_VERTEX);
        ColorPass pass(gl, 16, 16, 3);
        pass.draw.instance_count = 3;
        program.record().bind("position", vertices);
        pass.writeTo(program.record());
        ASSERT_EQ(program.call(), VERNON_STATUS_OK) << program.error();
        EXPECT_EQ(vertexAttributeType, format.nativeType);
        EXPECT_EQ(vertexAttributePointerKind, format.pointerKind);
        EXPECT_EQ(vertexAttributeStride, static_cast<GlSize>(stride));
    }
}
#endif

#undef GL_CALL
