#ifndef VERNON_RHI_OPENGL_DRIVER_H
#define VERNON_RHI_OPENGL_DRIVER_H

#include "VernonOpenGLContext.h"

#include <cstdint>
#include <string>

namespace vernon::rhi::opengl {

using Boolean = unsigned char;
using Bitfield = unsigned int;
using Enum = unsigned int;
using Int = int;
using Size = int;
using SizePtr = std::intptr_t;
using IntPtr = std::intptr_t;
using Uint = unsigned int;

constexpr Enum kArrayBuffer = 0x8892;
constexpr Enum kElementArrayBuffer = 0x8893;
constexpr Enum kUniformBuffer = 0x8A11;
constexpr Enum kShaderStorageBuffer = 0x90D2;
constexpr Enum kDynamicCopy = 0x88EA;
constexpr Bitfield kMapReadBit = 0x0001;
constexpr Enum kVertexShader = 0x8B31;
constexpr Enum kFragmentShader = 0x8B30;
constexpr Enum kComputeShader = 0x91B9;
constexpr Enum kCompileStatus = 0x8B81;
constexpr Enum kLinkStatus = 0x8B82;
constexpr Enum kInfoLogLength = 0x8B84;
constexpr Enum kFloat = 0x1406;
constexpr Enum kHalfFloat = 0x140B;
constexpr Enum kDouble = 0x140A;
constexpr Enum kInt = 0x1404;
constexpr Enum kUnsignedInt = 0x1405;
constexpr Enum kMaxVertexAttribs = 0x8869;
constexpr Enum kTexture2D = 0x0DE1;
constexpr Enum kTexture3D = 0x806F;
constexpr Enum kTextureCubeMap = 0x8513;
constexpr Enum kTextureCubeMapPositiveX = 0x8515;
constexpr Enum kTextureCubeMapSeamless = 0x884F;
constexpr Enum kTexture0 = 0x84C0;
constexpr Enum kFramebuffer = 0x8D40;
constexpr Enum kColorAttachment0 = 0x8CE0;
constexpr Enum kDepthAttachment = 0x8D00;
constexpr Enum kStencilAttachment = 0x8D20;
constexpr Enum kDepthStencilAttachment = 0x821A;
constexpr Enum kFramebufferComplete = 0x8CD5;
constexpr Enum kColor = 0x1800;
constexpr Enum kDepth = 0x1801;
constexpr Enum kStencil = 0x1802;
constexpr Enum kDepthStencil = 0x84F9;
constexpr Enum kDepthComponent32f = 0x8CAC;
constexpr Enum kDepth32fStencil8 = 0x8CAD;
constexpr Enum kDepthTest = 0x0B71;
constexpr Enum kStencilTest = 0x0B90;
constexpr Enum kCullFace = 0x0B44;
constexpr Enum kBlend = 0x0BE2;
constexpr Enum kScissorTest = 0x0C11;
constexpr Enum kLess = 0x0201;
constexpr Enum kTriangles = 0x0004;
constexpr Enum kLines = 0x0001;
constexpr Enum kPoints = 0x0000;
constexpr Enum kNone = 0;
constexpr unsigned kShaderStorageBarrierBit = 0x00002000;
constexpr unsigned kVertexAttribArrayBarrierBit = 0x00000001;
constexpr unsigned kElementArrayBarrierBit = 0x00000002;
constexpr unsigned kUniformBarrierBit = 0x00000004;
constexpr unsigned kTextureFetchBarrierBit = 0x00000008;
constexpr unsigned kShaderImageAccessBarrierBit = 0x00000020;
constexpr unsigned kTextureUpdateBarrierBit = 0x00000100;
constexpr unsigned kBufferUpdateBarrierBit = 0x00000200;
constexpr unsigned kFramebufferBarrierBit = 0x00000400;

#if defined(_WIN32)
#define VERNON_GL_CALL __stdcall
#else
#define VERNON_GL_CALL
#endif

struct Driver {
    Uint(VERNON_GL_CALL *createShader)(Enum) {};
    void(VERNON_GL_CALL *shaderSource)(Uint, Size, const char *const *, const Int *){};
    void(VERNON_GL_CALL *compileShader)(Uint){};
    void(VERNON_GL_CALL *getShaderiv)(Uint, Enum, Int *){};
    void(VERNON_GL_CALL *getShaderInfoLog)(Uint, Size, Size *, char *){};
    void(VERNON_GL_CALL *deleteShader)(Uint){};
    Uint(VERNON_GL_CALL *createProgram)() {};
    void(VERNON_GL_CALL *attachShader)(Uint, Uint){};
    void(VERNON_GL_CALL *linkProgram)(Uint){};
    void(VERNON_GL_CALL *getProgramiv)(Uint, Enum, Int *){};
    void(VERNON_GL_CALL *getProgramInfoLog)(Uint, Size, Size *, char *){};
    void(VERNON_GL_CALL *deleteProgram)(Uint){};
    void(VERNON_GL_CALL *useProgram)(Uint){};
    void(VERNON_GL_CALL *getIntegerv)(Enum, Int *){};
    void(VERNON_GL_CALL *genBuffers)(Size, Uint *){};
    void(VERNON_GL_CALL *deleteBuffers)(Size, const Uint *){};
    void(VERNON_GL_CALL *bindBuffer)(Enum, Uint){};
    void(VERNON_GL_CALL *bufferData)(Enum, SizePtr, const void *, Enum){};
    void(VERNON_GL_CALL *bufferSubData)(Enum, IntPtr, SizePtr, const void *){};
    void *(VERNON_GL_CALL *mapBufferRange)(Enum, IntPtr, SizePtr, Bitfield){};
    Boolean(VERNON_GL_CALL *unmapBuffer)(Enum) {};
    void(VERNON_GL_CALL *bindBufferBase)(Enum, Uint, Uint){};
    void(VERNON_GL_CALL *genVertexArrays)(Size, Uint *){};
    void(VERNON_GL_CALL *deleteVertexArrays)(Size, const Uint *){};
    void(VERNON_GL_CALL *bindVertexArray)(Uint){};
    void(VERNON_GL_CALL *enableVertexAttribArray)(Uint){};
    void(VERNON_GL_CALL *vertexAttribPointer)(Uint, Int, Enum, Boolean, Size, const void *){};
    void(VERNON_GL_CALL *vertexAttribIPointer)(Uint, Int, Enum, Size, const void *){};
    void(VERNON_GL_CALL *vertexAttribLPointer)(Uint, Int, Enum, Size, const void *){};
    void(VERNON_GL_CALL *vertexAttribDivisor)(Uint, Uint){};
    void(VERNON_GL_CALL *genFramebuffers)(Size, Uint *){};
    void(VERNON_GL_CALL *deleteFramebuffers)(Size, const Uint *){};
    void(VERNON_GL_CALL *bindFramebuffer)(Enum, Uint){};
    void(VERNON_GL_CALL *framebufferTexture2D)(Enum, Enum, Enum, Uint, Int){};
    Enum(VERNON_GL_CALL *checkFramebufferStatus)(Enum) {};
    void(VERNON_GL_CALL *drawBuffers)(Size, const Enum *){};
    void(VERNON_GL_CALL *readBuffer)(Enum){};
    void(VERNON_GL_CALL *invalidateFramebuffer)(Enum, Size, const Enum *){};
    void(VERNON_GL_CALL *clearBufferfv)(Enum, Int, const float *){};
    void(VERNON_GL_CALL *clearBufferiv)(Enum, Int, const Int *){};
    void(VERNON_GL_CALL *clearBufferfi)(Enum, Int, float, Int){};
    void(VERNON_GL_CALL *enable)(Enum){};
    void(VERNON_GL_CALL *disable)(Enum){};
    void(VERNON_GL_CALL *depthFunc)(Enum){};
    void(VERNON_GL_CALL *depthMask)(Boolean){};
    void(VERNON_GL_CALL *cullFace)(Enum){};
    void(VERNON_GL_CALL *frontFace)(Enum){};
    void(VERNON_GL_CALL *polygonOffset)(float, float){};
    void(VERNON_GL_CALL *blendFuncSeparate)(Enum, Enum, Enum, Enum){};
    void(VERNON_GL_CALL *blendEquationSeparate)(Enum, Enum){};
    void(VERNON_GL_CALL *colorMask)(Boolean, Boolean, Boolean, Boolean){};
    void(VERNON_GL_CALL *enablei)(Enum, Uint){};
    void(VERNON_GL_CALL *disablei)(Enum, Uint){};
    void(VERNON_GL_CALL *blendFuncSeparatei)(Uint, Enum, Enum, Enum, Enum){};
    void(VERNON_GL_CALL *blendEquationSeparatei)(Uint, Enum, Enum){};
    void(VERNON_GL_CALL *colorMaski)(Uint, Boolean, Boolean, Boolean, Boolean){};
    void(VERNON_GL_CALL *stencilFuncSeparate)(Enum, Enum, Int, Uint){};
    void(VERNON_GL_CALL *stencilOpSeparate)(Enum, Enum, Enum, Enum){};
    void(VERNON_GL_CALL *stencilMaskSeparate)(Enum, Uint){};
    void(VERNON_GL_CALL *viewport)(Int, Int, Size, Size){};
    void(VERNON_GL_CALL *scissor)(Int, Int, Size, Size){};
    void(VERNON_GL_CALL *drawArrays)(Enum, Int, Size){};
    void(VERNON_GL_CALL *drawArraysInstanced)(Enum, Int, Size, Size){};
    void(VERNON_GL_CALL *drawElementsInstanced)(Enum, Size, Enum, const void *, Size){};
    Int(VERNON_GL_CALL *getUniformLocation)(Uint, const char *) {};
    void(VERNON_GL_CALL *uniform1fv)(Int, Size, const float *){};
    void(VERNON_GL_CALL *uniform2fv)(Int, Size, const float *){};
    void(VERNON_GL_CALL *uniform3fv)(Int, Size, const float *){};
    void(VERNON_GL_CALL *uniform4fv)(Int, Size, const float *){};
    void(VERNON_GL_CALL *uniform1iv)(Int, Size, const Int *){};
    void(VERNON_GL_CALL *uniform2iv)(Int, Size, const Int *){};
    void(VERNON_GL_CALL *uniform3iv)(Int, Size, const Int *){};
    void(VERNON_GL_CALL *uniform4iv)(Int, Size, const Int *){};
    void(VERNON_GL_CALL *uniform1uiv)(Int, Size, const Uint *){};
    void(VERNON_GL_CALL *uniform2uiv)(Int, Size, const Uint *){};
    void(VERNON_GL_CALL *uniform3uiv)(Int, Size, const Uint *){};
    void(VERNON_GL_CALL *uniform4uiv)(Int, Size, const Uint *){};
    void(VERNON_GL_CALL *uniformMatrix2fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix2x3fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix2x4fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix3x2fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix3fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix3x4fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix4x2fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix4x3fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniformMatrix4fv)(Int, Size, Boolean, const float *){};
    void(VERNON_GL_CALL *uniform1i)(Int, Int){};
    void(VERNON_GL_CALL *activeTexture)(Enum){};
    void(VERNON_GL_CALL *genTextures)(Size, Uint *){};
    void(VERNON_GL_CALL *deleteTextures)(Size, const Uint *){};
    void(VERNON_GL_CALL *bindTexture)(Enum, Uint){};
    void(VERNON_GL_CALL *texImage2D)(Enum, Int, Int, Size, Size, Int, Enum, Enum, const void *){};
    void(VERNON_GL_CALL *texImage3D)(Enum, Int, Int, Size, Size, Size, Int, Enum, Enum, const void *){};
    void(VERNON_GL_CALL *texSubImage2D)(Enum, Int, Int, Int, Size, Size, Enum, Enum, const void *){};
    void(VERNON_GL_CALL *texSubImage3D)(Enum, Int, Int, Int, Int, Size, Size, Size, Enum, Enum, const void *){};
    void(VERNON_GL_CALL *texParameteri)(Enum, Enum, Int){};
    void(VERNON_GL_CALL *generateMipmap)(Enum){};
    void(VERNON_GL_CALL *pixelStorei)(Enum, Int){};
    void(VERNON_GL_CALL *readPixels)(Int, Int, Size, Size, Enum, Enum, void *){};
    void(VERNON_GL_CALL *genSamplers)(Size, Uint *){};
    void(VERNON_GL_CALL *deleteSamplers)(Size, const Uint *){};
    void(VERNON_GL_CALL *samplerParameteri)(Uint, Enum, Int){};
    void(VERNON_GL_CALL *bindSampler)(Uint, Uint){};
    void(VERNON_GL_CALL *dispatchCompute)(Uint, Uint, Uint){};
    void(VERNON_GL_CALL *memoryBarrier)(unsigned){};
    void(VERNON_GL_CALL *finish)(){};
};

#undef VERNON_GL_CALL

bool loadDriver(const VernonOpenGLContextCallbacks &callbacks, Driver &driver, std::string &error);

} // namespace vernon::rhi::opengl

#endif
