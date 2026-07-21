#ifndef VERNON_RUNTIME_BACKEND_OPENGL_DRIVER_H
#define VERNON_RUNTIME_BACKEND_OPENGL_DRIVER_H

#include "VernonRuntime.h"

#include <cstdint>
#include <string>

namespace vernon::runtime {

using GlBoolean = unsigned char;
using GlEnum = unsigned int;
using GlInt = int;
using GlSize = int;
using GlSizePtr = std::intptr_t;
using GlUint = unsigned int;

constexpr GlEnum kArrayBuffer = 0x8892;
constexpr GlEnum kElementArrayBuffer = 0x8893;
constexpr GlEnum kShaderStorageBuffer = 0x90D2;
constexpr GlEnum kDynamicCopy = 0x88EA;
constexpr GlEnum kVertexShader = 0x8B31;
constexpr GlEnum kFragmentShader = 0x8B30;
constexpr GlEnum kComputeShader = 0x91B9;
constexpr GlEnum kCompileStatus = 0x8B81;
constexpr GlEnum kLinkStatus = 0x8B82;
constexpr GlEnum kInfoLogLength = 0x8B84;
constexpr GlEnum kFloat = 0x1406;
constexpr GlEnum kUnsignedInt = 0x1405;
constexpr GlEnum kTexture2D = 0x0DE1;
constexpr GlEnum kFramebuffer = 0x8D40;
constexpr GlEnum kColorAttachment0 = 0x8CE0;
constexpr GlEnum kFramebufferComplete = 0x8CD5;
constexpr GlEnum kTriangles = 0x0004;
constexpr GlEnum kLines = 0x0001;
constexpr GlEnum kPoints = 0x0000;
constexpr GlEnum kNone = 0;
constexpr unsigned kShaderStorageBarrierBit = 0x00002000;
constexpr unsigned kVertexAttribArrayBarrierBit = 0x00000001;

#if defined(_WIN32)
#define VERNON_GL_CALL __stdcall
#else
#define VERNON_GL_CALL
#endif

struct OpenGLDriver {
  GlUint(VERNON_GL_CALL *createShader)(GlEnum) {};
  void(VERNON_GL_CALL *shaderSource)(GlUint, GlSize, const char *const *,
                                     const GlInt *){};
  void(VERNON_GL_CALL *compileShader)(GlUint){};
  void(VERNON_GL_CALL *getShaderiv)(GlUint, GlEnum, GlInt *){};
  void(VERNON_GL_CALL *getShaderInfoLog)(GlUint, GlSize, GlSize *, char *){};
  void(VERNON_GL_CALL *deleteShader)(GlUint){};
  GlUint(VERNON_GL_CALL *createProgram)() {};
  void(VERNON_GL_CALL *attachShader)(GlUint, GlUint){};
  void(VERNON_GL_CALL *linkProgram)(GlUint){};
  void(VERNON_GL_CALL *getProgramiv)(GlUint, GlEnum, GlInt *){};
  void(VERNON_GL_CALL *getProgramInfoLog)(GlUint, GlSize, GlSize *, char *){};
  void(VERNON_GL_CALL *deleteProgram)(GlUint){};
  void(VERNON_GL_CALL *useProgram)(GlUint){};
  void(VERNON_GL_CALL *genBuffers)(GlSize, GlUint *){};
  void(VERNON_GL_CALL *deleteBuffers)(GlSize, const GlUint *){};
  void(VERNON_GL_CALL *bindBuffer)(GlEnum, GlUint){};
  void(VERNON_GL_CALL *bufferData)(GlEnum, GlSizePtr, const void *, GlEnum){};
  void(VERNON_GL_CALL *bindBufferBase)(GlEnum, GlUint, GlUint){};
  void(VERNON_GL_CALL *genVertexArrays)(GlSize, GlUint *){};
  void(VERNON_GL_CALL *deleteVertexArrays)(GlSize, const GlUint *){};
  void(VERNON_GL_CALL *bindVertexArray)(GlUint){};
  void(VERNON_GL_CALL *enableVertexAttribArray)(GlUint){};
  void(VERNON_GL_CALL *vertexAttribPointer)(GlUint, GlInt, GlEnum, GlBoolean,
                                            GlSize, const void *){};
  void(VERNON_GL_CALL *vertexAttribDivisor)(GlUint, GlUint){};
  void(VERNON_GL_CALL *genFramebuffers)(GlSize, GlUint *){};
  void(VERNON_GL_CALL *deleteFramebuffers)(GlSize, const GlUint *){};
  void(VERNON_GL_CALL *bindFramebuffer)(GlEnum, GlUint){};
  void(VERNON_GL_CALL *framebufferTexture2D)(GlEnum, GlEnum, GlEnum, GlUint,
                                             GlInt){};
  GlEnum(VERNON_GL_CALL *checkFramebufferStatus)(GlEnum) {};
  void(VERNON_GL_CALL *drawBuffers)(GlSize, const GlEnum *){};
  void(VERNON_GL_CALL *viewport)(GlInt, GlInt, GlSize, GlSize){};
  void(VERNON_GL_CALL *drawArrays)(GlEnum, GlInt, GlSize){};
  void(VERNON_GL_CALL *drawArraysInstanced)(GlEnum, GlInt, GlSize, GlSize){};
  void(VERNON_GL_CALL *drawElementsInstanced)(GlEnum, GlSize, GlEnum,
                                              const void *, GlSize){};
  GlInt(VERNON_GL_CALL *getUniformLocation)(GlUint, const char *) {};
  void(VERNON_GL_CALL *uniform1fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_CALL *uniform2fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_CALL *uniform3fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_CALL *uniform4fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_CALL *uniformMatrix2fv)(GlInt, GlSize, GlBoolean,
                                         const float *){};
  void(VERNON_GL_CALL *uniformMatrix3fv)(GlInt, GlSize, GlBoolean,
                                         const float *){};
  void(VERNON_GL_CALL *uniformMatrix4fv)(GlInt, GlSize, GlBoolean,
                                         const float *){};
  void(VERNON_GL_CALL *dispatchCompute)(GlUint, GlUint, GlUint){};
  void(VERNON_GL_CALL *memoryBarrier)(unsigned){};
  void(VERNON_GL_CALL *finish)(){};
};

#undef VERNON_GL_CALL

bool loadOpenGLDriver(const VernonExternalOpenGLContext &external,
                      OpenGLDriver &driver, std::string &error);

} // namespace vernon::runtime

#endif
