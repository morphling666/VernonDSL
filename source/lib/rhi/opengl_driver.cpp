#include "opengl_driver.h"

namespace vernon::rhi::opengl {
namespace {

template <typename Function>
bool loadProc(const VernonOpenGLContextCallbacks &callbacks, Function &function, const char *name, std::string &error) {
    function = reinterpret_cast<Function>(callbacks.get_proc_address(callbacks.user_data, name));
    if (function)
        return true;
    error = std::string("external OpenGL context is missing ") + name;
    return false;
}

} // namespace

bool loadDriver(const VernonOpenGLContextCallbacks &callbacks, Driver &driver, std::string &error) {
    if (callbacks.struct_size < sizeof(VernonOpenGLContextCallbacks) || !callbacks.make_current ||
        !callbacks.get_proc_address) {
        error = "external OpenGL context callbacks are missing";
        return false;
    }
    callbacks.make_current(callbacks.user_data);
#define LOAD(member, name)                                                                                             \
    if (!loadProc(callbacks, driver.member, name, error))                                                              \
    return false
    LOAD(createShader, "glCreateShader");
    LOAD(shaderSource, "glShaderSource");
    LOAD(compileShader, "glCompileShader");
    LOAD(getShaderiv, "glGetShaderiv");
    LOAD(getShaderInfoLog, "glGetShaderInfoLog");
    LOAD(deleteShader, "glDeleteShader");
    LOAD(createProgram, "glCreateProgram");
    LOAD(attachShader, "glAttachShader");
    LOAD(linkProgram, "glLinkProgram");
    LOAD(getProgramiv, "glGetProgramiv");
    LOAD(getProgramInfoLog, "glGetProgramInfoLog");
    LOAD(deleteProgram, "glDeleteProgram");
    LOAD(useProgram, "glUseProgram");
    LOAD(getIntegerv, "glGetIntegerv");
    LOAD(genBuffers, "glGenBuffers");
    LOAD(deleteBuffers, "glDeleteBuffers");
    LOAD(bindBuffer, "glBindBuffer");
    LOAD(bufferData, "glBufferData");
    LOAD(bufferSubData, "glBufferSubData");
    LOAD(mapBufferRange, "glMapBufferRange");
    LOAD(unmapBuffer, "glUnmapBuffer");
    LOAD(bindBufferBase, "glBindBufferBase");
    LOAD(genVertexArrays, "glGenVertexArrays");
    LOAD(deleteVertexArrays, "glDeleteVertexArrays");
    LOAD(bindVertexArray, "glBindVertexArray");
    LOAD(enableVertexAttribArray, "glEnableVertexAttribArray");
    LOAD(vertexAttribPointer, "glVertexAttribPointer");
    LOAD(vertexAttribIPointer, "glVertexAttribIPointer");
    driver.vertexAttribLPointer = reinterpret_cast<decltype(driver.vertexAttribLPointer)>(
        callbacks.get_proc_address(callbacks.user_data, "glVertexAttribLPointer"));
    LOAD(vertexAttribDivisor, "glVertexAttribDivisor");
    LOAD(genFramebuffers, "glGenFramebuffers");
    LOAD(deleteFramebuffers, "glDeleteFramebuffers");
    LOAD(bindFramebuffer, "glBindFramebuffer");
    LOAD(framebufferTexture2D, "glFramebufferTexture2D");
    LOAD(checkFramebufferStatus, "glCheckFramebufferStatus");
    LOAD(drawBuffers, "glDrawBuffers");
    driver.invalidateFramebuffer = reinterpret_cast<decltype(driver.invalidateFramebuffer)>(
        callbacks.get_proc_address(callbacks.user_data, "glInvalidateFramebuffer"));
    LOAD(clearBufferfv, "glClearBufferfv");
    LOAD(enable, "glEnable");
    LOAD(disable, "glDisable");
    LOAD(depthFunc, "glDepthFunc");
    LOAD(viewport, "glViewport");
    driver.scissor =
        reinterpret_cast<decltype(driver.scissor)>(callbacks.get_proc_address(callbacks.user_data, "glScissor"));
    LOAD(drawArrays, "glDrawArrays");
    LOAD(drawArraysInstanced, "glDrawArraysInstanced");
    LOAD(drawElementsInstanced, "glDrawElementsInstanced");
    LOAD(getUniformLocation, "glGetUniformLocation");
    LOAD(uniform1fv, "glUniform1fv");
    LOAD(uniform2fv, "glUniform2fv");
    LOAD(uniform3fv, "glUniform3fv");
    LOAD(uniform4fv, "glUniform4fv");
    LOAD(uniform1iv, "glUniform1iv");
    LOAD(uniform2iv, "glUniform2iv");
    LOAD(uniform3iv, "glUniform3iv");
    LOAD(uniform4iv, "glUniform4iv");
    LOAD(uniform1uiv, "glUniform1uiv");
    LOAD(uniform2uiv, "glUniform2uiv");
    LOAD(uniform3uiv, "glUniform3uiv");
    LOAD(uniform4uiv, "glUniform4uiv");
    LOAD(uniformMatrix2fv, "glUniformMatrix2fv");
    LOAD(uniformMatrix2x3fv, "glUniformMatrix2x3fv");
    LOAD(uniformMatrix2x4fv, "glUniformMatrix2x4fv");
    LOAD(uniformMatrix3x2fv, "glUniformMatrix3x2fv");
    LOAD(uniformMatrix3fv, "glUniformMatrix3fv");
    LOAD(uniformMatrix3x4fv, "glUniformMatrix3x4fv");
    LOAD(uniformMatrix4x2fv, "glUniformMatrix4x2fv");
    LOAD(uniformMatrix4x3fv, "glUniformMatrix4x3fv");
    LOAD(uniformMatrix4fv, "glUniformMatrix4fv");
    LOAD(uniform1i, "glUniform1i");
    LOAD(activeTexture, "glActiveTexture");
    LOAD(genTextures, "glGenTextures");
    LOAD(deleteTextures, "glDeleteTextures");
    LOAD(bindTexture, "glBindTexture");
    LOAD(texImage2D, "glTexImage2D");
    LOAD(texImage3D, "glTexImage3D");
    LOAD(texSubImage2D, "glTexSubImage2D");
    LOAD(texSubImage3D, "glTexSubImage3D");
    LOAD(texParameteri, "glTexParameteri");
    LOAD(generateMipmap, "glGenerateMipmap");
    LOAD(pixelStorei, "glPixelStorei");
    LOAD(readPixels, "glReadPixels");
    LOAD(genSamplers, "glGenSamplers");
    LOAD(deleteSamplers, "glDeleteSamplers");
    LOAD(samplerParameteri, "glSamplerParameteri");
    LOAD(bindSampler, "glBindSampler");
    LOAD(finish, "glFinish");
#undef LOAD
    driver.dispatchCompute = reinterpret_cast<decltype(driver.dispatchCompute)>(
        callbacks.get_proc_address(callbacks.user_data, "glDispatchCompute"));
    driver.memoryBarrier = reinterpret_cast<decltype(driver.memoryBarrier)>(
        callbacks.get_proc_address(callbacks.user_data, "glMemoryBarrier"));
    return true;
}

} // namespace vernon::rhi::opengl
