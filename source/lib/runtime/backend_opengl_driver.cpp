#include "backend_opengl_driver.h"

namespace vernon::runtime {
namespace {

template <typename Function>
bool loadProc(const VernonExternalOpenGLContext &external, Function &function, const char *name, std::string &error) {
    function = reinterpret_cast<Function>(external.get_proc_address(external.user_data, name));
    if (function)
        return true;
    error = std::string("external OpenGL context is missing ") + name;
    return false;
}

} // namespace

bool loadOpenGLDriver(const VernonExternalOpenGLContext &external, OpenGLDriver &driver, std::string &error) {
#define LOAD(member, name)                                                                                             \
    if (!loadProc(external, driver.member, name, error))                                                               \
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
    LOAD(vertexAttribDivisor, "glVertexAttribDivisor");
    LOAD(genFramebuffers, "glGenFramebuffers");
    LOAD(deleteFramebuffers, "glDeleteFramebuffers");
    LOAD(bindFramebuffer, "glBindFramebuffer");
    LOAD(framebufferTexture2D, "glFramebufferTexture2D");
    LOAD(checkFramebufferStatus, "glCheckFramebufferStatus");
    LOAD(drawBuffers, "glDrawBuffers");
    LOAD(clearBufferfv, "glClearBufferfv");
    LOAD(viewport, "glViewport");
    LOAD(drawArrays, "glDrawArrays");
    LOAD(drawArraysInstanced, "glDrawArraysInstanced");
    LOAD(drawElementsInstanced, "glDrawElementsInstanced");
    LOAD(getUniformLocation, "glGetUniformLocation");
    LOAD(uniform1fv, "glUniform1fv");
    LOAD(uniform2fv, "glUniform2fv");
    LOAD(uniform3fv, "glUniform3fv");
    LOAD(uniform4fv, "glUniform4fv");
    LOAD(uniformMatrix2fv, "glUniformMatrix2fv");
    LOAD(uniformMatrix3fv, "glUniformMatrix3fv");
    LOAD(uniformMatrix4fv, "glUniformMatrix4fv");
    LOAD(uniform1i, "glUniform1i");
    LOAD(activeTexture, "glActiveTexture");
    LOAD(genTextures, "glGenTextures");
    LOAD(deleteTextures, "glDeleteTextures");
    LOAD(bindTexture, "glBindTexture");
    LOAD(texImage2D, "glTexImage2D");
    LOAD(texSubImage2D, "glTexSubImage2D");
    LOAD(pixelStorei, "glPixelStorei");
    LOAD(readPixels, "glReadPixels");
    LOAD(genSamplers, "glGenSamplers");
    LOAD(deleteSamplers, "glDeleteSamplers");
    LOAD(samplerParameteri, "glSamplerParameteri");
    LOAD(bindSampler, "glBindSampler");
    LOAD(finish, "glFinish");
#undef LOAD
    driver.dispatchCompute = reinterpret_cast<decltype(driver.dispatchCompute)>(
        external.get_proc_address(external.user_data, "glDispatchCompute"));
    driver.memoryBarrier = reinterpret_cast<decltype(driver.memoryBarrier)>(
        external.get_proc_address(external.user_data, "glMemoryBarrier"));
    return true;
}

} // namespace vernon::runtime
