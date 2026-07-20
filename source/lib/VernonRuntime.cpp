#include "VernonRuntime.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include <vulkan/vulkan.h>
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
#include <GLFW/glfw3.h>
#endif

namespace {

#if defined(VERNON_HAS_CUDA_RUNTIME)
using CudaDevice = int;
using CudaDevicePointer = unsigned long long;
using CudaContext = struct CudaContextOpaque *;
using CudaModule = struct CudaModuleOpaque *;
using CudaFunction = struct CudaFunctionOpaque *;
using CudaStream = struct CudaStreamOpaque *;
using CudaResult = int;

constexpr CudaResult kCudaSuccess = 0;

struct CudaDriver {
  using Init = CudaResult (*)(unsigned);
  using DeviceGet = CudaResult (*)(CudaDevice *, int);
  using PrimaryContextRetain = CudaResult (*)(CudaContext *, CudaDevice);
  using PrimaryContextRelease = CudaResult (*)(CudaDevice);
  using ContextSetCurrent = CudaResult (*)(CudaContext);
  using ContextSynchronize = CudaResult (*)();
  using ErrorName = CudaResult (*)(CudaResult, const char **);
  using ErrorString = CudaResult (*)(CudaResult, const char **);
  using MemoryAllocate = CudaResult (*)(CudaDevicePointer *, size_t);
  using MemoryFree = CudaResult (*)(CudaDevicePointer);
  using CopyHostToDevice = CudaResult (*)(CudaDevicePointer, const void *,
                                          size_t);
  using CopyDeviceToHost = CudaResult (*)(void *, CudaDevicePointer, size_t);
  using ModuleLoadData = CudaResult (*)(CudaModule *, const void *, unsigned,
                                        int *, void **);
  using ModuleGetFunction = CudaResult (*)(CudaFunction *, CudaModule,
                                           const char *);
  using ModuleUnload = CudaResult (*)(CudaModule);
  using LaunchKernel = CudaResult (*)(CudaFunction, unsigned, unsigned,
                                      unsigned, unsigned, unsigned, unsigned,
                                      unsigned, CudaStream, void **, void **);

  bool load() {
    std::lock_guard<std::mutex> guard(mutex);
    if (attempted)
      return available;
    attempted = true;
#if defined(_WIN32)
    constexpr const char *libraryName = "nvcuda.dll";
#else
    constexpr const char *libraryName = "libcuda.so.1";
#endif
    library =
        llvm::sys::DynamicLibrary::getPermanentLibrary(libraryName, &error);
    if (!library.isValid())
      return false;

#define VERNON_LOAD_CUDA(member, symbol)                                       \
  member =                                                                     \
      reinterpret_cast<decltype(member)>(library.getAddressOfSymbol(symbol));  \
  if (!member) {                                                               \
    error = std::string("CUDA Driver is missing symbol ") + symbol;            \
    return false;                                                              \
  }
    VERNON_LOAD_CUDA(init, "cuInit");
    VERNON_LOAD_CUDA(deviceGet, "cuDeviceGet");
    VERNON_LOAD_CUDA(primaryContextRetain, "cuDevicePrimaryCtxRetain");
    VERNON_LOAD_CUDA(primaryContextRelease, "cuDevicePrimaryCtxRelease");
    VERNON_LOAD_CUDA(contextSetCurrent, "cuCtxSetCurrent");
    VERNON_LOAD_CUDA(contextSynchronize, "cuCtxSynchronize");
    VERNON_LOAD_CUDA(errorName, "cuGetErrorName");
    VERNON_LOAD_CUDA(errorString, "cuGetErrorString");
    memoryAllocate = reinterpret_cast<MemoryAllocate>(
        library.getAddressOfSymbol("cuMemAlloc_v2"));
    if (!memoryAllocate)
      memoryAllocate = reinterpret_cast<MemoryAllocate>(
          library.getAddressOfSymbol("cuMemAlloc"));
    if (!memoryAllocate) {
      error = "CUDA Driver is missing symbol cuMemAlloc_v2";
      return false;
    }
    memoryFree = reinterpret_cast<MemoryFree>(
        library.getAddressOfSymbol("cuMemFree_v2"));
    if (!memoryFree)
      memoryFree =
          reinterpret_cast<MemoryFree>(library.getAddressOfSymbol("cuMemFree"));
    if (!memoryFree) {
      error = "CUDA Driver is missing symbol cuMemFree_v2";
      return false;
    }
    copyHostToDevice = reinterpret_cast<CopyHostToDevice>(
        library.getAddressOfSymbol("cuMemcpyHtoD_v2"));
    if (!copyHostToDevice)
      copyHostToDevice = reinterpret_cast<CopyHostToDevice>(
          library.getAddressOfSymbol("cuMemcpyHtoD"));
    if (!copyHostToDevice) {
      error = "CUDA Driver is missing symbol cuMemcpyHtoD_v2";
      return false;
    }
    copyDeviceToHost = reinterpret_cast<CopyDeviceToHost>(
        library.getAddressOfSymbol("cuMemcpyDtoH_v2"));
    if (!copyDeviceToHost)
      copyDeviceToHost = reinterpret_cast<CopyDeviceToHost>(
          library.getAddressOfSymbol("cuMemcpyDtoH"));
    if (!copyDeviceToHost) {
      error = "CUDA Driver is missing symbol cuMemcpyDtoH_v2";
      return false;
    }
    VERNON_LOAD_CUDA(moduleLoadData, "cuModuleLoadDataEx");
    VERNON_LOAD_CUDA(moduleGetFunction, "cuModuleGetFunction");
    VERNON_LOAD_CUDA(moduleUnload, "cuModuleUnload");
    VERNON_LOAD_CUDA(launchKernel, "cuLaunchKernel");
#undef VERNON_LOAD_CUDA
    available = true;
    return true;
  }

  llvm::sys::DynamicLibrary library;
  std::mutex mutex;
  std::string error;
  bool attempted{false};
  bool available{false};
  Init init{};
  DeviceGet deviceGet{};
  PrimaryContextRetain primaryContextRetain{};
  PrimaryContextRelease primaryContextRelease{};
  ContextSetCurrent contextSetCurrent{};
  ContextSynchronize contextSynchronize{};
  ErrorName errorName{};
  ErrorString errorString{};
  MemoryAllocate memoryAllocate{};
  MemoryFree memoryFree{};
  CopyHostToDevice copyHostToDevice{};
  CopyDeviceToHost copyDeviceToHost{};
  ModuleLoadData moduleLoadData{};
  ModuleGetFunction moduleGetFunction{};
  ModuleUnload moduleUnload{};
  LaunchKernel launchKernel{};
};

CudaDriver &cudaDriver() {
  static CudaDriver driver;
  return driver;
}
#endif

#if defined(VERNON_HAS_VULKAN_RUNTIME)
struct VulkanDriver {
  bool load() {
    std::lock_guard<std::mutex> guard(mutex);
    if (attempted)
      return available;
    attempted = true;
#if defined(_WIN32)
    constexpr const char *libraryName = "vulkan-1.dll";
#elif defined(__APPLE__)
    constexpr const char *libraryName = "libvulkan.1.dylib";
#else
    constexpr const char *libraryName = "libvulkan.so.1";
#endif
    library =
        llvm::sys::DynamicLibrary::getPermanentLibrary(libraryName, &error);
    if (!library.isValid())
      return false;
    getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        library.getAddressOfSymbol("vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr) {
      error = "Vulkan loader is missing vkGetInstanceProcAddr";
      return false;
    }
    createInstance = reinterpret_cast<PFN_vkCreateInstance>(
        getInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!createInstance) {
      error = "Vulkan loader is missing vkCreateInstance";
      return false;
    }
    available = true;
    return true;
  }

  bool loadInstance(VkInstance instance) {
#define VERNON_LOAD_VULKAN_INSTANCE(member, symbol)                            \
  member = reinterpret_cast<decltype(member)>(                                 \
      getInstanceProcAddr(instance, symbol));                                  \
  if (!member) {                                                               \
    error = std::string("Vulkan loader is missing ") + symbol;                 \
    return false;                                                              \
  }
    VERNON_LOAD_VULKAN_INSTANCE(destroyInstance, "vkDestroyInstance");
    VERNON_LOAD_VULKAN_INSTANCE(enumeratePhysicalDevices,
                                "vkEnumeratePhysicalDevices");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceQueueFamilyProperties,
                                "vkGetPhysicalDeviceQueueFamilyProperties");
    VERNON_LOAD_VULKAN_INSTANCE(getPhysicalDeviceMemoryProperties,
                                "vkGetPhysicalDeviceMemoryProperties");
    VERNON_LOAD_VULKAN_INSTANCE(createDevice, "vkCreateDevice");
    VERNON_LOAD_VULKAN_INSTANCE(getDeviceProcAddr, "vkGetDeviceProcAddr");
#undef VERNON_LOAD_VULKAN_INSTANCE
    return true;
  }

  bool loadDevice(VkDevice device) {
#define VERNON_LOAD_VULKAN_DEVICE(member, symbol)                              \
  member =                                                                     \
      reinterpret_cast<decltype(member)>(getDeviceProcAddr(device, symbol));   \
  if (!member) {                                                               \
    error = std::string("Vulkan device is missing ") + symbol;                 \
    return false;                                                              \
  }
    VERNON_LOAD_VULKAN_DEVICE(destroyDevice, "vkDestroyDevice");
    VERNON_LOAD_VULKAN_DEVICE(deviceWaitIdle, "vkDeviceWaitIdle");
    VERNON_LOAD_VULKAN_DEVICE(getDeviceQueue, "vkGetDeviceQueue");
    VERNON_LOAD_VULKAN_DEVICE(createCommandPool, "vkCreateCommandPool");
    VERNON_LOAD_VULKAN_DEVICE(destroyCommandPool, "vkDestroyCommandPool");
    VERNON_LOAD_VULKAN_DEVICE(allocateCommandBuffers,
                              "vkAllocateCommandBuffers");
    VERNON_LOAD_VULKAN_DEVICE(freeCommandBuffers, "vkFreeCommandBuffers");
    VERNON_LOAD_VULKAN_DEVICE(beginCommandBuffer, "vkBeginCommandBuffer");
    VERNON_LOAD_VULKAN_DEVICE(endCommandBuffer, "vkEndCommandBuffer");
    VERNON_LOAD_VULKAN_DEVICE(queueSubmit, "vkQueueSubmit");
    VERNON_LOAD_VULKAN_DEVICE(queueWaitIdle, "vkQueueWaitIdle");
    VERNON_LOAD_VULKAN_DEVICE(createBuffer, "vkCreateBuffer");
    VERNON_LOAD_VULKAN_DEVICE(destroyBuffer, "vkDestroyBuffer");
    VERNON_LOAD_VULKAN_DEVICE(getBufferMemoryRequirements,
                              "vkGetBufferMemoryRequirements");
    VERNON_LOAD_VULKAN_DEVICE(allocateMemory, "vkAllocateMemory");
    VERNON_LOAD_VULKAN_DEVICE(freeMemory, "vkFreeMemory");
    VERNON_LOAD_VULKAN_DEVICE(bindBufferMemory, "vkBindBufferMemory");
    VERNON_LOAD_VULKAN_DEVICE(mapMemory, "vkMapMemory");
    VERNON_LOAD_VULKAN_DEVICE(unmapMemory, "vkUnmapMemory");
    VERNON_LOAD_VULKAN_DEVICE(createShaderModule, "vkCreateShaderModule");
    VERNON_LOAD_VULKAN_DEVICE(destroyShaderModule, "vkDestroyShaderModule");
    VERNON_LOAD_VULKAN_DEVICE(createDescriptorSetLayout,
                              "vkCreateDescriptorSetLayout");
    VERNON_LOAD_VULKAN_DEVICE(destroyDescriptorSetLayout,
                              "vkDestroyDescriptorSetLayout");
    VERNON_LOAD_VULKAN_DEVICE(createPipelineLayout, "vkCreatePipelineLayout");
    VERNON_LOAD_VULKAN_DEVICE(destroyPipelineLayout, "vkDestroyPipelineLayout");
    VERNON_LOAD_VULKAN_DEVICE(createComputePipelines,
                              "vkCreateComputePipelines");
    VERNON_LOAD_VULKAN_DEVICE(destroyPipeline, "vkDestroyPipeline");
    VERNON_LOAD_VULKAN_DEVICE(createDescriptorPool, "vkCreateDescriptorPool");
    VERNON_LOAD_VULKAN_DEVICE(destroyDescriptorPool, "vkDestroyDescriptorPool");
    VERNON_LOAD_VULKAN_DEVICE(allocateDescriptorSets,
                              "vkAllocateDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(updateDescriptorSets, "vkUpdateDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(cmdBindPipeline, "vkCmdBindPipeline");
    VERNON_LOAD_VULKAN_DEVICE(cmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    VERNON_LOAD_VULKAN_DEVICE(cmdDispatch, "vkCmdDispatch");
#undef VERNON_LOAD_VULKAN_DEVICE
    return true;
  }

  llvm::sys::DynamicLibrary library;
  std::mutex mutex;
  std::string error;
  bool attempted{false};
  bool available{false};
  PFN_vkGetInstanceProcAddr getInstanceProcAddr{};
  PFN_vkCreateInstance createInstance{};
  PFN_vkDestroyInstance destroyInstance{};
  PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices{};
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      getPhysicalDeviceQueueFamilyProperties{};
  PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties{};
  PFN_vkCreateDevice createDevice{};
  PFN_vkGetDeviceProcAddr getDeviceProcAddr{};
  PFN_vkDestroyDevice destroyDevice{};
  PFN_vkDeviceWaitIdle deviceWaitIdle{};
  PFN_vkGetDeviceQueue getDeviceQueue{};
  PFN_vkCreateCommandPool createCommandPool{};
  PFN_vkDestroyCommandPool destroyCommandPool{};
  PFN_vkAllocateCommandBuffers allocateCommandBuffers{};
  PFN_vkFreeCommandBuffers freeCommandBuffers{};
  PFN_vkBeginCommandBuffer beginCommandBuffer{};
  PFN_vkEndCommandBuffer endCommandBuffer{};
  PFN_vkQueueSubmit queueSubmit{};
  PFN_vkQueueWaitIdle queueWaitIdle{};
  PFN_vkCreateBuffer createBuffer{};
  PFN_vkDestroyBuffer destroyBuffer{};
  PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements{};
  PFN_vkAllocateMemory allocateMemory{};
  PFN_vkFreeMemory freeMemory{};
  PFN_vkBindBufferMemory bindBufferMemory{};
  PFN_vkMapMemory mapMemory{};
  PFN_vkUnmapMemory unmapMemory{};
  PFN_vkCreateShaderModule createShaderModule{};
  PFN_vkDestroyShaderModule destroyShaderModule{};
  PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout{};
  PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout{};
  PFN_vkCreatePipelineLayout createPipelineLayout{};
  PFN_vkDestroyPipelineLayout destroyPipelineLayout{};
  PFN_vkCreateComputePipelines createComputePipelines{};
  PFN_vkDestroyPipeline destroyPipeline{};
  PFN_vkCreateDescriptorPool createDescriptorPool{};
  PFN_vkDestroyDescriptorPool destroyDescriptorPool{};
  PFN_vkAllocateDescriptorSets allocateDescriptorSets{};
  PFN_vkUpdateDescriptorSets updateDescriptorSets{};
  PFN_vkCmdBindPipeline cmdBindPipeline{};
  PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets{};
  PFN_vkCmdDispatch cmdDispatch{};
};

VulkanDriver &vulkanDriver() {
  static VulkanDriver driver;
  return driver;
}
#endif

#if defined(VERNON_HAS_OPENGL_RUNTIME)
#if defined(_WIN32)
#define VERNON_GL_APIENTRY __stdcall
#else
#define VERNON_GL_APIENTRY
#endif
using GlEnum = unsigned int;
using GlUint = unsigned int;
using GlInt = int;
using GlSize = int;
using GlBoolean = unsigned char;
using GlBitfield = unsigned int;
using GlSizePtr = intptr_t;
using GlChar = char;

constexpr GlEnum kGlComputeShader = 0x91B9;
constexpr GlEnum kGlVertexShader = 0x8B31;
constexpr GlEnum kGlFragmentShader = 0x8B30;
constexpr GlEnum kGlCompileStatus = 0x8B81;
constexpr GlEnum kGlLinkStatus = 0x8B82;
constexpr GlEnum kGlInfoLogLength = 0x8B84;
constexpr GlEnum kGlShaderStorageBuffer = 0x90D2;
constexpr GlEnum kGlArrayBuffer = 0x8892;
constexpr GlEnum kGlElementArrayBuffer = 0x8893;
constexpr GlEnum kGlDynamicCopy = 0x88EA;
constexpr GlEnum kGlFloat = 0x1406;
constexpr GlEnum kGlUnsignedByte = 0x1401;
constexpr GlEnum kGlUnsignedInt = 0x1405;
constexpr GlEnum kGlMapReadBit = 0x0001;
constexpr GlEnum kGlMapWriteBit = 0x0002;
constexpr GlEnum kGlTexture2D = 0x0DE1;
constexpr GlEnum kGlRgba = 0x1908;
constexpr GlEnum kGlRgba8 = 0x8058;
constexpr GlEnum kGlTextureMinFilter = 0x2801;
constexpr GlEnum kGlTextureMagFilter = 0x2800;
constexpr GlEnum kGlNearest = 0x2600;
constexpr GlEnum kGlFramebuffer = 0x8D40;
constexpr GlEnum kGlColorAttachment0 = 0x8CE0;
constexpr GlEnum kGlFramebufferComplete = 0x8CD5;
constexpr GlEnum kGlTriangles = 0x0004;
constexpr GlEnum kGlLines = 0x0001;
constexpr GlEnum kGlPoints = 0x0000;
constexpr GlEnum kGlNone = 0;
constexpr GlBitfield kGlColorBufferBit = 0x00004000;
constexpr GlBitfield kGlVertexAttribArrayBarrierBit = 0x00000001;
constexpr GlBitfield kGlShaderStorageBarrierBit = 0x00002000;

struct OpenGLDriver {
  bool initialize(std::string &error) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!initialized && !glfwInit()) {
      error = "GLFW could not initialize a window-system backend";
      return false;
    }
    initialized = true;
    ++contexts;
    return true;
  }

  void release() {
    std::lock_guard<std::mutex> guard(mutex);
    if (contexts)
      --contexts;
    if (!contexts && initialized) {
      glfwTerminate();
      initialized = false;
      loaded = false;
    }
  }

  bool load(std::string &error) {
    if (loaded)
      return true;
#define VERNON_LOAD_GL(member, symbol)                                         \
  member = reinterpret_cast<decltype(member)>(glfwGetProcAddress(symbol));     \
  if (!member) {                                                               \
    error = std::string("OpenGL context is missing ") + symbol;                \
    return false;                                                              \
  }
    VERNON_LOAD_GL(createShader, "glCreateShader");
    VERNON_LOAD_GL(shaderSource, "glShaderSource");
    VERNON_LOAD_GL(compileShader, "glCompileShader");
    VERNON_LOAD_GL(getShaderiv, "glGetShaderiv");
    VERNON_LOAD_GL(getShaderInfoLog, "glGetShaderInfoLog");
    VERNON_LOAD_GL(deleteShader, "glDeleteShader");
    VERNON_LOAD_GL(createProgram, "glCreateProgram");
    VERNON_LOAD_GL(attachShader, "glAttachShader");
    VERNON_LOAD_GL(linkProgram, "glLinkProgram");
    VERNON_LOAD_GL(getProgramiv, "glGetProgramiv");
    VERNON_LOAD_GL(getProgramInfoLog, "glGetProgramInfoLog");
    VERNON_LOAD_GL(deleteProgram, "glDeleteProgram");
    VERNON_LOAD_GL(useProgram, "glUseProgram");
    VERNON_LOAD_GL(genBuffers, "glGenBuffers");
    VERNON_LOAD_GL(deleteBuffers, "glDeleteBuffers");
    VERNON_LOAD_GL(bindBuffer, "glBindBuffer");
    VERNON_LOAD_GL(bufferData, "glBufferData");
    VERNON_LOAD_GL(bufferSubData, "glBufferSubData");
    VERNON_LOAD_GL(mapBufferRange, "glMapBufferRange");
    VERNON_LOAD_GL(unmapBuffer, "glUnmapBuffer");
    VERNON_LOAD_GL(bindBufferBase, "glBindBufferBase");
    VERNON_LOAD_GL(genVertexArrays, "glGenVertexArrays");
    VERNON_LOAD_GL(deleteVertexArrays, "glDeleteVertexArrays");
    VERNON_LOAD_GL(bindVertexArray, "glBindVertexArray");
    VERNON_LOAD_GL(enableVertexAttribArray, "glEnableVertexAttribArray");
    VERNON_LOAD_GL(vertexAttribPointer, "glVertexAttribPointer");
    VERNON_LOAD_GL(vertexAttribDivisor, "glVertexAttribDivisor");
    VERNON_LOAD_GL(genTextures, "glGenTextures");
    VERNON_LOAD_GL(deleteTextures, "glDeleteTextures");
    VERNON_LOAD_GL(bindTexture, "glBindTexture");
    VERNON_LOAD_GL(texParameteri, "glTexParameteri");
    VERNON_LOAD_GL(texImage2D, "glTexImage2D");
    VERNON_LOAD_GL(texSubImage2D, "glTexSubImage2D");
    VERNON_LOAD_GL(getTexImage, "glGetTexImage");
    VERNON_LOAD_GL(genFramebuffers, "glGenFramebuffers");
    VERNON_LOAD_GL(deleteFramebuffers, "glDeleteFramebuffers");
    VERNON_LOAD_GL(bindFramebuffer, "glBindFramebuffer");
    VERNON_LOAD_GL(framebufferTexture2D, "glFramebufferTexture2D");
    VERNON_LOAD_GL(checkFramebufferStatus, "glCheckFramebufferStatus");
    VERNON_LOAD_GL(viewport, "glViewport");
    VERNON_LOAD_GL(clearColor, "glClearColor");
    VERNON_LOAD_GL(clear, "glClear");
    VERNON_LOAD_GL(drawArrays, "glDrawArrays");
    VERNON_LOAD_GL(drawArraysInstanced, "glDrawArraysInstanced");
    VERNON_LOAD_GL(drawElementsInstanced, "glDrawElementsInstanced");
    VERNON_LOAD_GL(drawBuffers, "glDrawBuffers");
    VERNON_LOAD_GL(getUniformLocation, "glGetUniformLocation");
    VERNON_LOAD_GL(uniform1fv, "glUniform1fv");
    VERNON_LOAD_GL(uniform2fv, "glUniform2fv");
    VERNON_LOAD_GL(uniform3fv, "glUniform3fv");
    VERNON_LOAD_GL(uniform4fv, "glUniform4fv");
    VERNON_LOAD_GL(uniformMatrix2fv, "glUniformMatrix2fv");
    VERNON_LOAD_GL(uniformMatrix3fv, "glUniformMatrix3fv");
    VERNON_LOAD_GL(uniformMatrix4fv, "glUniformMatrix4fv");
    VERNON_LOAD_GL(finish, "glFinish");
#undef VERNON_LOAD_GL
    dispatchCompute = reinterpret_cast<decltype(dispatchCompute)>(
        glfwGetProcAddress("glDispatchCompute"));
    memoryBarrier = reinterpret_cast<decltype(memoryBarrier)>(
        glfwGetProcAddress("glMemoryBarrier"));
    loaded = true;
    return true;
  }

  std::mutex mutex;
  bool initialized{false};
  bool loaded{false};
  size_t contexts{0};
  GlUint(VERNON_GL_APIENTRY *createShader)(GlEnum) {};
  void(VERNON_GL_APIENTRY *shaderSource)(GlUint, GlSize, const GlChar *const *,
                                         const GlInt *){};
  void(VERNON_GL_APIENTRY *compileShader)(GlUint){};
  void(VERNON_GL_APIENTRY *getShaderiv)(GlUint, GlEnum, GlInt *){};
  void(VERNON_GL_APIENTRY *getShaderInfoLog)(GlUint, GlSize, GlSize *,
                                             GlChar *){};
  void(VERNON_GL_APIENTRY *deleteShader)(GlUint){};
  GlUint(VERNON_GL_APIENTRY *createProgram)() {};
  void(VERNON_GL_APIENTRY *attachShader)(GlUint, GlUint){};
  void(VERNON_GL_APIENTRY *linkProgram)(GlUint){};
  void(VERNON_GL_APIENTRY *getProgramiv)(GlUint, GlEnum, GlInt *){};
  void(VERNON_GL_APIENTRY *getProgramInfoLog)(GlUint, GlSize, GlSize *,
                                              GlChar *){};
  void(VERNON_GL_APIENTRY *deleteProgram)(GlUint){};
  void(VERNON_GL_APIENTRY *useProgram)(GlUint){};
  void(VERNON_GL_APIENTRY *genBuffers)(GlSize, GlUint *){};
  void(VERNON_GL_APIENTRY *deleteBuffers)(GlSize, const GlUint *){};
  void(VERNON_GL_APIENTRY *bindBuffer)(GlEnum, GlUint){};
  void(VERNON_GL_APIENTRY *bufferData)(GlEnum, GlSizePtr, const void *,
                                       GlEnum){};
  void(VERNON_GL_APIENTRY *bufferSubData)(GlEnum, GlSizePtr, GlSizePtr,
                                          const void *){};
  void *(VERNON_GL_APIENTRY *mapBufferRange)(GlEnum, GlSizePtr, GlSizePtr,
                                             GlBitfield){};
  GlBoolean(VERNON_GL_APIENTRY *unmapBuffer)(GlEnum) {};
  void(VERNON_GL_APIENTRY *bindBufferBase)(GlEnum, GlUint, GlUint){};
  void(VERNON_GL_APIENTRY *genVertexArrays)(GlSize, GlUint *){};
  void(VERNON_GL_APIENTRY *deleteVertexArrays)(GlSize, const GlUint *){};
  void(VERNON_GL_APIENTRY *bindVertexArray)(GlUint){};
  void(VERNON_GL_APIENTRY *enableVertexAttribArray)(GlUint){};
  void(VERNON_GL_APIENTRY *vertexAttribPointer)(GlUint, GlInt, GlEnum,
                                                GlBoolean, GlSize,
                                                const void *){};
  void(VERNON_GL_APIENTRY *vertexAttribDivisor)(GlUint, GlUint){};
  void(VERNON_GL_APIENTRY *genTextures)(GlSize, GlUint *){};
  void(VERNON_GL_APIENTRY *deleteTextures)(GlSize, const GlUint *){};
  void(VERNON_GL_APIENTRY *bindTexture)(GlEnum, GlUint){};
  void(VERNON_GL_APIENTRY *texParameteri)(GlEnum, GlEnum, GlInt){};
  void(VERNON_GL_APIENTRY *texImage2D)(GlEnum, GlInt, GlInt, GlSize, GlSize,
                                       GlInt, GlEnum, GlEnum, const void *){};
  void(VERNON_GL_APIENTRY *texSubImage2D)(GlEnum, GlInt, GlInt, GlInt, GlSize,
                                          GlSize, GlEnum, GlEnum,
                                          const void *){};
  void(VERNON_GL_APIENTRY *getTexImage)(GlEnum, GlInt, GlEnum, GlEnum,
                                        void *){};
  void(VERNON_GL_APIENTRY *genFramebuffers)(GlSize, GlUint *){};
  void(VERNON_GL_APIENTRY *deleteFramebuffers)(GlSize, const GlUint *){};
  void(VERNON_GL_APIENTRY *bindFramebuffer)(GlEnum, GlUint){};
  void(VERNON_GL_APIENTRY *framebufferTexture2D)(GlEnum, GlEnum, GlEnum, GlUint,
                                                 GlInt){};
  GlEnum(VERNON_GL_APIENTRY *checkFramebufferStatus)(GlEnum) {};
  void(VERNON_GL_APIENTRY *viewport)(GlInt, GlInt, GlSize, GlSize){};
  void(VERNON_GL_APIENTRY *clearColor)(float, float, float, float){};
  void(VERNON_GL_APIENTRY *clear)(GlBitfield){};
  void(VERNON_GL_APIENTRY *drawArrays)(GlEnum, GlInt, GlSize){};
  void(VERNON_GL_APIENTRY *drawArraysInstanced)(GlEnum, GlInt, GlSize,
                                                GlSize){};
  void(VERNON_GL_APIENTRY *drawElementsInstanced)(GlEnum, GlSize, GlEnum,
                                                  const void *, GlSize){};
  void(VERNON_GL_APIENTRY *drawBuffers)(GlSize, const GlEnum *){};
  GlInt(VERNON_GL_APIENTRY *getUniformLocation)(GlUint, const GlChar *) {};
  void(VERNON_GL_APIENTRY *uniform1fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_APIENTRY *uniform2fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_APIENTRY *uniform3fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_APIENTRY *uniform4fv)(GlInt, GlSize, const float *){};
  void(VERNON_GL_APIENTRY *uniformMatrix2fv)(GlInt, GlSize, GlBoolean,
                                             const float *){};
  void(VERNON_GL_APIENTRY *uniformMatrix3fv)(GlInt, GlSize, GlBoolean,
                                             const float *){};
  void(VERNON_GL_APIENTRY *uniformMatrix4fv)(GlInt, GlSize, GlBoolean,
                                             const float *){};
  void(VERNON_GL_APIENTRY *dispatchCompute)(GlUint, GlUint, GlUint){};
  void(VERNON_GL_APIENTRY *memoryBarrier)(GlBitfield){};
  void(VERNON_GL_APIENTRY *finish)(){};
};

OpenGLDriver &openGLDriver() {
  static OpenGLDriver driver;
  return driver;
}
#undef VERNON_GL_APIENTRY
#endif

struct ReflectedArgument {
  std::string kind;
  std::string builtin;
  size_t cpuOffset{0};
  size_t cpuSize{0};
  size_t tensorBytes{0};
  size_t tensorElements{0};
  size_t tensorElementSize{0};
  size_t alignment{1};
  uint32_t descriptorSet{0};
  uint32_t binding{UINT32_MAX};
};

struct ReflectedEntry {
  std::vector<ReflectedArgument> arguments;
  size_t cpuArgumentsSize{0};
  uint32_t workgroup[3]{1, 1, 1};
};

bool parseReflection(llvm::StringRef text, llvm::StringRef selected,
                     ReflectedEntry &output, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    error = "invalid reflection JSON";
    return false;
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getInteger("gpu_launch_abi_version").value_or(0) != 1) {
    error = "unsupported GPU launch ABI version";
    return false;
  }
  llvm::json::Array *entries = root->getArray("entries");
  if (!entries) {
    error = "reflection has no entries";
    return false;
  }
  for (llvm::json::Value &value : *entries) {
    llvm::json::Object *entry = value.getAsObject();
    if (!entry || entry->getString("name").value_or("") != selected)
      continue;
    output.cpuArgumentsSize = static_cast<size_t>(
        entry->getInteger("cpu_arguments_size").value_or(0));
    if (llvm::json::Array *workgroup = entry->getArray("workgroup_size")) {
      if (workgroup->size() == 3) {
        for (size_t index = 0; index < 3; ++index)
          output.workgroup[index] = static_cast<uint32_t>(
              (*workgroup)[index].getAsInteger().value_or(1));
      }
    }
    llvm::json::Array *arguments = entry->getArray("arguments");
    if (!arguments) {
      error = "entry has no argument reflection";
      return false;
    }
    for (llvm::json::Value &argumentValue : *arguments) {
      llvm::json::Object *argument = argumentValue.getAsObject();
      if (!argument) {
        error = "invalid argument reflection";
        return false;
      }
      ReflectedArgument reflected;
      reflected.kind = argument->getString("kind").value_or("scalar").str();
      reflected.builtin = argument->getString("builtin").value_or("").str();
      reflected.cpuOffset =
          static_cast<size_t>(argument->getInteger("cpu_offset").value_or(0));
      reflected.cpuSize =
          static_cast<size_t>(argument->getInteger("cpu_size").value_or(0));
      reflected.alignment =
          static_cast<size_t>(argument->getInteger("alignment").value_or(1));
      reflected.descriptorSet =
          static_cast<uint32_t>(argument->getInteger("vernon.set").value_or(0));
      if (std::optional<int64_t> binding =
              argument->getInteger("vernon.binding"))
        reflected.binding = static_cast<uint32_t>(*binding);
      llvm::StringRef dtype = argument->getString("dtype").value_or("");
      reflected.tensorElementSize = dtype == "f64"    ? 8
                                    : dtype == "f16"  ? 2
                                    : dtype == "bool" ? 1
                                                      : 4;
      if (llvm::json::Array *shape = argument->getArray("shape")) {
        size_t elements = 1;
        for (llvm::json::Value &dimension : *shape)
          elements *= static_cast<size_t>(dimension.getAsInteger().value_or(0));
        reflected.tensorElements = elements;
        reflected.tensorBytes = elements * reflected.tensorElementSize;
      }
      output.arguments.push_back(std::move(reflected));
    }
    return true;
  }
  error = "selected entry is absent from reflection";
  return false;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

} // namespace

struct VernonRuntimeContext {
  VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
  std::string error;
  size_t liveBuffers{0};
  size_t liveKernels{0};
  size_t liveTextures{0};
  size_t livePrograms{0};
  uint16_t apiVersionMajor{0};
  uint16_t apiVersionMinor{0};
  uint16_t requestedApiVersionMajor{0};
  uint16_t requestedApiVersionMinor{0};
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaDevice device{};
  CudaContext cudaContext{};
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkInstance vulkanInstance{};
  VkPhysicalDevice vulkanPhysicalDevice{};
  VkDevice vulkanDevice{};
  VkQueue vulkanQueue{};
  uint32_t vulkanQueueFamily{};
  VkCommandPool vulkanCommandPool{};
  VkPhysicalDeviceMemoryProperties vulkanMemoryProperties{};
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  GLFWwindow *openGLWindow{};
#endif
};

struct VernonDeviceBuffer {
  VernonRuntimeContext *context{};
  size_t size{};
  size_t alignment{};
  std::vector<unsigned char> host;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaDevicePointer device{};
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkBuffer vulkanBuffer{};
  VkDeviceMemory vulkanMemory{};
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  GlUint openGLBuffer{};
#endif
};

struct VernonDeviceTexture {
  VernonRuntimeContext *context{};
  uint32_t width{};
  uint32_t height{};
  VernonTextureFormat format{VERNON_TEXTURE_RGBA8_UNORM};
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  GlUint openGLTexture{};
#endif
};

struct VernonLoadedKernel {
  VernonRuntimeContext *context{};
  ReflectedEntry reflection;
  VernonCpuEntryPoint cpuEntry{};
  std::unique_ptr<llvm::orc::LLJIT> cpuJit;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaModule cudaModule{};
  CudaFunction cudaFunction{};
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkShaderModule vulkanShader{};
  VkDescriptorSetLayout vulkanDescriptorSetLayout{};
  VkPipelineLayout vulkanPipelineLayout{};
  VkPipeline vulkanPipeline{};
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  GlUint openGLProgram{};
#endif
};

struct VernonLoadedProgram {
  VernonRuntimeContext *context{};
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  GlUint openGLProgram{};
  GlUint openGLVertexArray{};
  GlUint openGLFramebuffer{};
  std::vector<uint32_t> openGLAttachmentLocations;
#endif
};

namespace {

VernonStatus fail(VernonRuntimeContext *context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
  if (context)
    context->error = std::move(message);
  return status;
}

#if defined(VERNON_HAS_CUDA_RUNTIME)
VernonStatus cudaFail(VernonRuntimeContext *context, CudaResult result,
                      const char *operation) {
  if (result == kCudaSuccess)
    return VERNON_STATUS_OK;
  const char *name = nullptr;
  const char *description = nullptr;
  CudaDriver &driver = cudaDriver();
  driver.errorName(result, &name);
  driver.errorString(result, &description);
  return fail(context,
              std::string(operation) +
                  " failed: " + (name ? name : "CUDA_ERROR") + " (" +
                  (description ? description : "unknown") + ")",
              VERNON_STATUS_INTERNAL_ERROR);
}
#endif

#if defined(VERNON_HAS_VULKAN_RUNTIME)
VernonStatus vulkanFail(VernonRuntimeContext *context, VkResult result,
                        const char *operation) {
  if (result == VK_SUCCESS)
    return VERNON_STATUS_OK;
  return fail(context,
              std::string(operation) + " failed with VkResult " +
                  std::to_string(result),
              VERNON_STATUS_INTERNAL_ERROR);
}

std::optional<uint32_t>
findVulkanMemoryType(const VernonRuntimeContext *context, uint32_t typeBits,
                     VkMemoryPropertyFlags required) {
  for (uint32_t index = 0;
       index < context->vulkanMemoryProperties.memoryTypeCount; ++index) {
    if ((typeBits & (uint32_t{1} << index)) &&
        (context->vulkanMemoryProperties.memoryTypes[index].propertyFlags &
         required) == required)
      return index;
  }
  return std::nullopt;
}

bool createVulkanBuffer(VernonRuntimeContext *context, VkDeviceSize size,
                        VkBuffer &buffer, VkDeviceMemory &memory) {
  VulkanDriver &driver = vulkanDriver();
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vulkanFail(context,
                 driver.createBuffer(context->vulkanDevice, &bufferInfo,
                                     nullptr, &buffer),
                 "vkCreateBuffer") != VERNON_STATUS_OK)
    return false;
  VkMemoryRequirements requirements{};
  driver.getBufferMemoryRequirements(context->vulkanDevice, buffer,
                                     &requirements);
  std::optional<uint32_t> memoryType =
      findVulkanMemoryType(context, requirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (!memoryType) {
    driver.destroyBuffer(context->vulkanDevice, buffer, nullptr);
    buffer = VK_NULL_HANDLE;
    fail(context, "Vulkan device has no host-visible coherent memory type");
    return false;
  }
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = *memoryType;
  if (vulkanFail(context,
                 driver.allocateMemory(context->vulkanDevice, &allocation,
                                       nullptr, &memory),
                 "vkAllocateMemory") != VERNON_STATUS_OK) {
    driver.destroyBuffer(context->vulkanDevice, buffer, nullptr);
    buffer = VK_NULL_HANDLE;
    return false;
  }
  if (vulkanFail(
          context,
          driver.bindBufferMemory(context->vulkanDevice, buffer, memory, 0),
          "vkBindBufferMemory") != VERNON_STATUS_OK) {
    driver.freeMemory(context->vulkanDevice, memory, nullptr);
    driver.destroyBuffer(context->vulkanDevice, buffer, nullptr);
    memory = VK_NULL_HANDLE;
    buffer = VK_NULL_HANDLE;
    return false;
  }
  return true;
}

bool initializeVulkanContext(VernonRuntimeContext *context,
                             uint32_t deviceIndex) {
  VulkanDriver &driver = vulkanDriver();
  if (!driver.load()) {
    context->error = driver.error;
    return false;
  }
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "VernonDSL";
  application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  application.pEngineName = "VernonDSL";
  application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  application.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &application;
  if (vulkanFail(context,
                 driver.createInstance(&instanceInfo, nullptr,
                                       &context->vulkanInstance),
                 "vkCreateInstance") != VERNON_STATUS_OK)
    return false;
  if (!driver.loadInstance(context->vulkanInstance)) {
    context->error = driver.error;
    return false;
  }
  uint32_t physicalDeviceCount = 0;
  if (driver.enumeratePhysicalDevices(context->vulkanInstance,
                                      &physicalDeviceCount,
                                      nullptr) != VK_SUCCESS ||
      deviceIndex >= physicalDeviceCount) {
    context->error = "Vulkan device index is unavailable";
    return false;
  }
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  if (driver.enumeratePhysicalDevices(context->vulkanInstance,
                                      &physicalDeviceCount,
                                      physicalDevices.data()) != VK_SUCCESS)
    return false;
  context->vulkanPhysicalDevice = physicalDevices[deviceIndex];
  uint32_t queueCount = 0;
  driver.getPhysicalDeviceQueueFamilyProperties(context->vulkanPhysicalDevice,
                                                &queueCount, nullptr);
  std::vector<VkQueueFamilyProperties> queues(queueCount);
  driver.getPhysicalDeviceQueueFamilyProperties(context->vulkanPhysicalDevice,
                                                &queueCount, queues.data());
  auto queue = llvm::find_if(queues, [](const VkQueueFamilyProperties &family) {
    return family.queueFlags & VK_QUEUE_COMPUTE_BIT;
  });
  if (queue == queues.end()) {
    context->error = "Vulkan device has no compute queue";
    return false;
  }
  context->vulkanQueueFamily =
      static_cast<uint32_t>(std::distance(queues.begin(), queue));
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = context->vulkanQueueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;
  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  if (vulkanFail(context,
                 driver.createDevice(context->vulkanPhysicalDevice, &deviceInfo,
                                     nullptr, &context->vulkanDevice),
                 "vkCreateDevice") != VERNON_STATUS_OK)
    return false;
  if (!driver.loadDevice(context->vulkanDevice)) {
    context->error = driver.error;
    return false;
  }
  driver.getDeviceQueue(context->vulkanDevice, context->vulkanQueueFamily, 0,
                        &context->vulkanQueue);
  driver.getPhysicalDeviceMemoryProperties(context->vulkanPhysicalDevice,
                                           &context->vulkanMemoryProperties);
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = context->vulkanQueueFamily;
  return vulkanFail(context,
                    driver.createCommandPool(context->vulkanDevice, &poolInfo,
                                             nullptr,
                                             &context->vulkanCommandPool),
                    "vkCreateCommandPool") == VERNON_STATUS_OK;
}

void destroyVulkanContext(VernonRuntimeContext *context) {
  VulkanDriver &driver = vulkanDriver();
  if (context->vulkanDevice) {
    driver.deviceWaitIdle(context->vulkanDevice);
    if (context->vulkanCommandPool)
      driver.destroyCommandPool(context->vulkanDevice,
                                context->vulkanCommandPool, nullptr);
    driver.destroyDevice(context->vulkanDevice, nullptr);
  }
  if (context->vulkanInstance && driver.destroyInstance)
    driver.destroyInstance(context->vulkanInstance, nullptr);
  context->vulkanCommandPool = VK_NULL_HANDLE;
  context->vulkanDevice = VK_NULL_HANDLE;
  context->vulkanInstance = VK_NULL_HANDLE;
}
#endif

#if defined(VERNON_HAS_OPENGL_RUNTIME)
bool initializeOpenGLContext(VernonRuntimeContext *context, uint16_t major = 4,
                             uint16_t minor = 3) {
  OpenGLDriver &driver = openGLDriver();
  if (!driver.initialize(context->error))
    return false;
  glfwDefaultWindowHints();
  context->requestedApiVersionMajor = major;
  context->requestedApiVersionMinor = minor;
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  context->openGLWindow =
      glfwCreateWindow(1, 1, "VernonDSL Runtime", nullptr, nullptr);
  if (!context->openGLWindow) {
    context->error = "cannot create requested OpenGL " + std::to_string(major) +
                     "." + std::to_string(minor) + " context";
    driver.release();
    return false;
  }
  glfwMakeContextCurrent(context->openGLWindow);
  if (!driver.load(context->error)) {
    glfwDestroyWindow(context->openGLWindow);
    context->openGLWindow = nullptr;
    driver.release();
    return false;
  }
  context->apiVersionMajor = static_cast<uint16_t>(
      glfwGetWindowAttrib(context->openGLWindow, GLFW_CONTEXT_VERSION_MAJOR));
  context->apiVersionMinor = static_cast<uint16_t>(
      glfwGetWindowAttrib(context->openGLWindow, GLFW_CONTEXT_VERSION_MINOR));
  if ((major > 4 || (major == 4 && minor >= 3)) &&
      (!driver.dispatchCompute || !driver.memoryBarrier)) {
    context->error =
        "requested OpenGL context does not expose compute shader operations";
    glfwDestroyWindow(context->openGLWindow);
    context->openGLWindow = nullptr;
    driver.release();
    return false;
  }
  return true;
}

void destroyOpenGLContext(VernonRuntimeContext *context) {
  if (!context->openGLWindow)
    return;
  glfwMakeContextCurrent(context->openGLWindow);
  openGLDriver().finish();
  glfwDestroyWindow(context->openGLWindow);
  context->openGLWindow = nullptr;
  openGLDriver().release();
}

std::string getOpenGLShaderLog(OpenGLDriver &driver, GlUint shader) {
  GlInt length = 0;
  driver.getShaderiv(shader, kGlInfoLogLength, &length);
  std::string log(std::max(length, 1), '\0');
  GlSize written = 0;
  driver.getShaderInfoLog(shader, static_cast<GlSize>(log.size()), &written,
                          log.data());
  log.resize(std::max(written, 0));
  return log;
}

std::string getOpenGLProgramLog(OpenGLDriver &driver, GlUint program) {
  GlInt length = 0;
  driver.getProgramiv(program, kGlInfoLogLength, &length);
  std::string log(std::max(length, 1), '\0');
  GlSize written = 0;
  driver.getProgramInfoLog(program, static_cast<GlSize>(log.size()), &written,
                           log.data());
  log.resize(std::max(written, 0));
  return log;
}
#endif

} // namespace

extern "C" {

VernonRuntimeCapabilities
vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
  static std::string diagnostic;
  diagnostic.clear();
  VernonRuntimeCapabilities capabilities{};
  capabilities.supports_compute = 1;
  if (backend == VERNON_RUNTIME_CPU) {
    capabilities.available = 1;
  } else if (backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaDriver &driver = cudaDriver();
    if (!driver.load()) {
      diagnostic = driver.error.empty() ? "CUDA Driver API is unavailable"
                                        : driver.error;
    } else {
      CudaDevice device{};
      CudaResult status = driver.init(0);
      if (status == kCudaSuccess)
        status = driver.deviceGet(&device, 0);
      capabilities.available = status == kCudaSuccess;
      if (!capabilities.available)
        diagnostic = "CUDA Driver API loaded but no usable device was found";
    }
#else
    diagnostic = "VernonDSLRuntime was built without CUDA Driver support";
#endif
  } else if (backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VernonRuntimeContext probe;
    probe.backend = VERNON_RUNTIME_VULKAN;
    capabilities.available = initializeVulkanContext(&probe, 0);
    diagnostic = probe.error;
    destroyVulkanContext(&probe);
#else
    diagnostic = "VernonDSLRuntime was built without Vulkan support";
#endif
  } else if (backend == VERNON_RUNTIME_OPENGL ||
             backend == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_HAS_OPENGL_RUNTIME)
    VernonRuntimeContext probe;
    probe.backend = backend;
    capabilities.available = initializeOpenGLContext(&probe);
    capabilities.supports_graphics = capabilities.available;
    capabilities.supports_storage_buffers = capabilities.available;
    capabilities.api_version_major = probe.apiVersionMajor;
    capabilities.api_version_minor = probe.apiVersionMinor;
    capabilities.graphics_draw_abi_version = capabilities.available ? 2 : 0;
    diagnostic = probe.error;
    destroyOpenGLContext(&probe);
#else
    diagnostic = "VernonDSLRuntime was built without OpenGL support";
#endif
  } else {
    diagnostic = "unknown runtime backend";
  }
  if (capabilities.available && backend != VERNON_RUNTIME_OPENGL &&
      backend != VERNON_RUNTIME_OPENGL_ES)
    capabilities.supports_storage_buffers = 1;
  capabilities.diagnostic = {diagnostic.data(), diagnostic.size()};
  return capabilities;
}

VernonRuntimeCapabilities
vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context) {
  VernonRuntimeCapabilities capabilities{};
  if (!context)
    return capabilities;
  capabilities.available = 1;
  capabilities.diagnostic = {context->error.data(), context->error.size()};
  if (context->backend == VERNON_RUNTIME_OPENGL ||
      context->backend == VERNON_RUNTIME_OPENGL_ES) {
    capabilities.supports_graphics = 1;
    capabilities.supports_compute = context->requestedApiVersionMajor > 4 ||
                                    (context->requestedApiVersionMajor == 4 &&
                                     context->requestedApiVersionMinor >= 3);
    capabilities.supports_storage_buffers = capabilities.supports_compute;
    capabilities.api_version_major = context->apiVersionMajor;
    capabilities.api_version_minor = context->apiVersionMinor;
    capabilities.graphics_draw_abi_version = 2;
  } else {
    capabilities.supports_compute = 1;
    capabilities.supports_storage_buffers = 1;
  }
  return capabilities;
}

VernonRuntimeContext *vernonRuntimeCreate(VernonRuntimeBackend backend,
                                          uint32_t deviceIndex) {
  VernonRuntimeCreateOptions options{};
  options.struct_size = sizeof(options);
  options.device_index = deviceIndex;
  if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
    options.api_version_major = 4;
    options.api_version_minor = 3;
  }
  return vernonRuntimeCreateWithOptions(backend, &options);
}

VernonRuntimeContext *
vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                               const VernonRuntimeCreateOptions *options) {
  if (!options || options->struct_size < sizeof(VernonRuntimeCreateOptions))
    return nullptr;
  const uint32_t deviceIndex = options->device_index;
  static std::once_flag nativeTargetInitialization;
  std::call_once(nativeTargetInitialization, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
  auto context = std::make_unique<VernonRuntimeContext>();
  context->backend = backend;
  if (backend == VERNON_RUNTIME_CPU)
    return context.release();
  if (backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!initializeVulkanContext(context.get(), deviceIndex)) {
      destroyVulkanContext(context.get());
      return nullptr;
    }
    return context.release();
#else
    (void)deviceIndex;
    return nullptr;
#endif
  }
  if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_HAS_OPENGL_RUNTIME)
    const uint16_t major =
        options->api_version_major ? options->api_version_major : 4;
    const uint16_t minor =
        options->api_version_major ? options->api_version_minor : 3;
    if (major < 3 || (major == 3 && minor < 3) ||
        !initializeOpenGLContext(context.get(), major, minor))
      return nullptr;
    return context.release();
#else
    (void)deviceIndex;
    return nullptr;
#endif
  }
  if (backend != VERNON_RUNTIME_CUDA)
    return nullptr;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaDriver &driver = cudaDriver();
  if (!driver.load() || driver.init(0) != kCudaSuccess ||
      driver.deviceGet(&context->device, static_cast<int>(deviceIndex)) !=
          kCudaSuccess ||
      driver.primaryContextRetain(&context->cudaContext, context->device) !=
          kCudaSuccess)
    return nullptr;
  if (driver.contextSetCurrent(context->cudaContext) != kCudaSuccess) {
    driver.primaryContextRelease(context->device);
    return nullptr;
  }
  return context.release();
#else
  (void)deviceIndex;
  return nullptr;
#endif
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_OK;
  if (context->liveBuffers || context->liveKernels || context->liveTextures ||
      context->livePrograms)
    return fail(context, "runtime context still owns live handles");
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA) {
    cudaDriver().contextSynchronize();
    cudaDriver().primaryContextRelease(context->device);
  }
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (context->backend == VERNON_RUNTIME_VULKAN)
    destroyVulkanContext(context);
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  if (context->backend == VERNON_RUNTIME_OPENGL ||
      context->backend == VERNON_RUNTIME_OPENGL_ES)
    destroyOpenGLContext(context);
#endif
  delete context;
  return VERNON_STATUS_OK;
}

VernonStringView
vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
  if (!context)
    return {nullptr, 0};
  return {context->error.data(), context->error.size()};
}

VernonDeviceBuffer *vernonRuntimeBufferAllocate(VernonRuntimeContext *context,
                                                size_t size, size_t alignment) {
  if (!context || !size || !alignment || (alignment & (alignment - 1))) {
    fail(context, "buffer size and power-of-two alignment are required");
    return nullptr;
  }
  auto buffer = std::make_unique<VernonDeviceBuffer>();
  buffer->context = context;
  buffer->size = size;
  buffer->alignment = alignment;
  if (context->backend == VERNON_RUNTIME_CPU) {
    buffer->host.resize(size);
  } else if (context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!createVulkanBuffer(context, size, buffer->vulkanBuffer,
                            buffer->vulkanMemory))
      return nullptr;
#else
    return nullptr;
#endif
  } else if (context->backend == VERNON_RUNTIME_OPENGL ||
             context->backend == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_HAS_OPENGL_RUNTIME)
    glfwMakeContextCurrent(context->openGLWindow);
    OpenGLDriver &driver = openGLDriver();
    driver.genBuffers(1, &buffer->openGLBuffer);
    driver.bindBuffer(kGlShaderStorageBuffer, buffer->openGLBuffer);
    driver.bufferData(kGlShaderStorageBuffer, static_cast<GlSizePtr>(size),
                      nullptr, kGlDynamicCopy);
#else
    return nullptr;
#endif
  } else {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (cudaFail(context, cudaDriver().memoryAllocate(&buffer->device, size),
                 "cuMemAlloc") != VERNON_STATUS_OK)
      return nullptr;
#else
    return nullptr;
#endif
  }
  ++context->liveBuffers;
  return buffer.release();
}

VernonStatus vernonRuntimeBufferFree(VernonDeviceBuffer *buffer) {
  if (!buffer)
    return VERNON_STATUS_OK;
  VernonRuntimeContext *context = buffer->context;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA && buffer->device)
    if (cudaFail(context, cudaDriver().memoryFree(buffer->device),
                 "cuMemFree") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (context->backend == VERNON_RUNTIME_VULKAN) {
    VulkanDriver &driver = vulkanDriver();
    if (buffer->vulkanBuffer)
      driver.destroyBuffer(context->vulkanDevice, buffer->vulkanBuffer,
                           nullptr);
    if (buffer->vulkanMemory)
      driver.freeMemory(context->vulkanDevice, buffer->vulkanMemory, nullptr);
  }
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  if (context->backend == VERNON_RUNTIME_OPENGL ||
      context->backend == VERNON_RUNTIME_OPENGL_ES) {
    glfwMakeContextCurrent(context->openGLWindow);
    if (buffer->openGLBuffer)
      openGLDriver().deleteBuffers(1, &buffer->openGLBuffer);
  }
#endif
  --context->liveBuffers;
  delete buffer;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyFromHost(VernonDeviceBuffer *buffer,
                                       size_t offset, const void *source,
                                       size_t size) {
  if (!buffer || !source || offset > buffer->size ||
      size > buffer->size - offset)
    return fail(buffer ? buffer->context : nullptr,
                "invalid host upload range");
  if (buffer->context->backend == VERNON_RUNTIME_CPU)
    std::memcpy(buffer->host.data() + offset, source, size);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  else if (buffer->context->backend == VERNON_RUNTIME_VULKAN) {
    void *mapped = nullptr;
    VulkanDriver &driver = vulkanDriver();
    if (vulkanFail(buffer->context,
                   driver.mapMemory(buffer->context->vulkanDevice,
                                    buffer->vulkanMemory, offset, size, 0,
                                    &mapped),
                   "vkMapMemory") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(mapped, source, size);
    driver.unmapMemory(buffer->context->vulkanDevice, buffer->vulkanMemory);
  }
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  else if (buffer->context->backend == VERNON_RUNTIME_OPENGL ||
           buffer->context->backend == VERNON_RUNTIME_OPENGL_ES) {
    glfwMakeContextCurrent(buffer->context->openGLWindow);
    OpenGLDriver &driver = openGLDriver();
    driver.bindBuffer(kGlShaderStorageBuffer, buffer->openGLBuffer);
    driver.bufferSubData(kGlShaderStorageBuffer, static_cast<GlSizePtr>(offset),
                         static_cast<GlSizePtr>(size), source);
  }
#endif
#if defined(VERNON_HAS_CUDA_RUNTIME)
  else
    return cudaFail(
        buffer->context,
        cudaDriver().copyHostToDevice(buffer->device + offset, source, size),
        "cuMemcpyHtoD");
#endif
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyToHost(const VernonDeviceBuffer *buffer,
                                     size_t offset, void *destination,
                                     size_t size) {
  if (!buffer || !destination || offset > buffer->size ||
      size > buffer->size - offset)
    return fail(buffer ? buffer->context : nullptr,
                "invalid host download range");
  if (buffer->context->backend == VERNON_RUNTIME_CPU)
    std::memcpy(destination, buffer->host.data() + offset, size);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  else if (buffer->context->backend == VERNON_RUNTIME_VULKAN) {
    void *mapped = nullptr;
    VulkanDriver &driver = vulkanDriver();
    if (vulkanFail(buffer->context,
                   driver.mapMemory(buffer->context->vulkanDevice,
                                    buffer->vulkanMemory, offset, size, 0,
                                    &mapped),
                   "vkMapMemory") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
    std::memcpy(destination, mapped, size);
    driver.unmapMemory(buffer->context->vulkanDevice, buffer->vulkanMemory);
  }
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  else if (buffer->context->backend == VERNON_RUNTIME_OPENGL ||
           buffer->context->backend == VERNON_RUNTIME_OPENGL_ES) {
    glfwMakeContextCurrent(buffer->context->openGLWindow);
    OpenGLDriver &driver = openGLDriver();
    driver.bindBuffer(kGlShaderStorageBuffer, buffer->openGLBuffer);
    void *mapped = driver.mapBufferRange(
        kGlShaderStorageBuffer, static_cast<GlSizePtr>(offset),
        static_cast<GlSizePtr>(size), kGlMapReadBit);
    if (!mapped)
      return fail(buffer->context, "glMapBufferRange failed",
                  VERNON_STATUS_INTERNAL_ERROR);
    std::memcpy(destination, mapped, size);
    driver.unmapBuffer(kGlShaderStorageBuffer);
  }
#endif
#if defined(VERNON_HAS_CUDA_RUNTIME)
  else
    return cudaFail(buffer->context,
                    cudaDriver().copyDeviceToHost(
                        destination, buffer->device + offset, size),
                    "cuMemcpyDtoH");
#endif
  return VERNON_STATUS_OK;
}

VernonDeviceTexture *vernonRuntimeTextureCreate2D(VernonRuntimeContext *context,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  VernonTextureFormat format) {
  if (!context || !width || !height || format != VERNON_TEXTURE_RGBA8_UNORM) {
    fail(context, "RGBA8 texture dimensions are required");
    return nullptr;
  }
  if (context->backend != VERNON_RUNTIME_OPENGL &&
      context->backend != VERNON_RUNTIME_OPENGL_ES) {
    fail(context, "textures are currently supported by OpenGL only",
         VERNON_STATUS_UNSUPPORTED_TARGET);
    return nullptr;
  }
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  auto texture = std::make_unique<VernonDeviceTexture>();
  texture->context = context;
  texture->width = width;
  texture->height = height;
  texture->format = format;
  glfwMakeContextCurrent(context->openGLWindow);
  OpenGLDriver &driver = openGLDriver();
  driver.genTextures(1, &texture->openGLTexture);
  driver.bindTexture(kGlTexture2D, texture->openGLTexture);
  driver.texParameteri(kGlTexture2D, kGlTextureMinFilter, kGlNearest);
  driver.texParameteri(kGlTexture2D, kGlTextureMagFilter, kGlNearest);
  driver.texImage2D(kGlTexture2D, 0, kGlRgba8, static_cast<GlSize>(width),
                    static_cast<GlSize>(height), 0, kGlRgba, kGlUnsignedByte,
                    nullptr);
  ++context->liveTextures;
  return texture.release();
#else
  return nullptr;
#endif
}

VernonStatus vernonRuntimeTextureFree(VernonDeviceTexture *texture) {
  if (!texture)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  glfwMakeContextCurrent(texture->context->openGLWindow);
  if (texture->openGLTexture)
    openGLDriver().deleteTextures(1, &texture->openGLTexture);
#endif
  --texture->context->liveTextures;
  delete texture;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeTextureCopyFromHost(VernonDeviceTexture *texture,
                                              const void *source, size_t size) {
  const size_t expected =
      texture ? static_cast<size_t>(texture->width) * texture->height * 4 : 0;
  if (!texture || !source || size != expected)
    return fail(texture ? texture->context : nullptr,
                "texture upload must contain width * height * 4 bytes");
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  glfwMakeContextCurrent(texture->context->openGLWindow);
  OpenGLDriver &driver = openGLDriver();
  driver.bindTexture(kGlTexture2D, texture->openGLTexture);
  driver.texSubImage2D(
      kGlTexture2D, 0, 0, 0, static_cast<GlSize>(texture->width),
      static_cast<GlSize>(texture->height), kGlRgba, kGlUnsignedByte, source);
  return VERNON_STATUS_OK;
#else
  return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus vernonRuntimeTextureCopyToHost(const VernonDeviceTexture *texture,
                                            void *destination, size_t size) {
  const size_t expected =
      texture ? static_cast<size_t>(texture->width) * texture->height * 4 : 0;
  if (!texture || !destination || size != expected)
    return fail(texture ? texture->context : nullptr,
                "texture readback must contain width * height * 4 bytes");
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  glfwMakeContextCurrent(texture->context->openGLWindow);
  OpenGLDriver &driver = openGLDriver();
  driver.bindTexture(kGlTexture2D, texture->openGLTexture);
  driver.getTexImage(kGlTexture2D, 0, kGlRgba, kGlUnsignedByte, destination);
  return VERNON_STATUS_OK;
#else
  return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonLoadedKernel *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context,
                                              VernonCpuEntryPoint entryPoint,
                                              const char *reflection,
                                              size_t reflectionSize,
                                              const char *entry,
                                              size_t entrySize) {
  if (!context || context->backend != VERNON_RUNTIME_CPU || !entryPoint ||
      !reflection || !entry)
    return nullptr;
  auto kernel = std::make_unique<VernonLoadedKernel>();
  kernel->context = context;
  kernel->cpuEntry = entryPoint;
  if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                       llvm::StringRef(entry, entrySize), kernel->reflection,
                       context->error))
    return nullptr;
  ++context->liveKernels;
  return kernel.release();
}

VernonLoadedKernel *
vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                          size_t artifactSize, const char *reflection,
                          size_t reflectionSize, const char *entry,
                          size_t entrySize) {
  if (!context || !artifact || !reflection || !entry)
    return nullptr;
  if (context->backend == VERNON_RUNTIME_CPU) {
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = context;
    if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                         llvm::StringRef(entry, entrySize), kernel->reflection,
                         context->error))
      return nullptr;
    auto targetMachine = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!targetMachine) {
      fail(context, llvm::toString(targetMachine.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    auto dataLayout = targetMachine->getDefaultDataLayoutForTarget();
    if (!dataLayout) {
      fail(context, llvm::toString(dataLayout.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    auto jit = llvm::orc::LLJITBuilder()
                   .setJITTargetMachineBuilder(std::move(*targetMachine))
                   .setDataLayout(*dataLayout)
                   .create();
    if (!jit) {
      fail(context, llvm::toString(jit.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
        llvm::StringRef(static_cast<const char *>(artifact), artifactSize),
        diagnostic, *llvmContext);
    if (!module) {
      std::string message;
      llvm::raw_string_ostream stream(message);
      diagnostic.print("VernonDSLRuntime", stream);
      fail(context, stream.str(), VERNON_STATUS_PARSE_ERROR);
      return nullptr;
    }
    module->setDataLayout(*dataLayout);
    module->setTargetTriple((*jit)->getTargetTriple());
    if (llvm::Error error = (*jit)->addIRModule(llvm::orc::ThreadSafeModule(
            std::move(module), std::move(llvmContext)))) {
      fail(context, llvm::toString(std::move(error)),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    std::string symbolName = "__vernon_cpu_" + std::string(entry, entrySize);
    auto symbol = (*jit)->lookup(symbolName);
    if (!symbol) {
      fail(context, llvm::toString(symbol.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    kernel->cpuEntry = symbol->toPtr<VernonCpuEntryPoint>();
    kernel->cpuJit = std::move(*jit);
    ++context->liveKernels;
    return kernel.release();
  }
  if (context->backend == VERNON_RUNTIME_OPENGL ||
      context->backend == VERNON_RUNTIME_OPENGL_ES) {
#if defined(VERNON_HAS_OPENGL_RUNTIME)
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = context;
    if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                         llvm::StringRef(entry, entrySize), kernel->reflection,
                         context->error))
      return nullptr;
    glfwMakeContextCurrent(context->openGLWindow);
    OpenGLDriver &driver = openGLDriver();
    if (context->requestedApiVersionMajor < 4 ||
        (context->requestedApiVersionMajor == 4 &&
         context->requestedApiVersionMinor < 3) ||
        !driver.dispatchCompute || !driver.memoryBarrier) {
      fail(context, "OpenGL 4.3 is required for compute shaders",
           VERNON_STATUS_UNSUPPORTED_TARGET);
      return nullptr;
    }
    GlUint shader = driver.createShader(kGlComputeShader);
    const GlChar *source = static_cast<const GlChar *>(artifact);
    GlInt sourceLength = static_cast<GlInt>(artifactSize);
    driver.shaderSource(shader, 1, &source, &sourceLength);
    driver.compileShader(shader);
    GlInt compiled = 0;
    driver.getShaderiv(shader, kGlCompileStatus, &compiled);
    if (!compiled) {
      context->error = "OpenGL compute shader compilation failed: " +
                       getOpenGLShaderLog(driver, shader);
      driver.deleteShader(shader);
      return nullptr;
    }
    kernel->openGLProgram = driver.createProgram();
    driver.attachShader(kernel->openGLProgram, shader);
    driver.linkProgram(kernel->openGLProgram);
    driver.deleteShader(shader);
    GlInt linked = 0;
    driver.getProgramiv(kernel->openGLProgram, kGlLinkStatus, &linked);
    if (!linked) {
      context->error = "OpenGL compute program link failed: " +
                       getOpenGLProgramLog(driver, kernel->openGLProgram);
      driver.deleteProgram(kernel->openGLProgram);
      kernel->openGLProgram = 0;
      return nullptr;
    }
    uint32_t kernelArgumentIndex = 0;
    for (ReflectedArgument &argument : kernel->reflection.arguments) {
      if (argument.kind == "builtin")
        continue;
      if (argument.binding == UINT32_MAX)
        argument.binding = kernelArgumentIndex;
      ++kernelArgumentIndex;
    }
    ++context->liveKernels;
    return kernel.release();
#else
    return nullptr;
#endif
  }
  if (context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (artifactSize % sizeof(uint32_t) != 0) {
      fail(context, "Vulkan artifact is not aligned SPIR-V");
      return nullptr;
    }
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = context;
    if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                         llvm::StringRef(entry, entrySize), kernel->reflection,
                         context->error))
      return nullptr;
    VulkanDriver &driver = vulkanDriver();
    VkShaderModuleCreateInfo shaderInfo{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = artifactSize;
    shaderInfo.pCode = static_cast<const uint32_t *>(artifact);
    if (vulkanFail(context,
                   driver.createShaderModule(context->vulkanDevice, &shaderInfo,
                                             nullptr, &kernel->vulkanShader),
                   "vkCreateShaderModule") != VERNON_STATUS_OK)
      return nullptr;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint32_t kernelArgumentIndex = 0;
    for (ReflectedArgument &argument : kernel->reflection.arguments) {
      if (argument.kind == "builtin")
        continue;
      if (argument.descriptorSet != 0) {
        fail(context,
             "Vulkan runtime currently supports descriptor set zero only");
        driver.destroyShaderModule(context->vulkanDevice, kernel->vulkanShader,
                                   nullptr);
        return nullptr;
      }
      if (argument.binding == UINT32_MAX)
        argument.binding = kernelArgumentIndex;
      bindings.push_back({argument.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
      ++kernelArgumentIndex;
    }
    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    descriptorLayoutInfo.pBindings = bindings.data();
    if (vulkanFail(context,
                   driver.createDescriptorSetLayout(
                       context->vulkanDevice, &descriptorLayoutInfo, nullptr,
                       &kernel->vulkanDescriptorSetLayout),
                   "vkCreateDescriptorSetLayout") != VERNON_STATUS_OK) {
      driver.destroyShaderModule(context->vulkanDevice, kernel->vulkanShader,
                                 nullptr);
      return nullptr;
    }
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &kernel->vulkanDescriptorSetLayout;
    if (vulkanFail(context,
                   driver.createPipelineLayout(context->vulkanDevice,
                                               &pipelineLayoutInfo, nullptr,
                                               &kernel->vulkanPipelineLayout),
                   "vkCreatePipelineLayout") != VERNON_STATUS_OK) {
      driver.destroyDescriptorSetLayout(
          context->vulkanDevice, kernel->vulkanDescriptorSetLayout, nullptr);
      driver.destroyShaderModule(context->vulkanDevice, kernel->vulkanShader,
                                 nullptr);
      return nullptr;
    }
    std::string symbol(entry, entrySize);
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = kernel->vulkanShader;
    stage.pName = symbol.c_str();
    VkComputePipelineCreateInfo pipelineInfo{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stage;
    pipelineInfo.layout = kernel->vulkanPipelineLayout;
    if (vulkanFail(context,
                   driver.createComputePipelines(
                       context->vulkanDevice, VK_NULL_HANDLE, 1, &pipelineInfo,
                       nullptr, &kernel->vulkanPipeline),
                   "vkCreateComputePipelines") != VERNON_STATUS_OK) {
      driver.destroyPipelineLayout(context->vulkanDevice,
                                   kernel->vulkanPipelineLayout, nullptr);
      driver.destroyDescriptorSetLayout(
          context->vulkanDevice, kernel->vulkanDescriptorSetLayout, nullptr);
      driver.destroyShaderModule(context->vulkanDevice, kernel->vulkanShader,
                                 nullptr);
      return nullptr;
    }
    ++context->liveKernels;
    return kernel.release();
#else
    return nullptr;
#endif
  }
  if (context->backend != VERNON_RUNTIME_CUDA)
    return nullptr;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  auto kernel = std::make_unique<VernonLoadedKernel>();
  kernel->context = context;
  if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                       llvm::StringRef(entry, entrySize), kernel->reflection,
                       context->error))
    return nullptr;
  std::string image(static_cast<const char *>(artifact), artifactSize);
  image.push_back('\0');
  if (cudaFail(context,
               cudaDriver().moduleLoadData(&kernel->cudaModule, image.data(), 0,
                                           nullptr, nullptr),
               "cuModuleLoadDataEx") != VERNON_STATUS_OK)
    return nullptr;
  std::string symbol(entry, entrySize);
  if (cudaFail(context,
               cudaDriver().moduleGetFunction(
                   &kernel->cudaFunction, kernel->cudaModule, symbol.c_str()),
               "cuModuleGetFunction") != VERNON_STATUS_OK) {
    cudaDriver().moduleUnload(kernel->cudaModule);
    return nullptr;
  }
  ++context->liveKernels;
  return kernel.release();
#else
  (void)artifactSize;
  (void)reflectionSize;
  (void)entrySize;
  return nullptr;
#endif
}

VernonLoadedKernel *
vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context,
                               const char *directory) {
  if (!context || !directory)
    return nullptr;
  std::filesystem::path root(directory);
  std::string manifestText = readFile(root / "compute.json");
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(manifestText);
  if (!parsed || !parsed->getAsObject()) {
    fail(context, "cannot parse compute.json");
    return nullptr;
  }
  llvm::json::Object &manifest = *parsed->getAsObject();
  if (manifest.getInteger("schema_version").value_or(0) != 1 ||
      manifest.getInteger("gpu_launch_abi_version").value_or(0) != 1) {
    fail(context, "unsupported compute bundle schema or launch ABI");
    return nullptr;
  }
  llvm::StringRef expectedTarget =
      context->backend == VERNON_RUNTIME_CPU      ? "cpu"
      : context->backend == VERNON_RUNTIME_CUDA   ? "cuda"
      : context->backend == VERNON_RUNTIME_VULKAN ? "vulkan"
      : context->backend == VERNON_RUNTIME_OPENGL ? "opengl"
                                                  : "opengles";
  if (manifest.getString("target").value_or("") != expectedTarget) {
    fail(context, "compute bundle target does not match runtime backend");
    return nullptr;
  }
  std::string entry = manifest.getString("entry").value_or("").str();
  std::string artifactName = manifest.getString("artifact").value_or("").str();
  llvm::json::Value *reflectionValue = manifest.get("reflection");
  if (entry.empty() || artifactName.empty() || !reflectionValue) {
    fail(context, "compute bundle is missing required fields");
    return nullptr;
  }
  std::string reflection;
  llvm::raw_string_ostream reflectionStream(reflection);
  reflectionStream << *reflectionValue;
  reflectionStream.flush();
  std::string artifact = readFile(root / artifactName);
  if (artifact.empty()) {
    fail(context, "compute bundle artifact is empty or absent");
    return nullptr;
  }
  if (manifest.getInteger("artifact_size").value_or(-1) !=
      static_cast<int64_t>(artifact.size())) {
    fail(context, "compute bundle artifact size mismatch");
    return nullptr;
  }
  std::string expectedHash =
      manifest.getString("artifact_sha256").value_or("").str();
  llvm::ArrayRef<uint8_t> bytes(
      reinterpret_cast<const uint8_t *>(artifact.data()), artifact.size());
  std::string actualHash = llvm::toHex(llvm::SHA256::hash(bytes), true);
  if (expectedHash.empty() ||
      !llvm::StringRef(expectedHash).equals_insensitive(actualHash)) {
    fail(context, "compute bundle artifact SHA-256 mismatch");
    return nullptr;
  }
  return vernonRuntimeLoadArtifact(context, artifact.data(), artifact.size(),
                                   reflection.data(), reflection.size(),
                                   entry.data(), entry.size());
}

VernonStatus vernonRuntimeKernelUnload(VernonLoadedKernel *kernel) {
  if (!kernel)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_CUDA && kernel->cudaModule)
    if (cudaFail(kernel->context, cudaDriver().moduleUnload(kernel->cudaModule),
                 "cuModuleUnload") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_VULKAN) {
    VulkanDriver &driver = vulkanDriver();
    driver.deviceWaitIdle(kernel->context->vulkanDevice);
    if (kernel->vulkanPipeline)
      driver.destroyPipeline(kernel->context->vulkanDevice,
                             kernel->vulkanPipeline, nullptr);
    if (kernel->vulkanPipelineLayout)
      driver.destroyPipelineLayout(kernel->context->vulkanDevice,
                                   kernel->vulkanPipelineLayout, nullptr);
    if (kernel->vulkanDescriptorSetLayout)
      driver.destroyDescriptorSetLayout(kernel->context->vulkanDevice,
                                        kernel->vulkanDescriptorSetLayout,
                                        nullptr);
    if (kernel->vulkanShader)
      driver.destroyShaderModule(kernel->context->vulkanDevice,
                                 kernel->vulkanShader, nullptr);
  }
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_OPENGL ||
      kernel->context->backend == VERNON_RUNTIME_OPENGL_ES) {
    glfwMakeContextCurrent(kernel->context->openGLWindow);
    if (kernel->openGLProgram)
      openGLDriver().deleteProgram(kernel->openGLProgram);
  }
#endif
  --kernel->context->liveKernels;
  delete kernel;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeLaunch(VernonLoadedKernel *kernel,
                                 VernonLaunchSize globalSize,
                                 const VernonLaunchArgument *arguments,
                                 size_t argumentCount) {
  if (!kernel || !globalSize.x || !globalSize.y || !globalSize.z)
    return fail(kernel ? kernel->context : nullptr,
                "grid dimensions must be positive");
  const size_t expected = static_cast<size_t>(std::count_if(
      kernel->reflection.arguments.begin(), kernel->reflection.arguments.end(),
      [](const ReflectedArgument &argument) {
        return argument.kind != "builtin";
      }));
  if (argumentCount != expected || (expected && !arguments))
    return fail(kernel->context,
                "launch argument count does not match reflection");
  size_t validatedIndex = 0;
  for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
    if (reflected.kind == "builtin")
      continue;
    const VernonLaunchArgument &argument = arguments[validatedIndex++];
    if (reflected.kind == "tensor") {
      if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer ||
          argument.buffer->context != kernel->context ||
          (reflected.tensorBytes &&
           argument.buffer->size < reflected.tensorBytes) ||
          argument.buffer->alignment < reflected.alignment)
        return fail(kernel->context,
                    "Tensor launch argument does not match reflection");
    } else if (argument.kind != VERNON_LAUNCH_SCALAR || !argument.scalar_data ||
               (reflected.cpuSize &&
                argument.scalar_size != reflected.cpuSize)) {
      return fail(kernel->context,
                  "scalar launch argument does not match reflection");
    }
  }

  if (kernel->context->backend == VERNON_RUNTIME_CPU) {
    std::vector<unsigned char> packed(kernel->reflection.cpuArgumentsSize);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
      if (reflected.kind == "builtin")
        continue;
      const VernonLaunchArgument &argument = arguments[supplied++];
      if (reflected.kind == "tensor") {
        if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer ||
            argument.buffer->context != kernel->context ||
            reflected.cpuSize != sizeof(uintptr_t) ||
            (reflected.tensorBytes &&
             argument.buffer->size < reflected.tensorBytes) ||
            argument.buffer->alignment < reflected.alignment)
          return fail(kernel->context, "invalid Tensor launch argument");
        uintptr_t pointer =
            reinterpret_cast<uintptr_t>(argument.buffer->host.data());
        std::memcpy(packed.data() + reflected.cpuOffset, &pointer,
                    sizeof(pointer));
      } else {
        if (argument.kind != VERNON_LAUNCH_SCALAR ||
            argument.scalar_size != reflected.cpuSize || !argument.scalar_data)
          return fail(kernel->context, "invalid scalar launch argument");
        std::memcpy(packed.data() + reflected.cpuOffset, argument.scalar_data,
                    argument.scalar_size);
      }
    }
    for (uint32_t z = 0; z < globalSize.z; ++z)
      for (uint32_t y = 0; y < globalSize.y; ++y)
        for (uint32_t x = 0; x < globalSize.x; ++x) {
          uint32_t id[3]{x, y, z};
          for (const ReflectedArgument &reflected :
               kernel->reflection.arguments)
            if (reflected.kind == "builtin" &&
                reflected.builtin == "global_invocation_id")
              std::memcpy(packed.data() + reflected.cpuOffset, id,
                          std::min(reflected.cpuSize, sizeof(id)));
          VernonCpuInvocation invocation{packed.data(), packed.size(), nullptr,
                                         0, nullptr};
          VernonStatus status = kernel->cpuEntry(&invocation);
          if (status != VERNON_STATUS_OK)
            return fail(kernel->context, "CPU kernel invocation failed",
                        status);
        }
    return VERNON_STATUS_OK;
  }

#if defined(VERNON_HAS_OPENGL_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_OPENGL ||
      kernel->context->backend == VERNON_RUNTIME_OPENGL_ES) {
    VernonRuntimeContext *context = kernel->context;
    glfwMakeContextCurrent(context->openGLWindow);
    OpenGLDriver &driver = openGLDriver();
    driver.useProgram(kernel->openGLProgram);
    std::vector<GlUint> scalarBuffers;
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
      if (reflected.kind == "builtin")
        continue;
      const VernonLaunchArgument &argument = arguments[supplied++];
      GlUint buffer = 0;
      if (argument.kind == VERNON_LAUNCH_TENSOR) {
        buffer = argument.buffer->openGLBuffer;
      } else {
        driver.genBuffers(1, &buffer);
        scalarBuffers.push_back(buffer);
        driver.bindBuffer(kGlShaderStorageBuffer, buffer);
        driver.bufferData(kGlShaderStorageBuffer,
                          static_cast<GlSizePtr>(argument.scalar_size),
                          argument.scalar_data, kGlDynamicCopy);
      }
      driver.bindBufferBase(kGlShaderStorageBuffer, reflected.binding, buffer);
    }
    uint32_t *workgroup = kernel->reflection.workgroup;
    driver.dispatchCompute((globalSize.x + workgroup[0] - 1) / workgroup[0],
                           (globalSize.y + workgroup[1] - 1) / workgroup[1],
                           (globalSize.z + workgroup[2] - 1) / workgroup[2]);
    driver.memoryBarrier(kGlShaderStorageBarrierBit);
    driver.finish();
    if (!scalarBuffers.empty())
      driver.deleteBuffers(static_cast<GlSize>(scalarBuffers.size()),
                           scalarBuffers.data());
    return VERNON_STATUS_OK;
  }
#endif

#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_VULKAN) {
    VernonRuntimeContext *context = kernel->context;
    VulkanDriver &driver = vulkanDriver();
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> scalarBuffers;
    auto cleanup = [&] {
      if (commandBuffer)
        driver.freeCommandBuffers(context->vulkanDevice,
                                  context->vulkanCommandPool, 1,
                                  &commandBuffer);
      if (descriptorPool)
        driver.destroyDescriptorPool(context->vulkanDevice, descriptorPool,
                                     nullptr);
      for (auto [buffer, memory] : scalarBuffers) {
        driver.destroyBuffer(context->vulkanDevice, buffer, nullptr);
        driver.freeMemory(context->vulkanDevice, memory, nullptr);
      }
    };

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  static_cast<uint32_t>(expected)};
    VkDescriptorPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = expected ? 1u : 0u;
    poolInfo.pPoolSizes = expected ? &poolSize : nullptr;
    if (vulkanFail(context,
                   driver.createDescriptorPool(context->vulkanDevice, &poolInfo,
                                               nullptr, &descriptorPool),
                   "vkCreateDescriptorPool") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
    VkDescriptorSetAllocateInfo setInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &kernel->vulkanDescriptorSetLayout;
    VkDescriptorSet descriptorSet{};
    if (vulkanFail(context,
                   driver.allocateDescriptorSets(context->vulkanDevice,
                                                 &setInfo, &descriptorSet),
                   "vkAllocateDescriptorSets") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> writes;
    bufferInfos.reserve(expected);
    writes.reserve(expected);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
      if (reflected.kind == "builtin")
        continue;
      const VernonLaunchArgument &argument = arguments[supplied++];
      VkBuffer buffer = VK_NULL_HANDLE;
      VkDeviceSize size = 0;
      if (argument.kind == VERNON_LAUNCH_TENSOR) {
        buffer = argument.buffer->vulkanBuffer;
        size = argument.buffer->size;
      } else {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        if (!createVulkanBuffer(context, argument.scalar_size, buffer,
                                memory)) {
          cleanup();
          return VERNON_STATUS_INTERNAL_ERROR;
        }
        scalarBuffers.emplace_back(buffer, memory);
        void *mapped = nullptr;
        if (vulkanFail(context,
                       driver.mapMemory(context->vulkanDevice, memory, 0,
                                        argument.scalar_size, 0, &mapped),
                       "vkMapMemory") != VERNON_STATUS_OK) {
          cleanup();
          return VERNON_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(mapped, argument.scalar_data, argument.scalar_size);
        driver.unmapMemory(context->vulkanDevice, memory);
        size = argument.scalar_size;
      }
      bufferInfos.push_back({buffer, 0, size});
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = descriptorSet;
      write.dstBinding = reflected.binding;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write.pBufferInfo = &bufferInfos.back();
      writes.push_back(write);
    }
    driver.updateDescriptorSets(context->vulkanDevice,
                                static_cast<uint32_t>(writes.size()),
                                writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo commandInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandInfo.commandPool = context->vulkanCommandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    if (vulkanFail(context,
                   driver.allocateCommandBuffers(context->vulkanDevice,
                                                 &commandInfo, &commandBuffer),
                   "vkAllocateCommandBuffers") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    VkCommandBufferBeginInfo beginInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vulkanFail(context,
                   driver.beginCommandBuffer(commandBuffer, &beginInfo),
                   "vkBeginCommandBuffer") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    driver.cmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           kernel->vulkanPipeline);
    driver.cmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 kernel->vulkanPipelineLayout, 0, 1,
                                 &descriptorSet, 0, nullptr);
    uint32_t *workgroup = kernel->reflection.workgroup;
    driver.cmdDispatch(commandBuffer,
                       (globalSize.x + workgroup[0] - 1) / workgroup[0],
                       (globalSize.y + workgroup[1] - 1) / workgroup[1],
                       (globalSize.z + workgroup[2] - 1) / workgroup[2]);
    if (vulkanFail(context, driver.endCommandBuffer(commandBuffer),
                   "vkEndCommandBuffer") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;
    if (vulkanFail(context,
                   driver.queueSubmit(context->vulkanQueue, 1, &submit,
                                      VK_NULL_HANDLE),
                   "vkQueueSubmit") != VERNON_STATUS_OK ||
        vulkanFail(context, driver.queueWaitIdle(context->vulkanQueue),
                   "vkQueueWaitIdle") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    cleanup();
    return VERNON_STATUS_OK;
  }
#endif

#if defined(VERNON_HAS_CUDA_RUNTIME)
  struct CudaMemRefDescriptor {
    CudaDevicePointer allocated;
    CudaDevicePointer aligned;
    uint64_t offset;
    uint64_t size;
    uint64_t stride;
  };
  std::vector<CudaMemRefDescriptor> descriptors;
  std::vector<void *> parameters;
  descriptors.reserve(argumentCount);
  parameters.reserve(argumentCount * 5);
  size_t suppliedIndex = 0;
  for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
    if (reflected.kind == "builtin")
      continue;
    const VernonLaunchArgument &argument = arguments[suppliedIndex++];
    if (argument.kind == VERNON_LAUNCH_TENSOR) {
      if (!argument.buffer || argument.buffer->context != kernel->context)
        return fail(kernel->context, "invalid CUDA Tensor argument");
      descriptors.push_back(
          {argument.buffer->device, argument.buffer->device, 0,
           reflected.tensorElements
               ? reflected.tensorElements
               : argument.buffer->size /
                     std::max(reflected.tensorElementSize, size_t{1}),
           1});
      CudaMemRefDescriptor &descriptor = descriptors.back();
      parameters.push_back(&descriptor.allocated);
      parameters.push_back(&descriptor.aligned);
      parameters.push_back(&descriptor.offset);
      parameters.push_back(&descriptor.size);
      parameters.push_back(&descriptor.stride);
    } else {
      if (!argument.scalar_data || !argument.scalar_size)
        return fail(kernel->context, "invalid CUDA scalar argument");
      parameters.push_back(const_cast<void *>(argument.scalar_data));
    }
  }
  uint32_t *workgroup = kernel->reflection.workgroup;
  return cudaFail(kernel->context,
                  cudaDriver().launchKernel(
                      kernel->cudaFunction,
                      (globalSize.x + workgroup[0] - 1) / workgroup[0],
                      (globalSize.y + workgroup[1] - 1) / workgroup[1],
                      (globalSize.z + workgroup[2] - 1) / workgroup[2],
                      workgroup[0], workgroup[1], workgroup[2], 0, nullptr,
                      parameters.data(), nullptr),
                  "cuLaunchKernel");
#else
  return fail(kernel->context, "CUDA runtime support is not built",
              VERNON_STATUS_UNSUPPORTED_TARGET);
#endif
}

VernonLoadedProgram *
vernonRuntimeProgramLoadGraphics(VernonRuntimeContext *context,
                                 VernonGraphicsStageArtifact vertex,
                                 VernonGraphicsStageArtifact fragment) {
  if (!context || !vertex.data || !vertex.size || !fragment.data ||
      !fragment.size)
    return nullptr;
  if (context->backend != VERNON_RUNTIME_OPENGL &&
      context->backend != VERNON_RUNTIME_OPENGL_ES) {
    fail(context, "graphics programs are currently supported by OpenGL only",
         VERNON_STATUS_UNSUPPORTED_TARGET);
    return nullptr;
  }
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  glfwMakeContextCurrent(context->openGLWindow);
  OpenGLDriver &driver = openGLDriver();
  auto compileStage = [&](GlEnum stage, VernonGraphicsStageArtifact artifact,
                          const char *name) -> GlUint {
    GlUint shader = driver.createShader(stage);
    const GlChar *source = static_cast<const GlChar *>(artifact.data);
    GlInt length = static_cast<GlInt>(artifact.size);
    driver.shaderSource(shader, 1, &source, &length);
    driver.compileShader(shader);
    GlInt compiled = 0;
    driver.getShaderiv(shader, kGlCompileStatus, &compiled);
    if (!compiled) {
      context->error =
          std::string("OpenGL ") + name +
          " shader compilation failed: " + getOpenGLShaderLog(driver, shader);
      driver.deleteShader(shader);
      return 0;
    }
    return shader;
  };
  GlUint vertexShader = compileStage(kGlVertexShader, vertex, "vertex");
  if (!vertexShader)
    return nullptr;
  GlUint fragmentShader = compileStage(kGlFragmentShader, fragment, "fragment");
  if (!fragmentShader) {
    driver.deleteShader(vertexShader);
    return nullptr;
  }
  auto program = std::make_unique<VernonLoadedProgram>();
  program->context = context;
  program->openGLProgram = driver.createProgram();
  driver.attachShader(program->openGLProgram, vertexShader);
  driver.attachShader(program->openGLProgram, fragmentShader);
  driver.linkProgram(program->openGLProgram);
  driver.deleteShader(vertexShader);
  driver.deleteShader(fragmentShader);
  GlInt linked = 0;
  driver.getProgramiv(program->openGLProgram, kGlLinkStatus, &linked);
  if (!linked) {
    context->error = "OpenGL graphics program link failed: " +
                     getOpenGLProgramLog(driver, program->openGLProgram);
    driver.deleteProgram(program->openGLProgram);
    return nullptr;
  }
  driver.genVertexArrays(1, &program->openGLVertexArray);
  driver.genFramebuffers(1, &program->openGLFramebuffer);
  ++context->livePrograms;
  return program.release();
#else
  return nullptr;
#endif
}

VernonStatus vernonRuntimeProgramUnload(VernonLoadedProgram *program) {
  if (!program)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  glfwMakeContextCurrent(program->context->openGLWindow);
  OpenGLDriver &driver = openGLDriver();
  if (program->openGLFramebuffer)
    driver.deleteFramebuffers(1, &program->openGLFramebuffer);
  if (program->openGLVertexArray)
    driver.deleteVertexArrays(1, &program->openGLVertexArray);
  if (program->openGLProgram)
    driver.deleteProgram(program->openGLProgram);
#endif
  --program->context->livePrograms;
  delete program;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeDraw(VernonLoadedProgram *program,
                               const VernonDrawDescription *description) {
  constexpr size_t legacyDrawSize =
      offsetof(VernonDrawDescription, index_binding);
  if (!program || !description || description->struct_size < legacyDrawSize ||
      !description->instance_count ||
      (description->binding_count && !description->bindings) ||
      (description->uniform_count && !description->uniforms))
    return fail(program ? program->context : nullptr,
                "invalid graphics draw description");
  auto hasField = [&](size_t offset, size_t size) {
    return description->struct_size >= offset + size;
  };
  const VernonIndexBinding *indexBinding =
      hasField(offsetof(VernonDrawDescription, index_binding),
               sizeof(description->index_binding))
          ? description->index_binding
          : nullptr;
  const VernonColorAttachment *colorAttachments =
      hasField(offsetof(VernonDrawDescription, color_attachments),
               sizeof(description->color_attachments))
          ? description->color_attachments
          : nullptr;
  const size_t colorAttachmentCount =
      hasField(offsetof(VernonDrawDescription, color_attachment_count),
               sizeof(description->color_attachment_count))
          ? description->color_attachment_count
          : 0;
  const VernonPrimitiveTopology topology =
      hasField(offsetof(VernonDrawDescription, topology),
               sizeof(description->topology))
          ? description->topology
          : VERNON_TOPOLOGY_TRIANGLE_LIST;
  if ((colorAttachmentCount && !colorAttachments) ||
      (!colorAttachmentCount && !description->target))
    return fail(program->context, "graphics draw requires a color attachment");
  std::vector<VernonColorAttachment> normalizedAttachments;
  if (colorAttachmentCount) {
    normalizedAttachments.assign(colorAttachments,
                                 colorAttachments + colorAttachmentCount);
  } else {
    normalizedAttachments.push_back({0, description->target});
  }
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint32_t> attachmentLocations;
  attachmentLocations.reserve(normalizedAttachments.size());
  for (const VernonColorAttachment &attachment : normalizedAttachments) {
    if (!attachment.texture || attachment.texture->context != program->context)
      return fail(program->context,
                  "color attachment belongs to another runtime");
    if (attachment.location >= 32)
      return fail(program->context, "color attachment location is too large");
    if (std::find(attachmentLocations.begin(), attachmentLocations.end(),
                  attachment.location) != attachmentLocations.end())
      return fail(program->context, "duplicate color attachment location");
    attachmentLocations.push_back(attachment.location);
    if (!width) {
      width = attachment.texture->width;
      height = attachment.texture->height;
    } else if (width != attachment.texture->width ||
               height != attachment.texture->height) {
      return fail(program->context,
                  "all color attachments must have the same extent");
    }
  }
  uint32_t elementCount = description->vertex_count;
  if (indexBinding) {
    if (!indexBinding->buffer ||
        indexBinding->buffer->context != program->context ||
        indexBinding->type != VERNON_INDEX_U32 || !indexBinding->index_count ||
        indexBinding->offset + static_cast<size_t>(indexBinding->index_count) *
                                   sizeof(uint32_t) >
            indexBinding->buffer->size)
      return fail(program->context, "invalid graphics index binding");
    elementCount = indexBinding->index_count;
  } else if (!elementCount) {
    return fail(program->context, "graphics draw has no vertices");
  }
  if ((topology == VERNON_TOPOLOGY_TRIANGLE_LIST && elementCount % 3 != 0) ||
      (topology == VERNON_TOPOLOGY_LINE_LIST && elementCount % 2 != 0) ||
      (topology != VERNON_TOPOLOGY_TRIANGLE_LIST &&
       topology != VERNON_TOPOLOGY_LINE_LIST &&
       topology != VERNON_TOPOLOGY_POINT_LIST))
    return fail(program->context,
                "draw count is incompatible with primitive topology");
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  VernonRuntimeContext *context = program->context;
  glfwMakeContextCurrent(context->openGLWindow);
  OpenGLDriver &driver = openGLDriver();
  driver.useProgram(program->openGLProgram);
  for (size_t index = 0; index < description->uniform_count; ++index) {
    const VernonUniformBinding &uniform = description->uniforms[index];
    if (!uniform.name || !uniform.values)
      return fail(context, "invalid uniform binding");
    GlInt location =
        driver.getUniformLocation(program->openGLProgram, uniform.name);
    if (location < 0)
      return fail(context, std::string("graphics program has no uniform '") +
                               uniform.name + "'");
    switch (uniform.value_count) {
    case 1:
      driver.uniform1fv(location, 1, uniform.values);
      break;
    case 2:
      driver.uniform2fv(location, 1, uniform.values);
      break;
    case 3:
      driver.uniform3fv(location, 1, uniform.values);
      break;
    case 4:
      driver.uniform4fv(location, 1, uniform.values);
      break;
    case 9:
      driver.uniformMatrix3fv(location, 1, 1, uniform.values);
      break;
    case 16:
      driver.uniformMatrix4fv(location, 1, 1, uniform.values);
      break;
    default:
      return fail(context, "unsupported uniform component count");
    }
  }
  driver.bindVertexArray(program->openGLVertexArray);
  if (indexBinding)
    driver.bindBuffer(kGlElementArrayBuffer,
                      indexBinding->buffer->openGLBuffer);
  for (size_t index = 0; index < description->binding_count; ++index) {
    const VernonDrawBinding &binding = description->bindings[index];
    if (!binding.buffer || binding.buffer->context != context ||
        binding.component_count < 1 || binding.component_count > 4)
      return fail(context, "vertex binding does not match the runtime");
    const uint32_t stride =
        binding.stride ? binding.stride : binding.component_count * 4;
    const uint32_t elementCount =
        binding.instance_divisor
            ? (description->instance_count + binding.instance_divisor - 1) /
                  binding.instance_divisor
            : description->vertex_count;
    const size_t required = binding.offset +
                            static_cast<size_t>(stride) * (elementCount - 1) +
                            binding.component_count * 4;
    if (required > binding.buffer->size)
      return fail(context, "vertex binding buffer is too small");
    driver.bindBuffer(kGlArrayBuffer, binding.buffer->openGLBuffer);
    driver.enableVertexAttribArray(binding.location);
    driver.vertexAttribPointer(binding.location,
                               static_cast<GlInt>(binding.component_count),
                               kGlFloat, 0, static_cast<GlSize>(stride),
                               reinterpret_cast<const void *>(binding.offset));
    driver.vertexAttribDivisor(binding.location, binding.instance_divisor);
  }
  driver.bindFramebuffer(kGlFramebuffer, program->openGLFramebuffer);
  for (uint32_t location : program->openGLAttachmentLocations)
    if (std::find(attachmentLocations.begin(), attachmentLocations.end(),
                  location) == attachmentLocations.end())
      driver.framebufferTexture2D(
          kGlFramebuffer, kGlColorAttachment0 + location, kGlTexture2D, 0, 0);
  const uint32_t maximumLocation =
      *std::max_element(attachmentLocations.begin(), attachmentLocations.end());
  std::vector<GlEnum> drawBuffers(maximumLocation + 1, kGlNone);
  for (const VernonColorAttachment &attachment : normalizedAttachments) {
    const GlEnum slot = kGlColorAttachment0 + attachment.location;
    driver.framebufferTexture2D(kGlFramebuffer, slot, kGlTexture2D,
                                attachment.texture->openGLTexture, 0);
    drawBuffers[attachment.location] = slot;
  }
  driver.drawBuffers(static_cast<GlSize>(drawBuffers.size()),
                     drawBuffers.data());
  program->openGLAttachmentLocations = std::move(attachmentLocations);
  if (driver.checkFramebufferStatus(kGlFramebuffer) != kGlFramebufferComplete)
    return fail(context, "OpenGL framebuffer is incomplete",
                VERNON_STATUS_INTERNAL_ERROR);
  driver.viewport(0, 0, static_cast<GlSize>(width),
                  static_cast<GlSize>(height));
  driver.clearColor(description->clear_color[0], description->clear_color[1],
                    description->clear_color[2], description->clear_color[3]);
  driver.clear(kGlColorBufferBit);
  const GlEnum mode = topology == VERNON_TOPOLOGY_LINE_LIST    ? kGlLines
                      : topology == VERNON_TOPOLOGY_POINT_LIST ? kGlPoints
                                                               : kGlTriangles;
  if (indexBinding) {
    driver.drawElementsInstanced(
        mode, static_cast<GlSize>(indexBinding->index_count), kGlUnsignedInt,
        reinterpret_cast<const void *>(indexBinding->offset),
        static_cast<GlSize>(description->instance_count));
  } else if (description->instance_count == 1) {
    driver.drawArrays(mode, 0, static_cast<GlSize>(description->vertex_count));
  } else {
    driver.drawArraysInstanced(
        mode, 0, static_cast<GlSize>(description->vertex_count),
        static_cast<GlSize>(description->instance_count));
  }
  return VERNON_STATUS_OK;
#else
  return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus
vernonRuntimeComputeToGraphicsBarrier(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_INVALID_ARGUMENT;
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  if (context->backend == VERNON_RUNTIME_OPENGL ||
      context->backend == VERNON_RUNTIME_OPENGL_ES) {
    glfwMakeContextCurrent(context->openGLWindow);
    OpenGLDriver &driver = openGLDriver();
    if (!driver.memoryBarrier)
      return fail(context, "OpenGL context does not support memory barriers",
                  VERNON_STATUS_UNSUPPORTED_TARGET);
    driver.memoryBarrier(kGlShaderStorageBarrierBit |
                         kGlVertexAttribArrayBarrierBit);
    return VERNON_STATUS_OK;
  }
#endif
  return fail(context, "compute-to-graphics barriers are unsupported",
              VERNON_STATUS_UNSUPPORTED_TARGET);
}

VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_INVALID_ARGUMENT;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA)
    return cudaFail(context, cudaDriver().contextSynchronize(),
                    "cuCtxSynchronize");
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (context->backend == VERNON_RUNTIME_VULKAN)
    return vulkanFail(context,
                      vulkanDriver().deviceWaitIdle(context->vulkanDevice),
                      "vkDeviceWaitIdle");
#endif
#if defined(VERNON_HAS_OPENGL_RUNTIME)
  if (context->backend == VERNON_RUNTIME_OPENGL ||
      context->backend == VERNON_RUNTIME_OPENGL_ES) {
    glfwMakeContextCurrent(context->openGLWindow);
    openGLDriver().finish();
  }
#endif
  return VERNON_STATUS_OK;
}

} // extern "C"
