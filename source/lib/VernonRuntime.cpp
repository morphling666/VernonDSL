#include "VernonRuntime.h"
#include "runtime/backend_cuda_driver.h"
#include "runtime/backend_opengl_driver.h"
#include "runtime/content_hash.h"
#include "runtime/platform_library.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "runtime/backend_vulkan_driver.h"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace vernon::runtime;

struct SampledTextureBinding {
  uint32_t descriptorSet{};
  uint32_t binding{UINT32_MAX};
};

struct ParameterUse {
  std::string stage;
  std::string interfaceKind;
  std::string uniformName;
  std::string dtype;
  std::vector<uint64_t> shape;
  uint32_t index{};
  uint32_t location{UINT32_MAX};
  uint32_t divisor{};
  uint32_t descriptorSet{};
  uint32_t binding{UINT32_MAX};
  std::vector<SampledTextureBinding> sampledTextureBindings;
};

struct Parameter {
  uint32_t slot{};
  std::string name;
  std::string kind;
  std::string dtype;
  std::string access;
  std::string dimension;
  std::vector<uint64_t> shape;
  std::vector<ParameterUse> uses;
};

struct Output {
  std::string name;
  std::string kind;
  std::string dtype;
  std::string access;
  std::vector<uint64_t> shape;
  uint32_t location{UINT32_MAX};
};

struct CpuNativeArtifact {
  std::filesystem::path root;
  std::filesystem::path relativeLibrary;
  std::string format{"native_library"};
  std::string entry;
  std::string symbol;
  std::string operatingSystem;
  std::string architecture;
  std::string targetTriple;
  std::string objectFormat;
  uint32_t invocationAbiVersion{};
  uint64_t size{};
  std::string sha256;
  nlohmann::json reflection;
};

struct ResolvedArtifact {
  std::string format;
  std::vector<uint8_t> bytes;
  std::filesystem::path path;
  bool external{};
};

struct Stage {
  std::string stage;
  std::string entry;
  std::string source;
  std::string reflection;
  std::vector<uint8_t> binary;
  std::optional<CpuNativeArtifact> cpuArtifact;
  uint32_t workgroup[3]{1, 1, 1};
};

struct Variant {
  std::vector<std::string> key;
  std::vector<Parameter> parameters;
  std::vector<Output> outputs;
  std::string compute;
  std::string vertex;
  std::string fragment;
  bool barrier{};
};

struct ReflectedArgument {
  std::string kind;
  std::string builtin;
  size_t cpuOffset{};
  size_t cpuSize{};
  size_t tensorBytes{};
  size_t tensorElements{};
  size_t tensorElementSize{};
  size_t alignment{1};
  uint32_t descriptorSet{};
  uint32_t binding{UINT32_MAX};
};

struct ReflectedEntry {
  std::vector<ReflectedArgument> arguments;
  size_t cpuArgumentsSize{};
  uint32_t workgroup[3]{1, 1, 1};
};

std::string shaderLog(OpenGLDriver &gl, GlUint shader) {
  GlInt size = 0;
  gl.getShaderiv(shader, kInfoLogLength, &size);
  std::string result(static_cast<size_t>(std::max(size, 1)), '\0');
  GlSize written = 0;
  gl.getShaderInfoLog(shader, size, &written, result.data());
  result.resize(static_cast<size_t>(std::max(written, 0)));
  return result;
}

std::string programLog(OpenGLDriver &gl, GlUint program) {
  GlInt size = 0;
  gl.getProgramiv(program, kInfoLogLength, &size);
  std::string result(static_cast<size_t>(std::max(size, 1)), '\0');
  GlSize written = 0;
  gl.getProgramInfoLog(program, size, &written, result.data());
  result.resize(static_cast<size_t>(std::max(written, 0)));
  return result;
}

} // namespace

struct VernonRuntimeContext {
  VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
  VernonExternalOpenGLContext external{};
  OpenGLDriver gl{};
  std::string error;
  size_t liveBuffers{};
  size_t liveKernels{};
  size_t liveTextures{};
  size_t liveSamplers{};
  size_t liveBundles{};
  size_t livePipelines{};
#if defined(VERNON_HAS_CUDA_RUNTIME)
  vernon::runtime::CudaDevice cudaDevice{};
  vernon::runtime::CudaContext cudaContext{};
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
};

struct VernonDeviceBuffer {
  VernonRuntimeContext *context{};
  GlUint name{};
  size_t size{};
  size_t alignment{};
  std::vector<unsigned char> host;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  vernon::runtime::CudaDevicePointer cudaDevice{};
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkBuffer vulkanBuffer{};
  VkDeviceMemory vulkanMemory{};
#endif
};

struct VernonDeviceTexture {
  VernonRuntimeContext *context{};
  GlUint name{};
  uint32_t width{};
  uint32_t height{};
  uint32_t depth{1};
  uint32_t mipLevels{1};
  VernonTextureDimension dimension{VERNON_TEXTURE_2D};
  VernonTextureFormat format{VERNON_TEXTURE_RGBA8_UNORM};
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkImage vulkanImage{};
  VkDeviceMemory vulkanMemory{};
  VkImageView vulkanView{};
  VkImageLayout vulkanLayout{VK_IMAGE_LAYOUT_UNDEFINED};
  VkFormat vulkanFormat{VK_FORMAT_UNDEFINED};
  bool vulkanColorAttachment{};
#endif
};

struct VernonDeviceSampler {
  VernonRuntimeContext *context{};
  GlUint name{};
  VernonSamplerDescriptor descriptor{};
  bool imported{};
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkSampler vulkanSampler{};
#endif
};

struct VernonPipelineBundle {
  VernonRuntimeContext *context{};
  std::string id;
  std::vector<std::string> features;
  std::unordered_map<std::string, Stage> stages;
  std::vector<Variant> variants;
};

struct VernonLoadedPipeline {
  VernonRuntimeContext *context{};
  Variant variant;
  VernonLoadedKernel *computeKernel{};
  GlUint computeProgram{};
  GlUint graphicsProgram{};
  GlUint vertexArray{};
  GlUint framebuffer{};
  uint32_t workgroup[3]{1, 1, 1};
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkShaderModule vulkanVertex{};
  VkShaderModule vulkanFragment{};
  std::string vulkanVertexEntry;
  std::string vulkanFragmentEntry;
#endif
};

struct VernonLoadedKernel {
  VernonRuntimeContext *context{};
  ReflectedEntry reflection;
  VernonCpuEntryPoint cpuEntry{};
  vernon::runtime::PlatformLibrary nativeLibrary;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  vernon::runtime::CudaModule cudaModule{};
  vernon::runtime::CudaFunction cudaFunction{};
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VkShaderModule vulkanShader{};
  VkDescriptorSetLayout vulkanDescriptorSetLayout{};
  VkPipelineLayout vulkanPipelineLayout{};
  VkPipeline vulkanPipeline{};
#endif
};

namespace {

std::mutex &staticCpuEntriesMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, VernonCpuEntryPoint> &staticCpuEntries() {
  static std::unordered_map<std::string, VernonCpuEntryPoint> entries;
  return entries;
}

VernonStatus fail(VernonRuntimeContext *context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
  if (context)
    context->error = std::move(error);
  return status;
}

#if defined(VERNON_HAS_CUDA_RUNTIME)
VernonStatus cudaFail(VernonRuntimeContext *context,
                      vernon::runtime::CudaResult result,
                      const char *operation) {
  if (result == vernon::runtime::kCudaSuccess)
    return VERNON_STATUS_OK;
  const char *name = nullptr;
  const char *description = nullptr;
  vernon::runtime::CudaDriver &driver = vernon::runtime::cudaDriver();
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
       index < context->vulkanMemoryProperties.memoryTypeCount; ++index)
    if ((typeBits & (uint32_t{1} << index)) &&
        (context->vulkanMemoryProperties.memoryTypes[index].propertyFlags &
         required) == required)
      return index;
  return std::nullopt;
}

bool createVulkanBuffer(VernonRuntimeContext *context, VkDeviceSize size,
                        VkBuffer &buffer, VkDeviceMemory &memory) {
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = size;
  bufferInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vulkanFail(context,
                 driver.createBuffer(context->vulkanDevice, &bufferInfo,
                                     nullptr, &buffer),
                 "vkCreateBuffer") != VERNON_STATUS_OK)
    return false;
  VkMemoryRequirements requirements{};
  driver.getBufferMemoryRequirements(context->vulkanDevice, buffer,
                                     &requirements);
  const std::optional<uint32_t> memoryType =
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

template <typename Record>
bool submitVulkanCommands(VernonRuntimeContext *context, Record record) {
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  VkCommandBufferAllocateInfo allocation{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocation.commandPool = context->vulkanCommandPool;
  allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocation.commandBufferCount = 1;
  VkCommandBuffer command = VK_NULL_HANDLE;
  if (vulkanFail(context,
                 driver.allocateCommandBuffers(context->vulkanDevice,
                                               &allocation, &command),
                 "vkAllocateCommandBuffers") != VERNON_STATUS_OK)
    return false;
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vulkanFail(context, driver.beginCommandBuffer(command, &begin),
                 "vkBeginCommandBuffer") != VERNON_STATUS_OK) {
    driver.freeCommandBuffers(context->vulkanDevice, context->vulkanCommandPool,
                              1, &command);
    return false;
  }
  record(command);
  const bool recorded = vulkanFail(context, driver.endCommandBuffer(command),
                                   "vkEndCommandBuffer") == VERNON_STATUS_OK;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command;
  const bool submitted =
      recorded &&
      vulkanFail(
          context,
          driver.queueSubmit(context->vulkanQueue, 1, &submit, VK_NULL_HANDLE),
          "vkQueueSubmit") == VERNON_STATUS_OK &&
      vulkanFail(context, driver.queueWaitIdle(context->vulkanQueue),
                 "vkQueueWaitIdle") == VERNON_STATUS_OK;
  driver.freeCommandBuffers(context->vulkanDevice, context->vulkanCommandPool,
                            1, &command);
  return submitted;
}

void transitionVulkanImage(VkCommandBuffer command,
                           VernonDeviceTexture *texture,
                           VkImageLayout newLayout) {
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = texture->vulkanLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = texture->vulkanImage;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = texture->mipLevels;
  barrier.subresourceRange.layerCount =
      texture->dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
  VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  if (texture->vulkanLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
    barrier.srcAccessMask = 0;
    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  } else if (texture->vulkanLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (texture->vulkanLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (texture->vulkanLayout ==
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  } else if (texture->vulkanLayout ==
             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  } else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    destinationStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  vernon::runtime::vulkanDriver().cmdPipelineBarrier(
      command, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
      &barrier);
  texture->vulkanLayout = newLayout;
}

void destroyVulkanContext(VernonRuntimeContext *context);

bool initializeVulkanContext(VernonRuntimeContext *context,
                             uint32_t deviceIndex) {
  // VulkanDriver stores dispatch functions globally, so serialize updates to
  // its instance/device dispatch table while a context is initialized.
  static std::mutex initializationMutex;
  std::lock_guard<std::mutex> guard(initializationMutex);
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  const auto failInitialization = [&]() {
    // Keep the diagnostic set by the failing operation while releasing every
    // handle that was successfully created before it.
    destroyVulkanContext(context);
    return false;
  };
  if (!driver.load()) {
    context->error = driver.error;
    return failInitialization();
  }
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "VernonRuntime";
  application.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  application.pEngineName = "VernonRuntime";
  application.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  application.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &application;
  if (vulkanFail(context,
                 driver.createInstance(&instanceInfo, nullptr,
                                       &context->vulkanInstance),
                 "vkCreateInstance") != VERNON_STATUS_OK)
    return failInitialization();
  if (!driver.loadInstance(context->vulkanInstance)) {
    context->error = driver.error;
    return failInitialization();
  }
  uint32_t physicalDeviceCount = 0;
  if (driver.enumeratePhysicalDevices(context->vulkanInstance,
                                      &physicalDeviceCount,
                                      nullptr) != VK_SUCCESS ||
      deviceIndex >= physicalDeviceCount) {
    context->error = "Vulkan device index is unavailable";
    return failInitialization();
  }
  std::vector<VkPhysicalDevice> devices(physicalDeviceCount);
  if (driver.enumeratePhysicalDevices(context->vulkanInstance,
                                      &physicalDeviceCount,
                                      devices.data()) != VK_SUCCESS)
    return failInitialization();
  context->vulkanPhysicalDevice = devices[deviceIndex];
  uint32_t queueCount = 0;
  driver.getPhysicalDeviceQueueFamilyProperties(context->vulkanPhysicalDevice,
                                                &queueCount, nullptr);
  std::vector<VkQueueFamilyProperties> queues(queueCount);
  driver.getPhysicalDeviceQueueFamilyProperties(context->vulkanPhysicalDevice,
                                                &queueCount, queues.data());
  const auto queue = std::find_if(
      queues.begin(), queues.end(), [](const VkQueueFamilyProperties &family) {
        return family.queueFlags & VK_QUEUE_COMPUTE_BIT;
      });
  if (queue == queues.end()) {
    context->error = "Vulkan device has no compute queue";
    return failInitialization();
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
    return failInitialization();
  if (!driver.loadDevice(context->vulkanDevice)) {
    context->error = driver.error;
    return failInitialization();
  }
  driver.getDeviceQueue(context->vulkanDevice, context->vulkanQueueFamily, 0,
                        &context->vulkanQueue);
  driver.getPhysicalDeviceMemoryProperties(context->vulkanPhysicalDevice,
                                           &context->vulkanMemoryProperties);
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = context->vulkanQueueFamily;
  if (vulkanFail(context,
                 driver.createCommandPool(context->vulkanDevice, &poolInfo,
                                          nullptr, &context->vulkanCommandPool),
                 "vkCreateCommandPool") != VERNON_STATUS_OK)
    return failInitialization();
  return true;
}

void destroyVulkanContext(VernonRuntimeContext *context) {
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  if (context->vulkanDevice) {
    if (driver.deviceWaitIdle)
      driver.deviceWaitIdle(context->vulkanDevice);
    if (context->vulkanCommandPool && driver.destroyCommandPool)
      driver.destroyCommandPool(context->vulkanDevice,
                                context->vulkanCommandPool, nullptr);
    PFN_vkDestroyDevice destroyDevice = driver.destroyDevice;
    if (!destroyDevice && driver.getDeviceProcAddr)
      destroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
          driver.getDeviceProcAddr(context->vulkanDevice, "vkDestroyDevice"));
    if (destroyDevice)
      destroyDevice(context->vulkanDevice, nullptr);
  }
  if (context->vulkanInstance) {
    PFN_vkDestroyInstance destroyInstance = driver.destroyInstance;
    if (!destroyInstance && driver.getInstanceProcAddr)
      destroyInstance =
          reinterpret_cast<PFN_vkDestroyInstance>(driver.getInstanceProcAddr(
              context->vulkanInstance, "vkDestroyInstance"));
    if (destroyInstance)
      destroyInstance(context->vulkanInstance, nullptr);
  }
  context->vulkanCommandPool = VK_NULL_HANDLE;
  context->vulkanQueue = VK_NULL_HANDLE;
  context->vulkanDevice = VK_NULL_HANDLE;
  context->vulkanPhysicalDevice = VK_NULL_HANDLE;
  context->vulkanInstance = VK_NULL_HANDLE;
}

std::optional<VkFormat> vulkanTextureFormat(VernonTextureFormat format) {
  switch (format) {
  case VERNON_TEXTURE_R8_UNORM:
    return VK_FORMAT_R8_UNORM;
  case VERNON_TEXTURE_RG8_UNORM:
    return VK_FORMAT_R8G8_UNORM;
  case VERNON_TEXTURE_RGB8_UNORM:
    return VK_FORMAT_R8G8B8_UNORM;
  case VERNON_TEXTURE_RGBA8_UNORM:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case VERNON_TEXTURE_RGBA8_SRGB:
    return VK_FORMAT_R8G8B8A8_SRGB;
  case VERNON_TEXTURE_RGBA16_FLOAT:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case VERNON_TEXTURE_RGBA32_FLOAT:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  case VERNON_TEXTURE_R11G11B10_FLOAT:
    return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  case VERNON_TEXTURE_R16_FLOAT:
    return VK_FORMAT_R16_SFLOAT;
  case VERNON_TEXTURE_R32_FLOAT:
    return VK_FORMAT_R32_SFLOAT;
  }
  return std::nullopt;
}

std::optional<VkSamplerAddressMode>
vulkanSamplerAddressMode(VernonSamplerWrapMode mode) {
  switch (mode) {
  case VERNON_SAMPLER_REPEAT:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case VERNON_SAMPLER_MIRRORED_REPEAT:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  case VERNON_SAMPLER_CLAMP_TO_EDGE:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case VERNON_SAMPLER_CLAMP_TO_BORDER:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  }
  return std::nullopt;
}

std::optional<VkFilter> vulkanSamplerFilter(VernonSamplerFilter filter) {
  switch (filter) {
  case VERNON_SAMPLER_NEAREST:
    return VK_FILTER_NEAREST;
  case VERNON_SAMPLER_LINEAR:
    return VK_FILTER_LINEAR;
  }
  return std::nullopt;
}
#endif

void makeCurrent(VernonRuntimeContext *context) {
  context->external.make_current(context->external.user_data);
}

GlUint compileShader(VernonRuntimeContext *context, GlEnum kind,
                     const std::string &source) {
  OpenGLDriver &gl = context->gl;
  const GlUint shader = gl.createShader(kind);
  const char *data = source.data();
  const GlInt size = static_cast<GlInt>(source.size());
  gl.shaderSource(shader, 1, &data, &size);
  gl.compileShader(shader);
  GlInt compiled = 0;
  gl.getShaderiv(shader, kCompileStatus, &compiled);
  if (compiled)
    return shader;
  fail(context, "OpenGL shader compilation failed: " + shaderLog(gl, shader));
  gl.deleteShader(shader);
  return 0;
}

GlUint linkProgram(VernonRuntimeContext *context,
                   const std::vector<GlUint> &shaders) {
  OpenGLDriver &gl = context->gl;
  const GlUint program = gl.createProgram();
  for (GlUint shader : shaders)
    gl.attachShader(program, shader);
  gl.linkProgram(program);
  for (GlUint shader : shaders)
    gl.deleteShader(shader);
  GlInt linked = 0;
  gl.getProgramiv(program, kLinkStatus, &linked);
  if (linked)
    return program;
  fail(context, "OpenGL program link failed: " + programLog(gl, program));
  gl.deleteProgram(program);
  return 0;
}

bool parseUse(const nlohmann::json &value, ParameterUse &use,
              std::string &error) {
  if (!value.is_object()) {
    error = "pipeline parameter use must be an object";
    return false;
  }
  use.stage = value.value("stage", "");
  use.interfaceKind = value.value("interface", "");
  use.uniformName = value.value("uniform_name", "");
  if (value.contains("dtype") && value["dtype"].is_string())
    use.dtype = value["dtype"].get<std::string>();
  use.index = value.value("index", 0u);
  use.location = value.value("vernon.location", UINT32_MAX);
  use.divisor = value.value("vernon.instance_divisor", 0u);
  use.descriptorSet = value.value("vernon.set", 0u);
  use.binding = value.value("vernon.binding", UINT32_MAX);
  if (value.contains("sampled_texture_bindings")) {
    const nlohmann::json &bindings = value["sampled_texture_bindings"];
    if (!bindings.is_array()) {
      error = "sampled_texture_bindings must be an array";
      return false;
    }
    for (const nlohmann::json &binding : bindings) {
      if (!binding.is_object() || !binding.contains("set") ||
          !binding["set"].is_number_unsigned() ||
          !binding.contains("binding") ||
          !binding["binding"].is_number_unsigned()) {
        error = "sampled texture binding must contain unsigned set/binding";
        return false;
      }
      use.sampledTextureBindings.push_back(
          {binding["set"].get<uint32_t>(),
           binding["binding"].get<uint32_t>()});
    }
  } else {
    const uint32_t sampledTextureSet =
        value.value("sampled_texture_set", UINT32_MAX);
    const uint32_t sampledTextureBinding =
        value.value("sampled_texture_binding", UINT32_MAX);
    if (sampledTextureSet != UINT32_MAX ||
        sampledTextureBinding != UINT32_MAX) {
      if (sampledTextureSet == UINT32_MAX ||
          sampledTextureBinding == UINT32_MAX) {
        error = "legacy sampled texture pairing is incomplete";
        return false;
      }
      use.sampledTextureBindings.push_back(
          {sampledTextureSet, sampledTextureBinding});
    }
  }
  if (value.contains("shape") && value["shape"].is_array())
    for (const nlohmann::json &dimension : value["shape"])
      use.shape.push_back(dimension.is_number_unsigned()
                              ? dimension.get<uint64_t>()
                              : uint64_t{0});
  if (use.stage.empty() || use.interfaceKind.empty()) {
    error = "pipeline parameter use is missing stage/interface metadata";
    return false;
  }
  return true;
}

void parseStaticType(const std::string &type, std::string &dtype,
                     std::vector<uint64_t> &shape) {
  if (type.rfind("tensor<", 0) != 0 || type.size() < 9 || type.back() != '>') {
    dtype = type;
    return;
  }
  const std::string body = type.substr(7, type.size() - 8);
  size_t begin = 0;
  while (true) {
    const size_t separator = body.find('x', begin);
    if (separator == std::string::npos) {
      dtype = body.substr(begin);
      return;
    }
    const std::string dimension = body.substr(begin, separator - begin);
    if (dimension == "?")
      shape.push_back(0);
    else {
      try {
        shape.push_back(std::stoull(dimension));
      } catch (...) {
        dtype.clear();
        shape.clear();
        return;
      }
    }
    begin = separator + 1;
  }
}

bool parseVariant(const nlohmann::json &value, Variant &variant,
                  std::string &error) {
  if (!value.is_object() || !value.contains("key") ||
      !value.contains("parameters") || !value.contains("steps") ||
      !value["key"].is_array() || !value["parameters"].is_array() ||
      !value["steps"].is_array()) {
    error = "pipeline variant tables are invalid";
    return false;
  }
  for (const nlohmann::json &feature : value["key"]) {
    if (!feature.is_string()) {
      error = "pipeline variant feature is not a string";
      return false;
    }
    variant.key.push_back(feature.get<std::string>());
  }
  if (!std::is_sorted(variant.key.begin(), variant.key.end()) ||
      std::adjacent_find(variant.key.begin(), variant.key.end()) !=
          variant.key.end()) {
    error = "pipeline variant feature key is not canonical";
    return false;
  }
  for (const nlohmann::json &row : value["parameters"]) {
    if (!row.is_object() || !row.contains("slot") || !row.contains("uses") ||
        !row["uses"].is_array()) {
      error = "pipeline parameter record is invalid";
      return false;
    }
    Parameter parameter;
    parameter.slot = row["slot"].get<uint32_t>();
    parameter.name = row.value("name", "");
    parameter.kind = row.value("kind", "");
    parameter.dtype = row.value("dtype", "");
    parameter.access = row.value("access", "read");
    parameter.dimension = row.value("dimension", "");
    if (row.contains("shape") && row["shape"].is_array())
      for (const nlohmann::json &dimension : row["shape"])
        parameter.shape.push_back(dimension.is_number_unsigned()
                                      ? dimension.get<uint64_t>()
                                      : uint64_t{0});
    for (const nlohmann::json &useValue : row["uses"]) {
      ParameterUse use;
      if (!parseUse(useValue, use, error))
        return false;
      parameter.uses.push_back(std::move(use));
    }
    if (parameter.name.empty() || parameter.uses.empty()) {
      error = "pipeline parameter has no name or uses";
      return false;
    }
    variant.parameters.push_back(std::move(parameter));
  }
  std::sort(variant.parameters.begin(), variant.parameters.end(),
            [](const Parameter &left, const Parameter &right) {
              return left.slot < right.slot;
            });
  if (std::adjacent_find(variant.parameters.begin(), variant.parameters.end(),
                         [](const Parameter &left, const Parameter &right) {
                           return left.slot == right.slot;
                         }) != variant.parameters.end()) {
    error = "pipeline variant contains duplicate parameter slots";
    return false;
  }
  for (const nlohmann::json &row :
       value.value("outputs", nlohmann::json::array())) {
    if (!row.is_object()) {
      error = "pipeline output record is invalid";
      return false;
    }
    Output output;
    output.name = row.value("name", "");
    output.kind = row.value("kind", "texture");
    output.dtype = row.value("dtype", "");
    output.access = row.value("access", "write");
    output.location = row.value("location", UINT32_MAX);
    if (output.name.empty() && output.location != UINT32_MAX)
      output.name = "output_" + std::to_string(output.location);
    if (row.contains("shape") && row["shape"].is_array())
      for (const nlohmann::json &dimension : row["shape"])
        output.shape.push_back(dimension.is_number_unsigned()
                                   ? dimension.get<uint64_t>()
                                   : uint64_t{0});
    if (output.dtype.empty())
      parseStaticType(row.value("type", ""), output.dtype, output.shape);
    if (output.name.empty() || output.dtype.empty() ||
        output.location == UINT32_MAX) {
      error = "pipeline output metadata is incomplete";
      return false;
    }
    variant.outputs.push_back(std::move(output));
  }
  for (const nlohmann::json &step : value["steps"]) {
    const std::string kind = step.value("kind", "");
    if (kind == "dispatch")
      variant.compute = step.value("stage", "");
    else if (kind == "barrier")
      variant.barrier = true;
    else if (kind == "draw") {
      variant.vertex = step.value("vertex", "");
      variant.fragment = step.value("fragment", "");
    } else {
      error = "pipeline step kind is unsupported";
      return false;
    }
  }
  if (variant.vertex.empty() != variant.fragment.empty()) {
    error = "graphics pipeline requires both vertex and fragment stages";
    return false;
  }
  return !variant.compute.empty() || !variant.vertex.empty();
}

bool parseReflection(const nlohmann::json &root, const std::string &selected,
                     ReflectedEntry &output, std::string &error) {
  if (!root.is_object() || root.value("gpu_launch_abi_version", 0) != 1 ||
      !root.contains("entries") || !root["entries"].is_array()) {
    error = "unsupported or invalid compute reflection";
    return false;
  }
  for (const nlohmann::json &entry : root["entries"]) {
    if (!entry.is_object() || entry.value("name", "") != selected)
      continue;
    output.cpuArgumentsSize = entry.value("cpu_arguments_size", size_t{0});
    if (entry.contains("workgroup_size") &&
        entry["workgroup_size"].is_array() &&
        entry["workgroup_size"].size() == 3)
      for (size_t index = 0; index < 3; ++index)
        output.workgroup[index] =
            entry["workgroup_size"][index].get<uint32_t>();
    if (!output.cpuArgumentsSize || !entry.contains("arguments") ||
        !entry["arguments"].is_array()) {
      error = "CPU entry has no argument layout";
      return false;
    }
    for (const nlohmann::json &value : entry["arguments"]) {
      if (!value.is_object()) {
        error = "CPU entry contains invalid argument reflection";
        return false;
      }
      ReflectedArgument argument;
      argument.kind = value.value("kind", "scalar");
      argument.builtin = value.value("builtin", "");
      argument.cpuOffset = value.value("cpu_offset", size_t{0});
      argument.cpuSize = value.value("cpu_size", size_t{0});
      argument.alignment = value.value("alignment", size_t{1});
      argument.descriptorSet = value.value("vernon.set", uint32_t{0});
      argument.binding = value.value("vernon.binding", UINT32_MAX);
      const std::string dtype = value.value("dtype", "");
      const size_t elementSize = dtype == "f64"    ? 8
                                 : dtype == "f16"  ? 2
                                 : dtype == "bool" ? 1
                                                   : 4;
      argument.tensorElementSize = elementSize;
      if (value.contains("shape") && value["shape"].is_array()) {
        size_t elements = 1;
        for (const nlohmann::json &dimension : value["shape"]) {
          if (!dimension.is_number_unsigned()) {
            elements = 0;
            break;
          }
          elements *= dimension.get<size_t>();
        }
        argument.tensorElements = elements;
        argument.tensorBytes = elements * elementSize;
      }
      if (!argument.cpuSize || argument.cpuOffset > output.cpuArgumentsSize ||
          argument.cpuSize > output.cpuArgumentsSize - argument.cpuOffset ||
          (argument.kind != "tensor" && argument.kind != "scalar" &&
           argument.kind != "builtin")) {
        error = "CPU argument layout is invalid";
        return false;
      }
      output.arguments.push_back(std::move(argument));
    }
    return true;
  }
  error = "selected CPU entry is absent from reflection";
  return false;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

bool isSha256(const std::string &value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool validateManifestHash(const nlohmann::json &root, bool required,
                          std::string &error) {
  if (!root.contains("content_hash")) {
    if (!required)
      return true;
    error = "pipeline manifest content_hash is missing";
    return false;
  }
  if (!root["content_hash"].is_string()) {
    error = "pipeline manifest content_hash is invalid";
    return false;
  }
  const std::string expected = root["content_hash"].get<std::string>();
  if (!isSha256(expected)) {
    error = "pipeline manifest content_hash is invalid";
    return false;
  }
  nlohmann::json canonical = root;
  canonical.erase("content_hash");
  const std::string bytes =
      canonical.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
  if (vernon::runtime::sha256Hex(bytes.data(), bytes.size()) != expected) {
    error = "pipeline manifest content_hash does not match canonical content";
    return false;
  }
  return true;
}

std::optional<std::vector<uint8_t>>
decodeBase64Strict(const std::string &encoded) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (encoded.size() % 4 != 0)
    return std::nullopt;
  std::vector<uint8_t> result;
  result.reserve(encoded.size() / 4 * 3);
  for (size_t offset = 0; offset < encoded.size(); offset += 4) {
    uint32_t value = 0;
    size_t padding = 0;
    for (size_t index = 0; index < 4; ++index) {
      const unsigned char character = encoded[offset + index];
      if (character == '=') {
        if (index < 2 || offset + 4 != encoded.size())
          return std::nullopt;
        ++padding;
        value <<= 6;
        continue;
      }
      if (padding)
        return std::nullopt;
      const char *position =
          std::find(std::begin(alphabet), std::end(alphabet) - 1, character);
      if (position == std::end(alphabet) - 1)
        return std::nullopt;
      value = (value << 6) | static_cast<uint32_t>(position - alphabet);
    }
    if (padding > 2 || (padding == 1 && (value & 0xffu) != 0) ||
        (padding == 2 && (value & 0xffffu) != 0))
      return std::nullopt;
    result.push_back(static_cast<uint8_t>(value >> 16));
    if (padding < 2)
      result.push_back(static_cast<uint8_t>(value >> 8));
    if (!padding)
      result.push_back(static_cast<uint8_t>(value));
  }
  return result;
}

bool resolveArtifact(VernonRuntimeContext *context,
                     const nlohmann::json &descriptor,
                     const std::optional<std::filesystem::path> &directory,
                     ResolvedArtifact &output) {
  if (!descriptor.is_object() || !descriptor.contains("format") ||
      !descriptor["format"].is_string() || !descriptor.contains("storage") ||
      !descriptor["storage"].is_string() || !descriptor.contains("size") ||
      !descriptor["size"].is_number_unsigned() ||
      !descriptor.contains("sha256") || !descriptor["sha256"].is_string()) {
    fail(context, "pipeline artifact descriptor is invalid");
    return false;
  }
  output.format = descriptor["format"].get<std::string>();
  const std::string storage = descriptor["storage"].get<std::string>();
  const uint64_t declaredSize = descriptor["size"].get<uint64_t>();
  const std::string expectedHash = descriptor["sha256"].get<std::string>();
  if (output.format.empty() || !isSha256(expectedHash)) {
    fail(context, "pipeline artifact format or SHA-256 is invalid");
    return false;
  }

  if (storage == "inline") {
    if (!descriptor.contains("encoding") ||
        !descriptor["encoding"].is_string() || !descriptor.contains("data") ||
        !descriptor["data"].is_string()) {
      fail(context, "inline pipeline artifact is invalid");
      return false;
    }
    const std::string encoding = descriptor["encoding"].get<std::string>();
    const std::string data = descriptor["data"].get<std::string>();
    if (encoding == "utf8")
      output.bytes.assign(data.begin(), data.end());
    else if (encoding == "base64") {
      const auto decoded = decodeBase64Strict(data);
      if (!decoded) {
        fail(context, "inline pipeline artifact base64 is invalid");
        return false;
      }
      output.bytes = *decoded;
    } else {
      fail(context, "inline pipeline artifact encoding is unsupported");
      return false;
    }
  } else if (storage == "external") {
    if (!directory || !descriptor.contains("path") ||
        !descriptor["path"].is_string()) {
      fail(context,
           "external pipeline artifacts require a bundle directory and path");
      return false;
    }
    const std::filesystem::path relative =
        std::filesystem::u8path(descriptor["path"].get<std::string>());
    if (relative.empty() || relative.is_absolute() ||
        relative.has_root_path() || relative == std::filesystem::path(".") ||
        relative.lexically_normal() != relative ||
        std::find(relative.begin(), relative.end(),
                  std::filesystem::path("..")) != relative.end()) {
      fail(context, "external pipeline artifact path is not normalized");
      return false;
    }
    std::error_code filesystemError;
    const std::filesystem::path root =
        std::filesystem::canonical(*directory, filesystemError);
    if (filesystemError ||
        !std::filesystem::is_directory(root, filesystemError)) {
      fail(context, "pipeline bundle directory is invalid");
      return false;
    }
    output.path = std::filesystem::canonical(root / relative, filesystemError);
    if (filesystemError ||
        !std::filesystem::is_regular_file(output.path, filesystemError)) {
      fail(context, "external pipeline artifact cannot be resolved");
      return false;
    }
    const std::filesystem::path contained =
        output.path.lexically_relative(root);
    if (contained.empty() || contained.is_absolute() ||
        *contained.begin() == std::filesystem::path("..")) {
      fail(context, "external pipeline artifact escapes the bundle directory");
      return false;
    }
    const std::string bytes = readFile(output.path);
    output.bytes.assign(bytes.begin(), bytes.end());
    output.external = true;
  } else {
    fail(context, "pipeline artifact storage is unsupported");
    return false;
  }

  if (output.bytes.size() != declaredSize ||
      vernon::runtime::sha256Hex(output.bytes.data(), output.bytes.size()) !=
          expectedHash) {
    fail(context, "pipeline artifact size or SHA-256 mismatch");
    return false;
  }
  return true;
}

const char *hostOperatingSystem() {
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

const char *hostArchitecture() {
#if defined(_M_X64) || defined(__x86_64__)
  return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
  return "aarch64";
#else
  return "unknown";
#endif
}

bool resolveCpuNativeArtifact(VernonRuntimeContext *context,
                              const CpuNativeArtifact &artifact,
                              std::filesystem::path &libraryPath,
                              ReflectedEntry *reflection) {
  const bool nativeLibrary = artifact.format == "native_library";
  const bool relocatableObject = artifact.format == "relocatable_object";
  if (artifact.entry.empty() || artifact.symbol.empty() ||
      artifact.relativeLibrary.empty() || artifact.sha256.size() != 64 ||
      (!nativeLibrary && !relocatableObject) ||
      (nativeLibrary && (artifact.operatingSystem != hostOperatingSystem() ||
                         artifact.architecture != hostArchitecture())) ||
      (relocatableObject &&
       (artifact.targetTriple.empty() ||
        (artifact.objectFormat != "coff" && artifact.objectFormat != "elf" &&
         artifact.objectFormat != "macho" &&
         artifact.objectFormat != "wasm"))) ||
      artifact.invocationAbiVersion != VERNON_CPU_INVOCATION_ABI_VERSION) {
    fail(context, "unsupported or invalid CPU AOT artifact");
    return false;
  }
  if (artifact.relativeLibrary.is_absolute() ||
      artifact.relativeLibrary.has_root_path() ||
      std::find(artifact.relativeLibrary.begin(),
                artifact.relativeLibrary.end(), std::filesystem::path("..")) !=
          artifact.relativeLibrary.end()) {
    fail(context, "CPU AOT artifact path is invalid");
    return false;
  }

  std::error_code error;
  const std::filesystem::path canonicalRoot =
      std::filesystem::canonical(artifact.root, error);
  if (error || !std::filesystem::is_directory(canonicalRoot, error)) {
    fail(context, "CPU AOT bundle directory is invalid");
    return false;
  }
  const std::filesystem::path candidate = std::filesystem::weakly_canonical(
      canonicalRoot / artifact.relativeLibrary, error);
  if (error) {
    fail(context, "CPU AOT artifact path cannot be resolved");
    return false;
  }
  const std::filesystem::path relative =
      candidate.lexically_relative(canonicalRoot);
  if (relative.empty() || relative.is_absolute() ||
      (!relative.empty() && *relative.begin() == std::filesystem::path(".."))) {
    fail(context, "CPU AOT artifact escapes the bundle directory");
    return false;
  }

  const std::string bytes = readFile(candidate);
  if (bytes.empty() || artifact.size != bytes.size() ||
      vernon::runtime::sha256Hex(bytes.data(), bytes.size()) !=
          artifact.sha256) {
    fail(context, "CPU AOT artifact size or SHA-256 mismatch");
    return false;
  }
  ReflectedEntry parsed;
  if (!parseReflection(artifact.reflection, artifact.entry, parsed,
                       context->error))
    return false;
  libraryPath = candidate;
  if (reflection)
    *reflection = std::move(parsed);
  return true;
}

VernonLoadedKernel *loadCpuNativeArtifact(VernonRuntimeContext *context,
                                          const CpuNativeArtifact &artifact) {
  std::filesystem::path libraryPath;
  auto kernel = std::make_unique<VernonLoadedKernel>();
  kernel->context = context;
  if (!resolveCpuNativeArtifact(context, artifact, libraryPath,
                                &kernel->reflection))
    return nullptr;
  if (artifact.format == "relocatable_object") {
    std::lock_guard<std::mutex> lock(staticCpuEntriesMutex());
    auto found = staticCpuEntries().find(artifact.symbol);
    if (found != staticCpuEntries().end())
      kernel->cpuEntry = found->second;
  } else {
    const std::string nativePath = libraryPath.u8string();
    if (!kernel->nativeLibrary.open(nativePath.c_str(), context->error))
      return nullptr;
    kernel->cpuEntry = reinterpret_cast<VernonCpuEntryPoint>(
        kernel->nativeLibrary.symbol(artifact.symbol.c_str()));
  }
  if (!kernel->cpuEntry) {
    fail(context, artifact.format == "relocatable_object"
                      ? "CPU AOT object symbol '" + artifact.symbol +
                            "' was not statically registered"
                      : "CPU AOT library does not export symbol '" +
                            artifact.symbol + "'");
    return nullptr;
  }
  ++context->liveKernels;
  return kernel.release();
}

bool argumentKindMatches(const Parameter &parameter,
                         const VernonPipelineArgument &argument) {
  return (parameter.kind == "tensor" &&
          argument.kind == VERNON_PIPELINE_TENSOR) ||
         (parameter.kind == "texture" &&
          argument.kind == VERNON_PIPELINE_TEXTURE) ||
         (parameter.kind == "sampler" &&
          argument.kind == VERNON_PIPELINE_SAMPLER) ||
         (parameter.kind == "inline" &&
          argument.kind == VERNON_PIPELINE_INLINE_VALUE);
}

std::optional<VernonPipelineArgumentKind>
pipelineArgumentKind(const std::string &kind) {
  if (kind == "tensor")
    return VERNON_PIPELINE_TENSOR;
  if (kind == "texture")
    return VERNON_PIPELINE_TEXTURE;
  if (kind == "sampler")
    return VERNON_PIPELINE_SAMPLER;
  if (kind == "inline")
    return VERNON_PIPELINE_INLINE_VALUE;
  return std::nullopt;
}

std::optional<VernonDataType> pipelineDataType(const std::string &dtype) {
  if (dtype == "bool")
    return VERNON_DATA_BOOL;
  if (dtype == "i32")
    return VERNON_DATA_I32;
  if (dtype == "u32")
    return VERNON_DATA_U32;
  if (dtype == "f16")
    return VERNON_DATA_F16;
  if (dtype == "f32")
    return VERNON_DATA_F32;
  if (dtype == "f64")
    return VERNON_DATA_F64;
  return std::nullopt;
}

std::optional<VernonValueAccess>
pipelineValueAccess(const std::string &access) {
  if (access == "read")
    return VERNON_ACCESS_READ;
  if (access == "write")
    return VERNON_ACCESS_WRITE;
  if (access == "read_write")
    return VERNON_ACCESS_READ_WRITE;
  return std::nullopt;
}

using PipelineArgumentMap =
    std::unordered_map<uint32_t, const VernonPipelineArgument *>;

VernonStatus
prepareComputeLaunch(VernonRuntimeContext *context, const Variant &variant,
                     const PipelineArgumentMap &arguments,
                     const VernonPipelineInvocation &invocation,
                     std::vector<VernonLaunchArgument> &launchArguments,
                     VernonLaunchSize &grid) {
  struct IndexedArgument {
    uint32_t index;
    VernonLaunchArgument argument;
  };
  std::vector<IndexedArgument> indexed;
  for (const Parameter &parameter : variant.parameters) {
    const VernonPipelineArgument &supplied = *arguments.at(parameter.slot);
    for (const ParameterUse &use : parameter.uses) {
      if (use.stage != "compute")
        continue;
      VernonLaunchArgument argument{};
      if (supplied.kind == VERNON_PIPELINE_TENSOR) {
        if (!supplied.tensor.buffer ||
            supplied.tensor.buffer->context != context ||
            supplied.tensor.byte_offset != 0)
          return fail(context, "compute Tensor view is invalid");
        argument.kind = VERNON_LAUNCH_TENSOR;
        argument.buffer = supplied.tensor.buffer;
      } else if (supplied.kind == VERNON_PIPELINE_INLINE_VALUE) {
        if (!supplied.inline_value.data || !supplied.inline_value.data_size)
          return fail(context, "compute inline value is empty");
        argument.kind = VERNON_LAUNCH_SCALAR;
        argument.scalar_data = supplied.inline_value.data;
        argument.scalar_size = supplied.inline_value.data_size;
      } else {
        return fail(context, "compute argument kind is unsupported");
      }
      indexed.push_back({use.index, argument});
    }
  }
  std::sort(indexed.begin(), indexed.end(),
            [](const IndexedArgument &left, const IndexedArgument &right) {
              return left.index < right.index;
            });
  for (const IndexedArgument &value : indexed)
    launchArguments.push_back(value.argument);

  grid = invocation.compute_grid;
  if (!grid.x || !grid.y || !grid.z) {
    for (const auto &[slot, argument] : arguments) {
      (void)slot;
      if (argument->kind != VERNON_PIPELINE_TENSOR || !argument->tensor.rank ||
          !argument->tensor.shape)
        continue;
      grid = {1, 1, 1};
      const uint32_t rank = argument->tensor.rank;
      grid.x = static_cast<uint32_t>(argument->tensor.shape[rank - 1]);
      if (rank > 1)
        grid.y = static_cast<uint32_t>(argument->tensor.shape[rank - 2]);
      if (rank > 2)
        grid.z = static_cast<uint32_t>(argument->tensor.shape[rank - 3]);
      break;
    }
  }
  if (!grid.x || !grid.y || !grid.z)
    return fail(context, "compute grid cannot be inferred");
  return VERNON_STATUS_OK;
}

GlEnum topologyMode(VernonPrimitiveTopology topology) {
  if (topology == VERNON_TOPOLOGY_LINE_LIST)
    return kLines;
  if (topology == VERNON_TOPOLOGY_POINT_LIST)
    return kPoints;
  return kTriangles;
}

#if defined(VERNON_HAS_VULKAN_RUNTIME)
VkPrimitiveTopology vulkanTopology(VernonPrimitiveTopology topology) {
  if (topology == VERNON_TOPOLOGY_LINE_LIST)
    return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  if (topology == VERNON_TOPOLOGY_POINT_LIST)
    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkFormat vulkanVertexFormat(uint32_t components) {
  if (components == 1)
    return VK_FORMAT_R32_SFLOAT;
  if (components == 2)
    return VK_FORMAT_R32G32_SFLOAT;
  if (components == 3)
    return VK_FORMAT_R32G32B32_SFLOAT;
  if (components == 4)
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  return VK_FORMAT_UNDEFINED;
}

struct VulkanCapabilityProbe {
  bool available{};
  std::string diagnostic;
};

const VulkanCapabilityProbe &cachedVulkanCapabilityProbe() {
  // Function-local static initialization makes the expensive device probe
  // happen once and safely publishes both its result and diagnostic.
  static const VulkanCapabilityProbe probe = [] {
    VulkanCapabilityProbe result;
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_VULKAN;
    result.available = initializeVulkanContext(&context, 0);
    result.diagnostic = context.error;
    destroyVulkanContext(&context);
    return result;
  }();
  return probe;
}
#endif

} // namespace

extern "C" {

VernonRuntimeCapabilities
vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
  static thread_local std::string diagnostic;
  diagnostic.clear();
  VernonRuntimeCapabilities result{};
  if (backend == VERNON_RUNTIME_CPU) {
    result.available = 1;
    result.supports_compute = 1;
    result.supports_storage_buffers = 1;
  } else if (backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    vernon::runtime::CudaDriver &driver = vernon::runtime::cudaDriver();
    if (!driver.load()) {
      diagnostic = driver.error;
    } else {
      vernon::runtime::CudaDevice device{};
      vernon::runtime::CudaResult status = driver.init(0);
      if (status == vernon::runtime::kCudaSuccess)
        status = driver.deviceGet(&device, 0);
      result.available = status == vernon::runtime::kCudaSuccess;
      result.supports_compute = result.available;
      result.supports_storage_buffers = result.available;
      if (!result.available)
        diagnostic = "CUDA Driver loaded but no usable device was found";
    }
#else
    diagnostic = "VernonRuntime was built without CUDA support";
#endif
  } else if (backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    const VulkanCapabilityProbe &probe = cachedVulkanCapabilityProbe();
    result.available = probe.available;
    result.supports_compute = result.available;
    result.supports_storage_buffers = result.available;
    result.diagnostic = {probe.diagnostic.data(), probe.diagnostic.size()};
    return result;
#else
    diagnostic = "VernonRuntime was built without Vulkan support";
#endif
  } else if (backend == VERNON_RUNTIME_OPENGL ||
             backend == VERNON_RUNTIME_OPENGL_ES) {
    result.supports_graphics = 1;
    diagnostic = "backend requires a host-owned external context";
  } else {
    diagnostic = "VernonRuntime backend is not enabled";
  }
  result.diagnostic = {diagnostic.data(), diagnostic.size()};
  return result;
}

VernonRuntimeContext *vernonRuntimeCreate(VernonRuntimeBackend backend,
                                          uint32_t deviceIndex) {
  VernonRuntimeCreateOptions options{};
  options.struct_size = sizeof(options);
  options.device_index = deviceIndex;
  return vernonRuntimeCreateWithOptions(backend, &options);
}

VernonRuntimeContext *
vernonRuntimeCreateWithOptions(VernonRuntimeBackend backend,
                               const VernonRuntimeCreateOptions *options) {
  if (!options || options->struct_size < sizeof(VernonRuntimeCreateOptions))
    return nullptr;
  auto context = std::make_unique<VernonRuntimeContext>();
  context->backend = backend;
  if (backend == VERNON_RUNTIME_CPU)
    return options->device_index == 0 ? context.release() : nullptr;
  if (backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!initializeVulkanContext(context.get(), options->device_index))
      return nullptr;
    return context.release();
#else
    return nullptr;
#endif
  }
  if (backend != VERNON_RUNTIME_CUDA)
    return nullptr;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  vernon::runtime::CudaDriver &driver = vernon::runtime::cudaDriver();
  if (!driver.load() || driver.init(0) != vernon::runtime::kCudaSuccess ||
      driver.deviceGet(&context->cudaDevice,
                       static_cast<int>(options->device_index)) !=
          vernon::runtime::kCudaSuccess ||
      driver.primaryContextRetain(&context->cudaContext, context->cudaDevice) !=
          vernon::runtime::kCudaSuccess)
    return nullptr;
  if (driver.contextSetCurrent(context->cudaContext) !=
      vernon::runtime::kCudaSuccess) {
    driver.primaryContextRelease(context->cudaDevice);
    return nullptr;
  }
  return context.release();
#else
  return nullptr;
#endif
}

VernonRuntimeContext *vernonRuntimeCreateExternalOpenGL(
    const VernonExternalOpenGLContext *externalContext) {
  return vernonRuntimeCreateExternalOpenGLForBackend(VERNON_RUNTIME_OPENGL,
                                                     externalContext);
}

VernonRuntimeContext *vernonRuntimeCreateExternalOpenGLForBackend(
    VernonRuntimeBackend backend,
    const VernonExternalOpenGLContext *externalContext) {
  if (!externalContext ||
      externalContext->struct_size < sizeof(VernonExternalOpenGLContext) ||
      !externalContext->make_current || !externalContext->get_proc_address ||
      (backend != VERNON_RUNTIME_OPENGL &&
       backend != VERNON_RUNTIME_OPENGL_ES) ||
      externalContext->api_version_major < 2)
    return nullptr;
  auto context = std::make_unique<VernonRuntimeContext>();
  context->backend = backend;
  context->external = *externalContext;
  makeCurrent(context.get());
  if (!loadOpenGLDriver(context->external, context->gl, context->error))
    return nullptr;
  const bool computeExpected =
      backend == VERNON_RUNTIME_OPENGL_ES
          ? (externalContext->api_version_major > 3 ||
             (externalContext->api_version_major == 3 &&
              externalContext->api_version_minor >= 1))
          : (externalContext->api_version_major > 4 ||
             (externalContext->api_version_major == 4 &&
              externalContext->api_version_minor >= 3));
  if (computeExpected) {
    if (!context->gl.dispatchCompute || !context->gl.memoryBarrier)
      return nullptr;
  }
  return context.release();
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_OK;
  if (context->liveBuffers || context->liveKernels || context->liveTextures ||
      context->liveSamplers || context->liveBundles || context->livePipelines)
    return fail(context, "runtime context still owns live handles");
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA) {
    vernon::runtime::cudaDriver().contextSynchronize();
    vernon::runtime::cudaDriver().primaryContextRelease(context->cudaDevice);
  }
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (context->backend == VERNON_RUNTIME_VULKAN)
    destroyVulkanContext(context);
#endif
  delete context;
  return VERNON_STATUS_OK;
}

VernonStringView
vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
  return context
             ? VernonStringView{context->error.data(), context->error.size()}
             : VernonStringView{nullptr, 0};
}

VernonRuntimeCapabilities
vernonRuntimeGetContextCapabilities(const VernonRuntimeContext *context) {
  VernonRuntimeCapabilities result{};
  if (!context)
    return result;
  result.available = 1;
  if (context->backend == VERNON_RUNTIME_CPU ||
      context->backend == VERNON_RUNTIME_CUDA ||
      context->backend == VERNON_RUNTIME_VULKAN) {
    result.supports_compute = 1;
    result.supports_storage_buffers = 1;
    result.diagnostic = {context->error.data(), context->error.size()};
    return result;
  }
  result.supports_graphics = 1;
  result.api_version_major = context->external.api_version_major;
  result.api_version_minor = context->external.api_version_minor;
  result.supports_compute =
      context->backend == VERNON_RUNTIME_OPENGL_ES
          ? (result.api_version_major > 3 ||
             (result.api_version_major == 3 && result.api_version_minor >= 1))
          : (result.api_version_major > 4 ||
             (result.api_version_major == 4 && result.api_version_minor >= 3));
  result.supports_storage_buffers = result.supports_compute;
  result.graphics_draw_abi_version = 2;
  result.diagnostic = {context->error.data(), context->error.size()};
  return result;
}

VernonDeviceBuffer *vernonRuntimeBufferAllocate(VernonRuntimeContext *context,
                                                size_t size, size_t alignment) {
  if (!context ||
      (context->backend != VERNON_RUNTIME_CPU &&
       context->backend != VERNON_RUNTIME_CUDA &&
       context->backend != VERNON_RUNTIME_VULKAN) ||
      !size || !alignment || (alignment & (alignment - 1))) {
    fail(context, "invalid compute buffer allocation");
    return nullptr;
  }
  auto result = std::make_unique<VernonDeviceBuffer>();
  result->context = context;
  result->size = size;
  result->alignment = alignment;
  if (context->backend == VERNON_RUNTIME_CPU) {
    result->host.resize(size);
  } else if (context->backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (cudaFail(context,
                 vernon::runtime::cudaDriver().memoryAllocate(
                     &result->cudaDevice, size),
                 "cuMemAlloc") != VERNON_STATUS_OK)
      return nullptr;
#else
    return nullptr;
#endif
  } else {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!createVulkanBuffer(context, size, result->vulkanBuffer,
                            result->vulkanMemory))
      return nullptr;
#else
    return nullptr;
#endif
  }
  ++context->liveBuffers;
  return result.release();
}

VernonDeviceBuffer *
vernonRuntimeImportOpenGLBuffer(VernonRuntimeContext *context, uint32_t buffer,
                                size_t size, size_t alignment) {
  if (!context || !buffer || !size || !alignment ||
      (alignment & (alignment - 1)) ||
      (context->backend != VERNON_RUNTIME_OPENGL &&
       context->backend != VERNON_RUNTIME_OPENGL_ES))
    return nullptr;
  auto result = std::make_unique<VernonDeviceBuffer>();
  result->context = context;
  result->name = buffer;
  result->size = size;
  result->alignment = alignment;
  ++context->liveBuffers;
  return result.release();
}

VernonStatus vernonRuntimeBufferFree(VernonDeviceBuffer *buffer) {
  if (!buffer)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (buffer->context->backend == VERNON_RUNTIME_CUDA && buffer->cudaDevice &&
      cudaFail(buffer->context,
               vernon::runtime::cudaDriver().memoryFree(buffer->cudaDevice),
               "cuMemFree") != VERNON_STATUS_OK)
    return VERNON_STATUS_INTERNAL_ERROR;
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (buffer->context->backend == VERNON_RUNTIME_VULKAN) {
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
    if (buffer->vulkanBuffer)
      driver.destroyBuffer(buffer->context->vulkanDevice, buffer->vulkanBuffer,
                           nullptr);
    if (buffer->vulkanMemory)
      driver.freeMemory(buffer->context->vulkanDevice, buffer->vulkanMemory,
                        nullptr);
  }
#endif
  --buffer->context->liveBuffers;
  delete buffer;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyFromHost(VernonDeviceBuffer *buffer,
                                       size_t offset, const void *source,
                                       size_t size) {
  if (!buffer || buffer->context->backend != VERNON_RUNTIME_CPU || !source ||
      offset > buffer->size || size > buffer->size - offset) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (buffer && buffer->context->backend == VERNON_RUNTIME_CUDA && source &&
        offset <= buffer->size && size <= buffer->size - offset)
      return cudaFail(buffer->context,
                      vernon::runtime::cudaDriver().copyHostToDevice(
                          buffer->cudaDevice + offset, source, size),
                      "cuMemcpyHtoD");
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (buffer && buffer->context->backend == VERNON_RUNTIME_VULKAN && source &&
        offset <= buffer->size && size <= buffer->size - offset) {
      void *mapped = nullptr;
      vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
      if (vulkanFail(buffer->context,
                     driver.mapMemory(buffer->context->vulkanDevice,
                                      buffer->vulkanMemory, offset, size, 0,
                                      &mapped),
                     "vkMapMemory") != VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
      std::memcpy(mapped, source, size);
      driver.unmapMemory(buffer->context->vulkanDevice, buffer->vulkanMemory);
      return VERNON_STATUS_OK;
    }
#endif
    return fail(buffer ? buffer->context : nullptr,
                "invalid compute buffer upload");
  }
  std::memcpy(buffer->host.data() + offset, source, size);
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyToHost(const VernonDeviceBuffer *buffer,
                                     size_t offset, void *destination,
                                     size_t size) {
  if (!buffer || buffer->context->backend != VERNON_RUNTIME_CPU ||
      !destination || offset > buffer->size || size > buffer->size - offset) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (buffer && buffer->context->backend == VERNON_RUNTIME_CUDA &&
        destination && offset <= buffer->size && size <= buffer->size - offset)
      return cudaFail(buffer->context,
                      vernon::runtime::cudaDriver().copyDeviceToHost(
                          destination, buffer->cudaDevice + offset, size),
                      "cuMemcpyDtoH");
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (buffer && buffer->context->backend == VERNON_RUNTIME_VULKAN &&
        destination && offset <= buffer->size &&
        size <= buffer->size - offset) {
      void *mapped = nullptr;
      vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
      if (vulkanFail(buffer->context,
                     driver.mapMemory(buffer->context->vulkanDevice,
                                      buffer->vulkanMemory, offset, size, 0,
                                      &mapped),
                     "vkMapMemory") != VERNON_STATUS_OK)
        return VERNON_STATUS_INTERNAL_ERROR;
      std::memcpy(destination, mapped, size);
      driver.unmapMemory(buffer->context->vulkanDevice, buffer->vulkanMemory);
      return VERNON_STATUS_OK;
    }
#endif
    return fail(buffer ? buffer->context : nullptr,
                "invalid compute buffer readback");
  }
  std::memcpy(destination, buffer->host.data() + offset, size);
  return VERNON_STATUS_OK;
}

VernonDeviceTexture *
vernonRuntimeTextureCreate(VernonRuntimeContext *context,
                           const VernonTextureDescriptor *descriptor) {
  if (!context || !descriptor ||
      descriptor->struct_size < sizeof(VernonTextureDescriptor) ||
      !descriptor->width || !descriptor->height || !descriptor->depth ||
      !descriptor->mip_levels)
    return nullptr;
  if (context->backend != VERNON_RUNTIME_VULKAN) {
    fail(context, "owned sampled textures require the Vulkan backend",
         VERNON_STATUS_UNSUPPORTED_TARGET);
    return nullptr;
  }
  if (descriptor->dimension > VERNON_TEXTURE_CUBE ||
      (descriptor->dimension != VERNON_TEXTURE_3D && descriptor->depth != 1) ||
      (descriptor->dimension == VERNON_TEXTURE_CUBE &&
       descriptor->width != descriptor->height)) {
    fail(context, "sampled texture descriptor dimensions are invalid");
    return nullptr;
  }
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  const std::optional<VkFormat> format =
      vulkanTextureFormat(descriptor->format);
  if (!format) {
    fail(context, "sampled texture format is unsupported");
    return nullptr;
  }
  uint32_t maximumMipLevels = 1;
  uint32_t maximumExtent =
      std::max({descriptor->width, descriptor->height, descriptor->depth});
  while (maximumExtent > 1) {
    maximumExtent >>= 1;
    ++maximumMipLevels;
  }
  if (descriptor->mip_levels > maximumMipLevels) {
    fail(context, "sampled texture mip count exceeds its extent");
    return nullptr;
  }
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  VkFormatProperties formatProperties{};
  driver.getPhysicalDeviceFormatProperties(context->vulkanPhysicalDevice,
                                           *format, &formatProperties);
  if (!(formatProperties.optimalTilingFeatures &
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
    fail(context, "Vulkan device cannot sample the requested texture format",
         VERNON_STATUS_UNSUPPORTED_TARGET);
    return nullptr;
  }
  auto texture = std::make_unique<VernonDeviceTexture>();
  texture->context = context;
  texture->width = descriptor->width;
  texture->height = descriptor->height;
  texture->depth = descriptor->depth;
  texture->mipLevels = descriptor->mip_levels;
  texture->dimension = descriptor->dimension;
  texture->format = descriptor->format;
  texture->vulkanFormat = *format;
  texture->vulkanColorAttachment = descriptor->dimension == VERNON_TEXTURE_2D &&
                                   (formatProperties.optimalTilingFeatures &
                                    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
  VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.flags = descriptor->dimension == VERNON_TEXTURE_CUBE
                        ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
                        : 0;
  imageInfo.imageType = descriptor->dimension == VERNON_TEXTURE_3D
                            ? VK_IMAGE_TYPE_3D
                            : VK_IMAGE_TYPE_2D;
  imageInfo.format = *format;
  imageInfo.extent = {
      descriptor->width, descriptor->height,
      descriptor->dimension == VERNON_TEXTURE_3D ? descriptor->depth : 1};
  imageInfo.mipLevels = descriptor->mip_levels;
  imageInfo.arrayLayers = descriptor->dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (texture->vulkanColorAttachment)
    imageInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vulkanFail(context,
                 driver.createImage(context->vulkanDevice, &imageInfo, nullptr,
                                    &texture->vulkanImage),
                 "vkCreateImage") != VERNON_STATUS_OK)
    return nullptr;
  VkMemoryRequirements requirements{};
  driver.getImageMemoryRequirements(context->vulkanDevice, texture->vulkanImage,
                                    &requirements);
  const std::optional<uint32_t> memoryType =
      findVulkanMemoryType(context, requirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (!memoryType) {
    driver.destroyImage(context->vulkanDevice, texture->vulkanImage, nullptr);
    fail(context, "Vulkan device has no device-local image memory");
    return nullptr;
  }
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = *memoryType;
  if (vulkanFail(context,
                 driver.allocateMemory(context->vulkanDevice, &allocation,
                                       nullptr, &texture->vulkanMemory),
                 "vkAllocateMemory") != VERNON_STATUS_OK) {
    driver.destroyImage(context->vulkanDevice, texture->vulkanImage, nullptr);
    return nullptr;
  }
  if (vulkanFail(context,
                 driver.bindImageMemory(context->vulkanDevice,
                                        texture->vulkanImage,
                                        texture->vulkanMemory, 0),
                 "vkBindImageMemory") != VERNON_STATUS_OK) {
    driver.destroyImage(context->vulkanDevice, texture->vulkanImage, nullptr);
    driver.freeMemory(context->vulkanDevice, texture->vulkanMemory, nullptr);
    return nullptr;
  }
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image = texture->vulkanImage;
  viewInfo.viewType =
      descriptor->dimension == VERNON_TEXTURE_3D     ? VK_IMAGE_VIEW_TYPE_3D
      : descriptor->dimension == VERNON_TEXTURE_CUBE ? VK_IMAGE_VIEW_TYPE_CUBE
                                                     : VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = *format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.levelCount = descriptor->mip_levels;
  viewInfo.subresourceRange.layerCount =
      descriptor->dimension == VERNON_TEXTURE_CUBE ? 6 : 1;
  if (vulkanFail(context,
                 driver.createImageView(context->vulkanDevice, &viewInfo,
                                        nullptr, &texture->vulkanView),
                 "vkCreateImageView") != VERNON_STATUS_OK) {
    driver.destroyImage(context->vulkanDevice, texture->vulkanImage, nullptr);
    driver.freeMemory(context->vulkanDevice, texture->vulkanMemory, nullptr);
    return nullptr;
  }
  ++context->liveTextures;
  return texture.release();
#else
  return nullptr;
#endif
}

VernonDeviceTexture *vernonRuntimeTextureCreate2D(VernonRuntimeContext *context,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  VernonTextureFormat format) {
  const VernonTextureDescriptor descriptor{sizeof(VernonTextureDescriptor),
                                           VERNON_TEXTURE_2D,
                                           format,
                                           width,
                                           height,
                                           1,
                                           1,
                                           {0, 0, 0, 0}};
  return vernonRuntimeTextureCreate(context, &descriptor);
}

VernonDeviceTexture *
vernonRuntimeImportOpenGLTexture(VernonRuntimeContext *context,
                                 uint32_t texture,
                                 const VernonTextureDescriptor *descriptor) {
  if (!context ||
      (context->backend != VERNON_RUNTIME_OPENGL &&
       context->backend != VERNON_RUNTIME_OPENGL_ES) ||
      !texture || !descriptor ||
      descriptor->struct_size < sizeof(VernonTextureDescriptor) ||
      !descriptor->width || !descriptor->height || !descriptor->depth ||
      !descriptor->mip_levels ||
      descriptor->format > VERNON_TEXTURE_R11G11B10_FLOAT ||
      descriptor->dimension > VERNON_TEXTURE_CUBE ||
      (descriptor->dimension != VERNON_TEXTURE_3D && descriptor->depth != 1) ||
      (descriptor->dimension == VERNON_TEXTURE_CUBE &&
       descriptor->width != descriptor->height))
    return nullptr;
  auto result = std::make_unique<VernonDeviceTexture>();
  result->context = context;
  result->name = texture;
  result->width = descriptor->width;
  result->height = descriptor->height;
  result->depth = descriptor->depth;
  result->mipLevels = descriptor->mip_levels;
  result->dimension = descriptor->dimension;
  result->format = descriptor->format;
  ++context->liveTextures;
  return result.release();
}

VernonDeviceTexture *vernonRuntimeImportOpenGLTexture2D(
    VernonRuntimeContext *context, uint32_t texture, uint32_t width,
    uint32_t height, VernonTextureFormat format) {
  const VernonTextureDescriptor descriptor{sizeof(VernonTextureDescriptor),
                                           VERNON_TEXTURE_2D,
                                           format,
                                           width,
                                           height,
                                           1,
                                           1,
                                           {0, 0, 0, 0}};
  return vernonRuntimeImportOpenGLTexture(context, texture, &descriptor);
}

VernonStatus vernonRuntimeTextureFree(VernonDeviceTexture *texture) {
  if (!texture)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (texture->context->backend == VERNON_RUNTIME_VULKAN) {
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
    driver.deviceWaitIdle(texture->context->vulkanDevice);
    if (texture->vulkanView)
      driver.destroyImageView(texture->context->vulkanDevice,
                              texture->vulkanView, nullptr);
    if (texture->vulkanImage)
      driver.destroyImage(texture->context->vulkanDevice, texture->vulkanImage,
                          nullptr);
    if (texture->vulkanMemory)
      driver.freeMemory(texture->context->vulkanDevice, texture->vulkanMemory,
                        nullptr);
  }
#endif
  --texture->context->liveTextures;
  delete texture;
  return VERNON_STATUS_OK;
}

VernonDeviceSampler *
vernonRuntimeSamplerCreate(VernonRuntimeContext *context,
                           const VernonSamplerDescriptor *descriptor) {
  if (!context || !descriptor ||
      descriptor->struct_size < sizeof(VernonSamplerDescriptor))
    return nullptr;
  if (context->backend != VERNON_RUNTIME_VULKAN) {
    fail(context,
         "owned sampler creation requires Vulkan; import an OpenGL sampler",
         VERNON_STATUS_UNSUPPORTED_TARGET);
    return nullptr;
  }
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  const auto wrapU = vulkanSamplerAddressMode(descriptor->wrap_u);
  const auto wrapV = vulkanSamplerAddressMode(descriptor->wrap_v);
  const auto wrapW = vulkanSamplerAddressMode(descriptor->wrap_w);
  const auto minFilter = vulkanSamplerFilter(descriptor->min_filter);
  const auto magFilter = vulkanSamplerFilter(descriptor->mag_filter);
  if (!wrapU || !wrapV || !wrapW || !minFilter || !magFilter ||
      descriptor->mip_filter > VERNON_SAMPLER_LINEAR) {
    fail(context, "sampler descriptor contains an invalid enum value");
    return nullptr;
  }
  auto sampler = std::make_unique<VernonDeviceSampler>();
  sampler->context = context;
  sampler->descriptor = *descriptor;
  VkSamplerCreateInfo createInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  createInfo.magFilter = *magFilter;
  createInfo.minFilter = *minFilter;
  createInfo.mipmapMode = descriptor->mip_filter == VERNON_SAMPLER_LINEAR
                              ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                              : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  createInfo.addressModeU = *wrapU;
  createInfo.addressModeV = *wrapV;
  createInfo.addressModeW = *wrapW;
  createInfo.minLod = 0.0f;
  createInfo.maxLod = VK_LOD_CLAMP_NONE;
  createInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  if (vulkanFail(context,
                 vernon::runtime::vulkanDriver().createSampler(
                     context->vulkanDevice, &createInfo, nullptr,
                     &sampler->vulkanSampler),
                 "vkCreateSampler") != VERNON_STATUS_OK)
    return nullptr;
  ++context->liveSamplers;
  return sampler.release();
#else
  return nullptr;
#endif
}

VernonDeviceSampler *
vernonRuntimeImportOpenGLSampler(VernonRuntimeContext *context,
                                 uint32_t sampler) {
  if (!context ||
      (context->backend != VERNON_RUNTIME_OPENGL &&
       context->backend != VERNON_RUNTIME_OPENGL_ES) ||
      !sampler)
    return nullptr;
  auto result = std::make_unique<VernonDeviceSampler>();
  result->context = context;
  result->name = sampler;
  result->imported = true;
  ++context->liveSamplers;
  return result.release();
}

VernonStatus vernonRuntimeSamplerFree(VernonDeviceSampler *sampler) {
  if (!sampler)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (sampler->context->backend == VERNON_RUNTIME_VULKAN &&
      sampler->vulkanSampler) {
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
    driver.deviceWaitIdle(sampler->context->vulkanDevice);
    driver.destroySampler(sampler->context->vulkanDevice,
                          sampler->vulkanSampler, nullptr);
  }
#endif
  --sampler->context->liveSamplers;
  delete sampler;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeTextureCopyFromHost(VernonDeviceTexture *texture,
                                              const void *source, size_t size) {
  const size_t expected =
      texture ? static_cast<size_t>(texture->width) * texture->height * 4 : 0;
  if (!texture || texture->context->backend != VERNON_RUNTIME_VULKAN ||
      texture->dimension != VERNON_TEXTURE_2D ||
      texture->format != VERNON_TEXTURE_RGBA8_UNORM ||
      texture->mipLevels != 1 || !source || size != expected)
    return fail(texture ? texture->context : nullptr,
                "Vulkan host upload supports one-mip RGBA8 2D textures only",
                VERNON_STATUS_UNSUPPORTED_TARGET);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  VernonRuntimeContext *context = texture->context;
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (!createVulkanBuffer(context, size, staging, memory))
    return VERNON_STATUS_INTERNAL_ERROR;
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  void *mapped = nullptr;
  if (vulkanFail(
          context,
          driver.mapMemory(context->vulkanDevice, memory, 0, size, 0, &mapped),
          "vkMapMemory") != VERNON_STATUS_OK) {
    driver.destroyBuffer(context->vulkanDevice, staging, nullptr);
    driver.freeMemory(context->vulkanDevice, memory, nullptr);
    return VERNON_STATUS_INTERNAL_ERROR;
  }
  std::memcpy(mapped, source, size);
  driver.unmapMemory(context->vulkanDevice, memory);
  const bool copied =
      submitVulkanCommands(context, [&](VkCommandBuffer command) {
        transitionVulkanImage(command, texture,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {texture->width, texture->height, 1};
        driver.cmdCopyBufferToImage(command, staging, texture->vulkanImage,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                    &region);
        transitionVulkanImage(command, texture,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      });
  driver.destroyBuffer(context->vulkanDevice, staging, nullptr);
  driver.freeMemory(context->vulkanDevice, memory, nullptr);
  return copied ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#else
  return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus vernonRuntimeTextureCopyToHost(const VernonDeviceTexture *texture,
                                            void *destination, size_t size) {
  const size_t expected =
      texture ? static_cast<size_t>(texture->width) * texture->height * 4 : 0;
  if (!texture || texture->context->backend != VERNON_RUNTIME_VULKAN ||
      texture->dimension != VERNON_TEXTURE_2D ||
      texture->format != VERNON_TEXTURE_RGBA8_UNORM ||
      texture->mipLevels != 1 || !destination || size != expected)
    return fail(texture ? texture->context : nullptr,
                "Vulkan host readback supports one-mip RGBA8 2D textures only",
                VERNON_STATUS_UNSUPPORTED_TARGET);
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  auto *mutableTexture = const_cast<VernonDeviceTexture *>(texture);
  VernonRuntimeContext *context = texture->context;
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (!createVulkanBuffer(context, size, staging, memory))
    return VERNON_STATUS_INTERNAL_ERROR;
  vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
  const VkImageLayout restoreLayout =
      texture->vulkanLayout == VK_IMAGE_LAYOUT_UNDEFINED
          ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
          : texture->vulkanLayout;
  const bool copied =
      submitVulkanCommands(context, [&](VkCommandBuffer command) {
        transitionVulkanImage(command, mutableTexture,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {texture->width, texture->height, 1};
        driver.cmdCopyImageToBuffer(command, texture->vulkanImage,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    staging, 1, &region);
        transitionVulkanImage(command, mutableTexture, restoreLayout);
      });
  if (copied) {
    void *mapped = nullptr;
    if (vulkanFail(context,
                   driver.mapMemory(context->vulkanDevice, memory, 0, size, 0,
                                    &mapped),
                   "vkMapMemory") == VERNON_STATUS_OK) {
      std::memcpy(destination, mapped, size);
      driver.unmapMemory(context->vulkanDevice, memory);
    } else {
      driver.destroyBuffer(context->vulkanDevice, staging, nullptr);
      driver.freeMemory(context->vulkanDevice, memory, nullptr);
      return VERNON_STATUS_INTERNAL_ERROR;
    }
  }
  driver.destroyBuffer(context->vulkanDevice, staging, nullptr);
  driver.freeMemory(context->vulkanDevice, memory, nullptr);
  return copied ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
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
      !reflection || !reflectionSize || !entry || !entrySize)
    return nullptr;
  try {
    const nlohmann::json parsed = nlohmann::json::parse(
        reflection, reflection + reflectionSize, nullptr, false);
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = context;
    kernel->cpuEntry = entryPoint;
    if (parsed.is_discarded() ||
        !parseReflection(parsed, std::string(entry, entrySize),
                         kernel->reflection, context->error))
      return nullptr;
    ++context->liveKernels;
    return kernel.release();
  } catch (const std::exception &error) {
    fail(context, std::string("failed to load CPU entry: ") + error.what(),
         VERNON_STATUS_INTERNAL_ERROR);
    return nullptr;
  }
}

VernonStatus
vernonRuntimeRegisterStaticCpuEntry(VernonStringView symbol,
                                    VernonCpuEntryPoint entryPoint) {
  if (!symbol.data || !symbol.size || !entryPoint)
    return VERNON_STATUS_INVALID_ARGUMENT;
  const std::string name(symbol.data, symbol.size);
  std::lock_guard<std::mutex> lock(staticCpuEntriesMutex());
  auto [found, inserted] = staticCpuEntries().emplace(name, entryPoint);
  return inserted || found->second == entryPoint
             ? VERNON_STATUS_OK
             : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonLoadedKernel *
vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                          size_t artifactSize, const char *reflection,
                          size_t reflectionSize, const char *entry,
                          size_t entrySize) {
  if (!context || !artifact || !artifactSize || !reflection ||
      !reflectionSize || !entry || !entrySize)
    return nullptr;
  if (context->backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    try {
      const nlohmann::json parsed = nlohmann::json::parse(
          reflection, reflection + reflectionSize, nullptr, false);
      auto kernel = std::make_unique<VernonLoadedKernel>();
      kernel->context = context;
      const std::string entryName(entry, entrySize);
      if (parsed.is_discarded() ||
          !parseReflection(parsed, entryName, kernel->reflection,
                           context->error))
        return nullptr;
      std::string image(static_cast<const char *>(artifact), artifactSize);
      image.push_back('\0');
      vernon::runtime::CudaDriver &driver = vernon::runtime::cudaDriver();
      if (cudaFail(context,
                   driver.moduleLoadData(&kernel->cudaModule, image.data(), 0,
                                         nullptr, nullptr),
                   "cuModuleLoadDataEx") != VERNON_STATUS_OK)
        return nullptr;
      if (cudaFail(context,
                   driver.moduleGetFunction(&kernel->cudaFunction,
                                            kernel->cudaModule,
                                            entryName.c_str()),
                   "cuModuleGetFunction") != VERNON_STATUS_OK) {
        driver.moduleUnload(kernel->cudaModule);
        return nullptr;
      }
      ++context->liveKernels;
      return kernel.release();
    } catch (const std::exception &error) {
      fail(context,
           std::string("failed to load CUDA artifact: ") + error.what(),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
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
    try {
      const nlohmann::json parsed = nlohmann::json::parse(
          reflection, reflection + reflectionSize, nullptr, false);
      auto kernel = std::make_unique<VernonLoadedKernel>();
      kernel->context = context;
      const std::string entryName(entry, entrySize);
      if (parsed.is_discarded() ||
          !parseReflection(parsed, entryName, kernel->reflection,
                           context->error))
        return nullptr;
      vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
      VkShaderModuleCreateInfo shaderInfo{
          VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      shaderInfo.codeSize = artifactSize;
      shaderInfo.pCode = static_cast<const uint32_t *>(artifact);
      if (vulkanFail(context,
                     driver.createShaderModule(context->vulkanDevice,
                                               &shaderInfo, nullptr,
                                               &kernel->vulkanShader),
                     "vkCreateShaderModule") != VERNON_STATUS_OK)
        return nullptr;
      std::vector<VkDescriptorSetLayoutBinding> bindings;
      uint32_t index = 0;
      for (ReflectedArgument &argument : kernel->reflection.arguments) {
        if (argument.kind == "builtin")
          continue;
        if (argument.descriptorSet != 0) {
          fail(context, "Vulkan compute supports descriptor set zero only");
          return nullptr;
        }
        if (argument.binding == UINT32_MAX)
          argument.binding = index;
        bindings.push_back({argument.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        ++index;
      }
      VkDescriptorSetLayoutCreateInfo descriptorInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      descriptorInfo.bindingCount = static_cast<uint32_t>(bindings.size());
      descriptorInfo.pBindings = bindings.data();
      if (vulkanFail(context,
                     driver.createDescriptorSetLayout(
                         context->vulkanDevice, &descriptorInfo, nullptr,
                         &kernel->vulkanDescriptorSetLayout),
                     "vkCreateDescriptorSetLayout") != VERNON_STATUS_OK)
        return nullptr;
      VkPipelineLayoutCreateInfo layoutInfo{
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layoutInfo.setLayoutCount = 1;
      layoutInfo.pSetLayouts = &kernel->vulkanDescriptorSetLayout;
      if (vulkanFail(context,
                     driver.createPipelineLayout(context->vulkanDevice,
                                                 &layoutInfo, nullptr,
                                                 &kernel->vulkanPipelineLayout),
                     "vkCreatePipelineLayout") != VERNON_STATUS_OK)
        return nullptr;
      VkPipelineShaderStageCreateInfo stage{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stage.module = kernel->vulkanShader;
      stage.pName = entryName.c_str();
      VkComputePipelineCreateInfo pipelineInfo{
          VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipelineInfo.stage = stage;
      pipelineInfo.layout = kernel->vulkanPipelineLayout;
      if (vulkanFail(context,
                     driver.createComputePipelines(
                         context->vulkanDevice, VK_NULL_HANDLE, 1,
                         &pipelineInfo, nullptr, &kernel->vulkanPipeline),
                     "vkCreateComputePipelines") != VERNON_STATUS_OK)
        return nullptr;
      ++context->liveKernels;
      return kernel.release();
    } catch (const std::exception &error) {
      fail(context,
           std::string("failed to load Vulkan artifact: ") + error.what(),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
#else
    return nullptr;
#endif
  }
  fail(context,
       "CPU AOT artifacts must be loaded from a compute bundle so the native "
       "library target and content hash can be validated",
       VERNON_STATUS_UNSUPPORTED_TARGET);
  return nullptr;
}

VernonLoadedKernel *
vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context,
                               const char *directory) {
  if (!context || context->backend != VERNON_RUNTIME_CPU || !directory)
    return nullptr;
  try {
    const std::filesystem::path root = std::filesystem::u8path(directory);
    const std::string manifestText = readFile(root / "compute.json");
    const nlohmann::json manifest =
        nlohmann::json::parse(manifestText, nullptr, false);
    if (manifest.is_discarded() || !manifest.is_object() ||
        manifest.value("schema_version", 0) != 2 ||
        manifest.value("target", "") != "cpu" ||
        manifest.value("artifact_format", "") != "native_library" ||
        !manifest.contains("reflection")) {
      fail(context, "unsupported or invalid CPU AOT bundle");
      return nullptr;
    }
    CpuNativeArtifact artifact;
    artifact.root = root;
    artifact.relativeLibrary =
        std::filesystem::u8path(manifest.value("artifact", ""));
    artifact.entry = manifest.value("entry", "");
    artifact.symbol = manifest.value("symbol", "");
    artifact.operatingSystem = manifest.value("operating_system", "");
    artifact.architecture = manifest.value("architecture", "");
    artifact.invocationAbiVersion =
        manifest.value("cpu_invocation_abi_version", 0u);
    artifact.size = manifest.value("artifact_size", uint64_t{0});
    artifact.sha256 = manifest.value("artifact_sha256", "");
    artifact.reflection = manifest["reflection"];
    return loadCpuNativeArtifact(context, artifact);
  } catch (const std::exception &error) {
    fail(context, std::string("failed to load CPU AOT bundle: ") + error.what(),
         VERNON_STATUS_INTERNAL_ERROR);
    return nullptr;
  }
}

VernonStatus vernonRuntimeKernelUnload(VernonLoadedKernel *kernel) {
  if (!kernel)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_CUDA && kernel->cudaModule &&
      cudaFail(kernel->context,
               vernon::runtime::cudaDriver().moduleUnload(kernel->cudaModule),
               "cuModuleUnload") != VERNON_STATUS_OK)
    return VERNON_STATUS_INTERNAL_ERROR;
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_VULKAN) {
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
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
                "compute launch grid dimensions must be positive");
  const size_t expected = static_cast<size_t>(std::count_if(
      kernel->reflection.arguments.begin(), kernel->reflection.arguments.end(),
      [](const ReflectedArgument &argument) {
        return argument.kind != "builtin";
      }));
  if (argumentCount != expected || (expected && !arguments))
    return fail(kernel->context,
                "compute launch argument count does not match reflection");

  size_t validated = 0;
  for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
    if (reflected.kind == "builtin")
      continue;
    const VernonLaunchArgument &argument = arguments[validated++];
    if (reflected.kind == "tensor") {
      if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer ||
          argument.buffer->context != kernel->context ||
          (reflected.tensorBytes &&
           argument.buffer->size < reflected.tensorBytes) ||
          argument.buffer->alignment < reflected.alignment)
        return fail(kernel->context,
                    "compute Tensor argument does not match reflection");
    } else if (argument.kind != VERNON_LAUNCH_SCALAR || !argument.scalar_data ||
               argument.scalar_size != reflected.cpuSize) {
      return fail(kernel->context,
                  "compute scalar argument does not match reflection");
    }
  }

#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_CUDA) {
    struct CudaMemRefDescriptor {
      vernon::runtime::CudaDevicePointer allocated;
      vernon::runtime::CudaDevicePointer aligned;
      uint64_t offset;
      uint64_t size;
      uint64_t stride;
    };
    std::vector<CudaMemRefDescriptor> descriptors;
    std::vector<void *> parameters;
    descriptors.reserve(argumentCount);
    parameters.reserve(argumentCount * 5);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
      if (reflected.kind == "builtin")
        continue;
      const VernonLaunchArgument &argument = arguments[supplied++];
      if (argument.kind == VERNON_LAUNCH_TENSOR) {
        descriptors.push_back(
            {argument.buffer->cudaDevice, argument.buffer->cudaDevice, 0,
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
        parameters.push_back(const_cast<void *>(argument.scalar_data));
      }
    }
    const uint32_t *workgroup = kernel->reflection.workgroup;
    return cudaFail(kernel->context,
                    vernon::runtime::cudaDriver().launchKernel(
                        kernel->cudaFunction,
                        (globalSize.x + workgroup[0] - 1) / workgroup[0],
                        (globalSize.y + workgroup[1] - 1) / workgroup[1],
                        (globalSize.z + workgroup[2] - 1) / workgroup[2],
                        workgroup[0], workgroup[1], workgroup[2], 0, nullptr,
                        parameters.data(), nullptr),
                    "cuLaunchKernel");
  }
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_VULKAN) {
    VernonRuntimeContext *context = kernel->context;
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
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
      for (const auto &[buffer, memory] : scalarBuffers) {
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
    const uint32_t *workgroup = kernel->reflection.workgroup;
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
    const bool submitted =
        vulkanFail(context,
                   driver.queueSubmit(context->vulkanQueue, 1, &submit,
                                      VK_NULL_HANDLE),
                   "vkQueueSubmit") == VERNON_STATUS_OK &&
        vulkanFail(context, driver.queueWaitIdle(context->vulkanQueue),
                   "vkQueueWaitIdle") == VERNON_STATUS_OK;
    cleanup();
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
  }
#endif
  if (kernel->context->backend != VERNON_RUNTIME_CPU)
    return fail(kernel->context, "compute backend is unsupported",
                VERNON_STATUS_UNSUPPORTED_TARGET);

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
        return fail(kernel->context, "invalid CPU Tensor launch argument");
      const uintptr_t pointer =
          reinterpret_cast<uintptr_t>(argument.buffer->host.data());
      std::memcpy(packed.data() + reflected.cpuOffset, &pointer,
                  sizeof(pointer));
    } else {
      if (argument.kind != VERNON_LAUNCH_SCALAR || !argument.scalar_data ||
          argument.scalar_size != reflected.cpuSize)
        return fail(kernel->context, "invalid CPU scalar launch argument");
      std::memcpy(packed.data() + reflected.cpuOffset, argument.scalar_data,
                  argument.scalar_size);
    }
  }
  for (uint32_t z = 0; z < globalSize.z; ++z)
    for (uint32_t y = 0; y < globalSize.y; ++y)
      for (uint32_t x = 0; x < globalSize.x; ++x) {
        const uint32_t id[3]{x, y, z};
        for (const ReflectedArgument &reflected : kernel->reflection.arguments)
          if (reflected.kind == "builtin" &&
              reflected.builtin == "global_invocation_id")
            std::memcpy(packed.data() + reflected.cpuOffset, id,
                        std::min(reflected.cpuSize, sizeof(id)));
        VernonCpuInvocation invocation{packed.data(), packed.size(), nullptr, 0,
                                       nullptr};
        const VernonStatus status = kernel->cpuEntry(&invocation);
        if (status != VERNON_STATUS_OK)
          return fail(kernel->context, "CPU AOT entry invocation failed",
                      status);
      }
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimePipelineBundleInspectTarget(
    const void *bundleData, size_t bundleSize, VernonRuntimeBackend *target) {
  if (!bundleData || !bundleSize || !target)
    return VERNON_STATUS_INVALID_ARGUMENT;
  try {
    const nlohmann::json root = nlohmann::json::parse(
        static_cast<const char *>(bundleData),
        static_cast<const char *>(bundleData) + bundleSize, nullptr, false);
    if (root.is_discarded() || !root.is_object())
      return VERNON_STATUS_PARSE_ERROR;
    const bool legacy = root.value("pipeline_bundle_schema_version", 0) == 1 &&
                        root.value("type", "") == "vernon_pipeline_bundle";
    const bool schema2 = root.value("schema_version", 0) == 2 &&
                         root.value("type", "") == "pipeline";
    if (!legacy && !schema2)
      return VERNON_STATUS_PARSE_ERROR;
    const std::string name = root.value("target", "");
    if (name == "cpu")
      *target = VERNON_RUNTIME_CPU;
    else if (name == "cuda")
      *target = VERNON_RUNTIME_CUDA;
    else if (name == "vulkan")
      *target = VERNON_RUNTIME_VULKAN;
    else if (name == "opengl")
      *target = VERNON_RUNTIME_OPENGL;
    else if (name == "opengles")
      *target = VERNON_RUNTIME_OPENGL_ES;
    else
      return VERNON_STATUS_UNSUPPORTED_TARGET;
    return VERNON_STATUS_OK;
  } catch (...) {
    return VERNON_STATUS_PARSE_ERROR;
  }
}

VernonPipelineBundle *
vernonRuntimeLoadPipelineBundle(VernonRuntimeContext *context,
                                const void *bundleData, size_t bundleSize) {
  return vernonRuntimeLoadPipelineBundleWithOptions(context, bundleData,
                                                    bundleSize, nullptr);
}

VernonPipelineBundle *vernonRuntimeLoadPipelineBundleWithOptions(
    VernonRuntimeContext *context, const void *bundleData, size_t bundleSize,
    const VernonPipelineBundleLoadOptions *options) {
  if (!context ||
      (context->backend != VERNON_RUNTIME_CPU &&
       context->backend != VERNON_RUNTIME_OPENGL &&
       context->backend != VERNON_RUNTIME_OPENGL_ES &&
       context->backend != VERNON_RUNTIME_VULKAN &&
       context->backend != VERNON_RUNTIME_CUDA) ||
      !bundleData || !bundleSize ||
      (options &&
       options->struct_size < sizeof(VernonPipelineBundleLoadOptions)))
    return nullptr;
  try {
    std::optional<std::filesystem::path> bundleDirectory;
    if (options && options->bundle_directory &&
        options->bundle_directory[0] != '\0')
      bundleDirectory = std::filesystem::u8path(options->bundle_directory);
    const nlohmann::json root = nlohmann::json::parse(
        static_cast<const char *>(bundleData),
        static_cast<const char *>(bundleData) + bundleSize);
    const char *expectedTarget =
        context->backend == VERNON_RUNTIME_CPU    ? "cpu"
        : context->backend == VERNON_RUNTIME_CUDA ? "cuda"
        : context->backend == VERNON_RUNTIME_VULKAN
            ? "vulkan"
            : (context->backend == VERNON_RUNTIME_OPENGL_ES ? "opengles"
                                                            : "opengl");
    const bool legacy = root.is_object() &&
                        root.value("pipeline_bundle_schema_version", 0) == 1 &&
                        root.value("type", "") == "vernon_pipeline_bundle";
    const bool schema2 = root.is_object() &&
                         root.value("schema_version", 0) == 2 &&
                         root.value("type", "") == "pipeline";
    if ((!legacy && !schema2) ||
        root.value("invocation_abi_version", 0) !=
            VERNON_PIPELINE_INVOCATION_ABI_VERSION ||
        root.value("target", "") != expectedTarget ||
        !root.contains("stage_artifacts") ||
        !root["stage_artifacts"].is_object() || !root.contains("variants") ||
        !root["variants"].is_array()) {
      fail(context, "unsupported or invalid pipeline bundle");
      return nullptr;
    }
    if (!validateManifestHash(root, schema2, context->error))
      return nullptr;
    auto bundle = std::make_unique<VernonPipelineBundle>();
    bundle->context = context;
    bundle->id = root.value("id", "");
    for (const nlohmann::json &feature :
         root.value("features", nlohmann::json::array()))
      bundle->features.push_back(feature.get<std::string>());
    if (schema2 &&
        (!std::is_sorted(bundle->features.begin(), bundle->features.end()) ||
         std::adjacent_find(bundle->features.begin(), bundle->features.end()) !=
             bundle->features.end())) {
      fail(context, "pipeline feature table is not canonical");
      return nullptr;
    }
    for (const auto &[id, value] : root["stage_artifacts"].items()) {
      if (!value.is_object() ||
          (value.contains("id") && (!value["id"].is_string() ||
                                    value["id"].get<std::string>() != id))) {
        fail(context, "pipeline stage id is invalid");
        return nullptr;
      }
      Stage stage;
      stage.stage = value.value("stage", "");
      stage.entry = value.value("entry", "");
      if (legacy)
        stage.source = value.value("source", "");
      if (value.contains("reflection") &&
          value["reflection"].contains("entries")) {
        stage.reflection = value["reflection"].dump();
        for (const nlohmann::json &entry : value["reflection"]["entries"]) {
          if (entry.value("name", "") != stage.entry)
            continue;
          const auto workgroup =
              entry.value("workgroup_size", nlohmann::json::array());
          if (workgroup.size() == 3)
            for (size_t index = 0; index < 3; ++index) {
              stage.workgroup[index] = workgroup[index].get<uint32_t>();
              if (!stage.workgroup[index]) {
                fail(context, "compute workgroup dimensions must be non-zero");
                return nullptr;
              }
            }
        }
      }
      std::optional<ResolvedArtifact> resolved;
      if (schema2) {
        if (!value.contains("artifact")) {
          fail(context, "pipeline stage artifact descriptor is missing");
          return nullptr;
        }
        ResolvedArtifact artifact;
        if (!resolveArtifact(context, value["artifact"], bundleDirectory,
                             artifact))
          return nullptr;
        const char *expectedFormat =
            context->backend == VERNON_RUNTIME_CUDA        ? "ptx"
            : context->backend == VERNON_RUNTIME_VULKAN    ? "spirv"
            : context->backend == VERNON_RUNTIME_OPENGL_ES ? "gles"
                                                           : "glsl";
        const std::string encoding = value["artifact"].value("encoding", "");
        const bool validCpuFormat = context->backend == VERNON_RUNTIME_CPU &&
                                    (artifact.format == "native_library" ||
                                     artifact.format == "relocatable_object");
        const bool validFormat = context->backend == VERNON_RUNTIME_CPU
                                     ? validCpuFormat
                                     : artifact.format == expectedFormat;
        if (!validFormat ||
            (artifact.external && value["artifact"].contains("encoding")) ||
            (!artifact.external &&
             ((artifact.format == "spirv" && encoding != "base64") ||
              (artifact.format != "spirv" && encoding != "utf8"))) ||
            value.value("target", "") != expectedTarget) {
          fail(context, "pipeline stage artifact format is invalid for target");
          return nullptr;
        }
        resolved = std::move(artifact);
      }
      if (context->backend == VERNON_RUNTIME_CPU) {
        if (!bundleDirectory ||
            (schema2 && (!resolved || !resolved->external))) {
          fail(context,
               "CPU pipeline artifacts require an external bundle directory");
          return nullptr;
        }
        const nlohmann::json &nativeArtifact =
            value.contains("artifact") && value["artifact"].is_object()
                ? value["artifact"]
                : value;
        CpuNativeArtifact artifact;
        artifact.root = *bundleDirectory;
        artifact.relativeLibrary = std::filesystem::u8path(
            nativeArtifact.value("path", value.value("native_library", "")));
        artifact.entry = stage.entry;
        artifact.format =
            schema2 ? resolved->format : value.value("format", "");
        artifact.symbol = value.value("symbol", "");
        artifact.operatingSystem = value.value("operating_system", "");
        artifact.architecture = value.value("architecture", "");
        artifact.targetTriple = value.value("target_triple", "");
        artifact.objectFormat = value.value("object_format", "");
        artifact.invocationAbiVersion =
            value.value("cpu_invocation_abi_version", 0u);
        artifact.size = nativeArtifact.value("size", uint64_t{0});
        artifact.sha256 = nativeArtifact.value("sha256", "");
        if (value.contains("reflection"))
          artifact.reflection = value["reflection"];
        std::filesystem::path validatedPath;
        const std::string format =
            schema2 ? resolved->format : value.value("format", "");
        if ((format != "native_library" && format != "relocatable_object") ||
            !resolveCpuNativeArtifact(context, artifact, validatedPath,
                                      nullptr))
          return nullptr;
        stage.cpuArtifact = std::move(artifact);
      }
      if (schema2 && context->backend == VERNON_RUNTIME_VULKAN)
        stage.binary = std::move(resolved->bytes);
      else if (schema2 && context->backend != VERNON_RUNTIME_CPU)
        stage.source.assign(resolved->bytes.begin(), resolved->bytes.end());
      if (legacy && context->backend == VERNON_RUNTIME_VULKAN &&
          value.contains("artifact") && value["artifact"].is_object()) {
        const nlohmann::json &artifact = value["artifact"];
        if (artifact.value("format", "") == "spirv" &&
            artifact.value("encoding", "") == "base64") {
          const auto decoded = decodeBase64Strict(artifact.value("data", ""));
          if (decoded)
            stage.binary = *decoded;
          if (!decoded || vernon::runtime::sha256Hex(stage.binary.data(),
                                                     stage.binary.size()) !=
                              artifact.value("sha256", ""))
            stage.binary.clear();
        }
      }
      const bool hasArtifact =
          context->backend == VERNON_RUNTIME_CPU ? stage.cpuArtifact.has_value()
          : context->backend == VERNON_RUNTIME_VULKAN
              ? !stage.binary.empty() &&
                    stage.binary.size() % sizeof(uint32_t) == 0 &&
                    !stage.reflection.empty()
          : context->backend == VERNON_RUNTIME_CUDA
              ? (schema2 || value.value("format", "") == "ptx") &&
                    !stage.source.empty() && !stage.reflection.empty()
              : !stage.source.empty();
      if (stage.stage.empty() || stage.entry.empty() || !hasArtifact) {
        fail(context, "pipeline stage artifact is invalid");
        return nullptr;
      }
      bundle->stages.emplace(id, std::move(stage));
    }
    for (const nlohmann::json &value : root["variants"]) {
      Variant variant;
      if (!parseVariant(value, variant, context->error))
        return nullptr;
      auto validStage = [&](const std::string &id, const char *kind) {
        if (id.empty())
          return true;
        const auto found = bundle->stages.find(id);
        return found != bundle->stages.end() && found->second.stage == kind;
      };
      if (!validStage(variant.compute, "compute") ||
          !validStage(variant.vertex, "vertex") ||
          !validStage(variant.fragment, "fragment")) {
        fail(context, "pipeline variant references an invalid stage");
        return nullptr;
      }
      if ((context->backend == VERNON_RUNTIME_CPU ||
           context->backend == VERNON_RUNTIME_CUDA) &&
          (variant.compute.empty() || !variant.vertex.empty() ||
           variant.barrier)) {
        fail(context,
             std::string(expectedTarget) +
                 " pipeline bundles support dispatch steps only",
             VERNON_STATUS_UNSUPPORTED_TARGET);
        return nullptr;
      }
      bundle->variants.push_back(std::move(variant));
    }
    std::sort(bundle->variants.begin(), bundle->variants.end(),
              [](const Variant &left, const Variant &right) {
                return left.key < right.key;
              });
    if (std::adjacent_find(bundle->variants.begin(), bundle->variants.end(),
                           [](const Variant &left, const Variant &right) {
                             return left.key == right.key;
                           }) != bundle->variants.end()) {
      fail(context, "pipeline bundle contains duplicate feature variants");
      return nullptr;
    }
    if (bundle->id.empty()) {
      fail(context, "pipeline bundle id is missing or empty");
      return nullptr;
    }
    if (bundle->variants.empty()) {
      fail(context, "pipeline bundle contains no variants");
      return nullptr;
    }
    ++context->liveBundles;
    return bundle.release();
  } catch (const nlohmann::json::exception &error) {
    fail(context, std::string("invalid pipeline bundle: ") + error.what());
    return nullptr;
  } catch (const std::exception &error) {
    fail(context,
         std::string("failed to load pipeline bundle: ") + error.what(),
         VERNON_STATUS_INTERNAL_ERROR);
    return nullptr;
  } catch (...) {
    fail(context, "failed to load pipeline bundle",
         VERNON_STATUS_INTERNAL_ERROR);
    return nullptr;
  }
}

VernonPipelineBundle *
vernonRuntimeLoadPipelineBundleFromDirectory(VernonRuntimeContext *context,
                                             const char *directory) {
  if (!context || !directory || !directory[0])
    return nullptr;
  try {
    const std::filesystem::path root = std::filesystem::u8path(directory);
    const std::string bundle = readFile(root / "pipeline.bundle");
    if (bundle.empty()) {
      fail(context, "pipeline.bundle is empty or cannot be read");
      return nullptr;
    }
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory;
    return vernonRuntimeLoadPipelineBundleWithOptions(context, bundle.data(),
                                                      bundle.size(), &options);
  } catch (const std::exception &error) {
    fail(context,
         std::string("failed to load pipeline bundle directory: ") +
             error.what(),
         VERNON_STATUS_INTERNAL_ERROR);
    return nullptr;
  }
}

VernonStringView
vernonRuntimePipelineBundleGetId(const VernonPipelineBundle *bundle) {
  return bundle ? VernonStringView{bundle->id.data(), bundle->id.size()}
                : VernonStringView{nullptr, 0};
}

namespace {

bool fillParameterView(const Parameter &source,
                       VernonPipelineParameterView &destination) {
  const auto kind = pipelineArgumentKind(source.kind);
  const auto dtype = source.kind == "sampler"
                         ? std::optional<VernonDataType>(VERNON_DATA_F32)
                         : pipelineDataType(source.dtype);
  const auto access = pipelineValueAccess(source.access);
  if (!kind || !dtype || !access)
    return false;
  destination = {source.slot,
                 {source.name.data(), source.name.size()},
                 *kind,
                 *dtype,
                 *access,
                 static_cast<uint32_t>(source.shape.size()),
                 source.shape.empty() ? nullptr : source.shape.data()};
  return true;
}

bool fillOutputView(const Output &source,
                    VernonPipelineOutputView &destination) {
  const auto kind = pipelineArgumentKind(source.kind);
  const auto dtype = pipelineDataType(source.dtype);
  const auto access = pipelineValueAccess(source.access);
  if (!kind || !dtype || !access)
    return false;
  destination = {{source.name.data(), source.name.size()},
                 *kind,
                 *dtype,
                 *access,
                 static_cast<uint32_t>(source.shape.size()),
                 source.shape.empty() ? nullptr : source.shape.data(),
                 source.location};
  return true;
}

bool stringViewEquals(VernonStringView view, const std::string &value) {
  return view.size == value.size() &&
         (!view.size || std::memcmp(view.data, value.data(), view.size) == 0);
}

} // namespace

void vernonRuntimePipelineBundleDestroy(VernonPipelineBundle *bundle) {
  if (!bundle)
    return;
  --bundle->context->liveBundles;
  delete bundle;
}

VernonLoadedPipeline *
vernonRuntimeResolvePipeline(VernonPipelineBundle *bundle,
                             VernonFeatureSetView features) {
  if (!bundle || (features.count && !features.names))
    return nullptr;
  std::vector<std::string> key;
  for (size_t index = 0; index < features.count; ++index) {
    if (!features.names[index])
      return nullptr;
    key.emplace_back(features.names[index]);
  }
  std::sort(key.begin(), key.end());
  const auto found =
      std::find_if(bundle->variants.begin(), bundle->variants.end(),
                   [&](const Variant &variant) { return variant.key == key; });
  if (found == bundle->variants.end()) {
    fail(bundle->context, "pipeline bundle has no exact feature variant");
    return nullptr;
  }
  auto pipeline = std::make_unique<VernonLoadedPipeline>();
  pipeline->context = bundle->context;
  pipeline->variant = *found;
  if (bundle->context->backend == VERNON_RUNTIME_CPU) {
    const Stage &stage = bundle->stages.at(found->compute);
    pipeline->computeKernel =
        loadCpuNativeArtifact(bundle->context, *stage.cpuArtifact);
    if (!pipeline->computeKernel)
      return nullptr;
    ++bundle->context->livePipelines;
    return pipeline.release();
  }
  if (bundle->context->backend == VERNON_RUNTIME_CUDA) {
    const Stage &stage = bundle->stages.at(found->compute);
    pipeline->computeKernel = vernonRuntimeLoadArtifact(
        bundle->context, stage.source.data(), stage.source.size(),
        stage.reflection.data(), stage.reflection.size(), stage.entry.data(),
        stage.entry.size());
    if (!pipeline->computeKernel)
      return nullptr;
    ++bundle->context->livePipelines;
    return pipeline.release();
  }
  if (bundle->context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!found->compute.empty()) {
      const Stage &stage = bundle->stages.at(found->compute);
      pipeline->computeKernel = vernonRuntimeLoadArtifact(
          bundle->context, stage.binary.data(), stage.binary.size(),
          stage.reflection.data(), stage.reflection.size(), stage.entry.data(),
          stage.entry.size());
      if (!pipeline->computeKernel)
        return nullptr;
    }
    if (!found->vertex.empty()) {
      const Stage &vertex = bundle->stages.at(found->vertex);
      const Stage &fragment = bundle->stages.at(found->fragment);
      vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
      auto createModule = [&](const Stage &stage, VkShaderModule &module) {
        VkShaderModuleCreateInfo info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = stage.binary.size();
        info.pCode = reinterpret_cast<const uint32_t *>(stage.binary.data());
        return vulkanFail(
                   bundle->context,
                   driver.createShaderModule(bundle->context->vulkanDevice,
                                             &info, nullptr, &module),
                   "vkCreateShaderModule") == VERNON_STATUS_OK;
      };
      if (!createModule(vertex, pipeline->vulkanVertex) ||
          !createModule(fragment, pipeline->vulkanFragment)) {
        if (pipeline->computeKernel)
          vernonRuntimeKernelUnload(pipeline->computeKernel);
        if (pipeline->vulkanVertex)
          driver.destroyShaderModule(bundle->context->vulkanDevice,
                                     pipeline->vulkanVertex, nullptr);
        return nullptr;
      }
      pipeline->vulkanVertexEntry = vertex.entry;
      pipeline->vulkanFragmentEntry = fragment.entry;
    }
    ++bundle->context->livePipelines;
    return pipeline.release();
#else
    return nullptr;
#endif
  }
  makeCurrent(bundle->context);
  if (!found->compute.empty()) {
    const Stage &stage = bundle->stages.at(found->compute);
    std::copy(std::begin(stage.workgroup), std::end(stage.workgroup),
              std::begin(pipeline->workgroup));
    const GlUint shader =
        compileShader(bundle->context, kComputeShader, stage.source);
    if (!shader)
      return nullptr;
    pipeline->computeProgram = linkProgram(bundle->context, {shader});
    if (!pipeline->computeProgram)
      return nullptr;
  }
  if (!found->vertex.empty()) {
    const GlUint vertex =
        compileShader(bundle->context, kVertexShader,
                      bundle->stages.at(found->vertex).source);
    const GlUint fragment =
        compileShader(bundle->context, kFragmentShader,
                      bundle->stages.at(found->fragment).source);
    if (!vertex || !fragment) {
      if (vertex)
        bundle->context->gl.deleteShader(vertex);
      if (fragment)
        bundle->context->gl.deleteShader(fragment);
      return nullptr;
    }
    pipeline->graphicsProgram =
        linkProgram(bundle->context, {vertex, fragment});
    if (!pipeline->graphicsProgram)
      return nullptr;
    bundle->context->gl.genVertexArrays(1, &pipeline->vertexArray);
    bundle->context->gl.genFramebuffers(1, &pipeline->framebuffer);
  }
  ++bundle->context->livePipelines;
  return pipeline.release();
}

size_t vernonRuntimeLoadedPipelineGetParameterCount(
    const VernonLoadedPipeline *pipeline) {
  return pipeline ? pipeline->variant.parameters.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetParameterByIndex(
    const VernonLoadedPipeline *pipeline, size_t index,
    VernonPipelineParameterView *parameter) {
  if (!pipeline || !parameter || index >= pipeline->variant.parameters.size())
    return VERNON_STATUS_INVALID_ARGUMENT;
  return fillParameterView(pipeline->variant.parameters[index], *parameter)
             ? VERNON_STATUS_OK
             : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus vernonRuntimeLoadedPipelineFindParameter(
    const VernonLoadedPipeline *pipeline, VernonStringView name,
    VernonPipelineParameterView *parameter) {
  if (!pipeline || !parameter || (name.size && !name.data))
    return VERNON_STATUS_INVALID_ARGUMENT;
  const auto found = std::find_if(
      pipeline->variant.parameters.begin(), pipeline->variant.parameters.end(),
      [&](const Parameter &p) { return stringViewEquals(name, p.name); });
  if (found == pipeline->variant.parameters.end())
    return VERNON_STATUS_INVALID_ARGUMENT;
  return fillParameterView(*found, *parameter) ? VERNON_STATUS_OK
                                               : VERNON_STATUS_PARSE_ERROR;
}

size_t vernonRuntimeLoadedPipelineGetOutputCount(
    const VernonLoadedPipeline *pipeline) {
  return pipeline ? pipeline->variant.outputs.size() : 0;
}

VernonStatus vernonRuntimeLoadedPipelineGetOutputByIndex(
    const VernonLoadedPipeline *pipeline, size_t index,
    VernonPipelineOutputView *output) {
  if (!pipeline || !output || index >= pipeline->variant.outputs.size())
    return VERNON_STATUS_INVALID_ARGUMENT;
  return fillOutputView(pipeline->variant.outputs[index], *output)
             ? VERNON_STATUS_OK
             : VERNON_STATUS_PARSE_ERROR;
}

VernonStatus
vernonRuntimeLoadedPipelineFindOutput(const VernonLoadedPipeline *pipeline,
                                      VernonStringView name,
                                      VernonPipelineOutputView *output) {
  if (!pipeline || !output || (name.size && !name.data))
    return VERNON_STATUS_INVALID_ARGUMENT;
  const auto found = std::find_if(
      pipeline->variant.outputs.begin(), pipeline->variant.outputs.end(),
      [&](const Output &value) { return stringViewEquals(name, value.name); });
  if (found == pipeline->variant.outputs.end())
    return VERNON_STATUS_INVALID_ARGUMENT;
  return fillOutputView(*found, *output) ? VERNON_STATUS_OK
                                         : VERNON_STATUS_PARSE_ERROR;
}

void vernonRuntimeLoadedPipelineDestroy(VernonLoadedPipeline *pipeline) {
  if (!pipeline)
    return;
  if (pipeline->context->backend == VERNON_RUNTIME_CPU ||
      pipeline->context->backend == VERNON_RUNTIME_CUDA) {
    if (pipeline->computeKernel)
      vernonRuntimeKernelUnload(pipeline->computeKernel);
    --pipeline->context->livePipelines;
    delete pipeline;
    return;
  }
  if (pipeline->context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
    driver.deviceWaitIdle(pipeline->context->vulkanDevice);
    if (pipeline->computeKernel)
      vernonRuntimeKernelUnload(pipeline->computeKernel);
    if (pipeline->vulkanVertex)
      driver.destroyShaderModule(pipeline->context->vulkanDevice,
                                 pipeline->vulkanVertex, nullptr);
    if (pipeline->vulkanFragment)
      driver.destroyShaderModule(pipeline->context->vulkanDevice,
                                 pipeline->vulkanFragment, nullptr);
#endif
    --pipeline->context->livePipelines;
    delete pipeline;
    return;
  }
  makeCurrent(pipeline->context);
  OpenGLDriver &gl = pipeline->context->gl;
  if (pipeline->framebuffer)
    gl.deleteFramebuffers(1, &pipeline->framebuffer);
  if (pipeline->vertexArray)
    gl.deleteVertexArrays(1, &pipeline->vertexArray);
  if (pipeline->graphicsProgram)
    gl.deleteProgram(pipeline->graphicsProgram);
  if (pipeline->computeProgram)
    gl.deleteProgram(pipeline->computeProgram);
  --pipeline->context->livePipelines;
  delete pipeline;
}

VernonStatus
vernonRuntimePipelineInvoke(VernonLoadedPipeline *pipeline,
                            const VernonPipelineInvocation *invocation) {
  if (!pipeline || !invocation ||
      invocation->struct_size < sizeof(VernonPipelineInvocation) ||
      invocation->abi_version != VERNON_PIPELINE_INVOCATION_ABI_VERSION ||
      (invocation->argument_count && !invocation->arguments))
    return fail(pipeline ? pipeline->context : nullptr,
                "invalid pipeline invocation");
  PipelineArgumentMap arguments;
  for (size_t index = 0; index < invocation->argument_count; ++index)
    if (!arguments
             .emplace(invocation->arguments[index].slot,
                      &invocation->arguments[index])
             .second)
      return fail(pipeline->context, "duplicate pipeline argument slot");
  if (arguments.size() != pipeline->variant.parameters.size())
    return fail(pipeline->context,
                "pipeline argument count does not match layout");
  for (const Parameter &parameter : pipeline->variant.parameters) {
    const auto found = arguments.find(parameter.slot);
    if (found == arguments.end() ||
        !argumentKindMatches(parameter, *found->second))
      return fail(pipeline->context,
                  "pipeline argument kind does not match layout");
    const VernonPipelineArgument &argument = *found->second;
    if (argument.kind == VERNON_PIPELINE_TEXTURE) {
      if (!argument.texture.texture ||
          argument.texture.texture->context != pipeline->context ||
          argument.texture.format != argument.texture.texture->format ||
          argument.texture.dimension != argument.texture.texture->dimension ||
          argument.texture.width != argument.texture.texture->width ||
          argument.texture.height != argument.texture.texture->height ||
          argument.texture.depth != argument.texture.texture->depth ||
          (!parameter.dimension.empty() &&
           ((parameter.dimension == "2d" &&
             argument.texture.dimension != VERNON_TEXTURE_2D) ||
            (parameter.dimension == "3d" &&
             argument.texture.dimension != VERNON_TEXTURE_3D) ||
            (parameter.dimension == "cube" &&
             argument.texture.dimension != VERNON_TEXTURE_CUBE))))
        return fail(pipeline->context,
                    "pipeline texture argument does not match layout");
    } else if (argument.kind == VERNON_PIPELINE_SAMPLER &&
               (!argument.sampler ||
                argument.sampler->context != pipeline->context)) {
      return fail(pipeline->context,
                  "pipeline sampler belongs to another runtime");
    }
  }

  if (pipeline->computeKernel) {
    std::vector<VernonLaunchArgument> launchArguments;
    VernonLaunchSize grid{};
    const VernonStatus prepareStatus =
        prepareComputeLaunch(pipeline->context, pipeline->variant, arguments,
                             *invocation, launchArguments, grid);
    if (prepareStatus != VERNON_STATUS_OK)
      return prepareStatus;
    const VernonStatus launchStatus =
        vernonRuntimeLaunch(pipeline->computeKernel, grid,
                            launchArguments.data(), launchArguments.size());
    if (launchStatus != VERNON_STATUS_OK ||
        pipeline->context->backend == VERNON_RUNTIME_CPU ||
        pipeline->context->backend == VERNON_RUNTIME_CUDA)
      return launchStatus;
  }

  if (pipeline->context->backend == VERNON_RUNTIME_VULKAN) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!pipeline->vulkanVertex)
      return VERNON_STATUS_OK;
    if (!invocation->color_attachment_count || !invocation->color_attachments)
      return fail(pipeline->context,
                  "Vulkan graphics pipeline requires color attachments");

    std::vector<VkVertexInputBindingDescription> bindingDescriptions;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceSize> vertexOffsets;
    struct PushConstantValue {
      uint32_t slot;
      const void *data;
      uint32_t size;
      VkShaderStageFlags stages;
    };
    std::vector<PushConstantValue> pushConstants;
    struct SampledResource {
      VernonDeviceTexture *texture{};
      VernonDeviceSampler *sampler{};
      VkShaderStageFlags stages{};
    };
    std::map<std::pair<uint32_t, uint32_t>, SampledResource> sampledResources;
    uint32_t vertexCount = invocation->vertex_count;
    uint32_t instanceCount = invocation->instance_count;
    for (const Parameter &parameter : pipeline->variant.parameters) {
      const VernonPipelineArgument &argument = *arguments.at(parameter.slot);
      for (const ParameterUse &use : parameter.uses) {
        if (use.stage != "vertex" && use.stage != "fragment")
          continue;
        const VkShaderStageFlags shaderStage =
            use.stage == "vertex" ? VK_SHADER_STAGE_VERTEX_BIT
                                  : VK_SHADER_STAGE_FRAGMENT_BIT;
        if (argument.kind == VERNON_PIPELINE_TEXTURE) {
          if (use.binding == UINT32_MAX)
            return fail(pipeline->context,
                        "Vulkan sampled texture is missing set/binding");
          SampledResource &resource =
              sampledResources[{use.descriptorSet, use.binding}];
          if (resource.texture && resource.texture != argument.texture.texture)
            return fail(pipeline->context,
                        "Vulkan sampled texture binding is ambiguous");
          resource.texture = argument.texture.texture;
          resource.stages |= shaderStage;
          continue;
        }
        if (argument.kind == VERNON_PIPELINE_SAMPLER) {
          if (use.sampledTextureBindings.empty())
            return fail(
                pipeline->context,
                "sampler reflection has no paired sampled texture binding");
          for (const SampledTextureBinding &binding :
               use.sampledTextureBindings) {
            SampledResource &resource =
                sampledResources[{binding.descriptorSet, binding.binding}];
            if (resource.sampler && resource.sampler != argument.sampler)
              return fail(pipeline->context,
                          "Vulkan sampler pairing is ambiguous");
            resource.sampler = argument.sampler;
            resource.stages |= shaderStage;
          }
          continue;
        }
        if (use.interfaceKind == "uniform") {
          if (use.binding != UINT32_MAX)
            return fail(pipeline->context,
                        "Vulkan descriptor uniforms are not implemented",
                        VERNON_STATUS_UNSUPPORTED_TARGET);
          if (argument.kind != VERNON_PIPELINE_INLINE_VALUE ||
              !argument.inline_value.data || !argument.inline_value.data_size ||
              argument.inline_value.data_size > 128 ||
              argument.inline_value.data_size % sizeof(uint32_t) != 0)
            return fail(pipeline->context,
                        "Vulkan push constant value is invalid");
          const VkShaderStageFlags stage = shaderStage;
          auto existing =
              std::find_if(pushConstants.begin(), pushConstants.end(),
                           [&](const PushConstantValue &value) {
                             return value.slot == parameter.slot;
                           });
          if (existing != pushConstants.end()) {
            existing->stages |= stage;
          } else {
            if (std::any_of(pushConstants.begin(), pushConstants.end(),
                            [&](const PushConstantValue &value) {
                              return (value.stages & stage) != 0;
                            }))
              return fail(
                  pipeline->context,
                  "Vulkan supports one push-constant parameter per stage",
                  VERNON_STATUS_UNSUPPORTED_TARGET);
            pushConstants.push_back(
                {parameter.slot, argument.inline_value.data,
                 static_cast<uint32_t>(argument.inline_value.data_size),
                 stage});
          }
          continue;
        }
        if (use.interfaceKind != "input" && use.interfaceKind != "instance")
          continue;
        if (argument.kind != VERNON_PIPELINE_TENSOR ||
            argument.tensor.dtype != VERNON_DATA_F32 ||
            !argument.tensor.buffer ||
            argument.tensor.buffer->context != pipeline->context ||
            !argument.tensor.rank || !argument.tensor.shape ||
            !argument.tensor.byte_strides || use.location == UINT32_MAX)
          return fail(pipeline->context,
                      "Vulkan graphics Tensor view is invalid");
        uint32_t components = 1;
        for (uint32_t dimension = 1; dimension < argument.tensor.rank;
             ++dimension)
          components *= static_cast<uint32_t>(argument.tensor.shape[dimension]);
        const uint32_t leading =
            static_cast<uint32_t>(argument.tensor.shape[0]);
        const bool instanced =
            use.interfaceKind == "instance" || use.divisor != 0;
        uint32_t &inferred = instanced ? instanceCount : vertexCount;
        if (inferred && inferred != leading)
          return fail(pipeline->context,
                      "Vulkan graphics Tensor leading dimensions conflict");
        inferred = leading;
        const uint32_t binding =
            static_cast<uint32_t>(bindingDescriptions.size());
        bindingDescriptions.push_back(
            {binding, static_cast<uint32_t>(argument.tensor.byte_strides[0]),
             instanced ? VK_VERTEX_INPUT_RATE_INSTANCE
                       : VK_VERTEX_INPUT_RATE_VERTEX});
        vertexBuffers.push_back(argument.tensor.buffer->vulkanBuffer);
        vertexOffsets.push_back(argument.tensor.byte_offset);
        if (components <= 4) {
          attributeDescriptions.push_back(
              {use.location, binding, vulkanVertexFormat(components), 0});
        } else if (argument.tensor.rank == 3 && argument.tensor.shape[2] <= 4) {
          const uint32_t columns =
              static_cast<uint32_t>(argument.tensor.shape[1]);
          const uint32_t rows = static_cast<uint32_t>(argument.tensor.shape[2]);
          for (uint32_t column = 0; column < columns; ++column)
            attributeDescriptions.push_back({use.location + column, binding,
                                             vulkanVertexFormat(rows),
                                             column * rows * sizeof(float)});
        } else {
          return fail(pipeline->context,
                      "Vulkan vertex attribute shape is unsupported");
        }
      }
    }
    for (const auto &[binding, resource] : sampledResources)
      if (!resource.texture || !resource.sampler ||
          !resource.sampler->vulkanSampler)
        return fail(pipeline->context,
                    "Vulkan sampled image requires paired texture and sampler");
    if (!vertexCount || !instanceCount)
      return fail(pipeline->context,
                  "Vulkan graphics draw counts cannot be inferred");
    if (invocation->index_binding &&
        (!invocation->index_binding->buffer ||
         invocation->index_binding->buffer->context != pipeline->context ||
         invocation->index_binding->type != VERNON_INDEX_U32 ||
         !invocation->index_binding->index_count))
      return fail(pipeline->context, "Vulkan index binding is invalid");

    std::vector<const VernonColorAttachment *> attachments;
    for (size_t index = 0; index < invocation->color_attachment_count;
         ++index) {
      const VernonColorAttachment &attachment =
          invocation->color_attachments[index];
      if (!attachment.texture ||
          attachment.texture->context != pipeline->context ||
          attachment.texture->dimension != VERNON_TEXTURE_2D ||
          !attachment.texture->vulkanColorAttachment)
        return fail(pipeline->context, "Vulkan render target is invalid");
      attachments.push_back(&attachment);
    }
    for (const auto &[binding, resource] : sampledResources)
      if (std::any_of(attachments.begin(), attachments.end(),
                      [&](const VernonColorAttachment *attachment) {
                        return attachment->texture == resource.texture;
                      }))
        return fail(pipeline->context,
                    "a Vulkan texture cannot be sampled and rendered to in "
                    "the same invocation");
    std::sort(attachments.begin(), attachments.end(),
              [](const VernonColorAttachment *left,
                 const VernonColorAttachment *right) {
                return left->location < right->location;
              });
    const uint32_t width = attachments.front()->texture->width;
    const uint32_t height = attachments.front()->texture->height;
    const uint32_t maximumLocation = attachments.back()->location;
    std::vector<VkAttachmentDescription> attachmentDescriptions;
    std::vector<VkAttachmentReference> attachmentReferences(
        maximumLocation + 1,
        {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    std::vector<VkImageView> imageViews;
    for (size_t index = 0; index < attachments.size(); ++index) {
      if (attachments[index]->texture->width != width ||
          attachments[index]->texture->height != height)
        return fail(pipeline->context, "Vulkan render target extents differ");
      VkAttachmentDescription description{};
      description.format = attachments[index]->texture->vulkanFormat;
      description.samples = VK_SAMPLE_COUNT_1_BIT;
      description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      description.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      description.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      attachmentDescriptions.push_back(description);
      attachmentReferences[attachments[index]->location] = {
          static_cast<uint32_t>(index),
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      imageViews.push_back(attachments[index]->texture->vulkanView);
    }

    VernonRuntimeContext *context = pipeline->context;
    vernon::runtime::VulkanDriver &driver = vernon::runtime::vulkanDriver();
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkDescriptorSet> descriptorSets;
    auto cleanup = [&] {
      if (framebuffer)
        driver.destroyFramebuffer(context->vulkanDevice, framebuffer, nullptr);
      if (graphicsPipeline)
        driver.destroyPipeline(context->vulkanDevice, graphicsPipeline,
                               nullptr);
      if (pipelineLayout)
        driver.destroyPipelineLayout(context->vulkanDevice, pipelineLayout,
                                     nullptr);
      if (descriptorPool)
        driver.destroyDescriptorPool(context->vulkanDevice, descriptorPool,
                                     nullptr);
      for (VkDescriptorSetLayout layout : descriptorSetLayouts)
        driver.destroyDescriptorSetLayout(context->vulkanDevice, layout,
                                          nullptr);
      if (renderPass)
        driver.destroyRenderPass(context->vulkanDevice, renderPass, nullptr);
    };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount =
        static_cast<uint32_t>(attachmentReferences.size());
    subpass.pColorAttachments = attachmentReferences.data();
    VkRenderPassCreateInfo renderPassInfo{
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount =
        static_cast<uint32_t>(attachmentDescriptions.size());
    renderPassInfo.pAttachments = attachmentDescriptions.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    if (vulkanFail(context,
                   driver.createRenderPass(context->vulkanDevice,
                                           &renderPassInfo, nullptr,
                                           &renderPass),
                   "vkCreateRenderPass") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
    VkPipelineLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (!sampledResources.empty()) {
      const uint32_t maximumSet = sampledResources.rbegin()->first.first;
      descriptorSetLayouts.resize(maximumSet + 1, VK_NULL_HANDLE);
      for (uint32_t set = 0; set <= maximumSet; ++set) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (const auto &[key, resource] : sampledResources)
          if (key.first == set)
            bindings.push_back({key.second,
                                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                resource.stages, nullptr});
        VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptorLayoutInfo.bindingCount =
            static_cast<uint32_t>(bindings.size());
        descriptorLayoutInfo.pBindings = bindings.data();
        if (vulkanFail(context,
                       driver.createDescriptorSetLayout(
                           context->vulkanDevice, &descriptorLayoutInfo,
                           nullptr, &descriptorSetLayouts[set]),
                       "vkCreateDescriptorSetLayout") != VERNON_STATUS_OK) {
          cleanup();
          return VERNON_STATUS_INTERNAL_ERROR;
        }
      }
      VkDescriptorPoolSize poolSize{
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          static_cast<uint32_t>(sampledResources.size())};
      VkDescriptorPoolCreateInfo poolInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      poolInfo.maxSets = static_cast<uint32_t>(descriptorSetLayouts.size());
      poolInfo.poolSizeCount = 1;
      poolInfo.pPoolSizes = &poolSize;
      if (vulkanFail(context,
                     driver.createDescriptorPool(context->vulkanDevice,
                                                 &poolInfo, nullptr,
                                                 &descriptorPool),
                     "vkCreateDescriptorPool") != VERNON_STATUS_OK) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
      }
      descriptorSets.resize(descriptorSetLayouts.size());
      VkDescriptorSetAllocateInfo allocateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      allocateInfo.descriptorPool = descriptorPool;
      allocateInfo.descriptorSetCount =
          static_cast<uint32_t>(descriptorSetLayouts.size());
      allocateInfo.pSetLayouts = descriptorSetLayouts.data();
      if (vulkanFail(context,
                     driver.allocateDescriptorSets(context->vulkanDevice,
                                                   &allocateInfo,
                                                   descriptorSets.data()),
                     "vkAllocateDescriptorSets") != VERNON_STATUS_OK) {
        cleanup();
        return VERNON_STATUS_INTERNAL_ERROR;
      }
      std::vector<VkDescriptorImageInfo> imageInfos;
      std::vector<VkWriteDescriptorSet> writes;
      imageInfos.reserve(sampledResources.size());
      writes.reserve(sampledResources.size());
      for (const auto &[key, resource] : sampledResources) {
        imageInfos.push_back({resource.sampler->vulkanSampler,
                              resource.texture->vulkanView,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptorSets[key.first];
        write.dstBinding = key.second;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
      }
      driver.updateDescriptorSets(context->vulkanDevice,
                                  static_cast<uint32_t>(writes.size()),
                                  writes.data(), 0, nullptr);
      layoutInfo.setLayoutCount =
          static_cast<uint32_t>(descriptorSetLayouts.size());
      layoutInfo.pSetLayouts = descriptorSetLayouts.data();
    }
    std::vector<VkPushConstantRange> pushConstantRanges;
    for (const PushConstantValue &value : pushConstants)
      pushConstantRanges.push_back({value.stages, 0, value.size});
    layoutInfo.pushConstantRangeCount =
        static_cast<uint32_t>(pushConstantRanges.size());
    layoutInfo.pPushConstantRanges = pushConstantRanges.data();
    if (vulkanFail(context,
                   driver.createPipelineLayout(context->vulkanDevice,
                                               &layoutInfo, nullptr,
                                               &pipelineLayout),
                   "vkCreatePipelineLayout") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    const VkPipelineShaderStageCreateInfo shaderStages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, pipeline->vulkanVertex,
         pipeline->vulkanVertexEntry.c_str(), nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, pipeline->vulkanFragment,
         pipeline->vulkanFragmentEntry.c_str(), nullptr}};
    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount =
        static_cast<uint32_t>(bindingDescriptions.size());
    vertexInput.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = vulkanTopology(invocation->topology);
    VkPipelineViewportStateCreateInfo viewportState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterization{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkPipelineColorBlendAttachmentState> blendStates(
        attachmentDescriptions.size());
    for (auto &blend : blendStates)
      blend.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = static_cast<uint32_t>(blendStates.size());
    blend.pAttachments = blendStates.data();
    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;
    VkGraphicsPipelineCreateInfo pipelineInfo{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    if (vulkanFail(context,
                   driver.createGraphicsPipelines(
                       context->vulkanDevice, VK_NULL_HANDLE, 1, &pipelineInfo,
                       nullptr, &graphicsPipeline),
                   "vkCreateGraphicsPipelines") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    VkFramebufferCreateInfo framebufferInfo{
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(imageViews.size());
    framebufferInfo.pAttachments = imageViews.data();
    framebufferInfo.width = width;
    framebufferInfo.height = height;
    framebufferInfo.layers = 1;
    if (vulkanFail(context,
                   driver.createFramebuffer(context->vulkanDevice,
                                            &framebufferInfo, nullptr,
                                            &framebuffer),
                   "vkCreateFramebuffer") != VERNON_STATUS_OK) {
      cleanup();
      return VERNON_STATUS_INTERNAL_ERROR;
    }
    const bool submitted = submitVulkanCommands(context, [&](VkCommandBuffer
                                                                 command) {
      if (pipeline->variant.barrier) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        driver.cmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1,
                                  &barrier, 0, nullptr, 0, nullptr);
      }
      for (const VernonColorAttachment *attachment : attachments)
        if (attachment->texture->vulkanLayout !=
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
          transitionVulkanImage(command, attachment->texture,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      for (const auto &[binding, resource] : sampledResources)
        if (resource.texture->vulkanLayout !=
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          transitionVulkanImage(command, resource.texture,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      std::vector<VkClearValue> clearValues(attachments.size());
      VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      begin.renderPass = renderPass;
      begin.framebuffer = framebuffer;
      begin.renderArea.extent = {width, height};
      begin.clearValueCount = static_cast<uint32_t>(clearValues.size());
      begin.pClearValues = clearValues.data();
      driver.cmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
      driver.cmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             graphicsPipeline);
      if (!descriptorSets.empty())
        driver.cmdBindDescriptorSets(
            command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
            static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(),
            0, nullptr);
      for (const PushConstantValue &value : pushConstants)
        driver.cmdPushConstants(command, pipelineLayout, value.stages, 0,
                                value.size, value.data);
      const bool hasViewport =
          invocation->viewport[2] && invocation->viewport[3];
      const bool hasScissor = invocation->scissor[2] && invocation->scissor[3];
      VkViewport viewport{
          static_cast<float>(hasViewport ? invocation->viewport[0] : 0),
          static_cast<float>(hasViewport ? invocation->viewport[1] : 0),
          static_cast<float>(hasViewport ? invocation->viewport[2] : width),
          static_cast<float>(hasViewport ? invocation->viewport[3] : height),
          0.0f,
          1.0f};
      VkRect2D scissor{
          {static_cast<int32_t>(hasScissor ? invocation->scissor[0] : 0),
           static_cast<int32_t>(hasScissor ? invocation->scissor[1] : 0)},
          {hasScissor ? invocation->scissor[2] : width,
           hasScissor ? invocation->scissor[3] : height}};
      driver.cmdSetViewport(command, 0, 1, &viewport);
      driver.cmdSetScissor(command, 0, 1, &scissor);
      if (!vertexBuffers.empty())
        driver.cmdBindVertexBuffers(command, 0,
                                    static_cast<uint32_t>(vertexBuffers.size()),
                                    vertexBuffers.data(), vertexOffsets.data());
      if (invocation->index_binding) {
        driver.cmdBindIndexBuffer(
            command, invocation->index_binding->buffer->vulkanBuffer,
            invocation->index_binding->offset, VK_INDEX_TYPE_UINT32);
        driver.cmdDrawIndexed(command, invocation->index_binding->index_count,
                              instanceCount, 0, 0, 0);
      } else {
        driver.cmdDraw(command, vertexCount, instanceCount, 0, 0);
      }
      driver.cmdEndRenderPass(command);
    });
    cleanup();
    return submitted ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#else
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
  }

  makeCurrent(pipeline->context);
  OpenGLDriver &gl = pipeline->context->gl;
  if (pipeline->computeProgram) {
    gl.useProgram(pipeline->computeProgram);
    std::vector<GlUint> temporaryBuffers;
    for (const Parameter &parameter : pipeline->variant.parameters) {
      const VernonPipelineArgument &argument = *arguments.at(parameter.slot);
      for (const ParameterUse &use : parameter.uses) {
        if (use.stage != "compute")
          continue;
        if (use.binding == UINT32_MAX)
          return fail(pipeline->context,
                      "compute parameter has no resource binding");
        GlUint buffer = 0;
        if (argument.kind == VERNON_PIPELINE_TENSOR) {
          if (!argument.tensor.buffer ||
              argument.tensor.buffer->context != pipeline->context ||
              argument.tensor.byte_offset != 0)
            return fail(pipeline->context, "compute Tensor view is invalid");
          buffer = argument.tensor.buffer->name;
        } else if (argument.kind == VERNON_PIPELINE_INLINE_VALUE) {
          if (!argument.inline_value.data || !argument.inline_value.data_size)
            return fail(pipeline->context, "compute inline value is empty");
          gl.genBuffers(1, &buffer);
          temporaryBuffers.push_back(buffer);
          gl.bindBuffer(kShaderStorageBuffer, buffer);
          gl.bufferData(kShaderStorageBuffer,
                        static_cast<GlSizePtr>(argument.inline_value.data_size),
                        argument.inline_value.data, kDynamicCopy);
        } else {
          return fail(pipeline->context,
                      "compute textures are not supported yet");
        }
        gl.bindBufferBase(kShaderStorageBuffer, use.binding, buffer);
      }
    }
    VernonLaunchSize grid = invocation->compute_grid;
    if (!grid.x || !grid.y || !grid.z) {
      for (const auto &[slot, argument] : arguments) {
        (void)slot;
        if (argument->kind != VERNON_PIPELINE_TENSOR ||
            !argument->tensor.rank || !argument->tensor.shape)
          continue;
        grid = {1, 1, 1};
        const uint32_t rank = argument->tensor.rank;
        grid.x = static_cast<uint32_t>(argument->tensor.shape[rank - 1]);
        if (rank > 1)
          grid.y = static_cast<uint32_t>(argument->tensor.shape[rank - 2]);
        if (rank > 2)
          grid.z = static_cast<uint32_t>(argument->tensor.shape[rank - 3]);
        break;
      }
    }
    if (!grid.x || !grid.y || !grid.z)
      return fail(pipeline->context, "compute grid cannot be inferred");
    gl.dispatchCompute(
        (grid.x + pipeline->workgroup[0] - 1) / pipeline->workgroup[0],
        (grid.y + pipeline->workgroup[1] - 1) / pipeline->workgroup[1],
        (grid.z + pipeline->workgroup[2] - 1) / pipeline->workgroup[2]);
    gl.memoryBarrier(kShaderStorageBarrierBit);
    if (!temporaryBuffers.empty())
      gl.deleteBuffers(static_cast<GlSize>(temporaryBuffers.size()),
                       temporaryBuffers.data());
  }
  if (pipeline->variant.barrier)
    gl.memoryBarrier(kShaderStorageBarrierBit | kVertexAttribArrayBarrierBit);
  if (!pipeline->graphicsProgram)
    return VERNON_STATUS_OK;

  gl.useProgram(pipeline->graphicsProgram);
  gl.bindVertexArray(pipeline->vertexArray);
  uint32_t vertexCount = invocation->vertex_count;
  uint32_t instanceCount = invocation->instance_count;
  for (const Parameter &parameter : pipeline->variant.parameters) {
    const VernonPipelineArgument &argument = *arguments.at(parameter.slot);
    for (const ParameterUse &use : parameter.uses) {
      if (use.stage != "vertex" && use.stage != "fragment")
        continue;
      if (argument.kind == VERNON_PIPELINE_TEXTURE) {
        if (use.descriptorSet != 0 || use.binding == UINT32_MAX)
          return fail(pipeline->context,
                      "OpenGL sampled texture requires set zero and a binding");
        const VernonDeviceTexture *texture = argument.texture.texture;
        const GlEnum target =
            texture->dimension == VERNON_TEXTURE_2D   ? kTexture2D
            : texture->dimension == VERNON_TEXTURE_3D ? kTexture3D
                                                      : kTextureCubeMap;
        gl.activeTexture(kTexture0 + use.binding);
        gl.bindTexture(target, texture->name);
        const std::string uniformName =
            use.uniformName.empty() ? "main_arg_" + std::to_string(use.index)
                                    : use.uniformName;
        const GlInt location = gl.getUniformLocation(pipeline->graphicsProgram,
                                                     uniformName.c_str());
        if (location >= 0)
          gl.uniform1i(location, static_cast<GlInt>(use.binding));
        continue;
      }
      if (argument.kind == VERNON_PIPELINE_SAMPLER) {
        if (use.sampledTextureBindings.empty())
          return fail(pipeline->context,
                      "OpenGL sampler has no paired sampled texture binding");
        for (const SampledTextureBinding &binding :
             use.sampledTextureBindings) {
          if (binding.descriptorSet != 0 || binding.binding == UINT32_MAX)
            return fail(
                pipeline->context,
                "OpenGL sampled texture requires set zero and a binding");
          gl.bindSampler(binding.binding, argument.sampler->name);
        }
        continue;
      }
      if (use.interfaceKind == "uniform") {
        if (argument.kind != VERNON_PIPELINE_INLINE_VALUE ||
            argument.inline_value.dtype != VERNON_DATA_F32 ||
            !argument.inline_value.data ||
            argument.inline_value.data_size % sizeof(float) != 0)
          return fail(pipeline->context, "graphics uniform is invalid");
        const GlInt location = gl.getUniformLocation(pipeline->graphicsProgram,
                                                     use.uniformName.c_str());
        if (location < 0)
          return fail(pipeline->context,
                      "graphics uniform location is missing");
        const auto *data =
            static_cast<const float *>(argument.inline_value.data);
        const uint32_t count = static_cast<uint32_t>(
            argument.inline_value.data_size / sizeof(float));
        if (count == 1)
          gl.uniform1fv(location, 1, data);
        else if (count == 2)
          gl.uniform2fv(location, 1, data);
        else if (count == 3)
          gl.uniform3fv(location, 1, data);
        else if (count == 4)
          gl.uniform4fv(location, 1, data);
        else if (count == 9)
          gl.uniformMatrix3fv(location, 1, 1, data);
        else if (count == 16)
          gl.uniformMatrix4fv(location, 1, 1, data);
        else
          return fail(pipeline->context,
                      "graphics uniform size is unsupported");
        continue;
      }
      if (use.interfaceKind != "input" && use.interfaceKind != "instance")
        continue;
      if (argument.kind != VERNON_PIPELINE_TENSOR ||
          argument.tensor.dtype != VERNON_DATA_F32 || !argument.tensor.buffer ||
          argument.tensor.buffer->context != pipeline->context ||
          !argument.tensor.rank || !argument.tensor.shape ||
          !argument.tensor.byte_strides || use.location == UINT32_MAX)
        return fail(pipeline->context, "graphics Tensor view is invalid");
      uint32_t components = 1;
      for (uint32_t dimension = 1; dimension < argument.tensor.rank;
           ++dimension)
        components *= static_cast<uint32_t>(argument.tensor.shape[dimension]);
      const uint32_t leading = static_cast<uint32_t>(argument.tensor.shape[0]);
      const bool instanced =
          use.interfaceKind == "instance" || use.divisor != 0;
      uint32_t &inferred = instanced ? instanceCount : vertexCount;
      if (inferred && inferred != leading)
        return fail(pipeline->context,
                    "graphics Tensor leading dimensions conflict");
      inferred = leading;
      gl.bindBuffer(kArrayBuffer, argument.tensor.buffer->name);
      if (components <= 4) {
        gl.enableVertexAttribArray(use.location);
        gl.vertexAttribPointer(
            use.location, static_cast<GlInt>(components), kFloat, 0,
            static_cast<GlSize>(argument.tensor.byte_strides[0]),
            reinterpret_cast<const void *>(argument.tensor.byte_offset));
        gl.vertexAttribDivisor(use.location,
                               instanced ? std::max(use.divisor, 1u) : 0);
      } else if (argument.tensor.rank == 3 && argument.tensor.shape[2] <= 4) {
        const uint32_t columns =
            static_cast<uint32_t>(argument.tensor.shape[1]);
        const uint32_t rows = static_cast<uint32_t>(argument.tensor.shape[2]);
        for (uint32_t column = 0; column < columns; ++column) {
          gl.enableVertexAttribArray(use.location + column);
          gl.vertexAttribPointer(
              use.location + column, static_cast<GlInt>(rows), kFloat, 0,
              static_cast<GlSize>(argument.tensor.byte_strides[0]),
              reinterpret_cast<const void *>(argument.tensor.byte_offset +
                                             column * rows * sizeof(float)));
          gl.vertexAttribDivisor(use.location + column,
                                 instanced ? std::max(use.divisor, 1u) : 0);
        }
      } else {
        return fail(pipeline->context,
                    "graphics Tensor component shape is unsupported");
      }
    }
  }
  if (!vertexCount || !instanceCount)
    return fail(pipeline->context, "graphics draw counts cannot be inferred");
  if (!invocation->color_attachment_count || !invocation->color_attachments)
    return fail(pipeline->context, "graphics pipeline requires targets");
  gl.bindFramebuffer(kFramebuffer, pipeline->framebuffer);
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t maximumLocation = 0;
  for (size_t index = 0; index < invocation->color_attachment_count; ++index) {
    const VernonColorAttachment &attachment =
        invocation->color_attachments[index];
    if (!attachment.texture || attachment.texture->context != pipeline->context)
      return fail(pipeline->context, "render target is invalid");
    if (!width) {
      width = attachment.texture->width;
      height = attachment.texture->height;
    } else if (width != attachment.texture->width ||
               height != attachment.texture->height) {
      return fail(pipeline->context, "render target extents differ");
    }
    maximumLocation = std::max(maximumLocation, attachment.location);
    gl.framebufferTexture2D(kFramebuffer,
                            kColorAttachment0 + attachment.location, kTexture2D,
                            attachment.texture->name, 0);
  }
  std::vector<GlEnum> drawBuffers(maximumLocation + 1, kNone);
  for (size_t index = 0; index < invocation->color_attachment_count; ++index) {
    const uint32_t location = invocation->color_attachments[index].location;
    drawBuffers[location] = kColorAttachment0 + location;
  }
  gl.drawBuffers(static_cast<GlSize>(drawBuffers.size()), drawBuffers.data());
  if (gl.checkFramebufferStatus(kFramebuffer) != kFramebufferComplete)
    return fail(pipeline->context, "OpenGL framebuffer is incomplete",
                VERNON_STATUS_INTERNAL_ERROR);
  gl.viewport(0, 0, static_cast<GlSize>(width), static_cast<GlSize>(height));
  const GlEnum mode = topologyMode(invocation->topology);
  if (invocation->index_binding) {
    const VernonIndexBinding &index = *invocation->index_binding;
    if (!index.buffer || index.buffer->context != pipeline->context ||
        index.type != VERNON_INDEX_U32 || !index.index_count)
      return fail(pipeline->context, "index binding is invalid");
    gl.bindBuffer(kElementArrayBuffer, index.buffer->name);
    gl.drawElementsInstanced(mode, static_cast<GlSize>(index.index_count),
                             kUnsignedInt,
                             reinterpret_cast<const void *>(index.offset),
                             static_cast<GlSize>(instanceCount));
  } else if (instanceCount == 1) {
    gl.drawArrays(mode, 0, static_cast<GlSize>(vertexCount));
  } else {
    gl.drawArraysInstanced(mode, 0, static_cast<GlSize>(vertexCount),
                           static_cast<GlSize>(instanceCount));
  }
  return VERNON_STATUS_OK;
}

VernonStatus
vernonRuntimeComputeToGraphicsBarrier(VernonRuntimeContext *context) {
  if (!context ||
      (context->backend != VERNON_RUNTIME_OPENGL &&
       context->backend != VERNON_RUNTIME_OPENGL_ES) ||
      !context->gl.memoryBarrier)
    return VERNON_STATUS_UNSUPPORTED_TARGET;
  makeCurrent(context);
  context->gl.memoryBarrier(kShaderStorageBarrierBit |
                            kVertexAttribArrayBarrierBit);
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_INVALID_ARGUMENT;
  if (context->backend == VERNON_RUNTIME_CPU)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA)
    return cudaFail(context, vernon::runtime::cudaDriver().contextSynchronize(),
                    "cuCtxSynchronize");
#endif
#if defined(VERNON_HAS_VULKAN_RUNTIME)
  if (context->backend == VERNON_RUNTIME_VULKAN)
    return vulkanFail(
        context,
        vernon::runtime::vulkanDriver().deviceWaitIdle(context->vulkanDevice),
        "vkDeviceWaitIdle");
#endif
  makeCurrent(context);
  context->gl.finish();
  return VERNON_STATUS_OK;
}

} // extern "C"
