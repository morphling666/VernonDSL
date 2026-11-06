#ifndef VERNON_HAL_VULKAN_RESOURCES_H
#define VERNON_HAL_VULKAN_RESOURCES_H

#include <cstdint>
#include <memory>

namespace vernon::hal::vulkan {
struct Buffer;
struct DeviceState;
} // namespace vernon::hal::vulkan

namespace vernon::hal {

class DeviceBuffers;
class Resources;

// The resources of a Vulkan HAL device over `state`. Buffers are named by native buffer references
// (hal/native_buffers.h), views and samplers by the entries of hal/native_images.h, whose `native` is the view's
// vulkan::Image or the vulkan::Sampler.
std::unique_ptr<Resources> createVulkanResources(std::shared_ptr<vulkan::DeviceState> state);

// The buffers a Vulkan HAL device allocates for its schedules, named the same way.
std::unique_ptr<DeviceBuffers> createVulkanDeviceBuffers(std::shared_ptr<vulkan::DeviceState> state);

// The buffer of `serial`; null once it is destroyed. The caller keeps the buffer alive while it uses the result.
vulkan::Buffer *vulkanBuffer(uint64_t serial);

} // namespace vernon::hal

#endif
