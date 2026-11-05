#ifndef VERNON_HAL_VULKAN_RESOURCES_H
#define VERNON_HAL_VULKAN_RESOURCES_H

#include "VernonRHI.h"

#include <cstdint>
#include <memory>

namespace vernon::rhi::vulkan {
struct Buffer;
}

namespace vernon::hal {

class DeviceBuffers;
class Resources;

// The resources of a Vulkan HAL device: Vulkan objects outside the RHI's registry. Buffers are named by native buffer
// references (hal/native_buffers.h), views and samplers by the entries of hal/native_images.h, whose `native` is the
// view's rhi::vulkan::Image or the rhi::vulkan::Sampler. Null for a device of another backend; the RHI device must
// outlive them.
std::unique_ptr<Resources> createVulkanResources(VernonRhiDevice device);

// The buffers a Vulkan HAL device allocates for its schedules, named the same way. Null for a device of another
// backend; the RHI device must outlive them.
std::unique_ptr<DeviceBuffers> createVulkanDeviceBuffers(VernonRhiDevice device);

// The buffer of `serial`; null once it is destroyed. The caller keeps the buffer alive while it uses the result.
rhi::vulkan::Buffer *vulkanBuffer(uint64_t serial);

} // namespace vernon::hal

#endif
