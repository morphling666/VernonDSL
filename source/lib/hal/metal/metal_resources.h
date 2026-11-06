#ifndef VERNON_HAL_METAL_METAL_RESOURCES_H
#define VERNON_HAL_METAL_METAL_RESOURCES_H

#include <cstdint>
#include <memory>

namespace vernon::hal::metal {
struct Buffer;
struct DeviceState;
} // namespace vernon::hal::metal

namespace vernon::hal {

class DeviceBuffers;
class Resources;

// The resources of a Metal HAL device over `state`. Buffers are named by native buffer references
// (hal/native_buffers.h), views and samplers by the entries of hal/native_images.h, whose `native` is the view's
// metal::View or the metal::Sampler.
std::unique_ptr<Resources> createMetalResources(std::shared_ptr<metal::DeviceState> state);

// The buffers a Metal HAL device allocates for its schedules, named the same way.
std::unique_ptr<DeviceBuffers> createMetalDeviceBuffers(std::shared_ptr<metal::DeviceState> state);

// The buffer of `serial`; null once it is destroyed. The caller keeps the buffer alive while it uses the result.
metal::Buffer *metalBuffer(uint64_t serial);

} // namespace vernon::hal

#endif
