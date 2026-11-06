#ifndef VERNON_HAL_D3D12_D3D12_RESOURCES_H
#define VERNON_HAL_D3D12_D3D12_RESOURCES_H

#include <cstdint>
#include <memory>

namespace vernon::hal::d3d12 {
struct Buffer;
struct DeviceState;
} // namespace vernon::hal::d3d12

namespace vernon::hal {

class DeviceBuffers;
class Resources;

// The resources of a DirectX 12 HAL device over `state`. Buffers are named by native buffer references
// (hal/native_buffers.h), views and samplers by the entries of hal/native_images.h, whose `native` is the view's
// d3d12::View or the d3d12::Sampler.
std::unique_ptr<Resources> createDirectX12Resources(std::shared_ptr<d3d12::DeviceState> state);

// The buffers a DirectX 12 HAL device allocates for its schedules, named the same way.
std::unique_ptr<DeviceBuffers> createDirectX12DeviceBuffers(std::shared_ptr<d3d12::DeviceState> state);

// The buffer of `serial`; null once it is destroyed. The caller keeps the buffer alive while it uses the result.
d3d12::Buffer *directX12Buffer(uint64_t serial);

} // namespace vernon::hal

#endif
