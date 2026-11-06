#ifndef VERNON_HAL_CUDA_DEVICE_H
#define VERNON_HAL_CUDA_DEVICE_H

#include "hal/hal.h"

#include <cstdint>
#include <memory>

namespace vernon::hal {

class Resources;

namespace cuda {
struct DeviceState;
}

// A CUDA device as a HAL device: a command buffer is a tape of kernel launches, and the queue is the device's stream,
// which its uploads and downloads share. A submission enqueues its launches and returns. With `labels`, each labelled
// dispatch launches inside an NVTX range of the "Vernon" domain carrying its label. Null without CUDA support.
std::unique_ptr<Device> createCudaDevice(std::shared_ptr<cuda::DeviceState> state, bool labels);

// The resources of a CUDA HAL device: buffers are device allocations of the device's CUDA context; the device has no
// images, samplers or surfaces. A buffer's reference is a native buffer reference (hal/native_buffers.h) whose
// resource value is its device address. Null without CUDA support.
std::unique_ptr<Resources> createCudaResources(std::shared_ptr<cuda::DeviceState> state);

} // namespace vernon::hal

#endif
