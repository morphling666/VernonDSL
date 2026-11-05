#ifndef VERNON_HAL_CUDA_DEVICE_H
#define VERNON_HAL_CUDA_DEVICE_H

#include "hal/hal.h"

#include "VernonRHI.h"

#include <cstdint>
#include <memory>

namespace vernon::hal {

class Resources;

// A CUDA RHI device as a HAL device: a command buffer is a tape of kernel launches, and the queue is the device's
// stream, which its uploads and downloads share. A submission enqueues its launches and returns. Null without CUDA
// support or for a device of another backend; the RHI device must outlive the HAL device.
std::unique_ptr<Device> createCudaDevice(VernonRhiDevice device);

// The resources of a CUDA HAL device: buffers are device allocations of the RHI device's CUDA context, outside the
// RHI's registry; the device has no images, samplers or surfaces. A buffer's reference is a native buffer reference
// (hal/native_buffers.h) whose resource value is its device address. Null without CUDA support.
std::unique_ptr<Resources> createCudaResources(VernonRhiDevice device);

} // namespace vernon::hal

#endif
