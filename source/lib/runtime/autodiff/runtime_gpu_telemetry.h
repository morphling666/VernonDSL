#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_TELEMETRY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_TELEMETRY_H

#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/runtime_gpu_resources.h"

namespace vernon::runtime::ad::gpu {

bool retainedValueBytes(const DeviceValues &devices, const HostValues &hosts, size_t &bytes);
std::shared_ptr<AutodiffMemoryReservation> reserveRetainedValues(VernonRuntimeContext &context,
                                                                 const DeviceValues &devices, const HostValues &hosts,
                                                                 size_t &bytes);
std::shared_ptr<AutodiffMemoryReservation> reserveApplyMemory(VernonRuntimeContext &context, size_t requested,
                                                              size_t &limit);

} // namespace vernon::runtime::ad::gpu

#endif
