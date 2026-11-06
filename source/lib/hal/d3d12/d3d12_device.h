#ifndef VERNON_HAL_D3D12_D3D12_DEVICE_H
#define VERNON_HAL_D3D12_D3D12_DEVICE_H

// The DirectX 12 device of linked schedules (specs/runtime/architecture_v2.md section 16.3): it records each command
// buffer's passes into native command lists once per frame slot and submits a call as one ExecuteCommandLists.

#include "hal/hal.h"

#include <memory>

struct VernonRuntimeContext;

namespace vernon::hal {

// Null unless the context is a DirectX 12 context with its device state.
std::unique_ptr<Device> createDirectX12Device(VernonRuntimeContext &context);

} // namespace vernon::hal

#endif
