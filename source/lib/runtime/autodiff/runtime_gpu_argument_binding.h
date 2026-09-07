#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_ARGUMENT_BINDING_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_ARGUMENT_BINDING_H

#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/program_execution/physical_buffer_view.h"
#include "runtime/shape_layout.h"

namespace vernon::runtime::ad::gpu {

bool appendInternalBufferArgument(VernonRuntimeContext &context, const Parameter &parameter,
                                  const program_execution::DeviceBuffer &buffer, size_t logicalBytes,
                                  program_execution::PhysicalBufferView &view,
                                  std::vector<VernonProgramArgument> &arguments);
bool appendBindingArgument(const Binding &binding, std::vector<VernonProgramArgument> &arguments);

} // namespace vernon::runtime::ad::gpu

#endif
