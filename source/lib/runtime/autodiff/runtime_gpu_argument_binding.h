#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_ARGUMENT_BINDING_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_ARGUMENT_BINDING_H

#include "runtime/autodiff/runtime_gpu_bindings.h"
#include "runtime/shape_layout.h"

namespace vernon::runtime::ad::gpu {

struct InternalBufferView {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

bool materializeInternalBufferView(const shape::DeclaredShape &declaredShape, const ValueLayout &layout,
                                   size_t logicalBytes, InternalBufferView &view);
bool appendInternalBufferArgument(VernonRuntimeContext &context, const Parameter &parameter, const DeviceBuffer &buffer,
                                  size_t logicalBytes, InternalBufferView &view,
                                  std::vector<VernonPipelineArgument> &arguments);
bool appendBindingArgument(const Binding &binding, std::vector<VernonPipelineArgument> &arguments);

} // namespace vernon::runtime::ad::gpu

#endif
