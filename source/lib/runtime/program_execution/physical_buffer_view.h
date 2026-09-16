#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PHYSICAL_BUFFER_VIEW_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PHYSICAL_BUFFER_VIEW_H

#include "runtime/shape_layout.h"
#include "runtime/stage_binding_plan.h"

#include <vector>

namespace vernon::runtime::program_execution {

struct PhysicalBufferView {
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
};

bool materializePhysicalBufferView(const shape::DeclaredShape &declaredShape, const ValueLayout &layout,
                                   size_t logicalBytes, PhysicalBufferView &view);

} // namespace vernon::runtime::program_execution

#endif
