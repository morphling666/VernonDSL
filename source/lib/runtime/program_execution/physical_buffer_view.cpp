#include "physical_buffer_view.h"

#include "runtime/pipeline_metadata.h"
#include "runtime/tensor_bridge.h"

#include <limits>
#include <optional>

namespace vernon::runtime::program_execution {
namespace {

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

} // namespace

bool materializePhysicalBufferView(const shape::DeclaredShape &declaredShape, const ValueLayout &layout,
                                   size_t logicalBytes, PhysicalBufferView &view) {
    if (layout.leaves.size() != 1 || !logicalBytes)
        return false;
    const ValueLeaf &leaf = layout.leaves.front();
    const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
    if (!dtype)
        return false;
    auto scalarBytes = dataTypeSize(*dtype);
    size_t elementBytes = 0;
    if (scalarBytes.isErr() || !leaf.scalarCount ||
        !checkedMultiply(scalarBytes.value(), static_cast<size_t>(leaf.scalarCount), elementBytes) || !elementBytes)
        return false;
    size_t elements = 0;
    return shape::resolveSingleDynamicExtent(declaredShape, elementBytes, logicalBytes, view.shape) &&
           shape::rowMajorByteStrides(view.shape, elementBytes, view.strides) &&
           shape::checkedElementCount(view.shape, elements) && elements <= logicalBytes / elementBytes;
}

} // namespace vernon::runtime::program_execution
