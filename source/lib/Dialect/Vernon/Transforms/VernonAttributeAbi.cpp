#include "mlir/Dialect/Vernon/Transforms/VernonAttributeAbi.h"

#include <algorithm>
#include <limits>

namespace mlir::vernon {

FailureOr<AttributeAbiLayout> getAttributeAbiLayout(Type type, ModuleOp module, ArrayRef<StringRef> logicalLeafDtypes) {
    FailureOr<ValueAbiLayout> valueLayout = getValueAbiLayout(type, module, logicalLeafDtypes);
    if (failed(valueLayout))
        return failure();

    AttributeAbiLayout result;
    result.valueLayout = std::move(*valueLayout);
    for (const ValueAbiLeaf &valueLeaf : result.valueLayout.leaves) {
        Type scalar = valueLeaf.scalarType;
        const unsigned bitWidth = scalar.getIntOrFloatBitWidth();
        if (valueLeaf.dtype == "bool" ||
            (!scalar.isInteger(32) && !scalar.isF16() && !scalar.isF32() && !scalar.isF64()) || bitWidth == 0 ||
            bitWidth % 8 != 0)
            return failure();
        const uint64_t scalarSize = bitWidth / 8;
        const uint64_t componentLimit = std::min<uint64_t>(4, 16 / scalarSize);
        for (uint64_t scalarIndex = 0; scalarIndex < valueLeaf.scalarCount; scalarIndex += componentLimit) {
            if (result.leaves.size() >= std::numeric_limits<uint32_t>::max())
                return failure();
            result.leaves.push_back(
                {valueLeaf.path, scalar, valueLeaf.dtype, static_cast<uint32_t>(result.leaves.size()),
                 static_cast<uint32_t>(std::min(componentLimit, valueLeaf.scalarCount - scalarIndex)),
                 valueLeaf.byteOffset + scalarIndex * scalarSize});
        }
    }
    return result;
}

} // namespace mlir::vernon
