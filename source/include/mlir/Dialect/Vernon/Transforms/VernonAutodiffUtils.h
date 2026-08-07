#pragma once

#include "mlir/Support/LLVM.h"

namespace mlir::vernon {

inline SmallVector<SmallVector<int64_t>> enumerateStaticCoordinates(ArrayRef<int64_t> shape) {
    SmallVector<SmallVector<int64_t>> coordinates(1);
    for (int64_t extent : shape) {
        SmallVector<SmallVector<int64_t>> expanded;
        for (const SmallVector<int64_t> &prefix : coordinates)
            for (int64_t index = 0; index < extent; ++index) {
                SmallVector<int64_t> coordinate(prefix);
                coordinate.push_back(index);
                expanded.push_back(std::move(coordinate));
            }
        coordinates = std::move(expanded);
    }
    return coordinates;
}

} // namespace mlir::vernon
