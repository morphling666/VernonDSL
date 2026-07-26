#ifndef VERNON_LIB_DIALECT_VERNON_TRANSFORMS_VERNONCPULOWERINGUTILS_H
#define VERNON_LIB_DIALECT_VERNON_TRANSFORMS_VERNONCPULOWERINGUTILS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::vernon {

enum class CpuIntrinsicKind {
    SharedValue,
    TensorViewLoad,
    TensorViewStore,
    TextureSample,
    Unknown,
};

inline CpuIntrinsicKind classifyCpuIntrinsic(llvm::StringRef name) {
    if (name == "tensor_view_load")
        return CpuIntrinsicKind::TensorViewLoad;
    if (name == "tensor_view_store")
        return CpuIntrinsicKind::TensorViewStore;
    if (name == "texture_sample")
        return CpuIntrinsicKind::TextureSample;
    if (name == "construct" || name == "broadcast" || name == "dot" || name == "normalize" || name == "cross" ||
        name == "reflect" || name == "matmul" || name == "min" || name == "max" || name == "pow" || name == "clamp")
        return CpuIntrinsicKind::SharedValue;
    return CpuIntrinsicKind::Unknown;
}

inline bool isCpuResourceIntrinsic(CpuIntrinsicKind kind) {
    return kind == CpuIntrinsicKind::TensorViewLoad || kind == CpuIntrinsicKind::TensorViewStore ||
           kind == CpuIntrinsicKind::TextureSample;
}

} // namespace mlir::vernon

#endif
