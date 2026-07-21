#ifndef VERNON_LIB_DIALECT_VERNON_TRANSFORMS_VERNONCPULOWERINGUTILS_H
#define VERNON_LIB_DIALECT_VERNON_TRANSFORMS_VERNONCPULOWERINGUTILS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::vernon {

enum class CpuIntrinsicKind {
  SharedValue,
  BufferLoad,
  BufferStore,
  TextureSample,
  Unknown,
};

inline CpuIntrinsicKind classifyCpuIntrinsic(llvm::StringRef name) {
  if (name == "buffer_load")
    return CpuIntrinsicKind::BufferLoad;
  if (name == "buffer_store")
    return CpuIntrinsicKind::BufferStore;
  if (name == "texture_sample")
    return CpuIntrinsicKind::TextureSample;
  if (name == "construct" || name == "dot" || name == "normalize" ||
      name == "cross" || name == "reflect" || name == "matmul" ||
      name == "min" || name == "max" || name == "pow" || name == "clamp")
    return CpuIntrinsicKind::SharedValue;
  return CpuIntrinsicKind::Unknown;
}

inline bool isCpuResourceIntrinsic(CpuIntrinsicKind kind) {
  return kind == CpuIntrinsicKind::BufferLoad ||
         kind == CpuIntrinsicKind::BufferStore ||
         kind == CpuIntrinsicKind::TextureSample;
}

} // namespace mlir::vernon

#endif
