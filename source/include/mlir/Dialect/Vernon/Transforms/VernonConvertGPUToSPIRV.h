#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONCONVERTGPUTOSPIRV_H_
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONCONVERTGPUTOSPIRV_H_

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

// The access of a kernel's texture or storage-buffer argument, which the GPU lowering erases from its type: the
// texture's or TensorView's access, and "read" for a value passed by storage. Read storage buffers are decorated
// NonWritable, which D3D12 binds as SRVs.
inline constexpr llvm::StringLiteral kResourceAccessAttrName = "vernon.resource_access";
// The interface ABIs of a SPIR-V entry function's read storage-buffer arguments, kept on the function while the
// generic ABI lowering replaces those arguments with globals.
inline constexpr llvm::StringLiteral kNonWritableStorageBindingsAttrName = "vernon.non_writable_storage_bindings";

std::unique_ptr<Pass> createVernonConvertGPUToSPIRVPass();
void registerVernonConvertGPUToSPIRVPass();
// Runs after the generic SPIR-V ABI lowering.
std::unique_ptr<Pass> createVernonDecorateNonWritableStoragePass();
void registerVernonDecorateNonWritableStoragePass();

} // namespace vernon
} // namespace mlir

#endif
