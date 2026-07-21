//===- VernonAttrs.h - Vernon shader interface schema -----------*- C++ -*-===//

#ifndef MLIR_DIALECT_VERNON_IR_VERNONATTRS_H_
#define MLIR_DIALECT_VERNON_IR_VERNONATTRS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::vernon {

/// Stable textual schema for shader entry points:
///
///   func.func @main(...) -> (...) attributes {
///     vernon.entry,
///     vernon.stage = "vertex" | "fragment" | "compute",
///     vernon.workgroup_size = array<i32: x, y, z>
///   }
///
/// Function arguments and results use:
///
///   vernon.interface = "input" | "output" | "uniform" | "resource"
///   vernon.location = <non-negative integer>
///   vernon.builtin = "<target-independent builtin name>"
///   vernon.set = <non-negative integer>
///   vernon.binding = <non-negative integer>
///   vernon.instance_divisor = <positive integer>
///
/// Input/output interfaces have exactly one of location or builtin.
/// Resource interfaces have both set and binding. Uniform interfaces may omit
/// both for legacy OpenGL uniforms, or provide both when backed by a descriptor.
/// Ordinary values use standard tensor/vector/linalg types and operations; this
/// schema introduces no Vernon-specific vector or matrix type.
inline constexpr llvm::StringLiteral kEntryAttrName = "vernon.entry";
inline constexpr llvm::StringLiteral kStageAttrName = "vernon.stage";
inline constexpr llvm::StringLiteral kWorkgroupSizeAttrName =
    "vernon.workgroup_size";
inline constexpr llvm::StringLiteral kInterfaceAttrName = "vernon.interface";
inline constexpr llvm::StringLiteral kLocationAttrName = "vernon.location";
inline constexpr llvm::StringLiteral kBuiltinAttrName = "vernon.builtin";
inline constexpr llvm::StringLiteral kDescriptorSetAttrName = "vernon.set";
inline constexpr llvm::StringLiteral kBindingAttrName = "vernon.binding";
inline constexpr llvm::StringLiteral kInstanceDivisorAttrName =
    "vernon.instance_divisor";

enum class ShaderStage { Vertex, Fragment, Compute };
enum class InterfaceKind { Input, Output, Uniform, Resource };

StringRef stringifyShaderStage(ShaderStage stage);
StringRef stringifyInterfaceKind(InterfaceKind kind);
FailureOr<ShaderStage> parseShaderStage(Attribute attr);
FailureOr<InterfaceKind> parseInterfaceKind(Attribute attr);

/// Parsed views keep the original attributes so validation can distinguish a
/// missing attribute from an attribute with the wrong storage type.
struct InterfaceAttrs {
  Attribute kind;
  Attribute location;
  Attribute builtin;
  Attribute descriptorSet;
  Attribute binding;
  Attribute instanceDivisor;

  bool empty() const;
};

InterfaceAttrs parseInterfaceAttrs(DictionaryAttr attrs);

} // namespace mlir::vernon

#endif // MLIR_DIALECT_VERNON_IR_VERNONATTRS_H_
