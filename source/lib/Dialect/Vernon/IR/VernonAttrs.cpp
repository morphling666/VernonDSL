#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"

#include "llvm/ADT/StringSwitch.h"

namespace mlir::vernon {

StringRef stringifyShaderStage(ShaderStage stage) {
  switch (stage) {
  case ShaderStage::Vertex:
    return "vertex";
  case ShaderStage::Fragment:
    return "fragment";
  case ShaderStage::Compute:
    return "compute";
  }
  llvm_unreachable("unknown Vernon shader stage");
}

StringRef stringifyInterfaceKind(InterfaceKind kind) {
  switch (kind) {
  case InterfaceKind::Input:
    return "input";
  case InterfaceKind::Output:
    return "output";
  case InterfaceKind::Uniform:
    return "uniform";
  case InterfaceKind::Resource:
    return "resource";
  }
  llvm_unreachable("unknown Vernon interface kind");
}

FailureOr<ShaderStage> parseShaderStage(Attribute attr) {
  auto value = dyn_cast_if_present<StringAttr>(attr);
  if (!value)
    return failure();
  return llvm::StringSwitch<FailureOr<ShaderStage>>(value.getValue())
      .Case("vertex", ShaderStage::Vertex)
      .Case("fragment", ShaderStage::Fragment)
      .Case("compute", ShaderStage::Compute)
      .Default(failure());
}

FailureOr<InterfaceKind> parseInterfaceKind(Attribute attr) {
  auto value = dyn_cast_if_present<StringAttr>(attr);
  if (!value)
    return failure();
  return llvm::StringSwitch<FailureOr<InterfaceKind>>(value.getValue())
      .Case("input", InterfaceKind::Input)
      .Case("output", InterfaceKind::Output)
      .Case("uniform", InterfaceKind::Uniform)
      .Case("resource", InterfaceKind::Resource)
      .Default(failure());
}

bool InterfaceAttrs::empty() const {
  return !kind && !location && !builtin && !descriptorSet && !binding &&
         !instanceDivisor;
}

InterfaceAttrs parseInterfaceAttrs(DictionaryAttr attrs) {
  if (!attrs)
    return {};
  return {attrs.get(kInterfaceAttrName), attrs.get(kLocationAttrName),
          attrs.get(kBuiltinAttrName),   attrs.get(kDescriptorSetAttrName),
          attrs.get(kBindingAttrName),   attrs.get(kInstanceDivisorAttrName)};
}

} // namespace mlir::vernon
