#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"

#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "llvm/ADT/STLExtras.h"
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

StringRef stringifyInterpolation(Interpolation interpolation) {
    switch (interpolation) {
    case Interpolation::Smooth:
        return "smooth";
    case Interpolation::NoPerspective:
        return "noperspective";
    case Interpolation::Flat:
        return "flat";
    }
    llvm_unreachable("unknown Vernon interpolation");
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

FailureOr<Interpolation> parseInterpolation(Attribute attr) {
    auto value = dyn_cast_if_present<StringAttr>(attr);
    if (!value)
        return failure();
    return llvm::StringSwitch<FailureOr<Interpolation>>(value.getValue())
        .Case("smooth", Interpolation::Smooth)
        .Case("noperspective", Interpolation::NoPerspective)
        .Case("flat", Interpolation::Flat)
        .Default(failure());
}

FailureOr<Interpolation> resolveInterpolation(DictionaryAttr attrs, Type type, ModuleOp module) {
    if (Attribute explicitValue = attrs ? attrs.get(kInterpolationAttrName) : Attribute())
        return parseInterpolation(explicitValue);
    FailureOr<ValueAbiLayout> layout = getValueStorageLayout(type, module);
    if (failed(layout) || layout->leaves.empty())
        return failure();
    const bool integer =
        llvm::any_of(layout->leaves, [](const ValueAbiLeaf &leaf) { return isa<IntegerType>(leaf.scalarType); });
    return integer ? Interpolation::Flat : Interpolation::Smooth;
}

bool InterfaceAttrs::empty() const {
    return !kind && !location && !builtin && !descriptorSet && !binding && !instanceDivisor;
}

InterfaceAttrs parseInterfaceAttrs(DictionaryAttr attrs) {
    if (!attrs)
        return {};
    return {attrs.get(kInterfaceAttrName),     attrs.get(kLocationAttrName), attrs.get(kBuiltinAttrName),
            attrs.get(kDescriptorSetAttrName), attrs.get(kBindingAttrName),  attrs.get(kInstanceDivisorAttrName)};
}

} // namespace mlir::vernon
