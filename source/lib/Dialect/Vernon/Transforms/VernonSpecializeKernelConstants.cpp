#include "mlir/Dialect/Vernon/Transforms/VernonSpecializeKernelConstants.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::vernon {
namespace {

bool isResourceType(Type type) { return isa<TensorViewType, TextureType, SamplerType>(type); }

LogicalResult fitsInteger(IntegerType type, int64_t value) {
    const unsigned width = type.getWidth();
    if (width == 1)
        return success(value == 0 || value == 1);
    if (type.isUnsigned()) {
        if (value < 0)
            return failure();
        return success(APInt(64, static_cast<uint64_t>(value)).isIntN(width));
    }
    return success(APInt(64, value, /*isSigned=*/true).isSignedIntN(width));
}

FailureOr<Value> materializeConstant(OpBuilder &builder, Location location, Type type,
                                     const KernelHostConstant &constant, StringRef name) {
    if (auto integerType = dyn_cast<IntegerType>(type)) {
        int64_t value = constant.integer;
        if (constant.kind == KernelHostConstant::Kind::Boolean) {
            if (integerType.getWidth() != 1)
                return emitError(location)
                       << "host constant '" << name << "' is a boolean but kernel argument type is " << type;
            value = constant.boolean ? 1 : 0;
        } else if (constant.kind != KernelHostConstant::Kind::Integer) {
            return emitError(location) << "host constant '" << name << "' is not an integer for kernel argument type "
                                       << type;
        }
        if (failed(fitsInteger(integerType, value)))
            return emitError(location) << "host constant '" << name << "' does not fit kernel argument type " << type;
        return arith::ConstantOp::create(builder, location, IntegerAttr::get(integerType, value)).getResult();
    }
    if (auto floatType = dyn_cast<FloatType>(type)) {
        if (constant.kind != KernelHostConstant::Kind::Float)
            return emitError(location) << "host constant '" << name << "' is not a float for kernel argument type "
                                       << type;
        return arith::ConstantOp::create(builder, location, FloatAttr::get(floatType, constant.floating)).getResult();
    }
    return emitError(location) << "host constant '" << name << "' cannot specialize kernel argument type " << type;
}

} // namespace

LogicalResult specializeKernelHostConstants(func::FuncOp entry, ArrayRef<KernelHostConstant> constants) {
    if (constants.empty())
        return success();
    if (entry.isExternal() || !llvm::hasSingleElement(entry.getBody()))
        return entry.emitError("kernel host-constant specialization requires a defined single-block function");

    llvm::StringMap<unsigned> argumentsByName;
    for (BlockArgument argument : entry.getArguments()) {
        if (entry.getArgAttr(argument.getArgNumber(), kBuiltinAttrName))
            continue;
        auto sourceName = entry.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "vernon.source_name");
        if (!sourceName || sourceName.getValue().empty())
            continue;
        if (!argumentsByName.try_emplace(sourceName.getValue(), argument.getArgNumber()).second)
            return entry.emitError() << "kernel argument source name '" << sourceName.getValue() << "' is not unique";
    }

    llvm::BitVector erase(entry.getNumArguments(), false);
    mlir::Block &block = entry.getBody().front();
    OpBuilder builder(&block, block.begin());
    for (const KernelHostConstant &constant : constants) {
        if (constant.name.empty())
            return entry.emitError("kernel host constant has an empty name");
        auto found = argumentsByName.find(constant.name);
        if (found == argumentsByName.end())
            return entry.emitError() << "kernel has no argument to specialize for host constant '" << constant.name
                                     << "'";
        const unsigned index = found->second;
        if (erase.test(index))
            return entry.emitError() << "host constant '" << constant.name << "' is specified more than once";
        Type type = entry.getArgument(index).getType();
        if (isResourceType(type))
            return entry.emitError() << "host constant '" << constant.name << "' cannot specialize a resource argument";
        FailureOr<Value> materialized = materializeConstant(builder, entry.getLoc(), type, constant, constant.name);
        if (failed(materialized))
            return failure();
        entry.getArgument(index).replaceAllUsesWith(*materialized);
        erase.set(index);
    }
    if (erase.any() && failed(entry.eraseArguments(erase)))
        return entry.emitError("failed to erase specialized kernel host constants");
    return success();
}

} // namespace mlir::vernon
