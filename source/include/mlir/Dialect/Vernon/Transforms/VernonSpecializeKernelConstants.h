#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSPECIALIZEKERNELCONSTANTS_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSPECIALIZEKERNELCONSTANTS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <string>

namespace mlir::vernon {

/// A host-static kernel parameter captured on a Program compute node.
/// The value is converted to the kernel argument's actual scalar type.
struct KernelHostConstant {
    enum class Kind { Integer, Float, Boolean };

    std::string name;
    Kind kind = Kind::Integer;
    int64_t integer = 0;
    double floating = 0.0;
    bool boolean = false;
};

/// Replaces matching entry arguments with `arith.constant` and erases them.
/// Resources, builtins, and non-scalar types cannot be specialized.
LogicalResult specializeKernelHostConstants(func::FuncOp entry, ArrayRef<KernelHostConstant> constants);

} // namespace mlir::vernon

#endif
