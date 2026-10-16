#include "VernonTensorElementAccess.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::vernon {

std::optional<int64_t> constantLinearIndex(ArrayRef<int64_t> shape, ValueRange indices) {
    int64_t linear = 0;
    for (auto [dimension, index] : llvm::zip_equal(shape, indices)) {
        if (auto cast = index.getDefiningOp<arith::IndexCastOp>())
            index = cast.getIn();
        auto constant = index.getDefiningOp<arith::ConstantOp>();
        auto value = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
        if (!value || value.getInt() < 0 || value.getInt() >= dimension)
            return std::nullopt;
        linear = linear * dimension + value.getInt();
    }
    return linear;
}

Value linearTensorIndex(Location location, ArrayRef<int64_t> shape, ValueRange indices, OpBuilder &builder) {
    Value linear = indices.front();
    for (auto [extent, index] : llvm::zip_equal(shape.drop_front(), indices.drop_front())) {
        linear =
            arith::MulIOp::create(builder, location, linear, arith::ConstantIndexOp::create(builder, location, extent));
        linear = arith::AddIOp::create(builder, location, linear, index);
    }
    return linear;
}

FailureOr<Value> extractLLVMArrayElement(Operation *op, Value array, Value linear, RewriterBase &rewriter) {
    auto arrayType = dyn_cast<LLVM::LLVMArrayType>(array.getType());
    auto function = op->getParentOfType<FunctionOpInterface>();
    if (!arrayType || !function || function.isExternal())
        return failure();
    Location location = op->getLoc();
    auto pointer = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value storage;
    {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&function.getFunctionBody().front());
        Value one = LLVM::ConstantOp::create(rewriter, location, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
        storage = LLVM::AllocaOp::create(rewriter, location, pointer, arrayType, one);
    }
    LLVM::StoreOp::create(rewriter, location, array, storage);
    Value index = arith::IndexCastOp::create(rewriter, location, rewriter.getI64Type(), linear);
    Value element =
        LLVM::GEPOp::create(rewriter, location, pointer, arrayType, storage, ArrayRef<LLVM::GEPArg>{0, index});
    return LLVM::LoadOp::create(rewriter, location, arrayType.getElementType(), element).getResult();
}

} // namespace mlir::vernon
