#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUTensors.h"

#include "VernonCpuLoweringUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::vernon {
namespace {

struct ReturnPattern final : OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

FailureOr<Value> buildAggregateValue(Type sourceType, ValueRange leaves, unsigned &cursor,
                                     const TypeConverter &converter, ModuleOp module,
                                     ConversionPatternRewriter &rewriter, Location location) {
    if (sourceType.isIntOrFloat()) {
        if (cursor >= leaves.size())
            return failure();
        return leaves[cursor++];
    }
    auto buildProduct = [&](TypeRange fields, Type targetType) -> FailureOr<Value> {
        auto target = dyn_cast<LLVM::LLVMStructType>(targetType);
        if (!target)
            return failure();
        Value result = LLVM::UndefOp::create(rewriter, location, target);
        for (auto [index, field] : llvm::enumerate(fields)) {
            FailureOr<Value> value = buildAggregateValue(field, leaves, cursor, converter, module, rewriter, location);
            if (failed(value))
                return failure();
            result = LLVM::InsertValueOp::create(rewriter, location, result, *value,
                                                 ArrayRef<int64_t>{static_cast<int64_t>(index)});
        }
        return result;
    };
    if (auto tuple = dyn_cast<TupleType>(sourceType))
        return buildProduct(tuple.getTypes(), converter.convertType(sourceType));
    if (auto structure = dyn_cast<StructType>(sourceType)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        if (failed(fields))
            return failure();
        return buildProduct(fields->second, converter.convertType(sourceType));
    }

    Type element;
    ArrayRef<int64_t> shape;
    if (auto tensor = dyn_cast<RankedTensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto tensor = dyn_cast<TensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else {
        return failure();
    }
    int64_t count = 1;
    for (int64_t dimension : shape)
        count *= dimension;
    SmallVector<Value> elements;
    for (int64_t index = 0; index < count; ++index) {
        FailureOr<Value> value = buildAggregateValue(element, leaves, cursor, converter, module, rewriter, location);
        if (failed(value))
            return failure();
        elements.push_back(*value);
    }
    Type targetType = converter.convertType(sourceType);
    if (auto vector = dyn_cast<VectorType>(targetType))
        return vector::FromElementsOp::create(rewriter, location, vector, elements).getResult();
    auto array = dyn_cast<LLVM::LLVMArrayType>(targetType);
    if (!array)
        return failure();
    Value result = LLVM::UndefOp::create(rewriter, location, array);
    for (auto [index, value] : llvm::enumerate(elements))
        result = LLVM::InsertValueOp::create(rewriter, location, result, value,
                                             ArrayRef<int64_t>{static_cast<int64_t>(index)});
    return result;
}

LogicalResult decomposeAggregateValue(Type sourceType, Value value, SmallVectorImpl<Value> &leaves,
                                      const TypeConverter &converter, ModuleOp module,
                                      ConversionPatternRewriter &rewriter, Location location) {
    if (sourceType.isIntOrFloat()) {
        leaves.push_back(value);
        return success();
    }
    auto decomposeProduct = [&](TypeRange fields) -> LogicalResult {
        for (auto [index, field] : llvm::enumerate(fields)) {
            Value extracted =
                LLVM::ExtractValueOp::create(rewriter, location, value, ArrayRef<int64_t>{static_cast<int64_t>(index)});
            if (failed(decomposeAggregateValue(field, extracted, leaves, converter, module, rewriter, location)))
                return failure();
        }
        return success();
    };
    if (auto tuple = dyn_cast<TupleType>(sourceType))
        return decomposeProduct(tuple.getTypes());
    if (auto structure = dyn_cast<StructType>(sourceType)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        return failed(fields) ? failure() : decomposeProduct(fields->second);
    }

    Type element;
    ArrayRef<int64_t> shape;
    if (auto tensor = dyn_cast<RankedTensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto tensor = dyn_cast<TensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else {
        return failure();
    }
    int64_t count = 1;
    for (int64_t dimension : shape)
        count *= dimension;
    for (int64_t index = 0; index < count; ++index) {
        Value extracted;
        if (isa<VectorType>(value.getType()))
            extracted = vector::ExtractOp::create(rewriter, location, value, index);
        else
            extracted = LLVM::ExtractValueOp::create(rewriter, location, value, ArrayRef<int64_t>{index});
        if (failed(decomposeAggregateValue(element, extracted, leaves, converter, module, rewriter, location)))
            return failure();
    }
    return success();
}

Value storageLeafIndex(Value recordIndex, uint64_t recordSize, const StorageLeaf &leaf, uint64_t scalarIndex,
                       ConversionPatternRewriter &rewriter, Location location) {
    uint64_t leafSize = std::max<uint64_t>(leaf.type.getIntOrFloatBitWidth() / 8, 1);
    Value stride = arith::ConstantIndexOp::create(rewriter, location, recordSize / leafSize);
    Value result = arith::MulIOp::create(rewriter, location, recordIndex, stride);
    const uint64_t scalarOffset = leaf.byteOffset / leafSize + scalarIndex;
    if (scalarOffset) {
        Value offset = arith::ConstantIndexOp::create(rewriter, location, scalarOffset);
        result = arith::AddIOp::create(rewriter, location, result, offset);
    }
    return result;
}

struct AggregateViewIntrinsicPattern final : ConversionPattern {
    AggregateViewIntrinsicPattern(TypeConverter &converter, MLIRContext *context, ModuleOp module)
        : ConversionPattern(converter, IntrinsicOp::getOperationName(), 2, context), module(module) {}

    LogicalResult matchAndRewrite(Operation *operation, ArrayRef<ValueRange> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto op = cast<IntrinsicOp>(operation);
        if (op.getName() != "tensor_view_load" && op.getName() != "tensor_view_store")
            return failure();
        auto view = dyn_cast<TensorViewType>(op.getOperand(0).getType());
        if (!view || view.getElementType().isIntOrFloat())
            return failure();
        FailureOr<StorageLayout> layout = resolveStorageLayout(view.getElementType(), module);
        if (failed(layout) || layout->leaves.empty())
            return op.emitError("cannot resolve aggregate TensorView storage layout");
        if (operands.size() != (op.getName() == "tensor_view_load" ? 2 : 3) ||
            operands[0].size() != layout->leaves.size() || operands[1].size() != 1)
            return op.emitError("aggregate TensorView conversion received an invalid operand mapping");
        ValueRange storageOperands = operands[0];
        Value recordIndex = operands[1].front();
        if (op.getName() == "tensor_view_load") {
            SmallVector<Value> leaves;
            for (auto [leaf, storage] : llvm::zip_equal(layout->leaves, storageOperands))
                for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex)
                    leaves.push_back(memref::LoadOp::create(
                        rewriter, op.getLoc(), storage,
                        storageLeafIndex(recordIndex, layout->size, leaf, scalarIndex, rewriter, op.getLoc())));
            unsigned cursor = 0;
            FailureOr<Value> value = buildAggregateValue(view.getElementType(), leaves, cursor, *getTypeConverter(),
                                                         module, rewriter, op.getLoc());
            if (failed(value) || cursor != leaves.size())
                return op.emitError("cannot reconstruct aggregate TensorView value");
            rewriter.replaceOp(op, *value);
            return success();
        }

        SmallVector<Value> leaves;
        if (operands[2].size() != 1 ||
            failed(decomposeAggregateValue(view.getElementType(), operands[2].front(), leaves, *getTypeConverter(),
                                           module, rewriter, op.getLoc())))
            return op.emitError("cannot decompose aggregate TensorView value");
        unsigned cursor = 0;
        for (auto [leaf, storage] : llvm::zip_equal(layout->leaves, storageOperands))
            for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex) {
                if (cursor >= leaves.size())
                    return op.emitError("cannot decompose aggregate TensorView value");
                memref::StoreOp::create(
                    rewriter, op.getLoc(), leaves[cursor++], storage,
                    storageLeafIndex(recordIndex, layout->size, leaf, scalarIndex, rewriter, op.getLoc()));
            }
        if (cursor != leaves.size())
            return op.emitError("cannot decompose aggregate TensorView value");
        rewriter.eraseOp(op);
        return success();
    }

    ModuleOp module;
};

struct ResourceIntrinsicTypePattern final : OpConversionPattern<IntrinsicOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!isCpuResourceIntrinsic(classifyCpuIntrinsic(op.getName())))
            return failure();
        if (auto view = dyn_cast<TensorViewType>(op.getOperand(0).getType());
            view && !view.getElementType().isIntOrFloat())
            return failure();
        SmallVector<Type> resultTypes;
        if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
            return failure();
        OperationState state(op.getLoc(), IntrinsicOp::getOperationName());
        state.addOperands(adaptor.getOperands());
        state.addTypes(resultTypes);
        state.addAttributes(op->getAttrs());
        Operation *replacement = rewriter.create(state);
        rewriter.replaceOp(op, replacement->getResults());
        return success();
    }
};

struct TupleCreatePattern final : OpConversionPattern<TupleCreateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(TupleCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type converted = getTypeConverter()->convertType(op.getResult().getType());
        auto structType = dyn_cast_if_present<LLVM::LLVMStructType>(converted);
        if (!structType)
            return failure();
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), structType);
        for (auto [index, element] : llvm::enumerate(adaptor.getElements()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, element,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }
};

struct TupleGetPattern final : OpConversionPattern<TupleGetOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(TupleGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{static_cast<int64_t>(op.getIndex())});
        return success();
    }
};

struct StructCreatePattern final : OpConversionPattern<StructCreateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto structType =
            dyn_cast_if_present<LLVM::LLVMStructType>(getTypeConverter()->convertType(op.getResult().getType()));
        if (!structType)
            return failure();
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), structType);
        for (auto [index, field] : llvm::enumerate(adaptor.getFields()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, field,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }
};

struct StructGetPattern final : OpConversionPattern<StructGetOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StructGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{static_cast<int64_t>(op.getIndex())});
        return success();
    }
};

struct AggregateTensorConstructPattern final : OpConversionPattern<IntrinsicOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op.getName() != "construct" || !isa<TensorType>(op.getResult().getType()))
            return failure();
        auto arrayType =
            dyn_cast_if_present<LLVM::LLVMArrayType>(getTypeConverter()->convertType(op.getResult().getType()));
        if (!arrayType || adaptor.getOperands().size() != arrayType.getNumElements())
            return failure();
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), arrayType);
        for (auto [index, element] : llvm::enumerate(adaptor.getOperands()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, element,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }
};

struct AggregateTensorGetPattern final : OpConversionPattern<TensorGetOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(TensorGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<TensorType>(op.getInput().getType());
        if (!sourceType || adaptor.getIndices().size() != sourceType.getShape().size())
            return failure();
        Location location = op.getLoc();
        Value linear = adaptor.getIndices().front();
        for (auto [dimension, index] :
             llvm::zip_equal(sourceType.getShape().drop_front(), adaptor.getIndices().drop_front())) {
            Value extent = arith::ConstantIndexOp::create(rewriter, location, dimension);
            linear = arith::MulIOp::create(rewriter, location, linear, extent);
            linear = arith::AddIOp::create(rewriter, location, linear, index);
        }
        int64_t elementCount = 1;
        for (int64_t dimension : sourceType.getShape())
            elementCount *= dimension;
        Value selected = LLVM::ExtractValueOp::create(rewriter, location, adaptor.getInput(), ArrayRef<int64_t>{0});
        for (int64_t index = 1; index < elementCount; ++index) {
            Value candidate =
                LLVM::ExtractValueOp::create(rewriter, location, adaptor.getInput(), ArrayRef<int64_t>{index});
            Value expected = arith::ConstantIndexOp::create(rewriter, location, index);
            Value matches = arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::eq, linear, expected);
            selected = arith::SelectOp::create(rewriter, location, matches, candidate, selected);
        }
        rewriter.replaceOp(op, selected);
        return success();
    }
};

struct VernonLowerCPUTensorsPass final : PassWrapper<VernonLowerCPUTensorsPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerCPUTensorsPass)

    StringRef getArgument() const final { return "vernon-lower-cpu-tensors"; }
    StringRef getDescription() const final { return "Lower static Vernon CPU value tensors to unrestricted vectors"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect, LLVM::LLVMDialect, scf::SCFDialect,
                        memref::MemRefDialect, tensor::TensorDialect, vector::VectorDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        bool invalid = false;
        getOperation().walk([&](IntrinsicOp intrinsic) {
            if (classifyCpuIntrinsic(intrinsic.getName()) != CpuIntrinsicKind::Unknown)
                return;
            auto function = intrinsic->getParentOfType<func::FuncOp>();
            intrinsic.emitError() << "unknown Vernon intrinsic '" << intrinsic.getName() << "' in CPU entry '"
                                  << (function ? function.getSymName() : StringRef("<unknown>")) << "'";
            invalid = true;
        });
        if (invalid) {
            signalPassFailure();
            return;
        }

        MLIRContext *context = &getContext();
        TypeConverter converter;
        // CPU value tensors retain the legacy emitter's unrestricted static size.
        addVernonSharedValueTypeConversions(converter);
        converter.addConversion([&converter](TupleType tuple) -> std::optional<Type> {
            SmallVector<Type> elements;
            if (failed(converter.convertTypes(tuple.getTypes(), elements)))
                return std::nullopt;
            return LLVM::LLVMStructType::getLiteral(tuple.getContext(), elements, true);
        });
        converter.addConversion([&converter](TensorType tensor) -> std::optional<Type> {
            Type element = converter.convertType(tensor.getElementType());
            if (!element)
                return std::nullopt;
            int64_t count = 1;
            for (int64_t dimension : tensor.getShape())
                count *= dimension;
            return LLVM::LLVMArrayType::get(element, count);
        });
        converter.addConversion([&converter, module](StructType structure) -> std::optional<Type> {
            FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
            if (failed(fields))
                return std::nullopt;
            SmallVector<Type> converted;
            if (failed(converter.convertTypes(fields->second, converted)))
                return std::nullopt;
            return LLVM::LLVMStructType::getLiteral(structure.getContext(), converted, true);
        });
        converter.addConversion(
            [module](TensorViewType view, SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
                if (view.getElementType().isIntOrFloat())
                    return std::nullopt;
                FailureOr<StorageLayout> layout = resolveStorageLayout(view.getElementType(), module);
                if (failed(layout))
                    return failure();
                for (const StorageLeaf &leaf : layout->leaves)
                    types.push_back(MemRefType::get({ShapedType::kDynamic}, leaf.type));
                return success();
            });

        RewritePatternSet patterns(context);
        populateVernonSharedValuePatterns(converter, patterns);
        patterns
            .add<ResourceIntrinsicTypePattern, ReturnPattern, AggregateTensorConstructPattern,
                 AggregateTensorGetPattern, StructCreatePattern, StructGetPattern, TupleCreatePattern, TupleGetPattern>(
                converter, context);
        patterns.add<AggregateViewIntrinsicPattern>(converter, context, module);
        populateFunctionOpInterfaceTypeConversionPattern(func::FuncOp::getOperationName(), patterns, converter);

        ConversionTarget target(*context);
        target.addLegalOp<ModuleOp, StructDeclOp>();
        target.addLegalDialect<memref::MemRefDialect>();
        target.addDynamicallyLegalDialect<arith::ArithDialect, func::FuncDialect, math::MathDialect, scf::SCFDialect,
                                          tensor::TensorDialect, vector::VectorDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
            return converter.isSignatureLegal(function.getFunctionType()) && converter.isLegal(&function.getBody());
        });
        target.addDynamicallyLegalOp<IntrinsicOp>([&](IntrinsicOp intrinsic) {
            return isCpuResourceIntrinsic(classifyCpuIntrinsic(intrinsic.getName())) &&
                   converter.isLegal(intrinsic.getOperation());
        });
        target.addIllegalOp<SwizzleOp, StructCreateOp, StructGetOp, TupleCreateOp, TupleGetOp>();
        populateVernonSharedValueStructuralTypeConversions(converter, patterns, target);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
            return;
        }
        for (StructDeclOp declaration : llvm::make_early_inc_range(module.getOps<StructDeclOp>()))
            declaration.erase();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerCPUTensorsPass() { return std::make_unique<VernonLowerCPUTensorsPass>(); }

void registerVernonLowerCPUTensorsPass() { PassRegistration<VernonLowerCPUTensorsPass>(); }

} // namespace mlir::vernon
