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
#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::vernon {
namespace {

struct HostAggregateType {
    LLVM::LLVMStructType type;
    SmallVector<int64_t> fieldIndices;
};

FailureOr<HostAggregateType> getHostAggregateType(Type source, const TypeConverter &converter, ModuleOp module) {
    SmallVector<Type> fields;
    if (auto tuple = dyn_cast<TupleType>(source)) {
        fields.append(tuple.getTypes().begin(), tuple.getTypes().end());
    } else if (auto structure = dyn_cast<StructType>(source)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> resolved = resolveStructFields(structure, module);
        if (failed(resolved))
            return failure();
        fields = std::move(resolved->second);
    } else {
        return failure();
    }
    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(source, module);
    FailureOr<PhysicalValueAbiLayout> physical =
        getPhysicalValueAbiLayout(source, module, PhysicalAbiProfile::HostValue);
    if (failed(layout) || failed(physical) || layout->size != physical->size ||
        layout->alignment != physical->alignment || layout->fieldOffsets.size() != fields.size())
        return failure();

    SmallVector<Type> elements;
    SmallVector<int64_t> fieldIndices;
    uint64_t offset = 0;
    Type byte = IntegerType::get(source.getContext(), 8);
    for (auto [index, field] : llvm::enumerate(fields)) {
        const uint64_t fieldOffset = layout->fieldOffsets[index];
        if (fieldOffset < offset)
            return failure();
        if (fieldOffset != offset)
            elements.push_back(LLVM::LLVMArrayType::get(byte, fieldOffset - offset));
        fieldIndices.push_back(static_cast<int64_t>(elements.size()));
        Type converted = converter.convertType(field);
        FailureOr<ValueAbiLayout> fieldLayout = getValueAbiLayout(field, module);
        if (!converted || failed(fieldLayout))
            return failure();
        elements.push_back(converted);
        offset = fieldOffset + fieldLayout->size;
    }
    if (layout->size < offset)
        return failure();
    if (layout->size != offset)
        elements.push_back(LLVM::LLVMArrayType::get(byte, layout->size - offset));
    return HostAggregateType{LLVM::LLVMStructType::getLiteral(source.getContext(), elements, true),
                             std::move(fieldIndices)};
}

struct ReturnPattern final : OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
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
    TupleCreatePattern(TypeConverter &converter, MLIRContext *context, ModuleOp module)
        : OpConversionPattern(converter, context), module(module) {}

    LogicalResult matchAndRewrite(TupleCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        FailureOr<HostAggregateType> aggregate =
            getHostAggregateType(op.getResult().getType(), *getTypeConverter(), module);
        if (failed(aggregate) || aggregate->fieldIndices.size() != adaptor.getElements().size())
            return failure();
        Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), aggregate->type);
        for (auto [index, element] : llvm::enumerate(adaptor.getElements()))
            result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result, element,
                                                 ArrayRef<int64_t>{aggregate->fieldIndices[index]});
        rewriter.replaceOp(op, result);
        return success();
    }

private:
    ModuleOp module;
};

struct TupleGetPattern final : OpConversionPattern<TupleGetOp> {
    TupleGetPattern(TypeConverter &converter, MLIRContext *context, ModuleOp module)
        : OpConversionPattern(converter, context), module(module) {}

    LogicalResult matchAndRewrite(TupleGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        FailureOr<HostAggregateType> aggregate =
            getHostAggregateType(op.getInput().getType(), *getTypeConverter(), module);
        if (failed(aggregate) || op.getIndex() >= aggregate->fieldIndices.size())
            return failure();
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{aggregate->fieldIndices[op.getIndex()]});
        return success();
    }

private:
    ModuleOp module;
};

struct StructCreatePattern final : OpConversionPattern<StructCreateOp> {
    StructCreatePattern(TypeConverter &converter, MLIRContext *context, ModuleOp module)
        : OpConversionPattern(converter, context), module(module) {}

    LogicalResult matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        FailureOr<HostAggregateType> aggregate =
            getHostAggregateType(op.getResult().getType(), *getTypeConverter(), module);
        if (failed(aggregate) || aggregate->fieldIndices.size() != adaptor.getFields().size())
            return failure();
        Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), aggregate->type);
        for (auto [index, field] : llvm::enumerate(adaptor.getFields()))
            result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result, field,
                                                 ArrayRef<int64_t>{aggregate->fieldIndices[index]});
        rewriter.replaceOp(op, result);
        return success();
    }

private:
    ModuleOp module;
};

struct StructGetPattern final : OpConversionPattern<StructGetOp> {
    StructGetPattern(TypeConverter &converter, MLIRContext *context, ModuleOp module)
        : OpConversionPattern(converter, context), module(module) {}

    LogicalResult matchAndRewrite(StructGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        FailureOr<HostAggregateType> aggregate =
            getHostAggregateType(op.getInput().getType(), *getTypeConverter(), module);
        if (failed(aggregate) || op.getIndex() >= aggregate->fieldIndices.size())
            return failure();
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{aggregate->fieldIndices[op.getIndex()]});
        return success();
    }

private:
    ModuleOp module;
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
        // CPU value tensors preserve their complete static shape.
        addVernonSharedValueTypeConversions(converter);
        converter.addConversion([&converter, module](TupleType tuple) -> std::optional<Type> {
            FailureOr<HostAggregateType> aggregate = getHostAggregateType(tuple, converter, module);
            return succeeded(aggregate) ? std::optional<Type>(aggregate->type) : std::nullopt;
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
            FailureOr<HostAggregateType> aggregate = getHostAggregateType(structure, converter, module);
            return succeeded(aggregate) ? std::optional<Type>(aggregate->type) : std::nullopt;
        });
        converter.addConversion(
            [module](TensorViewType view, SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
                if (view.getElementType().isIntOrFloat())
                    return std::nullopt;
                FailureOr<ValueAbiLayout> layout = getValueAbiLayout(view.getElementType(), module);
                if (failed(layout))
                    return failure();
                for (const ValueAbiLeaf &leaf : layout->leaves)
                    types.push_back(MemRefType::get({ShapedType::kDynamic}, leaf.scalarType));
                return success();
            });

        RewritePatternSet patterns(context);
        populateVernonSharedValuePatterns(converter, patterns);
        patterns.add<ResourceIntrinsicTypePattern, ReturnPattern, AggregateTensorConstructPattern,
                     AggregateTensorGetPattern>(converter, context);
        patterns.add<StructCreatePattern, StructGetPattern, TupleCreatePattern, TupleGetPattern>(converter, context,
                                                                                                 module);
        populateCpuAggregateTensorViewPatterns(converter, patterns, module);
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
        target.addDynamicallyLegalOp<LoadOp, StoreOp, PhysicalLoadOp, PhysicalStoreOp, PhysicalAtomicOp>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addIllegalOp<SwizzleOp, StructCreateOp, StructGetOp, TupleCreateOp, TupleGetOp, ReduceSumOp,
                            ScatterAddOp>();
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
