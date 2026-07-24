#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"

#include <functional>

namespace mlir::vernon {
namespace {

constexpr int64_t kRegisterTensorElementLimit = 16;

struct GpuTupleCreatePattern final : OpConversionPattern<TupleCreateOp> {
    GpuTupleCreatePattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(TupleCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type converted = getTypeConverter()->convertType(op.getResult().getType());
        if (useSpirv) {
            auto structType = dyn_cast_if_present<spirv::StructType>(converted);
            if (!structType)
                return failure();
            rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, structType, adaptor.getElements());
            return success();
        }
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

    bool useSpirv;
};

struct GpuTupleGetPattern final : OpConversionPattern<TupleGetOp> {
    GpuTupleGetPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(TupleGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (useSpirv) {
            rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
                op, adaptor.getInput(), ArrayRef<int32_t>{static_cast<int32_t>(op.getIndex())});
            return success();
        }
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{static_cast<int64_t>(op.getIndex())});
        return success();
    }

    bool useSpirv;
};

struct GpuStructCreatePattern final : OpConversionPattern<StructCreateOp> {
    GpuStructCreatePattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type> fieldTypes;
        llvm::transform(adaptor.getFields(), std::back_inserter(fieldTypes),
                        [](Value field) { return field.getType(); });
        if (useSpirv) {
            auto structType = spirv::StructType::get(fieldTypes);
            rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, structType, adaptor.getFields());
            return success();
        }
        auto structType = LLVM::LLVMStructType::getLiteral(op.getContext(), fieldTypes);
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), structType);
        for (auto [index, field] : llvm::enumerate(adaptor.getFields()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, field,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }

    bool useSpirv;
};

struct GpuStructGetPattern final : OpConversionPattern<StructGetOp> {
    GpuStructGetPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(StructGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (useSpirv) {
            rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
                op, adaptor.getInput(), ArrayRef<int32_t>{static_cast<int32_t>(op.getIndex())});
            return success();
        }
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{static_cast<int64_t>(op.getIndex())});
        return success();
    }

    bool useSpirv;
};

struct GpuAggregateTensorConstructPattern final : OpConversionPattern<IntrinsicOp> {
    GpuAggregateTensorConstructPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op.getName() != "construct" || !isa<TensorType>(op.getResult().getType()))
            return failure();
        Type converted = getTypeConverter()->convertType(op.getResult().getType());
        if (useSpirv) {
            auto arrayType = dyn_cast_if_present<spirv::ArrayType>(converted);
            if (!arrayType || adaptor.getOperands().size() != arrayType.getNumElements())
                return failure();
            rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, arrayType, adaptor.getOperands());
            return success();
        }
        auto arrayType = dyn_cast_if_present<LLVM::LLVMArrayType>(converted);
        if (!arrayType || adaptor.getOperands().size() != arrayType.getNumElements())
            return failure();
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), arrayType);
        for (auto [index, element] : llvm::enumerate(adaptor.getOperands()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, element,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }

    bool useSpirv;
};

struct GpuAggregateTensorGetPattern final : OpConversionPattern<TensorGetOp> {
    GpuAggregateTensorGetPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

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
        auto extract = [&](int64_t index) -> Value {
            if (useSpirv)
                return spirv::CompositeExtractOp::create(rewriter, location, adaptor.getInput(),
                                                         ArrayRef<int32_t>{static_cast<int32_t>(index)});
            return LLVM::ExtractValueOp::create(rewriter, location, adaptor.getInput(), ArrayRef<int64_t>{index});
        };
        Value selected = extract(0);
        for (int64_t index = 1; index < elementCount; ++index) {
            Value candidate = extract(index);
            Value expected = arith::ConstantIndexOp::create(rewriter, location, index);
            Value matches = arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::eq, linear, expected);
            if (useSpirv) {
                std::function<Value(Type, Value, Value)> selectValue = [&](Type type, Value whenTrue,
                                                                           Value whenFalse) -> Value {
                    if (auto structType = dyn_cast<spirv::StructType>(type)) {
                        SmallVector<Value> fields;
                        for (unsigned field = 0; field < structType.getNumElements(); ++field) {
                            Value trueField = spirv::CompositeExtractOp::create(
                                rewriter, location, whenTrue, ArrayRef<int32_t>{static_cast<int32_t>(field)});
                            Value falseField = spirv::CompositeExtractOp::create(
                                rewriter, location, whenFalse, ArrayRef<int32_t>{static_cast<int32_t>(field)});
                            fields.push_back(selectValue(structType.getElementType(field), trueField, falseField));
                        }
                        return spirv::CompositeConstructOp::create(rewriter, location, structType, fields);
                    }
                    if (auto arrayType = dyn_cast<spirv::ArrayType>(type)) {
                        SmallVector<Value> elements;
                        for (unsigned element = 0; element < arrayType.getNumElements(); ++element) {
                            Value trueElement = spirv::CompositeExtractOp::create(
                                rewriter, location, whenTrue, ArrayRef<int32_t>{static_cast<int32_t>(element)});
                            Value falseElement = spirv::CompositeExtractOp::create(
                                rewriter, location, whenFalse, ArrayRef<int32_t>{static_cast<int32_t>(element)});
                            elements.push_back(selectValue(arrayType.getElementType(), trueElement, falseElement));
                        }
                        return spirv::CompositeConstructOp::create(rewriter, location, arrayType, elements);
                    }
                    return spirv::SelectOp::create(rewriter, location, type, matches, whenTrue, whenFalse);
                };
                selected = selectValue(candidate.getType(), candidate, selected);
            } else {
                selected = arith::SelectOp::create(rewriter, location, matches, candidate, selected);
            }
        }
        rewriter.replaceOp(op, selected);
        return success();
    }

    bool useSpirv;
};

struct VernonLowerGPUTensorsPass final : PassWrapper<VernonLowerGPUTensorsPass, OperationPass<gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerGPUTensorsPass)

    VernonLowerGPUTensorsPass() = default;
    explicit VernonLowerGPUTensorsPass(bool useSpirvTupleAbi) : useSpirvTupleAbi(useSpirvTupleAbi) {}

    StringRef getArgument() const final { return "vernon-lower-gpu-tensors"; }
    StringRef getDescription() const final { return "Lower Vernon value tensors to GPU register vectors"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, LLVM::LLVMDialect, math::MathDialect, scf::SCFDialect, spirv::SPIRVDialect,
                        vector::VectorDialect>();
    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        WalkResult dynamicTensor = getOperation().walk([&](Operation *operation) {
            for (Type type : operation->getResultTypes()) {
                auto tensor = dyn_cast<RankedTensorType>(type);
                if (tensor && !tensor.hasStaticShape()) {
                    operation->emitError("dynamic local value Tensor cannot be allocated on this GPU "
                                         "backend; "
                                         "specialize its shape or pass it as an addressable Tensor");
                    return WalkResult::interrupt();
                }
            }
            for (Region &region : operation->getRegions()) {
                for (Block &block : region) {
                    for (BlockArgument argument : block.getArguments()) {
                        auto tensor = dyn_cast<RankedTensorType>(argument.getType());
                        if (tensor && !tensor.hasStaticShape()) {
                            operation->emitError("dynamic local value Tensor cannot be allocated on this GPU "
                                                 "backend; "
                                                 "specialize its shape or pass it as an addressable Tensor");
                            return WalkResult::interrupt();
                        }
                    }
                }
            }
            return WalkResult::advance();
        });
        if (dynamicTensor.wasInterrupted()) {
            signalPassFailure();
            return;
        }

        TypeConverter converter;
        llvm::StringMap<SmallVector<Type>> structFields;
        getOperation().walk([&](StructCreateOp create) {
            auto structure = dyn_cast<StructType>(create.getResult().getType());
            if (!structure)
                return;
            SmallVector<Type> fields;
            llvm::transform(create.getFields(), std::back_inserter(fields),
                            [](Value field) { return field.getType(); });
            structFields[structure.getName()] = std::move(fields);
        });
        addVernonSharedValueTypeConversions(converter, kRegisterTensorElementLimit);
        converter.addConversion([&](TupleType tuple) -> std::optional<Type> {
            SmallVector<Type> elements;
            if (failed(converter.convertTypes(tuple.getTypes(), elements)))
                return std::nullopt;
            if (useSpirvTupleAbi)
                return spirv::StructType::get(elements);
            return LLVM::LLVMStructType::getLiteral(tuple.getContext(), elements);
        });
        converter.addConversion([&](TensorType tensor) -> std::optional<Type> {
            Type element = converter.convertType(tensor.getElementType());
            if (!element)
                return std::nullopt;
            int64_t count = 1;
            for (int64_t dimension : tensor.getShape())
                count *= dimension;
            if (useSpirvTupleAbi)
                return spirv::ArrayType::get(element, static_cast<unsigned>(count));
            return LLVM::LLVMArrayType::get(element, count);
        });
        converter.addConversion([&](StructType structure) -> std::optional<Type> {
            auto found = structFields.find(structure.getName());
            if (found == structFields.end())
                return std::nullopt;
            SmallVector<Type> fields;
            if (failed(converter.convertTypes(found->second, fields)))
                return std::nullopt;
            if (useSpirvTupleAbi)
                return spirv::StructType::get(fields);
            return LLVM::LLVMStructType::getLiteral(structure.getContext(), fields);
        });

        RewritePatternSet patterns(context);
        populateVernonSharedValuePatterns(converter, patterns);
        patterns.add<GpuAggregateTensorConstructPattern, GpuAggregateTensorGetPattern, GpuStructCreatePattern,
                     GpuStructGetPattern, GpuTupleCreatePattern, GpuTupleGetPattern>(converter, context,
                                                                                     useSpirvTupleAbi);
        populateFunctionOpInterfaceTypeConversionPattern(gpu::GPUFuncOp::getOperationName(), patterns, converter);

        ConversionTarget target(*context);
        target.addLegalDialect<vector::VectorDialect>();
        if (useSpirvTupleAbi)
            target.addLegalDialect<spirv::SPIRVDialect>();
        else
            target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalDialect<VernonDialect>();
        target.addDynamicallyLegalDialect<tensor::TensorDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *operation) {
            if (auto constant = dyn_cast<arith::ConstantOp>(operation))
                if (isa<RankedTensorType>(constant.getType()))
                    return false;
            return converter.isLegal(operation);
        });
        target.addDynamicallyLegalDialect<gpu::GPUDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addDynamicallyLegalDialect<math::MathDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
            [&](gpu::GPUFuncOp op) { return converter.isSignatureLegal(op.getFunctionType()); });
        target.markUnknownOpDynamicallyLegal([&](Operation *operation) { return converter.isLegal(operation); });
        populateVernonSharedValueStructuralTypeConversions(converter, patterns, target);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }

    bool useSpirvTupleAbi = false;
};

} // namespace

std::unique_ptr<Pass> createVernonLowerGPUTensorsPass(bool useSpirvTupleAbi) {
    return std::make_unique<VernonLowerGPUTensorsPass>(useSpirvTupleAbi);
}

void registerVernonLowerGPUTensorsPass() { PassRegistration<VernonLowerGPUTensorsPass>(); }

} // namespace mlir::vernon
