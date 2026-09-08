#include "mlir/Dialect/Vernon/Transforms/VernonConvertGPUToSPIRV.h"

#include "VernonSpirvMath.h"

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::vernon {
namespace {

bool isCasBackedF32MemRef(Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    if (!memref || !memref.getElementType().isF32())
        return false;
    auto storageClass = dyn_cast_or_null<spirv::StorageClassAttr>(memref.getMemorySpace());
    return storageClass && (storageClass.getValue() == spirv::StorageClass::StorageBuffer ||
                            storageClass.getValue() == spirv::StorageClass::Workgroup);
}

struct CasBackedLoadToSPIRVPattern final : OpConversionPattern<memref::LoadOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto memref = dyn_cast<MemRefType>(op.getMemRefType());
        if (!memref || !isCasBackedF32MemRef(memref))
            return failure();
        const auto &converter = *getTypeConverter<SPIRVTypeConverter>();
        Value pointer =
            spirv::getElementPtr(converter, memref, adaptor.getMemref(), adaptor.getIndices(), op.getLoc(), rewriter);
        auto pointerType = dyn_cast_or_null<spirv::PointerType>(pointer.getType());
        if (!pointerType || !pointerType.getPointeeType().isInteger(32))
            return failure();
        Value bits = spirv::LoadOp::create(rewriter, op.getLoc(), pointer);
        rewriter.replaceOpWithNewOp<spirv::BitcastOp>(op, rewriter.getF32Type(), bits);
        return success();
    }
};

struct CasBackedStoreToSPIRVPattern final : OpConversionPattern<memref::StoreOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto memref = dyn_cast<MemRefType>(op.getMemRefType());
        if (!memref || !isCasBackedF32MemRef(memref))
            return failure();
        const auto &converter = *getTypeConverter<SPIRVTypeConverter>();
        Value pointer =
            spirv::getElementPtr(converter, memref, adaptor.getMemref(), adaptor.getIndices(), op.getLoc(), rewriter);
        auto pointerType = dyn_cast_or_null<spirv::PointerType>(pointer.getType());
        if (!pointerType || !pointerType.getPointeeType().isInteger(32))
            return failure();
        Value bits = spirv::BitcastOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), adaptor.getValue());
        spirv::StoreOp::create(rewriter, op.getLoc(), pointer, bits);
        rewriter.eraseOp(op);
        return success();
    }
};

struct Atan2ToSPIRVPattern final : OpConversionPattern<math::Atan2Op> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(math::Atan2Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type resultType = getTypeConverter()->convertType(op.getType());
        if (!resultType)
            return failure();
        FailureOr<Value> result =
            lowerAtan2ToSpirv(op.getLoc(), resultType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
        if (failed(result))
            return failure();
        rewriter.replaceOp(op, *result);
        return success();
    }
};

struct AtomicRMWToSPIRVPattern final : OpConversionPattern<memref::AtomicRMWOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(memref::AtomicRMWOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        const bool exchange = op.getKind() == arith::AtomicRMWKind::assign;
        const bool floatAdd = op.getKind() == arith::AtomicRMWKind::addf && op.getValue().getType().isF32();
        if (!exchange && !floatAdd)
            return failure();
        auto memref = cast<MemRefType>(op.getMemref().getType());
        auto storageClass = dyn_cast_or_null<spirv::StorageClassAttr>(memref.getMemorySpace());
        if (!storageClass)
            return failure();
        std::optional<spirv::Scope> scope;
        if (storageClass.getValue() == spirv::StorageClass::StorageBuffer)
            scope = spirv::Scope::Device;
        else if (storageClass.getValue() == spirv::StorageClass::Workgroup)
            scope = spirv::Scope::Workgroup;
        if (!scope)
            return failure();

        const auto &converter = *getTypeConverter<SPIRVTypeConverter>();
        Type resultType = converter.convertType(op.getType());
        Value pointer =
            spirv::getElementPtr(converter, memref, adaptor.getMemref(), adaptor.getIndices(), op.getLoc(), rewriter);
        if (!resultType || !pointer)
            return failure();
        if (exchange) {
            rewriter.replaceOpWithNewOp<spirv::AtomicExchangeOp>(
                op, resultType, pointer, *scope, spirv::MemorySemantics::AcquireRelease, adaptor.getValue());
            return success();
        }
        auto implementation = op->getAttrOfType<StringAttr>(kAtomicImplementationAttrName);
        if (floatAdd && !implementation)
            return op.emitError("floating atomic add reached SPIR-V conversion without a selected implementation");
        if (floatAdd && implementation.getValue() == kNativeAtomicImplementation) {
            rewriter.replaceOpWithNewOp<spirv::EXTAtomicFAddOp>(op, resultType, pointer, *scope,
                                                                spirv::MemorySemantics::None, adaptor.getValue());
        } else if (floatAdd && implementation.getValue() == kIntegerCasAtomicImplementation) {
            auto function = op->getParentOfType<spirv::FuncOp>();
            auto integerPointer = dyn_cast<spirv::PointerType>(pointer.getType());
            if (!function || !integerPointer || !integerPointer.getPointeeType().isInteger(32))
                return failure();
            Type i32 = rewriter.getI32Type();
            Value zero = spirv::ConstantOp::create(rewriter, op.getLoc(), i32, rewriter.getI32IntegerAttr(0));
            Value initial = spirv::AtomicCompareExchangeOp::create(rewriter, op.getLoc(), i32, pointer, *scope,
                                                                   spirv::MemorySemantics::None,
                                                                   spirv::MemorySemantics::None, zero, zero);

            OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.front());
            auto resultPointer = spirv::PointerType::get(i32, spirv::StorageClass::Function);
            Value result = spirv::VariableOp::create(entryBuilder, op.getLoc(), resultPointer,
                                                     spirv::StorageClass::Function, nullptr);

            auto loop = spirv::LoopOp::create(rewriter, op.getLoc(), spirv::LoopControl::None);
            loop.addEntryAndMergeBlock(rewriter);
            {
                OpBuilder::InsertionGuard guard(rewriter);
                Region &body = loop.getBody();
                Block *entry = loop.getEntryBlock();
                Block *merge = loop.getMergeBlock();
                Block *header = rewriter.createBlock(&body, std::prev(body.end()));
                Block *success = rewriter.createBlock(&body, std::prev(body.end()));
                Block *retry = rewriter.createBlock(&body, std::prev(body.end()));
                BlockArgument expected = header->addArgument(i32, op.getLoc());
                BlockArgument observed = retry->addArgument(i32, op.getLoc());

                OpBuilder headerBuilder = OpBuilder::atBlockBegin(header);
                Value current = spirv::BitcastOp::create(headerBuilder, op.getLoc(), rewriter.getF32Type(), expected);
                Value sum = spirv::FAddOp::create(headerBuilder, op.getLoc(), rewriter.getF32Type(), current,
                                                  adaptor.getValue());
                Value desired = spirv::BitcastOp::create(headerBuilder, op.getLoc(), i32, sum);
                Value exchanged = spirv::AtomicCompareExchangeOp::create(
                    headerBuilder, op.getLoc(), i32, pointer, *scope, spirv::MemorySemantics::None,
                    spirv::MemorySemantics::None, desired, expected);
                Value matched =
                    spirv::IEqualOp::create(headerBuilder, op.getLoc(), rewriter.getI1Type(), exchanged, expected);
                spirv::BranchConditionalOp::create(headerBuilder, op.getLoc(), matched, success, ValueRange{}, retry,
                                                   ValueRange{exchanged});

                OpBuilder successBuilder = OpBuilder::atBlockBegin(success);
                spirv::StoreOp::create(successBuilder, op.getLoc(), result, exchanged);
                spirv::BranchOp::create(successBuilder, op.getLoc(), merge);

                OpBuilder retryBuilder = OpBuilder::atBlockBegin(retry);
                spirv::BranchOp::create(retryBuilder, op.getLoc(), header, ValueRange{observed});

                OpBuilder loopEntryBuilder = OpBuilder::atBlockBegin(entry);
                spirv::BranchOp::create(loopEntryBuilder, op.getLoc(), header, ValueRange{initial});
            }
            rewriter.setInsertionPointAfter(loop);
            Value previousBits = spirv::LoadOp::create(rewriter, op.getLoc(), result);
            rewriter.replaceOpWithNewOp<spirv::BitcastOp>(op, resultType, previousBits);
        } else
            return op.emitError("unsupported floating atomic implementation '") << implementation.getValue() << "'";
        return success();
    }
};

FailureOr<Type> convertStorageTextureType(TextureType texture) {
    if (texture.getAccess() == "sampled" || texture.getDimension() == "cube")
        return failure();
    std::optional<spirv::Dim> dimension = llvm::StringSwitch<std::optional<spirv::Dim>>(texture.getDimension())
                                              .Case("2d", spirv::Dim::Dim2D)
                                              .Case("3d", spirv::Dim::Dim3D)
                                              .Default(std::nullopt);
    std::optional<spirv::ImageFormat> format =
        llvm::StringSwitch<std::optional<spirv::ImageFormat>>(texture.getFormat())
            .Case("r8_unorm", spirv::ImageFormat::R8)
            .Case("r16_float", spirv::ImageFormat::R16f)
            .Case("r32_float", spirv::ImageFormat::R32f)
            .Case("rg8_unorm", spirv::ImageFormat::Rg8)
            .Case("rgba8_unorm", spirv::ImageFormat::Rgba8)
            .Case("rgba16_float", spirv::ImageFormat::Rgba16f)
            .Case("rgba32_float", spirv::ImageFormat::Rgba32f)
            .Default(std::nullopt);
    if (!dimension || !format)
        return failure();
    Type image = spirv::ImageType::get(texture.getElementType(), *dimension, spirv::ImageDepthInfo::NoDepth,
                                       spirv::ImageArrayedInfo::NonArrayed, spirv::ImageSamplingInfo::SingleSampled,
                                       spirv::ImageSamplerUseInfo::NoSampler, *format);
    return spirv::PointerType::get(image, spirv::StorageClass::UniformConstant);
}

struct StorageTextureIntrinsicToSPIRVPattern final : OpConversionPattern<IntrinsicOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (adaptor.getOperands().empty())
            return failure();
        Value image = adaptor.getOperands()[0];
        auto pointer = dyn_cast<spirv::PointerType>(image.getType());
        if (!pointer || !isa<spirv::ImageType>(pointer.getPointeeType()))
            return failure();
        image = spirv::LoadOp::create(rewriter, op.getLoc(), image);
        if (op.getName() == "texture_load") {
            Type resultType = getTypeConverter()->convertType(op.getResult().getType());
            if (!resultType || adaptor.getOperands().size() != 2)
                return failure();
            rewriter.replaceOpWithNewOp<spirv::ImageReadOp>(op, resultType, image, adaptor.getOperands()[1],
                                                            spirv::ImageOperandsAttr(), ValueRange{});
            return success();
        }
        if (op.getName() == "texture_store") {
            if (adaptor.getOperands().size() != 3)
                return failure();
            spirv::ImageWriteOp::create(rewriter, op.getLoc(), image, adaptor.getOperands()[1],
                                        adaptor.getOperands()[2], spirv::ImageOperandsAttr(), ValueRange{});
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

// MLIR's generic SPIR-V ABI lowering materializes SampledImage arguments but
// intentionally does not support unsampled Image arguments. Materialize those
// opaque UniformConstant resources here so the generic pass only receives ABI
// argument kinds it supports.
LogicalResult materializeStorageImageInterfaceVariables(ModuleOp module) {
    SmallVector<spirv::FuncOp> functions;
    module.walk([&](spirv::FuncOp function) { functions.push_back(function); });
    const StringRef abiName = spirv::getInterfaceVarABIAttrName();
    for (spirv::FuncOp function : functions) {
        llvm::BitVector erase(function.getNumArguments());
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto pointer = dyn_cast<spirv::PointerType>(function.getArgument(index).getType());
            if (!pointer || !isa<spirv::ImageType>(pointer.getPointeeType()) ||
                pointer.getStorageClass() != spirv::StorageClass::UniformConstant)
                continue;
            auto abi = function.getArgAttrOfType<spirv::InterfaceVarABIAttr>(index, abiName);
            if (!abi)
                return function.emitError() << "storage image argument #" << index << " has no interface ABI";
            OpBuilder moduleBuilder(function);
            const std::string name = (function.getName() + "_storage_image_" + Twine(index)).str();
            auto global = spirv::GlobalVariableOp::create(moduleBuilder, function.getLoc(), pointer, name,
                                                          abi.getDescriptorSet(), abi.getBinding());
            OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&function.front());
            Value address = spirv::AddressOfOp::create(bodyBuilder, function.getLoc(), global);
            function.getArgument(index).replaceAllUsesWith(address);
            erase.set(index);
        }
        if (erase.any() && failed(function.eraseArguments(erase)))
            return function.emitError("cannot remove materialized storage image arguments");
    }
    return success();
}

struct ConvertGPUToSPIRVPass final : PassWrapper<ConvertGPUToSPIRVPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPUToSPIRVPass)

    ConvertGPUToSPIRVPass() = default;
    ConvertGPUToSPIRVPass(const ConvertGPUToSPIRVPass &other) : PassWrapper(other) {}

    StringRef getArgument() const override { return "vernon-convert-gpu-to-spirv"; }
    StringRef getDescription() const override {
        return "Convert GPU modules to SPIR-V with Vernon atomic exchange support";
    }

    void runOnOperation() override {
        SmallVector<gpu::GPUModuleOp> gpuModules;
        getOperation().walk([&](gpu::GPUModuleOp module) { gpuModules.push_back(module); });
        for (gpu::GPUModuleOp gpuModule : gpuModules) {
            OpBuilder builder(gpuModule);
            auto clone = cast<gpu::GPUModuleOp>(builder.clone(*gpuModule.getOperation()));
            bool usesIntegerCasStorage = false;
            clone.walk([&](memref::AtomicRMWOp atomic) {
                auto implementation = atomic->getAttrOfType<StringAttr>(kAtomicImplementationAttrName);
                usesIntegerCasStorage |= implementation && implementation.getValue() == kIntegerCasAtomicImplementation;
            });
            spirv::TargetEnvAttr targetEnvironment = spirv::lookupTargetEnvOrDefault(clone);
            std::unique_ptr<ConversionTarget> target = SPIRVConversionTarget::get(targetEnvironment);
            SPIRVTypeConverter typeConverter(targetEnvironment);
            populateMMAToSPIRVCoopMatrixTypeConversion(typeConverter);
            if (usesIntegerCasStorage) {
                typeConverter.addConversion([&](MemRefType memref) -> std::optional<Type> {
                    if (!isCasBackedF32MemRef(memref))
                        return std::nullopt;
                    MemRefType integerStorage =
                        MemRefType::get(memref.getShape(), IntegerType::get(memref.getContext(), 32),
                                        memref.getLayout(), memref.getMemorySpace());
                    return typeConverter.convertType(integerStorage);
                });
            }
            typeConverter.addConversion([](TextureType texture) -> std::optional<Type> {
                FailureOr<Type> converted = convertStorageTextureType(texture);
                return succeeded(converted) ? std::optional<Type>(*converted) : std::nullopt;
            });

            RewritePatternSet patterns(&getContext());
            populateGPUToSPIRVPatterns(typeConverter, patterns);
            populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(typeConverter, patterns);
            ScfToSPIRVContext scfContext;
            populateSCFToSPIRVPatterns(typeConverter, scfContext, patterns);
            arith::populateArithToSPIRVPatterns(typeConverter, patterns);
            populateMathToSPIRVPatterns(typeConverter, patterns);
            populateMemRefToSPIRVPatterns(typeConverter, patterns);
            populateFuncToSPIRVPatterns(typeConverter, patterns);
            populateVectorToSPIRVPatterns(typeConverter, patterns);
            patterns.add<Atan2ToSPIRVPattern>(typeConverter, &getContext(), PatternBenefit(2));
            patterns.add<AtomicRMWToSPIRVPattern>(typeConverter, &getContext(), PatternBenefit(2));
            patterns.add<StorageTextureIntrinsicToSPIRVPattern>(typeConverter, &getContext());
            if (usesIntegerCasStorage) {
                patterns.add<CasBackedLoadToSPIRVPattern, CasBackedStoreToSPIRVPattern>(typeConverter, &getContext(),
                                                                                        PatternBenefit(2));
            }

            if (failed(applyFullConversion(clone, *target, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
        }
        if (failed(materializeStorageImageInterfaceVariables(getOperation())))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonConvertGPUToSPIRVPass() { return std::make_unique<ConvertGPUToSPIRVPass>(); }

void registerVernonConvertGPUToSPIRVPass() { PassRegistration<ConvertGPUToSPIRVPass>(); }

} // namespace mlir::vernon
