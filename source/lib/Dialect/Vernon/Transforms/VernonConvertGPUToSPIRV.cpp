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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::vernon {
namespace {

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

struct AtomicExchangeToSPIRVPattern final : OpConversionPattern<memref::AtomicRMWOp> {
    AtomicExchangeToSPIRVPattern(const SPIRVTypeConverter &converter, MLIRContext *context)
        : OpConversionPattern(converter, context, PatternBenefit(2)) {}

    LogicalResult matchAndRewrite(memref::AtomicRMWOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op.getKind() != arith::AtomicRMWKind::assign)
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
        rewriter.replaceOpWithNewOp<spirv::AtomicExchangeOp>(
            op, resultType, pointer, *scope, spirv::MemorySemantics::AcquireRelease, adaptor.getValue());
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
            spirv::TargetEnvAttr targetEnvironment = spirv::lookupTargetEnvOrDefault(clone);
            std::unique_ptr<ConversionTarget> target = SPIRVConversionTarget::get(targetEnvironment);
            SPIRVTypeConverter typeConverter(targetEnvironment);
            populateMMAToSPIRVCoopMatrixTypeConversion(typeConverter);
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
            patterns.add<AtomicExchangeToSPIRVPattern>(typeConverter, &getContext());
            patterns.add<StorageTextureIntrinsicToSPIRVPattern>(typeConverter, &getContext());

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
