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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

            if (failed(applyFullConversion(clone, *target, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
        }
    }
};

} // namespace

std::unique_ptr<Pass> createVernonConvertGPUToSPIRVPass() { return std::make_unique<ConvertGPUToSPIRVPass>(); }

void registerVernonConvertGPUToSPIRVPass() { PassRegistration<ConvertGPUToSPIRVPass>(); }

} // namespace mlir::vernon
