#include "mlir/Dialect/Vernon/Transforms/VernonConvertGPUToSPIRV.h"

#include "VernonSpirvImageFormat.h"
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
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/BitVector.h"

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

struct TextureIntrinsicToSPIRVPattern final : OpConversionPattern<IntrinsicOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (adaptor.getOperands().empty())
            return failure();
        Value descriptor = adaptor.getOperands()[0];
        auto pointer = dyn_cast<spirv::PointerType>(descriptor.getType());
        if (!pointer || !isa<spirv::ImageType, spirv::SampledImageType>(pointer.getPointeeType()))
            return failure();
        const bool sampled = isa<spirv::SampledImageType>(pointer.getPointeeType());
        const StringRef name = op.getName();
        const size_t operandCount = adaptor.getOperands().size();
        const bool matches =
            sampled ? name == "texture_sample" && operandCount == 4
                    : (name == "texture_load" && operandCount == 2) || (name == "texture_store" && operandCount == 3);
        if (!matches)
            return failure();
        Type resultType = op.getNumResults() == 1 ? getTypeConverter()->convertType(op.getResult().getType()) : Type();
        if (op.getNumResults() == 1 && !resultType)
            return failure();

        Value image = spirv::LoadOp::create(rewriter, op.getLoc(), descriptor);
        if (sampled) {
            rewriter.replaceOp(op, buildSpirvTextureSample(rewriter, op.getLoc(), resultType, image,
                                                           adaptor.getOperands()[2], adaptor.getOperands()[3]));
        } else if (name == "texture_load") {
            rewriter.replaceOp(
                op, buildSpirvTextureLoad(rewriter, op.getLoc(), resultType, image, adaptor.getOperands()[1]));
        } else {
            spirv::ImageWriteOp::create(rewriter, op.getLoc(), image, adaptor.getOperands()[1],
                                        adaptor.getOperands()[2], spirv::ImageOperandsAttr(), ValueRange{});
            rewriter.eraseOp(op);
        }
        return success();
    }
};

// A sampler binds through the combined image-sampler of the texture it samples, so it has no SPIR-V value. It stays a
// kernel argument until conversion because texture_sample needs it as an operand and the function conversion is
// one-to-one; this pass-private mark survives the conversion, which replaces the sampler type with a placeholder.
constexpr llvm::StringLiteral kCombinedSamplerAttrName = "vernon.combined_sampler";

/* Marks every sampler argument after checking that the combined image-sampler model can lower its uses: the sampler
 * placeholder has no SPIR-V value, so it may reach only the sampler operand of texture_sample, and compute has no
 * implicit derivatives, so a sampled texture is only ever sampled at an explicit lod. */
LogicalResult prepareComputeSampling(gpu::GPUModuleOp module) {
    WalkResult result = module.walk([](Operation *op) -> WalkResult {
        if (auto function = dyn_cast<gpu::GPUFuncOp>(op)) {
            for (BlockArgument argument : function.getArguments()) {
                if (!isa<SamplerType>(argument.getType()))
                    continue;
                for (OpOperand &use : argument.getUses()) {
                    auto intrinsic = dyn_cast<IntrinsicOp>(use.getOwner());
                    if (!intrinsic || intrinsic.getName() != "texture_sample" || use.getOperandNumber() != 1) {
                        use.getOwner()->emitError() << "compute sampler argument #" << argument.getArgNumber()
                                                    << " may be used only as the sampler of texture_sample";
                        return WalkResult::interrupt();
                    }
                }
                function.setArgAttr(argument.getArgNumber(), kCombinedSamplerAttrName,
                                    UnitAttr::get(function.getContext()));
            }
            return WalkResult::advance();
        }
        auto intrinsic = dyn_cast<IntrinsicOp>(op);
        auto texture = intrinsic && intrinsic.getNumOperands() != 0
                           ? dyn_cast<TextureType>(intrinsic.getOperand(0).getType())
                           : TextureType();
        if (!texture || texture.getAccess() != "sampled")
            return WalkResult::advance();
        if (intrinsic.getName() != "texture_sample") {
            intrinsic.emitOpError() << intrinsic.getName()
                                    << " from a sampled texture is supported only in graphics stages";
            return WalkResult::interrupt();
        }
        if (intrinsic.getNumOperands() != 4) {
            intrinsic.emitOpError("texture_sample without lod is supported only in graphics stages");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

LogicalResult eraseCombinedSamplerArguments(ModuleOp module) {
    SmallVector<spirv::FuncOp> functions;
    module.walk([&](spirv::FuncOp function) { functions.push_back(function); });
    for (spirv::FuncOp function : functions) {
        llvm::BitVector erase(function.getNumArguments());
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (!function.getArgAttr(index, kCombinedSamplerAttrName))
                continue;
            if (!function.getArgument(index).use_empty())
                return function.emitError() << "combined sampler argument #" << index << " is still used";
            erase.set(index);
        }
        if (erase.any() && failed(function.eraseArguments(erase)))
            return function.emitError("cannot remove combined sampler arguments");
    }
    return success();
}

// Aggregate metadata is one explicit Uniform block. The generic SPIR-V ABI
// lowering can assign storage classes only to scalar function arguments, so
// materialize the compiler-owned block before that pass sees the function.
LogicalResult materializeMetadataInterfaceVariables(ModuleOp module) {
    SmallVector<spirv::FuncOp> functions;
    module.walk([&](spirv::FuncOp function) { functions.push_back(function); });
    const StringRef abiName = spirv::getInterfaceVarABIAttrName();
    for (spirv::FuncOp function : functions) {
        llvm::BitVector erase(function.getNumArguments());
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (!function.getArgAttr(index, kTensorMetadataCarrierAttrName))
                continue;
            auto valueType = dyn_cast<spirv::StructType>(function.getArgument(index).getType());
            auto abi = function.getArgAttrOfType<spirv::InterfaceVarABIAttr>(index, abiName);
            auto profile = function.getArgAttrOfType<StringAttr>(index, kTensorMetadataProfileAttrName);
            if (!valueType || !abi || !profile || profile.getValue() != "portable_shader_metadata_i32")
                return function.emitError()
                       << "metadata argument #" << index << " has no portable shader aggregate ABI";

            OpBuilder moduleBuilder(function);
            SmallVector<spirv::StructType::MemberDecorationInfo> memberDecorations;
            valueType.getMemberDecorations(memberDecorations);
            SmallVector<spirv::StructType::StructDecorationInfo> structDecorations;
            structDecorations.emplace_back(spirv::Decoration::Block, moduleBuilder.getUnitAttr());
            SmallVector<uint32_t> memberOffsets;
            SmallVector<Type> memberTypes(valueType.getElementTypes());
            memberOffsets.reserve(valueType.getNumElements());
            for (unsigned member = 0; member < valueType.getNumElements(); ++member)
                memberOffsets.push_back(static_cast<uint32_t>(valueType.getMemberOffset(member)));
            auto blockType = spirv::StructType::get(memberTypes, memberOffsets, memberDecorations, structDecorations);
            auto pointerType = spirv::PointerType::get(blockType, spirv::StorageClass::Uniform);
            const std::string name = (function.getName() + "_metadata").str();
            auto global = spirv::GlobalVariableOp::create(moduleBuilder, function.getLoc(), pointerType, name,
                                                          abi.getDescriptorSet(), abi.getBinding());

            OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&function.front());
            Value address = spirv::AddressOfOp::create(bodyBuilder, function.getLoc(), global);
            SmallVector<spirv::CompositeExtractOp> reads;
            for (Operation *user : function.getArgument(index).getUsers()) {
                auto read = dyn_cast<spirv::CompositeExtractOp>(user);
                if (!read || read.getComposite() != function.getArgument(index) || read.getIndices().size() != 1)
                    return function.emitError() << "metadata argument #" << index << " has a non-canonical use";
                reads.push_back(read);
            }
            for (spirv::CompositeExtractOp read : reads) {
                auto member = dyn_cast<IntegerAttr>(read.getIndices()[0]);
                if (!member)
                    return read.emitError("metadata field index is not an integer");
                OpBuilder readBuilder(read);
                Value memberIndex = spirv::ConstantOp::create(readBuilder, read.getLoc(), readBuilder.getI32Type(),
                                                              readBuilder.getI32IntegerAttr(member.getInt()));
                Value pointer =
                    spirv::AccessChainOp::create(readBuilder, read.getLoc(), address, ValueRange{memberIndex});
                Value loaded = spirv::LoadOp::create(readBuilder, read.getLoc(), pointer);
                read.getResult().replaceAllUsesWith(loaded);
                read.erase();
            }
            erase.set(index);
        }
        if (erase.any() && failed(function.eraseArguments(erase)))
            return function.emitError("cannot remove materialized metadata arguments");
    }
    return success();
}

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
            auto access = function.getArgAttrOfType<StringAttr>(index, "vernon.texture_access");
            OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&function.front());
            auto global = spirv::GlobalVariableOp::create(moduleBuilder, function.getLoc(), pointer, name,
                                                          abi.getDescriptorSet(), abi.getBinding());
            if (access && access.getValue() == "read")
                global->setAttr("non_writable", moduleBuilder.getUnitAttr());
            else if (access && access.getValue() == "write")
                global->setAttr("non_readable", moduleBuilder.getUnitAttr());
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
            if (failed(prepareComputeSampling(clone))) {
                signalPassFailure();
                return;
            }
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
                std::optional<Type> descriptor = spirvTextureDescriptorType(texture);
                if (!descriptor)
                    return std::nullopt;
                return spirv::PointerType::get(*descriptor, spirv::StorageClass::UniformConstant);
            });
            typeConverter.addConversion(
                [](SamplerType sampler) -> Type { return IntegerType::get(sampler.getContext(), 32); });

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
            patterns.add<TextureIntrinsicToSPIRVPattern>(typeConverter, &getContext());
            if (usesIntegerCasStorage) {
                patterns.add<CasBackedLoadToSPIRVPattern, CasBackedStoreToSPIRVPattern>(typeConverter, &getContext(),
                                                                                        PatternBenefit(2));
            }

            if (failed(applyFullConversion(clone, *target, std::move(patterns)))) {
                signalPassFailure();
                return;
            }
        }
        if (failed(eraseCombinedSamplerArguments(getOperation())) ||
            failed(materializeMetadataInterfaceVariables(getOperation())) ||
            failed(materializeStorageImageInterfaceVariables(getOperation())))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonConvertGPUToSPIRVPass() { return std::make_unique<ConvertGPUToSPIRVPass>(); }

void registerVernonConvertGPUToSPIRVPass() { PassRegistration<ConvertGPUToSPIRVPass>(); }

} // namespace mlir::vernon
