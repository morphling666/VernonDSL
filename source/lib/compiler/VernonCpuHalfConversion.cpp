#include "VernonCpuHalfConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/X86TargetParser.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace vernon {
namespace {

// These helpers implement IEEE-754 widening and round-to-nearest-even narrowing
// with integer representations. The cases mirror LLVM compiler-rt's generic
// fp_extend/fp_trunc algorithms, but use a Vernon-private integer ABI so cooked
// objects never depend on target-specific half-conversion libcalls.
llvm::ConstantInt *integer(llvm::IntegerType *type, uint64_t value) { return llvm::ConstantInt::get(type, value); }

struct HalfConversionHelpers {
    llvm::Function *halfToFloat{};
    llvm::Function *halfToDouble{};
    llvm::Function *floatToHalf{};
    llvm::Function *doubleToHalf{};
};

bool featureEnabled(llvm::StringRef featureString, llvm::StringRef feature, bool defaultValue) {
    llvm::SmallVector<llvm::StringRef> entries;
    featureString.split(entries, ',', -1, false);
    bool enabled = defaultValue;
    for (llvm::StringRef entry : entries) {
        entry = entry.trim();
        const bool setting = !entry.consume_front("-");
        entry.consume_front("+");
        if (entry == feature)
            enabled = setting;
    }
    return enabled;
}

bool hasNativeHalfOperations(const llvm::TargetMachine &targetMachine) {
    const llvm::Triple &triple = targetMachine.getTargetTriple();
    const llvm::StringRef cpu = targetMachine.getTargetCPU();
    const llvm::StringRef features = targetMachine.getTargetFeatureString();
    if (triple.isAArch64()) {
        bool native = triple.isOSDarwin();
        if (!cpu.empty() && cpu != "generic")
            if (std::optional<llvm::AArch64::CpuInfo> info = llvm::AArch64::parseCpu(cpu)) {
                std::vector<llvm::StringRef> defaults;
                if (llvm::AArch64::getExtensionFeatures(info->DefaultExtensions, defaults))
                    native = llvm::is_contained(defaults, "+fullfp16");
            }
        native = featureEnabled(features, "fp16", native);
        return featureEnabled(features, "fullfp16", native);
    }
    if (triple.isX86()) {
        llvm::SmallVector<llvm::StringRef> defaults;
        if (!cpu.empty() && cpu != "generic")
            llvm::X86::getFeaturesForCPU(cpu, defaults);
        const bool native = llvm::is_contained(defaults, "avx512fp16");
        return featureEnabled(features, "avx512fp16", native);
    }
    return false;
}

llvm::Function *defineHalfToWideHelper(llvm::Module &module, llvm::Type *destinationType,
                                       HalfConversionHelpers &helpers) {
    llvm::LLVMContext &context = module.getContext();
    const bool toDouble = destinationType->isDoubleTy();
    auto *sourceInteger = llvm::Type::getInt16Ty(context);
    auto *destinationInteger =
        llvm::IntegerType::get(context, destinationType->getPrimitiveSizeInBits().getFixedValue());
    const unsigned destinationBits = destinationInteger->getBitWidth();
    const unsigned destinationExponentBits = toDouble ? 11 : 8;
    const unsigned destinationFractionBits = toDouble ? 52 : 23;
    const unsigned destinationExponentBias = toDouble ? 1023 : 127;
    const char *name = toDouble ? "__vernon_cpu_f16_to_f64_bits" : "__vernon_cpu_f16_to_f32_bits";
    llvm::Function *&cached = toDouble ? helpers.halfToDouble : helpers.halfToFloat;
    if (cached)
        return cached;

    auto *functionType = llvm::FunctionType::get(destinationInteger, {sourceInteger}, false);
    llvm::Function *function = llvm::Function::Create(functionType, llvm::GlobalValue::InternalLinkage, name, module);
    cached = function;
    function->addFnAttr(llvm::Attribute::NoUnwind);
    function->addFnAttr(llvm::Attribute::WillReturn);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", function);
    llvm::IRBuilder<> builder(entry);
    llvm::Value *bits = function->getArg(0);

    llvm::Value *sign = builder.CreateZExt(builder.CreateLShr(bits, integer(sourceInteger, 15)), destinationInteger);
    sign = builder.CreateShl(sign, integer(destinationInteger, destinationBits - 1));
    llvm::Value *exponent =
        builder.CreateAnd(builder.CreateLShr(bits, integer(sourceInteger, 10)), integer(sourceInteger, 0x1f));
    llvm::Value *fraction = builder.CreateAnd(bits, integer(sourceInteger, 0x3ff));
    llvm::Value *wideExponent = builder.CreateZExt(exponent, destinationInteger);
    llvm::Value *wideFraction = builder.CreateZExt(fraction, destinationInteger);

    llvm::Value *normalExponent =
        builder.CreateAdd(wideExponent, integer(destinationInteger, destinationExponentBias - 15));
    llvm::Value *normal = builder.CreateOr(
        sign,
        builder.CreateOr(builder.CreateShl(normalExponent, integer(destinationInteger, destinationFractionBits)),
                         builder.CreateShl(wideFraction, integer(destinationInteger, destinationFractionBits - 10))));

    llvm::Value *special = builder.CreateOr(
        sign,
        builder.CreateOr(
            integer(destinationInteger, ((uint64_t{1} << destinationExponentBits) - 1) << destinationFractionBits),
            builder.CreateShl(wideFraction, integer(destinationInteger, destinationFractionBits - 10))));

    llvm::Function *countLeadingZeros =
        llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::ctlz, sourceInteger);
    llvm::Value *leadingZeros = builder.CreateCall(countLeadingZeros, {fraction, llvm::ConstantInt::getFalse(context)});
    llvm::Value *scale = builder.CreateSub(leadingZeros, integer(sourceInteger, 5));
    llvm::Value *denormalExponent = builder.CreateSub(integer(destinationInteger, destinationExponentBias - 14),
                                                      builder.CreateZExt(scale, destinationInteger));
    llvm::Value *denormalFraction =
        builder.CreateShl(wideFraction, builder.CreateAdd(integer(destinationInteger, destinationFractionBits - 10),
                                                          builder.CreateZExt(scale, destinationInteger)));
    denormalFraction =
        builder.CreateXor(denormalFraction, integer(destinationInteger, uint64_t{1} << destinationFractionBits));
    llvm::Value *denormal = builder.CreateOr(
        sign,
        builder.CreateOr(builder.CreateShl(denormalExponent, integer(destinationInteger, destinationFractionBits)),
                         denormalFraction));

    llvm::Value *zeroExponent = builder.CreateICmpEQ(exponent, integer(sourceInteger, 0));
    llvm::Value *zeroFraction = builder.CreateICmpEQ(fraction, integer(sourceInteger, 0));
    llvm::Value *infiniteExponent = builder.CreateICmpEQ(exponent, integer(sourceInteger, 0x1f));
    llvm::Value *finite = builder.CreateSelect(zeroFraction, sign, denormal);
    llvm::Value *nonSpecial = builder.CreateSelect(zeroExponent, finite, normal);
    builder.CreateRet(builder.CreateSelect(infiniteExponent, special, nonSpecial));
    return function;
}

llvm::Function *defineWideToHalfHelper(llvm::Module &module, llvm::Type *sourceType, HalfConversionHelpers &helpers) {
    llvm::LLVMContext &context = module.getContext();
    const bool fromDouble = sourceType->isDoubleTy();
    auto *sourceInteger = llvm::IntegerType::get(context, sourceType->getPrimitiveSizeInBits().getFixedValue());
    auto *destinationInteger = llvm::Type::getInt16Ty(context);
    const unsigned sourceBits = sourceInteger->getBitWidth();
    const unsigned sourceExponentBits = fromDouble ? 11 : 8;
    const unsigned sourceFractionBits = fromDouble ? 52 : 23;
    const unsigned sourceExponentBias = fromDouble ? 1023 : 127;
    const unsigned fractionTailBits = sourceFractionBits - 10;
    const unsigned sourceInfinityExponent = (1u << sourceExponentBits) - 1;
    const unsigned overflowExponent = sourceExponentBias + 31 - 15;
    const char *name = fromDouble ? "__vernon_cpu_f64_to_f16_bits" : "__vernon_cpu_f32_to_f16_bits";
    llvm::Function *&cached = fromDouble ? helpers.doubleToHalf : helpers.floatToHalf;
    if (cached)
        return cached;

    auto *functionType = llvm::FunctionType::get(destinationInteger, {sourceInteger}, false);
    llvm::Function *function = llvm::Function::Create(functionType, llvm::GlobalValue::InternalLinkage, name, module);
    cached = function;
    function->addFnAttr(llvm::Attribute::NoUnwind);
    function->addFnAttr(llvm::Attribute::WillReturn);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", function);
    llvm::IRBuilder<> builder(entry);
    llvm::Value *bits = function->getArg(0);

    llvm::Value *sign =
        builder.CreateTrunc(builder.CreateLShr(bits, integer(sourceInteger, sourceBits - 1)), destinationInteger);
    sign = builder.CreateShl(sign, integer(destinationInteger, 15));
    llvm::Value *exponentBits = builder.CreateAnd(builder.CreateLShr(bits, integer(sourceInteger, sourceFractionBits)),
                                                  integer(sourceInteger, sourceInfinityExponent));
    llvm::Value *fraction = builder.CreateAnd(bits, integer(sourceInteger, (uint64_t{1} << sourceFractionBits) - 1));
    llvm::Value *exponent = builder.CreateZExtOrTrunc(exponentBits, builder.getInt32Ty());
    llvm::Value *destinationExponentCandidate =
        builder.CreateAdd(builder.CreateSub(exponent, builder.getInt32(sourceExponentBias)), builder.getInt32(15));

    const uint64_t roundMask = (uint64_t{1} << fractionTailBits) - 1;
    const uint64_t halfway = uint64_t{1} << (fractionTailBits - 1);
    llvm::Value *truncatedFraction =
        builder.CreateTrunc(builder.CreateLShr(fraction, integer(sourceInteger, fractionTailBits)), destinationInteger);
    llvm::Value *roundBits = builder.CreateAnd(fraction, integer(sourceInteger, roundMask));
    llvm::Value *roundAbove = builder.CreateICmpUGT(roundBits, integer(sourceInteger, halfway));
    llvm::Value *roundTie =
        builder.CreateAnd(builder.CreateICmpEQ(roundBits, integer(sourceInteger, halfway)),
                          builder.CreateICmpNE(builder.CreateAnd(truncatedFraction, integer(destinationInteger, 1)),
                                               integer(destinationInteger, 0)));
    llvm::Value *roundedFraction = builder.CreateAdd(
        truncatedFraction, builder.CreateZExt(builder.CreateOr(roundAbove, roundTie), destinationInteger));
    llvm::Value *normalCarry = builder.CreateICmpUGE(roundedFraction, integer(destinationInteger, 0x400));
    llvm::Value *normalExponent =
        builder.CreateAdd(destinationExponentCandidate, builder.CreateZExt(normalCarry, builder.getInt32Ty()));
    llvm::Value *normalFraction = builder.CreateSelect(
        normalCarry, builder.CreateXor(roundedFraction, integer(destinationInteger, 0x400)), roundedFraction);
    llvm::Value *normalMagnitude = builder.CreateOr(
        builder.CreateShl(builder.CreateTrunc(normalExponent, destinationInteger), integer(destinationInteger, 10)),
        normalFraction);

    const uint64_t sourceQuietNaN = uint64_t{1} << (sourceFractionBits - 1);
    const uint64_t sourceNaNCode = sourceQuietNaN - 1;
    llvm::Value *nanPayload =
        builder.CreateAnd(builder.CreateLShr(builder.CreateAnd(fraction, integer(sourceInteger, sourceNaNCode)),
                                             integer(sourceInteger, fractionTailBits)),
                          integer(sourceInteger, 0x1ff));
    llvm::Value *nanMagnitude =
        builder.CreateOr(integer(destinationInteger, 0x7e00), builder.CreateTrunc(nanPayload, destinationInteger));

    llvm::Value *sourceNormal = builder.CreateICmpNE(exponent, builder.getInt32(0));
    llvm::Value *significand = builder.CreateOr(
        fraction, builder.CreateSelect(sourceNormal, integer(sourceInteger, uint64_t{1} << sourceFractionBits),
                                       integer(sourceInteger, 0)));
    llvm::Value *shift = builder.CreateSub(builder.getInt32(sourceExponentBias - 15), exponent);
    shift = builder.CreateAdd(shift, builder.CreateZExt(sourceNormal, builder.getInt32Ty()));
    llvm::Value *negativeShift = builder.CreateICmpSLT(shift, builder.getInt32(0));
    llvm::Value *safeShift = builder.CreateSelect(negativeShift, builder.getInt32(0), shift);
    safeShift = builder.CreateSelect(builder.CreateICmpSGT(safeShift, builder.getInt32(sourceFractionBits)),
                                     builder.getInt32(sourceFractionBits), safeShift);
    llvm::Value *wideShift = builder.CreateZExtOrTrunc(safeShift, sourceInteger);
    llvm::Value *rightShifted = builder.CreateLShr(significand, wideShift);
    llvm::Value *leftShift =
        builder.CreateSelect(builder.CreateICmpEQ(safeShift, builder.getInt32(0)), builder.getInt32(0),
                             builder.CreateSub(builder.getInt32(sourceBits), safeShift));
    llvm::Value *sticky = builder.CreateAnd(
        builder.CreateICmpNE(safeShift, builder.getInt32(0)),
        builder.CreateICmpNE(builder.CreateShl(significand, builder.CreateZExtOrTrunc(leftShift, sourceInteger)),
                             integer(sourceInteger, 0)));
    llvm::Value *denormalized = builder.CreateOr(rightShifted, builder.CreateZExt(sticky, sourceInteger));
    llvm::Value *denormalFraction = builder.CreateTrunc(
        builder.CreateLShr(denormalized, integer(sourceInteger, fractionTailBits)), destinationInteger);
    llvm::Value *denormalRoundBits = builder.CreateAnd(denormalized, integer(sourceInteger, roundMask));
    llvm::Value *denormalRoundAbove = builder.CreateICmpUGT(denormalRoundBits, integer(sourceInteger, halfway));
    llvm::Value *denormalRoundTie =
        builder.CreateAnd(builder.CreateICmpEQ(denormalRoundBits, integer(sourceInteger, halfway)),
                          builder.CreateICmpNE(builder.CreateAnd(denormalFraction, integer(destinationInteger, 1)),
                                               integer(destinationInteger, 0)));
    llvm::Value *roundedDenormal =
        builder.CreateAdd(denormalFraction, builder.CreateZExt(builder.CreateOr(denormalRoundAbove, denormalRoundTie),
                                                               destinationInteger));
    llvm::Value *underflowMagnitude =
        builder.CreateSelect(builder.CreateICmpSGT(shift, builder.getInt32(sourceFractionBits)),
                             integer(destinationInteger, 0), roundedDenormal);

    llvm::Value *normalRange =
        builder.CreateAnd(builder.CreateICmpSGE(destinationExponentCandidate, builder.getInt32(1)),
                          builder.CreateICmpSLT(destinationExponentCandidate, builder.getInt32(31)));
    llvm::Value *isNaN = builder.CreateAnd(builder.CreateICmpEQ(exponent, builder.getInt32(sourceInfinityExponent)),
                                           builder.CreateICmpNE(fraction, integer(sourceInteger, 0)));
    llvm::Value *overflow = builder.CreateICmpUGE(exponent, builder.getInt32(overflowExponent));
    llvm::Value *magnitude = builder.CreateSelect(
        normalRange, normalMagnitude,
        builder.CreateSelect(isNaN, nanMagnitude,
                             builder.CreateSelect(overflow, integer(destinationInteger, 0x7c00), underflowMagnitude)));
    builder.CreateRet(builder.CreateOr(sign, magnitude));
    return function;
}

llvm::Type *scalarType(llvm::Type *type) {
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type))
        return vector->getElementType();
    return type->isVectorTy() ? nullptr : type;
}

llvm::Type *elementType(llvm::Type *type) {
    if (auto *vector = llvm::dyn_cast<llvm::VectorType>(type))
        return vector->getElementType();
    return type;
}

bool instructionTouchesHalf(const llvm::Instruction &instruction) {
    if (elementType(instruction.getType())->isHalfTy())
        return true;
    return llvm::any_of(instruction.operands(),
                        [](const llvm::Use &operand) { return elementType(operand->getType())->isHalfTy(); });
}

llvm::Error validateSoftwareHalfOperations(llvm::Module &module) {
    for (llvm::Function &function : module)
        for (llvm::BasicBlock &block : function)
            for (llvm::Instruction &instruction : block) {
                if (!instructionTouchesHalf(instruction))
                    continue;
                if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction)) {
                    if (intrinsic->getIntrinsicID() == llvm::Intrinsic::experimental_constrained_fpext ||
                        intrinsic->getIntrinsicID() == llvm::Intrinsic::experimental_constrained_fptrunc)
                        return llvm::createStringError(
                            "constrained CPU f16 conversions are unsupported by software legalization");
                    return llvm::createStringError("CPU f16 intrinsic '%s' is unsupported by software legalization",
                                                   intrinsic->getCalledFunction()->getName().str().c_str());
                }
                if (auto *cast = llvm::dyn_cast<llvm::CastInst>(&instruction);
                    cast && cast->getOpcode() != llvm::Instruction::FPExt &&
                    cast->getOpcode() != llvm::Instruction::FPTrunc && cast->getOpcode() != llvm::Instruction::BitCast)
                    return llvm::createStringError("CPU f16 cast '%s' is unsupported by software legalization",
                                                   cast->getOpcodeName());
            }
    return llvm::Error::success();
}

llvm::Type *floatEquivalent(llvm::Type *type) {
    llvm::Type *floatType = llvm::Type::getFloatTy(type->getContext());
    if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type))
        return llvm::FixedVectorType::get(floatType, vector->getNumElements());
    return type->isHalfTy() ? floatType : nullptr;
}

llvm::Value *lowerScalarConversion(llvm::IRBuilder<> &builder, llvm::Module &module, llvm::Value *value,
                                   llvm::Type *destinationType, HalfConversionHelpers &helpers) {
    llvm::Type *sourceType = value->getType();
    llvm::Function *helper = sourceType->isHalfTy() ? defineHalfToWideHelper(module, destinationType, helpers)
                                                    : defineWideToHalfHelper(module, sourceType, helpers);
    auto *sourceInteger = llvm::cast<llvm::IntegerType>(helper->getFunctionType()->getParamType(0));
    llvm::Value *sourceBits = builder.CreateBitCast(value, sourceInteger);
    llvm::Value *destinationBits = builder.CreateCall(helper, sourceBits);
    return builder.CreateBitCast(destinationBits, destinationType);
}

} // namespace

llvm::Error lowerCpuHalfConversions(llvm::Module &module, const llvm::TargetMachine &targetMachine) {
    if (hasNativeHalfOperations(targetMachine))
        return llvm::Error::success();
    if (llvm::Error error = validateSoftwareHalfOperations(module))
        return error;

    llvm::SmallVector<llvm::BinaryOperator *> arithmetic;
    llvm::SmallVector<llvm::UnaryOperator *> negations;
    llvm::SmallVector<llvm::FCmpInst *> comparisons;
    for (llvm::Function &function : module)
        for (llvm::BasicBlock &block : function)
            for (llvm::Instruction &instruction : block) {
                if (auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
                    binary && elementType(binary->getType())->isHalfTy() &&
                    (binary->getOpcode() == llvm::Instruction::FAdd || binary->getOpcode() == llvm::Instruction::FSub ||
                     binary->getOpcode() == llvm::Instruction::FMul || binary->getOpcode() == llvm::Instruction::FDiv ||
                     binary->getOpcode() == llvm::Instruction::FRem))
                    arithmetic.push_back(binary);
                else if (auto *unary = llvm::dyn_cast<llvm::UnaryOperator>(&instruction);
                         unary && unary->getOpcode() == llvm::Instruction::FNeg &&
                         elementType(unary->getType())->isHalfTy())
                    negations.push_back(unary);
                else if (auto *comparison = llvm::dyn_cast<llvm::FCmpInst>(&instruction);
                         comparison && elementType(comparison->getOperand(0)->getType())->isHalfTy())
                    comparisons.push_back(comparison);
            }

    for (llvm::BinaryOperator *binary : arithmetic) {
        llvm::Type *wideType = floatEquivalent(binary->getType());
        if (!wideType)
            return llvm::createStringError("CPU f16 arithmetic uses a scalable vector type");
        llvm::IRBuilder<> builder(binary);
        builder.setFastMathFlags(binary->getFastMathFlags());
        llvm::Value *left = builder.CreateFPExt(binary->getOperand(0), wideType);
        llvm::Value *right = builder.CreateFPExt(binary->getOperand(1), wideType);
        llvm::Value *wide =
            builder.CreateBinOp(static_cast<llvm::Instruction::BinaryOps>(binary->getOpcode()), left, right);
        llvm::Value *replacement = builder.CreateFPTrunc(wide, binary->getType());
        binary->replaceAllUsesWith(replacement);
        binary->eraseFromParent();
    }
    for (llvm::UnaryOperator *negation : negations) {
        llvm::Type *wideType = floatEquivalent(negation->getType());
        if (!wideType)
            return llvm::createStringError("CPU f16 negation uses a scalable vector type");
        llvm::IRBuilder<> builder(negation);
        builder.setFastMathFlags(negation->getFastMathFlags());
        llvm::Value *wide = builder.CreateFPExt(negation->getOperand(0), wideType);
        llvm::Value *replacement = builder.CreateFPTrunc(builder.CreateFNeg(wide), negation->getType());
        negation->replaceAllUsesWith(replacement);
        negation->eraseFromParent();
    }
    for (llvm::FCmpInst *comparison : comparisons) {
        llvm::Type *wideType = floatEquivalent(comparison->getOperand(0)->getType());
        if (!wideType)
            return llvm::createStringError("CPU f16 comparison uses a scalable vector type");
        llvm::IRBuilder<> builder(comparison);
        builder.setFastMathFlags(comparison->getFastMathFlags());
        llvm::Value *left = builder.CreateFPExt(comparison->getOperand(0), wideType);
        llvm::Value *right = builder.CreateFPExt(comparison->getOperand(1), wideType);
        llvm::Value *replacement = builder.CreateFCmp(comparison->getPredicate(), left, right);
        comparison->replaceAllUsesWith(replacement);
        comparison->eraseFromParent();
    }

    llvm::SmallVector<llvm::CastInst *> conversions;
    for (llvm::Function &function : module)
        for (llvm::BasicBlock &block : function)
            for (llvm::Instruction &instruction : block)
                if (auto *cast = llvm::dyn_cast<llvm::CastInst>(&instruction);
                    cast && (cast->getOpcode() == llvm::Instruction::FPExt ||
                             cast->getOpcode() == llvm::Instruction::FPTrunc)) {
                    llvm::Type *source = elementType(cast->getSrcTy());
                    llvm::Type *destination = elementType(cast->getDestTy());
                    if (source->isHalfTy() || destination->isHalfTy())
                        conversions.push_back(cast);
                }

    HalfConversionHelpers helpers;
    for (llvm::CastInst *conversion : conversions) {
        llvm::Type *source = scalarType(conversion->getSrcTy());
        llvm::Type *destination = scalarType(conversion->getDestTy());
        if (!source || !destination ||
            !((source->isHalfTy() && (destination->isFloatTy() || destination->isDoubleTy())) ||
              (destination->isHalfTy() && (source->isFloatTy() || source->isDoubleTy()))))
            return llvm::createStringError("CPU f16 conversion uses an unsupported floating-point type");

        llvm::IRBuilder<> builder(conversion);
        llvm::Value *replacement = nullptr;
        if (auto *sourceVector = llvm::dyn_cast<llvm::FixedVectorType>(conversion->getSrcTy())) {
            auto *destinationVector = llvm::cast<llvm::FixedVectorType>(conversion->getDestTy());
            replacement = llvm::PoisonValue::get(destinationVector);
            for (unsigned index = 0; index < sourceVector->getNumElements(); ++index) {
                llvm::Value *sourceElement = builder.CreateExtractElement(conversion->getOperand(0), index);
                llvm::Value *destinationElement =
                    lowerScalarConversion(builder, module, sourceElement, destinationVector->getElementType(), helpers);
                replacement = builder.CreateInsertElement(replacement, destinationElement, index);
            }
        } else {
            replacement = lowerScalarConversion(builder, module, conversion->getOperand(0), destination, helpers);
        }
        conversion->replaceAllUsesWith(replacement);
        conversion->eraseFromParent();
    }
    return llvm::Error::success();
}

} // namespace vernon
