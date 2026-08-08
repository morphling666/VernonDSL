#include "VernonCpuAbiWrapper.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

namespace vernon {
namespace {

llvm::Error invalidAbi(const llvm::Twine &message) { return llvm::createStringError(message); }

llvm::Constant *integerConstant(llvm::Type *type, uint64_t value) {
    auto *integerType = llvm::dyn_cast<llvm::IntegerType>(type);
    return integerType ? llvm::ConstantInt::get(integerType, value) : nullptr;
}

} // namespace

llvm::Error defineCpuTextureSampleHelper(llvm::Module &module) {
    llvm::Function *helper = module.getFunction("__vernon_cpu_texture_sample");
    if (!helper)
        return llvm::Error::success();
    if (!helper->empty())
        return invalidAbi("CPU texture helper unexpectedly has a body");

    llvm::LLVMContext &context = module.getContext();
    llvm::Type *pointerType = llvm::PointerType::get(context, 0);
    auto *uvType = llvm::FixedVectorType::get(llvm::Type::getFloatTy(context), 2);
    auto *resultType = llvm::FixedVectorType::get(llvm::Type::getFloatTy(context), 4);
    if (helper->arg_size() != 4 || helper->getArg(0)->getType() != llvm::Type::getInt64Ty(context) ||
        helper->getArg(1)->getType() != llvm::Type::getInt64Ty(context) || helper->getArg(2)->getType() != uvType ||
        helper->getArg(3)->getType() != pointerType || helper->getReturnType() != resultType)
        return invalidAbi("lowered CPU texture helper has an incompatible type");

    helper->setLinkage(llvm::GlobalValue::InternalLinkage);
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", helper);
    llvm::IRBuilder<> builder(entry);
    llvm::Value *callbacks = helper->getArg(3);
    llvm::LoadInst *userData = builder.CreateLoad(pointerType, callbacks, "user_data");
    userData->setAlignment(llvm::Align(1));
    llvm::Value *sampleAddress = builder.CreateGEP(llvm::Type::getInt8Ty(context), callbacks,
                                                   llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 8));
    llvm::LoadInst *sampleFunction = builder.CreateLoad(pointerType, sampleAddress, "sample_2d");
    sampleFunction->setAlignment(llvm::Align(1));
    llvm::Value *output = builder.CreateAlloca(llvm::ArrayType::get(llvm::Type::getFloatTy(context), 4));
    auto callbackType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                {pointerType, llvm::Type::getInt64Ty(context), llvm::Type::getFloatTy(context),
                                 llvm::Type::getFloatTy(context), pointerType},
                                false);
    llvm::Value *uv = helper->getArg(2);
    builder.CreateCall(callbackType, sampleFunction,
                       {userData, helper->getArg(0), builder.CreateExtractElement(uv, uint64_t{0}),
                        builder.CreateExtractElement(uv, uint64_t{1}), output});
    llvm::LoadInst *sample = builder.CreateLoad(resultType, output);
    sample->setAlignment(llvm::Align(1));
    builder.CreateRet(sample);
    return llvm::Error::success();
}

llvm::Error emitCpuAbiWrapper(llvm::Module &module, const CpuAbiWrapperMetadata &metadata) {
    llvm::Function *function = module.getFunction(metadata.internalFunctionSymbol);
    if (!function)
        return invalidAbi("CPU ABI wrapper internal function '" + metadata.internalFunctionSymbol + "' does not exist");

    size_t loweredArgumentCount = 1;
    for (const CpuAbiArgumentPacking &argument : metadata.sourceArguments) {
        loweredArgumentCount += argument.kind == CpuAbiArgumentKind::TensorView
                                    ? 5 * argument.tensorLeafElementSizes.size()
                                    : argument.callLanes.size();
        if (argument.kind == CpuAbiArgumentKind::TensorView)
            loweredArgumentCount += 1 + 2 * argument.tensorRank;
    }
    if (function->arg_size() != loweredArgumentCount)
        return invalidAbi("lowered CPU entry '" + metadata.internalFunctionSymbol +
                          "' has an incompatible argument count");

    llvm::LLVMContext &context = module.getContext();
    llvm::Type *pointerType = llvm::PointerType::get(context, 0);
    auto wrapperType = llvm::FunctionType::get(llvm::Type::getInt32Ty(context), {pointerType}, false);
    llvm::Function *wrapper =
        llvm::Function::Create(wrapperType, llvm::GlobalValue::ExternalLinkage, metadata.exportedWrapperSymbol, module);
    if (llvm::Triple(module.getTargetTriple()).isOSWindows())
        wrapper->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    else
        wrapper->setVisibility(llvm::GlobalValue::DefaultVisibility);
    llvm::BasicBlock *entryBlock = llvm::BasicBlock::Create(context, "entry", wrapper);
    llvm::BasicBlock *sizeBlock = llvm::BasicBlock::Create(context, "check_sizes", wrapper);
    llvm::BasicBlock *callBlock = llvm::BasicBlock::Create(context, "call", wrapper);
    llvm::BasicBlock *invalidBlock = llvm::BasicBlock::Create(context, "invalid", wrapper);

    llvm::IRBuilder<> builder(entryBlock);
    llvm::Value *invocation = wrapper->getArg(0);
    builder.CreateCondBr(
        builder.CreateICmpEQ(invocation, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
        invalidBlock, sizeBlock);

    builder.SetInsertPoint(sizeBlock);
    llvm::LoadInst *argumentsLoad = builder.CreateLoad(pointerType, invocation, "arguments");
    argumentsLoad->setAlignment(llvm::Align(1));
    llvm::Value *arguments = argumentsLoad;
    llvm::Value *argumentsSizeAddress = builder.CreateGEP(llvm::Type::getInt8Ty(context), invocation,
                                                          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 8));
    llvm::LoadInst *argumentsSize = builder.CreateLoad(llvm::Type::getInt64Ty(context), argumentsSizeAddress);
    argumentsSize->setAlignment(llvm::Align(1));
    llvm::Value *resultsAddress = builder.CreateGEP(llvm::Type::getInt8Ty(context), invocation,
                                                    llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 16));
    llvm::LoadInst *resultsLoad = builder.CreateLoad(pointerType, resultsAddress, "results");
    resultsLoad->setAlignment(llvm::Align(1));
    llvm::Value *results = resultsLoad;
    llvm::Value *resultsSizeAddress = builder.CreateGEP(llvm::Type::getInt8Ty(context), invocation,
                                                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 24));
    llvm::LoadInst *resultsSize = builder.CreateLoad(llvm::Type::getInt64Ty(context), resultsSizeAddress);
    resultsSize->setAlignment(llvm::Align(1));
    llvm::Value *validArguments =
        metadata.argumentsSize == 0
            ? llvm::ConstantInt::getTrue(context)
            : builder.CreateAnd(
                  builder.CreateICmpNE(arguments,
                                       llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
                  builder.CreateICmpUGE(
                      argumentsSize, llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), metadata.argumentsSize)));
    llvm::Value *validResults =
        metadata.resultsSize == 0
            ? llvm::ConstantInt::getTrue(context)
            : builder.CreateAnd(
                  builder.CreateICmpNE(results,
                                       llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
                  builder.CreateICmpUGE(resultsSize,
                                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), metadata.resultsSize)));
    llvm::Value *texturesAddress = builder.CreateGEP(llvm::Type::getInt8Ty(context), invocation,
                                                     llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 32));
    llvm::LoadInst *textures = builder.CreateLoad(pointerType, texturesAddress, "textures");
    textures->setAlignment(llvm::Align(1));
    llvm::Value *validTextures =
        metadata.requiresTextureCallbacks
            ? builder.CreateICmpNE(textures, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType)))
            : llvm::ConstantInt::getTrue(context);
    builder.CreateCondBr(builder.CreateAnd(builder.CreateAnd(validArguments, validResults), validTextures), callBlock,
                         invalidBlock);

    builder.SetInsertPoint(callBlock);
    llvm::SmallVector<llvm::Value *> argumentsToCall;
    size_t loweredIndex = 0;
    for (const CpuAbiArgumentPacking &packing : metadata.sourceArguments) {
        llvm::Value *address =
            builder.CreateGEP(llvm::Type::getInt8Ty(context), arguments,
                              llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), packing.offset));
        if (packing.kind != CpuAbiArgumentKind::TensorView) {
            for (const CpuCallLanePacking &lane : packing.callLanes) {
                llvm::Type *argumentType = function->getArg(loweredIndex++)->getType();
                if (module.getDataLayout().getTypeStoreSize(argumentType) != lane.size)
                    return invalidAbi("CPU ABI scalar argument lane size does not match its lowered type");
                llvm::Value *laneAddress =
                    builder.CreateGEP(builder.getInt8Ty(), address, builder.getInt64(lane.offset));
                llvm::LoadInst *load = builder.CreateLoad(argumentType, laneAddress);
                load->setAlignment(llvm::Align(1));
                argumentsToCall.push_back(load);
            }
            continue;
        }

        const uint64_t descriptorSize = 8 * (2 + 2 * static_cast<uint64_t>(packing.tensorRank));
        if (packing.size != descriptorSize || module.getDataLayout().getPointerSize() != 8)
            return invalidAbi("CPU TensorView descriptor has an incompatible size");
        llvm::LoadInst *rawPointer = builder.CreateLoad(pointerType, address, "buffer");
        rawPointer->setAlignment(llvm::Align(1));
        for (size_t leafIndex = 0; leafIndex < packing.tensorLeafElementSizes.size(); ++leafIndex) {
            llvm::Type *allocatedType = function->getArg(loweredIndex++)->getType();
            llvm::Type *alignedType = function->getArg(loweredIndex++)->getType();
            llvm::Type *offsetType = function->getArg(loweredIndex++)->getType();
            llvm::Type *sizeType = function->getArg(loweredIndex++)->getType();
            llvm::Type *strideType = function->getArg(loweredIndex++)->getType();
            if (allocatedType != pointerType || alignedType != pointerType)
                return invalidAbi("lowered CPU buffer descriptor pointer types are "
                                  "incompatible");
            llvm::Constant *offset = integerConstant(offsetType, 0);
            llvm::Constant *stride = integerConstant(strideType, 1);
            llvm::Value *extent = integerConstant(sizeType, 1);
            if (!offset || !extent || !stride)
                return invalidAbi("lowered CPU buffer descriptor index types are "
                                  "incompatible");
            for (uint32_t dimension = 0; dimension < packing.tensorRank; ++dimension) {
                llvm::Value *extentAddress = builder.CreateGEP(
                    builder.getInt8Ty(), address, builder.getInt64(8 * (2 + static_cast<uint64_t>(dimension))));
                llvm::LoadInst *runtimeExtent = builder.CreateLoad(sizeType, extentAddress);
                runtimeExtent->setAlignment(llvm::Align(1));
                extent = builder.CreateMul(extent, runtimeExtent);
            }
            argumentsToCall.append({rawPointer, rawPointer, offset, extent, stride});
        }
    }
    for (const CpuAbiArgumentPacking &packing : metadata.sourceArguments) {
        if (packing.kind != CpuAbiArgumentKind::TensorView)
            continue;
        llvm::Value *descriptor =
            builder.CreateGEP(llvm::Type::getInt8Ty(context), arguments,
                              llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), packing.offset));
        const uint32_t fieldCount = 1 + 2 * packing.tensorRank;
        for (uint32_t field = 0; field < fieldCount; ++field) {
            llvm::Type *fieldType = function->getArg(loweredIndex++)->getType();
            if (!llvm::isa<llvm::IntegerType>(fieldType))
                return invalidAbi("CPU TensorView descriptor field type is not an integer");
            llvm::Value *fieldAddress = builder.CreateGEP(
                llvm::Type::getInt8Ty(context), descriptor,
                llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 8 * (1 + static_cast<uint64_t>(field))));
            llvm::LoadInst *fieldValue = builder.CreateLoad(fieldType, fieldAddress);
            fieldValue->setAlignment(llvm::Align(1));
            argumentsToCall.push_back(fieldValue);
        }
    }
    argumentsToCall.push_back(textures);
    llvm::CallInst *call = builder.CreateCall(function, argumentsToCall);
    if (!metadata.resultCallLanes.empty()) {
        llvm::SmallVector<llvm::Value *> lanes;
        if (metadata.resultCallLanes.size() == 1) {
            lanes.push_back(call);
        } else {
            auto *resultType = llvm::dyn_cast<llvm::StructType>(call->getType());
            if (!resultType || resultType->getNumElements() != metadata.resultCallLanes.size())
                return invalidAbi("CPU ABI scalar result lane count does not match its lowered type");
            for (unsigned index = 0; index < resultType->getNumElements(); ++index)
                lanes.push_back(builder.CreateExtractValue(call, index));
        }
        for (auto [lane, packing] : llvm::zip_equal(lanes, metadata.resultCallLanes)) {
            if (module.getDataLayout().getTypeStoreSize(lane->getType()) != packing.size)
                return invalidAbi("CPU ABI scalar result lane size does not match its lowered type");
            llvm::Value *target = builder.CreateGEP(builder.getInt8Ty(), results, builder.getInt64(packing.offset));
            llvm::StoreInst *store = builder.CreateStore(lane, target);
            store->setAlignment(llvm::Align(1));
        }
    } else if (!function->getReturnType()->isVoidTy()) {
        return invalidAbi("lowered CPU entry unexpectedly returns a value without result lanes");
    }
    builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0));

    builder.SetInsertPoint(invalidBlock);
    builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1));
    return llvm::Error::success();
}

} // namespace vernon
