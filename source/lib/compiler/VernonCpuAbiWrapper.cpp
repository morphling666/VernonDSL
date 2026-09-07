#include "VernonCpuAbiWrapper.h"

#include "VernonCommon.h"
#include "VernonCpuWorkgroupABI.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace vernon {
namespace {

constexpr uint32_t kLanePhaseComplete = 6;

llvm::Error invalidAbi(const llvm::Twine &message) { return llvm::createStringError(message); }

uint64_t hostProvidedArgumentsSize(const CpuAbiWrapperMetadata &metadata) {
    uint64_t size = 0;
    for (const CpuAbiArgumentPacking &packing : metadata.sourceArguments) {
        if (!packing.builtin.empty() && packing.builtin != "ad_tape_allocator" &&
            packing.builtin != "ad_tape_root_region")
            continue;
        size = std::max(size, packing.offset + packing.size);
    }
    return size;
}

llvm::Constant *integerConstant(llvm::Type *type, uint64_t value) {
    auto *integerType = llvm::dyn_cast<llvm::IntegerType>(type);
    return integerType ? llvm::ConstantInt::get(integerType, value) : nullptr;
}

llvm::Expected<llvm::Function *> createPhaseCoroutine(llvm::Module &module, llvm::Function &source,
                                                      llvm::ArrayRef<CpuCallLanePacking> resultPacking) {
    llvm::LLVMContext &context = module.getContext();
    llvm::Type *pointerType = llvm::PointerType::get(context, 0);
    llvm::SmallVector<llvm::Type *> parameterTypes;
    for (llvm::Argument &argument : source.args())
        parameterTypes.push_back(argument.getType());
    parameterTypes.push_back(pointerType);
    auto *functionType = llvm::FunctionType::get(pointerType, parameterTypes, false);
    llvm::Function *coroutine = llvm::Function::Create(functionType, llvm::GlobalValue::InternalLinkage,
                                                       source.getName() + ".range_phase", module);

    llvm::ValueToValueMapTy mapping;
    for (auto [index, argument] : llvm::enumerate(source.args()))
        mapping[&argument] = coroutine->getArg(index);
    llvm::SmallVector<llvm::ReturnInst *> returns;
    llvm::CloneFunctionInto(coroutine, &source, mapping, llvm::CloneFunctionChangeType::LocalChangesOnly, returns);
    coroutine->setLinkage(llvm::GlobalValue::InternalLinkage);
    coroutine->setVisibility(llvm::GlobalValue::DefaultVisibility);
    coroutine->setDSOLocal(true);
    coroutine->addFnAttr(llvm::Attribute::PresplitCoroutine);
    if (coroutine->empty())
        return invalidAbi("barrier-bearing CPU entry has no body");

    llvm::BasicBlock *originalEntry = &coroutine->getEntryBlock();
    llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "coro.entry", coroutine, originalEntry);
    llvm::BasicBlock *cleanup = llvm::BasicBlock::Create(context, "coro.cleanup", coroutine);
    llvm::BasicBlock *suspend = llvm::BasicBlock::Create(context, "coro.suspend", coroutine);
    llvm::BasicBlock *trap = llvm::BasicBlock::Create(context, "coro.final.resume", coroutine);
    llvm::IRBuilder<> builder(entry);

    llvm::Function *coroId = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_id);
    llvm::Value *nullPointer = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType));
    llvm::Value *id =
        builder.CreateCall(coroId, {builder.getInt32(0), nullPointer, nullPointer, nullPointer}, "coro.id");
    llvm::Function *coroSize =
        llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_size, {builder.getInt64Ty()});
    llvm::Value *frameSize = builder.CreateCall(coroSize, {}, "coro.size");
    llvm::Function *coroAlign =
        llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_align, {builder.getInt64Ty()});
    llvm::Value *frameAlignment = builder.CreateCall(coroAlign, {}, "coro.align");
    llvm::FunctionCallee laneAddress = module.getOrInsertFunction(
        VERNON_CPU_LANE_ADDRESS_V1_SYMBOL,
        llvm::FunctionType::get(
            builder.getInt64Ty(),
            {builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty()}, false));
    llvm::Value *frameInteger = builder.CreateCall(
        laneAddress,
        {builder.getInt64(VERNON_CPU_LANE_COROUTINE_FRAME_SITE_V1), frameSize, frameAlignment, builder.getInt64(0)},
        "coro.frame.address");
    llvm::BasicBlock *frameReady = llvm::BasicBlock::Create(context, "coro.frame.ready", coroutine, originalEntry);
    llvm::BasicBlock *frameFailed = llvm::BasicBlock::Create(context, "coro.frame.failed", coroutine, originalEntry);
    builder.CreateCondBr(builder.CreateICmpNE(frameInteger, builder.getInt64(0)), frameReady, frameFailed);
    builder.SetInsertPoint(frameFailed);
    builder.CreateRet(nullPointer);
    builder.SetInsertPoint(frameReady);
    llvm::Value *frame = builder.CreateIntToPtr(frameInteger, pointerType, "coro.frame");
    llvm::Function *coroBegin = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_begin);
    llvm::Value *handle = builder.CreateCall(coroBegin, {id, frame}, "coro.handle");
    builder.CreateBr(originalEntry);

    llvm::SmallVector<llvm::CallInst *> barriers;
    for (llvm::BasicBlock &block : *coroutine)
        for (llvm::Instruction &instruction : block)
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction))
                if (llvm::Function *callee = call->getCalledFunction();
                    callee && callee->getName() == VERNON_CPU_WORKGROUP_BARRIER_V1_SYMBOL)
                    barriers.push_back(call);

    llvm::Function *coroSuspend = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_suspend);
    for (llvm::CallInst *barrier : barriers) {
        llvm::BasicBlock *block = barrier->getParent();
        llvm::BasicBlock *resume = block->splitBasicBlock(std::next(barrier->getIterator()), "phase.resume");
        block->getTerminator()->eraseFromParent();
        builder.SetInsertPoint(block);
        llvm::Value *suspended =
            builder.CreateCall(coroSuspend, {llvm::ConstantTokenNone::get(context), builder.getFalse()}, "phase");
        llvm::SwitchInst *dispatch = builder.CreateSwitch(suspended, suspend, 2);
        dispatch->addCase(builder.getInt8(0), resume);
        dispatch->addCase(builder.getInt8(1), cleanup);
    }

    for (llvm::ReturnInst *returnInstruction : returns) {
        builder.SetInsertPoint(returnInstruction);
        if (llvm::Value *returned = returnInstruction->getReturnValue()) {
            llvm::SmallVector<llvm::Value *> lanes;
            if (resultPacking.size() == 1) {
                lanes.push_back(returned);
            } else {
                auto *resultType = llvm::dyn_cast<llvm::StructType>(returned->getType());
                if (!resultType || resultType->getNumElements() != resultPacking.size())
                    return invalidAbi("barrier-bearing CPU result lane count does not match its lowered type");
                for (unsigned index = 0; index < resultType->getNumElements(); ++index)
                    lanes.push_back(builder.CreateExtractValue(returned, index));
            }
            llvm::Argument *resultStorage = coroutine->getArg(coroutine->arg_size() - 1);
            for (auto [lane, packing] : llvm::zip_equal(lanes, resultPacking)) {
                if (module.getDataLayout().getTypeStoreSize(lane->getType()) != packing.size)
                    return invalidAbi("barrier-bearing CPU result lane size does not match its lowered type");
                llvm::Value *target =
                    builder.CreateGEP(builder.getInt8Ty(), resultStorage, builder.getInt64(packing.offset));
                builder.CreateStore(lane, target)->setAlignment(llvm::Align(1));
            }
        }
        llvm::Value *finished =
            builder.CreateCall(coroSuspend, {llvm::ConstantTokenNone::get(context), builder.getTrue()}, "complete");
        llvm::SwitchInst *dispatch = builder.CreateSwitch(finished, suspend, 2);
        dispatch->addCase(builder.getInt8(0), trap);
        dispatch->addCase(builder.getInt8(1), cleanup);
        returnInstruction->eraseFromParent();
    }

    builder.SetInsertPoint(cleanup);
    llvm::Function *coroFree = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_free);
    builder.CreateCall(coroFree, {id, handle});
    builder.CreateBr(suspend);

    builder.SetInsertPoint(suspend);
    llvm::Function *coroEnd = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_end);
    builder.CreateCall(coroEnd, {handle, builder.getFalse(), llvm::ConstantTokenNone::get(context)});
    builder.CreateRet(handle);

    builder.SetInsertPoint(trap);
    builder.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::trap));
    builder.CreateUnreachable();
    return coroutine;
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
    const unsigned pointerBits = module.getDataLayout().getPointerSizeInBits();
    if (pointerBits != 32 && pointerBits != 64)
        return invalidAbi("CPU texture callback ABI requires 32-bit or 64-bit pointers");
    llvm::IntegerType *uintptrType = llvm::IntegerType::get(context, pointerBits);
    auto *callbacksType = llvm::StructType::get(context, {pointerType, pointerType, pointerType});
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
    llvm::LoadInst *userData =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(callbacksType, callbacks, 0), "user_data");
    userData->setAlignment(llvm::Align(1));
    llvm::Value *sampleAddress = builder.CreateStructGEP(callbacksType, callbacks, 1);
    llvm::LoadInst *sampleFunction = builder.CreateLoad(pointerType, sampleAddress, "sample_2d");
    sampleFunction->setAlignment(llvm::Align(1));
    llvm::Value *output = builder.CreateAlloca(llvm::ArrayType::get(llvm::Type::getFloatTy(context), 4));
    auto callbackType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {pointerType, uintptrType, llvm::Type::getFloatTy(context), llvm::Type::getFloatTy(context), pointerType},
        false);
    llvm::Value *uv = helper->getArg(2);
    llvm::Value *texture = builder.CreateZExtOrTrunc(helper->getArg(0), uintptrType);
    builder.CreateCall(callbackType, sampleFunction,
                       {userData, texture, builder.CreateExtractElement(uv, uint64_t{0}),
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
    if (metadata.requiresPhases) {
        llvm::Expected<llvm::Function *> coroutine = createPhaseCoroutine(module, *function, metadata.resultCallLanes);
        if (!coroutine)
            return coroutine.takeError();
        function = *coroutine;
    }

    size_t loweredArgumentCount = 1;
    for (const CpuAbiArgumentPacking &argument : metadata.sourceArguments) {
        loweredArgumentCount += argument.kind == CpuAbiArgumentKind::TensorView
                                    ? 5 * argument.tensorLeafElementSizes.size()
                                    : argument.callLanes.size();
        if (argument.kind == CpuAbiArgumentKind::TensorView)
            loweredArgumentCount += 1 + 2 * argument.tensorRank;
    }
    if (function->arg_size() != loweredArgumentCount + (metadata.requiresPhases ? 1 : 0))
        return invalidAbi("lowered CPU entry '" + metadata.internalFunctionSymbol +
                          "' has an incompatible argument count");

    llvm::LLVMContext &context = module.getContext();
    llvm::Type *pointerType = llvm::PointerType::get(context, 0);
    const unsigned pointerBits = module.getDataLayout().getPointerSizeInBits();
    if (pointerBits != 32 && pointerBits != 64)
        return invalidAbi("CPU ABI requires 32-bit or 64-bit pointers");
    const uint64_t pointerBytes = pointerBits / 8;
    llvm::IntegerType *sizeType = llvm::IntegerType::get(context, pointerBits);
    auto *invocationType = llvm::StructType::get(context, {pointerType, sizeType, pointerType, sizeType, pointerType});
    auto wrapperType = llvm::FunctionType::get(llvm::Type::getInt32Ty(context), {pointerType}, false);
    llvm::Function *laneWrapper = llvm::Function::Create(wrapperType, llvm::GlobalValue::InternalLinkage,
                                                         metadata.exportedWrapperSymbol + ".lane", module);
    laneWrapper->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::BasicBlock *entryBlock = llvm::BasicBlock::Create(context, "entry", laneWrapper);
    llvm::BasicBlock *sizeBlock = llvm::BasicBlock::Create(context, "check_sizes", laneWrapper);
    llvm::BasicBlock *callBlock = llvm::BasicBlock::Create(context, "call", laneWrapper);
    llvm::BasicBlock *invalidBlock = llvm::BasicBlock::Create(context, "invalid", laneWrapper);

    llvm::IRBuilder<> builder(entryBlock);
    llvm::Value *invocation = laneWrapper->getArg(0);
    builder.CreateCondBr(
        builder.CreateICmpEQ(invocation, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
        invalidBlock, sizeBlock);

    builder.SetInsertPoint(sizeBlock);
    llvm::LoadInst *argumentsLoad =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(invocationType, invocation, 0), "arguments");
    argumentsLoad->setAlignment(llvm::Align(1));
    llvm::Value *arguments = argumentsLoad;
    llvm::LoadInst *argumentsSize =
        builder.CreateLoad(sizeType, builder.CreateStructGEP(invocationType, invocation, 1), "arguments_size");
    argumentsSize->setAlignment(llvm::Align(1));
    llvm::LoadInst *resultsLoad =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(invocationType, invocation, 2), "results");
    resultsLoad->setAlignment(llvm::Align(1));
    llvm::Value *results = resultsLoad;
    llvm::LoadInst *resultsSize =
        builder.CreateLoad(sizeType, builder.CreateStructGEP(invocationType, invocation, 3), "results_size");
    resultsSize->setAlignment(llvm::Align(1));
    llvm::Value *validArguments =
        metadata.argumentsSize == 0
            ? llvm::ConstantInt::getTrue(context)
            : builder.CreateAnd(
                  builder.CreateICmpNE(arguments,
                                       llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
                  builder.CreateICmpUGE(argumentsSize, llvm::ConstantInt::get(sizeType, metadata.argumentsSize)));
    llvm::Value *validResults =
        metadata.resultsSize == 0
            ? llvm::ConstantInt::getTrue(context)
            : builder.CreateAnd(
                  builder.CreateICmpNE(results,
                                       llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
                  builder.CreateICmpUGE(resultsSize, llvm::ConstantInt::get(sizeType, metadata.resultsSize)));
    llvm::LoadInst *textures =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(invocationType, invocation, 4), "textures");
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

        const uint64_t descriptorSize = pointerBytes * (2 + 2 * static_cast<uint64_t>(packing.tensorRank));
        if (packing.size != descriptorSize)
            return invalidAbi("CPU TensorView descriptor has an incompatible size");
        llvm::LoadInst *rawPointerBits = builder.CreateLoad(sizeType, address, "buffer_address");
        rawPointerBits->setAlignment(llvm::Align(1));
        llvm::Value *rawPointer = builder.CreateIntToPtr(rawPointerBits, pointerType, "buffer");
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
                llvm::Value *extentAddress =
                    builder.CreateGEP(builder.getInt8Ty(), address,
                                      builder.getInt64(pointerBytes * (2 + static_cast<uint64_t>(dimension))));
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
            llvm::Value *fieldAddress =
                builder.CreateGEP(llvm::Type::getInt8Ty(context), descriptor,
                                  llvm::ConstantInt::get(llvm::Type::getInt64Ty(context),
                                                         pointerBytes * (1 + static_cast<uint64_t>(field))));
            llvm::LoadInst *fieldValue = builder.CreateLoad(fieldType, fieldAddress);
            fieldValue->setAlignment(llvm::Align(1));
            argumentsToCall.push_back(fieldValue);
        }
    }
    argumentsToCall.push_back(textures);
    if (metadata.requiresPhases) {
        argumentsToCall.push_back(results);
        llvm::FunctionCallee laneAddress = module.getOrInsertFunction(
            VERNON_CPU_LANE_ADDRESS_V1_SYMBOL,
            llvm::FunctionType::get(
                builder.getInt64Ty(),
                {builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty()}, false));
        llvm::Value *slotInteger =
            builder.CreateCall(laneAddress, {builder.getInt64(VERNON_CPU_LANE_COROUTINE_HANDLE_SITE_V1),
                                             builder.getInt64(module.getDataLayout().getPointerSize()),
                                             builder.getInt64(module.getDataLayout().getPointerABIAlignment(0).value()),
                                             builder.getInt64(0)});
        llvm::BasicBlock *slotReady = llvm::BasicBlock::Create(context, "phase.slot.ready", laneWrapper);
        llvm::BasicBlock *slotFailed = llvm::BasicBlock::Create(context, "phase.slot.failed", laneWrapper);
        builder.CreateCondBr(builder.CreateICmpNE(slotInteger, builder.getInt64(0)), slotReady, slotFailed);
        builder.SetInsertPoint(slotFailed);
        builder.CreateRet(builder.getInt32(VERNON_STATUS_INTERNAL_ERROR));
        builder.SetInsertPoint(slotReady);
        llvm::Value *slot = builder.CreateIntToPtr(slotInteger, pointerType);
        llvm::LoadInst *existingHandle = builder.CreateLoad(pointerType, slot, "coro.handle");
        existingHandle->setAlignment(module.getDataLayout().getPointerABIAlignment(0));

        llvm::BasicBlock *create = llvm::BasicBlock::Create(context, "phase.create", laneWrapper);
        llvm::BasicBlock *resume = llvm::BasicBlock::Create(context, "phase.resume", laneWrapper);
        llvm::BasicBlock *joined = llvm::BasicBlock::Create(context, "phase.join", laneWrapper);
        llvm::BasicBlock *createReady = llvm::BasicBlock::Create(context, "phase.create.ready", laneWrapper);
        llvm::BasicBlock *createFailed = llvm::BasicBlock::Create(context, "phase.create.failed", laneWrapper);
        llvm::BasicBlock *completed = llvm::BasicBlock::Create(context, "phase.complete", laneWrapper);
        llvm::BasicBlock *yielded = llvm::BasicBlock::Create(context, "phase.yield", laneWrapper);
        builder.CreateCondBr(builder.CreateICmpEQ(existingHandle, llvm::ConstantPointerNull::get(
                                                                      llvm::cast<llvm::PointerType>(pointerType))),
                             create, resume);

        builder.SetInsertPoint(create);
        llvm::Value *createdHandle = builder.CreateCall(function, argumentsToCall, "coro.created");
        builder.CreateCondBr(builder.CreateICmpNE(createdHandle, llvm::ConstantPointerNull::get(
                                                                     llvm::cast<llvm::PointerType>(pointerType))),
                             createReady, createFailed);

        builder.SetInsertPoint(createFailed);
        builder.CreateRet(builder.getInt32(VERNON_STATUS_INTERNAL_ERROR));

        builder.SetInsertPoint(createReady);
        builder.CreateStore(createdHandle, slot)->setAlignment(module.getDataLayout().getPointerABIAlignment(0));
        builder.CreateBr(joined);

        builder.SetInsertPoint(resume);
        llvm::Function *coroResume = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_resume);
        builder.CreateCall(coroResume, {existingHandle});
        builder.CreateBr(joined);

        builder.SetInsertPoint(joined);
        llvm::PHINode *handle = builder.CreatePHI(pointerType, 2, "coro.active");
        handle->addIncoming(createdHandle, createReady);
        handle->addIncoming(existingHandle, resume);
        llvm::Function *coroDone = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_done);
        builder.CreateCondBr(builder.CreateCall(coroDone, {handle}), completed, yielded);

        builder.SetInsertPoint(completed);
        llvm::Function *coroDestroy = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::coro_destroy);
        builder.CreateCall(coroDestroy, {handle});
        builder.CreateStore(llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType)), slot)
            ->setAlignment(module.getDataLayout().getPointerABIAlignment(0));
        builder.CreateRet(builder.getInt32(kLanePhaseComplete));

        builder.SetInsertPoint(yielded);
        builder.CreateRet(builder.getInt32(0));
    } else {
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
        builder.CreateRet(builder.getInt32(0));
    }

    builder.SetInsertPoint(invalidBlock);
    builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1));

    llvm::Function *wrapper =
        llvm::Function::Create(wrapperType, llvm::GlobalValue::ExternalLinkage, metadata.exportedWrapperSymbol, module);
    if (llvm::Triple(module.getTargetTriple()).isOSWindows())
        wrapper->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    else
        wrapper->setVisibility(llvm::GlobalValue::DefaultVisibility);

    llvm::Type *i8 = llvm::Type::getInt8Ty(context);
    llvm::Type *i32 = llvm::Type::getInt32Ty(context);
    llvm::Type *i64 = llvm::Type::getInt64Ty(context);
    auto *rangeType = llvm::StructType::get(
        context, {sizeType, pointerType, sizeType, pointerType, sizeType, pointerType, llvm::ArrayType::get(i32, 3),
                  llvm::ArrayType::get(i32, 3), llvm::ArrayType::get(i32, 3), sizeType, sizeType, sizeType, i64, i64,
                  sizeType, i32, pointerType, pointerType, sizeType});
    const llvm::StructLayout *rangeLayout = module.getDataLayout().getStructLayout(rangeType);
    if (pointerBits == sizeof(void *) * 8 &&
        (rangeLayout->getSizeInBytes() != sizeof(VernonCpuRangeV1) ||
         rangeLayout->getElementOffset(6) != offsetof(VernonCpuRangeV1, grid) ||
         rangeLayout->getElementOffset(9) != offsetof(VernonCpuRangeV1, lane_begin) ||
         rangeLayout->getElementOffset(16) != offsetof(VernonCpuRangeV1, lane_arguments) ||
         rangeLayout->getElementOffset(18) != offsetof(VernonCpuRangeV1, lane_table_count)))
        return invalidAbi("CPU range descriptor layout does not match the host ABI");
    const uint64_t targetRangeSize = rangeLayout->getSizeInBytes();

    llvm::BasicBlock *dispatchEntry = llvm::BasicBlock::Create(context, "entry", wrapper);
    llvm::BasicBlock *rangeCheck = llvm::BasicBlock::Create(context, "range_check", wrapper);
    llvm::BasicBlock *rangeSetup = llvm::BasicBlock::Create(context, "range_setup", wrapper);
    llvm::BasicBlock *rangeFields = llvm::BasicBlock::Create(context, "range_fields", wrapper);
    llvm::BasicBlock *rangeLoop = llvm::BasicBlock::Create(context, "range_loop", wrapper);
    llvm::BasicBlock *rangeCall = llvm::BasicBlock::Create(context, "range_call", wrapper);
    llvm::BasicBlock *rangeNext = llvm::BasicBlock::Create(context, "range_next", wrapper);
    llvm::BasicBlock *rangeDone = llvm::BasicBlock::Create(context, "range_done", wrapper);
    llvm::BasicBlock *rangeInvalid = llvm::BasicBlock::Create(context, "range_invalid", wrapper);

    builder.SetInsertPoint(dispatchEntry);
    llvm::Value *publicInvocation = wrapper->getArg(0);
    llvm::Value *invocationPresent = builder.CreateICmpNE(
        publicInvocation, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType)));
    llvm::BasicBlock *dispatchKind = llvm::BasicBlock::Create(context, "dispatch_kind", wrapper);
    builder.CreateCondBr(invocationPresent, dispatchKind, rangeInvalid);

    builder.SetInsertPoint(dispatchKind);
    llvm::Value *publicArgumentsSizeAddress =
        builder.CreateStructGEP(invocationType, publicInvocation, 1, "arguments_size_address");
    llvm::LoadInst *publicArgumentsSize = builder.CreateLoad(sizeType, publicArgumentsSizeAddress, "arguments_size");
    publicArgumentsSize->setAlignment(llvm::Align(1));
    builder.CreateCondBr(builder.CreateICmpEQ(publicArgumentsSize, llvm::ConstantInt::getAllOnesValue(sizeType)),
                         rangeCheck, rangeInvalid);

    builder.SetInsertPoint(rangeCheck);
    llvm::LoadInst *rangePointer = builder.CreateLoad(pointerType, publicInvocation, "range");
    rangePointer->setAlignment(llvm::Align(1));
    llvm::Value *rangePresent =
        builder.CreateICmpNE(rangePointer, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType)));
    builder.CreateCondBr(rangePresent, rangeSetup, rangeInvalid);

    builder.SetInsertPoint(rangeSetup);
    llvm::LoadInst *rangeStructSize = builder.CreateLoad(sizeType, builder.CreateStructGEP(rangeType, rangePointer, 0));
    rangeStructSize->setAlignment(llvm::Align(1));
    builder.CreateCondBr(builder.CreateICmpEQ(rangeStructSize, llvm::ConstantInt::get(sizeType, targetRangeSize)),
                         rangeFields, rangeInvalid);

    builder.SetInsertPoint(rangeFields);
    llvm::LoadInst *rangeArguments =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(rangeType, rangePointer, 1));
    rangeArguments->setAlignment(llvm::Align(1));
    llvm::LoadInst *rangeArgumentsSizeValue =
        builder.CreateLoad(sizeType, builder.CreateStructGEP(rangeType, rangePointer, 2));
    rangeArgumentsSizeValue->setAlignment(llvm::Align(1));
    llvm::Value *rangeArgumentsSize = builder.CreateZExtOrTrunc(rangeArgumentsSizeValue, i64);
    llvm::LoadInst *rangeResults = builder.CreateLoad(pointerType, builder.CreateStructGEP(rangeType, rangePointer, 3));
    rangeResults->setAlignment(llvm::Align(1));
    llvm::LoadInst *rangeResultsSizeValue =
        builder.CreateLoad(sizeType, builder.CreateStructGEP(rangeType, rangePointer, 4));
    rangeResultsSizeValue->setAlignment(llvm::Align(1));
    llvm::Value *rangeResultsSize = builder.CreateZExtOrTrunc(rangeResultsSizeValue, i64);
    llvm::LoadInst *rangeTextures =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(rangeType, rangePointer, 5));
    rangeTextures->setAlignment(llvm::Align(1));
    llvm::LoadInst *laneArguments =
        builder.CreateLoad(pointerType, builder.CreateStructGEP(rangeType, rangePointer, 16));
    laneArguments->setAlignment(llvm::Align(1));
    llvm::LoadInst *laneResults = builder.CreateLoad(pointerType, builder.CreateStructGEP(rangeType, rangePointer, 17));
    laneResults->setAlignment(llvm::Align(1));
    llvm::LoadInst *laneTableCountValue =
        builder.CreateLoad(sizeType, builder.CreateStructGEP(rangeType, rangePointer, 18));
    laneTableCountValue->setAlignment(llvm::Align(1));
    llvm::Value *laneTableCount = builder.CreateZExtOrTrunc(laneTableCountValue, i64);
    llvm::LoadInst *laneBeginValue = builder.CreateLoad(sizeType, builder.CreateStructGEP(rangeType, rangePointer, 9));
    laneBeginValue->setAlignment(llvm::Align(1));
    llvm::Value *laneBegin = builder.CreateZExtOrTrunc(laneBeginValue, i64);
    llvm::LoadInst *laneEndValue = builder.CreateLoad(sizeType, builder.CreateStructGEP(rangeType, rangePointer, 10));
    laneEndValue->setAlignment(llvm::Align(1));
    llvm::Value *laneEnd = builder.CreateZExtOrTrunc(laneEndValue, i64);

    llvm::Value *workgroupValidationPointer = builder.CreateStructGEP(rangeType, rangePointer, 7);
    llvm::Value *gridValidationPointer = builder.CreateStructGEP(rangeType, rangePointer, 6);
    llvm::Value *groupValidationPointer = builder.CreateStructGEP(rangeType, rangePointer, 8);
    llvm::SmallVector<llvm::Value *, 3> validationWorkgroup;
    llvm::SmallVector<llvm::Value *, 3> validationGrid;
    llvm::Value *validDimensions = llvm::ConstantInt::getTrue(context);
    for (uint64_t dimension = 0; dimension < 3; ++dimension) {
        llvm::Value *index[] = {builder.getInt64(0), builder.getInt64(dimension)};
        llvm::Value *workgroupDimension = builder.CreateZExt(
            builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3), workgroupValidationPointer, index)),
            i64);
        llvm::Value *gridDimension = builder.CreateZExt(
            builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3), gridValidationPointer, index)),
            i64);
        llvm::Value *groupDimension = builder.CreateZExt(
            builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3), groupValidationPointer, index)),
            i64);
        validationWorkgroup.push_back(workgroupDimension);
        validationGrid.push_back(gridDimension);
        validDimensions =
            builder.CreateAnd(validDimensions, builder.CreateICmpNE(workgroupDimension, builder.getInt64(0)));
        validDimensions = builder.CreateAnd(validDimensions, builder.CreateICmpNE(gridDimension, builder.getInt64(0)));
        validDimensions = builder.CreateAnd(validDimensions, builder.CreateICmpULT(groupDimension, gridDimension));
    }
    llvm::Function *multiplyWithOverflow =
        llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::umul_with_overflow, {i64});
    auto checkedMultiply = [&](llvm::Value *left, llvm::Value *right) {
        llvm::Value *product = builder.CreateCall(multiplyWithOverflow, {left, right});
        return std::pair<llvm::Value *, llvm::Value *>{builder.CreateExtractValue(product, 0),
                                                       builder.CreateExtractValue(product, 1)};
    };
    auto [workgroupXY, workgroupXYOverflow] = checkedMultiply(validationWorkgroup[0], validationWorkgroup[1]);
    auto [workgroupVolume, workgroupVolumeOverflow] = checkedMultiply(workgroupXY, validationWorkgroup[2]);
    llvm::Value *validationGlobalWidth = builder.CreateMul(validationGrid[0], validationWorkgroup[0]);
    llvm::Value *validationGlobalHeight = builder.CreateMul(validationGrid[1], validationWorkgroup[1]);
    llvm::Value *validationGlobalDepth = builder.CreateMul(validationGrid[2], validationWorkgroup[2]);
    auto [globalXY, globalXYOverflow] = checkedMultiply(validationGlobalWidth, validationGlobalHeight);
    auto [globalVolume, globalVolumeOverflow] = checkedMultiply(globalXY, validationGlobalDepth);
    llvm::Value *validRange = builder.CreateAnd(builder.CreateICmpULT(laneBegin, laneEnd), validDimensions);
    validRange = builder.CreateAnd(validRange, builder.CreateNot(workgroupXYOverflow));
    validRange = builder.CreateAnd(validRange, builder.CreateNot(workgroupVolumeOverflow));
    validRange = builder.CreateAnd(validRange, builder.CreateNot(globalXYOverflow));
    validRange = builder.CreateAnd(validRange, builder.CreateNot(globalVolumeOverflow));
    validRange = builder.CreateAnd(validRange, builder.CreateICmpULE(laneEnd, workgroupVolume));
    llvm::Value *hasLaneArguments =
        builder.CreateICmpNE(laneArguments, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType)));
    llvm::Value *hasLaneResults =
        builder.CreateICmpNE(laneResults, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType)));
    llvm::Value *hasLaneTables = builder.CreateOr(hasLaneArguments, hasLaneResults);
    validRange =
        builder.CreateAnd(validRange, builder.CreateOr(builder.CreateNot(hasLaneTables),
                                                       builder.CreateICmpUGE(laneTableCount, workgroupVolume)));
    if (const uint64_t hostSize = hostProvidedArgumentsSize(metadata); hostSize)
        validRange = builder.CreateAnd(
            validRange,
            builder.CreateAnd(
                builder.CreateOr(builder.CreateICmpNE(rangeArguments, llvm::ConstantPointerNull::get(
                                                                          llvm::cast<llvm::PointerType>(pointerType))),
                                 builder.CreateICmpNE(laneArguments, llvm::ConstantPointerNull::get(
                                                                         llvm::cast<llvm::PointerType>(pointerType)))),
                builder.CreateICmpUGE(rangeArgumentsSize, builder.getInt64(hostSize))));
    if (metadata.resultsSize)
        validRange = builder.CreateAnd(
            validRange,
            builder.CreateAnd(
                builder.CreateOr(builder.CreateICmpNE(rangeResults, llvm::ConstantPointerNull::get(
                                                                        llvm::cast<llvm::PointerType>(pointerType))),
                                 builder.CreateICmpNE(laneResults, llvm::ConstantPointerNull::get(
                                                                       llvm::cast<llvm::PointerType>(pointerType)))),
                builder.CreateICmpUGE(rangeResultsSize, builder.getInt64(metadata.resultsSize))));

    auto *packedType = llvm::ArrayType::get(i8, std::max<uint64_t>(metadata.argumentsSize, 1));
    llvm::AllocaInst *packed = builder.CreateAlloca(packedType, nullptr, "range_arguments");
    packed->setAlignment(llvm::Align(16));
    llvm::AllocaInst *laneInvocation = builder.CreateAlloca(invocationType, nullptr, "lane_invocation");
    laneInvocation->setAlignment(module.getDataLayout().getABITypeAlign(invocationType));
    builder.CreateStore(packed, builder.CreateStructGEP(invocationType, laneInvocation, 0));
    builder.CreateStore(llvm::ConstantInt::get(sizeType, metadata.argumentsSize),
                        builder.CreateStructGEP(invocationType, laneInvocation, 1));
    builder.CreateStore(llvm::ConstantInt::get(sizeType, metadata.resultsSize),
                        builder.CreateStructGEP(invocationType, laneInvocation, 3));
    builder.CreateStore(rangeTextures, builder.CreateStructGEP(invocationType, laneInvocation, 4));
    builder.CreateCondBr(validRange, rangeLoop, rangeInvalid);

    builder.SetInsertPoint(rangeLoop);
    llvm::PHINode *lane = builder.CreatePHI(i64, 2, "lane");
    lane->addIncoming(laneBegin, rangeFields);
    builder.CreateCondBr(builder.CreateICmpULT(lane, laneEnd), rangeCall, rangeDone);

    builder.SetInsertPoint(rangeCall);
    builder.CreateStore(builder.CreateZExtOrTrunc(lane, sizeType),
                        builder.CreateStructGEP(rangeType, rangePointer, 11));
    llvm::Value *workgroupPointer = builder.CreateStructGEP(rangeType, rangePointer, 7);
    llvm::Value *groupPointer = builder.CreateStructGEP(rangeType, rangePointer, 8);
    llvm::SmallVector<llvm::Value *, 3> workgroup;
    llvm::SmallVector<llvm::Value *, 3> global;
    llvm::SmallVector<llvm::Value *, 3> local;
    llvm::SmallVector<llvm::Value *, 3> group;
    for (uint64_t dimension = 0; dimension < 3; ++dimension) {
        llvm::Value *index[] = {builder.getInt64(0), builder.getInt64(dimension)};
        workgroup.push_back(builder.CreateZExt(
            builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3), workgroupPointer, index)), i64));
        group.push_back(builder.CreateZExt(
            builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3), groupPointer, index)), i64));
    }
    local.push_back(builder.CreateURem(lane, workgroup[0]));
    local.push_back(builder.CreateURem(builder.CreateUDiv(lane, workgroup[0]), workgroup[1]));
    local.push_back(builder.CreateUDiv(lane, builder.CreateMul(workgroup[0], workgroup[1])));
    for (size_t dimension = 0; dimension < 3; ++dimension)
        global.push_back(
            builder.CreateAdd(builder.CreateMul(group[dimension], workgroup[dimension]), local[dimension]));
    llvm::Value *gridPointer = builder.CreateStructGEP(rangeType, rangePointer, 6);
    llvm::SmallVector<llvm::Value *, 3> grid;
    for (uint64_t dimension = 0; dimension < 3; ++dimension) {
        llvm::Value *index[] = {builder.getInt64(0), builder.getInt64(dimension)};
        grid.push_back(builder.CreateZExt(
            builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3), gridPointer, index)), i64));
    }
    llvm::Value *globalWidth = builder.CreateMul(grid[0], workgroup[0]);
    llvm::Value *globalHeight = builder.CreateMul(grid[1], workgroup[1]);
    llvm::Value *globalLinear = builder.CreateAdd(
        global[0],
        builder.CreateMul(globalWidth, builder.CreateAdd(global[1], builder.CreateMul(globalHeight, global[2]))));
    llvm::Value *laneTableIndex =
        builder.CreateSelect(builder.CreateICmpULT(laneTableCount, globalVolume), lane, globalLinear);

    auto selectLanePointer = [&](llvm::Value *common, llvm::Value *table, llvm::StringRef name) {
        llvm::BasicBlock *commonBlock = llvm::BasicBlock::Create(context, name + ".common", wrapper);
        llvm::BasicBlock *tableBlock = llvm::BasicBlock::Create(context, name + ".table", wrapper);
        llvm::BasicBlock *joinedBlock = llvm::BasicBlock::Create(context, name + ".joined", wrapper);
        builder.CreateCondBr(
            builder.CreateICmpEQ(table, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))),
            commonBlock, tableBlock);
        builder.SetInsertPoint(commonBlock);
        builder.CreateBr(joinedBlock);
        builder.SetInsertPoint(tableBlock);
        llvm::Value *lanePointer =
            builder.CreateLoad(pointerType, builder.CreateGEP(pointerType, table, laneTableIndex), name);
        builder.CreateBr(joinedBlock);
        builder.SetInsertPoint(joinedBlock);
        llvm::PHINode *selected = builder.CreatePHI(pointerType, 2, name + ".selected");
        selected->addIncoming(common, commonBlock);
        selected->addIncoming(lanePointer, tableBlock);
        return static_cast<llvm::Value *>(selected);
    };
    llvm::Value *selectedArguments = selectLanePointer(rangeArguments, laneArguments, "lane_arguments");
    llvm::Value *selectedResults = selectLanePointer(rangeResults, laneResults, "lane_results");
    llvm::Value *validLanePointers = llvm::ConstantInt::getTrue(context);
    if (hostProvidedArgumentsSize(metadata))
        validLanePointers = builder.CreateAnd(
            validLanePointers,
            builder.CreateICmpNE(selectedArguments,
                                 llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType))));
    if (metadata.resultsSize)
        validLanePointers = builder.CreateAnd(
            validLanePointers, builder.CreateICmpNE(selectedResults, llvm::ConstantPointerNull::get(
                                                                         llvm::cast<llvm::PointerType>(pointerType))));
    llvm::BasicBlock *laneReady = llvm::BasicBlock::Create(context, "lane_ready", wrapper);
    builder.CreateCondBr(validLanePointers, laneReady, rangeInvalid);
    builder.SetInsertPoint(laneReady);
    if (metadata.argumentsSize) {
        builder.CreateMemSet(packed, builder.getInt8(0), metadata.argumentsSize, llvm::MaybeAlign(16));
        if (hostProvidedArgumentsSize(metadata)) {
            llvm::Value *copySize = builder.CreateSelect(
                builder.CreateICmpULT(rangeArgumentsSize, builder.getInt64(metadata.argumentsSize)), rangeArgumentsSize,
                builder.getInt64(metadata.argumentsSize));
            builder.CreateMemCpy(packed, llvm::Align(16), selectedArguments, llvm::Align(1), copySize);
        }
    }
    builder.CreateStore(selectedResults, builder.CreateStructGEP(invocationType, laneInvocation, 2));

    for (const CpuAbiArgumentPacking &packing : metadata.sourceArguments) {
        llvm::ArrayRef<llvm::Value *> coordinates =
            packing.builtin == "global_invocation_id"  ? llvm::ArrayRef<llvm::Value *>(global)
            : packing.builtin == "local_invocation_id" ? llvm::ArrayRef<llvm::Value *>(local)
            : packing.builtin == "workgroup_id"        ? llvm::ArrayRef<llvm::Value *>(group)
                                                       : llvm::ArrayRef<llvm::Value *>();
        if (coordinates.empty())
            continue;
        llvm::Value *target = builder.CreateGEP(i8, packed, builder.getInt64(packing.offset));
        if (packing.size == sizeof(uint64_t)) {
            llvm::StoreInst *store = builder.CreateStore(coordinates.front(), target);
            store->setAlignment(llvm::Align(1));
            continue;
        }
        for (size_t dimension = 0; dimension < 3 && (dimension + 1) * sizeof(uint32_t) <= packing.size; ++dimension) {
            llvm::Value *component = builder.CreateTrunc(coordinates[dimension], i32);
            llvm::StoreInst *store =
                builder.CreateStore(component, builder.CreateGEP(i8, target, builder.getInt64(dimension * 4)));
            store->setAlignment(llvm::Align(1));
        }
    }
    llvm::CallInst *laneStatus = builder.CreateCall(laneWrapper, {laneInvocation});
    if (metadata.requiresPhases) {
        llvm::BasicBlock *phaseStatus = llvm::BasicBlock::Create(context, "phase_status", wrapper);
        llvm::BasicBlock *phaseCompleted = llvm::BasicBlock::Create(context, "phase_completed", wrapper);
        llvm::BasicBlock *phaseFailure = llvm::BasicBlock::Create(context, "phase_failure", wrapper);
        builder.CreateCondBr(builder.CreateICmpEQ(laneStatus, builder.getInt32(0)), rangeNext, phaseStatus);

        builder.SetInsertPoint(phaseStatus);
        builder.CreateCondBr(builder.CreateICmpEQ(laneStatus, builder.getInt32(kLanePhaseComplete)), phaseCompleted,
                             phaseFailure);

        builder.SetInsertPoint(phaseCompleted);
        llvm::LoadInst *phaseOutcome = builder.CreateLoad(i32, builder.CreateStructGEP(rangeType, rangePointer, 15));
        llvm::BasicBlock *recordCompletion = llvm::BasicBlock::Create(context, "record_completion", wrapper);
        builder.CreateCondBr(builder.CreateICmpEQ(phaseOutcome, builder.getInt32(VERNON_CPU_RANGE_COMPLETE_V1)),
                             recordCompletion, rangeInvalid);

        builder.SetInsertPoint(recordCompletion);
        llvm::Value *completedAddress = builder.CreateStructGEP(rangeType, rangePointer, 14);
        llvm::Value *completed = builder.CreateLoad(sizeType, completedAddress);
        builder.CreateStore(builder.CreateAdd(completed, llvm::ConstantInt::get(sizeType, 1)), completedAddress);
        builder.CreateBr(rangeNext);

        builder.SetInsertPoint(phaseFailure);
        builder.CreateRet(laneStatus);
    } else {
        llvm::BasicBlock *laneCompleted = llvm::BasicBlock::Create(context, "lane_completed", wrapper);
        llvm::BasicBlock *laneFailure = llvm::BasicBlock::Create(context, "lane_failure", wrapper);
        builder.CreateCondBr(builder.CreateICmpEQ(laneStatus, builder.getInt32(0)), laneCompleted, laneFailure);
        builder.SetInsertPoint(laneCompleted);
        llvm::Value *completedAddress = builder.CreateStructGEP(rangeType, rangePointer, 14);
        llvm::Value *completed = builder.CreateLoad(sizeType, completedAddress);
        builder.CreateStore(builder.CreateAdd(completed, llvm::ConstantInt::get(sizeType, 1)), completedAddress);
        builder.CreateBr(rangeNext);
        builder.SetInsertPoint(laneFailure);
        builder.CreateRet(laneStatus);
    }

    builder.SetInsertPoint(rangeNext);
    llvm::Value *nextLane = builder.CreateAdd(lane, builder.getInt64(1));
    lane->addIncoming(nextLane, rangeNext);
    builder.CreateBr(rangeLoop);

    builder.SetInsertPoint(rangeDone);
    builder.CreateRet(builder.getInt32(0));

    builder.SetInsertPoint(rangeInvalid);
    builder.CreateRet(builder.getInt32(1));
    return llvm::Error::success();
}

} // namespace vernon
