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
#include "llvm/Transforms/Utils/ModuleUtils.h"

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
        VERNON_CPU_LANE_ADDRESS_SYMBOL,
        llvm::FunctionType::get(
            builder.getInt64Ty(),
            {builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty()}, false));
    llvm::Value *frameInteger = builder.CreateCall(
        laneAddress,
        {builder.getInt64(VERNON_CPU_LANE_COROUTINE_FRAME_SITE), frameSize, frameAlignment, builder.getInt64(0)},
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
                    callee && callee->getName() == VERNON_CPU_WORKGROUP_BARRIER_SYMBOL)
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
    function->setLinkage(llvm::GlobalValue::InternalLinkage);
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

        if (packing.size != pointerBytes)
            return invalidAbi("CPU TensorView storage handle has an incompatible size");
        llvm::LoadInst *rawPointerBits = builder.CreateLoad(sizeType, address, "buffer_address");
        rawPointerBits->setAlignment(llvm::Align(1));
        llvm::Value *rawPointer = builder.CreateIntToPtr(rawPointerBits, pointerType, "buffer");
        for (size_t leafIndex = 0; leafIndex < packing.tensorLeafElementSizes.size(); ++leafIndex) {
            llvm::Type *allocatedType = function->getArg(loweredIndex++)->getType();
            // Every dispatch path rejects a written view that overlaps another argument (the language contract's
            // writable-overlap rule), so a view's accesses never meet another's writes. The leaves of one view share a
            // base, so only a single-leaf view is marked.
            if (packing.tensorLeafElementSizes.size() == 1)
                function->addParamAttr(static_cast<unsigned>(loweredIndex), llvm::Attribute::NoAlias);
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
            argumentsToCall.append({rawPointer, rawPointer, offset, extent, stride});
        }
    }
    argumentsToCall.push_back(textures);
    if (metadata.requiresPhases) {
        argumentsToCall.push_back(results);
        llvm::FunctionCallee laneAddress = module.getOrInsertFunction(
            VERNON_CPU_LANE_ADDRESS_SYMBOL,
            llvm::FunctionType::get(
                builder.getInt64Ty(),
                {builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty(), builder.getInt64Ty()}, false));
        llvm::Value *slotInteger =
            builder.CreateCall(laneAddress, {builder.getInt64(VERNON_CPU_LANE_COROUTINE_HANDLE_SITE),
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
        // Both range loops call the lane wrapper, so the entry has two call sites and the inliner's single-caller
        // bonus no longer applies. The fast loop hoists and vectorizes only with the body inlined.
        call->addFnAttr(llvm::Attribute::AlwaysInline);
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
        (rangeLayout->getSizeInBytes() != sizeof(VernonCpuRange) ||
         rangeLayout->getElementOffset(6) != offsetof(VernonCpuRange, grid) ||
         rangeLayout->getElementOffset(9) != offsetof(VernonCpuRange, lane_begin) ||
         rangeLayout->getElementOffset(16) != offsetof(VernonCpuRange, lane_arguments) ||
         rangeLayout->getElementOffset(18) != offsetof(VernonCpuRange, lane_table_count)))
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
    llvm::SmallVector<llvm::Value *, 3> validationGroup;
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
        validationGroup.push_back(groupDimension);
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

    // Entry-block allocas are static, so SROA and mem2reg can promote the packed record after inlining.
    llvm::IRBuilder<> allocaBuilder(dispatchEntry, dispatchEntry->getFirstInsertionPt());
    auto *packedType = llvm::ArrayType::get(i8, std::max<uint64_t>(metadata.argumentsSize, 1));
    llvm::AllocaInst *packed = allocaBuilder.CreateAlloca(packedType, nullptr, "range_arguments");
    packed->setAlignment(llvm::Align(16));
    llvm::AllocaInst *laneInvocation = allocaBuilder.CreateAlloca(invocationType, nullptr, "lane_invocation");
    laneInvocation->setAlignment(module.getDataLayout().getABITypeAlign(invocationType));
    builder.CreateStore(packed, builder.CreateStructGEP(invocationType, laneInvocation, 0));
    builder.CreateStore(llvm::ConstantInt::get(sizeType, metadata.argumentsSize),
                        builder.CreateStructGEP(invocationType, laneInvocation, 1));
    builder.CreateStore(llvm::ConstantInt::get(sizeType, metadata.resultsSize),
                        builder.CreateStructGEP(invocationType, laneInvocation, 3));
    builder.CreateStore(rangeTextures, builder.CreateStructGEP(invocationType, laneInvocation, 4));
    llvm::BasicBlock *rangeDispatch = llvm::BasicBlock::Create(context, "range_dispatch", wrapper);
    builder.CreateCondBr(validRange, rangeDispatch, rangeInvalid);

    auto copyArguments = [&](llvm::Value *target, llvm::Value *source, llvm::Value *sourceSize) {
        if (!metadata.argumentsSize)
            return;
        builder.CreateMemSet(target, builder.getInt8(0), metadata.argumentsSize, llvm::MaybeAlign(16));
        if (hostProvidedArgumentsSize(metadata)) {
            llvm::Value *copySize =
                builder.CreateSelect(builder.CreateICmpULT(sourceSize, builder.getInt64(metadata.argumentsSize)),
                                     sourceSize, builder.getInt64(metadata.argumentsSize));
            builder.CreateMemCpy(target, llvm::Align(16), source, llvm::Align(1), copySize);
        }
    };
    auto storeBuiltins = [&](llvm::Value *packed, llvm::ArrayRef<llvm::Value *> global,
                             llvm::ArrayRef<llvm::Value *> local, llvm::ArrayRef<llvm::Value *> group) {
        for (const CpuAbiArgumentPacking &packing : metadata.sourceArguments) {
            llvm::ArrayRef<llvm::Value *> coordinates = packing.builtin == "global_invocation_id"  ? global
                                                        : packing.builtin == "local_invocation_id" ? local
                                                        : packing.builtin == "workgroup_id"
                                                            ? group
                                                            : llvm::ArrayRef<llvm::Value *>();
            if (coordinates.empty())
                continue;
            llvm::Value *target = builder.CreateGEP(i8, packed, builder.getInt64(packing.offset));
            if (packing.size == sizeof(uint64_t)) {
                llvm::StoreInst *store = builder.CreateStore(coordinates.front(), target);
                store->setAlignment(llvm::Align(1));
                continue;
            }
            for (size_t dimension = 0; dimension < 3 && (dimension + 1) * sizeof(uint32_t) <= packing.size;
                 ++dimension) {
                llvm::Value *component = builder.CreateTrunc(coordinates[dimension], i32);
                llvm::StoreInst *store =
                    builder.CreateStore(component, builder.CreateGEP(i8, target, builder.getInt64(dimension * 4)));
                store->setAlignment(llvm::Align(1));
            }
        }
    };

    builder.SetInsertPoint(rangeDispatch);
    if (metadata.requiresPhases) {
        builder.CreateBr(rangeLoop);
    } else {
        // Without barriers no lane observes the range's progress fields while the range runs, so the fast loop keeps
        // them in registers and publishes them once on exit. The lane helpers read `active_lane` during the call, so
        // kernels that use them still store it per lane. The inner loop runs along one workgroup row with a linear
        // induction variable, which keeps it free of divisions and vectorizable.
        //
        // The loop runs in its own function whose buffer pointers are `noalias` parameters, written into the packed
        // record before the loop so the kernel's pointers derive from them. Inlining the kernel declares its own
        // `noalias` scopes inside the loop, which cover one iteration only; these cover the whole range, which the
        // vectorizer needs.
        const bool storesActiveLane = module.getFunction(VERNON_CPU_LANE_ADDRESS_SYMBOL) != nullptr ||
                                      module.getFunction(VERNON_CPU_WORKGROUP_IS_LEADER_SYMBOL) != nullptr;
        llvm::SmallVector<const CpuAbiArgumentPacking *> exclusiveBuffers;
        for (const CpuAbiArgumentPacking &packing : metadata.sourceArguments)
            if (packing.kind == CpuAbiArgumentKind::TensorView && packing.tensorLeafElementSizes.size() == 1)
                exclusiveBuffers.push_back(&packing);
        enum FastParameter : unsigned {
            FastRange,
            FastArguments,
            FastArgumentsSize,
            FastResults,
            FastTextures,
            FastLaneBegin,
            FastLaneEnd,
            FastWorkgroup,
            FastGroup = FastWorkgroup + 3,
            FastGridX = FastGroup + 3,
            FastGridY,
            FastGroupCount,
            FastBuffers,
        };
        llvm::SmallVector<llvm::Type *> fastParameters{pointerType, pointerType, i64, pointerType, pointerType};
        fastParameters.append(FastBuffers - FastLaneBegin, i64);
        fastParameters.append(exclusiveBuffers.size(), pointerType);
        llvm::Function *fast = llvm::Function::Create(llvm::FunctionType::get(i32, fastParameters, false),
                                                      llvm::GlobalValue::InternalLinkage,
                                                      metadata.exportedWrapperSymbol + ".range", module);
        for (size_t index = 0; index < exclusiveBuffers.size(); ++index)
            fast->addParamAttr(static_cast<unsigned>(FastBuffers + index), llvm::Attribute::NoAlias);

        llvm::BasicBlock *fastCall = llvm::BasicBlock::Create(context, "fast_call", wrapper);
        builder.CreateCondBr(hasLaneTables, rangeLoop, fastCall);
        builder.SetInsertPoint(fastCall);
        llvm::SmallVector<llvm::Value *> fastArguments{
            rangePointer, rangeArguments, rangeArgumentsSize, rangeResults, rangeTextures, laneBegin, laneEnd};
        fastArguments.append(validationWorkgroup.begin(), validationWorkgroup.end());
        fastArguments.append(validationGroup.begin(), validationGroup.end());
        fastArguments.append({validationGrid[0], validationGrid[1], builder.getInt64(1)});
        auto appendExclusiveBuffers = [&](llvm::SmallVectorImpl<llvm::Value *> &target, llvm::Value *record) {
            for (const CpuAbiArgumentPacking *packing : exclusiveBuffers) {
                llvm::LoadInst *bits = builder.CreateLoad(
                    sizeType, builder.CreateGEP(i8, record, builder.getInt64(packing->offset)), "exclusive_bits");
                bits->setAlignment(llvm::Align(1));
                target.push_back(builder.CreateIntToPtr(bits, pointerType, "exclusive_buffer"));
            }
        };
        appendExclusiveBuffers(fastArguments, rangeArguments);
        builder.CreateRet(builder.CreateCall(fast, fastArguments));

        llvm::BasicBlock *fastSetup = llvm::BasicBlock::Create(context, "fast_setup", fast);
        llvm::BasicBlock *fastRow = llvm::BasicBlock::Create(context, "fast_row", fast);
        llvm::BasicBlock *fastLane = llvm::BasicBlock::Create(context, "fast_lane", fast);
        llvm::BasicBlock *fastNext = llvm::BasicBlock::Create(context, "fast_next", fast);
        llvm::BasicBlock *fastRowNext = llvm::BasicBlock::Create(context, "fast_row_next", fast);
        llvm::BasicBlock *fastDone = llvm::BasicBlock::Create(context, "fast_done", fast);
        llvm::BasicBlock *fastFailure = llvm::BasicBlock::Create(context, "fast_failure", fast);
        builder.SetInsertPoint(fastSetup);
        llvm::Value *fastRange = fast->getArg(FastRange);
        llvm::Value *fastBegin = fast->getArg(FastLaneBegin);
        llvm::Value *fastEnd = fast->getArg(FastLaneEnd);
        llvm::Value *fastWorkgroup[] = {fast->getArg(FastWorkgroup), fast->getArg(FastWorkgroup + 1),
                                        fast->getArg(FastWorkgroup + 2)};
        llvm::Value *fastGroup[] = {fast->getArg(FastGroup), fast->getArg(FastGroup + 1), fast->getArg(FastGroup + 2)};
        llvm::AllocaInst *fastPacked = builder.CreateAlloca(packedType, nullptr, "range_arguments");
        fastPacked->setAlignment(llvm::Align(16));
        llvm::AllocaInst *fastInvocation = builder.CreateAlloca(invocationType, nullptr, "lane_invocation");
        fastInvocation->setAlignment(module.getDataLayout().getABITypeAlign(invocationType));
        builder.CreateStore(fastPacked, builder.CreateStructGEP(invocationType, fastInvocation, 0));
        builder.CreateStore(llvm::ConstantInt::get(sizeType, metadata.argumentsSize),
                            builder.CreateStructGEP(invocationType, fastInvocation, 1));
        builder.CreateStore(fast->getArg(FastResults), builder.CreateStructGEP(invocationType, fastInvocation, 2));
        builder.CreateStore(llvm::ConstantInt::get(sizeType, metadata.resultsSize),
                            builder.CreateStructGEP(invocationType, fastInvocation, 3));
        builder.CreateStore(fast->getArg(FastTextures), builder.CreateStructGEP(invocationType, fastInvocation, 4));
        copyArguments(fastPacked, fast->getArg(FastArguments), fast->getArg(FastArgumentsSize));
        for (size_t index = 0; index < exclusiveBuffers.size(); ++index) {
            llvm::StoreInst *store = builder.CreateStore(
                builder.CreatePtrToInt(fast->getArg(static_cast<unsigned>(FastBuffers + index)), sizeType),
                builder.CreateGEP(i8, fastPacked, builder.getInt64(exclusiveBuffers[index]->offset)));
            store->setAlignment(llvm::Align(1));
        }
        llvm::Value *rowBeginX = builder.CreateURem(fastBegin, fastWorkgroup[0]);
        llvm::Value *rowBeginY = builder.CreateURem(builder.CreateUDiv(fastBegin, fastWorkgroup[0]), fastWorkgroup[1]);
        llvm::Value *rowBeginZ = builder.CreateUDiv(fastBegin, builder.CreateMul(fastWorkgroup[0], fastWorkgroup[1]));
        llvm::BasicBlock *fastGroupHead = llvm::BasicBlock::Create(context, "fast_group", fast, fastRow);
        llvm::BasicBlock *fastGroupNext = llvm::BasicBlock::Create(context, "fast_group_next", fast, fastDone);
        builder.CreateBr(fastGroupHead);

        // Groups advance in workgroup-linear order, x fastest, carrying into y and z.
        builder.SetInsertPoint(fastGroupHead);
        llvm::PHINode *groupIndex = builder.CreatePHI(i64, 2, "group_index");
        llvm::PHINode *groupCoordinates[] = {builder.CreatePHI(i64, 2, "group_x"), builder.CreatePHI(i64, 2, "group_y"),
                                             builder.CreatePHI(i64, 2, "group_z")};
        groupIndex->addIncoming(builder.getInt64(0), fastSetup);
        for (size_t dimension = 0; dimension < 3; ++dimension)
            groupCoordinates[dimension]->addIncoming(fastGroup[dimension], fastSetup);
        llvm::Value *currentGroup[] = {groupCoordinates[0], groupCoordinates[1], groupCoordinates[2]};
        llvm::SmallVector<llvm::Value *, 3> groupBase;
        for (size_t dimension = 0; dimension < 3; ++dimension)
            groupBase.push_back(builder.CreateMul(currentGroup[dimension], fastWorkgroup[dimension]));
        builder.CreateBr(fastRow);

        builder.SetInsertPoint(fastRow);
        llvm::PHINode *rowLane = builder.CreatePHI(i64, 2, "row_lane");
        llvm::PHINode *rowX = builder.CreatePHI(i64, 2, "row_x");
        llvm::PHINode *rowY = builder.CreatePHI(i64, 2, "row_y");
        llvm::PHINode *rowZ = builder.CreatePHI(i64, 2, "row_z");
        rowLane->addIncoming(fastBegin, fastGroupHead);
        rowX->addIncoming(rowBeginX, fastGroupHead);
        rowY->addIncoming(rowBeginY, fastGroupHead);
        rowZ->addIncoming(rowBeginZ, fastGroupHead);
        llvm::Value *rowLimit = builder.CreateAdd(rowLane, builder.CreateSub(fastWorkgroup[0], rowX));
        llvm::Value *rowEnd = builder.CreateSelect(builder.CreateICmpULT(rowLimit, fastEnd), rowLimit, fastEnd);
        llvm::Value *globalY = builder.CreateAdd(groupBase[1], rowY);
        llvm::Value *globalZ = builder.CreateAdd(groupBase[2], rowZ);
        llvm::Value *rowGlobalX = builder.CreateAdd(groupBase[0], rowX);
        builder.CreateBr(fastLane);

        builder.SetInsertPoint(fastLane);
        llvm::PHINode *fastLaneIndex = builder.CreatePHI(i64, 2, "fast_lane_index");
        fastLaneIndex->addIncoming(rowLane, fastRow);
        if (storesActiveLane)
            builder.CreateStore(builder.CreateZExtOrTrunc(fastLaneIndex, sizeType),
                                builder.CreateStructGEP(rangeType, fastRange, 11));
        llvm::Value *step = builder.CreateSub(fastLaneIndex, rowLane);
        llvm::Value *fastLocal[] = {builder.CreateAdd(rowX, step), rowY, rowZ};
        llvm::Value *fastGlobal[] = {builder.CreateAdd(rowGlobalX, step), globalY, globalZ};
        storeBuiltins(fastPacked, fastGlobal, fastLocal, currentGroup);
        llvm::CallInst *fastStatus = builder.CreateCall(laneWrapper, {fastInvocation});
        builder.CreateCondBr(builder.CreateICmpEQ(fastStatus, builder.getInt32(0)), fastNext, fastFailure);

        builder.SetInsertPoint(fastNext);
        llvm::Value *fastNextLane = builder.CreateAdd(fastLaneIndex, builder.getInt64(1));
        fastLaneIndex->addIncoming(fastNextLane, fastNext);
        builder.CreateCondBr(builder.CreateICmpULT(fastNextLane, rowEnd), fastLane, fastRowNext);

        builder.SetInsertPoint(fastRowNext);
        llvm::Value *nextY = builder.CreateAdd(rowY, builder.getInt64(1));
        llvm::Value *wrapsY = builder.CreateICmpEQ(nextY, fastWorkgroup[1]);
        rowLane->addIncoming(rowEnd, fastRowNext);
        rowX->addIncoming(builder.getInt64(0), fastRowNext);
        rowY->addIncoming(builder.CreateSelect(wrapsY, builder.getInt64(0), nextY), fastRowNext);
        rowZ->addIncoming(builder.CreateSelect(wrapsY, builder.CreateAdd(rowZ, builder.getInt64(1)), rowZ),
                          fastRowNext);
        builder.CreateCondBr(builder.CreateICmpULT(rowEnd, fastEnd), fastRow, fastGroupNext);

        builder.SetInsertPoint(fastGroupNext);
        llvm::Value *nextGroupIndex = builder.CreateAdd(groupIndex, builder.getInt64(1));
        llvm::Value *nextGroupX = builder.CreateAdd(currentGroup[0], builder.getInt64(1));
        llvm::Value *wrapsGroupX = builder.CreateICmpEQ(nextGroupX, fast->getArg(FastGridX));
        llvm::Value *carriedGroupY =
            builder.CreateSelect(wrapsGroupX, builder.CreateAdd(currentGroup[1], builder.getInt64(1)), currentGroup[1]);
        llvm::Value *wrapsGroupY = builder.CreateICmpEQ(carriedGroupY, fast->getArg(FastGridY));
        groupIndex->addIncoming(nextGroupIndex, fastGroupNext);
        groupCoordinates[0]->addIncoming(builder.CreateSelect(wrapsGroupX, builder.getInt64(0), nextGroupX),
                                         fastGroupNext);
        groupCoordinates[1]->addIncoming(builder.CreateSelect(wrapsGroupY, builder.getInt64(0), carriedGroupY),
                                         fastGroupNext);
        groupCoordinates[2]->addIncoming(
            builder.CreateSelect(wrapsGroupY, builder.CreateAdd(currentGroup[2], builder.getInt64(1)), currentGroup[2]),
            fastGroupNext);
        builder.CreateCondBr(builder.CreateICmpULT(nextGroupIndex, fast->getArg(FastGroupCount)), fastGroupHead,
                             fastDone);

        // A chunk has no range descriptor to publish to; a range runs one group.
        auto publishProgress = [&](llvm::Value *lastLane, llvm::Value *finished) {
            llvm::BasicBlock *publish = llvm::BasicBlock::Create(context, "publish", fast);
            llvm::BasicBlock *published = llvm::BasicBlock::Create(context, "published", fast);
            builder.CreateCondBr(builder.CreateIsNull(fastRange), published, publish);
            builder.SetInsertPoint(publish);
            builder.CreateStore(builder.CreateZExtOrTrunc(lastLane, sizeType),
                                builder.CreateStructGEP(rangeType, fastRange, 11));
            llvm::Value *completedAddress = builder.CreateStructGEP(rangeType, fastRange, 14);
            llvm::Value *completed = builder.CreateLoad(sizeType, completedAddress);
            builder.CreateStore(builder.CreateAdd(completed, builder.CreateZExtOrTrunc(finished, sizeType)),
                                completedAddress);
            builder.CreateBr(published);
            builder.SetInsertPoint(published);
        };
        builder.SetInsertPoint(fastDone);
        publishProgress(builder.CreateSub(fastEnd, builder.getInt64(1)), builder.CreateSub(fastEnd, fastBegin));
        builder.CreateRet(builder.getInt32(0));

        builder.SetInsertPoint(fastFailure);
        publishProgress(fastLaneIndex, builder.CreateSub(fastLaneIndex, fastBegin));
        builder.CreateRet(fastStatus);

        // A view's stride is a runtime value, so the loop above addresses its elements with gathers and scatters.
        // The copy below pins every innermost stride to 1 after the arguments are copied, which makes those
        // accesses consecutive; both copies compute the same addresses whenever the copy runs.
        llvm::SmallVector<CpuCallLanePacking> unitStrides;
        const uint64_t hostSize = hostProvidedArgumentsSize(metadata);
        for (const CpuCallLanePacking &stride : metadata.innermostStrides)
            if (stride.offset + stride.size <= hostSize && (stride.size == 4 || stride.size == 8))
                unitStrides.push_back(stride);
        if (!unitStrides.empty()) {
            llvm::ValueToValueMapTy cloned;
            llvm::Function *unit = llvm::CloneFunction(fast, cloned);
            unit->setName(metadata.exportedWrapperSymbol + ".range.unit");
            builder.SetInsertPoint(llvm::cast<llvm::BasicBlock>(cloned[fastSetup])->getTerminator());
            auto *unitPacked = llvm::cast<llvm::Value>(cloned[fastPacked]);
            for (const CpuCallLanePacking &stride : unitStrides) {
                llvm::StoreInst *store =
                    builder.CreateStore(llvm::ConstantInt::get(builder.getIntNTy(8 * stride.size), 1),
                                        builder.CreateGEP(i8, unitPacked, builder.getInt64(stride.offset)));
                store->setAlignment(llvm::Align(1));
            }

            llvm::Function *select = llvm::Function::Create(fast->getFunctionType(), llvm::GlobalValue::InternalLinkage,
                                                            metadata.exportedWrapperSymbol + ".range.select", module);
            fast->replaceAllUsesWith(select);
            llvm::BasicBlock *selectEntry = llvm::BasicBlock::Create(context, "entry", select);
            llvm::BasicBlock *selectUnit = llvm::BasicBlock::Create(context, "unit_strides", select);
            llvm::BasicBlock *selectGeneral = llvm::BasicBlock::Create(context, "general_strides", select);
            builder.SetInsertPoint(selectEntry);
            llvm::Value *allUnit = builder.getTrue();
            for (const CpuCallLanePacking &stride : unitStrides) {
                llvm::Type *strideType = builder.getIntNTy(8 * stride.size);
                llvm::LoadInst *value = builder.CreateLoad(
                    strideType, builder.CreateGEP(i8, select->getArg(FastArguments), builder.getInt64(stride.offset)));
                value->setAlignment(llvm::Align(1));
                allUnit =
                    builder.CreateAnd(allUnit, builder.CreateICmpEQ(value, llvm::ConstantInt::get(strideType, 1)));
            }
            builder.CreateCondBr(allUnit, selectUnit, selectGeneral);
            llvm::SmallVector<llvm::Value *> forwarded;
            for (llvm::Argument &argument : select->args())
                forwarded.push_back(&argument);
            builder.SetInsertPoint(selectUnit);
            builder.CreateRet(builder.CreateCall(unit, forwarded));
            builder.SetInsertPoint(selectGeneral);
            builder.CreateRet(builder.CreateCall(fast, forwarded));
            // Inlined into the selector, the loops lose the `noalias` parameters that let them vectorize.
            fast->addFnAttr(llvm::Attribute::NoInline);
            unit->addFnAttr(llvm::Attribute::NoInline);
            fast = select;
        }

        bool usesWorkgroupHelpers = false;
        for (const char *helper : {VERNON_CPU_WORKGROUP_ADDRESS_SYMBOL, VERNON_CPU_LANE_ADDRESS_SYMBOL,
                                   VERNON_CPU_WORKGROUP_BARRIER_SYMBOL, VERNON_CPU_WORKGROUP_IS_LEADER_SYMBOL})
            if (const llvm::Function *declared = module.getFunction(helper))
                usesWorkgroupHelpers |= !declared->use_empty();
        if (!usesWorkgroupHelpers && !metadata.resultsSize && !metadata.requiresTextureCallbacks) {
            auto *chunkType =
                llvm::StructType::get(context, {sizeType, pointerType, sizeType, llvm::ArrayType::get(i32, 3),
                                                llvm::ArrayType::get(i32, 3), i64, i64});
            const llvm::StructLayout *chunkLayout = module.getDataLayout().getStructLayout(chunkType);
            if (pointerBits == sizeof(void *) * 8 &&
                (chunkLayout->getSizeInBytes() != sizeof(VernonCpuChunk) ||
                 chunkLayout->getElementOffset(3) != offsetof(VernonCpuChunk, grid) ||
                 chunkLayout->getElementOffset(5) != offsetof(VernonCpuChunk, group_begin)))
                return invalidAbi("CPU chunk descriptor layout does not match the host ABI");
            llvm::Function *chunkEntry = llvm::Function::Create(wrapperType, llvm::GlobalValue::ExternalLinkage,
                                                                cpuChunkEntrySymbol(metadata), module);
            if (llvm::Triple(module.getTargetTriple()).isOSWindows())
                chunkEntry->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
            else
                chunkEntry->setVisibility(llvm::GlobalValue::DefaultVisibility);
            llvm::BasicBlock *chunkBegin = llvm::BasicBlock::Create(context, "entry", chunkEntry);
            llvm::BasicBlock *chunkSized = llvm::BasicBlock::Create(context, "chunk_sized", chunkEntry);
            llvm::BasicBlock *chunkFields = llvm::BasicBlock::Create(context, "chunk_fields", chunkEntry);
            llvm::BasicBlock *chunkRun = llvm::BasicBlock::Create(context, "chunk_run", chunkEntry);
            llvm::BasicBlock *chunkInvalid = llvm::BasicBlock::Create(context, "chunk_invalid", chunkEntry);
            llvm::Value *chunk = chunkEntry->getArg(0);
            builder.SetInsertPoint(chunkBegin);
            builder.CreateCondBr(builder.CreateIsNull(chunk), chunkInvalid, chunkSized);
            builder.SetInsertPoint(chunkSized);
            llvm::LoadInst *chunkSize = builder.CreateLoad(sizeType, builder.CreateStructGEP(chunkType, chunk, 0));
            chunkSize->setAlignment(llvm::Align(1));
            builder.CreateCondBr(
                builder.CreateICmpEQ(chunkSize, llvm::ConstantInt::get(sizeType, chunkLayout->getSizeInBytes())),
                chunkFields, chunkInvalid);

            builder.SetInsertPoint(chunkFields);
            llvm::LoadInst *chunkArguments =
                builder.CreateLoad(pointerType, builder.CreateStructGEP(chunkType, chunk, 1));
            chunkArguments->setAlignment(llvm::Align(1));
            llvm::LoadInst *chunkArgumentsSizeValue =
                builder.CreateLoad(sizeType, builder.CreateStructGEP(chunkType, chunk, 2));
            chunkArgumentsSizeValue->setAlignment(llvm::Align(1));
            llvm::Value *chunkArgumentsSize = builder.CreateZExtOrTrunc(chunkArgumentsSizeValue, i64);
            llvm::SmallVector<llvm::Value *, 3> chunkGrid;
            llvm::SmallVector<llvm::Value *, 3> chunkWorkgroup;
            llvm::Value *chunkValid = llvm::ConstantInt::getTrue(context);
            for (uint64_t dimension = 0; dimension < 3; ++dimension) {
                llvm::Value *index[] = {builder.getInt64(0), builder.getInt64(dimension)};
                llvm::LoadInst *grid =
                    builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3),
                                                              builder.CreateStructGEP(chunkType, chunk, 3), index));
                grid->setAlignment(llvm::Align(1));
                llvm::LoadInst *workgroup =
                    builder.CreateLoad(i32, builder.CreateGEP(llvm::ArrayType::get(i32, 3),
                                                              builder.CreateStructGEP(chunkType, chunk, 4), index));
                workgroup->setAlignment(llvm::Align(1));
                chunkGrid.push_back(builder.CreateZExt(grid, i64));
                chunkWorkgroup.push_back(builder.CreateZExt(workgroup, i64));
                chunkValid = builder.CreateAnd(chunkValid, builder.CreateICmpNE(grid, builder.getInt32(0)));
                chunkValid = builder.CreateAnd(chunkValid, builder.CreateICmpNE(workgroup, builder.getInt32(0)));
            }
            llvm::LoadInst *groupBegin = builder.CreateLoad(i64, builder.CreateStructGEP(chunkType, chunk, 5));
            groupBegin->setAlignment(llvm::Align(1));
            llvm::LoadInst *groupEnd = builder.CreateLoad(i64, builder.CreateStructGEP(chunkType, chunk, 6));
            groupEnd->setAlignment(llvm::Align(1));
            auto [chunkWorkgroupXY, chunkWorkgroupXYOverflow] = checkedMultiply(chunkWorkgroup[0], chunkWorkgroup[1]);
            auto [chunkWorkgroupVolume, chunkWorkgroupVolumeOverflow] =
                checkedMultiply(chunkWorkgroupXY, chunkWorkgroup[2]);
            auto [chunkGlobalXY, chunkGlobalXYOverflow] = checkedMultiply(
                builder.CreateMul(chunkGrid[0], chunkWorkgroup[0]), builder.CreateMul(chunkGrid[1], chunkWorkgroup[1]));
            auto [chunkGlobalVolume, chunkGlobalVolumeOverflow] =
                checkedMultiply(chunkGlobalXY, builder.CreateMul(chunkGrid[2], chunkWorkgroup[2]));
            llvm::Value *gridXY = builder.CreateMul(chunkGrid[0], chunkGrid[1]);
            auto [gridVolume, gridVolumeOverflow] = checkedMultiply(gridXY, chunkGrid[2]);
            for (llvm::Value *overflow : {chunkWorkgroupXYOverflow, chunkWorkgroupVolumeOverflow, chunkGlobalXYOverflow,
                                          chunkGlobalVolumeOverflow, gridVolumeOverflow})
                chunkValid = builder.CreateAnd(chunkValid, builder.CreateNot(overflow));
            chunkValid = builder.CreateAnd(chunkValid, builder.CreateICmpULT(groupBegin, groupEnd));
            chunkValid = builder.CreateAnd(chunkValid, builder.CreateICmpULE(groupEnd, gridVolume));
            if (const uint64_t hostSize = hostProvidedArgumentsSize(metadata); hostSize)
                chunkValid = builder.CreateAnd(
                    chunkValid,
                    builder.CreateAnd(builder.CreateIsNotNull(chunkArguments),
                                      builder.CreateICmpUGE(chunkArgumentsSize, builder.getInt64(hostSize))));
            builder.CreateCondBr(chunkValid, chunkRun, chunkInvalid);

            builder.SetInsertPoint(chunkRun);
            llvm::Value *nullPointer = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pointerType));
            llvm::SmallVector<llvm::Value *> chunkCall{nullPointer,         chunkArguments, chunkArgumentsSize,
                                                       nullPointer,         nullPointer,    builder.getInt64(0),
                                                       chunkWorkgroupVolume};
            chunkCall.append(chunkWorkgroup.begin(), chunkWorkgroup.end());
            chunkCall.push_back(builder.CreateURem(groupBegin, chunkGrid[0]));
            chunkCall.push_back(builder.CreateURem(builder.CreateUDiv(groupBegin, chunkGrid[0]), chunkGrid[1]));
            chunkCall.push_back(builder.CreateUDiv(groupBegin, gridXY));
            chunkCall.append({chunkGrid[0], chunkGrid[1], builder.CreateSub(groupEnd, groupBegin)});
            appendExclusiveBuffers(chunkCall, chunkArguments);
            builder.CreateRet(builder.CreateCall(fast, chunkCall));

            builder.SetInsertPoint(chunkInvalid);
            builder.CreateRet(builder.getInt32(1));
        }
    }

    builder.SetInsertPoint(rangeLoop);
    llvm::PHINode *lane = builder.CreatePHI(i64, 2, "lane");
    lane->addIncoming(laneBegin, rangeDispatch);
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
    copyArguments(packed, selectedArguments, rangeArgumentsSize);
    builder.CreateStore(selectedResults, builder.CreateStructGEP(invocationType, laneInvocation, 2));
    storeBuiltins(packed, global, local, group);
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
        builder.CreateCondBr(builder.CreateICmpEQ(phaseOutcome, builder.getInt32(VERNON_CPU_RANGE_COMPLETE)),
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

llvm::Error emitCpuStaticRegistration(llvm::Module &module, const std::vector<CpuAbiWrapperMetadata> &entries) {
    if (entries.empty())
        return llvm::Error::success();
    llvm::LLVMContext &context = module.getContext();
    const llvm::DataLayout &layout = module.getDataLayout();
    auto *pointerType = llvm::PointerType::getUnqual(context);
    llvm::IntegerType *sizeType = layout.getIntPtrType(context);
    auto *entryType = llvm::StructType::get(context, {pointerType, sizeType, pointerType});
    const llvm::StructLayout *entryLayout = layout.getStructLayout(entryType);
    if (sizeType->getBitWidth() == sizeof(void *) * 8 &&
        (entryLayout->getSizeInBytes() != sizeof(VernonStaticCpuEntry) ||
         entryLayout->getElementOffset(1) != offsetof(VernonStaticCpuEntry, symbol.size) ||
         entryLayout->getElementOffset(2) != offsetof(VernonStaticCpuEntry, entry_point)))
        return invalidAbi("CPU static entry layout does not match the host ABI");

    const bool supportsComdat = llvm::Triple(module.getTargetTriple()).supportsCOMDAT();
    std::vector<llvm::Constant *> rows;
    rows.reserve(entries.size());
    for (const CpuAbiWrapperMetadata &entry : entries) {
        llvm::Function *wrapper = module.getFunction(entry.exportedWrapperSymbol);
        if (!wrapper || wrapper->isDeclaration())
            return invalidAbi("CPU static registration names missing wrapper '" + entry.exportedWrapperSymbol + "'");
        // Wrapper names hash the module content, and Programs embed identical
        // helper modules, so one application may link several copies of an
        // object. The copies must fold instead of colliding; each copy's
        // constructor then registers the surviving definition.
        for (llvm::Function *folded : {wrapper, module.getFunction(cpuChunkEntrySymbol(entry))}) {
            if (!folded || folded->isDeclaration())
                continue;
            folded->setLinkage(llvm::GlobalValue::WeakODRLinkage);
            if (supportsComdat)
                folded->setComdat(module.getOrInsertComdat(folded->getName()));
        }
        llvm::Constant *name = llvm::ConstantDataArray::getString(context, entry.exportedWrapperSymbol, false);
        auto *nameGlobal = new llvm::GlobalVariable(module, name->getType(), true, llvm::GlobalValue::PrivateLinkage,
                                                    name, "vernon.cpu.entry_name");
        nameGlobal->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
        rows.push_back(llvm::ConstantStruct::get(
            entryType, {nameGlobal, llvm::ConstantInt::get(sizeType, entry.exportedWrapperSymbol.size()), wrapper}));
    }
    auto *tableType = llvm::ArrayType::get(entryType, rows.size());
    auto *table = new llvm::GlobalVariable(module, tableType, true, llvm::GlobalValue::PrivateLinkage,
                                           llvm::ConstantArray::get(tableType, rows), "vernon.cpu.entries");

    // VernonStatus is a C enum, which every supported target lowers to i32.
    auto *registryType = llvm::FunctionType::get(llvm::Type::getInt32Ty(context), {pointerType, sizeType}, false);
    // A failure can only mean malformed arguments or allocation failure; a
    // missing entry is then reported when the Program loads.
    const auto emitRegistryCall = [&](const char *callee, const char *name) {
        llvm::Function *function =
            llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
                                   llvm::GlobalValue::InternalLinkage, name, module);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
        builder.CreateCall(module.getOrInsertFunction(callee, registryType),
                           {table, llvm::ConstantInt::get(sizeType, rows.size())});
        builder.CreateRetVoid();
        return function;
    };
    llvm::appendToGlobalCtors(module, emitRegistryCall(cpuStaticRegistrationSymbol, "vernon.cpu.register_entries"),
                              65535);
    llvm::appendToGlobalDtors(module, emitRegistryCall(cpuStaticUnregistrationSymbol, "vernon.cpu.unregister_entries"),
                              65535);
    return llvm::Error::success();
}

} // namespace vernon
