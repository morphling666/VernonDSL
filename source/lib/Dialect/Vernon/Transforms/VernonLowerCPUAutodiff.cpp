#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUAutodiff.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::vernon {
namespace {

constexpr StringLiteral kAllocatorBuiltin = "ad_tape_allocator";
constexpr StringLiteral kRootRegionBuiltin = "ad_tape_root_region";

// Field ordinals in the frozen VernonAdTapeAllocator ABI.
enum class AllocatorField : int32_t {
    RequiredBytes = VERNON_AD_TAPE_ALLOCATOR_FIELD_REQUIRED_BYTES,
    Reset = VERNON_AD_TAPE_ALLOCATOR_FIELD_RESET,
    BeginRegion = VERNON_AD_TAPE_ALLOCATOR_FIELD_BEGIN_REGION,
    ReserveRecord = VERNON_AD_TAPE_ALLOCATOR_FIELD_RESERVE_RECORD,
    WriteLeaf = VERNON_AD_TAPE_ALLOCATOR_FIELD_WRITE_LEAF,
    SetChild = VERNON_AD_TAPE_ALLOCATOR_FIELD_SET_CHILD,
    EndRegion = VERNON_AD_TAPE_ALLOCATOR_FIELD_END_REGION,
    Seal = VERNON_AD_TAPE_ALLOCATOR_FIELD_SEAL,
    ReadLeaf = VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_LEAF,
    ReadChild = VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_CHILD,
    ReadExecutedCount = VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_EXECUTED_COUNT,
    ReadExitKind = VERNON_AD_TAPE_ALLOCATOR_FIELD_READ_EXIT_KIND,
};

enum class SemanticCallback : int32_t {
    Reset,
    BeginRegion,
    ReserveRecord,
    WriteLeaf,
    SetChild,
    EndRegion,
    Seal,
    ReadLeaf,
    ReadChild,
    ReadExecutedCount,
    ReadExitKind,
};

AllocatorField physicalField(SemanticCallback callback) {
    switch (callback) {
    case SemanticCallback::Reset:
        return AllocatorField::Reset;
    case SemanticCallback::BeginRegion:
        return AllocatorField::BeginRegion;
    case SemanticCallback::ReserveRecord:
        return AllocatorField::ReserveRecord;
    case SemanticCallback::WriteLeaf:
        return AllocatorField::WriteLeaf;
    case SemanticCallback::SetChild:
        return AllocatorField::SetChild;
    case SemanticCallback::EndRegion:
        return AllocatorField::EndRegion;
    case SemanticCallback::Seal:
        return AllocatorField::Seal;
    case SemanticCallback::ReadLeaf:
        return AllocatorField::ReadLeaf;
    case SemanticCallback::ReadChild:
        return AllocatorField::ReadChild;
    case SemanticCallback::ReadExecutedCount:
        return AllocatorField::ReadExecutedCount;
    case SemanticCallback::ReadExitKind:
        return AllocatorField::ReadExitKind;
    }
}

DictionaryAttr builtinAttributes(MLIRContext *context, StringRef builtin) {
    NamedAttrList attributes;
    attributes.set("vernon.builtin", StringAttr::get(context, builtin));
    attributes.set("vernon.interface", StringAttr::get(context, "input"));
    return attributes.getDictionary(context);
}

bool hasBuiltin(func::FuncOp function, unsigned index, StringRef builtin) {
    auto attribute = function.getArgAttrOfType<StringAttr>(index, "vernon.builtin");
    return attribute && attribute.getValue() == builtin;
}

Value indexConstant(OpBuilder &builder, Location location, uint64_t value) {
    return arith::ConstantIndexOp::create(builder, location, static_cast<int64_t>(value));
}

Value integerConstant(OpBuilder &builder, Location location, unsigned width, uint64_t value) {
    return arith::ConstantIntOp::create(builder, location, value, width);
}

FailureOr<Value> toRawBits(OpBuilder &builder, Location location, Value value) {
    Type type = value.getType();
    Value bits = value;
    unsigned width = 0;
    if (auto floatType = dyn_cast<FloatType>(type)) {
        width = floatType.getWidth();
        bits = arith::BitcastOp::create(builder, location, IntegerType::get(builder.getContext(), width), value);
    } else if (auto integerType = dyn_cast<IntegerType>(type)) {
        width = integerType.getWidth();
    } else if (type.isIndex()) {
        width = 64;
        bits = arith::IndexCastOp::create(builder, location, builder.getI64Type(), value);
    } else {
        return failure();
    }
    if (width > 64)
        return failure();
    if (width < 64)
        bits = arith::ExtUIOp::create(builder, location, builder.getI64Type(), bits);
    return bits;
}

FailureOr<Value> fromRawBits(OpBuilder &builder, Location location, Value raw, Type type) {
    if (type.isIndex())
        return arith::IndexCastOp::create(builder, location, type, raw).getResult();
    Type integerType;
    if (auto floatType = dyn_cast<FloatType>(type))
        integerType = IntegerType::get(builder.getContext(), floatType.getWidth());
    else if (auto targetInteger = dyn_cast<IntegerType>(type))
        integerType = targetInteger;
    else
        return failure();
    Value bits = raw;
    const unsigned width = cast<IntegerType>(integerType).getWidth();
    if (width < 64)
        bits = arith::TruncIOp::create(builder, location, integerType, raw);
    if (isa<FloatType>(type))
        return arith::BitcastOp::create(builder, location, type, bits).getResult();
    return bits;
}

struct RecordAddress {
    Value handle;
    uint64_t payloadSize{};
    uint64_t childCount{};
};

class FunctionLowering {
public:
    explicit FunctionLowering(func::FuncOp function) : function(function) {}

    LogicalResult run() {
        bool hasCapture = false;
        bool hasAutodiffHandles = false;
        function.walk([&](Operation *operation) {
            hasCapture = hasCapture || isa<AdCaptureOp>(operation);
            for (Type type : operation->getOperandTypes())
                hasAutodiffHandles = hasAutodiffHandles || containsLogicalAutodiffHandle(type);
            for (Type type : operation->getResultTypes())
                hasAutodiffHandles = hasAutodiffHandles || containsLogicalAutodiffHandle(type);
        });
        if (!hasCapture && !hasAutodiffHandles)
            return success();
        if (hasCapture) {
            if (failed(prepareForward()))
                return failure();
        } else {
            if (failed(prepareBackward()))
                return failure();
        }

        SmallVector<Operation *> operations;
        function.walk<WalkOrder::PreOrder>([&](Operation *operation) { operations.push_back(operation); });
        for (Operation *operation : operations) {
            if (!operation->getBlock())
                continue;
            if (auto begin = dyn_cast<AdBeginInvocationOp>(operation)) {
                begin.getTape().replaceAllUsesWith(descriptor);
                begin.erase();
            } else if (auto begin = dyn_cast<AdBeginRegionOp>(operation)) {
                if (failed(lowerBeginRegion(begin)))
                    return failure();
            } else if (auto reserve = dyn_cast<AdReserveRecordOp>(operation)) {
                if (failed(lowerReserve(reserve)))
                    return failure();
            } else if (auto write = dyn_cast<AdWriteLeafOp>(operation)) {
                if (failed(lowerWrite(write)))
                    return failure();
            } else if (auto end = dyn_cast<AdEndRegionOp>(operation)) {
                if (failed(lowerEnd(end)))
                    return failure();
            } else if (auto increment = dyn_cast<AdCheckedIncrementOp>(operation)) {
                lowerCheckedIncrement(increment);
            }
        }
        for (AdCaptureOp capture : llvm::make_early_inc_range(function.getOps<AdCaptureOp>()))
            if (failed(lowerCapture(capture)))
                return failure();
        operations.clear();
        function.walk<WalkOrder::PreOrder>([&](Operation *operation) { operations.push_back(operation); });
        for (Operation *operation : operations) {
            if (!operation->getBlock())
                continue;
            if (auto read = dyn_cast<AdReadRecordOffsetOp>(operation)) {
                OpBuilder builder(read);
                read.getRecordOffset().replaceAllUsesWith(indexConstant(builder, read.getLoc(), 0));
                read.erase();
            } else if (auto read = dyn_cast<AdReadExecutedCountOp>(operation)) {
                lowerExecutedCountRead(read);
            } else if (auto read = dyn_cast<AdReadExitKindOp>(operation)) {
                lowerExitKindRead(read);
            } else if (auto read = dyn_cast<AdReadNestedRegionOp>(operation)) {
                lowerNestedRead(read);
            } else if (auto read = dyn_cast<AdReadLeafOp>(operation)) {
                if (failed(lowerLeafRead(read)))
                    return failure();
            }
        }
        for (AdCommitOp commit : llvm::make_early_inc_range(function.getOps<AdCommitOp>()))
            if (failed(lowerCommit(commit)))
                return failure();
        return success();
    }

private:
    OpBuilder builder() const { return OpBuilder(function->getContext()); }

    Value asI64(OpBuilder &builder, Location location, Value value) const {
        if (value.getType().isIndex())
            return arith::IndexCastOp::create(builder, location, builder.getI64Type(), value);
        return value;
    }

    CpuAdCallbackOp emitCallback(OpBuilder &builder, Location location, Value descriptorBits, SemanticCallback callback,
                                 ValueRange arguments, TypeRange results = {}) {
        OperationState state(location, CpuAdCallbackOp::getOperationName());
        state.addOperands(descriptorBits);
        state.addAttribute("callback", builder.getI32IntegerAttr(static_cast<int32_t>(callback)));
        state.addOperands(arguments);
        state.addTypes(results);
        return cast<CpuAdCallbackOp>(builder.create(state));
    }

    LogicalResult prepareForward() {
        if (function.getNumResults() != 1)
            return function.emitError("dynamic CPU autodiff forward requires one result");
        auto resultType = dyn_cast<TupleType>(function.getResultTypes().front());
        if (!resultType || resultType.size() != 3 || !isa<AdTapeType>(resultType.getType(1)) ||
            !isa<AdRegionHeaderType>(resultType.getType(2)))
            return function.emitError("dynamic CPU autodiff forward has an invalid logical result");
        SmallVector<func::ReturnOp> returns;
        function.walk([&](func::ReturnOp operation) { returns.push_back(operation); });
        for (func::ReturnOp operation : returns) {
            auto tuple = operation.getOperand(0).getDefiningOp<TupleCreateOp>();
            if (!tuple || tuple.getElements().size() != 3)
                return operation.emitError("dynamic CPU autodiff forward must return output, tape, and root region");
            operation->setOperand(0, tuple.getElements().front());
            if (tuple->use_empty())
                tuple.erase();
        }
        function.setType(
            FunctionType::get(function.getContext(), function.getArgumentTypes(), {resultType.getType(0)}));

        Block &entry = function.getBody().front();
        for (unsigned index = 0; index < function.getNumArguments(); ++index)
            if (hasBuiltin(function, index, kAllocatorBuiltin)) {
                descriptor = entry.getArgument(index);
                break;
            }
        if (!descriptor)
            return function.emitError("CPU autodiff signature preparation did not materialize the allocator builtin");
        return success();
    }

    LogicalResult prepareBackward() {
        if (function.getNumArguments() < 3 || !isa<AdTapeType>(function.getArgumentTypes()[0]) ||
            !isa<AdRegionHeaderType>(function.getArgumentTypes()[1]))
            return function.emitError("dynamic CPU autodiff backward has an invalid logical tape signature");
        Block &entry = function.getBody().front();
        descriptor = entry.getArgument(0);
        Value rootArgument = entry.getArgument(1);
        descriptor.setType(IndexType::get(function.getContext()));
        rootArgument.setType(IndexType::get(function.getContext()));
        SmallVector<Type> inputs(function.getArgumentTypes());
        inputs[0] = descriptor.getType();
        inputs[1] = rootArgument.getType();
        function.setType(FunctionType::get(function.getContext(), inputs, function.getResultTypes()));
        function.setArgAttrs(0, builtinAttributes(function.getContext(), kAllocatorBuiltin));
        function.setArgAttrs(1, builtinAttributes(function.getContext(), kRootRegionBuiltin));
        OpBuilder entryBuilder = OpBuilder::atBlockBegin(&entry);
        auto rootCast =
            arith::IndexCastOp::create(entryBuilder, function.getLoc(), entryBuilder.getI64Type(), rootArgument);
        rootRegion = rootCast.getResult();
        rootArgument.replaceAllUsesExcept(rootRegion, rootCast);
        regionDescriptors.try_emplace(rootRegion, descriptor);
        return success();
    }

    LogicalResult lowerBeginRegion(AdBeginRegionOp operation) {
        OpBuilder builder(operation);
        Value logicalTape = operation->getOperand(0);
        Value tapeDescriptor = descriptors.lookup(logicalTape);
        if (!tapeDescriptor)
            tapeDescriptor = logicalTape == descriptor ? descriptor : Value{};
        if (!tapeDescriptor)
            return operation.emitError("CPU autodiff region has no allocator descriptor");
        Value parent = integerConstant(builder, operation.getLoc(), 64, 0);
        if (!operation.getParentLink().empty())
            parent = operation.getParentLink().front();
        Value handle = emitCallback(builder, operation.getLoc(), tapeDescriptor, SemanticCallback::BeginRegion,
                                    {parent}, {builder.getI64Type()})
                           .getResult(0);
        descriptors.try_emplace(handle, tapeDescriptor);
        regionDescriptors.try_emplace(handle, tapeDescriptor);

        if (!operation.getParentLink().empty()) {
            Value record = operation.getParentLink()[1];
            auto found = recordAddresses.find(record);
            if (found == recordAddresses.end())
                return operation.emitError("nested CPU autodiff region has no writable parent record");
            const uint64_t ordinal = static_cast<uint64_t>(operation.getChildOrdinalAttr().getInt());
            if (ordinal >= found->second.childCount)
                return operation.emitError("nested CPU autodiff child ordinal exceeds its parent record");
            emitCallback(builder, operation.getLoc(), descriptor, SemanticCallback::SetChild,
                         {found->second.handle, integerConstant(builder, operation.getLoc(), 64, ordinal), handle});
        }
        operation.getRegion().replaceAllUsesWith(handle);
        operation.erase();
        return success();
    }

    LogicalResult lowerReserve(AdReserveRecordOp operation) {
        OpBuilder builder(operation);
        uint64_t childCount = 0;
        for (Operation *user : operation.getRecordOffset().getUsers())
            if (auto child = dyn_cast<AdBeginRegionOp>(user))
                childCount =
                    std::max<uint64_t>(childCount, static_cast<uint64_t>(child.getChildOrdinalAttr().getInt()) + 1);
        const uint64_t payloadSize = static_cast<uint64_t>(operation.getRecordSizeAttr().getInt());
        Value region = operation->getOperand(0);
        Value tapeDescriptor = regionDescriptors.lookup(region);
        if (!tapeDescriptor)
            return operation.emitError("CPU autodiff reservation has no allocator descriptor");
        Value rawResult =
            emitCallback(builder, operation.getLoc(), tapeDescriptor, SemanticCallback::ReserveRecord,
                         {region, integerConstant(builder, operation.getLoc(), 64, payloadSize),
                          integerConstant(builder, operation.getLoc(), 64,
                                          static_cast<uint64_t>(operation.getRecordAlignmentAttr().getInt())),
                          integerConstant(builder, operation.getLoc(), 64, childCount)},
                         {builder.getI64Type()})
                .getResult(0);
        Value result = arith::IndexCastOp::create(builder, operation.getLoc(), builder.getIndexType(), rawResult);
        Value oldOffset = operation.getRecordOffset();
        RecordAddress address{result, payloadSize, childCount};
        recordAddresses.try_emplace(oldOffset, address);
        recordAddresses.try_emplace(result, address);
        oldOffset.replaceAllUsesWith(result);
        operation.erase();
        return success();
    }

    LogicalResult lowerWrite(AdWriteLeafOp operation) {
        auto found = recordAddresses.find(operation.getRecordOffset());
        if (found == recordAddresses.end())
            return operation.emitError("CPU autodiff write has no checked reservation address");
        OpBuilder builder(operation);
        FailureOr<Value> raw = toRawBits(builder, operation.getLoc(), operation.getValue());
        if (failed(raw))
            return operation.emitError("CPU autodiff tape leaf is not a supported scalar");
        const uint64_t size = std::max<uint64_t>(operation.getValue().getType().getIntOrFloatBitWidth() / 8, 1);
        emitCallback(builder, operation.getLoc(), regionDescriptors.lookup(operation->getOperand(0)),
                     SemanticCallback::WriteLeaf,
                     {found->second.handle,
                      integerConstant(builder, operation.getLoc(), 64,
                                      static_cast<uint64_t>(operation.getLeafOffsetAttr().getInt())),
                      *raw, integerConstant(builder, operation.getLoc(), 64, size)});
        operation.erase();
        return success();
    }

    LogicalResult lowerEnd(AdEndRegionOp operation) {
        OpBuilder builder(operation);
        Value region = operation->getOperand(0);
        Value tapeDescriptor = regionDescriptors.lookup(region);
        if (!tapeDescriptor)
            return operation.emitError("CPU autodiff region finalization has no allocator descriptor");
        emitCallback(
            builder, operation.getLoc(), tapeDescriptor, SemanticCallback::EndRegion,
            {region, asI64(builder, operation.getLoc(), operation.getExecutedCount()), operation.getExitKind()});
        operation.erase();
        return success();
    }

    void lowerCheckedIncrement(AdCheckedIncrementOp operation) {
        OpBuilder builder(operation);
        Value counter = asI64(builder, operation.getLoc(), operation.getCounter());
        Value maximum = integerConstant(builder, operation.getLoc(), 64, std::numeric_limits<uint64_t>::max());
        Value overflow = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::eq, counter, maximum);
        Value incremented = arith::AddIOp::create(builder, operation.getLoc(), counter,
                                                  integerConstant(builder, operation.getLoc(), 64, 1));
        Value rawResult = arith::SelectOp::create(builder, operation.getLoc(), overflow, counter, incremented);

        Value result = arith::IndexCastOp::create(builder, operation.getLoc(), builder.getIndexType(), rawResult);
        operation.getResult().replaceAllUsesWith(result);
        operation.erase();
    }

    void lowerExecutedCountRead(AdReadExecutedCountOp operation) {
        OpBuilder builder(operation);
        Value region = operation->getOperand(0);
        Value tapeDescriptor = regionDescriptors.lookup(region);
        Value raw = emitCallback(builder, operation.getLoc(), tapeDescriptor, SemanticCallback::ReadExecutedCount,
                                 {region}, {builder.getI64Type()})
                        .getResult(0);
        Value result = arith::IndexCastOp::create(builder, operation.getLoc(), builder.getIndexType(), raw);
        operation.getExecutedCount().replaceAllUsesWith(result);
        operation.erase();
    }

    void lowerExitKindRead(AdReadExitKindOp operation) {
        OpBuilder builder(operation);
        Value region = operation->getOperand(0);
        Value tapeDescriptor = regionDescriptors.lookup(region);
        Value result = emitCallback(builder, operation.getLoc(), tapeDescriptor, SemanticCallback::ReadExitKind,
                                    {region}, {builder.getI32Type()})
                           .getResult(0);
        operation.getExitKind().replaceAllUsesWith(result);
        operation.erase();
    }

    void lowerNestedRead(AdReadNestedRegionOp operation) {
        OpBuilder builder(operation);
        Value region = operation->getOperand(0);
        Value tapeDescriptor = regionDescriptors.lookup(region);
        Value result = emitCallback(builder, operation.getLoc(), tapeDescriptor, SemanticCallback::ReadChild,
                                    {region, asI64(builder, operation.getLoc(), operation.getRecordIndex()),
                                     integerConstant(builder, operation.getLoc(), 64,
                                                     static_cast<uint64_t>(operation.getChildOrdinalAttr().getInt()))},
                                    {builder.getI64Type()})
                           .getResult(0);
        regionDescriptors.try_emplace(result, tapeDescriptor);
        operation.getNestedRegion().replaceAllUsesWith(result);
        operation.erase();
    }

    LogicalResult lowerLeafRead(AdReadLeafOp operation) {
        OpBuilder builder(operation);
        Value region = operation->getOperand(0);
        Value tapeDescriptor = regionDescriptors.lookup(region);
        Type type = operation.getValue().getType();
        const uint64_t size = std::max<uint64_t>(type.getIntOrFloatBitWidth() / 8, 1);
        Value raw = emitCallback(builder, operation.getLoc(), tapeDescriptor, SemanticCallback::ReadLeaf,
                                 {region, asI64(builder, operation.getLoc(), operation.getRecordIndex()),
                                  integerConstant(builder, operation.getLoc(), 64,
                                                  static_cast<uint64_t>(operation.getLeafOffsetAttr().getInt())),
                                  integerConstant(builder, operation.getLoc(), 64, size)},
                                 {builder.getI64Type()})
                        .getResult(0);
        FailureOr<Value> value = fromRawBits(builder, operation.getLoc(), raw, type);
        if (failed(value))
            return operation.emitError("CPU autodiff tape read has an unsupported scalar type");
        operation.getValue().replaceAllUsesWith(*value);
        operation.erase();
        return success();
    }

    LogicalResult lowerCapture(AdCaptureOp operation) {
        OpBuilder builder(operation);
        emitCallback(builder, operation.getLoc(), descriptor, SemanticCallback::Reset, {});
        Block &body = operation.getBody().front();
        auto yield = cast<AdCaptureYieldOp>(body.getTerminator());
        Value tape = yield->getOperand(0);
        Value root = yield->getOperand(1);
        for (Operation &nested : llvm::make_early_inc_range(body.without_terminator()))
            nested.moveBefore(operation);
        Value sealStatus =
            emitCallback(builder, operation.getLoc(), descriptor, SemanticCallback::Seal, {}, {builder.getI32Type()})
                .getResult(0);
        Value captureSuccess = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::eq, sealStatus,
                                                     integerConstant(builder, operation.getLoc(), 32, 0));
        Value required = CpuAdRequiredBytesOp::create(builder, operation.getLoc(), builder.getIndexType(), descriptor)
                             .getRequiredBytes();
        Value overflowCode = integerConstant(builder, operation.getLoc(), 32, 2);
        Value overflow =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::eq, sealStatus, overflowCode);
        operation.getTape().replaceAllUsesWith(tape);
        operation.getRootRegion().replaceAllUsesWith(root);
        operation.getSuccess().replaceAllUsesWith(captureSuccess);
        operation.getRequiredBytes().replaceAllUsesWith(required);
        operation.getOverflow().replaceAllUsesWith(overflow);
        operation.erase();
        return success();
    }

    LogicalResult lowerCommit(AdCommitOp operation) {
        OpBuilder builder(operation);
        scf::IfOp conditional = scf::IfOp::create(builder, operation.getLoc(), operation.getCaptureSuccess(), false);
        Block &target = conditional.getThenRegion().front();
        if (!target.empty())
            target.getTerminator()->erase();
        for (Operation &nested : operation.getBody().front().without_terminator())
            target.getOperations().push_back(nested.clone());
        OpBuilder targetBuilder = OpBuilder::atBlockEnd(&target);
        scf::YieldOp::create(targetBuilder, operation.getLoc());
        operation.erase();
        return success();
    }

    func::FuncOp function;
    Value descriptor;
    Value rootRegion;
    DenseMap<Value, Value> descriptors;
    DenseMap<Value, Value> regionDescriptors;
    DenseMap<Value, RecordAddress> recordAddresses;
};

class CallbackConversion {
public:
    explicit CallbackConversion(func::FuncOp function) : function(function) {}

    LogicalResult run() {
        SmallVector<CpuAdCallbackOp> callbacks;
        function.walk([&](CpuAdCallbackOp operation) { callbacks.push_back(operation); });
        for (CpuAdCallbackOp callback : callbacks)
            if (failed(lower(callback)))
                return failure();
        SmallVector<CpuAdRequiredBytesOp> requiredBytesReads;
        function.walk([&](CpuAdRequiredBytesOp operation) { requiredBytesReads.push_back(operation); });
        for (CpuAdRequiredBytesOp read : requiredBytesReads)
            lower(read);
        return success();
    }

private:
    LLVM::LLVMStructType allocatorType() {
        MLIRContext *context = function.getContext();
        Type i64 = IntegerType::get(context, 64);
        Type i32 = IntegerType::get(context, 32);
        Type ptr = LLVM::LLVMPointerType::get(context);
        return LLVM::LLVMStructType::getLiteral(
            context, {i64, i32, i32, ptr, i64, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr});
    }

    Value asI64(OpBuilder &builder, Location location, Value value) {
        if (value.getType().isIndex())
            return arith::IndexCastOp::create(builder, location, builder.getI64Type(), value);
        return value;
    }

    Value stackSlot(Location location, Type elementType) {
        OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.getBody().front());
        Value one = LLVM::ConstantOp::create(entryBuilder, location, entryBuilder.getI64Type(),
                                             entryBuilder.getI64IntegerAttr(1));
        return LLVM::AllocaOp::create(entryBuilder, location, LLVM::LLVMPointerType::get(function.getContext()),
                                      elementType, one);
    }

    Value zeroInitializedSlot(OpBuilder &builder, Location location, Type elementType) {
        Value slot = stackSlot(location, elementType);
        Value zero = LLVM::ConstantOp::create(builder, location, elementType, builder.getIntegerAttr(elementType, 0));
        LLVM::StoreOp::create(builder, location, zero, slot);
        return slot;
    }

    void lower(CpuAdRequiredBytesOp operation) {
        OpBuilder builder(operation);
        Type ptr = LLVM::LLVMPointerType::get(function.getContext());
        Value descriptorBits = asI64(builder, operation.getLoc(), operation.getDescriptor());
        Value descriptor = LLVM::IntToPtrOp::create(builder, operation.getLoc(), ptr, descriptorBits);
        Value address =
            LLVM::GEPOp::create(builder, operation.getLoc(), ptr, allocatorType(), descriptor,
                                ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(AllocatorField::RequiredBytes)});
        Value raw = LLVM::LoadOp::create(builder, operation.getLoc(), builder.getI64Type(), address);
        Value required =
            arith::IndexCastOp::create(builder, operation.getLoc(), builder.getIndexType(), raw).getResult();
        operation.getRequiredBytes().replaceAllUsesWith(required);
        operation.erase();
    }

    LogicalResult lower(CpuAdCallbackOp operation) {
        OpBuilder builder(operation);
        const int64_t callbackValue = operation.getCallback();
        if (callbackValue < static_cast<int64_t>(SemanticCallback::Reset) ||
            callbackValue > static_cast<int64_t>(SemanticCallback::ReadExitKind))
            return operation.emitError("references an unknown semantic tape callback");
        const auto callback = static_cast<SemanticCallback>(callbackValue);
        SmallVector<Value> arguments(operation.getArguments().begin(), operation.getArguments().end());
        Value outputSlot;
        Type outputType;
        unsigned expectedArguments = 0;
        switch (callback) {
        case SemanticCallback::Reset:
        case SemanticCallback::Seal:
            expectedArguments = 0;
            break;
        case SemanticCallback::BeginRegion:
            expectedArguments = 1;
            outputType = builder.getI64Type();
            break;
        case SemanticCallback::ReserveRecord:
            expectedArguments = 4;
            outputType = builder.getI64Type();
            break;
        case SemanticCallback::WriteLeaf:
            expectedArguments = 4;
            break;
        case SemanticCallback::SetChild:
            expectedArguments = 3;
            break;
        case SemanticCallback::EndRegion:
            expectedArguments = 3;
            break;
        case SemanticCallback::ReadLeaf:
            expectedArguments = 4;
            outputType = builder.getI64Type();
            break;
        case SemanticCallback::ReadChild:
            expectedArguments = 3;
            outputType = builder.getI64Type();
            break;
        case SemanticCallback::ReadExecutedCount:
            expectedArguments = 1;
            outputType = builder.getI64Type();
            break;
        case SemanticCallback::ReadExitKind:
            expectedArguments = 1;
            outputType = builder.getI32Type();
            break;
        }
        if (arguments.size() != expectedArguments)
            return operation.emitError("has the wrong semantic callback argument count");
        const auto isWideInteger = [](Value value) {
            return value.getType().isInteger(64) || value.getType().isIndex();
        };
        bool validArgumentTypes = llvm::all_of(arguments, isWideInteger);
        if (callback == SemanticCallback::EndRegion)
            validArgumentTypes = arguments.size() == 3 && isWideInteger(arguments[0]) && isWideInteger(arguments[1]) &&
                                 arguments[2].getType().isInteger(32);
        if (!validArgumentTypes)
            return operation.emitError("has the wrong semantic callback argument type");
        if (callback == SemanticCallback::Seal) {
            if (operation.getNumResults() != 1 || !operation.getResult(0).getType().isInteger(32))
                return operation.emitError("seal callback must produce one i32 status");
        } else if (outputType) {
            if (operation.getNumResults() != 1 || operation.getResult(0).getType() != outputType)
                return operation.emitError("semantic callback has the wrong result type");
        } else if (operation.getNumResults() != 0) {
            return operation.emitError("write-only semantic callback unexpectedly produces a result");
        }

        if (callback == SemanticCallback::WriteLeaf) {
            Value rawSlot = stackSlot(operation.getLoc(), builder.getI64Type());
            LLVM::StoreOp::create(builder, operation.getLoc(), asI64(builder, operation.getLoc(), arguments[2]),
                                  rawSlot);
            arguments[2] = rawSlot;
        } else if (outputType) {
            outputSlot = zeroInitializedSlot(builder, operation.getLoc(), outputType);
            if (callback == SemanticCallback::ReadLeaf)
                arguments.insert(arguments.end() - 1, outputSlot);
            else
                arguments.push_back(outputSlot);
        }

        Type ptr = LLVM::LLVMPointerType::get(function.getContext());
        Value descriptorBits = asI64(builder, operation.getLoc(), operation.getDescriptor());
        Value descriptor = LLVM::IntToPtrOp::create(builder, operation.getLoc(), ptr, descriptorBits);
        Value callbackAddress =
            LLVM::GEPOp::create(builder, operation.getLoc(), ptr, allocatorType(), descriptor,
                                ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(physicalField(callback))});
        Value callbackPointer = LLVM::LoadOp::create(builder, operation.getLoc(), ptr, callbackAddress);
        SmallVector<Value> physicalArguments;
        physicalArguments.reserve(arguments.size());
        for (Value argument : arguments)
            physicalArguments.push_back(isa<LLVM::LLVMPointerType>(argument.getType())
                                            ? argument
                                            : asI64(builder, operation.getLoc(), argument));
        SmallVector<Value> operands = {callbackPointer, descriptor};
        operands.append(physicalArguments);
        SmallVector<Type> parameterTypes = {ptr};
        llvm::append_range(parameterTypes,
                           llvm::map_range(physicalArguments, [](Value value) { return value.getType(); }));
        auto callbackType = LLVM::LLVMFunctionType::get(builder.getI32Type(), parameterTypes);
        Value status = LLVM::CallOp::create(builder, operation.getLoc(), callbackType, operands).getResult();

        if (callback == SemanticCallback::Seal) {
            operation.getResult(0).replaceAllUsesWith(status);
        } else if (outputType) {
            Value result = LLVM::LoadOp::create(builder, operation.getLoc(), outputType, outputSlot);
            operation.getResult(0).replaceAllUsesWith(result);
        }
        operation.erase();
        return success();
    }

    func::FuncOp function;
};

struct VernonPrepareCPUAutodiffSignaturesPass final
    : PassWrapper<VernonPrepareCPUAutodiffSignaturesPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonPrepareCPUAutodiffSignaturesPass)

    StringRef getArgument() const final { return "vernon-prepare-cpu-autodiff-signatures"; }
    StringRef getDescription() const final {
        return "Materialize hidden CPU autodiff builtins before physical lowering";
    }

    void runOnOperation() override {
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            if (function.isDeclaration())
                continue;
            bool hasCapture = false;
            bool hasHandles = false;
            function.walk([&](Operation *operation) {
                hasCapture = hasCapture || isa<AdCaptureOp>(operation);
                for (Type type : operation->getOperandTypes())
                    hasHandles = hasHandles || containsLogicalAutodiffHandle(type);
                for (Type type : operation->getResultTypes())
                    hasHandles = hasHandles || containsLogicalAutodiffHandle(type);
            });
            if (!hasCapture && !hasHandles)
                continue;
            if (hasCapture) {
                bool found = false;
                for (unsigned index = 0; index < function.getNumArguments(); ++index)
                    found = found || hasBuiltin(function, index, kAllocatorBuiltin);
                if (!found && failed(function.insertArgument(
                                  function.getNumArguments(), IndexType::get(function.getContext()),
                                  builtinAttributes(function.getContext(), kAllocatorBuiltin), function.getLoc()))) {
                    function.emitError("cannot append the hidden CPU autodiff allocator argument");
                    signalPassFailure();
                    return;
                }
                continue;
            }
            if (function.getNumArguments() < 2 || !isa<AdTapeType>(function.getArgumentTypes()[0]) ||
                !isa<AdRegionHeaderType>(function.getArgumentTypes()[1])) {
                function.emitError("dynamic CPU autodiff backward has an invalid logical tape signature");
                signalPassFailure();
                return;
            }
            function.setArgAttrs(0, builtinAttributes(function.getContext(), kAllocatorBuiltin));
            function.setArgAttrs(1, builtinAttributes(function.getContext(), kRootRegionBuiltin));
        }
    }
};

struct VernonLowerCPUAutodiffPass final : PassWrapper<VernonLowerCPUAutodiffPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerCPUAutodiffPass)

    StringRef getArgument() const final { return "vernon-lower-cpu-autodiff"; }
    StringRef getDescription() const final {
        return "Lower logical autodiff tape operations to the hidden CPU allocator ABI";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect, VernonDialect>();
    }

    void runOnOperation() override {
        SmallVector<func::FuncOp> functions;
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
            if (!function.isDeclaration())
                functions.push_back(function);
        for (func::FuncOp function : functions)
            if (failed(FunctionLowering(function).run())) {
                signalPassFailure();
                return;
            }
    }
};

struct VernonCPUAutodiffToLLVMPass final : PassWrapper<VernonCPUAutodiffToLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonCPUAutodiffToLLVMPass)

    StringRef getArgument() const final { return "vernon-cpu-autodiff-to-llvm"; }
    StringRef getDescription() const final { return "Convert typed CPU autodiff callbacks to the frozen host ABI"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect, VernonDialect>();
    }

    void runOnOperation() override {
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
            if (!function.isDeclaration() && failed(CallbackConversion(function).run())) {
                signalPassFailure();
                return;
            }
    }
};

} // namespace

std::unique_ptr<Pass> createVernonPrepareCPUAutodiffSignaturesPass() {
    return std::make_unique<VernonPrepareCPUAutodiffSignaturesPass>();
}

std::unique_ptr<Pass> createVernonLowerCPUAutodiffPass() { return std::make_unique<VernonLowerCPUAutodiffPass>(); }

std::unique_ptr<Pass> createVernonCPUAutodiffToLLVMPass() { return std::make_unique<VernonCPUAutodiffToLLVMPass>(); }

void registerVernonPrepareCPUAutodiffSignaturesPass() { PassRegistration<VernonPrepareCPUAutodiffSignaturesPass>(); }

void registerVernonLowerCPUAutodiffPass() { PassRegistration<VernonLowerCPUAutodiffPass>(); }

void registerVernonCPUAutodiffToLLVMPass() { PassRegistration<VernonCPUAutodiffToLLVMPass>(); }

} // namespace mlir::vernon
