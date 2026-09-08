#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUAutodiff.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir::vernon {
namespace {

constexpr uint32_t kInvocationHeaderBytes = 16;
constexpr uint32_t kRegionHeaderBytes = 20;
constexpr unsigned kSegmentVirtualWorkgroup = 0;
constexpr unsigned kSegmentVirtualGlobalBase = 3;
constexpr unsigned kSegmentTapeBase = 6;
constexpr unsigned kSegmentTapeStride = 7;
constexpr unsigned kSegmentTapeCapacity = 8;
constexpr unsigned kSegmentCarrierBase = 9;
constexpr StringLiteral kCotangentCarriersMaterializedAttrName = "vernon.gpu.cotangent_carriers_materialized";
constexpr uint32_t kSegmentWords = 20;

bool containsTapeHandle(Type type) {
    if (isa<AdTapeType, AdRegionHeaderType>(type))
        return true;
    if (auto tuple = dyn_cast<TupleType>(type))
        return llvm::any_of(tuple.getTypes(), containsTapeHandle);
    return false;
}

StringRef scalarDtype(Type type) {
    if (type.isF16())
        return "f16";
    if (type.isF32())
        return "f32";
    if (type.isF64())
        return "f64";
    return {};
}

void setLanguageDtypes(NamedAttrList &attrs, MLIRContext *context, StringRef dtype, bool container) {
    Attribute leaf = StringAttr::get(context, dtype);
    attrs.set("vernon.dtype", StringAttr::get(context, dtype));
    attrs.set("vernon.abi_leaf_dtypes", ArrayAttr::get(context, {leaf}));
    if (container)
        attrs.set("vernon.element_abi_leaf_dtypes", ArrayAttr::get(context, {leaf}));
}

Value createPhysicalLoad(OpBuilder &builder, Location location, Type type, Value storage, Value index) {
    OperationState state(location, PhysicalLoadOp::getOperationName());
    state.addOperands({storage, index});
    state.addTypes(type);
    return builder.create(state)->getResult(0);
}

void createPhysicalStore(OpBuilder &builder, Location location, Value value, Value storage, Value index) {
    OperationState state(location, PhysicalStoreOp::getOperationName());
    state.addOperands({value, storage, index});
    builder.create(state);
}

void createPhysicalAtomicMax(OpBuilder &builder, Location location, Value value, Value storage, Value index) {
    OperationState state(location, PhysicalAtomicOp::getOperationName());
    state.addOperands({storage, index, value});
    state.addTypes(value.getType());
    state.addAttribute("atomic_kind", builder.getStringAttr("umax"));
    state.addAttribute("ordering", builder.getStringAttr("relaxed"));
    builder.create(state);
}

class FunctionLowering {
public:
    explicit FunctionLowering(func::FuncOp function) : function(function), context(function.getContext()) {}

    LogicalResult run() {
        if (failed(materializeGradientResults()))
            return failure();
        auto residualStorage = function->getAttrOfType<StringAttr>("vernon.ad.residual_storage");
        if (!residualStorage)
            return success();
        if (residualStorage.getValue() == "none") {
            if (!hasCotangentParameter())
                return success();
            if (function->hasAttr(kCotangentCarriersMaterializedAttrName))
                return success();
            FailureOr<Value> carrierIndex = appendNoTapeCarrierIndex();
            if (failed(carrierIndex) || failed(materializeCotangentCarriers(*carrierIndex)))
                return failure();
            function->setAttr(kCotangentCarriersMaterializedAttrName, UnitAttr::get(context));
            return success();
        }

        bool hasLogicalTape = llvm::any_of(function.getArgumentTypes(), containsTapeHandle) ||
                              llvm::any_of(function.getResultTypes(), containsTapeHandle);
        function.walk([&](Operation *operation) {
            hasLogicalTape =
                hasLogicalTape ||
                isa<AdCaptureOp, AdBeginInvocationOp, AdBeginRegionOp, AdReserveRecordOp, AdReadLeafOp>(operation);
        });
        if (!hasLogicalTape)
            return success();

        if (!function->hasAttr(kEntryAttrName) ||
            function->getAttrOfType<StringAttr>(kStageAttrName).getValue() != "compute")
            return function.emitError("GPU Tape lowering requires a compute entry");
        if (!function.getBody().hasOneBlock())
            return function.emitError("GPU Tape lowering requires a single entry block");

        forward =
            llvm::any_of(function.getResultTypes(), containsTapeHandle) || !function.getOps<AdCaptureOp>().empty();
        if (failed(appendResources()) || failed(materializeReplayIds()))
            return failure();
        OpBuilder carrierBuilder(laneBase.getDefiningOp());
        Value carrier = arith::AddIOp::create(carrierBuilder, function.getLoc(), carrierBase, physicalLaneLinear);
        Value carrierIndex =
            arith::IndexCastUIOp::create(carrierBuilder, function.getLoc(), carrierBuilder.getIndexType(), carrier);
        if (failed(materializeCotangentCarriers(carrierIndex)) || failed(prepareSignature()))
            return failure();

        SmallVector<Operation *> operations;
        function.walk<WalkOrder::PreOrder>([&](Operation *operation) { operations.push_back(operation); });
        for (Operation *operation : operations) {
            if (!operation->getBlock())
                continue;
            if (auto begin = dyn_cast<AdBeginInvocationOp>(operation))
                lowerBeginInvocation(begin);
            else if (auto begin = dyn_cast<AdBeginRegionOp>(operation)) {
                if (failed(lowerBeginRegion(begin)))
                    return failure();
            } else if (auto reserve = dyn_cast<AdReserveRecordOp>(operation)) {
                if (failed(lowerReserve(reserve)))
                    return failure();
            } else if (auto write = dyn_cast<AdWriteLeafOp>(operation)) {
                if (failed(lowerWrite(write)))
                    return failure();
            } else if (auto end = dyn_cast<AdEndRegionOp>(operation))
                lowerEndRegion(end);
            else if (auto increment = dyn_cast<AdCheckedIncrementOp>(operation))
                lowerCheckedIncrement(increment);
        }

        for (AdCaptureOp capture : llvm::make_early_inc_range(function.getOps<AdCaptureOp>()))
            lowerCapture(capture);

        operations.clear();
        function.walk<WalkOrder::PreOrder>([&](Operation *operation) { operations.push_back(operation); });
        for (Operation *operation : operations) {
            if (!operation->getBlock())
                continue;
            if (auto read = dyn_cast<AdReadRecordOffsetOp>(operation))
                lowerRecordOffsetRead(read);
            else if (auto read = dyn_cast<AdReadExecutedCountOp>(operation))
                lowerExecutedCountRead(read);
            else if (auto read = dyn_cast<AdReadExitKindOp>(operation))
                lowerExitKindRead(read);
            else if (auto read = dyn_cast<AdReadNestedRegionOp>(operation))
                lowerNestedRegionRead(read);
            else if (auto read = dyn_cast<AdReadLeafOp>(operation)) {
                if (failed(lowerLeafRead(read)))
                    return failure();
            }
        }
        for (AdCommitOp commit : llvm::make_early_inc_range(function.getOps<AdCommitOp>()))
            lowerCommit(commit);
        return finalizeSignature();
    }

private:
    bool hasCotangentParameter() {
        for (unsigned index = 0; index < function.getNumArguments(); ++index)
            if (auto role = function.getArgAttrOfType<StringAttr>(index, "vernon.autodiff_role");
                role && role.getValue() == "cotangent")
                return true;
        return false;
    }

    Value i32Constant(OpBuilder &builder, Location location, uint64_t value) const {
        return arith::ConstantIntOp::create(builder, location, static_cast<int32_t>(value), 32);
    }

    Value indexConstant(OpBuilder &builder, Location location, uint64_t value) const {
        return arith::ConstantIndexOp::create(builder, location, static_cast<int64_t>(value));
    }

    Value asI32(OpBuilder &builder, Location location, Value value) const {
        if (value.getType().isIndex())
            return arith::IndexCastUIOp::create(builder, location, builder.getI32Type(), value);
        if (auto integer = dyn_cast<IntegerType>(value.getType())) {
            if (integer.getWidth() < 32)
                return arith::ExtUIOp::create(builder, location, builder.getI32Type(), value);
            if (integer.getWidth() > 32)
                return arith::TruncIOp::create(builder, location, builder.getI32Type(), value);
        }
        return value;
    }

    Value add(OpBuilder &builder, Location location, Value left, uint64_t right) const {
        return arith::AddIOp::create(builder, location, left, i32Constant(builder, location, right));
    }

    Value absoluteAddress(OpBuilder &builder, Location location, Value relative, uint64_t field = 0) const {
        Value address = arith::AddIOp::create(builder, location, laneBase, relative);
        return field ? add(builder, location, address, field) : address;
    }

    Value loadWord(OpBuilder &builder, Location location, Value byteAddress) const {
        Value word = arith::DivUIOp::create(builder, location, byteAddress, i32Constant(builder, location, 4));
        Value index = arith::IndexCastUIOp::create(builder, location, builder.getIndexType(), word);
        return createPhysicalLoad(builder, location, builder.getI32Type(), tape, index);
    }

    Value loadByte(OpBuilder &builder, Location location, Value byteAddress) const {
        Value word = loadWord(builder, location, byteAddress);
        Value remainder = arith::RemUIOp::create(builder, location, byteAddress, i32Constant(builder, location, 4));
        Value shift = arith::MulIOp::create(builder, location, remainder, i32Constant(builder, location, 8));
        return arith::AndIOp::create(builder, location, arith::ShRUIOp::create(builder, location, word, shift),
                                     arith::ConstantIntOp::create(builder, location, 0xff, 32));
    }

    void storeByte(OpBuilder &builder, Location location, Value byteAddress, Value byte) const {
        Value word = loadWord(builder, location, byteAddress);
        Value remainder = arith::RemUIOp::create(builder, location, byteAddress, i32Constant(builder, location, 4));
        Value shift = arith::MulIOp::create(builder, location, remainder, i32Constant(builder, location, 8));
        Value mask =
            arith::ShLIOp::create(builder, location, arith::ConstantIntOp::create(builder, location, 0xff, 32), shift);
        Value cleared = arith::AndIOp::create(
            builder, location, word,
            arith::XOrIOp::create(builder, location, mask, arith::ConstantIntOp::create(builder, location, -1, 32)));
        Value inserted = arith::ShLIOp::create(
            builder, location,
            arith::AndIOp::create(builder, location, byte, arith::ConstantIntOp::create(builder, location, 0xff, 32)),
            shift);
        Value updated = arith::OrIOp::create(builder, location, cleared, inserted);
        Value wordIndexValue =
            arith::DivUIOp::create(builder, location, byteAddress, i32Constant(builder, location, 4));
        Value wordIndex = arith::IndexCastUIOp::create(builder, location, builder.getIndexType(), wordIndexValue);
        createPhysicalStore(builder, location, updated, tape, wordIndex);
    }

    Value loadI32(OpBuilder &builder, Location location, Value byteAddress) const {
        Value result = i32Constant(builder, location, 0);
        for (unsigned byte = 0; byte < 4; ++byte) {
            Value part = loadByte(builder, location, add(builder, location, byteAddress, byte));
            if (byte)
                part = arith::ShLIOp::create(builder, location, part, i32Constant(builder, location, byte * 8));
            result = arith::OrIOp::create(builder, location, result, part);
        }
        return result;
    }

    void storeI32(OpBuilder &builder, Location location, Value byteAddress, Value value) const {
        value = asI32(builder, location, value);
        for (unsigned byte = 0; byte < 4; ++byte) {
            Value shifted =
                byte ? arith::ShRUIOp::create(builder, location, value, i32Constant(builder, location, byte * 8))
                     : value;
            storeByte(builder, location, add(builder, location, byteAddress, byte), shifted);
        }
    }

    Value loadRelativeI32(OpBuilder &builder, Location location, Value relative, uint64_t field = 0) const {
        return loadI32(builder, location, absoluteAddress(builder, location, relative, field));
    }

    void storeRelativeI32(OpBuilder &builder, Location location, Value relative, uint64_t field, Value value) const {
        storeI32(builder, location, absoluteAddress(builder, location, relative, field), value);
    }

    Value loadMetadataI32(OpBuilder &builder, Location location, unsigned field) {
        Value lowIndex =
            arith::AddIOp::create(builder, location, metadataWordBase, i32Constant(builder, location, 2 * field));
        Value highIndex = arith::AddIOp::create(builder, location, lowIndex, i32Constant(builder, location, 1));
        Value low =
            createPhysicalLoad(builder, location, builder.getI32Type(), segment,
                               arith::IndexCastUIOp::create(builder, location, builder.getIndexType(), lowIndex));
        Value high =
            createPhysicalLoad(builder, location, builder.getI32Type(), segment,
                               arith::IndexCastUIOp::create(builder, location, builder.getIndexType(), highIndex));
        Value highIsZero =
            arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq, high, i32Constant(builder, location, 0));
        metadataFitsI32 = arith::AndIOp::create(builder, location, metadataFitsI32, highIsZero);
        return low;
    }

    DictionaryAttr resourceAttrs(StringRef sourceName, StringRef role, int64_t binding) const {
        NamedAttrList attrs;
        attrs.set(kInterfaceAttrName, StringAttr::get(context, "resource"));
        attrs.set(kDescriptorSetAttrName, IntegerAttr::get(IntegerType::get(context, 64), 0));
        attrs.set(kBindingAttrName, IntegerAttr::get(IntegerType::get(context, 64), binding));
        attrs.set("vernon.source_name", StringAttr::get(context, sourceName));
        attrs.set("vernon.autodiff_role", StringAttr::get(context, role));
        setLanguageDtypes(attrs, context, "i32", /*container=*/true);
        return attrs.getDictionary(context);
    }

    LogicalResult materializeGradientResults() {
        if (!function.getNumResults())
            return success();
        ArrayAttr paths = function.getResultAttrOfType<ArrayAttr>(0, "vernon.autodiff_gradient_paths");
        if (!paths)
            return success();
        SmallVector<Type> gradientTypes;
        Type resultType = function.getResultTypes().front();
        if (auto tuple = dyn_cast<TupleType>(resultType))
            gradientTypes.append(tuple.getTypes().begin(), tuple.getTypes().end());
        else
            gradientTypes.push_back(resultType);
        if (paths.size() != gradientTypes.size())
            return function.emitError("GPU autodiff gradient path count does not match the backward result ABI");

        int64_t nextBinding = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto descriptorSet = function.getArgAttrOfType<IntegerAttr>(index, kDescriptorSetAttrName);
            if (descriptorSet && descriptorSet.getInt() != 0)
                continue;
            if (auto binding = function.getArgAttrOfType<IntegerAttr>(index, kBindingAttrName))
                nextBinding = std::max(nextBinding, binding.getInt() + 1);
        }
        const unsigned firstGradient = function.getNumArguments();
        SmallVector<Type> resourceTypes;
        SmallVector<DictionaryAttr> resourceAttributes;
        SmallVector<unsigned> indices(paths.size(), firstGradient);
        SmallVector<Location> locations(paths.size(), function.getLoc());
        for (auto [index, pair] : llvm::enumerate(llvm::zip(paths, gradientTypes))) {
            auto [pathAttribute, gradientType] = pair;
            auto path = dyn_cast<StringAttr>(pathAttribute);
            StringRef dtype = scalarDtype(gradientType);
            if (!path || dtype.empty())
                return function.emitError("GPU autodiff value gradients require named floating-point scalar leaves");
            resourceTypes.push_back(TensorViewType::get(context, gradientType, {}, "read_write", "device"));
            NamedAttrList attributes;
            attributes.set(kInterfaceAttrName, StringAttr::get(context, "resource"));
            attributes.set(kDescriptorSetAttrName, IntegerAttr::get(IntegerType::get(context, 64), 0));
            attributes.set(kBindingAttrName,
                           IntegerAttr::get(IntegerType::get(context, 64), nextBinding + static_cast<int64_t>(index)));
            attributes.set("vernon.source_name", path);
            attributes.set("vernon.autodiff_source", path);
            attributes.set("vernon.autodiff_role", StringAttr::get(context, "gradient"));
            setLanguageDtypes(attributes, context, dtype, /*container=*/true);
            resourceAttributes.push_back(attributes.getDictionary(context));
        }
        if (failed(function.insertArguments(indices, resourceTypes, resourceAttributes, locations)))
            return function.emitError("cannot append reflected GPU gradient resources");

        SmallVector<func::ReturnOp> returns;
        function.walk([&](func::ReturnOp operation) { returns.push_back(operation); });
        for (func::ReturnOp operation : returns) {
            if (operation.getNumOperands() != 1)
                return operation.emitError("GPU autodiff backward return must contain one canonical gradient value");
            SmallVector<Value> gradients;
            Value result = operation.getOperand(0);
            Operation *aggregate = nullptr;
            if (gradientTypes.size() == 1) {
                gradients.push_back(result);
            } else {
                aggregate = result.getDefiningOp<TupleCreateOp>();
                if (!aggregate || aggregate->getNumOperands() != gradientTypes.size())
                    return operation.emitError("GPU autodiff gradient tuple must be materialized at the return");
                gradients.append(aggregate->getOperands().begin(), aggregate->getOperands().end());
            }
            OpBuilder builder(operation);
            for (auto [index, gradient] : llvm::enumerate(gradients)) {
                OperationState state(operation.getLoc(), ReduceSumOp::getOperationName());
                state.addOperands({gradient, function.getArgument(firstGradient + index)});
                state.addAttribute("deterministic", builder.getBoolAttr(false));
                builder.create(state);
            }
            operation->setOperands({});
            if (aggregate && aggregate->use_empty())
                aggregate->erase();
        }
        function.setType(FunctionType::get(context, function.getArgumentTypes(), {}));
        return success();
    }

    LogicalResult appendResources() {
        int64_t nextBinding = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index)
            if (auto binding = function.getArgAttrOfType<IntegerAttr>(index, kBindingAttrName))
                nextBinding = std::max(nextBinding, binding.getInt() + 1);
        Type tapeType = TensorViewType::get(context, IntegerType::get(context, 32), {-1}, "read_write", "device");
        Type segmentType = TensorViewType::get(context, IntegerType::get(context, 32), {-1}, "read", "device");
        Type statusType = TensorViewType::get(context, IntegerType::get(context, 32), {-1}, "read_write", "device");
        Type physicalIdType = RankedTensorType::get({3}, IntegerType::get(context, 32));
        const unsigned first = function.getNumArguments();
        SmallVector<Type> types = {tapeType, segmentType};
        NamedAttrList physicalLocalAttrs;
        physicalLocalAttrs.set(kBuiltinAttrName, StringAttr::get(context, "local_invocation_id"));
        physicalLocalAttrs.set(kInterfaceAttrName, StringAttr::get(context, "input"));
        setLanguageDtypes(physicalLocalAttrs, context, "u32", /*container=*/true);
        NamedAttrList physicalWorkgroupAttrs;
        physicalWorkgroupAttrs.set(kBuiltinAttrName, StringAttr::get(context, "workgroup_id"));
        physicalWorkgroupAttrs.set(kInterfaceAttrName, StringAttr::get(context, "input"));
        setLanguageDtypes(physicalWorkgroupAttrs, context, "u32", /*container=*/true);
        SmallVector<DictionaryAttr> attrs = {resourceAttrs("__vernon_ad_tape", "tape", nextBinding),
                                             resourceAttrs("__vernon_ad_segment", "replay_segment", nextBinding + 1)};
        if (forward) {
            types.push_back(statusType);
            attrs.push_back(resourceAttrs("__vernon_ad_status", "replay_status", nextBinding + 2));
        }
        types.append({physicalIdType, physicalIdType});
        attrs.append({physicalLocalAttrs.getDictionary(context), physicalWorkgroupAttrs.getDictionary(context)});
        SmallVector<unsigned> indices(types.size(), first);
        SmallVector<Location> locations(types.size(), function.getLoc());
        if (failed(function.insertArguments(indices, types, attrs, locations)))
            return function.emitError("cannot append reflected GPU Tape resources");
        tape = function.getArgument(first);
        segment = function.getArgument(first + 1);
        const unsigned builtinBase = first + (forward ? 3 : 2);
        if (forward)
            statusBuffer = function.getArgument(first + 2);
        physicalLocal = function.getArgument(builtinBase);
        physicalWorkgroup = function.getArgument(builtinBase + 1);

        OpBuilder builder = OpBuilder::atBlockBegin(&function.front());
        metadataFitsI32 = arith::ConstantIntOp::create(builder, function.getLoc(), 1, 1);
        physicalGroupLinear = tensor::ExtractOp::create(builder, function.getLoc(), physicalWorkgroup,
                                                        indexConstant(builder, function.getLoc(), 0));
        metadataWordBase = arith::MulIOp::create(builder, function.getLoc(), physicalGroupLinear,
                                                 i32Constant(builder, function.getLoc(), kSegmentWords));
        for (unsigned dimension = 0; dimension < 3; ++dimension)
            virtualWorkgroup[dimension] =
                loadMetadataI32(builder, function.getLoc(), kSegmentVirtualWorkgroup + dimension);
        for (unsigned dimension = 0; dimension < 3; ++dimension)
            virtualGlobalBase[dimension] =
                loadMetadataI32(builder, function.getLoc(), kSegmentVirtualGlobalBase + dimension);
        tapeBase = loadMetadataI32(builder, function.getLoc(), kSegmentTapeBase);
        tapeStride = loadMetadataI32(builder, function.getLoc(), kSegmentTapeStride);
        tapeCapacity = loadMetadataI32(builder, function.getLoc(), kSegmentTapeCapacity);
        carrierBase = loadMetadataI32(builder, function.getLoc(), kSegmentCarrierBase);
        auto localComponent = [&](unsigned dimension) {
            return tensor::ExtractOp::create(builder, function.getLoc(), physicalLocal,
                                             indexConstant(builder, function.getLoc(), dimension))
                .getResult();
        };
        Value localX = localComponent(0);
        Value localY = localComponent(1);
        Value localZ = localComponent(2);
        auto workgroup = function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName);
        if (!workgroup || workgroup.size() != 3)
            return function.emitError("GPU Tape replay requires a three-dimensional workgroup size");
        Value x = asI32(builder, function.getLoc(), localX);
        Value y = asI32(builder, function.getLoc(), localY);
        Value z = asI32(builder, function.getLoc(), localZ);
        Value yz = arith::AddIOp::create(
            builder, function.getLoc(), y,
            arith::MulIOp::create(builder, function.getLoc(), z,
                                  i32Constant(builder, function.getLoc(), workgroup.asArrayRef()[1])));
        physicalLaneLinear = arith::AddIOp::create(
            builder, function.getLoc(), x,
            arith::MulIOp::create(builder, function.getLoc(), yz,
                                  i32Constant(builder, function.getLoc(), workgroup.asArrayRef()[0])));
        Value laneOffset = arith::MulIOp::create(builder, function.getLoc(), physicalLaneLinear, tapeStride);
        Value plannedLaneBase = arith::AddIOp::create(builder, function.getLoc(), tapeBase, laneOffset);
        Value laneEnd = arith::AddIOp::create(builder, function.getLoc(), plannedLaneBase, tapeStride);
        Value linearIsZero = arith::CmpIOp::create(builder, function.getLoc(), arith::CmpIPredicate::eq,
                                                   physicalLaneLinear, i32Constant(builder, function.getLoc(), 0));
        Value checkedDivisor = arith::SelectOp::create(builder, function.getLoc(), linearIsZero,
                                                       i32Constant(builder, function.getLoc(), 1), physicalLaneLinear);
        Value offsetValid = arith::OrIOp::create(
            builder, function.getLoc(), linearIsZero,
            arith::CmpIOp::create(builder, function.getLoc(), arith::CmpIPredicate::eq,
                                  arith::DivUIOp::create(builder, function.getLoc(), laneOffset, checkedDivisor),
                                  tapeStride));
        Value baseValid =
            arith::CmpIOp::create(builder, function.getLoc(), arith::CmpIPredicate::uge, plannedLaneBase, tapeBase);
        Value endValid =
            arith::CmpIOp::create(builder, function.getLoc(), arith::CmpIPredicate::uge, laneEnd, plannedLaneBase);
        Value headerFits = arith::CmpIOp::create(builder, function.getLoc(), arith::CmpIPredicate::uge, tapeStride,
                                                 i32Constant(builder, function.getLoc(), kInvocationHeaderBytes));
        Value withinCapacity =
            arith::CmpIOp::create(builder, function.getLoc(), arith::CmpIPredicate::ule, laneEnd, tapeCapacity);
        laneValid = arith::AndIOp::create(
            builder, function.getLoc(), metadataFitsI32,
            arith::AndIOp::create(
                builder, function.getLoc(), offsetValid,
                arith::AndIOp::create(builder, function.getLoc(), baseValid,
                                      arith::AndIOp::create(builder, function.getLoc(), endValid,
                                                            arith::AndIOp::create(builder, function.getLoc(),
                                                                                  headerFits, withinCapacity)))));
        laneBase = arith::SelectOp::create(builder, function.getLoc(), laneValid, plannedLaneBase,
                                           i32Constant(builder, function.getLoc(), 0));
        return success();
    }

    LogicalResult materializeReplayIds() {
        OpBuilder builder(laneBase.getDefiningOp());
        builder.setInsertionPointAfter(laneBase.getDefiningOp());
        SmallVector<std::pair<BlockArgument, StringRef>> replacements;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto builtin = function.getArgAttrOfType<StringAttr>(index, kBuiltinAttrName);
            if (function.getArgument(index) != physicalWorkgroup && builtin &&
                (builtin.getValue() == "global_invocation_id" || builtin.getValue() == "workgroup_id"))
                replacements.emplace_back(function.getArgument(index), builtin.getValue());
        }
        for (auto [argument, builtin] : replacements) {
            SmallVector<Value> components;
            for (unsigned dimension = 0; dimension < 3; ++dimension) {
                Value value = builtin == "workgroup_id" ? virtualWorkgroup[dimension] : virtualGlobalBase[dimension];
                if (builtin == "global_invocation_id") {
                    Value local = gpuBuiltinComponent(builder, argument, dimension, "local_invocation_id");
                    value = arith::AddIOp::create(builder, function.getLoc(), value,
                                                  asI32(builder, function.getLoc(), local));
                }
                components.push_back(value);
            }
            Value replacement;
            if (auto tensor = dyn_cast<RankedTensorType>(argument.getType())) {
                SmallVector<Value> typed;
                for (Value component : components)
                    typed.push_back(component);
                replacement = tensor::FromElementsOp::create(builder, function.getLoc(), tensor, typed);
            } else {
                replacement = argument.getType().isIndex()
                                  ? arith::IndexCastUIOp::create(builder, function.getLoc(), builder.getIndexType(),
                                                                 components.front())
                                        .getResult()
                                  : components.front();
            }
            argument.replaceAllUsesExcept(replacement, replacement.getDefiningOp());
        }
        return success();
    }

    FailureOr<Value> appendNoTapeCarrierIndex() {
        BlockArgument physicalGlobalId;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto builtin = function.getArgAttrOfType<StringAttr>(index, kBuiltinAttrName);
            auto type = dyn_cast<RankedTensorType>(function.getArgument(index).getType());
            if (builtin && builtin.getValue() == "global_invocation_id" && type && type.getRank() == 1 &&
                type.getShape()[0] == 3 && type.getElementType().isInteger(32)) {
                physicalGlobalId = function.getArgument(index);
                break;
            }
        }
        auto workgroup = function->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName);
        if (!workgroup || workgroup.size() != 3)
            return function.emitError("GPU no-Tape cotangent carriers require a three-dimensional workgroup size");

        int64_t nextBinding = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index)
            if (auto binding = function.getArgAttrOfType<IntegerAttr>(index, kBindingAttrName))
                nextBinding = std::max(nextBinding, binding.getInt() + 1);
        const unsigned argumentIndex = function.getNumArguments();
        Type launchType = TensorViewType::get(context, IntegerType::get(context, 32), {3}, "read", "device");
        NamedAttrList attributes;
        attributes.set(kInterfaceAttrName, StringAttr::get(context, "resource"));
        attributes.set(kDescriptorSetAttrName, IntegerAttr::get(IntegerType::get(context, 64), 0));
        attributes.set(kBindingAttrName, IntegerAttr::get(IntegerType::get(context, 64), nextBinding));
        attributes.set("vernon.source_name", StringAttr::get(context, "__vernon_ad_launch"));
        attributes.set("vernon.autodiff_role", StringAttr::get(context, "launch_metadata"));
        setLanguageDtypes(attributes, context, "i32", /*container=*/true);
        if (failed(function.insertArgument(argumentIndex, launchType, attributes.getDictionary(context),
                                           function.getLoc())))
            return function.emitError("cannot append GPU no-Tape launch metadata");
        if (!physicalGlobalId) {
            NamedAttrList physicalAttributes;
            physicalAttributes.set(kBuiltinAttrName, StringAttr::get(context, "global_invocation_id"));
            physicalAttributes.set(kInterfaceAttrName, StringAttr::get(context, "input"));
            setLanguageDtypes(physicalAttributes, context, "u32", /*container=*/true);
            Type physicalIdType = RankedTensorType::get({3}, IntegerType::get(context, 32));
            if (failed(function.insertArgument(function.getNumArguments(), physicalIdType,
                                               physicalAttributes.getDictionary(context), function.getLoc())))
                return function.emitError("cannot append GPU no-Tape physical invocation ID");
            physicalGlobalId = function.getArgument(function.getNumArguments() - 1);
        }

        OpBuilder builder = OpBuilder::atBlockBegin(&function.front());
        Value launch = function.getArgument(argumentIndex);
        auto loadLaunchDimension = [&](unsigned dimension) {
            OperationState state(function.getLoc(), LoadOp::getOperationName());
            state.addOperands({launch, indexConstant(builder, function.getLoc(), dimension)});
            state.addTypes(builder.getI32Type());
            return builder.create(state)->getResult(0);
        };
        auto globalComponent = [&](unsigned dimension) {
            return tensor::ExtractOp::create(builder, function.getLoc(), physicalGlobalId,
                                             indexConstant(builder, function.getLoc(), dimension))
                .getResult();
        };
        Value extentX = arith::MulIOp::create(builder, function.getLoc(), loadLaunchDimension(0),
                                              i32Constant(builder, function.getLoc(), workgroup.asArrayRef()[0]));
        Value extentY = arith::MulIOp::create(builder, function.getLoc(), loadLaunchDimension(1),
                                              i32Constant(builder, function.getLoc(), workgroup.asArrayRef()[1]));
        Value yz =
            arith::AddIOp::create(builder, function.getLoc(), globalComponent(1),
                                  arith::MulIOp::create(builder, function.getLoc(), globalComponent(2), extentY));
        Value linear = arith::AddIOp::create(builder, function.getLoc(), globalComponent(0),
                                             arith::MulIOp::create(builder, function.getLoc(), yz, extentX));
        return arith::IndexCastUIOp::create(builder, function.getLoc(), builder.getIndexType(), linear).getResult();
    }

    LogicalResult materializeCotangentCarriers(Value carrierIndex) {
        if (forward)
            return success();
        for (unsigned argumentIndex = 0; argumentIndex < function.getNumArguments(); ++argumentIndex) {
            auto role = function.getArgAttrOfType<StringAttr>(argumentIndex, "vernon.autodiff_role");
            auto view = dyn_cast<TensorViewType>(function.getArgument(argumentIndex).getType());
            if (!role || role.getValue() != "cotangent" || !view)
                continue;
            for (OpOperand &use : function.getArgument(argumentIndex).getUses()) {
                auto load = dyn_cast<LoadOp>(use.getOwner());
                if (!load || use.getOperandNumber() != 0)
                    return function.emitError(
                        "GPU cotangent TensorViews must be loaded directly before carrier materialization");
            }
            SmallVector<int64_t> carrierShape = {-1};
            carrierShape.append(view.getShape().begin(), view.getShape().end());
            function.getArgument(argumentIndex)
                .setType(TensorViewType::get(context, view.getElementType(), carrierShape, view.getAccess(),
                                             view.getAddressSpace()));
            function.setArgAttr(argumentIndex, "vernon.autodiff_carrier",
                                StringAttr::get(context, "invocation_linear"));
            SmallVector<LoadOp> loads;
            function.walk([&](LoadOp load) {
                if (load.getStorage() == function.getArgument(argumentIndex))
                    loads.push_back(load);
            });
            for (LoadOp load : loads) {
                OpBuilder loadBuilder(load);
                OperationState state(load.getLoc(), LoadOp::getOperationName());
                state.addOperands(function.getArgument(argumentIndex));
                state.addOperands(carrierIndex);
                state.addOperands(load.getIndices());
                state.addTypes(load.getResult().getType());
                Operation *replacement = loadBuilder.create(state);
                load.getResult().replaceAllUsesWith(replacement->getResult(0));
                load.erase();
            }
        }
        SmallVector<Type> argumentTypes;
        argumentTypes.reserve(function.getNumArguments());
        for (BlockArgument argument : function.getArguments())
            argumentTypes.push_back(argument.getType());
        function.setType(FunctionType::get(context, argumentTypes, function.getResultTypes()));
        return success();
    }

    Value gpuBuiltinComponent(OpBuilder &builder, BlockArgument, unsigned dimension, StringRef) {
        return tensor::ExtractOp::create(builder, function.getLoc(), physicalLocal,
                                         indexConstant(builder, function.getLoc(), dimension));
    }

    LogicalResult prepareSignature() {
        if (!forward) {
            if (function.getNumArguments() < 4 || !isa<AdTapeType>(function.getArgumentTypes()[0]) ||
                !isa<AdRegionHeaderType>(function.getArgumentTypes()[1]))
                return function.emitError("GPU Tape backward profile has an invalid logical signature");
            OpBuilder builder(laneBase.getDefiningOp());
            builder.setInsertionPointAfter(laneBase.getDefiningOp());
            rootRegion = loadRelativeI32(builder, function.getLoc(), i32Constant(builder, function.getLoc(), 0), 4);
            function.getArgument(1).setType(builder.getI32Type());
            function.getArgument(1).replaceAllUsesExcept(rootRegion, rootRegion.getDefiningOp());
        }
        return success();
    }

    void lowerBeginInvocation(AdBeginInvocationOp operation) {
        OpBuilder builder(operation);
        Value zero = i32Constant(builder, operation.getLoc(), 0);
        storeRelativeI32(builder, operation.getLoc(), zero, 0,
                         i32Constant(builder, operation.getLoc(), kInvocationHeaderBytes));
        storeRelativeI32(builder, operation.getLoc(), zero, 4, zero);
        storeRelativeI32(builder, operation.getLoc(), zero, 8,
                         i32Constant(builder, operation.getLoc(), kInvocationHeaderBytes));
        storeRelativeI32(builder, operation.getLoc(), zero, 12,
                         arith::SelectOp::create(builder, operation.getLoc(), laneValid, zero,
                                                 i32Constant(builder, operation.getLoc(), 2)));
        operation->getResult(0).setType(builder.getI32Type());
        operation->getResult(0).replaceAllUsesWith(laneBase);
        operation.erase();
    }

    LogicalResult lowerBeginRegion(AdBeginRegionOp operation) {
        OpBuilder builder(operation);
        Value zero = i32Constant(builder, operation.getLoc(), 0);
        Value cursor = loadRelativeI32(builder, operation.getLoc(), zero, 0);
        Value next = add(builder, operation.getLoc(), cursor, kRegionHeaderBytes);
        Value nextValid = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::uge, next, cursor);
        Value withinStride =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ule, next, tapeStride);
        Value absoluteEnd = arith::AddIOp::create(builder, operation.getLoc(), laneBase, next);
        Value absoluteValid =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::uge, absoluteEnd, laneBase);
        Value withinCapacity =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ule, absoluteEnd, tapeCapacity);
        Value valid = arith::AndIOp::create(
            builder, operation.getLoc(), arith::AndIOp::create(builder, operation.getLoc(), nextValid, withinStride),
            arith::AndIOp::create(builder, operation.getLoc(), absoluteValid, withinCapacity));
        Value region = arith::SelectOp::create(builder, operation.getLoc(), valid, cursor, zero);
        storeRelativeI32(builder, operation.getLoc(), zero, 0,
                         arith::SelectOp::create(builder, operation.getLoc(), valid, next, cursor));
        Value required = loadRelativeI32(builder, operation.getLoc(), zero, 8);
        Value largerRequired =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ugt, next, required);
        storeRelativeI32(builder, operation.getLoc(), zero, 8,
                         arith::SelectOp::create(builder, operation.getLoc(), largerRequired, next, required));
        Value oldStatus = loadRelativeI32(builder, operation.getLoc(), zero, 12);
        storeRelativeI32(builder, operation.getLoc(), zero, 12,
                         arith::SelectOp::create(builder, operation.getLoc(), valid, oldStatus,
                                                 i32Constant(builder, operation.getLoc(), 1)));
        scf::IfOp initialize = scf::IfOp::create(builder, operation.getLoc(), valid, false);
        OpBuilder initializeBuilder(initialize.getThenRegion().front().getTerminator());
        for (uint64_t field : {uint64_t{0}, uint64_t{4}, uint64_t{8}, uint64_t{12}, uint64_t{16}})
            storeRelativeI32(initializeBuilder, operation.getLoc(), region, field, zero);
        if (operation.getParentLink().empty()) {
            Value previous = loadRelativeI32(builder, operation.getLoc(), zero, 4);
            storeRelativeI32(builder, operation.getLoc(), zero, 4,
                             arith::SelectOp::create(builder, operation.getLoc(), valid, region, previous));
        } else {
            Value parentRecord = asI32(builder, operation.getLoc(), operation.getParentLink()[1]);
            uint64_t ordinal = static_cast<uint64_t>(operation.getChildOrdinalAttr().getInt());
            Value childAddress = absoluteAddress(builder, operation.getLoc(), parentRecord);
            childAddress = arith::SubIOp::create(builder, operation.getLoc(), childAddress,
                                                 i32Constant(builder, operation.getLoc(), 12 + 4 * ordinal));
            Value previous = loadI32(builder, operation.getLoc(), childAddress);
            storeI32(builder, operation.getLoc(), childAddress,
                     arith::SelectOp::create(builder, operation.getLoc(), valid, region, previous));
        }
        Value old = operation.getRegion();
        old.setType(builder.getI32Type());
        old.replaceAllUsesWith(region);
        operation.erase();
        return success();
    }

    LogicalResult lowerReserve(AdReserveRecordOp operation) {
        OpBuilder builder(operation);
        Value region = asI32(builder, operation.getLoc(), operation->getOperand(0));
        uint64_t childCount = 0;
        for (Operation *user : operation.getRecordOffset().getUsers())
            if (auto child = dyn_cast<AdBeginRegionOp>(user))
                childCount =
                    std::max<uint64_t>(childCount, static_cast<uint64_t>(child.getChildOrdinalAttr().getInt()) + 1);
        const uint64_t prefix = 8 + 4 * childCount;
        const uint64_t alignment = static_cast<uint64_t>(operation.getRecordAlignmentAttr().getInt());
        const uint64_t recordSize = static_cast<uint64_t>(operation.getRecordSizeAttr().getInt());
        if (alignment > uint64_t{1} << 31 || recordSize > std::numeric_limits<uint32_t>::max() ||
            prefix + alignment - 1 > std::numeric_limits<uint32_t>::max())
            return operation.emitError("GPU Tape record layout exceeds the portable 32-bit arena ABI");
        Value zero = i32Constant(builder, operation.getLoc(), 0);
        Value cursor = loadRelativeI32(builder, operation.getLoc(), zero, 0);
        Value biased = add(builder, operation.getLoc(), cursor, prefix + alignment - 1);
        Value plannedPayload = arith::AndIOp::create(builder, operation.getLoc(), biased,
                                                     i32Constant(builder, operation.getLoc(), ~(alignment - 1)));
        Value plannedNext = add(builder, operation.getLoc(), plannedPayload, recordSize);
        Value biasedValid =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::uge, biased, cursor);
        Value nextValid =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::uge, plannedNext, plannedPayload);
        Value withinStride =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ule, plannedNext, tapeStride);
        Value absoluteEnd = arith::AddIOp::create(builder, operation.getLoc(), laneBase, plannedNext);
        Value absoluteValid =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::uge, absoluteEnd, laneBase);
        Value withinCapacity =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ule, absoluteEnd, tapeCapacity);
        Value regionValid = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ne, region, zero);
        Value valid = arith::AndIOp::create(
            builder, operation.getLoc(), regionValid,
            arith::AndIOp::create(
                builder, operation.getLoc(), withinStride,
                arith::AndIOp::create(
                    builder, operation.getLoc(),
                    arith::AndIOp::create(builder, operation.getLoc(), biasedValid, nextValid),
                    arith::AndIOp::create(builder, operation.getLoc(), absoluteValid, withinCapacity))));
        Value payload = arith::SelectOp::create(builder, operation.getLoc(), valid, plannedPayload, zero);
        Value previous = loadRelativeI32(builder, operation.getLoc(), region, 4);
        scf::IfOp initialize = scf::IfOp::create(builder, operation.getLoc(), valid, false);
        OpBuilder initializeBuilder(initialize.getThenRegion().front().getTerminator());
        Value previousAddress = arith::SubIOp::create(initializeBuilder, operation.getLoc(),
                                                      absoluteAddress(initializeBuilder, operation.getLoc(), payload),
                                                      i32Constant(initializeBuilder, operation.getLoc(), 4));
        storeI32(initializeBuilder, operation.getLoc(), previousAddress, previous);
        Value nextAddress = arith::SubIOp::create(initializeBuilder, operation.getLoc(),
                                                  absoluteAddress(initializeBuilder, operation.getLoc(), payload),
                                                  i32Constant(initializeBuilder, operation.getLoc(), 8));
        storeI32(initializeBuilder, operation.getLoc(), nextAddress, zero);
        for (uint64_t child = 0; child < childCount; ++child) {
            Value childAddress = arith::SubIOp::create(
                initializeBuilder, operation.getLoc(), absoluteAddress(initializeBuilder, operation.getLoc(), payload),
                i32Constant(initializeBuilder, operation.getLoc(), 12 + 4 * child));
            storeI32(initializeBuilder, operation.getLoc(), childAddress, zero);
        }
        Value hasPrevious =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ne, previous, zero);
        Value updateLink = arith::AndIOp::create(builder, operation.getLoc(), hasPrevious, valid);
        scf::IfOp updatePrevious = scf::IfOp::create(builder, operation.getLoc(), updateLink, false);
        OpBuilder previousBuilder(updatePrevious.getThenRegion().front().getTerminator());
        Value previousNext = arith::SubIOp::create(previousBuilder, operation.getLoc(),
                                                   absoluteAddress(previousBuilder, operation.getLoc(), previous),
                                                   i32Constant(previousBuilder, operation.getLoc(), 8));
        storeI32(previousBuilder, operation.getLoc(), previousNext, payload);
        Value count = loadRelativeI32(builder, operation.getLoc(), region, 8);
        Value incremented = add(builder, operation.getLoc(), count, 1);
        Value first = loadRelativeI32(builder, operation.getLoc(), region, 0);
        Value candidateFirst = arith::SelectOp::create(builder, operation.getLoc(), hasPrevious, first, payload);
        Value selectedFirst = arith::SelectOp::create(builder, operation.getLoc(), valid, candidateFirst, first);
        Value selectedLast = arith::SelectOp::create(builder, operation.getLoc(), valid, payload, previous);
        Value selectedCount = arith::SelectOp::create(builder, operation.getLoc(), valid, incremented, count);
        storeRelativeI32(builder, operation.getLoc(), region, 0, selectedFirst);
        storeRelativeI32(builder, operation.getLoc(), region, 4, selectedLast);
        storeRelativeI32(builder, operation.getLoc(), region, 8, selectedCount);
        Value selectedCursor = arith::SelectOp::create(builder, operation.getLoc(), valid, plannedNext, cursor);
        storeRelativeI32(builder, operation.getLoc(), zero, 0, selectedCursor);
        Value required = loadRelativeI32(builder, operation.getLoc(), zero, 8);
        Value largerRequired =
            arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ugt, plannedNext, required);
        storeRelativeI32(builder, operation.getLoc(), zero, 8,
                         arith::SelectOp::create(builder, operation.getLoc(), largerRequired, plannedNext, required));
        Value oldStatus = loadRelativeI32(builder, operation.getLoc(), zero, 12);
        Value status = arith::SelectOp::create(builder, operation.getLoc(), valid, oldStatus,
                                               i32Constant(builder, operation.getLoc(), 1));
        storeRelativeI32(builder, operation.getLoc(), zero, 12, status);
        Value old = operation.getRecordOffset();
        old.replaceAllUsesWith(
            arith::IndexCastUIOp::create(builder, operation.getLoc(), builder.getIndexType(), payload));
        operation.erase();
        return success();
    }

    FailureOr<Value> toRawBits(OpBuilder &builder, Location location, Value value) const {
        Type type = value.getType();
        if (type.isIndex())
            return asI32(builder, location, value);
        if (auto floating = dyn_cast<FloatType>(type)) {
            if (floating.getWidth() > 32)
                return failure();
            Type bitsType = IntegerType::get(context, floating.getWidth());
            value = arith::BitcastOp::create(builder, location, bitsType, value);
        }
        if (auto integer = dyn_cast<IntegerType>(value.getType())) {
            if (integer.getWidth() < 32)
                return arith::ExtUIOp::create(builder, location, builder.getI32Type(), value).getResult();
            if (integer.getWidth() == 32)
                return value;
        }
        return failure();
    }

    FailureOr<Value> fromRawBits(OpBuilder &builder, Location location, Value raw, Type type) const {
        if (type.isIndex())
            return arith::IndexCastUIOp::create(builder, location, type, raw).getResult();
        Type bitsType = type;
        if (auto floating = dyn_cast<FloatType>(type))
            bitsType = IntegerType::get(context, floating.getWidth());
        Value bits = raw;
        if (cast<IntegerType>(bitsType).getWidth() < 32)
            bits = arith::TruncIOp::create(builder, location, bitsType, raw);
        if (isa<FloatType>(type))
            return arith::BitcastOp::create(builder, location, type, bits).getResult();
        return bits;
    }

    LogicalResult lowerWrite(AdWriteLeafOp operation) {
        OpBuilder builder(operation);
        Type type = operation.getValue().getType();
        Value record = asI32(builder, operation.getLoc(), operation.getRecordOffset());
        Value recordValid = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ne, record,
                                                  i32Constant(builder, operation.getLoc(), 0));
        Value address = absoluteAddress(builder, operation.getLoc(), record,
                                        static_cast<uint64_t>(operation.getLeafOffsetAttr().getInt()));
        address = arith::SelectOp::create(builder, operation.getLoc(), recordValid, address, laneBase);
        if (type.isF64()) {
            auto sourceType = VectorType::get({1}, type);
            auto wordsType = VectorType::get({2}, builder.getI32Type());
            Value source =
                vector::FromElementsOp::create(builder, operation.getLoc(), sourceType, operation.getValue());
            Value words = vector::BitCastOp::create(builder, operation.getLoc(), wordsType, source);
            for (int64_t word = 0; word < 2; ++word) {
                Value wordAddress = add(builder, operation.getLoc(), address, 4 * word);
                Value previous = loadI32(builder, operation.getLoc(), wordAddress);
                Value value = vector::ExtractOp::create(builder, operation.getLoc(), words, word);
                storeI32(builder, operation.getLoc(), wordAddress,
                         arith::SelectOp::create(builder, operation.getLoc(), recordValid, value, previous));
            }
            operation.erase();
            return success();
        }
        FailureOr<Value> raw = toRawBits(builder, operation.getLoc(), operation.getValue());
        if (failed(raw))
            return operation.emitError("GPU Tape supports only scalar canonical ABI leaves");
        const unsigned bytes = std::max<unsigned>(operation.getValue().getType().getIntOrFloatBitWidth() / 8, 1);
        for (unsigned byte = 0; byte < bytes; ++byte) {
            Value shifted = byte ? arith::ShRUIOp::create(builder, operation.getLoc(), *raw,
                                                          i32Constant(builder, operation.getLoc(), byte * 8))
                                 : *raw;
            Value byteAddress = add(builder, operation.getLoc(), address, byte);
            Value previous = loadByte(builder, operation.getLoc(), byteAddress);
            storeByte(builder, operation.getLoc(), byteAddress,
                      arith::SelectOp::create(builder, operation.getLoc(), recordValid, shifted, previous));
        }
        operation.erase();
        return success();
    }

    void lowerEndRegion(AdEndRegionOp operation) {
        OpBuilder builder(operation);
        Value region = asI32(builder, operation.getLoc(), operation->getOperand(0));
        Value valid = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::ne, region,
                                            i32Constant(builder, operation.getLoc(), 0));
        scf::IfOp write = scf::IfOp::create(builder, operation.getLoc(), valid, false);
        OpBuilder writeBuilder(write.getThenRegion().front().getTerminator());
        storeRelativeI32(writeBuilder, operation.getLoc(), region, 12,
                         asI32(writeBuilder, operation.getLoc(), operation.getExecutedCount()));
        storeRelativeI32(writeBuilder, operation.getLoc(), region, 16,
                         asI32(writeBuilder, operation.getLoc(), operation.getExitKind()));
        operation.erase();
    }

    void lowerCheckedIncrement(AdCheckedIncrementOp operation) {
        OpBuilder builder(operation);
        Value counter = asI32(builder, operation.getLoc(), operation.getCounter());
        Value maximum = i32Constant(builder, operation.getLoc(), std::numeric_limits<uint32_t>::max());
        Value overflow = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::eq, counter, maximum);
        Value incremented = add(builder, operation.getLoc(), counter, 1);
        Value selected = arith::SelectOp::create(builder, operation.getLoc(), overflow, counter, incremented);
        Value result = arith::IndexCastUIOp::create(builder, operation.getLoc(), builder.getIndexType(), selected);
        operation.getResult().replaceAllUsesWith(result);
        operation.erase();
    }

    void lowerCapture(AdCaptureOp operation) {
        OpBuilder builder(operation);
        auto yield = cast<AdCaptureYieldOp>(operation.getBody().front().getTerminator());
        Value logicalTape = yield->getOperand(0);
        Value logicalRoot = yield->getOperand(1);
        for (Operation &nested : llvm::make_early_inc_range(operation.getBody().front().without_terminator()))
            nested.moveBefore(operation);
        Value zero = i32Constant(builder, operation.getLoc(), 0);
        Value status = loadRelativeI32(builder, operation.getLoc(), zero, 12);
        Value success = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::eq, status, zero);
        Value required32 = loadRelativeI32(builder, operation.getLoc(), zero, 8);
        createPhysicalAtomicMax(builder, operation.getLoc(), required32, statusBuffer,
                                indexConstant(builder, operation.getLoc(), 0));
        createPhysicalAtomicMax(builder, operation.getLoc(), status, statusBuffer,
                                indexConstant(builder, operation.getLoc(), 1));
        Value required = arith::IndexCastUIOp::create(builder, operation.getLoc(), builder.getIndexType(), required32);
        Value overflow = arith::CmpIOp::create(builder, operation.getLoc(), arith::CmpIPredicate::eq, status,
                                               i32Constant(builder, operation.getLoc(), 2));
        operation->getResult(0).setType(builder.getI32Type());
        operation->getResult(1).setType(builder.getI32Type());
        operation->getResult(0).replaceAllUsesWith(logicalTape);
        operation->getResult(1).replaceAllUsesWith(logicalRoot);
        operation.getSuccess().replaceAllUsesWith(success);
        operation.getRequiredBytes().replaceAllUsesWith(required);
        operation.getOverflow().replaceAllUsesWith(overflow);
        operation.erase();
    }

    Value recordAt(OpBuilder &builder, Location location, Value region, Value recordIndex) const {
        Value first = loadRelativeI32(builder, location, region, 0);
        Value upper = recordIndex.getType().isIndex()
                          ? recordIndex
                          : arith::IndexCastUIOp::create(builder, location, builder.getIndexType(), recordIndex);
        SmallVector<Value> initial = {first};
        scf::ForOp loop = scf::ForOp::create(
            builder, location, indexConstant(builder, location, 0), upper, indexConstant(builder, location, 1), initial,
            [&](OpBuilder &body, Location bodyLocation, Value, ValueRange iterArgs) {
                Value nextAddress =
                    arith::SubIOp::create(body, bodyLocation, absoluteAddress(body, bodyLocation, iterArgs.front()),
                                          i32Constant(body, bodyLocation, 8));
                Value next = loadI32(body, bodyLocation, nextAddress);
                scf::YieldOp::create(body, bodyLocation, next);
            });
        return loop.getResult(0);
    }

    void lowerRecordOffsetRead(AdReadRecordOffsetOp operation) {
        OpBuilder builder(operation);
        Value region = asI32(builder, operation.getLoc(), operation->getOperand(0));
        Value last = loadRelativeI32(builder, operation.getLoc(), region, 4);
        Value result = arith::IndexCastUIOp::create(builder, operation.getLoc(), builder.getIndexType(), last);
        operation.getRecordOffset().replaceAllUsesWith(result);
        operation.erase();
    }

    void lowerExecutedCountRead(AdReadExecutedCountOp operation) {
        OpBuilder builder(operation);
        Value value = loadRelativeI32(builder, operation.getLoc(),
                                      asI32(builder, operation.getLoc(), operation->getOperand(0)), 12);
        operation.getExecutedCount().replaceAllUsesWith(
            arith::IndexCastUIOp::create(builder, operation.getLoc(), builder.getIndexType(), value));
        operation.erase();
    }

    void lowerExitKindRead(AdReadExitKindOp operation) {
        OpBuilder builder(operation);
        Value value = loadRelativeI32(builder, operation.getLoc(),
                                      asI32(builder, operation.getLoc(), operation->getOperand(0)), 16);
        operation.getExitKind().replaceAllUsesWith(value);
        operation.erase();
    }

    void lowerNestedRegionRead(AdReadNestedRegionOp operation) {
        OpBuilder builder(operation);
        Value record =
            recordAt(builder, operation.getLoc(), asI32(builder, operation.getLoc(), operation->getOperand(0)),
                     operation.getRecordIndex());
        Value address = arith::SubIOp::create(
            builder, operation.getLoc(), absoluteAddress(builder, operation.getLoc(), record),
            i32Constant(builder, operation.getLoc(),
                        12 + 4 * static_cast<uint64_t>(operation.getChildOrdinalAttr().getInt())));
        Value child = loadI32(builder, operation.getLoc(), address);
        operation->getResult(0).setType(builder.getI32Type());
        operation->getResult(0).replaceAllUsesWith(child);
        operation.erase();
    }

    LogicalResult lowerLeafRead(AdReadLeafOp operation) {
        OpBuilder builder(operation);
        Value record =
            recordAt(builder, operation.getLoc(), asI32(builder, operation.getLoc(), operation->getOperand(0)),
                     operation.getRecordIndex());
        Value address = absoluteAddress(builder, operation.getLoc(), record,
                                        static_cast<uint64_t>(operation.getLeafOffsetAttr().getInt()));
        Type type = operation.getValue().getType();
        if (type.isF64()) {
            auto wordsType = VectorType::get({2}, builder.getI32Type());
            SmallVector<Value> words;
            for (uint64_t word = 0; word < 2; ++word)
                words.push_back(
                    loadI32(builder, operation.getLoc(), add(builder, operation.getLoc(), address, 4 * word)));
            Value packed = vector::FromElementsOp::create(builder, operation.getLoc(), wordsType, words);
            Value values = vector::BitCastOp::create(builder, operation.getLoc(), VectorType::get({1}, type), packed);
            operation.getValue().replaceAllUsesWith(
                vector::ExtractOp::create(builder, operation.getLoc(), values, int64_t{0}));
            operation.erase();
            return success();
        }
        const unsigned bytes = std::max<unsigned>(type.getIntOrFloatBitWidth() / 8, 1);
        Value raw = i32Constant(builder, operation.getLoc(), 0);
        for (unsigned byte = 0; byte < bytes; ++byte) {
            Value part = loadByte(builder, operation.getLoc(), add(builder, operation.getLoc(), address, byte));
            if (byte)
                part = arith::ShLIOp::create(builder, operation.getLoc(), part,
                                             i32Constant(builder, operation.getLoc(), byte * 8));
            raw = arith::OrIOp::create(builder, operation.getLoc(), raw, part);
        }
        FailureOr<Value> value = fromRawBits(builder, operation.getLoc(), raw, type);
        if (failed(value))
            return operation.emitError("GPU Tape read has an unsupported scalar type");
        operation.getValue().replaceAllUsesWith(*value);
        operation.erase();
        return success();
    }

    void lowerCommit(AdCommitOp operation) {
        OpBuilder builder(operation);
        scf::IfOp conditional = scf::IfOp::create(builder, operation.getLoc(), operation.getCaptureSuccess(), false);
        Block *target = &conditional.getThenRegion().front();
        target->getTerminator()->erase();
        for (Operation &nested : operation.getBody().front().without_terminator())
            target->getOperations().push_back(nested.clone());
        OpBuilder targetBuilder = OpBuilder::atBlockEnd(target);
        scf::YieldOp::create(targetBuilder, operation.getLoc());
        operation.erase();
    }

    LogicalResult finalizeSignature() {
        if (forward) {
            SmallVector<func::ReturnOp> returns;
            function.walk([&](func::ReturnOp operation) { returns.push_back(operation); });
            for (func::ReturnOp operation : returns) {
                SmallVector<Operation *> definitions;
                for (Value value : operation.getOperands())
                    if (Operation *definition = value.getDefiningOp())
                        definitions.push_back(definition);
                operation->setOperands({});
                for (Operation *definition : definitions)
                    if (definition->use_empty())
                        definition->erase();
            }
            function.setType(FunctionType::get(context, function.getArgumentTypes(), {}));
        } else {
            llvm::BitVector erase(function.getNumArguments(), false);
            erase.set(0);
            erase.set(1);
            if (failed(function.eraseArguments(erase)))
                return function.emitError("cannot erase logical GPU Tape handle arguments");
        }
        return success();
    }

    func::FuncOp function;
    MLIRContext *context;
    Value tape;
    Value segment;
    Value statusBuffer;
    Value physicalLocal;
    Value physicalWorkgroup;
    Value physicalGroupLinear;
    Value physicalLaneLinear;
    Value metadataWordBase;
    Value metadataFitsI32;
    Value virtualWorkgroup[3];
    Value virtualGlobalBase[3];
    Value tapeBase;
    Value tapeStride;
    Value tapeCapacity;
    Value carrierBase;
    Value laneBase;
    Value laneValid;
    Value rootRegion;
    bool forward{};
};

struct VernonLowerGPUAutodiffPass : public PassWrapper<VernonLowerGPUAutodiffPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerGPUAutodiffPass)

    StringRef getArgument() const final { return "vernon-lower-gpu-autodiff"; }
    StringRef getDescription() const final { return "Lower logical autodiff Tape to bounded GPU replay resources"; }
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect, tensor::TensorDialect,
                        vector::VectorDialect, VernonDialect>();
    }

    void runOnOperation() override {
        for (func::FuncOp function : getOperation().getOps<func::FuncOp>())
            if (failed(FunctionLowering(function).run())) {
                signalPassFailure();
                return;
            }
    }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerGPUAutodiffPass() { return std::make_unique<VernonLowerGPUAutodiffPass>(); }

void registerVernonLowerGPUAutodiffPass() { PassRegistration<VernonLowerGPUAutodiffPass>(); }

} // namespace mlir::vernon
