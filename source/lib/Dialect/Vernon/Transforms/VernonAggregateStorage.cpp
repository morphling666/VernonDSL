#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::vernon {
namespace {

FailureOr<Value> buildAggregateValueVernon(Type type, ValueRange leaves, unsigned &cursor, ModuleOp module,
                                           OpBuilder &builder, Location location) {
    if (type.isIntOrFloat()) {
        if (cursor >= leaves.size())
            return failure();
        return leaves[cursor++];
    }
    auto create = [&](StringRef name, Type resultType, ValueRange operands,
                      ArrayRef<NamedAttribute> attributes = {}) -> Value {
        OperationState state(location, name);
        state.addOperands(operands);
        state.addTypes(resultType);
        state.addAttributes(attributes);
        return builder.create(state)->getResult(0);
    };
    auto buildProduct = [&](TypeRange fields, bool structure) -> FailureOr<Value> {
        SmallVector<Value> values;
        for (Type field : fields) {
            FailureOr<Value> value = buildAggregateValueVernon(field, leaves, cursor, module, builder, location);
            if (failed(value))
                return failure();
            values.push_back(*value);
        }
        if (structure) {
            auto structType = dyn_cast<StructType>(type);
            if (!structType)
                return failure();
            NamedAttribute name(builder.getStringAttr("type_name"), builder.getStringAttr(structType.getName()));
            return create(StructCreateOp::getOperationName(), type, values, name);
        }
        return create(TupleCreateOp::getOperationName(), type, values);
    };
    if (auto tuple = dyn_cast<TupleType>(type))
        return buildProduct(tuple.getTypes(), false);
    if (auto structure = dyn_cast<StructType>(type)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        if (failed(fields))
            return failure();
        return buildProduct(fields->second, true);
    }

    Type element;
    ArrayRef<int64_t> shape;
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto tensor = dyn_cast<TensorType>(type)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else {
        return failure();
    }
    int64_t count = 1;
    for (int64_t dimension : shape)
        count *= dimension;
    SmallVector<Value> values;
    for (int64_t index = 0; index < count; ++index) {
        FailureOr<Value> value = buildAggregateValueVernon(element, leaves, cursor, module, builder, location);
        if (failed(value))
            return failure();
        values.push_back(*value);
    }
    if (isa<RankedTensorType>(type))
        return tensor::FromElementsOp::create(builder, location, cast<RankedTensorType>(type), values).getResult();
    NamedAttribute name(builder.getStringAttr("name"), builder.getStringAttr("construct"));
    return create(IntrinsicOp::getOperationName(), type, values, name);
}

LogicalResult decomposeAggregateValueVernon(Type type, Value value, SmallVectorImpl<Value> &leaves, ModuleOp module,
                                            OpBuilder &builder, Location location) {
    if (type.isIntOrFloat()) {
        leaves.push_back(value);
        return success();
    }
    auto extract = [&](StringRef name, Type resultType, ValueRange operands,
                       ArrayRef<NamedAttribute> attributes) -> Value {
        OperationState state(location, name);
        state.addOperands(operands);
        state.addTypes(resultType);
        state.addAttributes(attributes);
        return builder.create(state)->getResult(0);
    };
    auto decomposeProduct = [&](TypeRange fields, bool structure) -> LogicalResult {
        for (auto [index, field] : llvm::enumerate(fields)) {
            SmallVector<NamedAttribute> attributes;
            attributes.emplace_back(builder.getStringAttr("index"), builder.getI64IntegerAttr(index));
            StringRef operationName = TupleGetOp::getOperationName();
            if (structure) {
                attributes.emplace_back(builder.getStringAttr("field"), builder.getStringAttr(""));
                operationName = StructGetOp::getOperationName();
            }
            Value fieldValue = extract(operationName, field, value, attributes);
            if (failed(decomposeAggregateValueVernon(field, fieldValue, leaves, module, builder, location)))
                return failure();
        }
        return success();
    };
    if (auto tuple = dyn_cast<TupleType>(type))
        return decomposeProduct(tuple.getTypes(), false);
    if (auto structure = dyn_cast<StructType>(type)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        return failed(fields) ? failure() : decomposeProduct(fields->second, true);
    }

    Type element;
    ArrayRef<int64_t> shape;
    bool builtinTensor = false;
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
        builtinTensor = true;
    } else if (auto tensor = dyn_cast<TensorType>(type)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else {
        return failure();
    }
    int64_t count = 1;
    for (int64_t dimension : shape)
        count *= dimension;
    for (int64_t linear = 0; linear < count; ++linear) {
        int64_t remaining = linear;
        SmallVector<Value> indices(shape.size());
        for (int64_t dimension = shape.size() - 1; dimension >= 0; --dimension) {
            indices[dimension] = arith::ConstantIndexOp::create(builder, location, remaining % shape[dimension]);
            remaining /= shape[dimension];
        }
        Value elementValue;
        if (builtinTensor) {
            elementValue = tensor::ExtractOp::create(builder, location, value, indices);
        } else {
            SmallVector<Value> operands{value};
            operands.append(indices);
            elementValue = extract(TensorGetOp::getOperationName(), element, operands, {});
        }
        if (failed(decomposeAggregateValueVernon(element, elementValue, leaves, module, builder, location)))
            return failure();
    }
    return success();
}

FailureOr<Value> buildAggregateValueLlvm(Type sourceType, ValueRange leaves, unsigned &cursor,
                                         const TypeConverter &converter, ModuleOp module, OpBuilder &builder,
                                         Location location) {
    if (sourceType.isIntOrFloat()) {
        if (cursor >= leaves.size())
            return failure();
        return leaves[cursor++];
    }
    auto buildProduct = [&](TypeRange fields, Type targetType) -> FailureOr<Value> {
        auto target = dyn_cast<LLVM::LLVMStructType>(targetType);
        if (!target)
            return failure();
        Value result = LLVM::UndefOp::create(builder, location, target);
        for (auto [index, field] : llvm::enumerate(fields)) {
            FailureOr<Value> value =
                buildAggregateValueLlvm(field, leaves, cursor, converter, module, builder, location);
            if (failed(value))
                return failure();
            result = LLVM::InsertValueOp::create(builder, location, result, *value,
                                                 ArrayRef<int64_t>{static_cast<int64_t>(index)});
        }
        return result;
    };
    if (auto tuple = dyn_cast<TupleType>(sourceType))
        return buildProduct(tuple.getTypes(), converter.convertType(sourceType));
    if (auto structure = dyn_cast<StructType>(sourceType)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        if (failed(fields))
            return failure();
        return buildProduct(fields->second, converter.convertType(sourceType));
    }

    Type element;
    ArrayRef<int64_t> shape;
    if (auto tensor = dyn_cast<RankedTensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto tensor = dyn_cast<TensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else {
        return failure();
    }
    int64_t count = 1;
    for (int64_t dimension : shape)
        count *= dimension;
    SmallVector<Value> elements;
    for (int64_t index = 0; index < count; ++index) {
        FailureOr<Value> value = buildAggregateValueLlvm(element, leaves, cursor, converter, module, builder, location);
        if (failed(value))
            return failure();
        elements.push_back(*value);
    }
    Type targetType = converter.convertType(sourceType);
    if (auto vector = dyn_cast<VectorType>(targetType))
        return vector::FromElementsOp::create(builder, location, vector, elements).getResult();
    auto array = dyn_cast<LLVM::LLVMArrayType>(targetType);
    if (!array)
        return failure();
    Value result = LLVM::UndefOp::create(builder, location, array);
    for (auto [index, value] : llvm::enumerate(elements))
        result = LLVM::InsertValueOp::create(builder, location, result, value,
                                             ArrayRef<int64_t>{static_cast<int64_t>(index)});
    return result;
}

LogicalResult decomposeAggregateValueLlvm(Type sourceType, Value value, SmallVectorImpl<Value> &leaves,
                                          const TypeConverter &converter, ModuleOp module, OpBuilder &builder,
                                          Location location) {
    if (sourceType.isIntOrFloat()) {
        leaves.push_back(value);
        return success();
    }
    auto decomposeProduct = [&](TypeRange fields) -> LogicalResult {
        for (auto [index, field] : llvm::enumerate(fields)) {
            Value extracted =
                LLVM::ExtractValueOp::create(builder, location, value, ArrayRef<int64_t>{static_cast<int64_t>(index)});
            if (failed(decomposeAggregateValueLlvm(field, extracted, leaves, converter, module, builder, location)))
                return failure();
        }
        return success();
    };
    if (auto tuple = dyn_cast<TupleType>(sourceType))
        return decomposeProduct(tuple.getTypes());
    if (auto structure = dyn_cast<StructType>(sourceType)) {
        FailureOr<std::pair<StructDeclOp, SmallVector<Type>>> fields = resolveStructFields(structure, module);
        return failed(fields) ? failure() : decomposeProduct(fields->second);
    }

    Type element;
    ArrayRef<int64_t> shape;
    if (auto tensor = dyn_cast<RankedTensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else if (auto tensor = dyn_cast<TensorType>(sourceType)) {
        element = tensor.getElementType();
        shape = tensor.getShape();
    } else {
        return failure();
    }
    int64_t count = 1;
    for (int64_t dimension : shape)
        count *= dimension;
    for (int64_t index = 0; index < count; ++index) {
        Value extracted;
        if (isa<VectorType>(value.getType()))
            extracted = vector::ExtractOp::create(builder, location, value, index);
        else
            extracted = LLVM::ExtractValueOp::create(builder, location, value, ArrayRef<int64_t>{index});
        if (failed(decomposeAggregateValueLlvm(element, extracted, leaves, converter, module, builder, location)))
            return failure();
    }
    return success();
}

Value loadStorageLeaf(Value storage, Value index, AggregateStorageBackend backend, OpBuilder &builder,
                      Location location, Type leafType) {
    if (backend == AggregateStorageBackend::MemRef || isa<MemRefType>(storage.getType()))
        return memref::LoadOp::create(builder, location, storage, index);
    OperationState state(location, LoadOp::getOperationName());
    state.addOperands({storage, index});
    state.addTypes(leafType);
    return builder.create(state)->getResult(0);
}

void storeStorageLeaf(Value value, Value storage, Value index, AggregateStorageBackend backend, OpBuilder &builder,
                      Location location) {
    if (backend == AggregateStorageBackend::MemRef || isa<MemRefType>(storage.getType())) {
        memref::StoreOp::create(builder, location, value, storage, index);
        return;
    }
    OperationState state(location, StoreOp::getOperationName());
    state.addOperands({value, storage, index});
    builder.create(state);
}

Value workgroupCompactLeafIndex(Value recordIndex, const ValueAbiLeaf &leaf, uint64_t scalarIndex, OpBuilder &builder,
                                Location location) {
    Value result = recordIndex;
    if (leaf.scalarCount != 1) {
        Value stride = arith::ConstantIndexOp::create(builder, location, leaf.scalarCount);
        result = arith::MulIOp::create(builder, location, recordIndex, stride);
    }
    if (scalarIndex) {
        Value offset = arith::ConstantIndexOp::create(builder, location, scalarIndex);
        result = arith::AddIOp::create(builder, location, result, offset);
    }
    return result;
}

Value aggregateLeafIndex(Value recordIndex, const ValueAbiLayout &layout, const ValueAbiLeaf &leaf,
                         uint64_t scalarIndex, bool compact, OpBuilder &builder, Location location) {
    if (compact)
        return workgroupCompactLeafIndex(recordIndex, leaf, scalarIndex, builder, location);
    const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
    Value stride = arith::ConstantIndexOp::create(builder, location, layout.size / scalarSize);
    Value result = arith::MulIOp::create(builder, location, recordIndex, stride);
    const uint64_t scalarOffset = leaf.byteOffset / scalarSize + scalarIndex;
    if (scalarOffset) {
        Value offset = arith::ConstantIndexOp::create(builder, location, scalarOffset);
        result = arith::AddIOp::create(builder, location, result, offset);
    }
    return result;
}

struct AggregateViewPattern final : ConversionPattern {
    AggregateViewPattern(TypeConverter &converter, MLIRContext *context, ModuleOp module, StringRef operationName)
        : ConversionPattern(converter, operationName, 2, context), module(module) {}

    LogicalResult matchAndRewrite(Operation *operation, ArrayRef<ValueRange> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto load = dyn_cast<LoadOp>(operation);
        auto store = dyn_cast<StoreOp>(operation);
        Value sourceStorage = load ? load.getStorage() : store.getStorage();
        auto view = dyn_cast<TensorViewType>(sourceStorage.getType());
        if (!view)
            return failure();
        if (view.getElementType().isIntOrFloat())
            return failure();
        FailureOr<ValueAbiLayout> layout = getValueAbiLayout(view.getElementType(), module);
        if (failed(layout) || layout->leaves.empty())
            return operation->emitError("cannot resolve aggregate TensorView storage layout");
        const unsigned storagePosition = load ? 0 : 1;
        const unsigned indicesPosition = storagePosition + 1;
        if (operands.size() != indicesPosition + view.getShape().size() ||
            operands[storagePosition].size() != layout->leaves.size())
            return operation->emitError("aggregate TensorView conversion received an invalid operand mapping");
        ValueRange storageOperands = operands[storagePosition];
        SmallVector<Value> indices;
        for (unsigned position = 0; position < view.getShape().size(); ++position) {
            if (operands[indicesPosition + position].size() != 1)
                return operation->emitError("aggregate TensorView index conversion is invalid");
            indices.push_back(operands[indicesPosition + position].front());
        }
        if (!operation->hasAttr(kPhysicalIndexAttrName) || indices.empty())
            return operation->emitError("aggregate TensorView is missing its materialized physical index");
        Value recordIndex = indices.front();
        Location loc = operation->getLoc();
        if (load) {
            SmallVector<Value> leaves;
            for (auto [leaf, storage] : llvm::zip_equal(layout->leaves, storageOperands))
                for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex)
                    leaves.push_back(
                        loadStorageLeaf(storage,
                                        aggregateLeafIndex(recordIndex, *layout, leaf, scalarIndex,
                                                           view.getAddressSpace() == "workgroup", rewriter, loc),
                                        AggregateStorageBackend::MemRef, rewriter, loc, leaf.scalarType));
            unsigned cursor = 0;
            FailureOr<Value> value = buildAggregateValueLlvm(view.getElementType(), leaves, cursor, *getTypeConverter(),
                                                             module, rewriter, loc);
            if (failed(value) || cursor != leaves.size())
                return operation->emitError("cannot reconstruct aggregate TensorView value");
            rewriter.replaceOp(operation, *value);
            return success();
        }

        SmallVector<Value> leaves;
        if (operands[0].size() != 1)
            return operation->emitError("cannot decompose aggregate TensorView value");
        if (failed(decomposeAggregateValueLlvm(view.getElementType(), operands[0].front(), leaves, *getTypeConverter(),
                                               module, rewriter, loc))) {
            return operation->emitError("cannot decompose aggregate TensorView value");
        }
        unsigned cursor = 0;
        for (auto [leaf, storage] : llvm::zip_equal(layout->leaves, storageOperands))
            for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex) {
                if (cursor >= leaves.size())
                    return operation->emitError("cannot decompose aggregate TensorView value");
                storeStorageLeaf(leaves[cursor++], storage,
                                 aggregateLeafIndex(recordIndex, *layout, leaf, scalarIndex,
                                                    view.getAddressSpace() == "workgroup", rewriter, loc),
                                 AggregateStorageBackend::MemRef, rewriter, loc);
            }
        if (cursor != leaves.size())
            return operation->emitError("cannot decompose aggregate TensorView value");
        rewriter.eraseOp(operation);
        return success();
    }

    ModuleOp module;
};

struct AggregateWorkgroupAllocPattern final : OpConversionPattern<WorkgroupAllocOp> {
    AggregateWorkgroupAllocPattern(TypeConverter &converter, MLIRContext *context, ModuleOp module)
        : OpConversionPattern(converter, context), module(module) {}

    LogicalResult matchAndRewrite(WorkgroupAllocOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
        TensorViewType view = op.getResult().getType();
        if (view.getElementType().isIntOrFloat())
            return failure();
        FailureOr<WorkgroupPhysicalStoragePlan> plan = getWorkgroupPhysicalStoragePlan(view, module);
        if (failed(plan) || plan->leaves.empty())
            return op.emitError("cannot resolve aggregate workgroup storage layout");
        SmallVector<Value> storages;
        storages.reserve(plan->leaves.size());
        for (const WorkgroupPhysicalLeaf &leaf : plan->leaves) {
            auto memref = MemRefType::get({ShapedType::kDynamic}, leaf.scalarType);
            Value size = arith::ConstantIndexOp::create(rewriter, op.getLoc(), leaf.scalarCount);
            storages.push_back(memref::AllocaOp::create(rewriter, op.getLoc(), memref, ValueRange{size}));
        }
        rewriter.replaceOpWithMultiple(op, {storages});
        return success();
    }

    ModuleOp module;
};

} // namespace

FailureOr<Value> loadAggregateRecordFromStorages(Type elementType, ValueRange storages, Value recordIndex,
                                                 const ValueAbiLayout &layout, ModuleOp module, OpBuilder &builder,
                                                 Location location, AggregateStorageBackend backend) {
    SmallVector<Value> leaves;
    for (auto [leaf, storage] : llvm::zip_equal(layout.leaves, storages))
        for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex)
            leaves.push_back(loadStorageLeaf(storage,
                                             aggregateLeafIndex(recordIndex, layout, leaf, scalarIndex,
                                                                backend == AggregateStorageBackend::WorkgroupTensorView,
                                                                builder, location),
                                             backend, builder, location, leaf.scalarType));
    unsigned cursor = 0;
    FailureOr<Value> result = buildAggregateValueVernon(elementType, leaves, cursor, module, builder, location);
    if (failed(result) || cursor != leaves.size())
        return failure();
    return result;
}

LogicalResult storeAggregateRecordToStorages(Type elementType, ValueRange storages, Value recordIndex, Value value,
                                             const ValueAbiLayout &layout, ModuleOp module, OpBuilder &builder,
                                             Location location, AggregateStorageBackend backend) {
    SmallVector<Value> leaves;
    if (failed(decomposeAggregateValueVernon(elementType, value, leaves, module, builder, location)))
        return failure();
    unsigned cursor = 0;
    for (auto [leaf, storage] : llvm::zip_equal(layout.leaves, storages))
        for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex) {
            if (cursor >= leaves.size())
                return failure();
            storeStorageLeaf(leaves[cursor++], storage,
                             aggregateLeafIndex(recordIndex, layout, leaf, scalarIndex,
                                                backend == AggregateStorageBackend::WorkgroupTensorView, builder,
                                                location),
                             backend, builder, location);
        }
    return success(cursor == leaves.size());
}

void populateCpuAggregateTensorViewPatterns(TypeConverter &converter, RewritePatternSet &patterns, ModuleOp module) {
    MLIRContext *context = patterns.getContext();
    patterns.add<AggregateViewPattern>(converter, context, module, LoadOp::getOperationName());
    patterns.add<AggregateViewPattern>(converter, context, module, StoreOp::getOperationName());
    patterns.add<AggregateWorkgroupAllocPattern>(converter, context, module);
}

LogicalResult lowerGpuAggregateWorkgroupStorage(gpu::GPUFuncOp kernel, ModuleOp module) {
    MLIRContext *context = kernel.getContext();
    SmallVector<WorkgroupAllocOp> aggregateAllocations;
    kernel.walk([&](WorkgroupAllocOp allocation) {
        if (!allocation.getResult().getType().getElementType().isIntOrFloat())
            aggregateAllocations.push_back(allocation);
    });
    IRRewriter storageRewriter(context);
    for (WorkgroupAllocOp allocation : aggregateAllocations) {
        TensorViewType view = allocation.getResult().getType();
        FailureOr<WorkgroupPhysicalStoragePlan> plan = getWorkgroupPhysicalStoragePlan(view, module);
        if (failed(plan) || plan->leaves.empty())
            return allocation.emitError("cannot resolve aggregate workgroup physical storage");
        storageRewriter.setInsertionPoint(allocation);
        SmallVector<Value> storages;
        for (const WorkgroupPhysicalLeaf &leaf : plan->leaves) {
            auto leafView = TensorViewType::get(context, leaf.scalarType, {static_cast<int64_t>(leaf.scalarCount)},
                                                "read_write", "workgroup");
            storages.push_back(WorkgroupAllocOp::create(storageRewriter, allocation.getLoc(), leafView).getResult());
        }
        SmallVector<Operation *> users;
        for (Operation *user : allocation.getResult().getUsers())
            users.push_back(user);
        for (Operation *user : users) {
            storageRewriter.setInsertionPoint(user);
            if (auto load = dyn_cast<LoadOp>(user)) {
                if (!load->hasAttr(kPhysicalIndexAttrName) || load.getIndices().empty())
                    return load.emitError("aggregate workgroup load is missing its materialized physical index");
                FailureOr<Value> value = loadAggregateRecordFromStorages(
                    view.getElementType(), storages, load.getIndices().front(), plan->layout, module, storageRewriter,
                    load.getLoc(), AggregateStorageBackend::WorkgroupTensorView);
                if (failed(value))
                    return load.emitError("cannot reconstruct aggregate workgroup value");
                storageRewriter.replaceOp(load, *value);
                continue;
            }
            if (auto store = dyn_cast<StoreOp>(user)) {
                if (!store->hasAttr(kPhysicalIndexAttrName) || store.getIndices().empty())
                    return store.emitError("aggregate workgroup store is missing its materialized physical index");
                if (failed(storeAggregateRecordToStorages(
                        view.getElementType(), storages, store.getIndices().front(), store.getValue(), plan->layout,
                        module, storageRewriter, store.getLoc(), AggregateStorageBackend::WorkgroupTensorView)))
                    return store.emitError("cannot decompose aggregate workgroup value");
                storageRewriter.eraseOp(store);
                continue;
            }
            return user->emitError("aggregate workgroup storage has an unsupported use");
        }
        storageRewriter.eraseOp(allocation);
    }

    return success();
}

} // namespace mlir::vernon
