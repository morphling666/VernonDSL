#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vernon {
namespace {

Type convertStorageLeaf(Type type, bool useSpirvStorage) {
    if (!useSpirvStorage)
        return MemRefType::get({ShapedType::kDynamic}, type);
    return MemRefType::get({ShapedType::kDynamic}, type, AffineMap(),
                           spirv::StorageClassAttr::get(type.getContext(), spirv::StorageClass::StorageBuffer));
}

struct ViewExpansion {
    Type elementType;
    ValueAbiLayout layout;
    SmallVector<Value> storages;
};

FailureOr<Value> buildAggregateValue(Type type, ValueRange leaves, unsigned &cursor, ModuleOp module,
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
            FailureOr<Value> value = buildAggregateValue(field, leaves, cursor, module, builder, location);
            if (failed(value))
                return failure();
            values.push_back(*value);
        }
        if (structure) {
            auto structType = cast<StructType>(type);
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
        FailureOr<Value> value = buildAggregateValue(element, leaves, cursor, module, builder, location);
        if (failed(value))
            return failure();
        values.push_back(*value);
    }
    if (isa<RankedTensorType>(type))
        return tensor::FromElementsOp::create(builder, location, cast<RankedTensorType>(type), values).getResult();
    NamedAttribute name(builder.getStringAttr("name"), builder.getStringAttr("construct"));
    return create(IntrinsicOp::getOperationName(), type, values, name);
}

LogicalResult decomposeAggregateValue(Type type, Value value, SmallVectorImpl<Value> &leaves, ModuleOp module,
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
            if (failed(decomposeAggregateValue(field, fieldValue, leaves, module, builder, location)))
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
        if (failed(decomposeAggregateValue(element, elementValue, leaves, module, builder, location)))
            return failure();
    }
    return success();
}

Value storageLeafIndex(Value recordIndex, uint64_t recordSize, const ValueAbiLeaf &leaf, uint64_t scalarIndex,
                       OpBuilder &builder, Location location) {
    uint64_t leafSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
    Value stride = arith::ConstantIndexOp::create(builder, location, recordSize / leafSize);
    Value result = arith::MulIOp::create(builder, location, recordIndex, stride);
    const uint64_t scalarOffset = leaf.byteOffset / leafSize + scalarIndex;
    if (scalarOffset) {
        Value offset = arith::ConstantIndexOp::create(builder, location, scalarOffset);
        result = arith::AddIOp::create(builder, location, result, offset);
    }
    return result;
}

FailureOr<Value> emitStorageLoad(const ViewExpansion &expansion, Value recordIndex, ModuleOp module, OpBuilder &builder,
                                 Location location) {
    SmallVector<Value> leaves;
    for (auto [leaf, storage] : llvm::zip_equal(expansion.layout.leaves, expansion.storages))
        for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex)
            leaves.push_back(memref::LoadOp::create(
                builder, location, storage,
                storageLeafIndex(recordIndex, expansion.layout.size, leaf, scalarIndex, builder, location)));
    unsigned cursor = 0;
    FailureOr<Value> result = buildAggregateValue(expansion.elementType, leaves, cursor, module, builder, location);
    if (failed(result) || cursor != leaves.size())
        return failure();
    return result;
}

struct VernonToGPUPass : public PassWrapper<VernonToGPUPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonToGPUPass)

    VernonToGPUPass() = default;
    explicit VernonToGPUPass(bool useSpirvStorage) : useSpirvStorage(useSpirvStorage) {}
    VernonToGPUPass(const VernonToGPUPass &other) : PassWrapper(other), useSpirvStorage(other.useSpirvStorage) {}

    StringRef getArgument() const final { return "vernon-to-gpu"; }
    StringRef getDescription() const final { return "Outline Vernon compute entries as MLIR GPU kernels"; }
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect, spirv::SPIRVDialect,
                        tensor::TensorDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SmallVector<func::FuncOp> computeEntries;
        for (func::FuncOp function : module.getOps<func::FuncOp>()) {
            auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
            if (function->hasAttr(kEntryAttrName) && stage && stage.getValue() == "compute")
                computeEntries.push_back(function);
        }
        if (computeEntries.empty())
            return;

        OpBuilder topBuilder(module.getContext());
        topBuilder.setInsertionPointToEnd(module.getBody());
        auto gpuModule = gpu::GPUModuleOp::create(topBuilder, module.getLoc(), "vernon_kernels");
        SmallVector<Attribute> reflectedStructs;
        for (StructDeclOp declaration : module.getOps<StructDeclOp>()) {
            SmallVector<Attribute> fields;
            for (Attribute fieldAttribute : declaration.getFields()) {
                StringRef spelling = cast<StringAttr>(fieldAttribute).getValue();
                const size_t separator = spelling.find(':');
                Type fieldType = separator == StringRef::npos
                                     ? Type{}
                                     : parseType(spelling.drop_front(separator + 1), module.getContext());
                if (!fieldType) {
                    declaration.emitError("cannot preserve struct field types for GPU lowering");
                    return signalPassFailure();
                }
                fields.push_back(TypeAttr::get(fieldType));
            }
            reflectedStructs.push_back(DictionaryAttr::get(
                module.getContext(),
                {topBuilder.getNamedAttr("name", topBuilder.getStringAttr(declaration.getSymName())),
                 topBuilder.getNamedAttr("fields", topBuilder.getArrayAttr(fields))}));
        }
        gpuModule->setAttr("vernon.struct_definitions", topBuilder.getArrayAttr(reflectedStructs));
        if (useSpirvStorage) {
            auto targetTriple = spirv::VerCapExtAttr::get(spirv::Version::V_1_3, {spirv::Capability::Shader},
                                                          llvm::ArrayRef<spirv::Extension>(), module.getContext());
            gpuModule->setAttr(
                spirv::getTargetEnvAttrName(),
                spirv::TargetEnvAttr::get(targetTriple, spirv::getDefaultResourceLimits(module.getContext()),
                                          spirv::ClientAPI::Vulkan, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
                                          spirv::TargetEnvAttr::kUnknownDeviceID));
        }
        OpBuilder moduleBuilder = OpBuilder::atBlockBegin(gpuModule.getBody());

        for (func::FuncOp source : computeEntries) {
            struct InlineTensorArgument {
                unsigned sourceIndex;
                unsigned kernelIndex;
                RankedTensorType type;
            };
            SmallVector<Type> kernelArgumentTypes;
            DenseMap<unsigned, std::pair<unsigned, unsigned>> sourceArgumentRanges;
            SmallVector<InlineTensorArgument> inlineTensorArguments;
            DenseMap<unsigned, TensorType> aggregateTensorArguments;
            SmallVector<std::pair<unsigned, std::pair<unsigned, unsigned>>> resourceBindings;
            for (auto [index, type] : llvm::enumerate(source.getArgumentTypes())) {
                InterfaceAttrs attrs = parseInterfaceAttrs(source.getArgAttrDict(index));
                auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
                if (!attrs.builtin) {
                    if (auto tensor = dyn_cast<TensorType>(type)) {
                        if (!useSpirvStorage) {
                            unsigned kernelIndex = kernelArgumentTypes.size();
                            kernelArgumentTypes.push_back(type);
                            sourceArgumentRanges[index] = {kernelIndex, 1};
                            continue;
                        }
                        FailureOr<ValueAbiLayout> layout = getValueAbiLayout(tensor.getElementType(), module);
                        if (failed(layout) || layout->leaves.empty()) {
                            source.emitError() << "cannot lower aggregate Tensor-by-value argument #" << index;
                            return signalPassFailure();
                        }
                        unsigned firstKernelIndex = kernelArgumentTypes.size();
                        for (const ValueAbiLeaf &leaf : layout->leaves) {
                            unsigned kernelIndex = kernelArgumentTypes.size();
                            kernelArgumentTypes.push_back(convertStorageLeaf(leaf.scalarType, useSpirvStorage));
                            resourceBindings.emplace_back(kernelIndex, std::make_pair(0u, kernelIndex));
                        }
                        sourceArgumentRanges[index] = {firstKernelIndex, layout->leaves.size()};
                        aggregateTensorArguments[index] = tensor;
                        continue;
                    }
                }
                if (!attrs.builtin) {
                    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
                        FailureOr<PhysicalValueAbiLayout> layout =
                            getPhysicalValueAbiLayout(tensor, module, PhysicalAbiProfile::VulkanStd430StorageBuffer);
                        if (failed(layout) || layout->byteStrides.size() != static_cast<size_t>(tensor.getRank())) {
                            source.emitError() << "compute Tensor-by-value argument #" << index
                                               << " must have a positive static shape and scalar element type";
                            return signalPassFailure();
                        }
                        if (!useSpirvStorage) {
                            unsigned kernelIndex = kernelArgumentTypes.size();
                            kernelArgumentTypes.push_back(type);
                            sourceArgumentRanges[index] = {kernelIndex, 1};
                            continue;
                        }
                        unsigned kernelIndex = kernelArgumentTypes.size();
                        kernelArgumentTypes.push_back(convertStorageLeaf(tensor.getElementType(), true));
                        resourceBindings.emplace_back(kernelIndex, std::make_pair(0u, kernelIndex));
                        inlineTensorArguments.push_back({static_cast<unsigned>(index), kernelIndex, tensor});
                        continue;
                    }
                }
                if (kind && kind.getValue() == "resource") {
                    auto view = dyn_cast<TensorViewType>(type);
                    FailureOr<ValueAbiLayout> layout =
                        view ? getValueAbiLayout(view.getElementType(), module) : FailureOr<ValueAbiLayout>(failure());
                    if (!view || failed(layout) || layout->leaves.empty()) {
                        source.emitError() << "cannot lower compute resource argument #" << index << " type " << type;
                        return signalPassFailure();
                    }
                    unsigned firstKernelIndex = kernelArgumentTypes.size();
                    auto descriptorSet = cast<IntegerAttr>(attrs.descriptorSet);
                    for (const ValueAbiLeaf &leaf : layout->leaves) {
                        unsigned kernelIndex = kernelArgumentTypes.size();
                        kernelArgumentTypes.push_back(convertStorageLeaf(leaf.scalarType, useSpirvStorage));
                        resourceBindings.emplace_back(kernelIndex, std::make_pair(descriptorSet.getInt(), kernelIndex));
                    }
                    sourceArgumentRanges[index] = {firstKernelIndex, layout->leaves.size()};
                } else if (!attrs.builtin && (type.isIntOrIndexOrFloat() || isa<VectorType>(type))) {
                    unsigned kernelIndex = kernelArgumentTypes.size();
                    kernelArgumentTypes.push_back(type);
                    sourceArgumentRanges[index] = {kernelIndex, 1};
                }
            }
            auto functionType = moduleBuilder.getFunctionType(kernelArgumentTypes, TypeRange{});
            auto kernel = gpu::GPUFuncOp::create(moduleBuilder, source.getLoc(), source.getSymName(), functionType);
            kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), moduleBuilder.getUnitAttr());
            if (auto workgroup = source->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName)) {
                kernel.setKnownBlockSizeAttr(workgroup);
                kernel->setAttr(spirv::getEntryPointABIAttrName(),
                                spirv::getEntryPointABIAttr(source.getContext(), workgroup.asArrayRef()));
            }
            for (auto [index, binding] : resourceBindings) {
                kernel.setArgAttr(
                    index, spirv::getInterfaceVarABIAttrName(),
                    spirv::getInterfaceVarABIAttr(binding.first, binding.second, std::nullopt, source.getContext()));
            }
            if (useSpirvStorage) {
                for (unsigned index = 0; index < kernel.getNumArguments(); ++index) {
                    if (kernel.getArgAttr(index, spirv::getInterfaceVarABIAttrName()))
                        continue;
                    std::optional<spirv::StorageClass> storageClass;
                    if (kernel.getArgument(index).getType().isIntOrIndexOrFloat())
                        storageClass = spirv::StorageClass::StorageBuffer;
                    kernel.setArgAttr(index, spirv::getInterfaceVarABIAttrName(),
                                      spirv::getInterfaceVarABIAttr(0, index, storageClass, source.getContext()));
                }
            }

            Block *entry = &kernel.front();
            OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
            IRMapping mapping;
            DenseMap<Value, ViewExpansion> sourceViewExpansions;
            DenseMap<Value, ViewExpansion> kernelViewExpansions;
            for (auto [sourceIndex, range] : sourceArgumentRanges) {
                Value first = entry->getArgument(range.first);
                auto aggregateTensor = aggregateTensorArguments.find(sourceIndex);
                if (aggregateTensor != aggregateTensorArguments.end()) {
                    FailureOr<ValueAbiLayout> layout =
                        getValueAbiLayout(aggregateTensor->second.getElementType(), module);
                    if (failed(layout))
                        return signalPassFailure();
                    ViewExpansion expansion{aggregateTensor->second.getElementType(), *layout, {}};
                    expansion.storages.append(entry->getArguments().begin() + range.first,
                                              entry->getArguments().begin() + range.first + range.second);
                    int64_t elementCount = 1;
                    for (int64_t dimension : aggregateTensor->second.getShape())
                        elementCount *= dimension;
                    SmallVector<Value> elements;
                    for (int64_t index = 0; index < elementCount; ++index) {
                        Value recordIndex = arith::ConstantIndexOp::create(bodyBuilder, source.getLoc(), index);
                        FailureOr<Value> element =
                            emitStorageLoad(expansion, recordIndex, module, bodyBuilder, source.getLoc());
                        if (failed(element))
                            return signalPassFailure();
                        elements.push_back(*element);
                    }
                    OperationState state(source.getLoc(), IntrinsicOp::getOperationName());
                    state.addOperands(elements);
                    state.addTypes(aggregateTensor->second);
                    state.addAttribute("name", bodyBuilder.getStringAttr("construct"));
                    mapping.map(source.getArgument(sourceIndex), bodyBuilder.create(state)->getResult(0));
                    continue;
                }
                mapping.map(source.getArgument(sourceIndex), first);
                auto view = dyn_cast<TensorViewType>(source.getArgument(sourceIndex).getType());
                if (!view)
                    continue;
                FailureOr<ValueAbiLayout> layout = getValueAbiLayout(view.getElementType(), module);
                if (failed(layout))
                    return signalPassFailure();
                ViewExpansion expansion{view.getElementType(), *layout, {}};
                expansion.storages.append(entry->getArguments().begin() + range.first,
                                          entry->getArguments().begin() + range.first + range.second);
                sourceViewExpansions[source.getArgument(sourceIndex)] = expansion;
                kernelViewExpansions[first] = std::move(expansion);
            }
            for (const InlineTensorArgument &inlineTensor : inlineTensorArguments) {
                RankedTensorType tensor = inlineTensor.type;
                FailureOr<PhysicalValueAbiLayout> layout =
                    getPhysicalValueAbiLayout(tensor, module, PhysicalAbiProfile::VulkanStd430StorageBuffer);
                if (failed(layout) || layout->byteStrides.size() != static_cast<size_t>(tensor.getRank()))
                    return signalPassFailure();
                const uint64_t elementSize = std::max<uint64_t>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
                SmallVector<Value> elements;
                elements.reserve(tensor.getNumElements());
                for (int64_t linear = 0; linear < tensor.getNumElements(); ++linear) {
                    int64_t remaining = linear;
                    uint64_t byteOffset = 0;
                    for (int64_t dimension = tensor.getRank(); dimension-- > 0;) {
                        const uint64_t index = remaining % tensor.getDimSize(dimension);
                        remaining /= tensor.getDimSize(dimension);
                        byteOffset += index * layout->byteStrides[dimension];
                    }
                    Value elementIndex =
                        arith::ConstantIndexOp::create(bodyBuilder, source.getLoc(), byteOffset / elementSize);
                    elements.push_back(memref::LoadOp::create(
                        bodyBuilder, source.getLoc(), entry->getArgument(inlineTensor.kernelIndex), elementIndex));
                }
                mapping.map(source.getArgument(inlineTensor.sourceIndex),
                            tensor::FromElementsOp::create(bodyBuilder, source.getLoc(), tensor, elements));
            }

            for (auto [index, argument] : llvm::enumerate(source.getArguments())) {
                if (mapping.contains(argument))
                    continue;
                InterfaceAttrs attrs = parseInterfaceAttrs(source.getArgAttrDict(index));
                auto builtin = dyn_cast_if_present<StringAttr>(attrs.builtin);
                if (!builtin || (builtin.getValue() != "global_invocation_id" &&
                                 builtin.getValue() != "local_invocation_id" && builtin.getValue() != "workgroup_id")) {
                    source.emitError() << "compute argument #" << index
                                       << " is neither a resource nor a supported builtin";
                    return signalPassFailure();
                }
                auto createId = [&](gpu::Dimension dimension) -> Value {
                    if (builtin.getValue() == "local_invocation_id")
                        return gpu::ThreadIdOp::create(bodyBuilder, source.getLoc(), dimension);
                    if (builtin.getValue() == "workgroup_id")
                        return gpu::BlockIdOp::create(bodyBuilder, source.getLoc(), dimension);
                    return gpu::GlobalIdOp::create(bodyBuilder, source.getLoc(), dimension);
                };
                if (auto tensorType = dyn_cast<RankedTensorType>(argument.getType())) {
                    if (tensorType.getRank() != 1 || tensorType.getDimSize(0) != 3 ||
                        !tensorType.getElementType().isInteger(32)) {
                        source.emitError() << builtin.getValue() << " Tensor must have type tensor<3xi32>";
                        return signalPassFailure();
                    }
                    SmallVector<Value> components;
                    for (gpu::Dimension dimension : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
                        Value id = createId(dimension);
                        components.push_back(arith::IndexCastUIOp::create(bodyBuilder, source.getLoc(),
                                                                          tensorType.getElementType(), id));
                    }
                    mapping.map(argument,
                                tensor::FromElementsOp::create(bodyBuilder, source.getLoc(), tensorType, components));
                } else {
                    Value globalId = createId(gpu::Dimension::x);
                    if (!argument.getType().isIndex())
                        globalId =
                            arith::IndexCastUIOp::create(bodyBuilder, source.getLoc(), argument.getType(), globalId);
                    mapping.map(argument, globalId);
                }
            }

            auto emitViewLoad = [&](const ViewExpansion &expansion, Value recordIndex, OpBuilder &builder,
                                    Location location) -> FailureOr<Value> {
                SmallVector<Value> leaves;
                for (auto [leaf, storage] : llvm::zip_equal(expansion.layout.leaves, expansion.storages))
                    for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex)
                        leaves.push_back(
                            memref::LoadOp::create(builder, location, storage,
                                                   storageLeafIndex(recordIndex, expansion.layout.size, leaf,
                                                                    scalarIndex, builder, location)));
                unsigned cursor = 0;
                FailureOr<Value> result =
                    buildAggregateValue(expansion.elementType, leaves, cursor, module, builder, location);
                if (failed(result) || cursor != leaves.size())
                    return failure();
                return result;
            };
            auto emitViewStore = [&](const ViewExpansion &expansion, Value recordIndex, Value value, OpBuilder &builder,
                                     Location location) -> LogicalResult {
                SmallVector<Value> leaves;
                if (failed(decomposeAggregateValue(expansion.elementType, value, leaves, module, builder, location)))
                    return failure();
                unsigned cursor = 0;
                for (auto [leaf, storage] : llvm::zip_equal(expansion.layout.leaves, expansion.storages))
                    for (uint64_t scalarIndex = 0; scalarIndex < leaf.scalarCount; ++scalarIndex) {
                        if (cursor >= leaves.size())
                            return failure();
                        memref::StoreOp::create(
                            builder, location, leaves[cursor++], storage,
                            storageLeafIndex(recordIndex, expansion.layout.size, leaf, scalarIndex, builder, location));
                    }
                return success(cursor == leaves.size());
            };

            for (Operation &operation : source.front()) {
                if (isa<func::ReturnOp>(operation)) {
                    gpu::ReturnOp::create(bodyBuilder, operation.getLoc());
                    continue;
                }
                if (auto intrinsic = dyn_cast<IntrinsicOp>(operation)) {
                    if (intrinsic.getName() == "tensor_view_load") {
                        auto expansion = sourceViewExpansions.find(intrinsic.getOperand(0));
                        if (expansion == sourceViewExpansions.end())
                            return signalPassFailure();
                        FailureOr<Value> loaded =
                            emitViewLoad(expansion->second, mapping.lookup(intrinsic.getOperand(1)), bodyBuilder,
                                         intrinsic.getLoc());
                        if (failed(loaded))
                            return signalPassFailure();
                        mapping.map(intrinsic.getResult(), *loaded);
                        continue;
                    }
                    if (intrinsic.getName() == "tensor_view_store") {
                        auto expansion = sourceViewExpansions.find(intrinsic.getOperand(0));
                        if (expansion == sourceViewExpansions.end() ||
                            failed(emitViewStore(expansion->second, mapping.lookup(intrinsic.getOperand(1)),
                                                 mapping.lookup(intrinsic.getOperand(2)), bodyBuilder,
                                                 intrinsic.getLoc())))
                            return signalPassFailure();
                        continue;
                    }
                }
                bodyBuilder.clone(operation, mapping);
            }

            // Intrinsics nested under structured control flow are cloned recursively,
            // so lower them after the complete kernel body has been materialized.
            SmallVector<IntrinsicOp> nestedIntrinsics;
            kernel.walk([&](IntrinsicOp intrinsic) { nestedIntrinsics.push_back(intrinsic); });
            for (IntrinsicOp intrinsic : nestedIntrinsics) {
                OpBuilder builder(intrinsic);
                if (intrinsic.getName() == "tensor_view_load") {
                    auto expansion = kernelViewExpansions.find(intrinsic.getOperand(0));
                    if (expansion == kernelViewExpansions.end())
                        return signalPassFailure();
                    FailureOr<Value> loaded =
                        emitViewLoad(expansion->second, intrinsic.getOperand(1), builder, intrinsic.getLoc());
                    if (failed(loaded))
                        return signalPassFailure();
                    intrinsic.getResult().replaceAllUsesWith(*loaded);
                    intrinsic.erase();
                    continue;
                }
                if (intrinsic.getName() == "tensor_view_store") {
                    auto expansion = kernelViewExpansions.find(intrinsic.getOperand(0));
                    if (expansion == kernelViewExpansions.end() ||
                        failed(emitViewStore(expansion->second, intrinsic.getOperand(1), intrinsic.getOperand(2),
                                             builder, intrinsic.getLoc())))
                        return signalPassFailure();
                    intrinsic.erase();
                }
            }
        }
    }

    bool useSpirvStorage{false};
};

} // namespace

std::unique_ptr<Pass> createVernonToGPUPass(bool useSpirvStorage) {
    return std::make_unique<VernonToGPUPass>(useSpirvStorage);
}

void registerVernonToGPUPass() { PassRegistration<VernonToGPUPass>(); }

} // namespace mlir::vernon
