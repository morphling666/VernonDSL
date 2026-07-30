#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::vernon {
namespace {

Type convertStorageLeaf(Type type, bool useSpirvStorage) {
    if (!useSpirvStorage)
        return MemRefType::get({ShapedType::kDynamic}, type);
    return MemRefType::get({ShapedType::kDynamic}, type, AffineMap(),
                           spirv::StorageClassAttr::get(type.getContext(), spirv::StorageClass::StorageBuffer));
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
            SmallVector<UnrealizedConversionCastOp> viewBridges;
            DenseMap<Value, Value> scalarViewStorages;
            struct AggregateViewExpansion {
                TensorViewType view;
                ValueAbiLayout layout;
                SmallVector<Value> storages;
            };
            DenseMap<Value, AggregateViewExpansion> aggregateViewStorages;
            for (auto [sourceIndex, range] : sourceArgumentRanges) {
                Value first = entry->getArgument(range.first);
                auto aggregateTensor = aggregateTensorArguments.find(sourceIndex);
                if (aggregateTensor != aggregateTensorArguments.end()) {
                    FailureOr<ValueAbiLayout> layout =
                        getValueAbiLayout(aggregateTensor->second.getElementType(), module);
                    if (failed(layout))
                        return signalPassFailure();
                    SmallVector<Value> storages;
                    storages.reserve(range.second);
                    for (unsigned offset = 0; offset < range.second; ++offset)
                        storages.push_back(entry->getArgument(range.first + offset));
                    int64_t elementCount = 1;
                    for (int64_t dimension : aggregateTensor->second.getShape())
                        elementCount *= dimension;
                    SmallVector<Value> elements;
                    for (int64_t index = 0; index < elementCount; ++index) {
                        Value recordIndex = arith::ConstantIndexOp::create(bodyBuilder, source.getLoc(), index);
                        FailureOr<Value> element = loadAggregateRecordFromStorages(
                            aggregateTensor->second.getElementType(), storages, recordIndex, *layout, module,
                            bodyBuilder, source.getLoc(), AggregateStorageBackend::MemRef);
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
                auto view = dyn_cast<TensorViewType>(source.getArgument(sourceIndex).getType());
                if (!view) {
                    mapping.map(source.getArgument(sourceIndex), first);
                    continue;
                }
                FailureOr<ValueAbiLayout> layout = getValueAbiLayout(view.getElementType(), module);
                if (failed(layout))
                    return signalPassFailure();
                SmallVector<Value> storages;
                storages.reserve(range.second);
                for (unsigned offset = 0; offset < range.second; ++offset)
                    storages.push_back(entry->getArgument(range.first + offset));
                auto bridge = UnrealizedConversionCastOp::create(bodyBuilder, source.getLoc(), TypeRange{view},
                                                                 ValueRange(storages));
                mapping.map(source.getArgument(sourceIndex), bridge.getResult(0));
                viewBridges.push_back(bridge);
                if (view.getElementType().isIntOrFloat())
                    scalarViewStorages[bridge.getResult(0)] = storages.front();
                else
                    aggregateViewStorages[bridge.getResult(0)] = {view, *layout, std::move(storages)};
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

            for (Operation &operation : source.front()) {
                if (isa<func::ReturnOp>(operation)) {
                    gpu::ReturnOp::create(bodyBuilder, operation.getLoc());
                    continue;
                }
                bodyBuilder.clone(operation, mapping);
            }

            SmallVector<Operation *> aggregateStorageOperations;
            kernel.walk([&](Operation *operation) {
                if (isa<LoadOp, StoreOp>(operation))
                    aggregateStorageOperations.push_back(operation);
            });
            IRRewriter aggregateRewriter(kernel.getContext());
            for (Operation *operation : aggregateStorageOperations) {
                Value logicalStorage =
                    isa<StoreOp>(operation) ? cast<StoreOp>(operation).getStorage() : operation->getOperand(0);
                auto found = aggregateViewStorages.find(logicalStorage);
                if (found == aggregateViewStorages.end())
                    continue;
                ValueRange indices = isa<StoreOp>(operation) ? cast<StoreOp>(operation).getIndices()
                                                             : cast<LoadOp>(operation).getIndices();
                if (!operation->hasAttr(kPhysicalIndexAttrName) || indices.empty()) {
                    operation->emitError("aggregate device TensorView operation has no materialized physical index");
                    return signalPassFailure();
                }
                aggregateRewriter.setInsertionPoint(operation);
                if (auto load = dyn_cast<LoadOp>(operation)) {
                    FailureOr<Value> value = loadAggregateRecordFromStorages(
                        found->second.view.getElementType(), found->second.storages, indices.front(),
                        found->second.layout, module, aggregateRewriter, load.getLoc(),
                        AggregateStorageBackend::MemRef);
                    if (failed(value)) {
                        load.emitError("cannot reconstruct aggregate device TensorView value");
                        return signalPassFailure();
                    }
                    aggregateRewriter.replaceOp(load, *value);
                } else {
                    auto store = cast<StoreOp>(operation);
                    if (failed(storeAggregateRecordToStorages(found->second.view.getElementType(),
                                                              found->second.storages, indices.front(), store.getValue(),
                                                              found->second.layout, module, aggregateRewriter,
                                                              store.getLoc(), AggregateStorageBackend::MemRef))) {
                        store.emitError("cannot decompose aggregate device TensorView value");
                        return signalPassFailure();
                    }
                    aggregateRewriter.eraseOp(store);
                }
            }

            SmallVector<Operation *> scalarStorageOperations;
            kernel.walk([&](Operation *operation) {
                if (isa<LoadOp, StoreOp, AtomicOp>(operation))
                    scalarStorageOperations.push_back(operation);
            });
            IRRewriter storageRewriter(kernel.getContext());
            for (Operation *operation : scalarStorageOperations) {
                Value logicalStorage =
                    isa<StoreOp>(operation) ? cast<StoreOp>(operation).getStorage() : operation->getOperand(0);
                auto found = scalarViewStorages.find(logicalStorage);
                if (found == scalarViewStorages.end())
                    continue;
                if (!operation->hasAttr(kPhysicalIndexAttrName)) {
                    operation->emitError("device TensorView operation is missing its materialized physical index");
                    return signalPassFailure();
                }
                ValueRange indices = isa<LoadOp>(operation)    ? cast<LoadOp>(operation).getIndices()
                                     : isa<StoreOp>(operation) ? cast<StoreOp>(operation).getIndices()
                                                               : cast<AtomicOp>(operation).getIndices();
                if (indices.empty()) {
                    operation->emitError("device TensorView operation has no physical index");
                    return signalPassFailure();
                }
                storageRewriter.setInsertionPoint(operation);
                if (auto load = dyn_cast<LoadOp>(operation)) {
                    storageRewriter.replaceOpWithNewOp<memref::LoadOp>(load, found->second, indices.front());
                } else if (auto store = dyn_cast<StoreOp>(operation)) {
                    memref::StoreOp::create(storageRewriter, store.getLoc(), store.getValue(), found->second,
                                            indices.front());
                    storageRewriter.eraseOp(store);
                } else {
                    auto atomic = cast<AtomicOp>(operation);
                    arith::AtomicRMWKind kind = atomic.getAtomicKind() == "add"    ? arith::AtomicRMWKind::addi
                                                : atomic.getAtomicKind() == "min"  ? arith::AtomicRMWKind::mins
                                                : atomic.getAtomicKind() == "max"  ? arith::AtomicRMWKind::maxs
                                                : atomic.getAtomicKind() == "umin" ? arith::AtomicRMWKind::minu
                                                : atomic.getAtomicKind() == "umax" ? arith::AtomicRMWKind::maxu
                                                                                   : arith::AtomicRMWKind::assign;
                    storageRewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(atomic, kind, atomic.getValue(),
                                                                            found->second, indices.front());
                }
            }

            if (failed(lowerGpuAggregateWorkgroupStorage(kernel, module)))
                return signalPassFailure();

            for (UnrealizedConversionCastOp bridge : viewBridges) {
                if (!bridge->use_empty()) {
                    bridge.emitError("TensorView conversion bridge still has uses after GPU storage lowering");
                    return signalPassFailure();
                }
                bridge.erase();
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
