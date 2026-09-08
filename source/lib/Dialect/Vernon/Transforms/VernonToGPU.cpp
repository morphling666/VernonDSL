#include "mlir/Dialect/Vernon/Transforms/VernonToGPU.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStorageProjection.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <limits>

namespace mlir::vernon {
namespace {

LogicalResult publishWorkgroupReduction(ReduceSumOp reduce, OpBuilder &builder, Value storage, ValueRange indices,
                                        Value reduced) {
    Value zero = arith::ConstantIndexOp::create(builder, reduce.getLoc(), 0);
    Value leader;
    for (gpu::Dimension dimension : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
        Value thread = gpu::ThreadIdOp::create(builder, reduce.getLoc(), dimension);
        Value component = arith::CmpIOp::create(builder, reduce.getLoc(), arith::CmpIPredicate::eq, thread, zero);
        leader = leader ? arith::AndIOp::create(builder, reduce.getLoc(), leader, component).getResult() : component;
    }
    scf::IfOp publish = scf::IfOp::create(builder, reduce.getLoc(), leader, false);
    OpBuilder publishBuilder(publish.getThenRegion().front().getTerminator());
    OperationState state(reduce.getLoc(), AtomicOp::getOperationName());
    state.addOperands(storage);
    state.addOperands(indices);
    state.addOperands(reduced);
    state.addTypes(reduced.getType());
    state.addAttribute("atomic_kind", publishBuilder.getStringAttr("add"));
    state.addAttribute("ordering", publishBuilder.getStringAttr("relaxed"));
    if (Attribute implementation = reduce->getAttr(kAtomicImplementationAttrName))
        state.addAttribute(kAtomicImplementationAttrName, implementation);
    publishBuilder.create(state);
    return success();
}

LogicalResult lowerWorkgroupReduction(ReduceSumOp reduce, OpBuilder &builder, IRMapping &mapping,
                                      bool useSpirvWorkgroupReduction) {
    auto strategy = reduce->getAttrOfType<StringAttr>(kAccumulationStrategyAttrName);
    if (!strategy || strategy.getValue() != kWorkgroupReductionAccumulationStrategy)
        return failure();
    Value value = mapping.lookupOrNull(reduce.getValue());
    Value storage = mapping.lookupOrNull(reduce.getStorage());
    SmallVector<Value> indices;
    for (Value index : reduce.getIndices()) {
        Value mapped = mapping.lookupOrNull(index);
        if (!mapped)
            return reduce.emitError("workgroup reduction index is not mapped");
        indices.push_back(mapped);
    }
    if (!value || !storage)
        return reduce.emitError("workgroup reduction operands are not mapped");
    if (!useSpirvWorkgroupReduction) {
        auto add = gpu::AllReduceOperationAttr::get(builder.getContext(), gpu::AllReduceOperation::ADD);
        Value reduced = gpu::AllReduceOp::create(builder, reduce.getLoc(), value, add, /*uniform=*/true).getResult();
        return publishWorkgroupReduction(reduce, builder, storage, indices, reduced);
    }

    auto kernel = dyn_cast<gpu::GPUFuncOp>(builder.getInsertionBlock()->getParentOp());
    DenseI32ArrayAttr workgroup = kernel ? kernel.getKnownBlockSizeAttr() : DenseI32ArrayAttr{};
    if (!kernel || !workgroup || workgroup.size() != 3)
        return reduce.emitError("workgroup reduction requires a statically sized GPU kernel");
    uint64_t laneCount = 1;
    for (int32_t extent : workgroup.asArrayRef()) {
        if (extent <= 0)
            return reduce.emitError("workgroup reduction requires positive workgroup dimensions");
        if (laneCount > std::numeric_limits<uint64_t>::max() / static_cast<uint32_t>(extent))
            return reduce.emitError("workgroup reduction lane count overflows");
        laneCount *= static_cast<uint32_t>(extent);
    }
    if (laneCount > std::numeric_limits<unsigned>::max())
        return reduce.emitError("workgroup reduction lane count exceeds the SPIR-V array limit");
    gpu::GPUModuleOp gpuModule = kernel->getParentOfType<gpu::GPUModuleOp>();
    if (!gpuModule)
        return reduce.emitError("workgroup reduction has no enclosing GPU module");
    if (!isa<FloatType>(value.getType()))
        return reduce.emitError("workgroup reduction requires a floating-point value");
    const unsigned elementBytes = value.getType().getIntOrFloatBitWidth() / 8;
    auto arrayType = spirv::ArrayType::get(value.getType(), static_cast<unsigned>(laneCount), elementBytes);
    auto pointerType = spirv::PointerType::get(arrayType, spirv::StorageClass::Workgroup);
    const std::string globalName =
        (Twine("__vernon_workgroup_reduce_") + kernel.getName() + "_f" + Twine(value.getType().getIntOrFloatBitWidth()))
            .str();
    spirv::GlobalVariableOp global;
    for (spirv::GlobalVariableOp candidate : gpuModule.getOps<spirv::GlobalVariableOp>())
        if (candidate.getSymName() == globalName) {
            global = candidate;
            break;
        }
    if (global && global.getType() != pointerType)
        return reduce.emitError("reused workgroup reduction scratch has an incompatible type");
    if (!global) {
        OpBuilder globalBuilder(kernel);
        global = spirv::GlobalVariableOp::create(globalBuilder, reduce.getLoc(), pointerType, globalName,
                                                 FlatSymbolRefAttr(), IntegerAttr(), IntegerAttr(), IntegerAttr(),
                                                 StringAttr(), spirv::LinkageAttributesAttr());
    }
    Value buffer = spirv::AddressOfOp::create(builder, reduce.getLoc(), global);
    auto load = [&](OpBuilder &loadBuilder, Value index) -> Value {
        Value converted = arith::IndexCastUIOp::create(loadBuilder, reduce.getLoc(), loadBuilder.getI32Type(), index);
        Value pointer = spirv::AccessChainOp::create(loadBuilder, reduce.getLoc(), buffer, ValueRange{converted});
        return spirv::LoadOp::create(loadBuilder, reduce.getLoc(), pointer);
    };
    auto store = [&](OpBuilder &storeBuilder, Value stored, Value index) {
        Value converted = arith::IndexCastUIOp::create(storeBuilder, reduce.getLoc(), storeBuilder.getI32Type(), index);
        Value pointer = spirv::AccessChainOp::create(storeBuilder, reduce.getLoc(), buffer, ValueRange{converted});
        spirv::StoreOp::create(storeBuilder, reduce.getLoc(), pointer, stored);
    };
    Value threadX = gpu::ThreadIdOp::create(builder, reduce.getLoc(), gpu::Dimension::x);
    Value threadY = gpu::ThreadIdOp::create(builder, reduce.getLoc(), gpu::Dimension::y);
    Value threadZ = gpu::ThreadIdOp::create(builder, reduce.getLoc(), gpu::Dimension::z);
    Value linear = arith::AddIOp::create(
        builder, reduce.getLoc(), threadX,
        arith::MulIOp::create(
            builder, reduce.getLoc(),
            arith::AddIOp::create(builder, reduce.getLoc(), threadY,
                                  arith::MulIOp::create(builder, reduce.getLoc(), threadZ,
                                                        arith::ConstantIndexOp::create(builder, reduce.getLoc(),
                                                                                       workgroup.asArrayRef()[1]))),
            arith::ConstantIndexOp::create(builder, reduce.getLoc(), workgroup.asArrayRef()[0])));
    store(builder, value, linear);
    gpu::BarrierOp::create(builder, reduce.getLoc());
    uint64_t span = 1;
    while (span < laneCount)
        span *= 2;
    for (uint64_t offset = span / 2; offset; offset /= 2) {
        Value offsetValue = arith::ConstantIndexOp::create(builder, reduce.getLoc(), offset);
        Value peer = arith::AddIOp::create(builder, reduce.getLoc(), linear, offsetValue);
        Value active = arith::AndIOp::create(
            builder, reduce.getLoc(),
            arith::CmpIOp::create(builder, reduce.getLoc(), arith::CmpIPredicate::ult, linear, offsetValue),
            arith::CmpIOp::create(builder, reduce.getLoc(), arith::CmpIPredicate::ult, peer,
                                  arith::ConstantIndexOp::create(builder, reduce.getLoc(), laneCount)));
        scf::IfOp accumulate = scf::IfOp::create(builder, reduce.getLoc(), active, false);
        OpBuilder accumulateBuilder(accumulate.getThenRegion().front().getTerminator());
        Value left = load(accumulateBuilder, linear);
        Value right = load(accumulateBuilder, peer);
        Value sum = arith::AddFOp::create(accumulateBuilder, reduce.getLoc(), left, right);
        store(accumulateBuilder, sum, linear);
        gpu::BarrierOp::create(builder, reduce.getLoc());
    }
    Value reduced = load(builder, arith::ConstantIndexOp::create(builder, reduce.getLoc(), 0));
    return publishWorkgroupReduction(reduce, builder, storage, indices, reduced);
}

Type convertStorageLeaf(Type type, bool useSpirvStorage) {
    if (!useSpirvStorage)
        return MemRefType::get({ShapedType::kDynamic}, type);
    return MemRefType::get({ShapedType::kDynamic}, type, AffineMap(),
                           spirv::StorageClassAttr::get(type.getContext(), spirv::StorageClass::StorageBuffer));
}

struct PhysicalLoadConversion final : OpConversionPattern<PhysicalLoadOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PhysicalLoadOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!llvm::hasSingleElement(adaptor.getIndex()))
            return rewriter.notifyMatchFailure(op, "expected one converted physical index");
        auto module = op->getParentOfType<ModuleOp>();
        FailureOr<ValueAbiLayout> layout = getValueStorageLayout(op.getResult().getType(), module);
        if (failed(layout) || adaptor.getStorage().size() != layout->leaves.size())
            return rewriter.notifyMatchFailure(op, "TensorView storage expansion does not match its value ABI");
        FailureOr<Value> value =
            loadAggregateRecordFromStorages(op.getResult().getType(), adaptor.getStorage(), adaptor.getIndex().front(),
                                            *layout, module, rewriter, op.getLoc(), AggregateStorageBackend::MemRef);
        if (failed(value))
            return failure();
        rewriter.replaceOp(op, *value);
        return success();
    }
};

struct PhysicalStoreConversion final : OpConversionPattern<PhysicalStoreOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PhysicalStoreOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!llvm::hasSingleElement(adaptor.getIndex()) || !llvm::hasSingleElement(adaptor.getValue()))
            return rewriter.notifyMatchFailure(op, "expected scalar converted store index and value");
        auto module = op->getParentOfType<ModuleOp>();
        FailureOr<ValueAbiLayout> layout = getValueStorageLayout(op.getValue().getType(), module);
        if (failed(layout) || adaptor.getStorage().size() != layout->leaves.size())
            return rewriter.notifyMatchFailure(op, "TensorView storage expansion does not match its value ABI");
        if (failed(storeAggregateRecordToStorages(op.getValue().getType(), adaptor.getStorage(),
                                                  adaptor.getIndex().front(), adaptor.getValue().front(), *layout,
                                                  module, rewriter, op.getLoc(), AggregateStorageBackend::MemRef)))
            return failure();
        rewriter.eraseOp(op);
        return success();
    }
};

struct PhysicalAtomicConversion final : OpConversionPattern<PhysicalAtomicOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PhysicalAtomicOp op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!llvm::hasSingleElement(adaptor.getIndex()) || !llvm::hasSingleElement(adaptor.getValue()))
            return rewriter.notifyMatchFailure(op, "expected one converted physical index and value");
        Type valueType = op.getValue().getType();
        if (valueType.isIntOrFloat()) {
            if (!llvm::hasSingleElement(adaptor.getStorage()))
                return rewriter.notifyMatchFailure(op, "scalar atomic TensorView storage must lower to one memref");
            arith::AtomicRMWKind kind =
                op.getAtomicKind() == "add"
                    ? (isa<FloatType>(valueType) ? arith::AtomicRMWKind::addf : arith::AtomicRMWKind::addi)
                : op.getAtomicKind() == "min"  ? arith::AtomicRMWKind::mins
                : op.getAtomicKind() == "max"  ? arith::AtomicRMWKind::maxs
                : op.getAtomicKind() == "umin" ? arith::AtomicRMWKind::minu
                : op.getAtomicKind() == "umax" ? arith::AtomicRMWKind::maxu
                                               : arith::AtomicRMWKind::assign;
            auto replacement = memref::AtomicRMWOp::create(rewriter, op.getLoc(), kind, adaptor.getValue().front(),
                                                           adaptor.getStorage().front(), adaptor.getIndex().front());
            if (Attribute implementation = op->getAttr(kAtomicImplementationAttrName))
                replacement->setAttr(kAtomicImplementationAttrName, implementation);
            rewriter.replaceOp(op, replacement.getResult());
            return success();
        }
        if (op.getAtomicKind() != "add")
            return rewriter.notifyMatchFailure(op, "aggregate TensorView atomic currently supports add");
        auto module = op->getParentOfType<ModuleOp>();
        FailureOr<ValueAbiLayout> layout = getValueStorageLayout(valueType, module);
        if (failed(layout) || adaptor.getStorage().size() != layout->leaves.size())
            return rewriter.notifyMatchFailure(op, "TensorView storage expansion does not match its value ABI");
        FailureOr<Value> old = atomicAddAggregateRecordToStorages(
            valueType, adaptor.getStorage(), adaptor.getIndex().front(), adaptor.getValue().front(), *layout, module,
            rewriter, op.getLoc(), AggregateStorageBackend::MemRef, op->getAttr(kAtomicImplementationAttrName));
        if (failed(old))
            return failure();
        rewriter.replaceOp(op, *old);
        return success();
    }
};

struct VernonToGPUPass : public PassWrapper<VernonToGPUPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonToGPUPass)

    VernonToGPUPass() = default;
    VernonToGPUPass(bool useSpirvStorage, bool useSpirvWorkgroupReduction) : VernonToGPUPass() {
        this->useSpirvStorage = useSpirvStorage;
        this->useSpirvWorkgroupReduction = useSpirvWorkgroupReduction;
    }
    VernonToGPUPass(const VernonToGPUPass &other) : PassWrapper(other) {}

    StringRef getArgument() const final { return "vernon-to-gpu"; }
    StringRef getDescription() const final { return "Outline Vernon compute entries as MLIR GPU kernels"; }
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect, scf::SCFDialect,
                        spirv::SPIRVDialect, tensor::TensorDialect>();
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
                        FailureOr<ValueAbiLayout> layout = getValueStorageLayout(tensor.getElementType(), module);
                        if (failed(layout) || layout->leaves.empty() || layout->size % sizeof(uint32_t) != 0 ||
                            llvm::any_of(layout->leaves, [](const ValueAbiLeaf &leaf) {
                                return leaf.scalarType.getIntOrFloatBitWidth() != 32 ||
                                       leaf.byteOffset % sizeof(uint32_t) != 0;
                            })) {
                            source.emitError() << "cannot lower aggregate Tensor-by-value argument #" << index;
                            return signalPassFailure();
                        }
                        unsigned kernelIndex = kernelArgumentTypes.size();
                        kernelArgumentTypes.push_back(convertStorageLeaf(moduleBuilder.getI32Type(), true));
                        resourceBindings.emplace_back(kernelIndex, std::make_pair(0u, kernelIndex));
                        sourceArgumentRanges[index] = {kernelIndex, 1};
                        aggregateTensorArguments[index] = tensor;
                        continue;
                    }
                }
                if (!attrs.builtin) {
                    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
                        FailureOr<ByteTransportPlan> layout =
                            getByteTransportPlan(tensor, module, PhysicalAbiProfile::VulkanStd430StorageBuffer);
                        if (failed(layout) ||
                            layout->root->byteStrides.size() != static_cast<size_t>(tensor.getRank())) {
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
                    if (isa<TextureType>(type)) {
                        unsigned kernelIndex = kernelArgumentTypes.size();
                        auto descriptorSet = cast<IntegerAttr>(attrs.descriptorSet);
                        kernelArgumentTypes.push_back(type);
                        resourceBindings.emplace_back(kernelIndex, std::make_pair(descriptorSet.getInt(), kernelIndex));
                        sourceArgumentRanges[index] = {kernelIndex, 1};
                        continue;
                    }
                    auto view = dyn_cast<TensorViewType>(type);
                    FailureOr<ValueAbiLayout> layout = view ? getValueStorageLayout(view.getElementType(), module)
                                                            : FailureOr<ValueAbiLayout>(failure());
                    if (!view || failed(layout) || layout->leaves.empty()) {
                        source.emitError() << "cannot lower compute resource argument #" << index << " type " << type;
                        return signalPassFailure();
                    }
                    unsigned kernelIndex = kernelArgumentTypes.size();
                    auto descriptorSet = cast<IntegerAttr>(attrs.descriptorSet);
                    kernelArgumentTypes.push_back(view);
                    resourceBindings.emplace_back(kernelIndex, std::make_pair(descriptorSet.getInt(), kernelIndex));
                    sourceArgumentRanges[index] = {kernelIndex, 1};
                } else if (!attrs.builtin && (type.isIntOrIndexOrFloat() || isa<VectorType>(type))) {
                    unsigned kernelIndex = kernelArgumentTypes.size();
                    kernelArgumentTypes.push_back(type);
                    sourceArgumentRanges[index] = {kernelIndex, 1};
                }
            }
            auto functionType = moduleBuilder.getFunctionType(kernelArgumentTypes, TypeRange{});
            auto kernel = gpu::GPUFuncOp::create(moduleBuilder, source.getLoc(), source.getSymName(), functionType);
            kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), moduleBuilder.getUnitAttr());
            for (const auto &[sourceIndex, range] : sourceArgumentRanges)
                for (StringRef name : {"vernon.autodiff_role", "vernon.autodiff_source", "vernon.autodiff_carrier"})
                    if (Attribute value = source.getArgAttr(sourceIndex, name))
                        for (unsigned offset = 0; offset < range.second; ++offset)
                            kernel.setArgAttr(range.first + offset, name, value);
            for (unsigned sourceIndex = 0; sourceIndex < source.getNumArguments(); ++sourceIndex) {
                auto component = source.getArgAttrOfType<StringAttr>(sourceIndex, kTensorDescriptorComponentAttrName);
                auto range = sourceArgumentRanges.find(sourceIndex);
                if (!component || range == sourceArgumentRanges.end() || range->second.second != 1)
                    continue;
                const unsigned kernelIndex = range->second.first;
                auto owner = source.getArgAttrOfType<IntegerAttr>(sourceIndex, kTensorDescriptorOwnerAttrName);
                auto ownerRange = owner ? sourceArgumentRanges.find(owner.getInt()) : sourceArgumentRanges.end();
                if (!owner || ownerRange == sourceArgumentRanges.end())
                    return signalPassFailure();
                kernel.setArgAttr(kernelIndex, kTensorDescriptorOwnerAttrName,
                                  moduleBuilder.getI64IntegerAttr(ownerRange->second.first));
                kernel.setArgAttr(kernelIndex, kTensorDescriptorComponentAttrName, component);
                if (auto dimension = source.getArgAttr(sourceIndex, kTensorDescriptorDimensionAttrName))
                    kernel.setArgAttr(kernelIndex, kTensorDescriptorDimensionAttrName, dimension);
            }
            if (auto workgroup = source->getAttrOfType<DenseI32ArrayAttr>(kWorkgroupSizeAttrName)) {
                kernel.setKnownBlockSizeAttr(workgroup);
                kernel->setAttr(spirv::getEntryPointABIAttrName(),
                                spirv::getEntryPointABIAttr(source.getContext(), workgroup.asArrayRef()));
            }
            Block *entry = &kernel.front();
            OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
            IRMapping mapping;
            for (auto [sourceIndex, range] : sourceArgumentRanges) {
                Value first = entry->getArgument(range.first);
                auto aggregateTensor = aggregateTensorArguments.find(sourceIndex);
                if (aggregateTensor != aggregateTensorArguments.end()) {
                    FailureOr<ValueAbiLayout> layout =
                        getValueStorageLayout(aggregateTensor->second.getElementType(), module);
                    if (failed(layout) || layout->size % sizeof(uint32_t) != 0)
                        return signalPassFailure();
                    Value recordStride = arith::ConstantIndexOp::create(
                        bodyBuilder, source.getLoc(), static_cast<int64_t>(layout->size / sizeof(uint32_t)));
                    int64_t elementCount = 1;
                    for (int64_t dimension : aggregateTensor->second.getShape())
                        elementCount *= dimension;
                    SmallVector<Value> elements;
                    for (int64_t index = 0; index < elementCount; ++index) {
                        Value recordIndex = arith::ConstantIndexOp::create(bodyBuilder, source.getLoc(), index);
                        Value recordBase =
                            arith::MulIOp::create(bodyBuilder, source.getLoc(), recordIndex, recordStride);
                        SmallVector<Value> scalars;
                        for (const ValueAbiLeaf &leaf : layout->leaves)
                            for (uint64_t scalar = 0; scalar < leaf.scalarCount; ++scalar) {
                                const uint64_t word = leaf.byteOffset / sizeof(uint32_t) + scalar;
                                Value wordOffset = arith::ConstantIndexOp::create(bodyBuilder, source.getLoc(),
                                                                                  static_cast<int64_t>(word));
                                Value wordIndex =
                                    arith::AddIOp::create(bodyBuilder, source.getLoc(), recordBase, wordOffset);
                                Value loaded = memref::LoadOp::create(bodyBuilder, source.getLoc(), first, wordIndex);
                                if (leaf.scalarType.isF32())
                                    loaded =
                                        arith::BitcastOp::create(bodyBuilder, source.getLoc(), leaf.scalarType, loaded);
                                scalars.push_back(loaded);
                            }
                        FailureOr<Value> element = buildAggregateValueFromScalars(
                            aggregateTensor->second.getElementType(), scalars, module, bodyBuilder, source.getLoc());
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
                mapping.map(source.getArgument(sourceIndex), first);
            }
            for (const InlineTensorArgument &inlineTensor : inlineTensorArguments) {
                RankedTensorType tensor = inlineTensor.type;
                FailureOr<ByteTransportPlan> layout =
                    getByteTransportPlan(tensor, module, PhysicalAbiProfile::VulkanStd430StorageBuffer);
                if (failed(layout) || layout->root->byteStrides.size() != static_cast<size_t>(tensor.getRank()))
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
                        byteOffset += index * layout->root->byteStrides[dimension];
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
                if (auto reduce = dyn_cast<ReduceSumOp>(operation);
                    reduce && reduce->hasAttr(kAccumulationStrategyAttrName)) {
                    if (failed(lowerWorkgroupReduction(reduce, bodyBuilder, mapping, useSpirvWorkgroupReduction)))
                        return signalPassFailure();
                    continue;
                }
                bodyBuilder.clone(operation, mapping);
            }
            if (failed(materializeTensorViewProjections(kernel)))
                return signalPassFailure();

            TypeConverter storageConverter;
            storageConverter.addConversion([](Type type) { return type; });
            storageConverter.addConversion([&](TensorViewType view, SmallVectorImpl<Type> &converted) {
                FailureOr<ValueAbiLayout> layout = getValueStorageLayout(view.getElementType(), module);
                if (failed(layout) || layout->leaves.empty())
                    return failure();
                for (const ValueAbiLeaf &leaf : layout->leaves)
                    converted.push_back(convertStorageLeaf(leaf.scalarType, useSpirvStorage));
                return success();
            });

            if (llvm::any_of(kernelArgumentTypes, [](Type type) { return isa<TensorViewType>(type); })) {
                RewritePatternSet storagePatterns(module.getContext());
                storagePatterns.add<PhysicalLoadConversion, PhysicalStoreConversion, PhysicalAtomicConversion>(
                    storageConverter, module.getContext());
                populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(storagePatterns, storageConverter);

                ConversionTarget storageTarget(*module.getContext());
                auto isNonResourceStorage = [](Value storage) {
                    while (auto cast = storage.getDefiningOp<UnrealizedConversionCastOp>()) {
                        if (cast.getInputs().size() != 1)
                            break;
                        storage = cast.getInputs().front();
                    }
                    return !isa<BlockArgument>(storage);
                };
                storageTarget.addDynamicallyLegalOp<PhysicalLoadOp>(
                    [&](PhysicalLoadOp op) { return isNonResourceStorage(op.getStorage()); });
                storageTarget.addDynamicallyLegalOp<PhysicalStoreOp>(
                    [&](PhysicalStoreOp op) { return isNonResourceStorage(op.getStorage()); });
                storageTarget.addDynamicallyLegalOp<PhysicalAtomicOp>(
                    [&](PhysicalAtomicOp op) { return isNonResourceStorage(op.getStorage()); });
                storageTarget.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp function) {
                    return storageConverter.isSignatureLegal(function.getFunctionType());
                });
                storageTarget.addIllegalOp<ReduceSumOp, ScatterAddOp>();
                storageTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
                if (failed(applyPartialConversion(kernel, storageTarget, std::move(storagePatterns))))
                    return signalPassFailure();
            }

            SmallVector<std::pair<unsigned, unsigned>> convertedArgumentRanges;
            unsigned convertedIndex = 0;
            for (Type type : kernelArgumentTypes) {
                SmallVector<Type> convertedTypes;
                if (failed(storageConverter.convertTypes(type, convertedTypes)))
                    return signalPassFailure();
                convertedArgumentRanges.emplace_back(convertedIndex, convertedTypes.size());
                convertedIndex += convertedTypes.size();
            }
            for (auto [originalIndex, binding] : resourceBindings) {
                if (originalIndex >= convertedArgumentRanges.size()) {
                    source.emitError() << "resource argument index " << originalIndex
                                       << " exceeds converted kernel signature size " << convertedArgumentRanges.size();
                    return signalPassFailure();
                }
                auto [first, count] = convertedArgumentRanges[originalIndex];
                for (unsigned offset = 0; offset < count; ++offset) {
                    const unsigned index = first + offset;
                    if (index >= kernel.getNumArguments()) {
                        source.emitError() << "converted resource argument index " << index
                                           << " exceeds kernel argument count " << kernel.getNumArguments();
                        return signalPassFailure();
                    }
                    kernel.setArgAttr(
                        index, spirv::getInterfaceVarABIAttrName(),
                        spirv::getInterfaceVarABIAttr(binding.first, index, std::nullopt, source.getContext()));
                }
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

            if (failed(lowerGpuAggregateWorkgroupStorage(kernel, module)))
                return signalPassFailure();
        }
    }

    Option<bool> useSpirvStorage{*this, "use-spirv-storage", llvm::cl::desc("Use SPIR-V storage-buffer ABI types"),
                                 llvm::cl::init(false)};
    Option<bool> useSpirvWorkgroupReduction{
        *this, "use-spirv-workgroup-reduction",
        llvm::cl::desc("Lower workgroup reductions to explicit SPIR-V shared-memory trees"), llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<Pass> createVernonToGPUPass(bool useSpirvStorage, bool useSpirvWorkgroupReduction) {
    return std::make_unique<VernonToGPUPass>(useSpirvStorage, useSpirvWorkgroupReduction);
}

void registerVernonToGPUPass() { PassRegistration<VernonToGPUPass>(); }

} // namespace mlir::vernon
