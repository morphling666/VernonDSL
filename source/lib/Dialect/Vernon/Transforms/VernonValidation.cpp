#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAttributeAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <set>
#include <string>

namespace mlir::vernon {
namespace {

struct LocationEndpoint {
    func::FuncOp function;
    unsigned index;
    Type type;
    int64_t location;
};

struct VernonValidatePass : public PassWrapper<VernonValidatePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonValidatePass)

    StringRef getArgument() const final { return "vernon-validate"; }
    StringRef getDescription() const final { return "Validate Vernon shader stages and interface attributes"; }

    void runOnOperation() override {
        bool invalid = false;
        ModuleOp module = getOperation();
        auto compilerContractVersion = module->getAttrOfType<IntegerAttr>("vernon.compiler_contract_version");
        auto pipelineVersion = module->getAttrOfType<IntegerAttr>("vernon.pipeline_version");
        if (!compilerContractVersion || compilerContractVersion.getInt() != VERNON_COMPILER_CONTRACT_VERSION) {
            module.emitError() << "requires vernon.compiler_contract_version = " << VERNON_COMPILER_CONTRACT_VERSION;
            invalid = true;
        }
        if (!pipelineVersion || pipelineVersion.getInt() != VERNON_PIPELINE_VERSION) {
            module.emitError() << "requires vernon.pipeline_version = " << VERNON_PIPELINE_VERSION;
            invalid = true;
        }
        std::map<ShaderStage, SmallVector<LocationEndpoint>> stageInputs;
        std::map<ShaderStage, SmallVector<LocationEndpoint>> stageOutputs;
        std::set<ShaderStage> entryStages;

        for (StructDeclOp declaration : module.getOps<StructDeclOp>()) {
            auto structure = StructType::get(module.getContext(), declaration.getSymName());
            FailureOr<ValueAbiLayout> layout = getValueAbiLayout(structure, module);
            if (failed(layout)) {
                declaration.emitError("does not define a finite canonical Value ABI layout");
                invalid = true;
                continue;
            }
            if (declaration->hasAttr("abi_size") || declaration->hasAttr("abi_alignment") ||
                declaration->hasAttr("abi_field_offsets") || declaration->hasAttr("abi_element_stride")) {
                declaration.emitError("contains obsolete frontend-authored Value ABI layout metadata");
                invalid = true;
            }
        }

        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            Attribute entryAttr = function->getAttr(kEntryAttrName);
            Attribute stageAttr = function->getAttr(kStageAttrName);
            Attribute workgroupAttr = function->getAttr(kWorkgroupSizeAttrName);

            for (unsigned index = 0; index < function.getNumArguments(); ++index) {
                if (function.getArgAttr(index, kTensorDescriptorOwnerAttrName) ||
                    function.getArgAttr(index, kTensorDescriptorComponentAttrName) ||
                    function.getArgAttr(index, kTensorDescriptorDimensionAttrName)) {
                    function.emitError() << "argument #" << index
                                         << " contains internal TensorView descriptor metadata";
                    invalid = true;
                }
            }

            if (entryAttr && !isa<UnitAttr>(entryAttr)) {
                function.emitError() << "'" << kEntryAttrName << "' must be a unit attribute";
                invalid = true;
            }

            if (stageAttr && !entryAttr) {
                function.emitError() << "'" << kStageAttrName << "' requires '" << kEntryAttrName << "'";
                invalid = true;
            }
            if (entryAttr && !stageAttr) {
                function.emitError() << "'" << kEntryAttrName << "' requires '" << kStageAttrName << "'";
                invalid = true;
            }

            std::optional<ShaderStage> stage;
            if (stageAttr) {
                FailureOr<ShaderStage> parsed = parseShaderStage(stageAttr);
                if (failed(parsed)) {
                    function.emitError() << "'" << kStageAttrName
                                         << "' must be one of \"vertex\", \"fragment\", or \"compute\"";
                    invalid = true;
                } else {
                    stage = *parsed;
                }
            }

            if (entryAttr && stage)
                entryStages.insert(*stage);

            if (stage == ShaderStage::Compute) {
                auto workgroup = dyn_cast_if_present<DenseI32ArrayAttr>(workgroupAttr);
                if (!workgroup || workgroup.size() != 3 ||
                    llvm::any_of(workgroup.asArrayRef(), [](int32_t value) { return value <= 0; })) {
                    function.emitError() << "compute entry requires '" << kWorkgroupSizeAttrName
                                         << "' as three positive i32 values";
                    invalid = true;
                }
            } else if (workgroupAttr) {
                function.emitError() << "'" << kWorkgroupSizeAttrName << "' is only valid on a compute entry";
                invalid = true;
            }

            std::map<std::string, unsigned> inputSlots;
            std::map<std::string, unsigned> outputSlots;
            std::set<std::pair<int64_t, int64_t>> bindings;

            auto validateInterface = [&](DictionaryAttr dictionary, Type type, unsigned index, bool isResult) {
                Type abiType = type;
                StringRef abiPrefix = "vernon.abi_";
                if (auto view = dyn_cast<TensorViewType>(type)) {
                    abiType = view.getElementType();
                    abiPrefix = "vernon.element_abi_";
                }
                const bool abiBearing = abiType.isIntOrFloat() ||
                                        isa<TensorType, RankedTensorType, VectorType, StructType, TupleType>(abiType);
                if (abiBearing && failed(verifyValueAbiType(abiType, getOperation()))) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " has no finite canonical Value ABI layout";
                    invalid = true;
                }
                std::string sizeName = (abiPrefix + "size").str();
                if (dictionary && dictionary.get(sizeName)) {
                    auto dtypes = dictionary.getAs<ArrayAttr>((abiPrefix + "leaf_dtypes").str());
                    SmallVector<StringRef> logicalDtypes;
                    if (dtypes)
                        for (Attribute dtype : dtypes) {
                            auto value = dyn_cast<StringAttr>(dtype);
                            logicalDtypes.push_back(value ? value.getValue() : StringRef());
                        }
                    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(abiType, getOperation(), logicalDtypes);
                    auto reportAbiError = [&](StringRef detail) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index
                                             << " has invalid canonical Value ABI metadata: " << detail;
                        invalid = true;
                    };
                    if (failed(layout)) {
                        reportAbiError("type has no finite layout");
                    } else {
                        auto integerMatches = [&](StringRef suffix, uint64_t expected) {
                            auto value = dictionary.getAs<IntegerAttr>((abiPrefix + suffix).str());
                            return value && value.getValue().isNonNegative() &&
                                   value.getValue().getZExtValue() == expected;
                        };
                        auto hash = dictionary.getAs<StringAttr>((abiPrefix + "layout_hash").str());
                        if (!integerMatches("size", layout->size) || !integerMatches("alignment", layout->alignment))
                            reportAbiError("size or alignment does not match the canonical layout");
                        else if (!hash || hash.getValue() != layout->layoutHash)
                            reportAbiError("layout hash does not match the canonical layout");

                        auto offsets = dictionary.getAs<DenseI64ArrayAttr>((abiPrefix + "leaf_offsets").str());
                        auto counts = dictionary.getAs<DenseI64ArrayAttr>((abiPrefix + "leaf_counts").str());
                        auto paths = dictionary.getAs<ArrayAttr>((abiPrefix + "leaf_paths").str());
                        if (!offsets || !counts || !dtypes || !paths || offsets.size() != layout->leaves.size() ||
                            counts.size() != layout->leaves.size() || dtypes.size() != layout->leaves.size() ||
                            paths.size() != layout->leaves.size()) {
                            reportAbiError("leaf arrays do not match the canonical leaf count");
                        } else {
                            for (auto [leafIndex, leaf] : llvm::enumerate(layout->leaves)) {
                                std::string path;
                                llvm::raw_string_ostream pathStream(path);
                                for (auto [componentIndex, component] : llvm::enumerate(leaf.path)) {
                                    if (componentIndex != 0)
                                        pathStream << '/';
                                    if (component.field)
                                        pathStream << *component.field;
                                    else
                                        pathStream << '[' << component.index << ']';
                                }
                                pathStream.flush();
                                auto dtypeAttr = dyn_cast<StringAttr>(dtypes[leafIndex]);
                                auto pathAttr = dyn_cast<StringAttr>(paths[leafIndex]);
                                if (static_cast<uint64_t>(offsets[leafIndex]) != leaf.byteOffset ||
                                    static_cast<uint64_t>(counts[leafIndex]) != leaf.scalarCount || !dtypeAttr ||
                                    dtypeAttr.getValue() != leaf.dtype || !pathAttr || pathAttr.getValue() != path) {
                                    reportAbiError("leaf contract does not match the canonical layout");
                                    break;
                                }
                            }
                        }
                    }
                }

                if (isa<TensorViewType>(type) && dictionary &&
                    (dictionary.get("vernon.tensor_shape") || dictionary.get("vernon.tensor_strides") ||
                     dictionary.get("vernon.tensor_offset"))) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uses retired TensorView layout attributes; shape, strides, and offset "
                                            "are supplied by the dispatch descriptor";
                    invalid = true;
                }

                InterfaceAttrs attrs = parseInterfaceAttrs(dictionary);
                if (attrs.empty()) {
                    if (entryAttr) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index
                                             << " of a Vernon entry must have '" << kInterfaceAttrName << "'";
                        invalid = true;
                    }
                    return;
                }

                if (!entryAttr || !stage) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uses Vernon interface attributes outside a valid entry";
                    invalid = true;
                    return;
                }

                FailureOr<InterfaceKind> parsedKind = parseInterfaceKind(attrs.kind);
                if (failed(parsedKind)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                         << kInterfaceAttrName
                                         << "' must be one of \"input\", \"output\", \"uniform\", "
                                            "or \"resource\"";
                    invalid = true;
                    return;
                }
                InterfaceKind kind = *parsedKind;

                if (isResult && kind != InterfaceKind::Output) {
                    function.emitError() << "result #" << index << " must use vernon.interface = \"output\"";
                    invalid = true;
                }
                if (!isResult && kind == InterfaceKind::Output) {
                    function.emitError() << "argument #" << index << " cannot use vernon.interface = \"output\"";
                    invalid = true;
                }

                std::optional<int64_t> location;
                if (attrs.location) {
                    auto integer = dyn_cast<IntegerAttr>(attrs.location);
                    if (!integer || integer.getInt() < 0) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kLocationAttrName << "' must be a non-negative integer";
                        invalid = true;
                    } else {
                        location = integer.getInt();
                    }
                }

                StringAttr builtin;
                if (attrs.builtin) {
                    builtin = dyn_cast<StringAttr>(attrs.builtin);
                    if (!builtin || builtin.getValue().empty()) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kBuiltinAttrName << "' must be a non-empty string";
                        invalid = true;
                    }
                }

                if (attrs.location && attrs.builtin) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " cannot have both '"
                                         << kLocationAttrName << "' and '" << kBuiltinAttrName << "'";
                    invalid = true;
                }

                bool isIo = kind == InterfaceKind::Input || kind == InterfaceKind::Output;
                bool isDescriptor = kind == InterfaceKind::Uniform || kind == InterfaceKind::Resource;
                if (isIo && !attrs.location && !attrs.builtin) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " "
                                         << stringifyInterfaceKind(kind) << " requires exactly one of '"
                                         << kLocationAttrName << "' or '" << kBuiltinAttrName << "'";
                    invalid = true;
                }

                auto parseNonNegative = [&](Attribute attr, StringRef name) -> std::optional<int64_t> {
                    if (!attr)
                        return std::nullopt;
                    auto integer = dyn_cast<IntegerAttr>(attr);
                    if (!integer || integer.getInt() < 0) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '" << name
                                             << "' must be a non-negative integer";
                        invalid = true;
                        return std::nullopt;
                    }
                    return integer.getInt();
                };

                std::optional<int64_t> descriptorSet = parseNonNegative(attrs.descriptorSet, kDescriptorSetAttrName);
                std::optional<int64_t> binding = parseNonNegative(attrs.binding, kBindingAttrName);

                if (kind == InterfaceKind::Resource && (!attrs.descriptorSet || !attrs.binding)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index << " "
                                         << stringifyInterfaceKind(kind) << " requires both '" << kDescriptorSetAttrName
                                         << "' and '" << kBindingAttrName << "'";
                    invalid = true;
                }
                if (kind == InterfaceKind::Uniform &&
                    static_cast<bool>(attrs.descriptorSet) != static_cast<bool>(attrs.binding)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uniform must provide both '" << kDescriptorSetAttrName << "' and '"
                                         << kBindingAttrName << "', or neither";
                    invalid = true;
                }
                if (isIo && (attrs.descriptorSet || attrs.binding)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " input/output cannot carry descriptor set or binding";
                    invalid = true;
                }
                if (isDescriptor && (attrs.location || attrs.builtin)) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " uniform/resource cannot carry location or builtin";
                    invalid = true;
                }

                if (descriptorSet && binding && !bindings.insert({*descriptorSet, *binding}).second) {
                    function.emitError() << (isResult ? "result #" : "argument #") << index
                                         << " conflicts with another interface at descriptor set " << *descriptorSet
                                         << ", binding " << *binding;
                    invalid = true;
                }

                if (attrs.instanceDivisor) {
                    auto divisor = dyn_cast<IntegerAttr>(attrs.instanceDivisor);
                    if (!divisor || divisor.getInt() <= 0) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kInstanceDivisorAttrName << "' must be a positive integer";
                        invalid = true;
                    }
                    if (isResult || *stage != ShaderStage::Vertex || kind != InterfaceKind::Input) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " '"
                                             << kInstanceDivisorAttrName << "' is only valid on vertex inputs";
                        invalid = true;
                    }
                }

                if (!isIo || (!location && !builtin))
                    return;

                auto &slots = kind == InterfaceKind::Input ? inputSlots : outputSlots;
                int64_t locationSpan = 1;
                if (location && !isResult && *stage == ShaderStage::Vertex && kind == InterfaceKind::Input) {
                    SmallVector<StringRef> logicalDtypes;
                    if (auto dtypes = dictionary.getAs<ArrayAttr>("vernon.abi_leaf_dtypes"))
                        for (Attribute dtype : dtypes) {
                            auto value = dyn_cast<StringAttr>(dtype);
                            logicalDtypes.push_back(value ? value.getValue() : StringRef());
                        }
                    FailureOr<AttributeAbiLayout> plan = getAttributeAbiLayout(type, getOperation(), logicalDtypes);
                    if (failed(plan)) {
                        function.emitError() << "argument #" << index << " has no numeric vertex attribute layout";
                        invalid = true;
                        return;
                    }
                    locationSpan = static_cast<int64_t>(plan->getLocationSpan());
                }
                for (int64_t offset = 0; offset < locationSpan; ++offset) {
                    std::string slot = location ? ("location:" + std::to_string(*location + offset))
                                                : ("builtin:" + builtin.getValue().str());
                    auto [iterator, inserted] = slots.emplace(slot, index);
                    if (!inserted) {
                        function.emitError() << (isResult ? "result #" : "argument #") << index << " conflicts with "
                                             << (kind == InterfaceKind::Input ? "input argument #" : "output result #")
                                             << iterator->second << " at " << slot;
                        invalid = true;
                    }
                }

                if (!location)
                    return;
                auto &endpoints = kind == InterfaceKind::Input ? stageInputs[*stage] : stageOutputs[*stage];
                endpoints.push_back({function, index, type, *location});
            };

            for (unsigned index = 0; index < function.getNumArguments(); ++index)
                validateInterface(function.getArgAttrDict(index), function.getArgumentTypes()[index], index,
                                  /*isResult=*/false);
            for (unsigned index = 0; index < function.getNumResults(); ++index)
                validateInterface(function.getResultAttrDict(index), function.getResultTypes()[index], index,
                                  /*isResult=*/true);
        }

        for (func::FuncOp function : module.getOps<func::FuncOp>()) {
            if (!function->hasAttr(kEntryAttrName))
                continue;
            auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
            if (!stage || stage.getValue() != "compute")
                continue;
            uint64_t totalPhysical = 0;
            function.walk([&](WorkgroupAllocOp allocation) {
                TensorViewType view = allocation.getResult().getType();
                FailureOr<WorkgroupPhysicalStoragePlan> plan = getWorkgroupPhysicalStoragePlan(view, module);
                if (failed(plan)) {
                    allocation.emitError("workgroup allocation has an invalid physical storage plan");
                    invalid = true;
                    return;
                }
                if (totalPhysical > kPortableWorkgroupStorageLimit ||
                    plan->totalPhysicalBytes > kPortableWorkgroupStorageLimit - totalPhysical) {
                    allocation.emitError(
                        "combined workgroup storage exceeds the portable 16 KiB workgroup storage limit");
                    invalid = true;
                    return;
                }
                totalPhysical += plan->totalPhysicalBytes;
            });
        }

        getOperation().walk([&](Operation *operation) {
            if (operation->hasAttr("physical_index")) {
                operation->emitError("retired 'physical_index' attribute is not part of the Vernon IR contract");
                invalid = true;
            }
            if (isa<PhysicalLoadOp, PhysicalStoreOp, PhysicalAtomicOp>(operation)) {
                operation->emitError("physical TensorView operations are reserved for internal lowering");
                invalid = true;
            }
            bool workgroupStorage = isa<WorkgroupAllocOp>(operation);
            if (isa<LoadOp, StoreOp>(operation)) {
                auto view = dyn_cast<TensorViewType>(operation->getOperand(isa<StoreOp>(operation) ? 1 : 0).getType());
                workgroupStorage = view && view.getAddressSpace() == "workgroup";
            }
            if (!workgroupStorage && !isa<AtomicOp, BarrierOp>(operation))
                return;
            func::FuncOp function = operation->getParentOfType<func::FuncOp>();
            auto stage = function ? function->getAttrOfType<StringAttr>(kStageAttrName) : nullptr;
            if (!stage || stage.getValue() != "compute") {
                operation->emitError("workgroup storage, atomics, and barriers require a compute entry");
                invalid = true;
            }
            if (isa<WorkgroupAllocOp>(operation) && (!function || operation->getBlock() != &function.front())) {
                operation->emitError("workgroup storage must be allocated in the compute entry block");
                invalid = true;
            }
        });

        const auto validateStageInterface = [&](ShaderStage producer, ShaderStage consumer) {
            for (LocationEndpoint &input : stageInputs[consumer]) {
                auto matchingLocation = llvm::find_if(stageOutputs[producer], [&](const LocationEndpoint &output) {
                    return output.location == input.location;
                });
                if (matchingLocation == stageOutputs[producer].end()) {
                    input.function.emitError()
                        << stringifyShaderStage(consumer) << " input #" << input.index << " at location "
                        << input.location << " has no " << stringifyShaderStage(producer) << " output";
                    invalid = true;
                } else if (matchingLocation->type != input.type) {
                    input.function.emitError()
                        << stringifyShaderStage(consumer) << " input #" << input.index << " at location "
                        << input.location << " has type " << input.type << ", but the "
                        << stringifyShaderStage(producer) << " output has type " << matchingLocation->type;
                    invalid = true;
                }
            }
        };
        if (entryStages.count(ShaderStage::Vertex) && entryStages.count(ShaderStage::Fragment))
            validateStageInterface(ShaderStage::Vertex, ShaderStage::Fragment);

        if (invalid)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonValidatePass() { return std::make_unique<VernonValidatePass>(); }

void registerVernonValidatePass() { PassRegistration<VernonValidatePass>(); }

} // namespace mlir::vernon
