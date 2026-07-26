#include "mlir/Dialect/Vernon/Transforms/VernonValidation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
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
        SmallVector<LocationEndpoint> vertexOutputs;
        SmallVector<LocationEndpoint> fragmentInputs;
        bool hasVertexEntry = false;
        bool hasFragmentEntry = false;

        for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
            Attribute entryAttr = function->getAttr(kEntryAttrName);
            Attribute stageAttr = function->getAttr(kStageAttrName);
            Attribute workgroupAttr = function->getAttr(kWorkgroupSizeAttrName);

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

            if (entryAttr && stage == ShaderStage::Vertex)
                hasVertexEntry = true;
            if (entryAttr && stage == ShaderStage::Fragment)
                hasFragmentEntry = true;

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
                    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
                        FailureOr<StaticAttributePlan> plan =
                            getStaticAttributePlan(tensor.getElementType(), tensor.getShape());
                        if (failed(plan)) {
                            function.emitError() << "argument #" << index << " has no numeric vertex attribute layout";
                            invalid = true;
                            return;
                        }
                        locationSpan = static_cast<int64_t>(plan->leaves.size());
                    }
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
                if (*stage == ShaderStage::Vertex && kind == InterfaceKind::Output) {
                    vertexOutputs.push_back({function, index, type, *location});
                } else if (*stage == ShaderStage::Fragment && kind == InterfaceKind::Input) {
                    fragmentInputs.push_back({function, index, type, *location});
                }
            };

            for (unsigned index = 0; index < function.getNumArguments(); ++index)
                validateInterface(function.getArgAttrDict(index), function.getArgumentTypes()[index], index,
                                  /*isResult=*/false);
            for (unsigned index = 0; index < function.getNumResults(); ++index)
                validateInterface(function.getResultAttrDict(index), function.getResultTypes()[index], index,
                                  /*isResult=*/true);
        }

        if (hasVertexEntry && hasFragmentEntry) {
            for (const LocationEndpoint &fragmentInput : fragmentInputs) {
                func::FuncOp fragmentFunction = fragmentInput.function;
                auto matchingLocation = llvm::find_if(vertexOutputs, [&](const LocationEndpoint &output) {
                    return output.location == fragmentInput.location;
                });
                if (matchingLocation == vertexOutputs.end()) {
                    fragmentFunction.emitError() << "fragment input #" << fragmentInput.index << " at location "
                                                 << fragmentInput.location << " has no vertex output";
                    invalid = true;
                } else if (matchingLocation->type != fragmentInput.type) {
                    fragmentFunction.emitError() << "fragment input #" << fragmentInput.index << " at location "
                                                 << fragmentInput.location << " has type " << fragmentInput.type
                                                 << ", but the vertex output has type " << matchingLocation->type;
                    invalid = true;
                }
            }
        }

        if (invalid)
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonValidatePass() { return std::make_unique<VernonValidatePass>(); }

void registerVernonValidatePass() { PassRegistration<VernonValidatePass>(); }

} // namespace mlir::vernon
