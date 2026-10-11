#include "mlir/Dialect/Vernon/Transforms/VernonGraphicsBindings.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace mlir::vernon {
namespace {

bool isGraphicsEntry(func::FuncOp function) {
    auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
    return function->hasAttr(kEntryAttrName) && stage && stage.getValue() != "compute";
}

std::string typeKey(Type type) {
    std::string key;
    llvm::raw_string_ostream stream(key);
    type.print(stream);
    return key;
}

std::string sourceName(func::FuncOp function, unsigned index) {
    auto name = function.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
    return name ? name.getValue().str() : std::string();
}

} // namespace

const SmallVector<uint32_t> *GraphicsBindingPlan::find(StringRef entry, unsigned argument) const {
    auto found = arguments.find({entry.str(), argument});
    return found == arguments.end() ? nullptr : &found->second;
}

FailureOr<GraphicsBindingRequests> collectGraphicsBindingRequests(ModuleOp module) {
    GraphicsBindingRequests collected;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (!isGraphicsEntry(function))
            continue;
        const std::string symbol = function.getSymName().str();
        if (collected.stage.empty())
            collected.stage = function->getAttrOfType<StringAttr>(kStageAttrName).getValue().str();
        uint64_t inlineSize = 0;
        bool readsStorage = false;
        for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
            const unsigned argument = static_cast<unsigned>(index);
            DictionaryAttr dictionary = function.getArgAttrDict(argument);
            InterfaceAttrs attrs = parseInterfaceAttrs(dictionary);
            auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
            const bool generated = dictionary && dictionary.get(kGeneratedBindingAttrName);
            const auto request = [&](StringRef requestKind, uint32_t count) {
                collected.requests.push_back(
                    {symbol, argument, sourceName(function, argument), requestKind.str(), typeKey(type), count});
            };
            if (function.getArgAttr(argument, kTensorMetadataCarrierAttrName))
                continue;
            if (attrs.binding && !generated) {
                auto set = dyn_cast_if_present<IntegerAttr>(attrs.descriptorSet);
                collected.authored.push_back({symbol, argument, sourceName(function, argument), set ? set.getInt() : 0,
                                              cast<IntegerAttr>(attrs.binding).getInt()});
            }
            if (auto view = dyn_cast<TensorViewType>(type)) {
                if (view.getAddressSpace() != "device")
                    continue;
                if (view.getAccess() != "read")
                    return function.emitError()
                           << "graphics TensorView argument #" << index << " must have read access";
                if (attrs.binding || attrs.descriptorSet)
                    return function.emitError()
                           << "graphics TensorView argument #" << index << " cannot declare an explicit set or binding";
                FailureOr<ValueAbiLayout> layout = getValueStorageLayout(view.getElementType(), module);
                if (failed(layout) || layout->leaves.empty())
                    return function.emitError()
                           << "graphics TensorView argument #" << index << " has no storage layout";
                request("storage", static_cast<uint32_t>(layout->leaves.size()));
                readsStorage = true;
                continue;
            }
            if (isa<TextureType, SamplerType>(type)) {
                if (generated || (!attrs.binding && !attrs.descriptorSet))
                    request(isa<TextureType>(type) ? "texture" : "sampler", 1);
                continue;
            }
            if (!kind || kind.getValue() != "uniform" || attrs.binding)
                continue;
            if (isa<TensorType>(type)) {
                request("uniform_buffer", 1);
                continue;
            }
            FailureOr<ByteTransportPlan> layout =
                getByteTransportPlan(type, module, PhysicalAbiProfile::VulkanPushConstant);
            if (failed(layout))
                return function.emitError() << "cannot plan graphics inline layout for argument #" << index;
            const uint64_t offset = llvm::alignTo(inlineSize, layout->root->alignment);
            const bool spill = (isa<RankedTensorType>(type) && cast<RankedTensorType>(type).getRank() > 2) ||
                               offset > 128 || layout->root->size > 128 - std::min<uint64_t>(offset, 128);
            if (spill)
                request("uniform_buffer", 1);
            else
                inlineSize = offset + layout->root->size;
        }
        if (readsStorage)
            collected.requests.push_back({symbol, std::nullopt, std::string(), "metadata", std::string(), 1});
    }
    return collected;
}

FailureOr<std::vector<GraphicsBindingLayout>> linkGraphicsBindings(ArrayRef<GraphicsBindingRequests> modules,
                                                                   std::string &error) {
    struct Owner {
        std::string stage;
        std::string sourceName;
    };
    std::map<std::pair<int64_t, int64_t>, Owner> authored;
    std::set<uint32_t> used;
    for (const GraphicsBindingRequests &module : modules)
        for (const GraphicsAuthoredBinding &binding : module.authored) {
            auto [owner, inserted] =
                authored.try_emplace({binding.set, binding.binding}, Owner{module.stage, binding.sourceName});
            if (!inserted && owner->second.stage != module.stage &&
                (binding.sourceName.empty() || owner->second.sourceName != binding.sourceName)) {
                error = "graphics endpoints " + owner->second.stage + " '" + owner->second.sourceName + "' and " +
                        module.stage + " '" + binding.sourceName + "' both use descriptor (set " +
                        std::to_string(binding.set) + ", binding " + std::to_string(binding.binding) +
                        ") for different Program Values";
                return failure();
            }
            if (binding.set == 0)
                used.insert(static_cast<uint32_t>(binding.binding));
        }

    using ShareKey = std::tuple<std::string, std::string, std::string, uint32_t>;
    std::map<ShareKey, SmallVector<uint32_t>> shared;
    std::vector<GraphicsBindingLayout> layouts;
    for (const GraphicsBindingRequests &module : modules) {
        GraphicsBindingLayout &layout = layouts.emplace_back();
        for (const GraphicsBindingRequest &request : module.requests) {
            const bool shareable = request.argument && !request.sourceName.empty();
            const ShareKey key{request.sourceName, request.kind, request.typeKey, request.count};
            if (auto found = shareable ? shared.find(key) : shared.end(); found != shared.end()) {
                layout.assignments.emplace_back(request, found->second);
                continue;
            }
            SmallVector<uint32_t> bindings;
            for (uint32_t candidate = 0; bindings.size() < request.count; ++candidate)
                if (used.insert(candidate).second)
                    bindings.push_back(candidate);
            if (shareable)
                shared.emplace(key, bindings);
            layout.assignments.emplace_back(request, std::move(bindings));
        }
    }
    return layouts;
}

void attachGraphicsBindingLayout(ModuleOp module, const GraphicsBindingLayout &layout) {
    Builder builder(module.getContext());
    SmallVector<Attribute> rows;
    for (const auto &[request, bindings] : layout.assignments) {
        SmallVector<int32_t> values(bindings.begin(), bindings.end());
        rows.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("entry", builder.getStringAttr(request.entry)),
            builder.getNamedAttr(
                "argument",
                builder.getI64IntegerAttr(request.argument ? static_cast<int64_t>(*request.argument) : int64_t{-1})),
            builder.getNamedAttr("bindings", builder.getDenseI32ArrayAttr(values)),
        }));
    }
    module->setAttr(kGraphicsBindingLayoutAttrName, builder.getArrayAttr(rows));
}

FailureOr<GraphicsBindingPlan> planGraphicsBindings(ModuleOp module) {
    FailureOr<GraphicsBindingRequests> requests = collectGraphicsBindingRequests(module);
    if (failed(requests))
        return failure();
    std::map<std::pair<std::string, int64_t>, SmallVector<uint32_t>> assigned;
    if (auto rows = module->getAttrOfType<ArrayAttr>(kGraphicsBindingLayoutAttrName)) {
        for (Attribute rowAttr : rows) {
            auto row = dyn_cast<DictionaryAttr>(rowAttr);
            auto entry = row ? row.getAs<StringAttr>("entry") : nullptr;
            auto argument = row ? row.getAs<IntegerAttr>("argument") : nullptr;
            auto bindings = row ? row.getAs<DenseI32ArrayAttr>("bindings") : nullptr;
            if (!entry || !argument || !bindings)
                return module.emitError("graphics binding layout has a malformed row");
            SmallVector<uint32_t> values;
            for (int32_t binding : bindings.asArrayRef()) {
                if (binding < 0)
                    return module.emitError("graphics binding layout has a negative binding");
                values.push_back(static_cast<uint32_t>(binding));
            }
            if (!assigned.emplace(std::pair{entry.getValue().str(), argument.getInt()}, std::move(values)).second)
                return module.emitError("graphics binding layout assigns one argument twice");
        }
        if (assigned.size() != requests->requests.size())
            return module.emitError("graphics binding layout does not match the module's binding requests");
    } else {
        std::string error;
        FailureOr<std::vector<GraphicsBindingLayout>> layouts = linkGraphicsBindings({*requests}, error);
        if (failed(layouts))
            return module.emitError(error);
        for (const auto &[request, bindings] : layouts->front().assignments)
            assigned.emplace(
                std::pair{request.entry, request.argument ? static_cast<int64_t>(*request.argument) : int64_t{-1}},
                bindings);
    }

    GraphicsBindingPlan plan;
    for (const GraphicsBindingRequest &request : requests->requests) {
        auto found =
            assigned.find({request.entry, request.argument ? static_cast<int64_t>(*request.argument) : int64_t{-1}});
        if (found == assigned.end() || found->second.size() != request.count)
            return module.emitError() << "graphics binding layout does not cover " << request.kind
                                      << " request of entry '" << request.entry << "'";
        if (request.argument)
            plan.arguments[{request.entry, *request.argument}] = found->second;
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (!isGraphicsEntry(function))
            continue;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (!function.getArgAttr(index, kTensorMetadataCarrierAttrName))
                continue;
            auto carrier = assigned.find({function.getSymName().str(), int64_t{-1}});
            if (carrier == assigned.end())
                return function.emitError("TensorView metadata carrier has no planned binding");
            plan.arguments[{function.getSymName().str(), index}] = carrier->second;
        }
    }
    return plan;
}

LogicalResult materializeGraphicsResourceBindings(ModuleOp module) {
    FailureOr<GraphicsBindingPlan> plan = planGraphicsBindings(module);
    if (failed(plan))
        return failure();
    Builder builder(module.getContext());
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        if (!isGraphicsEntry(function))
            continue;
        for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
            if (!isa<TextureType, SamplerType>(type))
                continue;
            const SmallVector<uint32_t> *planned = plan->find(function.getSymName(), static_cast<unsigned>(index));
            if (!planned)
                continue;
            NamedAttrList attrs(function.getArgAttrDict(index));
            attrs.set(kDescriptorSetAttrName, builder.getI64IntegerAttr(0));
            attrs.set(kBindingAttrName, builder.getI64IntegerAttr(planned->front()));
            attrs.set(kGeneratedBindingAttrName, builder.getUnitAttr());
            function.setArgAttrs(index, attrs.getDictionary(module.getContext()));
        }
    }
    return success();
}

} // namespace mlir::vernon
