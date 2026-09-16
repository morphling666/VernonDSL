#include "mlir/Dialect/Vernon/Transforms/VernonGPUProfileABI.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/IR/BuiltinOps.h"

#include <set>
#include <utility>

namespace mlir::vernon {

LogicalResult materializeGPUAutodiffProfileBindings(ModuleOp module) {
    if (!module->getAttrOfType<StringAttr>("vernon.ad_profile"))
        return success();

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
        auto stage = function->getAttrOfType<StringAttr>(kStageAttrName);
        if (!function->hasAttr(kEntryAttrName) || !stage || stage.getValue() != "compute")
            continue;

        std::set<std::pair<int64_t, int64_t>> usedBindings;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            InterfaceAttrs attrs = parseInterfaceAttrs(function.getArgAttrDict(index));
            auto descriptorSet = dyn_cast_if_present<IntegerAttr>(attrs.descriptorSet);
            auto binding = dyn_cast_if_present<IntegerAttr>(attrs.binding);
            if (descriptorSet && binding)
                usedBindings.emplace(descriptorSet.getInt(), binding.getInt());
        }

        int64_t nextBinding = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (!isa<TensorViewType>(function.getArgumentTypes()[index]))
                continue;
            DictionaryAttr dictionary = function.getArgAttrDict(index);
            InterfaceAttrs attrs = parseInterfaceAttrs(dictionary);
            auto kind = dyn_cast_if_present<StringAttr>(attrs.kind);
            auto location = dyn_cast_if_present<IntegerAttr>(attrs.location);
            if (!kind || kind.getValue() != "input" || !location || location.getInt() < 0)
                continue;

            int64_t binding = static_cast<int64_t>(index);
            if (usedBindings.find({0, binding}) != usedBindings.end()) {
                while (usedBindings.find({0, nextBinding}) != usedBindings.end())
                    ++nextBinding;
                binding = nextBinding++;
            }
            usedBindings.emplace(0, binding);

            NamedAttrList materialized(dictionary);
            materialized.set(kInterfaceAttrName, StringAttr::get(module.getContext(), "resource"));
            materialized.erase(kLocationAttrName);
            materialized.set(kDescriptorSetAttrName, IntegerAttr::get(IntegerType::get(module.getContext(), 64), 0));
            materialized.set(kBindingAttrName, IntegerAttr::get(IntegerType::get(module.getContext(), 64), binding));
            function.setArgAttrs(index, materialized.getDictionary(module.getContext()));
        }
    }
    return success();
}

} // namespace mlir::vernon
