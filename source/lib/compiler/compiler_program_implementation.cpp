#include "compiler_program_implementation.h"

#include "compiler_program_capture.h"
#include "compiler_program_compute.h"
#include "compiler_program_stage.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace vernon::compiler {
namespace {

struct InterfaceValue {
    std::string name;
    std::string role;
    std::string source;
};

void collectInterface(const llvm::json::Array *rows, std::vector<InterfaceValue> &interface) {
    if (!rows)
        return;
    for (const llvm::json::Value &value : *rows) {
        const llvm::json::Object *row = value.getAsObject();
        const std::optional<llvm::StringRef> name = row ? row->getString("vernon.source_name") : std::nullopt;
        if (!name)
            continue;
        InterfaceValue reflected{name->str(), "", ""};
        if (std::optional<llvm::StringRef> role = row->getString("vernon.autodiff_role"))
            reflected.role = role->str();
        if (std::optional<llvm::StringRef> source = row->getString("vernon.autodiff_source"))
            reflected.source = source->str();
        else if (!reflected.role.empty())
            reflected.source = reflected.name;
        interface.push_back(std::move(reflected));
        if (const llvm::json::Array *gradientPaths = row->getArray("vernon.autodiff_gradient_paths"))
            for (const llvm::json::Value &pathValue : *gradientPaths)
                if (const std::optional<llvm::StringRef> path = pathValue.getAsString())
                    interface.push_back(InterfaceValue{name->str(), "gradient", path->str()});
    }
}

bool matchesAutodiffBinding(llvm::StringRef logicalRole, llvm::StringRef logicalSource,
                            const InterfaceValue &physical) {
    if (logicalRole == "tape")
        return isProgramTapeCarrierRole(physical.role);
    return physical.role == logicalRole && physical.source == logicalSource;
}

bool compiledKernelHasTapeAbi(const llvm::json::Object &compiledEntry) {
    const auto scan = [](const llvm::json::Array *rows) {
        if (!rows)
            return false;
        for (const llvm::json::Value &value : *rows) {
            const llvm::json::Object *row = value.getAsObject();
            const std::optional<llvm::StringRef> builtin =
                row ? (row->getString("vernon.builtin") ? row->getString("vernon.builtin") : row->getString("builtin"))
                    : std::nullopt;
            const std::optional<llvm::StringRef> role = row ? row->getString("vernon.autodiff_role") : std::nullopt;
            if ((builtin && isProgramKernelTapeBuiltin(*builtin)) || (role && isProgramTapeCarrierRole(*role)))
                return true;
        }
        return false;
    };
    return scan(compiledEntry.getArray("arguments")) || scan(compiledEntry.getArray("results"));
}

void annotateProgramValueLeafProjections(const llvm::json::Object &execution, llvm::json::Array &bindings) {
    const llvm::json::Array *values = execution.getArray("values");
    if (!values)
        return;
    for (llvm::json::Value &bindingValue : bindings) {
        llvm::json::Object *binding = bindingValue.getAsObject();
        const std::optional<llvm::StringRef> role = binding ? binding->getString("autodiff_role") : std::nullopt;
        const std::optional<llvm::StringRef> source = binding ? binding->getString("autodiff_source") : std::nullopt;
        const std::optional<llvm::StringRef> parameter = binding ? binding->getString("parameter") : std::nullopt;
        const std::optional<int64_t> valueId = binding ? binding->getInteger("value") : std::nullopt;
        if (!binding || (role != "gradient" && role != "cotangent") || !source || !parameter || !valueId ||
            *valueId < 0 || static_cast<size_t>(*valueId) >= values->size())
            continue;
        const llvm::json::Object *value = (*values)[*valueId].getAsObject();
        const llvm::json::Object *layout = value ? value->getObject("value_layout") : nullptr;
        if (!layout)
            continue;
        if (const std::optional<size_t> leaf = resolveProgramValueLeafIndex(*layout, *source, *parameter))
            (*binding)["leaf"] = static_cast<int64_t>(*leaf);
    }
}

llvm::json::Object *findProgramNode(llvm::json::Object &execution, llvm::StringRef graphName,
                                    llvm::StringRef requestId) {
    llvm::json::Array *graphs = execution.getArray("graphs");
    if (!graphs)
        return nullptr;
    for (llvm::json::Value &graphValue : *graphs) {
        llvm::json::Object *graph = graphValue.getAsObject();
        if (!graph || graph->getString("name") != graphName)
            continue;
        llvm::json::Array *nodes = graph->getArray("nodes");
        if (!nodes)
            return nullptr;
        for (llvm::json::Value &nodeValue : *nodes)
            if (llvm::json::Object *node = nodeValue.getAsObject(); node && node->getString("stage") == requestId)
                return node;
    }
    return nullptr;
}

void copyProgramValueLeafProjections(const llvm::json::Array &source, llvm::json::Array &destination) {
    for (llvm::json::Value &destinationValue : destination) {
        llvm::json::Object *binding = destinationValue.getAsObject();
        const std::optional<llvm::StringRef> parameter = binding ? binding->getString("parameter") : std::nullopt;
        const std::optional<int64_t> value = binding ? binding->getInteger("value") : std::nullopt;
        if (!binding || !parameter || !value)
            continue;
        const auto projected = llvm::find_if(source, [&](const llvm::json::Value &sourceValue) {
            const llvm::json::Object *candidate = sourceValue.getAsObject();
            return candidate && candidate->getString("parameter") == parameter &&
                   candidate->getInteger("value") == value && candidate->getInteger("leaf");
        });
        if (projected != source.end())
            (*binding)["leaf"] = *projected->getAsObject()->getInteger("leaf");
    }
}

} // namespace

bool normalizeProgramImplementationAbi(llvm::json::Object &execution, llvm::json::Object &request,
                                       const llvm::json::Object &compiledEntry, std::string &error) {
    llvm::json::Array *requestBindings = request.getArray("bindings");
    const std::optional<llvm::StringRef> graphName = request.getString("graph");
    const std::optional<llvm::StringRef> requestId = request.getString("id");
    if (!requestBindings || !graphName || !requestId) {
        error = "Program implementation request has no graph ABI";
        return false;
    }
    std::vector<InterfaceValue> interface;
    collectInterface(compiledEntry.getArray("arguments"), interface);
    collectInterface(compiledEntry.getArray("results"), interface);

    ProgramImplementationBindingPlan bindingPlan;
    std::set<std::string> matched;
    for (size_t bindingIndex = 0; bindingIndex < requestBindings->size();) {
        llvm::json::Object *binding = (*requestBindings)[bindingIndex].getAsObject();
        const std::optional<llvm::StringRef> parameter = binding ? binding->getString("parameter") : std::nullopt;
        if (!parameter) {
            error = "Program implementation request has an invalid binding";
            return false;
        }
        const std::optional<llvm::StringRef> role = binding->getString("autodiff_role");
        const auto exact = std::find_if(interface.begin(), interface.end(), [&](const InterfaceValue &value) {
            if (value.name != *parameter)
                return false;
            return !role || role->empty() || value.role == *role;
        });
        if (exact != interface.end()) {
            matched.insert(exact->name);
            ++bindingIndex;
            continue;
        }
        const std::optional<llvm::StringRef> source = binding->getString("autodiff_source");
        if (!role || !source) {
            const std::optional<int64_t> valueId = binding->getInteger("value");
            const llvm::json::Array *values = execution.getArray("values");
            const llvm::json::Object *programValue =
                values && valueId && *valueId >= 0 && static_cast<size_t>(*valueId) < values->size()
                    ? (*values)[*valueId].getAsObject()
                    : nullptr;
            const std::optional<llvm::StringRef> valueName =
                programValue ? programValue->getString("name") : std::nullopt;
            const auto canonical =
                valueName ? std::find_if(interface.begin(), interface.end(),
                                         [&](const InterfaceValue &value) { return value.name == *valueName; })
                          : interface.end();
            if (canonical != interface.end()) {
                matched.insert(canonical->name);
                bindingPlan.parameterAliases.emplace(parameter->str(), std::vector<std::string>{canonical->name});
                (*binding)["parameter"] = canonical->name;
                ++bindingIndex;
                continue;
            }
            error = "compiled kernel ABI does not bind Program parameter '" + parameter->str() + "'";
            return false;
        }
        std::vector<const InterfaceValue *> semantic;
        for (const InterfaceValue &value : interface)
            if (matchesAutodiffBinding(*role, *source, value))
                semantic.push_back(&value);
        if (!semantic.empty()) {
            std::vector<std::string> names;
            names.reserve(semantic.size());
            for (const InterfaceValue *value : semantic) {
                matched.insert(value->name);
                names.push_back(value->name);
            }
            bindingPlan.parameterAliases.emplace(parameter->str(), names);
            (*binding)["parameter"] = names.front();
            const std::optional<int64_t> value = binding->getInteger("value");
            for (size_t index = 1; index < names.size(); ++index) {
                llvm::json::Object expanded{{"parameter", names[index]}};
                if (value)
                    expanded["value"] = *value;
                expanded["autodiff_role"] = role->str();
                expanded["autodiff_source"] = source->str();
                requestBindings->insert(requestBindings->begin() + bindingIndex + index,
                                        llvm::json::Value(std::move(expanded)));
            }
            bindingIndex += names.size();
        } else if (*role == "retained_primal" || *role == "cotangent" ||
                   (*role == "tape" && !compiledKernelHasTapeAbi(compiledEntry))) {
            bindingPlan.omittedParameters.insert(parameter->str());
            requestBindings->erase(requestBindings->begin() + bindingIndex);
        } else if (*role == "tape") {
            matched.insert(parameter->str());
            ++bindingIndex;
        } else {
            error = "compiled kernel ABI does not provide Program " + role->str() + " '" + source->str() + "'";
            return false;
        }
    }
    annotateProgramValueLeafProjections(execution, *requestBindings);
    for (const InterfaceValue &value : interface)
        if (!value.role.empty() && !matched.count(value.name)) {
            error = "compiled kernel ABI requires unmapped Program value '" + value.name + "'";
            return false;
        }
    return applyProgramImplementationBindingPlan(execution, *graphName, *requestId, *requestBindings, bindingPlan,
                                                 error);
}

bool applyProgramImplementationBindingPlan(llvm::json::Object &execution, llvm::StringRef graphName,
                                           llvm::StringRef requestId, const llvm::json::Array &requestBindings,
                                           const ProgramImplementationBindingPlan &plan, std::string &error) {
    llvm::json::Object *node = findProgramNode(execution, graphName, requestId);
    if (!node) {
        error = "Program implementation request does not identify an executable node";
        return false;
    }
    std::set<int64_t> boundValues;
    llvm::json::Array *nodeBindings = node->getArray("bindings");
    if (!nodeBindings) {
        error = "Program executable node has no value bindings";
        return false;
    }
    for (size_t bindingIndex = 0; bindingIndex < nodeBindings->size();) {
        llvm::json::Object *object = (*nodeBindings)[bindingIndex].getAsObject();
        const std::optional<llvm::StringRef> parameter = object ? object->getString("parameter") : std::nullopt;
        if (!parameter) {
            error = "Program executable node has an invalid value binding";
            return false;
        }
        if (plan.omittedParameters.count(parameter->str())) {
            nodeBindings->erase(nodeBindings->begin() + bindingIndex);
            continue;
        }
        const std::optional<int64_t> value = object->getInteger("value");
        if (auto alias = plan.parameterAliases.find(parameter->str()); alias != plan.parameterAliases.end()) {
            (*object)["parameter"] = alias->second.front();
            for (size_t index = 1; index < alias->second.size(); ++index) {
                llvm::json::Object expanded{{"parameter", alias->second[index]}};
                if (value)
                    expanded["value"] = *value;
                nodeBindings->insert(nodeBindings->begin() + bindingIndex + index,
                                     llvm::json::Value(std::move(expanded)));
            }
            bindingIndex += alias->second.size();
        } else
            ++bindingIndex;
        if (value)
            boundValues.insert(*value);
    }
    copyProgramValueLeafProjections(requestBindings, *nodeBindings);
    if (llvm::json::Array *grid = node->getArray("grid"))
        for (const llvm::json::Value &componentValue : *grid)
            if (const llvm::json::Object *component = componentValue.getAsObject())
                if (const llvm::json::Object *control = component->getObject("control"))
                    if (std::optional<int64_t> argument = control->getInteger("argument"))
                        boundValues.insert(*argument);
    const auto retainBound = [&](llvm::json::Array *values) {
        if (!values)
            return;
        for (auto value = values->begin(); value != values->end();)
            if (std::optional<int64_t> id = value->getAsInteger(); id && !boundValues.count(*id))
                value = values->erase(value);
            else
                ++value;
    };
    retainBound(node->getArray("operands"));
    std::set<int64_t> liveValues = boundValues;
    if (llvm::json::Array *results = node->getArray("results"))
        for (const llvm::json::Value &result : *results)
            if (std::optional<int64_t> value = result.getAsInteger())
                liveValues.insert(*value);
    if (llvm::json::Array *resources = node->getArray("resources")) {
        for (auto resource = resources->begin(); resource != resources->end();) {
            if (const llvm::json::Object *object = resource->getAsObject();
                object && object->getInteger("value") && !liveValues.count(*object->getInteger("value")))
                resource = resources->erase(resource);
            else
                ++resource;
        }
    }
    rebuildProgramDependencies(execution, graphName);
    rebuildProgramCaptures(execution);
    return true;
}

} // namespace vernon::compiler
