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
    int64_t physicalLeaf{};
};

std::optional<std::string> projectedGradientPath(llvm::StringRef root, const llvm::json::Array *path,
                                                 size_t rootCount) {
    std::string result = root.str();
    if (!path)
        return result;
    size_t begin = 0;
    if (rootCount > 1 && !path->empty()) {
        const std::optional<int64_t> rootIndex = (*path)[0].getAsInteger();
        if (!rootIndex || *rootIndex < 0 || static_cast<size_t>(*rootIndex) >= rootCount)
            return std::nullopt;
        begin = 1;
    }
    for (size_t index = begin; index < path->size(); ++index) {
        result.push_back('.');
        if (const std::optional<llvm::StringRef> field = (*path)[index].getAsString())
            result += *field;
        else if (const std::optional<int64_t> element = (*path)[index].getAsInteger())
            result += std::to_string(*element);
        else
            return std::nullopt;
    }
    return result;
}

void collectInterface(const llvm::json::Array *rows, std::vector<InterfaceValue> &interface) {
    if (!rows)
        return;
    for (const llvm::json::Value &value : *rows) {
        const llvm::json::Object *row = value.getAsObject();
        const std::optional<llvm::StringRef> name = row ? row->getString("vernon.source_name") : std::nullopt;
        if (!name)
            continue;
        InterfaceValue reflected{name->str(), "", "", 0};
        if (std::optional<llvm::StringRef> role = row->getString("vernon.autodiff_role"))
            reflected.role = role->str();
        if (std::optional<llvm::StringRef> source = row->getString("vernon.autodiff_source"))
            reflected.source = source->str();
        else if (!reflected.role.empty())
            reflected.source = reflected.name;
        const llvm::json::Array *gradientPaths = row->getArray("vernon.autodiff_gradient_paths");
        if (!gradientPaths) {
            interface.push_back(std::move(reflected));
            continue;
        }
        const llvm::json::Object *layout = row->getObject("value_layout");
        const llvm::json::Array *leaves = layout ? layout->getArray("leaves") : nullptr;
        if (!leaves) {
            for (auto [physicalLeaf, pathValue] : llvm::enumerate(*gradientPaths))
                if (const std::optional<llvm::StringRef> path = pathValue.getAsString())
                    interface.push_back(
                        InterfaceValue{name->str(), "gradient", path->str(), static_cast<int64_t>(physicalLeaf)});
            continue;
        }
        for (auto [physicalLeaf, leafValue] : llvm::enumerate(*leaves)) {
            const llvm::json::Object *leaf = leafValue.getAsObject();
            const llvm::json::Array *leafPath = leaf ? leaf->getArray("path") : nullptr;
            size_t rootIndex = 0;
            if (gradientPaths->size() > 1) {
                const std::optional<int64_t> index =
                    leafPath && !leafPath->empty() ? (*leafPath)[0].getAsInteger() : std::nullopt;
                if (!index || *index < 0 || static_cast<size_t>(*index) >= gradientPaths->size())
                    continue;
                rootIndex = static_cast<size_t>(*index);
            }
            const std::optional<llvm::StringRef> root = (*gradientPaths)[rootIndex].getAsString();
            const std::optional<std::string> path =
                root ? projectedGradientPath(*root, leafPath, gradientPaths->size()) : std::nullopt;
            if (path)
                interface.push_back(InterfaceValue{name->str(), "gradient", *path, static_cast<int64_t>(physicalLeaf)});
        }
    }
}

bool matchesAutodiffBinding(llvm::StringRef logicalRole, llvm::StringRef logicalSource,
                            const InterfaceValue &physical) {
    if (logicalRole == "tape")
        return isProgramTapeCarrierRole(physical.role);
    return physical.role == logicalRole && physical.source == logicalSource;
}

std::optional<size_t> logicalProjection(const llvm::json::Object *layout, llvm::StringRef source,
                                        const InterfaceValue &physical, bool allowOrdinal) {
    if (!layout)
        return std::nullopt;
    const llvm::json::Array *leaves = layout->getArray("leaves");
    const auto nonRootProjection = [&](std::optional<size_t> leaf) -> std::optional<size_t> {
        if (!leaf || !leaves || *leaf >= leaves->size())
            return std::nullopt;
        const llvm::json::Object *row = (*leaves)[*leaf].getAsObject();
        const llvm::json::Array *path = row ? row->getArray("path") : nullptr;
        return leaves->size() == 1 && path && path->empty() ? std::nullopt : leaf;
    };
    if (const std::optional<size_t> leaf =
            nonRootProjection(resolveProgramValueLeafIndex(*layout, source, physical.name)))
        return leaf;
    if (const std::optional<size_t> leaf =
            nonRootProjection(resolveProgramValueLeafIndex(*layout, source, physical.source)))
        return leaf;
    if (allowOrdinal && physical.source == source && physical.physicalLeaf >= 0 && leaves &&
        static_cast<size_t>(physical.physicalLeaf) < leaves->size())
        return nonRootProjection(static_cast<size_t>(physical.physicalLeaf));
    return std::nullopt;
}

bool isWholeValueProjection(const llvm::json::Object *layout) {
    const llvm::json::Array *leaves = layout ? layout->getArray("leaves") : nullptr;
    const llvm::json::Object *leaf = leaves && leaves->size() == 1 ? (*leaves)[0].getAsObject() : nullptr;
    const llvm::json::Array *path = leaf ? leaf->getArray("path") : nullptr;
    return path && path->empty();
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

    llvm::json::Array sourceBindings = std::move(*requestBindings);
    llvm::json::Array normalizedBindings;
    std::set<std::string> matched;
    for (const llvm::json::Value &bindingValue : sourceBindings) {
        const llvm::json::Object *binding = bindingValue.getAsObject();
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
            llvm::json::Object normalized = *binding;
            const std::optional<llvm::StringRef> source = binding->getString("autodiff_source");
            const std::optional<int64_t> logicalValueId = binding->getInteger("value");
            const llvm::json::Array *values = execution.getArray("values");
            const llvm::json::Object *logicalValue = values && logicalValueId && *logicalValueId >= 0 &&
                                                             static_cast<size_t>(*logicalValueId) < values->size()
                                                         ? (*values)[*logicalValueId].getAsObject()
                                                         : nullptr;
            if (source)
                if (const std::optional<size_t> leaf = logicalProjection(
                        logicalValue ? logicalValue->getObject("value_layout") : nullptr, *source, *exact, false))
                    normalized["leaf"] = static_cast<int64_t>(*leaf);
            normalizedBindings.emplace_back(std::move(normalized));
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
                llvm::json::Object normalized = *binding;
                normalized["parameter"] = canonical->name;
                normalizedBindings.emplace_back(std::move(normalized));
                continue;
            }
            error = "compiled kernel ABI does not bind Program parameter '" + parameter->str() + "'";
            return false;
        }
        std::vector<std::pair<const InterfaceValue *, std::optional<int64_t>>> semantic;
        const std::optional<int64_t> logicalValueId = binding->getInteger("value");
        const llvm::json::Array *values = execution.getArray("values");
        const llvm::json::Object *logicalValue =
            values && logicalValueId && *logicalValueId >= 0 && static_cast<size_t>(*logicalValueId) < values->size()
                ? (*values)[*logicalValueId].getAsObject()
                : nullptr;
        const llvm::json::Object *logicalLayout = logicalValue ? logicalValue->getObject("value_layout") : nullptr;
        for (const InterfaceValue &value : interface) {
            if (*role == "gradient" && value.role == *role && logicalLayout) {
                if (const std::optional<size_t> leaf = logicalProjection(logicalLayout, *source, value, true))
                    semantic.emplace_back(&value, static_cast<int64_t>(*leaf));
                else if (value.source == *source && isWholeValueProjection(logicalLayout))
                    semantic.emplace_back(&value, std::nullopt);
            } else if (matchesAutodiffBinding(*role, *source, value)) {
                semantic.emplace_back(&value, std::nullopt);
            }
        }
        if (!semantic.empty()) {
            for (const auto &[value, logicalLeaf] : semantic) {
                llvm::json::Object normalized = *binding;
                if (*role == "gradient") {
                    normalized["endpoint"] = value->name;
                    normalized["physical_leaf"] = value->physicalLeaf;
                    if (logicalLeaf)
                        normalized["leaf"] = *logicalLeaf;
                } else {
                    normalized["parameter"] = value->name;
                    if (const std::optional<size_t> leaf = logicalProjection(logicalLayout, *source, *value, false))
                        normalized["leaf"] = static_cast<int64_t>(*leaf);
                }
                normalizedBindings.emplace_back(std::move(normalized));
                matched.insert(value->name);
            }
        } else if (*role == "retained_primal" || *role == "cotangent" ||
                   (*role == "tape" && !compiledKernelHasTapeAbi(compiledEntry))) {
            continue;
        } else if (*role == "tape") {
            matched.insert(parameter->str());
            llvm::json::Object normalized = *binding;
            normalizedBindings.emplace_back(std::move(normalized));
        } else {
            error = "compiled kernel ABI does not provide Program " + role->str() + " '" + source->str() + "'";
            return false;
        }
    }
    *requestBindings = std::move(normalizedBindings);
    for (const InterfaceValue &value : interface)
        if (!value.role.empty() && !matched.count(value.name)) {
            error = "compiled kernel ABI requires unmapped Program value '" + value.name + "'";
            return false;
        }

    llvm::json::Object *node = findProgramNode(execution, *graphName, *requestId);
    if (!node) {
        error = "Program implementation request does not identify an executable node";
        return false;
    }
    llvm::json::Array *nodeBindings = node->getArray("bindings");
    if (!nodeBindings) {
        error = "Program executable node has no value bindings";
        return false;
    }
    *nodeBindings = *requestBindings;

    std::set<int64_t> boundValues;
    for (const llvm::json::Value &normalizedValue : *nodeBindings)
        if (const llvm::json::Object *normalized = normalizedValue.getAsObject())
            if (const std::optional<int64_t> value = normalized->getInteger("value"))
                boundValues.insert(*value);
    if (llvm::json::Array *grid = node->getArray("grid"))
        for (const llvm::json::Value &componentValue : *grid)
            if (const llvm::json::Object *component = componentValue.getAsObject())
                if (const llvm::json::Object *control = component->getObject("control"))
                    if (std::optional<int64_t> value = control->getInteger("value"))
                        boundValues.insert(*value);
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
    rebuildProgramDependencies(execution, *graphName);
    rebuildProgramCaptures(execution);
    return true;
}

} // namespace vernon::compiler
