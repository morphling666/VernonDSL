#include "compiler_program_finalization.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

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
    }
}

llvm::json::Object *findNode(llvm::json::Object &execution, llvm::StringRef graphName, llvm::StringRef requestId) {
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

void rebuildDependencies(llvm::json::Object &execution, llvm::StringRef graphName) {
    llvm::json::Array *graphs = execution.getArray("graphs");
    if (!graphs)
        return;
    for (llvm::json::Value &graphValue : *graphs) {
        llvm::json::Object *graph = graphValue.getAsObject();
        if (!graph || graph->getString("name") != graphName)
            continue;
        std::map<int64_t, int64_t> producer;
        if (llvm::json::Array *nodes = graph->getArray("nodes"))
            for (llvm::json::Value &nodeValue : *nodes) {
                llvm::json::Object *node = nodeValue.getAsObject();
                if (!node)
                    continue;
                std::set<int64_t> dependencies;
                if (llvm::json::Array *operands = node->getArray("operands"))
                    for (const llvm::json::Value &operand : *operands)
                        if (std::optional<int64_t> value = operand.getAsInteger())
                            if (auto found = producer.find(*value); found != producer.end())
                                dependencies.insert(found->second);
                llvm::json::Array reflectedDependencies;
                for (int64_t dependency : dependencies)
                    reflectedDependencies.emplace_back(dependency);
                (*node)["dependencies"] = std::move(reflectedDependencies);
                const int64_t nodeId = node->getInteger("id").value_or(-1);
                if (llvm::json::Array *results = node->getArray("results"))
                    for (const llvm::json::Value &result : *results)
                        if (std::optional<int64_t> value = result.getAsInteger())
                            producer[*value] = nodeId;
            }
    }
}

const llvm::json::Object *valueById(const llvm::json::Array &values, int64_t id);

bool valueHasDynamicExtents(const llvm::json::Object &value) {
    const llvm::json::Array *shape = value.getArray("shape");
    if (!shape)
        return false;
    for (const llvm::json::Value &extentValue : *shape) {
        const std::optional<int64_t> extent = extentValue.getAsInteger();
        if (!extent || *extent <= 0)
            return true;
    }
    return false;
}

void considerCapture(int64_t value, const std::set<int64_t> &available, const std::set<int64_t> &forwardValues,
                     std::set<int64_t> &captures) {
    if (!available.count(value) && forwardValues.count(value))
        captures.insert(value);
}

void collectLikeCaptures(const llvm::json::Array &values, int64_t start, const std::set<int64_t> &available,
                         const std::set<int64_t> &forwardValues, std::set<int64_t> &captures) {
    std::set<int64_t> seen;
    int64_t current = start;
    while (seen.insert(current).second) {
        considerCapture(current, available, forwardValues, captures);
        const llvm::json::Object *value = valueById(values, current);
        if (!value)
            return;
        if (const std::optional<int64_t> like = value->getInteger("like"))
            current = *like;
        else
            return;
    }
}

void collectBackwardCaptures(const llvm::json::Object &execution, std::set<int64_t> &captures) {
    const llvm::json::Array *graphs = execution.getArray("graphs");
    const llvm::json::Array *values = execution.getArray("values");
    if (!graphs)
        return;
    std::set<int64_t> forwardValues;
    const llvm::json::Object *backward = nullptr;
    for (const llvm::json::Value &graphValue : *graphs) {
        const llvm::json::Object *graph = graphValue.getAsObject();
        if (!graph)
            continue;
        if (graph->getString("direction") == "forward") {
            if (const llvm::json::Array *arguments = graph->getArray("arguments"))
                for (const llvm::json::Value &argument : *arguments)
                    if (std::optional<int64_t> value = argument.getAsInteger())
                        forwardValues.insert(*value);
            if (const llvm::json::Array *nodes = graph->getArray("nodes"))
                for (const llvm::json::Value &nodeValue : *nodes)
                    if (const llvm::json::Object *node = nodeValue.getAsObject()) {
                        if (const llvm::json::Array *operands = node->getArray("operands"))
                            for (const llvm::json::Value &operand : *operands)
                                if (std::optional<int64_t> value = operand.getAsInteger())
                                    forwardValues.insert(*value);
                        if (const llvm::json::Array *results = node->getArray("results"))
                            for (const llvm::json::Value &result : *results)
                                if (std::optional<int64_t> value = result.getAsInteger())
                                    forwardValues.insert(*value);
                    }
        } else if (graph->getString("direction") == "backward") {
            backward = graph;
        }
    }
    if (!backward)
        return;
    std::set<int64_t> available;
    if (const llvm::json::Array *arguments = backward->getArray("arguments"))
        for (const llvm::json::Value &argument : *arguments)
            if (std::optional<int64_t> value = argument.getAsInteger())
                available.insert(*value);
    if (const llvm::json::Array *nodes = backward->getArray("nodes"))
        for (const llvm::json::Value &nodeValue : *nodes)
            if (const llvm::json::Object *node = nodeValue.getAsObject()) {
                if (const llvm::json::Array *operands = node->getArray("operands"))
                    for (const llvm::json::Value &operand : *operands)
                        if (std::optional<int64_t> value = operand.getAsInteger()) {
                            considerCapture(*value, available, forwardValues, captures);
                            const llvm::json::Object *row = values ? valueById(*values, *value) : nullptr;
                            if (row && valueHasDynamicExtents(*row))
                                collectLikeCaptures(*values, *value, available, forwardValues, captures);
                        }
                if (const llvm::json::Array *results = node->getArray("results"))
                    for (const llvm::json::Value &result : *results)
                        if (std::optional<int64_t> value = result.getAsInteger())
                            available.insert(*value);
            }
    if (!values)
        return;
    for (const llvm::json::Value &rowValue : *values) {
        const llvm::json::Object *row = rowValue.getAsObject();
        const std::optional<int64_t> id = row ? row->getInteger("id") : std::nullopt;
        if (!id || !row->getInteger("like") || forwardValues.count(*id) || !valueHasDynamicExtents(*row))
            continue;
        collectLikeCaptures(*values, *id, available, forwardValues, captures);
    }
}

void rebuildCaptures(llvm::json::Object &execution) {
    llvm::json::Object *signature = execution.getObject("signature");
    if (!signature)
        return;
    std::set<int64_t> captures;
    collectBackwardCaptures(execution, captures);
    llvm::json::Array reflected;
    for (int64_t capture : captures)
        reflected.emplace_back(capture);
    (*signature)["captures"] = std::move(reflected);
}

llvm::json::Value canonicalized(const llvm::json::Value &value) {
    if (const llvm::json::Object *object = value.getAsObject()) {
        std::vector<std::string> keys;
        keys.reserve(object->size());
        for (const auto &[key, unused] : *object) {
            (void)unused;
            keys.push_back(key.str());
        }
        std::sort(keys.begin(), keys.end());
        llvm::json::Object result;
        for (const std::string &key : keys)
            result[key] = canonicalized(*object->get(key));
        return result;
    }
    if (const llvm::json::Array *array = value.getAsArray()) {
        llvm::json::Array result;
        result.reserve(array->size());
        for (const llvm::json::Value &element : *array)
            result.emplace_back(canonicalized(element));
        return result;
    }
    return value;
}

llvm::json::Array copyArray(const llvm::json::Array &array) {
    llvm::json::Array result;
    result.reserve(array.size());
    for (const llvm::json::Value &value : array)
        result.emplace_back(value);
    return result;
}

bool canonicalValueIds(const llvm::json::Array &ids, llvm::json::Array &canonical, std::string &error) {
    std::set<int64_t> unique;
    for (const llvm::json::Value &value : ids) {
        const std::optional<int64_t> id = value.getAsInteger();
        if (!id) {
            error = "canonical compute node has invalid operand or result ids";
            return false;
        }
        if (!unique.insert(*id).second) {
            error = "canonical compute node has duplicate operand or result ids";
            return false;
        }
    }
    canonical.clear();
    canonical.reserve(unique.size());
    for (int64_t id : unique)
        canonical.emplace_back(id);
    return true;
}

llvm::json::Object copyObject(const llvm::json::Object &object) {
    llvm::json::Object result;
    for (const auto &[key, value] : object)
        result[key] = value;
    return result;
}

llvm::json::Array attributeCellShape(const llvm::json::Object &row) {
    if (const llvm::json::Array *shape = row.getArray("shape"))
        return copyArray(*shape);
    const llvm::json::Object *valueLayout = row.getObject("value_layout");
    const llvm::json::Array *leaves = valueLayout ? valueLayout->getArray("leaves") : nullptr;
    if (!leaves || leaves->size() != 1)
        return {};
    const llvm::json::Object *leaf = (*leaves)[0].getAsObject();
    const llvm::json::Array *path = leaf ? leaf->getArray("path") : nullptr;
    const llvm::json::Array *leafShape = leaf ? leaf->getArray("shape") : nullptr;
    if (!leafShape || (path && !path->empty()))
        return {};
    return copyArray(*leafShape);
}

std::string canonicalJson(const llvm::json::Value &value) {
    std::string result;
    llvm::raw_string_ostream stream(result);
    stream << canonicalized(value);
    return result;
}

std::string sha256(const llvm::json::Value &value) {
    const std::string bytes = canonicalJson(value);
    llvm::SHA256 hash;
    hash.update(bytes);
    return llvm::toHex(hash.final(), true);
}

bool validAccess(llvm::StringRef access) { return access == "read" || access == "write" || access == "read_write"; }

// Program operand_accesses is the required edge. Kernel TensorView access is the
// provided capability. A write dest may RMW (read_write) under accumulation
// ownership; that is not a Program-level access upgrade.
bool resourceAccessSatisfies(llvm::StringRef physical, llvm::StringRef logical) {
    if (logical == "read")
        return physical == "read" || physical == "read_write";
    if (logical == "write")
        return physical == "write" || physical == "read_write";
    return logical == "read_write" && physical == "read_write";
}

bool kernelHiddenBuiltin(llvm::StringRef builtin) {
    return builtin == "global_invocation_id" || builtin == "local_invocation_id" || builtin == "workgroup_id";
}

const llvm::json::Object *valueById(const llvm::json::Array &values, int64_t id) {
    for (const llvm::json::Value &value : values)
        if (const llvm::json::Object *object = value.getAsObject(); object && object->getInteger("id") == id)
            return object;
    return nullptr;
}

bool identicalConcreteShape(const llvm::json::Array *left, const llvm::json::Array *right) {
    if (!left || !right || left->size() != right->size())
        return false;
    for (size_t index = 0; index < left->size(); ++index) {
        const std::optional<int64_t> expected = (*left)[index].getAsInteger();
        const std::optional<int64_t> actual = (*right)[index].getAsInteger();
        if (!expected || !actual)
            return false;
        if (*expected > 0 && *actual > 0 && *expected != *actual)
            return false;
        if ((*expected > 0) != (*actual > 0))
            return false;
    }
    return true;
}

bool compatibleAbiShape(const llvm::json::Array *logical, const llvm::json::Array *physical) {
    const size_t logicalRank = logical ? logical->size() : 0;
    const size_t physicalRank = physical ? physical->size() : 0;
    if (logicalRank != physicalRank)
        return false;
    if (!logicalRank)
        return true;
    for (size_t index = 0; index < logicalRank; ++index) {
        const std::optional<int64_t> concrete = (*logical)[index].getAsInteger();
        const std::optional<int64_t> declared = (*physical)[index].getAsInteger();
        if (!concrete || !declared)
            return false;
        if (*concrete > 0 && *declared > 0 && *concrete != *declared)
            return false;
    }
    return true;
}

std::optional<std::string> reflectedAccess(const llvm::json::Object &row) {
    if (std::optional<llvm::StringRef> access = row.getString("vernon.access"); access && validAccess(*access))
        return access->str();
    if (std::optional<llvm::StringRef> access = row.getString("access"); access && validAccess(*access))
        return access->str();
    return std::nullopt;
}

const llvm::json::Object *endpointLayout(const llvm::json::Object &row, bool resource) {
    if (resource)
        if (const llvm::json::Object *layout = row.getObject("element_layout"))
            return layout;
    return row.getObject("value_layout");
}

bool isTextureType(llvm::StringRef type) { return type.starts_with("!vernon.texture<"); }

bool isSamplerType(llvm::StringRef type) { return type.starts_with("!vernon.sampler"); }

bool isAdTapeType(llvm::StringRef type) { return type == "!vernon.ad_tape" || type.starts_with("!vernon.ad_tape<"); }

bool isTensorViewType(llvm::StringRef type) { return type.starts_with("!vernon.tensor_view<"); }

bool kernelTapeBuiltin(llvm::StringRef builtin) {
    return builtin == "ad_tape_allocator" || builtin == "ad_tape_root_region";
}

bool isTapeCarrierRole(llvm::StringRef role) {
    return role == "tape" || role == "replay_segment" || role == "replay_status";
}

bool matchesAutodiffBinding(llvm::StringRef logicalRole, llvm::StringRef logicalSource,
                            const InterfaceValue &physical) {
    if (logicalRole == "tape")
        return isTapeCarrierRole(physical.role);
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
            if ((builtin && kernelTapeBuiltin(*builtin)) || (role && isTapeCarrierRole(*role)))
                return true;
        }
        return false;
    };
    return scan(compiledEntry.getArray("arguments")) || scan(compiledEntry.getArray("results"));
}

std::vector<std::string> quotedTypeFields(llvm::StringRef type) {
    std::vector<std::string> fields;
    while (true) {
        const size_t begin = type.find('"');
        if (begin == llvm::StringRef::npos)
            return fields;
        type = type.drop_front(begin + 1);
        const size_t end = type.find('"');
        if (end == llvm::StringRef::npos)
            return {};
        fields.push_back(type.take_front(end).str());
        type = type.drop_front(end + 1);
    }
}

llvm::json::Object manifestLayout(const llvm::json::Object &layout, llvm::StringRef scope) {
    llvm::json::Object result;
    for (const auto &[key, value] : layout)
        if (key != "logical_type" && key != "struct_name")
            result[key] = value;
    result["scope"] = scope.str();
    return result;
}

bool checkedByteLength(const llvm::json::Object &layout, const llvm::json::Array &shape, uint64_t &length) {
    std::optional<int64_t> byteSize = layout.getInteger("byte_size");
    if (!byteSize || *byteSize <= 0)
        return false;
    length = static_cast<uint64_t>(*byteSize);
    for (const llvm::json::Value &extentValue : shape) {
        std::optional<int64_t> extent = extentValue.getAsInteger();
        if (!extent || *extent <= 0 || length > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(*extent))
            return false;
        length *= static_cast<uint64_t>(*extent);
    }
    return length <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

bool dynamicExtent(const llvm::json::Value &extentValue) {
    const std::optional<int64_t> extent = extentValue.getAsInteger();
    return !extent || *extent <= 0;
}

llvm::json::Object valueCarrier(llvm::StringRef tag, int64_t slot, const llvm::json::Object &layout) {
    return llvm::json::Object{{"tag", tag.str()},
                              {"slot", slot},
                              {"byte_offset", int64_t{0}},
                              {"byte_size", layout.getInteger("byte_size").value_or(0)},
                              {"alignment", layout.getInteger("alignment").value_or(0)}};
}

struct LogicalResource {
    int64_t before{};
    std::optional<int64_t> after;
    std::string access;
    int64_t storage{-1};
};

} // namespace

bool compatibleProgramBindingShape(llvm::StringRef role, llvm::StringRef carrier, const llvm::json::Array *logical,
                                   const llvm::json::Array *physical) {
    if (compatibleAbiShape(logical, physical))
        return true;
    if (role != "cotangent" || carrier != "invocation_linear" || !logical || !physical ||
        physical->size() != logical->size() + 1)
        return false;
    for (size_t index = 0; index < logical->size(); ++index) {
        const std::optional<int64_t> expected = (*logical)[index].getAsInteger();
        const std::optional<int64_t> actual = (*physical)[index + 1].getAsInteger();
        if (!expected || !actual || (*expected > 0 && *actual > 0 && *expected != *actual))
            return false;
    }
    return true;
}

std::optional<size_t> resolveProgramValueLeafIndex(const llvm::json::Object &layout, llvm::StringRef source,
                                                   llvm::StringRef parameter) {
    const llvm::json::Array *leaves = layout.getArray("leaves");
    if (!leaves || source.empty())
        return std::nullopt;
    for (auto [leafIndex, leafValue] : llvm::enumerate(*leaves)) {
        const llvm::json::Object *leaf = leafValue.getAsObject();
        const llvm::json::Array *path = leaf ? leaf->getArray("path") : nullptr;
        if (!path)
            continue;
        if (path->empty()) {
            if (parameter == source)
                return leafIndex;
            continue;
        }
        std::string canonical = source.str();
        for (const llvm::json::Value &component : *path) {
            canonical.push_back('.');
            if (std::optional<llvm::StringRef> field = component.getAsString())
                canonical += *field;
            else if (std::optional<int64_t> index = component.getAsInteger())
                canonical += std::to_string(*index);
            else
                return std::nullopt;
        }
        if (parameter == canonical)
            return leafIndex;
    }
    return std::nullopt;
}

static void annotateProgramValueLeafProjections(const llvm::json::Object &execution, llvm::json::Array &bindings) {
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

static void copyProgramValueLeafProjections(const llvm::json::Array &source, llvm::json::Array &destination) {
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

    std::set<std::string> omitted;
    std::set<std::string> matched;
    std::map<std::string, std::vector<std::string>> aliases;
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
            if (!role || role->empty())
                return true;
            return value.role == *role;
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
                aliases.emplace(parameter->str(), std::vector<std::string>{canonical->name});
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
            aliases.emplace(parameter->str(), names);
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
            // Kernel structured VJP may drop an output that Program still seeded
            // because that seed does not depend on this node's wrt. Extra retained
            // primals are the same class. Inactive results are not zero-seeded.
            // Elementwise / validNoTape kernels have no tape builtins; unmatched
            // Program tape bindings are not a second reverse-construction authority.
            omitted.insert(parameter->str());
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

    llvm::json::Object *node = findNode(execution, *graphName, *requestId);
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
        if (omitted.count(parameter->str())) {
            nodeBindings->erase(nodeBindings->begin() + bindingIndex);
            continue;
        }
        const std::optional<int64_t> value = object->getInteger("value");
        if (auto alias = aliases.find(parameter->str()); alias != aliases.end()) {
            (*object)["parameter"] = alias->second.front();
            for (size_t index = 1; index < alias->second.size(); ++index) {
                llvm::json::Object expanded{{"parameter", alias->second[index]}};
                if (value)
                    expanded["value"] = *value;
                nodeBindings->insert(nodeBindings->begin() + bindingIndex + index,
                                     llvm::json::Value(std::move(expanded)));
            }
            bindingIndex += alias->second.size();
        } else {
            ++bindingIndex;
        }
        if (value)
            boundValues.insert(*value);
    }
    copyProgramValueLeafProjections(*requestBindings, *nodeBindings);
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
        for (auto resource = resources->begin(); resource != resources->end();)
            if (const llvm::json::Object *object = resource->getAsObject();
                object && object->getInteger("value") && !liveValues.count(*object->getInteger("value")))
                resource = resources->erase(resource);
            else
                ++resource;
    }
    rebuildDependencies(execution, *graphName);
    rebuildCaptures(execution);
    return true;
}

const llvm::json::Object *selectedTransportPlan(const llvm::json::Object &row) {
    const llvm::json::Object *layouts = row.getObject("physical_layouts");
    if (!layouts)
        return nullptr;
    const llvm::json::Object *fallback = nullptr;
    for (const auto &[name, value] : *layouts) {
        const llvm::json::Object *plan = value.getAsObject();
        if (!plan)
            continue;
        const llvm::StringRef kind = plan->getString("kind").value_or("");
        if (kind != "byte_transport" && kind != "native_uniform" && kind != "cpu_call" && kind != "kernel_parameter")
            continue;
        fallback = plan;
        if (std::optional<llvm::StringRef> profile = plan->getString("profile");
            profile && *profile == llvm::StringRef(name))
            return plan;
    }
    return fallback;
}

llvm::json::Object compiledEndpointAbi(const llvm::json::Object &row, llvm::StringRef module, int64_t index) {
    llvm::json::Object compiled{{"module", module.str()}, {"index", index}};
    if (const std::optional<llvm::StringRef> builtin =
            row.getString("vernon.builtin") ? row.getString("vernon.builtin") : row.getString("builtin"))
        compiled["builtin"] = builtin->str();
    if (std::optional<llvm::StringRef> transport = row.getString("value_transport"))
        compiled["value_transport"] = transport->str();
    if (std::optional<int64_t> set = row.getInteger("vernon.set"))
        compiled["set"] = *set;
    if (std::optional<int64_t> binding = row.getInteger("vernon.binding"))
        compiled["binding"] = *binding;
    if (const llvm::json::Array *sampled = row.getArray("sampled_image_bindings"))
        compiled["sampled_image_bindings"] = copyArray(*sampled);
    if (const llvm::json::Object *element = row.getObject("element_layout"))
        compiled["element_layout"] = copyObject(*element);
    if (const llvm::json::Object *plan = selectedTransportPlan(row))
        compiled["interface_plan"] = copyObject(*plan);
    if (const llvm::json::Object *layouts = row.getObject("physical_layouts"))
        if (const llvm::json::Object *host = layouts->getObject("host_value"))
            if (std::optional<int64_t> offset = host->getInteger("frame_offset"); offset && *offset >= 0)
                compiled["packed_frame_offset"] = *offset;
    return compiled;
}

struct CanonicalGraph {
    const llvm::json::Object *graph = nullptr;
    const llvm::json::Array *nodes = nullptr;
    const llvm::json::Array *argumentIds = nullptr;
    const llvm::json::Array *resultIds = nullptr;
    std::string name;
    std::string direction;
};

struct ArgumentSlot {
    std::string graph;
    int64_t slot = 0;
};

struct ValueProducer {
    std::string graph;
    int64_t node = -1;
};

int64_t storageRootOf(const std::map<int64_t, int64_t> &parent, int64_t value) {
    int64_t root = value;
    while (parent.count(root) && parent.at(root) != root)
        root = parent.at(root);
    return root;
}

bool thisGraphArgument(int64_t value, llvm::StringRef allocGraph,
                       const std::map<int64_t, ArgumentSlot> &argumentSlots) {
    const auto argument = argumentSlots.find(value);
    return argument != argumentSlots.end() && argument->second.graph == allocGraph;
}

bool entryAvailable(int64_t value, llvm::StringRef allocGraph, bool captureLegal,
                    const std::map<int64_t, ArgumentSlot> &argumentSlots, const std::set<int64_t> &captures) {
    if (thisGraphArgument(value, allocGraph, argumentSlots))
        return true;
    return captureLegal && captures.count(value) != 0;
}

int64_t walkLikeSource(const llvm::json::Array &values, int64_t root, llvm::StringRef allocGraph, bool captureLegal,
                       const std::map<int64_t, ArgumentSlot> &argumentSlots, const std::set<int64_t> &captures,
                       const std::map<int64_t, ValueProducer> &producerByValue,
                       const std::map<int64_t, int64_t> &parent, std::string &error) {
    std::set<int64_t> seen;
    int64_t current = root;
    while (true) {
        if (!seen.insert(current).second) {
            error = "owned compute Storage like-source cycle";
            return -1;
        }
        const llvm::json::Object *value = valueById(values, current);
        if (!value) {
            error = "owned compute Storage like-source is unknown";
            return -1;
        }
        if (const std::optional<int64_t> like = value->getInteger("like")) {
            current = *like;
            continue;
        }
        if (entryAvailable(current, allocGraph, captureLegal, argumentSlots, captures))
            return current;
        if (!producerByValue.count(current)) {
            error = "owned compute Storage like-source is not a ControlValueRef";
            return -1;
        }
        const llvm::json::Object *allocated = valueById(values, storageRootOf(parent, current));
        if (allocated)
            if (const std::optional<int64_t> like = allocated->getInteger("like"); like && *like != current) {
                current = *like;
                continue;
            }
        error = "owned compute Storage like-source has NodeResultOrigin";
        return -1;
    }
}

std::optional<llvm::json::Object> controlValueRef(int64_t likeId, llvm::StringRef allocGraph, bool captureLegal,
                                                  const std::map<int64_t, ArgumentSlot> &argumentSlots,
                                                  const std::set<int64_t> &captures,
                                                  const std::map<int64_t, ValueProducer> &producerByValue,
                                                  std::string &error) {
    if (thisGraphArgument(likeId, allocGraph, argumentSlots)) {
        const auto argument = argumentSlots.find(likeId);
        return llvm::json::Object{{"argument", argument->second.slot}};
    }
    if (captureLegal && captures.count(likeId))
        return llvm::json::Object{{"capture", likeId}};
    if (producerByValue.count(likeId)) {
        error = "owned compute Storage like-source has NodeResultOrigin";
        return std::nullopt;
    }
    error = "owned compute Storage like-source is not a ControlValueRef";
    return std::nullopt;
}

llvm::json::Value dimensionExtent(const llvm::json::Object &reference, int64_t axis) {
    return llvm::json::Object{{"dimension", llvm::json::Object{{"control", copyObject(reference)}, {"axis", axis}}}};
}

bool ownedDynamicExtents(const llvm::json::Array &rawValues, int64_t root, llvm::StringRef allocGraph,
                         bool captureLegal, const llvm::json::Array &shape,
                         const std::map<int64_t, ArgumentSlot> &argumentSlots, const std::set<int64_t> &captures,
                         const std::map<int64_t, ValueProducer> &producerByValue,
                         const std::map<int64_t, int64_t> &parent, llvm::json::Array &extents, std::string &error) {
    const int64_t like = walkLikeSource(rawValues, root, allocGraph, captureLegal, argumentSlots, captures,
                                        producerByValue, parent, error);
    if (like < 0)
        return false;
    const std::optional<llvm::json::Object> reference =
        controlValueRef(like, allocGraph, captureLegal, argumentSlots, captures, producerByValue, error);
    if (!reference)
        return false;
    const llvm::json::Object *likeValue = valueById(rawValues, like);
    const llvm::json::Array *likeShape = likeValue ? likeValue->getArray("shape") : nullptr;
    if (!likeShape || likeShape->size() != shape.size()) {
        error = "owned compute Storage like-source rank does not match the allocated value";
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        const llvm::json::Value &extentValue = shape[axis];
        if (!dynamicExtent(extentValue)) {
            extents.emplace_back(*extentValue.getAsInteger());
            continue;
        }
        if (!dynamicExtent((*likeShape)[axis])) {
            error = "owned compute Storage dyn axis has a static like-source extent";
            return false;
        }
        extents.emplace_back(dimensionExtent(*reference, static_cast<int64_t>(axis)));
    }
    return true;
}

bool selectCanonicalGraphs(const llvm::json::Array &rawGraphs, std::vector<CanonicalGraph> &graphs,
                           std::string &error) {
    const llvm::json::Object *forward = nullptr;
    const llvm::json::Object *backward = nullptr;
    for (const llvm::json::Value &value : rawGraphs) {
        const llvm::json::Object *graph = value.getAsObject();
        const std::optional<llvm::StringRef> direction = graph ? graph->getString("direction") : std::nullopt;
        if (!graph || !direction) {
            error = "canonical compute finalization requires named forward/backward graphs";
            return false;
        }
        if (*direction == "forward") {
            if (forward) {
                error = "canonical compute finalization requires exactly one forward graph";
                return false;
            }
            forward = graph;
        } else if (*direction == "backward") {
            if (backward) {
                error = "canonical compute finalization requires at most one backward graph";
                return false;
            }
            backward = graph;
        } else {
            error = "canonical compute finalization received an unknown graph direction";
            return false;
        }
    }
    if (!forward) {
        error = "canonical compute finalization requires exactly one forward graph and signature";
        return false;
    }
    const auto push = [&](const llvm::json::Object *graph) {
        CanonicalGraph view;
        view.graph = graph;
        view.direction = graph->getString("direction")->str();
        view.name = graph->getString("name").value_or(view.direction).str();
        view.nodes = graph->getArray("nodes");
        view.argumentIds = graph->getArray("arguments");
        view.resultIds = graph->getArray("results");
        if (view.name.empty() || !view.nodes || view.nodes->empty() || !view.argumentIds || !view.resultIds) {
            error = "canonical compute finalization requires one non-empty " + view.direction + " compute graph";
            return false;
        }
        graphs.push_back(std::move(view));
        return true;
    };
    return push(forward) && (!backward || push(backward));
}

bool buildCanonicalComputeProgram(const llvm::json::Object &execution,
                                  const std::vector<CanonicalComputeStage> &compiledStages, llvm::json::Object &program,
                                  llvm::json::Object &stageContracts, llvm::json::Object &targetImplementations,
                                  std::string &error) {
    const llvm::json::Array *rawValues = execution.getArray("values");
    const llvm::json::Array *rawGraphs = execution.getArray("graphs");
    const llvm::json::Object *rawSignature = execution.getObject("signature");
    std::vector<CanonicalGraph> selectedGraphs;
    if (!rawValues || !rawGraphs || !rawSignature || !selectCanonicalGraphs(*rawGraphs, selectedGraphs, error)) {
        if (error.empty())
            error = "canonical compute finalization requires exactly one forward graph and signature";
        return false;
    }
    size_t nodeCount = 0;
    for (const CanonicalGraph &view : selectedGraphs)
        nodeCount += view.nodes->size();
    if (nodeCount != compiledStages.size()) {
        error = "compiled stages do not exactly cover canonical compute nodes";
        return false;
    }

    std::map<int64_t, ArgumentSlot> argumentSlots;
    std::map<std::string, std::string> graphDirections;
    for (const CanonicalGraph &view : selectedGraphs) {
        graphDirections[view.name] = view.direction;
        for (size_t slot = 0; slot < view.argumentIds->size(); ++slot) {
            std::optional<int64_t> id = (*view.argumentIds)[slot].getAsInteger();
            if (!id || !argumentSlots.emplace(*id, ArgumentSlot{view.name, static_cast<int64_t>(slot)}).second) {
                error = "canonical compute graph has invalid argument ids";
                return false;
            }
        }
    }
    std::set<int64_t> capturedValues;
    collectBackwardCaptures(execution, capturedValues);
    std::map<int64_t, ValueProducer> producerByValue;
    std::map<int64_t, std::string> allocationGraph;
    std::map<std::string, const llvm::json::Object *> nodeByStage;
    std::map<std::string, const CanonicalComputeStage *> compiledByRequest;
    std::map<std::string, std::map<int64_t, LogicalResource>> resourcesByStage;
    std::set<int64_t> resourceVersions;
    std::map<int64_t, int64_t> parent;
    for (const CanonicalComputeStage &compiled : compiledStages)
        if (compiled.requestId.empty() || !compiledByRequest.emplace(compiled.requestId, &compiled).second) {
            error = "canonical compute stages have invalid or duplicate logical request ids";
            return false;
        }
    for (const CanonicalGraph &view : selectedGraphs) {
        for (const llvm::json::Value &nodeValue : *view.nodes) {
            const llvm::json::Object *node = nodeValue.getAsObject();
            const std::optional<llvm::StringRef> requestId = node ? node->getString("stage") : std::nullopt;
            const llvm::json::Array *nodeResults = node ? node->getArray("results") : nullptr;
            const llvm::json::Array *rawResources = node ? node->getArray("resources") : nullptr;
            const std::optional<llvm::StringRef> kind = node ? node->getString("kind") : std::nullopt;
            if (!node || (kind != "compute" && kind != "render") || !requestId ||
                !compiledByRequest.count(requestId->str()) || !node->getArray("operands") || !nodeResults ||
                !node->getArray("bindings") || !rawResources || !node->getArray("grid") ||
                node->getArray("grid")->size() != 3 || !nodeByStage.emplace(requestId->str(), node).second) {
                error = "canonical compute node has incomplete or duplicate logical execution metadata";
                return false;
            }
            if (const llvm::json::Array *operands = node->getArray("operands")) {
                for (const llvm::json::Value &operand : *operands)
                    if (std::optional<int64_t> id = operand.getAsInteger();
                        id && !argumentSlots.count(*id) && !producerByValue.count(*id))
                        allocationGraph.try_emplace(*id, view.name);
            }
            for (const llvm::json::Value &value : *nodeResults) {
                std::optional<int64_t> id = value.getAsInteger();
                if (!id || !valueById(*rawValues, *id) ||
                    !producerByValue.emplace(*id, ValueProducer{view.name, node->getInteger("id").value_or(-1)})
                         .second) {
                    error = "canonical compute graph has invalid or multiply-produced result ids";
                    return false;
                }
            }
            std::map<int64_t, LogicalResource> &resources = resourcesByStage[requestId->str()];
            for (const llvm::json::Value &resourceValue : *rawResources) {
                const llvm::json::Object *resource = resourceValue.getAsObject();
                const std::optional<int64_t> value = resource ? resource->getInteger("value") : std::nullopt;
                const std::optional<llvm::StringRef> access = resource ? resource->getString("access") : std::nullopt;
                if (!value || !access || !validAccess(*access) || resources.count(*value)) {
                    error = "canonical compute node has invalid or duplicate resource accesses";
                    return false;
                }
                const llvm::json::Object *logicalValue = valueById(*rawValues, *value);
                const llvm::StringRef type = logicalValue ? logicalValue->getString("type").value_or("") : "";
                const bool actualResource =
                    isTensorViewType(type) || isTextureType(type) || isSamplerType(type) || isAdTapeType(type);
                if (!actualResource)
                    continue;
                LogicalResource logical{*value, resource->getInteger("after"), access->str(), -1};
                const bool writable = logical.access == "write" || logical.access == "read_write";
                if ((!writable && logical.after) || (writable && !logical.after && !producerByValue.count(*value))) {
                    error = "canonical compute writable resources require exact before/after Storage SSA";
                    return false;
                }
                parent.emplace(*value, *value);
                resourceVersions.insert(*value);
                if (logical.after) {
                    parent.emplace(*logical.after, *logical.after);
                    const int64_t beforeRoot = storageRootOf(parent, *value);
                    const int64_t afterRoot = storageRootOf(parent, *logical.after);
                    parent[afterRoot] = beforeRoot;
                    resourceVersions.insert(*logical.after);
                }
                resources.emplace(*value, std::move(logical));
            }
        }
    }
    bool assignedAllocGraph = true;
    while (assignedAllocGraph) {
        assignedAllocGraph = false;
        for (const llvm::json::Value &rowValue : *rawValues) {
            const llvm::json::Object *value = rowValue.getAsObject();
            const std::optional<int64_t> id = value ? value->getInteger("id") : std::nullopt;
            const std::optional<int64_t> like = value ? value->getInteger("like") : std::nullopt;
            if (!id || !like)
                continue;
            const auto found = allocationGraph.find(*id);
            if (found == allocationGraph.end() || argumentSlots.count(*like) || producerByValue.count(*like))
                continue;
            assignedAllocGraph |= allocationGraph.try_emplace(*like, found->second).second;
        }
    }
    for (const auto &[id, graphName] : allocationGraph) {
        if (graphDirections[graphName] != "backward")
            continue;
        const llvm::json::Object *allocated = valueById(*rawValues, id);
        const llvm::json::Array *shape = allocated ? allocated->getArray("shape") : nullptr;
        bool needsRuntimeExtents = false;
        if (shape)
            for (const llvm::json::Value &extent : *shape)
                needsRuntimeExtents |= dynamicExtent(extent);
        if (!needsRuntimeExtents)
            continue;
        std::string walkError;
        const int64_t like = walkLikeSource(*rawValues, id, graphName, true, argumentSlots, capturedValues,
                                            producerByValue, parent, walkError);
        if (like >= 0 && !thisGraphArgument(like, graphName, argumentSlots))
            capturedValues.insert(like);
    }

    llvm::json::Array storages;
    std::map<int64_t, int64_t> storageByRoot;
    std::map<int64_t, bool> mutableByRoot;
    for (const auto &[unusedStage, resources] : resourcesByStage) {
        (void)unusedStage;
        for (const auto &[before, resource] : resources) {
            const int64_t root = storageRootOf(parent, before);
            mutableByRoot[root] |= resource.access != "read";
        }
    }
    for (const auto &[version, unusedParent] : parent) {
        (void)unusedParent;
        const int64_t root = storageRootOf(parent, version);
        if (storageByRoot.count(root))
            continue;
        const llvm::json::Object *value = valueById(*rawValues, root);
        const llvm::json::Object *layout = value ? value->getObject("value_layout") : nullptr;
        const llvm::json::Array *shape = value ? value->getArray("shape") : nullptr;
        const llvm::StringRef type = value ? value->getString("type").value_or("") : "";
        if (!value || !shape || type.empty()) {
            error = "canonical compute resource has no valid static type or shape";
            return false;
        }
        const int64_t storage = static_cast<int64_t>(storages.size());
        storageByRoot[root] = storage;
        const auto foundGraph = allocationGraph.find(root);
        const llvm::StringRef allocGraph = foundGraph != allocationGraph.end()
                                               ? llvm::StringRef(foundGraph->second)
                                               : llvm::StringRef(selectedGraphs.front().name);
        const auto direction = graphDirections.find(allocGraph.str());
        const bool captureLegal = direction != graphDirections.end() && direction->second == "backward";
        llvm::json::Object descriptor;
        if (isTextureType(type)) {
            const std::vector<std::string> fields = quotedTypeFields(type);
            if (fields.size() < 2 || shape->size() > 3) {
                error = "canonical compute texture has no valid dimension, format, or rank";
                return false;
            }
            size_t spatialRank = shape->size();
            if (spatialRank == 0) {
                if (fields[0] == "1d")
                    spatialRank = 1;
                else if (fields[0] == "3d")
                    spatialRank = 3;
                else
                    spatialRank = 2;
            }
            const bool borrowed = argumentSlots.count(root);
            llvm::json::Array extent;
            llvm::json::Array dynamicExtents;
            bool hasDynamic = false;
            for (size_t axis = 0; axis < spatialRank; ++axis)
                if (axis < shape->size() && dynamicExtent((*shape)[axis]))
                    hasDynamic = true;
            if (hasDynamic && !borrowed &&
                !ownedDynamicExtents(*rawValues, root, allocGraph, captureLegal, *shape, argumentSlots, capturedValues,
                                     producerByValue, parent, dynamicExtents, error))
                return false;
            for (size_t axis = 0; axis < 3; ++axis) {
                const size_t sourceAxis = spatialRank > axis ? spatialRank - axis - 1 : spatialRank;
                if (sourceAxis >= spatialRank) {
                    extent.emplace_back(int64_t{1});
                    continue;
                }
                const bool hasExtent = sourceAxis < shape->size();
                const llvm::json::Value *planned = hasExtent ? &(*shape)[sourceAxis] : nullptr;
                if (planned && !dynamicExtent(*planned)) {
                    extent.emplace_back(*planned->getAsInteger());
                } else if (borrowed) {
                    extent.emplace_back(int64_t{0});
                } else if (sourceAxis < dynamicExtents.size()) {
                    extent.emplace_back(dynamicExtents[sourceAxis]);
                } else {
                    error = "owned compute texture requires a concrete extent";
                    return false;
                }
            }
            descriptor = llvm::json::Object{{"tag", "image"},
                                            {"dimension", fields[0]},
                                            {"extent", std::move(extent)},
                                            {"format", fields[1]},
                                            {"sample_count", int64_t{1}},
                                            {"mip_levels", int64_t{1}},
                                            {"array_layers", int64_t{1}},
                                            {"aspects", llvm::json::Array{"color"}},
                                            {"usage", llvm::json::Array{"storage"}}};
        } else if (isSamplerType(type) || isAdTapeType(type)) {
            descriptor = llvm::json::Object{
                {"tag", "opaque"},
                {"contract_hash", sha256(llvm::json::Value(llvm::json::Object{{"type", type.str()}}))}};
        } else {
            uint64_t byteLength = 0;
            const std::optional<int64_t> alignment = layout ? layout->getInteger("alignment") : std::nullopt;
            const std::optional<int64_t> byteSize = layout ? layout->getInteger("byte_size") : std::nullopt;
            const bool borrowed = argumentSlots.count(root);
            if (!layout || !alignment || *alignment <= 0 || !byteSize || *byteSize <= 0) {
                error = "canonical compute buffer has no valid static layout";
                return false;
            }
            llvm::json::Value byteLengthValue = nullptr;
            if (checkedByteLength(*layout, *shape, byteLength)) {
                byteLengthValue = static_cast<int64_t>(byteLength);
            } else if (borrowed) {
                byteLengthValue = int64_t{0};
            } else {
                llvm::json::Array extents;
                if (!ownedDynamicExtents(*rawValues, root, allocGraph, captureLegal, *shape, argumentSlots,
                                         capturedValues, producerByValue, parent, extents, error))
                    return false;
                byteLengthValue = std::move(extents);
            }
            descriptor = llvm::json::Object{{"tag", "buffer"},
                                            {"byte_length", std::move(byteLengthValue)},
                                            {"alignment", *alignment},
                                            {"memory", "device"},
                                            {"usage", llvm::json::Array{"storage"}}};
        }
        storages.emplace_back(llvm::json::Object{
            {"id", storage},
            {"name", value->getString("name").value_or("value").str()},
            {"initial_value", root},
            {"ownership", argumentSlots.count(root) ? "borrowed" : "owned"},
            {"lifetime", isAdTapeType(type) ? "pullback" : "invocation"},
            {"mutability", mutableByRoot[root] ? "mutable" : "read_only"},
            {"descriptor", std::move(descriptor)},
        });
    }

    std::map<int64_t, int64_t> storageByValue;
    for (const auto &[version, unusedParent] : parent) {
        (void)unusedParent;
        const int64_t root = storageRootOf(parent, version);
        const llvm::json::Object *initial = valueById(*rawValues, root);
        const llvm::json::Object *current = valueById(*rawValues, version);
        const llvm::json::Object *initialLayout = initial ? initial->getObject("value_layout") : nullptr;
        const llvm::json::Object *currentLayout = current ? current->getObject("value_layout") : nullptr;
        const llvm::json::Array *initialShape = initial ? initial->getArray("shape") : nullptr;
        const llvm::json::Array *currentShape = current ? current->getArray("shape") : nullptr;
        const llvm::StringRef initialType = initial ? initial->getString("type").value_or("") : "";
        const llvm::StringRef currentType = current ? current->getString("type").value_or("") : "";
        const bool opaqueResource =
            isTextureType(initialType) || isSamplerType(initialType) || isAdTapeType(initialType);
        if (initialType.empty() || initialType != currentType ||
            (!opaqueResource && (!initialLayout || !currentLayout ||
                                 initialLayout->getString("layout_hash") != currentLayout->getString("layout_hash"))) ||
            !identicalConcreteShape(initialShape, currentShape) ||
            !identicalConcreteShape(currentShape, initialShape)) {
            error = "canonical compute Storage SSA versions disagree on layout or concrete shape";
            return false;
        }
        storageByValue[version] = storageByRoot[root];
    }
    for (auto &[unusedStage, resources] : resourcesByStage) {
        (void)unusedStage;
        for (auto &[before, resource] : resources)
            resource.storage = storageByValue[before];
    }
    llvm::json::Array values;
    for (size_t expectedId = 0; expectedId < rawValues->size(); ++expectedId) {
        const llvm::json::Object *value = valueById(*rawValues, static_cast<int64_t>(expectedId));
        const llvm::json::Object *layout = value ? value->getObject("value_layout") : nullptr;
        const llvm::json::Array *shape = value ? value->getArray("shape") : nullptr;
        const std::optional<llvm::StringRef> type = value ? value->getString("type") : std::nullopt;
        if (!value || !shape || !type) {
            error = "canonical compute values must have contiguous ids, types, and shapes";
            return false;
        }
        llvm::json::Object origin;
        if (auto argument = argumentSlots.find(static_cast<int64_t>(expectedId)); argument != argumentSlots.end())
            origin = llvm::json::Object{
                {"tag", "argument"}, {"graph", argument->second.graph}, {"slot", argument->second.slot}};
        else if (auto producer = producerByValue.find(static_cast<int64_t>(expectedId));
                 producer != producerByValue.end())
            origin = llvm::json::Object{
                {"tag", "node_result"}, {"graph", producer->second.graph}, {"node", producer->second.node}};
        else if (auto found = allocationGraph.find(static_cast<int64_t>(expectedId)); found != allocationGraph.end())
            origin = llvm::json::Object{{"tag", "allocation"}, {"graph", found->second}};
        else {
            error = "canonical compute value has no origin";
            return false;
        }
        const bool resource = resourceVersions.count(static_cast<int64_t>(expectedId));
        const bool opaqueResource = resource && (isTextureType(*type) || isSamplerType(*type) || isAdTapeType(*type));
        if (!opaqueResource && !layout) {
            error = "canonical compute byte values must have layouts";
            return false;
        }
        llvm::json::Object row{{"id", static_cast<int64_t>(expectedId)},
                               {"name", value->getString("name").value_or("value").str()},
                               {"type", type->str()},
                               {"origin", std::move(origin)}};
        if (!opaqueResource) {
            row["shape"] = copyArray(*shape);
            row["value_layout"] = manifestLayout(*layout, resource ? "element" : "value");
        }
        if (resource)
            row["storage"] = storageByValue[static_cast<int64_t>(expectedId)];
        values.emplace_back(std::move(row));
    }

    llvm::json::Object stages;
    std::map<std::string, llvm::json::Array> canonicalNodesByGraph;
    for (const CanonicalGraph &view : selectedGraphs) {
        llvm::json::Array &canonicalNodes = canonicalNodesByGraph[view.name];
        for (const llvm::json::Value &nodeValue : *view.nodes) {
            const llvm::json::Object *node = nodeValue.getAsObject();
            const std::string logicalStage = node->getString("stage")->str();
            const CanonicalComputeStage &compiled = *compiledByRequest[logicalStage];
            const llvm::StringRef requestId = compiled.requestId;
            const llvm::json::Object &compiledReflection = compiled.compiledReflection;
            const llvm::json::Object &compiledEntry = compiled.compiledEntry;
            const llvm::json::Array *nodeOperands = node->getArray("operands");
            const llvm::json::Array *nodeResults = node->getArray("results");
            const llvm::json::Array *rawBindings = node->getArray("bindings");
            const llvm::json::Array *workgroups = node->getArray("grid");
            std::map<int64_t, LogicalResource> &resources = resourcesByStage[compiled.requestId];
            std::map<std::string, int64_t> boundValues;
            for (const llvm::json::Value &bindingValue : *rawBindings) {
                const llvm::json::Object *binding = bindingValue.getAsObject();
                const std::optional<llvm::StringRef> name = binding ? binding->getString("parameter") : std::nullopt;
                const std::optional<int64_t> value = binding ? binding->getInteger("value") : std::nullopt;
                if (!name || name->empty() || !value || !boundValues.emplace(name->str(), *value).second) {
                    error = "canonical compute node has invalid or duplicate parameter bindings";
                    return false;
                }
            }

            if (node->getString("kind") == "render") {
                const llvm::json::Array *entries = compiledReflection.getArray("entries");
                if (!entries || entries->empty()) {
                    error = "compiled graphics pipeline has no reflected modules";
                    return false;
                }
                std::map<std::string, const llvm::json::Object *> modules;
                for (const llvm::json::Value &entryValue : *entries) {
                    const llvm::json::Object *entry = entryValue.getAsObject();
                    const std::optional<llvm::StringRef> role = entry ? entry->getString("stage") : std::nullopt;
                    if (!role || (*role != "vertex" && *role != "fragment") ||
                        !modules.emplace(role->str(), entry).second) {
                        error = "compiled graphics pipeline has invalid or duplicate module roles";
                        return false;
                    }
                }
                if (!modules.count("vertex")) {
                    error = "compiled graphics pipeline requires a vertex module";
                    return false;
                }

                llvm::json::Array endpoints;
                llvm::json::Array endpointBindings;
                llvm::json::Array implementationEndpoints;
                llvm::json::Array accesses;
                std::map<int64_t, int64_t> accessByBefore;
                int64_t nextSlot = 0;
                const auto accessFor = [&](int64_t valueId, bool attachment) -> std::optional<int64_t> {
                    if (auto found = accessByBefore.find(valueId); found != accessByBefore.end())
                        return found->second;
                    auto resource = resources.find(valueId);
                    if (resource == resources.end())
                        return std::nullopt;
                    const int64_t accessIndex = static_cast<int64_t>(accesses.size());
                    accessByBefore[valueId] = accessIndex;
                    if (attachment) {
                        if (!resource->second.after)
                            return std::nullopt;
                        accesses.emplace_back(llvm::json::Object{{"tag", "attachment"},
                                                                 {"storage", resource->second.storage},
                                                                 {"before", valueId},
                                                                 {"after", *resource->second.after}});
                    } else if (resource->second.access == "read") {
                        accesses.emplace_back(llvm::json::Object{
                            {"tag", "read"}, {"storage", resource->second.storage}, {"value", valueId}});
                    } else if (!resource->second.after) {
                        accesses.emplace_back(llvm::json::Object{
                            {"tag", "initialize"}, {"storage", resource->second.storage}, {"after", valueId}});
                    } else {
                        accesses.emplace_back(llvm::json::Object{{"tag", "write"},
                                                                 {"storage", resource->second.storage},
                                                                 {"before", valueId},
                                                                 {"after", *resource->second.after},
                                                                 {"access", resource->second.access}});
                    }
                    return accessIndex;
                };

                llvm::json::Array vertexInputs;
                llvm::json::Array vertexOutputs;
                llvm::json::Array fragmentInputs;
                llvm::json::Array fragmentOutputs;
                std::set<std::string> coveredBindings;
                const auto appendLinkage = [&](llvm::json::Array &rows, const llvm::json::Object &row) {
                    llvm::json::Object linkage;
                    if (std::optional<int64_t> location = row.getInteger("vernon.location"))
                        linkage["location"] = *location;
                    else if (std::optional<llvm::StringRef> builtin = row.getString("vernon.builtin"))
                        linkage["builtin"] = builtin->str();
                    else
                        return false;
                    linkage["type"] = row.getString("type").value_or("").str();
                    if (linkage.get("location"))
                        linkage["interpolation"] = row.getString("vernon.interpolation").value_or("smooth").str();
                    rows.emplace_back(std::move(linkage));
                    return true;
                };
                const auto vertexFormat = [](llvm::StringRef dtype, int64_t components) {
                    const llvm::StringRef suffix = dtype == "f32" ? "float" : dtype == "u32" ? "uint" : "sint";
                    if (components <= 1)
                        return ("r32_" + suffix).str();
                    if (components == 2)
                        return ("rg32_" + suffix).str();
                    if (components == 3)
                        return ("rgb32_" + suffix).str();
                    return ("rgba32_" + suffix).str();
                };

                for (llvm::StringRef role : {llvm::StringRef("vertex"), llvm::StringRef("fragment")}) {
                    auto moduleIt = modules.find(role.str());
                    if (moduleIt == modules.end())
                        continue;
                    const llvm::json::Object &entry = *moduleIt->second;
                    const auto appendRows = [&](const llvm::json::Array *rows, llvm::StringRef interfaceKind) -> bool {
                        if (!rows)
                            return true;
                        for (size_t ordinal = 0; ordinal < rows->size(); ++ordinal) {
                            const llvm::json::Object *row = (*rows)[ordinal].getAsObject();
                            if (!row) {
                                error = "compiled graphics endpoint is not an object";
                                return false;
                            }
                            const int64_t endpointIndex = row->getInteger("index").value_or(ordinal);
                            const bool moduleLinkage =
                                (role == "vertex" && interfaceKind == "result") ||
                                (role == "fragment" && interfaceKind == "argument" &&
                                 (row->getInteger("vernon.location") || row->getString("vernon.builtin")));
                            if (moduleLinkage) {
                                if (!appendLinkage(role == "vertex" ? vertexOutputs : fragmentInputs, *row)) {
                                    error = "compiled graphics linkage has no location or builtin";
                                    return false;
                                }
                                continue;
                            }
                            if (role == "fragment" && interfaceKind == "result") {
                                const std::optional<int64_t> location = row->getInteger("vernon.location");
                                if (!location) {
                                    error = "compiled fragment output has no location";
                                    return false;
                                }
                                fragmentOutputs.emplace_back(llvm::json::Object{
                                    {"location", *location}, {"type", row->getString("type").value_or("").str()}});
                                continue;
                            }
                            const std::optional<llvm::StringRef> builtin = row->getString("vernon.builtin")
                                                                               ? row->getString("vernon.builtin")
                                                                               : row->getString("builtin");
                            const std::optional<llvm::StringRef> implicit = row->getString("vernon.implicit");
                            if (implicit == "resolution") {
                                const llvm::json::Object *layout = endpointLayout(*row, false);
                                if (!layout) {
                                    error = "compiled graphics resolution endpoint has no canonical layout";
                                    return false;
                                }
                                llvm::json::Array abiBindings;
                                abiBindings.emplace_back(
                                    llvm::json::Object{{"semantic", "value"},
                                                       {"carrier", valueCarrier("value_slot", nextSlot++, *layout)}});
                                endpoints.emplace_back(llvm::json::Object{
                                    {"tag", "value"},
                                    {"module", role.str()},
                                    {"interface", "system_value"},
                                    {"index", endpointIndex},
                                    {"type", row->getString("type").value_or("").str()},
                                    {"layout_hash", layout->getString("layout_hash").value_or("").str()},
                                    {"transport", "by_value"},
                                    {"access", "read"},
                                    {"builtin", "resolution"},
                                    {"abi", llvm::json::Object{{"bindings", std::move(abiBindings)}}},
                                });
                                implementationEndpoints.emplace_back(compiledEndpointAbi(*row, role, endpointIndex));
                                continue;
                            }
                            if (builtin || implicit) {
                                endpoints.emplace_back(llvm::json::Object{
                                    {"tag", "system"},
                                    {"module", role.str()},
                                    {"interface", interfaceKind.str()},
                                    {"index", endpointIndex},
                                    {"semantic", builtin ? builtin->str() : implicit->str()},
                                    {"abi", llvm::json::Object{{"bindings", llvm::json::Array()}}},
                                });
                                continue;
                            }
                            const std::optional<llvm::StringRef> source = row->getString("vernon.source_name");
                            auto logicalBinding = source ? boundValues.find(source->str()) : boundValues.end();
                            if (!source || logicalBinding == boundValues.end()) {
                                error = "compiled graphics endpoint is not present in logical bindings";
                                return false;
                            }
                            coveredBindings.insert(source->str());
                            const int64_t valueId = logicalBinding->second;
                            const llvm::json::Object *logicalValue = valueById(*rawValues, valueId);
                            if (!logicalValue) {
                                error = "compiled graphics endpoint references an unknown logical value";
                                return false;
                            }
                            const llvm::StringRef kind = row->getString("kind").value_or("");
                            const bool vertexAttribute = role == "vertex" && row->getArray("attribute_leaves");
                            const bool resourceEndpoint =
                                vertexAttribute || kind == "image" || kind == "sampler" || resources.count(valueId);
                            llvm::json::Array abiBindings;
                            if (resourceEndpoint) {
                                abiBindings.emplace_back(llvm::json::Object{
                                    {"semantic", kind == "sampler" ? "sampler" : "resource"},
                                    {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", nextSlot++}}}});
                                const std::optional<int64_t> accessIndex = accessFor(valueId, false);
                                const llvm::json::Object *layout = endpointLayout(*row, true);
                                if (!accessIndex || (!layout && kind != "image" && kind != "sampler")) {
                                    error = "compiled graphics resource endpoint has no logical access or layout";
                                    return false;
                                }
                                llvm::json::Object resourceLayout;
                                if (kind == "image")
                                    resourceLayout = llvm::json::Object{
                                        {"tag", "image"},
                                        {"dimension", row->getString("dimension").value_or("").str()},
                                        {"format", row->getString("exact_storage_format").value_or("any").str()},
                                        {"sample_count", int64_t{0}},
                                        {"aspects", llvm::json::Array{"color"}}};
                                else if (kind == "sampler")
                                    resourceLayout = llvm::json::Object{{"tag", "sampler"}};
                                else if (vertexAttribute) {
                                    llvm::json::Array cellShape = attributeCellShape(*row);
                                    const int64_t viewRank = static_cast<int64_t>(cellShape.size());
                                    resourceLayout = llvm::json::Object{
                                        {"tag", "buffer"},
                                        {"view_rank", viewRank},
                                        {"shape", std::move(cellShape)},
                                        {"descriptor", false},
                                        {"element_layout_hash", layout->getString("layout_hash").value_or("").str()},
                                        {"minimum_alignment", layout->getInteger("alignment").value_or(0)}};
                                } else
                                    resourceLayout = llvm::json::Object{
                                        {"tag", "buffer"},
                                        {"view_rank",
                                         logicalValue->getArray("shape")
                                             ? static_cast<int64_t>(logicalValue->getArray("shape")->size())
                                             : int64_t{0}},
                                        {"descriptor", true},
                                        {"element_layout_hash", layout->getString("layout_hash").value_or("").str()},
                                        {"minimum_alignment", layout->getInteger("alignment").value_or(0)}};
                                endpoints.emplace_back(llvm::json::Object{
                                    {"tag", "resource"},
                                    {"module", role.str()},
                                    {"interface", interfaceKind.str()},
                                    {"index", endpointIndex},
                                    {"role", vertexAttribute ? "vertex"
                                                             : row->getString("binding_role")
                                                                   .value_or(kind == "sampler" ? "sampler" : "storage")
                                                                   .str()},
                                    {"type", logicalValue->getString("type").value_or("").str()},
                                    {"layout", std::move(resourceLayout)},
                                    {"address_space", row->getString("address_space").value_or("device").str()},
                                    {"transport", "resource_handle"},
                                    {"access", resources.at(valueId).access},
                                    {"abi", llvm::json::Object{{"bindings", std::move(abiBindings)}}},
                                });
                                endpointBindings.emplace_back(llvm::json::Object{{"module", role.str()},
                                                                                 {"interface", interfaceKind.str()},
                                                                                 {"index", endpointIndex},
                                                                                 {"tag", "resource"},
                                                                                 {"access", *accessIndex}});
                                implementationEndpoints.emplace_back(compiledEndpointAbi(*row, role, endpointIndex));
                                if (vertexAttribute) {
                                    const int64_t divisor = row->getInteger("vernon.instance_divisor").value_or(0);
                                    if (divisor < 0) {
                                        error = "compiled graphics vertex input has a negative instance divisor";
                                        return false;
                                    }
                                    for (const llvm::json::Value &leafValue : *row->getArray("attribute_leaves"))
                                        if (const llvm::json::Object *leaf = leafValue.getAsObject())
                                            vertexInputs.emplace_back(llvm::json::Object{
                                                {"location", leaf->getInteger("location").value_or(0)},
                                                {"endpoint_index", endpointIndex},
                                                {"format",
                                                 vertexFormat(leaf->getString("dtype").value_or("f32"),
                                                              leaf->getInteger("component_count").value_or(1))},
                                                {"byte_offset", leaf->getInteger("byte_offset").value_or(0)},
                                                {"byte_stride", layout->getInteger("byte_size").value_or(0)},
                                                {"step", divisor > 0 ? "instance" : "vertex"},
                                                {"divisor", divisor},
                                            });
                                }
                            } else {
                                const llvm::json::Object *layout = endpointLayout(*row, false);
                                if (!layout) {
                                    error = "compiled graphics value endpoint has no canonical layout";
                                    return false;
                                }
                                abiBindings.emplace_back(
                                    llvm::json::Object{{"semantic", "value"},
                                                       {"carrier", valueCarrier("value_slot", nextSlot++, *layout)}});
                                endpoints.emplace_back(llvm::json::Object{
                                    {"tag", "value"},
                                    {"module", role.str()},
                                    {"interface", interfaceKind.str()},
                                    {"index", endpointIndex},
                                    {"type", logicalValue->getString("type").value_or("").str()},
                                    {"layout_hash", layout->getString("layout_hash").value_or("").str()},
                                    {"transport", "by_value"},
                                    {"access", "read"},
                                    {"abi", llvm::json::Object{{"bindings", std::move(abiBindings)}}},
                                });
                                endpointBindings.emplace_back(llvm::json::Object{{"module", role.str()},
                                                                                 {"interface", interfaceKind.str()},
                                                                                 {"index", endpointIndex},
                                                                                 {"tag", "value"},
                                                                                 {"value", valueId}});
                                implementationEndpoints.emplace_back(compiledEndpointAbi(*row, role, endpointIndex));
                            }
                        }
                        return true;
                    };
                    if (!appendRows(entry.getArray("arguments"), "argument") ||
                        !appendRows(entry.getArray("results"), "result"))
                        return false;
                }
                if (coveredBindings.size() != boundValues.size()) {
                    error = "compiled graphics ABI does not exactly cover logical bindings";
                    return false;
                }

                const int64_t colorCount = node->getInteger("color_count").value_or(1);
                if (colorCount < 1 || static_cast<uint64_t>(colorCount) > nodeOperands->size() ||
                    nodeOperands->size() < static_cast<uint64_t>(colorCount) ||
                    (nodeResults && nodeResults->size() < static_cast<uint64_t>(colorCount))) {
                    error = "canonical graphics node has an invalid color attachment count";
                    return false;
                }
                const bool hasDepth = nodeResults && nodeResults->size() == static_cast<uint64_t>(colorCount) + 1;
                if (nodeOperands->size() < static_cast<size_t>(colorCount + (hasDepth ? 1 : 0))) {
                    error = "canonical graphics node is missing attachment operands";
                    return false;
                }
                struct AttachmentImage {
                    int64_t value{};
                    int64_t access{};
                    const llvm::json::Object *descriptor{};
                    std::string format;
                };
                const auto attachmentImage = [&](size_t operandIndex,
                                                 llvm::StringRef role) -> std::optional<AttachmentImage> {
                    const std::optional<int64_t> valueId = (*nodeOperands)[operandIndex].getAsInteger();
                    const std::optional<int64_t> access = valueId ? accessFor(*valueId, true) : std::nullopt;
                    if (!valueId || !access) {
                        error = ("canonical graphics " + role + " has no attachment access").str();
                        return std::nullopt;
                    }
                    const LogicalResource &resource = resources.at(*valueId);
                    const llvm::json::Object *storage =
                        resource.storage >= 0 && static_cast<size_t>(resource.storage) < storages.size()
                            ? storages[static_cast<size_t>(resource.storage)].getAsObject()
                            : nullptr;
                    const llvm::json::Object *descriptor = storage ? storage->getObject("descriptor") : nullptr;
                    const llvm::json::Array *extent = descriptor ? descriptor->getArray("extent") : nullptr;
                    if (!descriptor || descriptor->getString("tag") != "image" || !extent || extent->size() < 2) {
                        error = ("canonical graphics " + role + " requires a concrete image Storage").str();
                        return std::nullopt;
                    }
                    return AttachmentImage{*valueId, *access, descriptor,
                                           descriptor->getString("format").value_or("").str()};
                };
                std::vector<AttachmentImage> colorImages;
                for (int64_t index = 0; index < colorCount; ++index) {
                    std::optional<AttachmentImage> image =
                        attachmentImage(static_cast<size_t>(index), "color attachment");
                    if (!image)
                        return false;
                    colorImages.push_back(*image);
                }
                std::optional<AttachmentImage> depthImage;
                if (hasDepth) {
                    depthImage = attachmentImage(static_cast<size_t>(colorCount), "depth attachment");
                    if (!depthImage)
                        return false;
                }
                const llvm::json::Array *extent = colorImages.front().descriptor->getArray("extent");
                llvm::json::Array attachmentConstraints;
                for (const llvm::json::Value &outputValue : fragmentOutputs) {
                    const llvm::json::Object *output = outputValue.getAsObject();
                    if (!output)
                        continue;
                    const int64_t location = output->getInteger("location").value_or(0);
                    if (location < 0 || static_cast<uint64_t>(location) >= colorImages.size()) {
                        error = "compiled fragment output has no matching color attachment";
                        return false;
                    }
                    const AttachmentImage &color = colorImages[static_cast<size_t>(location)];
                    attachmentConstraints.emplace_back(llvm::json::Object{
                        {"location", location},
                        {"formats", llvm::json::Array{color.format}},
                        {"sample_counts", llvm::json::Array{color.descriptor->getInteger("sample_count").value_or(1)}},
                        {"aspects", llvm::json::Array{"color"}}});
                }
                if (depthImage)
                    attachmentConstraints.emplace_back(llvm::json::Object{
                        {"location", int64_t{-1}},
                        {"formats", llvm::json::Array{depthImage->format}},
                        {"sample_counts",
                         llvm::json::Array{depthImage->descriptor->getInteger("sample_count").value_or(1)}},
                        {"aspects", llvm::json::Array{"depth"}}});
                llvm::json::Array requiredFeatures;
                if (const llvm::json::Array *rawFeatures = compiledReflection.getArray("required_features"))
                    requiredFeatures = copyArray(*rawFeatures);
                llvm::json::Object graphics{
                    {"topology", node->getString("topology").value_or("triangle_list").str()},
                    {"vertex_inputs", std::move(vertexInputs)},
                    {"fragment_outputs", copyArray(fragmentOutputs)},
                    {"linkage", llvm::json::Object{{"vertex_outputs", std::move(vertexOutputs)},
                                                   {"fragment_inputs", std::move(fragmentInputs)}}},
                    {"attachment_constraints", std::move(attachmentConstraints)},
                    {"index_formats", llvm::json::Array{"u16", "u32"}},
                    {"capabilities", llvm::json::Array{"direct_draw"}},
                };
                llvm::json::Object reflection{{"required_features", std::move(requiredFeatures)},
                                              {"endpoints", std::move(endpoints)},
                                              {"graphics", std::move(graphics)}};
                llvm::json::Object stageContract{{"operation", "graphics"}, {"reflection", std::move(reflection)}};
                const std::string contractHash = sha256(llvm::json::Value(copyObject(stageContract)));
                stageContracts[requestId] = std::move(stageContract);
                if (!implementationEndpoints.empty()) {
                    llvm::json::Object implementation{{"endpoints", std::move(implementationEndpoints)}};
                    if (const llvm::json::Array *slots = compiledReflection.getArray("metal_resource_slots"))
                        implementation["metal_resource_slots"] = copyArray(*slots);
                    targetImplementations[requestId] = std::move(implementation);
                }
                stages[requestId] = llvm::json::Object{{"operation", "graphics"}, {"contract_hash", contractHash}};

                const int64_t vertexCount = [&]() {
                    for (const auto &[name, valueId] : boundValues)
                        if (resources.count(valueId))
                            if (const llvm::json::Object *value = valueById(*rawValues, valueId))
                                if (const llvm::json::Array *shape = value->getArray("shape"); shape && !shape->empty())
                                    return (*shape)[0].getAsInteger().value_or(1);
                    return int64_t{1};
                }();
                llvm::json::Array colors;
                for (size_t index = 0; index < colorImages.size(); ++index)
                    colors.emplace_back(llvm::json::Object{{"location", static_cast<int64_t>(index)},
                                                           {"access", colorImages[index].access},
                                                           {"load", llvm::json::Object{{"tag", "discard"}}},
                                                           {"store", "store"}});
                llvm::json::Value depthStencil = nullptr;
                if (depthImage)
                    depthStencil = llvm::json::Object{{"access", depthImage->access},
                                                      {"load", llvm::json::Object{{"tag", "discard"}}},
                                                      {"store", "store"},
                                                      {"depth", true},
                                                      {"stencil", false}};
                llvm::json::Object operation{
                    {"tag", "graphics"},
                    {"attachments",
                     llvm::json::Object{
                         {"colors", std::move(colors)},
                         {"depth_stencil", std::move(depthStencil)},
                         {"render_area", llvm::json::Object{{"x", int64_t{0}},
                                                            {"y", int64_t{0}},
                                                            {"width", (*extent)[0].getAsInteger().value_or(1)},
                                                            {"height", (*extent)[1].getAsInteger().value_or(1)}}},
                         {"layer_count", int64_t{1}}}},
                    {"state",
                     llvm::json::Object{
                         {"raster", llvm::json::Object{{"front_face", "counter_clockwise"},
                                                       {"cull_mode", "none"},
                                                       {"fill_mode", "fill"}}},
                         {"depth_stencil", llvm::json::Object{{"depth_test", hasDepth},
                                                              {"depth_write", hasDepth},
                                                              {"depth_compare", hasDepth ? "less" : "always"},
                                                              {"stencil_test", false}}},
                         {"multisample",
                          llvm::json::Object{{"sample_mask", int64_t{4294967295ULL}}, {"alpha_to_coverage", false}}},
                         {"blend", llvm::json::Array()},
                         {"viewport", nullptr},
                         {"scissor", nullptr}}},
                    {"draw", llvm::json::Object{{"tag", "direct"},
                                                {"vertex_count", vertexCount},
                                                {"instance_count", int64_t{1}}}},
                };
                llvm::json::Array operands;
                llvm::json::Array results;
                if (!canonicalValueIds(*nodeOperands, operands, error) ||
                    !canonicalValueIds(*nodeResults, results, error))
                    return false;
                canonicalNodes.emplace_back(llvm::json::Object{{"id", node->getInteger("id").value_or(0)},
                                                               {"name", node->getString("name").value_or("").str()},
                                                               {"stage", requestId.str()},
                                                               {"operands", std::move(operands)},
                                                               {"results", std::move(results)},
                                                               {"bindings", std::move(endpointBindings)},
                                                               {"accesses", std::move(accesses)},
                                                               {"operation", std::move(operation)}});
                continue;
            }

            std::map<std::string, const llvm::json::Object *> interface;
            std::set<std::string> endpointNames;
            size_t logicalInterfaceCount = 0;
            const auto collect = [&](const llvm::json::Array *rows) {
                if (!rows)
                    return true;
                for (const llvm::json::Value &rowValue : *rows) {
                    const llvm::json::Object *row = rowValue.getAsObject();
                    if (!row)
                        return false;
                    const std::optional<llvm::StringRef> builtin =
                        row->getString("vernon.builtin") ? row->getString("vernon.builtin") : row->getString("builtin");
                    if (builtin)
                        continue;
                    const std::optional<llvm::StringRef> source = row->getString("vernon.source_name");
                    if (!source || source->empty() || !endpointNames.insert(source->str()).second)
                        return false;
                    interface.emplace(source->str(), row);
                    const std::optional<llvm::StringRef> role = row->getString("vernon.autodiff_role");
                    if (!role || !isTapeCarrierRole(*role))
                        ++logicalInterfaceCount;
                }
                return true;
            };
            std::map<std::string, int64_t> namedBound;
            for (const auto &[name, value] : boundValues) {
                const llvm::json::Object *logical = valueById(*rawValues, value);
                const llvm::StringRef type = logical ? logical->getString("type").value_or("") : "";
                if (name == "tape" || isAdTapeType(type))
                    continue;
                namedBound.emplace(name, value);
            }
            if (!collect(compiledEntry.getArray("arguments")) || !collect(compiledEntry.getArray("results")) ||
                logicalInterfaceCount != namedBound.size()) {
                error = "compiled compute ABI source names must be unique and exactly cover logical bindings";
                return false;
            }
            for (const auto &[name, value] : namedBound)
                if (!interface.count(name) || !valueById(*rawValues, value)) {
                    error = "compiled compute ABI does not exactly cover logical endpoint '" + name + "'";
                    return false;
                }

            std::set<int64_t> portableSlots;
            const auto collectSlot = [&](const llvm::json::Object &row) {
                if (std::optional<int64_t> slot = row.getInteger("vernon.binding"))
                    portableSlots.insert(*slot);
                if (const llvm::json::Array *leaves = row.getArray("storage_leaves"))
                    for (const llvm::json::Value &leafValue : *leaves)
                        if (const llvm::json::Object *leaf = leafValue.getAsObject())
                            if (std::optional<int64_t> slot = leaf->getInteger("binding"))
                                portableSlots.insert(*slot);
                if (const llvm::json::Object *descriptor = row.getObject("tensor_view_descriptor")) {
                    if (std::optional<int64_t> slot = descriptor->getInteger("offset_binding"))
                        portableSlots.insert(*slot);
                    for (llvm::StringRef field : {"extent_bindings", "stride_bindings"})
                        if (const llvm::json::Array *slots = descriptor->getArray(field))
                            for (const llvm::json::Value &slotValue : *slots)
                                if (std::optional<int64_t> slot = slotValue.getAsInteger())
                                    portableSlots.insert(*slot);
                }
            };
            if (const llvm::json::Array *rows = compiledEntry.getArray("arguments"))
                for (const llvm::json::Value &rowValue : *rows)
                    if (const llvm::json::Object *row = rowValue.getAsObject())
                        if (!row->getString("vernon.builtin") && !row->getString("builtin"))
                            collectSlot(*row);
            int64_t nextSlot = portableSlots.empty() ? 0 : *portableSlots.rbegin() + 1;

            llvm::json::Array endpoints;
            llvm::json::Array endpointBindings;
            llvm::json::Array implementationEndpoints;
            llvm::json::Array accesses;
            std::map<int64_t, int64_t> accessByBefore;
            std::map<std::string, llvm::json::Object> footprints;
            if (const llvm::json::Array *rows = compiledEntry.getArray("tensor_view_write_footprints"))
                for (const llvm::json::Value &rowValue : *rows)
                    if (const llvm::json::Object *row = rowValue.getAsObject())
                        if (std::optional<llvm::StringRef> owner = row->getString("owner"))
                            footprints.emplace(owner->str(), *row);

            const auto appendEndpoint = [&](const llvm::json::Object &row, llvm::StringRef interfaceKind,
                                            size_t fallbackIndex) -> bool {
                const std::optional<llvm::StringRef> builtin =
                    row.getString("vernon.builtin") ? row.getString("vernon.builtin") : row.getString("builtin");
                const int64_t endpointIndex = row.getInteger("index").value_or(static_cast<int64_t>(fallbackIndex));
                if (builtin) {
                    if (kernelTapeBuiltin(*builtin)) {
                        implementationEndpoints.emplace_back(compiledEndpointAbi(row, "compute", endpointIndex));
                        return true;
                    }
                    if (kernelHiddenBuiltin(*builtin))
                        return true;
                    const llvm::json::Object *layout = row.getObject("value_layout");
                    if (!layout) {
                        error = "compiled compute system value has no canonical layout (" + builtin->str() + ")";
                        return false;
                    }
                    llvm::json::Array abi;
                    abi.emplace_back(llvm::json::Object{
                        {"semantic", "value"}, {"carrier", valueCarrier("constant_region", nextSlot++, *layout)}});
                    endpoints.emplace_back(llvm::json::Object{
                        {"tag", "value"},
                        {"module", "compute"},
                        {"interface", "system_value"},
                        {"index", endpointIndex},
                        {"type", row.getString("type").value_or("").str()},
                        {"layout_hash", layout->getString("layout_hash").value_or("").str()},
                        {"transport", "by_value"},
                        {"access", "read"},
                        {"builtin", builtin->str()},
                        {"abi", llvm::json::Object{{"bindings", std::move(abi)}}},
                    });
                    implementationEndpoints.emplace_back(compiledEndpointAbi(row, "compute", endpointIndex));
                    return true;
                }
                const std::optional<llvm::StringRef> source = row.getString("vernon.source_name");
                auto binding = source ? boundValues.find(source->str()) : boundValues.end();
                if (!source || binding == boundValues.end()) {
                    error = "compiled compute endpoint is not present in logical bindings";
                    return false;
                }
                const int64_t valueId = binding->second;
                const llvm::json::Object *logicalValue = valueById(*rawValues, valueId);
                auto resourceIt = resources.find(valueId);
                const bool resource = resourceIt != resources.end();
                const llvm::StringRef endpointKind = row.getString("kind").value_or("");
                const bool opaqueResourceEndpoint = endpointKind == "image" || endpointKind == "sampler";
                const std::optional<llvm::StringRef> autodiffRole = row.getString("vernon.autodiff_role");
                const bool tapeCarrier = logicalValue && isAdTapeType(logicalValue->getString("type").value_or("")) &&
                                         autodiffRole && isTapeCarrierRole(*autodiffRole);
                const llvm::json::Object *logicalLayout =
                    logicalValue ? logicalValue->getObject("value_layout") : nullptr;
                const llvm::json::Array *logicalShape = logicalValue ? logicalValue->getArray("shape") : nullptr;
                const llvm::json::Object *wholeLayout = endpointLayout(row, resource);
                const llvm::json::Array *physicalShape =
                    row.getArray("source_shape") ? row.getArray("source_shape") : row.getArray("shape");
                const bool opaqueMatches = opaqueResourceEndpoint && resource && logicalValue &&
                                           logicalValue->getString("type") == row.getString("type");
                const bool byteValueMatches =
                    !opaqueResourceEndpoint && logicalValue && wholeLayout && logicalLayout && logicalShape &&
                    wholeLayout->getString("layout_hash") == logicalLayout->getString("layout_hash") &&
                    compatibleAbiShape(logicalShape, physicalShape ? physicalShape : logicalShape);
                bool projectedLeafMatches = false;
                const llvm::json::Array *logicalLeaves = logicalLayout ? logicalLayout->getArray("leaves") : nullptr;
                const llvm::json::Array *physicalLeaves = wholeLayout ? wholeLayout->getArray("leaves") : nullptr;
                const llvm::json::Object *physicalLeaf =
                    physicalLeaves && physicalLeaves->size() == 1 ? (*physicalLeaves)[0].getAsObject() : nullptr;
                std::optional<size_t> projectedLeafIndex;
                const std::optional<llvm::StringRef> autodiffSource = row.getString("vernon.autodiff_source");
                if (!opaqueResourceEndpoint && source && autodiffSource && logicalLayout && logicalLeaves &&
                    physicalLeaf && logicalShape &&
                    compatibleProgramBindingShape(autodiffRole.value_or(""),
                                                  row.getString("vernon.autodiff_carrier").value_or(""), logicalShape,
                                                  physicalShape ? physicalShape : logicalShape)) {
                    projectedLeafIndex = resolveProgramValueLeafIndex(*logicalLayout, *autodiffSource, *source);
                    const llvm::json::Object *projectedLeaf =
                        projectedLeafIndex && *projectedLeafIndex < logicalLeaves->size()
                            ? (*logicalLeaves)[*projectedLeafIndex].getAsObject()
                            : nullptr;
                    projectedLeafMatches =
                        projectedLeaf && projectedLeaf->getString("dtype") == physicalLeaf->getString("dtype") &&
                        projectedLeaf->getInteger("scalar_count") == physicalLeaf->getInteger("scalar_count") &&
                        compatibleAbiShape(projectedLeaf->getArray("shape"), physicalLeaf->getArray("shape"));
                    if (!projectedLeafMatches)
                        projectedLeafIndex.reset();
                }
                if (!tapeCarrier && !opaqueMatches && !byteValueMatches && !projectedLeafMatches) {
                    error = "compiled compute endpoint layout/type/shape does not match logical value '" +
                            source->str() + "'";
                    return false;
                }
                if (std::optional<llvm::StringRef> logicalType = logicalValue->getString("type");
                    !resource && logicalType && row.getString("type") && *logicalType != *row.getString("type")) {
                    error = "compiled compute endpoint type does not match logical value '" + source->str() + "'";
                    return false;
                }
                if (!resource) {
                    const std::optional<int64_t> reflectedSlot = row.getInteger("vernon.binding");
                    const int64_t slot = reflectedSlot ? *reflectedSlot : nextSlot++;
                    llvm::json::Array abi;
                    abi.emplace_back(llvm::json::Object{{"semantic", "value"},
                                                        {"carrier", valueCarrier("value_slot", slot, *wholeLayout)}});
                    llvm::json::Object endpoint{
                        {"tag", "value"},
                        {"module", "compute"},
                        {"interface", interfaceKind.str()},
                        {"index", endpointIndex},
                        {"type", logicalValue->getString("type").value_or("").str()},
                        {"layout_hash", wholeLayout->getString("layout_hash").value_or("").str()},
                        {"transport", "by_value"},
                        {"access", "read"},
                        {"abi", llvm::json::Object{{"bindings", std::move(abi)}}}};
                    if (const llvm::json::Object *elementLayout = row.getObject("element_layout"))
                        endpoint["element_layout_hash"] = elementLayout->getString("layout_hash").value_or("").str();
                    endpoints.emplace_back(std::move(endpoint));
                    endpointBindings.emplace_back(llvm::json::Object{{"module", "compute"},
                                                                     {"interface", interfaceKind.str()},
                                                                     {"index", endpointIndex},
                                                                     {"tag", "value"},
                                                                     {"value", valueId}});
                    implementationEndpoints.emplace_back(compiledEndpointAbi(row, "compute", endpointIndex));
                    return true;
                }
                const std::optional<std::string> physicalAccess = reflectedAccess(row);
                if ((!physicalAccess && interfaceKind != "result") ||
                    (physicalAccess && !tapeCarrier &&
                     !resourceAccessSatisfies(*physicalAccess, resourceIt->second.access))) {
                    error =
                        "compiled compute resource access disagrees with logical access for '" + source->str() + "'";
                    return false;
                }
                const llvm::json::Object *descriptor = row.getObject("tensor_view_descriptor");
                llvm::json::Array abi;
                if (opaqueResourceEndpoint)
                    if (std::optional<int64_t> slot = row.getInteger("vernon.binding"))
                        abi.emplace_back(llvm::json::Object{
                            {"semantic", "resource"},
                            {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", *slot}}}});
                if (const llvm::json::Array *leaves = row.getArray("storage_leaves"))
                    for (size_t leafIndex = 0; leafIndex < leaves->size(); ++leafIndex)
                        if (const llvm::json::Object *leaf = (*leaves)[leafIndex].getAsObject())
                            if (std::optional<int64_t> slot = leaf->getInteger("binding"))
                                abi.emplace_back(llvm::json::Object{
                                    {"semantic", llvm::json::Object{{"storage_leaf",
                                                                     static_cast<int64_t>(
                                                                         projectedLeafIndex.value_or(0) + leafIndex)}}},
                                    {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", *slot}}}});
                if (descriptor) {
                    const int64_t rank = descriptor->getInteger("rank").value_or(0);
                    abi.emplace_back(llvm::json::Object{
                        {"semantic", "byte_offset"},
                        {"carrier", llvm::json::Object{{"tag", "value_slot"},
                                                       {"slot", descriptor->getInteger("offset_binding").value_or(-1)},
                                                       {"byte_offset", int64_t{0}},
                                                       {"byte_size", int64_t{8}},
                                                       {"alignment", int64_t{8}}}}});
                    for (llvm::StringRef field : {"extent_bindings", "stride_bindings"}) {
                        const llvm::json::Array *slots = descriptor->getArray(field);
                        if (!slots || static_cast<int64_t>(slots->size()) != rank) {
                            error = "compiled compute TensorView descriptor is incomplete";
                            return false;
                        }
                        for (size_t axis = 0; axis < slots->size(); ++axis)
                            abi.emplace_back(llvm::json::Object{
                                {"semantic", llvm::json::Object{{field == "extent_bindings" ? "extent" : "byte_stride",
                                                                 static_cast<int64_t>(axis)}}},
                                {"carrier", llvm::json::Object{{"tag", "value_slot"},
                                                               {"slot", (*slots)[axis].getAsInteger().value_or(-1)},
                                                               {"byte_offset", int64_t{0}},
                                                               {"byte_size", int64_t{8}},
                                                               {"alignment", int64_t{8}}}}});
                    }
                }
                if (abi.empty()) {
                    error = "compiled compute resource endpoint has no storage leaves";
                    return false;
                }
                llvm::json::Object resourceLayout;
                if (endpointKind == "image") {
                    resourceLayout =
                        llvm::json::Object{{"tag", "image"},
                                           {"dimension", row.getString("dimension").value_or("").str()},
                                           {"format", row.getString("exact_storage_format").value_or("").str()},
                                           {"sample_count", int64_t{1}},
                                           {"aspects", llvm::json::Array{"color"}}};
                } else if (endpointKind == "sampler") {
                    resourceLayout = llvm::json::Object{{"tag", "sampler"}};
                } else {
                    resourceLayout = llvm::json::Object{
                        {"tag", "buffer"},
                        {"view_rank", descriptor ? descriptor->getInteger("rank").value_or(0) : 0},
                        {"shape", copyArray(*(physicalShape ? physicalShape : logicalShape))},
                        {"descriptor", descriptor != nullptr},
                        {"element_layout_hash", wholeLayout->getString("layout_hash").value_or("").str()},
                        {"minimum_alignment", wholeLayout->getInteger("alignment").value_or(0)}};
                    if (const std::optional<llvm::StringRef> carrier = row.getString("vernon.autodiff_carrier"))
                        resourceLayout["autodiff_carrier"] = carrier->str();
                }
                llvm::json::Object endpoint{
                    {"tag", "resource"},
                    {"module", "compute"},
                    {"interface", interfaceKind.str()},
                    {"index", endpointIndex},
                    {"role", autodiffRole ? autodiffRole->str()
                                          : row.getString("binding_role")
                                                .value_or(endpointKind == "sampler" ? "sampler" : "storage")
                                                .str()},
                    {"type", logicalValue->getString("type").value_or("").str()},
                    {"layout", std::move(resourceLayout)},
                    {"address_space", row.getString("address_space").value_or("device").str()},
                    {"transport", "resource_handle"},
                    {"access", resourceIt->second.access},
                    {"abi", llvm::json::Object{{"bindings", std::move(abi)}}},
                };
                if (auto footprint = footprints.find(source->str()); footprint != footprints.end())
                    endpoint["write_footprint"] =
                        llvm::json::Object{{"kind", footprint->second.getString("kind").value_or("").str()},
                                           {"indices", footprint->second.getArray("indices")
                                                           ? copyArray(*footprint->second.getArray("indices"))
                                                           : llvm::json::Array()}};
                endpoints.emplace_back(std::move(endpoint));
                int64_t accessIndex = 0;
                if (auto found = accessByBefore.find(valueId); found != accessByBefore.end()) {
                    accessIndex = found->second;
                } else {
                    accessIndex = static_cast<int64_t>(accesses.size());
                    accessByBefore[valueId] = accessIndex;
                    if (resourceIt->second.access == "read")
                        accesses.emplace_back(llvm::json::Object{
                            {"tag", "read"}, {"storage", resourceIt->second.storage}, {"value", valueId}});
                    else if (!resourceIt->second.after)
                        accesses.emplace_back(llvm::json::Object{
                            {"tag", "initialize"}, {"storage", resourceIt->second.storage}, {"after", valueId}});
                    else
                        accesses.emplace_back(llvm::json::Object{{"tag", "write"},
                                                                 {"storage", resourceIt->second.storage},
                                                                 {"before", valueId},
                                                                 {"after", *resourceIt->second.after},
                                                                 {"access", resourceIt->second.access}});
                }
                llvm::json::Object endpointBinding{{"module", "compute"},
                                                   {"interface", interfaceKind.str()},
                                                   {"index", endpointIndex},
                                                   {"tag", "resource"},
                                                   {"access", accessIndex}};
                if (projectedLeafIndex)
                    endpointBinding["leaf"] = static_cast<int64_t>(*projectedLeafIndex);
                endpointBindings.emplace_back(std::move(endpointBinding));
                implementationEndpoints.emplace_back(compiledEndpointAbi(row, "compute", endpointIndex));
                return true;
            };

            const auto emitTapeAccess = [&](int64_t valueId, const LogicalResource &resource) {
                if (accessByBefore.count(valueId))
                    return;
                accessByBefore[valueId] = static_cast<int64_t>(accesses.size());
                if (resource.access == "read")
                    accesses.emplace_back(
                        llvm::json::Object{{"tag", "read"}, {"storage", resource.storage}, {"value", valueId}});
                else if (!resource.after)
                    accesses.emplace_back(
                        llvm::json::Object{{"tag", "initialize"}, {"storage", resource.storage}, {"after", valueId}});
                else
                    accesses.emplace_back(llvm::json::Object{{"tag", "write"},
                                                             {"storage", resource.storage},
                                                             {"before", valueId},
                                                             {"after", *resource.after},
                                                             {"access", resource.access}});
            };

            size_t endpointOrdinal = 0;
            if (const llvm::json::Array *rows = compiledEntry.getArray("arguments"))
                for (const llvm::json::Value &rowValue : *rows) {
                    const llvm::json::Object *row = rowValue.getAsObject();
                    if (!row || !appendEndpoint(*row, "argument", endpointOrdinal++))
                        return false;
                }
            if (const llvm::json::Array *rows = compiledEntry.getArray("results"))
                for (const llvm::json::Value &rowValue : *rows) {
                    const llvm::json::Object *row = rowValue.getAsObject();
                    if (!row || !appendEndpoint(*row, "result", endpointOrdinal++))
                        return false;
                }
            for (const auto &[valueId, resource] : resources) {
                const llvm::json::Object *logical = valueById(*rawValues, valueId);
                const llvm::StringRef type = logical ? logical->getString("type").value_or("") : "";
                if (isAdTapeType(type))
                    emitTapeAccess(valueId, resource);
            }

            std::vector<int64_t> emittedSlots;
            for (const llvm::json::Value &endpointValue : endpoints)
                if (const llvm::json::Object *endpoint = endpointValue.getAsObject())
                    if (const llvm::json::Object *abi = endpoint->getObject("abi"))
                        if (const llvm::json::Array *bindings = abi->getArray("bindings"))
                            for (const llvm::json::Value &bindingValue : *bindings)
                                if (const llvm::json::Object *binding = bindingValue.getAsObject())
                                    if (const llvm::json::Object *carrier = binding->getObject("carrier"))
                                        if (std::optional<int64_t> slot = carrier->getInteger("slot"))
                                            emittedSlots.push_back(*slot);
            std::sort(emittedSlots.begin(), emittedSlots.end());
            for (size_t index = 0; index < emittedSlots.size(); ++index)
                if (emittedSlots[index] != static_cast<int64_t>(index)) {
                    error = "compiled compute portable ABI slots must be contiguous and unique";
                    return false;
                }

            llvm::json::Array features;
            std::set<std::string> uniqueFeatures;
            if (const llvm::json::Array *required = compiledReflection.getArray("required_features")) {
                for (const llvm::json::Value &feature : *required)
                    if (std::optional<llvm::StringRef> name = feature.getAsString())
                        uniqueFeatures.insert(name->str());
                    else {
                        error = "compiled compute reflection has invalid required features";
                        return false;
                    }
            }
            for (const std::string &feature : uniqueFeatures)
                features.emplace_back(feature);
            const llvm::json::Array *workgroupSize = compiledEntry.getArray("workgroup_size");
            if (!workgroupSize || workgroupSize->size() != 3) {
                error = "compiled compute reflection has no workgroup size";
                return false;
            }
            llvm::json::Object compute{{"workgroup_size", copyArray(*workgroupSize)},
                                       {"subgroup", nullptr},
                                       {"capabilities", llvm::json::Array{"direct_dispatch"}}};
            // The phase-one runtime consumes this validated compiler contract to reject
            // dispatches that violate lowering assumptions, so it remains in reflection.
            if (const llvm::json::Object *dispatch = compiledEntry.getObject("dispatch_contract"))
                compute["dispatch_contract"] = copyObject(*dispatch);
            llvm::json::Object reflection{{"required_features", std::move(features)},
                                          {"endpoints", std::move(endpoints)},
                                          {"compute", std::move(compute)}};
            llvm::json::Object stageContract{{"operation", "compute"}, {"reflection", std::move(reflection)}};
            const std::string contractHash = sha256(llvm::json::Value(copyObject(stageContract)));
            stageContracts[requestId] = std::move(stageContract);
            if (!implementationEndpoints.empty()) {
                llvm::json::Object implementation{{"endpoints", std::move(implementationEndpoints)}};
                if (const llvm::json::Array *slots = compiledReflection.getArray("metal_resource_slots"))
                    implementation["metal_resource_slots"] = copyArray(*slots);
                targetImplementations[requestId] = std::move(implementation);
            }
            stages[requestId] = llvm::json::Object{{"operation", "compute"}, {"contract_hash", contractHash}};
            llvm::json::Array operands;
            llvm::json::Array results;
            if (!canonicalValueIds(*nodeOperands, operands, error) || !canonicalValueIds(*nodeResults, results, error))
                return false;
            canonicalNodes.emplace_back(llvm::json::Object{
                {"id", node->getInteger("id").value_or(0)},
                {"name", node->getString("name").value_or("").str()},
                {"stage", requestId.str()},
                {"operands", std::move(operands)},
                {"results", std::move(results)},
                {"bindings", std::move(endpointBindings)},
                {"accesses", std::move(accesses)},
                {"operation", llvm::json::Object{{"tag", "compute"}, {"workgroups", copyArray(*workgroups)}}}});
        }
    }

    llvm::json::Array canonicalGraphs;
    for (const CanonicalGraph &view : selectedGraphs) {
        llvm::json::Array graphInputs;
        for (size_t slot = 0; slot < view.argumentIds->size(); ++slot)
            graphInputs.emplace_back(llvm::json::Object{
                {"tag", "user_input"}, {"value", (*view.argumentIds)[slot]}, {"slot", static_cast<int64_t>(slot)}});
        for (const llvm::json::Value &rowValue : values) {
            const llvm::json::Object *row = rowValue.getAsObject();
            const llvm::json::Object *origin = row ? row->getObject("origin") : nullptr;
            if (!origin || origin->getString("tag") != "allocation" || origin->getString("graph") != view.name)
                continue;
            const int64_t valueId = row->getInteger("id").value_or(-1);
            auto storage = storageByValue.find(valueId);
            if (storage == storageByValue.end()) {
                error = ("allocation origin has no Storage for value " + llvm::Twine(valueId) + " '" +
                         row->getString("name").value_or("") + "' type '" + row->getString("type").value_or("") + "'")
                            .str();
                return false;
            }
            graphInputs.emplace_back(
                llvm::json::Object{{"tag", "allocation"}, {"value", valueId}, {"storage", storage->second}});
        }
        llvm::json::Array graphOutputs;
        for (const llvm::json::Value &value : *view.resultIds)
            graphOutputs.emplace_back(
                llvm::json::Object{{"tag", "user_output"}, {"value", value}, {"disposition", "transfer"}});
        llvm::json::Array captures;
        if (view.direction == "backward")
            for (int64_t value : capturedValues)
                captures.emplace_back(llvm::json::Object{{"value", value}});
        canonicalGraphs.emplace_back(llvm::json::Object{{"name", view.name},
                                                        {"direction", view.direction},
                                                        {"inputs", std::move(graphInputs)},
                                                        {"captures", std::move(captures)},
                                                        {"outputs", std::move(graphOutputs)},
                                                        {"nodes", std::move(canonicalNodesByGraph[view.name])}});
    }
    const auto signatureRows = [&](llvm::StringRef field, bool output) {
        llvm::json::Array result;
        if (const llvm::json::Array *rows = rawSignature->getArray(field))
            for (const llvm::json::Value &rowValue : *rows)
                if (const llvm::json::Object *row = rowValue.getAsObject()) {
                    llvm::json::Object binding{{"path", row->getString("path").value_or("").str()},
                                               {"value", row->getInteger("value").value_or(-1)}};
                    if (output)
                        binding["disposition"] = "transfer";
                    result.emplace_back(std::move(binding));
                }
        return result;
    };
    program = llvm::json::Object{
        {"stages", std::move(stages)},
        {"parameters", llvm::json::Array()},
        {"storages", std::move(storages)},
        {"values", std::move(values)},
        {"shape_symbols", llvm::json::Array()},
        {"shape_constraints", llvm::json::Array()},
        {"alias_preconditions", llvm::json::Array()},
        {"graphs", std::move(canonicalGraphs)},
        {"signature", llvm::json::Object{{"inputs", signatureRows("inputs", false)},
                                         {"outputs", signatureRows("outputs", true)},
                                         {"cotangents", signatureRows("cotangents", false)},
                                         {"gradients", signatureRows("gradients", true)}}},
    };
    if (selectedGraphs.size() > 1) {
        llvm::json::Array residualCaptures;
        for (int64_t value : capturedValues)
            residualCaptures.emplace_back(llvm::json::Object{
                {"value", value},
                {"replay", llvm::json::Object{
                               {"legal", false}, {"required_values", llvm::json::Array()}, {"cost", int64_t{0}}}}});
        program["residual_contract"] =
            llvm::json::Object{{"captures", std::move(residualCaptures)}, {"shape_symbols", llvm::json::Array()}};
    }
    return true;
}

} // namespace vernon::compiler
