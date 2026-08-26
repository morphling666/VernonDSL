#include "program_manifest.h"
#include "pipeline_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <string_view>

namespace vernon::runtime {
namespace {

bool parseUint32(const nlohmann::json &value, uint32_t &result) {
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed > std::numeric_limits<uint32_t>::max())
            return false;
        result = static_cast<uint32_t>(parsed);
        return true;
    }
    if (!value.is_number_integer())
        return false;
    const int64_t parsed = value.get<int64_t>();
    if (parsed < 0 || static_cast<uint64_t>(parsed) > std::numeric_limits<uint32_t>::max())
        return false;
    result = static_cast<uint32_t>(parsed);
    return true;
}

bool parseUint64(const nlohmann::json &value, uint64_t &result) {
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
        return true;
    }
    if (!value.is_number_integer())
        return false;
    const int64_t parsed = value.get<int64_t>();
    if (parsed < 0)
        return false;
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool hasOnlyKeys(const nlohmann::json &value, std::initializer_list<std::string_view> allowed) {
    for (auto row = value.begin(); row != value.end(); ++row)
        if (std::find(allowed.begin(), allowed.end(), row.key()) == allowed.end())
            return false;
    return true;
}

std::string publicPath(std::string path) {
    for (std::string_view prefix : {"input.", "output.", "cotangent.", "gradient."})
        if (path.compare(0, prefix.size(), prefix) == 0)
            path = path.substr(prefix.size());
    for (std::string_view suffix : {"_cotangent", "_gradient"})
        if (path.size() > suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0)
            path.resize(path.size() - suffix.size());
    return path;
}

} // namespace

bool ExecutableProgram::validate(const std::map<std::string, std::string> &stages, std::string &error) const {
    if (graphs.empty()) {
        error = "executable program requires at least one graph";
        return false;
    }
    for (uint32_t index = 0; index < values.size(); ++index) {
        const ProgramValueSlot &value = values[index];
        if (value.id != index || value.name.empty() || value.type.empty() ||
            (value.dtype.empty() && !value.valueLayout)) {
            error = "executable program values must have canonical contiguous ids and complete types";
            return false;
        }
    }
    std::set<uint32_t> forwardValues;
    for (const ProgramGraph &graph : graphs)
        if (graph.direction == "forward") {
            forwardValues.insert(graph.arguments.begin(), graph.arguments.end());
            for (const ProgramNode &node : graph.nodes)
                forwardValues.insert(node.results.begin(), node.results.end());
        }
    std::set<std::string> directions;
    for (const ProgramGraph &graph : graphs) {
        if (graph.name.empty() || (graph.direction != "forward" && graph.direction != "backward") ||
            !directions.insert(graph.direction).second) {
            error = "executable program graphs must have unique forward/backward directions";
            return false;
        }
        std::set<uint32_t> seenNodes;
        std::set<uint32_t> availableValues(graph.arguments.begin(), graph.arguments.end());
        std::vector<std::optional<uint32_t>> producers(values.size());
        const auto validateValues = [&](const std::vector<uint32_t> &ids, std::string_view role) {
            for (uint32_t id : ids)
                if (id >= values.size()) {
                    error = "executable program " + std::string(role) + " references an unknown value";
                    return false;
                }
            return true;
        };
        if (!validateValues(graph.arguments, "argument") || !validateValues(graph.results, "result"))
            return false;
        for (uint32_t argument : graph.arguments)
            if (!values[argument].external) {
                error = "executable program graph argument must be externally bound";
                return false;
            }
        for (const ProgramNode &node : graph.nodes) {
            if (static_cast<size_t>(node.id) != seenNodes.size() || node.name.empty() ||
                (node.kind != "compute" && node.kind != "render") || stages.find(node.stage) == stages.end() ||
                !seenNodes.insert(node.id).second || !validateValues(node.operands, "node operand") ||
                !validateValues(node.results, "node result")) {
                error = "executable program node identity, kind, stage, or values are invalid";
                return false;
            }
            if (!std::is_sorted(node.dependencies.begin(), node.dependencies.end()) ||
                std::adjacent_find(node.dependencies.begin(), node.dependencies.end()) != node.dependencies.end()) {
                error = "executable program node dependencies must be canonical";
                return false;
            }
            for (uint32_t dependency : node.dependencies)
                if (!seenNodes.count(dependency) || dependency == node.id) {
                    error = "executable program node dependencies must be unique earlier nodes";
                    return false;
                }
            for (uint32_t operand : node.operands) {
                if (!availableValues.count(operand) &&
                    (graph.direction != "backward" || !forwardValues.count(operand))) {
                    error = "executable program node reads a value unavailable to its graph";
                    return false;
                }
                if (producers[operand] && std::find(node.dependencies.begin(), node.dependencies.end(),
                                                    *producers[operand]) == node.dependencies.end()) {
                    error = "executable program node is missing a producer dependency";
                    return false;
                }
            }
            for (uint32_t result : node.results) {
                if (producers[result]) {
                    error = "executable program value has multiple producers";
                    return false;
                }
                producers[result] = node.id;
                availableValues.insert(result);
            }
            std::set<std::string> bindingParameters;
            std::set<uint32_t> boundValues;
            for (const ProgramValueBinding &binding : node.bindings) {
                if (binding.parameter.empty() || binding.value >= values.size() ||
                    !bindingParameters.insert(binding.parameter).second ||
                    (std::find(node.operands.begin(), node.operands.end(), binding.value) == node.operands.end() &&
                     std::find(node.results.begin(), node.results.end(), binding.value) == node.results.end())) {
                    error = "executable program node value bindings are invalid";
                    return false;
                }
                boundValues.insert(binding.value);
            }
            if (node.kind == "compute")
                for (uint32_t value : node.operands)
                    if (!boundValues.count(value)) {
                        error = "executable program compute operand has no stage parameter binding";
                        return false;
                    }
            for (uint32_t value : node.results)
                if (!boundValues.count(value)) {
                    error = "executable program result has no stage parameter binding";
                    return false;
                }
            std::set<uint32_t> resourceValues;
            for (const ProgramResourceUse &use : node.resources) {
                if (use.value >= values.size() ||
                    (use.access != "read" && use.access != "write" && use.access != "read_write") ||
                    !resourceValues.insert(use.value).second) {
                    error = "executable program node resource uses are invalid";
                    return false;
                }
            }
            for (uint32_t operand : node.operands)
                if (!resourceValues.count(operand)) {
                    error = "executable program operand has no declared resource use";
                    return false;
                }
            for (uint32_t result : node.results)
                if (!resourceValues.count(result)) {
                    error = "executable program result has no declared resource use";
                    return false;
                }
        }
        for (uint32_t result : graph.results)
            if (!values[result].external && !producers[result]) {
                error = "executable program graph result has no producer";
                return false;
            }
    }
    if (!directions.count("forward")) {
        error = "executable program requires one forward graph";
        return false;
    }
    const auto graph = [&](std::string_view direction) -> const ProgramGraph * {
        const auto found = std::find_if(graphs.begin(), graphs.end(), [&](const ProgramGraph &candidate) {
            return candidate.direction == direction;
        });
        return found == graphs.end() ? nullptr : &*found;
    };
    const ProgramGraph *forward = graph("forward");
    const ProgramGraph *backward = graph("backward");
    const auto validateBoundary = [&](const std::vector<ProgramAdSignatureBinding> &bindings,
                                      const std::vector<uint32_t> &expected, std::string_view role) {
        if (bindings.size() != expected.size()) {
            error = "executable program AD " + std::string(role) + " does not match graph boundary";
            return false;
        }
        std::set<std::string> paths;
        for (size_t index = 0; index < bindings.size(); ++index)
            if (bindings[index].value != expected[index] || bindings[index].value >= values.size() ||
                bindings[index].path.empty() || !paths.insert(bindings[index].path).second) {
                error = "executable program AD " + std::string(role) + " binding is invalid";
                return false;
            }
        return true;
    };
    static const std::vector<uint32_t> empty;
    if (!validateBoundary(adSignature.inputs, forward->arguments, "inputs") ||
        !validateBoundary(adSignature.outputs, forward->results, "outputs") ||
        !validateBoundary(adSignature.cotangents, backward ? backward->arguments : empty, "cotangents") ||
        !validateBoundary(adSignature.gradients, backward ? backward->results : empty, "gradients") ||
        adSignature.captures != backwardCaptures()) {
        if (error.empty())
            error = "executable program AD captures do not match backward graph";
        return false;
    }
    const auto validateDerivative = [&](const ProgramAdSignatureBinding &derivative,
                                        const std::vector<ProgramAdSignatureBinding> &primals, std::string_view role) {
        const auto primal = std::find_if(primals.begin(), primals.end(), [&](const ProgramAdSignatureBinding &value) {
            return value.path == derivative.path;
        });
        if (primal == primals.end()) {
            error = "executable program AD " + std::string(role) + " path has no primal boundary";
            return false;
        }
        const ProgramValueSlot &primalValue = values[primal->value];
        const ProgramValueSlot &derivativeValue = values[derivative.value];
        if (primalValue.dtype.empty() || derivativeValue.dtype.empty()) {
            if (!primalValue.valueLayout || !derivativeValue.valueLayout ||
                primalValue.valueLayout->leaves.size() != derivativeValue.valueLayout->leaves.size()) {
                error = "executable program AD aggregate derivative ABI does not match primal boundary";
                return false;
            }
            for (size_t index = 0; index < primalValue.valueLayout->leaves.size(); ++index) {
                const ValueLeaf &primalLeaf = primalValue.valueLayout->leaves[index];
                const ValueLeaf &derivativeLeaf = derivativeValue.valueLayout->leaves[index];
                const bool dtypeOk = primalLeaf.dtype == derivativeLeaf.dtype ||
                                     (primalLeaf.dtype == "f16" && derivativeLeaf.dtype == "f32");
                std::vector<uint64_t> tangentShape(primalValue.shape);
                tangentShape.insert(tangentShape.end(), primalLeaf.shape.begin(), primalLeaf.shape.end());
                if (!dtypeOk || (derivativeLeaf.shape != primalLeaf.shape && derivativeLeaf.shape != tangentShape)) {
                    error = "executable program AD aggregate derivative ABI does not match primal boundary";
                    return false;
                }
            }
            return true;
        }
        if (primalValue.shape != derivativeValue.shape ||
            (primalValue.dtype != derivativeValue.dtype &&
             !(primalValue.dtype == "f16" && derivativeValue.dtype == "f32"))) {
            error = "executable program AD derivative ABI does not match primal boundary";
            return false;
        }
        return true;
    };
    if (adSignature.declared) {
        for (const ProgramAdSignatureBinding &cotangent : adSignature.cotangents)
            if (!validateDerivative(cotangent, adSignature.outputs, "cotangent"))
                return false;
        for (const ProgramAdSignatureBinding &gradient : adSignature.gradients)
            if (!validateDerivative(gradient, adSignature.inputs, "gradient"))
                return false;
    }
    return true;
}

std::vector<uint32_t> ExecutableProgram::backwardCaptures() const {
    std::set<uint32_t> forwardValues;
    for (const ProgramGraph &graph : graphs) {
        if (graph.direction != "forward")
            continue;
        forwardValues.insert(graph.arguments.begin(), graph.arguments.end());
        for (const ProgramNode &node : graph.nodes)
            forwardValues.insert(node.results.begin(), node.results.end());
    }
    std::vector<uint32_t> captures;
    for (const ProgramGraph &graph : graphs) {
        if (graph.direction != "backward")
            continue;
        std::set<uint32_t> available(graph.arguments.begin(), graph.arguments.end());
        std::set<uint32_t> captured;
        for (const ProgramNode &node : graph.nodes) {
            for (uint32_t operand : node.operands)
                if (!available.count(operand) && forwardValues.count(operand) && captured.insert(operand).second)
                    captures.push_back(operand);
            available.insert(node.results.begin(), node.results.end());
        }
    }
    return captures;
}

std::vector<uint32_t> ExecutableProgram::residualCaptures() const {
    if (adSignature.declared)
        return adSignature.captures;
    return backwardCaptures();
}

void markProgramGraphValues(const ProgramGraph &graph, std::vector<char> &live) {
    const auto mark = [&](uint32_t value) {
        if (value < live.size())
            live[value] = 1;
    };
    for (uint32_t value : graph.arguments)
        mark(value);
    for (uint32_t value : graph.captures)
        mark(value);
    for (uint32_t value : graph.results)
        mark(value);
    for (const ProgramNode &node : graph.nodes) {
        for (uint32_t value : node.operands)
            mark(value);
        for (uint32_t value : node.results)
            mark(value);
        for (const ProgramValueBinding &binding : node.bindings)
            mark(binding.value);
        for (const ProgramResourceUse &resource : node.resources)
            mark(resource.value);
    }
}

bool parseExecutableProgram(const nlohmann::json &value, ExecutableProgram &program, std::string &error) {
    if (!value.is_object() || !hasOnlyKeys(value, {"values", "graphs", "signature"}) || !value.contains("values") ||
        !value["values"].is_array() || !value.contains("graphs") || !value["graphs"].is_array()) {
        error = "executable program must contain values and graphs";
        return false;
    }
    const auto parseIds = [&](const nlohmann::json &array, std::vector<uint32_t> &ids, std::string_view field) {
        if (!array.is_array()) {
            error = "executable program " + std::string(field) + " must be an array";
            return false;
        }
        for (const nlohmann::json &item : array) {
            uint32_t id = UINT32_MAX;
            if (!parseUint32(item, id)) {
                error = "executable program " + std::string(field) + " must contain uint32 ids";
                return false;
            }
            ids.push_back(id);
        }
        return true;
    };
    for (const nlohmann::json &row : value["values"]) {
        if (!row.is_object() ||
            !hasOnlyKeys(row, {"id", "name", "type", "dtype", "shape", "value_layout", "external", "output"}) ||
            !row.contains("id") || !row.contains("name") || !row["name"].is_string() || !row.contains("type") ||
            !row["type"].is_string() || !row.contains("dtype") || !row["dtype"].is_string() || !row.contains("shape") ||
            !row["shape"].is_array() || !row.contains("external") || !row["external"].is_boolean() ||
            !row.contains("output") || !row["output"].is_boolean()) {
            error = "executable program value record is invalid";
            return false;
        }
        ProgramValueSlot slot;
        if (!parseUint32(row["id"], slot.id)) {
            error = "executable program value id must be uint32";
            return false;
        }
        slot.name = row["name"].get<std::string>();
        slot.type = row["type"].get<std::string>();
        slot.dtype = row["dtype"].get<std::string>();
        slot.external = row["external"].get<bool>();
        slot.output = row["output"].get<bool>();
        for (const nlohmann::json &extentValue : row["shape"]) {
            uint64_t extent = 0;
            if (!parseUint64(extentValue, extent)) {
                error = "executable program value shape must contain unsigned extents";
                return false;
            }
            slot.shape.push_back(extent);
        }
        if (row.contains("value_layout")) {
            slot.valueLayout = std::make_shared<ValueLayout>();
            if (!parsePipelineValueLayout(row["value_layout"], *slot.valueLayout, error)) {
                error = "executable program value has invalid canonical layout: " + error;
                return false;
            }
        }
        program.values.push_back(std::move(slot));
    }
    for (const nlohmann::json &graphValue : value["graphs"]) {
        if (!graphValue.is_object() ||
            !hasOnlyKeys(graphValue, {"name", "direction", "arguments", "results", "nodes"}) ||
            !graphValue.contains("name") || !graphValue["name"].is_string() || !graphValue.contains("direction") ||
            !graphValue["direction"].is_string() || !graphValue.contains("arguments") ||
            !graphValue.contains("results") || !graphValue.contains("nodes") || !graphValue["nodes"].is_array()) {
            error = "executable program graph record is invalid";
            return false;
        }
        ProgramGraph graph;
        graph.name = graphValue["name"].get<std::string>();
        graph.direction = graphValue["direction"].get<std::string>();
        if (!parseIds(graphValue["arguments"], graph.arguments, "graph arguments") ||
            !parseIds(graphValue["results"], graph.results, "graph results"))
            return false;
        for (const nlohmann::json &nodeValue : graphValue["nodes"]) {
            if (!nodeValue.is_object() ||
                !hasOnlyKeys(nodeValue, {"id", "name", "kind", "stage", "operands", "results", "dependencies",
                                         "bindings", "resources", "grid"}) ||
                !nodeValue.contains("id") || !nodeValue.contains("name") || !nodeValue["name"].is_string() ||
                !nodeValue.contains("kind") || !nodeValue["kind"].is_string() || !nodeValue.contains("stage") ||
                !nodeValue["stage"].is_string() || !nodeValue.contains("operands") || !nodeValue.contains("results") ||
                !nodeValue.contains("dependencies") || !nodeValue.contains("resources") ||
                !nodeValue["resources"].is_array() || !nodeValue.contains("bindings") ||
                !nodeValue["bindings"].is_array() || !nodeValue.contains("grid") || !nodeValue["grid"].is_array() ||
                nodeValue["grid"].size() != 3) {
                error = "executable program node record is invalid";
                return false;
            }
            ProgramNode node;
            if (!parseUint32(nodeValue["id"], node.id)) {
                error = "executable program node id must be uint32";
                return false;
            }
            node.name = nodeValue["name"].get<std::string>();
            node.kind = nodeValue["kind"].get<std::string>();
            node.stage = nodeValue["stage"].get<std::string>();
            if (!parseIds(nodeValue["operands"], node.operands, "node operands") ||
                !parseIds(nodeValue["results"], node.results, "node results") ||
                !parseIds(nodeValue["dependencies"], node.dependencies, "node dependencies"))
                return false;
            for (unsigned index = 0; index < 3; ++index)
                if (!parseUint64(nodeValue["grid"][index], node.grid[index]) || !node.grid[index]) {
                    error = "executable program node grid must contain three positive extents";
                    return false;
                }
            for (const nlohmann::json &bindingValue : nodeValue["bindings"]) {
                if (!bindingValue.is_object() || !hasOnlyKeys(bindingValue, {"parameter", "value"}) ||
                    !bindingValue.contains("parameter") || !bindingValue["parameter"].is_string() ||
                    !bindingValue.contains("value")) {
                    error = "executable program node value binding is invalid";
                    return false;
                }
                ProgramValueBinding binding;
                binding.parameter = bindingValue["parameter"].get<std::string>();
                if (!parseUint32(bindingValue["value"], binding.value)) {
                    error = "executable program node binding value must be uint32";
                    return false;
                }
                node.bindings.push_back(std::move(binding));
            }
            for (const nlohmann::json &resourceValue : nodeValue["resources"]) {
                if (!resourceValue.is_object() || !hasOnlyKeys(resourceValue, {"value", "access"}) ||
                    !resourceValue.contains("value") || !resourceValue.contains("access") ||
                    !resourceValue["access"].is_string()) {
                    error = "executable program resource use is invalid";
                    return false;
                }
                ProgramResourceUse use;
                if (!parseUint32(resourceValue["value"], use.value)) {
                    error = "executable program resource value must be uint32";
                    return false;
                }
                use.access = resourceValue["access"].get<std::string>();
                node.resources.push_back(std::move(use));
            }
            graph.nodes.push_back(std::move(node));
        }
        program.graphs.push_back(std::move(graph));
    }
    const auto parseSignatureBindings = [&](const nlohmann::json &rows,
                                            std::vector<ProgramAdSignatureBinding> &bindings, std::string_view role) {
        if (!rows.is_array()) {
            error = "executable program AD " + std::string(role) + " must be an array";
            return false;
        }
        for (const nlohmann::json &row : rows) {
            if (!row.is_object() || !hasOnlyKeys(row, {"value", "path"}) || !row.contains("value") ||
                !row.contains("path") || !row["path"].is_string()) {
                error = "executable program AD " + std::string(role) + " binding is invalid";
                return false;
            }
            ProgramAdSignatureBinding binding;
            if (!parseUint32(row["value"], binding.value)) {
                error = "executable program AD " + std::string(role) + " value must be uint32";
                return false;
            }
            binding.path = row["path"].get<std::string>();
            bindings.push_back(std::move(binding));
        }
        return true;
    };
    if (value.contains("signature")) {
        program.adSignature.declared = true;
        const nlohmann::json &signature = value["signature"];
        if (!signature.is_object() ||
            !hasOnlyKeys(signature, {"inputs", "outputs", "cotangents", "gradients", "captures"}) ||
            !signature.contains("inputs") || !signature.contains("outputs") || !signature.contains("cotangents") ||
            !signature.contains("gradients") || !signature.contains("captures") ||
            !parseSignatureBindings(signature["inputs"], program.adSignature.inputs, "inputs") ||
            !parseSignatureBindings(signature["outputs"], program.adSignature.outputs, "outputs") ||
            !parseSignatureBindings(signature["cotangents"], program.adSignature.cotangents, "cotangents") ||
            !parseSignatureBindings(signature["gradients"], program.adSignature.gradients, "gradients") ||
            !parseIds(signature["captures"], program.adSignature.captures, "AD captures")) {
            if (error.empty())
                error = "executable program AD signature is invalid";
            return false;
        }
    } else {
        const auto appendBoundary = [&](const std::vector<uint32_t> &ids,
                                        std::vector<ProgramAdSignatureBinding> &bindings) {
            for (uint32_t id : ids)
                bindings.push_back({id, id < program.values.size() ? publicPath(program.values[id].name) : ""});
        };
        for (const ProgramGraph &graph : program.graphs)
            if (graph.direction == "forward") {
                appendBoundary(graph.arguments, program.adSignature.inputs);
                appendBoundary(graph.results, program.adSignature.outputs);
            } else if (graph.direction == "backward") {
                appendBoundary(graph.arguments, program.adSignature.cotangents);
                appendBoundary(graph.results, program.adSignature.gradients);
            }
        program.adSignature.captures = program.backwardCaptures();
    }
    return true;
}

} // namespace vernon::runtime
