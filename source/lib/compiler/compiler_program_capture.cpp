#include "compiler_program_capture.h"

#include "compiler_json.h"

#include <map>

namespace vernon::compiler {
namespace {

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
        const llvm::json::Object *value = findJsonObjectByIntegerId(values, current);
        if (!value)
            return;
        if (const std::optional<int64_t> like = value->getInteger("like"))
            current = *like;
        else
            return;
    }
}

} // namespace

void rebuildProgramDependencies(llvm::json::Object &execution, llvm::StringRef graphName) {
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

void collectProgramBackwardCaptures(const llvm::json::Object &execution, std::set<int64_t> &captures) {
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
                        for (llvm::StringRef field : {"operands", "results"})
                            if (const llvm::json::Array *ids = node->getArray(field))
                                for (const llvm::json::Value &id : *ids)
                                    if (std::optional<int64_t> value = id.getAsInteger())
                                        forwardValues.insert(*value);
                    }
        } else if (graph->getString("direction") == "backward")
            backward = graph;
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
                            const llvm::json::Object *row =
                                values ? findJsonObjectByIntegerId(*values, *value) : nullptr;
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

void rebuildProgramCaptures(llvm::json::Object &execution) {
    llvm::json::Object *signature = execution.getObject("signature");
    if (!signature)
        return;
    std::set<int64_t> captures;
    collectProgramBackwardCaptures(execution, captures);
    llvm::json::Array reflected;
    for (int64_t capture : captures)
        reflected.emplace_back(capture);
    (*signature)["captures"] = std::move(reflected);
}

} // namespace vernon::compiler
