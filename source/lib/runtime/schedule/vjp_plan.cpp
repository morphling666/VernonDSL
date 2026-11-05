#include "runtime/schedule/vjp_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <set>

namespace vernon::schedule {
namespace {

using Planned = Result<VjpPlan, std::string>;

Planned failPlan(std::string message) { return Planned{err(std::move(message))}; }

const nlohmann::json *findGraph(const nlohmann::json &program, const std::string &name) {
    for (const nlohmann::json &graph : program.at("graphs"))
        if (graph.value("name", std::string()) == name)
            return &graph;
    return nullptr;
}

bool isStorage(const nlohmann::json &value) { return value.contains("storage") && !value.at("storage").is_null(); }

// The Storages and Values the graph's nodes bind as arguments or take workgroup counts from; a VJP graph may capture
// Values none of them read. The fragment builder makes a slot of the same captured scalars.
void boundByNodes(const nlohmann::json &graph, std::set<uint32_t> &storages, std::set<uint32_t> &values) {
    for (const nlohmann::json &node : graph.at("nodes")) {
        const nlohmann::json accesses = node.value("accesses", nlohmann::json::array());
        for (const nlohmann::json &binding : node.value("bindings", nlohmann::json::array())) {
            const std::string tag = binding.value("tag", std::string());
            if (tag == "resource" && binding.contains("access") && binding.at("access").is_number_unsigned())
                storages.insert(accesses.at(binding.at("access").get<size_t>()).at("storage").get<uint32_t>());
            else if (tag == "value")
                for (const nlohmann::json &projection : binding.value("projections", nlohmann::json::array()))
                    values.insert(projection.at("value").get<uint32_t>());
        }
        const nlohmann::json operation = node.value("operation", nlohmann::json::object());
        for (const nlohmann::json &axis : operation.value("workgroups", nlohmann::json::array()))
            if (axis.is_object() && axis.contains("control") && axis.at("control").contains("value"))
                values.insert(axis.at("control").at("value").get<uint32_t>());
    }
}

// The borrowed or captured Storages whose extents size the graph's allocations, through the byte lengths of the
// owned Storages they name; the graph's schedule reads those extents from its arguments, a captured Storage's from
// its residual or argument.
void extentSources(const nlohmann::json &program, const nlohmann::json &graph, const std::set<uint32_t> &captured,
                   std::set<uint32_t> &sources) {
    const nlohmann::json &storages = program.at("storages");
    const nlohmann::json &values = program.at("values");
    std::vector<uint32_t> pending;
    for (const nlohmann::json &row : graph.at("inputs"))
        if (row.value("tag", std::string()) == "allocation")
            pending.push_back(row.at("storage").get<uint32_t>());
    std::set<uint32_t> seen;
    while (!pending.empty()) {
        const uint32_t storage = pending.back();
        pending.pop_back();
        if (!seen.insert(storage).second)
            continue;
        const nlohmann::json &row = storages.at(storage);
        if (row.value("ownership", std::string()) == "borrowed" || captured.count(storage)) {
            sources.insert(storage);
            continue;
        }
        const nlohmann::json length = row.at("descriptor").value("byte_length", nlohmann::json());
        if (!length.is_array())
            continue;
        for (const nlohmann::json &component : length) {
            if (!component.is_object() || !component.contains("dimension"))
                continue;
            const nlohmann::json control = component.at("dimension").value("control", nlohmann::json::object());
            if (!control.contains("value"))
                continue;
            const nlohmann::json &value = values.at(control.at("value").get<size_t>());
            if (isStorage(value))
                pending.push_back(value.at("storage").get<uint32_t>());
        }
    }
}

} // namespace

Planned planVjp(const nlohmann::json &program, bool cpuTapes) {
    if (!program.contains("graphs") || program.at("graphs").size() != 2)
        return failPlan("a linked VJP takes a Program of a forward and a backward graph");
    const nlohmann::json *forward = findGraph(program, "forward");
    const nlohmann::json *backward = findGraph(program, "backward");
    if (!forward || !backward)
        return failPlan("a linked VJP takes a Program of a forward and a backward graph");
    const nlohmann::json &storages = program.at("storages");
    const nlohmann::json &values = program.at("values");

    std::set<uint32_t> allocated;
    for (const nlohmann::json &row : forward->at("inputs"))
        if (row.value("tag", std::string()) == "allocation")
            allocated.insert(row.at("storage").get<uint32_t>());
    // The forward boundary slot passing each Storage and scalar Value; an output that republishes an input's Storage
    // passes it through that input.
    std::map<uint32_t, uint32_t> boundaryOfStorage;
    std::map<uint32_t, uint32_t> boundaryOfScalar;
    for (const nlohmann::json &row : program.at("abi").at("boundary_slots")) {
        const nlohmann::json &value = values.at(row.at("value").get<size_t>());
        if (value.value("origin", nlohmann::json::object()).value("graph", std::string()) != "forward")
            continue;
        const auto id = row.at("id").get<uint32_t>();
        if (row.contains("storage_id") && !row.at("storage_id").is_null())
            boundaryOfStorage.emplace(row.at("storage_id").get<uint32_t>(), id);
        if (row.value("category", std::string()) == "value")
            boundaryOfScalar.emplace(row.at("value").get<uint32_t>(), id);
    }

    std::set<uint32_t> boundStorages;
    std::set<uint32_t> boundValues;
    boundByNodes(*backward, boundStorages, boundValues);
    std::set<uint32_t> forwardStorages;
    std::set<uint32_t> forwardValues;
    boundByNodes(*forward, forwardStorages, forwardValues);
    std::set<uint32_t> capturedStorages;
    for (const nlohmann::json &capture : backward->value("captures", nlohmann::json::array()))
        if (const nlohmann::json &value = values.at(capture.at("value").get<size_t>()); isStorage(value))
            capturedStorages.insert(value.at("storage").get<uint32_t>());
    extentSources(program, *backward, capturedStorages, boundStorages);
    const auto isTape = [&](uint32_t storage) {
        return storages.at(storage).at("descriptor").value("contract", std::string()) == "vernon.ad_tape";
    };

    VjpPlan plan;
    std::set<uint32_t> captured;
    std::set<uint32_t> tapes;
    for (const nlohmann::json &capture : backward->value("captures", nlohmann::json::array())) {
        const auto valueId = capture.at("value").get<uint32_t>();
        const nlohmann::json &value = values.at(valueId);
        if (!isStorage(value)) {
            if (!boundValues.count(valueId))
                continue;
            const auto found = boundaryOfScalar.find(valueId);
            if (found == boundaryOfScalar.end())
                return failPlan("the backward graph reads a scalar the forward arguments do not pass");
            plan.scalarBoundaries.push_back(found->second);
            continue;
        }
        const auto storage = value.at("storage").get<uint32_t>();
        if (isTape(storage)) {
            if (cpuTapes || forwardStorages.count(storage))
                tapes.insert(storage);
        } else if (boundStorages.count(storage)) {
            captured.insert(storage);
        }
    }
    for (uint32_t storage : captured) {
        if (storages.at(storage).at("descriptor").value("tag", std::string()) != "buffer")
            return failPlan("the backward graph reads a forward Storage that is not a buffer");
        const auto boundary = boundaryOfStorage.find(storage);
        if (boundary != boundaryOfStorage.end()) {
            plan.capturedBoundary.push_back(boundary->second);
        } else if (allocated.count(storage)) {
            plan.capturedBoundary.push_back(UINT32_MAX);
            plan.residuals.push_back(storage);
        } else {
            return failPlan("the backward graph reads a forward Storage that is neither an argument nor an allocation");
        }
        plan.captured.push_back(storage);
    }
    plan.tapes.assign(tapes.begin(), tapes.end());

    std::vector<uint32_t> exposed = plan.residuals;
    exposed.insert(exposed.end(), plan.tapes.begin(), plan.tapes.end());
    std::map<uint32_t, size_t> passOfNode;
    for (const nlohmann::json &node : forward->at("nodes")) {
        passOfNode.emplace(node.at("id").get<uint32_t>(), plan.passes.size());
        plan.passes.push_back({node.at("id").get<uint32_t>(), node.value("name", std::string()), {}});
    }
    // A Storage several forward nodes write is credited to the node that produced the Value the backward graph reads.
    std::set<uint32_t> credited;
    for (const nlohmann::json &capture : backward->value("captures", nlohmann::json::array())) {
        const nlohmann::json &value = values.at(capture.at("value").get<size_t>());
        const nlohmann::json origin = value.value("origin", nlohmann::json::object());
        if (!isStorage(value) || origin.value("graph", std::string()) != "forward" ||
            origin.value("tag", std::string()) != "node_result")
            continue;
        const auto at = std::find(exposed.begin(), exposed.end(), value.at("storage").get<uint32_t>());
        const auto pass = passOfNode.find(origin.at("node").get<uint32_t>());
        if (at == exposed.end() || pass == passOfNode.end() ||
            !credited.insert(static_cast<uint32_t>(at - exposed.begin())).second)
            continue;
        plan.passes[pass->second].exposed.push_back(static_cast<uint32_t>(at - exposed.begin()));
    }
    return Planned{ok(std::move(plan))};
}

} // namespace vernon::schedule
