#include "compiler_program_compute.h"

#include "VernonProgramCapabilities.h"
#include "compiler_json.h"
#include "compiler_program_stage.h"
#include "compiler_program_storage.h"

#include <algorithm>
#include <set>
#include <vector>

namespace vernon::compiler {
namespace {

void collectPortableSlots(const llvm::json::Object &row, std::set<int64_t> &slots) {
    if (std::optional<int64_t> slot = row.getInteger("vernon.binding"))
        slots.insert(*slot);
    if (const llvm::json::Array *leaves = row.getArray("storage_leaves"))
        for (const llvm::json::Value &leafValue : *leaves)
            if (const llvm::json::Object *leaf = leafValue.getAsObject())
                if (std::optional<int64_t> slot = leaf->getInteger("binding"))
                    slots.insert(*slot);
}

} // namespace

bool isProgramTapeCarrierRole(llvm::StringRef role) {
    return program_plan::tapeCarrierFromRoleName(std::string_view(role.data(), role.size())).has_value();
}

bool isProgramKernelHiddenBuiltin(llvm::StringRef builtin) {
    return builtin == "global_invocation_id" || builtin == "local_invocation_id" || builtin == "workgroup_id";
}

bool isProgramKernelTapeBuiltin(llvm::StringRef builtin) {
    return builtin == "ad_tape_allocator" || builtin == "ad_tape_root_region";
}

bool prepareCanonicalComputeInterface(const llvm::json::Object &compiledEntry, const llvm::json::Array &rawValues,
                                      const ProgramNodeBindingIndex &boundValues, int64_t &nextPortableSlot,
                                      std::string &error) {
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
            const std::optional<llvm::StringRef> role = row->getString("vernon.autodiff_role");
            if (!role || !isProgramTapeCarrierRole(*role))
                ++logicalInterfaceCount;
        }
        return true;
    };
    std::map<std::string, int64_t> namedBound;
    for (const auto &[name, projections] : boundValues) {
        if (projections.empty())
            continue;
        const int64_t value = projections.front().value;
        const llvm::json::Object *logical = findJsonObjectByIntegerId(rawValues, value);
        const llvm::StringRef type = logical ? logical->getString("type").value_or("") : "";
        if (name == "tape" || isProgramAdTapeType(type))
            continue;
        namedBound.emplace(name, value);
    }
    if (!collect(compiledEntry.getArray("arguments")) || !collect(compiledEntry.getArray("results")) ||
        logicalInterfaceCount != namedBound.size()) {
        error = "compiled compute ABI source names must be unique and exactly cover logical bindings";
        return false;
    }
    for (const auto &[name, value] : namedBound)
        if (!endpointNames.count(name) || !findJsonObjectByIntegerId(rawValues, value)) {
            error = "compiled compute ABI does not exactly cover logical endpoint '" + name + "'";
            return false;
        }

    std::set<int64_t> portableSlots;
    if (const llvm::json::Array *rows = compiledEntry.getArray("arguments"))
        for (const llvm::json::Value &rowValue : *rows)
            if (const llvm::json::Object *row = rowValue.getAsObject())
                if (!row->getString("vernon.builtin") && !row->getString("builtin"))
                    collectPortableSlots(*row, portableSlots);
    nextPortableSlot = portableSlots.empty() ? 0 : *portableSlots.rbegin() + 1;
    return true;
}

bool buildCanonicalComputeEndpoints(const llvm::json::Object &compiledEntry, const llvm::json::Array &rawValues,
                                    const ProgramNodeBindingIndex &boundValues,
                                    std::map<int64_t, ProgramLogicalResource> &resources, int64_t nextPortableSlot,
                                    ProgramComputeEndpointPlan &plan, std::string &error) {
    std::set<int64_t> metadataArguments;
    std::map<int64_t, int64_t> metadataRanks;
    if (const llvm::json::Object *metadata = compiledEntry.getObject("metadata_carrier")) {
        const llvm::json::Array *fields = metadata->getArray("fields");
        if (!fields || fields->empty()) {
            error = "compiled compute metadata carrier has no semantic fields";
            return false;
        }
        llvm::json::Array semanticFields;
        for (const auto &[ordinal, fieldValue] : llvm::enumerate(*fields)) {
            const llvm::json::Object *field = fieldValue.getAsObject();
            const std::optional<int64_t> argument = field ? field->getInteger("argument") : std::nullopt;
            const std::optional<llvm::StringRef> kind = field ? field->getString("kind") : std::nullopt;
            if (!argument || *argument < 0 || !kind || (*kind != "offset" && *kind != "extent" && *kind != "stride") ||
                field->getInteger("ordinal") != static_cast<int64_t>(ordinal) ||
                field->getString("units") != "logical_elements") {
                error = "compiled compute metadata carrier has an invalid semantic field";
                return false;
            }
            metadataArguments.insert(*argument);
            llvm::json::Object projected{{"ordinal", static_cast<int64_t>(ordinal)},
                                         {"argument", *argument},
                                         {"kind", kind->str()},
                                         {"units", "logical_elements"}};
            if (*kind == "extent" || *kind == "stride") {
                const std::optional<int64_t> dimension = field->getInteger("dimension");
                if (!dimension || *dimension < 0) {
                    error = "compiled compute metadata carrier has an invalid field dimension";
                    return false;
                }
                projected["dimension"] = *dimension;
                metadataRanks[*argument] = std::max(metadataRanks[*argument], *dimension + 1);
            } else if (field->get("dimension")) {
                error = "compiled compute offset metadata cannot have a dimension";
                return false;
            }
            semanticFields.emplace_back(std::move(projected));
        }
        plan.semanticMetadataCarrier = llvm::json::Object{{"fields", std::move(semanticFields)}};
        plan.physicalMetadataCarrier = copyJsonObject(*metadata);
        plan.physicalMetadataCarrier->erase("fields");
    }
    std::map<int64_t, int64_t> accessByValue;
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
            if (isProgramKernelTapeBuiltin(*builtin)) {
                plan.implementationEndpoints.emplace_back(
                    compiledProgramEndpointAbi(row, "compute", interfaceKind, endpointIndex));
                return true;
            }
            if (isProgramKernelHiddenBuiltin(*builtin))
                return true;
            const llvm::json::Object *layout = row.getObject("value_layout");
            if (!layout) {
                error = "compiled compute system value has no canonical layout (" + builtin->str() + ")";
                return false;
            }
            llvm::json::Array abi;
            abi.emplace_back(
                llvm::json::Object{{"semantic", "value"},
                                   {"carrier", programValueCarrier("constant_region", nextPortableSlot++, *layout)}});
            plan.endpoints.emplace_back(llvm::json::Object{
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
            plan.implementationEndpoints.emplace_back(
                compiledProgramEndpointAbi(row, "compute", interfaceKind, endpointIndex));
            return true;
        }
        const std::optional<llvm::StringRef> source = row.getString("vernon.source_name");
        auto binding = source ? boundValues.find(source->str()) : boundValues.end();
        if (!source || binding == boundValues.end()) {
            error = "compiled compute endpoint is not present in logical bindings";
            return false;
        }
        std::vector<ProgramLogicalBinding> logicalBindings = binding->second;
        if (logicalBindings.empty()) {
            error = "compiled compute endpoint has no logical projections";
            return false;
        }
        if (row.get("vernon.program_storage_shape") && logicalBindings.size() > 1) {
            const auto firstResource = resources.find(logicalBindings.front().value);
            if (firstResource == resources.end() || firstResource->second.storage < 0) {
                error = "Program storage-shape carrier does not project a canonical Storage";
                return false;
            }
            const int64_t storage = firstResource->second.storage;
            for (const ProgramLogicalBinding &projection : llvm::drop_begin(logicalBindings))
                if (const auto resource = resources.find(projection.value);
                    resource == resources.end() || resource->second.storage != storage) {
                    error = "Program storage-shape carrier combines distinct canonical Storages";
                    return false;
                }
            logicalBindings.resize(1);
        }
        const int64_t valueId = logicalBindings.front().value;
        const llvm::json::Object *logicalValue = findJsonObjectByIntegerId(rawValues, valueId);
        auto resourceIt = resources.find(valueId);
        const bool resource = resourceIt != resources.end();
        const llvm::StringRef endpointKind = row.getString("kind").value_or("");
        if (endpointKind == "sampler") {
            const program_capabilities::Entry &capability =
                program_capabilities::get(program_capabilities::Id::ComputeSamplerBinding);
            error = std::string(capability.diagnosticCode) + ": " + std::string(capability.diagnostic);
            return false;
        }
        const bool opaqueResourceEndpoint = endpointKind == "image" || endpointKind == "sampler";
        const std::optional<llvm::StringRef> autodiffRole = row.getString("vernon.autodiff_role");
        const bool tapeCarrier = logicalValue && isProgramAdTapeType(logicalValue->getString("type").value_or("")) &&
                                 autodiffRole && isProgramTapeCarrierRole(*autodiffRole);
        const llvm::json::Object *logicalLayout = logicalValue ? logicalValue->getObject("value_layout") : nullptr;
        const llvm::json::Array *logicalShape = logicalValue ? logicalValue->getArray("shape") : nullptr;
        const llvm::json::Array *physicalShape =
            row.getArray("source_shape") ? row.getArray("source_shape") : row.getArray("shape");
        const bool resourceBackedValue = !resource && endpointKind == "tensor" && autodiffRole == "gradient" &&
                                         logicalShape && logicalShape->empty() && physicalShape &&
                                         physicalShape->empty();
        const llvm::json::Object *wholeLayout = programEndpointLayout(row, resource || resourceBackedValue);
        if (logicalBindings.size() > 1) {
            const llvm::json::Array *physicalLeaves = wholeLayout ? wholeLayout->getArray("leaves") : nullptr;
            if (interfaceKind != "result" || !autodiffRole || *autodiffRole != "gradient" || !physicalLeaves ||
                physicalLeaves->size() != logicalBindings.size()) {
                error = "aggregate physical endpoint '" + source->str() + "' has " +
                        std::to_string(physicalLeaves ? physicalLeaves->size() : 0) + " physical leaves but " +
                        std::to_string(logicalBindings.size()) + " canonical projections (Values";
                for (const ProgramLogicalBinding &projection : logicalBindings)
                    error += " " + std::to_string(projection.value);
                error += ")";
                return false;
            }
            llvm::json::Array projections;
            std::set<int64_t> coveredPhysicalLeaves;
            for (const ProgramLogicalBinding &projection : logicalBindings) {
                if (projection.physicalLeaf < 0 ||
                    static_cast<size_t>(projection.physicalLeaf) >= physicalLeaves->size() ||
                    !coveredPhysicalLeaves.insert(projection.physicalLeaf).second) {
                    error = "aggregate physical endpoint has an invalid or duplicate leaf projection";
                    return false;
                }
                const llvm::json::Object *value = findJsonObjectByIntegerId(rawValues, projection.value);
                const llvm::json::Object *layout = value ? value->getObject("value_layout") : nullptr;
                const llvm::json::Array *leaves = layout ? layout->getArray("leaves") : nullptr;
                const size_t logicalLeaf = projection.leaf ? static_cast<size_t>(*projection.leaf) : 0;
                const llvm::json::Object *canonicalLeaf =
                    leaves && logicalLeaf < leaves->size() ? (*leaves)[logicalLeaf].getAsObject() : nullptr;
                const llvm::json::Object *implementationLeaf =
                    (*physicalLeaves)[static_cast<size_t>(projection.physicalLeaf)].getAsObject();
                if (!canonicalLeaf || !implementationLeaf ||
                    canonicalLeaf->getString("dtype") != implementationLeaf->getString("dtype")) {
                    error = "aggregate physical endpoint leaf dtype does not match its canonical projection";
                    return false;
                }
                llvm::json::Object item{
                    {"value", projection.value}, {"physical_leaf", projection.physicalLeaf}, {"direction", "result"}};
                if (projection.leaf)
                    item["leaf"] = *projection.leaf;
                projections.emplace_back(std::move(item));
            }
            const std::optional<int64_t> reflectedSlot = row.getInteger("vernon.binding");
            const int64_t slot = reflectedSlot ? *reflectedSlot : nextPortableSlot++;
            llvm::json::Array abi{llvm::json::Object{
                {"semantic", "value"}, {"carrier", programValueCarrier("value_slot", slot, *wholeLayout)}}};
            plan.endpoints.emplace_back(llvm::json::Object{
                {"tag", "value"},
                {"module", "compute"},
                {"interface", interfaceKind.str()},
                {"index", endpointIndex},
                {"role", autodiffRole->str()},
                {"type", row.getString("type").value_or("").str()},
                {"layout_hash", wholeLayout->getString("layout_hash").value_or("").str()},
                {"transport", "by_value"},
                {"access", "read"},
                {"abi", llvm::json::Object{{"bindings", std::move(abi)}}},
            });
            plan.endpointBindings.emplace_back(llvm::json::Object{{"module", "compute"},
                                                                  {"interface", interfaceKind.str()},
                                                                  {"index", endpointIndex},
                                                                  {"tag", "value"},
                                                                  {"projections", std::move(projections)}});
            plan.implementationEndpoints.emplace_back(
                compiledProgramEndpointAbi(row, "compute", interfaceKind, endpointIndex));
            return true;
        }
        const bool opaqueMatches = opaqueResourceEndpoint && resource && logicalValue &&
                                   logicalValue->getString("type") == row.getString("type");
        const bool byteValueMatches =
            !opaqueResourceEndpoint && logicalValue && wholeLayout && logicalLayout && logicalShape &&
            wholeLayout->getString("layout_hash") == logicalLayout->getString("layout_hash") &&
            (!resource || programAbiShapesCompatible(logicalShape, physicalShape ? physicalShape : logicalShape));
        bool projectedLeafMatches = false;
        const llvm::json::Array *logicalLeaves = logicalLayout ? logicalLayout->getArray("leaves") : nullptr;
        const llvm::json::Array *physicalLeaves = wholeLayout ? wholeLayout->getArray("leaves") : nullptr;
        const llvm::json::Object *physicalLeaf =
            physicalLeaves && physicalLeaves->size() == 1 ? (*physicalLeaves)[0].getAsObject() : nullptr;
        std::optional<size_t> projectedLeafIndex;
        const std::optional<llvm::StringRef> autodiffSource = row.getString("vernon.autodiff_source");
        if (!opaqueResourceEndpoint && source && autodiffSource && logicalLayout && logicalLeaves && physicalLeaf &&
            logicalShape &&
            compatibleProgramBindingShape(autodiffRole.value_or(""),
                                          row.getString("vernon.autodiff_carrier").value_or(""), logicalShape,
                                          physicalShape ? physicalShape : logicalShape)) {
            projectedLeafIndex = resolveProgramValueLeafIndex(*logicalLayout, *autodiffSource, *source);
            const llvm::json::Object *projectedLeaf = projectedLeafIndex && *projectedLeafIndex < logicalLeaves->size()
                                                          ? (*logicalLeaves)[*projectedLeafIndex].getAsObject()
                                                          : nullptr;
            projectedLeafMatches =
                projectedLeaf && projectedLeaf->getString("dtype") == physicalLeaf->getString("dtype") &&
                projectedLeaf->getInteger("scalar_count") == physicalLeaf->getInteger("scalar_count") &&
                programAbiShapesCompatible(projectedLeaf->getArray("shape"), physicalLeaf->getArray("shape"));
            if (!projectedLeafMatches)
                projectedLeafIndex.reset();
        }
        if (!tapeCarrier && !opaqueMatches && !byteValueMatches && !projectedLeafMatches) {
            error = "compiled compute endpoint layout/type/shape does not match logical value '" + source->str() +
                    "' (logical type '" +
                    (logicalValue ? logicalValue->getString("type").value_or("").str() : std::string()) +
                    "', endpoint type '" + row.getString("type").value_or("").str() + "', logical layout '" +
                    (logicalLayout ? logicalLayout->getString("layout_hash").value_or("").str() : std::string()) +
                    "', endpoint layout '" +
                    (wholeLayout ? wholeLayout->getString("layout_hash").value_or("").str() : std::string()) +
                    "', endpoint kind '" + endpointKind.str() + "', resource " + (resource ? "true" : "false") +
                    ", has value layout " + (row.getObject("value_layout") ? "true" : "false") +
                    ", has element layout " + (row.getObject("element_layout") ? "true" : "false") + ")";
            return false;
        }
        if (std::optional<llvm::StringRef> logicalType = logicalValue->getString("type");
            !resource && !resourceBackedValue && logicalType && row.getString("type") &&
            *logicalType != *row.getString("type")) {
            error = "compiled compute endpoint type does not match logical value '" + source->str() + "'";
            return false;
        }
        if (resourceBackedValue) {
            const std::optional<int64_t> slot = row.getInteger("vernon.binding");
            if (!slot || !metadataArguments.count(endpointIndex)) {
                error = "compiled resource-backed value has an incomplete TensorView carrier";
                return false;
            }
            llvm::json::Array abi{llvm::json::Object{
                {"semantic", "resource"},
                {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", *slot}}},
            }};
            llvm::json::Object endpoint{
                {"tag", "resource"},
                {"module", "compute"},
                {"interface", interfaceKind.str()},
                {"index", endpointIndex},
                {"role", autodiffRole->str()},
                {"type", row.getString("type").value_or("").str()},
                {"layout",
                 llvm::json::Object{{"tag", "buffer"},
                                    {"view_rank", int64_t{0}},
                                    {"shape", llvm::json::Array{}},
                                    {"descriptor", true},
                                    {"element_layout_hash", wholeLayout->getString("layout_hash").value_or("").str()},
                                    {"minimum_alignment", wholeLayout->getInteger("alignment").value_or(0)}}},
                {"address_space", row.getString("address_space").value_or("device").str()},
                {"transport", "resource_handle"},
                {"access", reflectedProgramResourceAccess(row).value_or("read_write")},
                {"abi", llvm::json::Object{{"bindings", std::move(abi)}}},
            };
            plan.endpoints.emplace_back(std::move(endpoint));
            llvm::json::Array projections;
            projections.emplace_back(
                llvm::json::Object{{"value", valueId}, {"physical_leaf", int64_t{0}}, {"direction", "result"}});
            plan.endpointBindings.emplace_back(llvm::json::Object{{"module", "compute"},
                                                                  {"interface", interfaceKind.str()},
                                                                  {"index", endpointIndex},
                                                                  {"tag", "value"},
                                                                  {"projections", std::move(projections)}});
            llvm::json::Object compiled = compiledProgramEndpointAbi(row, "compute", interfaceKind, endpointIndex);
            compiled["value_transport"] = "storage_buffer";
            plan.implementationEndpoints.emplace_back(std::move(compiled));
            return true;
        }
        if (!resource) {
            const std::optional<int64_t> reflectedSlot = row.getInteger("vernon.binding");
            const int64_t slot = reflectedSlot ? *reflectedSlot : nextPortableSlot++;
            llvm::json::Array abi{llvm::json::Object{
                {"semantic", "value"}, {"carrier", programValueCarrier("value_slot", slot, *wholeLayout)}}};
            llvm::json::Object endpoint{{"tag", "value"},
                                        {"module", "compute"},
                                        {"interface", interfaceKind.str()},
                                        {"index", endpointIndex},
                                        {"type", logicalValue->getString("type").value_or("").str()},
                                        {"layout_hash", wholeLayout->getString("layout_hash").value_or("").str()},
                                        {"transport", "by_value"},
                                        {"access", "read"},
                                        {"abi", llvm::json::Object{{"bindings", std::move(abi)}}}};
            if (autodiffRole)
                endpoint["role"] = autodiffRole->str();
            if (const llvm::json::Object *elementLayout = row.getObject("element_layout"))
                endpoint["element_layout_hash"] = elementLayout->getString("layout_hash").value_or("").str();
            plan.endpoints.emplace_back(std::move(endpoint));
            llvm::json::Array projections;
            llvm::json::Object projection{{"value", valueId},
                                          {"physical_leaf", int64_t{0}},
                                          {"direction", interfaceKind == "result" ? "result" : "input"}};
            if (projectedLeafIndex)
                projection["leaf"] = static_cast<int64_t>(*projectedLeafIndex);
            projections.emplace_back(std::move(projection));
            plan.endpointBindings.emplace_back(llvm::json::Object{{"module", "compute"},
                                                                  {"interface", interfaceKind.str()},
                                                                  {"index", endpointIndex},
                                                                  {"tag", "value"},
                                                                  {"projections", std::move(projections)}});
            plan.implementationEndpoints.emplace_back(
                compiledProgramEndpointAbi(row, "compute", interfaceKind, endpointIndex));
            return true;
        }
        const std::optional<std::string> physicalAccess = reflectedProgramResourceAccess(row);
        if ((!physicalAccess && interfaceKind != "result") ||
            (physicalAccess && !tapeCarrier &&
             !programResourceAccessSatisfies(*physicalAccess, resourceIt->second.access))) {
            error = "compiled compute resource access disagrees with logical access for '" + source->str() + "'";
            return false;
        }
        const bool descriptor = metadataArguments.count(endpointIndex) != 0;
        const int64_t descriptorRank = descriptor ? metadataRanks[endpointIndex] : 0;
        llvm::json::Array abi;
        if (opaqueResourceEndpoint)
            if (std::optional<int64_t> slot = row.getInteger("vernon.binding"))
                abi.emplace_back(
                    llvm::json::Object{{"semantic", "resource"},
                                       {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", *slot}}}});
        if (const llvm::json::Array *leaves = row.getArray("storage_leaves"))
            for (size_t leafIndex = 0; leafIndex < leaves->size(); ++leafIndex)
                if (const llvm::json::Object *leaf = (*leaves)[leafIndex].getAsObject())
                    if (std::optional<int64_t> slot = leaf->getInteger("binding"))
                        abi.emplace_back(llvm::json::Object{
                            {"semantic",
                             llvm::json::Object{
                                 {"storage_leaf", static_cast<int64_t>(projectedLeafIndex.value_or(0) + leafIndex)}}},
                            {"carrier", llvm::json::Object{{"tag", "resource_slot"}, {"slot", *slot}}}});
        if (abi.empty()) {
            error = "compiled compute resource endpoint has no storage leaves";
            return false;
        }
        llvm::json::Object resourceLayout;
        if (endpointKind == "image")
            resourceLayout = llvm::json::Object{{"tag", "image"},
                                                {"dimension", row.getString("dimension").value_or("").str()},
                                                {"format", row.getString("exact_storage_format").value_or("").str()},
                                                {"sample_count", int64_t{1}},
                                                {"aspects", llvm::json::Array{"color"}}};
        else if (endpointKind == "sampler")
            resourceLayout = llvm::json::Object{{"tag", "sampler"}};
        else {
            resourceLayout =
                llvm::json::Object{{"tag", "buffer"},
                                   {"view_rank", descriptorRank},
                                   {"shape", copyJsonArray(*(physicalShape ? physicalShape : logicalShape))},
                                   {"descriptor", descriptor},
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
            {"role",
             autodiffRole
                 ? autodiffRole->str()
                 : row.getString("binding_role").value_or(endpointKind == "sampler" ? "sampler" : "storage").str()},
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
                                                   ? copyJsonArray(*footprint->second.getArray("indices"))
                                                   : llvm::json::Array()}};
        plan.endpoints.emplace_back(std::move(endpoint));
        const std::optional<int64_t> accessIndex =
            ensureProgramResourceAccess(valueId, false, resources, accessByValue, plan.accesses.json());
        if (!accessIndex) {
            error = "compiled compute resource endpoint has no logical access";
            return false;
        }
        llvm::json::Object endpointBinding{{"module", "compute"},
                                           {"interface", interfaceKind.str()},
                                           {"index", endpointIndex},
                                           {"tag", "resource"},
                                           {"access", *accessIndex}};
        if (projectedLeafIndex)
            endpointBinding["leaf"] = static_cast<int64_t>(*projectedLeafIndex);
        plan.endpointBindings.emplace_back(std::move(endpointBinding));
        plan.implementationEndpoints.emplace_back(
            compiledProgramEndpointAbi(row, "compute", interfaceKind, endpointIndex));
        return true;
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
        (void)resource;
        const llvm::json::Object *logical = findJsonObjectByIntegerId(rawValues, valueId);
        const llvm::StringRef type = logical ? logical->getString("type").value_or("") : "";
        if (isProgramAdTapeType(type))
            (void)ensureProgramResourceAccess(valueId, false, resources, accessByValue, plan.accesses.json());
    }
    return validateCanonicalComputePortableSlots(plan.endpoints.json(), error);
}

bool validateCanonicalComputePortableSlots(const llvm::json::Array &endpoints, std::string &error) {
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
    return true;
}

bool buildCanonicalComputeStageContract(const llvm::json::Object &compiledEntry,
                                        const llvm::json::Object &compiledReflection, llvm::json::Array endpoints,
                                        std::optional<llvm::json::Object> metadataCarrier,
                                        llvm::json::Object &stageContract, std::string &error) {
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
    llvm::json::Object compute{{"workgroup_size", copyJsonArray(*workgroupSize)},
                               {"subgroup", nullptr},
                               {"capabilities", llvm::json::Array{"direct_dispatch"}}};
    if (const llvm::json::Object *dispatch = compiledEntry.getObject("dispatch_contract"))
        compute["dispatch_contract"] = copyJsonObject(*dispatch);
    llvm::json::Object reflection{{"required_features", std::move(features)},
                                  {"endpoints", std::move(endpoints)},
                                  {"compute", std::move(compute)}};
    if (metadataCarrier)
        reflection["metadata_carrier"] = std::move(*metadataCarrier);
    stageContract = llvm::json::Object{{"operation", "compute"}, {"reflection", std::move(reflection)}};
    return true;
}

} // namespace vernon::compiler
