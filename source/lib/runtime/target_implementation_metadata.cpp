#include "target_implementation_metadata.h"

#include <nlohmann/json.hpp>

#include <set>

namespace vernon::runtime {
namespace {

bool parseMetalResourceSlots(const nlohmann::json &slots, std::vector<NativeResourceSlot> &output, std::string &error) {
    static const std::set<std::string> fields{
        "entry_point",         "stage", "kind", "name", "set", "binding", "argument_buffer_index", "member_id",
        "direct_buffer_index", "count"};
    if (!slots.is_array())
        return error = "Metal target resource_slots must be an array", false;
    output.clear();
    for (const nlohmann::json &slot : slots) {
        if (!slot.is_object() || slot.size() != fields.size())
            return error = "Metal target resource slot has an invalid schema", false;
        for (const auto &[name, unused] : slot.items())
            if (!fields.count(name))
                return error = "Metal target resource slot has an unknown field", false;
        if (!slot["entry_point"].is_string() || !slot["stage"].is_string() || !slot["kind"].is_string() ||
            !slot["name"].is_string() || !slot["set"].is_number_unsigned() || !slot["binding"].is_number_unsigned() ||
            !slot["argument_buffer_index"].is_number_unsigned() || !slot["member_id"].is_number_unsigned() ||
            !slot["direct_buffer_index"].is_number_unsigned() || !slot["count"].is_number_unsigned() ||
            slot["count"].get<uint32_t>() == 0)
            return error = "Metal target resource slot has invalid field types", false;
        output.push_back({slot["entry_point"].get<std::string>(), slot["stage"].get<std::string>(),
                          slot["kind"].get<std::string>(), slot["name"].get<std::string>(), slot["set"].get<uint32_t>(),
                          slot["binding"].get<uint32_t>(), slot["argument_buffer_index"].get<uint32_t>(),
                          slot["member_id"].get<uint32_t>(), slot["direct_buffer_index"].get<uint32_t>(),
                          slot["count"].get<uint32_t>()});
    }
    return true;
}

} // namespace

bool parseTargetImplementationMetadata(std::string_view target, const nlohmann::json &metadata,
                                       std::vector<NativeResourceSlot> &nativeSlots, std::string &error) {
    if (!metadata.is_object())
        return error = "target implementation metadata must be an object", false;
    nativeSlots.clear();
    if (target == "metal") {
        if (metadata.size() != 1 || !metadata.contains("resource_slots"))
            return error = "Metal target implementation metadata must contain only resource_slots", false;
        return parseMetalResourceSlots(metadata["resource_slots"], nativeSlots, error);
    }
    if (!metadata.empty())
        return error = "target implementation metadata is unsupported for this backend", false;
    return true;
}

} // namespace vernon::runtime
