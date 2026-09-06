#include "program_execution_manifest.h"

#include "content_hash.h"
#include "shape_layout.h"
#include "target_implementation_metadata.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string_view>

namespace vernon::runtime::program {
namespace {

bool fail(Diagnostic &diagnostic, std::string code, std::string phase, std::string path, std::string message) {
    diagnostic = {std::move(code), std::move(phase), std::move(path), std::move(message)};
    return false;
}

bool isTapeCarrierRole(std::string_view role) { return program_plan::tapeCarrierFromRoleName(role).has_value(); }

bool parseTapeCarriers(const nlohmann::json &rows, std::vector<program_plan::TapeCarrier> &carriers) {
    if (!rows.is_array())
        return false;
    for (const nlohmann::json &row : rows) {
        if (!row.is_string())
            return false;
        const std::optional<program_plan::TapeCarrier> carrier =
            program_plan::tapeCarrierFromPlanName(row.get<std::string>());
        if (!carrier)
            return false;
        carriers.push_back(*carrier);
    }
    return true;
}

bool isResourceEndpointRole(std::string_view role) {
    return role == "storage" || role == "image" || role == "sampled" || role == "sampler" || role == "vertex" ||
           role == "primal" || role == "retained_primal" || role == "cotangent" || role == "gradient" ||
           isTapeCarrierRole(role);
}

bool uint32Value(const nlohmann::json &value, uint32_t &result) {
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed > std::numeric_limits<uint32_t>::max())
            return false;
        result = static_cast<uint32_t>(parsed);
        return true;
    }
    if (!value.is_number_integer())
        return false;
    const auto parsed = value.get<int64_t>();
    if (parsed < 0 || static_cast<uint64_t>(parsed) > std::numeric_limits<uint32_t>::max())
        return false;
    result = static_cast<uint32_t>(parsed);
    return true;
}

bool digestValue(const nlohmann::json &value) {
    if (!value.is_string() || value.get_ref<const std::string &>().size() != 64)
        return false;
    return std::all_of(value.get_ref<const std::string &>().begin(), value.get_ref<const std::string &>().end(),
                       [](char character) {
                           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
                       });
}

bool uint64Value(const nlohmann::json &value, uint64_t &result) {
    if (!value.is_number_integer() && !value.is_number_unsigned())
        return false;
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
        return true;
    }
    const auto parsed = value.get<int64_t>();
    if (parsed < 0)
        return false;
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool exactObject(const nlohmann::json &value, std::initializer_list<std::string_view> required,
                 std::initializer_list<std::string_view> optional, Diagnostic &diagnostic, const std::string &path);

template <typename Enum>
bool enumName(const nlohmann::json &value, std::initializer_list<std::string_view> names, Enum &result) {
    if (!value.is_string())
        return false;
    const std::string_view name = value.get_ref<const std::string &>();
    size_t ordinal = 0;
    for (std::string_view candidate : names) {
        if (name == candidate) {
            result = static_cast<Enum>(ordinal);
            return true;
        }
        ++ordinal;
    }
    return false;
}

bool parseStencilFace(const nlohmann::json &value, VernonStencilFaceState &face, Diagnostic &diagnostic,
                      const std::string &path) {
    if (!exactObject(value, {"stencil_fail", "depth_fail", "pass_operation", "compare"}, {}, diagnostic, path) ||
        !enumName(value["stencil_fail"],
                  {"KEEP", "ZERO", "REPLACE", "INCREMENT_CLAMP", "DECREMENT_CLAMP", "INVERT", "INCREMENT_WRAP",
                   "DECREMENT_WRAP"},
                  face.stencil_fail) ||
        !enumName(value["depth_fail"],
                  {"KEEP", "ZERO", "REPLACE", "INCREMENT_CLAMP", "DECREMENT_CLAMP", "INVERT", "INCREMENT_WRAP",
                   "DECREMENT_WRAP"},
                  face.depth_fail) ||
        !enumName(value["pass_operation"],
                  {"KEEP", "ZERO", "REPLACE", "INCREMENT_CLAMP", "DECREMENT_CLAMP", "INVERT", "INCREMENT_WRAP",
                   "DECREMENT_WRAP"},
                  face.pass) ||
        !enumName(value["compare"],
                  {"NEVER", "LESS", "EQUAL", "LESS_EQUAL", "GREATER", "NOT_EQUAL", "GREATER_EQUAL", "ALWAYS"},
                  face.compare))
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path, "invalid stencil face state");
    return true;
}

bool parseColorBlend(const nlohmann::json &value, VernonColorBlendState &blend, Diagnostic &diagnostic,
                     const std::string &path) {
    if (!exactObject(value,
                     {"enabled", "source_color", "destination_color", "color_operation", "source_alpha",
                      "destination_alpha", "alpha_operation", "write_mask"},
                     {}, diagnostic, path) ||
        !value["enabled"].is_boolean() ||
        !enumName(value["source_color"],
                  {"ZERO", "ONE", "SOURCE_COLOR", "ONE_MINUS_SOURCE_COLOR", "DESTINATION_COLOR",
                   "ONE_MINUS_DESTINATION_COLOR", "SOURCE_ALPHA", "ONE_MINUS_SOURCE_ALPHA", "DESTINATION_ALPHA",
                   "ONE_MINUS_DESTINATION_ALPHA"},
                  blend.source_color_factor) ||
        !enumName(value["destination_color"],
                  {"ZERO", "ONE", "SOURCE_COLOR", "ONE_MINUS_SOURCE_COLOR", "DESTINATION_COLOR",
                   "ONE_MINUS_DESTINATION_COLOR", "SOURCE_ALPHA", "ONE_MINUS_SOURCE_ALPHA", "DESTINATION_ALPHA",
                   "ONE_MINUS_DESTINATION_ALPHA"},
                  blend.destination_color_factor) ||
        !enumName(value["color_operation"], {"ADD", "SUBTRACT", "REVERSE_SUBTRACT", "MINIMUM", "MAXIMUM"},
                  blend.color_operation) ||
        !enumName(value["source_alpha"],
                  {"ZERO", "ONE", "SOURCE_COLOR", "ONE_MINUS_SOURCE_COLOR", "DESTINATION_COLOR",
                   "ONE_MINUS_DESTINATION_COLOR", "SOURCE_ALPHA", "ONE_MINUS_SOURCE_ALPHA", "DESTINATION_ALPHA",
                   "ONE_MINUS_DESTINATION_ALPHA"},
                  blend.source_alpha_factor) ||
        !enumName(value["destination_alpha"],
                  {"ZERO", "ONE", "SOURCE_COLOR", "ONE_MINUS_SOURCE_COLOR", "DESTINATION_COLOR",
                   "ONE_MINUS_DESTINATION_COLOR", "SOURCE_ALPHA", "ONE_MINUS_SOURCE_ALPHA", "DESTINATION_ALPHA",
                   "ONE_MINUS_DESTINATION_ALPHA"},
                  blend.destination_alpha_factor) ||
        !enumName(value["alpha_operation"], {"ADD", "SUBTRACT", "REVERSE_SUBTRACT", "MINIMUM", "MAXIMUM"},
                  blend.alpha_operation) ||
        !uint32Value(value["write_mask"], blend.write_mask) || (blend.write_mask & ~VERNON_RHI_COLOR_WRITE_ALL))
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path, "invalid color blend state");
    blend.blend_enabled = value["enabled"].get<bool>();
    return true;
}

bool exactObject(const nlohmann::json &value, std::initializer_list<std::string_view> required,
                 std::initializer_list<std::string_view> optional, Diagnostic &diagnostic, const std::string &path) {
    if (!value.is_object())
        return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path, "expected an object");
    for (std::string_view key : required)
        if (!value.contains(std::string(key)))
            return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path + "/" + std::string(key),
                        "missing required member");
    for (auto item = value.begin(); item != value.end(); ++item) {
        const std::string_view key = item.key();
        if (std::find(required.begin(), required.end(), key) == required.end() &&
            std::find(optional.begin(), optional.end(), key) == optional.end())
            return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path + "/" + item.key(), "unknown member");
    }
    return true;
}

bool parseGraphicsPipelineState(const nlohmann::json &value, GraphicsPipelineState &state, Diagnostic &diagnostic,
                                const std::string &path) {
    if (!exactObject(value, {"topology", "rasterization", "depth_stencil", "color_blends"}, {}, diagnostic, path) ||
        !enumName(value["topology"], {"triangle_list", "line_list", "point_list"}, state.topology))
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path, "invalid graphics pipeline state");
    const auto &raster = value["rasterization"];
    if (!exactObject(raster, {"cull_mode", "front_face", "depth_clamp", "depth_bias_constant", "depth_bias_slope"}, {},
                     diagnostic, path + "/rasterization") ||
        !enumName(raster["cull_mode"], {"NONE", "FRONT", "BACK"}, state.rasterization.cull_mode) ||
        !enumName(raster["front_face"], {"COUNTER_CLOCKWISE", "CLOCKWISE"}, state.rasterization.front_face) ||
        !raster["depth_clamp"].is_boolean() || !raster["depth_bias_constant"].is_number() ||
        !raster["depth_bias_slope"].is_number())
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/rasterization",
                    "invalid rasterization state");
    state.rasterization.depth_clamp = raster["depth_clamp"].get<bool>();
    state.rasterization.depth_bias_constant = raster["depth_bias_constant"].get<float>();
    state.rasterization.depth_bias_slope = raster["depth_bias_slope"].get<float>();
    if (!std::isfinite(state.rasterization.depth_bias_constant) || !std::isfinite(state.rasterization.depth_bias_slope))
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/rasterization",
                    "rasterization bias must be finite");
    state.rasterization.depth_bias_enabled =
        state.rasterization.depth_bias_constant != 0.0f || state.rasterization.depth_bias_slope != 0.0f;

    const auto &depth = value["depth_stencil"];
    if (!exactObject(depth,
                     {"depth_test", "depth_write", "depth_compare", "stencil_test", "front", "back",
                      "stencil_read_mask", "stencil_write_mask"},
                     {}, diagnostic, path + "/depth_stencil") ||
        !depth["depth_test"].is_boolean() || !depth["depth_write"].is_boolean() ||
        !depth["stencil_test"].is_boolean() ||
        !enumName(depth["depth_compare"],
                  {"NEVER", "LESS", "EQUAL", "LESS_EQUAL", "GREATER", "NOT_EQUAL", "GREATER_EQUAL", "ALWAYS"},
                  state.depthStencil.depth_compare) ||
        !uint32Value(depth["stencil_read_mask"], state.depthStencil.stencil_read_mask) ||
        !uint32Value(depth["stencil_write_mask"], state.depthStencil.stencil_write_mask) ||
        state.depthStencil.stencil_read_mask > 0xff || state.depthStencil.stencil_write_mask > 0xff ||
        !parseStencilFace(depth["front"], state.depthStencil.front, diagnostic, path + "/depth_stencil/front") ||
        !parseStencilFace(depth["back"], state.depthStencil.back, diagnostic, path + "/depth_stencil/back"))
        return false;
    state.depthStencil.depth_test = depth["depth_test"].get<bool>();
    state.depthStencil.depth_write = depth["depth_write"].get<bool>();
    state.depthStencil.stencil_test = depth["stencil_test"].get<bool>();

    const auto &blends = value["color_blends"];
    if (!blends.is_array())
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/color_blends",
                    "color blends must be an array");
    for (size_t index = 0; index < blends.size(); ++index) {
        const auto &entry = blends[index];
        uint32_t location{};
        VernonColorBlendState blend{};
        const std::string entryPath = path + "/color_blends/" + std::to_string(index);
        if (!entry.is_array() || entry.size() != 2 || !uint32Value(entry[0], location) ||
            !parseColorBlend(entry[1], blend, diagnostic, entryPath + "/1") ||
            !state.colorBlends.emplace(location, blend).second)
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", entryPath,
                        "invalid or duplicate color blend location");
    }
    return true;
}

bool parseTextureFormat(const nlohmann::json &value, VernonTextureFormat &format) {
    return enumName(value,
                    {"rgba8_unorm", "rgba8_srgb", "rgba16_float", "rgba32_float", "r8_unorm", "r16_float", "r32_float",
                     "rg8_unorm", "rgb8_unorm", "r11g11b10_float", "d32_float", "d32_float_s8_uint"},
                    format);
}

bool parseAttachmentSignature(const nlohmann::json &value, bool color, GraphicsAttachmentSignature &signature,
                              Diagnostic &diagnostic, const std::string &path) {
    const std::initializer_list<std::string_view> required =
        color ? std::initializer_list<std::string_view>{"location", "access", "formats", "sample_counts"}
              : std::initializer_list<std::string_view>{"access", "formats", "sample_counts", "aspects"};
    if (!exactObject(value, required, {}, diagnostic, path) || !uint32Value(value["access"], signature.access) ||
        !value["formats"].is_array() || value["formats"].empty() || !value["sample_counts"].is_array() ||
        value["sample_counts"].empty())
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path,
                    "invalid graphics attachment signature");
    if (color) {
        if (!uint32Value(value["location"], signature.location))
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/location",
                        "invalid color attachment location");
        signature.aspects = VERNON_IMAGE_ASPECT_COLOR;
    } else {
        if (!value["aspects"].is_array() || value["aspects"].empty())
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/aspects",
                        "invalid depth/stencil aspects");
        for (const auto &aspect : value["aspects"]) {
            if (aspect == "depth")
                signature.aspects |= VERNON_IMAGE_ASPECT_DEPTH;
            else if (aspect == "stencil")
                signature.aspects |= VERNON_IMAGE_ASPECT_STENCIL;
            else
                return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/aspects",
                            "unknown depth/stencil aspect");
        }
    }
    for (const auto &formatValue : value["formats"]) {
        VernonTextureFormat format{};
        if (!parseTextureFormat(formatValue, format))
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/formats",
                        "unknown attachment format");
        signature.formats.push_back(format);
    }
    for (const auto &sampleValue : value["sample_counts"]) {
        uint32_t samples{};
        if (!uint32Value(sampleValue, samples) || !samples)
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/sample_counts",
                        "invalid attachment sample count");
        signature.sampleCounts.push_back(samples);
    }
    return true;
}

bool parseIdArray(const nlohmann::json &value, std::vector<uint32_t> &result, Diagnostic &diagnostic,
                  const std::string &path) {
    if (!value.is_array())
        return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path, "expected an array");
    for (size_t index = 0; index < value.size(); ++index) {
        uint32_t id = 0;
        if (!uint32Value(value[index], id))
            return fail(diagnostic, "PROGRAM_ID_SEQUENCE", "parse", path + "/" + std::to_string(index),
                        "expected a uint32 value id");
        result.push_back(id);
    }
    return true;
}

bool parseShape(const nlohmann::json &value, std::vector<uint64_t> &result, Diagnostic &diagnostic,
                const std::string &path, bool allowEmpty = true, bool allowDynamic = false) {
    if (!value.is_array() || (!allowEmpty && value.empty()))
        return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "expected a shape array");
    std::vector<int64_t> reflected;
    reflected.reserve(value.size());
    for (size_t index = 0; index < value.size(); ++index) {
        if (!value[index].is_number_integer())
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path + "/" + std::to_string(index),
                        allowDynamic ? "TensorView extent must be a positive static size or dyn"
                                     : "phase-one compute requires positive static extents");
        int64_t extent = 0;
        if (value[index].is_number_unsigned()) {
            const uint64_t unsignedExtent = value[index].get<uint64_t>();
            if (unsignedExtent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path + "/" + std::to_string(index),
                            "shape extent exceeds the signed reflection range");
            extent = static_cast<int64_t>(unsignedExtent);
        } else {
            extent = value[index].get<int64_t>();
        }
        if ((!allowDynamic && extent <= 0) || (allowDynamic && extent != -1 && extent <= 0))
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path + "/" + std::to_string(index),
                        allowDynamic ? "TensorView extent must be a positive static size or dyn"
                                     : "phase-one compute requires positive static extents");
        reflected.push_back(extent);
    }
    const std::optional<shape::DeclaredShape> decoded = shape::decodeReflectedShape(reflected);
    if (!decoded)
        return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "shape contains an invalid extent");
    result = shape::encodeRuntimeContractShape(*decoded);
    return true;
}

bool parseCanonicalValueType(std::string_view type, CanonicalValueType &parsed) {
    parsed = {};
    if (type.rfind("tensor<", 0) != 0 && type.rfind("vector<", 0) != 0) {
        if (type.find('<') == std::string_view::npos)
            parsed.dtype = std::string(type);
        return true;
    }
    if (type.size() < 9 || type.back() != '>')
        return false;
    const std::string_view body = type.substr(7, type.size() - 8);
    size_t begin = 0;
    while (true) {
        const size_t separator = body.find('x', begin);
        if (separator == std::string_view::npos)
            break;
        const std::string_view extent = body.substr(begin, separator - begin);
        if (extent.empty() ||
            !std::all_of(extent.begin(), extent.end(), [](char value) { return value >= '0' && value <= '9'; }))
            break;
        uint64_t parsedExtent = 0;
        for (char digit : extent) {
            const uint64_t value = static_cast<uint64_t>(digit - '0');
            if (parsedExtent > (std::numeric_limits<uint64_t>::max() - value) / 10)
                return false;
            parsedExtent = parsedExtent * 10 + value;
        }
        if (!parsedExtent)
            return false;
        parsed.innerShape.push_back(parsedExtent);
        begin = separator + 1;
    }
    if (begin == body.size())
        return false;
    parsed.dtype = std::string(body.substr(begin));
    parsed.rankedValue = true;
    return !parsed.dtype.empty();
}

bool sortedUnique(const std::vector<uint32_t> &values) {
    return std::is_sorted(values.begin(), values.end()) &&
           std::adjacent_find(values.begin(), values.end()) == values.end();
}

bool parseLayout(const nlohmann::json &value, ValueLayout &layout, Diagnostic &diagnostic, const std::string &path) {
    if (!exactObject(value, {"scope", "layout_hash", "byte_size", "alignment", "leaves"}, {}, diagnostic, path))
        return false;
    if (!value["scope"].is_string() || (value["scope"] != "element" && value["scope"] != "value") ||
        !digestValue(value["layout_hash"]) || !uint64Value(value["byte_size"], layout.byteSize) || !layout.byteSize ||
        !uint64Value(value["alignment"], layout.alignment) || !layout.alignment ||
        (layout.alignment & (layout.alignment - 1)) || !value["leaves"].is_array()) {
        return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse", path, "invalid ValueLayout");
    }
    layout.scope = value["scope"].get<std::string>();
    layout.layoutHash = value["layout_hash"].get<std::string>();
    uint64_t previousEnd = 0;
    for (size_t index = 0; index < value["leaves"].size(); ++index) {
        const auto &row = value["leaves"][index];
        const std::string leafPath = path + "/leaves/" + std::to_string(index);
        if (!exactObject(row, {"path", "dtype", "byte_offset", "scalar_count", "shape"}, {}, diagnostic, leafPath))
            return false;
        LayoutLeaf leaf;
        if (!row["path"].is_array() || !row["dtype"].is_string() || !uint64Value(row["byte_offset"], leaf.byteOffset) ||
            !uint64Value(row["scalar_count"], leaf.scalarCount) || !leaf.scalarCount ||
            !parseShape(row["shape"], leaf.shape, diagnostic, leafPath + "/shape")) {
            return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse", leafPath, "invalid canonical layout leaf");
        }
        for (size_t component = 0; component < row["path"].size(); ++component) {
            LayoutPathComponent pathComponent;
            if (row["path"][component].is_string() && !row["path"][component].get_ref<const std::string &>().empty()) {
                pathComponent.field = row["path"][component].get<std::string>();
            } else {
                uint32_t componentIndex{};
                if (!uint32Value(row["path"][component], componentIndex))
                    return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse",
                                leafPath + "/path/" + std::to_string(component), "invalid canonical layout path");
                pathComponent.index = componentIndex;
            }
            leaf.path.push_back(std::move(pathComponent));
        }
        leaf.dtype = row["dtype"].get<std::string>();
        const uint64_t scalarSize = leaf.dtype == "bool" || leaf.dtype == "i8" || leaf.dtype == "u8"    ? 1
                                    : leaf.dtype == "i16" || leaf.dtype == "u16" || leaf.dtype == "f16" ? 2
                                    : leaf.dtype == "i32" || leaf.dtype == "u32" || leaf.dtype == "f32" ? 4
                                    : leaf.dtype == "i64" || leaf.dtype == "u64" || leaf.dtype == "f64" ? 8
                                                                                                        : 0;
        if (!scalarSize ||
            leaf.scalarCount > (layout.byteSize - std::min(layout.byteSize, leaf.byteOffset)) / scalarSize)
            return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse", leafPath, "layout leaf exceeds its byte range");
        const uint64_t end = leaf.byteOffset + leaf.scalarCount * scalarSize;
        if (leaf.byteOffset < previousEnd || end > layout.byteSize)
            return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse", leafPath, "layout leaves overlap or exceed size");
        previousEnd = end;
        layout.leaves.push_back(std::move(leaf));
    }
    if (layout.leaves.empty())
        return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse", path + "/leaves",
                    "canonical layout requires at least one leaf");
    return true;
}

bool parseOrigin(const nlohmann::json &value, Origin &origin, Diagnostic &diagnostic, const std::string &path) {
    if (!value.is_object() || !value.contains("tag") || !value["tag"].is_string())
        return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "invalid Value origin");
    const std::string tag = value["tag"].get<std::string>();
    if (tag == "argument") {
        if (!exactObject(value, {"tag", "graph", "slot"}, {}, diagnostic, path) || !value["graph"].is_string() ||
            !uint32Value(value["slot"], origin.slot))
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "invalid ArgumentOrigin");
        origin.kind = OriginKind::Argument;
        origin.graph = value["graph"].get<std::string>();
    } else if (tag == "parameter") {
        if (!exactObject(value, {"tag", "parameter"}, {}, diagnostic, path) ||
            !uint32Value(value["parameter"], origin.parameter))
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "invalid ParameterOrigin");
        origin.kind = OriginKind::Parameter;
    } else if (tag == "allocation") {
        if (!exactObject(value, {"tag", "graph"}, {}, diagnostic, path) || !value["graph"].is_string())
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "invalid AllocationOrigin");
        origin.kind = OriginKind::Allocation;
        origin.graph = value["graph"].get<std::string>();
    } else if (tag == "node_result") {
        if (!exactObject(value, {"tag", "graph", "node"}, {}, diagnostic, path) || !value["graph"].is_string() ||
            !uint32Value(value["node"], origin.node))
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path, "invalid NodeResultOrigin");
        origin.kind = OriginKind::NodeResult;
        origin.graph = value["graph"].get<std::string>();
    } else {
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/tag",
                    "value origin is not implemented by phase-one compute");
    }
    return true;
}

bool parseControl(const nlohmann::json &value, ControlComponent &control, Diagnostic &diagnostic,
                  const std::string &path) {
    if (uint64Value(value, control.value)) {
        if (!control.value)
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path, "workgroup count must be positive");
        control.kind = ControlKind::Static;
        return true;
    }
    if (!exactObject(value, {"control"}, {}, diagnostic, path) || !value["control"].is_object())
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path, "invalid control component");
    const auto &reference = value["control"];
    if (reference.size() != 1)
        return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "parse", path + "/control",
                    "control reference must have one source");
    if (reference.contains("value") && uint32Value(reference["value"], control.reference))
        control.kind = ControlKind::Value;
    else if (reference.contains("parameter") && uint32Value(reference["parameter"], control.reference))
        control.kind = ControlKind::Parameter;
    else if (reference.contains("capture") && uint32Value(reference["capture"], control.reference))
        control.kind = ControlKind::Capture;
    else
        return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "parse", path + "/control",
                    "control supports value, parameter, and capture");
    return true;
}

bool parseExtentComponent(const nlohmann::json &value, ControlComponent &component, Diagnostic &diagnostic,
                          const std::string &path, bool allowZero) {
    if (uint64Value(value, component.value)) {
        if (!component.value && !allowZero)
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path,
                        "owned extent must be a positive static size or a control dimension");
        component.kind = ControlKind::Static;
        return true;
    }
    if (value.is_object() && value.contains("dimension")) {
        if (!exactObject(value, {"dimension"}, {}, diagnostic, path) || !value["dimension"].is_object())
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path, "invalid dimension control");
        const auto &dimension = value["dimension"];
        if (!exactObject(dimension, {"control", "axis"}, {}, diagnostic, path + "/dimension") ||
            !dimension["control"].is_object() || !uint32Value(dimension["axis"], component.axis))
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/dimension",
                        "dimension control requires control and axis");
        nlohmann::json wrapped = {{"control", dimension["control"]}};
        if (!parseControl(wrapped, component, diagnostic, path + "/dimension"))
            return false;
        component.hasAxis = true;
        return true;
    }
    if (!parseControl(value, component, diagnostic, path))
        return false;
    return true;
}

bool stringArray(const nlohmann::json &value, std::vector<std::string> &result, Diagnostic &diagnostic,
                 const std::string &path, bool allowEmpty = true) {
    if (!value.is_array() || (!allowEmpty && value.empty()))
        return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path, "expected a string array");
    for (size_t index = 0; index < value.size(); ++index) {
        if (!value[index].is_string() || value[index].get_ref<const std::string &>().empty())
            return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path + "/" + std::to_string(index),
                        "expected a non-empty string");
        result.push_back(value[index].get<std::string>());
    }
    if (!std::is_sorted(result.begin(), result.end()) ||
        std::adjacent_find(result.begin(), result.end()) != result.end())
        return fail(diagnostic, "PROGRAM_NON_CANONICAL_ORDER", "parse", path, "string set must be sorted and unique");
    return true;
}

BoundaryAccess canonicalBoundaryAccess(const Program &program, BoundaryDirection direction,
                                       const std::optional<uint32_t> &storageId) {
    if (direction == BoundaryDirection::Output)
        return BoundaryAccess::Write;
    if (!storageId)
        return BoundaryAccess::Read;
    bool reads = false;
    bool writes = false;
    const Graph *forward = findGraph(program, "forward");
    if (forward)
        for (const Node &node : forward->nodes)
            for (const ResourceAccess &access : node.accesses) {
                if (access.storage != *storageId)
                    continue;
                reads |= access.kind == AccessKind::Read || access.kind == AccessKind::Attachment ||
                         (access.kind == AccessKind::Write && access.access == "read_write");
                writes |= access.kind == AccessKind::Initialize || access.kind == AccessKind::Write ||
                          access.kind == AccessKind::Attachment;
            }
    return writes ? (reads ? BoundaryAccess::ReadWrite : BoundaryAccess::Write) : BoundaryAccess::Read;
}

} // namespace

const Graph *findGraph(const Program &program, std::string_view direction) {
    const auto found = std::find_if(program.graphs.begin(), program.graphs.end(),
                                    [&](const Graph &graph) { return graph.direction == direction; });
    return found == program.graphs.end() ? nullptr : &*found;
}

PublicationPlan derivePublicationPlan(const std::vector<BoundarySlot> &slots) {
    PublicationPlan plan;
    for (const BoundarySlot &slot : slots)
        if (slot.publication == BoundaryPublication::CommitAfterSuccess)
            plan.targets.push_back({slot.id, slot.value, slot.role, slot.aliasOwner});
    return plan;
}

const PublicationTarget *findPublicationTarget(const ProgramAbi &abi, uint32_t slot) {
    const auto found = std::find_if(abi.publication.targets.begin(), abi.publication.targets.end(),
                                    [&](const PublicationTarget &target) { return target.slot == slot; });
    return found == abi.publication.targets.end() ? nullptr : &*found;
}

std::vector<uint32_t> residualCaptures(const Program &program) {
    std::vector<uint32_t> captures;
    if (!program.residualContract)
        return captures;
    captures.reserve(program.residualContract->captures.size());
    for (const ResidualCapture &capture : program.residualContract->captures)
        captures.push_back(capture.value);
    return captures;
}

void markGraphValues(const Graph &graph, std::vector<char> &live) {
    const auto mark = [&](uint32_t value) {
        if (value < live.size())
            live[value] = 1;
    };
    for (const GraphInput &input : graph.inputs)
        mark(input.value);
    for (uint32_t capture : graph.captures)
        mark(capture);
    for (const GraphOutput &output : graph.outputs)
        mark(output.value);
    for (const Node &node : graph.nodes) {
        for (uint32_t value : node.operands)
            mark(value);
        for (uint32_t value : node.results)
            mark(value);
    }
}

bool isTapeValueType(std::string_view type) {
    return type == "!vernon.ad_tape" ||
           (type.rfind("!vernon.ad_tape<", 0) == 0 && type.size() > 17 && type.back() == '>');
}

bool parse(const nlohmann::json &value, Program &program, Diagnostic &diagnostic) {
    program = {};
    diagnostic = {};
    if (!exactObject(value, {"stages", "parameters", "storages", "values", "graphs", "abi"}, {"residual_contract"},
                     diagnostic, ""))
        return false;
    if (!value["stages"].is_object() || !value["parameters"].is_array() || !value["storages"].is_array() ||
        !value["values"].is_array() || !value["graphs"].is_array())
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", "", "Program collections are invalid");

    for (const auto &[stageId, row] : value["stages"].items()) {
        const std::string path = "/stages/" + stageId;
        const std::string operation = row.value("operation", "");
        if (stageId.empty() || !exactObject(row, {"operation", "contract_hash"}, {}, diagnostic, path) ||
            !row["operation"].is_string() || (operation != "compute" && operation != "graphics") ||
            !digestValue(row["contract_hash"]))
            return fail(diagnostic, "PROGRAM_STAGE_MISSING", "parse", path,
                        "stage must be a compute or graphics StageContract");
        program.stages.emplace(stageId, StageContract{operation, row["contract_hash"].get<std::string>()});
    }

    for (size_t index = 0; index < value["parameters"].size(); ++index) {
        const auto &row = value["parameters"][index];
        const std::string path = "/parameters/" + std::to_string(index);
        Parameter parameter;
        if (!exactObject(row, {"id", "path", "value"}, {}, diagnostic, path) || !uint32Value(row["id"], parameter.id) ||
            parameter.id != index || !row["path"].is_string() || row["path"].get_ref<const std::string &>().empty() ||
            !uint32Value(row["value"], parameter.value))
            return fail(diagnostic, "PROGRAM_PARAMETER_BINDING", "parse", path, "invalid Parameter");
        program.parameters.push_back(std::move(parameter));
    }

    for (size_t index = 0; index < value["storages"].size(); ++index) {
        const auto &row = value["storages"][index];
        const std::string path = "/storages/" + std::to_string(index);
        Storage storage;
        if (!exactObject(row, {"id", "initial_value", "ownership", "lifetime", "mutability", "descriptor"}, {"name"},
                         diagnostic, path) ||
            !uint32Value(row["id"], storage.id) || storage.id != index ||
            !uint32Value(row["initial_value"], storage.initialValue) || !row["ownership"].is_string() ||
            !row["lifetime"].is_string() || !row["mutability"].is_string()) {
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path, "invalid Storage");
        }
        storage.name = row.value("name", "");
        const std::string ownership = row["ownership"].get<std::string>();
        const std::string lifetime = row["lifetime"].get<std::string>();
        const std::string mutability = row["mutability"].get<std::string>();
        if (ownership == "owned")
            storage.ownership = StorageOwnership::Owned;
        else if (ownership != "borrowed")
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/ownership", "invalid ownership");
        if (lifetime == "instance")
            storage.lifetime = StorageLifetime::Instance;
        else if (lifetime == "pullback")
            storage.lifetime = StorageLifetime::Pullback;
        else if (lifetime != "invocation")
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/lifetime", "invalid lifetime");
        if (mutability == "mutable")
            storage.mutability = StorageMutability::Mutable;
        else if (mutability != "read_only")
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/mutability", "invalid mutability");
        const auto &descriptor = row["descriptor"];
        if (!descriptor.is_object() || !descriptor.contains("tag") || !descriptor["tag"].is_string())
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor",
                        "storage descriptor requires a tag");
        const std::string descriptorTag = descriptor["tag"].get<std::string>();
        if (descriptorTag == "buffer") {
            if (!exactObject(descriptor, {"tag", "byte_length", "alignment", "memory", "usage"}, {}, diagnostic,
                             path + "/descriptor") ||
                !uint64Value(descriptor["alignment"], storage.buffer.alignment) || !storage.buffer.alignment ||
                (storage.buffer.alignment & (storage.buffer.alignment - 1)) || !descriptor["memory"].is_string() ||
                !descriptor["usage"].is_array())
                return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor",
                            "invalid buffer descriptor");
            if (descriptor["byte_length"].is_array()) {
                if (storage.ownership != StorageOwnership::Owned)
                    return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor/byte_length",
                                "control byte_length is legal only on owned Storage");
                for (size_t axis = 0; axis < descriptor["byte_length"].size(); ++axis) {
                    ControlComponent component;
                    if (!parseExtentComponent(descriptor["byte_length"][axis], component, diagnostic,
                                              path + "/descriptor/byte_length/" + std::to_string(axis), false))
                        return false;
                    if (component.kind != ControlKind::Static && !component.hasAxis)
                        return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse",
                                    path + "/descriptor/byte_length/" + std::to_string(axis),
                                    "owned buffer dyn extent must name a like-source dimension");
                    storage.buffer.byteLengthExtents.push_back(component);
                }
                if (storage.buffer.byteLengthExtents.empty())
                    return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor/byte_length",
                                "owned dyn buffer byte_length has no extents");
            } else if (!uint64Value(descriptor["byte_length"], storage.buffer.byteLength) ||
                       (!storage.buffer.byteLength && storage.ownership != StorageOwnership::Borrowed)) {
                return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor",
                            "invalid buffer descriptor");
            }
            storage.buffer.memory = descriptor["memory"].get<std::string>();
            for (const auto &usage : descriptor["usage"]) {
                if (!usage.is_string())
                    return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor/usage",
                                "usage must contain strings");
                storage.buffer.usage.push_back(usage.get<std::string>());
            }
            if (storage.buffer.usage.empty() ||
                !std::is_sorted(storage.buffer.usage.begin(), storage.buffer.usage.end()) ||
                std::adjacent_find(storage.buffer.usage.begin(), storage.buffer.usage.end()) !=
                    storage.buffer.usage.end())
                return fail(diagnostic, "PROGRAM_NON_CANONICAL_ORDER", "parse", path + "/descriptor/usage",
                            "usage must be sorted and unique");
        } else if (descriptorTag == "image") {
            storage.descriptorKind = StorageDescriptorKind::Image;
            if (!exactObject(descriptor,
                             {"tag", "dimension", "extent", "format", "sample_count", "mip_levels", "array_layers",
                              "aspects", "usage"},
                             {}, diagnostic, path + "/descriptor") ||
                !descriptor["dimension"].is_string() || !descriptor["extent"].is_array() ||
                descriptor["extent"].size() != 3 || !descriptor["format"].is_string() ||
                !uint32Value(descriptor["sample_count"], storage.image.sampleCount) ||
                !uint32Value(descriptor["mip_levels"], storage.image.mipLevels) ||
                !uint32Value(descriptor["array_layers"], storage.image.arrayLayers) || !storage.image.sampleCount ||
                !storage.image.mipLevels || !storage.image.arrayLayers || !descriptor["aspects"].is_array() ||
                !descriptor["usage"].is_array())
                return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor",
                            "invalid static image descriptor");
            storage.image.dimension = descriptor["dimension"].get<std::string>();
            storage.image.format = descriptor["format"].get<std::string>();
            const bool borrowedImage = storage.ownership == StorageOwnership::Borrowed;
            for (size_t axis = 0; axis < descriptor["extent"].size(); ++axis) {
                ControlComponent component;
                if (!parseExtentComponent(descriptor["extent"][axis], component, diagnostic,
                                          path + "/descriptor/extent/" + std::to_string(axis), borrowedImage))
                    return false;
                if (component.kind == ControlKind::Static) {
                    storage.image.extent.push_back(component.value);
                } else {
                    if (borrowedImage)
                        return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse",
                                    path + "/descriptor/extent/" + std::to_string(axis),
                                    "borrowed image extent uses 0 for dyn, not a control component");
                    storage.image.extent.push_back(0);
                    storage.image.extentControls.resize(3);
                    storage.image.extentControls[axis] = component;
                }
            }
            if (!storage.image.extentControls.empty())
                for (size_t axis = 0; axis < storage.image.extent.size(); ++axis)
                    if (storage.image.extentControls.size() <= axis ||
                        (storage.image.extent[axis] && storage.image.extentControls[axis].kind == ControlKind::Static &&
                         !storage.image.extentControls[axis].value))
                        storage.image.extentControls[axis] =
                            ControlComponent{ControlKind::Static, storage.image.extent[axis], 0, false, 0};
            for (const auto &[key, target] :
                 {std::pair<const char *, std::vector<std::string> *>("aspects", &storage.image.aspects),
                  std::pair<const char *, std::vector<std::string> *>("usage", &storage.image.usage)}) {
                for (const auto &item : descriptor[key]) {
                    if (!item.is_string())
                        return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor/" + key,
                                    "entries must be strings");
                    target->push_back(item.get<std::string>());
                }
                if (target->empty() || !std::is_sorted(target->begin(), target->end()) ||
                    std::adjacent_find(target->begin(), target->end()) != target->end())
                    return fail(diagnostic, "PROGRAM_NON_CANONICAL_ORDER", "parse", path + "/descriptor/" + key,
                                "entries must be sorted and unique");
            }
        } else if (descriptorTag == "opaque") {
            storage.descriptorKind = StorageDescriptorKind::Opaque;
            if (!exactObject(descriptor, {"tag", "contract_hash"}, {}, diagnostic, path + "/descriptor") ||
                !digestValue(descriptor["contract_hash"]))
                return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor",
                            "invalid opaque descriptor");
            storage.opaqueContractHash = descriptor["contract_hash"].get<std::string>();
        } else {
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/descriptor",
                        "unknown storage descriptor tag");
        }
        program.storages.push_back(std::move(storage));
    }

    for (size_t index = 0; index < value["values"].size(); ++index) {
        const auto &row = value["values"][index];
        const std::string path = "/values/" + std::to_string(index);
        Value parsed;
        if (!exactObject(row, {"id", "type", "origin"}, {"name", "shape", "storage", "value_layout"}, diagnostic,
                         path) ||
            !uint32Value(row["id"], parsed.id) || parsed.id != index || !row["type"].is_string() ||
            row["type"].get_ref<const std::string &>().empty() ||
            !parseOrigin(row["origin"], parsed.origin, diagnostic, path + "/origin"))
            return false;
        parsed.name = row.value("name", "");
        parsed.type = row["type"].get<std::string>();
        if (!parseCanonicalValueType(parsed.type, parsed.canonicalType))
            return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", path + "/type",
                        "Value has an invalid canonical tensor type");
        if (row.contains("shape") && !parseShape(row["shape"], parsed.shape, diagnostic, path + "/shape", true, true))
            return false;
        if (row.contains("storage")) {
            uint32_t storage = 0;
            if (!uint32Value(row["storage"], storage))
                return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "parse", path + "/storage", "invalid Storage id");
            parsed.storage = storage;
        }
        if (row.contains("value_layout")) {
            ValueLayout layout;
            if (!parseLayout(row["value_layout"], layout, diagnostic, path + "/value_layout"))
                return false;
            parsed.layout = std::move(layout);
        }
        if (parsed.canonicalType.rankedValue && parsed.layout && parsed.layout->scope == "value" &&
            parsed.layout->leaves.size() == 1 && parsed.layout->leaves.front().path.empty() &&
            (parsed.layout->leaves.front().dtype != parsed.canonicalType.dtype ||
             parsed.layout->leaves.front().shape != parsed.canonicalType.innerShape))
            return fail(diagnostic, "PROGRAM_LAYOUT_HASH", "parse", path + "/value_layout",
                        "canonical tensor type disagrees with its ValueLayout");
        program.values.push_back(std::move(parsed));
    }

    for (size_t graphIndex = 0; graphIndex < value["graphs"].size(); ++graphIndex) {
        const auto &row = value["graphs"][graphIndex];
        const std::string path = "/graphs/" + std::to_string(graphIndex);
        Graph graph;
        if (!exactObject(row, {"name", "direction", "inputs", "captures", "outputs", "nodes"}, {}, diagnostic, path) ||
            !row["name"].is_string() || !row["direction"].is_string() || !row["inputs"].is_array() ||
            !row["captures"].is_array() || !row["outputs"].is_array() || !row["nodes"].is_array())
            return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", path, "invalid Graph");
        graph.name = row["name"].get<std::string>();
        graph.direction = row["direction"].get<std::string>();
        if (graph.name != graph.direction || (graph.direction != "forward" && graph.direction != "backward"))
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/direction",
                        "graph name must equal direction and be forward or backward");
        if ((graphIndex == 0 && graph.direction != "forward") || (graphIndex == 1 && graph.direction != "backward") ||
            graphIndex >= 2)
            return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", path + "/direction",
                        "graphs must be exactly one forward graph followed by an optional backward graph");
        if (graph.direction == "forward") {
            if (!row["captures"].empty())
                return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", path + "/captures",
                            "forward captures must be empty");
        } else {
            for (size_t captureIndex = 0; captureIndex < row["captures"].size(); ++captureIndex) {
                const auto &captureValue = row["captures"][captureIndex];
                const std::string capturePath = path + "/captures/" + std::to_string(captureIndex);
                uint32_t capture = 0;
                if (!exactObject(captureValue, {"value"}, {}, diagnostic, capturePath) ||
                    !uint32Value(captureValue["value"], capture))
                    return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", capturePath, "invalid capture");
                graph.captures.push_back(capture);
            }
        }
        for (size_t inputIndex = 0; inputIndex < row["inputs"].size(); ++inputIndex) {
            const auto &inputValue = row["inputs"][inputIndex];
            const std::string inputPath = path + "/inputs/" + std::to_string(inputIndex);
            if (!inputValue.is_object() || !inputValue.contains("tag") || !inputValue["tag"].is_string())
                return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", inputPath, "invalid GraphInput");
            GraphInput input;
            const std::string tag = inputValue["tag"].get<std::string>();
            if (tag == "user_input") {
                if (!exactObject(inputValue, {"tag", "value", "slot"}, {}, diagnostic, inputPath) ||
                    !uint32Value(inputValue["value"], input.value) || !uint32Value(inputValue["slot"], input.slot))
                    return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "parse", inputPath, "invalid user input");
                input.kind = GraphInputKind::UserInput;
            } else if (tag == "invocation_control") {
                if (!exactObject(inputValue, {"tag", "value", "axis"}, {}, diagnostic, inputPath) ||
                    !uint32Value(inputValue["value"], input.value) || !uint32Value(inputValue["axis"], input.axis) ||
                    input.axis >= 3)
                    return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "parse", inputPath,
                                "invalid invocation control input");
                input.kind = GraphInputKind::InvocationControl;
            } else if (tag == "control") {
                if (!exactObject(inputValue, {"tag", "value"}, {}, diagnostic, inputPath) ||
                    !uint32Value(inputValue["value"], input.value))
                    return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "parse", inputPath,
                                "invalid typed control input");
                input.kind = GraphInputKind::Control;
            } else if (tag == "parameter") {
                if (!exactObject(inputValue, {"tag", "value", "parameter"}, {}, diagnostic, inputPath) ||
                    !uint32Value(inputValue["value"], input.value) ||
                    !uint32Value(inputValue["parameter"], input.parameter))
                    return fail(diagnostic, "PROGRAM_PARAMETER_BINDING", "parse", inputPath, "invalid parameter input");
                input.kind = GraphInputKind::Parameter;
            } else if (tag == "allocation") {
                if (!exactObject(inputValue, {"tag", "value", "storage"}, {}, diagnostic, inputPath) ||
                    !uint32Value(inputValue["value"], input.value) ||
                    !uint32Value(inputValue["storage"], input.storage))
                    return fail(diagnostic, "PROGRAM_STORAGE_INITIAL_VALUE", "parse", inputPath,
                                "invalid allocation input");
                input.kind = GraphInputKind::Allocation;
            } else {
                return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", inputPath + "/tag",
                            "GraphInput is not implemented by phase-one compute");
            }
            graph.inputs.push_back(input);
        }
        for (size_t outputIndex = 0; outputIndex < row["outputs"].size(); ++outputIndex) {
            const auto &outputValue = row["outputs"][outputIndex];
            const std::string outputPath = path + "/outputs/" + std::to_string(outputIndex);
            GraphOutput output;
            if (!exactObject(outputValue, {"tag", "value", "disposition"}, {}, diagnostic, outputPath) ||
                !outputValue["tag"].is_string() || outputValue["tag"] != "user_output" ||
                !uint32Value(outputValue["value"], output.value) || !outputValue["disposition"].is_string())
                return fail(diagnostic, "PROGRAM_RESOURCE_OUTPUT", "parse", outputPath, "invalid user output");
            output.disposition = outputValue["disposition"].get<std::string>();
            graph.outputs.push_back(std::move(output));
        }
        for (size_t nodeIndex = 0; nodeIndex < row["nodes"].size(); ++nodeIndex) {
            const auto &nodeValue = row["nodes"][nodeIndex];
            const std::string nodePath = path + "/nodes/" + std::to_string(nodeIndex);
            Node node;
            if (!exactObject(nodeValue, {"id", "stage", "operands", "results", "bindings", "accesses", "operation"},
                             {"name"}, diagnostic, nodePath))
                return false;
            if (!uint32Value(nodeValue["id"], node.id) || node.id != nodeIndex || !nodeValue["stage"].is_string() ||
                !parseIdArray(nodeValue["operands"], node.operands, diagnostic, nodePath + "/operands") ||
                !parseIdArray(nodeValue["results"], node.results, diagnostic, nodePath + "/results") ||
                !sortedUnique(node.operands) || !sortedUnique(node.results) || !nodeValue["bindings"].is_array() ||
                !nodeValue["accesses"].is_array()) {
                return fail(diagnostic, "PROGRAM_NON_CANONICAL_ORDER", "parse", nodePath,
                            "invalid or non-canonical Node");
            }
            node.name = nodeValue.value("name", "");
            node.stage = nodeValue["stage"].get<std::string>();
            for (size_t bindingIndex = 0; bindingIndex < nodeValue["bindings"].size(); ++bindingIndex) {
                const auto &bindingValue = nodeValue["bindings"][bindingIndex];
                const std::string bindingPath = nodePath + "/bindings/" + std::to_string(bindingIndex);
                EndpointBinding binding;
                const std::string module = bindingValue.value("module", "");
                if (!exactObject(bindingValue, {"module", "interface", "index", "tag"},
                                 {"projections", "access", "leaf"}, diagnostic, bindingPath) ||
                    !bindingValue["module"].is_string() ||
                    (module != "compute" && module != "vertex" && module != "fragment") ||
                    !bindingValue["interface"].is_string() || !uint32Value(bindingValue["index"], binding.index) ||
                    !bindingValue["tag"].is_string())
                    return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", bindingPath,
                                "invalid EndpointBinding");
                binding.module = module;
                binding.interfaceKind = bindingValue["interface"].get<std::string>();
                if (bindingValue["tag"] == "value") {
                    if (!bindingValue.contains("projections") || !bindingValue["projections"].is_array() ||
                        bindingValue["projections"].empty() || bindingValue.contains("access"))
                        return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", bindingPath,
                                    "invalid value binding");
                    binding.tag = BindingTag::Value;
                    std::set<uint32_t> physicalLeaves;
                    for (size_t projectionIndex = 0; projectionIndex < bindingValue["projections"].size();
                         ++projectionIndex) {
                        const auto &projectionValue = bindingValue["projections"][projectionIndex];
                        const std::string projectionPath =
                            bindingPath + "/projections/" + std::to_string(projectionIndex);
                        ValueEndpointProjection projection;
                        if (!exactObject(projectionValue, {"value", "physical_leaf", "direction"}, {"leaf"}, diagnostic,
                                         projectionPath) ||
                            !uint32Value(projectionValue["value"], projection.value) ||
                            !uint32Value(projectionValue["physical_leaf"], projection.physicalLeaf) ||
                            !physicalLeaves.insert(projection.physicalLeaf).second ||
                            !projectionValue["direction"].is_string())
                            return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", projectionPath,
                                        "invalid value projection");
                        const std::string direction = projectionValue["direction"].get<std::string>();
                        if (direction == "input")
                            projection.direction = ValueBindingDirection::Input;
                        else if (direction == "result")
                            projection.direction = ValueBindingDirection::Result;
                        else
                            return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", projectionPath + "/direction",
                                        "unknown value projection direction");
                        if (projectionValue.contains("leaf")) {
                            uint32_t leaf = 0;
                            if (!uint32Value(projectionValue["leaf"], leaf))
                                return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", projectionPath + "/leaf",
                                            "invalid value projection leaf");
                            projection.leaf = leaf;
                        }
                        binding.projections.push_back(std::move(projection));
                    }
                } else if (bindingValue["tag"] == "resource") {
                    if (!bindingValue.contains("access") || bindingValue.contains("projections") ||
                        !uint32Value(bindingValue["access"], binding.access))
                        return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", bindingPath,
                                    "invalid resource binding");
                    binding.tag = BindingTag::Resource;
                } else {
                    return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", bindingPath + "/tag",
                                "unknown binding tag");
                }
                if (bindingValue.contains("leaf")) {
                    if (binding.tag != BindingTag::Resource)
                        return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", bindingPath + "/leaf",
                                    "only resource bindings may project a leaf directly");
                    uint32_t leaf = 0;
                    if (!uint32Value(bindingValue["leaf"], leaf))
                        return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "parse", bindingPath + "/leaf",
                                    "invalid leaf");
                    binding.resourceLeaf = leaf;
                }
                node.bindings.push_back(std::move(binding));
            }
            for (size_t accessIndex = 0; accessIndex < nodeValue["accesses"].size(); ++accessIndex) {
                const auto &accessValue = nodeValue["accesses"][accessIndex];
                const std::string accessPath = nodePath + "/accesses/" + std::to_string(accessIndex);
                if (!accessValue.is_object() || !accessValue.contains("tag") || !accessValue["tag"].is_string())
                    return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse", accessPath, "invalid ResourceAccess");
                ResourceAccess access;
                const std::string tag = accessValue["tag"].get<std::string>();
                if (tag == "read") {
                    if (!exactObject(accessValue, {"tag", "storage", "value"}, {"view"}, diagnostic, accessPath) ||
                        !uint32Value(accessValue["storage"], access.storage) ||
                        !uint32Value(accessValue["value"], access.value))
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse", accessPath, "invalid read access");
                    access.kind = AccessKind::Read;
                } else if (tag == "initialize") {
                    if (!exactObject(accessValue, {"tag", "storage", "after"}, {"view"}, diagnostic, accessPath) ||
                        !uint32Value(accessValue["storage"], access.storage) ||
                        !uint32Value(accessValue["after"], access.after))
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse", accessPath,
                                    "invalid initialize access");
                    access.kind = AccessKind::Initialize;
                } else if (tag == "write") {
                    if (!exactObject(accessValue, {"tag", "storage", "before", "after", "access"}, {"view"}, diagnostic,
                                     accessPath) ||
                        !uint32Value(accessValue["storage"], access.storage) ||
                        !uint32Value(accessValue["before"], access.before) ||
                        !uint32Value(accessValue["after"], access.after) || !accessValue["access"].is_string())
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse", accessPath, "invalid write access");
                    access.kind = AccessKind::Write;
                    access.access = accessValue["access"].get<std::string>();
                } else if (tag == "attachment") {
                    if (!exactObject(accessValue, {"tag", "storage", "before", "after"}, {"view"}, diagnostic,
                                     accessPath) ||
                        !uint32Value(accessValue["storage"], access.storage) ||
                        !uint32Value(accessValue["before"], access.before) ||
                        !uint32Value(accessValue["after"], access.after))
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse", accessPath,
                                    "invalid attachment access");
                    access.kind = AccessKind::Attachment;
                    access.access = "read_write";
                } else {
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", accessPath + "/tag",
                                "ResourceAccess is not implemented by phase-one compute");
                }
                if (accessValue.contains("view")) {
                    uint32_t view = 0;
                    if (!uint32Value(accessValue["view"], view))
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse", accessPath + "/view",
                                    "invalid access view");
                    access.view = view;
                }
                node.accesses.push_back(std::move(access));
            }
            const auto &operation = nodeValue["operation"];
            if (!operation.is_object() || !operation.contains("tag") || !operation["tag"].is_string())
                return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", nodePath + "/operation",
                            "node operation requires a tag");
            const std::string operationTag = operation["tag"].get<std::string>();
            if (operationTag == "compute") {
                ComputeOperation compute;
                if (!exactObject(operation, {"tag", "workgroups"}, {}, diagnostic, nodePath + "/operation") ||
                    !operation["workgroups"].is_array() || operation["workgroups"].size() != 3)
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", nodePath + "/operation",
                                "compute node must be a direct dispatch");
                for (size_t axis = 0; axis < 3; ++axis)
                    if (!parseControl(operation["workgroups"][axis], compute.workgroups[axis], diagnostic,
                                      nodePath + "/operation/workgroups/" + std::to_string(axis)))
                        return false;
                node.operation = std::move(compute);
            } else if (operationTag == "graphics") {
                GraphicsOperation graphics;
                if (!exactObject(operation, {"tag", "pipeline_state", "render_pass", "draw", "dynamic_state"}, {},
                                 diagnostic, nodePath + "/operation") ||
                    !parseGraphicsPipelineState(operation["pipeline_state"], graphics.pipelineState, diagnostic,
                                                nodePath + "/operation/pipeline_state"))
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", nodePath + "/operation",
                                "invalid graphics operation");
                const auto &renderPass = operation["render_pass"];
                if (!exactObject(renderPass, {"control", "colors", "depth_stencil"}, {}, diagnostic,
                                 nodePath + "/operation/render_pass") ||
                    !uint32Value(renderPass["control"], graphics.renderPassControl) || !renderPass["colors"].is_array())
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse",
                                nodePath + "/operation/render_pass", "invalid graphics render-pass control");
                if (renderPass["colors"].empty() && renderPass["depth_stencil"].is_null())
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse",
                                nodePath + "/operation/render_pass",
                                "graphics render pass requires a color or depth-stencil attachment");
                for (size_t colorIndex = 0; colorIndex < renderPass["colors"].size(); ++colorIndex) {
                    GraphicsAttachmentSignature signature;
                    if (!parseAttachmentSignature(renderPass["colors"][colorIndex], true, signature, diagnostic,
                                                  nodePath + "/operation/render_pass/colors/" +
                                                      std::to_string(colorIndex)) ||
                        signature.location != colorIndex)
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "parse",
                                    nodePath + "/operation/render_pass/colors/" + std::to_string(colorIndex),
                                    "color attachment locations must be contiguous from zero");
                    graphics.colorAttachments.push_back(std::move(signature));
                }
                if (!renderPass["depth_stencil"].is_null()) {
                    GraphicsAttachmentSignature signature;
                    if (!parseAttachmentSignature(renderPass["depth_stencil"], false, signature, diagnostic,
                                                  nodePath + "/operation/render_pass/depth_stencil"))
                        return false;
                    graphics.depthStencilAttachment = std::move(signature);
                }
                if (graphics.pipelineState.colorBlends.size() != graphics.colorAttachments.size())
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse",
                                nodePath + "/operation/pipeline_state/color_blends",
                                "pipeline state must define one blend for every color attachment");
                for (size_t location = 0; location < graphics.colorAttachments.size(); ++location)
                    if (!graphics.pipelineState.colorBlends.count(static_cast<uint32_t>(location)))
                        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse",
                                    nodePath + "/operation/pipeline_state/color_blends",
                                    "pipeline color blend locations must be contiguous from zero");
                const auto &draw = operation["draw"];
                if (!exactObject(draw, {"control", "default"}, {}, diagnostic, nodePath + "/operation/draw") ||
                    !uint32Value(draw["control"], graphics.drawCommandControl) || !draw["default"].is_object())
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", nodePath + "/operation/draw",
                                "invalid graphics draw control");
                const auto &defaultDraw = draw["default"];
                if (!exactObject(defaultDraw, {"tag", "vertex_count", "instance_count"}, {}, diagnostic,
                                 nodePath + "/operation/draw") ||
                    defaultDraw.value("tag", "") != "direct" ||
                    !uint32Value(defaultDraw["vertex_count"], graphics.vertexCount) ||
                    !uint32Value(defaultDraw["instance_count"], graphics.instanceCount) || !graphics.instanceCount)
                    return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", nodePath + "/operation/draw",
                                "invalid direct draw");
                const auto &dynamicState = operation["dynamic_state"];
                if (!exactObject(dynamicState, {"control"}, {}, diagnostic, nodePath + "/operation/dynamic_state") ||
                    !uint32Value(dynamicState["control"], graphics.dynamicStateControl))
                    return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "parse",
                                nodePath + "/operation/dynamic_state", "invalid graphics dynamic-state control");
                node.operation = std::move(graphics);
            } else {
                return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "parse", nodePath + "/operation/tag",
                            "unknown operation tag");
            }
            graph.nodes.push_back(std::move(node));
        }
        program.graphs.push_back(std::move(graph));
    }

    const auto &abi = value["abi"];
    if (!exactObject(abi, {"boundary_slots", "derivative_projections", "tape_plans"}, {}, diagnostic, "/abi") ||
        !abi["boundary_slots"].is_array() || !abi["derivative_projections"].is_array() || !abi["tape_plans"].is_array())
        return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", "/abi", "invalid ProgramABI");
    for (size_t index = 0; index < abi["boundary_slots"].size(); ++index) {
        const auto &row = abi["boundary_slots"][index];
        const std::string path = "/abi/boundary_slots/" + std::to_string(index);
        BoundarySlot slot;
        if (!exactObject(row,
                         {"id", "path", "value", "role", "direction", "category", "access", "logical_type",
                          "outer_shape", "alias_owner"},
                         {"value_layout", "storage_id", "storage_descriptor", "publication"}, diagnostic, path))
            return false;
        if (!uint32Value(row["id"], slot.id) || slot.id != index || !row["path"].is_string() ||
            row["path"].get_ref<const std::string &>().empty() || !uint32Value(row["value"], slot.value) ||
            !row["role"].is_string() || !row["direction"].is_string() || !row["category"].is_string() ||
            !row["access"].is_string() || !row["logical_type"].is_string() || !row["alias_owner"].is_string())
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path, "invalid ProgramABI boundary slot");
        slot.path = row["path"].get<std::string>();
        const std::string role = row["role"].get<std::string>();
        if (role == "input")
            slot.role = BoundaryRole::Input;
        else if (role == "output")
            slot.role = BoundaryRole::Output;
        else if (role == "cotangent")
            slot.role = BoundaryRole::Cotangent;
        else if (role == "gradient")
            slot.role = BoundaryRole::Gradient;
        else
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/role",
                        "unknown ProgramABI boundary role");
        const BoundaryDirection roleDirection = slot.role == BoundaryRole::Input || slot.role == BoundaryRole::Cotangent
                                                    ? BoundaryDirection::Input
                                                    : BoundaryDirection::Output;
        if (row["direction"] == "input")
            slot.direction = BoundaryDirection::Input;
        else if (row["direction"] == "output")
            slot.direction = BoundaryDirection::Output;
        else
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/direction",
                        "unknown ProgramABI boundary direction");
        if (slot.direction != roleDirection)
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/direction",
                        "ProgramABI boundary role and direction disagree");
        if (slot.value >= program.values.size())
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path, "ProgramABI boundary Value is inconsistent");
        const Value &logicalValue = program.values[slot.value];
        slot.logicalType = row["logical_type"].get<std::string>();
        if (slot.logicalType != logicalValue.type)
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/logical_type",
                        "ProgramABI logical type disagrees with its Value");
        if (!parseShape(row["outer_shape"], slot.outerShape, diagnostic, path + "/outer_shape", true, true))
            return false;
        if (slot.outerShape != logicalValue.shape)
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/outer_shape",
                        "ProgramABI outer shape disagrees with its Value");
        const bool hasValueLayout = row.contains("value_layout");
        if (hasValueLayout != logicalValue.layout.has_value())
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/value_layout",
                        "ProgramABI ValueLayout presence disagrees with its Value");
        if (hasValueLayout) {
            ValueLayout layout;
            if (!parseLayout(row["value_layout"], layout, diagnostic, path + "/value_layout"))
                return false;
            if (row["value_layout"] != value["values"][slot.value]["value_layout"])
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/value_layout",
                            "ProgramABI ValueLayout disagrees with its Value");
            slot.layout = std::move(layout);
        }
        const std::string category = row["category"].get<std::string>();
        if (category == "value")
            slot.category = BoundaryCategory::Value;
        else if (category == "storage_view")
            slot.category = BoundaryCategory::StorageView;
        else if (category == "texture")
            slot.category = BoundaryCategory::Texture;
        else if (category == "sampler")
            slot.category = BoundaryCategory::Sampler;
        else
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/category",
                        "unknown ProgramABI boundary category");
        const std::string access = row["access"].get<std::string>();
        if (access == "read")
            slot.access = BoundaryAccess::Read;
        else if (access == "write")
            slot.access = BoundaryAccess::Write;
        else if (access == "read_write")
            slot.access = BoundaryAccess::ReadWrite;
        else
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/access",
                        "unknown ProgramABI boundary access");
        if (logicalValue.storage) {
            uint32_t serializedStorage{};
            if (!row.contains("storage_id") || !uint32Value(row["storage_id"], serializedStorage) ||
                serializedStorage != *logicalValue.storage)
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/storage_id",
                            "ProgramABI Storage id disagrees with its Value");
            if (!row.contains("storage_descriptor"))
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/storage_descriptor",
                            "resource ProgramABI boundary requires a Storage descriptor");
            BoundaryStorage boundaryStorage;
            boundaryStorage.id = serializedStorage;
            if (boundaryStorage.id >= program.storages.size())
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path,
                            "ProgramABI boundary Value references an unknown Storage");
            const Storage &storage = program.storages[boundaryStorage.id];
            if (row["storage_descriptor"] != value["storages"][boundaryStorage.id]["descriptor"])
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/storage_descriptor",
                            "ProgramABI Storage descriptor disagrees with its Storage");
            boundaryStorage.descriptorKind = storage.descriptorKind;
            boundaryStorage.buffer = storage.buffer;
            boundaryStorage.image = storage.image;
            boundaryStorage.opaqueContractHash = storage.opaqueContractHash;
            const BoundaryCategory storageCategory =
                storage.descriptorKind == StorageDescriptorKind::Buffer  ? BoundaryCategory::StorageView
                : storage.descriptorKind == StorageDescriptorKind::Image ? BoundaryCategory::Texture
                                                                         : BoundaryCategory::Sampler;
            if (slot.category != storageCategory ||
                (storage.descriptorKind == StorageDescriptorKind::Opaque && logicalValue.type != "!vernon.sampler"))
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/category",
                            "ProgramABI category disagrees with its Storage");
            slot.storage = std::move(boundaryStorage);
            slot.aliasOwner = {ProgramOwnerKind::Storage, *logicalValue.storage};
        } else {
            if (row.contains("storage_id") || row.contains("storage_descriptor") ||
                slot.category != BoundaryCategory::Value)
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/category",
                            "by-value ProgramABI boundary has resource metadata");
            slot.aliasOwner = {ProgramOwnerKind::Value, logicalValue.id};
        }
        const std::string expectedOwner =
            std::string(slot.aliasOwner.kind == ProgramOwnerKind::Storage ? "storage:" : "value:") +
            std::to_string(slot.aliasOwner.id);
        if (row["alias_owner"] != expectedOwner)
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/alias_owner",
                        "ProgramABI alias owner disagrees with its Value");
        if (slot.access != canonicalBoundaryAccess(program, slot.direction, logicalValue.storage))
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/access",
                        "ProgramABI access disagrees with the canonical Program");
        if (slot.direction == BoundaryDirection::Input) {
            if (row.contains("publication"))
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/publication",
                            "input ProgramABI boundary cannot publish");
        } else {
            if (!row.contains("publication") || !row["publication"].is_string())
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/publication",
                            "output ProgramABI boundary requires publication");
            if (row["publication"] == "commit_after_success") {
                slot.publication = BoundaryPublication::CommitAfterSuccess;
            } else if (row["publication"] == "in_place") {
                if (!logicalValue.storage ||
                    program.storages[*logicalValue.storage].mutability != StorageMutability::Mutable)
                    return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/publication",
                                "in_place publication requires mutable Storage-backed output");
                slot.publication = BoundaryPublication::InPlace;
            } else {
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/publication",
                            "unknown ProgramABI publication policy");
            }
        }
        program.abi.boundarySlots.push_back(std::move(slot));
    }
    program.abi.publication = derivePublicationPlan(program.abi.boundarySlots);

    std::set<uint32_t> projectedDerivatives;
    for (size_t index = 0; index < abi["derivative_projections"].size(); ++index) {
        const auto &row = abi["derivative_projections"][index];
        const std::string path = "/abi/derivative_projections/" + std::to_string(index);
        DerivativeProjection projection;
        if (!exactObject(row, {"derivative", "primal", "value_path"}, {}, diagnostic, path) ||
            !exactObject(row["derivative"], {"slot", "path"}, {}, diagnostic, path + "/derivative") ||
            !exactObject(row["primal"], {"slot", "path"}, {}, diagnostic, path + "/primal") ||
            !uint32Value(row["derivative"]["slot"], projection.derivative.slot) ||
            !uint32Value(row["primal"]["slot"], projection.primal.slot) || !row["derivative"]["path"].is_string() ||
            !row["primal"]["path"].is_string() || !row["value_path"].is_array() ||
            projection.derivative.slot >= program.abi.boundarySlots.size() ||
            projection.primal.slot >= program.abi.boundarySlots.size())
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path, "invalid derivative projection");
        projection.derivative.path = row["derivative"]["path"].get<std::string>();
        projection.primal.path = row["primal"]["path"].get<std::string>();
        const BoundarySlot &derivative = program.abi.boundarySlots[projection.derivative.slot];
        const BoundarySlot &primal = program.abi.boundarySlots[projection.primal.slot];
        const bool roleMatch = (derivative.role == BoundaryRole::Cotangent && primal.role == BoundaryRole::Output) ||
                               (derivative.role == BoundaryRole::Gradient && primal.role == BoundaryRole::Input);
        if (!projectedDerivatives.insert(projection.derivative.slot).second ||
            projection.derivative.path != derivative.path || projection.primal.path != primal.path || !roleMatch)
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path,
                        "derivative projection does not reference compatible boundary slots and paths");
        for (size_t component = 0; component < row["value_path"].size(); ++component) {
            LayoutPathComponent parsed;
            const auto &componentValue = row["value_path"][component];
            if (componentValue.is_string() && !componentValue.get_ref<const std::string &>().empty())
                parsed.field = componentValue.get<std::string>();
            else if (!uint32Value(componentValue, parsed.index.emplace()))
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse",
                            path + "/value_path/" + std::to_string(component), "invalid projection path component");
            projection.valuePath.push_back(std::move(parsed));
        }
        std::string projectedPath = projection.primal.path;
        for (const LayoutPathComponent &component : projection.valuePath) {
            projectedPath.push_back('.');
            projectedPath += component.index ? std::to_string(*component.index) : component.field;
        }
        if (projectedPath != projection.derivative.path)
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path + "/value_path",
                        "projection path does not resolve to the derivative boundary path");
        program.abi.derivativeProjections.push_back(std::move(projection));
    }
    const size_t derivativeCount = static_cast<size_t>(
        std::count_if(program.abi.boundarySlots.begin(), program.abi.boundarySlots.end(), [](const BoundarySlot &slot) {
            return slot.role == BoundaryRole::Cotangent || slot.role == BoundaryRole::Gradient;
        }));
    if (program.abi.derivativeProjections.size() != derivativeCount)
        return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", "/abi/derivative_projections",
                    "every derivative boundary slot requires exactly one projection");

    std::vector<uint32_t> tapeValues;
    for (const Value &logicalValue : program.values)
        if (isTapeValueType(logicalValue.type))
            tapeValues.push_back(logicalValue.id);
    if (abi["tape_plans"].size() != tapeValues.size())
        return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", "/abi/tape_plans",
                    "typed tape plans must exactly cover Program tape values");
    for (size_t index = 0; index < abi["tape_plans"].size(); ++index) {
        const auto &row = abi["tape_plans"][index];
        const std::string path = "/abi/tape_plans/" + std::to_string(index);
        TapePlan plan;
        if (!exactObject(row,
                         {"value", "forward_producer", "backward_consumer", "required_carriers", "optional_carriers"},
                         {}, diagnostic, path) ||
            !uint32Value(row["value"], plan.value) || plan.value != tapeValues[index] ||
            !row["forward_producer"].is_boolean() || !row["backward_consumer"].is_boolean() ||
            !parseTapeCarriers(row["required_carriers"], plan.requiredCarriers) ||
            !parseTapeCarriers(row["optional_carriers"], plan.optionalCarriers) ||
            plan.requiredCarriers != std::vector<program_plan::TapeCarrier>{program_plan::TapeCarrier::TapeData,
                                                                            program_plan::TapeCarrier::ReplaySegment} ||
            plan.optionalCarriers != std::vector<program_plan::TapeCarrier>{program_plan::TapeCarrier::LaunchMetadata,
                                                                            program_plan::TapeCarrier::ReplayStatus})
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "parse", path, "invalid typed tape plan");
        plan.forwardProducer = row["forward_producer"].get<bool>();
        plan.backwardConsumer = row["backward_consumer"].get<bool>();
        program.abi.tapePlans.push_back(std::move(plan));
    }

    const bool hasBackward = program.graphs.size() == 2;
    if (hasBackward != value.contains("residual_contract"))
        return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", "/residual_contract",
                    "residual_contract is required exactly when a backward graph exists");
    if (!hasBackward)
        return true;
    const auto &residual = value["residual_contract"];
    if (!exactObject(residual, {"captures"}, {}, diagnostic, "/residual_contract") || !residual["captures"].is_array())
        return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", "/residual_contract",
                    "invalid residual_contract");
    ResidualContract contract;
    const Graph &backward = program.graphs.back();
    if (residual["captures"].size() != backward.captures.size())
        return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", "/residual_contract/captures",
                    "residual captures must equal the backward graph capture array");
    for (size_t index = 0; index < residual["captures"].size(); ++index) {
        const auto &row = residual["captures"][index];
        const std::string path = "/residual_contract/captures/" + std::to_string(index);
        ResidualCapture capture;
        if (!exactObject(row, {"value", "replay"}, {}, diagnostic, path) || !uint32Value(row["value"], capture.value) ||
            capture.value != backward.captures[index] || !row["replay"].is_object())
            return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", path, "invalid residual capture");
        const auto &replay = row["replay"];
        if (!exactObject(replay, {"legal", "required_values", "cost"}, {}, diagnostic, path + "/replay") ||
            !replay["legal"].is_boolean() || !replay["required_values"].is_array() ||
            (!replay["legal"].get<bool>() && !replay["required_values"].empty()))
            return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", path + "/replay", "invalid capture replay");
        uint64_t cost = 0;
        if (!uint64Value(replay["cost"], cost))
            return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "parse", path + "/replay/cost", "invalid replay cost");
        contract.captures.push_back(capture);
    }
    program.residualContract = std::move(contract);
    return true;
}

bool parseArtifactSystem(const nlohmann::json &target, const nlohmann::json &blobs, const nlohmann::json &value,
                         ArtifactSystem &artifacts, Diagnostic &diagnostic) {
    artifacts = {};
    diagnostic = {};
    if (!exactObject(value, {"runtime_requirements", "artifacts"}, {}, diagnostic, "/artifact_system"))
        return false;
    if (!exactObject(target, {"kind", "options"}, {}, diagnostic, "/target") || !target["kind"].is_string() ||
        !target["options"].is_object())
        return fail(diagnostic, "PROGRAM_ARTIFACT_TARGET", "parse", "/target",
                    "invalid single-compute ArtifactSystem target");
    artifacts.target = target["kind"].get<std::string>();
    if (artifacts.target != "cpu" && artifacts.target != "cuda" && artifacts.target != "vulkan" &&
        artifacts.target != "opengl" && artifacts.target != "opengles" && artifacts.target != "metal" &&
        artifacts.target != "directx")
        return fail(diagnostic, "PROGRAM_ARTIFACT_TARGET", "parse", "/target/kind",
                    "ArtifactSystem target kind must be cpu, cuda, vulkan, opengl, opengles, metal, or directx");
    if (!blobs.is_object() || !value["runtime_requirements"].is_object() || !value["artifacts"].is_object())
        return fail(diagnostic, "PROGRAM_UNKNOWN_FIELD", "parse", "/artifact_system",
                    "runtime_requirements, artifacts, and root blobs must be objects");

    const auto parseCodeBlob = [&](const std::string &blobId) {
        if (artifacts.blobs.find(blobId) != artifacts.blobs.end())
            return true;
        const std::string path = "/blobs/" + blobId;
        const auto found = blobs.find(blobId);
        if (found == blobs.end())
            return fail(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "parse", path,
                        "CodeModule references an unknown Blob");
        const nlohmann::json &row = *found;
        Blob blob;
        if (blobId.empty() || !exactObject(row, {"byte_length", "sha256", "location"}, {}, diagnostic, path) ||
            !uint64Value(row["byte_length"], blob.byteLength) || !blob.byteLength || !digestValue(row["sha256"]))
            return fail(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "parse", path, "invalid Blob");
        const auto &location = row["location"];
        if (!exactObject(location, {"tag", "uri"}, {}, diagnostic, path + "/location") ||
            !location["tag"].is_string() || location["tag"] != "external" || !location["uri"].is_string() ||
            location["uri"].get_ref<const std::string &>().empty())
            return fail(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "parse", path + "/location",
                        "code Blob must use an external location");
        blob.sha256 = row["sha256"].get<std::string>();
        blob.uri = location["uri"].get<std::string>();
        if (blob.uri.front() == '/' || blob.uri.find("://") != std::string::npos ||
            blob.uri.find("..") != std::string::npos)
            return fail(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "parse", path + "/location/uri",
                        "external Blob URI is not normalized beneath the bundle");
        artifacts.blobs.emplace(blobId, std::move(blob));
        return true;
    };

    std::vector<nlohmann::json> stageRequirements;
    for (const auto &[artifactId, row] : value["artifacts"].items()) {
        const std::string path = "/artifact_system/artifacts/" + artifactId;
        StageArtifact stage;
        const std::string operation = row.value("operation", "");
        if (artifactId.empty() ||
            !exactObject(row, {"tag", "operation", "contract_hash", "runtime_requirements", "modules", "reflection"},
                         {"implementation"}, diagnostic, path) ||
            !row["tag"].is_string() || row["tag"] != "stage" || !row["operation"].is_string() ||
            (operation != "compute" && operation != "graphics") || !digestValue(row["contract_hash"]) ||
            !row["modules"].is_array() || row["modules"].empty() ||
            (operation == "compute" ? row["modules"].size() != 1 : row["modules"].size() > 2))
            return fail(diagnostic, "PROGRAM_ARTIFACT_MODULE", "parse", path,
                        "artifact has an invalid compute or graphics module set");
        stage.operation = operation;
        stage.contractHash = row["contract_hash"].get<std::string>();

        const auto &requirements = row["runtime_requirements"];
        const std::initializer_list<std::string_view> requirementFields =
            artifacts.target == "cpu"
                ? std::initializer_list<std::string_view>{"backend", "features", "target_triple", "object_format"}
            : artifacts.target == "metal"
                ? std::initializer_list<std::string_view>{"backend", "features", "apple_platform", "msl_version",
                                                          "minimum_os_version"}
            : artifacts.target == "opengl" || artifacts.target == "opengles"
                ? std::initializer_list<std::string_view>{"backend", "features", "glsl_version", "api_version",
                                                          "profile"}
            : artifacts.target == "directx"
                ? std::initializer_list<std::string_view>{"backend",      "features",
                                                          "api_version",  "minimum_feature_level",
                                                          "shader_model", "root_signature_version"}
            : artifacts.target == "cuda"
                ? std::initializer_list<std::string_view>{"backend", "features", "ptx_version", "address_size",
                                                          "minimum_compute_capability"}
                : std::initializer_list<std::string_view>{"backend", "features", "api_version", "spirv_version"};
        if (!exactObject(requirements, requirementFields, {}, diagnostic, path + "/runtime_requirements") ||
            !requirements["backend"].is_string() || requirements["backend"] != artifacts.target ||
            !stringArray(requirements["features"], stage.requiredFeatures, diagnostic,
                         path + "/runtime_requirements/features"))
            return false;
        if (artifacts.target == "cpu" &&
            (!requirements["target_triple"].is_string() ||
             requirements["target_triple"].get_ref<const std::string &>().empty() ||
             !requirements["object_format"].is_string() ||
             (requirements["object_format"] != "elf" && requirements["object_format"] != "macho" &&
              requirements["object_format"] != "coff" && requirements["object_format"] != "wasm")))
            return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                        "invalid CPU runtime requirements");
        const auto versionPair = [](const nlohmann::json &pair) {
            uint32_t first{}, second{};
            return pair.is_array() && pair.size() == 2 && uint32Value(pair[0], first) && uint32Value(pair[1], second);
        };
        if (artifacts.target == "metal" &&
            (!requirements["apple_platform"].is_string() ||
             (requirements["apple_platform"] != "macos" && requirements["apple_platform"] != "ios") ||
             !versionPair(requirements["msl_version"]) || !versionPair(requirements["minimum_os_version"])))
            return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                        "invalid Metal runtime requirements");
        if (artifacts.target == "vulkan" &&
            (!versionPair(requirements["api_version"]) || !versionPair(requirements["spirv_version"])))
            return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                        "invalid Vulkan runtime requirements");
        if ((artifacts.target == "opengl" || artifacts.target == "opengles") &&
            (!requirements["glsl_version"].is_number_integer() || !requirements["profile"].is_string() ||
             !versionPair(requirements["api_version"])))
            return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                        "invalid OpenGL runtime requirements");
        if (artifacts.target == "opengl" || artifacts.target == "opengles") {
            uint32_t glslVersion{};
            vernon::runtime::RuntimeVersion apiVersion{};
            if (!uint32Value(requirements["glsl_version"], glslVersion) ||
                !uint32Value(requirements["api_version"][0], apiVersion.major) ||
                !uint32Value(requirements["api_version"][1], apiVersion.minor) ||
                glslVersion != vernon::runtime::glslVersionForApi(apiVersion) ||
                (artifacts.target == "opengles"
                     ? requirements["profile"] != "es"
                     : requirements["profile"] != "core" && requirements["profile"] != "compatibility"))
                return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                            "invalid OpenGL runtime requirements");
        }
        if (artifacts.target == "directx" &&
            (!versionPair(requirements["api_version"]) || !versionPair(requirements["minimum_feature_level"]) ||
             !versionPair(requirements["shader_model"]) || !versionPair(requirements["root_signature_version"])))
            return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                        "invalid DirectX runtime requirements");
        if (artifacts.target == "cuda") {
            uint32_t addressSize{};
            if (!versionPair(requirements["ptx_version"]) || !versionPair(requirements["minimum_compute_capability"]) ||
                !uint32Value(requirements["address_size"], addressSize) || (addressSize != 32 && addressSize != 64))
                return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements",
                            "invalid CUDA runtime requirements");
        }
        stage.backend = artifacts.target;
        stageRequirements.push_back(requirements);

        std::set<std::string> moduleRoles;
        for (size_t moduleIndex = 0; moduleIndex < row["modules"].size(); ++moduleIndex) {
            const auto &module = row["modules"][moduleIndex];
            const std::string modulePath = path + "/modules/" + std::to_string(moduleIndex);
            CodeModule code;
            const std::string role = module.value("role", "");
            if (!exactObject(module, {"role", "format", "entry_point", "blob", "offset", "byte_length", "sha256"}, {},
                             diagnostic, modulePath) ||
                !module["role"].is_string() ||
                (operation == "compute" ? role != "compute" : role != "vertex" && role != "fragment") ||
                !moduleRoles.insert(role).second || !module["format"].is_string() ||
                module["format"] != (artifacts.target == "cpu"        ? "relocatable_object"
                                     : artifacts.target == "metal"    ? "msl"
                                     : artifacts.target == "opengl"   ? "glsl"
                                     : artifacts.target == "opengles" ? "gles"
                                     : artifacts.target == "directx"  ? "dxil"
                                     : artifacts.target == "cuda"     ? "ptx"
                                                                      : "spirv") ||
                !module["entry_point"].is_string() || module["entry_point"].get_ref<const std::string &>().empty() ||
                !module["blob"].is_string() || !uint64Value(module["offset"], code.offset) ||
                !uint64Value(module["byte_length"], code.byteLength) || !code.byteLength ||
                !digestValue(module["sha256"]))
                return fail(diagnostic, "PROGRAM_ARTIFACT_MODULE", "parse", modulePath, "invalid target CodeModule");
            code.role = role;
            code.format = module["format"].get<std::string>();
            code.entryPoint = module["entry_point"].get<std::string>();
            code.blob = module["blob"].get<std::string>();
            code.sha256 = module["sha256"].get<std::string>();
            if (!parseCodeBlob(code.blob))
                return false;
            const auto blob = artifacts.blobs.find(code.blob);
            if (blob == artifacts.blobs.end() || code.offset > blob->second.byteLength ||
                code.byteLength > blob->second.byteLength - code.offset)
                return fail(diagnostic, "PROGRAM_BLOB_AUTHENTICATION", "parse", modulePath + "/blob",
                            "CodeModule range is outside its Blob");
            stage.modules.push_back(std::move(code));
        }
        if (operation == "graphics" && !moduleRoles.count("vertex"))
            return fail(diagnostic, "PROGRAM_ARTIFACT_MODULE", "parse", path + "/modules",
                        "graphics artifact requires a vertex module");

        const auto &reflection = row["reflection"];
        std::vector<std::string> reflectedFeatures;
        if (!exactObject(reflection, {"required_features", "endpoints", operation}, {}, diagnostic,
                         path + "/reflection") ||
            !reflection["endpoints"].is_array() ||
            !stringArray(reflection["required_features"], reflectedFeatures, diagnostic,
                         path + "/reflection/required_features"))
            return false;
        if (reflectedFeatures != stage.requiredFeatures)
            return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", path + "/runtime_requirements/features",
                        "Runtime requirements must equal reflected features");
        for (size_t endpointIndex = 0; endpointIndex < reflection["endpoints"].size(); ++endpointIndex) {
            const auto &endpointValue = reflection["endpoints"][endpointIndex];
            const std::string endpointPath = path + "/reflection/endpoints/" + std::to_string(endpointIndex);
            if (!endpointValue.is_object() || !endpointValue.contains("tag") || !endpointValue["tag"].is_string())
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath,
                            "invalid reflected endpoint");
            ReflectedEndpoint endpoint;
            endpoint.tag = endpointValue["tag"].get<std::string>();
            const bool resource = endpoint.tag == "resource";
            const bool system = endpoint.tag == "system";
            const bool systemValue =
                !resource && endpointValue.contains("interface") && endpointValue["interface"] == "system_value";
            if (system) {
                if (!exactObject(endpointValue, {"tag", "module", "interface", "index", "semantic", "abi"}, {},
                                 diagnostic, endpointPath) ||
                    !endpointValue["module"].is_string() || !endpointValue["interface"].is_string() ||
                    !uint32Value(endpointValue["index"], endpoint.index) || !endpointValue["semantic"].is_string() ||
                    !endpointValue["abi"].is_object() || !endpointValue["abi"].contains("bindings") ||
                    !endpointValue["abi"]["bindings"].is_array() || !endpointValue["abi"]["bindings"].empty())
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath,
                                "invalid graphics system endpoint");
                endpoint.module = endpointValue["module"].get<std::string>();
                endpoint.interfaceKind = endpointValue["interface"].get<std::string>();
                endpoint.builtin = endpointValue["semantic"].get<std::string>();
                stage.endpoints.push_back(std::move(endpoint));
                continue;
            }
            const std::string endpointModule = endpointValue.value("module", "");
            if (!exactObject(
                    endpointValue,
                    resource ? std::initializer_list<std::string_view>{"tag", "module", "interface", "index", "role",
                                                                       "type", "layout", "address_space", "transport",
                                                                       "access", "abi"}
                    : systemValue
                        ? std::initializer_list<std::string_view>{"tag", "module", "interface", "index", "type",
                                                                  "layout_hash", "transport", "access", "builtin",
                                                                  "abi"}
                        : std::initializer_list<std::string_view>{"tag", "module", "interface", "index", "type",
                                                                  "layout_hash", "transport", "access", "abi"},
                    resource ? std::initializer_list<std::string_view>{"write_footprint"}
                             : std::initializer_list<std::string_view>{"element_layout_hash", "role"},
                    diagnostic, endpointPath) ||
                (!resource && endpoint.tag != "value") || !endpointValue["module"].is_string() ||
                (operation == "compute" ? endpointModule != "compute"
                                        : endpointModule != "vertex" && endpointModule != "fragment") ||
                !endpointValue["interface"].is_string() || !uint32Value(endpointValue["index"], endpoint.index) ||
                !endpointValue["type"].is_string() || !endpointValue["transport"].is_string() ||
                !endpointValue["access"].is_string() || !endpointValue["abi"].is_object())
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath,
                            "invalid compute endpoint reflection");
            endpoint.module = endpointModule;
            endpoint.interfaceKind = endpointValue["interface"].get<std::string>();
            endpoint.type = endpointValue["type"].get<std::string>();
            endpoint.transport = endpointValue["transport"].get<std::string>();
            endpoint.access = endpointValue["access"].get<std::string>();
            if (!resource && endpointValue.contains("role")) {
                if (!endpointValue["role"].is_string())
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/role",
                                "invalid value endpoint role");
                endpoint.role = endpointValue["role"].get<std::string>();
            }
            if (systemValue) {
                if (!endpointValue["builtin"].is_string() ||
                    endpointValue["builtin"].get_ref<const std::string &>().empty())
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/builtin",
                                "invalid compute system value");
                endpoint.builtin = endpointValue["builtin"].get<std::string>();
            }
            if (resource) {
                if (!endpointValue["role"].is_string() || !endpointValue["layout"].is_object() ||
                    !endpointValue["address_space"].is_string())
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath,
                                "invalid resource endpoint reflection");
                endpoint.role = endpointValue["role"].get<std::string>();
                const auto &layout = endpointValue["layout"];
                const bool image = layout.contains("tag") && layout["tag"] == "image";
                const bool sampler = layout.contains("tag") && layout["tag"] == "sampler";
                uint64_t minimumAlignment{};
                if (image) {
                    uint32_t sampleCount = 0;
                    if (!exactObject(layout, {"tag", "dimension", "format", "sample_count", "aspects"}, {}, diagnostic,
                                     endpointPath + "/layout") ||
                        !layout["dimension"].is_string() || !layout["format"].is_string() ||
                        !uint32Value(layout["sample_count"], sampleCount) || !layout["aspects"].is_array() ||
                        layout["aspects"].empty())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout",
                                    "invalid image endpoint layout");
                    for (const auto &aspect : layout["aspects"])
                        if (!aspect.is_string())
                            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                        endpointPath + "/layout/aspects", "image aspects must be strings");
                    endpoint.imageDimension = layout["dimension"].get<std::string>();
                    endpoint.imageFormat = layout["format"].get<std::string>();
                } else if (sampler) {
                    if (!exactObject(layout, {"tag"}, {}, diagnostic, endpointPath + "/layout"))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout",
                                    "invalid sampler endpoint layout");
                } else if (!exactObject(layout, {"tag", "view_rank", "element_layout_hash", "minimum_alignment"},
                                        {"shape", "descriptor", "autodiff_carrier"}, diagnostic,
                                        endpointPath + "/layout") ||
                           !layout["tag"].is_string() || layout["tag"] != "buffer" ||
                           !uint32Value(layout["view_rank"], endpoint.viewRank) ||
                           !digestValue(layout["element_layout_hash"]) ||
                           !uint64Value(layout["minimum_alignment"], minimumAlignment) || minimumAlignment == 0) {
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout",
                                "invalid buffer endpoint layout");
                }
                endpoint.viewDescriptor =
                    !image && !sampler && (!layout.contains("descriptor") || layout["descriptor"] == true);
                if (!image && !sampler && layout.contains("autodiff_carrier")) {
                    if (!layout["autodiff_carrier"].is_string() || layout["autodiff_carrier"] != "invocation_linear")
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/layout/autodiff_carrier", "invalid autodiff carrier");
                    endpoint.autodiffCarrier = layout["autodiff_carrier"].get<std::string>();
                }
                if (layout.contains("descriptor") && !layout["descriptor"].is_boolean())
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout/descriptor",
                                "invalid descriptor marker");
                const nlohmann::json shape = image ? nlohmann::json::array()
                                             : layout.contains("shape")
                                                 ? layout["shape"]
                                                 : nlohmann::json(std::vector<int64_t>(endpoint.viewRank, -1));
                if (!shape.is_array() || (endpoint.viewDescriptor && shape.size() != endpoint.viewRank))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout/shape",
                                "invalid reflected TensorView shape");
                for (const auto &extentValue : shape) {
                    if (!extentValue.is_number_integer())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout/shape",
                                    "invalid reflected TensorView shape");
                    const int64_t extent = extentValue.get<int64_t>();
                    if (!extent || extent < -1)
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout/shape",
                                    "invalid reflected TensorView extent");
                    endpoint.viewShape.push_back(extent);
                }
                if (!image && !sampler)
                    endpoint.layoutHash = layout["element_layout_hash"].get<std::string>();
                const auto &footprint = endpointValue.contains("write_footprint") ? endpointValue["write_footprint"]
                                                                                  : nlohmann::json(nullptr);
                if (!footprint.is_null()) {
                    if (!exactObject(footprint, {"kind", "indices"}, {}, diagnostic,
                                     endpointPath + "/write_footprint") ||
                        !footprint["kind"].is_string() || !footprint["indices"].is_array())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/write_footprint", "invalid write footprint");
                    endpoint.writeFootprintKind = footprint["kind"].get<std::string>();
                    for (size_t index = 0; index < footprint["indices"].size(); ++index) {
                        uint32_t component{};
                        if (!uint32Value(footprint["indices"][index], component))
                            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                        endpointPath + "/write_footprint/indices", "invalid footprint index");
                        endpoint.writeFootprintIndices.push_back(component);
                    }
                }
                const auto &abi = endpointValue["abi"];
                if (!exactObject(abi, {"bindings"}, {}, diagnostic, endpointPath + "/abi") ||
                    !abi["bindings"].is_array() ||
                    abi["bindings"].size() < 1 + (endpoint.viewDescriptor ? 1 + 2 * endpoint.viewRank : 0)) {
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/abi",
                                "buffer endpoint has an incomplete portable ABI");
                }
                for (size_t bindingIndex = 0; bindingIndex < abi["bindings"].size(); ++bindingIndex) {
                    const auto &abiBinding = abi["bindings"][bindingIndex];
                    const std::string bindingPath = endpointPath + "/abi/bindings/" + std::to_string(bindingIndex);
                    if (!exactObject(abiBinding, {"semantic", "carrier"}, {}, diagnostic, bindingPath) ||
                        !abiBinding["carrier"].is_object())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", bindingPath,
                                    "invalid portable ABI binding");
                    EndpointAbiBinding binding;
                    if (abiBinding["semantic"].is_string()) {
                        binding.semantic = abiBinding["semantic"].get<std::string>();
                        if (binding.semantic != "resource" && binding.semantic != "sampler" &&
                            binding.semantic != "byte_offset")
                            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", bindingPath + "/semantic",
                                        "unknown portable ABI semantic");
                    } else if (abiBinding["semantic"].is_object()) {
                        const auto &semantic = abiBinding["semantic"];
                        const bool storageLeaf = semantic.contains("storage_leaf");
                        const bool extent = semantic.contains("extent");
                        const std::string_view semanticName = storageLeaf ? "storage_leaf"
                                                              : extent    ? "extent"
                                                                          : "byte_stride";
                        if (!exactObject(semantic, {semanticName}, {}, diagnostic, bindingPath + "/semantic")) {
                            return false;
                        }
                        uint32_t axis{};
                        const auto &axisValue = semantic[semanticName];
                        if (!uint32Value(axisValue, axis) || (!storageLeaf && axis >= endpoint.viewRank))
                            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", bindingPath + "/semantic",
                                        "portable ABI axis is out of range");
                        binding.semantic = std::string(semanticName);
                        binding.axis = axis;
                    } else {
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", bindingPath + "/semantic",
                                    "invalid portable ABI semantic");
                    }
                    const auto &carrier = abiBinding["carrier"];
                    const bool resourceCarrier =
                        binding.semantic == "resource" || binding.semantic == "sampler" ||
                        binding.semantic == "storage_leaf" ||
                        (binding.semantic == "value" && carrier.value("tag", "") == "resource_slot");
                    if (!exactObject(carrier,
                                     resourceCarrier
                                         ? std::initializer_list<std::string_view>{"tag", "slot"}
                                         : std::initializer_list<std::string_view>{"tag", "slot", "byte_offset",
                                                                                   "byte_size", "alignment"},
                                     {}, diagnostic, bindingPath + "/carrier") ||
                        !carrier["tag"].is_string() ||
                        carrier["tag"] != (resourceCarrier ? "resource_slot" : "value_slot") ||
                        !uint32Value(carrier["slot"], binding.slot))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", bindingPath + "/carrier",
                                    "portable ABI carrier does not match its semantic");
                    binding.carrier = carrier["tag"].get<std::string>();
                    if (!resourceCarrier && (!uint64Value(carrier["byte_offset"], binding.byteOffset) ||
                                             !uint64Value(carrier["byte_size"], binding.byteSize) ||
                                             !uint64Value(carrier["alignment"], binding.alignment) ||
                                             !binding.byteSize || !binding.alignment))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", bindingPath + "/carrier",
                                    "invalid value-slot carrier");
                    endpoint.abiBindings.push_back(std::move(binding));
                }
                const auto semanticCount = [&](const std::string &name) {
                    return std::count_if(endpoint.abiBindings.begin(), endpoint.abiBindings.end(),
                                         [&](const EndpointAbiBinding &binding) { return binding.semantic == name; });
                };
                if (semanticCount("resource") + semanticCount("sampler") + semanticCount("storage_leaf") == 0 ||
                    semanticCount("byte_offset") != (endpoint.viewDescriptor ? 1 : 0) ||
                    semanticCount("extent") != (endpoint.viewDescriptor ? endpoint.viewRank : 0) ||
                    semanticCount("byte_stride") != (endpoint.viewDescriptor ? endpoint.viewRank : 0))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/abi",
                                "buffer endpoint portable ABI semantics are incomplete");
            } else {
                if (!digestValue(endpointValue["layout_hash"]))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/layout_hash",
                                "invalid value endpoint layout hash");
                endpoint.layoutHash = endpointValue["layout_hash"].get<std::string>();
                if (endpointValue.contains("element_layout_hash")) {
                    if (!digestValue(endpointValue["element_layout_hash"]))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/element_layout_hash", "invalid element layout hash");
                    endpoint.elementLayoutHash = endpointValue["element_layout_hash"].get<std::string>();
                }
                const auto &abi = endpointValue["abi"];
                if (!exactObject(abi, {"bindings"}, {}, diagnostic, endpointPath + "/abi") ||
                    !abi["bindings"].is_array() || abi["bindings"].size() != 1)
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/abi",
                                "value endpoint has an incomplete portable ABI");
                const auto &abiBinding = abi["bindings"][0];
                const auto &carrier = abiBinding["carrier"];
                EndpointAbiBinding binding;
                if (!exactObject(abiBinding, {"semantic", "carrier"}, {}, diagnostic,
                                 endpointPath + "/abi/bindings/0") ||
                    !abiBinding["semantic"].is_string() || abiBinding["semantic"] != "value" || !carrier.is_object() ||
                    !exactObject(carrier, {"tag", "slot", "byte_offset", "byte_size", "alignment"}, {}, diagnostic,
                                 endpointPath + "/abi/bindings/0/carrier") ||
                    !carrier["tag"].is_string() ||
                    (carrier["tag"] != "value_slot" && carrier["tag"] != "constant_region") ||
                    !uint32Value(carrier["slot"], binding.slot) ||
                    !uint64Value(carrier["byte_offset"], binding.byteOffset) ||
                    !uint64Value(carrier["byte_size"], binding.byteSize) ||
                    !uint64Value(carrier["alignment"], binding.alignment) || !binding.byteSize || !binding.alignment)
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/abi/bindings/0",
                                "invalid value-slot carrier");
                binding.semantic = "value";
                binding.carrier = carrier["tag"].get<std::string>();
                endpoint.abiBindings.push_back(std::move(binding));
            }
            stage.endpoints.push_back(std::move(endpoint));
        }
        std::vector<uint32_t> allSlots;
        for (const ReflectedEndpoint &endpoint : stage.endpoints) {
            for (const EndpointAbiBinding &binding : endpoint.abiBindings) {
                allSlots.push_back(binding.slot);
            }
        }
        std::sort(allSlots.begin(), allSlots.end());
        for (size_t index = 0; index < allSlots.size(); ++index)
            if (allSlots[index] != index)
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", path + "/reflection/endpoints",
                            "portable ABI slots must be contiguous and unique");
        if (row.contains("implementation")) {
            const auto &implementation = row["implementation"];
            const std::string implementationPath = path + "/implementation";
            if (!exactObject(implementation, {"target", "metadata", "endpoints"}, {}, diagnostic, implementationPath) ||
                !implementation["target"].is_string() || implementation["target"] != stage.backend ||
                !implementation["metadata"].is_object() || !implementation["endpoints"].is_array())
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", implementationPath,
                            "invalid target implementation table");
            const auto &metadata = implementation["metadata"];
            std::string targetError;
            if (!parseTargetImplementationMetadata(stage.backend, metadata, stage.nativeSlots, targetError))
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", implementationPath + "/metadata",
                            std::move(targetError));
            for (size_t index = 0; index < implementation["endpoints"].size(); ++index) {
                const auto &rowValue = implementation["endpoints"][index];
                const std::string endpointPath = implementationPath + "/endpoints/" + std::to_string(index);
                CompiledEndpointAbi compiled;
                if (!exactObject(rowValue, {"module", "interface", "index"},
                                 {"builtin", "value_transport", "set", "binding", "interface_plan",
                                  "packed_frame_offset", "element_layout", "sampled_image_bindings"},
                                 diagnostic, endpointPath) ||
                    !rowValue["module"].is_string() || !rowValue["interface"].is_string() ||
                    !uint32Value(rowValue["index"], compiled.index))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath,
                                "invalid compiled endpoint ABI");
                compiled.module = rowValue["module"].get<std::string>();
                compiled.interfaceKind = rowValue["interface"].get<std::string>();
                if (rowValue.contains("builtin")) {
                    if (!rowValue["builtin"].is_string())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/builtin",
                                    "invalid compiled builtin");
                    compiled.builtin = rowValue["builtin"].get<std::string>();
                }
                if (rowValue.contains("value_transport")) {
                    if (!rowValue["value_transport"].is_string())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/value_transport", "invalid compiled value transport");
                    compiled.valueTransport = rowValue["value_transport"].get<std::string>();
                }
                if (rowValue.contains("set") && !uint32Value(rowValue["set"], compiled.descriptorSet))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/set",
                                "invalid compiled descriptor set");
                if (rowValue.contains("binding") && !uint32Value(rowValue["binding"], compiled.binding))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", endpointPath + "/binding",
                                "invalid compiled descriptor binding");
                if (rowValue.contains("interface_plan")) {
                    vernon::runtime::InterfacePlan plan;
                    std::string error;
                    if (!vernon::runtime::parsePipelineInterfacePlan(rowValue["interface_plan"], plan, error))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/interface_plan", error);
                    compiled.interfacePlan = std::move(plan);
                }
                if (rowValue.contains("packed_frame_offset")) {
                    uint64_t offset = 0;
                    if (!uint64Value(rowValue["packed_frame_offset"], offset))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/packed_frame_offset", "invalid packed frame offset");
                    compiled.packedFrameOffset = offset;
                }
                if (rowValue.contains("element_layout")) {
                    vernon::runtime::ValueLayout layout;
                    std::string error;
                    if (!vernon::runtime::parsePipelineValueLayout(rowValue["element_layout"], layout, error))
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/element_layout", error);
                    compiled.elementLayout = std::move(layout);
                }
                if (rowValue.contains("sampled_image_bindings")) {
                    const auto &bindings = rowValue["sampled_image_bindings"];
                    if (!bindings.is_array())
                        return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                    endpointPath + "/sampled_image_bindings",
                                    "sampled image bindings must be an array");
                    for (const auto &binding : bindings) {
                        uint32_t descriptorSet{};
                        uint32_t descriptorBinding{};
                        if (!binding.is_object() || !binding.contains("set") || !binding.contains("binding") ||
                            !uint32Value(binding["set"], descriptorSet) ||
                            !uint32Value(binding["binding"], descriptorBinding))
                            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                                        endpointPath + "/sampled_image_bindings",
                                        "sampled image binding must contain unsigned set/binding");
                        compiled.sampledImageBindings.push_back({descriptorSet, descriptorBinding});
                    }
                }
                stage.compiledAbi.push_back(std::move(compiled));
            }
        }
        if (operation == "graphics") {
            const auto &graphics = reflection["graphics"];
            if (!exactObject(graphics,
                             {"topology", "vertex_inputs", "fragment_outputs", "linkage", "attachment_constraints",
                              "index_formats", "capabilities"},
                             {}, diagnostic, path + "/reflection/graphics") ||
                !graphics["topology"].is_string() || !graphics["vertex_inputs"].is_array() ||
                !graphics["fragment_outputs"].is_array() || !graphics["linkage"].is_object() ||
                !graphics["attachment_constraints"].is_array() ||
                !stringArray(graphics["index_formats"], stage.capabilities, diagnostic,
                             path + "/reflection/graphics/index_formats", false)) {
                return false;
            }
            stage.graphicsTopology = graphics["topology"].get<std::string>();
            for (size_t inputIndex = 0; inputIndex < graphics["vertex_inputs"].size(); ++inputIndex) {
                const auto &input = graphics["vertex_inputs"][inputIndex];
                const std::string inputPath = path + "/reflection/graphics/vertex_inputs/" + std::to_string(inputIndex);
                GraphicsVertexInput parsed;
                if (!exactObject(
                        input,
                        {"location", "endpoint_index", "format", "byte_offset", "byte_stride", "step", "divisor"}, {},
                        diagnostic, inputPath) ||
                    !uint32Value(input["location"], parsed.location) ||
                    !uint32Value(input["endpoint_index"], parsed.endpointIndex) || !input["format"].is_string() ||
                    !uint64Value(input["byte_offset"], parsed.byteOffset) ||
                    !uint64Value(input["byte_stride"], parsed.byteStride) || !parsed.byteStride ||
                    !input["step"].is_string() || !uint32Value(input["divisor"], parsed.divisor))
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", inputPath,
                                "invalid graphics vertex input");
                parsed.format = input["format"].get<std::string>();
                parsed.step = input["step"].get<std::string>();
                stage.vertexInputs.push_back(std::move(parsed));
            }
            for (size_t outputIndex = 0; outputIndex < graphics["fragment_outputs"].size(); ++outputIndex) {
                const auto &output = graphics["fragment_outputs"][outputIndex];
                const std::string outputPath =
                    path + "/reflection/graphics/fragment_outputs/" + std::to_string(outputIndex);
                GraphicsFragmentOutput parsed;
                if (!exactObject(output, {"location", "type"}, {}, diagnostic, outputPath) ||
                    !uint32Value(output["location"], parsed.location) || !output["type"].is_string())
                    return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", outputPath,
                                "invalid graphics fragment output");
                parsed.type = output["type"].get<std::string>();
                stage.fragmentOutputs.push_back(std::move(parsed));
            }
            stage.capabilities.clear();
            if (!stringArray(graphics["capabilities"], stage.capabilities, diagnostic,
                             path + "/reflection/graphics/capabilities", false))
                return false;
            nlohmann::json contract{{"operation", operation}, {"reflection", reflection}};
            const std::string contractBytes = contract.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
            if (sha256Hex(contractBytes.data(), contractBytes.size()) != stage.contractHash)
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", path + "/contract_hash",
                            "StageArtifact contract hash does not match reflection");
            artifacts.stages.emplace(artifactId, std::move(stage));
            continue;
        }
        const auto &compute = reflection["compute"];
        if (!exactObject(compute, {"workgroup_size", "subgroup", "capabilities"}, {"dispatch_contract"}, diagnostic,
                         path + "/reflection/compute") ||
            !compute["workgroup_size"].is_array() || compute["workgroup_size"].size() != 3 ||
            !compute["subgroup"].is_null() ||
            !stringArray(compute["capabilities"], stage.capabilities, diagnostic,
                         path + "/reflection/compute/capabilities", false) ||
            stage.capabilities != std::vector<std::string>{"direct_dispatch"})
            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", path + "/reflection/compute",
                        "invalid direct compute reflection");
        const nlohmann::json dispatch =
            compute.contains("dispatch_contract")
                ? compute["dispatch_contract"]
                : nlohmann::json{{"requires_unit_workgroup", true}, {"unit_grid_axes", {0, 1, 2}}};
        if (!exactObject(dispatch, {"requires_unit_workgroup", "unit_grid_axes"}, {}, diagnostic,
                         path + "/reflection/compute/dispatch_contract") ||
            !dispatch["requires_unit_workgroup"].is_boolean() || !dispatch["unit_grid_axes"].is_array())
            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                        path + "/reflection/compute/dispatch_contract", "invalid dispatch contract");
        stage.requiresUnitWorkgroup = dispatch["requires_unit_workgroup"].get<bool>();
        for (const auto &axisValue : dispatch["unit_grid_axes"]) {
            uint32_t axis{};
            if (!uint32Value(axisValue, axis) || axis >= 3)
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                            path + "/reflection/compute/dispatch_contract/unit_grid_axes",
                            "dispatch contract axis is out of range");
            stage.unitGridAxes.push_back(axis);
        }
        for (size_t axis = 0; axis < 3; ++axis)
            if (!uint32Value(compute["workgroup_size"][axis], stage.workgroupSize[axis]) || !stage.workgroupSize[axis])
                return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse",
                            path + "/reflection/compute/workgroup_size/" + std::to_string(axis),
                            "workgroup size must be positive uint32");

        nlohmann::json contract{{"operation", operation}, {"reflection", reflection}};
        const std::string contractBytes = contract.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
        if (sha256Hex(contractBytes.data(), contractBytes.size()) != stage.contractHash)
            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "parse", path + "/contract_hash",
                        "StageArtifact contract hash does not match reflection");
        artifacts.stages.emplace(artifactId, std::move(stage));
    }
    if (stageRequirements.empty())
        return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", "/artifact_system/runtime_requirements",
                    "ArtifactSystem requires at least one StageArtifact");
    nlohmann::json aggregate = stageRequirements.front();
    std::set<std::string> aggregateFeatures;
    const std::set<std::string> invariantFields{
        "target_triple", "object_format", "address_size", "profile", "apple_platform",
    };
    const auto pairLess = [](const nlohmann::json &left, const nlohmann::json &right) {
        return std::pair<uint32_t, uint32_t>{left[0].get<uint32_t>(), left[1].get<uint32_t>()} <
               std::pair<uint32_t, uint32_t>{right[0].get<uint32_t>(), right[1].get<uint32_t>()};
    };
    for (const nlohmann::json &requirements : stageRequirements) {
        for (const nlohmann::json &feature : requirements["features"])
            aggregateFeatures.insert(feature.get<std::string>());
        for (const auto &[name, member] : requirements.items()) {
            if (name == "backend" || name == "features")
                continue;
            if (invariantFields.count(name)) {
                if (aggregate[name] != member)
                    return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse",
                                "/artifact_system/runtime_requirements/" + name,
                                "StageArtifacts disagree on an invariant runtime requirement");
            } else if ((member.is_array() && pairLess(aggregate[name], member)) ||
                       (member.is_number_integer() && aggregate[name].get<int64_t>() < member.get<int64_t>())) {
                aggregate[name] = member;
            }
        }
    }
    aggregate["features"] = nlohmann::json::array();
    for (const std::string &feature : aggregateFeatures)
        aggregate["features"].push_back(feature);
    if (value["runtime_requirements"] != aggregate)
        return fail(diagnostic, "PROGRAM_RUNTIME_REQUIREMENTS", "parse", "/artifact_system/runtime_requirements",
                    "ArtifactSystem aggregate does not match its StageArtifacts");
    return true;
}

bool resolve(Program program, const ArtifactSystem &artifacts, ResolvedProgram &resolved, Diagnostic &diagnostic) {
    resolved = {};
    diagnostic = {};
    if (program.graphs.empty() || program.graphs.front().direction != "forward" ||
        (program.graphs.size() == 2 && program.graphs.back().direction != "backward") || program.graphs.size() > 2)
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "resolve", "/graphs",
                    "phase-one compute requires one forward graph and at most one backward graph");
    if (artifacts.stages.size() != program.stages.size())
        return fail(diagnostic, "PROGRAM_STAGE_MISSING", "resolve", "/artifact_system/artifacts",
                    "StageArtifacts must exactly cover every Program stage");
    for (const auto &[stageId, contract] : program.stages) {
        const auto artifact = artifacts.stages.find(stageId);
        if (artifact == artifacts.stages.end())
            return fail(diagnostic, "PROGRAM_STAGE_MISSING", "resolve", "/artifact_system/artifacts/" + stageId,
                        "Program stage has no directly keyed StageArtifact");
        if (artifact->second.operation != contract.operation || artifact->second.contractHash != contract.contractHash)
            return fail(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "resolve", "/artifact_system/artifacts/" + stageId,
                        "StageArtifact does not implement the portable StageContract");
        resolved.stages.emplace(stageId, ResolvedStage{stageId, artifact->second});
    }
    for (size_t index = 0; index < program.values.size(); ++index) {
        const Value &value = program.values[index];
        if (value.id != index)
            return fail(diagnostic, "PROGRAM_ID_SEQUENCE", "resolve", "/values/" + std::to_string(index) + "/id",
                        "Value ids must equal array indices");
        if (value.storage && *value.storage >= program.storages.size())
            return fail(diagnostic, "PROGRAM_STORAGE_DESCRIPTOR", "resolve",
                        "/values/" + std::to_string(index) + "/storage", "Value references an unknown Storage");
    }
    for (size_t index = 0; index < program.storages.size(); ++index) {
        const Storage &storage = program.storages[index];
        if (storage.id != index || storage.initialValue >= program.values.size() ||
            !program.values[storage.initialValue].storage ||
            *program.values[storage.initialValue].storage != storage.id)
            return fail(diagnostic, "PROGRAM_STORAGE_INITIAL_VALUE", "resolve",
                        "/storages/" + std::to_string(index) + "/initial_value",
                        "Storage initial root is missing or belongs to another Storage");
    }
    for (size_t index = 0; index < program.parameters.size(); ++index) {
        const Parameter &parameter = program.parameters[index];
        if (parameter.id != index || parameter.value >= program.values.size() ||
            program.values[parameter.value].origin.kind != OriginKind::Parameter ||
            program.values[parameter.value].origin.parameter != parameter.id)
            return fail(diagnostic, "PROGRAM_PARAMETER_BINDING", "resolve", "/parameters/" + std::to_string(index),
                        "Parameter and ParameterOrigin disagree");
    }

    std::set<std::string> usedStages;
    for (size_t graphIndex = 0; graphIndex < program.graphs.size(); ++graphIndex) {
        const Graph &graph = program.graphs[graphIndex];
        const std::string graphPath = "/graphs/" + std::to_string(graphIndex);
        std::set<uint32_t> entryValues;
        uint32_t nextUserSlot = 0;
        for (size_t index = 0; index < graph.inputs.size(); ++index) {
            const GraphInput &input = graph.inputs[index];
            if (input.value >= program.values.size() || !entryValues.insert(input.value).second)
                return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "resolve",
                            graphPath + "/inputs/" + std::to_string(index), "invalid or duplicate graph input");
            const Value &value = program.values[input.value];
            const bool valid =
                (input.kind == GraphInputKind::UserInput && input.slot == nextUserSlot++ &&
                 value.origin.kind == OriginKind::Argument && value.origin.graph == graph.direction &&
                 value.origin.slot == input.slot) ||
                (input.kind == GraphInputKind::InvocationControl && value.origin.kind == OriginKind::Argument &&
                 value.origin.graph == graph.direction) ||
                (input.kind == GraphInputKind::Control && value.origin.kind == OriginKind::Argument &&
                 value.origin.graph == graph.direction) ||
                (input.kind == GraphInputKind::Parameter && input.parameter < program.parameters.size() &&
                 program.parameters[input.parameter].value == input.value &&
                 value.origin.kind == OriginKind::Parameter) ||
                (input.kind == GraphInputKind::Allocation && input.storage < program.storages.size() &&
                 program.storages[input.storage].initialValue == input.value &&
                 value.origin.kind == OriginKind::Allocation && value.origin.graph == graph.direction);
            if (!valid)
                return fail(diagnostic, "PROGRAM_VALUE_ORIGIN", "resolve",
                            graphPath + "/inputs/" + std::to_string(index), "GraphInput and Value origin disagree");
        }
        for (size_t captureIndex = 0; captureIndex < graph.captures.size(); ++captureIndex) {
            const uint32_t capture = graph.captures[captureIndex];
            if (capture >= program.values.size() || !entryValues.insert(capture).second)
                return fail(diagnostic, "PROGRAM_RESIDUAL_CONTRACT", "resolve",
                            graphPath + "/captures/" + std::to_string(captureIndex),
                            "invalid or duplicate graph capture");
        }

        std::vector<std::optional<uint32_t>> producers(program.values.size());
        std::vector<std::set<uint32_t>> readers(program.values.size());
        std::vector<std::optional<uint32_t>> successors(program.values.size());
        ResolvedGraph resolvedGraph;
        resolvedGraph.predecessors.resize(graph.nodes.size());
        for (size_t nodeIndex = 0; nodeIndex < graph.nodes.size(); ++nodeIndex) {
            const Node &node = graph.nodes[nodeIndex];
            const std::string nodePath = graphPath + "/nodes/" + std::to_string(nodeIndex);
            const auto stage = program.stages.find(node.stage);
            const std::string_view operationName =
                executionKind(node) == ExecutionKind::Compute ? "compute" : "graphics";
            if (node.id != nodeIndex || stage == program.stages.end() || stage->second.operation != operationName)
                return fail(diagnostic, "PROGRAM_STAGE_MISSING", "resolve", nodePath + "/stage",
                            "Node references an unknown or incompatible stage");
            usedStages.insert(node.stage);
            if (executionKind(node) == ExecutionKind::Graphics) {
                const GraphicsOperation &graphics = graphicsOperation(node);
                const auto project = [&](const GraphicsAttachmentSignature &attachment) {
                    if (attachment.access >= node.accesses.size())
                        return;
                    resolvedGraph.controlResources.push_back({node.id, graphics.renderPassControl,
                                                              node.accesses[attachment.access].storage,
                                                              attachment.location, attachment.aspects});
                };
                for (const GraphicsAttachmentSignature &attachment : graphics.colorAttachments)
                    project(attachment);
                if (graphics.depthStencilAttachment)
                    project(*graphics.depthStencilAttachment);
            }
            const StageArtifact &stageArtifact = resolved.stages.at(node.stage).stage;
            std::vector<const ReflectedEndpoint *> bindableEndpoints;
            for (const ReflectedEndpoint &endpoint : stageArtifact.endpoints)
                if (endpoint.tag != "system" && endpoint.interfaceKind != "system_value")
                    bindableEndpoints.push_back(&endpoint);
            if (node.bindings.size() != bindableEndpoints.size())
                return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "resolve", nodePath + "/bindings",
                            "Program bindings do not cover reflected endpoints exactly");
            std::set<uint32_t> expectedOperands;
            std::set<uint32_t> expectedResults;
            for (size_t accessIndex = 0; accessIndex < node.accesses.size(); ++accessIndex) {
                const ResourceAccess &access = node.accesses[accessIndex];
                if (access.storage >= program.storages.size())
                    return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve",
                                nodePath + "/accesses/" + std::to_string(accessIndex) + "/storage",
                                "ResourceAccess names an unknown Storage");
                const auto checkRoot = [&](uint32_t value) {
                    return value < program.values.size() && program.values[value].storage &&
                           *program.values[value].storage == access.storage;
                };
                if (access.kind == AccessKind::Read) {
                    if (!checkRoot(access.value))
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/accesses",
                                    "read root does not belong to its Storage");
                    expectedOperands.insert(access.value);
                    readers[access.value].insert(node.id);
                } else if (access.kind == AccessKind::Initialize) {
                    if (!checkRoot(access.after) ||
                        program.storages[access.storage].mutability == StorageMutability::ReadOnly)
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/accesses",
                                    "initialize result is incompatible with its Storage");
                    expectedResults.insert(access.after);
                    const uint32_t initial = program.storages[access.storage].initialValue;
                    const OriginKind initialOrigin = program.values[initial].origin.kind;
                    if ((initialOrigin == OriginKind::Allocation && access.after == initial) ||
                        (initialOrigin == OriginKind::NodeResult && access.after != initial) ||
                        (initialOrigin != OriginKind::Allocation && initialOrigin != OriginKind::NodeResult) ||
                        (initialOrigin == OriginKind::Allocation && successors[initial])) {
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/accesses",
                                    "initialize does not produce the unique first readable Storage version");
                    }
                    if (initialOrigin == OriginKind::Allocation) {
                        successors[initial] = node.id;
                        expectedOperands.insert(initial);
                    }
                } else {
                    if (!checkRoot(access.before) || !checkRoot(access.after) ||
                        program.storages[access.storage].mutability == StorageMutability::ReadOnly)
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/accesses",
                                    "write versions are incompatible with their Storage");
                    if (access.before == access.after || successors[access.before])
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/accesses",
                                    "Storage version has multiple or self successors");
                    successors[access.before] = node.id;
                    if (access.access == "read_write")
                        readers[access.before].insert(node.id);
                    else if (access.access != "write")
                        return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/accesses",
                                    "write access mode must be write or read_write");
                    expectedOperands.insert(access.before);
                    expectedResults.insert(access.after);
                }
                if (access.view)
                    expectedOperands.insert(*access.view);
            }
            for (size_t bindingIndex = 0; bindingIndex < node.bindings.size(); ++bindingIndex) {
                const EndpointBinding &binding = node.bindings[bindingIndex];
                const ReflectedEndpoint &endpoint = *bindableEndpoints[bindingIndex];
                if (binding.module != endpoint.module || binding.interfaceKind != endpoint.interfaceKind ||
                    binding.index != endpoint.index)
                    return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "resolve",
                                nodePath + "/bindings/" + std::to_string(bindingIndex),
                                "EndpointBinding identity does not match artifact reflection");
                if (binding.tag == BindingTag::Value) {
                    for (const ValueEndpointProjection &projection : binding.projections) {
                        if (projection.value >= program.values.size() ||
                            (projection.leaf &&
                             (!program.values[projection.value].layout ||
                              *projection.leaf >= program.values[projection.value].layout->leaves.size())))
                            return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "resolve",
                                        nodePath + "/bindings/" + std::to_string(bindingIndex),
                                        "value projection references an unknown canonical Value leaf");
                        if (projection.direction == ValueBindingDirection::Input)
                            expectedOperands.insert(projection.value);
                        else
                            expectedResults.insert(projection.value);
                    }
                } else {
                    if (binding.access >= node.accesses.size())
                        return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "resolve", nodePath + "/bindings",
                                    "resource binding names an unknown access");
                    const ResourceAccess &access = node.accesses[binding.access];
                    const uint32_t physicalValue = access.kind == AccessKind::Read         ? access.value
                                                   : access.kind == AccessKind::Initialize ? access.after
                                                                                           : access.before;
                    const std::string requiredAccess = access.kind == AccessKind::Read         ? "read"
                                                       : access.kind == AccessKind::Initialize ? "write"
                                                                                               : access.access;
                    if (physicalValue >= program.values.size() || program.values[physicalValue].type != endpoint.type ||
                        !isResourceEndpointRole(endpoint.role) ||
                        (endpoint.transport != "resource_handle" && endpoint.transport != "device_address") ||
                        (!isTapeCarrierRole(endpoint.role) && endpoint.access != requiredAccess))
                        return fail(diagnostic, "PROGRAM_BINDING_MISMATCH", "resolve",
                                    nodePath + "/bindings/" + std::to_string(bindingIndex),
                                    "resource binding ABI or access does not match reflection");
                }
            }
            if (executionKind(node) == ExecutionKind::Compute) {
                for (const ControlComponent &control : computeOperation(node).workgroups) {
                    if (control.kind == ControlKind::Parameter) {
                        if (control.reference >= program.parameters.size())
                            return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "resolve", nodePath + "/operation",
                                        "dispatch parameter is unavailable");
                        expectedOperands.insert(program.parameters[control.reference].value);
                    } else if (control.kind == ControlKind::Value) {
                        if (control.reference >= program.values.size())
                            return fail(diagnostic, "PROGRAM_CONTROL_UNAVAILABLE", "resolve", nodePath + "/operation",
                                        "dispatch Value is unavailable");
                        expectedOperands.insert(control.reference);
                    }
                }
            } else {
                const GraphicsOperation &graphics = graphicsOperation(node);
                const auto validateAttachment = [&](const GraphicsAttachmentSignature &attachment) {
                    return attachment.access < node.accesses.size() &&
                           node.accesses[attachment.access].kind == AccessKind::Attachment;
                };
                if (!std::all_of(graphics.colorAttachments.begin(), graphics.colorAttachments.end(),
                                 validateAttachment) ||
                    (graphics.depthStencilAttachment && !validateAttachment(*graphics.depthStencilAttachment)))
                    return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/operation/attachments",
                                "graphics attachment does not select an attachment access");
                for (const auto &[location, blend] : graphics.pipelineState.colorBlends) {
                    (void)blend;
                    if (location >= graphics.colorAttachments.size())
                        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "resolve",
                                    nodePath + "/operation/pipeline_state/color_blends",
                                    "color blend names an absent attachment");
                }
            }
            if (std::vector<uint32_t>(expectedOperands.begin(), expectedOperands.end()) != node.operands ||
                std::vector<uint32_t>(expectedResults.begin(), expectedResults.end()) != node.results)
                return fail(diagnostic, "PROGRAM_OPERAND_CLOSURE", "resolve", nodePath,
                            "Node operands or results do not equal their exact binding/access/control closure");
            std::set<uint32_t> predecessors;
            for (uint32_t operand : node.operands) {
                if (operand >= program.values.size())
                    return fail(diagnostic, "PROGRAM_OPERAND_CLOSURE", "resolve", nodePath + "/operands",
                                "Node operand is unknown");
                if (producers[operand])
                    predecessors.insert(*producers[operand]);
                else if (!entryValues.count(operand))
                    return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/operands",
                                "Node operand is unavailable");
            }
            for (uint32_t result : node.results) {
                if (result >= program.values.size() || producers[result] ||
                    program.values[result].origin.kind != OriginKind::NodeResult ||
                    program.values[result].origin.graph != graph.direction ||
                    program.values[result].origin.node != node.id)
                    return fail(diagnostic, "PROGRAM_ACCESS_CHAIN", "resolve", nodePath + "/results",
                                "Node result producer metadata disagrees");
                producers[result] = node.id;
            }
            resolvedGraph.predecessors[node.id] = {predecessors.begin(), predecessors.end()};
        }
        for (size_t value = 0; value < successors.size(); ++value) {
            if (!successors[value])
                continue;
            const uint32_t successor = *successors[value];
            for (uint32_t reader : readers[value]) {
                if (reader == successor)
                    continue;
                if (reader > successor)
                    return fail(diagnostic, "PROGRAM_DEPENDENCY_ORDER", "resolve",
                                graphPath + "/nodes/" + std::to_string(successor),
                                "Storage successor precedes a reader of its predecessor version");
                std::vector<uint32_t> &predecessors = resolvedGraph.predecessors[successor];
                if (std::find(predecessors.begin(), predecessors.end(), reader) == predecessors.end()) {
                    predecessors.push_back(reader);
                    std::sort(predecessors.begin(), predecessors.end());
                }
            }
        }
        for (size_t index = 0; index < graph.outputs.size(); ++index)
            if (graph.outputs[index].value >= producers.size() || !producers[graph.outputs[index].value])
                return fail(diagnostic, "PROGRAM_RESOURCE_OUTPUT", "resolve",
                            graphPath + "/outputs/" + std::to_string(index), "output has no producer");
        resolved.graphs.push_back(std::move(resolvedGraph));
    }
    if (usedStages.size() != program.stages.size())
        return fail(diagnostic, "PROGRAM_STAGE_MISSING", "resolve", "/stages",
                    "every StageContract must be referenced");

    const Graph *forward = findGraph(program, "forward");
    const Graph *backward = findGraph(program, "backward");
    if (!forward)
        return fail(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "resolve", "/graphs",
                    "phase-one compute requires one forward graph");
    const auto userInputCount = [](const Graph &graph) {
        uint32_t count = 0;
        for (const GraphInput &input : graph.inputs)
            if (input.kind == GraphInputKind::UserInput)
                ++count;
        return count;
    };
    const auto boundaries = [&](BoundaryRole role) {
        std::vector<const BoundarySlot *> result;
        for (const BoundarySlot &slot : program.abi.boundarySlots)
            if (slot.role == role)
                result.push_back(&slot);
        return result;
    };
    const auto matchUserInputs = [&](const Graph &graph, BoundaryRole role, const std::string &field) {
        const std::vector<const BoundarySlot *> slots = boundaries(role);
        if (slots.size() != userInputCount(graph))
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "resolve", "/abi/" + field,
                        "ProgramABI does not match graph user inputs");
        for (size_t index = 0; index < slots.size(); ++index) {
            const auto input = std::find_if(graph.inputs.begin(), graph.inputs.end(), [&](const GraphInput &candidate) {
                return candidate.kind == GraphInputKind::UserInput && candidate.slot == index;
            });
            if (input == graph.inputs.end() || slots[index]->value != input->value)
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "resolve",
                            "/abi/" + field + "/" + std::to_string(index), "input boundary mismatch");
        }
        return true;
    };
    const auto matchOutputs = [&](const Graph &graph, BoundaryRole role, const std::string &field) {
        const std::vector<const BoundarySlot *> slots = boundaries(role);
        if (slots.size() != graph.outputs.size())
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "resolve", "/abi/" + field,
                        "ProgramABI does not match graph outputs");
        for (size_t index = 0; index < graph.outputs.size(); ++index) {
            const PublicationTarget *publication = findPublicationTarget(program.abi, slots[index]->id);
            const bool transactionMatches = slots[index]->publication == BoundaryPublication::InPlace
                                                ? publication == nullptr
                                                : publication && publication->value == slots[index]->value &&
                                                      publication->role == role &&
                                                      publication->aliasOwner.kind == slots[index]->aliasOwner.kind &&
                                                      publication->aliasOwner.id == slots[index]->aliasOwner.id;
            if (slots[index]->value != graph.outputs[index].value || !transactionMatches)
                return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "resolve",
                            "/abi/" + field + "/" + std::to_string(index), "output boundary mismatch");
        }
        return true;
    };
    if (!matchUserInputs(*forward, BoundaryRole::Input, "inputs") ||
        !matchOutputs(*forward, BoundaryRole::Output, "outputs"))
        return false;
    if (!backward) {
        if (!boundaries(BoundaryRole::Cotangent).empty() || !boundaries(BoundaryRole::Gradient).empty())
            return fail(diagnostic, "PROGRAM_ABI_MISMATCH", "resolve", "/abi",
                        "ProgramABI does not match forward graph boundaries");
    } else if (!matchUserInputs(*backward, BoundaryRole::Cotangent, "cotangents") ||
               !matchOutputs(*backward, BoundaryRole::Gradient, "gradients")) {
        return false;
    }

    resolved.program = std::move(program);
    return true;
}

bool resolveControlValue(const Program &program, const ControlComponent &control, std::string_view graph,
                         uint32_t &valueId) {
    if (control.kind == ControlKind::Capture) {
        valueId = control.reference;
        return valueId < program.values.size();
    }
    if (control.kind == ControlKind::Parameter) {
        if (control.reference >= program.parameters.size())
            return false;
        valueId = program.parameters[control.reference].value;
        return valueId < program.values.size();
    }
    if (control.kind != ControlKind::Value || control.reference >= program.values.size())
        return false;
    valueId = control.reference;
    return true;
}

} // namespace vernon::runtime::program
