#include "pipeline_manifest.h"
#include "content_hash.h"
#include "pipeline_metadata.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <string_view>
#include <tuple>

namespace vernon::runtime {

std::optional<VernonTextureDimension> pipelineTextureDimension(const std::string &dimension) {
    if (dimension == "2d")
        return VERNON_TEXTURE_2D;
    if (dimension == "3d")
        return VERNON_TEXTURE_3D;
    if (dimension == "cube")
        return VERNON_TEXTURE_CUBE;
    return std::nullopt;
}

std::optional<VernonTextureFormat> pipelineTextureFormat(const std::string &format) {
    if (format == "rgba8_unorm")
        return VERNON_TEXTURE_RGBA8_UNORM;
    if (format == "rgba8_srgb")
        return VERNON_TEXTURE_RGBA8_SRGB;
    if (format == "rgba16_float")
        return VERNON_TEXTURE_RGBA16_FLOAT;
    if (format == "rgba32_float")
        return VERNON_TEXTURE_RGBA32_FLOAT;
    if (format == "r8_unorm")
        return VERNON_TEXTURE_R8_UNORM;
    if (format == "r16_float")
        return VERNON_TEXTURE_R16_FLOAT;
    if (format == "r32_float")
        return VERNON_TEXTURE_R32_FLOAT;
    if (format == "rg8_unorm")
        return VERNON_TEXTURE_RG8_UNORM;
    if (format == "rgb8_unorm")
        return VERNON_TEXTURE_RGB8_UNORM;
    if (format == "r11g11b10_float")
        return VERNON_TEXTURE_R11G11B10_FLOAT;
    if (format == "d32_float")
        return VERNON_TEXTURE_D32_FLOAT;
    if (format == "d32_float_s8_uint")
        return VERNON_TEXTURE_D32_FLOAT_S8_UINT;
    return std::nullopt;
}

namespace {

bool parseUint32(const nlohmann::json &value, uint32_t &result);
bool hasLegacyManifestKey(const nlohmann::json &value);
bool parseValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error);

template <size_t N> bool hasOnlyKeys(const nlohmann::json &value, const std::string_view (&allowed)[N]) {
    for (auto row = value.begin(); row != value.end(); ++row)
        if (std::find(std::begin(allowed), std::end(allowed), row.key()) == std::end(allowed))
            return false;
    return true;
}

constexpr std::string_view kVariantKeys[] = {"key", "program", "parameters", "internal_parameters", "outputs"};
constexpr std::string_view kExternalParameterKeys[] = {
    "slot",         "name",           "kind",  "type",          "uses",      "access",         "shape",
    "value_layout", "element_layout", "dtype", "address_space", "dimension", "texture_format",
};
constexpr std::string_view kInternalParameterKeys[] = {
    "name",           "kind",  "type",          "uses",      "access",         "shape",  "value_layout",
    "element_layout", "dtype", "address_space", "dimension", "texture_format", "source", "system_value",
};
constexpr std::string_view kParameterUseKeys[] = {
    "stage",
    "interface",
    "index",
    "shape",
    "dtype",
    "uniform_name",
    "vernon.location",
    "vernon.instance_divisor",
    "vernon.set",
    "vernon.binding",
    "transport",
    "value_layout",
    "interface_plan",
    "attribute_leaves",
    "sampled_texture_bindings",
    "tensor_view_descriptor",
};
constexpr std::string_view kOutputKeys[] = {"name", "kind", "dtype", "shape", "access", "location", "type"};
constexpr std::string_view kAttributeLeafKeys[] = {"path",  "location",        "location_offset",
                                                   "dtype", "component_count", "byte_offset"};
constexpr std::string_view kSampledTextureBindingKeys[] = {"set", "binding"};
constexpr std::string_view kInterfacePlanKeys[] = {"kind", "profile", "canonical_layout_hash", "root", "frame_offset"};
constexpr std::string_view kTransportNodeKeys[] = {"kind",      "representation", "offset",       "size",
                                                   "alignment", "shape",          "byte_strides", "children"};
constexpr std::string_view kTensorViewDescriptorKeys[] = {"rank", "offset_binding", "extent_bindings",
                                                          "stride_bindings"};

bool parseUint64(const nlohmann::json &value, uint64_t &result) {
    if (!value.is_number_integer())
        return false;
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
        return true;
    }
    const int64_t parsed = value.get<int64_t>();
    if (parsed < 0)
        return false;
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool parseInt64(const nlohmann::json &value, int64_t &result) {
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return false;
        result = static_cast<int64_t>(parsed);
        return true;
    }
    if (!value.is_number_integer())
        return false;
    result = value.get<int64_t>();
    return true;
}

bool parseTransportNode(const nlohmann::json &value, TransportNode &node, std::string &error) {
    if (!value.is_object() || !hasOnlyKeys(value, kTransportNodeKeys) || !value.contains("kind") ||
        !value["kind"].is_string() || !value.contains("offset") || !parseUint64(value["offset"], node.offset) ||
        !value.contains("size") || !parseUint64(value["size"], node.size) || !value.contains("alignment") ||
        !parseUint64(value["alignment"], node.alignment) || node.size == 0 || node.alignment == 0 ||
        (node.alignment & (node.alignment - 1))) {
        error = "interface plan node has invalid kind, offset, size, or alignment";
        return false;
    }
    const std::string kind = value["kind"].get<std::string>();
    if (kind == "scalar")
        node.kind = TransportNodeKind::Scalar;
    else if (kind == "product")
        node.kind = TransportNodeKind::Product;
    else if (kind == "array")
        node.kind = TransportNodeKind::Array;
    else {
        error = "interface plan node has an unsupported kind";
        return false;
    }
    if (value.contains("representation")) {
        if (!value["representation"].is_string()) {
            error = "interface plan scalar representation must be a string";
            return false;
        }
        node.representation = value["representation"].get<std::string>();
    }
    const auto parseUnsignedArray = [&](const char *key, std::vector<uint64_t> &target) {
        if (!value.contains(key))
            return true;
        if (!value[key].is_array())
            return false;
        for (const nlohmann::json &item : value[key]) {
            uint64_t parsed = 0;
            if (!parseUint64(item, parsed))
                return false;
            target.push_back(parsed);
        }
        return true;
    };
    if (!parseUnsignedArray("shape", node.shape) || !parseUnsignedArray("byte_strides", node.byteStrides)) {
        error = "interface plan node shape and strides must contain unsigned integers";
        return false;
    }
    if (value.contains("children")) {
        if (!value["children"].is_array()) {
            error = "interface plan node children must be an array";
            return false;
        }
        for (const nlohmann::json &childValue : value["children"]) {
            TransportNode child;
            if (!parseTransportNode(childValue, child, error))
                return false;
            if (child.offset > node.size || child.size > node.size - child.offset ||
                child.offset % child.alignment != 0) {
                error = "interface plan child exceeds its parent bounds";
                return false;
            }
            node.children.push_back(std::move(child));
        }
    }
    if ((node.kind == TransportNodeKind::Scalar &&
         (!node.children.empty() || node.representation.empty() || !node.shape.empty() || !node.byteStrides.empty())) ||
        (node.kind == TransportNodeKind::Product &&
         (node.children.empty() || !node.representation.empty() || !node.shape.empty() || !node.byteStrides.empty())) ||
        (node.kind == TransportNodeKind::Array &&
         (node.children.size() != 1 || !node.representation.empty() || node.shape.empty() ||
          node.byteStrides.size() != node.shape.size()))) {
        error = "interface plan node topology does not match its kind";
        return false;
    }
    if (node.kind == TransportNodeKind::Scalar) {
        const uint64_t representationSize =
            node.representation == "bool"                                                                  ? 1
            : node.representation == "f16"                                                                 ? 2
            : node.representation == "i32" || node.representation == "u32" || node.representation == "f32" ? 4
            : node.representation == "f64" || node.representation == "index"                               ? 8
                                                                                                           : 0;
        if (!representationSize || node.size != representationSize || node.alignment > representationSize) {
            error = "interface plan scalar representation does not match its size and alignment";
            return false;
        }
    }
    if (node.kind == TransportNodeKind::Product) {
        uint64_t end = 0;
        for (const TransportNode &child : node.children) {
            if (child.offset < end) {
                error = "interface plan product children overlap or are not ordered";
                return false;
            }
            end = child.offset + child.size;
        }
    }
    if (node.kind == TransportNodeKind::Array) {
        uint64_t maximumOffset = 0;
        for (size_t dimension = 0; dimension < node.shape.size(); ++dimension) {
            if (!node.shape[dimension] || !node.byteStrides[dimension] ||
                node.shape[dimension] - 1 >
                    (std::numeric_limits<uint64_t>::max() - maximumOffset) / node.byteStrides[dimension]) {
                error = "interface plan array extent and strides overflow";
                return false;
            }
            maximumOffset += (node.shape[dimension] - 1) * node.byteStrides[dimension];
        }
        const TransportNode &element = node.children.front();
        if (maximumOffset > node.size || element.offset > node.size - maximumOffset ||
            element.size > node.size - maximumOffset - element.offset) {
            error = "interface plan array extent exceeds its parent bounds";
            return false;
        }
    }
    return true;
}

bool parseInterfacePlan(const nlohmann::json &value, InterfacePlan &plan, std::string &error) {
    if (!value.is_object() || !hasOnlyKeys(value, kInterfacePlanKeys) || !value.contains("kind") ||
        !value["kind"].is_string() || !value.contains("profile") || !value["profile"].is_string() ||
        !value.contains("canonical_layout_hash") || !value["canonical_layout_hash"].is_string()) {
        error = "interface_plan is missing typed plan metadata";
        return false;
    }
    const std::string kind = value["kind"].get<std::string>();
    if (kind == "cpu_call")
        plan.kind = InterfacePlanKind::CpuCall;
    else if (kind == "kernel_parameter")
        plan.kind = InterfacePlanKind::KernelParameter;
    else if (kind == "byte_transport")
        plan.kind = InterfacePlanKind::ByteTransport;
    else if (kind == "native_uniform")
        plan.kind = InterfacePlanKind::NativeUniform;
    else {
        error = "interface_plan has an unsupported kind";
        return false;
    }
    plan.profile = value["profile"].get<std::string>();
    plan.canonicalLayoutHash = value["canonical_layout_hash"].get<std::string>();
    if (value.contains("frame_offset") && !parseUint64(value["frame_offset"], plan.frameOffset)) {
        error = "interface_plan frame_offset must be unsigned";
        return false;
    }
    if (plan.canonicalLayoutHash.empty()) {
        error = "interface_plan canonical layout hash is empty";
        return false;
    }
    if (!value.contains("root")) {
        error = "interface_plan must contain a recursive root";
        return false;
    }
    TransportNode root;
    if (!parseTransportNode(value["root"], root, error))
        return false;
    if (root.offset != 0) {
        error = "interface_plan root offset must be zero";
        return false;
    }
    plan.root = std::move(root);
    return true;
}

bool parseUse(const nlohmann::json &value, ParameterUse &use, std::string &error) {
    if (!value.is_object()) {
        error = "pipeline parameter use must be an object";
        return false;
    }
    if (hasLegacyManifestKey(value)) {
        error = "legacy compiler-generated parameter metadata is unsupported";
        return false;
    }
    if (!hasOnlyKeys(value, kParameterUseKeys)) {
        error = "pipeline parameter use contains an unknown field";
        return false;
    }
    if (!value.contains("stage") || !value["stage"].is_string() || value["stage"].get<std::string>().empty() ||
        !value.contains("interface") || !value["interface"].is_string() ||
        value["interface"].get<std::string>().empty()) {
        error = "pipeline parameter use is missing stage/interface metadata";
        return false;
    }
    use.stage = value["stage"].get<std::string>();
    use.interfaceKind = value["interface"].get<std::string>();
    if (value.contains("uniform_name")) {
        if (!value["uniform_name"].is_string()) {
            error = "pipeline parameter use uniform_name must be a string";
            return false;
        }
        use.uniformName = value["uniform_name"].get<std::string>();
    }
    if (value.contains("dtype")) {
        if (!value["dtype"].is_string()) {
            error = "pipeline parameter use dtype must be a string";
            return false;
        }
        use.dtype = value["dtype"].get<std::string>();
    }
    if (value.contains("index")) {
        if (!parseUint32(value["index"], use.index)) {
            error = "pipeline parameter use index must be a non-negative integer";
            return false;
        }
    }
    if (value.contains("vernon.location")) {
        if (!parseUint32(value["vernon.location"], use.location)) {
            error = "pipeline parameter use vernon.location must be a non-negative integer";
            return false;
        }
    }
    if (value.contains("vernon.instance_divisor")) {
        if (!parseUint32(value["vernon.instance_divisor"], use.divisor) || use.divisor == 0) {
            error = "pipeline parameter use vernon.instance_divisor must be a positive integer";
            return false;
        }
    }
    if (value.contains("vernon.set")) {
        if (!parseUint32(value["vernon.set"], use.descriptorSet)) {
            error = "pipeline parameter use vernon.set must be a non-negative integer";
            return false;
        }
    }
    if (value.contains("vernon.binding")) {
        if (!parseUint32(value["vernon.binding"], use.binding)) {
            error = "pipeline parameter use vernon.binding must be a non-negative integer";
            return false;
        }
    }
    if (value.contains("attribute_leaves")) {
        const nlohmann::json &leaves = value["attribute_leaves"];
        if (!leaves.is_array()) {
            error = "attribute_leaves must be an array";
            return false;
        }
        for (const nlohmann::json &leaf : leaves) {
            if (!leaf.is_object() || hasLegacyManifestKey(leaf) || !hasOnlyKeys(leaf, kAttributeLeafKeys)) {
                error =
                    "attribute leaf must contain dtype and unsigned location_offset, component_count, and byte_offset";
                return false;
            }
            uint64_t locationOffset = 0;
            uint64_t componentCount = 0;
            uint64_t byteOffset = 0;
            if (!leaf.contains("location_offset") || !parseUint64(leaf["location_offset"], locationOffset) ||
                !leaf.contains("component_count") || !parseUint64(leaf["component_count"], componentCount) ||
                !leaf.contains("byte_offset") || !parseUint64(leaf["byte_offset"], byteOffset) ||
                !leaf.contains("dtype") || !leaf["dtype"].is_string() || locationOffset > UINT32_MAX ||
                componentCount > UINT32_MAX || byteOffset > UINT32_MAX) {
                error =
                    "attribute leaf must contain dtype and unsigned location_offset, component_count, and byte_offset";
                return false;
            }
            use.attributeLeaves.push_back({static_cast<uint32_t>(locationOffset), leaf["dtype"].get<std::string>(),
                                           static_cast<uint32_t>(componentCount), static_cast<uint32_t>(byteOffset)});
        }
    }
    if (value.contains("transport")) {
        if (!value["transport"].is_string()) {
            error = "parameter use transport must be a string";
            return false;
        }
        use.transport = value["transport"].get<std::string>();
    }
    if (value.contains("value_layout")) {
        ValueLayout parsed;
        if (!parseValueLayout(value["value_layout"], parsed, error))
            return false;
        use.valueLayout = std::move(parsed);
    }
    if (value.contains("interface_plan")) {
        InterfacePlan parsed;
        if (!parseInterfacePlan(value["interface_plan"], parsed, error))
            return false;
        use.interfacePlan = std::move(parsed);
    }
    if (use.interfacePlan && use.valueLayout && use.interfacePlan->canonicalLayoutHash != use.valueLayout->layoutHash) {
        error = "interface_plan canonical layout hash does not match value_layout";
        return false;
    }
    if (value.contains("sampled_texture_bindings")) {
        const nlohmann::json &bindings = value["sampled_texture_bindings"];
        if (!bindings.is_array()) {
            error = "sampled_texture_bindings must be an array";
            return false;
        }
        for (const nlohmann::json &binding : bindings) {
            const auto parseBindingIndex = [](const nlohmann::json &field, uint32_t &result) {
                if (!field.is_number_integer())
                    return false;
                const int64_t value = field.get<int64_t>();
                if (value < 0 || static_cast<uint64_t>(value) > UINT32_MAX)
                    return false;
                result = static_cast<uint32_t>(value);
                return true;
            };
            uint32_t descriptorSet = 0;
            uint32_t descriptorBinding = 0;
            if (!binding.is_object() || hasLegacyManifestKey(binding) ||
                !hasOnlyKeys(binding, kSampledTextureBindingKeys) || !binding.contains("set") ||
                !parseBindingIndex(binding["set"], descriptorSet) || !binding.contains("binding") ||
                !parseBindingIndex(binding["binding"], descriptorBinding)) {
                error = "sampled texture binding must contain unsigned set/binding";
                return false;
            }
            use.sampledTextureBindings.push_back({descriptorSet, descriptorBinding});
        }
    }
    if (value.contains("shape")) {
        if (!value["shape"].is_array()) {
            error = "pipeline parameter use shape must be an array";
            return false;
        }
        for (const nlohmann::json &dimension : value["shape"]) {
            uint64_t extent = 0;
            if (!parseUint64(dimension, extent)) {
                error = "pipeline parameter use shape must contain unsigned extents";
                return false;
            }
            use.shape.push_back(extent);
        }
    }
    if (value.contains("tensor_view_descriptor")) {
        const nlohmann::json &descriptor = value["tensor_view_descriptor"];
        TensorViewDescriptorUse parsed;
        if (!descriptor.is_object() || !hasOnlyKeys(descriptor, kTensorViewDescriptorKeys) ||
            !descriptor.contains("rank") || !parseUint32(descriptor["rank"], parsed.rank) ||
            !descriptor.contains("offset_binding") ||
            !parseUint32(descriptor["offset_binding"], parsed.offsetBinding) ||
            !descriptor.contains("extent_bindings") || !descriptor["extent_bindings"].is_array() ||
            !descriptor.contains("stride_bindings") || !descriptor["stride_bindings"].is_array()) {
            error = "TensorView descriptor use is invalid";
            return false;
        }
        for (const nlohmann::json &binding : descriptor["extent_bindings"]) {
            uint32_t parsedBinding = 0;
            if (!parseUint32(binding, parsedBinding)) {
                error = "TensorView extent binding must be unsigned";
                return false;
            }
            parsed.extentBindings.push_back(parsedBinding);
        }
        for (const nlohmann::json &binding : descriptor["stride_bindings"]) {
            uint32_t parsedBinding = 0;
            if (!parseUint32(binding, parsedBinding)) {
                error = "TensorView stride binding must be unsigned";
                return false;
            }
            parsed.strideBindings.push_back(parsedBinding);
        }
        if (!parsed.rank || parsed.extentBindings.size() != parsed.rank ||
            parsed.strideBindings.size() != parsed.rank || (!use.shape.empty() && use.shape.size() != parsed.rank)) {
            error = "TensorView descriptor rank does not match shape";
            return false;
        }
        use.tensorViewDescriptor = std::move(parsed);
    }
    if (use.interfacePlan && use.interfacePlan->root && !use.interfacePlan->root->byteStrides.empty() &&
        use.interfacePlan->root->byteStrides.size() != use.shape.size()) {
        error = "interface_plan byte-stride rank " + std::to_string(use.interfacePlan->root->byteStrides.size()) +
                " does not match logical Tensor rank " + std::to_string(use.shape.size());
        return false;
    }
    return true;
}

bool parseValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error) {
    uint64_t byteSize = 0;
    uint64_t alignment = 0;
    if (!value.is_object() || !value.contains("logical_type") || !value["logical_type"].is_string() ||
        !value.contains("layout_hash") || !value["layout_hash"].is_string() || !value.contains("byte_size") ||
        !parseUint64(value["byte_size"], byteSize) || !value.contains("alignment") ||
        !parseUint64(value["alignment"], alignment) || !value.contains("leaves") || !value["leaves"].is_array() ||
        !byteSize || !alignment || byteSize > UINT32_MAX || alignment > UINT32_MAX) {
        error = "element_layout must contain logical_type, layout_hash, byte_size, alignment, and leaves";
        return false;
    }
    layout.logicalType = value["logical_type"].get<std::string>();
    layout.structName = value.value("struct_name", "");
    layout.layoutHash = value["layout_hash"].get<std::string>();
    layout.byteSize = static_cast<uint32_t>(byteSize);
    layout.alignment = static_cast<uint32_t>(alignment);
    if (layout.logicalType.empty() || layout.layoutHash.empty() || (layout.alignment & (layout.alignment - 1))) {
        error = "element_layout contains invalid type, hash, or alignment";
        return false;
    }
    for (const nlohmann::json &leaf : value["leaves"]) {
        uint64_t scalarCount = 0;
        uint64_t byteOffset = 0;
        if (!leaf.is_object() || !leaf.contains("dtype") || !leaf["dtype"].is_string() ||
            !leaf.contains("scalar_count") || !parseUint64(leaf["scalar_count"], scalarCount) ||
            !leaf.contains("byte_offset") || !parseUint64(leaf["byte_offset"], byteOffset) || !leaf.contains("path") ||
            !leaf["path"].is_array() || !scalarCount || scalarCount > UINT32_MAX || byteOffset > UINT32_MAX) {
            error = "element_layout leaf must contain path, dtype, scalar_count, and byte_offset";
            return false;
        }
        const std::string dtype = leaf["dtype"].get<std::string>();
        const auto parsedDtype = pipelineDataType(dtype);
        if (!parsedDtype || byteOffset >= byteSize) {
            error = "element_layout leaf has unsupported dtype or byte offset";
            return false;
        }
        ValueLeaf parsedLeaf;
        parsedLeaf.dtype = dtype;
        parsedLeaf.scalarCount = static_cast<uint32_t>(scalarCount);
        parsedLeaf.byteOffset = static_cast<uint32_t>(byteOffset);
        for (const nlohmann::json &component : leaf["path"]) {
            if (component.is_string())
                parsedLeaf.path.push_back({component.get<std::string>(), 0});
            else {
                uint64_t index = 0;
                if (!parseUint64(component, index)) {
                    error = "element_layout leaf path components must be field names or unsigned indices";
                    return false;
                }
                parsedLeaf.path.push_back({std::nullopt, index});
            }
        }
        if (leaf.contains("shape")) {
            if (!leaf["shape"].is_array()) {
                error = "element_layout leaf shape must be an array";
                return false;
            }
            uint64_t shapeCount = 1;
            for (const nlohmann::json &extentValue : leaf["shape"]) {
                uint64_t extent = 0;
                if (!parseUint64(extentValue, extent) || !extent ||
                    shapeCount > std::numeric_limits<uint64_t>::max() / extent) {
                    error = "element_layout leaf shape must contain positive non-overflowing extents";
                    return false;
                }
                shapeCount *= extent;
                parsedLeaf.shape.push_back(extent);
            }
            if (shapeCount != scalarCount) {
                error = "element_layout leaf shape does not match scalar_count";
                return false;
            }
        }
        layout.leaves.push_back(std::move(parsedLeaf));
        layout.abiLeaves.push_back({static_cast<uint32_t>(*parsedDtype), static_cast<uint32_t>(scalarCount),
                                    static_cast<uint32_t>(byteOffset)});
    }
    if (layout.leaves.empty()) {
        error = "element_layout must contain at least one semantic leaf";
        return false;
    }
    rebuildValueLayoutPathViews(layout);
    return true;
}

bool isScalarLayout(const ValueLayout &layout, const char *dtype) {
    return layout.leaves.size() == 1 && layout.leaves[0].dtype == dtype && layout.leaves[0].scalarCount == 1 &&
           layout.leaves[0].byteOffset == 0;
}

void parseStaticType(const std::string &type, std::string &dtype, std::vector<uint64_t> &shape) {
    if (type.rfind("tensor<", 0) != 0 || type.size() < 9 || type.back() != '>') {
        dtype = type;
        return;
    }
    const std::string body = type.substr(7, type.size() - 8);
    size_t begin = 0;
    while (true) {
        const size_t separator = body.find('x', begin);
        if (separator == std::string::npos) {
            dtype = body.substr(begin);
            return;
        }
        const std::string dimension = body.substr(begin, separator - begin);
        if (dimension == "?")
            shape.push_back(0);
        else {
            try {
                shape.push_back(std::stoull(dimension));
            } catch (...) {
                dtype.clear();
                shape.clear();
                return;
            }
        }
        begin = separator + 1;
    }
}

bool parseUint32(const nlohmann::json &value, uint32_t &result) {
    if (!value.is_number_integer())
        return false;
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed > UINT32_MAX)
            return false;
        result = static_cast<uint32_t>(parsed);
    } else {
        const int64_t parsed = value.get<int64_t>();
        if (parsed < 0 || static_cast<uint64_t>(parsed) > UINT32_MAX)
            return false;
        result = static_cast<uint32_t>(parsed);
    }
    return true;
}

bool parseVersion(const nlohmann::json &value, RuntimeVersion &version) {
    if (!value.is_array() || value.size() != 2 || !parseUint32(value[0], version.major) ||
        !parseUint32(value[1], version.minor))
        return false;
    return version.major != 0;
}

bool hasOnlyKeys(const nlohmann::json &value, std::initializer_list<std::string_view> allowed) {
    for (auto row = value.begin(); row != value.end(); ++row)
        if (std::find(allowed.begin(), allowed.end(), row.key()) == allowed.end())
            return false;
    return true;
}

bool hasLegacyManifestKey(const nlohmann::json &value) {
    static constexpr std::string_view legacyKeys[] = {"vernon.compiler_generated", "vernon.implicit_sampler",
                                                      "vernon.system_value"};
    for (std::string_view key : legacyKeys)
        if (value.contains(std::string(key)))
            return true;
    return false;
}

enum class LogicalParameterTypeKind { TensorView, Texture, Sampler, Tensor, Struct, Scalar, Invalid };

struct ParsedLogicalParameterType {
    LogicalParameterTypeKind kind{LogicalParameterTypeKind::Invalid};
    std::string addressSpace;
    std::string textureDimension;
};

bool parseQuotedManifestToken(std::string_view text, size_t &cursor, std::string &token) {
    if (cursor >= text.size() || text[cursor] != '"')
        return false;
    ++cursor;
    token.clear();
    while (cursor < text.size()) {
        if (text[cursor] == '"') {
            ++cursor;
            return !token.empty();
        }
        token.push_back(text[cursor++]);
    }
    return false;
}

bool parseTensorViewLogicalType(const std::string &type, ParsedLogicalParameterType &parsed) {
    constexpr std::string_view prefix = "!vernon.tensor_view<";
    if (type.rfind(prefix, 0) != 0 || type.back() != '>' || type.size() <= prefix.size() + 1)
        return false;
    std::string_view body(type.data() + prefix.size(), type.size() - prefix.size() - 1);
    const size_t shapeOpen = body.find('[');
    const size_t shapeClose = shapeOpen == std::string_view::npos ? std::string_view::npos : body.find(']', shapeOpen);
    if (shapeOpen == std::string_view::npos || shapeClose == std::string_view::npos || shapeClose <= shapeOpen + 1)
        return false;
    size_t cursor = shapeClose + 1;
    while (cursor < body.size() && body[cursor] == ' ')
        ++cursor;
    if (cursor >= body.size() || body[cursor] != ',')
        return false;
    ++cursor;
    while (cursor < body.size() && body[cursor] == ' ')
        ++cursor;
    std::string access;
    if (!parseQuotedManifestToken(body, cursor, access))
        return false;
    if (access != "read" && access != "write" && access != "read_write")
        return false;
    while (cursor < body.size() && body[cursor] == ' ')
        ++cursor;
    if (cursor >= body.size() || body[cursor] != ',')
        return false;
    ++cursor;
    while (cursor < body.size() && body[cursor] == ' ')
        ++cursor;
    if (!parseQuotedManifestToken(body, cursor, parsed.addressSpace))
        return false;
    while (cursor < body.size() && body[cursor] == ' ')
        ++cursor;
    if (cursor != body.size())
        return false;
    if (parsed.addressSpace != "device" && parsed.addressSpace != "workgroup" && parsed.addressSpace != "private")
        return false;
    parsed.kind = LogicalParameterTypeKind::TensorView;
    return true;
}

bool parseTextureLogicalType(const std::string &type, ParsedLogicalParameterType &parsed) {
    constexpr std::string_view prefix = "!vernon.texture<";
    if (type.rfind(prefix, 0) != 0 || type.back() != '>' || type.size() <= prefix.size() + 1)
        return false;
    size_t cursor = prefix.size();
    if (!parseQuotedManifestToken(type, cursor, parsed.textureDimension))
        return false;
    while (cursor < type.size() && type[cursor] == ' ')
        ++cursor;
    if (cursor >= type.size() || type[cursor] != ',')
        return false;
    parsed.kind = LogicalParameterTypeKind::Texture;
    return true;
}

ParsedLogicalParameterType parseLogicalParameterType(const std::string &type) {
    ParsedLogicalParameterType parsed;
    if (parseTensorViewLogicalType(type, parsed))
        return parsed;
    if (parseTextureLogicalType(type, parsed))
        return parsed;
    if (type == "!vernon.sampler") {
        parsed.kind = LogicalParameterTypeKind::Sampler;
        return parsed;
    }
    if ((type.rfind("tensor<", 0) == 0 || type.rfind("vector<", 0) == 0 || type.rfind("!vernon.tensor<", 0) == 0) &&
        type.back() == '>') {
        parsed.kind = LogicalParameterTypeKind::Tensor;
        return parsed;
    }
    if ((type.rfind("!vernon.struct<", 0) == 0 || type.rfind("tuple<", 0) == 0) && type.back() == '>') {
        parsed.kind = LogicalParameterTypeKind::Struct;
        return parsed;
    }
    if (!type.empty() && type.find('<') == std::string::npos)
        parsed.kind = LogicalParameterTypeKind::Scalar;
    return parsed;
}

bool validateParameterKindTypeCoherence(const std::string &kind, const ParsedLogicalParameterType &logicalType,
                                        std::string &error) {
    if (logicalType.kind == LogicalParameterTypeKind::Invalid) {
        error = "pipeline parameter type is not a recognized schema-v5 logical type";
        return false;
    }
    if (kind == "texture") {
        if (logicalType.kind != LogicalParameterTypeKind::Texture) {
            error = "texture parameter kind does not match its logical type";
            return false;
        }
        return true;
    }
    if (kind == "sampler") {
        if (logicalType.kind != LogicalParameterTypeKind::Sampler) {
            error = "sampler parameter kind does not match its logical type";
            return false;
        }
        return true;
    }
    if (kind == "tensor") {
        if (logicalType.kind == LogicalParameterTypeKind::Texture ||
            logicalType.kind == LogicalParameterTypeKind::Sampler) {
            error = "tensor parameter kind does not match its logical type";
            return false;
        }
        return true;
    }
    error = "pipeline parameter kind must be tensor, texture, or sampler";
    return false;
}

bool validateParameterAddressSpace(const std::string &kind, const ParsedLogicalParameterType &logicalType,
                                   const std::string &addressSpace, std::string &error) {
    if (logicalType.kind == LogicalParameterTypeKind::TensorView) {
        if (addressSpace != "device" || logicalType.addressSpace != "device") {
            error = "pipeline TensorView parameter must use device address space";
            return false;
        }
        return true;
    }
    if (!addressSpace.empty()) {
        error = "non-TensorView parameter contains address_space";
        return false;
    }
    if (kind == "texture" && !logicalType.textureDimension.empty()) {
        // Dimension is validated separately against the reflected parameter.dimension field.
    }
    return true;
}

} // namespace

void rebuildValueLayoutPathViews(ValueLayout &layout) {
    for (ValueLeaf &leaf : layout.leaves) {
        leaf.abiPath.clear();
        leaf.abiPath.reserve(leaf.path.size());
        for (const ValuePathComponent &component : leaf.path) {
            if (component.field)
                leaf.abiPath.push_back(
                    {VERNON_VALUE_PATH_FIELD, {component.field->data(), component.field->size()}, 0});
            else
                leaf.abiPath.push_back({VERNON_VALUE_PATH_INDEX, {nullptr, 0}, component.index});
        }
    }
}

bool parsePipelineValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error) {
    return parseValueLayout(value, layout, error);
}

bool parsePipelineInterfacePlan(const nlohmann::json &value, InterfacePlan &plan, std::string &error) {
    return parseInterfacePlan(value, plan, error);
}

bool Variant::validate(std::string &error) const {
    const auto validTextureDimension = [](const std::string &dimension) {
        return dimension == "2d" || dimension == "3d" || dimension == "cube";
    };
    const auto validTextureFormat = [](const std::string &format) {
        return format.empty() || format == "rgba8_unorm" || format == "rgba8_srgb" || format == "rgba16_float" ||
               format == "rgba32_float" || format == "r8_unorm" || format == "r16_float" || format == "r32_float" ||
               format == "rg8_unorm" || format == "rgb8_unorm" || format == "r11g11b10_float";
    };
    if (!std::is_sorted(key.begin(), key.end()) || std::adjacent_find(key.begin(), key.end()) != key.end()) {
        error = "pipeline variant feature key is not canonical";
        return false;
    }
    if (!std::is_sorted(parameters.begin(), parameters.end(),
                        [](const Parameter &left, const Parameter &right) { return left.slot < right.slot; }) ||
        std::adjacent_find(parameters.begin(), parameters.end(), [](const Parameter &left, const Parameter &right) {
            return left.slot == right.slot;
        }) != parameters.end()) {
        error = "pipeline variant parameter slots are not unique and sorted";
        return false;
    }
    if (!std::is_sorted(internalParameters.begin(), internalParameters.end(),
                        [](const Parameter &left, const Parameter &right) { return left.name < right.name; }) ||
        std::adjacent_find(internalParameters.begin(), internalParameters.end(),
                           [](const Parameter &left, const Parameter &right) { return left.name == right.name; }) !=
            internalParameters.end()) {
        error = "pipeline variant internal parameters are not unique and sorted";
        return false;
    }
    for (const Parameter &parameter : parameters) {
        if (parameter.name.empty() || parameter.uses.empty() || !parameter.source.empty() ||
            !parameter.systemValue.empty()) {
            error = "external pipeline parameter invariant failed";
            return false;
        }
        const bool textureConstraintsValid =
            parameter.kind == "texture"
                ? validTextureDimension(parameter.dimension) && validTextureFormat(parameter.textureFormat)
                : parameter.dimension.empty() && parameter.textureFormat.empty();
        if (!textureConstraintsValid) {
            error = "pipeline parameter texture constraint invariant failed";
            return false;
        }
        for (const ParameterUse &use : parameter.uses)
            if (parameter.kind == "tensor" && (use.interfaceKind == "uniform" || use.interfaceKind == "value") &&
                !use.interfacePlan) {
                error = "packed Tensor value is missing its typed interface plan";
                return false;
            }
    }
    for (const Parameter &parameter : internalParameters) {
        const bool implicitSampler =
            parameter.source == "implicit_sampler" && parameter.kind == "sampler" && parameter.systemValue.empty();
        const bool resolution = parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                                parameter.kind == "tensor" && isScalarLayout(parameter.elementLayout, "f32") &&
                                parameter.shape == std::vector<uint64_t>{2};
        if (parameter.name.empty() || parameter.uses.empty() || (!implicitSampler && !resolution)) {
            error = "internal pipeline parameter invariant failed";
            return false;
        }
        for (const ParameterUse &use : parameter.uses) {
            if (parameter.kind == "tensor" && (use.interfaceKind == "uniform" || use.interfaceKind == "value") &&
                !use.interfacePlan) {
                error = "packed Tensor value is missing its typed interface plan";
                return false;
            }
            if (implicitSampler && use.sampledTextureBindings.empty()) {
                error = "implicit sampler requires reflected sampled texture bindings";
                return false;
            }
            if (resolution && !use.sampledTextureBindings.empty()) {
                error = "resolution cannot have sampled texture bindings";
                return false;
            }
        }
    }
    using BindingKey = std::tuple<std::string, uint32_t, uint32_t>;
    std::map<BindingKey, std::pair<std::string, std::string>> descriptorOwners;
    std::set<BindingKey> textureBindings;
    std::vector<BindingKey> samplerBindings;
    const auto validateBindings = [&](const Parameter &parameter) {
        for (const ParameterUse &use : parameter.uses) {
            if (program.find(use.stage) == program.end()) {
                error = "pipeline parameter use references a stage outside its program";
                return false;
            }
            const bool descriptorRequired =
                (use.interfaceKind == "resource" && (parameter.kind == "tensor" || parameter.kind == "texture")) ||
                ((use.interfaceKind == "uniform" || use.interfaceKind == "value") &&
                 (use.transport == "uniform_buffer" || use.transport == "storage_buffer"));
            if (descriptorRequired && use.binding == UINT32_MAX) {
                error = "descriptor-backed pipeline parameter is missing set/binding";
                return false;
            }
            if (use.binding != UINT32_MAX && parameter.kind != "sampler") {
                BindingKey key{use.stage, use.descriptorSet, use.binding};
                const auto [found, inserted] = descriptorOwners.emplace(key, std::pair{parameter.name, parameter.kind});
                if (!inserted && found->second.first != parameter.name) {
                    error = "pipeline descriptor binding is assigned to multiple parameters";
                    return false;
                }
                if (parameter.kind == "texture")
                    textureBindings.insert(std::move(key));
            }
            if (parameter.kind == "sampler") {
                if (use.sampledTextureBindings.empty()) {
                    error = "sampler parameter has no paired sampled texture binding";
                    return false;
                }
                for (const SampledTextureBinding &binding : use.sampledTextureBindings)
                    samplerBindings.emplace_back(use.stage, binding.descriptorSet, binding.binding);
            } else if (!use.sampledTextureBindings.empty()) {
                error = "non-sampler parameter contains sampled texture bindings";
                return false;
            }
        }
        return true;
    };
    for (const Parameter &parameter : parameters)
        if (!validateBindings(parameter))
            return false;
    for (const Parameter &parameter : internalParameters)
        if (!validateBindings(parameter))
            return false;
    for (const BindingKey &binding : samplerBindings)
        if (textureBindings.find(binding) == textureBindings.end()) {
            error = "sampler references an unknown sampled texture binding";
            return false;
        }
    const bool computeTopology = !compute.empty() && program.size() == 1;
    const bool graphicsTopology = compute.empty() && !vertex.empty() && !fragment.empty() && program.size() == 2;
    if (!computeTopology && !graphicsTopology) {
        error = "pipeline variant must contain either one compute program or one graphics program";
        return false;
    }
    return true;
}

bool parseRuntimeRequirements(const nlohmann::json &root, const std::string &target, RuntimeRequirements &requirements,
                              std::string &error) {
    if (!root.contains("runtime_requirements")) {
        error = "pipeline manifest requires runtime_requirements";
        return false;
    }
    const nlohmann::json &value = root["runtime_requirements"];
    if (!value.is_object() || !value.contains("backend") || !value["backend"].is_string() ||
        !value.contains("features") || !value["features"].is_array()) {
        error = "runtime_requirements must contain backend and features";
        return false;
    }
    requirements.backend = value["backend"].get<std::string>();
    if (requirements.backend != target) {
        error = "runtime_requirements backend does not match pipeline target";
        return false;
    }
    for (const nlohmann::json &feature : value["features"]) {
        if (!feature.is_string()) {
            error = "runtime requirement feature must be a string";
            return false;
        }
        requirements.features.push_back(feature.get<std::string>());
    }
    static constexpr std::string_view knownFeatures[] = {"atomics",  "barriers",     "compute",  "instancing",
                                                         "samplers", "tensor_views", "textures", "workgroup_storage"};
    if (!std::is_sorted(requirements.features.begin(), requirements.features.end()) ||
        std::adjacent_find(requirements.features.begin(), requirements.features.end()) != requirements.features.end() ||
        std::any_of(requirements.features.begin(), requirements.features.end(), [](const std::string &feature) {
            return std::find(std::begin(knownFeatures), std::end(knownFeatures), feature) == std::end(knownFeatures);
        })) {
        error = "runtime requirement features must be known, unique, and sorted";
        return false;
    }
    if (target == "cpu") {
        if (!hasOnlyKeys(value, {"backend", "features", "target_triple", "object_format"}) ||
            !value.contains("target_triple") || !value["target_triple"].is_string() ||
            value["target_triple"].get_ref<const std::string &>().empty() || !value.contains("object_format") ||
            !value["object_format"].is_string()) {
            error = "CPU runtime requirements are invalid";
            return false;
        }
        requirements.targetTriple = value["target_triple"].get<std::string>();
        requirements.objectFormat = value["object_format"].get<std::string>();
        if ((requirements.objectFormat != "coff" && requirements.objectFormat != "elf" &&
             requirements.objectFormat != "macho" && requirements.objectFormat != "wasm")) {
            error = "CPU runtime requirements are invalid";
            return false;
        }
        return true;
    }
    if (target == "opengl" || target == "opengles") {
        if (!hasOnlyKeys(value, {"backend", "features", "glsl_version", "profile", "api_version"}) ||
            !value.contains("glsl_version") || !parseUint32(value["glsl_version"], requirements.glslVersion) ||
            !value.contains("profile") || !value["profile"].is_string() || !value.contains("api_version") ||
            !parseVersion(value["api_version"], requirements.apiVersion)) {
            error = "OpenGL runtime requirements are invalid";
            return false;
        }
        requirements.profile = value["profile"].get<std::string>();
        if ((target == "opengles" && requirements.profile != "es") ||
            (target == "opengl" && requirements.profile != "core" && requirements.profile != "compatibility")) {
            error = "OpenGL runtime requirement profile does not match target";
            return false;
        }
        if (requirements.glslVersion != glslVersionForApi(requirements.apiVersion)) {
            error = "OpenGL GLSL and API versions are inconsistent";
            return false;
        }
        return true;
    }
    if (target == "vulkan") {
        if (!hasOnlyKeys(value, {"backend", "features", "api_version", "spirv_version", "compute_workgroup_size"}) ||
            !value.contains("api_version") || !parseVersion(value["api_version"], requirements.apiVersion) ||
            !value.contains("spirv_version") || !parseVersion(value["spirv_version"], requirements.shaderVersion)) {
            error = "Vulkan runtime requirements are invalid";
            return false;
        }
        if (value.contains("compute_workgroup_size")) {
            const nlohmann::json &workgroup = value["compute_workgroup_size"];
            if (!workgroup.is_array() || workgroup.size() != 3) {
                error = "Vulkan compute workgroup requirement must have three dimensions";
                return false;
            }
            for (size_t index = 0; index < 3; ++index) {
                if (!parseUint32(workgroup[index], requirements.computeWorkgroupSize[index]) ||
                    requirements.computeWorkgroupSize[index] == 0) {
                    error = "Vulkan compute workgroup dimensions must be positive";
                    return false;
                }
            }
        }
        return true;
    }
    if (target == "metal") {
        if (!hasOnlyKeys(value, {"backend", "features", "apple_platform", "msl_version", "minimum_os_version"}) ||
            !value.contains("apple_platform") || !value["apple_platform"].is_string() ||
            !value.contains("msl_version") || !parseVersion(value["msl_version"], requirements.shaderVersion) ||
            !value.contains("minimum_os_version") ||
            !parseVersion(value["minimum_os_version"], requirements.minimumOsVersion)) {
            error = "Metal runtime requirements are invalid";
            return false;
        }
        requirements.applePlatform = value["apple_platform"].get<std::string>();
        const RuntimeVersion minimumSupported =
            requirements.applePlatform == "ios" ? RuntimeVersion{15, 0} : RuntimeVersion{11, 0};
        if ((requirements.applePlatform != "macos" && requirements.applePlatform != "ios") ||
            requirements.shaderVersion.major != 2 || requirements.shaderVersion.minor != 4 ||
            !runtimeVersionAtLeast(requirements.minimumOsVersion, minimumSupported)) {
            error = "Metal runtime requirements contain an unsupported Apple platform or version";
            return false;
        }
        return true;
    }
    if (target == "directx") {
        if (!hasOnlyKeys(value, {"backend", "features", "api_version", "minimum_feature_level", "shader_model",
                                 "root_signature_version", "compute_workgroup_size"}) ||
            !value.contains("api_version") || !parseVersion(value["api_version"], requirements.apiVersion) ||
            !value.contains("minimum_feature_level") ||
            !parseVersion(value["minimum_feature_level"], requirements.minimumFeatureLevel) ||
            !value.contains("shader_model") || !parseVersion(value["shader_model"], requirements.shaderVersion) ||
            !value.contains("root_signature_version") ||
            !parseVersion(value["root_signature_version"], requirements.rootSignatureVersion)) {
            error = "DirectX runtime requirements are invalid";
            return false;
        }
        if (requirements.apiVersion.major != 12 || requirements.shaderVersion.major < 6 ||
            requirements.rootSignatureVersion.major != 1) {
            error = "DirectX runtime requires D3D12, Shader Model 6+, and root signature 1.x";
            return false;
        }
        if (value.contains("compute_workgroup_size")) {
            const nlohmann::json &workgroup = value["compute_workgroup_size"];
            if (!workgroup.is_array() || workgroup.size() != 3) {
                error = "DirectX compute workgroup requirement must have three dimensions";
                return false;
            }
            for (size_t index = 0; index < 3; ++index)
                if (!parseUint32(workgroup[index], requirements.computeWorkgroupSize[index]) ||
                    requirements.computeWorkgroupSize[index] == 0) {
                    error = "DirectX compute workgroup dimensions must be positive";
                    return false;
                }
        }
        return true;
    }
    if (target == "cuda") {
        if (!hasOnlyKeys(value, {"backend", "features", "ptx_version", "minimum_compute_capability", "address_size"}) ||
            !value.contains("ptx_version") || !parseVersion(value["ptx_version"], requirements.shaderVersion) ||
            !value.contains("minimum_compute_capability") ||
            !parseVersion(value["minimum_compute_capability"], requirements.minimumComputeCapability) ||
            !value.contains("address_size") || !parseUint32(value["address_size"], requirements.addressSize)) {
            error = "CUDA runtime requirements are invalid";
            return false;
        }
        if (requirements.addressSize != 32 && requirements.addressSize != 64) {
            error = "CUDA PTX address size must be 32 or 64";
            return false;
        }
        return true;
    }
    error = "runtime requirements are unsupported for pipeline target";
    return false;
}

bool runtimeVersionAtLeast(RuntimeVersion actual, RuntimeVersion required) {
    return actual.major > required.major || (actual.major == required.major && actual.minor >= required.minor);
}

uint32_t glslVersionForApi(RuntimeVersion apiVersion) { return apiVersion.major * 100 + apiVersion.minor * 10; }

bool parseVariant(const nlohmann::json &value, Variant &variant, std::string &error) {
    if (!value.is_object() || !value.contains("key") || !value["key"].is_array() || !value.contains("program") ||
        !value["program"].is_object() || !value.contains("parameters") || !value["parameters"].is_array() ||
        !value.contains("outputs") || !value["outputs"].is_array()) {
        error = "pipeline variant tables are invalid";
        return false;
    }
    if (hasLegacyManifestKey(value) || !hasOnlyKeys(value, kVariantKeys)) {
        error = "pipeline variant contains an unknown or legacy field";
        return false;
    }
    for (const nlohmann::json &feature : value["key"]) {
        if (!feature.is_string()) {
            error = "pipeline variant feature is not a string";
            return false;
        }
        variant.key.push_back(feature.get<std::string>());
    }
    if (!std::is_sorted(variant.key.begin(), variant.key.end()) ||
        std::adjacent_find(variant.key.begin(), variant.key.end()) != variant.key.end()) {
        error = "pipeline variant feature key is not canonical";
        return false;
    }
    auto parseParameter = [&](const nlohmann::json &row, bool internal, Parameter &parameter) {
        if (!row.is_object() || (!internal && !row.contains("slot")) || !row.contains("name") ||
            !row["name"].is_string() || row["name"].get_ref<const std::string &>().empty() || !row.contains("kind") ||
            !row["kind"].is_string() || row["kind"].get_ref<const std::string &>().empty() || !row.contains("type") ||
            !row["type"].is_string() || row["type"].get_ref<const std::string &>().empty() || !row.contains("access") ||
            !row["access"].is_string() || !row.contains("shape") || !row["shape"].is_array() || !row.contains("uses") ||
            !row["uses"].is_array() || (internal && (!row.contains("source") || !row["source"].is_string()))) {
            error = internal ? "internal pipeline parameter record is invalid" : "pipeline parameter record is invalid";
            return false;
        }
        if (hasLegacyManifestKey(row) ||
            !(internal ? hasOnlyKeys(row, kInternalParameterKeys) : hasOnlyKeys(row, kExternalParameterKeys))) {
            error = internal ? "internal pipeline parameter contains an unknown or legacy field"
                             : "pipeline parameter contains an unknown or legacy field";
            return false;
        }
        if (!internal && !parseUint32(row["slot"], parameter.slot)) {
            error = "pipeline parameter slot must be a non-negative uint32";
            return false;
        }
        for (std::string_view field : {"name", "kind", "access", "dtype", "address_space", "dimension",
                                       "texture_format", "source", "system_value"})
            if (row.contains(std::string(field)) && !row[std::string(field)].is_string()) {
                error = "pipeline parameter string metadata has an invalid type";
                return false;
            }
        parameter.name = row["name"].get<std::string>();
        parameter.kind = row["kind"].get<std::string>();
        parameter.source = row.value("source", "");
        parameter.systemValue = row.value("system_value", "");
        parameter.access = row["access"].get<std::string>();
        parameter.addressSpace = row.value("address_space", "");
        parameter.dimension = row.value("dimension", "");
        parameter.textureFormat = row.value("texture_format", "");
        for (const nlohmann::json &dimension : row["shape"]) {
            uint64_t extent = 0;
            if (!parseUint64(dimension, extent)) {
                error = "pipeline parameter shape must contain unsigned extents";
                return false;
            }
            parameter.shape.push_back(extent);
        }
        for (const nlohmann::json &useValue : row["uses"]) {
            ParameterUse use;
            if (!parseUse(useValue, use, error))
                return false;
            parameter.uses.push_back(std::move(use));
        }
        if (parameter.name.empty() || parameter.uses.empty()) {
            error =
                internal ? "internal pipeline parameter has no name or uses" : "pipeline parameter has no name or uses";
            return false;
        }
        const std::string logicalType = row["type"].get<std::string>();
        const ParsedLogicalParameterType parsedType = parseLogicalParameterType(logicalType);
        if (!validateParameterKindTypeCoherence(parameter.kind, parsedType, error) ||
            !validateParameterAddressSpace(parameter.kind, parsedType, parameter.addressSpace, error))
            return false;
        if (parsedType.kind == LogicalParameterTypeKind::Texture && !parsedType.textureDimension.empty() &&
            parsedType.textureDimension != parameter.dimension) {
            error = "texture parameter dimension does not match its logical type";
            return false;
        }
        if (parameter.kind == "texture") {
            if (!pipelineTextureDimension(parameter.dimension) ||
                (!parameter.textureFormat.empty() && !pipelineTextureFormat(parameter.textureFormat))) {
                error = "texture parameter has an invalid dimension or texture_format";
                return false;
            }
        } else if (!parameter.dimension.empty() || !parameter.textureFormat.empty()) {
            error = "non-texture parameter contains texture constraints";
            return false;
        }
        if (parameter.kind == "tensor") {
            if (row.contains("dtype") || row.contains("element_layout") == row.contains("value_layout")) {
                error = "Tensor parameter must contain exactly one value_layout or element_layout";
                return false;
            }
            if (row.contains("element_layout")) {
                if (!parseValueLayout(row["element_layout"], parameter.elementLayout, error))
                    return false;
            } else {
                ValueLayout layout;
                if (!parseValueLayout(row["value_layout"], layout, error))
                    return false;
                parameter.valueLayout = std::move(layout);
            }
        } else if (row.contains("element_layout") || row.contains("value_layout")) {
            error = "resource parameter contains a value layout";
            return false;
        }
        if (internal) {
            const bool implicitSampler =
                parameter.source == "implicit_sampler" && parameter.kind == "sampler" && parameter.systemValue.empty();
            const bool resolution = parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                                    parameter.kind == "tensor" && isScalarLayout(parameter.elementLayout, "f32") &&
                                    parameter.shape == std::vector<uint64_t>{2};
            if (!implicitSampler && !resolution) {
                error = "internal pipeline parameter metadata is unsupported";
                return false;
            }
        } else if (!parameter.source.empty() || !parameter.systemValue.empty()) {
            error = "external pipeline parameter contains internal metadata";
            return false;
        }
        return true;
    };
    for (const nlohmann::json &row : value["parameters"]) {
        Parameter parameter;
        if (!parseParameter(row, false, parameter))
            return false;
        variant.parameters.push_back(std::move(parameter));
    }
    const nlohmann::json internalRows = value.value("internal_parameters", nlohmann::json::array());
    if (!internalRows.is_array()) {
        error = "internal_parameters must be an array";
        return false;
    }
    for (const nlohmann::json &row : internalRows) {
        Parameter parameter;
        if (!parseParameter(row, true, parameter))
            return false;
        variant.internalParameters.push_back(std::move(parameter));
    }
    std::sort(variant.parameters.begin(), variant.parameters.end(),
              [](const Parameter &left, const Parameter &right) { return left.slot < right.slot; });
    if (std::adjacent_find(variant.parameters.begin(), variant.parameters.end(),
                           [](const Parameter &left, const Parameter &right) { return left.slot == right.slot; }) !=
        variant.parameters.end()) {
        error = "pipeline variant contains duplicate parameter slots";
        return false;
    }
    std::sort(variant.internalParameters.begin(), variant.internalParameters.end(),
              [](const Parameter &left, const Parameter &right) { return left.name < right.name; });
    if (std::adjacent_find(variant.internalParameters.begin(), variant.internalParameters.end(),
                           [](const Parameter &left, const Parameter &right) { return left.name == right.name; }) !=
        variant.internalParameters.end()) {
        error = "pipeline variant contains duplicate internal parameters";
        return false;
    }
    for (const Parameter &parameter : variant.internalParameters) {
        if (parameter.source != "implicit_sampler")
            continue;
        for (const ParameterUse &use : parameter.uses)
            if (use.sampledTextureBindings.empty()) {
                error = "implicit sampler has no paired sampled texture binding";
                return false;
            }
    }
    for (const nlohmann::json &row : value["outputs"]) {
        if (!row.is_object() || hasLegacyManifestKey(row) || !hasOnlyKeys(row, kOutputKeys) || !row.contains("name") ||
            !row["name"].is_string() || row["name"].get_ref<const std::string &>().empty() || !row.contains("kind") ||
            !row["kind"].is_string() || !row.contains("access") || !row["access"].is_string() ||
            !row.contains("location")) {
            error = "pipeline output record is invalid";
            return false;
        }
        Output output;
        output.name = row["name"].get<std::string>();
        output.kind = row["kind"].get<std::string>();
        output.dtype = row.value("dtype", "");
        output.access = row["access"].get<std::string>();
        if (!parseUint32(row["location"], output.location)) {
            error = "pipeline output location must be a non-negative uint32";
            return false;
        }
        if (row.contains("shape")) {
            if (!row["shape"].is_array()) {
                error = "pipeline output shape must be an array";
                return false;
            }
            for (const nlohmann::json &dimension : row["shape"]) {
                uint64_t extent = 0;
                if (!parseUint64(dimension, extent)) {
                    error = "pipeline output shape must contain unsigned extents";
                    return false;
                }
                output.shape.push_back(extent);
            }
        }
        if (output.dtype.empty())
            parseStaticType(row.value("type", ""), output.dtype, output.shape);
        if (output.name.empty() || output.dtype.empty() || output.location == UINT32_MAX) {
            error = "pipeline output metadata is incomplete";
            return false;
        }
        variant.outputs.push_back(std::move(output));
    }
    for (const auto &[stage, artifact] : value["program"].items()) {
        if (!artifact.is_string() || artifact.get_ref<const std::string &>().empty()) {
            error = "pipeline program stage artifact id is invalid";
            return false;
        }
        variant.program.emplace(stage, artifact.get<std::string>());
        if (stage == "compute")
            variant.compute = artifact.get<std::string>();
        else if (stage == "vertex")
            variant.vertex = artifact.get<std::string>();
        else if (stage == "fragment")
            variant.fragment = artifact.get<std::string>();
        else {
            error = "pipeline program contains an unknown shader stage";
            return false;
        }
    }
    if (variant.vertex.empty() != variant.fragment.empty()) {
        error = "graphics pipeline requires both vertex and fragment stages";
        return false;
    }
    return variant.validate(error);
}

bool parseAutodiffManifest(const nlohmann::json &root, AutodiffManifest &manifest, std::string &error) {
    const bool hasTransform = root.contains("program_transform");
    const bool hasProfiles = root.contains("autodiff_profiles");
    if (hasTransform != hasProfiles) {
        error = "pipeline autodiff manifest requires both program_transform and autodiff_profiles";
        return false;
    }
    if (!hasTransform)
        return true;
    const nlohmann::json &transform = root["program_transform"];
    const nlohmann::json &profiles = root["autodiff_profiles"];
    auto validIdentity = [](const nlohmann::json &value) {
        if (!value.is_string())
            return false;
        const std::string &text = value.get_ref<const std::string &>();
        return text.size() == 64 && std::all_of(text.begin(), text.end(), [](unsigned char character) {
                   return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
               });
    };
    auto verifyIdentity = [&](const nlohmann::json &value, const char *label) {
        if (!value.is_object() || !value.contains("identity") || !validIdentity(value["identity"]))
            return false;
        nlohmann::json canonical = value;
        const std::string expected = canonical["identity"].get<std::string>();
        canonical.erase("identity");
        const std::string bytes = canonical.dump(-1, ' ', false, nlohmann::json::error_handler_t::strict);
        if (sha256Hex(bytes.data(), bytes.size()) == expected)
            return true;
        error = std::string(label) + " identity does not match canonical content";
        return false;
    };
    auto parseLaunchSize = [](const nlohmann::json &value, VernonLaunchSize &size) {
        if (!value.is_array() || value.size() != 3)
            return false;
        uint64_t dimensions[3]{};
        for (size_t index = 0; index < 3; ++index)
            if (!parseUint64(value[index], dimensions[index]) || !dimensions[index] || dimensions[index] > UINT32_MAX)
                return false;
        size = {static_cast<uint32_t>(dimensions[0]), static_cast<uint32_t>(dimensions[1]),
                static_cast<uint32_t>(dimensions[2])};
        return true;
    };
    auto parseCanonicalStrings = [](const nlohmann::json &value, std::vector<std::string> &result, bool allowEmpty) {
        if (!value.is_array() || (!allowEmpty && value.empty()))
            return false;
        for (const nlohmann::json &item : value) {
            if (!item.is_string() || item.get_ref<const std::string &>().empty())
                return false;
            result.push_back(item.get<std::string>());
        }
        return std::is_sorted(result.begin(), result.end()) &&
               std::adjacent_find(result.begin(), result.end()) == result.end();
    };
    if (!transform.is_object() ||
        !hasOnlyKeys(transform,
                     {"kind", "wrt", "output_cotangents", "gradient_policy", "accumulation_policy", "tape_policy",
                      "derivative_rules_version", "identity", "rule_set", "rule_set_identity"}) ||
        transform.value("kind", "") != "vjp" || !transform.contains("wrt") || !transform["wrt"].is_array() ||
        !transform.contains("output_cotangents") || !transform["output_cotangents"].is_array() ||
        transform.value("gradient_policy", "") != "f16:f32,f32:f32,f64:f64" ||
        transform.value("accumulation_policy", "") != "fresh" || transform.value("tape_policy", "") != "bounded" ||
        transform.value("derivative_rules_version", 0) != 1 || !verifyIdentity(transform, "program transform")) {
        if (error.empty())
            error = "pipeline program_transform is invalid";
        return false;
    }
    for (const nlohmann::json &path : transform["wrt"]) {
        if (!path.is_string() || path.get_ref<const std::string &>().empty()) {
            error = "pipeline program_transform wrt paths are invalid";
            return false;
        }
        manifest.wrt.push_back(path.get<std::string>());
    }
    if (manifest.wrt.empty() || !std::is_sorted(manifest.wrt.begin(), manifest.wrt.end()) ||
        std::adjacent_find(manifest.wrt.begin(), manifest.wrt.end()) != manifest.wrt.end()) {
        error = "pipeline program_transform wrt paths are not canonical";
        return false;
    }
    for (const nlohmann::json &path : transform["output_cotangents"]) {
        if (!path.is_string() || path.get_ref<const std::string &>().empty()) {
            error = "pipeline program_transform output cotangent paths are invalid";
            return false;
        }
        manifest.outputCotangents.push_back(path.get<std::string>());
    }
    if (manifest.outputCotangents.empty() ||
        !std::is_sorted(manifest.outputCotangents.begin(), manifest.outputCotangents.end()) ||
        std::adjacent_find(manifest.outputCotangents.begin(), manifest.outputCotangents.end()) !=
            manifest.outputCotangents.end()) {
        error = "pipeline program_transform output cotangent paths are not canonical";
        return false;
    }
    manifest.transformIdentity = transform["identity"].get<std::string>();
    if (!profiles.is_object() || !hasOnlyKeys(profiles, {"identity", "variants"}) || !profiles.contains("variants") ||
        !profiles["variants"].is_array() || !verifyIdentity(profiles, "autodiff profiles")) {
        if (error.empty())
            error = "pipeline autodiff_profiles is invalid";
        return false;
    }
    manifest.profilesIdentity = profiles["identity"].get<std::string>();
    for (const nlohmann::json &value : profiles["variants"]) {
        if (!value.is_object() || !hasOnlyKeys(value, {"key", "plan", "programs"}) || !value.contains("key") ||
            !value["key"].is_array() || !value.contains("plan") || !value["plan"].is_object() ||
            !value.contains("programs") || !value["programs"].is_object()) {
            error = "autodiff variant profile is invalid";
            return false;
        }
        AutodiffVariant variant;
        for (const nlohmann::json &feature : value["key"]) {
            if (!feature.is_string()) {
                error = "autodiff variant feature key is invalid";
                return false;
            }
            variant.key.push_back(feature.get<std::string>());
        }
        if (!std::is_sorted(variant.key.begin(), variant.key.end()) ||
            std::adjacent_find(variant.key.begin(), variant.key.end()) != variant.key.end()) {
            error = "autodiff variant feature key is not canonical";
            return false;
        }
        const nlohmann::json &plan = value["plan"];
        if (!hasOnlyKeys(plan, {"transform_identity", "program_graph_identity", "tape_bytes", "derivative_rules",
                                "derivative_rules_version", "launch", "profiles", "identity"}) ||
            plan.value("transform_identity", "") != manifest.transformIdentity ||
            !plan.contains("program_graph_identity") || !validIdentity(plan["program_graph_identity"]) ||
            !plan.contains("tape_bytes") || !parseUint64(plan["tape_bytes"], variant.tapeBytes) ||
            plan.value("derivative_rules_version", 0) != 1 || !plan.contains("launch") || !plan["launch"].is_object() ||
            !plan.contains("profiles") || !plan["profiles"].is_array() || plan["profiles"].size() != 3 ||
            !verifyIdentity(plan, "autodiff plan")) {
            if (error.empty())
                error = "autodiff profile plan is invalid";
            return false;
        }
        variant.planIdentity = plan["identity"].get<std::string>();
        variant.programGraphIdentity = plan["program_graph_identity"].get<std::string>();
        const nlohmann::json &launch = plan["launch"];
        if (!hasOnlyKeys(launch, {"workgroup_size", "accumulation_plans"}) || !launch.contains("workgroup_size") ||
            !parseLaunchSize(launch["workgroup_size"], variant.launch.workgroupSize) ||
            !launch.contains("accumulation_plans") || !launch["accumulation_plans"].is_array() ||
            launch["accumulation_plans"].size() != manifest.wrt.size()) {
            error = "autodiff launch plan is invalid";
            return false;
        }
        std::vector<std::string> accumulationPaths;
        for (const nlohmann::json &resource : launch["accumulation_plans"]) {
            if (!resource.is_object() || !hasOnlyKeys(resource, {"path", "mode", "evidence", "invocation_axes"}) ||
                !resource.contains("path") || !resource["path"].is_string() ||
                resource["path"].get_ref<const std::string &>().empty() || !resource.contains("mode") ||
                !resource["mode"].is_string() || !resource.contains("evidence") ||
                !resource.contains("invocation_axes") || !resource["invocation_axes"].is_array()) {
                error = "autodiff resource accumulation plan is invalid";
                return false;
            }
            AutodiffAccumulationPlan accumulation;
            accumulation.path = resource["path"].get<std::string>();
            const std::string mode = resource.value("mode", "");
            if (mode == "reduce_sum")
                accumulation.operation = AutodiffAccumulationPlan::Operation::ReduceSum;
            else if (mode == "scatter_add")
                accumulation.operation = AutodiffAccumulationPlan::Operation::ScatterAdd;
            else {
                error = "autodiff resource accumulation operation is invalid";
                return false;
            }
            if (!parseCanonicalStrings(resource["evidence"], accumulation.evidence, true)) {
                error = "autodiff resource accumulation evidence is invalid";
                return false;
            }
            constexpr std::array<std::string_view, 5> evidenceKinds{"disjoint_scatter", "injective_global_index",
                                                                    "non_injective_index", "shared_value",
                                                                    "static_index_conflict"};
            if (std::any_of(accumulation.evidence.begin(), accumulation.evidence.end(), [&](const std::string &item) {
                    return std::find(evidenceKinds.begin(), evidenceKinds.end(), item) == evidenceKinds.end();
                })) {
                error = "autodiff resource accumulation evidence is unknown";
                return false;
            }
            for (const nlohmann::json &axisValue : resource["invocation_axes"]) {
                uint32_t axis = 0;
                if (!parseUint32(axisValue, axis) || axis >= 3 ||
                    (!accumulation.invocationAxes.empty() && accumulation.invocationAxes.back() >= axis)) {
                    error = "autodiff resource invocation axes are not canonical";
                    return false;
                }
                accumulation.invocationAxes.push_back(axis);
            }
            const bool disjoint = std::find(accumulation.evidence.begin(), accumulation.evidence.end(),
                                            "disjoint_scatter") != accumulation.evidence.end();
            const bool injective = std::find(accumulation.evidence.begin(), accumulation.evidence.end(),
                                             "injective_global_index") != accumulation.evidence.end();
            const bool shared = std::find(accumulation.evidence.begin(), accumulation.evidence.end(), "shared_value") !=
                                accumulation.evidence.end();
            if ((accumulation.operation == AutodiffAccumulationPlan::Operation::ReduceSum &&
                 (accumulation.evidence != std::vector<std::string>{"shared_value"} ||
                  !accumulation.invocationAxes.empty())) ||
                (accumulation.operation == AutodiffAccumulationPlan::Operation::ScatterAdd &&
                 (shared || injective != !accumulation.invocationAxes.empty())) ||
                (disjoint && (accumulation.operation != AutodiffAccumulationPlan::Operation::ScatterAdd ||
                              accumulation.invocationAxes != std::vector<uint32_t>{0, 1, 2}))) {
                error = "autodiff resource accumulation evidence is inconsistent";
                return false;
            }
            accumulationPaths.push_back(accumulation.path);
            variant.launch.accumulationPlans.push_back(std::move(accumulation));
        }
        if (accumulationPaths != manifest.wrt) {
            error = "autodiff launch resources do not match the transform";
            return false;
        }
        static constexpr std::string_view expectedNames[] = {"primal", "forward_with_tape", "backward"};
        std::vector<std::string> variantGradientPaths;
        for (size_t index = 0; index < 3; ++index) {
            const nlohmann::json &profile = plan["profiles"][index];
            if (!profile.is_object() || !hasOnlyKeys(profile, {"name", "symbol", "inputs", "outputs"}) ||
                profile.value("name", "") != expectedNames[index] || profile.value("symbol", "").empty() ||
                !profile.contains("inputs") || !profile["inputs"].is_array() || !profile.contains("outputs") ||
                !profile["outputs"].is_array()) {
                error = "autodiff profile ABI is invalid";
                return false;
            }
            auto validBindings = [&](const nlohmann::json &bindings) {
                return std::all_of(bindings.begin(), bindings.end(), [](const nlohmann::json &binding) {
                    return binding.is_object() && hasOnlyKeys(binding, {"path", "type", "role"}) &&
                           !binding.value("path", "").empty() && !binding.value("type", "").empty() &&
                           !binding.value("role", "").empty();
                });
            };
            if (!validBindings(profile["inputs"]) || !validBindings(profile["outputs"])) {
                error = "autodiff profile bindings are invalid";
                return false;
            }
            if (index == 2) {
                const nlohmann::json &inputs = profile["inputs"];
                if (inputs.empty() || inputs[0].value("path", "") != "tape" || inputs[0].value("role", "") != "tape") {
                    error = "autodiff backward tape binding is invalid";
                    return false;
                }
                std::vector<std::string> cotangentPaths;
                for (size_t binding = 1; binding < inputs.size(); ++binding) {
                    if (inputs[binding].value("role", "") != "cotangent") {
                        error = "autodiff backward cotangent binding is invalid";
                        return false;
                    }
                    cotangentPaths.push_back(inputs[binding].value("path", ""));
                }
                if (cotangentPaths != manifest.outputCotangents) {
                    error = "autodiff backward cotangents do not match program transform";
                    return false;
                }
                for (const nlohmann::json &binding : profile["outputs"]) {
                    if (binding.value("role", "") != "gradient") {
                        error = "autodiff backward gradient binding is invalid";
                        return false;
                    }
                    variantGradientPaths.push_back(binding.value("path", ""));
                }
                if (variantGradientPaths.empty() ||
                    !std::is_sorted(variantGradientPaths.begin(), variantGradientPaths.end()) ||
                    std::adjacent_find(variantGradientPaths.begin(), variantGradientPaths.end()) !=
                        variantGradientPaths.end()) {
                    error = "autodiff gradient paths are not canonical";
                    return false;
                }
            }
        }
        if (manifest.gradientPaths.empty())
            manifest.gradientPaths = variantGradientPaths;
        else if (manifest.gradientPaths != variantGradientPaths) {
            error = "autodiff variants expose inconsistent gradient paths";
            return false;
        }
        const nlohmann::json &programs = value["programs"];
        if (!hasOnlyKeys(programs, {"primal", "forward_with_tape", "backward"})) {
            error = "autodiff profile program table is invalid";
            return false;
        }
        auto program = [&](const char *name, std::string &output) {
            if (!programs.contains(name) || !programs[name].is_object() || !hasOnlyKeys(programs[name], {"compute"}) ||
                !programs[name].contains("compute") || !programs[name]["compute"].is_string() ||
                programs[name]["compute"].get_ref<const std::string &>().empty())
                return false;
            output = programs[name]["compute"].get<std::string>();
            return true;
        };
        if (!program("primal", variant.primal) || !program("forward_with_tape", variant.forwardWithTape) ||
            !program("backward", variant.backward)) {
            error = "autodiff profile program reference is invalid";
            return false;
        }
        manifest.variants.push_back(std::move(variant));
    }
    if (manifest.variants.empty()) {
        error = "autodiff profile table is empty";
        return false;
    }
    std::sort(manifest.variants.begin(), manifest.variants.end(),
              [](const AutodiffVariant &left, const AutodiffVariant &right) { return left.key < right.key; });
    if (std::adjacent_find(manifest.variants.begin(), manifest.variants.end(),
                           [](const AutodiffVariant &left, const AutodiffVariant &right) {
                               return left.key == right.key;
                           }) != manifest.variants.end()) {
        error = "autodiff profile table contains duplicate variants";
        return false;
    }
    return true;
}

} // namespace vernon::runtime
