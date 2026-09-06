#include "pipeline_manifest.h"
#include "pipeline_metadata.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include <string_view>
#include <tuple>

namespace vernon::runtime {
namespace {} // namespace

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
    if (format == "r8_unorm")
        return VERNON_TEXTURE_R8_UNORM;
    if (format == "r16_float")
        return VERNON_TEXTURE_R16_FLOAT;
    if (format == "r32_float")
        return VERNON_TEXTURE_R32_FLOAT;
    if (format == "rg8_unorm")
        return VERNON_TEXTURE_RG8_UNORM;
    if (format == "rgba8_unorm")
        return VERNON_TEXTURE_RGBA8_UNORM;
    if (format == "rgba16_float")
        return VERNON_TEXTURE_RGBA16_FLOAT;
    if (format == "rgba32_float")
        return VERNON_TEXTURE_RGBA32_FLOAT;
    return std::nullopt;
}

namespace {

bool parseValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error);

template <size_t N> bool hasOnlyKeys(const nlohmann::json &value, const std::string_view (&allowed)[N]) {
    for (auto row = value.begin(); row != value.end(); ++row)
        if (std::find(std::begin(allowed), std::end(allowed), row.key()) == std::end(allowed))
            return false;
    return true;
}

constexpr std::string_view kInterfacePlanKeys[] = {"kind", "profile", "canonical_layout_hash", "root", "frame_offset"};
constexpr std::string_view kTransportNodeKeys[] = {"kind",      "representation", "offset",       "size",
                                                   "alignment", "shape",          "byte_strides", "children"};
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

bool parseValueLayout(const nlohmann::json &value, ValueLayout &layout, std::string &error) {
    layout = {};
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
    std::vector<std::pair<uint64_t, uint64_t>> leafRanges;
    std::set<std::string> leafPaths;
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
        const auto scalarSize = [&]() -> uint64_t {
            if (!parsedDtype)
                return 0;
            switch (*parsedDtype) {
            case VERNON_DATA_BOOL:
            case VERNON_DATA_U8:
                return 1;
            case VERNON_DATA_F16:
                return 2;
            case VERNON_DATA_I32:
            case VERNON_DATA_U32:
            case VERNON_DATA_F32:
                return 4;
            case VERNON_DATA_F64:
                return 8;
            default:
                return 0;
            }
        }();
        if (!scalarSize || byteOffset >= byteSize || scalarCount > UINT64_MAX / scalarSize ||
            scalarCount * scalarSize > byteSize - byteOffset || byteOffset % scalarSize != 0) {
            error = "element_layout leaf has unsupported dtype or byte offset";
            return false;
        }
        const uint64_t byteEnd = byteOffset + scalarCount * scalarSize;
        for (const auto &[existingBegin, existingEnd] : leafRanges)
            if (byteOffset < existingEnd && existingBegin < byteEnd) {
                error = "element_layout semantic leaves overlap";
                return false;
            }
        leafRanges.emplace_back(byteOffset, byteEnd);
        ValueLeaf parsedLeaf;
        parsedLeaf.dtype = dtype;
        parsedLeaf.scalarCount = static_cast<uint32_t>(scalarCount);
        parsedLeaf.byteOffset = static_cast<uint32_t>(byteOffset);
        std::string canonicalPath;
        for (const nlohmann::json &component : leaf["path"]) {
            if (component.is_string()) {
                const std::string field = component.get<std::string>();
                if (field.empty() || field.find('.') != std::string::npos) {
                    error = "element_layout field path components must be non-empty and cannot contain '.'";
                    return false;
                }
                canonicalPath += "f" + std::to_string(field.size()) + ":" + field + ";";
                parsedLeaf.path.push_back({field, 0});
            } else {
                uint64_t index = 0;
                if (!parseUint64(component, index)) {
                    error = "element_layout leaf path components must be field names or unsigned indices";
                    return false;
                }
                canonicalPath += "i" + std::to_string(index) + ";";
                parsedLeaf.path.push_back({std::nullopt, index});
            }
        }
        if (!leafPaths.insert(std::move(canonicalPath)).second) {
            error = "element_layout semantic leaf paths are not unique";
            return false;
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

bool hasOnlyKeys(const nlohmann::json &value, std::initializer_list<std::string_view> allowed) {
    for (auto row = value.begin(); row != value.end(); ++row)
        if (std::find(allowed.begin(), allowed.end(), row.key()) == allowed.end())
            return false;
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

void rebuildVariantLayoutViews(Variant &variant) {
    const auto rebuild = [](Parameter &parameter) {
        if (parameter.valueLayout)
            rebuildValueLayoutPathViews(*parameter.valueLayout);
        rebuildValueLayoutPathViews(parameter.elementLayout);
    };
    for (Parameter &parameter : variant.parameters)
        rebuild(parameter);
    for (Parameter &parameter : variant.internalParameters)
        rebuild(parameter);
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
        const bool imageConstraintsValid =
            parameter.kind == "image" ? validTextureDimension(parameter.dimension) : parameter.dimension.empty();
        if (!imageConstraintsValid) {
            error = "pipeline parameter image constraint invariant failed";
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
        const bool programValue =
            parameter.source == "program_value" && parameter.kind == "tensor" && parameter.systemValue.empty();
        if (parameter.name.empty() || parameter.uses.empty() || (!implicitSampler && !resolution && !programValue)) {
            error = "internal pipeline parameter invariant failed";
            return false;
        }
        for (const ParameterUse &use : parameter.uses) {
            if (parameter.kind == "tensor" && (use.interfaceKind == "uniform" || use.interfaceKind == "value") &&
                !use.interfacePlan) {
                error = "packed Tensor value is missing its typed interface plan";
                return false;
            }
            if (implicitSampler && use.sampledImageBindings.empty()) {
                error = "implicit sampler requires reflected sampled image bindings";
                return false;
            }
            if (resolution && !use.sampledImageBindings.empty()) {
                error = "resolution cannot have sampled image bindings";
                return false;
            }
        }
    }
    using BindingKey = std::tuple<std::string, uint32_t, uint32_t>;
    std::map<BindingKey, std::pair<std::string, std::string>> descriptorOwners;
    std::set<BindingKey> imageBindings;
    std::vector<BindingKey> samplerBindings;
    const auto validateBindings = [&](const Parameter &parameter) {
        for (const ParameterUse &use : parameter.uses) {
            if (program.find(use.stage) == program.end()) {
                error = "pipeline parameter use references a stage outside its program";
                return false;
            }
            const bool descriptorRequired =
                (use.interfaceKind == "resource" && (parameter.kind == "tensor" || parameter.kind == "image")) ||
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
                if (parameter.kind == "image")
                    imageBindings.insert(std::move(key));
            }
            if (parameter.kind == "sampler") {
                if (use.sampledImageBindings.empty()) {
                    error = "sampler parameter has no paired sampled image binding";
                    return false;
                }
                for (const SampledImageBinding &binding : use.sampledImageBindings)
                    samplerBindings.emplace_back(use.stage, binding.descriptorSet, binding.binding);
            } else if (!use.sampledImageBindings.empty()) {
                error = "non-sampler parameter contains sampled image bindings";
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
        if (imageBindings.find(binding) == imageBindings.end()) {
            error = "sampler references an unknown sampled image binding";
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

bool runtimeVersionAtLeast(RuntimeVersion actual, RuntimeVersion required) {
    return actual.major > required.major || (actual.major == required.major && actual.minor >= required.minor);
}

uint32_t glslVersionForApi(RuntimeVersion apiVersion) { return apiVersion.major * 100 + apiVersion.minor * 10; }

} // namespace vernon::runtime
