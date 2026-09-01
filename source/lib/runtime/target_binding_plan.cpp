#include "target_binding_plan.h"

#include "pipeline_metadata.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <string_view>

namespace vernon::runtime::program {
namespace {

bool reject(Diagnostic &diagnostic, std::string code, std::string path, std::string message) {
    diagnostic = {std::move(code), "target_binding", std::move(path), std::move(message)};
    return false;
}

const CompiledEndpointAbi *findCompiledAbi(const StageArtifact &stage, const ReflectedEndpoint &endpoint) {
    const auto found =
        std::find_if(stage.compiledAbi.begin(), stage.compiledAbi.end(), [&](const CompiledEndpointAbi &compiled) {
            return compiled.module == endpoint.module && compiled.index == endpoint.index;
        });
    return found == stage.compiledAbi.end() ? nullptr : &*found;
}

const EndpointBinding *findBinding(const Node &node, const ReflectedEndpoint &endpoint) {
    const auto found = std::find_if(node.bindings.begin(), node.bindings.end(), [&](const EndpointBinding &binding) {
        return binding.module == endpoint.module && binding.interfaceKind == endpoint.interfaceKind &&
               binding.index == endpoint.index;
    });
    return found == node.bindings.end() ? nullptr : &*found;
}

uint32_t boundValueId(const Node &node, const EndpointBinding &binding) {
    if (binding.tag != BindingTag::Resource || binding.access >= node.accesses.size())
        return binding.value;
    const ResourceAccess &access = node.accesses[binding.access];
    return access.kind == AccessKind::Initialize ? access.after
           : access.kind == AccessKind::Write    ? access.before
                                                 : access.value;
}

const EndpointAbiBinding *semantic(const ReflectedEndpoint &endpoint, const std::string &name,
                                   std::optional<uint32_t> axis = std::nullopt) {
    const auto found =
        std::find_if(endpoint.abiBindings.begin(), endpoint.abiBindings.end(), [&](const EndpointAbiBinding &binding) {
            return binding.semantic == name && binding.axis == axis;
        });
    return found == endpoint.abiBindings.end() ? nullptr : &*found;
}

std::optional<VernonDataType> dataType(const std::string &dtype) { return pipelineDataType(dtype); }

uint64_t scalarByteSize(const std::string &dtype) {
    const std::optional<VernonDataType> type = dataType(dtype);
    return type ? dataTypeSize(*type) : 0;
}

std::optional<program::ValueLayout> elementValueLayout(const Value &value, const ReflectedEndpoint &endpoint,
                                                       const EndpointAbiBinding &carrier) {
    if (!value.layout || value.layout->leaves.empty())
        return std::nullopt;
    if (value.layout->scope != "value" || value.layout->layoutHash != endpoint.layoutHash ||
        value.layout->byteSize != carrier.byteSize || value.layout->alignment != carrier.alignment)
        return std::nullopt;
    if (value.shape.empty())
        return *value.layout;
    uint64_t elementCount = 1;
    for (uint64_t extent : value.shape) {
        if (!extent || extent > std::numeric_limits<uint32_t>::max() ||
            elementCount > std::numeric_limits<uint64_t>::max() / extent)
            return std::nullopt;
        elementCount *= extent;
    }
    if (value.layout->byteSize % elementCount)
        return std::nullopt;
    const LayoutLeaf &front = value.layout->leaves.front();
    if (value.layout->leaves.size() == 1 && front.path.empty() && front.shape == value.shape) {
        const uint64_t scalarSize = scalarByteSize(front.dtype);
        if (!scalarSize || front.scalarCount != elementCount)
            return std::nullopt;
        return program::ValueLayout{"element",
                                    endpoint.elementLayoutHash,
                                    value.layout->byteSize / elementCount,
                                    scalarSize,
                                    {LayoutLeaf{{}, front.dtype, 0, 1, {}}}};
    }
    if (endpoint.elementLayoutHash.empty())
        return std::nullopt;
    program::ValueLayout element{
        "element", endpoint.elementLayoutHash, value.layout->byteSize / elementCount, value.layout->alignment, {}};
    for (const LayoutLeaf &valueLeaf : value.layout->leaves) {
        if (valueLeaf.path.size() < value.shape.size())
            return std::nullopt;
        bool firstElement = true;
        for (size_t dimension = 0; dimension < value.shape.size(); ++dimension)
            firstElement &= valueLeaf.path[dimension].index == std::optional<uint32_t>(0);
        if (!firstElement)
            continue;
        LayoutLeaf leaf = valueLeaf;
        leaf.path.erase(leaf.path.begin(), leaf.path.begin() + static_cast<ptrdiff_t>(value.shape.size()));
        if (leaf.byteOffset >= element.byteSize)
            return std::nullopt;
        element.leaves.push_back(std::move(leaf));
    }
    if (element.leaves.empty() || element.leaves.size() > std::numeric_limits<uint64_t>::max() / elementCount ||
        element.leaves.size() * elementCount != value.layout->leaves.size())
        return std::nullopt;
    return element;
}

TransportNode leafTransport(const LayoutLeaf &leaf) {
    const uint64_t scalarSize = scalarByteSize(leaf.dtype);
    TransportNode scalar;
    scalar.kind = TransportNodeKind::Scalar;
    scalar.representation = leaf.dtype;
    scalar.size = scalarSize;
    scalar.alignment = scalarSize;
    if (leaf.shape.empty())
        return scalar;
    TransportNode array;
    array.kind = TransportNodeKind::Array;
    array.size = leaf.scalarCount * scalarSize;
    array.alignment = scalarSize;
    array.shape = leaf.shape;
    array.byteStrides.resize(leaf.shape.size());
    uint64_t stride = scalarSize;
    for (size_t dimension = leaf.shape.size(); dimension-- > 0;) {
        array.byteStrides[dimension] = stride;
        stride *= leaf.shape[dimension];
    }
    array.children.push_back(std::move(scalar));
    return array;
}

TransportNode valueTransport(const program::ValueLayout &layout) {
    if (layout.leaves.size() == 1 && layout.leaves.front().byteOffset == 0)
        return leafTransport(layout.leaves.front());
    TransportNode root;
    root.kind = TransportNodeKind::Product;
    root.size = layout.byteSize;
    root.alignment = layout.alignment;
    for (const LayoutLeaf &leaf : layout.leaves) {
        TransportNode child = leafTransport(leaf);
        child.offset = leaf.byteOffset;
        root.children.push_back(std::move(child));
    }
    return root;
}

TransportNode shapedValueTransport(const program::ValueLayout &elementLayout, const std::vector<uint64_t> &shape,
                                   uint64_t byteSize) {
    TransportNode root;
    root.kind = TransportNodeKind::Array;
    root.size = byteSize;
    root.alignment = elementLayout.alignment;
    root.shape = shape;
    root.byteStrides.resize(shape.size());
    uint64_t stride = elementLayout.byteSize;
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        root.byteStrides[dimension] = stride;
        stride *= shape[dimension];
    }
    root.children.push_back(valueTransport(elementLayout));
    return root;
}

InterfacePlan computeValuePlan(const TransportNode &root, const std::string &layoutHash, VernonRuntimeBackend backend) {
    InterfacePlan plan;
    plan.kind = backend == VERNON_RUNTIME_CPU ? InterfacePlanKind::CpuCall : InterfacePlanKind::ByteTransport;
    plan.profile = physicalValueProfileName(backend, "storage_buffer");
    plan.canonicalLayoutHash = layoutHash;
    plan.root = root;
    return plan;
}

bool vertexFormat(std::string_view format, std::string &dtype, uint32_t &components) {
    const size_t separator = format.rfind('_');
    if (separator == std::string_view::npos)
        return false;
    const std::string_view channels = format.substr(0, separator);
    const std::string_view scalar = format.substr(separator + 1);
    components = channels == "r32"      ? 1
                 : channels == "rg32"   ? 2
                 : channels == "rgb32"  ? 3
                 : channels == "rgba32" ? 4
                                        : 0;
    dtype = scalar == "float" ? "f32" : scalar == "uint" ? "u32" : scalar == "sint" ? "i32" : "";
    return components && !dtype.empty();
}

std::string valueName(const ResolvedProgram &program, uint32_t valueId) {
    for (const SignatureBinding &binding : program.program.signature.inputs)
        if (binding.value == valueId)
            return binding.path;
    return valueId < program.program.values.size() ? program.program.values[valueId].name : std::string{};
}

bool alignFrame(uint64_t &offset, uint64_t alignment) {
    if (!alignment || (alignment & (alignment - 1)))
        return false;
    const uint64_t mask = alignment - 1;
    if (offset > std::numeric_limits<uint64_t>::max() - mask)
        return false;
    offset = (offset + mask) & ~mask;
    return true;
}

bool assignCpuPhysical(uint64_t &cpuFrameOffset, const CompiledEndpointAbi *compiled, uint64_t alignment, uint64_t size,
                       PhysicalArgumentLayout &physical, InterfacePlan *plan, Diagnostic &diagnostic,
                       const char *what) {
    uint64_t offset = cpuFrameOffset;
    if (compiled && compiled->interfacePlan)
        offset = compiled->interfacePlan->frameOffset;
    else if (compiled && compiled->packedFrameOffset)
        offset = *compiled->packedFrameOffset;
    else if (!alignFrame(offset, alignment))
        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                      std::string(what) + " frame offset overflows");
    if (size && offset > std::numeric_limits<uint64_t>::max() - size)
        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                      std::string(what) + " frame offset overflows");
    physical = {static_cast<size_t>(offset), static_cast<size_t>(size), static_cast<size_t>(alignment)};
    if (plan)
        plan->frameOffset = offset;
    cpuFrameOffset = std::max(cpuFrameOffset, offset + size);
    return true;
}

} // namespace

vernon::runtime::ValueLayout materializeValueLayout(const ValueLayout &layout, std::string logicalType) {
    vernon::runtime::ValueLayout result;
    result.logicalType = std::move(logicalType);
    result.layoutHash = layout.layoutHash;
    result.byteSize = static_cast<uint32_t>(layout.byteSize);
    result.alignment = static_cast<uint32_t>(layout.alignment);
    for (const LayoutLeaf &leaf : layout.leaves) {
        ValueLeaf converted(leaf.dtype, static_cast<uint32_t>(leaf.scalarCount),
                            static_cast<uint32_t>(leaf.byteOffset));
        converted.shape = leaf.shape;
        for (const LayoutPathComponent &component : leaf.path) {
            ValuePathComponent path;
            if (!component.field.empty())
                path.field = component.field;
            if (component.index)
                path.index = *component.index;
            converted.path.push_back(std::move(path));
        }
        if (const std::optional<VernonDataType> dtype = pipelineDataType(converted.dtype))
            result.abiLeaves.push_back({static_cast<uint32_t>(*dtype), converted.scalarCount, converted.byteOffset});
        result.leaves.push_back(std::move(converted));
    }
    if (result.logicalType.empty() && result.leaves.size() == 1 && result.leaves.front().path.empty() &&
        result.leaves.front().shape.empty() && result.leaves.front().scalarCount == 1)
        result.logicalType = result.leaves.front().dtype;
    rebuildValueLayoutPathViews(result);
    return result;
}

bool buildTargetBindingPlan(const ResolvedProgram &program, const Node &node, const ResolvedStage &stage,
                            VernonRuntimeBackend backend, TargetBindingPlan &plan, Diagnostic &diagnostic) {
    diagnostic = {};
    plan = {};
    plan.backend = backend;
    plan.operation = node.operation;
    plan.topology = stage.stage.graphicsTopology;
    std::copy(std::begin(stage.stage.workgroupSize), std::end(stage.stage.workgroupSize), plan.workgroupSize);
    plan.dispatch.requiresUnitWorkgroup = stage.stage.requiresUnitWorkgroup;
    for (uint32_t axis : stage.stage.unitGridAxes)
        if (axis < 3)
            plan.dispatch.unitGridAxes[axis] = true;
    for (const CodeModule &module : stage.stage.modules)
        plan.modules.push_back({module.role, module.entryPoint, module.format});

    uint64_t cpuFrameOffset = 0;
    uint32_t maximumSlot = 0;
    bool hasSlots = false;
    const bool compute = node.operation != "graphics";
    const std::string entry = stage.stage.modules.empty() ? std::string() : stage.stage.modules.front().entryPoint;

    for (const ReflectedEndpoint &endpoint : stage.stage.endpoints) {
        TargetBinding target;
        target.endpoint = {endpoint.module, endpoint.interfaceKind, endpoint.index,
                           UINT32_MAX,      endpoint.access,        UINT32_MAX};
        target.role = endpoint.role;
        target.access = endpoint.access;
        target.dimension = endpoint.imageDimension;
        target.imageFormat = endpoint.imageFormat;
        target.viewShape = endpoint.viewShape;
        target.writeFootprintKind = endpoint.writeFootprintKind;
        target.writeFootprintIndices = endpoint.writeFootprintIndices;

        if (endpoint.tag == "system")
            continue;
        if (endpoint.interfaceKind == "system_value") {
            const EndpointAbiBinding *valueSlot = semantic(endpoint, "value");
            if (!valueSlot || !valueSlot->byteSize || !valueSlot->alignment)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "system value has an invalid portable carrier");
            target.source = SourceRepresentation::SystemValue;
            target.name = endpoint.builtin;
            target.kind = "tensor";
            target.reflectedKind = "builtin";
            target.builtin = endpoint.builtin;
            target.endpoint.portableSlot = valueSlot->slot;
            if (!compute) {
                const CompiledEndpointAbi *compiled = findCompiledAbi(stage.stage, endpoint);
                if (endpoint.builtin != "resolution" || !compiled || !compiled->interfacePlan)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "graphics resolution is missing the compiled physical plan");
                if (valueSlot->byteSize != 8)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "graphics resolution carrier must be two f32 scalars");
                target.carrier = compiled->valueTransport == "uniform_buffer"   ? TargetCarrier::UniformBuffer
                                 : compiled->valueTransport == "storage_buffer" ? TargetCarrier::StorageBuffer
                                                                                : TargetCarrier::InlineValue;
                target.shape = {2};
                LayoutLeaf scalar{{}, "f32", 0, 1, {}};
                target.elementLayout =
                    materializeValueLayout(ValueLayout{"element", endpoint.layoutHash, 4, 4, {scalar}}, "f32");
                LayoutLeaf vector{{}, "f32", 0, 2, {2}};
                target.wholeValueLayout = materializeValueLayout(
                    ValueLayout{"value", endpoint.layoutHash, 8, valueSlot->alignment, {vector}}, "tensor<2xf32>");
                target.native = {compiled->descriptorSet == UINT32_MAX ? 0 : compiled->descriptorSet,
                                 compiled->binding == UINT32_MAX ? valueSlot->slot : compiled->binding, UINT32_MAX};
                target.physical = {0, static_cast<size_t>(valueSlot->byteSize),
                                   static_cast<size_t>(valueSlot->alignment)};
                target.transport =
                    TargetPhysicalTransport{*target.wholeValueLayout, *compiled->interfacePlan, target.native};
                plan.bindings.push_back(std::move(target));
                continue;
            }
            if (valueSlot->byteSize % sizeof(uint32_t))
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "system value has an invalid portable carrier");
            const uint64_t scalarCount = valueSlot->byteSize / sizeof(uint32_t);
            target.carrier = backend == VERNON_RUNTIME_CPU ? TargetCarrier::InlineValue : TargetCarrier::StorageBuffer;
            target.shape = scalarCount == 1 ? std::vector<uint64_t>{} : std::vector<uint64_t>{scalarCount};
            LayoutLeaf leaf{{}, "u32", 0, scalarCount, target.shape};
            program::ValueLayout layout{
                "value", endpoint.layoutHash, valueSlot->byteSize, valueSlot->alignment, {leaf}};
            target.wholeValueLayout = materializeValueLayout(layout, endpoint.builtin);
            target.elementLayout = *target.wholeValueLayout;
            InterfacePlan physical = computeValuePlan(leafTransport(leaf), endpoint.layoutHash, backend);
            if (backend == VERNON_RUNTIME_CPU) {
                if (!assignCpuPhysical(cpuFrameOffset, findCompiledAbi(stage.stage, endpoint), valueSlot->alignment,
                                       valueSlot->byteSize, target.physical, &physical, diagnostic, "system value"))
                    return false;
            } else {
                target.physical = {0, static_cast<size_t>(valueSlot->byteSize),
                                   static_cast<size_t>(valueSlot->alignment)};
            }
            target.transport = TargetPhysicalTransport{
                *target.wholeValueLayout, std::move(physical), {0, valueSlot->slot, valueSlot->slot}};
            plan.bindings.push_back(std::move(target));
            continue;
        }

        const EndpointBinding *binding = findBinding(node, endpoint);
        if (!binding)
            return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                          "node does not bind a reflected endpoint");
        const uint32_t valueId = boundValueId(node, *binding);
        if (valueId >= program.program.values.size())
            return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                          "endpoint binding references an unknown Program value");
        const Value &value = program.program.values[valueId];
        target.endpoint.value = valueId;
        target.endpoint.leaf = binding->leaf;
        target.name = valueName(program, valueId);
        if (binding->leaf) {
            if (!value.layout || *binding->leaf >= value.layout->leaves.size())
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                              "endpoint binding projects an unknown Program value leaf");
            for (const LayoutPathComponent &component : value.layout->leaves[*binding->leaf].path) {
                target.name.push_back('.');
                target.name += component.field.empty() ? std::to_string(component.index.value_or(0)) : component.field;
            }
        }
        if (endpoint.role == "vertex" || !endpoint.viewShape.empty()) {
            target.shape.clear();
            target.shape.reserve(endpoint.viewShape.size());
            for (int64_t extent : endpoint.viewShape)
                target.shape.push_back(extent < 0 ? 0 : static_cast<uint64_t>(extent));
        } else {
            target.shape = value.shape;
        }
        target.kind = endpoint.role == "sampler" ? "sampler" : !endpoint.imageDimension.empty() ? "image" : "tensor";
        const CompiledEndpointAbi *compiled = findCompiledAbi(stage.stage, endpoint);

        if (target.kind == "sampler" || target.kind == "image") {
            const EndpointAbiBinding *resource = semantic(endpoint, target.kind == "sampler" ? "sampler" : "resource");
            if (!resource)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "opaque resource endpoint lacks its portable resource carrier");
            target.source = SourceRepresentation::ResourceHandle;
            target.carrier = target.kind == "sampler" ? TargetCarrier::Sampler : TargetCarrier::Image;
            target.reflectedKind = target.kind;
            target.endpoint.portableSlot = resource->slot;
            target.native = {0, resource->slot, resource->slot};
            if (compiled) {
                if (compiled->binding != UINT32_MAX)
                    target.native = {compiled->descriptorSet == UINT32_MAX ? 0 : compiled->descriptorSet,
                                     compiled->binding, UINT32_MAX};
                target.sampledImageBindings = compiled->sampledImageBindings;
            }
            maximumSlot = std::max(maximumSlot, resource->slot);
            hasSlots = true;
            if (backend == VERNON_RUNTIME_CPU) {
                if (!assignCpuPhysical(cpuFrameOffset, compiled, alignof(uintptr_t), sizeof(uintptr_t), target.physical,
                                       nullptr, diagnostic, "resource"))
                    return false;
            }
        } else {
            if (!value.layout)
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/values/" + std::to_string(valueId),
                              "typed target binding requires a canonical value layout");
            if (endpoint.tag == "value") {
                const EndpointAbiBinding *slot = semantic(endpoint, "value");
                if (!slot)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "value endpoint lacks its portable value carrier");
                if (value.layout->layoutHash != endpoint.layoutHash || value.layout->byteSize != slot->byteSize ||
                    value.layout->alignment != slot->alignment)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "value endpoint whole-value ABI disagrees with its Program Value");
                target.wholeValueLayout = materializeValueLayout(*value.layout, value.type);
                if (!compute) {
                    if (!compiled || !compiled->interfacePlan)
                        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                      "graphics value endpoint is missing the compiled physical plan");
                    target.source = SourceRepresentation::WholeValueBytes;
                    target.reflectedKind = value.shape.empty() ? "scalar" : "tensor_value";
                    target.elementLayout = *target.wholeValueLayout;
                    target.carrier = compiled->valueTransport == "uniform_buffer"   ? TargetCarrier::UniformBuffer
                                     : compiled->valueTransport == "storage_buffer" ? TargetCarrier::StorageBuffer
                                                                                    : TargetCarrier::InlineValue;
                    target.native = {compiled->descriptorSet == UINT32_MAX ? 0 : compiled->descriptorSet,
                                     compiled->binding == UINT32_MAX ? slot->slot : compiled->binding, UINT32_MAX};
                    target.transport =
                        TargetPhysicalTransport{*target.wholeValueLayout, *compiled->interfacePlan, target.native};
                } else {
                    std::optional<program::ValueLayout> element = elementValueLayout(value, endpoint, *slot);
                    if (!element)
                        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                      "value endpoint whole-value ABI disagrees with its Program Value");
                    target.source = value.shape.empty() ? SourceRepresentation::WholeValueBytes
                                                        : SourceRepresentation::ElementStream;
                    target.reflectedKind = value.shape.empty() ? "scalar" : "tensor_value";
                    target.elementLayout = materializeValueLayout(*element, value.type);
                    target.carrier = TargetCarrier::StorageBuffer;
                    InterfacePlan physical;
                    if (compiled && compiled->interfacePlan &&
                        compiled->interfacePlan->kind != InterfacePlanKind::NativeUniform)
                        physical = *compiled->interfacePlan;
                    else {
                        TransportNode root = target.source == SourceRepresentation::WholeValueBytes
                                                 ? valueTransport(*value.layout)
                                                 : shapedValueTransport(*element, value.shape, value.layout->byteSize);
                        physical = computeValuePlan(root, value.layout->layoutHash, backend);
                    }
                    if (!physical.root)
                        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                      "compute value endpoint is missing its physical transport tree");
                    if (backend == VERNON_RUNTIME_CPU) {
                        if (!assignCpuPhysical(cpuFrameOffset, compiled, value.layout->alignment,
                                               value.layout->byteSize, target.physical, &physical, diagnostic, "value"))
                            return false;
                    } else {
                        target.physical = {0, static_cast<size_t>(physical.root->size),
                                           static_cast<size_t>(physical.root->alignment)};
                    }
                    target.transport = TargetPhysicalTransport{
                        *target.wholeValueLayout, std::move(physical), {0, slot->slot, slot->slot}};
                }
                target.endpoint.portableSlot = slot->slot;
                maximumSlot = std::max(maximumSlot, slot->slot);
                hasSlots = true;
            } else {
                std::vector<std::pair<uint32_t, const EndpointAbiBinding *>> indexedStorageBindings;
                for (const EndpointAbiBinding &binding : endpoint.abiBindings)
                    if (binding.semantic == "storage_leaf" && binding.axis)
                        indexedStorageBindings.emplace_back(*binding.axis, &binding);
                std::sort(indexedStorageBindings.begin(), indexedStorageBindings.end(),
                          [](const auto &left, const auto &right) { return left.first < right.first; });
                if (target.endpoint.leaf && (indexedStorageBindings.size() != 1 ||
                                             indexedStorageBindings.front().first != *target.endpoint.leaf))
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "TensorView endpoint leaf binding disagrees with its storage projection");
                std::vector<const EndpointAbiBinding *> storageBindings;
                for (const auto &[leaf, binding] : indexedStorageBindings)
                    storageBindings.push_back(binding);
                if (storageBindings.empty())
                    if (const EndpointAbiBinding *resource = semantic(endpoint, "resource"))
                        storageBindings.push_back(resource);
                program::ValueLayout elementLayout = *value.layout;
                const bool vertexBuffer = endpoint.role == "vertex";
                if (!endpoint.viewDescriptor && !vertexBuffer) {
                    uint64_t elementCount = 1;
                    for (int64_t extent : endpoint.viewShape)
                        elementCount *= static_cast<uint64_t>(extent);
                    elementLayout.scope = "element";
                    elementLayout.layoutHash = endpoint.layoutHash;
                    if (!elementCount || value.layout->byteSize % elementCount)
                        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                      "TensorView whole-value size is not an element multiple");
                    elementLayout.byteSize = value.layout->byteSize / elementCount;
                    elementLayout.leaves.clear();
                    for (const LayoutLeaf &leaf : value.layout->leaves) {
                        if (leaf.byteOffset >= elementLayout.byteSize)
                            break;
                        LayoutLeaf elementLeaf = leaf;
                        if (elementLeaf.path.size() >= endpoint.viewShape.size())
                            elementLeaf.path.erase(elementLeaf.path.begin(),
                                                   elementLeaf.path.begin() +
                                                       static_cast<ptrdiff_t>(endpoint.viewShape.size()));
                        elementLayout.leaves.push_back(std::move(elementLeaf));
                    }
                }
                if (!indexedStorageBindings.empty()) {
                    std::vector<LayoutLeaf> projectedLeaves;
                    projectedLeaves.reserve(indexedStorageBindings.size());
                    for (const auto &[leaf, binding] : indexedStorageBindings) {
                        (void)binding;
                        if (leaf >= elementLayout.leaves.size())
                            return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                          "TensorView endpoint projects an unknown storage leaf");
                        projectedLeaves.push_back(elementLayout.leaves[leaf]);
                    }
                    elementLayout.leaves = std::move(projectedLeaves);
                }
                const EndpointAbiBinding *offset = semantic(endpoint, "byte_offset");
                if ((endpoint.viewDescriptor && !offset) ||
                    (vertexBuffer ? storageBindings.size() != 1
                                  : storageBindings.size() != elementLayout.leaves.size()) ||
                    elementLayout.leaves.empty())
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "TensorView endpoint lacks required storage-leaf or offset carriers");
                target.source = endpoint.viewDescriptor ? SourceRepresentation::TensorViewDescriptor
                                                        : SourceRepresentation::ElementStream;
                target.carrier = endpoint.role == "vertex" ? TargetCarrier::VertexBuffer : TargetCarrier::StorageBuffer;
                target.reflectedKind = "tensor";
                target.wholeValueLayout = materializeValueLayout(*value.layout, value.type);
                target.elementLayout = compiled && compiled->elementLayout
                                           ? *compiled->elementLayout
                                           : materializeValueLayout(elementLayout, value.type);
                if (!vertexBuffer && target.elementLayout.leaves.size() != storageBindings.size())
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "TensorView endpoint physical layout does not match its storage carriers");
                target.endpoint.portableSlot = storageBindings.front()->slot;
                target.native = {0, storageBindings.front()->slot, storageBindings.front()->slot};
                if (compiled && compiled->binding != UINT32_MAX)
                    target.native = {compiled->descriptorSet == UINT32_MAX ? 0 : compiled->descriptorSet,
                                     compiled->binding, target.native.location};
                for (size_t leaf = 0; leaf < target.elementLayout.leaves.size(); ++leaf) {
                    const vernon::runtime::ValueLeaf &layoutLeaf = target.elementLayout.leaves[leaf];
                    const uint64_t elementSize = scalarByteSize(layoutLeaf.dtype);
                    if (!elementSize)
                        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                      "TensorView leaf has an unsupported dtype");
                    target.storageLeaves.push_back({static_cast<size_t>(elementSize),
                                                    static_cast<size_t>(layoutLeaf.byteOffset),
                                                    storageBindings[vertexBuffer ? 0 : leaf]->slot});
                    maximumSlot = std::max(maximumSlot, storageBindings[vertexBuffer ? 0 : leaf]->slot);
                    hasSlots = true;
                }
                if (endpoint.viewDescriptor) {
                    TensorViewDescriptorUse descriptor;
                    descriptor.rank = endpoint.viewRank;
                    descriptor.offsetBinding = offset->slot;
                    for (uint32_t axis = 0; axis < endpoint.viewRank; ++axis) {
                        const EndpointAbiBinding *extent = semantic(endpoint, "extent", axis);
                        const EndpointAbiBinding *stride = semantic(endpoint, "byte_stride", axis);
                        if (!extent || !stride)
                            return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                          "TensorView endpoint lacks an extent or stride carrier");
                        descriptor.extentBindings.push_back(extent->slot);
                        descriptor.strideBindings.push_back(stride->slot);
                        maximumSlot = std::max({maximumSlot, extent->slot, stride->slot});
                    }
                    maximumSlot = std::max(maximumSlot, offset->slot);
                    hasSlots = true;
                    target.tensorViewDescriptor = std::move(descriptor);
                    if (backend == VERNON_RUNTIME_CPU) {
                        const uint64_t descriptorSize =
                            static_cast<uint64_t>(2 + 2 * endpoint.viewRank) * sizeof(uintptr_t);
                        if (!assignCpuPhysical(cpuFrameOffset, compiled, alignof(uintptr_t), descriptorSize,
                                               target.physical, nullptr, diagnostic, "TensorView descriptor"))
                            return false;
                    }
                } else if (backend == VERNON_RUNTIME_CPU) {
                    InterfacePlan physical =
                        computeValuePlan(valueTransport(*value.layout), value.layout->layoutHash, backend);
                    if (!assignCpuPhysical(cpuFrameOffset, compiled, value.layout->alignment, value.layout->byteSize,
                                           target.physical, &physical, diagnostic, "TensorView value"))
                        return false;
                    target.transport =
                        TargetPhysicalTransport{*target.wholeValueLayout,
                                                std::move(physical),
                                                {0, storageBindings.front()->slot, storageBindings.front()->slot}};
                }
            }
        }

        if (endpoint.role == "vertex") {
            const auto first =
                std::find_if(stage.stage.vertexInputs.begin(), stage.stage.vertexInputs.end(),
                             [&](const GraphicsVertexInput &input) { return input.endpointIndex == endpoint.index; });
            if (first == stage.stage.vertexInputs.end())
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/graphics/vertex_inputs",
                              "vertex endpoint has no physical input");
            std::string dtype;
            uint32_t components{};
            if (!vertexFormat(first->format, dtype, components))
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/graphics/vertex_inputs",
                              "vertex input format is unsupported");
            (void)components;
            target.divisor = first->divisor;
            target.native.location = first->location;
            for (const GraphicsVertexInput &input : stage.stage.vertexInputs) {
                if (input.endpointIndex != endpoint.index)
                    continue;
                std::string leafDtype;
                uint32_t leafComponents{};
                if (!vertexFormat(input.format, leafDtype, leafComponents))
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/graphics/vertex_inputs",
                                  "vertex input leaves use incompatible formats");
                target.attributeLeaves.push_back({input.location - first->location, leafDtype, leafComponents,
                                                  static_cast<uint32_t>(input.byteOffset)});
            }
        }
        plan.bindings.push_back(std::move(target));
    }

    if (backend == VERNON_RUNTIME_CPU) {
        for (const CompiledEndpointAbi &compiled : stage.stage.compiledAbi) {
            if (compiled.builtin != VERNON_AD_TAPE_ALLOCATOR_BUILTIN &&
                compiled.builtin != VERNON_AD_TAPE_ROOT_REGION_BUILTIN)
                continue;
            if (std::any_of(plan.bindings.begin(), plan.bindings.end(),
                            [&](const TargetBinding &bound) { return bound.builtin == compiled.builtin; }))
                continue;
            if (!compiled.interfacePlan)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/implementation/endpoints",
                              "tape packed ABI is missing its compiled physical plan");
            TargetBinding target;
            target.source = SourceRepresentation::SystemValue;
            target.carrier = TargetCarrier::InlineValue;
            target.name = compiled.builtin;
            target.kind = "tensor";
            target.reflectedKind = "builtin";
            target.builtin = compiled.builtin;
            target.access = "read";
            target.endpoint = {"compute", "system_value", compiled.index, UINT32_MAX, "read", UINT32_MAX};
            if (!assignCpuPhysical(cpuFrameOffset, &compiled, alignof(uintptr_t), sizeof(uintptr_t), target.physical,
                                   nullptr, diagnostic, "tape packed field"))
                return false;
            plan.bindings.push_back(std::move(target));
        }
    }

    for (const GraphicsFragmentOutput &output : stage.stage.fragmentOutputs)
        plan.outputs.push_back({output.location, output.type});
    if (backend == VERNON_RUNTIME_CPU)
        plan.packedArgumentsSize = cpuFrameOffset;
    if (backend == VERNON_RUNTIME_METAL && hasSlots) {
        std::set<uint32_t> imageSlots;
        for (const TargetBinding &binding : plan.bindings)
            if (binding.kind == "image")
                imageSlots.insert(binding.endpoint.portableSlot);
        for (uint32_t slot = 0; slot <= maximumSlot; ++slot) {
            NativeResourceSlot native;
            native.entry = entry;
            native.stage = "compute";
            native.kind = imageSlots.count(slot) ? "storage_image" : "storage_buffer";
            native.name = entry + "_arg_" + std::to_string(slot);
            native.set = 0;
            native.binding = slot;
            native.argumentBufferIndex = 0;
            native.memberId = slot;
            native.count = 1;
            plan.nativeSlots.push_back(std::move(native));
        }
    }
    return true;
}

bool materializeTargetBindingPlan(const TargetBindingPlan &plan, Variant &variant, ReflectedEntry &reflection,
                                  Diagnostic &diagnostic) {
    diagnostic = {};
    variant = {};
    reflection = {};
    std::copy(std::begin(plan.workgroupSize), std::end(plan.workgroupSize), reflection.workgroup);
    reflection.dispatchContract = plan.dispatch;
    if (plan.backend == VERNON_RUNTIME_CPU)
        reflection.packedArguments = PackedArgumentsLayout{static_cast<size_t>(plan.packedArgumentsSize)};
    if (!plan.modules.empty()) {
        variant.compute = plan.modules.front().entryPoint;
        variant.program.emplace("compute", variant.compute);
    }
    for (const TargetBinding &binding : plan.bindings) {
        ReflectedArgument argument;
        argument.sourceName = binding.name;
        argument.index = binding.endpoint.index;
        argument.kind = binding.reflectedKind.empty() ? binding.kind : binding.reflectedKind;
        if (argument.kind == "tensor_value")
            argument.kind = "scalar";
        argument.builtin = binding.builtin;
        argument.physical = binding.physical;
        if (!argument.physical.alignment)
            argument.physical.alignment = 1;
        argument.descriptorSet = 0;
        argument.binding = binding.endpoint.portableSlot;
        argument.storageLeaves = binding.storageLeaves;
        argument.tensorElementSize = binding.elementLayout.byteSize;
        argument.sourceShape = binding.viewShape;
        if (!binding.shape.empty()) {
            size_t elements = 1;
            for (uint64_t extent : binding.shape)
                elements *= static_cast<size_t>(extent);
            argument.tensorElements = elements;
            argument.tensorBytes = elements * argument.tensorElementSize;
        }
        if (binding.tensorViewDescriptor) {
            TensorViewDescriptorLayout descriptor;
            descriptor.rank = binding.tensorViewDescriptor->rank;
            descriptor.offsetBinding = binding.tensorViewDescriptor->offsetBinding;
            descriptor.extentBindings = binding.tensorViewDescriptor->extentBindings;
            descriptor.strideBindings = binding.tensorViewDescriptor->strideBindings;
            argument.tensorViewDescriptor = std::move(descriptor);
        }
        reflection.arguments.push_back(std::move(argument));
        if (binding.source == SourceRepresentation::SystemValue)
            continue;

        vernon::runtime::Parameter parameter;
        parameter.slot = static_cast<uint32_t>(variant.parameters.size());
        parameter.name = binding.name;
        parameter.source = "direct";
        parameter.kind = binding.kind;
        parameter.access = binding.access.empty() ? "read" : binding.access;
        parameter.elementLayout = binding.elementLayout;
        if (binding.source == SourceRepresentation::WholeValueBytes)
            parameter.valueLayout = binding.wholeValueLayout;
        parameter.dimension = binding.dimension;
        parameter.bindingRole = binding.role;
        parameter.exactStorageFormat = binding.imageFormat;
        parameter.shape = binding.shape;
        ParameterUse use;
        use.stage = "compute";
        use.index = binding.endpoint.index;
        use.interfaceKind = binding.kind == "image" || binding.kind == "sampler" ? "resource"
                            : binding.source == SourceRepresentation::WholeValueBytes ||
                                    (binding.source == SourceRepresentation::ElementStream && binding.transport)
                                ? "value"
                                : "storage";
        use.dtype = binding.elementLayout.logicalType.empty() && !binding.elementLayout.leaves.empty()
                        ? binding.elementLayout.leaves.front().dtype
                        : binding.elementLayout.logicalType;
        use.shape = binding.shape;
        use.transport = plan.backend == VERNON_RUNTIME_CPU ? "host_value" : "storage_buffer";
        use.valueLayout = binding.wholeValueLayout;
        use.interfacePlan = binding.transport ? std::optional<InterfacePlan>(binding.transport->targetAbi)
                                              : std::optional<InterfacePlan>{};
        use.tensorViewDescriptor = binding.tensorViewDescriptor;
        use.descriptorSet = 0;
        use.binding = binding.endpoint.portableSlot;
        parameter.uses.push_back(std::move(use));
        rebuildValueLayoutPathViews(parameter.elementLayout);
        if (parameter.valueLayout)
            rebuildValueLayoutPathViews(*parameter.valueLayout);
        variant.parameters.push_back(std::move(parameter));
    }

    const auto argumentIndex = [&](const std::string &owner) -> std::optional<uint32_t> {
        for (size_t index = 0; index < reflection.arguments.size(); ++index)
            if (reflection.arguments[index].sourceName == owner)
                return static_cast<uint32_t>(index);
        return std::nullopt;
    };
    for (const TargetBinding &binding : plan.bindings) {
        const bool storageTensor =
            binding.kind == "tensor" && (binding.source == SourceRepresentation::TensorViewDescriptor ||
                                         (binding.source == SourceRepresentation::ElementStream && !binding.transport));
        if (!storageTensor && binding.writeFootprintKind.empty())
            continue;
        if (binding.access == "read" || binding.access == "read_write") {
            const auto index = argumentIndex(binding.name);
            if (!index)
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/effects",
                              "read footprint names an unknown argument");
            reflection.readFootprints.push_back({*index, binding.name, true, {}});
        }
        if (!binding.writeFootprintKind.empty()) {
            const auto index = argumentIndex(binding.name);
            if (!index)
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/effects",
                              "write footprint names an unknown argument");
            TensorViewWriteFootprint footprint;
            footprint.argument = *index;
            footprint.owner = binding.name;
            footprint.wholeView = binding.writeFootprintKind != "element";
            footprint.indices.assign(binding.writeFootprintIndices.begin(), binding.writeFootprintIndices.end());
            reflection.writeFootprints.push_back(std::move(footprint));
        }
    }
    return true;
}

bool buildResolvedExecutablePlan(const ResolvedProgram &program, VernonRuntimeBackend backend,
                                 ResolvedExecutablePlan &plan, Diagnostic &diagnostic) {
    plan = {};
    for (const Graph &graph : program.program.graphs) {
        for (const Node &node : graph.nodes) {
            const auto resolvedStage = program.stages.find(node.stage);
            if (resolvedStage == program.stages.end())
                return reject(diagnostic, "PROGRAM_STAGE_BINDING", "/graphs/" + graph.name + "/nodes/" + node.name,
                              "node has no resolved StageArtifact");
            const char *expected = backend == VERNON_RUNTIME_CPU         ? "cpu"
                                   : backend == VERNON_RUNTIME_VULKAN    ? "vulkan"
                                   : backend == VERNON_RUNTIME_METAL     ? "metal"
                                   : backend == VERNON_RUNTIME_CUDA      ? "cuda"
                                   : backend == VERNON_RUNTIME_DIRECTX12 ? "directx"
                                   : backend == VERNON_RUNTIME_OPENGL    ? "opengl"
                                   : backend == VERNON_RUNTIME_OPENGL_ES ? "opengles"
                                                                         : "";
            if (resolvedStage->second.stage.backend != expected)
                return reject(diagnostic, "PROGRAM_ARTIFACT_TARGET", "/artifact_system/target",
                              "ArtifactSystem target does not match Runtime backend");
            ResolvedExecutableNode resolved;
            resolved.node = &node;
            resolved.stage = &resolvedStage->second;
            if (!buildTargetBindingPlan(program, node, resolvedStage->second, backend, resolved.plan, diagnostic))
                return false;
            plan.nodes.push_back(std::move(resolved));
        }
    }
    if (plan.nodes.empty())
        return reject(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "/graphs",
                      "canonical Program has no executable nodes");
    return true;
}

bool materializeGraphicsTargetBindingPlan(const TargetBindingPlan &plan, Variant &variant, Diagnostic &diagnostic) {
    diagnostic = {};
    variant = {};
    std::map<uint32_t, size_t> parameterByValue;
    for (const TargetBinding &binding : plan.bindings) {
        if (binding.source == SourceRepresentation::SystemValue) {
            vernon::runtime::Parameter parameter;
            parameter.name = binding.name.empty() ? "__vernon_system_value_" + binding.endpoint.module + "_" +
                                                        std::to_string(binding.endpoint.index)
                                                  : binding.name;
            parameter.kind = "tensor";
            parameter.source = "system_value";
            parameter.systemValue = "resolution";
            parameter.elementLayout = binding.elementLayout;
            parameter.valueLayout = binding.wholeValueLayout;
            parameter.access = binding.access.empty() ? "read" : binding.access;
            parameter.shape = binding.shape;
            ParameterUse use;
            use.stage = binding.endpoint.module;
            use.interfaceKind = "uniform";
            use.index = binding.endpoint.index;
            use.dtype = "f32";
            use.shape = binding.shape;
            use.uniformName = "_vernon_resolution";
            use.binding = binding.native.binding != UINT32_MAX ? binding.native.binding : binding.endpoint.portableSlot;
            use.descriptorSet = binding.native.descriptorSet == UINT32_MAX ? 0 : binding.native.descriptorSet;
            if (binding.transport) {
                use.transport = binding.carrier == TargetCarrier::StorageBuffer   ? "storage_buffer"
                                : binding.carrier == TargetCarrier::UniformBuffer ? "uniform_buffer"
                                                                                  : "push_constant";
                use.interfacePlan = binding.transport->targetAbi;
                use.valueLayout = binding.wholeValueLayout;
            }
            rebuildValueLayoutPathViews(parameter.elementLayout);
            if (parameter.valueLayout)
                rebuildValueLayoutPathViews(*parameter.valueLayout);
            parameter.uses.push_back(std::move(use));
            variant.internalParameters.push_back(std::move(parameter));
            continue;
        }
        auto found = parameterByValue.find(binding.endpoint.value);
        if (found == parameterByValue.end()) {
            vernon::runtime::Parameter parameter;
            parameter.slot = static_cast<uint32_t>(variant.parameters.size());
            parameter.name = binding.name;
            parameter.access = binding.access;
            parameter.kind = binding.kind;
            parameter.elementLayout = binding.wholeValueLayout ? *binding.wholeValueLayout : binding.elementLayout;
            parameter.valueLayout = binding.wholeValueLayout;
            parameter.dimension = binding.dimension;
            parameter.bindingRole = binding.role;
            parameter.exactStorageFormat = binding.imageFormat;
            if (binding.kind == "image" && binding.role == "sampled")
                parameter.sampleResultClass = "float";
            if (binding.source != SourceRepresentation::WholeValueBytes)
                parameter.shape = binding.shape;
            variant.parameters.push_back(std::move(parameter));
            found = parameterByValue.emplace(binding.endpoint.value, variant.parameters.size() - 1).first;
        }
        vernon::runtime::Parameter &parameter = variant.parameters[found->second];
        ParameterUse use;
        use.stage = binding.endpoint.module;
        use.interfaceKind = binding.carrier == TargetCarrier::VertexBuffer           ? "input"
                            : binding.source == SourceRepresentation::ResourceHandle ? "resource"
                                                                                     : "uniform";
        use.index = binding.endpoint.index;
        use.binding = binding.native.binding != UINT32_MAX ? binding.native.binding : binding.endpoint.portableSlot;
        use.descriptorSet = binding.native.descriptorSet == UINT32_MAX ? 0 : binding.native.descriptorSet;
        use.location = binding.native.location;
        use.divisor = binding.divisor;
        use.attributeLeaves = binding.attributeLeaves;
        if (binding.transport) {
            use.dtype = !binding.elementLayout.leaves.empty() ? binding.elementLayout.leaves.front().dtype
                                                              : binding.elementLayout.logicalType;
            use.transport = binding.carrier == TargetCarrier::StorageBuffer   ? "storage_buffer"
                            : binding.carrier == TargetCarrier::UniformBuffer ? "uniform_buffer"
                                                                              : "push_constant";
            use.interfacePlan = binding.transport->targetAbi;
            use.shape = binding.shape;
        }
        if (binding.carrier == TargetCarrier::VertexBuffer) {
            use.dtype = binding.attributeLeaves.front().dtype;
            use.shape = binding.shape;
            use.valueLayout = binding.elementLayout;
        }
        use.sampledImageBindings = binding.sampledImageBindings;
        if (use.interfaceKind != "input" && !binding.name.empty())
            use.uniformName = binding.name;
        parameter.uses.push_back(std::move(use));
    }
    for (const TargetOutput &output : plan.outputs) {
        Output reflected;
        reflected.name = "output_" + std::to_string(output.location);
        reflected.kind = "image";
        const bool tensor = output.type.compare(0, 7, "tensor<") == 0;
        const bool f32 = output.type.size() >= 5 && output.type.compare(output.type.size() - 5, 5, "xf32>") == 0;
        reflected.dtype = f32 ? "f32" : output.type;
        if (tensor) {
            const size_t separator = output.type.find('x', 7);
            if (separator != std::string::npos)
                if (const std::string count = output.type.substr(7, separator - 7);
                    !count.empty() && std::all_of(count.begin(), count.end(), ::isdigit))
                    reflected.shape = {std::stoull(count)};
        }
        reflected.access = "write";
        reflected.location = output.location;
        variant.outputs.push_back(std::move(reflected));
    }
    std::sort(variant.internalParameters.begin(), variant.internalParameters.end(),
              [](const vernon::runtime::Parameter &left, const vernon::runtime::Parameter &right) {
                  return left.name < right.name;
              });
    return true;
}

} // namespace vernon::runtime::program
