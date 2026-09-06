#include "target_binding_plan.h"

#include "VernonProgramCapabilities.h"
#include "pipeline_metadata.h"
#include "runtime/autodiff/tape_allocator_abi.h"
#include "shape_layout.h"
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

AutodiffResourceRole autodiffResourceRole(std::string_view role) {
    if (const std::optional<program_plan::TapeCarrier> carrier = program_plan::tapeCarrierFromRoleName(role)) {
        switch (*carrier) {
        case program_plan::TapeCarrier::TapeData:
            return AutodiffResourceRole::Tape;
        case program_plan::TapeCarrier::ReplaySegment:
            return AutodiffResourceRole::ReplaySegment;
        case program_plan::TapeCarrier::ReplayStatus:
            return AutodiffResourceRole::ReplayStatus;
        case program_plan::TapeCarrier::LaunchMetadata:
            return AutodiffResourceRole::LaunchMetadata;
        }
    }
    if (role == "cotangent")
        return AutodiffResourceRole::Cotangent;
    if (role == "gradient")
        return AutodiffResourceRole::Gradient;
    if (role == "primal")
        return AutodiffResourceRole::Primal;
    if (role == "retained_primal")
        return AutodiffResourceRole::RetainedPrimal;
    return AutodiffResourceRole::None;
}

const CompiledEndpointAbi *findCompiledAbi(const StageArtifact &stage, const ReflectedEndpoint &endpoint) {
    const auto found =
        std::find_if(stage.compiledAbi.begin(), stage.compiledAbi.end(), [&](const CompiledEndpointAbi &compiled) {
            return compiled.module == endpoint.module && compiled.interfaceKind == endpoint.interfaceKind &&
                   compiled.index == endpoint.index;
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
    if (binding.tag != BindingTag::Resource)
        return binding.projections.empty() ? UINT32_MAX : binding.projections.front().value;
    if (binding.access >= node.accesses.size())
        return UINT32_MAX;
    const ResourceAccess &access = node.accesses[binding.access];
    return access.kind == AccessKind::Initialize ? access.after
           : access.kind == AccessKind::Write    ? access.before
                                                 : access.value;
}

struct PhysicalValueLeaf {
    TransportNode transport;
    uint64_t offset{};
};

void collectPhysicalValueLeaves(const TransportNode &node, uint64_t base, std::vector<PhysicalValueLeaf> &leaves) {
    if (node.kind != TransportNodeKind::Product) {
        TransportNode leaf = node;
        leaf.offset = 0;
        leaves.push_back({std::move(leaf), base + node.offset});
        return;
    }
    for (const TransportNode &child : node.children)
        collectPhysicalValueLeaves(child, base + node.offset, leaves);
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
    for (const BoundarySlot &slot : program.program.abi.boundarySlots)
        if (slot.role == BoundaryRole::Input && slot.value == valueId)
            return slot.path;
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
    plan.operation = executionKind(node) == ExecutionKind::Compute ? "compute" : "graphics";
    plan.topology = stage.stage.graphicsTopology;
    std::copy(std::begin(stage.stage.workgroupSize), std::end(stage.stage.workgroupSize), plan.workgroupSize);
    plan.dispatch.requiresUnitWorkgroup = stage.stage.requiresUnitWorkgroup;
    for (uint32_t axis : stage.stage.unitGridAxes)
        if (axis < 3)
            plan.dispatch.unitGridAxes[axis] = true;
    for (const CodeModule &module : stage.stage.modules)
        plan.modules.push_back({module.role, module.entryPoint, module.format});

    uint64_t cpuArgumentFrameOffset = 0;
    uint64_t cpuResultFrameOffset = 0;
    const auto cpuFrameOffset = [&](const TargetBinding &binding) -> uint64_t & {
        return binding.endpoint.interfaceKind == "result" ? cpuResultFrameOffset : cpuArgumentFrameOffset;
    };
    bool hasSlots = false;
    const bool compute = executionKind(node) == ExecutionKind::Compute;
    const std::string entry = stage.stage.modules.empty() ? std::string() : stage.stage.modules.front().entryPoint;
    if (entry.find("_program_copy_") != std::string::npos || entry.find("_program_add_") != std::string::npos)
        plan.dispatchMapping = DispatchMapping::FirstTensorElementCount;

    for (const ReflectedEndpoint &endpoint : stage.stage.endpoints) {
        if (compute && endpoint.role == "sampler") {
            const program_capabilities::Entry &capability =
                program_capabilities::get(program_capabilities::Id::ComputeSamplerBinding);
            return reject(diagnostic, std::string(capability.diagnosticCode), "/endpoints",
                          std::string(capability.diagnostic));
        }
        TargetBinding target;
        target.endpoint = {endpoint.module, endpoint.interfaceKind, endpoint.index, UINT32_MAX, endpoint.access};
        target.role = endpoint.role;
        target.access = endpoint.access;
        target.dimension = endpoint.imageDimension;
        target.imageFormat = endpoint.imageFormat;
        target.viewShape = endpoint.viewShape;
        target.writeFootprintKind = endpoint.writeFootprintKind;
        target.writeFootprintIndices = endpoint.writeFootprintIndices;

        if (endpoint.tag == "system") {
            /* An implicit sampler is the one system endpoint a target must still bind. It has no boundary slot, so
             * it never becomes a Program parameter, but the backend needs the images it pairs with. */
            if (endpoint.builtin != "sampler")
                continue;
            if (compute) {
                const program_capabilities::Entry &capability =
                    program_capabilities::get(program_capabilities::Id::ComputeSamplerBinding);
                return reject(diagnostic, std::string(capability.diagnosticCode), "/endpoints",
                              std::string(capability.diagnostic));
            }
            const CompiledEndpointAbi *compiled = findCompiledAbi(stage.stage, endpoint);
            if (!compiled || compiled->sampledImageBindings.empty())
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "implicit sampler is missing the sampled images it pairs with");
            target.source = SourceRepresentation::ImplicitSampler;
            target.kind = "sampler";
            target.carrier = TargetCarrier::Sampler;
            target.sampledImageBindings = compiled->sampledImageBindings;
            target.native = {compiled->descriptorSet, compiled->binding, UINT32_MAX};
            plan.bindings.push_back(std::move(target));
            continue;
        }
        if (endpoint.interfaceKind == "system_value") {
            const EndpointAbiBinding *valueSlot = semantic(endpoint, "value");
            if (!valueSlot || !valueSlot->byteSize || !valueSlot->alignment)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "system value has an invalid portable carrier");
            target.source = SourceRepresentation::SystemValue;
            target.semantic = CarrierSemantic::Value;
            target.name = endpoint.builtin;
            target.sourceName = target.name;
            target.kind = "tensor";
            target.reflectedKind = "builtin";
            target.builtin = endpoint.builtin;
            target.endpoint.portableSlot = valueSlot->slot;
            if (!compute) {
                const CompiledEndpointAbi *compiled = findCompiledAbi(stage.stage, endpoint);
                if (endpoint.builtin != "resolution" || !compiled || !compiled->interfacePlan ||
                    !compiled->interfacePlan->root)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "graphics resolution is missing the compiled physical plan");
                if (valueSlot->byteSize != 8)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "graphics resolution carrier must be two f32 scalars");
                target.carrier = compiled->valueTransport == "uniform_buffer"   ? TargetCarrier::UniformBuffer
                                 : compiled->valueTransport == "storage_buffer" ? TargetCarrier::StorageBuffer
                                                                                : TargetCarrier::InlineValue;
                target.shape = {shape::Extent::fixed(2)};
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
                target.transport = TargetPhysicalTransport{*compiled->interfacePlan, target.native};
                plan.bindings.push_back(std::move(target));
                continue;
            }
            if (valueSlot->byteSize % sizeof(uint32_t))
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "system value has an invalid portable carrier");
            const uint64_t scalarCount = valueSlot->byteSize / sizeof(uint32_t);
            target.carrier = backend == VERNON_RUNTIME_CPU ? TargetCarrier::InlineValue : TargetCarrier::StorageBuffer;
            target.shape =
                scalarCount == 1 ? shape::DeclaredShape{} : shape::DeclaredShape{shape::Extent::fixed(scalarCount)};
            LayoutLeaf leaf{{}, "u32", 0, scalarCount, shape::encodeRuntimeContractShape(target.shape)};
            program::ValueLayout layout{
                "value", endpoint.layoutHash, valueSlot->byteSize, valueSlot->alignment, {leaf}};
            target.wholeValueLayout = materializeValueLayout(layout, endpoint.builtin);
            target.elementLayout = *target.wholeValueLayout;
            InterfacePlan physical = computeValuePlan(leafTransport(leaf), endpoint.layoutHash, backend);
            if (backend == VERNON_RUNTIME_CPU) {
                if (!assignCpuPhysical(cpuFrameOffset(target), findCompiledAbi(stage.stage, endpoint),
                                       valueSlot->alignment, valueSlot->byteSize, target.physical, &physical,
                                       diagnostic, "system value"))
                    return false;
            } else {
                target.physical = {0, static_cast<size_t>(valueSlot->byteSize),
                                   static_cast<size_t>(valueSlot->alignment)};
            }
            target.transport = TargetPhysicalTransport{std::move(physical), {0, valueSlot->slot, valueSlot->slot}};
            plan.bindings.push_back(std::move(target));
            continue;
        }

        const EndpointBinding *binding = findBinding(node, endpoint);
        if (!binding)
            return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                          "node does not bind a reflected endpoint");
        if (binding->tag == BindingTag::Value && binding->projections.size() > 1) {
            const CompiledEndpointAbi *compiled = findCompiledAbi(stage.stage, endpoint);
            const EndpointAbiBinding *slot = semantic(endpoint, "value");
            if (backend != VERNON_RUNTIME_CPU || endpoint.tag != "value" || !compiled || !compiled->interfacePlan ||
                !compiled->interfacePlan->root || !slot)
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                              "aggregate value projections require a compiled CPU value transport");
            std::vector<PhysicalValueLeaf> physicalLeaves;
            collectPhysicalValueLeaves(*compiled->interfacePlan->root, 0, physicalLeaves);
            for (const ValueEndpointProjection &projection : binding->projections) {
                if (projection.value >= program.program.values.size() ||
                    projection.physicalLeaf >= physicalLeaves.size())
                    return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                                  "aggregate projection references value " + std::to_string(projection.value) + " of " +
                                      std::to_string(program.program.values.size()) + " or physical leaf " +
                                      std::to_string(projection.physicalLeaf) + " of " +
                                      std::to_string(physicalLeaves.size()));
                const Value &logical = program.program.values[projection.value];
                if (!logical.layout)
                    return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                                  "aggregate value projection has no canonical layout");
                program::ValueLayout projectedLayout = *logical.layout;
                if (projection.leaf) {
                    if (*projection.leaf >= projectedLayout.leaves.size())
                        return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                                      "aggregate value projection references an unknown canonical leaf");
                    LayoutLeaf leaf = projectedLayout.leaves[*projection.leaf];
                    const uint64_t scalarSize = scalarByteSize(leaf.dtype);
                    leaf.byteOffset = 0;
                    projectedLayout.leaves = {leaf};
                    projectedLayout.byteSize = scalarSize * leaf.scalarCount;
                    projectedLayout.alignment = scalarSize;
                }
                const PhysicalValueLeaf &physicalLeaf = physicalLeaves[projection.physicalLeaf];
                if (projectedLayout.byteSize != physicalLeaf.transport.size)
                    return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                                  "aggregate physical leaf size disagrees with its canonical projection");
                TargetBinding projected = target;
                projected.projection = {projection.value, projection.leaf, projection.direction};
                if (projection.direction == ValueBindingDirection::Result) {
                    projected.access = "write";
                    projected.endpoint.access = "write";
                }
                projected.valueType = logical.canonicalType;
                projected.name = valueName(program, projection.value);
                projected.sourceName = projected.name;
                projected.kind = "tensor";
                projected.reflectedKind = logical.canonicalType.rankedValue ? "tensor_value" : "scalar";
                projected.source = SourceRepresentation::WholeValueBytes;
                projected.semantic = CarrierSemantic::Value;
                projected.carrier = TargetCarrier::InlineValue;
                projected.wholeValueLayout = materializeValueLayout(projectedLayout, logical.type);
                projected.elementLayout = *projected.wholeValueLayout;
                projected.endpoint.portableSlot = slot->slot;
                InterfacePlan leafPlan = *compiled->interfacePlan;
                leafPlan.root = physicalLeaf.transport;
                leafPlan.frameOffset += physicalLeaf.offset;
                projected.physical = {static_cast<size_t>(leafPlan.frameOffset),
                                      static_cast<size_t>(physicalLeaf.transport.size),
                                      static_cast<size_t>(physicalLeaf.transport.alignment)};
                projected.transport = TargetPhysicalTransport{std::move(leafPlan), {}};
                uint64_t &frameSize = cpuFrameOffset(projected);
                const uint64_t projectedEnd =
                    static_cast<uint64_t>(projected.physical.offset) + projected.physical.size;
                frameSize = std::max(frameSize, projectedEnd);
                plan.bindings.push_back(std::move(projected));
            }
            continue;
        }
        const uint32_t valueId = boundValueId(node, *binding);
        if (valueId >= program.program.values.size())
            return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                          "endpoint binding references an unknown Program value");
        const Value &value = program.program.values[valueId];
        target.projection.value = valueId;
        target.projection.leaf =
            binding->tag == BindingTag::Resource ? binding->resourceLeaf : binding->projections.front().leaf;
        target.projection.direction = binding->tag == BindingTag::Resource ? ValueBindingDirection::Input
                                                                           : binding->projections.front().direction;
        if (target.projection.direction == ValueBindingDirection::Result) {
            target.access = "write";
            target.endpoint.access = "write";
        }
        target.valueType = value.canonicalType;
        target.name = valueName(program, valueId);
        target.sourceName = target.name;
        if (target.projection.leaf) {
            if (!value.layout || *target.projection.leaf >= value.layout->leaves.size())
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/bindings",
                              "endpoint binding projects an unknown Program value leaf");
            for (const LayoutPathComponent &component : value.layout->leaves[*target.projection.leaf].path) {
                target.name.push_back('.');
                target.name += component.field.empty() ? std::to_string(component.index.value_or(0)) : component.field;
            }
        }
        if (endpoint.role == "vertex" || !endpoint.viewShape.empty()) {
            const std::optional<shape::DeclaredShape> decoded = shape::decodeReflectedShape(endpoint.viewShape);
            if (!decoded)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "endpoint shape contains an invalid extent");
            target.shape = *decoded;
        } else {
            target.shape = shape::decodeRuntimeContractShape(value.shape);
        }
        if (!endpoint.autodiffCarrier.empty()) {
            if (endpoint.role != "cotangent" || endpoint.autodiffCarrier != "invocation_linear" ||
                endpoint.viewShape.size() != value.shape.size() + 1)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "autodiff carrier is incompatible with its logical value");
            ViewTransform transform;
            transform.axes.push_back({ViewAxisSource::InvocationLinearCarrier, 0, 0, true});
            for (uint32_t axis = 0; axis < value.shape.size(); ++axis)
                transform.axes.push_back({ViewAxisSource::LogicalAxis, axis, 0, false});
            target.viewTransform = std::move(transform);
        } else if (endpoint.role == "cotangent" && endpoint.viewShape.size() != value.shape.size()) {
            return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                          "cotangent physical rank requires an explicit autodiff carrier");
        }
        target.kind = endpoint.role == "sampler" ? "sampler" : !endpoint.imageDimension.empty() ? "image" : "tensor";
        const CompiledEndpointAbi *compiled = findCompiledAbi(stage.stage, endpoint);
        const std::optional<program_plan::TapeCarrier> tapeCarrier =
            program_plan::tapeCarrierFromRoleName(endpoint.role);
        const bool typedTapeCarrier = value.type.rfind("!vernon.ad_tape", 0) == 0 && tapeCarrier;

        if (typedTapeCarrier) {
            if (backend == VERNON_RUNTIME_CPU || !compiled || !compiled->elementLayout || !endpoint.viewDescriptor)
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "opaque tape carrier lacks its GPU TensorView ABI");
            std::vector<const EndpointAbiBinding *> storageBindings;
            for (const EndpointAbiBinding &binding : endpoint.abiBindings)
                if (binding.semantic == "storage_leaf")
                    storageBindings.push_back(&binding);
            const EndpointAbiBinding *offset = semantic(endpoint, "byte_offset");
            if (storageBindings.size() != compiled->elementLayout->leaves.size() || !offset ||
                compiled->elementLayout->leaves.empty())
                return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                              "opaque tape carrier has an incomplete storage ABI");
            target.source = SourceRepresentation::ResourceHandle;
            target.semantic = CarrierSemantic::Tape;
            target.tapeCarrier = tapeCarrier;
            target.name += "." + endpoint.role;
            target.carrier = TargetCarrier::StorageBuffer;
            target.reflectedKind = "tensor";
            target.elementLayout = *compiled->elementLayout;
            target.endpoint.portableSlot = storageBindings.front()->slot;
            target.native = {compiled->descriptorSet == UINT32_MAX ? 0 : compiled->descriptorSet,
                             compiled->binding == UINT32_MAX ? storageBindings.front()->slot : compiled->binding,
                             storageBindings.front()->slot};
            for (size_t leaf = 0; leaf < target.elementLayout.leaves.size(); ++leaf) {
                const vernon::runtime::ValueLeaf &layoutLeaf = target.elementLayout.leaves[leaf];
                const uint64_t elementSize = scalarByteSize(layoutLeaf.dtype);
                if (!elementSize)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "opaque tape carrier has an unsupported element type");
                target.storageLeaves.push_back({static_cast<size_t>(elementSize),
                                                static_cast<size_t>(layoutLeaf.byteOffset),
                                                storageBindings[leaf]->slot});
            }
            TensorViewDescriptorUse descriptor;
            descriptor.rank = endpoint.viewRank;
            descriptor.offsetBinding = offset->slot;
            for (uint32_t axis = 0; axis < endpoint.viewRank; ++axis) {
                const EndpointAbiBinding *extent = semantic(endpoint, "extent", axis);
                const EndpointAbiBinding *stride = semantic(endpoint, "byte_stride", axis);
                if (!extent || !stride)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "opaque tape carrier lacks an extent or stride ABI");
                descriptor.extentBindings.push_back(extent->slot);
                descriptor.strideBindings.push_back(stride->slot);
            }
            hasSlots = true;
            target.tensorViewDescriptor = std::move(descriptor);
            plan.bindings.push_back(std::move(target));
            continue;
        }

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
            hasSlots = true;
            if (backend == VERNON_RUNTIME_CPU) {
                if (!assignCpuPhysical(cpuFrameOffset(target), compiled, alignof(uintptr_t), sizeof(uintptr_t),
                                       target.physical, nullptr, diagnostic, "resource"))
                    return false;
            }
        } else {
            if (!value.layout)
                return reject(diagnostic, "PROGRAM_BINDING_MISMATCH", "/values/" + std::to_string(valueId),
                              "typed target binding requires a canonical value layout");
            if (binding->tag == BindingTag::Value) {
                target.semantic = CarrierSemantic::Value;
                const EndpointAbiBinding *slot = semantic(endpoint, endpoint.tag == "value" ? "value" : "resource");
                if (!slot)
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "value endpoint lacks its portable value carrier");
                if (value.layout->layoutHash != endpoint.layoutHash ||
                    (endpoint.tag == "value" && !endpoint.viewDescriptor &&
                     (value.layout->byteSize != slot->byteSize || value.layout->alignment != slot->alignment)))
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "value endpoint whole-value ABI disagrees with its Program Value");
                target.wholeValueLayout = materializeValueLayout(*value.layout, value.type);
                if (!compiled || (compiled->interfacePlan && !compiled->interfacePlan->root) ||
                    (!compiled->interfacePlan && compiled->valueTransport != "storage_buffer" &&
                     compiled->valueTransport != "uniform_buffer"))
                    return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                  "value endpoint is missing the compiler-selected physical plan");
                target.source = SourceRepresentation::WholeValueBytes;
                target.reflectedKind = value.canonicalType.rankedValue ? "tensor_value" : "scalar";
                target.elementLayout = *target.wholeValueLayout;
                target.carrier = compiled->valueTransport == "uniform_buffer"   ? TargetCarrier::UniformBuffer
                                 : compiled->valueTransport == "storage_buffer" ? TargetCarrier::StorageBuffer
                                                                                : TargetCarrier::InlineValue;
                target.native = {compiled->descriptorSet == UINT32_MAX ? 0 : compiled->descriptorSet,
                                 compiled->binding == UINT32_MAX ? slot->slot : compiled->binding, UINT32_MAX};
                InterfacePlan physical = compiled->interfacePlan ? *compiled->interfacePlan
                                                                 : computeValuePlan(valueTransport(*value.layout),
                                                                                    value.layout->layoutHash, backend);
                if (backend == VERNON_RUNTIME_CPU) {
                    if (!assignCpuPhysical(cpuFrameOffset(target), compiled, value.layout->alignment,
                                           value.layout->byteSize, target.physical, &physical, diagnostic, "value"))
                        return false;
                } else {
                    target.physical = {0, static_cast<size_t>(physical.root->size),
                                       static_cast<size_t>(physical.root->alignment)};
                }
                target.transport = TargetPhysicalTransport{std::move(physical), target.native};
                target.endpoint.portableSlot = slot->slot;
                if (endpoint.tag == "resource") {
                    const EndpointAbiBinding *offset = semantic(endpoint, "byte_offset");
                    if (!offset || endpoint.viewRank != 0 || target.carrier != TargetCarrier::StorageBuffer)
                        return reject(diagnostic, "PROGRAM_REFLECTION_MISMATCH", "/endpoints",
                                      "physical resource projection has an incompatible TensorView descriptor");
                    target.storageLeaves.push_back(
                        {static_cast<size_t>(value.layout->byteSize), size_t{0}, slot->slot});
                    TensorViewDescriptorUse descriptor;
                    descriptor.rank = 0;
                    descriptor.offsetBinding = offset->slot;
                    target.tensorViewDescriptor = std::move(descriptor);
                }
                hasSlots = true;
            } else {
                std::vector<std::pair<uint32_t, const EndpointAbiBinding *>> indexedStorageBindings;
                for (const EndpointAbiBinding &binding : endpoint.abiBindings)
                    if (binding.semantic == "storage_leaf" && binding.axis)
                        indexedStorageBindings.emplace_back(*binding.axis, &binding);
                std::sort(indexedStorageBindings.begin(), indexedStorageBindings.end(),
                          [](const auto &left, const auto &right) { return left.first < right.first; });
                if (target.projection.leaf && (indexedStorageBindings.size() != 1 ||
                                               indexedStorageBindings.front().first != *target.projection.leaf))
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
                    }
                    hasSlots = true;
                    target.tensorViewDescriptor = std::move(descriptor);
                    if (backend == VERNON_RUNTIME_CPU) {
                        const uint64_t descriptorSize =
                            static_cast<uint64_t>(2 + 2 * endpoint.viewRank) * sizeof(uintptr_t);
                        if (!assignCpuPhysical(cpuFrameOffset(target), compiled, alignof(uintptr_t), descriptorSize,
                                               target.physical, nullptr, diagnostic, "TensorView descriptor"))
                            return false;
                    }
                } else if (backend == VERNON_RUNTIME_CPU) {
                    InterfacePlan physical =
                        computeValuePlan(valueTransport(*value.layout), value.layout->layoutHash, backend);
                    if (!assignCpuPhysical(cpuFrameOffset(target), compiled, value.layout->alignment,
                                           value.layout->byteSize, target.physical, &physical, diagnostic,
                                           "TensorView value"))
                        return false;
                    target.transport = TargetPhysicalTransport{
                        std::move(physical), {0, storageBindings.front()->slot, storageBindings.front()->slot}};
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
            target.endpoint = {"compute", "system_value", compiled.index, UINT32_MAX, "read"};
            target.sourceName = target.name;
            if (!assignCpuPhysical(cpuFrameOffset(target), &compiled, alignof(uintptr_t), sizeof(uintptr_t),
                                   target.physical, nullptr, diagnostic, "tape packed field"))
                return false;
            plan.bindings.push_back(std::move(target));
        }
    }

    for (const GraphicsFragmentOutput &output : stage.stage.fragmentOutputs)
        plan.outputs.push_back({output.location, output.type});
    if (backend == VERNON_RUNTIME_CPU)
        plan.packedArgumentsSize = cpuArgumentFrameOffset;
    if (backend == VERNON_RUNTIME_CPU)
        plan.packedResultsSize = cpuResultFrameOffset;
    if (backend == VERNON_RUNTIME_METAL && !stage.stage.nativeSlots.empty())
        plan.nativeSlots = stage.stage.nativeSlots;
    else if (backend == VERNON_RUNTIME_METAL && hasSlots && compute)
        return reject(diagnostic, "PROGRAM_TARGET_BINDING", "/stage",
                      "Metal stage has no explicit target resource slots");
    return true;
}

static bool buildComputeExecutableBindingView(const TargetBindingPlan &plan, ExecutableBindingView &variant,
                                              ReflectedEntry &reflection, Diagnostic &diagnostic) {
    diagnostic = {};
    variant = {};
    reflection = {};
    std::copy(std::begin(plan.workgroupSize), std::end(plan.workgroupSize), reflection.workgroup);
    reflection.dispatchContract = plan.dispatch;
    if (plan.backend == VERNON_RUNTIME_CPU)
        reflection.packedArguments = PackedArgumentsLayout{static_cast<size_t>(plan.packedArgumentsSize)};
    if (plan.backend == VERNON_RUNTIME_CPU && plan.packedResultsSize)
        reflection.packedResults = PackedArgumentsLayout{static_cast<size_t>(plan.packedResultsSize)};
    if (!plan.modules.empty()) {
        variant.compute = plan.modules.front().entryPoint;
        variant.program.emplace("compute", variant.compute);
    }
    for (const TargetBinding &binding : plan.bindings) {
        const uint32_t runtimeArgumentIndex = static_cast<uint32_t>(reflection.arguments.size());
        ReflectedArgument argument;
        argument.sourceName = binding.name;
        argument.index = runtimeArgumentIndex;
        argument.kind = binding.reflectedKind.empty() ? binding.kind : binding.reflectedKind;
        if (plan.backend != VERNON_RUNTIME_CPU && binding.carrier == TargetCarrier::StorageBuffer)
            argument.kind = "tensor";
        else if (argument.kind == "tensor_value")
            argument.kind = "scalar";
        argument.builtin = binding.builtin;
        argument.autodiffRole = binding.role;
        if (!binding.elementLayout.leaves.empty())
            argument.dtype = pipelineDataType(binding.elementLayout.leaves.front().dtype);
        else if (binding.wholeValueLayout && !binding.wholeValueLayout->leaves.empty())
            argument.dtype = pipelineDataType(binding.wholeValueLayout->leaves.front().dtype);
        argument.result = binding.endpoint.interfaceKind == "result";
        argument.physical = binding.physical;
        if (!argument.physical.alignment)
            argument.physical.alignment = 1;
        argument.descriptorSet = 0;
        argument.binding = binding.endpoint.portableSlot;
        argument.storageLeaves = binding.storageLeaves;
        argument.tensorElementSize = binding.elementLayout.byteSize;
        argument.sourceShape = binding.viewShape;
        if (const std::optional<shape::ConcreteShape> concrete = shape::concrete(binding.shape);
            concrete && !concrete->empty()) {
            size_t elements = 0;
            if (!shape::checkedElementCount(*concrete, elements))
                return false;
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
        parameter.autodiffRole = autodiffResourceRole(binding.role);
        parameter.autodiffSource = binding.sourceName;
        parameter.invocationCarrier =
            binding.viewTransform && !binding.viewTransform->axes.empty() &&
            binding.viewTransform->axes.front().source == ViewAxisSource::InvocationLinearCarrier;
        parameter.exactStorageFormat = binding.imageFormat;
        parameter.shape = shape::encodeRuntimeContractShape(binding.shape);
        ParameterUse use;
        use.stage = "compute";
        use.index = runtimeArgumentIndex;
        use.interfaceKind = binding.endpoint.interfaceKind == "result"             ? "result"
                            : binding.kind == "image" || binding.kind == "sampler" ? "resource"
                            : binding.source == SourceRepresentation::WholeValueBytes ||
                                    (binding.source == SourceRepresentation::ElementStream && binding.transport)
                                ? "value"
                                : "storage";
        use.dtype = binding.elementLayout.logicalType.empty() && !binding.elementLayout.leaves.empty()
                        ? binding.elementLayout.leaves.front().dtype
                        : binding.elementLayout.logicalType;
        use.shape = binding.source == SourceRepresentation::WholeValueBytes && binding.valueType &&
                            binding.valueType->rankedValue
                        ? binding.valueType->innerShape
                        : parameter.shape;
        use.transport = plan.backend == VERNON_RUNTIME_CPU                ? "host_value"
                        : binding.carrier == TargetCarrier::UniformBuffer ? "uniform_buffer"
                        : binding.carrier == TargetCarrier::InlineValue   ? "push_constant"
                                                                          : "storage_buffer";
        use.valueLayout = binding.wholeValueLayout;
        use.interfacePlan = binding.transport ? std::optional<InterfacePlan>(binding.transport->targetAbi)
                                              : std::optional<InterfacePlan>{};
        use.tensorViewDescriptor = binding.tensorViewDescriptor;
        use.descriptorSet = 0;
        use.binding = binding.endpoint.portableSlot;
        parameter.uses.push_back(std::move(use));
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
    rebuildVariantLayoutViews(variant);
    return true;
}

bool buildResolvedExecutablePlan(const ResolvedProgram &program, VernonRuntimeBackend backend,
                                 ResolvedExecutablePlan &plan, Diagnostic &diagnostic) {
    plan = {};
    const bool hasBackward = findGraph(program.program, "backward") != nullptr;
    if (hasBackward) {
        const bool graphicsVjp =
            std::any_of(program.program.graphs.begin(), program.program.graphs.end(), [](const Graph &graph) {
                return graph.direction == "backward" &&
                       std::any_of(graph.nodes.begin(), graph.nodes.end(),
                                   [](const Node &node) { return executionKind(node) == ExecutionKind::Graphics; });
            });
        if (graphicsVjp) {
            const program_capabilities::Entry &capability =
                program_capabilities::get(program_capabilities::Id::GraphicsVjp);
            return reject(diagnostic, std::string(capability.diagnosticCode), "/graphs",
                          std::string(capability.diagnostic));
        }
        const bool opaqueVjp = std::any_of(program.program.abi.boundarySlots.begin(),
                                           program.program.abi.boundarySlots.end(), [](const BoundarySlot &slot) {
                                               return slot.category == BoundaryCategory::Texture ||
                                                      slot.category == BoundaryCategory::Sampler;
                                           });
        if (opaqueVjp) {
            const program_capabilities::Entry &capability =
                program_capabilities::get(program_capabilities::Id::OpaqueResourceVjp);
            return reject(diagnostic, std::string(capability.diagnosticCode), "/abi/boundary_slots",
                          std::string(capability.diagnostic));
        }
    }
    if (backend != VERNON_RUNTIME_CPU) {
        const bool usesF16 =
            std::any_of(program.program.values.begin(), program.program.values.end(), [](const Value &value) {
                return value.layout && std::any_of(value.layout->leaves.begin(), value.layout->leaves.end(),
                                                   [](const LayoutLeaf &leaf) { return leaf.dtype == "f16"; });
            });
        if (usesF16) {
            const program_capabilities::Entry &capability = program_capabilities::get(program_capabilities::Id::GpuF16);
            return reject(diagnostic, std::string(capability.diagnosticCode), "/values",
                          std::string(capability.diagnostic));
        }
    }
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
                return reject(diagnostic, "PROGRAM_ARTIFACT_TARGET", "/target",
                              "ArtifactSystem target does not match Runtime backend");
            ResolvedExecutableNode resolved;
            resolved.graph = graph.direction;
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

static bool buildGraphicsExecutableBindingView(const TargetBindingPlan &plan, ExecutableBindingView &variant,
                                               Diagnostic &diagnostic) {
    diagnostic = {};
    variant = {};
    std::map<uint32_t, size_t> parameterByValue;
    for (const TargetBinding &binding : plan.bindings) {
        if (binding.source == SourceRepresentation::ImplicitSampler) {
            vernon::runtime::Parameter parameter;
            parameter.name =
                "__vernon_implicit_sampler_" + binding.endpoint.module + "_" + std::to_string(binding.endpoint.index);
            parameter.kind = "sampler";
            parameter.source = "implicit_sampler";
            parameter.access = "read";
            ParameterUse use;
            use.stage = binding.endpoint.module;
            use.interfaceKind = "resource";
            use.index = binding.endpoint.index;
            use.descriptorSet = binding.native.descriptorSet == UINT32_MAX ? 0 : binding.native.descriptorSet;
            use.binding = binding.native.binding;
            use.sampledImageBindings = binding.sampledImageBindings;
            parameter.uses.push_back(std::move(use));
            variant.internalParameters.push_back(std::move(parameter));
            continue;
        }
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
            parameter.shape = shape::encodeRuntimeContractShape(binding.shape);
            ParameterUse use;
            use.stage = binding.endpoint.module;
            use.interfaceKind = "uniform";
            use.index = binding.endpoint.index;
            use.dtype = "f32";
            use.shape = parameter.shape;
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
            parameter.uses.push_back(std::move(use));
            variant.internalParameters.push_back(std::move(parameter));
            continue;
        }
        auto found = parameterByValue.find(binding.projection.value);
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
            parameter.shape = shape::encodeRuntimeContractShape(binding.shape);
            variant.parameters.push_back(std::move(parameter));
            found = parameterByValue.emplace(binding.projection.value, variant.parameters.size() - 1).first;
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
            use.shape = binding.source == SourceRepresentation::WholeValueBytes && binding.valueType &&
                                binding.valueType->rankedValue
                            ? binding.valueType->innerShape
                            : parameter.shape;
        }
        if (binding.carrier == TargetCarrier::VertexBuffer) {
            use.dtype = binding.attributeLeaves.front().dtype;
            use.shape = shape::encodeRuntimeContractShape(binding.shape);
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
    rebuildVariantLayoutViews(variant);
    return true;
}

bool buildExecutableBindingView(const TargetBindingPlan &plan, ExecutableBindingView &view, ReflectedEntry &reflection,
                                Diagnostic &diagnostic) {
    if (plan.operation == "graphics") {
        reflection = {};
        return buildGraphicsExecutableBindingView(plan, view, diagnostic);
    }
    if (plan.operation == "compute")
        return buildComputeExecutableBindingView(plan, view, reflection, diagnostic);
    return reject(diagnostic, "PROGRAM_OPERATION_UNSUPPORTED", "/graphs",
                  "TargetBindingPlan has an unsupported operation");
}

} // namespace vernon::runtime::program
