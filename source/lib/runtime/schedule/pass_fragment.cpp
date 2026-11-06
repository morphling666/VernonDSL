#include "runtime/backend_stage_pipeline.h"
#include "runtime/capi/program_state.h"
#include "runtime/runtime_state.h"
#include "runtime/schedule/fragment.h"
#include "runtime/schedule/pass_plan.h"
#include "runtime/shape_layout.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vernon::schedule {
namespace {

using Built = Result<Fragment, std::string>;

Built failBuild(std::string message) { return Built{err(std::move(message))}; }

uint64_t alignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }

uint32_t addGraphicsSlot(Fragment &fragment, uint32_t kind, uint32_t access) {
    VernonScheduleSlot slot{};
    slot.kind = kind;
    slot.access = access;
    slot.size = graphicsSlotSize(kind);
    slot.offset = alignUp(fragment.recordSize, 16);
    fragment.recordSize = slot.offset + slot.size;
    fragment.slots.push_back(slot);
    fragment.shapes.emplace_back();
    fragment.extents.emplace_back();
    return static_cast<uint32_t>(fragment.slots.size() - 1);
}

std::optional<runtime::ValueLayout> valueLayoutOf(const runtime::program::Value &value) {
    if (!value.layout)
        return std::nullopt;
    return runtime::program::materializeValueLayout(*value.layout, value.type);
}

std::optional<uint64_t> valueByteSize(const runtime::program::Value &value) {
    const std::optional<runtime::ValueLayout> layout = valueLayoutOf(value);
    if (!layout || !layout->byteSize)
        return std::nullopt;
    uint64_t bytes = layout->byteSize;
    for (uint64_t extent : value.shape) {
        if (!extent || extent > UINT64_MAX / bytes)
            return std::nullopt;
        bytes *= extent;
    }
    return bytes;
}

} // namespace

Built graphicsProgramFragment(VernonProgramExecutable &executable, const std::string &graphName,
                              const std::vector<uint32_t> &exposed) {
    namespace program = runtime::program;
    const program::Program *source = runtime::capi::executableProgram(executable);
    const std::shared_ptr<const program::ResolvedExecutionPlan> plan = executable.executionPlan;
    if (!source || !plan)
        return failBuild("the Program executable has no resolved plan");
    const program::Graph *graph = nullptr;
    for (const program::Graph &candidate : source->graphs)
        if (candidate.name == graphName)
            graph = &candidate;
    if (!graph)
        return failBuild("the Program has no graph '" + graphName + "'");
    const std::optional<program::GraphDirection> direction = program::graphDirection(graph->direction);
    if (!direction)
        return failBuild("the graph has no direction");
    if (graph->nodes.empty())
        return failBuild("the graph has no node");

    Fragment fragment;
    // An ARGUMENT slot per input boundary slot of the graph, and per output boundary slot of a Storage no input has,
    // in ABI order; a Stage parameter names the slot of the Value it reads.
    std::map<uint32_t, uint32_t> slotOfValue;
    // A Value a node writes into a caller's view (a read_write Tensor a compute node rewrites, or an output the caller
    // allocates) lives in the view's Storage, so a later node that reads it reads the slot of that Storage.
    std::map<uint32_t, uint32_t> slotOfStorage;
    const auto storageOf = [&](uint32_t value) -> std::optional<uint32_t> {
        return value < source->values.size() ? source->values[value].storage : std::nullopt;
    };
    const auto ofGraph = [&](const program::BoundarySlot &boundary) {
        return boundary.value < source->values.size() && source->values[boundary.value].origin.graph == graphName;
    };
    std::set<uint32_t> inputStorages;
    for (const program::BoundarySlot &boundary : source->abi.boundarySlots)
        if (ofGraph(boundary) && boundary.direction == program::BoundaryDirection::Input)
            if (const std::optional<uint32_t> storage = storageOf(boundary.value))
                inputStorages.insert(*storage);
    // Slots the call zeroes before its passes: an output of a Storage the Program owns, such as a gradient its
    // pullback accumulates into, starts as zeros as the CPU and CUDA fragments' owned outputs do; so does a scalar
    // gradient, a buffer of one element its Stages sum their lanes into; and a residual a forward graph exposes.
    std::set<uint32_t> zeroedSlots;
    for (const program::BoundarySlot &boundary : source->abi.boundarySlots) {
        if (!ofGraph(boundary))
            continue;
        const bool input = boundary.direction == program::BoundaryDirection::Input;
        const std::optional<uint32_t> storage = storageOf(boundary.value);
        const bool scalarGradient = !input && !storage && boundary.role == program::BoundaryRole::Gradient &&
                                    boundary.category == program::BoundaryCategory::Value;
        if (scalarGradient) {
            const uint32_t slot =
                addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_ARGUMENT, VERNON_SCHEDULE_ACCESS_READ_WRITE);
            fragment.boundarySlots.push_back({boundary.id, slot});
            slotOfValue.emplace(boundary.value, slot);
            zeroedSlots.insert(slot);
            continue;
        }
        const bool output = !input && storage && *storage < source->storages.size() && !inputStorages.count(*storage) &&
                            source->storages[*storage].descriptorKind == program::StorageDescriptorKind::Buffer;
        if (!input && !output)
            continue;
        const uint32_t slot = addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_ARGUMENT,
                                              input ? VERNON_SCHEDULE_ACCESS_READ : VERNON_SCHEDULE_ACCESS_READ_WRITE);
        fragment.boundarySlots.push_back({boundary.id, slot});
        slotOfValue.emplace(boundary.value, slot);
        if (storage)
            slotOfStorage.emplace(*storage, slot);
        if (output && source->storages[*storage].ownership == program::StorageOwnership::Owned &&
            source->storages[*storage].lifetime == program::StorageLifetime::Invocation)
            zeroedSlots.insert(slot);
    }
    // Then an ARGUMENT slot per exposed Storage: a residual the graph allocates, which the caller allocates instead and
    // the call zeroes, as the Program would its owned Storage; or one it captures from the forward graph. Its slot
    // describes the Value's element size and shape, which size and check the residual block. A dynamic extent of a
    // residual is the one its Storage's byte length takes from an input; a captured one's is the block's.
    // An exposed tape is a TAPE slot: written in the graph whose node initializes its Value, read in the graph that
    // captures it. Its passes read their carriers from record bytes the executor writes on every call.
    std::map<uint32_t, uint32_t> tapeSlotOfStorage;
    for (const uint32_t storageId : exposed) {
        if (storageId >= source->storages.size())
            return failBuild("exposed Storage does not exist");
        const program::Storage &storage = source->storages[storageId];
        if (std::any_of(source->values.begin(), source->values.end(), [&](const program::Value &value) {
                return value.storage == storageId && program::isTapeValue(value);
            })) {
            if (storage.initialValue >= source->values.size() ||
                source->values[storage.initialValue].origin.kind != program::OriginKind::NodeResult)
                return failBuild("autodiff tape '" + storage.name + "' has no forward node");
            VernonScheduleSlot slot{};
            slot.kind = VERNON_SCHEDULE_SLOT_TAPE;
            slot.size = sizeof(void *);
            slot.access = source->values[storage.initialValue].origin.graph == graphName ? VERNON_SCHEDULE_ACCESS_WRITE
                                                                                         : VERNON_SCHEDULE_ACCESS_READ;
            slot.offset = alignUp(fragment.recordSize, 16);
            fragment.recordSize = slot.offset + slot.size;
            fragment.slots.push_back(slot);
            fragment.shapes.emplace_back();
            fragment.extents.emplace_back();
            tapeSlotOfStorage.emplace(storageId, static_cast<uint32_t>(fragment.slots.size() - 1));
            continue;
        }
        const program::Value *described = nullptr;
        bool exported = false;
        for (const program::GraphInput &input : graph->inputs)
            if (input.kind == program::GraphInputKind::Allocation && input.storage == storageId &&
                input.value < source->values.size()) {
                described = &source->values[input.value];
                exported = true;
            }
        for (const uint32_t captured : graph->captures)
            if (!described && captured < source->values.size() && source->values[captured].storage == storageId)
                described = &source->values[captured];
        if (!described)
            return failBuild("exposed Storage is neither an allocation nor a capture of the graph");
        const std::optional<runtime::ValueLayout> layout = valueLayoutOf(*described);
        if (!layout || !layout->byteSize)
            return failBuild("exposed Storage '" + storage.name + "' has no element layout");
        const uint32_t slot =
            addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_ARGUMENT,
                            exported ? VERNON_SCHEDULE_ACCESS_READ_WRITE : VERNON_SCHEDULE_ACCESS_READ);
        const runtime::shape::DeclaredShape declared = runtime::shape::decodeRuntimeContractShape(described->shape);
        fragment.slots[slot].rank = static_cast<uint32_t>(declared.size());
        fragment.slots[slot].element_size = layout->byteSize;
        for (uint32_t dimension = 0; dimension < declared.size(); ++dimension) {
            if (const std::optional<uint64_t> fixed = declared[dimension].staticValue()) {
                fragment.shapes[slot].push_back(static_cast<int64_t>(*fixed));
                fragment.extents[slot].push_back(Extent{static_cast<int64_t>(*fixed), kNoSlot, 0});
                continue;
            }
            fragment.shapes[slot].push_back(-1);
            if (!exported) {
                fragment.extents[slot].push_back(Extent{0, slot, dimension});
                continue;
            }
            const std::vector<program::ControlComponent> &lengths = storage.buffer.byteLengthExtents;
            uint32_t valueId = 0;
            const auto input =
                dimension < lengths.size() && lengths[dimension].hasAxis &&
                        program::resolveControlValue(*source, lengths[dimension], graph->direction, valueId)
                    ? slotOfValue.find(valueId)
                    : slotOfValue.end();
            if (input == slotOfValue.end())
                return failBuild("exposed Storage '" + storage.name + "' takes an extent from no input boundary slot");
            fragment.extents[slot].push_back(Extent{0, input->second, lengths[dimension].axis});
        }
        if (exported)
            zeroedSlots.insert(slot);
        slotOfStorage.emplace(storageId, slot);
    }
    // The forward compute node whose launch an autodiff tape records.
    const auto launchedNode = [&](const program::Value &tape) -> const program::Node * {
        const program::Graph *forward = tape.origin.kind == program::OriginKind::NodeResult
                                            ? program::findGraph(*source, tape.origin.graph)
                                            : nullptr;
        for (size_t index = 0; forward && index < forward->nodes.size(); ++index)
            if (forward->nodes[index].id == tape.origin.node &&
                program::executionKind(forward->nodes[index]) == program::ExecutionKind::Compute)
                return &forward->nodes[index];
        return nullptr;
    };
    const auto isLaunch = [](const program::NodeEndpointProjection &projection) {
        return projection.target.semantic == program::CarrierSemantic::Tape &&
               projection.target.tapeCarrier == vernon::program_plan::TapeCarrier::LaunchMetadata;
    };
    // Then an ARGUMENT slot per captured scalar a node reads or takes a workgroup count from, or that counts the
    // forward launch of a tape it reads, in capture order, as planVjp's scalar boundaries are.
    std::set<uint32_t> boundValues;
    const auto bindGrid = [&](const program::Node &node) {
        for (const program::ControlComponent &control : program::computeOperation(node).workgroups)
            if (control.kind != program::ControlKind::Static)
                boundValues.insert(control.reference);
    };
    for (const program::Node &node : graph->nodes) {
        if (const program::ResolvedNodePlan *resolved = plan->node(*direction, node.id))
            for (const program::NodeEndpointProjection &projection : resolved->projections) {
                boundValues.insert(projection.value);
                if (isLaunch(projection) && projection.value < source->values.size())
                    if (const program::Node *launched = launchedNode(source->values[projection.value]))
                        bindGrid(*launched);
            }
        if (program::executionKind(node) == program::ExecutionKind::Compute)
            bindGrid(node);
    }
    for (const uint32_t captured : graph->captures)
        if (captured < source->values.size() && !source->values[captured].storage && boundValues.count(captured))
            slotOfValue.emplace(captured,
                                addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_ARGUMENT, VERNON_SCHEDULE_ACCESS_READ));
    for (const program::Value &value : source->values)
        if (value.storage && !slotOfValue.count(value.id))
            if (const auto found = slotOfStorage.find(*value.storage); found != slotOfStorage.end())
                slotOfValue.emplace(value.id, found->second);
    const auto gridOf = [&](const program::Node &node, hal::ComputeGrid &grid) -> std::optional<std::string> {
        const program::ComputeOperation &compute = program::computeOperation(node);
        for (size_t axis = 0; axis < 3; ++axis) {
            const program::ControlComponent &control = compute.workgroups[axis];
            if (control.kind == program::ControlKind::Static) {
                grid.constant[axis] = control.value;
                continue;
            }
            const auto found = slotOfValue.find(control.reference);
            if (found == slotOfValue.end() || control.reference >= source->values.size())
                return "node '" + node.name + "' takes its grid from no input boundary slot";
            const std::string &dtype = source->values[control.reference].canonicalType.dtype;
            const bool signedScalar = dtype == "i32" || dtype == "si32" || dtype == "i64" || dtype == "si64";
            const bool scalar32 = dtype == "u32" || dtype == "ui32" || dtype == "i32" || dtype == "si32";
            const bool scalar64 =
                dtype == "u64" || dtype == "ui64" || dtype == "index" || dtype == "i64" || dtype == "si64";
            if (!scalar32 && !scalar64)
                return "node '" + node.name + "' takes its grid from a non-integer value";
            grid.argument[axis] = fragment.slots[found->second].offset;
            grid.scalarBytes[axis] = scalar32 ? 4 : 8;
            grid.scalarSigned[axis] = signedScalar;
        }
        return std::nullopt;
    };
    // The internal buffer of each Storage the Program owns for one call, shared by every pass that reads it, and the
    // extents it takes from the call, if any.
    struct Internal {
        std::shared_ptr<hal::InternalBuffer> buffer;
        std::vector<hal::InternalExtent> extents;
    };
    std::map<uint32_t, Internal> internalOfStorage;
    const auto internalBuffer = [&](const program::Value &value) -> Result<Internal, std::string> {
        using Resolved = Result<Internal, std::string>;
        if (!value.storage || *value.storage >= source->storages.size())
            return Resolved{ok(Internal{})};
        const program::Storage &storage = source->storages[*value.storage];
        if (storage.ownership != program::StorageOwnership::Owned ||
            storage.lifetime != program::StorageLifetime::Invocation ||
            storage.descriptorKind != program::StorageDescriptorKind::Buffer)
            return Resolved{ok(Internal{})};
        if (const auto found = internalOfStorage.find(storage.id); found != internalOfStorage.end())
            return Resolved{ok(found->second)};
        if (!storage.buffer.byteLengthExtents.empty()) {
            // Each extent is static or one of a Tensor the caller passes, which the pass reads at plan time.
            const std::optional<runtime::ValueLayout> element =
                storage.initialValue < source->values.size() ? valueLayoutOf(source->values[storage.initialValue])
                                                             : std::nullopt;
            if (!element || !element->byteSize)
                return Resolved{err("its owned Storage '" + storage.name + "' has no element layout")};
            Internal internal{std::make_shared<hal::InternalBuffer>(), {}};
            internal.buffer->bytes = element->byteSize;
            for (const program::ControlComponent &extent : storage.buffer.byteLengthExtents) {
                hal::InternalExtent resolved;
                internal.buffer->shape.push_back(extent.value);
                if (extent.kind != program::ControlKind::Static) {
                    uint32_t valueId = 0;
                    const auto slot = program::resolveControlValue(*source, extent, graph->direction, valueId)
                                          ? slotOfValue.find(valueId)
                                          : slotOfValue.end();
                    if (slot == slotOfValue.end() || !extent.hasAxis)
                        return Resolved{err("its owned Storage '" + storage.name +
                                            "' takes an extent from no input boundary slot")};
                    resolved = {fragment.slots[slot->second].offset, extent.axis};
                }
                internal.extents.push_back(resolved);
            }
            internal.buffer->strides.assign(internal.buffer->shape.size(), 0);
            return Resolved{ok(internalOfStorage.emplace(storage.id, std::move(internal)).first->second)};
        }
        const std::optional<runtime::ValueLayout> layout = valueLayoutOf(value);
        if (!runtime::shape::isConcrete(runtime::shape::decodeRuntimeContractShape(value.shape)) || !layout ||
            !layout->byteSize)
            return Resolved{err("its owned Storage '" + storage.name + "' has no static size")};
        auto internal = std::make_shared<hal::InternalBuffer>();
        internal->shape = value.shape;
        internal->strides.assign(value.shape.size(), 0);
        uint64_t bytes = layout->byteSize;
        for (size_t axis = value.shape.size(); axis-- > 0;) {
            internal->strides[axis] = static_cast<int64_t>(bytes);
            bytes *= value.shape[axis];
        }
        // Every Value of the Storage fits in it.
        for (const program::Value &other : source->values)
            if (other.storage == value.storage)
                if (const std::optional<size_t> size = valueByteSize(other))
                    bytes = std::max<uint64_t>(bytes, *size);
        internal->bytes = std::max<uint64_t>(bytes, storage.buffer.byteLength);
        if (!internal->bytes)
            return Resolved{err("its owned Storage '" + storage.name + "' is empty")};
        return Resolved{ok(internalOfStorage.emplace(storage.id, Internal{std::move(internal), {}}).first->second)};
    };
    // A Stage parameter that reads one leaf of a Struct Value reads a dense carrier: an internal buffer of the view's
    // extents, gathered from the Value's argument before the pass when the Stage reads it and scattered back after
    // the pass when the Stage writes it.
    std::vector<Dispatch> gathers;
    std::vector<Dispatch> scatters;
    const auto leafCarrier = [&](const program::ResolvedNodePlan &resolved, size_t parameter,
                                 Dispatch &pass) -> Result<void, std::string> {
        using Carried = Result<void, std::string>;
        const program::NodeEndpointProjection &projection = resolved.projections[parameter];
        const program::Value &read = source->values[projection.value];
        const auto slot = slotOfValue.find(projection.value);
        if (slot == slotOfValue.end())
            return Carried{err(std::string("it reads a leaf of a Value no argument holds"))};
        const std::optional<runtime::ValueLayout> layout = valueLayoutOf(read);
        if (!layout || !layout->byteSize || *projection.logicalLeaf >= layout->leaves.size() ||
            parameter >= resolved.stage->bindingProjection.parameters.size())
            return Carried{err(std::string("it reads a leaf of a Value with no element layout"))};
        hal::LeafCopy copy;
        copy.storageElement = layout->byteSize;
        copy.storageOffset = layout->leaves[*projection.logicalLeaf].byteOffset;
        if (const auto &endpoint = projection.target.endpointProjection) {
            copy.carrierElement = endpoint->carrierByteSize;
            copy.carrierOffset = endpoint->leafByteOffset;
            copy.bytes = endpoint->leafByteSize;
        } else {
            copy.carrierElement = resolved.stage->bindingProjection.parameters[parameter].elementLayout.byteSize;
            copy.bytes = copy.carrierElement;
        }
        if (!copy.bytes || copy.storageOffset > copy.storageElement ||
            copy.bytes > copy.storageElement - copy.storageOffset || copy.carrierOffset > copy.carrierElement ||
            copy.bytes > copy.carrierElement - copy.carrierOffset)
            return Carried{err(std::string("its leaf does not fit its Value's element or its carrier's"))};
        const uint64_t argument = fragment.slots[slot->second].offset;
        auto buffer = std::make_shared<hal::InternalBuffer>();
        buffer->bytes = copy.carrierElement;
        std::vector<hal::InternalExtent> extents;
        for (uint32_t axis = 0; axis < read.shape.size(); ++axis) {
            buffer->shape.push_back(read.shape[axis]);
            extents.push_back({argument, axis});
        }
        buffer->strides.assign(buffer->shape.size(), 0);
        pass.pass.internals.resize(resolved.projections.size());
        pass.pass.internals[parameter] = buffer;
        if (!extents.empty()) {
            pass.pass.internalExtents.resize(resolved.projections.size());
            pass.pass.internalExtents[parameter] = extents;
        }
        pass.pass.parameters.push_back(UINT64_MAX);
        const auto addCopy = [&](bool scatter) {
            Dispatch dispatch;
            dispatch.kind = OperationKind::Copy;
            dispatch.copy = copy;
            dispatch.copy.scatter = scatter;
            dispatch.pass.parameters = {argument, UINT64_MAX};
            dispatch.pass.internals = {nullptr, buffer};
            if (!extents.empty())
                dispatch.pass.internalExtents = {{}, extents};
            if (zeroedSlots.count(slot->second))
                dispatch.pass.zeroed = {1, 0};
            dispatch.accesses.push_back(
                {false, slot->second,
                 static_cast<uint32_t>(scatter ? VERNON_SCHEDULE_ACCESS_WRITE : VERNON_SCHEDULE_ACCESS_READ)});
            (scatter ? scatters : gathers).push_back(std::move(dispatch));
        };
        if (projection.target.access != "write")
            addCopy(false);
        if (projection.target.access != "read") {
            if (!(fragment.slots[slot->second].access & VERNON_SCHEDULE_ACCESS_WRITE))
                return Carried{err(std::string("it writes a leaf of a Value it may only read"))};
            addCopy(true);
        }
        return Carried{ok()};
    };
    const auto addPass = [&](Dispatch &&pass, const program::Node &node) {
        for (Dispatch &gather : gathers)
            fragment.operations.push_back(std::move(gather));
        fragment.operations.push_back(std::move(pass));
        for (Dispatch &scatter : scatters)
            fragment.operations.push_back(std::move(scatter));
        gathers.clear();
        scatters.clear();
        fragment.nodeOperationEnds.push_back(static_cast<uint32_t>(fragment.operations.size()));
        fragment.nodeLabels.push_back(hal::nodeLabel(node));
    };
    // The PASS slot and color attachment index (UINT32_MAX for depth) of each attachment Storage of a pass so far: a
    // later pass reads the attachment through the RenderPass the earliest pass rendering to it binds.
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> attachmentOfStorage;
    std::vector<std::pair<uint64_t, VernonDrawCommand>> defaultDraws;
    for (const program::Node &node : graph->nodes) {
        const program::ResolvedNodePlan *resolved = plan->node(*direction, node.id);
        if (!resolved || !resolved->stage)
            return failBuild("node '" + node.name + "' has no resolved Stage");
        Dispatch pass;
        pass.kind = OperationKind::Pass;
        pass.executable = static_cast<uint32_t>(fragment.executables.size());
        for (size_t parameter = 0; parameter < resolved->projections.size(); ++parameter) {
            const program::NodeEndpointProjection &projection = resolved->projections[parameter];
            const std::string name = "node '" + node.name + "' Stage parameter " + std::to_string(parameter);
            if (projection.value >= source->values.size())
                return failBuild(name + " reads no Value");
            // The one leaf of a Value is the whole Value, and a carrier of just that leaf is the Value's own buffer.
            const program::Value &read = source->values[projection.value];
            const auto &carrier = projection.target.endpointProjection;
            if (projection.logicalLeaf &&
                (*projection.logicalLeaf != 0 || !read.layout || read.layout->leaves.size() != 1 ||
                 (carrier && (carrier->leafByteOffset || carrier->leafByteSize != carrier->carrierByteSize)))) {
                if (auto carried = leafCarrier(*resolved, parameter, pass); carried.isErr())
                    return failBuild(name + ": " + carried.error());
                continue;
            }
            if (isLaunch(projection)) {
                const program::Node *launched = launchedNode(read);
                if (!launched)
                    return failBuild(name + " reads the launch of a tape that has no forward compute node");
                hal::ComputeGrid launch;
                if (const std::optional<std::string> failed = gridOf(*launched, launch))
                    return failBuild(name + ": " + *failed);
                pass.pass.launches.resize(resolved->projections.size());
                pass.pass.launches[parameter] = launch;
                pass.pass.parameters.push_back(UINT64_MAX);
                continue;
            }
            if (projection.target.semantic == program::CarrierSemantic::Tape) {
                const std::optional<uint32_t> storage = storageOf(projection.value);
                const auto tape = storage ? tapeSlotOfStorage.find(*storage) : tapeSlotOfStorage.end();
                if (tape == tapeSlotOfStorage.end() || !projection.target.tapeCarrier)
                    return failBuild(name + " reads an autodiff tape that is not exposed");
                if (program::executionKind(node) != program::ExecutionKind::Compute)
                    return failBuild(name + " keeps an autodiff tape in a graphics pass");
                if (pass.tape.slot != kNoSlot && pass.tape.slot != tape->second)
                    return failBuild("node '" + node.name + "' uses two autodiff tapes");
                const vernon::program_plan::TapeCarrier carrier = *projection.target.tapeCarrier;
                VernonScheduleField field{};
                field.slot = tape->second;
                field.kind = kTapeFieldArgument;
                field.size = sizeof(VernonProgramArgument);
                field.offset = alignUp(fragment.recordSize, 16);
                fragment.recordSize = field.offset + field.size;
                pass.tape.fields.push_back(
                    {carrier == vernon::program_plan::TapeCarrier::TapeData        ? DeviceTapeCarrier::Data
                     : carrier == vernon::program_plan::TapeCarrier::ReplaySegment ? DeviceTapeCarrier::Segments
                                                                                   : DeviceTapeCarrier::Status,
                     field});
                pass.tape.device = true;
                pass.tape.slot = tape->second;
                pass.tape.writes = fragment.slots[tape->second].access == VERNON_SCHEDULE_ACCESS_WRITE;
                for (const program::TapePlan &tapePlan : source->abi.tapePlans)
                    if (storageOf(tapePlan.value) == storage)
                        pass.tape.stride = std::max(pass.tape.stride, tapePlan.minimumTapeStrideBytes);
                const VernonLaunchSize workgroup = resolved->stage->workgroupSize;
                pass.tape.workgroup[0] = workgroup.x;
                pass.tape.workgroup[1] = workgroup.y;
                pass.tape.workgroup[2] = workgroup.z;
                pass.pass.parameters.push_back(field.offset);
                continue;
            }
            if (const auto found = slotOfValue.find(projection.value); found != slotOfValue.end()) {
                pass.pass.parameters.push_back(fragment.slots[found->second].offset);
                pass.accesses.push_back({false, found->second, fragment.slots[found->second].access});
                if (zeroedSlots.count(found->second)) {
                    pass.pass.zeroed.resize(resolved->projections.size());
                    pass.pass.zeroed[parameter] = 1;
                }
                continue;
            }
            auto internal = internalBuffer(source->values[projection.value]);
            if (internal.isErr())
                return failBuild(name + ": " + internal.error());
            if (internal.value().buffer) {
                pass.pass.internals.resize(resolved->projections.size());
                pass.pass.internals[parameter] = internal.value().buffer;
                if (!internal.value().extents.empty()) {
                    pass.pass.internalExtents.resize(resolved->projections.size());
                    pass.pass.internalExtents[parameter] = internal.value().extents;
                }
                pass.pass.parameters.push_back(UINT64_MAX);
                continue;
            }
            const std::optional<uint32_t> storage = storageOf(projection.value);
            const auto attachment = storage ? attachmentOfStorage.find(*storage) : attachmentOfStorage.end();
            if (attachment == attachmentOfStorage.end())
                return failBuild(name + " reads no input boundary slot, owned Storage or earlier attachment");
            const auto &[renderPass, color] = attachment->second;
            pass.pass.attachments.resize(resolved->projections.size());
            pass.pass.attachments[parameter] = {fragment.slots[renderPass].offset, color};
            pass.pass.parameters.push_back(UINT64_MAX);
            pass.accesses.push_back({false, renderPass, VERNON_SCHEDULE_ACCESS_READ_WRITE});
        }
        hal::ExecutableSource executableSource;
        executableSource.graphics =
            std::make_shared<const hal::GraphicsSource>(hal::GraphicsSource{plan, &node, resolved, &executable});
        if (program::executionKind(node) == program::ExecutionKind::Compute) {
            const auto *controls = std::get_if<program::ResolvedComputeControls>(&resolved->controls);
            if (!controls)
                return failBuild("node '" + node.name + "' has no resolved compute controls");
            pass.pass.compute = true;
            pass.pass.grid.linearized = controls->dispatchMapping != program::DispatchMapping::StaticGrid;
            if (const std::optional<std::string> failed = gridOf(node, pass.pass.grid))
                return failBuild(*failed);
            // The executor sizes a tape's carriers from the grid before the pass's arguments are planned.
            if (pass.tape.present() && pass.pass.grid.linearized)
                return failBuild("node '" + node.name + "' keeps an autodiff tape in a dispatch its arguments size");
            fragment.executables.push_back(std::move(executableSource));
            addPass(std::move(pass), node);
            continue;
        }
        const uint32_t renderPass =
            addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_PASS, VERNON_SCHEDULE_ACCESS_READ_WRITE);
        const uint32_t draw = addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_DRAW, VERNON_SCHEDULE_ACCESS_READ);
        const uint32_t dynamic = addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_DYNAMIC, VERNON_SCHEDULE_ACCESS_READ);
        const uint32_t stream = addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_STREAM, VERNON_SCHEDULE_ACCESS_READ);
        const uint32_t table = addGraphicsSlot(fragment, VERNON_SCHEDULE_SLOT_TABLE, VERNON_SCHEDULE_ACCESS_READ);
        pass.pass.renderPass = fragment.slots[renderPass].offset;
        pass.pass.draw = fragment.slots[draw].offset;
        pass.pass.dynamic = fragment.slots[dynamic].offset;
        pass.pass.stream = fragment.slots[stream].offset;
        pass.pass.table = fragment.slots[table].offset;
        pass.accesses.push_back({false, renderPass, VERNON_SCHEDULE_ACCESS_READ_WRITE});
        // Registered after the node's own parameters, which never read the attachments it renders to.
        if (const auto *controls = std::get_if<program::ResolvedGraphicsControls>(&resolved->controls)) {
            for (uint32_t color = 0; color < controls->colorAttachments.size(); ++color)
                attachmentOfStorage.emplace(controls->colorAttachments[color].storage,
                                            std::make_pair(renderPass, color));
            if (controls->depthStencilAttachment)
                attachmentOfStorage.emplace(controls->depthStencilAttachment->storage,
                                            std::make_pair(renderPass, UINT32_MAX));
        }
        const program::GraphicsOperation &graphics = program::graphicsOperation(node);
        VernonDrawCommand defaultDraw{};
        defaultDraw.struct_size = sizeof(defaultDraw);
        defaultDraw.vertex_count = static_cast<uint32_t>(graphics.vertexCount);
        defaultDraw.instance_count = static_cast<uint32_t>(graphics.instanceCount);
        defaultDraws.emplace_back(pass.pass.draw, defaultDraw);
        fragment.executables.push_back(std::move(executableSource));
        addPass(std::move(pass), node);
    }
    fragment.recordSize = alignUp(fragment.recordSize, 16);
    fragment.recordTemplate.assign(fragment.recordSize, 0);
    for (const auto &[offset, draw] : defaultDraws)
        std::memcpy(fragment.recordTemplate.data() + offset, &draw, sizeof(draw));
    if (auto valid = fragment.validate(); valid.isErr())
        return failBuild(valid.error());
    return Built{ok(std::move(fragment))};
}

Built computeProgramFragment(VernonProgramExecutable &executable, const std::string &graphName,
                             const std::vector<uint32_t> &exposed) {
    namespace program = runtime::program;
    const program::Program *source = runtime::capi::executableProgram(executable);
    const std::shared_ptr<const program::ResolvedExecutionPlan> plan = executable.executionPlan;
    if (!source || !plan)
        return failBuild("the Program executable has no resolved plan");
    if (!executable.programSource)
        return failBuild("a composed Program graph links as a graph of its nodes");
    const program::Graph *graph = program::findGraph(*source, graphName);
    const std::optional<program::GraphDirection> direction =
        graph ? program::graphDirection(graph->direction) : std::nullopt;
    if (!graph || !direction)
        return failBuild("the Program has no graph '" + graphName + "'");
    std::map<std::string, Fragment> stages;
    for (const program::Node &node : graph->nodes) {
        const program::ResolvedNodePlan *resolved = plan->node(*direction, node.id);
        if (!resolved || !resolved->stage)
            return failBuild("node '" + node.name + "' has no resolved Stage");
        if (program::executionKind(node) != program::ExecutionKind::Compute)
            return failBuild("graphics nodes link on Vulkan and OpenGL only");
        if (!resolved->stage->kernelFragment)
            return failBuild("node '" + node.name + "': " + resolved->stage->kernelFragmentError);
        stages.emplace(node.stage, *resolved->stage->kernelFragment);
    }
    // A Program's owned Storages start as zeros on every device.
    std::vector<uint32_t> zeroed;
    for (const program::Storage &storage : source->storages)
        if (storage.ownership == program::StorageOwnership::Owned &&
            storage.lifetime == program::StorageLifetime::Invocation &&
            storage.descriptorKind == program::StorageDescriptorKind::Buffer)
            zeroed.push_back(storage.id);
    return programFragment(*executable.programSource, graphName, stages, zeroed, exposed);
}

} // namespace vernon::schedule
