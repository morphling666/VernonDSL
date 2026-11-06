#include "runtime/schedule/pass_plan.h"

#include "runtime/capi/program_state.h"
#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <array>
#include <limits>

namespace vernon::hal {
namespace {

// Turns the caller's whole-Value argument into what Stage parameter `parameter` reads: the Stage's slot, the
// endpoint's access and the Stage's element or whole-Value layout. An element stream of a
// ranked Value bound without a shape takes the Value's inner shape, with row-major strides; an endpoint's view
// transform reorders and broadcasts the Value's axes, over the dispatch's `invocations` (0 when it has none) for an
// axis the dispatch derives. Either points the argument into `view`, which must outlive it.
Made<void> stageArgument(const runtime::program::Program &program, const VernonStageExecutable &stage,
                         const runtime::program::ResolvedNodePlan &resolved, size_t parameter, uint64_t invocations,
                         VernonProgramArgument &argument, StagedView &view) {
    if (parameter >= stage.bindingProjection.parameters.size() || parameter >= resolved.projections.size() ||
        resolved.projections[parameter].value >= program.values.size())
        return Made<void>{err(std::string("the Stage has no parameter ") + std::to_string(parameter))};
    const runtime::Parameter &stageParameter = stage.bindingProjection.parameters[parameter];
    const runtime::program::TargetBinding &target = resolved.projections[parameter].target;
    const runtime::program::CanonicalValueType &type =
        program.values[resolved.projections[parameter].value].canonicalType;
    std::vector<int64_t> &strides = view.strides;
    argument.slot = stageParameter.slot;
    if (argument.kind != VERNON_PROGRAM_TENSOR)
        return Made<void>{ok()};
    argument.tensor.access = target.access == "read"    ? VERNON_ACCESS_READ
                             : target.access == "write" ? VERNON_ACCESS_WRITE
                                                        : VERNON_ACCESS_READ_WRITE;
    const runtime::ValueLayout *layout = &stageParameter.elementLayout;
    if (stageParameter.tensorArgument == runtime::TensorRepresentation::WholeValue) {
        if (!stageParameter.valueLayout)
            return Made<void>{err("Stage parameter " + std::to_string(parameter) + " has no whole-Value layout")};
        layout = &*stageParameter.valueLayout;
        argument.tensor.rank = 0;
        argument.tensor.shape = nullptr;
        argument.tensor.byte_strides = nullptr;
    } else if (type.rankedValue && !argument.tensor.rank && !type.innerShape.empty()) {
        strides.assign(type.innerShape.size(), 0);
        int64_t stride = static_cast<int64_t>(layout->byteSize);
        for (size_t axis = type.innerShape.size(); axis-- > 0;) {
            strides[axis] = stride;
            stride *= static_cast<int64_t>(type.innerShape[axis]);
        }
        argument.tensor.rank = static_cast<uint32_t>(type.innerShape.size());
        argument.tensor.shape = type.innerShape.data();
        argument.tensor.byte_strides = strides.data();
    }
    if (target.viewTransform) {
        const std::string name = "Stage parameter " + std::to_string(parameter);
        const std::vector<runtime::program::ViewAxisTransform> &axes = target.viewTransform->axes;
        if ((argument.tensor.rank && (!argument.tensor.shape || !argument.tensor.byte_strides)) ||
            stageParameter.shape.size() != axes.size())
            return Made<void>{err(name + " has a view transform incompatible with its Value")};
        const std::vector<uint64_t> logicalShape(argument.tensor.shape, argument.tensor.shape + argument.tensor.rank);
        const std::vector<int64_t> logicalStrides(argument.tensor.byte_strides,
                                                  argument.tensor.byte_strides + argument.tensor.rank);
        view.shape.clear();
        view.strides.clear();
        for (size_t physicalAxis = 0; physicalAxis < axes.size(); ++physicalAxis) {
            const runtime::program::ViewAxisTransform &axis = axes[physicalAxis];
            if (axis.source == runtime::program::ViewAxisSource::Constant) {
                if (!axis.constantExtent || !axis.zeroStride)
                    return Made<void>{err(name + " has a constant view axis with no storage mapping")};
                view.shape.push_back(axis.constantExtent);
                view.strides.push_back(0);
                continue;
            }
            if (axis.source == runtime::program::ViewAxisSource::InvocationLinearCarrier) {
                if (!axis.zeroStride || !invocations)
                    return Made<void>{err(name + " broadcasts over the invocations of no static dispatch")};
                view.shape.push_back(invocations);
                view.strides.push_back(0);
                continue;
            }
            if (axis.logicalAxis >= logicalShape.size())
                return Made<void>{err(name + " has a view axis of no axis of its Value")};
            const uint64_t physicalExtent = stageParameter.shape[physicalAxis];
            const uint64_t logicalExtent = logicalShape[axis.logicalAxis];
            if (physicalExtent && logicalExtent && physicalExtent != logicalExtent)
                return Made<void>{err(name + " has a view axis whose extent disagrees with its Value")};
            view.shape.push_back(logicalExtent ? logicalExtent : physicalExtent);
            view.strides.push_back(axis.zeroStride ? 0 : logicalStrides[axis.logicalAxis]);
        }
        argument.tensor.rank = static_cast<uint32_t>(view.shape.size());
        argument.tensor.shape = view.shape.empty() ? nullptr : view.shape.data();
        argument.tensor.byte_strides = view.strides.empty() ? nullptr : view.strides.data();
    }
    argument.tensor.element_layout = runtime::pipelineValueLayout(*layout);
    return Made<void>{ok()};
}

runtime::GraphicsDrawStream drawStream(const VernonScheduleDrawStream &stream) {
    return {stream.host_records, stream.host_counts, stream.records,    stream.counts,
            stream.first_record, stream.max_records, stream.count_index};
}

std::string entryPrefix(const std::string &pass, size_t entry) {
    return entry == SIZE_MAX ? pass : pass + "table entry " + std::to_string(entry) + ": ";
}

Made<void> checkReads(const std::string &name, const PlannedPass &pass) {
    if (runtime::graphicsScopeReadsOverlap(pass.reads, pass.targets))
        return Made<void>{err(name + "samples an image it renders to")};
    return Made<void>{ok()};
}

constexpr size_t kCachedPlans = 4;

} // namespace

Made<std::array<uint32_t, 3>> gridCounts(const ComputeGrid &grid, const uint8_t *record) {
    using Counted = Made<std::array<uint32_t, 3>>;
    std::array<uint32_t, 3> counts{};
    for (size_t axis = 0; axis < 3; ++axis) {
        uint64_t groups = grid.constant[axis];
        if (grid.argument[axis] != UINT64_MAX) {
            const auto scalar = readRecord<VernonProgramArgument>(record, grid.argument[axis]);
            const size_t bytes = grid.scalarBytes[axis];
            if (scalar.kind != VERNON_PROGRAM_TENSOR || scalar.tensor.storage != VERNON_TENSOR_HOST ||
                !scalar.tensor.host_data || scalar.tensor.byte_offset > scalar.tensor.byte_size ||
                bytes > scalar.tensor.byte_size - scalar.tensor.byte_offset)
                return Counted{err(std::string("its grid reads no host scalar"))};
            const auto *data = static_cast<const uint8_t *>(scalar.tensor.host_data) + scalar.tensor.byte_offset;
            if (bytes == 4) {
                uint32_t value{};
                std::memcpy(&value, data, sizeof(value));
                groups = grid.scalarSigned[axis] && static_cast<int32_t>(value) < 0 ? UINT64_MAX : value;
            } else {
                std::memcpy(&groups, data, sizeof(groups));
                if (grid.scalarSigned[axis] && static_cast<int64_t>(groups) < 0)
                    groups = UINT64_MAX;
            }
        }
        if (!groups || groups > UINT32_MAX)
            return Counted{err(std::string("its grid is outside the portable uint32 range"))};
        counts[axis] = static_cast<uint32_t>(groups);
    }
    return Counted{ok(counts)};
}

struct PassPlanner::PassCall {
    const VernonRenderPass *renderPass{};
    VernonDynamicState dynamic{};
    VernonScheduleDrawStream stream{};
    const VernonSchedulePassTable *table{};
    // The node's own draw, of a pass without a table.
    VernonDrawCommand draw{};
};

namespace {

// The words of what a table pass's draws are planned from besides the table: the node's arguments but for the
// bytes of host Tensors, and the formats and extent of the attachments, the render area and the dynamic state.
std::vector<uint64_t> passKey(const PlannedPass &pass, const PassPlanner::PassCall &call) {
    std::vector<uint64_t> key;
    for (const VernonProgramArgument &argument : pass.nodeArguments) {
        key.insert(key.end(), {argument.slot, static_cast<uint64_t>(argument.kind)});
        if (argument.kind == VERNON_PROGRAM_IMAGE) {
            addReference(key, argument.image.view);
            continue;
        }
        if (argument.kind != VERNON_PROGRAM_TENSOR) {
            addReference(key, argument.resource);
            continue;
        }
        const VernonTensorView &tensor = argument.tensor;
        key.insert(key.end(),
                   {static_cast<uint64_t>(tensor.storage), static_cast<uint64_t>(tensor.access), tensor.rank,
                    tensor.byte_offset, tensor.byte_size, tensor.element_layout.byte_size,
                    tensor.element_layout.alignment,
                    reinterpret_cast<uintptr_t>(tensor.element_layout.layout_hash.data),
                    tensor.element_layout.layout_hash.size, reinterpret_cast<uintptr_t>(tensor.element_layout.leaves),
                    tensor.element_layout.leaf_count, tensor.shape != nullptr, tensor.byte_strides != nullptr});
        if (tensor.storage != VERNON_TENSOR_HOST)
            addReference(key, tensor.resource);
        if (tensor.shape)
            key.insert(key.end(), tensor.shape, tensor.shape + tensor.rank);
        if (tensor.byte_strides)
            for (uint32_t axis = 0; axis < tensor.rank; ++axis)
                key.push_back(static_cast<uint64_t>(tensor.byte_strides[axis]));
    }
    const runtime::PlannedRenderTargets &targets = pass.targets;
    key.insert(key.end(), {targets.attachmentFormats.size(), static_cast<uint64_t>(targets.depthFormat),
                           targets.depthAttachment != nullptr, targets.attachmentWidth, targets.attachmentHeight});
    for (const VernonTextureFormat format : targets.attachmentFormats)
        key.push_back(static_cast<uint64_t>(format));
    addWords(key, call.renderPass->render_area, sizeof(call.renderPass->render_area));
    key.push_back(call.dynamic.struct_size);
    if (call.dynamic.struct_size) {
        addWords(key, call.dynamic.viewport, sizeof(call.dynamic.viewport));
        addWords(key, call.dynamic.scissor, sizeof(call.dynamic.scissor));
        key.push_back(call.dynamic.stencil_reference);
    }
    return key;
}

} // namespace

PassExecutable::PassExecutable(std::shared_ptr<const GraphicsSource> source) : source_(std::move(source)) {
    if (runtime::program::executionKind(*source_->node) == runtime::program::ExecutionKind::Graphics)
        state_.describe(runtime::program::graphicsOperation(*source_->node));
}

const std::string &nodeLabel(const runtime::program::Node &node) { return node.name.empty() ? node.stage : node.name; }

VernonStatus describeImage(void *userData, VernonRuntimeProviderResourceReference resource,
                           VernonRuntimeProviderImageDescription *description) {
    auto described = runtime::describeBackendImage(*static_cast<VernonRuntimeContext *>(userData), resource);
    if (described.isErr())
        return toVernonStatus(std::move(described).error());
    *description = described.value();
    return VERNON_STATUS_OK;
}

std::string runtimeMessage(VernonRuntimeContext &context, const RuntimeError &error, const char *fallback) {
    std::string &diagnostic = runtime::invocationDiagnostic(context);
    if (!diagnostic.empty())
        return std::exchange(diagnostic, std::string());
    return error.context.operation ? error.context.operation : fallback;
}

Made<void> TableBindings::select(const std::shared_ptr<const RuntimeGraphicsPipelineState> &pipeline) {
    if (pipeline == pipeline_)
        return Made<void>{ok()};
    std::unordered_map<std::string_view, Bound> named;
    for (size_t endpoint = 0; endpoint < bound_.size(); ++endpoint)
        if (bound_[endpoint].set)
            named[runtime::capi::endpointPath(*pipeline_, endpoint)] = bound_[endpoint];
    pipeline_ = pipeline;
    frames_.clear();
    bound_.assign(pipeline->endpoints.size(), {});
    for (size_t endpoint = 0; endpoint < bound_.size() && !named.empty(); ++endpoint) {
        if (!pipeline->endpoints[endpoint].parameterSlot)
            continue;
        const auto found = named.find(runtime::capi::endpointPath(*pipeline, endpoint));
        if (found == named.end())
            continue;
        const std::shared_ptr<const RuntimeBindingSetMatch> match =
            runtime::capi::matchBindingSet(*found->second.set, pipeline);
        if (!match->failures[found->second.entry] && match->endpoints[endpoint])
            bound_[endpoint] = found->second;
    }
    return Made<void>{ok()};
}

Made<void> TableBindings::bind(const VernonBindingSet *set) {
    if (!set || !set->state || !set->state->sealed)
        return Made<void>{err(std::string("it binds a binding set that is not sealed"))};
    const std::shared_ptr<const RuntimeBindingSetMatch> match = runtime::capi::matchBindingSet(*set->state, pipeline_);
    for (const runtime::program::Diagnostic &failure : match->failures)
        if (failure)
            return Made<void>{err(failure.path + ": " + failure.message)};
    for (size_t entry = 0; entry < set->state->entries.size(); ++entry)
        for (uint32_t endpoint : match->entryEndpoints[entry])
            bound_[endpoint] = {set->state.get(), entry};
    return Made<void>{ok()};
}

Made<const runtime::capi::InvocationSetFrame *> TableBindings::frame(VernonRuntimeContext &context, size_t endpoint) {
    RuntimeBindingSetState *set = bound_[endpoint].set;
    if (const auto found = frames_.find(set); found != frames_.end())
        return Made<const runtime::capi::InvocationSetFrame *>{ok(&found->second)};
    RecordedSetFrame recorded{runtime::capi::matchBindingSet(*set, pipeline_), pipeline_};
    runtime::capi::InvocationSetFrame lowered;
    std::string error;
    if (!runtime::capi::lowerSetFrame(context, recorded, lowered, error))
        return Made<const runtime::capi::InvocationSetFrame *>{err(std::move(error))};
    retained_.push_back(std::move(recorded.match));
    return Made<const runtime::capi::InvocationSetFrame *>{ok(&frames_.emplace(set, std::move(lowered)).first->second)};
}

PassCommandBuffer::PassCommandBuffer(std::vector<DispatchCommand> commands)
    : commands_(std::move(commands)), passCache_(std::make_unique<PassCache>()) {
    passCache_->commands.resize(commands_.size());
}

Made<void> checkPassCommands(const std::vector<DispatchCommand> &commands, const char *device) {
    for (const DispatchCommand &command : commands) {
        const bool copy = command.kind == CommandKind::Copy && command.pass.parameters.size() == 2 &&
                          command.pass.internals.size() == 2 && command.pass.internals[1];
        if ((!copy && (command.kind != CommandKind::Pass || !command.executable || !command.pass.present())) ||
            !command.recordAddresses.empty())
            return Made<void>{err(std::string(device) + " devices record compute and graphics passes only")};
    }
    return Made<void>{ok()};
}

Made<InternalBuffers::Internal> InternalBuffers::allocate(std::shared_ptr<const InternalBuffer> buffer,
                                                          uint64_t bytes) {
    if (!buffers_)
        return Made<Internal>{err(std::string("this device allocates no internal buffer"))};
    auto allocated = buffers_->allocate(bytes);
    if (allocated.isErr())
        return Made<Internal>{err("cannot allocate an internal buffer: " + allocated.error())};
    std::unique_ptr<Buffer> allocation = std::move(allocated).value();
    const VernonRuntimeProviderResourceReference reference = allocation->reference(0, bytes);
    return Made<Internal>{ok(Internal{std::move(buffer), std::move(allocation), reference, bytes})};
}

Made<const InternalBuffers::Internal *> InternalBuffers::internal(const std::shared_ptr<const InternalBuffer> &buffer,
                                                                  uint64_t bytes) {
    const auto found = internals_.find(buffer.get());
    if (found != internals_.end() && found->second.bytes >= bytes)
        return Made<const Internal *>{ok(&found->second)};
    if (found != internals_.end())
        bytes = std::max(bytes, 2 * found->second.bytes);
    bytes = std::max(bytes, buffer->bytes);
    auto created = allocate(buffer, bytes);
    if (created.isErr())
        return Made<const Internal *>{err(std::move(created).error())};
    Internal entry = std::move(created).value();
    if (found == internals_.end())
        return Made<const Internal *>{ok(&internals_.emplace(buffer.get(), std::move(entry)).first->second)};
    outgrown_.push_back(std::exchange(found->second, std::move(entry)));
    return Made<const Internal *>{ok(&found->second)};
}

Made<const InternalBuffers::Internal *> InternalBuffers::launch(const std::array<uint32_t, 3> &counts) {
    if (const auto found = launches_.find(counts); found != launches_.end())
        return Made<const Internal *>{ok(&found->second)};
    constexpr uint64_t bytes = sizeof(counts);
    auto created = allocate(nullptr, bytes);
    if (created.isErr())
        return Made<const Internal *>{err(std::move(created).error())};
    if (auto uploaded = buffers_->upload(*created.value().allocation, counts.data(), bytes); uploaded.isErr())
        return Made<const Internal *>{err("cannot upload a tape's launch counts: " + uploaded.error())};
    return Made<const Internal *>{ok(&launches_.emplace(counts, std::move(created).value()).first->second)};
}

Made<std::vector<const InternalBuffers::Internal *>>
InternalBuffers::used(const std::vector<DispatchCommand> &commands) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const Internal *> used;
    for (const DispatchCommand &command : commands)
        for (const std::shared_ptr<const InternalBuffer> &buffer : command.pass.internals) {
            if (!buffer)
                continue;
            auto found = internal(buffer, 0);
            if (found.isErr())
                return Made<std::vector<const Internal *>>{err(std::move(found).error())};
            if (std::find(used.begin(), used.end(), found.value()) == used.end())
                used.push_back(found.value());
        }
    return Made<std::vector<const Internal *>>{ok(std::move(used))};
}

Made<VernonProgramArgument> InternalBuffers::argument(const PassOffsets &offsets, size_t parameter,
                                                      const uint8_t *record, InternalShape &shape) {
    if (offsets.parameters[parameter] != UINT64_MAX)
        return Made<VernonProgramArgument>{
            ok(readRecord<VernonProgramArgument>(record, offsets.parameters[parameter]))};
    if (!offsets.launches.empty() && offsets.launches[parameter]) {
        auto counts = gridCounts(*offsets.launches[parameter], record);
        if (counts.isErr())
            return Made<VernonProgramArgument>{err("a tape's launch: " + counts.error())};
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = launch(counts.value());
        if (found.isErr())
            return Made<VernonProgramArgument>{err(std::move(found).error())};
        shape.shape = {3};
        shape.strides = {sizeof(uint32_t)};
        VernonProgramArgument argument{};
        argument.kind = VERNON_PROGRAM_TENSOR;
        argument.tensor.struct_size = sizeof(argument.tensor);
        argument.tensor.storage = VERNON_TENSOR_DEVICE_RESOURCE;
        argument.tensor.resource = found.value()->reference;
        argument.tensor.access = VERNON_ACCESS_READ;
        argument.tensor.rank = 1;
        argument.tensor.shape = shape.shape.data();
        argument.tensor.byte_strides = shape.strides.data();
        argument.tensor.byte_size = 3 * sizeof(uint32_t);
        return Made<VernonProgramArgument>{ok(argument)};
    }
    if (!offsets.attachments.empty() && offsets.attachments[parameter].renderPass != UINT64_MAX) {
        const AttachmentRead &read = offsets.attachments[parameter];
        const auto *renderPass = readRecord<const VernonRenderPass *>(record, read.renderPass);
        const VernonRuntimeProviderResourceReference *view =
            !renderPass                ? nullptr
            : read.color == UINT32_MAX ? (renderPass->depth_attachment ? &renderPass->depth_attachment->view : nullptr)
            : read.color < renderPass->color_attachment_count ? &renderPass->color_attachments[read.color].view
                                                              : nullptr;
        if (!view)
            return Made<VernonProgramArgument>{err(std::string("the RenderPass binds no attachment it reads"))};
        VernonProgramArgument argument{};
        argument.kind = VERNON_PROGRAM_IMAGE;
        argument.image.view = *view;
        return Made<VernonProgramArgument>{ok(argument)};
    }
    const std::shared_ptr<const InternalBuffer> &buffer = offsets.internals[parameter];
    const std::vector<uint64_t> *extents = &buffer->shape;
    const std::vector<int64_t> *strides = &buffer->strides;
    uint64_t bytes = buffer->bytes;
    if (!offsets.internalExtents.empty() && !offsets.internalExtents[parameter].empty()) {
        const std::vector<InternalExtent> &dynamic = offsets.internalExtents[parameter];
        shape.shape.resize(dynamic.size());
        shape.strides.resize(dynamic.size());
        for (size_t axis = 0; axis < dynamic.size(); ++axis) {
            shape.shape[axis] = buffer->shape[axis];
            if (dynamic[axis].argument == UINT64_MAX)
                continue;
            const auto source = readRecord<VernonProgramArgument>(record, dynamic[axis].argument);
            if (source.kind != VERNON_PROGRAM_TENSOR || dynamic[axis].axis >= source.tensor.rank ||
                !source.tensor.shape)
                return Made<VernonProgramArgument>{
                    err(std::string("an internal buffer takes an extent of an argument that has none"))};
            shape.shape[axis] = source.tensor.shape[dynamic[axis].axis];
        }
        for (size_t axis = dynamic.size(); axis-- > 0;) {
            shape.strides[axis] = static_cast<int64_t>(bytes);
            if (shape.shape[axis] && bytes > std::numeric_limits<uint64_t>::max() / shape.shape[axis])
                return Made<VernonProgramArgument>{err(std::string("an internal buffer's size overflows"))};
            bytes *= shape.shape[axis];
        }
        extents = &shape.shape;
        strides = &shape.strides;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = internal(buffer, bytes);
    if (found.isErr())
        return Made<VernonProgramArgument>{err(std::move(found).error())};
    VernonProgramArgument argument{};
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(argument.tensor);
    argument.tensor.storage = VERNON_TENSOR_DEVICE_RESOURCE;
    argument.tensor.resource = found.value()->reference;
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = static_cast<uint32_t>(extents->size());
    argument.tensor.shape = extents->empty() ? nullptr : extents->data();
    argument.tensor.byte_strides = strides->empty() ? nullptr : strides->data();
    argument.tensor.byte_size = bytes;
    return Made<VernonProgramArgument>{ok(argument)};
}

void addWords(std::vector<uint64_t> &key, const void *data, size_t size) {
    const size_t at = key.size();
    key.resize(at + ((size + sizeof(uint64_t) - 1) / sizeof(uint64_t)), 0);
    std::memcpy(key.data() + at, data, size);
}

void addReference(std::vector<uint64_t> &key, const VernonRuntimeProviderResourceReference &reference) {
    key.insert(key.end(), {reference.identity, reference.resource.value, reference.offset, reference.size});
}

void addScopeKey(std::vector<uint64_t> &key, const runtime::GraphicsRenderScope &scope) {
    for (const VernonRuntimeProviderColorAttachment &color : scope.colors) {
        addReference(key, color.view);
        key.insert(key.end(), {color.location, static_cast<uint64_t>(color.load_operation),
                               static_cast<uint64_t>(color.store_operation)});
        addWords(key, &color.clear_color, sizeof(color.clear_color));
    }
    if (!scope.depth)
        return;
    const VernonRuntimeProviderDepthStencilAttachment &depth = *scope.depth;
    addReference(key, depth.view);
    uint32_t clearDepth{};
    std::memcpy(&clearDepth, &depth.clear_depth, sizeof(clearDepth));
    key.insert(key.end(),
               {static_cast<uint64_t>(depth.depth_load_operation), static_cast<uint64_t>(depth.depth_store_operation),
                clearDepth, static_cast<uint64_t>(depth.stencil_load_operation),
                static_cast<uint64_t>(depth.stencil_store_operation), depth.clear_stencil});
}

void addCopyKey(std::vector<uint64_t> &key, const PlannedCopy &copy) {
    key.push_back(3);
    addReference(key, copy.source);
    addReference(key, copy.destination);
    key.push_back(copy.regions.size());
    for (const CopyRegion &region : copy.regions)
        key.insert(key.end(), {region.source, region.destination, region.size});
}

// Every pass is planned before any scope begins, because a scope must name the images all of its passes sample when
// it begins.
Made<void> PassPlanner::plan(const PassCommandBuffer &commands, const uint8_t *record, PlannedPasses &planned) {
    PassCache &cache = commands.passCache();
    planned.passes.reserve(commands.commands().size());
    runtime::GraphicsScopeMaterializer materializer;
    bool interrupted = false;
    for (size_t index = 0; index < commands.commands().size(); ++index) {
        const DispatchCommand &command = commands.commands()[index];
        const bool copy = command.kind == CommandKind::Copy;
        for (size_t parameter = 0; parameter < command.pass.zeroed.size(); ++parameter) {
            if (!command.pass.zeroed[parameter])
                continue;
            const auto argument = readRecord<VernonProgramArgument>(record, command.pass.parameters[parameter]);
            if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_DEVICE_RESOURCE)
                return Made<void>{err(
                    (copy ? std::string("a leaf copy")
                          : "pass '" + static_cast<const PassExecutable &>(*command.executable).source().node->name +
                                "'") +
                    " zeroes an argument that names no device buffer")};
            const VernonRuntimeProviderResourceReference &range = argument.tensor.resource;
            const bool seen = std::any_of(planned.zeroed.begin(), planned.zeroed.end(), [&](const auto &zeroed) {
                return zeroed.identity == range.identity && zeroed.resource.value == range.resource.value &&
                       zeroed.offset == range.offset;
            });
            if (!seen)
                planned.zeroed.push_back(range);
        }
        if (copy) {
            PlannedPass &pass = *planned.passes.emplace_back(&planned.fresh.emplace_back());
            pass.command = &command;
            if (auto made = planCopy(command, record, pass); made.isErr())
                return made;
            interrupted = true;
            continue;
        }
        const auto &executable = static_cast<const PassExecutable &>(*command.executable);
        if (command.pass.compute) {
            PlannedPass &pass = *planned.passes.emplace_back(&planned.fresh.emplace_back());
            pass.command = &command;
            if (auto made = planCompute(executable, command.pass, record, pass); made.isErr())
                return made;
            // A dispatch ends the scope before it: no pass after it fuses with a pass before it.
            interrupted = true;
            continue;
        }
        auto made = planGraphics(executable, command.pass, record, cache, index, planned);
        if (made.isErr())
            return Made<void>{err(std::move(made).error())};
        PlannedPass &pass = *planned.passes.emplace_back(made.value());
        pass.command = &command;
        // One candidate region: adjacent passes fuse whenever their attachments allow it.
        if (materializer.materialize(0, pass.targets, pass.reads, std::exchange(interrupted, false)) ==
            runtime::GraphicsScopeMaterialization::Fuse) {
            planned.scopes.back() = materializer.scope();
        } else {
            planned.scopes.push_back(materializer.scope());
        }
        pass.scope = planned.scopes.size() - 1;
    }
    return Made<void>{ok()};
}

Made<void> PassPlanner::planCompute(const PassExecutable &executable, const PassOffsets &offsets, const uint8_t *record,
                                    PlannedPass &pass) {
    const std::string name = "pass '" + executable.source().node->name + "': ";
    pass.executable = &executable;
    pass.compute = true;
    pass.arguments.resize(offsets.parameters.size());
    pass.views.resize(offsets.parameters.size());
    pass.internalShapes.resize(offsets.parameters.size());
    const ComputeGrid &grid = offsets.grid;
    auto counts = gridCounts(grid, record);
    if (counts.isErr())
        return Made<void>{err(name + counts.error())};
    pass.grid = {counts.value()[0], counts.value()[1], counts.value()[2]};
    const VernonStageExecutable &stage = *executable.source().resolved->stage;
    // A linearized grid follows the arguments, so a view axis cannot broadcast over its invocations.
    uint64_t invocations = 0;
    if (!grid.linearized) {
        invocations = 1;
        for (uint64_t extent :
             {uint64_t{pass.grid.x}, uint64_t{pass.grid.y}, uint64_t{pass.grid.z}, uint64_t{stage.workgroupSize.x},
              uint64_t{stage.workgroupSize.y}, uint64_t{stage.workgroupSize.z}}) {
            if (!extent || invocations > std::numeric_limits<uint64_t>::max() / extent)
                return Made<void>{err(name + "its invocation count overflows")};
            invocations *= extent;
        }
    }
    for (size_t parameter = 0; parameter < pass.arguments.size(); ++parameter) {
        auto argument = internals_.argument(offsets, parameter, record, pass.internalShapes[parameter]);
        if (argument.isErr())
            return Made<void>{err(name + argument.error())};
        pass.arguments[parameter] = argument.value();
        if (auto adapted = stageArgument(*runtime::capi::executableProgram(*executable.source().executable), stage,
                                         *executable.source().resolved, parameter, invocations,
                                         pass.arguments[parameter], pass.views[parameter]);
            adapted.isErr())
            return Made<void>{err(name + adapted.error())};
    }
    if (!grid.linearized)
        return Made<void>{ok()};
    // A linearized grid covers the element count of the first Tensor argument with a shape.
    for (const VernonProgramArgument &argument : pass.arguments) {
        if (argument.kind != VERNON_PROGRAM_TENSOR || (argument.tensor.rank && !argument.tensor.shape))
            continue;
        uint64_t count = 1;
        for (uint32_t dimension = 0; dimension < argument.tensor.rank && count; ++dimension) {
            const uint64_t extent = argument.tensor.shape[dimension];
            if (extent && count > UINT32_MAX / extent)
                return Made<void>{err(name + "its linearized grid exceeds the portable uint32 range")};
            count *= extent;
        }
        pass.grid = {static_cast<uint32_t>(count), 1, 1};
        return Made<void>{ok()};
    }
    return Made<void>{err(name + "its linearized grid has no Tensor extent source")};
}

Made<void> PassPlanner::planCopy(const DispatchCommand &command, const uint8_t *record, PlannedPass &pass) {
    const LeafCopy &layout = command.copy;
    pass.compute = true;
    pass.internalShapes.resize(2);
    auto storage = internals_.argument(command.pass, 0, record, pass.internalShapes[0]);
    if (storage.isErr())
        return Made<void>{err("a leaf copy: " + storage.error())};
    auto carrier = internals_.argument(command.pass, 1, record, pass.internalShapes[1]);
    if (carrier.isErr())
        return Made<void>{err("a leaf copy: " + carrier.error())};
    const VernonTensorView &view = storage.value().tensor;
    const VernonTensorView &dense = carrier.value().tensor;
    if (storage.value().kind != VERNON_PROGRAM_TENSOR || view.storage != VERNON_TENSOR_DEVICE_RESOURCE ||
        (view.rank && !view.shape))
        return Made<void>{err(std::string("a leaf copy's Storage is no device Tensor"))};
    uint64_t elements = 1;
    for (uint32_t axis = 0; axis < view.rank; ++axis) {
        if (view.shape[axis] && elements > std::numeric_limits<uint64_t>::max() / view.shape[axis])
            return Made<void>{err(std::string("a leaf copy's element count overflows"))};
        elements *= view.shape[axis];
    }
    if (elements > dense.byte_size / layout.carrierElement)
        return Made<void>{err(std::string("a leaf copy's carrier is smaller than its view"))};
    // The view's strides, row-major over the Storage's elements when the caller passes none.
    std::vector<int64_t> strides(view.rank);
    int64_t stride = static_cast<int64_t>(layout.storageElement);
    for (uint32_t axis = view.rank; axis-- > 0;) {
        strides[axis] = view.byte_strides ? view.byte_strides[axis] : stride;
        stride *= static_cast<int64_t>(view.shape[axis]);
    }
    PlannedCopy copy;
    copy.source = layout.scatter ? dense.resource : view.resource;
    copy.destination = layout.scatter ? view.resource : dense.resource;
    std::vector<uint64_t> index(view.rank, 0);
    for (uint64_t element = 0; element < elements; ++element) {
        int64_t at = static_cast<int64_t>(view.byte_offset + layout.storageOffset);
        for (uint32_t axis = 0; axis < view.rank; ++axis)
            at += static_cast<int64_t>(index[axis]) * strides[axis];
        if (at < 0 || static_cast<uint64_t>(at) + layout.bytes > view.resource.size)
            return Made<void>{err(std::string("a leaf copy reads outside its Storage's buffer"))};
        const uint64_t inStorage = view.resource.offset + static_cast<uint64_t>(at);
        const uint64_t inCarrier =
            dense.resource.offset + dense.byte_offset + element * layout.carrierElement + layout.carrierOffset;
        CopyRegion region{layout.scatter ? inCarrier : inStorage, layout.scatter ? inStorage : inCarrier, layout.bytes};
        if (!copy.regions.empty() && copy.regions.back().source + copy.regions.back().size == region.source &&
            copy.regions.back().destination + copy.regions.back().size == region.destination)
            copy.regions.back().size += region.size;
        else
            copy.regions.push_back(region);
        for (uint32_t axis = view.rank; axis-- > 0;) {
            if (++index[axis] < view.shape[axis])
                break;
            index[axis] = 0;
        }
    }
    pass.copy = std::move(copy);
    return Made<void>{ok()};
}

// Reads the node's arguments and plans its render targets.
Made<void> PassPlanner::planInputs(const PassExecutable &executable, const PassOffsets &offsets, const uint8_t *record,
                                   PlannedPass &pass, PassCall &call) {
    const GraphicsSource &source = executable.source();
    const std::string name = "pass '" + source.node->name + "': ";
    pass.executable = &executable;
    pass.nodeArguments.resize(offsets.parameters.size());
    pass.internalShapes.resize(offsets.parameters.size());
    for (size_t parameter = 0; parameter < pass.nodeArguments.size(); ++parameter) {
        auto argument = internals_.argument(offsets, parameter, record, pass.internalShapes[parameter]);
        if (argument.isErr())
            return Made<void>{err(name + argument.error())};
        pass.nodeArguments[parameter] = argument.value();
    }
    call.renderPass = readRecord<const VernonRenderPass *>(record, offsets.renderPass);
    call.dynamic = readRecord<VernonDynamicState>(record, offsets.dynamic);
    call.stream = readRecord<VernonScheduleDrawStream>(record, offsets.stream);
    call.table = readRecord<const VernonSchedulePassTable *>(record, offsets.table);
    if (!call.table)
        call.draw = readRecord<VernonDrawCommand>(record, offsets.draw);
    if (!call.renderPass)
        return Made<void>{err(name + "no RenderPass is bound")};
    std::string error;
    if (!runtime::planRenderTargets(call.renderPass, describeImage, &context_, pass.targets, error) ||
        !runtime::checkProgramGraphicsAttachmentSignature(runtime::program::graphicsOperation(*source.node),
                                                          pass.targets, error))
        return Made<void>{err(name + error)};
    return Made<void>{ok()};
}

// Plans a graphics pass in `fresh`, or reuses the plan an earlier call made of its table or of its own draw, taking
// the host bytes of the node's arguments and the attachments of this call.
Made<PlannedPass *> PassPlanner::planGraphics(const PassExecutable &executable, const PassOffsets &offsets,
                                              const uint8_t *record, PassCache &cache, size_t command,
                                              PlannedPasses &planned) {
    std::deque<CachedPass> &cached = cache.commands[command];
    PlannedPass &pass = planned.fresh.emplace_back();
    PassCall call;
    if (auto read = planInputs(executable, offsets, record, pass, call); read.isErr())
        return Made<PlannedPass *>{err(std::move(read).error())};
    const VernonSchedulePassTable *table = call.table;
    // A table's revision stands for its entries; the node's own draw is in the key.
    const bool cacheable = !table || (table->struct_size >= sizeof(*table) && table->revision);
    if (!cacheable) {
        if (auto made = planPass(call, pass); made.isErr())
            return Made<PlannedPass *>{err(std::move(made).error())};
        return Made<PlannedPass *>{ok(&pass)};
    }
    const uint64_t revision = table ? table->revision : 0;
    std::vector<uint64_t> key = passKey(pass, call);
    if (!table) {
        addWords(key, &call.draw, sizeof(call.draw));
        addWords(key, &call.stream, sizeof(call.stream));
    }
    for (CachedPass &plan : cached) {
        if (plan.table != table || plan.revision != revision || plan.key != key)
            continue;
        PlannedPass &reused = plan.pass;
        std::copy(pass.nodeArguments.begin(), pass.nodeArguments.end(), reused.nodeArguments.begin());
        for (const auto &[draw, parameter, node] : plan.hostArguments)
            draw->arguments[parameter].tensor.host_data = reused.nodeArguments[node].tensor.host_data;
        reused.targets = std::move(pass.targets);
        planned.fresh.pop_back();
        if (auto checked = checkReads("pass '" + executable.source().node->name + "': ", reused); checked.isErr())
            return Made<PlannedPass *>{err(std::move(checked).error())};
        return Made<PlannedPass *>{ok(&reused)};
    }
    if (auto made = planPass(call, pass); made.isErr())
        return Made<PlannedPass *>{err(std::move(made).error())};
    if (cached.size() == kCachedPlans) {
        planned.evicted.push_back(cached.front().pass.plan);
        cached.pop_front();
    }
    pass.plan = cache.nextPlan++;
    CachedPass &made = cached.emplace_back();
    made.table = table;
    made.revision = revision;
    made.key = std::move(key);
    made.pass = std::move(pass);
    for (PlannedDraw &draw : made.pass.draws)
        for (const auto &[parameter, node] : draw.nodeParameters)
            if (draw.arguments[parameter].kind == VERNON_PROGRAM_TENSOR &&
                draw.arguments[parameter].tensor.storage == VERNON_TENSOR_HOST)
                made.hostArguments.emplace_back(&draw, parameter, node);
    planned.fresh.pop_back();
    return Made<PlannedPass *>{ok(&cached.back().pass)};
}

// Plans the draws of a pass whose inputs planInputs read.
Made<void> PassPlanner::planPass(const PassCall &call, PlannedPass &pass) {
    const PassExecutable &executable = *pass.executable;
    const GraphicsSource &source = executable.source();
    const std::string name = "pass '" + source.node->name + "': ";
    std::vector<VernonProgramArgument> &nodeArguments = pass.nodeArguments;
    const VernonRenderPass *renderPass = call.renderPass;
    const VernonDynamicState &dynamic = call.dynamic;
    const VernonScheduleDrawStream &stream = call.stream;
    const VernonSchedulePassTable *table = call.table;
    const VernonDynamicState *dynamicState = dynamic.struct_size ? &dynamic : nullptr;
    // A node argument's identity is its place in the pass, so draws of one pass that bind it share its groups.
    const auto nodeOwner = [&](size_t parameter) { return runtime::ArgumentOwner{&nodeArguments[parameter], nullptr}; };

    if (!table) {
        PlannedDraw &draw = pass.draws.emplace_back();
        draw.program = runtime::capi::executableProgram(*source.executable);
        draw.resolved = source.resolved;
        draw.stage = source.resolved->stage.get();
        draw.state = &executable.state();
        draw.arguments = nodeArguments;
        for (size_t parameter = 0; parameter < nodeArguments.size(); ++parameter)
            draw.owners.push_back(nodeOwner(parameter));
        draw.lowered.assign(nodeArguments.size(), nullptr);
        for (size_t parameter = 0; parameter < nodeArguments.size(); ++parameter)
            draw.nodeParameters.emplace_back(parameter, parameter);
        draw.drawCommand = call.draw;
        if (stream.struct_size != 0) {
            if (stream.struct_size < sizeof(stream))
                return Made<void>{err(name + "the draw stream's struct_size is too small")};
            draw.streamed = true;
            draw.stream = drawStream(stream);
        }
        if (auto made = planDraw(name, SIZE_MAX, *renderPass, dynamicState, pass, draw); made.isErr())
            return made;
        return checkReads(name, pass);
    }
    if (table->struct_size < sizeof(*table) || (table->entry_count && !table->entries))
        return Made<void>{err(name + "the pass table is malformed")};
    pass.draws.reserve(table->entry_count);
    TableBindings &bound = pass.bindings;
    size_t debugDepth = 0;
    const auto copyDebug = [&](const VernonScheduleDebugCommand *commands, size_t count,
                               std::vector<PlannedDebug> &copied) -> Made<void> {
        if (count && !commands)
            return Made<void>{err(name + "the pass table's debug commands are malformed")};
        for (size_t at = 0; at < count; ++at) {
            const VernonScheduleDebugCommand &command = commands[at];
            if (command.label.size && !command.label.data)
                return Made<void>{err(name + "a debug command's label is malformed")};
            if (command.kind == VERNON_GRAPHICS_COMMAND_PUSH_DEBUG_GROUP) {
                ++debugDepth;
            } else if (command.kind == VERNON_GRAPHICS_COMMAND_POP_DEBUG_GROUP) {
                if (!debugDepth)
                    return Made<void>{err(name + "the pass table pops a debug group it did not push")};
                --debugDepth;
            } else if (command.kind != VERNON_GRAPHICS_COMMAND_INSERT_DEBUG_MARKER) {
                return Made<void>{err(name + "a debug command has an unknown kind")};
            }
            copied.push_back({command.kind, std::string(command.label.data, command.label.size)});
        }
        return Made<void>{ok()};
    };
    for (size_t index = 0; index < table->entry_count; ++index) {
        const VernonSchedulePassEntry &entry = table->entries[index];
        // Formatted only on failure: a pass may have thousands of entries.
        const auto entryName = [&] { return entryPrefix(name, index); };
        const bool streamed = entry.stream.struct_size != 0;
        if (entry.struct_size < sizeof(entry) || (entry.argument_count && !entry.arguments) ||
            entry.draw.struct_size < sizeof(entry.draw) || (streamed && !entry.draw.index_binding) ||
            (streamed && entry.stream.struct_size < sizeof(entry.stream)) ||
            (entry.dynamic.struct_size && entry.dynamic.struct_size < sizeof(entry.dynamic)))
            return Made<void>{err(entryName() + "is malformed")};
        if (entry.set_count && !entry.sets)
            return Made<void>{err(entryName() + "is malformed")};
        PlannedDraw &draw = pass.draws.emplace_back();
        if (auto copied = copyDebug(entry.debug_commands, entry.debug_command_count, draw.debug); copied.isErr())
            return copied;
        std::shared_ptr<const RuntimeGraphicsPipelineState> pipeline;
        if (entry.pipeline) {
            pipeline = entry.pipeline->state;
        } else {
            // Pipeline 0 is cached in the executable, which the call otherwise leaves unchanged.
            auto own = runtime::capi::passPipelineState(const_cast<VernonProgramExecutable &>(*source.executable),
                                                        source.node->id);
            if (own.isErr())
                return Made<void>{err(entryName() + own.error())};
            pipeline = std::move(own).value();
        }
        if (pipeline->pass != source.executable || pipeline->node != source.node->id)
            return Made<void>{err(entryName() + "its pipeline draws in another graphics node")};
        const runtime::program::Program *program = runtime::capi::executableProgram(*pipeline->source);
        const runtime::program::ResolvedNodePlan *resolved = pipeline->sourcePlan;
        if (!program || !resolved || !resolved->stage)
            return Made<void>{err(entryName() + "its pipeline has no resolved Stage")};
        const std::vector<runtime::recorded::PipelineEndpoint> &endpoints = pipeline->endpoints;
        if (endpoints.size() != resolved->projections.size())
            return Made<void>{err(entryName() + "its pipeline's endpoints do not match its Stage")};
        draw.state = entry.pipeline ? &pipeline->graphicsState.state : &executable.state();
        draw.program = program;
        draw.resolved = resolved;
        draw.stage = resolved->stage.get();
        if (auto rebound = bound.select(pipeline); rebound.isErr())
            return Made<void>{err(entryName() + rebound.error())};
        for (size_t set = 0; set < entry.set_count; ++set)
            if (auto bind = bound.bind(entry.sets[set]); bind.isErr())
                return Made<void>{err(entryName() + bind.error())};
        // A Stage parameter is the pass Node's fixed endpoint, or a per-draw parameter of the pipeline's Program
        // that the entry's arguments or a bound set supply, or that draws with the Node's own binding.
        draw.arguments.resize(endpoints.size());
        draw.owners.resize(endpoints.size());
        draw.lowered.assign(endpoints.size(), nullptr);
        for (size_t parameter = 0; parameter < endpoints.size(); ++parameter) {
            const runtime::recorded::PipelineEndpoint &endpoint = endpoints[parameter];
            const VernonProgramArgument *found = nullptr;
            if (endpoint.fixedPassParameter && *endpoint.fixedPassParameter < nodeArguments.size()) {
                found = &nodeArguments[*endpoint.fixedPassParameter];
                draw.owners[parameter] = nodeOwner(*endpoint.fixedPassParameter);
                draw.nodeParameters.emplace_back(parameter, *endpoint.fixedPassParameter);
            }
            for (size_t argument = 0; !found && endpoint.parameterSlot && argument < entry.argument_count; ++argument) {
                const uint32_t slot = entry.arguments[argument].slot;
                if (slot == *endpoint.parameterSlot ||
                    (slot < program->abi.boundarySlots.size() &&
                     program->abi.boundarySlots[slot].value == resolved->projections[parameter].value)) {
                    found = &entry.arguments[argument];
                    draw.owners[parameter] = {found, nullptr};
                }
            }
            if (!found && endpoint.parameterSlot && bound.supplies(parameter)) {
                // Lowered for the pipeline's Stage once, and owning its groups across calls.
                auto frame = bound.frame(context_, parameter);
                if (frame.isErr())
                    return Made<void>{err(entryName() + frame.error())};
                draw.arguments[parameter] = frame.value()->lowered->frame.arguments[parameter];
                draw.owners[parameter] = frame.value()->owner;
                draw.lowered[parameter] = frame.value();
                continue;
            }
            if (!found && endpoint.initialPassParameter && *endpoint.initialPassParameter < nodeArguments.size()) {
                found = &nodeArguments[*endpoint.initialPassParameter];
                draw.owners[parameter] = nodeOwner(*endpoint.initialPassParameter);
                draw.nodeParameters.emplace_back(parameter, *endpoint.initialPassParameter);
            }
            if (!found)
                return Made<void>{err(entryName() + "no argument for Stage parameter " + std::to_string(parameter))};
            draw.arguments[parameter] = *found;
        }
        draw.drawCommand = entry.draw;
        if (streamed) {
            draw.drawCommand.instance_count = 1;
            draw.streamed = true;
            draw.stream = drawStream(entry.stream);
        }
        const VernonDynamicState *entryDynamic = dynamicState;
        if (entry.dynamic.struct_size) {
            // A rectangle the entry leaves empty is the pass's, and the pass's default is the render area.
            draw.dynamic = dynamicState ? *dynamicState : VernonDynamicState{};
            draw.dynamic.struct_size = sizeof(draw.dynamic);
            for (size_t axis = 0; axis < 4 && !dynamicState; ++axis)
                draw.dynamic.viewport[axis] = draw.dynamic.scissor[axis] = renderPass->render_area[axis];
            if (entry.dynamic.viewport[2] && entry.dynamic.viewport[3])
                std::copy_n(entry.dynamic.viewport, 4, draw.dynamic.viewport);
            if (entry.dynamic.scissor[2] && entry.dynamic.scissor[3])
                std::copy_n(entry.dynamic.scissor, 4, draw.dynamic.scissor);
            draw.dynamic.stencil_reference = entry.dynamic.stencil_reference;
            entryDynamic = &draw.dynamic;
        }
        if (auto made = planDraw(name, index, *renderPass, entryDynamic, pass, draw); made.isErr())
            return made;
    }
    if (auto copied = copyDebug(table->debug_commands, table->debug_command_count, pass.trailingDebug); copied.isErr())
        return copied;
    if (debugDepth)
        return Made<void>{err(name + "the pass table pushes debug groups it does not pop")};
    bound.release();
    return checkReads(name, pass);
}

// Plans draw `entry` of the pass, SIZE_MAX for the node's own draw. An argument a set frame lowered is the Stage's
// already; the call adapts the others.
Made<void> PassPlanner::planDraw(const std::string &pass, size_t entry, const VernonRenderPass &renderPass,
                                 const VernonDynamicState *dynamic, PlannedPass &planned, PlannedDraw &draw) {
    std::string error;
    if (!draw.program)
        return Made<void>{err(entryPrefix(pass, entry) + "the pass has no Program")};
    draw.views.resize(draw.arguments.size());
    draw.argumentPointers.resize(draw.arguments.size());
    const runtime::GraphicsDrawLayout &layout = draw.stage->drawLayout;
    for (uint32_t parameter = 0; parameter < draw.arguments.size(); ++parameter) {
        draw.argumentPointers[parameter] = &draw.arguments[parameter];
        const bool sampled = std::find(layout.sampledImages.begin(), layout.sampledImages.end(), parameter) !=
                             layout.sampledImages.end();
        if (const runtime::capi::InvocationSetFrame *frame = draw.lowered[parameter]) {
            if (sampled)
                runtime::addGraphicsScopeRead(planned.reads, draw.arguments[parameter].image.view,
                                              frame->lowered->planned[parameter].subresources);
            continue;
        }
        if (auto adapted = stageArgument(*draw.program, *draw.stage, *draw.resolved, parameter, 0,
                                         draw.arguments[parameter], draw.views[parameter]);
            adapted.isErr())
            return Made<void>{err(entryPrefix(pass, entry) + adapted.error())};
        runtime::PlannedGraphicsArgument argument;
        if (!runtime::planGraphicsArgument(draw.stage->bindingProjection, layout, parameter, draw.arguments[parameter],
                                           describeImage, &context_, argument, error))
            return Made<void>{err(entryPrefix(pass, entry) + error)};
        if (sampled)
            runtime::addGraphicsScopeRead(planned.reads, draw.arguments[parameter].image.view, argument.subresources);
    }
    if (!runtime::planGraphicsDraw(layout, draw.argumentPointers.data(), planned.targets, renderPass, dynamic,
                                   draw.drawCommand, draw.draw, error))
        return Made<void>{err(entryPrefix(pass, entry) + error)};
    return Made<void>{ok()};
}

} // namespace vernon::hal
