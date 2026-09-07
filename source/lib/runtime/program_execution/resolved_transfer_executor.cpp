#include "resolved_transfer_executor.h"

#include "device_commands.h"
#include "execution_graph/execution_graph_internal.h"
#include "program_tensor_copy.h"
#include "runtime/runtime_dispatch.h"

#include <algorithm>
#include <unordered_map>

namespace vernon::runtime::program_execution {

bool ResolvedTransferExecutor::prepareGraph(program::GraphDirection graph, std::string &error) {
    preparedGraph_ = graph;
    prepared_ = true;
    state_.deviceUploads_.clear();
    initialCopies_.clear();
    struct HostBacking {
        const void *data{};
        size_t bytes{};
        bool stagedDeviceInitial{};
        std::shared_ptr<DeviceBuffer> device;
    };
    std::unordered_map<uintptr_t, HostBacking> backings;
    for (size_t value = 0; value < state_.arguments_.size(); ++value) {
        if (!state_.plan_->requiresDevice(graph, static_cast<uint32_t>(value)))
            continue;
        const VernonProgramArgument &argument = state_.arguments_[value];
        if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
            !argument.tensor.host_data || !argument.tensor.byte_size)
            continue;
        HostBacking &backing = backings[reinterpret_cast<uintptr_t>(argument.tensor.host_data)];
        backing.data = argument.tensor.host_data;
        backing.bytes = std::max(backing.bytes, argument.tensor.byte_offset + argument.tensor.byte_size);
        backing.stagedDeviceInitial |=
            value < state_.values_.size() && state_.values_[value].stagedDeviceInitial.has_value();
    }
    for (auto &[identity, backing] : backings) {
        (void)identity;
        backing.device = std::make_shared<DeviceBuffer>(context_, backing.bytes);
        if (!backing.device->valid())
            return error = "resolved transfer cannot allocate device Storage", false;
    }
    for (size_t value = 0; value < state_.arguments_.size(); ++value) {
        if (!state_.plan_->requiresDevice(graph, static_cast<uint32_t>(value)))
            continue;
        VernonProgramArgument &argument = state_.arguments_[value];
        if (argument.kind != VERNON_PROGRAM_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
            !argument.tensor.host_data || !argument.tensor.byte_size)
            continue;
        const auto found = backings.find(reinterpret_cast<uintptr_t>(argument.tensor.host_data));
        if (found == backings.end())
            return error = "resolved transfer lost Storage alias backing", false;
        state_.deviceValues_[value] = found->second.device;
        VernonRuntimeProviderResourceReference reference{};
        if (!found->second.device->reference(reference))
            return error = "resolved transfer cannot reference device Storage", false;
        argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
        argument.tensor.resource = reference;
        if (!found->second.stagedDeviceInitial)
            state_.deviceUploads_.push_back({static_cast<uint32_t>(value), found->second.device->handle(),
                                             found->second.data, found->second.bytes});
    }
    for (size_t value = 0; value < state_.values_.size(); ++value) {
        const auto &initial = state_.values_[value].stagedDeviceInitial;
        if (!initial || value >= state_.deviceValues_.size() || !state_.deviceValues_[value])
            continue;
        VernonRhiBuffer source{};
        if (initial->retainedSourceBuffer)
            source = *initial->retainedSourceBuffer;
        else if (!resolveBackendRhiBufferReference(context_, initial->source, source))
            return error = "staged device input source for Value " + std::to_string(value) + " is no longer retained",
                   false;
        if (initial->byteSize > state_.deviceValues_[value]->size())
            return error = "staged device input exceeds fresh apply Storage", false;
        VernonTensorView sourceView{};
        sourceView.struct_size = sizeof(VernonTensorView);
        sourceView.storage = VERNON_TENSOR_RHI_RESOURCE;
        sourceView.resource = initial->source;
        sourceView.byte_offset = initial->byteOffset;
        sourceView.byte_size = initial->byteSize;
        sourceView.element_layout = initial->elementLayout;
        sourceView.rank = static_cast<uint32_t>(initial->shape.size());
        sourceView.shape = initial->shape.empty() ? nullptr : initial->shape.data();
        sourceView.byte_strides = initial->strides.empty() ? nullptr : initial->strides.data();
        std::vector<ProgramTensorCopyRegion> regions;
        std::string copyError;
        const VernonTensorView &destinationView = state_.values_[value].argument.tensor;
        if (!planProgramTensorCopy(sourceView, destinationView, regions, copyError)) {
            error = "staged device input Value " + std::to_string(value) + ": " + copyError + " (source rank " +
                    std::to_string(sourceView.rank) + ", destination rank " + std::to_string(destinationView.rank) +
                    ", source element bytes " + std::to_string(sourceView.element_layout.byte_size) +
                    ", destination element bytes " + std::to_string(destinationView.element_layout.byte_size) + ")";
            return false;
        }
        for (const ProgramTensorCopyRegion &region : regions)
            initialCopies_.push_back({source, state_.deviceValues_[value]->handle(),
                                      initial->source.offset + region.sourceOffset, region.destinationOffset,
                                      region.size});
    }
    initialCopiesPending_ = !state_.deviceUploads_.empty() || !initialCopies_.empty();
    return true;
}

VernonStatus ResolvedTransferExecutor::appendBeforeConsumer(
    program::NodeKey consumer, const std::vector<DeviceBufferCopy> &physicalCopies,
    vernon::execution::detail::RhiCommandExecutionPlan &commands, std::string &error) {
    if (!prepared_ || consumer.graph != preparedGraph_)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const bool plannedConsumer = std::any_of(state_.plan_->transfers.edges.begin(), state_.plan_->transfers.edges.end(),
                                             [&](const program::ResolvedTransferEdge &edge) {
                                                 return edge.consumer.kind == program::TransferEndpointKind::Node &&
                                                        edge.consumer.graph == consumer.graph &&
                                                        edge.consumer.id == consumer.node;
                                             });
    const bool appendInitial = initialCopiesPending_ && plannedConsumer;
    if (!appendInitial && physicalCopies.empty())
        return VERNON_STATUS_OK;
    std::vector<DeviceBufferUpload> uploads;
    std::vector<VernonRhiBuffer> uploaded;
    if (appendInitial)
        for (const ProgramDeviceUpload &upload : state_.deviceUploads_) {
            if (std::find_if(uploaded.begin(), uploaded.end(), [&](const VernonRhiBuffer &buffer) {
                    return buffer.index == upload.destination.index &&
                           buffer.generation == upload.destination.generation;
                }) != uploaded.end())
                continue;
            uploads.push_back({upload.destination, 0, upload.source, upload.size});
            uploaded.push_back(upload.destination);
        }
    vernon::execution::detail::RhiCommandExecutionPlan transfers;
    const VernonStatus status =
        buildBufferTransferCommandPlan(context_, appendInitial ? initialCopies_ : physicalCopies, uploads, transfers);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!transfers.commands.nodes.empty() &&
        !vernon::execution::detail::appendRhiCommandExecutionPlan(commands, std::move(transfers), true, error))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (appendInitial)
        initialCopiesPending_ = false;
    if (appendInitial && !physicalCopies.empty()) {
        vernon::execution::detail::RhiCommandExecutionPlan physical;
        const VernonStatus physicalStatus = buildBufferTransferCommandPlan(context_, physicalCopies, {}, physical);
        if (physicalStatus != VERNON_STATUS_OK)
            return physicalStatus;
        if (!physical.commands.nodes.empty() &&
            !vernon::execution::detail::appendRhiCommandExecutionPlan(commands, std::move(physical), true, error))
            return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
}

bool ResolvedTransferExecutor::restoreForRetry(program::GraphDirection graph, std::string &error) {
    return state_.restoreDeviceValuesFromHost(graph, error);
}

bool ResolvedTransferExecutor::readbackBoundaryValues(const std::vector<char> &required, std::string &error) const {
    if (required.size() != state_.arguments_.size())
        return error = "boundary readback set does not match invocation Values", false;
    for (size_t value = 0; value < required.size(); ++value) {
        if (!required[value] || !state_.deviceValues_[value])
            continue;
        const ProgramValueState &host = state_.values_[value];
        if (host.argument.kind != VERNON_PROGRAM_TENSOR || !host.argument.tensor.host_data ||
            !state_.deviceValues_[value]->download(host.argument.tensor.byte_offset,
                                                   const_cast<void *>(host.argument.tensor.host_data),
                                                   host.argument.tensor.byte_size))
            return error = "Program boundary readback failed", false;
    }
    return true;
}

} // namespace vernon::runtime::program_execution
