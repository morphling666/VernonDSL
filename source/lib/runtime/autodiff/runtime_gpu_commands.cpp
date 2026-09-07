#include "runtime/program_execution/device_commands.h"

#include "execution_graph/execution_graph_internal.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <memory>
#include <string>

namespace vernon::runtime::program_execution {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

execution::detail::CommandResourceAccess bufferAccess(VernonRhiBuffer buffer, uint64_t offset, uint64_t size,
                                                      execution::AccessMode access) {
    return execution::detail::rhiBufferAccess(buffer, offset, size, access,
                                              access == execution::AccessMode::Read
                                                  ? VERNON_RHI_STATE_TRANSFER_SOURCE
                                                  : VERNON_RHI_STATE_TRANSFER_DESTINATION);
}

execution::detail::CommandResourceAccess shaderBufferAccess(VernonRhiBuffer buffer, uint64_t offset, uint64_t size,
                                                            execution::AccessMode access) {
    execution::detail::CommandResourceAccess result = bufferAccess(buffer, offset, size, access);
    result.state = access == execution::AccessMode::Read ? VERNON_RHI_STATE_SHADER_READ : VERNON_RHI_STATE_SHADER_WRITE;
    result.stageMask = VERNON_RHI_STAGE_COMPUTE;
    return result;
}

void appendCopyBindings(std::vector<execution::detail::RhiCommandResourceBinding> &bindings,
                        const std::vector<DeviceBufferCopy> &copies) {
    for (const DeviceBufferCopy &copy : copies) {
        execution::detail::appendRhiBufferBinding(bindings, copy.source);
        execution::detail::appendRhiBufferBinding(bindings, copy.destination);
    }
}

void appendUploadBindings(std::vector<execution::detail::RhiCommandResourceBinding> &bindings,
                          const std::vector<DeviceBufferUpload> &uploads) {
    for (const DeviceBufferUpload &upload : uploads)
        execution::detail::appendRhiBufferBinding(bindings, upload.destination);
}

void appendCopyAccesses(execution::detail::CommandNode &node, const std::vector<DeviceBufferCopy> &copies) {
    node.accesses.reserve(copies.size() * 2);
    for (const DeviceBufferCopy &copy : copies) {
        node.accesses.push_back(bufferAccess(copy.source, copy.sourceOffset, copy.size, execution::AccessMode::Read));
        node.accesses.push_back(
            bufferAccess(copy.destination, copy.destinationOffset, copy.size, execution::AccessMode::Write));
    }
}

void appendUploadAccesses(execution::detail::CommandNode &node, const std::vector<DeviceBufferUpload> &uploads) {
    for (const DeviceBufferUpload &upload : uploads)
        node.accesses.push_back(
            bufferAccess(upload.destination, upload.destinationOffset, upload.size, execution::AccessMode::Write));
}

bool decodeBuffer(uint64_t key, VernonRhiBuffer &buffer) {
    const uint32_t encodedIndex = static_cast<uint32_t>(key);
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    buffer = {encodedIndex - 1, generation};
    return true;
}

bool appendPipelineResources(execution::detail::CommandNode &node,
                             std::vector<execution::detail::RhiCommandResourceBinding> &bindings,
                             const std::vector<VernonProgramArgument> &arguments) {
    for (const VernonProgramArgument &argument : arguments) {
        if (argument.kind != VERNON_PROGRAM_TENSOR)
            continue;
        const VernonTensorView &tensor = argument.tensor;
        if (tensor.storage != VERNON_TENSOR_RHI_RESOURCE)
            continue;
        VernonRhiBuffer buffer{};
        if (!tensor.byte_size || tensor.resource.offset > UINT64_MAX - tensor.byte_offset ||
            !decodeBuffer(tensor.resource.resource.value, buffer))
            return false;
        execution::AccessMode access = execution::AccessMode::Read;
        if (tensor.access == VERNON_ACCESS_WRITE)
            access = execution::AccessMode::Write;
        else if (tensor.access == VERNON_ACCESS_READ_WRITE)
            access = execution::AccessMode::ReadWrite;
        node.accesses.push_back(
            shaderBufferAccess(buffer, tensor.resource.offset + tensor.byte_offset, tensor.byte_size, access));
        execution::detail::appendRhiBufferBinding(bindings, buffer);
    }
    return true;
}

struct TransferCommandContext {
    TransferCommandContext(VernonRuntimeContext &runtimeValue, const std::vector<DeviceBufferCopy> &copyValues,
                           const std::vector<DeviceBufferUpload> &uploadValues)
        : runtime(runtimeValue), copies(copyValues), uploads(uploadValues) {
        uploadBytes.reserve(uploads.size());
        for (DeviceBufferUpload &upload : uploads) {
            const auto *begin = static_cast<const uint8_t *>(upload.source);
            uploadBytes.emplace_back(begin, begin + upload.size);
            upload.source = uploadBytes.back().data();
        }
    }

    VernonRuntimeContext &runtime;
    std::vector<DeviceBufferCopy> copies;
    std::vector<DeviceBufferUpload> uploads;
    std::vector<std::vector<uint8_t>> uploadBytes;
};

struct PipelineCommandContext {
    PipelineCommandContext(VernonRuntimeContext &runtimeValue, VernonStageExecutable &pipelineValue,
                           const std::vector<VernonProgramArgument> &argumentValues, VernonLaunchSize gridValue)
        : runtime(runtimeValue), pipeline(pipelineValue), arguments(argumentValues), grid(gridValue) {
        shapes.resize(arguments.size());
        strides.resize(arguments.size());
        for (size_t index = 0; index < arguments.size(); ++index) {
            VernonProgramArgument &argument = arguments[index];
            if (argument.kind != VERNON_PROGRAM_TENSOR || !argument.tensor.rank)
                continue;
            shapes[index].assign(argument.tensor.shape, argument.tensor.shape + argument.tensor.rank);
            strides[index].assign(argument.tensor.byte_strides, argument.tensor.byte_strides + argument.tensor.rank);
            argument.tensor.shape = shapes[index].data();
            argument.tensor.byte_strides = strides[index].data();
        }
    }

    VernonRuntimeContext &runtime;
    VernonStageExecutable &pipeline;
    std::vector<VernonProgramArgument> arguments;
    VernonLaunchSize grid;
    std::vector<std::vector<uint64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
};

} // namespace

bool encodeBufferCopies(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                        const std::vector<DeviceBufferCopy> &copies) {
    for (size_t index = 0; index < copies.size(); ++index) {
        const DeviceBufferCopy &copy = copies[index];
        if (injectFailure(FailureBoundary::Copy) ||
            vernonRhiCommandEncoderCopyBuffer(context.rhiDevice, encoder, copy.source, copy.sourceOffset,
                                              copy.destination, copy.destinationOffset,
                                              copy.size) != VERNON_RHI_STATUS_OK) {
            invocationDiagnostic(context) = "GPU buffer copy " + std::to_string(index) + " failed (source offset " +
                                            std::to_string(copy.sourceOffset) + ", destination offset " +
                                            std::to_string(copy.destinationOffset) + ", size " +
                                            std::to_string(copy.size) + ")";
            return false;
        }
    }
    return true;
}

bool encodeBufferUploads(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                         const std::vector<DeviceBufferUpload> &uploads) {
    for (const DeviceBufferUpload &upload : uploads)
        if (!upload.source || !upload.size || injectFailure(FailureBoundary::Upload) ||
            vernonRhiCommandEncoderUploadBuffer(context.rhiDevice, encoder, upload.destination,
                                                upload.destinationOffset, upload.source,
                                                upload.size) != VERNON_RHI_STATUS_OK)
            return false;
    return true;
}

namespace {

VernonRhiStatus encodeTransferCommand(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &state = *static_cast<TransferCommandContext *>(opaque);
    return encodeBufferCopies(state.runtime, encoder, state.copies) &&
                   encodeBufferUploads(state.runtime, encoder, state.uploads)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus encodePipelinePlanCommand(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &state = *static_cast<PipelineCommandContext *>(opaque);
    return encodePipelineCommand(state.runtime, encoder, state.pipeline, state.arguments, state.grid) ==
                   VERNON_STATUS_OK
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus encodeNoopCommand(void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }

class WaitFailureCompletion final : public execution::detail::RhiCommandCompletion {
public:
    explicit WaitFailureCompletion(VernonRuntimeContext &context) : context_(context) {}

    bool validationPhase() const override { return true; }

    void complete(bool succeeded, const execution::detail::RhiCommandDagExecutionStats &) override {
        if (!succeeded || !injectFailure(FailureBoundary::Wait))
            return;
        invocationDiagnostic(context_) = "injected GPU command DAG wait failure";
        throw std::runtime_error("injected GPU command DAG wait failure");
    }

private:
    VernonRuntimeContext &context_;
};

struct EncodeFailureContext {
    VernonRuntimeContext &runtime;
    execution::detail::RhiCommandNodeEncoder original;
};

VernonRhiStatus encodeWithFailureBoundary(void *opaque, VernonRhiCommandEncoder encoder) {
    auto &context = *static_cast<EncodeFailureContext *>(opaque);
    if (injectFailure(FailureBoundary::Encode)) {
        invocationDiagnostic(context.runtime) = "injected GPU command DAG encoding failure";
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return context.original.encode(context.original.context, encoder);
}

} // namespace

VernonStatus encodePipelineCommand(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                                   VernonStageExecutable &pipeline, std::vector<VernonProgramArgument> &arguments,
                                   VernonLaunchSize grid) {
    VernonRuntimeProviderObject provider{};
    if (referenceBackendCommandEncoder(context, encoder, provider) != VERNON_STATUS_OK)
        return fail(context, "cannot reference GPU autodiff command encoder", VERNON_STATUS_INTERNAL_ERROR);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = arguments.data();
    invocation.argument_count = arguments.size();
    invocation.compute_grid = grid;
    return vernonRuntimeStageEncode(provider, &pipeline, &invocation);
}

VernonStatus executeCommandPlanAndWait(VernonRuntimeContext &context,
                                       const execution::detail::RhiCommandExecutionPlan &plan,
                                       ExecutionControlPlaneUsage *telemetry,
                                       execution::detail::RhiCommandPlanSink *sink, bool flush) {
    if (sink) {
        execution::detail::RhiCommandExecutionPlan deferred = plan;
        const auto firstCommand =
            std::find_if(deferred.encoders.begin(), deferred.encoders.end(),
                         [](const execution::detail::RhiCommandNodeEncoder &encoder) { return encoder.encode; });
        if (firstCommand == deferred.encoders.end())
            return fail(context, "GPU autodiff command program has no encodable command", VERNON_STATUS_INTERNAL_ERROR);
        auto encodeContext = std::make_shared<EncodeFailureContext>(EncodeFailureContext{context, *firstCommand});
        *firstCommand = {encodeWithFailureBoundary, encodeContext.get(), nullptr};
        deferred.retainedContexts.push_back(std::move(encodeContext));
        if (sink->append(std::move(deferred)) != VERNON_RHI_STATUS_OK)
            return fail(context, "GPU autodiff command program failed", VERNON_STATUS_INTERNAL_ERROR);
        sink->onCompletion(std::make_shared<WaitFailureCompletion>(context));
        if (flush && sink->flush() != VERNON_RHI_STATUS_OK)
            return fail(context, "GPU autodiff command program failed", VERNON_STATUS_INTERNAL_ERROR);
        return VERNON_STATUS_OK;
    }
    if (injectFailure(FailureBoundary::Encode))
        return fail(context, "injected GPU command DAG encoding failure", VERNON_STATUS_INTERNAL_ERROR);
    const uint32_t requiredCapabilities =
        std::any_of(plan.commands.nodes.begin(), plan.commands.nodes.end(),
                    [](const auto &node) { return node.queue == execution::detail::CommandQueueClass::Graphics; })
            ? VERNON_RHI_QUEUE_GRAPHICS
            : VERNON_RHI_QUEUE_COMPUTE;
    execution::detail::RhiCommandDagExecutionStats stats;
    const VernonRhiStatus status =
        execution::detail::executeRhiCommandPlanAndWait(context.rhiDevice, requiredCapabilities, plan, &stats);
    if (telemetry) {
        telemetry->submissions += stats.submissions;
        telemetry->waits += stats.waits;
        telemetry->deviceWaitNanoseconds += stats.deviceWaitNanoseconds;
    }
    if (status == VERNON_RHI_STATUS_OK && injectFailure(FailureBoundary::Wait))
        return fail(context, "injected GPU command DAG wait failure", VERNON_STATUS_INTERNAL_ERROR);
    if (status == VERNON_RHI_STATUS_OK)
        return VERNON_STATUS_OK;
    const std::string detail = invocationDiagnostic(context);
    return fail(context, detail.empty() ? "GPU autodiff command program failed" : detail, VERNON_STATUS_INTERNAL_ERROR);
}

VernonStatus executeBufferCopiesAndWait(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copies) {
    if (copies.empty())
        return VERNON_STATUS_OK;
    execution::detail::RhiCommandExecutionPlan plan;
    appendCopyBindings(plan.bindings, copies);
    auto transferContext = std::make_shared<TransferCommandContext>(context, copies, std::vector<DeviceBufferUpload>{});
    execution::detail::CommandNode transfer;
    transfer.kind = execution::detail::CommandNodeKind::Transfer;
    transfer.queue = execution::detail::CommandQueueClass::Transfer;
    appendCopyAccesses(transfer, copies);
    plan.commands.nodes.push_back(std::move(transfer));
    plan.encoders.push_back({encodeTransferCommand, transferContext.get()});
    plan.retainedContexts.push_back(std::move(transferContext));
    std::string error;
    if (!execution::detail::validateRhiCommandExecutionPlan(plan, error))
        return fail(context, std::move(error));
    return executeCommandPlanAndWait(context, plan);
}

VernonStatus buildBufferTransferCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copies,
                                            const std::vector<DeviceBufferUpload> &uploads,
                                            execution::detail::RhiCommandExecutionPlan &plan) {
    plan = {};
    if (copies.empty() && uploads.empty())
        return VERNON_STATUS_OK;
    for (const DeviceBufferCopy &copy : copies)
        if (!copy.size)
            return fail(context, "Program Storage copy has no payload");
    for (const DeviceBufferUpload &upload : uploads)
        if (!upload.source || !upload.size)
            return fail(context, "Program Storage upload has no source bytes");
    appendCopyBindings(plan.bindings, copies);
    appendUploadBindings(plan.bindings, uploads);
    auto transferContext = std::make_shared<TransferCommandContext>(context, copies, uploads);
    execution::detail::CommandNode transfer;
    transfer.kind = execution::detail::CommandNodeKind::Transfer;
    transfer.queue = execution::detail::CommandQueueClass::Transfer;
    appendCopyAccesses(transfer, copies);
    appendUploadAccesses(transfer, uploads);
    plan.commands.nodes.push_back(std::move(transfer));
    plan.encoders.push_back({encodeTransferCommand, transferContext.get()});
    plan.retainedContexts.push_back(std::move(transferContext));
    return VERNON_STATUS_OK;
}

VernonStatus buildBufferUploadCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferUpload> &uploads,
                                          execution::detail::RhiCommandExecutionPlan &plan) {
    return buildBufferTransferCommandPlan(context, {}, uploads, plan);
}

VernonStatus executePipelineCommandDagAndWait(VernonStageExecutable &pipeline, VernonLaunchSize grid,
                                              std::vector<VernonProgramArgument> &arguments,
                                              const std::vector<DeviceBufferUpload> &uploadsBefore,
                                              execution::detail::CommandNodeKind kind,
                                              ExecutionControlPlaneUsage *telemetry,
                                              execution::detail::RhiCommandPlanSink *sink) {
    return executePipelineCommandDagAndWait(*pipeline.context, {}, uploadsBefore, pipeline, arguments, grid, {}, kind,
                                            telemetry, sink);
}

VernonStatus buildPipelineCommandPlan(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore,
                                      const std::vector<DeviceBufferUpload> &uploadsBefore,
                                      VernonStageExecutable &pipeline,
                                      const std::vector<VernonProgramArgument> &arguments, VernonLaunchSize grid,
                                      const std::vector<DeviceBufferCopy> &copiesAfter,
                                      execution::detail::CommandNodeKind kind,
                                      execution::detail::RhiCommandExecutionPlan &plan) {
    plan = {};
    for (const DeviceBufferUpload &upload : uploadsBefore)
        if (!upload.source || !upload.size)
            return fail(context, "GPU autodiff upload has no source bytes");
    for (const VernonProgramArgument &argument : arguments)
        if (argument.kind == VERNON_PROGRAM_TENSOR && argument.tensor.rank &&
            (!argument.tensor.shape || !argument.tensor.byte_strides))
            return fail(context, "GPU autodiff Tensor argument has incomplete layout metadata");
    appendCopyBindings(plan.bindings, copiesBefore);
    appendCopyBindings(plan.bindings, copiesAfter);
    appendUploadBindings(plan.bindings, uploadsBefore);
    if (!copiesBefore.empty() || !uploadsBefore.empty()) {
        auto transferContext = std::make_shared<TransferCommandContext>(context, copiesBefore, uploadsBefore);
        execution::detail::CommandNode transfer;
        transfer.kind = execution::detail::CommandNodeKind::Transfer;
        transfer.queue = execution::detail::CommandQueueClass::Transfer;
        appendCopyAccesses(transfer, copiesBefore);
        appendUploadAccesses(transfer, uploadsBefore);
        plan.commands.nodes.push_back(std::move(transfer));
        plan.encoders.push_back({encodeTransferCommand, transferContext.get()});
        plan.retainedContexts.push_back(std::move(transferContext));
    }
    auto pipelineContext = std::make_shared<PipelineCommandContext>(context, pipeline, arguments, grid);
    execution::detail::CommandNode derivative;
    derivative.kind = kind;
    derivative.queue = execution::detail::CommandQueueClass::Compute;
    if (!appendPipelineResources(derivative, plan.bindings, pipelineContext->arguments))
        return fail(context, "GPU autodiff pipeline has an invalid RHI resource argument");
    if (!plan.commands.nodes.empty())
        derivative.predecessors.push_back(static_cast<uint32_t>(plan.commands.nodes.size() - 1));
    plan.commands.nodes.push_back(std::move(derivative));
    plan.encoders.push_back({encodePipelinePlanCommand, pipelineContext.get()});
    plan.retainedContexts.push_back(std::move(pipelineContext));
    if (!copiesAfter.empty()) {
        auto transferContext =
            std::make_shared<TransferCommandContext>(context, copiesAfter, std::vector<DeviceBufferUpload>{});
        execution::detail::CommandNode transfer;
        transfer.kind = execution::detail::CommandNodeKind::Transfer;
        transfer.queue = execution::detail::CommandQueueClass::Transfer;
        transfer.predecessors.push_back(static_cast<uint32_t>(plan.commands.nodes.size() - 1));
        appendCopyAccesses(transfer, copiesAfter);
        plan.commands.nodes.push_back(std::move(transfer));
        plan.encoders.push_back({encodeTransferCommand, transferContext.get()});
        plan.retainedContexts.push_back(std::move(transferContext));
        execution::detail::CommandNode finalState;
        finalState.kind = execution::detail::CommandNodeKind::Transfer;
        finalState.queue = execution::detail::CommandQueueClass::Ordered;
        finalState.predecessors.push_back(static_cast<uint32_t>(plan.commands.nodes.size() - 1));
        for (const DeviceBufferCopy &copy : copiesAfter)
            finalState.accesses.push_back(shaderBufferAccess(copy.destination, copy.destinationOffset, copy.size,
                                                             execution::AccessMode::ReadWrite));
        plan.commands.nodes.push_back(std::move(finalState));
        plan.encoders.push_back({encodeNoopCommand, nullptr});
    }
    std::string error;
    if (!execution::detail::validateRhiCommandExecutionPlan(plan, error))
        return fail(context, std::move(error));
    return VERNON_STATUS_OK;
}

VernonStatus
executePipelineCommandDagAndWait(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore,
                                 const std::vector<DeviceBufferUpload> &uploadsBefore, VernonStageExecutable &pipeline,
                                 std::vector<VernonProgramArgument> &arguments, VernonLaunchSize grid,
                                 const std::vector<DeviceBufferCopy> &copiesAfter,
                                 execution::detail::CommandNodeKind kind, ExecutionControlPlaneUsage *telemetry,
                                 execution::detail::RhiCommandPlanSink *sink) {
    if (injectFailure(FailureBoundary::Submit))
        return fail(context, "injected GPU pipeline submission failure", VERNON_STATUS_INTERNAL_ERROR);
    execution::detail::RhiCommandExecutionPlan plan;
    if (const VernonStatus status = buildPipelineCommandPlan(context, copiesBefore, uploadsBefore, pipeline, arguments,
                                                             grid, copiesAfter, kind, plan);
        status != VERNON_STATUS_OK)
        return status;
    return executeCommandPlanAndWait(context, plan, telemetry, sink);
}

VernonStatus
executePipelineStatusCommandDagAndWait(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copiesBefore,
                                       VernonStageExecutable &pipeline, std::vector<VernonProgramArgument> &arguments,
                                       VernonLaunchSize grid, const std::vector<DeviceBufferUpload> &uploadsBefore,
                                       VernonRhiBuffer statusBuffer, size_t statusOffset, size_t statusSize,
                                       GpuCommandCompletionCallback complete, void *completionContext,
                                       execution::detail::CommandNodeKind kind, ExecutionControlPlaneUsage *telemetry,
                                       execution::detail::RhiCommandPlanSink *sink) {
    if (!complete || !statusSize)
        return fail(context, "GPU command DAG status callback is invalid");
    if (injectFailure(FailureBoundary::Submit))
        return fail(context, "injected GPU pipeline submission failure", VERNON_STATUS_INTERNAL_ERROR);
    execution::detail::RhiCommandExecutionPlan plan;
    if (const VernonStatus status =
            buildPipelineCommandPlan(context, copiesBefore, uploadsBefore, pipeline, arguments, grid, {}, kind, plan);
        status != VERNON_STATUS_OK)
        return status;
    execution::detail::CommandNode status;
    status.kind = execution::detail::CommandNodeKind::Status;
    status.queue = execution::detail::CommandQueueClass::Ordered;
    status.predecessors.push_back(static_cast<uint32_t>(plan.commands.nodes.size() - 1));
    status.accesses.push_back(bufferAccess(statusBuffer, statusOffset, statusSize, execution::AccessMode::Read));
    plan.commands.nodes.push_back(std::move(status));
    plan.encoders.push_back({nullptr, completionContext, complete});
    execution::detail::appendRhiBufferBinding(plan.bindings, statusBuffer);
    return executeCommandPlanAndWait(context, plan, telemetry, sink, true);
}

} // namespace vernon::runtime::program_execution
