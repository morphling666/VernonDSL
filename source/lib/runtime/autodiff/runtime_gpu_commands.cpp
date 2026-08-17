#include "runtime/autodiff/runtime_gpu_commands.h"

#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"
#include "runtime_gpu_failure_injection.h"

#include <chrono>
#include <string>

namespace vernon::runtime::ad::gpu {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

bool encodeCopies(VernonRuntimeContext &context, VernonRhiCommandEncoder encoder,
                  const std::vector<DeviceBufferCopy> &copies) {
    for (const DeviceBufferCopy &copy : copies)
        if (injectFailure(FailureBoundary::Copy) ||
            vernonRhiCommandEncoderCopyBuffer(context.rhiDevice, encoder, copy.source, copy.offset, copy.destination,
                                              copy.offset, copy.size) != VERNON_RHI_STATUS_OK)
            return false;
    return true;
}

} // namespace

VernonStatus submitAndWait(VernonLoadedPipeline &pipeline, VernonLaunchSize grid,
                           std::vector<VernonPipelineArgument> &arguments, PullbackControlPlaneUsage *telemetry) {
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments.data();
    invocation.argument_count = arguments.size();
    invocation.compute_grid = grid;
    VernonSubmission *submission = nullptr;
    VernonStatus status =
        injectFailure(FailureBoundary::Submit)
            ? fail(*pipeline.context, "injected GPU pipeline submission failure", VERNON_STATUS_INTERNAL_ERROR)
            : vernonRuntimePipelineSubmit(&pipeline, &invocation, &submission);
    if (status == VERNON_STATUS_OK) {
        if (telemetry)
            ++telemetry->submissions;
        const auto waitStarted = std::chrono::steady_clock::now();
        if (telemetry)
            ++telemetry->waits;
        status = injectFailure(FailureBoundary::Wait)
                     ? fail(*pipeline.context, "injected GPU pipeline wait failure", VERNON_STATUS_INTERNAL_ERROR)
                     : vernonSubmissionWait(submission);
        if (telemetry)
            telemetry->deviceWaitNanoseconds += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStarted)
                    .count());
    }
    vernonSubmissionDestroy(submission);
    return status;
}

VernonStatus submitWithCopiesAndWait(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &copies,
                                     VernonLoadedPipeline &pipeline, std::vector<VernonPipelineArgument> &arguments,
                                     VernonLaunchSize grid, PullbackControlPlaneUsage *telemetry) {
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE;
    VernonRhiCommandEncoder encoder{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (injectFailure(FailureBoundary::Encode) ||
        vernonRhiDeviceCreateCommandEncoder(context.rhiDevice, &descriptor, &encoder) != VERNON_RHI_STATUS_OK)
        return fail(context, "cannot create GPU replay restore chain", VERNON_STATUS_INTERNAL_ERROR);
    VernonStatus status = encodeCopies(context, encoder, copies)
                              ? VERNON_STATUS_OK
                              : fail(context, "cannot encode GPU replay restore", VERNON_STATUS_INTERNAL_ERROR);
    VernonRuntimeProviderObject provider{};
    if (status == VERNON_STATUS_OK)
        status = referenceBackendCommandEncoder(context, encoder, provider);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments.data();
    invocation.argument_count = arguments.size();
    invocation.compute_grid = grid;
    if (status == VERNON_STATUS_OK)
        status = injectFailure(FailureBoundary::Encode)
                     ? fail(context, "injected GPU replay encode failure", VERNON_STATUS_INTERNAL_ERROR)
                     : vernonRuntimePipelineEncode(provider, &pipeline, &invocation);
    if (status == VERNON_STATUS_OK && vernonRhiCommandEncoderFinish(context.rhiDevice, encoder) != VERNON_RHI_STATUS_OK)
        status = fail(context, "cannot finish GPU replay restore chain", VERNON_STATUS_INTERNAL_ERROR);
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (status == VERNON_STATUS_OK && !injectFailure(FailureBoundary::Submit) &&
        vernonRhiDeviceSubmit(context.rhiDevice, encoder, &completion) != VERNON_RHI_STATUS_OK)
        status = fail(context, "cannot submit GPU replay restore chain", VERNON_STATUS_INTERNAL_ERROR);
    else if (status == VERNON_STATUS_OK && completion.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        status = fail(context, "injected GPU replay submission failure", VERNON_STATUS_INTERNAL_ERROR);
    if (status != VERNON_STATUS_OK) {
        (void)vernonRhiDeviceDestroyCommandEncoder(context.rhiDevice, encoder);
        return status;
    }
    if (telemetry)
        ++telemetry->submissions;
    const auto waitStarted = std::chrono::steady_clock::now();
    if (telemetry)
        ++telemetry->waits;
    const VernonRhiStatus wait = injectFailure(FailureBoundary::Wait)
                                     ? VERNON_RHI_STATUS_INTERNAL_ERROR
                                     : vernonRhiCompletionWait(context.rhiDevice, completion);
    if (telemetry)
        telemetry->deviceWaitNanoseconds += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStarted)
                .count());
    (void)vernonRhiDeviceDestroyCompletion(context.rhiDevice, completion);
    return wait == VERNON_RHI_STATUS_OK
               ? VERNON_STATUS_OK
               : fail(context, "GPU replay restore chain failed", VERNON_STATUS_INTERNAL_ERROR);
}

} // namespace vernon::runtime::ad::gpu
