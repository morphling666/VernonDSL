#include "publication_executor.h"

#include "execution_graph/execution_graph_internal.h"
#include "failure_injection.h"
#include "runtime/runtime_state.h"

namespace vernon::runtime::program_execution {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INTERNAL_ERROR) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

VernonStatus executeCopies(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &buffers,
                           const std::vector<DeviceImageCopy> &images, std::string &error) {
    if (buffers.empty() && images.empty())
        return VERNON_STATUS_OK;
    execution::detail::RhiCommandExecutionPlan plan;
    VernonStatus status = buildDeviceTransferCommandPlan(context, buffers, images, {}, plan);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!execution::detail::validateRhiCommandExecutionPlan(plan, error))
        return fail(context, error, VERNON_STATUS_INVALID_ARGUMENT);
    return executeCommandPlanAndWait(context, plan);
}

} // namespace

VernonStatus executePublicationInitialization(VernonRuntimeContext &context, const PublicationTransaction &transaction,
                                              std::string &error) {
    return executeCopies(context, {}, transaction.initializationImageCopies(), error);
}

VernonStatus executePublicationCommit(VernonRuntimeContext &context, PublicationTransaction &transaction,
                                      const ProgramInvocationState &state, std::string &error) {
    std::vector<DeviceBufferCopy> buffers;
    std::vector<DeviceImageCopy> images;
    VernonStatus status = transaction.prepareCommit(state, buffers, images, error);
    if (status != VERNON_STATUS_OK)
        return status;
    if (injectFailure(FailureBoundary::Submission)) {
        transaction.rollback();
        return fail(context, "publication submission failed before destination mutation");
    }
    status = executeCopies(context, buffers, images, error);
    if (status != VERNON_STATUS_OK) {
        transaction.poison();
        if (error.empty())
            error = invocationDiagnostic(context);
        return status;
    }
    return transaction.completeCommit(error);
}

} // namespace vernon::runtime::program_execution
