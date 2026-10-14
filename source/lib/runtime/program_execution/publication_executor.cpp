#include "publication_executor.h"

#include "execution_graph/execution_graph_internal.h"
#include "failure_injection.h"
#include "runtime/runtime_state.h"
#include "submission_record.h"

namespace vernon::runtime::program_execution {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string message,
                  VernonStatus status = VERNON_STATUS_INTERNAL_ERROR) {
    invocationDiagnostic(context) = std::move(message);
    return status;
}

VernonStatus executeCopies(VernonRuntimeContext &context, const std::vector<DeviceBufferCopy> &buffers,
                           const std::vector<DeviceImageCopy> &images, SubmissionRecord &submission, std::string &error,
                           SubmissionState &step) {
    step = SubmissionState::NotSubmitted;
    if (buffers.empty() && images.empty())
        return VERNON_STATUS_OK;
    execution::detail::RhiCommandExecutionPlan plan;
    VernonStatus status = buildDeviceTransferCommandPlan(context, buffers, images, {}, plan);
    if (status != VERNON_STATUS_OK)
        return status;
    if (!execution::detail::validateRhiCommandExecutionPlan(plan, error))
        return fail(context, error, VERNON_STATUS_INVALID_ARGUMENT);
    return submission.execute(std::move(plan), step);
}

} // namespace

VernonStatus executePublicationInitialization(VernonRuntimeContext &context, const PublicationTransaction &transaction,
                                              SubmissionRecord &submission, std::string &error) {
    SubmissionState step;
    return executeCopies(context, {}, transaction.initializationImageCopies(), submission, error, step);
}

VernonStatus executePublicationCommit(VernonRuntimeContext &context, PublicationTransaction &transaction,
                                      const ProgramInvocationState &state, SubmissionRecord &submission,
                                      std::string &error) {
    std::vector<DeviceBufferCopy> buffers;
    std::vector<DeviceImageCopy> images;
    auto prepared = transaction.prepareCommit(state, buffers, images);
    if (prepared.isErr()) {
        error = publicationErrorMessage(prepared.error());
        return publicationErrorStatus(prepared.error());
    }
    if (injectFailure(FailureBoundary::Submission)) {
        transaction.rollback();
        return fail(context, "publication submission failed before destination mutation");
    }
    SubmissionState step;
    VernonStatus status = executeCopies(context, buffers, images, submission, error, step);
    transaction.noteSubmission(step);
    if (status != VERNON_STATUS_OK) {
        if (step == SubmissionState::Indeterminate)
            transaction.poison();
        else
            transaction.rollback();
        if (error.empty())
            error = invocationDiagnostic(context);
        return status;
    }
    auto committed = transaction.completeCommit();
    if (committed.isErr()) {
        error = publicationErrorMessage(committed.error());
        return publicationErrorStatus(committed.error());
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime::program_execution
