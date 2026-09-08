#ifndef VERNON_TESTS_SUPPORT_RESOLVED_STAGE_TEST_UTILS_H
#define VERNON_TESTS_SUPPORT_RESOLVED_STAGE_TEST_UTILS_H

#include "runtime/resolved_stage_invocation.h"

namespace vernon::tests {

inline VernonStatus completeResolvedStageSubmission(VernonStageExecutable *stage,
                                                    const VernonStageInvocationDescriptor *invocation) {
    VernonSubmission *submission{};
    const VernonStatus submitStatus = runtime::submitResolvedStage(stage, invocation, &submission);
    if (submitStatus != VERNON_STATUS_OK)
        return submitStatus;
    const VernonStatus completionStatus = vernonSubmissionWait(submission);
    vernonSubmissionDestroy(submission);
    return completionStatus;
}

} // namespace vernon::tests

#endif
