#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_EXECUTOR_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_EXECUTOR_H

#include "publication_transaction.h"

namespace vernon::runtime::program_execution {

class SubmissionRecord;

VernonStatus executePublicationInitialization(VernonRuntimeContext &context, const PublicationTransaction &transaction,
                                              SubmissionRecord &submission, std::string &error);
VernonStatus executePublicationCommit(VernonRuntimeContext &context, PublicationTransaction &transaction,
                                      const ProgramInvocationState &state, SubmissionRecord &submission,
                                      std::string &error);

} // namespace vernon::runtime::program_execution

#endif
