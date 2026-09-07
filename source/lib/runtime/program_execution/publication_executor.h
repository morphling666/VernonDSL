#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_EXECUTOR_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_PUBLICATION_EXECUTOR_H

#include "publication_transaction.h"

namespace vernon::runtime::program_execution {

VernonStatus executePublicationInitialization(VernonRuntimeContext &context, const PublicationTransaction &transaction,
                                              std::string &error);
VernonStatus executePublicationCommit(VernonRuntimeContext &context, PublicationTransaction &transaction,
                                      const ProgramInvocationState &state, std::string &error);

} // namespace vernon::runtime::program_execution

#endif
