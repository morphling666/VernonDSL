#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_EXECUTABLE_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_EXECUTABLE_H

#include "runtime_cpu_replay.h"

namespace vernon::runtime::ad::cpu {

bool loadStructuredCpuExecutable(VernonRuntimeContext &context, const Stage &primalStage, const Stage &forwardStage,
                                 const Stage &backwardStage, const std::vector<std::string> &gradientPaths,
                                 uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                 const std::string &selectedPolicy, bool wholeDispatchRetentionPermitted,
                                 std::shared_ptr<Executable> &executable);

bool loadStructuredCpuEntryExecutable(VernonRuntimeContext &context, const Stage &primalStage,
                                      VernonCpuEntryPoint primalEntry, const Stage &forwardStage,
                                      VernonCpuEntryPoint forwardEntry, const Stage &backwardStage,
                                      VernonCpuEntryPoint backwardEntry, const std::vector<std::string> &gradientPaths,
                                      uint64_t staticTapeBytesHint, const std::string &residualStorage,
                                      const std::string &selectedPolicy, bool wholeDispatchRetentionPermitted,
                                      std::shared_ptr<Executable> &executable);

} // namespace vernon::runtime::ad::cpu

#endif
