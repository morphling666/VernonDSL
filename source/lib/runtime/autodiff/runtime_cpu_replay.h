#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_REPLAY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_REPLAY_H

#include "runtime_cpu_pullback.h"

namespace vernon::runtime::ad::cpu {

std::shared_ptr<Executable> createNoTapeStructuredCpuExecutable(VernonRuntimeContext &context,
                                                                std::shared_ptr<const CpuAutodiffProgram> program);
std::shared_ptr<Executable> createTapedStructuredCpuExecutable(VernonRuntimeContext &context,
                                                               std::shared_ptr<const CpuAutodiffProgram> program,
                                                               std::shared_ptr<const CpuResidualPlan> plan);

} // namespace vernon::runtime::ad::cpu

#endif
