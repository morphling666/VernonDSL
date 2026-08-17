#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_PULLBACK_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_CPU_PULLBACK_H

#include "runtime_cpu_forward.h"

namespace vernon::runtime::ad::cpu {

std::unique_ptr<PullbackExecution>
createNoTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                        Signature signature, VernonLaunchSize computeGrid, RuntimeTensorShapes tensorShapes,
                        RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews);
std::unique_ptr<PullbackExecution>
createRetainedTapeCpuPullback(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                              Signature signature, VernonLaunchSize computeGrid,
                              std::shared_ptr<HostStaticTapeBatch> tapeBatch, RuntimeTensorShapes tensorShapes,
                              RetainedPrimalLeaves retainedPrimals, RetainedPrimalTensorViews retainedTensorViews);
VernonStatus applyCpuBackwardSegment(VernonRuntimeContext &context, std::shared_ptr<const CpuAutodiffProgram> program,
                                     const Signature &signature, const RuntimeTensorShapes &tensorShapes,
                                     const RetainedPrimalLeaves &retainedPrimals,
                                     const RetainedPrimalTensorViewRefs &retainedTensorViews,
                                     VernonLaunchSize computeGrid, std::shared_ptr<HostStaticTapeBatch> &tapeBatch,
                                     size_t groupLinear, const VernonAdValueSet *cotangents,
                                     std::vector<std::vector<uint8_t>> &stagedGradients);

} // namespace vernon::runtime::ad::cpu

#endif
