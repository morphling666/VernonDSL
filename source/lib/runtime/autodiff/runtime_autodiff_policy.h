#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_POLICY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_POLICY_H

namespace vernon::runtime::ad {

enum class PlanningPolicy {
    MinMemory,
    Balanced,
    MinRuntime,
};

} // namespace vernon::runtime::ad

#endif
