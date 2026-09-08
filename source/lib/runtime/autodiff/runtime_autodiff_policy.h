#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_POLICY_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_AUTODIFF_POLICY_H

#include <string_view>

namespace vernon::runtime::ad {

enum class PlanningPolicy {
    MinMemory,
    Balanced,
    MinRuntime,
};

bool parsePlanningPolicy(std::string_view value, PlanningPolicy &policy);

} // namespace vernon::runtime::ad

#endif
