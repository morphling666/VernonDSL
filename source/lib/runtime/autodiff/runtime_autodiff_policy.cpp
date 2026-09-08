#include "runtime/autodiff/runtime_autodiff_policy.h"

namespace vernon::runtime::ad {

bool parsePlanningPolicy(std::string_view value, PlanningPolicy &policy) {
    if (value == "min_memory")
        policy = PlanningPolicy::MinMemory;
    else if (value == "balanced")
        policy = PlanningPolicy::Balanced;
    else if (value == "min_runtime")
        policy = PlanningPolicy::MinRuntime;
    else
        return false;
    return true;
}

} // namespace vernon::runtime::ad
