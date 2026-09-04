#ifndef VERNON_RUNTIME_GRAPHICS_SCOPE_PLANNER_H
#define VERNON_RUNTIME_GRAPHICS_SCOPE_PLANNER_H

#include "graphics_invocation_planner.h"

namespace vernon::runtime {

class GraphicsScopePlanner {
public:
    bool canAppend(const PlannedGraphicsInvocation &next) const;
    void append(const PlannedGraphicsInvocation &invocation);
    void reset();

private:
    const PlannedGraphicsInvocation *current_{};
};

} // namespace vernon::runtime

#endif
