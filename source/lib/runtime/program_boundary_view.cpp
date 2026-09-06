#include "program_boundary_view.h"

#include "resolved_execution_plan.h"
#include "runtime_state.h"

#include <stdexcept>

namespace vernon::runtime {
namespace {

const char *roleName(program::BoundaryRole role) {
    switch (role) {
    case program::BoundaryRole::Input:
        return "input";
    case program::BoundaryRole::Output:
        return "output";
    case program::BoundaryRole::Cotangent:
        return "cotangent";
    case program::BoundaryRole::Gradient:
        return "gradient";
    }
    return "";
}

const char *categoryName(program::BoundaryCategory category) {
    switch (category) {
    case program::BoundaryCategory::Value:
        return "value";
    case program::BoundaryCategory::StorageView:
        return "storage_view";
    case program::BoundaryCategory::Texture:
        return "texture";
    case program::BoundaryCategory::Sampler:
        return "sampler";
    }
    return "";
}

} // namespace

std::vector<ProgramBoundaryView> programBoundaryViews(const VernonProgramExecutable &pipeline) {
    const program::ResolvedExecutionPlan &execution = *pipeline.executionPlan;
    std::vector<ProgramBoundaryView> result;
    for (const program::BoundarySlot &slot : execution.resolvedProgram->program.abi.boundarySlots)
        result.push_back({slot.id, slot.value, slot.path, roleName(slot.role), categoryName(slot.category)});
    return result;
}

} // namespace vernon::runtime
