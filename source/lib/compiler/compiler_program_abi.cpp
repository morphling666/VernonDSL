#include "compiler_program_abi.h"

#include "compiler_program_boundary.h"
#include "compiler_program_derivative.h"
#include "compiler_program_publication.h"
#include "compiler_program_tape.h"

#include <string>
#include <vector>

namespace vernon::compiler {

bool buildCanonicalProgramAbi(const llvm::json::Object &signature, const llvm::json::Array &values,
                              const llvm::json::Array &storages, const llvm::json::Array &graphs,
                              llvm::json::Object &abi, std::string &error) {
    ProgramBoundaryPlan boundaries;
    if (!planProgramBoundaries(signature, values, storages, graphs, boundaries, error))
        return false;

    llvm::json::Array boundarySlots = serializeProgramBoundarySlots(boundaries);
    std::vector<ProgramBoundaryIdentity> slots;
    slots.reserve(boundaries.slots.size());
    for (const ProgramBoundarySlotPlan &slot : boundaries.slots)
        slots.push_back(slot.identity);
    std::vector<ProgramDerivativeProjectionPlan> projectionPlan;
    if (!planProgramDerivativeProjections(slots, projectionPlan, error))
        return false;
    llvm::json::Array projections = serializeProgramDerivativeProjections(projectionPlan);
    llvm::json::Array tapePlans = serializeProgramTapePlans(planProgramTapes(values, graphs));
    abi = llvm::json::Object{{"boundary_slots", std::move(boundarySlots)},
                             {"derivative_projections", std::move(projections)},
                             {"tape_plans", std::move(tapePlans)}};
    return true;
}

} // namespace vernon::compiler
