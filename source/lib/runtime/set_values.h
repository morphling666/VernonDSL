#ifndef VERNON_RUNTIME_SET_VALUES_H
#define VERNON_RUNTIME_SET_VALUES_H

#include "VernonResult.hpp"
#include "runtime/program_execution_manifest.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/shape_layout.h"

#include <optional>
#include <vector>

namespace vernon::runtime {

enum class ProgramValueOwnership : uint8_t {
    Unbound,
    BorrowedHost,
    BorrowedDevice,
};

struct ProgramValueState {
    std::vector<int64_t> strides;
    std::optional<shape::ConcreteShape> concreteShape;
    VernonProgramArgument argument{};
    ProgramValueOwnership ownership{ProgramValueOwnership::Unbound};
};

// The Values a sealed binding set supplies to one pipeline's Program, which the set's frames are lowered from. A
// Tensor argument's shape and strides point into its own state, so the values never move after construction.
class SetValues {
public:
    SetValues(const program::ResolvedExecutionPlan &plan, std::vector<ProgramValueState> values);
    SetValues(const SetValues &) = delete;
    SetValues &operator=(const SetValues &) = delete;

    // A dynamic control must name a non-negative integer host scalar the set supplies.
    vernon::Result<uint64_t, std::string> resolveControl(const program::Program &program,
                                                         const program::ControlComponent &control) const;
    const VernonProgramArgument *argument(uint32_t value) const;
    // Whether the argument of `value` is the set's bound argument itself.
    bool borrowsBinding(uint32_t value) const;

    const program::ResolvedExecutionPlan &plan() const { return *plan_; }
    const std::vector<ProgramValueState> &values() const { return values_; }

private:
    const program::ResolvedExecutionPlan *plan_;
    std::vector<ProgramValueState> values_;
};

} // namespace vernon::runtime

#endif
