#include "runtime/backend_stage_pipeline.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <type_traits>

static_assert(!std::is_same_v<VernonStageExecutable, VernonProgramExecutable>);
static_assert(!std::is_convertible_v<VernonStageExecutable *, VernonProgramExecutable *>);
static_assert(!std::is_convertible_v<VernonProgramExecutable *, VernonStageExecutable *>);
static_assert(std::is_default_constructible_v<VernonStageExecutable>);
static_assert(!std::is_default_constructible_v<VernonProgramExecutable>);
static_assert(std::is_constructible_v<VernonProgramExecutable, VernonRuntimeContext &,
                                      std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan>>);

TEST(RuntimeArchitectureTypes, StageAndProgramHandlesAreNominallyDistinct) {
    EXPECT_FALSE((std::is_convertible_v<VernonStageExecutable *, VernonProgramExecutable *>));
}
