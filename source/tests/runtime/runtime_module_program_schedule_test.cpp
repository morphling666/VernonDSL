#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"
#include "schedule_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using vernon::tests::DeviceTensor;
using vernon::tests::LinkedProgram;
using vernon::tests::LinkedVjp;
using vernon::tests::ProgramFixtureManifest;
using vernon::tests::ResolvedProgram;

struct GraphDeleter {
    void operator()(VernonProgramGraph *graph) const { vernonRuntimeProgramGraphDestroy(graph); }
};
struct ExecutableDeleter {
    void operator()(VernonProgramExecutable *executable) const { vernonRuntimeProgramExecutableDestroy(executable); }
};

std::string testName(const testing::TestParamInfo<ProgramFixtureManifest> &info) {
    return std::string(info.param.target);
}

class ModuleProgramVjp : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::programVjpFixtureRequirements(GetParam().runtime);
    }

    // Runs `vjp` of a Program of x * x from source 3 and returns its output and the gradient of source.
    void differentiate(LinkedVjp &vjp, float &output, float &gradient) {
        const DeviceTensor source(device_, std::vector<float>{3.0f});
        const DeviceTensor forwardOutput(device_, std::vector<float>{0.0f});
        const DeviceTensor seed(device_, std::vector<float>{1.0f});
        const DeviceTensor sourceGradient(device_, std::vector<float>{0.0f});
        vjp.forwardRecord().bind("source", source);
        vjp.forwardRecord().bind("output", forwardOutput);
        vjp.forwardRecord().bindGrid(1);
        ASSERT_TRUE(vjp.forwardRecord().unbound().empty());
        ASSERT_EQ(vjp.forward(), VERNON_STATUS_OK) << vjp.error();
        vjp.backwardRecord().bind("output", seed, VERNON_PROGRAM_BOUNDARY_COTANGENT);
        vjp.backwardRecord().bind("source", sourceGradient, VERNON_PROGRAM_BOUNDARY_GRADIENT);
        ASSERT_TRUE(vjp.backwardRecord().unbound().empty());
        ASSERT_EQ(vjp.backward(), VERNON_STATUS_OK) << vjp.error();
        output = forwardOutput.values<float>()[0];
        gradient = sourceGradient.values<float>()[0];
    }
};

TEST_P(ModuleProgramVjp, ComputeModuleForward9AndVjpGradient6) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    ASSERT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(program.get()), 1u);
    LinkedVjp vjp(runtime(), program.get());
    ASSERT_TRUE(vjp) << vjp.error();
    float output = 0.0f;
    float gradient = 0.0f;
    differentiate(vjp, output, gradient);
    EXPECT_FLOAT_EQ(output, 9.0f);
    EXPECT_FLOAT_EQ(gradient, 6.0f);
}

TEST_P(ModuleProgramVjp, ComposedProgramGraphRunsItsNodesBackwardGraphs) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    const std::unique_ptr<VernonProgramGraph, GraphDeleter> graph(vernonRuntimeProgramGraphCreate(runtime()));
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId node = UINT32_MAX;
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph.get(), program.bundle(), nullptr, {}, &node), VERNON_STATUS_OK);
    for (const auto &[role, name] :
         {std::pair{VERNON_PROGRAM_BOUNDARY_INPUT, "source"}, std::pair{VERNON_PROGRAM_BOUNDARY_OUTPUT, "output"}}) {
        VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken)};
        ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph.get(), node, role, {name, std::strlen(name)}, &token),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph.get(), &token, {name, std::strlen(name)}),
                  VERNON_STATUS_OK);
    }
    const std::unique_ptr<VernonProgramExecutable, ExecutableDeleter> composite(
        vernonRuntimeResolveProgramGraph(graph.get()));
    ASSERT_NE(composite, nullptr) << vernon::tests::runtimeError(runtime());
    EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(composite.get()), 0u);
    LinkedVjp vjp(runtime(), composite.get());
    ASSERT_TRUE(vjp) << vjp.error();
    float output = 0.0f;
    float gradient = 0.0f;
    differentiate(vjp, output, gradient);
    EXPECT_FLOAT_EQ(output, 9.0f);
    EXPECT_FLOAT_EQ(gradient, 6.0f);
}

class ModuleProgramCompute : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::computeFixtureRequirements(GetParam().runtime);
    }

    // Calls the tensor_view_chain Program, which writes (source + 1) * 2, with timings when `timed`.
    void runTensorViewChain(bool timed) {
        const ResolvedProgram program(runtime(), fixture("tensor_view_chain").manifestPath);
        ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());

        using namespace vernon::runtime::program;
        const ResolvedExecutionPlan &plan = *program.get()->executionPlan;
        const ResolvedNodePlan *producer = plan.node(GraphDirection::Forward, 0);
        const ResolvedNodePlan *consumer = plan.node(GraphDirection::Forward, 1);
        ASSERT_NE(producer, nullptr);
        ASSERT_NE(consumer, nullptr);
        EXPECT_NE(producer->stage.get(), consumer->stage.get());
        const Graph *forward = findGraph(plan.resolvedProgram->program, "forward");
        ASSERT_NE(forward, nullptr);
        ASSERT_EQ(forward->nodes.size(), 2u);
        const auto intermediate =
            std::find_first_of(forward->nodes[0].results.begin(), forward->nodes[0].results.end(),
                               forward->nodes[1].operands.begin(), forward->nodes[1].operands.end());
        ASSERT_NE(intermediate, forward->nodes[0].results.end());
        ASSERT_LT(*intermediate, plan.resolvedProgram->program.values.size());
        EXPECT_TRUE(plan.resolvedProgram->program.values[*intermediate].storage.has_value());
        const size_t graph = static_cast<size_t>(forward - plan.resolvedProgram->program.graphs.data());
        ASSERT_LT(graph, plan.resolvedProgram->graphs.size());
        const std::vector<uint32_t> &predecessors = plan.resolvedProgram->graphs[graph].predecessors.at(1);
        EXPECT_NE(std::find(predecessors.begin(), predecessors.end(), 0), predecessors.end());

        LinkedProgram linked(runtime(), program.get());
        ASSERT_TRUE(linked) << linked.error();
        const DeviceTensor source(device_, std::vector<float>{3.0f});
        const DeviceTensor output(device_, std::vector<float>{0.0f});
        linked.record().bind("source", source);
        linked.record().bind("output", output);
        linked.record().bindGrid(1);
        ASSERT_TRUE(linked.record().unbound().empty());
        if (timed)
            ASSERT_EQ(vernonScheduleSetTimings(linked.get(), 1), VERNON_STATUS_OK);
        uint64_t point = 0;
        ASSERT_EQ(linked.call(&point), VERNON_STATUS_OK) << linked.error();
        EXPECT_FLOAT_EQ(output.values<float>()[0], 8.0f);

        std::array<VernonScopeTiming, 4> scopes{};
        size_t count = 0;
        const VernonStatus status =
            vernonScheduleGetScopeTimings(linked.get(), point, scopes.data(), scopes.size(), &count);
        if (!timed) {
            EXPECT_EQ(status, VERNON_STATUS_UNSUPPORTED_TARGET);
            return;
        }
        // The CPU and CUDA always time; a graphics device without timers reports the call untimed.
        if (status == VERNON_STATUS_UNSUPPORTED_TARGET && !device_.host() && device_.backend() != VERNON_RUNTIME_CUDA)
            return;
        ASSERT_EQ(status, VERNON_STATUS_OK) << linked.error();
        ASSERT_EQ(count, 2u);
        EXPECT_EQ(scopes[0].begin_nanoseconds, 0u);
        for (size_t index = 0; index < count; ++index) {
            EXPECT_EQ(scopes[index].first_node, index);
            EXPECT_EQ(scopes[index].node_count, 1u);
            EXPECT_LE(scopes[index].begin_nanoseconds, scopes[index].end_nanoseconds);
        }
    }
};

TEST_P(ModuleProgramCompute, ReusesOneStageAcrossDifferentNodeProjections) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());

    using namespace vernon::runtime::program;
    const ResolvedNodePlan *first = program.get()->executionPlan->node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *second = program.get()->executionPlan->node(GraphDirection::Forward, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->stage.get(), second->stage.get());
    ASSERT_EQ(first->projections.size(), second->projections.size());
    ASSERT_GE(first->projections.size(), 2u);
    EXPECT_NE(first->projections[0].value, second->projections[0].value);
    EXPECT_NE(first->projections[1].value, second->projections[1].value);

    LinkedProgram linked(runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();
    const DeviceTensor source(device_, std::vector<float>{1, 2, 3, 4});
    const DeviceTensor output(device_, std::vector<float>(4));
    linked.record().bind("source", source);
    linked.record().bind("output", output);
    linked.record().bindScalar<uint32_t>("grid_x", 4);
    linked.record().bindScalar<uint32_t>("grid_y", 1);
    linked.record().bindScalar<uint32_t>("grid_z", 1);
    linked.record().bindGrid(4);
    ASSERT_TRUE(linked.record().unbound().empty());
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    EXPECT_EQ(output.values<float>(), (std::vector<float>{3, 4, 5, 6}));
}

TEST_P(ModuleProgramCompute, ChainsDistinctComputeKernelsThroughTensorViews) { runTensorViewChain(false); }

TEST_P(ModuleProgramCompute, TimesEachComputeNode) { runTensorViewChain(true); }

TEST_P(ModuleProgramCompute, CallsOneScheduleAcrossDynamicShapesAndGrids) {
    const ResolvedProgram program(runtime(), fixture("dynamic_shape_grid").manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    LinkedProgram linked(runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();
    VernonProgramParameterView published{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(program.get(), VERNON_PROGRAM_BOUNDARY_OUTPUT, 0, &published),
        VERNON_STATUS_OK);
    const std::string publishedName(published.name.data, published.name.size);

    constexpr std::array<size_t, 2> counts{3, 5};
    constexpr std::array<float, 2> factors{2, 3};
    for (size_t iteration = 0; iteration < counts.size(); ++iteration) {
        const size_t count = counts[iteration];
        std::vector<float> values(count);
        for (size_t index = 0; index < count; ++index)
            values[index] = static_cast<float>(index + 1);
        const DeviceTensor source(device_, values);
        const DeviceTensor output(device_, std::vector<float>(count));
        linked.record().bind("source", source);
        linked.record().bind("output", output);
        if (publishedName != "output")
            linked.record().bind(publishedName, output);
        linked.record().bindScalar("factor", factors[iteration]);
        linked.record().bindGrid(static_cast<uint32_t>(count));
        ASSERT_TRUE(linked.record().unbound().empty());
        ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
        const std::vector<float> written = output.values<float>();
        for (size_t index = 0; index < count; ++index)
            EXPECT_FLOAT_EQ(written[index], values[index] * factors[iteration]);
    }
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, ModuleProgramVjp,
                         testing::ValuesIn(vernon::tests::programFixtureCases("module_program", true)), testName);

INSTANTIATE_TEST_SUITE_P(EnabledTargets, ModuleProgramCompute,
                         testing::ValuesIn(vernon::tests::programFixtureCases("reused_stage", true)), testName);

} // namespace
