#include "VernonCompiler.h"
#include "VernonVersions.h"
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
using vernon::tests::ProgramFixtureManifest;
using vernon::tests::ResolvedProgram;

std::string stringValue(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string{};
}

static constexpr char synchronizationModule[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @synchronize(
      %output: !vernon.tensor_view<i32, [10], "write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.dtype = "i32",
        vernon.element_abi_leaf_dtypes = ["i32"]
      },
      %lane: index {
        vernon.interface = "input",
        vernon.builtin = "local_invocation_id"
      },
      %group: index {
        vernon.interface = "input",
        vernon.builtin = "workgroup_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %shared = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<i32, [1], "read_write", "workgroup">
    %zero_index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %hundred = arith.constant 100 : i32
    %lane_is_zero = arith.cmpi eq, %lane, %zero_index : index
    scf.if %lane_is_zero {
      %group_i32 = arith.index_cast %group : index to i32
      %group_base = arith.muli %group_i32, %hundred : i32
      "vernon.store"(%group_base, %shared, %zero_index) :
        (i32, !vernon.tensor_view<i32, [1], "read_write", "workgroup">, index) -> ()
    }
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %previous = "vernon.atomic"(%shared, %zero_index, %one) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<i32, [1], "read_write", "workgroup">, index, i32) -> i32
    %four = arith.constant 4 : index
    %group_offset = arith.muli %group, %four : index
    %previous_index = arith.addi %group_offset, %lane : index
    "vernon.store"(%previous, %output, %previous_index) :
      (i32, !vernon.tensor_view<i32, [10], "write", "device">, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    scf.if %lane_is_zero {
      %final = "vernon.load"(%shared, %zero_index) :
        (!vernon.tensor_view<i32, [1], "read_write", "workgroup">, index) -> i32
      %eight = arith.constant 8 : index
      %final_index = arith.addi %eight, %group : index
      "vernon.store"(%final, %output, %final_index) :
        (i32, !vernon.tensor_view<i32, [10], "write", "device">, index) -> ()
    }
    return
  }
}
)mlir";

// Each of the 2 workgroups' 4 lanes stores what its atomic add returned, and lane 0 stores the final count.
void verifySynchronizationResult(const std::vector<int32_t> &result) {
    ASSERT_EQ(result.size(), 10u);
    for (size_t group = 0; group < 2; ++group) {
        std::array<int32_t, 4> previous{result[group * 4], result[group * 4 + 1], result[group * 4 + 2],
                                        result[group * 4 + 3]};
        std::sort(previous.begin(), previous.end());
        const int32_t base = static_cast<int32_t>(group * 100);
        EXPECT_EQ(previous, (std::array<int32_t, 4>{base, base + 1, base + 2, base + 3}));
        EXPECT_EQ(result[8 + group], base + 4);
    }
}

TEST(CompilerRuntimeSynchronization, CpuArtifactCompilationDoesNotUseProcessGlobalHelpers) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, synchronizationModule,
                                                              sizeof(synchronizationModule) - 1, VERNON_TARGET_CPU);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK)
        << stringValue(vernonCompileResultGetDiagnostics(compiled));
    EXPECT_EQ(vernonCompileResultGetArtifactCount(compiled), 1u);
    EXPECT_EQ(vernonCompileResultGetCpuEntry(compiled, "synchronize", std::strlen("synchronize")), nullptr);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

#if defined(VERNON_SYNCHRONIZATION_FIXTURES_AVAILABLE)
class RuntimeSynchronization : public vernon::tests::HalProgramTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        vernon::tests::BackendTestRequirements result;
        result.compute = true;
        result.storageBuffers = true;
        if (GetParam().runtime == VERNON_RUNTIME_OPENGL) {
            result.minimumApiMajor = 4;
            result.minimumApiMinor = 3;
        } else if (GetParam().runtime == VERNON_RUNTIME_OPENGL_ES) {
            result.minimumApiMajor = 3;
            result.minimumApiMinor = 1;
        }
        return result;
    }

    DeviceTensor output() const { return DeviceTensor(device_, std::vector<int32_t>(10)); }
};

TEST_P(RuntimeSynchronization, CanonicalProgramExecutesIndependentWorkgroupBarrierAndAtomic) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    LinkedProgram linked(runtime(), program.get());
    ASSERT_TRUE(linked) << linked.error();
    const DeviceTensor result = output();
    linked.record().bind("output", result);
    linked.record().bindGrid(2);
    ASSERT_TRUE(linked.record().unbound().empty());
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    verifySynchronizationResult(result.values<int32_t>());
}

// Two nodes of one Program export their boundaries under their own names, so each binds its own output and grid.
TEST_P(RuntimeSynchronization, ProgramGraphScopesDuplicateNodeBindings) {
    const ResolvedProgram program(runtime(), GetParam().manifestPath);
    ASSERT_TRUE(program) << vernon::tests::runtimeError(runtime());
    const std::unique_ptr<VernonProgramGraph, decltype(&vernonRuntimeProgramGraphDestroy)> graph(
        vernonRuntimeProgramGraphCreate(runtime()), vernonRuntimeProgramGraphDestroy);
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId nodes[2]{};
    for (size_t node = 0; node < 2; ++node) {
        ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph.get(), program.bundle(), nullptr, {}, &nodes[node]),
                  VERNON_STATUS_OK);
        const auto exportAs = [&](const std::string &name, const std::string &exported) {
            VernonProgramNodeBindingToken token{sizeof(VernonProgramNodeBindingToken)};
            ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph.get(), nodes[node], VERNON_PROGRAM_BOUNDARY_INPUT,
                                                            {name.data(), name.size()}, &token),
                      VERNON_STATUS_OK);
            ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph.get(), &token, {exported.data(), exported.size()}),
                      VERNON_STATUS_OK);
        };
        const std::string prefix = "node_" + std::to_string(node) + "_";
        exportAs("output", prefix + "output");
        for (const char axis : {'x', 'y', 'z'})
            exportAs(std::string("__grid_") + axis, prefix + "grid_" + axis);
    }
    const std::unique_ptr<VernonProgramExecutable, decltype(&vernonRuntimeProgramExecutableDestroy)> executable(
        vernonRuntimeResolveProgramGraph(graph.get()), vernonRuntimeProgramExecutableDestroy);
    ASSERT_NE(executable, nullptr) << vernon::tests::runtimeError(runtime());
    {
        const std::unique_ptr<VernonProgramExecutable, decltype(&vernonRuntimeProgramExecutableDestroy)> same(
            vernonRuntimeResolveProgramGraph(graph.get()), vernonRuntimeProgramExecutableDestroy);
        ASSERT_NE(same, nullptr) << vernon::tests::runtimeError(runtime());
        const VernonStringView identity = vernonRuntimeProgramExecutableGetId(executable.get());
        const VernonStringView sameIdentity = vernonRuntimeProgramExecutableGetId(same.get());
        EXPECT_EQ(std::string(identity.data, identity.size), std::string(sameIdentity.data, sameIdentity.size));
    }

    LinkedProgram linked(runtime(), executable.get());
    ASSERT_TRUE(linked) << linked.error();
    const DeviceTensor results[2]{output(), output()};
    for (size_t node = 0; node < 2; ++node) {
        const std::string prefix = "node_" + std::to_string(node) + "_";
        linked.record().bind(prefix + "output", results[node]);
        linked.record().bindScalar<uint32_t>(prefix + "grid_x", 2);
        linked.record().bindScalar<uint32_t>(prefix + "grid_y", 1);
        linked.record().bindScalar<uint32_t>(prefix + "grid_z", 1);
    }
    ASSERT_TRUE(linked.record().unbound().empty()) << linked.record().unbound().front();
    ASSERT_EQ(linked.call(), VERNON_STATUS_OK) << linked.error();
    verifySynchronizationResult(results[0].values<int32_t>());
    verifySynchronizationResult(results[1].values<int32_t>());
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, RuntimeSynchronization,
                         testing::ValuesIn(vernon::tests::programFixtureCases("synchronization", true)),
                         [](const testing::TestParamInfo<ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });
#endif

} // namespace
