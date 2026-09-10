#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "VernonVersions.h"
#include "backend_test_matrix.h"
#include "program_fixture_manifest_table.h"
#include "runtime_rhi_test_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <string_view>

namespace {

class RhiRuntimeSynchronization : public testing::TestWithParam<vernon::tests::BackendTestRow> {};

#if defined(VERNON_SYNCHRONIZATION_FIXTURES_AVAILABLE)
extern "C" VernonStatus vernonRegisterSynchronizationProgramFixture(void);
#endif

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

void verifySynchronizationResult(const std::array<int32_t, 10> &result) {
    for (size_t group = 0; group < 2; ++group) {
        std::array<int32_t, 4> previous{result[group * 4], result[group * 4 + 1], result[group * 4 + 2],
                                        result[group * 4 + 3]};
        std::sort(previous.begin(), previous.end());
        const int32_t base = static_cast<int32_t>(group * 100);
        EXPECT_EQ(previous, (std::array<int32_t, 4>{base, base + 1, base + 2, base + 3}));
        EXPECT_EQ(result[8 + group], base + 4);
    }
}

#if defined(VERNON_SYNCHRONIZATION_FIXTURES_AVAILABLE)
TEST(CompilerRuntimeSynchronization, CanonicalCpuProgramExecutesCooperativeWorkgroups) {
    ASSERT_EQ(vernonRegisterSynchronizationProgramFixture(), VERNON_STATUS_OK);
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    {
        const auto *fixture = vernon::tests::findProgramFixtureManifest("synchronization", VERNON_RUNTIME_CPU);
        ASSERT_NE(fixture, nullptr);
        vernon::tests::OwnedProgramExecutable program(runtime, std::string(fixture->manifestPath));
        ASSERT_TRUE(program) << stringValue(vernonRuntimeGetLastError(runtime));

        std::array<int32_t, 10> result{};
        constexpr uint64_t shape[]{result.size()};
        constexpr int64_t strides[]{sizeof(int32_t)};
        VernonProgramArgument argument{};
        VernonProgramParameterView parameter{};
        ASSERT_EQ(
            vernonRuntimeProgramExecutableFindParameter(program.get(), {"output", std::strlen("output")}, &parameter),
            VERNON_STATUS_OK);
        argument.slot = parameter.slot;
        argument.kind = VERNON_PROGRAM_TENSOR;
        argument.tensor.struct_size = sizeof(VernonTensorView);
        argument.tensor.storage = VERNON_TENSOR_HOST;
        argument.tensor.host_data = result.data();
        argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
        argument.tensor.access = VERNON_ACCESS_WRITE;
        argument.tensor.rank = 1;
        argument.tensor.shape = shape;
        argument.tensor.byte_strides = strides;
        argument.tensor.byte_size = sizeof(result);
        ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(program.get(), &argument, 1, {2, 1, 1}),
                  VERNON_STATUS_OK)
            << stringValue(vernonRuntimeGetLastError(runtime));
        verifySynchronizationResult(result);
    }

    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}

TEST(CompilerRuntimeSynchronization, ProgramGraphScopesDuplicateNodeBindings) {
    ASSERT_EQ(vernonRegisterSynchronizationProgramFixture(), VERNON_STATUS_OK);
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    const auto *fixture = vernon::tests::findProgramFixtureManifest("synchronization", VERNON_RUNTIME_CPU);
    ASSERT_NE(fixture, nullptr);
    const std::filesystem::path manifestPath = std::string(fixture->manifestPath);
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(manifest.empty());
    const std::string directory = manifestPath.parent_path().string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(runtime, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << stringValue(vernonRuntimeGetLastError(runtime));
    VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(runtime);
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId nodes[2]{};
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, &nodes[0]), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, &nodes[1]), VERNON_STATUS_OK);

    VernonProgramNodeBindingToken outputs[2]{{sizeof(VernonProgramNodeBindingToken)},
                                             {sizeof(VernonProgramNodeBindingToken)}};
    std::array<std::array<VernonProgramNodeBindingToken, 3>, 2> grids{};
    for (size_t node = 0; node < 2; ++node) {
        ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, nodes[node], VERNON_PROGRAM_BOUNDARY_INPUT,
                                                        {"output", 6}, &outputs[node]),
                  VERNON_STATUS_OK);
        for (size_t axis = 0; axis < 3; ++axis) {
            grids[node][axis].struct_size = sizeof(VernonProgramNodeBindingToken);
            const std::string name = "__grid_" + std::string(1, "xyz"[axis]);
            ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, nodes[node], VERNON_PROGRAM_BOUNDARY_INPUT,
                                                            {name.data(), name.size()}, &grids[node][axis]),
                      VERNON_STATUS_OK);
        }
    }
    VernonProgramExecutable *executable = vernonRuntimeResolveProgramGraph(graph);
    ASSERT_NE(executable, nullptr) << stringValue(vernonRuntimeGetLastError(runtime));
    VernonProgramExecutable *sameExecutable = vernonRuntimeResolveProgramGraph(graph);
    ASSERT_NE(sameExecutable, nullptr) << stringValue(vernonRuntimeGetLastError(runtime));
    const VernonStringView identity = vernonRuntimeProgramExecutableGetId(executable);
    const VernonStringView sameIdentity = vernonRuntimeProgramExecutableGetId(sameExecutable);
    EXPECT_EQ(std::string_view(identity.data, identity.size), std::string_view(sameIdentity.data, sameIdentity.size));
    vernonRuntimeProgramExecutableDestroy(sameExecutable);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(executable);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);

    std::array<std::array<int32_t, 10>, 2> results{};
    constexpr uint64_t shape[]{10};
    constexpr int64_t strides[]{sizeof(int32_t)};
    const uint32_t gridValues[3]{2, 1, 1};
    for (size_t node = 0; node < 2; ++node) {
        VernonProgramArgument output{};
        output.kind = VERNON_PROGRAM_TENSOR;
        output.tensor.struct_size = sizeof(VernonTensorView);
        output.tensor.storage = VERNON_TENSOR_HOST;
        output.tensor.host_data = results[node].data();
        output.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
        output.tensor.access = VERNON_ACCESS_WRITE;
        output.tensor.rank = 1;
        output.tensor.shape = shape;
        output.tensor.byte_strides = strides;
        output.tensor.byte_size = sizeof(results[node]);
        ASSERT_EQ(vernonRuntimeProgramInvocationBindNode(invocation, &outputs[node], &output, nullptr, 0, 0),
                  VERNON_STATUS_OK);
        for (size_t axis = 0; axis < 3; ++axis) {
            VernonProgramArgument grid{};
            grid.kind = VERNON_PROGRAM_TENSOR;
            grid.tensor.struct_size = sizeof(VernonTensorView);
            grid.tensor.storage = VERNON_TENSOR_HOST;
            grid.tensor.host_data = &gridValues[axis];
            grid.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_U32);
            grid.tensor.access = VERNON_ACCESS_READ;
            grid.tensor.byte_size = sizeof(uint32_t);
            ASSERT_EQ(vernonRuntimeProgramInvocationBindNode(invocation, &grids[node][axis], &grid, nullptr, 0, 0),
                      VERNON_STATUS_OK);
        }
    }
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK)
        << stringValue(vernonRuntimeGetLastError(runtime));
    verifySynchronizationResult(results[0]);
    verifySynchronizationResult(results[1]);

    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(executable);
    vernonRuntimeProgramGraphDestroy(graph);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}
#endif

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
TEST_P(RhiRuntimeSynchronization, CanonicalProgramExecutesIndependentWorkgroupBarrierAndAtomic) {
    const vernon::tests::BackendTestRow backend = GetParam();
    vernon::tests::BackendTestRequirements requirements;
    requirements.compute = true;
    requirements.storageBuffers = true;
    if (backend.runtime == VERNON_RUNTIME_OPENGL) {
        requirements.minimumApiMajor = 4;
        requirements.minimumApiMinor = 3;
    } else if (backend.runtime == VERNON_RUNTIME_OPENGL_ES) {
        requirements.minimumApiMajor = 3;
        requirements.minimumApiMinor = 1;
    }
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    const vernon::tests::BackendProbeResult compilerProbe =
        vernon::tests::probeCompilerBackend(compiler, backend, requirements);
    if (!compilerProbe.available()) {
        vernonCompilerDestroy(compiler);
        GTEST_SKIP() << compilerProbe.reason;
    }
    vernonCompilerDestroy(compiler);
    vernon::tests::OwnedRhiRuntime owned(backend.runtime, nullptr, backend.runtime == VERNON_RUNTIME_DIRECTX12);
    vernon::tests::RhiRuntime &context = owned.context();
    const vernon::tests::BackendProbeResult runtimeProbe =
        vernon::tests::probeRuntimeBackend(backend, requirements, context.runtime);
    if (!runtimeProbe.available()) {
        if (runtimeProbe.skippable())
            GTEST_SKIP() << runtimeProbe.reason;
        FAIL() << runtimeProbe.reason;
        return;
    }
    const auto *fixture = vernon::tests::findProgramFixtureManifest("synchronization", backend.runtime);
    ASSERT_NE(fixture, nullptr);
    vernon::tests::OwnedProgramExecutable program(context.runtime, std::string(fixture->manifestPath));
    ASSERT_TRUE(program) << stringValue(vernonRuntimeGetLastError(context.runtime));

    std::array<int32_t, 10> result{};
    vernon::tests::RhiBuffer buffer = vernon::tests::createBuffer(context, sizeof(result), alignof(int32_t),
                                                                  VERNON_RHI_BUFFER_STORAGE, result.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[]{result.size()};
    constexpr int64_t strides[]{sizeof(int32_t)};
    VernonProgramArgument argument{};
    VernonProgramParameterView parameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableFindParameter(program.get(), {"output", std::strlen("output")}, &parameter),
              VERNON_STATUS_OK);
    argument.slot = parameter.slot;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = buffer.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(result);
    ASSERT_EQ(vernon::tests::completeCanonicalComputeInvocation(program.get(), &argument, 1, {2, 1, 1}),
              VERNON_STATUS_OK)
        << "RHI: " << stringValue(vernonRhiDeviceGetLastError(context.device))
        << "; runtime: " << stringValue(vernonRuntimeGetLastError(context.runtime));
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, result.data(), sizeof(result)),
              VERNON_RHI_STATUS_OK);
    verifySynchronizationResult(result);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, RhiRuntimeSynchronization,
                         testing::ValuesIn(vernon::tests::backendTestMatrix.begin() + 1,
                                           vernon::tests::backendTestMatrix.end()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });
#endif

} // namespace
