#include "VernonCpuWorkgroupABI.h"
#include "VernonVersions.h"
#include "vernon-c/Compiler.h" // IWYU pragma: keep
#include "vernon_test_support.h"

#include <gtest/gtest.h>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view module = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @add_vectors(
      %left: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %right: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 1 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %sum = arith.addf %left, %right : tensor<4xf32>
    return %sum : tensor<4xf32>
  }
}
)mlir";

VernonStatus invokeRange(VernonCpuEntryPoint entry, const void *arguments, size_t argumentsSize, void *results,
                         size_t resultsSize) {
    VernonCpuRangeV1 range{};
    range.struct_size = sizeof(range);
    range.arguments = arguments;
    range.arguments_size = argumentsSize;
    range.results = results;
    range.results_size = resultsSize;
    range.grid[0] = range.grid[1] = range.grid[2] = 1;
    range.workgroup[0] = range.workgroup[1] = range.workgroup[2] = 1;
    range.lane_end = 1;
    const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
    return entry(&invocation);
}

} // namespace

TEST(CompiledProgramOwnership, OutlivesCompilerContext) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);

    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.target = VERNON_TARGET_CPU;
    constexpr std::string_view cpu = "generic";
    options.as.cpu.processor = {cpu.data(), cpu.size()};
    VernonCompileResult *program =
        vernonCompilerCompileMlirWithOptions(compiler, module.data(), module.size(), &options);
    ASSERT_TRUE(program);
    ASSERT_TRUE(vernonCompileResultGetStatus(program) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernon::test::contains(vernonCompileResultGetReflection(program), R"json("kind":"cpu")json"));
    ASSERT_TRUE(vernon::test::contains(vernonCompileResultGetReflection(program), R"json("processor":"generic")json"));
    ASSERT_TRUE(vernon::test::contains(vernonCompileResultGetReflection(program), R"json("triple":)json"));

    VernonCpuEntryPoint entry = vernonCompileResultGetCpuEntry(program, "add_vectors", 11);
    ASSERT_TRUE(entry);
    vernonCompilerDestroy(compiler);

    float arguments[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float results[4]{};
    const VernonCpuInvocation scalarInvocation{arguments, sizeof(arguments), results, sizeof(results), nullptr};
    ASSERT_EQ(entry(&scalarInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(invokeRange(entry, arguments, sizeof(arguments), results, sizeof(results)) == VERNON_STATUS_OK);
    ASSERT_TRUE(results[0] == 6.0f);
    ASSERT_TRUE(results[1] == 8.0f);
    ASSERT_TRUE(results[2] == 10.0f);
    ASSERT_TRUE(results[3] == 12.0f);
    float secondArguments[8] = {2.0f, 4.0f, 6.0f, 8.0f, 1.0f, 3.0f, 5.0f, 7.0f};
    float secondResults[4]{};
    const void *laneArguments[2]{arguments, secondArguments};
    void *laneResults[2]{results, secondResults};
    VernonCpuRangeV1 range{};
    range.struct_size = sizeof(range);
    range.arguments = arguments;
    range.arguments_size = sizeof(arguments);
    range.results = results;
    range.results_size = sizeof(results);
    range.grid[0] = range.grid[1] = range.grid[2] = 1;
    range.workgroup[0] = 2;
    range.workgroup[1] = range.workgroup[2] = 1;
    range.lane_end = 2;
    range.lane_arguments = laneArguments;
    range.lane_results = laneResults;
    range.lane_table_count = 2;
    const VernonCpuInvocation rangeInvocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
    ASSERT_EQ(entry(&rangeInvocation), VERNON_STATUS_OK);
    EXPECT_EQ(secondResults[0], 3.0f);
    EXPECT_EQ(secondResults[3], 15.0f);

    VernonCpuRangeV1 invalidRange = range;
    invalidRange.workgroup[0] = 0;
    VernonCpuInvocation invalidInvocation{&invalidRange, VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1, nullptr, 0, nullptr};
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    invalidRange = range;
    invalidRange.lane_end = 3;
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    invalidRange = range;
    invalidRange.lane_end = invalidRange.lane_begin;
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    invalidRange = range;
    invalidRange.group[0] = invalidRange.grid[0];
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    invalidRange = range;
    invalidRange.struct_size = sizeof(uint64_t);
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    invalidRange = range;
    invalidRange.arguments = nullptr;
    invalidRange.lane_begin = 1;
    laneArguments[1] = nullptr;
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    laneArguments[1] = secondArguments;
    invalidRange = range;
    invalidRange.lane_table_count = 1;
    EXPECT_EQ(entry(&invalidInvocation), VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(!vernonCompileResultGetCpuEntry(program, "missing", 7));

    vernonCompileResultDestroy(program);
}

TEST(CompiledProgramOwnership, MultipleResultsOutliveOneCompilerContext) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    options.target = VERNON_TARGET_CPU;
    std::vector<VernonCompileResult *> programs;
    for (unsigned index = 0; index < 12; ++index) {
        VernonCompileResult *program =
            vernonCompilerCompileMlirWithOptions(compiler, module.data(), module.size(), &options);
        ASSERT_TRUE(program);
        ASSERT_EQ(vernonCompileResultGetStatus(program), VERNON_STATUS_OK);
        programs.push_back(program);
    }
    vernonCompilerDestroy(compiler);

    for (VernonCompileResult *program : programs) {
        VernonCpuEntryPoint entry = vernonCompileResultGetCpuEntry(program, "add_vectors", 11);
        ASSERT_TRUE(entry);
        float arguments[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float results[4]{};
        ASSERT_EQ(invokeRange(entry, arguments, sizeof(arguments), results, sizeof(results)), VERNON_STATUS_OK);
        EXPECT_EQ(results[0], 6.0f);
        vernonCompileResultDestroy(program);
    }
}
