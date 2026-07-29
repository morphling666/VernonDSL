#include "VernonVersions.h"
#include "vernon-c/Compiler.h"
#include "vernon_test_support.h"

#include <gtest/gtest.h>
#include <string_view>

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

} // namespace

TEST(CompiledProgramOwnership, OutlivesCompilerContext) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);

    VernonCompileOptions options{};
    options.struct_size = sizeof(options);
    constexpr std::string_view cpu = "generic";
    options.cpu_name = {cpu.data(), cpu.size()};
    VernonCompileResult *program =
        vernonCompilerCompileMlirWithOptions(compiler, module.data(), module.size(), VERNON_TARGET_CPU, &options);
    ASSERT_TRUE(program);
    ASSERT_TRUE(vernonCompileResultGetStatus(program) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernon::test::contains(vernonCompileResultGetReflection(program), R"json("target":"cpu")json"));
    ASSERT_TRUE(vernon::test::contains(vernonCompileResultGetReflection(program), R"json("cpu":"generic")json"));
    ASSERT_TRUE(vernon::test::contains(vernonCompileResultGetReflection(program), R"json("target_triple":)json"));

    VernonCpuEntryPoint entry = vernonCompileResultGetCpuEntry(program, "add_vectors", 11);
    ASSERT_TRUE(entry);
    vernonCompilerDestroy(compiler);

    float arguments[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float results[4]{};
    VernonCpuInvocation invocation{arguments, sizeof(arguments), results, sizeof(results), nullptr};
    ASSERT_TRUE(entry(&invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(results[0] == 6.0f);
    ASSERT_TRUE(results[1] == 8.0f);
    ASSERT_TRUE(results[2] == 10.0f);
    ASSERT_TRUE(results[3] == 12.0f);
    ASSERT_TRUE(!vernonCompileResultGetCpuEntry(program, "missing", 7));

    vernonCompileResultDestroy(program);
}
