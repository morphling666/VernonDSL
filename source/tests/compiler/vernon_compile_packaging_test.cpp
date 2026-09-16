#include "vernon_compile_packaging.h"

#include "VernonVersions.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>

namespace {

std::string readFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    EXPECT_TRUE(input) << path;
    if (!input)
        return {};
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

} // namespace

TEST(CompilePackaging, WritesArtifactsAndReflection) {
    constexpr std::string_view computeModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @scale() attributes {
    vernon.entry,
    vernon.stage = "compute",
    vernon.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
)mlir";

    VernonCompilerContext *context = vernonCompilerCreate();
    ASSERT_TRUE(context);
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("vernon_compile_packaging_" + std::to_string(unique));
    std::filesystem::create_directories(root);
    std::ostringstream output;
    std::ostringstream error;

    VernonCompileOptions cpuOptions{};
    cpuOptions.struct_size = sizeof(cpuOptions);
    cpuOptions.target = VERNON_TARGET_CPU;
    constexpr std::string_view triple = "x86_64-pc-windows-msvc";
    cpuOptions.as.cpu.triple = {triple.data(), triple.size()};
    VernonCompileResult *cpuResult =
        vernonCompilerCompileMlirWithOptions(context, computeModule.data(), computeModule.size(), &cpuOptions);
    ASSERT_TRUE(cpuResult);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpuResult) == VERNON_STATUS_OK);

    vernon::tools::PackagingOptions outputPackaging;
    outputPackaging.outputDirectory = root / "output";
    ASSERT_TRUE(vernon::tools::packageCompileResult(cpuResult, VERNON_TARGET_CPU, outputPackaging, output, error) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(std::filesystem::is_regular_file(root / "output" / "reflection.json"));
    ASSERT_TRUE(std::filesystem::is_regular_file(root / "output" /
                                                 std::string(vernonCompileResultGetArtifactName(cpuResult, 0).data,
                                                             vernonCompileResultGetArtifactName(cpuResult, 0).size)));
    ASSERT_EQ(readFile(root / "output" / "reflection.json"),
              std::string(vernonCompileResultGetReflection(cpuResult).data,
                          vernonCompileResultGetReflection(cpuResult).size));
    vernonCompileResultDestroy(cpuResult);

    vernonCompilerDestroy(context);
    std::filesystem::remove_all(root);
}
