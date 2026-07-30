#include "vernon_compile_packaging.h"

#include "VernonVersions.h"
#include "vernon-c/Compiler.h"

#include <nlohmann/json.hpp>

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

TEST(CompilePackaging, WritesBundlesAndArtifacts) {
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
    constexpr std::string_view triple = "x86_64-pc-windows-msvc";
    cpuOptions.cpu_target_triple = {triple.data(), triple.size()};
    VernonCompileResult *cpuResult = vernonCompilerCompileMlirWithOptions(
        context, computeModule.data(), computeModule.size(), VERNON_TARGET_CPU, &cpuOptions);
    ASSERT_TRUE(cpuResult);
    ASSERT_TRUE(vernonCompileResultGetStatus(cpuResult) == VERNON_STATUS_OK);

    vernon::tools::PackagingOptions computePackaging;
    computePackaging.computeBundlePath = root / "compute";
    computePackaging.targetTriple = std::string(triple);
    ASSERT_TRUE(vernon::tools::packageCompileResult(context, cpuResult, VERNON_TARGET_CPU, computePackaging, output,
                                                    error) == VERNON_STATUS_OK);
    nlohmann::json computeManifest = nlohmann::json::parse(readFile(root / "compute" / "compute.json"));
    ASSERT_TRUE(computeManifest.at("pipeline_version") == VERNON_PIPELINE_VERSION);
    ASSERT_TRUE(computeManifest.at("target") == "cpu");
    ASSERT_TRUE(computeManifest.at("artifact_format") == "relocatable_object");
    ASSERT_TRUE(computeManifest.at("target_triple").get<std::string>() == triple);
    ASSERT_TRUE(computeManifest.at("object_format") == "coff");
    ASSERT_TRUE(computeManifest.at("release_version") == VERNON_RELEASE_VERSION);
    ASSERT_TRUE(std::filesystem::is_regular_file(root / "compute" / computeManifest.at("artifact").get<std::string>()));

    vernon::tools::PackagingOptions outputPackaging;
    outputPackaging.outputDirectory = root / "output";
    ASSERT_TRUE(vernon::tools::packageCompileResult(context, cpuResult, VERNON_TARGET_CPU, outputPackaging, output,
                                                    error) == VERNON_STATUS_OK);
    ASSERT_TRUE(std::filesystem::is_regular_file(root / "output" / "reflection.json"));
    ASSERT_TRUE(std::filesystem::is_regular_file(root / "output" /
                                                 std::string(vernonCompileResultGetArtifactName(cpuResult, 0).data,
                                                             vernonCompileResultGetArtifactName(cpuResult, 0).size)));
    vernonCompileResultDestroy(cpuResult);

    vernonCompilerDestroy(context);
    std::filesystem::remove_all(root);
}
