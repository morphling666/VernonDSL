#include "vernon_compile_packaging.h"

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
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

} // namespace

TEST(CompilePackaging, WritesBundlesAndArtifacts) {
  constexpr std::string_view computeModule = R"mlir(
module {
  func.func @scale() attributes {
    vernon.entry,
    vernon.stage = "compute",
    vernon.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
)mlir";
  constexpr std::string_view graphicsModule = R"mlir(
module {
  func.func @vertex_main(
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 1 : i64
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      },
      tensor<3xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "vertex"} {
    return %position, %color : tensor<4xf32>, tensor<3xf32>
  }
  func.func @fragment_main(
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (tensor<3xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    return %color : tensor<3xf32>
  }
}
)mlir";

  VernonCompilerContext *context = vernonCompilerCreate();
  ASSERT_TRUE(context);
  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("vernon_compile_packaging_" + std::to_string(unique));
  std::filesystem::create_directories(root);
  std::ostringstream output;
  std::ostringstream error;

  VernonCompileOptions cpuOptions{};
  cpuOptions.struct_size = sizeof(cpuOptions);
  constexpr std::string_view triple = "x86_64-pc-windows-msvc";
  cpuOptions.cpu_target_triple = {triple.data(), triple.size()};
  VernonCompileResult *cpuResult = vernonCompilerCompileMlirWithOptions(
      context, computeModule.data(), computeModule.size(), VERNON_TARGET_CPU,
      &cpuOptions);
  ASSERT_TRUE(cpuResult);
  ASSERT_TRUE(vernonCompileResultGetStatus(cpuResult) == VERNON_STATUS_OK);

  vernon::tools::LegacyPackagingOptions computePackaging;
  computePackaging.computeBundlePath = root / "compute";
  computePackaging.targetTriple = std::string(triple);
  ASSERT_TRUE(vernon::tools::packageCompileResult(
                  context, cpuResult, VERNON_TARGET_CPU, computePackaging,
                  output, error) == VERNON_STATUS_OK);
  nlohmann::json computeManifest =
      nlohmann::json::parse(readFile(root / "compute" / "compute.json"));
  ASSERT_TRUE(computeManifest.at("schema_version") == 3);
  ASSERT_TRUE(computeManifest.at("target") == "cpu");
  ASSERT_TRUE(computeManifest.at("artifact_format") == "relocatable_object");
  ASSERT_TRUE(computeManifest.at("target_triple").get<std::string>() == triple);
  ASSERT_TRUE(computeManifest.at("object_format") == "coff");
  ASSERT_TRUE(computeManifest.at("cpu_invocation_abi_version") == 1);
  ASSERT_TRUE(std::filesystem::is_regular_file(
      root / "compute" / computeManifest.at("artifact").get<std::string>()));

  vernon::tools::LegacyPackagingOptions outputPackaging;
  outputPackaging.outputDirectory = root / "output";
  ASSERT_TRUE(vernon::tools::packageCompileResult(
                  context, cpuResult, VERNON_TARGET_CPU, outputPackaging,
                  output, error) == VERNON_STATUS_OK);
  ASSERT_TRUE(
      std::filesystem::is_regular_file(root / "output" / "reflection.json"));
  ASSERT_TRUE(std::filesystem::is_regular_file(
      root / "output" /
      std::string(vernonCompileResultGetArtifactName(cpuResult, 0).data,
                  vernonCompileResultGetArtifactName(cpuResult, 0).size)));
  vernonCompileResultDestroy(cpuResult);

  if (vernonCompilerGetTargetCapabilities(context, VERNON_TARGET_OPENGL)
          .available) {
    VernonCompileOptions glOptions{};
    glOptions.struct_size = sizeof(glOptions);
    glOptions.glsl_version = 330;
    VernonCompileResult *graphicsResult = vernonCompilerCompileMlirWithOptions(
        context, graphicsModule.data(), graphicsModule.size(),
        VERNON_TARGET_OPENGL, &glOptions);
    ASSERT_TRUE(graphicsResult);
    ASSERT_TRUE(vernonCompileResultGetStatus(graphicsResult) ==
                VERNON_STATUS_OK);
    vernon::tools::LegacyPackagingOptions shaderPackaging;
    shaderPackaging.shaderBundlePath = root / "shader";
    shaderPackaging.assetId = "shaders/legacy";
    ASSERT_TRUE(vernon::tools::packageCompileResult(
                    context, graphicsResult, VERNON_TARGET_OPENGL,
                    shaderPackaging, output, error) == VERNON_STATUS_OK);
    nlohmann::json shaderManifest =
        nlohmann::json::parse(readFile(root / "shader" / "shader.json"));
    ASSERT_TRUE(shaderManifest.at("schema_version") == 1);
    ASSERT_TRUE(shaderManifest.at("type") == "compiled_shader_bundle");
    ASSERT_TRUE(shaderManifest.at("id") == "shaders/legacy");
    ASSERT_TRUE(shaderManifest.at("target") == "opengl");
    for (size_t index = 0;
         index < vernonCompileResultGetArtifactCount(graphicsResult); ++index) {
      VernonStringView name =
          vernonCompileResultGetArtifactName(graphicsResult, index);
      ASSERT_TRUE(std::filesystem::is_regular_file(
          root / "shader" / std::string(name.data, name.size)));
    }
    vernonCompileResultDestroy(graphicsResult);
  }

  vernonCompilerDestroy(context);
  std::filesystem::remove_all(root);
}
