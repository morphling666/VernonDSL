#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

using vernon::runtime::parseRuntimeRequirements;
using vernon::runtime::RuntimeRequirements;

TEST(PipelineManifestRequirements, RejectsMissingRuntimeRequirements) {
    RuntimeRequirements requirements;
    std::string error;
    EXPECT_FALSE(parseRuntimeRequirements(nlohmann::json::object(), "cpu", requirements, error));
    EXPECT_EQ(error, "pipeline manifest requires runtime_requirements");
}

TEST(PipelineManifestRequirements, ParsesEveryRuntimeBackendShape) {
    const std::pair<const char *, nlohmann::json> cases[] = {
        {"cpu",
         {{"backend", "cpu"},
          {"features", nlohmann::json::array({"compute", "tensor_views"})},
          {"target_triple", "x86_64-pc-windows-msvc"},
          {"object_format", "coff"},
          {"invocation_abi_version", 1}}},
        {"opengl",
         {{"backend", "opengl"},
          {"features", nlohmann::json::array({"textures"})},
          {"glsl_version", 330},
          {"profile", "core"},
          {"api_version", nlohmann::json::array({3, 3})}}},
        {"opengles",
         {{"backend", "opengles"},
          {"features", nlohmann::json::array()},
          {"glsl_version", 310},
          {"profile", "es"},
          {"api_version", nlohmann::json::array({3, 1})}}},
        {"vulkan",
         {{"backend", "vulkan"},
          {"features", nlohmann::json::array({"compute"})},
          {"api_version", nlohmann::json::array({1, 1})},
          {"spirv_version", nlohmann::json::array({1, 3})},
          {"compute_workgroup_size", nlohmann::json::array({8, 4, 1})}}},
        {"cuda",
         {{"backend", "cuda"},
          {"features", nlohmann::json::array({"compute"})},
          {"ptx_version", nlohmann::json::array({8, 0})},
          {"minimum_compute_capability", nlohmann::json::array({5, 0})},
          {"address_size", 64}}},
        {"directx",
         {{"backend", "directx"},
          {"features", nlohmann::json::array({"compute", "tensor_views"})},
          {"api_version", nlohmann::json::array({12, 0})},
          {"minimum_feature_level", nlohmann::json::array({11, 0})},
          {"shader_model", nlohmann::json::array({6, 0})},
          {"root_signature_version", nlohmann::json::array({1, 0})},
          {"compute_workgroup_size", nlohmann::json::array({8, 1, 1})}}},
    };
    for (const auto &[target, value] : cases) {
        RuntimeRequirements requirements;
        std::string error;
        const nlohmann::json root = {{"runtime_requirements", value}};
        EXPECT_TRUE(parseRuntimeRequirements(root, target, requirements, error)) << target << ": " << error;
        EXPECT_EQ(requirements.backend, target);
    }
}

TEST(PipelineManifestRequirements, RejectsNonCanonicalOrUnknownData) {
    RuntimeRequirements requirements;
    std::string error;
    nlohmann::json root = {{"runtime_requirements",
                            {{"backend", "vulkan"},
                             {"features", nlohmann::json::array({"compute", "compute"})},
                             {"api_version", nlohmann::json::array({1, 1})},
                             {"spirv_version", nlohmann::json::array({1, 3})}}}};
    EXPECT_FALSE(parseRuntimeRequirements(root, "vulkan", requirements, error));
    EXPECT_NE(error.find("unique"), std::string::npos);

    root["runtime_requirements"]["features"] = nlohmann::json::array({"future_feature"});
    error.clear();
    requirements = {};
    EXPECT_FALSE(parseRuntimeRequirements(root, "vulkan", requirements, error));
    EXPECT_NE(error.find("known"), std::string::npos);

    root["runtime_requirements"]["features"] = nlohmann::json::array();
    root["runtime_requirements"]["unknown"] = true;
    error.clear();
    requirements = {};
    EXPECT_FALSE(parseRuntimeRequirements(root, "vulkan", requirements, error));
    EXPECT_NE(error.find("invalid"), std::string::npos);
}

TEST(PipelineManifestRequirements, RejectsTargetMismatchAndMalformedVersions) {
    RuntimeRequirements requirements;
    std::string error;
    nlohmann::json root = {{"runtime_requirements",
                            {{"backend", "opengl"},
                             {"features", nlohmann::json::array()},
                             {"glsl_version", 330},
                             {"profile", "core"},
                             {"api_version", "3.3"}}}};
    EXPECT_FALSE(parseRuntimeRequirements(root, "opengl", requirements, error));

    root["runtime_requirements"]["api_version"] = nlohmann::json::array({3, 3});
    error.clear();
    requirements = {};
    EXPECT_FALSE(parseRuntimeRequirements(root, "opengles", requirements, error));
    EXPECT_NE(error.find("does not match"), std::string::npos);
}

TEST(PipelineManifestRequirements, ComparesApiAndCpuHostRequirements) {
    using vernon::runtime::glslVersionForApi;
    using vernon::runtime::RuntimeVersion;
    using vernon::runtime::runtimeVersionAtLeast;
    using vernon::runtime::validateCpuRuntimeRequirements;

    EXPECT_TRUE(runtimeVersionAtLeast({4, 6}, {4, 5}));
    EXPECT_TRUE(runtimeVersionAtLeast({5, 0}, {4, 6}));
    EXPECT_FALSE(runtimeVersionAtLeast({4, 5}, {4, 6}));
    EXPECT_EQ(glslVersionForApi(RuntimeVersion{3, 3}), 330u);

#if defined(_M_ARM64) || defined(__aarch64__)
#define VERNON_TEST_TRIPLE_ARCH "aarch64"
#else
#define VERNON_TEST_TRIPLE_ARCH "x86_64"
#endif
#if defined(_WIN32)
    constexpr const char *triple = VERNON_TEST_TRIPLE_ARCH "-pc-windows-msvc";
    constexpr const char *format = "coff";
#elif defined(__APPLE__)
    constexpr const char *triple = VERNON_TEST_TRIPLE_ARCH "-apple-darwin";
    constexpr const char *format = "macho";
#else
    constexpr const char *triple = VERNON_TEST_TRIPLE_ARCH "-unknown-linux-gnu";
    constexpr const char *format = "elf";
#endif
#undef VERNON_TEST_TRIPLE_ARCH
    std::string error;
    EXPECT_TRUE(validateCpuRuntimeRequirements(triple, format, VERNON_CPU_INVOCATION_ABI_VERSION, error)) << error;
    EXPECT_FALSE(validateCpuRuntimeRequirements(triple, "wasm", VERNON_CPU_INVOCATION_ABI_VERSION, error));
    EXPECT_NE(error.find("runtime provides"), std::string::npos);
}

} // namespace
