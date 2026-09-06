#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string_view>

namespace {
TEST(PipelineManifestRequirements, ValidatesCanonicalDerivativeGroupsAndRejectsDivergentMetadata) {
    using namespace vernon::runtime;
    std::string error;
    const std::vector<AutodiffDerivativeGroup> canonical{
        {AutodiffDerivativeRole::Gradient, "x", {"x.mass", "x.velocity"}},
        {AutodiffDerivativeRole::Cotangent, "output", {"output.mass", "output.velocity"}},
    };
    EXPECT_TRUE(validateAutodiffDerivativeGroups(canonical, error)) << error;

    std::vector<AutodiffDerivativeGroup> invalid = canonical;
    std::swap(invalid[0], invalid[1]);
    error.clear();
    EXPECT_FALSE(validateAutodiffDerivativeGroups(invalid, error));
    EXPECT_NE(error.find("role order"), std::string::npos);

    invalid = {
        {AutodiffDerivativeRole::Gradient, "x", {"x.mass"}},
        {AutodiffDerivativeRole::Gradient, "x.mass", {"x.mass.value"}},
        {AutodiffDerivativeRole::Cotangent, "output", {"output.mass"}},
    };
    error.clear();
    EXPECT_FALSE(validateAutodiffDerivativeGroups(invalid, error));
    EXPECT_NE(error.find("exactly one declared owner"), std::string::npos);

    invalid = canonical;
    invalid[1].leafPaths = {"output..mass"};
    error.clear();
    EXPECT_FALSE(validateAutodiffDerivativeGroups(invalid, error));
    EXPECT_NE(error.find("leaves"), std::string::npos);

    invalid = canonical;
    invalid[1].leafPaths = {"output.a速"};
    error.clear();
    EXPECT_FALSE(validateAutodiffDerivativeGroups(invalid, error));
    EXPECT_NE(error.find("leaves"), std::string::npos);
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

#if defined(VERNON_RUNTIME_PROFILE_WEB)
    constexpr const char *triple = "wasm32-unknown-emscripten";
    constexpr const char *format = "wasm";
#elif defined(_M_ARM64) || defined(__aarch64__)
#define VERNON_TEST_TRIPLE_ARCH "aarch64"
#else
#define VERNON_TEST_TRIPLE_ARCH "x86_64"
#endif
#if !defined(VERNON_RUNTIME_PROFILE_WEB)
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
#endif
    std::string error;
    EXPECT_TRUE(validateCpuRuntimeRequirements(triple, format, error)) << error;
#if defined(VERNON_RUNTIME_PROFILE_WEB)
    EXPECT_FALSE(validateCpuRuntimeRequirements(triple, "macho", error));
#else
    EXPECT_FALSE(validateCpuRuntimeRequirements(triple, "wasm", error));
#endif
    EXPECT_NE(error.find("runtime provides"), std::string::npos);
}

TEST(PipelineManifestRequirements, ParsesStructuredLeafPathsAndStaticShapes) {
    const nlohmann::json layout = {
        {"logical_type", "!vernon.struct<\"Payload\">"},
        {"struct_name", "Payload"},
        {"byte_size", 24},
        {"alignment", 4},
        {"layout_hash", "layout"},
        {"leaves", nlohmann::json::array({{{"path", nlohmann::json::array({"nested", 1})},
                                           {"dtype", "f32"},
                                           {"byte_offset", 8},
                                           {"scalar_count", 4},
                                           {"shape", nlohmann::json::array({2, 2})}}})},
    };
    vernon::runtime::ValueLayout parsed;
    std::string error;
    ASSERT_TRUE(vernon::runtime::parsePipelineValueLayout(layout, parsed, error)) << error;
    ASSERT_EQ(parsed.leaves.size(), 1u);
    ASSERT_EQ(parsed.leaves[0].path.size(), 2u);
    EXPECT_EQ(parsed.leaves[0].path[0].field, "nested");
    EXPECT_FALSE(parsed.leaves[0].path[1].field);
    EXPECT_EQ(parsed.leaves[0].path[1].index, 1u);
    EXPECT_EQ(parsed.leaves[0].shape, (std::vector<uint64_t>{2, 2}));
    ASSERT_EQ(parsed.leaves[0].abiPath.size(), 2u);
    EXPECT_EQ(parsed.leaves[0].abiPath[0].kind, VERNON_VALUE_PATH_FIELD);
    EXPECT_EQ(parsed.leaves[0].abiPath[1].kind, VERNON_VALUE_PATH_INDEX);

    nlohmann::json invalid = layout;
    invalid["leaves"][0]["shape"] = nlohmann::json::array({4, 2});
    EXPECT_FALSE(vernon::runtime::parsePipelineValueLayout(invalid, parsed = {}, error));
    EXPECT_NE(error.find("scalar_count"), std::string::npos);

    invalid = layout;
    invalid["leaves"][0]["byte_offset"] = 12;
    EXPECT_FALSE(vernon::runtime::parsePipelineValueLayout(invalid, parsed, error));
    EXPECT_NE(error.find("byte offset"), std::string::npos);
    EXPECT_TRUE(parsed.leaves.empty());

    invalid = layout;
    invalid["leaves"].push_back(
        {{"path", nlohmann::json::array({"overlap"})}, {"dtype", "f32"}, {"byte_offset", 20}, {"scalar_count", 1}});
    EXPECT_FALSE(vernon::runtime::parsePipelineValueLayout(invalid, parsed, error));
    EXPECT_NE(error.find("overlap"), std::string::npos);

    invalid = layout;
    invalid["leaves"][0]["path"] = nlohmann::json::array({"nested.field"});
    EXPECT_FALSE(vernon::runtime::parsePipelineValueLayout(invalid, parsed, error));
    EXPECT_NE(error.find("cannot contain"), std::string::npos);

    invalid = layout;
    invalid["byte_size"] = 40;
    invalid["leaves"].push_back(invalid["leaves"][0]);
    invalid["leaves"][1]["byte_offset"] = 24;
    EXPECT_FALSE(vernon::runtime::parsePipelineValueLayout(invalid, parsed, error));
    EXPECT_NE(error.find("not unique"), std::string::npos);
}

TEST(PipelineManifestRequirements, ParsesSharedInterfacePlansTransactionally) {
    const nlohmann::json source = {
        {"kind", "native_uniform"},
        {"profile", "opengl_native_uniform"},
        {"canonical_layout_hash", "layout"},
        {"frame_offset", 16},
        {"root",
         {{"kind", "array"},
          {"offset", 0},
          {"size", 16},
          {"alignment", 4},
          {"shape", nlohmann::json::array({4})},
          {"byte_strides", nlohmann::json::array({4})},
          {"children",
           nlohmann::json::array(
               {{{"kind", "scalar"}, {"representation", "f32"}, {"offset", 0}, {"size", 4}, {"alignment", 4}}})}}},
    };
    vernon::runtime::InterfacePlan plan;
    std::string error;
    ASSERT_TRUE(vernon::runtime::parsePipelineInterfacePlan(source, plan, error)) << error;
    EXPECT_EQ(plan.kind, vernon::runtime::InterfacePlanKind::NativeUniform);
    EXPECT_EQ(plan.profile, "opengl_native_uniform");
    EXPECT_EQ(plan.frameOffset, 16u);
    ASSERT_TRUE(plan.root.has_value());
    EXPECT_EQ(plan.root->children.size(), 1u);

    nlohmann::json invalid = source;
    invalid["root"]["byte_strides"] = nlohmann::json::array({0});
    EXPECT_FALSE(vernon::runtime::parsePipelineInterfacePlan(invalid, plan, error));

    invalid = source;
    invalid["unknown"] = true;
    EXPECT_FALSE(vernon::runtime::parsePipelineInterfacePlan(invalid, plan, error));
}

} // namespace
