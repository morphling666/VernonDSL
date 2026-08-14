#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string_view>

namespace {

using vernon::runtime::parseRuntimeRequirements;
using vernon::runtime::RuntimeRequirements;

nlohmann::json validAutodiffManifest() {
    const nlohmann::json primal = {{"path", "x"}, {"type", "f32"}, {"role", "primal"}};
    const nlohmann::json output = {{"path", "output"}, {"type", "f32"}, {"role", "primal"}};
    const nlohmann::json tape = {{"path", "tape"}, {"type", "!vernon.ad_tape<16>"}, {"role", "tape"}};
    const nlohmann::json cotangent = {{"path", "output"}, {"type", "f32"}, {"role", "cotangent"}};
    const nlohmann::json gradient = {{"path", "x"}, {"type", "f32"}, {"role", "gradient"}};
    return {{"autodiff",
             {{"kind", "vjp"},
              {"protocol", "dynamic_v2"},
              {"wrt", nlohmann::json::array({"x"})},
              {"output_cotangents", nlohmann::json::array({"output"})},
              {"variants", nlohmann::json::array({{{"key", nlohmann::json::array()},
                                                   {"workgroup_size", nlohmann::json::array({1, 1, 1})},
                                                   {"residual_storage", "static"},
                                                   {"static_tape_bytes_hint", 16},
                                                   {"required_primal_paths", nlohmann::json::array()},
                                                   {"source_kind_counts", {{"static_capture", 1}}},
                                                   {"cost_components",
                                                    {{"backward_load_bytes", 4},
                                                     {"capture_store_bytes", 4},
                                                     {"checkpoint_copy_bytes", 0},
                                                     {"graph_replay_cost", 0},
                                                     {"recomputation_cost", 0},
                                                     {"resource_reload_cost", 0},
                                                     {"retained_tape_bytes", 16}}},
                                                   {"selected_policy", "min_memory"},
                                                   {"profiles",
                                                    {{"primal",
                                                      {{"compute", "primal"},
                                                       {"inputs", nlohmann::json::array({primal})},
                                                       {"outputs", nlohmann::json::array({output})}}},
                                                     {"forward_with_tape",
                                                      {{"compute", "forward"},
                                                       {"inputs", nlohmann::json::array({primal})},
                                                       {"outputs", nlohmann::json::array({output, tape})}}},
                                                     {"backward",
                                                      {{"compute", "backward"},
                                                       {"inputs", nlohmann::json::array({tape, cotangent})},
                                                       {"outputs", nlohmann::json::array({gradient})}}}}}}})}}}};
}

nlohmann::json validTensorVariant() {
    return {{"key", nlohmann::json::array()},
            {"program", {{"compute", "compute.spv"}}},
            {"parameters",
             nlohmann::json::array(
                 {{{"slot", 0},
                   {"name", "values"},
                   {"kind", "tensor"},
                   {"type", "!vernon.tensor_view<f32, [1], \"read_write\", \"device\">"},
                   {"access", "read_write"},
                   {"address_space", "device"},
                   {"shape", nlohmann::json::array({1})},
                   {"element_layout",
                    {{"logical_type", "f32"},
                     {"layout_hash", "cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b"},
                     {"byte_size", 4},
                     {"alignment", 4},
                     {"leaves", nlohmann::json::array({{{"path", nlohmann::json::array()},
                                                        {"dtype", "f32"},
                                                        {"scalar_count", 1},
                                                        {"byte_offset", 0}}})}}},
                   {"uses", nlohmann::json::array({{{"stage", "compute"},
                                                    {"interface", "resource"},
                                                    {"vernon.set", 0},
                                                    {"vernon.binding", 0},
                                                    {"shape", nlohmann::json::array({1})},
                                                    {"tensor_view_descriptor",
                                                     {{"rank", 1},
                                                      {"offset_binding", 1},
                                                      {"extent_bindings", nlohmann::json::array({2})},
                                                      {"stride_bindings", nlohmann::json::array({3})}}}}})}}})},
            {"internal_parameters", nlohmann::json::array()},
            {"outputs", nlohmann::json::array()}};
}

TEST(PipelineManifestRequirements, RejectsMissingRuntimeRequirements) {
    RuntimeRequirements requirements;
    std::string error;
    EXPECT_FALSE(parseRuntimeRequirements(nlohmann::json::object(), "cpu", requirements, error));
    EXPECT_EQ(error, "pipeline manifest requires runtime_requirements");
}

TEST(PipelineManifestRequirements, ParsesCanonicalAutodiffProfiles) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    AutodiffManifest manifest;
    std::string error;
    const nlohmann::json root = validAutodiffManifest();
    ASSERT_TRUE(parseAutodiffManifest(root, manifest, error)) << error;
    EXPECT_EQ(manifest.protocol, "dynamic_v2");
    ASSERT_EQ(manifest.derivativeGroups.size(), 2u);
    EXPECT_EQ(manifest.derivativeGroups[0].role, vernon::runtime::AutodiffDerivativeRole::Gradient);
    EXPECT_EQ(manifest.derivativeGroups[0].declaredPath, "x");
    EXPECT_EQ(manifest.derivativeGroups[0].leafPaths, std::vector<std::string>{"x"});
    EXPECT_EQ(manifest.derivativeGroups[1].role, vernon::runtime::AutodiffDerivativeRole::Cotangent);
    EXPECT_EQ(manifest.derivativeGroups[1].declaredPath, "output");
    EXPECT_EQ(manifest.derivativeGroups[1].leafPaths, std::vector<std::string>{"output"});
    ASSERT_EQ(manifest.variants.size(), 1u);
    EXPECT_EQ(manifest.variants[0].forwardWithTape, "forward");
    EXPECT_EQ(manifest.variants[0].backward, "backward");
    EXPECT_EQ(manifest.variants[0].staticTapeBytesHint, 16u);
    EXPECT_EQ(manifest.variants[0].launch.workgroupSize.x, 1u);
}

TEST(PipelineManifestRequirements, ParsesNoTapeProfilesWithExplicitPrimals) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    nlohmann::json root = validAutodiffManifest();
    nlohmann::json &variant = root["autodiff"]["variants"][0];
    variant["residual_storage"] = "none";
    variant["static_tape_bytes_hint"] = 0;
    variant["required_primal_paths"] = nlohmann::json::array({"primal.x"});
    variant["source_kind_counts"] = {{"primal_argument", 1}};
    variant["cost_components"]["backward_load_bytes"] = 0;
    variant["cost_components"]["capture_store_bytes"] = 0;
    variant["cost_components"]["retained_tape_bytes"] = 0;
    variant["profiles"]["forward_with_tape"]["outputs"] = nlohmann::json::array();
    const nlohmann::json primal = {{"path", "primal.x"}, {"type", "f32"}, {"role", "primal"}};
    const nlohmann::json cotangent = {{"path", "output"}, {"type", "f32"}, {"role", "cotangent"}};
    variant["profiles"]["backward"]["inputs"] = nlohmann::json::array({primal, cotangent});

    AutodiffManifest manifest;
    std::string error;
    ASSERT_TRUE(parseAutodiffManifest(root, manifest, error)) << error;
    ASSERT_EQ(manifest.variants.size(), 1u);
    EXPECT_EQ(manifest.variants[0].residualStorage, "none");
    EXPECT_EQ(manifest.variants[0].staticTapeBytesHint, 0u);
    EXPECT_EQ(manifest.variants[0].requiredPrimalPaths, std::vector<std::string>{"primal.x"});
}

TEST(PipelineManifestRequirements, RejectsLegacyAutodiffMetadata) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    nlohmann::json root = validAutodiffManifest();
    root["autodiff"]["gradient_policy"] = "f16:f32,f32:f32,f64:f64";

    AutodiffManifest manifest;
    std::string error;
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("autodiff object"), std::string::npos);
}

TEST(PipelineManifestRequirements, RejectsUnsupportedAutodiffProtocol) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    nlohmann::json root = validAutodiffManifest();
    root["autodiff"]["protocol"] = "unsupported_protocol";

    AutodiffManifest manifest;
    std::string error;
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("autodiff object"), std::string::npos);
}

TEST(PipelineManifestRequirements, RejectsIncompleteOrUnknownAutodiffProfiles) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    std::string error;
    AutodiffManifest manifest;
    nlohmann::json root = validAutodiffManifest();
    root["autodiff"]["protocol"] = "unknown";
    error.clear();
    manifest = {};
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("autodiff object"), std::string::npos);

    root = validAutodiffManifest();
    root["autodiff"]["variants"][0]["profiles"]["backward"]["symbol"] = "__backward";
    error.clear();
    manifest = {};
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("profile"), std::string::npos);

    root = validAutodiffManifest();
    root["autodiff"]["variants"][0]["profiles"]["forward_with_tape"]["outputs"][1]["role"] = "primal";
    error.clear();
    manifest = {};
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("forward bindings"), std::string::npos);
}

TEST(PipelineManifestRequirements, RejectsNonCanonicalAutodiffPaths) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    AutodiffManifest manifest;
    std::string error;

    nlohmann::json root = validAutodiffManifest();
    root["autodiff"]["variants"][0]["profiles"]["backward"]["inputs"][1]["path"] = "output..mass";
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("bindings"), std::string::npos);
}

TEST(PipelineManifestRequirements, RejectsInconsistentAutodiffTapeBindings) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    AutodiffManifest manifest;
    std::string error;

    nlohmann::json root = validAutodiffManifest();
    root["autodiff"]["variants"][0]["profiles"]["backward"]["inputs"][0]["path"] = "saved";
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("backward tape binding"), std::string::npos);

    root = validAutodiffManifest();
    root["autodiff"]["variants"][0]["profiles"]["backward"]["inputs"][0]["type"] = "!vernon.ad_tape<32>";
    error.clear();
    manifest = {};
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("backward tape binding"), std::string::npos);
}

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

TEST(PipelineManifestRequirements, ParsesEveryRuntimeBackendShape) {
    const std::pair<const char *, nlohmann::json> cases[] = {
        {"cpu",
         {{"backend", "cpu"},
          {"features", nlohmann::json::array({"compute", "tensor_views"})},
          {"target_triple", "x86_64-pc-windows-msvc"},
          {"object_format", "coff"}}},
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
        {"metal",
         {{"backend", "metal"},
          {"features", nlohmann::json::array({"compute"})},
          {"apple_platform", "ios"},
          {"msl_version", nlohmann::json::array({2, 4})},
          {"minimum_os_version", nlohmann::json::array({15, 0})}}},
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

TEST(PipelineManifestRequirements, RejectsUnsupportedMetalRequirements) {
    RuntimeRequirements requirements;
    std::string error;
    nlohmann::json root = {{"runtime_requirements",
                            {{"backend", "metal"},
                             {"features", nlohmann::json::array()},
                             {"apple_platform", "ios"},
                             {"msl_version", nlohmann::json::array({2, 3})},
                             {"minimum_os_version", nlohmann::json::array({15, 0})}}}};
    EXPECT_FALSE(parseRuntimeRequirements(root, "metal", requirements, error));
    EXPECT_NE(error.find("unsupported"), std::string::npos);

    root["runtime_requirements"]["msl_version"] = nlohmann::json::array({2, 4});
    root["runtime_requirements"]["minimum_os_version"] = nlohmann::json::array({14, 9});
    error.clear();
    requirements = {};
    EXPECT_FALSE(parseRuntimeRequirements(root, "metal", requirements, error));
    EXPECT_NE(error.find("unsupported"), std::string::npos);

    root["runtime_requirements"]["minimum_os_version"] = nlohmann::json::array({15, 0});
    root["runtime_requirements"]["apple_platform"] = "tvos";
    error.clear();
    requirements = {};
    EXPECT_FALSE(parseRuntimeRequirements(root, "metal", requirements, error));
    EXPECT_NE(error.find("unsupported"), std::string::npos);
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

TEST(PipelineManifestRequirements, ParsesReflectedUniformTensorLayout) {
    const nlohmann::json manifest = {
        {"key", nlohmann::json::array()},
        {"program", {{"vertex", "vertex.spv"}, {"fragment", "fragment.spv"}}},
        {"parameters", nlohmann::json::array(
                           {{{"slot", 0},
                             {"name", "weights"},
                             {"kind", "tensor"},
                             {"type", "tensor<2x3xf32>"},
                             {"access", "read"},
                             {"element_layout",
                              {{"logical_type", "f32"},
                               {"byte_size", 4},
                               {"alignment", 4},
                               {"layout_hash", "cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b"},
                               {"leaves", nlohmann::json::array({{{"path", nlohmann::json::array()},
                                                                  {"dtype", "f32"},
                                                                  {"byte_offset", 0},
                                                                  {"scalar_count", 1}}})}}},
                             {"shape", nlohmann::json::array({2, 3})},
                             {"uses", nlohmann::json::array(
                                          {{{"stage", "vertex"},
                                            {"interface", "uniform"},
                                            {"index", 0},
                                            {"shape", nlohmann::json::array({2, 3})},
                                            {"vernon.set", 0},
                                            {"vernon.binding", 2},
                                            {"transport", "uniform_buffer"},
                                            {"interface_plan",
                                             {{"kind", "byte_transport"},
                                              {"profile", "vulkan_std140_uniform_buffer"},
                                              {"canonical_layout_hash", "tensor-layout"},
                                              {"root",
                                               {{"kind", "array"},
                                                {"offset", 0},
                                                {"size", 32},
                                                {"alignment", 16},
                                                {"shape", nlohmann::json::array({2, 3})},
                                                {"byte_strides", nlohmann::json::array({16, 4})},
                                                {"children", nlohmann::json::array({{{"kind", "scalar"},
                                                                                     {"representation", "f32"},
                                                                                     {"offset", 0},
                                                                                     {"size", 4},
                                                                                     {"alignment", 4}}})}}}}}}})}}})},
        {"outputs", nlohmann::json::array()}};
    vernon::runtime::Variant variant;
    std::string error;
    ASSERT_TRUE(vernon::runtime::parseVariant(manifest, variant, error)) << error;
    ASSERT_EQ(variant.parameters.size(), 1u);
    ASSERT_EQ(variant.parameters[0].uses.size(), 1u);
    const auto &layout = variant.parameters[0].uses[0].interfacePlan;
    ASSERT_TRUE(layout);
    EXPECT_EQ(layout->profile, "vulkan_std140_uniform_buffer");
    EXPECT_EQ(variant.parameters[0].uses[0].transport, "uniform_buffer");
    ASSERT_TRUE(layout->root);
    EXPECT_EQ(layout->root->size, 32u);
    EXPECT_EQ(layout->root->alignment, 16u);
    EXPECT_EQ(layout->root->byteStrides, (std::vector<uint64_t>{16, 4}));
    ASSERT_EQ(variant.parameters[0].elementLayout.leaves.size(), 1u);
    EXPECT_TRUE(variant.parameters[0].elementLayout.leaves[0].path.empty());

    nlohmann::json invalid = manifest;
    invalid["parameters"][0]["uses"][0]["interface_plan"]["root"]["children"] = nlohmann::json::array();
    EXPECT_FALSE(vernon::runtime::parseVariant(invalid, variant, error));
    EXPECT_NE(error.find("topology"), std::string::npos);

    invalid = manifest;
    invalid["parameters"][0]["uses"][0]["interface_plan"]["root"]["children"][0]["offset"] = 30;
    EXPECT_FALSE(vernon::runtime::parseVariant(invalid, variant, error));
    EXPECT_NE(error.find("bounds"), std::string::npos);

    invalid = manifest;
    invalid["parameters"][0]["uses"][0]["interface_plan"]["root"]["children"][0]["size"] = 8;
    EXPECT_FALSE(vernon::runtime::parseVariant(invalid, variant, error));
    EXPECT_NE(error.find("representation"), std::string::npos);

    invalid = manifest;
    invalid["parameters"][0]["uses"][0]["interface_plan"]["root"]["children"][0]["representation"] = "f64";
    EXPECT_FALSE(vernon::runtime::parseVariant(invalid, variant, error));
    EXPECT_NE(error.find("representation"), std::string::npos);

    invalid = manifest;
    invalid["parameters"][0]["uses"][0]["value_layout"] = invalid["parameters"][0]["element_layout"];
    EXPECT_FALSE(vernon::runtime::parseVariant(invalid, variant, error));
    EXPECT_NE(error.find("canonical layout hash"), std::string::npos);
}

TEST(PipelineManifestRequirements, RejectsUnknownAndLegacyVariantRecords) {
    vernon::runtime::Variant variant;
    std::string error;
    nlohmann::json manifest = validTensorVariant();
    manifest["unknown"] = true;
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));

    manifest = validTensorVariant();
    manifest["parameters"][0]["unknown"] = true;
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));

    manifest = validTensorVariant();
    manifest["parameters"][0]["uses"][0]["unknown"] = true;
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));

    for (std::string_view retired : {"entry", "kind", "type", "access", "address_space", "dimension", "location_span",
                                     "internal_source", "system_value"}) {
        manifest = validTensorVariant();
        manifest["parameters"][0]["uses"][0][retired] = "retired";
        EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error)) << retired;
    }

    manifest = validTensorVariant();
    manifest["parameters"][0]["vernon.compiler_generated"] = true;
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));

    manifest = validTensorVariant();
    manifest["outputs"].push_back(
        {{"name", "color"}, {"kind", "tensor"}, {"dtype", "f32"}, {"shape", {4}}, {"location", 0}, {"unknown", true}});
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));
}

TEST(PipelineManifestRequirements, RejectsParameterKindTypeAndAddressSpaceMismatch) {
    vernon::runtime::Variant variant;
    std::string error;
    nlohmann::json manifest = validTensorVariant();
    manifest["parameters"][0]["kind"] = "texture";
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));

    manifest = validTensorVariant();
    manifest["parameters"][0]["address_space"] = "workgroup";
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));

    manifest = validTensorVariant();
    manifest["parameters"][0]["type"] = "!vernon.sampler";
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));
}

TEST(PipelineManifestRequirements, RejectsAmbiguousOrUnresolvedResourceBindings) {
    const auto image = [](uint32_t slot, const char *name, uint32_t binding) {
        return nlohmann::json{{"slot", slot},
                              {"name", name},
                              {"kind", "image"},
                              {"type", "!vernon.texture<\"2d\", f32, \"unknown\", \"sampled\">"},
                              {"access", "read"},
                              {"shape", nlohmann::json::array()},
                              {"dimension", "2d"},
                              {"binding_role", "sampled"},
                              {"sample_result_class", "float"},
                              {"uses", nlohmann::json::array({{{"stage", "fragment"},
                                                               {"interface", "resource"},
                                                               {"vernon.set", 0},
                                                               {"vernon.binding", binding}}})}};
    };
    nlohmann::json manifest = {{"key", nlohmann::json::array()},
                               {"program", {{"vertex", "vertex"}, {"fragment", "fragment"}}},
                               {"parameters", nlohmann::json::array({image(0, "left", 3), image(1, "right", 3)})},
                               {"outputs", nlohmann::json::array()}};
    vernon::runtime::Variant variant;
    std::string error;
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));
    EXPECT_EQ(error, "pipeline descriptor binding is assigned to multiple parameters");

    manifest["parameters"] = nlohmann::json::array(
        {{{"slot", 0},
          {"name", "sampler"},
          {"kind", "sampler"},
          {"type", "!vernon.sampler"},
          {"access", "read"},
          {"shape", nlohmann::json::array()},
          {"uses", nlohmann::json::array(
                       {{{"stage", "fragment"},
                         {"interface", "resource"},
                         {"sampled_image_bindings", nlohmann::json::array({{{"set", 0}, {"binding", 3}}})}}})}}});
    variant = {};
    error.clear();
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));
    EXPECT_EQ(error, "sampler references an unknown sampled image binding");
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

} // namespace
