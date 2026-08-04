#include "runtime/content_hash.h"
#include "runtime/pipeline_bundle.h"
#include "runtime/pipeline_manifest.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string_view>

namespace {

using vernon::runtime::parseRuntimeRequirements;
using vernon::runtime::RuntimeRequirements;

nlohmann::json withIdentity(nlohmann::json value) {
    const std::string canonical = value.dump(-1, ' ', false);
    value["identity"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return value;
}

nlohmann::json validAutodiffManifest() {
    nlohmann::json transform = withIdentity({{"kind", "vjp"},
                                             {"wrt", nlohmann::json::array({"x"})},
                                             {"output_cotangents", nlohmann::json::array({"output"})},
                                             {"gradient_policy", "f16:f32,f32:f32,f64:f64"},
                                             {"accumulation_policy", "fresh"},
                                             {"tape_policy", "bounded"},
                                             {"derivative_rules_version", 1}});
    const std::string transformIdentity = transform["identity"].get<std::string>();
    const nlohmann::json primal = {{"path", "x"}, {"type", "f32"}, {"role", "primal"}};
    const nlohmann::json output = {{"path", "output"}, {"type", "f32"}, {"role", "primal"}};
    const nlohmann::json tape = {{"path", "tape"}, {"type", "!vernon.ad_tape<16>"}, {"role", "tape"}};
    const nlohmann::json cotangent = {{"path", "output"}, {"type", "f32"}, {"role", "cotangent"}};
    const nlohmann::json gradient = {{"path", "x"}, {"type", "f32"}, {"role", "gradient"}};
    nlohmann::json plan = withIdentity(
        {{"transform_identity", transformIdentity},
         {"program_graph_identity", std::string(64, 'a')},
         {"tape_bytes", 16},
         {"derivative_rules", nlohmann::json::array({"mul"})},
         {"derivative_rules_version", 1},
         {"launch",
          {{"workgroup_size", nlohmann::json::array({1, 1, 1})},
           {"accumulation_plans", nlohmann::json::array({{{"path", "x"},
                                                          {"mode", "reduce_sum"},
                                                          {"evidence", nlohmann::json::array({"shared_value"})},
                                                          {"invocation_axes", nlohmann::json::array()}}})}}},
         {"profiles", nlohmann::json::array({{{"name", "primal"},
                                              {"symbol", "main"},
                                              {"inputs", nlohmann::json::array({primal})},
                                              {"outputs", nlohmann::json::array({output})}},
                                             {{"name", "forward_with_tape"},
                                              {"symbol", "__forward"},
                                              {"inputs", nlohmann::json::array({primal})},
                                              {"outputs", nlohmann::json::array({output, tape})}},
                                             {{"name", "backward"},
                                              {"symbol", "__backward"},
                                              {"inputs", nlohmann::json::array({tape, cotangent})},
                                              {"outputs", nlohmann::json::array({gradient})}}})}});
    nlohmann::json profiles =
        withIdentity({{"variants", nlohmann::json::array({{{"key", nlohmann::json::array()},
                                                           {"plan", plan},
                                                           {"programs",
                                                            {{"primal", {{"compute", "primal"}}},
                                                             {"forward_with_tape", {{"compute", "forward"}}},
                                                             {"backward", {{"compute", "backward"}}}}}}})}});
    return {{"program_transform", std::move(transform)}, {"autodiff_profiles", std::move(profiles)}};
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
    EXPECT_EQ(manifest.wrt, std::vector<std::string>{"x"});
    EXPECT_EQ(manifest.outputCotangents, std::vector<std::string>{"output"});
    EXPECT_EQ(manifest.gradientPaths, std::vector<std::string>{"x"});
    ASSERT_EQ(manifest.variants.size(), 1u);
    EXPECT_EQ(manifest.variants[0].forwardWithTape, "forward");
    EXPECT_EQ(manifest.variants[0].backward, "backward");
    EXPECT_EQ(manifest.variants[0].tapeBytes, 16u);
    ASSERT_EQ(manifest.variants[0].launch.accumulationPlans.size(), 1u);
    EXPECT_EQ(manifest.variants[0].launch.accumulationPlans[0].path, "x");
    EXPECT_EQ(manifest.variants[0].launch.accumulationPlans[0].operation,
              vernon::runtime::AutodiffAccumulationPlan::Operation::ReduceSum);
}

TEST(PipelineManifestRequirements, RejectsIncompleteOrTamperedAutodiffProfiles) {
    using vernon::runtime::AutodiffManifest;
    using vernon::runtime::parseAutodiffManifest;
    std::string error;
    AutodiffManifest manifest;
    nlohmann::json root = validAutodiffManifest();
    root.erase("autodiff_profiles");
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));

    root = validAutodiffManifest();
    root["autodiff_profiles"]["variants"][0]["plan"]["tape_bytes"] = 32;
    error.clear();
    manifest = {};
    EXPECT_FALSE(parseAutodiffManifest(root, manifest, error));
    EXPECT_NE(error.find("identity"), std::string::npos);
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
    EXPECT_TRUE(validateCpuRuntimeRequirements(triple, format, error)) << error;
    EXPECT_FALSE(validateCpuRuntimeRequirements(triple, "wasm", error));
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
    const auto texture = [](uint32_t slot, const char *name, uint32_t binding) {
        return nlohmann::json{{"slot", slot},
                              {"name", name},
                              {"kind", "texture"},
                              {"type", "!vernon.texture<\"2d\", f32>"},
                              {"access", "read"},
                              {"shape", nlohmann::json::array()},
                              {"dimension", "2d"},
                              {"uses", nlohmann::json::array({{{"stage", "fragment"},
                                                               {"interface", "resource"},
                                                               {"vernon.set", 0},
                                                               {"vernon.binding", binding}}})}};
    };
    nlohmann::json manifest = {{"key", nlohmann::json::array()},
                               {"program", {{"vertex", "vertex"}, {"fragment", "fragment"}}},
                               {"parameters", nlohmann::json::array({texture(0, "left", 3), texture(1, "right", 3)})},
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
                         {"sampled_texture_bindings", nlohmann::json::array({{{"set", 0}, {"binding", 3}}})}}})}}});
    variant = {};
    error.clear();
    EXPECT_FALSE(vernon::runtime::parseVariant(manifest, variant, error));
    EXPECT_EQ(error, "sampler references an unknown sampled texture binding");
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
}

} // namespace
