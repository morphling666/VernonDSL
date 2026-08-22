#include "runtime/content_hash.h"
#include "runtime/pipeline_manifest.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/program_manifest.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

nlohmann::json validExecutableProgram() {
    return {
        {"values", nlohmann::json::array({{{"id", 0},
                                           {"name", "input"},
                                           {"type", "tensor<4xf32>"},
                                           {"dtype", "f32"},
                                           {"shape", nlohmann::json::array({4})},
                                           {"external", true},
                                           {"output", false}},
                                          {{"id", 1},
                                           {"name", "result"},
                                           {"type", "tensor<4xf32>"},
                                           {"dtype", "f32"},
                                           {"shape", nlohmann::json::array({4})},
                                           {"external", false},
                                           {"output", true}}})},
        {"graphs", nlohmann::json::array(
                       {{{"name", "forward"},
                         {"direction", "forward"},
                         {"arguments", nlohmann::json::array({0})},
                         {"results", nlohmann::json::array({1})},
                         {"nodes", nlohmann::json::array(
                                       {{{"id", 0},
                                         {"name", "square"},
                                         {"kind", "compute"},
                                         {"stage", "Module.square"},
                                         {"operands", nlohmann::json::array({0})},
                                         {"results", nlohmann::json::array({1})},
                                         {"dependencies", nlohmann::json::array()},
                                         {"bindings", nlohmann::json::array({{{"parameter", "input"}, {"value", 0}},
                                                                             {{"parameter", "output"}, {"value", 1}}})},
                                         {"resources", nlohmann::json::array({{{"value", 0}, {"access", "read"}},
                                                                              {{"value", 1}, {"access", "write"}}})},
                                         {"grid", nlohmann::json::array({4, 1, 1})}}})}}})}};
}

TEST(ProgramManifest, ParsesAndValidatesExecutableGraph) {
    vernon::runtime::ExecutableProgram program;
    std::string error;
    nlohmann::json manifest = validExecutableProgram();
    ASSERT_TRUE(vernon::runtime::parseExecutableProgram(manifest, program, error)) << error;
    ASSERT_TRUE(program.validate({{"Module.square", "compute.spv"}}, error)) << error;
    ASSERT_EQ(program.graphs.size(), 1u);
    ASSERT_EQ(program.graphs[0].nodes.size(), 1u);

    manifest["graphs"][0]["nodes"][0]["dependencies"] = nlohmann::json::array({1});
    program = {};
    ASSERT_TRUE(vernon::runtime::parseExecutableProgram(manifest, program, error)) << error;
    EXPECT_FALSE(program.validate({{"Module.square", "compute.spv"}}, error));
    EXPECT_NE(error.find("dependencies"), std::string::npos);
}

TEST(ProgramManifest, DerivesBackwardCapturesFromSharedForwardValues) {
    nlohmann::json manifest = validExecutableProgram();
    manifest["values"].push_back({{"id", 2},
                                  {"name", "output_cotangent"},
                                  {"type", "tensor<4xf32>"},
                                  {"dtype", "f32"},
                                  {"shape", nlohmann::json::array({4})},
                                  {"external", true},
                                  {"output", false}});
    manifest["values"].push_back({{"id", 3},
                                  {"name", "input_gradient"},
                                  {"type", "tensor<4xf32>"},
                                  {"dtype", "f32"},
                                  {"shape", nlohmann::json::array({4})},
                                  {"external", false},
                                  {"output", false}});
    manifest["graphs"].push_back(
        {{"name", "backward"},
         {"direction", "backward"},
         {"arguments", nlohmann::json::array({2})},
         {"results", nlohmann::json::array({3})},
         {"nodes", nlohmann::json::array(
                       {{{"id", 0},
                         {"name", "square_backward"},
                         {"kind", "compute"},
                         {"stage", "Module.square_backward"},
                         {"operands", nlohmann::json::array({0, 1, 2})},
                         {"results", nlohmann::json::array({3})},
                         {"dependencies", nlohmann::json::array()},
                         {"bindings", nlohmann::json::array({{{"parameter", "input"}, {"value", 0}},
                                                             {{"parameter", "output"}, {"value", 1}},
                                                             {{"parameter", "output_cotangent"}, {"value", 2}},
                                                             {{"parameter", "input_gradient"}, {"value", 3}}})},
                         {"resources", nlohmann::json::array({{{"value", 0}, {"access", "read"}},
                                                              {{"value", 1}, {"access", "read"}},
                                                              {{"value", 2}, {"access", "read"}},
                                                              {{"value", 3}, {"access", "write"}}})},
                         {"grid", nlohmann::json::array({4, 1, 1})}}})}});

    vernon::runtime::ExecutableProgram program;
    std::string error;
    ASSERT_TRUE(vernon::runtime::parseExecutableProgram(manifest, program, error)) << error;
    ASSERT_TRUE(
        program.validate({{"Module.square", "compute.spv"}, {"Module.square_backward", "compute_backward.spv"}}, error))
        << error;
    EXPECT_EQ(program.backwardCaptures(), std::vector<uint32_t>({0, 1}));

    program.graphs[1].nodes[0].operands[0] = 3;
    EXPECT_FALSE(program.validate(
        {{"Module.square", "compute.spv"}, {"Module.square_backward", "compute_backward.spv"}}, error));
    EXPECT_NE(error.find("unavailable"), std::string::npos);
}

TEST(ProgramManifest, NormalizesLegacySingleComputeAsOneNodePipelineTopology) {
    vernon::runtime::Variant variant;
    variant.compute = "square";
    variant.program.emplace("compute", "square");
    vernon::runtime::Parameter parameter;
    parameter.name = "values";
    parameter.kind = "tensor";
    parameter.access = "read_write";
    parameter.shape = {4};
    parameter.elementLayout.logicalType = "f32";
    parameter.elementLayout.leaves.emplace_back("f32", 1, 0);
    variant.parameters.push_back(std::move(parameter));

    vernon::runtime::ExecutableProgram execution;
    std::string error;
    ASSERT_TRUE(vernon::runtime::normalizeLegacySingleComputeExecution(variant, execution, error)) << error;
    ASSERT_EQ(execution.graphs.size(), 1u);
    const vernon::runtime::ProgramGraph &forward = execution.graphs.front();
    EXPECT_EQ(forward.direction, "forward");
    ASSERT_EQ(forward.nodes.size(), 1u);
    EXPECT_EQ(forward.nodes.front().stage, "compute");
    EXPECT_EQ(forward.nodes.front().resources.front().access, "read_write");
    EXPECT_EQ(execution.values.front().name, "values");
    EXPECT_TRUE(execution.values.front().external);
}

TEST(ProgramManifest, AllowsSideEffectingZeroValuePipelineTopology) {
    vernon::runtime::Variant variant;
    variant.compute = "tick";
    variant.program.emplace("compute", "tick");

    vernon::runtime::ExecutableProgram execution;
    std::string error;
    ASSERT_TRUE(vernon::runtime::normalizeLegacySingleComputeExecution(variant, execution, error)) << error;
    EXPECT_TRUE(execution.values.empty());
    ASSERT_EQ(execution.graphs.size(), 1u);
    ASSERT_EQ(execution.graphs.front().nodes.size(), 1u);
}

TEST(ProgramExecutionManifest, ResolvesAndExecutesCanonicalComputePrograms) {
    const std::string f32LayoutHash = "95f54cae607cf7751c0ec4327f86b0982056823c49534e9d8dddd66fc98c5f07";
    const std::string u32LayoutHash = "280f13e115d9ccfbe2a5a33aaafefb004f2ad59b8312a2807f2dfaa7b0966bc6";
    const auto layout = [](const std::string &scope, const std::string &hash, const std::string &dtype) {
        return nlohmann::json{{"scope", scope},
                              {"layout_hash", hash},
                              {"byte_size", 4},
                              {"alignment", 4},
                              {"leaves", nlohmann::json::array({{{"path", nlohmann::json::array()},
                                                                 {"dtype", dtype},
                                                                 {"byte_offset", 0},
                                                                 {"scalar_count", 1},
                                                                 {"shape", nlohmann::json::array()}}})}};
    };
    nlohmann::json manifest{
        {"stages",
         {{"scale",
           {{"operation", "compute"},
            {"contract_hash", "0000000000000000000000000000000000000000000000000000000000000000"}}}}},
        {"parameters", nlohmann::json::array({{{"id", 0}, {"path", "groups_x"}, {"value", 2}}})},
        {"storages", nlohmann::json::array({{{"id", 0},
                                             {"name", "x"},
                                             {"initial_value", 0},
                                             {"ownership", "borrowed"},
                                             {"lifetime", "invocation"},
                                             {"mutability", "read_only"},
                                             {"descriptor",
                                              {{"tag", "buffer"},
                                               {"byte_length", 1024},
                                               {"alignment", 16},
                                               {"memory", "device"},
                                               {"usage", nlohmann::json::array({"storage"})}}}},
                                            {{"id", 1},
                                             {"name", "y"},
                                             {"initial_value", 1},
                                             {"ownership", "owned"},
                                             {"lifetime", "invocation"},
                                             {"mutability", "mutable"},
                                             {"descriptor",
                                              {{"tag", "buffer"},
                                               {"byte_length", 1024},
                                               {"alignment", 16},
                                               {"memory", "device"},
                                               {"usage", nlohmann::json::array({"storage"})}}}}})},
        {"values", nlohmann::json::array({{{"id", 0},
                                           {"name", "x"},
                                           {"type", "tensor<256xf32>"},
                                           {"shape", nlohmann::json::array({256})},
                                           {"origin", {{"tag", "argument"}, {"graph", "forward"}, {"slot", 0}}},
                                           {"storage", 0},
                                           {"value_layout", layout("element", f32LayoutHash, "f32")}},
                                          {{"id", 1},
                                           {"name", "y"},
                                           {"type", "tensor<256xf32>"},
                                           {"shape", nlohmann::json::array({256})},
                                           {"origin", {{"tag", "node_result"}, {"graph", "forward"}, {"node", 0}}},
                                           {"storage", 1},
                                           {"value_layout", layout("element", f32LayoutHash, "f32")}},
                                          {{"id", 2},
                                           {"name", "groups_x"},
                                           {"type", "u32"},
                                           {"shape", nlohmann::json::array()},
                                           {"origin", {{"tag", "parameter"}, {"parameter", 0}}},
                                           {"value_layout", layout("value", u32LayoutHash, "u32")}}})},
        {"shape_symbols", nlohmann::json::array()},
        {"shape_constraints", nlohmann::json::array()},
        {"alias_preconditions", nlohmann::json::array()},
        {"graphs",
         nlohmann::json::array(
             {{{"name", "forward"},
               {"direction", "forward"},
               {"inputs", nlohmann::json::array({{{"tag", "user_input"}, {"value", 0}, {"slot", 0}},
                                                 {{"tag", "parameter"}, {"value", 2}, {"parameter", 0}}})},
               {"captures", nlohmann::json::array()},
               {"outputs",
                nlohmann::json::array({{{"tag", "user_output"}, {"value", 1}, {"disposition", "transfer"}}})},
               {"nodes",
                nlohmann::json::array(
                    {{{"id", 0},
                      {"stage", "scale"},
                      {"operands", nlohmann::json::array({0, 2})},
                      {"results", nlohmann::json::array({1})},
                      {"bindings", nlohmann::json::array({{{"module", "compute"},
                                                           {"interface", "argument"},
                                                           {"index", 0},
                                                           {"tag", "resource"},
                                                           {"access", 0}},
                                                          {{"module", "compute"},
                                                           {"interface", "argument"},
                                                           {"index", 1},
                                                           {"tag", "resource"},
                                                           {"access", 1}}})},
                      {"accesses", nlohmann::json::array({{{"tag", "read"}, {"storage", 0}, {"value", 0}},
                                                          {{"tag", "initialize"}, {"storage", 1}, {"after", 1}}})},
                      {"operation",
                       {{"tag", "compute"},
                        {"workgroups", nlohmann::json::array({{{"control", {{"parameter", 0}}}}, 1, 1})}}}}})}}})},
        {"signature",
         {{"inputs", nlohmann::json::array({{{"path", "x"}, {"value", 0}}})},
          {"outputs", nlohmann::json::array({{{"path", "y"}, {"value", 1}, {"disposition", "transfer"}}})},
          {"cotangents", nlohmann::json::array()},
          {"gradients", nlohmann::json::array()}}}};

    nlohmann::json reflection{
        {"required_features", nlohmann::json::array()},
        {"endpoints",
         nlohmann::json::array(
             {{{"tag", "resource"},
               {"module", "compute"},
               {"interface", "argument"},
               {"index", 0},
               {"role", "storage"},
               {"type", "tensor<256xf32>"},
               {"layout",
                {{"tag", "buffer"},
                 {"view_rank", 1},
                 {"element_layout_hash", f32LayoutHash},
                 {"minimum_alignment", 4}}},
               {"address_space", "device"},
               {"transport", "resource_handle"},
               {"access", "read"},
               {"abi",
                {{"bindings",
                  nlohmann::json::array(
                      {{{"semantic", "resource"}, {"carrier", {{"tag", "resource_slot"}, {"slot", 0}}}},
                       {{"semantic", "byte_offset"},
                        {"carrier",
                         {{"tag", "value_slot"}, {"slot", 1}, {"byte_offset", 0}, {"byte_size", 8}, {"alignment", 8}}}},
                       {{"semantic", {{"extent", 0}}},
                        {"carrier",
                         {{"tag", "value_slot"}, {"slot", 3}, {"byte_offset", 0}, {"byte_size", 8}, {"alignment", 8}}}},
                       {{"semantic", {{"byte_stride", 0}}},
                        {"carrier",
                         {{"tag", "value_slot"},
                          {"slot", 4},
                          {"byte_offset", 0},
                          {"byte_size", 8},
                          {"alignment", 8}}}}})}}}},
              {{"tag", "resource"},
               {"module", "compute"},
               {"interface", "argument"},
               {"index", 1},
               {"role", "storage"},
               {"type", "tensor<256xf32>"},
               {"layout",
                {{"tag", "buffer"},
                 {"view_rank", 1},
                 {"element_layout_hash", f32LayoutHash},
                 {"minimum_alignment", 4}}},
               {"address_space", "device"},
               {"transport", "resource_handle"},
               {"access", "write"},
               {"abi",
                {{"bindings",
                  nlohmann::json::array(
                      {{{"semantic", "resource"}, {"carrier", {{"tag", "resource_slot"}, {"slot", 2}}}},
                       {{"semantic", "byte_offset"},
                        {"carrier",
                         {{"tag", "value_slot"}, {"slot", 5}, {"byte_offset", 0}, {"byte_size", 8}, {"alignment", 8}}}},
                       {{"semantic", {{"extent", 0}}},
                        {"carrier",
                         {{"tag", "value_slot"}, {"slot", 6}, {"byte_offset", 0}, {"byte_size", 8}, {"alignment", 8}}}},
                       {{"semantic", {{"byte_stride", 0}}},
                        {"carrier",
                         {{"tag", "value_slot"},
                          {"slot", 7},
                          {"byte_offset", 0},
                          {"byte_size", 8},
                          {"alignment", 8}}}}})}}}}})},
        {"compute",
         {{"workgroup_size", nlohmann::json::array({64, 1, 1})},
          {"subgroup", nullptr},
          {"capabilities", nlohmann::json::array({"direct_dispatch"})}}}};
    const nlohmann::json contract{{"operation", "compute"}, {"reflection", reflection}};
    const std::string contractBytes = contract.dump(-1, ' ', false);
    const std::string contractHash = vernon::runtime::sha256Hex(contractBytes.data(), contractBytes.size());
    manifest["stages"]["scale"]["contract_hash"] = contractHash;
    const std::string codeBytes = "test";
    const std::string codeHash = vernon::runtime::sha256Hex(codeBytes.data(), codeBytes.size());
    nlohmann::json artifactSystem{{"target", {{"kind", "cpu"}, {"options", {{"triple", "aarch64-apple-darwin"}}}}},
                                  {"blobs",
                                   {{"code",
                                     {{"byte_length", 4},
                                      {"sha256", codeHash},
                                      {"location", {{"tag", "external"}, {"uri", "artifacts/scale.o"}}}}}}},
                                  {"artifacts",
                                   {{"scale-artifact",
                                     {{"tag", "stage"},
                                      {"operation", "compute"},
                                      {"contract_hash", contractHash},
                                      {"runtime_requirements",
                                       {{"backend", "cpu"},
                                        {"features", nlohmann::json::array()},
                                        {"target_triple", "aarch64-apple-darwin"},
                                        {"object_format", "macho"}}},
                                      {"modules", nlohmann::json::array({{{"role", "compute"},
                                                                          {"format", "relocatable_object"},
                                                                          {"entry_point", "scale"},
                                                                          {"blob", "code"},
                                                                          {"offset", 0},
                                                                          {"byte_length", 4},
                                                                          {"sha256", codeHash}}})},
                                      {"reflection", reflection}}}}}};

    vernon::runtime::program::Program program;
    vernon::runtime::program::ArtifactSystem artifacts;
    vernon::runtime::program::Diagnostic diagnostic;
    ASSERT_TRUE(vernon::runtime::program::parse(manifest, program, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(vernon::runtime::program::parseArtifactSystem(artifactSystem, artifacts, diagnostic))
        << diagnostic.message;
    vernon::runtime::program::ResolvedProgram resolved;
    ASSERT_TRUE(vernon::runtime::program::resolve(std::move(program), artifacts, {{"scale", "scale-artifact"}},
                                                  resolved, diagnostic))
        << diagnostic.message;
    ASSERT_EQ(resolved.stages.size(), 1u);
    EXPECT_EQ(resolved.stages.at("scale").stage.workgroupSize[0], 64u);
    ASSERT_EQ(resolved.graphs.size(), 1u);
    ASSERT_EQ(resolved.graphs[0].predecessors.size(), 1u);
    EXPECT_TRUE(resolved.graphs[0].predecessors[0].empty());

    std::vector<float> input(256);
    for (size_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<float>(index);
    vernon::runtime::program::Invocation invocation;
    invocation.arguments.push_back({input.data(), input.size() * sizeof(float)});
    invocation.parameters.push_back(4);
    vernon::runtime::program::ExecutionResult execution;
    const vernon::runtime::program::StageExecutor executor = [](const vernon::runtime::program::StageInvocation &stage,
                                                                vernon::runtime::program::Diagnostic &error) {
        if (!stage.stage || stage.resources.size() != 2 || !stage.resources[0].data || !stage.resources[1].data ||
            stage.resources[0].byteLength != stage.resources[1].byteLength) {
            error = {"PROGRAM_BINDING_MISMATCH", "execute", "", "invalid test stage resources"};
            return false;
        }
        const auto *source = static_cast<const float *>(stage.resources[0].data);
        auto *destination = static_cast<float *>(stage.resources[1].data);
        const size_t count = stage.resources[0].byteLength / sizeof(float);
        const bool bias = stage.stage->stage.modules.front().entryPoint == "bias";
        for (size_t index = 0; index < count; ++index)
            destination[index] = bias ? source[index] + 1.0f : source[index] * 2.0f;
        return true;
    };
    ASSERT_TRUE(vernon::runtime::program::execute(resolved, invocation, executor, execution, diagnostic))
        << diagnostic.message;
    ASSERT_EQ(execution.outputs.size(), 1u);
    ASSERT_EQ(execution.outputs[0].size(), 1024u);
    const auto *output = reinterpret_cast<const float *>(execution.outputs[0].data());
    EXPECT_EQ(output[0], 0.0f);
    EXPECT_EQ(output[17], 34.0f);
    EXPECT_EQ(output[255], 510.0f);

    nlohmann::json multiManifest = manifest;
    multiManifest["stages"]["bias"] = {{"operation", "compute"}, {"contract_hash", contractHash}};
    multiManifest["storages"].push_back({{"id", 2},
                                         {"name", "z"},
                                         {"initial_value", 3},
                                         {"ownership", "owned"},
                                         {"lifetime", "invocation"},
                                         {"mutability", "mutable"},
                                         {"descriptor",
                                          {{"tag", "buffer"},
                                           {"byte_length", 1024},
                                           {"alignment", 16},
                                           {"memory", "device"},
                                           {"usage", nlohmann::json::array({"storage"})}}}});
    multiManifest["values"].push_back({{"id", 3},
                                       {"name", "z"},
                                       {"type", "tensor<256xf32>"},
                                       {"shape", nlohmann::json::array({256})},
                                       {"origin", {{"tag", "node_result"}, {"graph", "forward"}, {"node", 1}}},
                                       {"storage", 2},
                                       {"value_layout", layout("element", f32LayoutHash, "f32")}});
    multiManifest["graphs"][0]["nodes"].push_back(
        {{"id", 1},
         {"stage", "bias"},
         {"operands", nlohmann::json::array({1, 2})},
         {"results", nlohmann::json::array({3})},
         {"bindings",
          nlohmann::json::array(
              {{{"module", "compute"}, {"interface", "argument"}, {"index", 0}, {"tag", "resource"}, {"access", 0}},
               {{"module", "compute"}, {"interface", "argument"}, {"index", 1}, {"tag", "resource"}, {"access", 1}}})},
         {"accesses", nlohmann::json::array({{{"tag", "read"}, {"storage", 1}, {"value", 1}},
                                             {{"tag", "initialize"}, {"storage", 2}, {"after", 3}}})},
         {"operation",
          {{"tag", "compute"}, {"workgroups", nlohmann::json::array({{{"control", {{"parameter", 0}}}}, 1, 1})}}}});
    multiManifest["graphs"][0]["outputs"][0]["value"] = 3;
    multiManifest["signature"]["outputs"][0]["value"] = 3;

    nlohmann::json multiArtifactSystem = artifactSystem;
    multiArtifactSystem["artifacts"]["bias-artifact"] = multiArtifactSystem["artifacts"]["scale-artifact"];
    multiArtifactSystem["artifacts"]["bias-artifact"]["modules"][0]["entry_point"] = "bias";
    vernon::runtime::program::Program multiProgram;
    vernon::runtime::program::ArtifactSystem multiArtifacts;
    ASSERT_TRUE(vernon::runtime::program::parse(multiManifest, multiProgram, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(vernon::runtime::program::parseArtifactSystem(multiArtifactSystem, multiArtifacts, diagnostic))
        << diagnostic.message;
    vernon::runtime::program::ResolvedProgram multiResolved;
    ASSERT_TRUE(vernon::runtime::program::resolve(std::move(multiProgram), multiArtifacts,
                                                  {{"bias", "bias-artifact"}, {"scale", "scale-artifact"}},
                                                  multiResolved, diagnostic))
        << diagnostic.message;
    ASSERT_EQ(multiResolved.graphs[0].predecessors.size(), 2u);
    EXPECT_EQ(multiResolved.graphs[0].predecessors[1], std::vector<uint32_t>({0}));
    execution = {};
    ASSERT_TRUE(vernon::runtime::program::execute(multiResolved, invocation, executor, execution, diagnostic))
        << diagnostic.message;
    ASSERT_EQ(execution.outputs.size(), 1u);
    output = reinterpret_cast<const float *>(execution.outputs[0].data());
    EXPECT_EQ(output[0], 1.0f);
    EXPECT_EQ(output[17], 35.0f);
    EXPECT_EQ(output[255], 511.0f);

    manifest["graphs"][0]["nodes"][0]["dependencies"] = nlohmann::json::array();
    EXPECT_FALSE(vernon::runtime::program::parse(manifest, program, diagnostic));
    EXPECT_EQ(diagnostic.code, "PROGRAM_UNKNOWN_FIELD");
    EXPECT_EQ(diagnostic.path, "/graphs/0/nodes/0/dependencies");
}

TEST(ProgramExecutionManifest, ResolvesCanonicalGraphicsAttachment) {
    nlohmann::json reflection{
        {"required_features", nlohmann::json::array()},
        {"endpoints", nlohmann::json::array()},
        {"graphics",
         {{"topology", "triangle_list"},
          {"vertex_inputs", nlohmann::json::array()},
          {"fragment_outputs", nlohmann::json::array({{{"location", 0}, {"type", "tensor<4xf32>"}}})},
          {"linkage", {{"vertex_outputs", nlohmann::json::array()}, {"fragment_inputs", nlohmann::json::array()}}},
          {"attachment_constraints", nlohmann::json::array()},
          {"index_formats", nlohmann::json::array({"u16", "u32"})},
          {"capabilities", nlohmann::json::array({"direct_draw"})}}}};
    const nlohmann::json contract{{"operation", "graphics"}, {"reflection", reflection}};
    const std::string contractBytes = contract.dump(-1, ' ', false);
    const std::string contractHash = vernon::runtime::sha256Hex(contractBytes.data(), contractBytes.size());
    nlohmann::json manifest{
        {"stages", {{"draw", {{"operation", "graphics"}, {"contract_hash", contractHash}}}}},
        {"parameters", nlohmann::json::array()},
        {"storages", nlohmann::json::array({{{"id", 0},
                                             {"initial_value", 0},
                                             {"ownership", "borrowed"},
                                             {"lifetime", "invocation"},
                                             {"mutability", "mutable"},
                                             {"descriptor",
                                              {{"tag", "image"},
                                               {"dimension", "2d"},
                                               {"extent", nlohmann::json::array({32, 32, 1})},
                                               {"format", "rgba8_unorm"},
                                               {"sample_count", 1},
                                               {"mip_levels", 1},
                                               {"array_layers", 1},
                                               {"aspects", nlohmann::json::array({"color"})},
                                               {"usage", nlohmann::json::array({"color_attachment"})}}}}})},
        {"values", nlohmann::json::array({{{"id", 0},
                                           {"type", "image<rgba8_unorm>"},
                                           {"origin", {{"tag", "argument"}, {"graph", "forward"}, {"slot", 0}}},
                                           {"storage", 0}},
                                          {{"id", 1},
                                           {"type", "image<rgba8_unorm>"},
                                           {"origin", {{"tag", "node_result"}, {"graph", "forward"}, {"node", 0}}},
                                           {"storage", 0}}})},
        {"shape_symbols", nlohmann::json::array()},
        {"shape_constraints", nlohmann::json::array()},
        {"alias_preconditions", nlohmann::json::array()},
        {"graphs",
         nlohmann::json::array(
             {{{"name", "forward"},
               {"direction", "forward"},
               {"inputs", nlohmann::json::array({{{"tag", "user_input"}, {"value", 0}, {"slot", 0}}})},
               {"captures", nlohmann::json::array()},
               {"outputs",
                nlohmann::json::array({{{"tag", "user_output"}, {"value", 1}, {"disposition", "transfer"}}})},
               {"nodes",
                nlohmann::json::array(
                    {{{"id", 0},
                      {"stage", "draw"},
                      {"operands", nlohmann::json::array({0})},
                      {"results", nlohmann::json::array({1})},
                      {"bindings", nlohmann::json::array()},
                      {"accesses",
                       nlohmann::json::array({{{"tag", "attachment"}, {"storage", 0}, {"before", 0}, {"after", 1}}})},
                      {"operation",
                       {{"tag", "graphics"},
                        {"attachments",
                         {{"colors",
                           nlohmann::json::array(
                               {{{"location", 0}, {"access", 0}, {"load", {{"tag", "discard"}}}, {"store", "store"}}})},
                          {"depth_stencil", nullptr},
                          {"render_area", {{"x", 0}, {"y", 0}, {"width", 32}, {"height", 32}}},
                          {"layer_count", 1}}},
                        {"state", nlohmann::json::object()},
                        {"draw", {{"tag", "direct"}, {"vertex_count", 3}, {"instance_count", 1}}}}}}})}}})},
        {"signature",
         {{"inputs", nlohmann::json::array({{{"path", "target"}, {"value", 0}}})},
          {"outputs", nlohmann::json::array({{{"path", "target"}, {"value", 1}, {"disposition", "transfer"}}})},
          {"cotangents", nlohmann::json::array()},
          {"gradients", nlohmann::json::array()}}}};
    const std::string codeBytes = "test";
    const std::string codeHash = vernon::runtime::sha256Hex(codeBytes.data(), codeBytes.size());
    nlohmann::json artifactSystem{{"target", {{"kind", "metal"}, {"options", {{"platform", "macos"}}}}},
                                  {"blobs",
                                   {{"vertex",
                                     {{"byte_length", 4},
                                      {"sha256", codeHash},
                                      {"location", {{"tag", "external"}, {"uri", "artifacts/vertex.metal"}}}}}}},
                                  {"artifacts",
                                   {{"draw-artifact",
                                     {{"tag", "stage"},
                                      {"operation", "graphics"},
                                      {"contract_hash", contractHash},
                                      {"runtime_requirements",
                                       {{"backend", "metal"},
                                        {"features", nlohmann::json::array()},
                                        {"apple_platform", "macos"},
                                        {"msl_version", nlohmann::json::array({2, 4})},
                                        {"minimum_os_version", nlohmann::json::array({11, 0})}}},
                                      {"modules", nlohmann::json::array({{{"role", "vertex"},
                                                                          {"format", "msl"},
                                                                          {"entry_point", "vertex_main"},
                                                                          {"blob", "vertex"},
                                                                          {"offset", 0},
                                                                          {"byte_length", 4},
                                                                          {"sha256", codeHash}}})},
                                      {"reflection", reflection}}}}}};

    vernon::runtime::program::Program program;
    vernon::runtime::program::ArtifactSystem artifacts;
    vernon::runtime::program::Diagnostic diagnostic;
    ASSERT_TRUE(vernon::runtime::program::parse(manifest, program, diagnostic)) << diagnostic.message;
    ASSERT_TRUE(vernon::runtime::program::parseArtifactSystem(artifactSystem, artifacts, diagnostic))
        << diagnostic.message;
    vernon::runtime::program::ResolvedProgram resolved;
    ASSERT_TRUE(vernon::runtime::program::resolve(std::move(program), artifacts, {{"draw", "draw-artifact"}}, resolved,
                                                  diagnostic))
        << diagnostic.message;
    EXPECT_EQ(resolved.program.graphs.front().nodes.front().operation, "graphics");
}

} // namespace
