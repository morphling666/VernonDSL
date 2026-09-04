#include "runtime/autodiff/program_publication.h"
#include "runtime/content_hash.h"
#include "runtime/program_execution_manifest.h"
#include "runtime/runtime_pipeline_backend.h"
#include "runtime/target_binding_plan.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

VernonPipelineArgument hostTensor(void *data, size_t size) {
    VernonPipelineArgument argument{};
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = data;
    argument.tensor.byte_size = size;
    argument.tensor.element_layout.byte_size = size;
    return argument;
}

struct BoundarySpec {
    const char *path;
    uint32_t value;
    const char *role;
    const char *direction;
};

void setProgramAbi(nlohmann::json &manifest, std::initializer_list<BoundarySpec> boundaries) {
    nlohmann::json slots = nlohmann::json::array();
    for (const BoundarySpec &boundary : boundaries) {
        const uint32_t valueId = boundary.value;
        const auto &value = manifest["values"][valueId];
        nlohmann::json slot{{"id", slots.size()},
                            {"path", boundary.path},
                            {"value", valueId},
                            {"role", boundary.role},
                            {"direction", boundary.direction},
                            {"logical_type", value["type"]},
                            {"alias_owner", "value:" + std::to_string(valueId)},
                            {"outer_shape", value.value("shape", nlohmann::json::array())}};
        if (std::string_view(boundary.direction) == "output")
            slot["publication"] = "commit_after_success";
        if (value.contains("value_layout"))
            slot["value_layout"] = value["value_layout"];
        if (value.contains("storage")) {
            const uint32_t storageId = value["storage"].get<uint32_t>();
            const auto &storage = manifest["storages"][storageId];
            const bool input = std::string_view(boundary.direction) == "input";
            const std::string tag = storage["descriptor"]["tag"];
            slot["category"] = tag == "buffer" ? "storage_view" : tag == "image" ? "texture" : "sampler";
            slot["storage_id"] = storageId;
            slot["alias_owner"] = "storage:" + std::to_string(storageId);
            slot["storage_descriptor"] = storage["descriptor"];
            slot["access"] = input && storage["mutability"] == "mutable" ? "read_write" : input ? "read" : "write";
        } else {
            slot["category"] = "value";
            slot["access"] = std::string(boundary.direction) == "input" ? "read" : "write";
        }
        slots.push_back(std::move(slot));
    }
    manifest["abi"] = {{"boundary_slots", std::move(slots)},
                       {"derivative_projections", nlohmann::json::array()},
                       {"tape_plans", nlohmann::json::array()}};
}

TEST(ProgramPublication, ValidatesEveryTargetBeforeCommit) {
    float firstSource = 3.0f;
    float secondSource = 4.0f;
    float firstDestination = -1.0f;
    float secondDestination = -2.0f;
    std::vector<vernon::runtime::ad::ProgramHostValue> storage(2);
    storage[0].argument = hostTensor(&firstSource, sizeof(firstSource));
    storage[1].argument = hostTensor(&secondSource, sizeof(secondSource));

    std::vector<vernon::runtime::program::PublicationTarget> targets{
        {0,
         0,
         vernon::runtime::program::BoundaryRole::Output,
         {vernon::runtime::program::ProgramOwnerKind::Storage, 0}},
        {1,
         1,
         vernon::runtime::program::BoundaryRole::Output,
         {vernon::runtime::program::ProgramOwnerKind::Storage, 1}},
    };
    std::vector<vernon::runtime::ad::PendingProgramPublication> publications{
        {&targets[0], hostTensor(&firstDestination, sizeof(firstDestination)), std::nullopt},
        {&targets[1], hostTensor(&secondDestination, sizeof(secondDestination) * 2), std::nullopt},
    };
    std::string error;
    EXPECT_FALSE(vernon::runtime::ad::commitProgramPublications(storage, publications, error));
    EXPECT_EQ(firstDestination, -1.0f);
    EXPECT_EQ(secondDestination, -2.0f);

    publications[1].destination = hostTensor(&secondDestination, sizeof(secondDestination));
    ASSERT_TRUE(vernon::runtime::ad::commitProgramPublications(storage, publications, error)) << error;
    EXPECT_EQ(firstDestination, firstSource);
    EXPECT_EQ(secondDestination, secondSource);
}

TEST(ProgramPublication, ProgramOwnerIdentityControlsBindingReuse) {
    float first = 1.0f;
    float second = 2.0f;
    const vernon::runtime::program::ProgramOwnerId sharedOwner{vernon::runtime::program::ProgramOwnerKind::Storage, 7};
    vernon::runtime::ad::ProgramOwnerBindings bindings;
    std::string error;
    ASSERT_TRUE(bindings.bind(sharedOwner, hostTensor(&first, sizeof(first)), error));
    EXPECT_FALSE(bindings.bind(sharedOwner, hostTensor(&second, sizeof(second)), error));
    EXPECT_TRUE(bindings.bind({vernon::runtime::program::ProgramOwnerKind::Storage, 8},
                              hostTensor(&second, sizeof(second)), error));
}

TEST(ProgramTargetBinding, PreservesCanonicalNumericShapeAcrossComputeAndGraphics) {
    using namespace vernon::runtime;
    using namespace vernon::runtime::program;

    ValueLeaf matrixLeaf{"f32", 6, 0};
    matrixLeaf.shape = {2, 3};
    vernon::runtime::ValueLayout matrixLayout{"tensor<2x3xf32>", "", "matrix-layout", 24, 4, {matrixLeaf}};
    TransportNode matrixCarrier;
    matrixCarrier.kind = TransportNodeKind::Array;
    matrixCarrier.size = 24;
    matrixCarrier.alignment = 4;
    matrixCarrier.shape = {3, 2};
    matrixCarrier.byteStrides = {8, 4};
    InterfacePlan nativeUniform;
    nativeUniform.kind = InterfacePlanKind::NativeUniform;
    nativeUniform.profile = "opengl_native_uniform";
    nativeUniform.canonicalLayoutHash = matrixLayout.layoutHash;
    nativeUniform.root = matrixCarrier;

    TargetBinding binding;
    binding.endpoint = {"vertex", "argument", 4, 4, "read"};
    binding.projection.value = 0;
    binding.source = SourceRepresentation::WholeValueBytes;
    binding.carrier = TargetCarrier::InlineValue;
    binding.name = "view_projection";
    binding.kind = "tensor";
    binding.access = "read";
    binding.valueType = CanonicalValueType{"f32", {2, 3}, true};
    binding.wholeValueLayout = matrixLayout;
    binding.elementLayout = matrixLayout;
    binding.transport = TargetPhysicalTransport{nativeUniform, {0, 4, UINT32_MAX}};
    binding.native = {0, 4, UINT32_MAX};

    TargetBindingPlan plan;
    plan.backend = VERNON_RUNTIME_OPENGL;
    plan.operation = "graphics";
    plan.bindings.push_back(std::move(binding));

    ExecutableBindingView variant;
    ReflectedEntry reflection;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildExecutableBindingView(plan, variant, reflection, diagnostic)) << diagnostic.message;
    ASSERT_EQ(variant.parameters.size(), 1u);
    EXPECT_TRUE(variant.parameters[0].shape.empty());
    ASSERT_EQ(variant.parameters[0].uses.size(), 1u);
    EXPECT_EQ(variant.parameters[0].uses[0].shape, std::vector<uint64_t>({2, 3}));

    plan.operation = "compute";
    ASSERT_TRUE(buildExecutableBindingView(plan, variant, reflection, diagnostic)) << diagnostic.message;
    ASSERT_EQ(variant.parameters.size(), 1u);
    EXPECT_TRUE(variant.parameters[0].shape.empty());
    ASSERT_EQ(variant.parameters[0].uses.size(), 1u);
    EXPECT_EQ(variant.parameters[0].uses[0].shape, std::vector<uint64_t>({2, 3}));
}

TEST(ProgramTargetBinding, ProjectsOnlyOpenGLNativeUniformMatrixMetadata) {
    using vernon::runtime::OpenGLNativeUniformShape;
    using vernon::runtime::resolveOpenGLNativeUniformShape;

    const auto expectShape = [&](std::string_view dtype, std::vector<uint64_t> shape, uint32_t scalars,
                                 uint32_t columns) {
        OpenGLNativeUniformShape result;
        ASSERT_TRUE(resolveOpenGLNativeUniformShape(dtype, shape, result));
        EXPECT_EQ(result.scalarCount, scalars);
        EXPECT_EQ(result.matrixColumns, columns);
    };
    expectShape("f32", {}, 1, 1);
    expectShape("f32", {2}, 2, 1);
    expectShape("i32", {3}, 3, 1);
    expectShape("u32", {4}, 4, 1);
    expectShape("f32", {2, 3}, 6, 3);
    expectShape("f32", {3, 2}, 6, 2);
    expectShape("f32", {4, 4}, 16, 4);

    OpenGLNativeUniformShape rejected;
    EXPECT_FALSE(resolveOpenGLNativeUniformShape("i32", {2, 2}, rejected));
    EXPECT_FALSE(resolveOpenGLNativeUniformShape("f32", {10, 10}, rejected));
}

TEST(ProgramTargetBinding, PreservesCompilerSelectedBufferCarrierForLargeMatrices) {
    using namespace vernon::runtime;
    using namespace vernon::runtime::program;

    ValueLeaf leaf{"f32", 100, 0};
    leaf.shape = {10, 10};
    vernon::runtime::ValueLayout layout{"tensor<10x10xf32>", "", "large-matrix-layout", 400, 4, {leaf}};
    TransportNode root;
    root.kind = TransportNodeKind::Array;
    root.size = 400;
    root.alignment = 16;
    root.shape = {10, 10};
    root.byteStrides = {40, 4};
    InterfacePlan buffer;
    buffer.kind = InterfacePlanKind::ByteTransport;
    buffer.profile = "std430_storage_buffer";
    buffer.canonicalLayoutHash = layout.layoutHash;
    buffer.root = root;

    TargetBinding binding;
    binding.endpoint = {"fragment", "argument", 0, 0, "read"};
    binding.projection.value = 0;
    binding.source = SourceRepresentation::WholeValueBytes;
    binding.carrier = TargetCarrier::StorageBuffer;
    binding.name = "large_matrix";
    binding.kind = "tensor";
    binding.access = "read";
    binding.valueType = CanonicalValueType{"f32", {10, 10}, true};
    binding.wholeValueLayout = layout;
    binding.elementLayout = layout;
    binding.transport = TargetPhysicalTransport{buffer, {0, 0, UINT32_MAX}};

    TargetBindingPlan plan;
    plan.backend = VERNON_RUNTIME_OPENGL;
    plan.operation = "graphics";
    plan.bindings.push_back(std::move(binding));
    ExecutableBindingView view;
    ReflectedEntry reflection;
    Diagnostic diagnostic;
    ASSERT_TRUE(buildExecutableBindingView(plan, view, reflection, diagnostic)) << diagnostic.message;
    ASSERT_EQ(view.parameters.size(), 1u);
    ASSERT_EQ(view.parameters[0].uses.size(), 1u);
    EXPECT_TRUE(view.parameters[0].shape.empty());
    EXPECT_EQ(view.parameters[0].uses[0].shape, std::vector<uint64_t>({10, 10}));
    EXPECT_EQ(view.parameters[0].uses[0].transport, "storage_buffer");
}

TEST(ProgramExecutionManifest, ResolvesCanonicalComputePrograms) {
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
                        {"workgroups", nlohmann::json::array({{{"control", {{"parameter", 0}}}}, 1, 1})}}}}})}}})}};
    setProgramAbi(manifest, {{"x", 0, "input", "input"}, {"y", 1, "output", "output"}});

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
    ASSERT_EQ(program.values.size(), 3u);
    EXPECT_TRUE(program.values[0].canonicalType.rankedValue);
    EXPECT_EQ(program.values[0].canonicalType.dtype, "f32");
    EXPECT_EQ(program.values[0].canonicalType.innerShape, std::vector<uint64_t>({256}));
    EXPECT_FALSE(program.values[2].canonicalType.rankedValue);
    EXPECT_EQ(program.values[2].canonicalType.dtype, "u32");
    ASSERT_EQ(program.abi.boundarySlots.size(), 2u);
    EXPECT_EQ(program.abi.boundarySlots[0].path, "x");
    EXPECT_EQ(program.abi.boundarySlots[0].category, vernon::runtime::program::BoundaryCategory::StorageView);
    ASSERT_TRUE(program.abi.boundarySlots[0].storage);
    EXPECT_EQ(program.abi.boundarySlots[0].storage->id, 0u);
    EXPECT_TRUE(program.abi.derivativeProjections.empty());
    ASSERT_EQ(program.abi.publication.targets.size(), 1u);
    EXPECT_EQ(program.abi.publication.targets[0].slot, program.abi.boundarySlots[1].id);
    EXPECT_EQ(program.abi.publication.targets[0].value, program.abi.boundarySlots[1].value);
    EXPECT_EQ(program.abi.publication.targets[0].aliasOwner.kind, vernon::runtime::program::ProgramOwnerKind::Storage);
    EXPECT_EQ(program.abi.publication.targets[0].aliasOwner.id, 1u);
    std::vector<vernon::runtime::program::BoundarySlot> unpublishedSlots = program.abi.boundarySlots;
    unpublishedSlots[1].publication = vernon::runtime::program::BoundaryPublication::None;
    EXPECT_TRUE(vernon::runtime::program::derivePublicationPlan(unpublishedSlots).targets.empty());
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
    setProgramAbi(multiManifest, {{"x", 0, "input", "input"}, {"y", 3, "output", "output"}});

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

    nlohmann::json missingAbi = manifest;
    missingAbi.erase("abi");
    EXPECT_FALSE(vernon::runtime::program::parse(missingAbi, program, diagnostic));
    EXPECT_EQ(diagnostic.path, "/abi");

    nlohmann::json legacySignature = manifest;
    legacySignature["signature"] = nlohmann::json::object();
    EXPECT_FALSE(vernon::runtime::program::parse(legacySignature, program, diagnostic));
    EXPECT_EQ(diagnostic.code, "PROGRAM_UNKNOWN_FIELD");
    EXPECT_EQ(diagnostic.path, "/signature");

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
                        {"pipeline_state",
                         {{"topology", "triangle_list"},
                          {"rasterization",
                           {{"cull_mode", "NONE"},
                            {"front_face", "COUNTER_CLOCKWISE"},
                            {"depth_clamp", false},
                            {"depth_bias_constant", 0.0},
                            {"depth_bias_slope", 0.0}}},
                          {"depth_stencil",
                           {{"depth_test", false},
                            {"depth_write", false},
                            {"depth_compare", "LESS"},
                            {"stencil_test", false},
                            {"front",
                             {{"stencil_fail", "KEEP"},
                              {"depth_fail", "KEEP"},
                              {"pass_operation", "KEEP"},
                              {"compare", "ALWAYS"}}},
                            {"back",
                             {{"stencil_fail", "KEEP"},
                              {"depth_fail", "KEEP"},
                              {"pass_operation", "KEEP"},
                              {"compare", "ALWAYS"}}},
                            {"stencil_read_mask", 255},
                            {"stencil_write_mask", 255}}},
                          {"color_blends", nlohmann::json::array({{0,
                                                                   {{"enabled", false},
                                                                    {"source_color", "ONE"},
                                                                    {"destination_color", "ZERO"},
                                                                    {"color_operation", "ADD"},
                                                                    {"source_alpha", "ONE"},
                                                                    {"destination_alpha", "ZERO"},
                                                                    {"alpha_operation", "ADD"},
                                                                    {"write_mask", 15}}}})}}},
                        {"render_pass",
                         {{"control", 0},
                          {"colors", nlohmann::json::array({{{"location", 0},
                                                             {"access", 0},
                                                             {"formats", nlohmann::json::array({"rgba8_unorm"})},
                                                             {"sample_counts", nlohmann::json::array({1})}}})},
                          {"depth_stencil", nullptr}}},
                        {"draw",
                         {{"control", 1},
                          {"default", {{"tag", "direct"}, {"vertex_count", 3}, {"instance_count", 1}}}}},
                        {"dynamic_state", {{"control", 2}}}}}}})}}})}};
    setProgramAbi(manifest, {{"target", 0, "input", "input"}, {"target", 1, "output", "output"}});
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
    EXPECT_EQ(vernon::runtime::program::executionKind(resolved.program.graphs.front().nodes.front()),
              vernon::runtime::program::ExecutionKind::Graphics);
}

} // namespace
