#include "VernonCpuWorkgroupABI.h"
#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/autodiff/program_value_arena.h"
#include "runtime/autodiff/runtime_direct_autodiff.h"
#include "runtime/content_hash.h"
#include "runtime/runtime_dispatch.h"
#include "runtime_rhi_test_utils.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>

#ifndef VERNON_CPU_BUNDLE_PATH
#error VERNON_CPU_BUNDLE_PATH must name the CPU bundle test fixture
#endif

#ifndef VERNON_RUNTIME_TEST_OS
#error VERNON_RUNTIME_TEST_OS must name the host operating system
#endif

#ifndef VERNON_RUNTIME_TEST_ARCH
#error VERNON_RUNTIME_TEST_ARCH must name the host architecture
#endif

namespace {

std::string readFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

VernonPipelineBundle *loadWithDirectory(VernonRuntimeContext *runtime, const std::string &bundle,
                                        const std::string &directory) {
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    return vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
}

std::string withContentHash(nlohmann::json root) {
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

VernonStatus staticallyLinkedFill(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto *range = reinterpret_cast<VernonCpuRangeV1 *>(const_cast<void *>(invocation->arguments));
    if (!range || range->struct_size != sizeof(*range))
        return VERNON_STATUS_INVALID_ARGUMENT;
    uintptr_t address = 0;
    std::memcpy(&address, range->arguments, sizeof(address));
    float *values = reinterpret_cast<float *>(address);
    for (size_t lane = range->lane_begin; lane < range->lane_end; ++lane) {
        const uint32_t localX = static_cast<uint32_t>(lane % range->workgroup[0]);
        const uint32_t localY = static_cast<uint32_t>((lane / range->workgroup[0]) % range->workgroup[1]);
        const uint32_t localZ =
            static_cast<uint32_t>(lane / (static_cast<size_t>(range->workgroup[0]) * range->workgroup[1]));
        const uint32_t x = range->group[0] * range->workgroup[0] + localX;
        const uint32_t y = range->group[1] * range->workgroup[1] + localY;
        const uint32_t z = range->group[2] * range->workgroup[2] + localZ;
        values[z * 8 + y * 4 + x] = static_cast<float>(x + 10 * y + 100 * z);
    }
    return VERNON_STATUS_OK;
}

} // namespace

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
TEST(RuntimeCpuPipeline, ReflectsImageConstraintsAndRejectsLegacyMetadata) {
    const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
    const std::string directoryUtf8 = directory.u8string();
    const std::string bundle = readFile(directory / "cpu_fill.pipeline.json");
    ASSERT_FALSE(bundle.empty());

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);

    VernonPipelineBundle *tensorBundle = loadWithDirectory(runtime, bundle, directoryUtf8);
    ASSERT_NE(tensorBundle, nullptr);
    VernonLoadedPipeline *tensorPipeline = vernonRuntimeResolvePipeline(tensorBundle, {nullptr, 0});
    ASSERT_NE(tensorPipeline, nullptr);
    VernonPipelineImageConstraintView constraint{};
    constraint.struct_size = sizeof(constraint);
    EXPECT_EQ(vernonRuntimeLoadedPipelineGetImageConstraintByParameterIndex(tensorPipeline, 0, &constraint),
              VERNON_STATUS_INVALID_ARGUMENT);

    nlohmann::json constrained = nlohmann::json::parse(bundle);
    nlohmann::json &parameter = constrained["variants"][0]["parameters"][0];
    parameter["kind"] = "image";
    parameter["type"] = "!vernon.texture<\"3d\", f32, \"unknown\", \"sampled\">";
    parameter.erase("address_space");
    parameter.erase("element_layout");
    parameter["access"] = "read";
    parameter["dimension"] = "3d";
    parameter["binding_role"] = "sampled";
    parameter["sample_result_class"] = "float";
    parameter["shape"] = nlohmann::json::array();
    VernonPipelineBundle *textureBundle = loadWithDirectory(runtime, withContentHash(constrained), directoryUtf8);
    ASSERT_NE(textureBundle, nullptr);
    VernonLoadedPipeline *texturePipeline = vernonRuntimeResolvePipeline(textureBundle, {nullptr, 0});
    ASSERT_NE(texturePipeline, nullptr);

    constraint = {};
    constraint.struct_size = sizeof(constraint);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetImageConstraintByParameterIndex(texturePipeline, 0, &constraint),
              VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);
    EXPECT_EQ(constraint.binding_role, VERNON_IMAGE_BINDING_SAMPLED);

    constraint = {};
    constraint.struct_size = sizeof(constraint);
    ASSERT_EQ(
        vernonRuntimeLoadedPipelineFindImageConstraint(texturePipeline, {"output", std::strlen("output")}, &constraint),
        VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);

    constraint = {};
    constraint.struct_size = sizeof(constraint) - 1;
    EXPECT_EQ(vernonRuntimeLoadedPipelineGetImageConstraintByParameterIndex(texturePipeline, 0, &constraint),
              VERNON_STATUS_INVALID_ARGUMENT);

    constrained["variants"][0]["parameters"][0]["texture_format"] = "rgba16_float";
    EXPECT_EQ(loadWithDirectory(runtime, withContentHash(constrained), directoryUtf8), nullptr);

    vernonRuntimeLoadedPipelineDestroy(texturePipeline);
    vernonRuntimePipelineBundleDestroy(textureBundle);
    vernonRuntimeLoadedPipelineDestroy(tensorPipeline);
    vernonRuntimePipelineBundleDestroy(tensorBundle);
    vernonRuntimeDestroy(runtime);
}

TEST(RuntimeCpuPipeline, LoadsValidatesAndInvokesBundles) {
    const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
    const std::string directoryUtf8 = directory.u8string();
    const std::string bundle = readFile(directory / "cpu_fill.pipeline.json");
    ASSERT_TRUE(!bundle.empty());

    VernonRuntimeBackend target = VERNON_RUNTIME_CUDA;
    ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(bundle.data(), bundle.size(), &target) == VERNON_STATUS_OK);
    ASSERT_TRUE(target == VERNON_RUNTIME_CPU);

    nlohmann::json mixedTargetOptions = nlohmann::json::parse(bundle);
    mixedTargetOptions["target"]["options"]["version"] = 330;
    const std::string mixedTargetBundle = withContentHash(mixedTargetOptions);
    EXPECT_EQ(vernonRuntimePipelineBundleInspectTarget(mixedTargetBundle.data(), mixedTargetBundle.size(), &target),
              VERNON_STATUS_PARSE_ERROR);

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_TRUE(runtime);

    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), nullptr));

    VernonPipelineBundle *loaded = loadWithDirectory(runtime, bundle, directoryUtf8);
    const VernonStringView loadError = vernonRuntimeGetLastError(runtime);
    ASSERT_TRUE(loaded) << std::string(loadError.data ? loadError.data : "", loadError.size);

    const std::string objectBytes = "test relocatable object";
    const std::filesystem::path objectPath = directory / "test_static.o";
    {
        std::ofstream objectOutput(objectPath, std::ios::binary);
        objectOutput << objectBytes;
    }
    nlohmann::json objectBundle = nlohmann::json::parse(bundle);
    nlohmann::json &objectStage = objectBundle["stage_artifacts"]["fill"];
    objectStage["artifact"]["format"] = "relocatable_object";
    objectStage["artifact"]["path"] = objectPath.filename().string();
    objectStage["artifact"]["size"] = objectBytes.size();
    objectStage["artifact"]["sha256"] = vernon::runtime::sha256Hex(objectBytes.data(), objectBytes.size());
    objectStage["symbol"] = "vernon_missing_static_fill";
    objectBundle.erase("content_hash");
    std::string objectCanonical = objectBundle.dump(-1, ' ', false);
    objectBundle["content_hash"] = vernon::runtime::sha256Hex(objectCanonical.data(), objectCanonical.size());
    const std::string missingObjectManifest = objectBundle.dump(-1, ' ', false);
    // The object is a link-time input, not a runtime artifact. A deployed
    // application carries the linked symbol and metadata, but not the .o/.obj.
    ASSERT_TRUE(std::filesystem::remove(objectPath));
    VernonPipelineBundle *missingObjectLoaded = vernonRuntimeLoadPipelineBundleWithOptions(
        runtime, missingObjectManifest.data(), missingObjectManifest.size(), nullptr);
    ASSERT_TRUE(missingObjectLoaded);
    EXPECT_EQ(vernonRuntimeResolvePipeline(missingObjectLoaded, {nullptr, 0}), nullptr);
    const VernonStringView missingRegistrationError = vernonRuntimeGetLastError(runtime);
    EXPECT_NE(
        std::string(missingRegistrationError.data ? missingRegistrationError.data : "", missingRegistrationError.size)
            .find("was not statically registered"),
        std::string::npos);
    vernonRuntimePipelineBundleDestroy(missingObjectLoaded);

    objectStage["symbol"] = "vernon_test_fill";
    objectBundle.erase("content_hash");
    objectCanonical = objectBundle.dump(-1, ' ', false);
    objectBundle["content_hash"] = vernon::runtime::sha256Hex(objectCanonical.data(), objectCanonical.size());
    const std::string objectManifest = objectBundle.dump(-1, ' ', false);
    ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry({"vernon_test_fill", std::strlen("vernon_test_fill")},
                                                    staticallyLinkedFill) == VERNON_STATUS_OK);
    VernonPipelineBundle *objectLoaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, objectManifest.data(), objectManifest.size(), nullptr);
    ASSERT_TRUE(objectLoaded);
    VernonLoadedPipeline *objectPipeline = vernonRuntimeResolvePipeline(objectLoaded, {nullptr, 0});
    ASSERT_TRUE(objectPipeline);
    vernonRuntimeLoadedPipelineDestroy(objectPipeline);
    vernonRuntimePipelineBundleDestroy(objectLoaded);

    nlohmann::json invalidVersion = nlohmann::json::parse(bundle);
    invalidVersion["pipeline_version"] = VERNON_PIPELINE_VERSION + 1;
    invalidVersion.erase("content_hash");
    std::string canonical = invalidVersion.dump(-1, ' ', false);
    invalidVersion["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    canonical = invalidVersion.dump(-1, ' ', false);
    ASSERT_TRUE(!loadWithDirectory(runtime, canonical, directoryUtf8));

    nlohmann::json previousVersion = nlohmann::json::parse(bundle);
    previousVersion["pipeline_version"] = 15;
    previousVersion.erase("content_hash");
    canonical = previousVersion.dump(-1, ' ', false);
    previousVersion["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    canonical = previousVersion.dump(-1, ' ', false);
    ASSERT_TRUE(!loadWithDirectory(runtime, canonical, directoryUtf8));

    VernonPipelineBundleLoadOptions shortOptions{};
    shortOptions.struct_size = sizeof(shortOptions) - 1;
    shortOptions.bundle_directory = directoryUtf8.c_str();
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &shortOptions));

    nlohmann::json invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["variants"][0]["program"] = nlohmann::json::object();
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["variants"][0]["program"] = {{"compute", "fill"}, {"vertex", "fill"}, {"fragment", "fill"}};
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    std::string &artifactHash =
        invalidDocument["stage_artifacts"]["fill"]["artifact"]["sha256"].get_ref<std::string &>();
    artifactHash[0] = artifactHash[0] == '0' ? '1' : '0';
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["stage_artifacts"]["fill"]["artifact"]["path"] =
        "../" + invalidDocument["stage_artifacts"]["fill"]["artifact"]["path"].get<std::string>();
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["stage_artifacts"]["fill"]["operating_system"] = "unsupported";
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["stage_artifacts"]["fill"]["architecture"] = "unsupported";
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["features"] = nlohmann::json::array();
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["stage_artifacts"]["fill"]["artifact"]["legacy"] = true;
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    const VernonStringView id = vernonRuntimePipelineBundleGetId(loaded);
    ASSERT_TRUE(id.size == std::strlen("cpu/fill"));
    ASSERT_TRUE(std::memcmp(id.data, "cpu/fill", id.size) == 0);

    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    EXPECT_EQ(pipeline->topology, nullptr);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterCount(pipeline) == 1);
    VernonPipelineParameterView parameter{};
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, 0, &parameter) == VERNON_STATUS_OK);
    ASSERT_TRUE(parameter.slot == 0 && parameter.kind == VERNON_PIPELINE_TENSOR &&
                parameter.element_layout.leaf_count == 1 &&
                parameter.element_layout.leaves[0].dtype == VERNON_DATA_F32 &&
                parameter.access == VERNON_ACCESS_WRITE && parameter.rank == 1 && parameter.static_shape[0] == 16);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"output", std::strlen("output")}, &parameter) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetOutputCount(pipeline) == 1);
    VernonPipelineOutputView outputView{};
    ASSERT_TRUE(vernonRuntimeLoadedPipelineFindOutput(pipeline, {"result", std::strlen("result")}, &outputView) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(outputView.kind == VERNON_PIPELINE_TENSOR && outputView.dtype == VERNON_DATA_F32 &&
                outputView.rank == 1 && outputView.static_shape[0] == 16 && outputView.location == 0);

    float output[16]{};
    const uint64_t shape[] = {16};
    const int64_t strides[] = {sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = output;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 16 * sizeof(float);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {2, 1, 2};
    ASSERT_TRUE(vernon::tests::completeSubmission(pipeline, &invocation) == VERNON_STATUS_OK);

    ASSERT_TRUE(output[0] == 0.0f && output[3] == 3.0f);
    ASSERT_TRUE(output[4] == 10.0f && output[15] == 113.0f);

    nlohmann::json constantWriteBundle = nlohmann::json::parse(bundle);
    nlohmann::json &constantEntry = constantWriteBundle["stage_artifacts"]["fill"]["reflection"]["entries"][0];
    constantEntry["dispatch_contract"] = {{"unit_grid_axes", {0, 1, 2}}, {"requires_unit_workgroup", true}};
    VernonPipelineBundle *constantLoaded =
        loadWithDirectory(runtime, withContentHash(constantWriteBundle), directoryUtf8);
    ASSERT_TRUE(constantLoaded);
    VernonLoadedPipeline *constantPipeline = vernonRuntimeResolvePipeline(constantLoaded, {nullptr, 0});
    ASSERT_TRUE(constantPipeline);
    EXPECT_EQ(vernon::tests::completeSubmission(constantPipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStringView constantError = vernonRuntimeGetLastError(runtime);
    EXPECT_NE(std::string(constantError.data ? constantError.data : "", constantError.size)
                  .find("compute dispatch grid axis"),
              std::string::npos);
    invocation.compute_grid = {1, 1, 1};
    EXPECT_EQ(vernon::tests::completeSubmission(constantPipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);

    vernonRuntimeLoadedPipelineDestroy(constantPipeline);
    vernonRuntimePipelineBundleDestroy(constantLoaded);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}

TEST(RuntimeCpuPipeline, RejectsLegacyExecutableTopology) {
    const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
    const std::string directoryUtf8 = directory.u8string();
    nlohmann::json bundle = nlohmann::json::parse(readFile(directory / "cpu_fill.pipeline.json"));
    nlohmann::json &variant = bundle["variants"][0];
    variant["program"] = {{"forward:0", "fill"}, {"forward:1", "fill"}};
    nlohmann::json internal = variant["parameters"][0];
    internal.erase("slot");
    internal["name"] = "temporary";
    internal["source"] = "program_value";
    internal["uses"][0]["stage"] = "forward:0";
    variant["internal_parameters"] = nlohmann::json::array({std::move(internal)});
    variant["parameters"][0]["uses"][0]["stage"] = "forward:1";
    variant["execution"] = {
        {"values", nlohmann::json::array({{{"id", 0},
                                           {"name", "temporary"},
                                           {"type", "tensor<16xf32>"},
                                           {"dtype", "f32"},
                                           {"shape", {16}},
                                           {"external", false},
                                           {"output", false}},
                                          {{"id", 1},
                                           {"name", "output"},
                                           {"type", "tensor<16xf32>"},
                                           {"dtype", "f32"},
                                           {"shape", {16}},
                                           {"external", true},
                                           {"output", true}}})},
        {"graphs",
         nlohmann::json::array(
             {{{"name", "forward"},
               {"direction", "forward"},
               {"arguments", nlohmann::json::array()},
               {"results", {1}},
               {"nodes", nlohmann::json::array(
                             {{{"id", 0},
                               {"name", "fill temporary"},
                               {"kind", "compute"},
                               {"stage", "forward:0"},
                               {"operands", nlohmann::json::array()},
                               {"results", {0}},
                               {"dependencies", nlohmann::json::array()},
                               {"bindings", nlohmann::json::array({{{"parameter", "temporary"}, {"value", 0}}})},
                               {"resources", nlohmann::json::array({{{"value", 0}, {"access", "write"}}})},
                               {"grid", {2, 1, 2}}},
                              {{"id", 1},
                               {"name", "fill output"},
                               {"kind", "compute"},
                               {"stage", "forward:1"},
                               {"operands", nlohmann::json::array()},
                               {"results", {1}},
                               {"dependencies", {0}},
                               {"bindings", nlohmann::json::array({{{"parameter", "output"}, {"value", 1}}})},
                               {"resources", nlohmann::json::array({{{"value", 1}, {"access", "write"}}})},
                               {"grid", {2, 1, 2}}}})}}})}};

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    VernonPipelineBundle *loaded = loadWithDirectory(runtime, withContentHash(bundle), directoryUtf8);
    const VernonStringView loadError = vernonRuntimeGetLastError(runtime);
    EXPECT_EQ(loaded, nullptr);
    EXPECT_NE(std::string(loadError.data ? loadError.data : "", loadError.size).find("unknown or legacy field"),
              std::string::npos);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    return;
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    const VernonStringView resolveError = vernonRuntimeGetLastError(runtime);
    ASSERT_NE(pipeline, nullptr) << std::string(resolveError.data ? resolveError.data : "", resolveError.size);
    ASSERT_NE(pipeline->topology, nullptr);
    EXPECT_EQ(pipeline->topology->stages.size(), 2u);

    VernonPipelineParameterView parameter{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, 0, &parameter), VERNON_STATUS_OK);
    float output[16]{};
    const uint64_t shape[] = {16};
    const int64_t strides[] = {sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = parameter.slot;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = output;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(output);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(output[0], 0.0f);
    EXPECT_EQ(output[3], 3.0f);
    EXPECT_EQ(output[4], 10.0f);
    EXPECT_EQ(output[15], 113.0f);

    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}

TEST(RuntimeCpuPipeline, ResolvesAndExecutesNativeBackwardProgramGraph) {
    const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
    const std::string directoryUtf8 = directory.u8string();
    nlohmann::json bundle = nlohmann::json::parse(readFile(directory / "cpu_fill.pipeline.json"));
    nlohmann::json &variant = bundle["variants"][0];
    nlohmann::json gradient = variant["parameters"][0];
    gradient["slot"] = 1;
    gradient["name"] = "gradient";
    gradient["uses"][0]["stage"] = "backward:0";
    nlohmann::json residual = variant["parameters"][0];
    residual["slot"] = 2;
    residual["name"] = "residual";
    residual["uses"][0]["stage"] = "forward:residual";
    variant["parameters"][0]["uses"][0]["stage"] = "forward:0";
    nlohmann::json captureUse = variant["parameters"][0]["uses"][0];
    captureUse["stage"] = "backward:capture_output";
    variant["parameters"][0]["uses"].push_back(std::move(captureUse));
    nlohmann::json residualCaptureUse = residual["uses"][0];
    residualCaptureUse["stage"] = "backward:capture_residual";
    residual["uses"].push_back(std::move(residualCaptureUse));
    variant["parameters"].push_back(std::move(gradient));
    variant["parameters"].push_back(std::move(residual));
    variant["program"] = {{"forward:residual", "fill"},
                          {"forward:0", "fill"},
                          {"backward:capture_output", "fill"},
                          {"backward:capture_residual", "fill"},
                          {"backward:0", "fill"}};
    const auto value = [](uint32_t id, const char *name) {
        return nlohmann::json{{"id", id},       {"name", name},  {"type", "tensor<16xf32>"},
                              {"dtype", "f32"}, {"shape", {16}}, {"external", true},
                              {"output", true}};
    };
    const auto node = [](uint32_t id, const char *name, const char *stage, uint32_t result, const char *parameter,
                         nlohmann::json dependencies) {
        return nlohmann::json{{"id", id},
                              {"name", name},
                              {"kind", "compute"},
                              {"stage", stage},
                              {"operands", nlohmann::json::array()},
                              {"results", {result}},
                              {"dependencies", std::move(dependencies)},
                              {"bindings", nlohmann::json::array({{{"parameter", parameter}, {"value", result}}})},
                              {"resources", nlohmann::json::array({{{"value", result}, {"access", "write"}}})},
                              {"grid", {2, 1, 2}}};
    };
    variant["execution"] = {
        {"values", nlohmann::json::array({value(0, "output"), value(1, "gradient"), value(2, "residual")})},
        {"graphs",
         nlohmann::json::array(
             {{{"name", "forward"},
               {"direction", "forward"},
               {"arguments", nlohmann::json::array()},
               {"results", {0}},
               {"nodes", nlohmann::json::array(
                             {node(0, "fill residual", "forward:residual", 2, "residual", nlohmann::json::array()),
                              node(1, "fill output", "forward:0", 0, "output", {0})})}},
              {{"name", "backward"},
               {"direction", "backward"},
               {"arguments", nlohmann::json::array()},
               {"results", {1}},
               {"nodes", nlohmann::json::array(
                             {{{"id", 0},
                               {"name", "consume output capture"},
                               {"kind", "compute"},
                               {"stage", "backward:capture_output"},
                               {"operands", {0}},
                               {"results", nlohmann::json::array()},
                               {"dependencies", nlohmann::json::array()},
                               {"bindings", nlohmann::json::array({{{"parameter", "output"}, {"value", 0}}})},
                               {"resources", nlohmann::json::array({{{"value", 0}, {"access", "read_write"}}})},
                               {"grid", {2, 1, 2}}},
                              {{"id", 1},
                               {"name", "consume residual capture"},
                               {"kind", "compute"},
                               {"stage", "backward:capture_residual"},
                               {"operands", {2}},
                               {"results", nlohmann::json::array()},
                               {"dependencies", {0}},
                               {"bindings", nlohmann::json::array({{{"parameter", "residual"}, {"value", 2}}})},
                               {"resources", nlohmann::json::array({{{"value", 2}, {"access", "read_write"}}})},
                               {"grid", {2, 1, 2}}},
                              {{"id", 2},
                               {"name", "fill gradient"},
                               {"kind", "compute"},
                               {"stage", "backward:0"},
                               {"operands", nlohmann::json::array()},
                               {"results", {1}},
                               {"dependencies", {1}},
                               {"bindings", nlohmann::json::array({{{"parameter", "gradient"}, {"value", 1}}})},
                               {"resources", nlohmann::json::array({{{"value", 1}, {"access", "write"}}})},
                               {"grid", {2, 1, 2}}}})}}})}};

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    VernonPipelineBundle *loaded = loadWithDirectory(runtime, withContentHash(bundle), directoryUtf8);
    const VernonStringView loadError = vernonRuntimeGetLastError(runtime);
    EXPECT_EQ(loaded, nullptr);
    EXPECT_NE(std::string(loadError.data ? loadError.data : "", loadError.size).find("unknown or legacy field"),
              std::string::npos);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}
#endif

#if defined(VERNON_RUNTIME_PROFILE_WEB)
TEST(RuntimeCpuPipeline, WebProfileLoadsMultipleStaticPipelinesWithoutFilesystem) {
    const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
    const std::string fixture = readFile(directory / "cpu_fill.pipeline.json");
    ASSERT_FALSE(fixture.empty());

    auto wasmManifest = [&](const char *id, const char *symbol) {
        nlohmann::json root = nlohmann::json::parse(fixture);
        root["id"] = id;
        root["target"]["options"] = {{"triple", "wasm32-unknown-emscripten"}};
        root["runtime_requirements"]["target_triple"] = "wasm32-unknown-emscripten";
        root["runtime_requirements"]["object_format"] = "wasm";
        nlohmann::json &stage = root["stage_artifacts"]["fill"];
        stage["symbol"] = symbol;
        stage["artifact"] = {{"format", "relocatable_object"},
                             {"storage", "external"},
                             {"path", std::string(symbol) + ".wasm.o"},
                             {"size", 16},
                             {"sha256", std::string(64, 'a')}};
        stage["reflection"]["target"]["options"] = {{"triple", "wasm32-unknown-emscripten"}};
        return withContentHash(std::move(root));
    };

    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({"vernon_web_fill_a", 17}, staticallyLinkedFill), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({"vernon_web_fill_b", 17}, staticallyLinkedFill), VERNON_STATUS_OK);
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);

    const std::string firstManifest = wasmManifest("cpu/web/a", "vernon_web_fill_a");
    const std::string secondManifest = wasmManifest("cpu/web/b", "vernon_web_fill_b");
    VernonPipelineBundle *firstBundle =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, firstManifest.data(), firstManifest.size(), nullptr);
    VernonPipelineBundle *secondBundle =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, secondManifest.data(), secondManifest.size(), nullptr);
    const VernonStringView loadError = vernonRuntimeGetLastError(runtime);
    ASSERT_NE(firstBundle, nullptr) << std::string(loadError.data ? loadError.data : "", loadError.size);
    ASSERT_NE(secondBundle, nullptr) << std::string(loadError.data ? loadError.data : "", loadError.size);
    VernonLoadedPipeline *first = vernonRuntimeResolvePipeline(firstBundle, {nullptr, 0});
    VernonLoadedPipeline *second = vernonRuntimeResolvePipeline(secondBundle, {nullptr, 0});
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    VernonPipelineParameterView parameter{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetParameterByIndex(first, 0, &parameter), VERNON_STATUS_OK);
    float output[16]{};
    const uint64_t shape[] = {16};
    const int64_t strides[] = {sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = output;
    argument.tensor.element_layout = parameter.element_layout;
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(output);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {2, 1, 2};
    EXPECT_EQ(vernon::tests::completeSubmission(first, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(vernon::tests::completeSubmission(second, &invocation), VERNON_STATUS_OK);
    EXPECT_EQ(output[15], 113.0f);

    vernonRuntimeLoadedPipelineDestroy(second);
    vernonRuntimeLoadedPipelineDestroy(first);
    vernonRuntimePipelineBundleDestroy(secondBundle);
    vernonRuntimePipelineBundleDestroy(firstBundle);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}
#endif
