#include "runtime/content_hash.h"
#include "vernon-c/Runtime.h"

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
    uintptr_t address = 0;
    uint32_t gid[3] = {0, 0, 0};
    if (!invocation || invocation->arguments_size < 20)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::memcpy(&address, invocation->arguments, sizeof(address));
    std::memcpy(gid, static_cast<const unsigned char *>(invocation->arguments) + 8, sizeof(gid));
    float *values = reinterpret_cast<float *>(address);
    values[gid[2] * 8 + gid[1] * 4 + gid[0]] = static_cast<float>(gid[0] + 10 * gid[1] + 100 * gid[2]);
    return VERNON_STATUS_OK;
}

} // namespace

TEST(RuntimeCpuPipeline, ReflectsTextureDimensionsAndRejectsFormatMetadata) {
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
    VernonPipelineTextureConstraintView constraint{};
    constraint.struct_size = sizeof(constraint);
    EXPECT_EQ(vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(tensorPipeline, 0, &constraint),
              VERNON_STATUS_INVALID_ARGUMENT);

    nlohmann::json constrained = nlohmann::json::parse(bundle);
    nlohmann::json &parameter = constrained["variants"][0]["parameters"][0];
    parameter["kind"] = "texture";
    parameter["type"] = "!vernon.texture<\"3d\", f32>";
    parameter.erase("address_space");
    parameter.erase("element_layout");
    parameter["access"] = "read";
    parameter["dimension"] = "3d";
    parameter["shape"] = nlohmann::json::array();
    VernonPipelineBundle *textureBundle = loadWithDirectory(runtime, withContentHash(constrained), directoryUtf8);
    ASSERT_NE(textureBundle, nullptr);
    VernonLoadedPipeline *texturePipeline = vernonRuntimeResolvePipeline(textureBundle, {nullptr, 0});
    ASSERT_NE(texturePipeline, nullptr);

    constraint = {};
    constraint.struct_size = sizeof(constraint);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(texturePipeline, 0, &constraint),
              VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);
    EXPECT_EQ(constraint.has_format_constraint, 0u);

    constraint = {};
    constraint.struct_size = sizeof(constraint);
    ASSERT_EQ(vernonRuntimeLoadedPipelineFindTextureConstraint(texturePipeline, {"output", std::strlen("output")},
                                                               &constraint),
              VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);

    constraint = {};
    constraint.struct_size = sizeof(constraint) - 1;
    EXPECT_EQ(vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(texturePipeline, 0, &constraint),
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
    VernonPipelineBundle *missingObjectLoaded = loadWithDirectory(runtime, missingObjectManifest, directoryUtf8);
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
    VernonPipelineBundle *objectLoaded = loadWithDirectory(runtime, objectManifest, directoryUtf8);
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
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);

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
    EXPECT_EQ(vernonRuntimePipelineInvoke(constantPipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);
    const VernonStringView constantError = vernonRuntimeGetLastError(runtime);
    EXPECT_NE(std::string(constantError.data ? constantError.data : "", constantError.size)
                  .find("compute dispatch grid axis"),
              std::string::npos);
    invocation.compute_grid = {1, 1, 1};
    EXPECT_EQ(vernonRuntimePipelineInvoke(constantPipeline, &invocation), VERNON_STATUS_INVALID_ARGUMENT);

    vernonRuntimeLoadedPipelineDestroy(constantPipeline);
    vernonRuntimePipelineBundleDestroy(constantLoaded);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
