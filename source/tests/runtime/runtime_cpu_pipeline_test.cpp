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
    values[gid[2] * 6 + gid[1] * 3 + gid[0]] = static_cast<float>(gid[0] + 10 * gid[1] + 100 * gid[2]);
    return VERNON_STATUS_OK;
}

} // namespace

TEST(RuntimeCpuPipeline, ReflectsVersionedTextureConstraints) {
    const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
    const std::string directoryUtf8 = directory.u8string();
    const std::string bundle = readFile(directory / "cpu_fill.pipeline.json");
    ASSERT_FALSE(bundle.empty());

    VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
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
    parameter["access"] = "read";
    parameter["dimension"] = "3d";
    parameter["texture_format"] = "rgba16_float";
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
    EXPECT_EQ(constraint.has_format_constraint, 1u);
    EXPECT_EQ(constraint.format, VERNON_TEXTURE_RGBA16_FLOAT);

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

    constrained["variants"][0]["parameters"][0].erase("texture_format");
    VernonPipelineBundle *unconstrainedBundle = loadWithDirectory(runtime, withContentHash(constrained), directoryUtf8);
    ASSERT_NE(unconstrainedBundle, nullptr);
    VernonLoadedPipeline *unconstrainedPipeline = vernonRuntimeResolvePipeline(unconstrainedBundle, {nullptr, 0});
    ASSERT_NE(unconstrainedPipeline, nullptr);
    constraint = {};
    constraint.struct_size = sizeof(constraint);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetTextureConstraintByParameterIndex(unconstrainedPipeline, 0, &constraint),
              VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);
    EXPECT_EQ(constraint.has_format_constraint, 0u);

    vernonRuntimeLoadedPipelineDestroy(unconstrainedPipeline);
    vernonRuntimePipelineBundleDestroy(unconstrainedBundle);
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

    VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
    ASSERT_TRUE(runtime);

    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundle(runtime, bundle.data(), bundle.size()));

    VernonPipelineBundle *loaded = loadWithDirectory(runtime, bundle, directoryUtf8);
    ASSERT_TRUE(loaded);

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
    objectStage["format"] = "relocatable_object";
    objectStage["target_triple"] = "test-host-triple";
    objectStage["object_format"] = "elf";
    objectBundle.erase("content_hash");
    std::string objectCanonical = objectBundle.dump(-1, ' ', false);
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

    nlohmann::json invalidSchema2 = nlohmann::json::parse(bundle);
    invalidSchema2["stage_artifacts"]["fill"]["cpu_invocation_abi_version"] = 2;
    invalidSchema2.erase("content_hash");
    std::string canonical = invalidSchema2.dump(-1, ' ', false);
    invalidSchema2["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    canonical = invalidSchema2.dump(-1, ' ', false);
    ASSERT_TRUE(!loadWithDirectory(runtime, canonical, directoryUtf8));

    VernonPipelineBundleLoadOptions shortOptions{};
    shortOptions.struct_size = sizeof(shortOptions) - 1;
    shortOptions.bundle_directory = directoryUtf8.c_str();
    ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &shortOptions));

    nlohmann::json invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["variants"][0]["steps"].push_back({{"kind", "barrier"}});
    ASSERT_TRUE(!loadWithDirectory(runtime, withContentHash(invalidDocument), directoryUtf8));

    invalidDocument = nlohmann::json::parse(bundle);
    invalidDocument["variants"][0]["steps"] = {{{"kind", "draw"}, {"vertex", "fill"}, {"fragment", "fill"}}};
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

    const VernonStringView id = vernonRuntimePipelineBundleGetId(loaded);
    ASSERT_TRUE(id.size == std::strlen("cpu/fill"));
    ASSERT_TRUE(std::memcmp(id.data, "cpu/fill", id.size) == 0);

    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetStepCount(pipeline) == 1);
    VernonPipelineStepView step{};
    step.struct_size = sizeof(step);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetStepByIndex(pipeline, 0, &step) == VERNON_STATUS_OK);
    ASSERT_TRUE(step.kind == VERNON_PIPELINE_DISPATCH);
    ASSERT_TRUE(step.stage.size == std::strlen("fill") && std::memcmp(step.stage.data, "fill", step.stage.size) == 0);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterCount(pipeline) == 1);
    VernonPipelineParameterView parameter{};
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, 0, &parameter) == VERNON_STATUS_OK);
    ASSERT_TRUE(parameter.slot == 0 && parameter.kind == VERNON_PIPELINE_TENSOR && parameter.dtype == VERNON_DATA_F32 &&
                parameter.access == VERNON_ACCESS_WRITE && parameter.rank == 1 && parameter.static_shape[0] == 12);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"output", std::strlen("output")}, &parameter) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetOutputCount(pipeline) == 1);
    VernonPipelineOutputView outputView{};
    ASSERT_TRUE(vernonRuntimeLoadedPipelineFindOutput(pipeline, {"result", std::strlen("result")}, &outputView) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(outputView.kind == VERNON_PIPELINE_TENSOR && outputView.dtype == VERNON_DATA_F32 &&
                outputView.rank == 1 && outputView.static_shape[0] == 12 && outputView.location == 0);

    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, 12 * sizeof(float), alignof(float));
    ASSERT_TRUE(buffer);
    const uint64_t shape[] = {12};
    const int64_t strides[] = {sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_DEVICE;
    argument.tensor.buffer = buffer;
    argument.tensor.dtype = VERNON_DATA_F32;
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = 12 * sizeof(float);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {3, 2, 2};
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);

    float output[12]{};
    ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) == VERNON_STATUS_OK);
    ASSERT_TRUE(output[0] == 0.0f && output[2] == 2.0f);
    ASSERT_TRUE(output[3] == 10.0f && output[11] == 112.0f);

    ASSERT_TRUE(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
