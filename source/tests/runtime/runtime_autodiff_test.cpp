#include "VernonRuntime.hpp"
#include "runtime/content_hash.h"
#include "runtime/runtime_autodiff_internal.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {

std::string lastError(const VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return error.data ? std::string(error.data, error.size) : std::string();
}

VernonStatus squareForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != sizeof(float) || invocation->results_size != 2 * sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float x = 0.0f;
    std::memcpy(&x, invocation->arguments, sizeof(x));
    const float results[]{x * x, x};
    std::memcpy(invocation->results, results, sizeof(results));
    return VERNON_STATUS_OK;
}

VernonStatus squareBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(float) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float arguments[2]{};
    std::memcpy(arguments, invocation->arguments, sizeof(arguments));
    const float gradient = 2.0f * arguments[0] * arguments[1];
    std::memcpy(invocation->results, &gradient, sizeof(gradient));
    return VERNON_STATUS_OK;
}

VernonStatus tensorSquareForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 * sizeof(float) || invocation->results_size != 4 * sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float x[2]{};
    std::memcpy(x, invocation->arguments, sizeof(x));
    const float results[]{x[0] * x[0], x[1] * x[1], x[0], x[1]};
    std::memcpy(invocation->results, results, sizeof(results));
    return VERNON_STATUS_OK;
}

VernonStatus tensorSquareBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 4 * sizeof(float) || invocation->results_size != 2 * sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float arguments[4]{};
    std::memcpy(arguments, invocation->arguments, sizeof(arguments));
    const float gradients[]{2.0f * arguments[0] * arguments[2], 2.0f * arguments[1] * arguments[3]};
    std::memcpy(invocation->results, gradients, sizeof(gradients));
    return VERNON_STATUS_OK;
}

VernonStatus aggregateForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(float) || invocation->results_size != 3 * sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float arguments[3]{};
    std::memcpy(arguments, invocation->arguments, sizeof(arguments));
    const float results[]{arguments[0] * arguments[1] * arguments[2], arguments[0], arguments[1]};
    std::memcpy(invocation->results, results, sizeof(results));
    return VERNON_STATUS_OK;
}

VernonStatus aggregateBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 3 * sizeof(float) || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    float arguments[3]{};
    std::memcpy(arguments, invocation->arguments, sizeof(arguments));
    const float gradient = arguments[0] * arguments[1] * arguments[2];
    std::memcpy(invocation->results, &gradient, sizeof(gradient));
    return VERNON_STATUS_OK;
}

VernonStatus halfIdentityForward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 2 || invocation->results_size != 4)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::memcpy(invocation->results, invocation->arguments, 2);
    std::memcpy(static_cast<std::byte *>(invocation->results) + 2, invocation->arguments, 2);
    return VERNON_STATUS_OK;
}

VernonStatus halfIdentityBackward(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != 8 || invocation->results_size != sizeof(float))
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::memcpy(invocation->results, static_cast<const std::byte *>(invocation->arguments) + 4, sizeof(float));
    return VERNON_STATUS_OK;
}

nlohmann::json leaf(size_t offset, size_t scalarCount = 1, const char *dtype = "f32", bool shaped = false) {
    nlohmann::json result = {
        {"path", nlohmann::json::array()}, {"dtype", dtype}, {"scalar_count", scalarCount}, {"byte_offset", offset}};
    if (scalarCount != 1 || shaped)
        result["shape"] = nlohmann::json::array({scalarCount});
    return result;
}

nlohmann::json transportRoot(const nlohmann::json &leaves, size_t size, size_t alignment) {
    const auto scalarSize = [](const std::string &dtype) {
        return dtype == "f16" ? size_t{2} : dtype == "f64" ? size_t{8} : size_t{4};
    };
    const auto child = [&](const nlohmann::json &value) {
        const std::string dtype = value["dtype"].get<std::string>();
        const size_t count = value.value("scalar_count", size_t{1});
        const size_t offset = value.value("byte_offset", size_t{0});
        const size_t laneSize = scalarSize(dtype);
        if (count == 1)
            return nlohmann::json{{"kind", "scalar"},
                                  {"representation", dtype},
                                  {"offset", offset},
                                  {"size", laneSize},
                                  {"alignment", laneSize}};
        return nlohmann::json{{"kind", "array"},
                              {"offset", offset},
                              {"size", count * laneSize},
                              {"alignment", laneSize},
                              {"shape", nlohmann::json::array({count})},
                              {"byte_strides", nlohmann::json::array({laneSize})},
                              {"children", nlohmann::json::array({{{"kind", "scalar"},
                                                                   {"representation", dtype},
                                                                   {"offset", 0},
                                                                   {"size", laneSize},
                                                                   {"alignment", laneSize}}})}};
    };
    if (leaves.size() == 1) {
        nlohmann::json root = child(leaves[0]);
        root["offset"] = 0;
        root["size"] = size;
        root["alignment"] = alignment;
        return root;
    }
    nlohmann::json children = nlohmann::json::array();
    for (const nlohmann::json &value : leaves)
        children.push_back(child(value));
    return {{"kind", "product"},
            {"offset", 0},
            {"size", size},
            {"alignment", alignment},
            {"children", std::move(children)}};
}

nlohmann::json argument(const char *name, size_t offset, nlohmann::json leaves, size_t size = 4, size_t alignment = 4) {
    const std::string dtype =
        leaves.size() == 1 && leaves[0].contains("dtype") ? leaves[0]["dtype"].get<std::string>() : "";
    nlohmann::json root = transportRoot(leaves, size, alignment);
    return {{"kind", "scalar"},
            {"dtype", dtype},
            {"vernon.source_name", name},
            {"value_layout", {{"leaves", std::move(leaves)}}},
            {"physical_layouts",
             {{"host_value",
               {{"profile", "host_value"},
                {"kind", "cpu_call"},
                {"canonical_layout_hash", "test-layout"},
                {"frame_offset", offset},
                {"root", std::move(root)}}}}}};
}

nlohmann::json profileReflection(const char *entry, size_t argumentBytes, size_t resultBytes, nlohmann::json arguments,
                                 nlohmann::json resultLeaves, size_t resultAlignment = 4) {
    nlohmann::json resultRoot = transportRoot(resultLeaves, resultBytes, resultAlignment);
    return {
        {"compiler_contract_version", VERNON_COMPILER_CONTRACT_VERSION},
        {"pipeline_version", VERNON_PIPELINE_VERSION},
        {"entries", nlohmann::json::array(
                        {{{"name", entry},
                          {"workgroup_size", nlohmann::json::array({1, 1, 1})},
                          {"physical_layouts",
                           {{"host_value",
                             {{"profile", "host_value"},
                              {"packed_arguments_size", argumentBytes},
                              {"packed_results_size", resultBytes}}}}},
                          {"arguments", std::move(arguments)},
                          {"results", nlohmann::json::array({{{"value_layout", {{"leaves", std::move(resultLeaves)}}},
                                                              {"physical_layouts",
                                                               {{"host_value",
                                                                 {{"profile", "host_value"},
                                                                  {"kind", "cpu_call"},
                                                                  {"canonical_layout_hash", "test-layout"},
                                                                  {"frame_offset", uint64_t{0}},
                                                                  {"root", std::move(resultRoot)}}}}}}})}}})}};
}

vernon::runtime::CpuNativeArtifact artifact(const std::filesystem::path &root, const char *entry, const char *symbol,
                                            nlohmann::json reflection) {
    vernon::runtime::CpuNativeArtifact result;
    result.root = root;
    result.relativeLibrary = "fixture.o";
    result.format = "relocatable_object";
    result.entry = entry;
    result.symbol = symbol;
#if defined(_WIN32)
    result.targetTriple = "x86_64-pc-windows-msvc";
    result.objectFormat = "coff";
#elif defined(__APPLE__)
#if defined(__aarch64__)
    result.targetTriple = "aarch64-apple-darwin";
#else
    result.targetTriple = "x86_64-apple-darwin";
#endif
    result.objectFormat = "macho";
#else
#if defined(__aarch64__)
    result.targetTriple = "aarch64-unknown-linux-gnu";
#else
    result.targetTriple = "x86_64-unknown-linux-gnu";
#endif
    result.objectFormat = "elf";
#endif
    constexpr char bytes[] = "registered object fixture";
    result.size = sizeof(bytes) - 1;
    result.sha256 = vernon::runtime::sha256Hex(bytes, sizeof(bytes) - 1);
    result.reflection = std::move(reflection);
    return result;
}

vernon::runtime::Stage reflectedStage(const char *entry, const nlohmann::json &reflection) {
    vernon::runtime::Stage result;
    result.stage = "compute";
    result.entry = entry;
    result.reflection = reflection.dump();
    return result;
}

vernon::runtime::Stage stage(vernon::runtime::CpuNativeArtifact artifact) {
    vernon::runtime::Stage result = reflectedStage(artifact.entry.c_str(), artifact.reflection);
    result.cpuArtifact = std::move(artifact);
    return result;
}

void attachCpuAutodiff(VernonRuntimeContext &context, VernonLoadedPipeline &pipeline, vernon::runtime::Stage forward,
                       vernon::runtime::Stage backward, std::vector<std::string> gradientPaths) {
    std::shared_ptr<vernon::runtime::ad::Executable> executable;
    ASSERT_TRUE(vernon::runtime::ad::createCpuExecutable(context, forward, backward, gradientPaths, executable))
        << lastError(&context);
    pipeline.autodiff = VernonLoadedAutodiff{std::move(executable)};
}

class MetadataExecutable final : public vernon::runtime::ad::HostExecutable {
public:
    explicit MetadataExecutable(vernon::runtime::ad::Signature signature) : signature_(std::move(signature)) {}

    const vernon::runtime::ad::Signature &signature() const override { return signature_; }

    VernonStatus forward(VernonLaunchSize, const VernonAdValueSet &, VernonAdValueSet &,
                         std::unique_ptr<vernon::runtime::ad::PullbackExecution> &) override {
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }

private:
    vernon::runtime::ad::Signature signature_;
};

TEST(RuntimeAutodiff, OrdinaryRuntimeFailureReplacesThreadLocalInvocationDiagnostic) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_CPU;
    vernon::runtime::invocationDiagnostic(context) = "stale autodiff diagnostic";
    constexpr char invalidBundle[] = "{";
    EXPECT_EQ(vernonRuntimeLoadPipelineBundleWithOptions(&context, invalidBundle, sizeof(invalidBundle) - 1, nullptr),
              nullptr);
    const VernonStringView error = vernonRuntimeGetLastError(&context);
    const std::string message(error.data, error.size);
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message, "stale autodiff diagnostic");
    vernon::runtime::clearInvocationDiagnostic(context);
}

TEST(RuntimeAutodiff, ExecutesReusableScalarPullbackFromRegisteredProfiles) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_square_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_square_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, squareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, squareBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_registered_profiles";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }

    const nlohmann::json forwardReflection =
        profileReflection("forward", 4, 8, nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0)}))}),
                          nlohmann::json::array({leaf(0), leaf(4)}));
    const nlohmann::json backwardReflection =
        profileReflection("backward", 8, 4,
                          nlohmann::json::array({argument("tape", 0, nlohmann::json::array({leaf(0)})),
                                                 argument("output", 4, nlohmann::json::array({leaf(0)}))}),
                          nlohmann::json::array({leaf(0)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    VernonDataType outputType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDataType(&pipeline, &outputType), VERNON_STATUS_OK);
    EXPECT_EQ(outputType, VERNON_DATA_F32);
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradientCount(&pipeline), 1u);
    VernonStringView gradientPath{};
    VernonDataType gradientType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradient(&pipeline, 0, &gradientPath, &gradientType), VERNON_STATUS_OK);
    EXPECT_EQ(std::string(gradientPath.data, gradientPath.size), "x");
    EXPECT_EQ(gradientType, VERNON_DATA_F32);

    float x = 3.0f;
    float outputValue = 0.0f;
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 0, nullptr};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {0, 0, 0, 0}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {0, 0, 0, 0}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);

    x = 100.0f;
    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue,
                           sizeof(gradientValue), 0,        nullptr};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {0, 0, 0, 0}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    float seedValue = 2.0f;
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &seedValue, sizeof(seedValue), 0, nullptr};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {0, 0, 0, 0}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 12.0f);

    vernonPullbackDestroy(pullback);

    outputValue = 0.0f;
    gradientValue = 0.0f;
    vernon::runtime::Pullback cppPullback = vernon::runtime::vjp(&pipeline, inputs, outputs, {1, 1, 1});
    vernon::runtime::Pullback movedPullback = std::move(cppPullback);
    EXPECT_FALSE(static_cast<bool>(cppPullback));
    movedPullback.apply(nullptr, gradients);
    EXPECT_FLOAT_EQ(outputValue, 10000.0f);
    EXPECT_FLOAT_EQ(gradientValue, 200.0f);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_INVALID_ARGUMENT);
    movedPullback = {};

    x = 2.0f;
    std::array<float, 24> batchedOutput{};
    const uint64_t batchedShape[]{4, 3, 2};
    output.data = batchedOutput.data();
    output.size = sizeof(batchedOutput);
    output.rank = 3;
    output.shape = batchedShape;
    VernonPullback *batchedPullback = nullptr;
    EXPECT_EQ(vernonAdPipelineForward(&pipeline, {0, 3, 4}, &inputs, &outputs, &batchedPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {2, 3, 4}, &inputs, &outputs, &batchedPullback), VERNON_STATUS_OK)
        << lastError(context);
    for (float value : batchedOutput)
        EXPECT_FLOAT_EQ(value, 4.0f);
    std::array<float, 24> batchedSeed{};
    for (size_t index = 0; index < batchedSeed.size(); ++index)
        batchedSeed[index] = static_cast<float>(index + 1);
    seed.data = batchedSeed.data();
    seed.size = sizeof(batchedSeed);
    seed.rank = 3;
    seed.shape = batchedShape;
    gradientValue = 0.0f;
    ASSERT_EQ(vernonPullbackApply(batchedPullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 1200.0f);
    vernonPullbackDestroy(batchedPullback);

    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExecutesContiguousTensorValuePullback) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_tensor_square_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_tensor_square_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, tensorSquareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, tensorSquareBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_tensor_profiles";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection = profileReflection(
        "forward", 8, 16, nlohmann::json::array({argument("values", 0, nlohmann::json::array({leaf(0, 2)}), 8)}),
        nlohmann::json::array({leaf(0, 2), leaf(8, 2)}));
    const nlohmann::json backwardReflection =
        profileReflection("backward", 16, 8,
                          nlohmann::json::array({argument("tape", 0, nlohmann::json::array({leaf(0, 2)}), 8),
                                                 argument("output", 8, nlohmann::json::array({leaf(0, 2)}), 8)}),
                          nlohmann::json::array({leaf(0, 2)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"values"});
    size_t outputRank{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputRank(&pipeline, &outputRank), VERNON_STATUS_OK);
    EXPECT_EQ(outputRank, 1u);
    uint64_t outputExtent{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDimension(&pipeline, 0, &outputExtent), VERNON_STATUS_OK);
    EXPECT_EQ(outputExtent, 2u);

    float values[]{2.0f, 3.0f};
    float outputsBuffer[2]{};
    const uint64_t tensorShape[]{2};
    VernonAdValue input{sizeof(VernonAdValue), {"values", 6}, VERNON_DATA_F32, values, sizeof(values), 1, tensorShape};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, outputsBuffer, sizeof(outputsBuffer), 1, tensorShape};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    const uint64_t wrongShape[]{1, 2};
    output.rank = 2;
    output.shape = wrongShape;
    EXPECT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pullback, nullptr);
    output.rank = 1;
    output.shape = tensorShape;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(outputsBuffer[0], 4.0f);
    EXPECT_FLOAT_EQ(outputsBuffer[1], 9.0f);

    float gradientsBuffer[2]{};
    VernonAdValue gradient{sizeof(VernonAdValue),
                           {"values", 6},
                           VERNON_DATA_F32,
                           gradientsBuffer,
                           sizeof(gradientsBuffer),
                           1,
                           tensorShape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_INVALID_ARGUMENT);

    float seedBuffer[]{1.0f, 2.0f};
    VernonAdValue seed{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, seedBuffer, sizeof(seedBuffer), 1, tensorShape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientsBuffer[0], 4.0f);
    EXPECT_FLOAT_EQ(gradientsBuffer[1], 12.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExecutesFlattenedAggregateInputLeaves) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_aggregate_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_aggregate_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, aggregateForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, aggregateBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_aggregate";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    nlohmann::json pairX = leaf(0);
    pairX["path"] = nlohmann::json::array({"x"});
    nlohmann::json pairY = leaf(4);
    pairY["path"] = nlohmann::json::array({"y"});
    const nlohmann::json forwardReflection =
        profileReflection("forward", 12, 12,
                          nlohmann::json::array({argument("pair", 0, nlohmann::json::array({pairX, pairY}), 8),
                                                 argument("scale", 8, nlohmann::json::array({leaf(0)}))}),
                          nlohmann::json::array({leaf(0), leaf(4), leaf(8)}));
    const nlohmann::json backwardReflection =
        profileReflection("backward", 12, 4,
                          nlohmann::json::array({argument("tape", 0, nlohmann::json::array({leaf(0), leaf(4)}), 8),
                                                 argument("output", 8, nlohmann::json::array({leaf(0)}))}),
                          nlohmann::json::array({leaf(0)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"scale"});

    float pairXValue = 2.0f;
    float pairYValue = 3.0f;
    float scale = 4.0f;
    VernonAdValue inputsBuffer[]{
        {sizeof(VernonAdValue), {"pair.x", 6}, VERNON_DATA_F32, &pairXValue, sizeof(pairXValue), 0, nullptr},
        {sizeof(VernonAdValue), {"pair.y", 6}, VERNON_DATA_F32, &pairYValue, sizeof(pairYValue), 0, nullptr},
        {sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &scale, sizeof(scale), 0, nullptr},
    };
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), inputsBuffer, std::size(inputsBuffer), {}};
    float outputValue = 0.0f;
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 0, nullptr};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(outputValue, 24.0f);

    float gradientValue = 0.0f;
    VernonAdValue gradient{
        sizeof(VernonAdValue), {"scale", 5}, VERNON_DATA_F32, &gradientValue, sizeof(gradientValue), 0, nullptr};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, ExposesMetadataFromResolvedSignature) {
    VernonRuntimeContext context;
    context.backend = VERNON_RUNTIME_VULKAN;
    VernonLoadedPipeline pipeline;
    pipeline.context = &context;
    vernon::runtime::ad::Signature signature;
    signature.output = {"output", VERNON_DATA_F32, 2 * sizeof(float), alignof(float), {2}};
    signature.cotangent = signature.output;
    signature.gradients.push_back({"values", VERNON_DATA_F32, 2 * sizeof(float), alignof(float), {2}});
    pipeline.autodiff = VernonLoadedAutodiff{std::make_shared<MetadataExecutable>(std::move(signature))};

    VernonDataType outputType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDataType(&pipeline, &outputType), VERNON_STATUS_OK);
    EXPECT_EQ(outputType, VERNON_DATA_F32);
    size_t rank{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputRank(&pipeline, &rank), VERNON_STATUS_OK);
    EXPECT_EQ(rank, 1u);
    uint64_t extent{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdOutputDimension(&pipeline, 0, &extent), VERNON_STATUS_OK);
    EXPECT_EQ(extent, 2u);
    VernonStringView path{};
    VernonDataType gradientType{};
    ASSERT_EQ(vernonRuntimeLoadedPipelineGetAdGradient(&pipeline, 0, &path, &gradientType), VERNON_STATUS_OK);
    EXPECT_EQ(std::string(path.data, path.size), "values");
    EXPECT_EQ(gradientType, VERNON_DATA_F32);
}

TEST(RuntimeAutodiff, OneElementTensorStillRequiresExplicitCotangent) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_one_tensor_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_one_tensor_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, squareForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, squareBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_one_tensor";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection = profileReflection(
        "forward", 4, 8, nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0, 1, "f32", true)}))}),
        nlohmann::json::array({leaf(0, 1, "f32", true), leaf(4, 1, "f32", true)}));
    const nlohmann::json backwardReflection = profileReflection(
        "backward", 8, 4,
        nlohmann::json::array({argument("tape", 0, nlohmann::json::array({leaf(0, 1, "f32", true)})),
                               argument("output", 4, nlohmann::json::array({leaf(0, 1, "f32", true)}))}),
        nlohmann::json::array({leaf(0, 1, "f32", true)}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    float x = 3.0f;
    float outputValue = 0.0f;
    const uint64_t tensorShape[]{1};
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &x, sizeof(x), 1, tensorShape};
    VernonAdValue output{
        sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F32, &outputValue, sizeof(outputValue), 1, tensorShape};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);

    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue,
                           sizeof(gradientValue), 1,        tensorShape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_INVALID_ARGUMENT);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

TEST(RuntimeAutodiff, PromotesF16ImplicitCotangentAndGradientToF32) {
    static constexpr char forwardSymbol[] = "__vernon_cpu_test_half_identity_forward";
    static constexpr char backwardSymbol[] = "__vernon_cpu_test_half_identity_backward";
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({forwardSymbol, sizeof(forwardSymbol) - 1}, halfIdentityForward),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeRegisterStaticCpuEntry({backwardSymbol, sizeof(backwardSymbol) - 1}, halfIdentityBackward),
              VERNON_STATUS_OK);

    const std::filesystem::path root = std::filesystem::temp_directory_path() / "vernon_runtime_autodiff_f16";
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "fixture.o", std::ios::binary);
        output.write("registered object fixture", 25);
    }
    const nlohmann::json forwardReflection = profileReflection(
        "forward", 2, 4, nlohmann::json::array({argument("x", 0, nlohmann::json::array({leaf(0, 1, "f16")}), 2, 2)}),
        nlohmann::json::array({leaf(0, 1, "f16"), leaf(2, 1, "f16")}), 2);
    const nlohmann::json backwardReflection =
        profileReflection("backward", 8, 4,
                          nlohmann::json::array({argument("tape", 0, nlohmann::json::array({leaf(0, 1, "f16")}), 2, 2),
                                                 argument("output", 4, nlohmann::json::array({leaf(0, 1, "f32")}))}),
                          nlohmann::json::array({leaf(0, 1, "f32")}));

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonLoadedPipeline pipeline;
    pipeline.context = context;
    attachCpuAutodiff(*context, pipeline, stage(artifact(root, "forward", forwardSymbol, forwardReflection)),
                      stage(artifact(root, "backward", backwardSymbol, backwardReflection)), {"x"});
    uint16_t x = 0x3c00;
    uint16_t outputValue = 0;
    VernonAdValue input{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F16, &x, sizeof(x), {}};
    VernonAdValue output{sizeof(VernonAdValue), {"output", 6}, VERNON_DATA_F16, &outputValue, sizeof(outputValue), {}};
    VernonAdValueSet inputs{sizeof(VernonAdValueSet), &input, 1, {}};
    VernonAdValueSet outputs{sizeof(VernonAdValueSet), &output, 1, {}};
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonAdPipelineForward(&pipeline, {1, 1, 1}, &inputs, &outputs, &pullback), VERNON_STATUS_OK)
        << lastError(context);

    float gradientValue = 0.0f;
    VernonAdValue gradient{sizeof(VernonAdValue), {"x", 1}, VERNON_DATA_F32, &gradientValue, sizeof(gradientValue), {}};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    EXPECT_EQ(vernonPullbackApply(pullback, nullptr, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 1.0f);

    vernonPullbackDestroy(pullback);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
    std::filesystem::remove_all(root);
}

} // namespace
