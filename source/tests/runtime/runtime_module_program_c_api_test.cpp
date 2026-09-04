#include "VernonRuntime.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

extern "C" VernonStatus vernonRegisterModuleProgramFixture(void);

namespace {

std::string lastError(VernonRuntimeContext *context) {
    const VernonStringView error = vernonRuntimeGetLastError(context);
    return std::string(error.data ? error.data : "", error.size);
}

VernonProgramParameterView parameter(VernonProgramExecutable *pipeline, const char *name) {
    VernonProgramParameterView result{};
    EXPECT_EQ(vernonRuntimeProgramExecutableFindParameter(pipeline, {name, std::strlen(name)}, &result),
              VERNON_STATUS_OK);
    return result;
}

VernonProgramArgument tensorArgument(const VernonProgramParameterView &parameter, float &value) {
    static const uint64_t shape[]{1};
    static const int64_t strides[]{sizeof(float)};
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = &value;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = 1;
    result.tensor.shape = shape;
    result.tensor.byte_strides = strides;
    result.tensor.byte_size = sizeof(value);
    return result;
}

VernonProgramBindingToken bindingToken(const char *value) {
    return {sizeof(VernonProgramBindingToken), value, std::strlen(value)};
}

} // namespace

TEST(RuntimeModuleProgramCApi, LoadsLinkedBundleAndExecutesPersistentForwardAndVjp) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_MODULE_PROGRAM_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonExecutableBundleKind bundleKind{};
    ASSERT_EQ(vernonRuntimeExecutableBundleInspectKind(manifest.data(), manifest.size(), &bundleKind),
              VERNON_STATUS_OK);
    ASSERT_EQ(bundleKind, VERNON_EXECUTABLE_BUNDLE_PROGRAM);

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramExecutable *pipeline = vernonRuntimeLoadManagedProgramBundleWithOptions(
        context, manifest.data(), manifest.size(), {nullptr, 0}, nullptr);
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(pipeline), 2u);
    ASSERT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline), 1u);

    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    const VernonProgramParameterView outputParameter = parameter(pipeline, "output");
    ASSERT_EQ(sourceParameter.kind, VERNON_PROGRAM_TENSOR);
    ASSERT_EQ(outputParameter.kind, VERNON_PROGRAM_TENSOR);

    VernonAdValueSet emptyValues{sizeof(VernonAdValueSet), nullptr, 0, {}};
    VernonPullback *legacyPullback = nullptr;
    EXPECT_EQ(vernonAdProgramForward(pipeline, {1, 1, 1}, &emptyValues, &emptyValues, &legacyPullback),
              VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(legacyPullback, nullptr);
    EXPECT_NE(lastError(context).find("vernonRuntimeProgramForward"), std::string::npos);

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    float source = 3.0f;
    float output = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
    VernonProgramArgument outputArgument = tensorArgument(outputParameter, output);
    const VernonProgramBindingToken sourceToken = bindingToken("source-v1");
    const VernonProgramBindingToken outputToken = bindingToken("output-v1");
    const VernonProgramBindingToken renderPassToken = bindingToken("render-pass-v1");
    const VernonProgramBindingToken dynamicStateToken = bindingToken("dynamic-state-v1");
    int controlLeaseCount = 0;
    VernonProgramResourceLease controlLease{
        sizeof(VernonProgramResourceLease),
        &controlLeaseCount,
        [](void *value) { ++*static_cast<int *>(value); },
        [](void *value) { --*static_cast<int *>(value); },
    };
    VernonRenderPass renderPass{};
    renderPass.struct_size = sizeof(renderPass);
    VernonDynamicState dynamicState{};
    dynamicState.struct_size = sizeof(dynamicState);
    dynamicState.viewport[2] = 1;
    dynamicState.viewport[3] = 1;

    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramInvocationBindRenderPass(invocation, 0, &renderPassToken, &renderPass, &controlLease, 1),
        VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBindDynamicState(invocation, 0, &dynamicStateToken, &dynamicState),
              VERNON_STATUS_OK);
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, &pullback), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(output, 9.0f);

    float seedValue = 1.0f;
    float gradientValue = 0.0f;
    const uint64_t shape[]{1};
    VernonAdValue seed{sizeof(VernonAdValue),
                       {"output", std::strlen("output")},
                       VERNON_DATA_F32,
                       &seedValue,
                       sizeof(seedValue),
                       1,
                       shape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    VernonAdValue gradient{sizeof(VernonAdValue),
                           {"source", std::strlen("source")},
                           VERNON_DATA_F32,
                           &gradientValue,
                           sizeof(gradientValue),
                           1,
                           shape};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);
    vernonPullbackDestroy(pullback);

    invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_FLOAT_EQ(output, 9.0f);

    const VernonProgramBindingToken changedRenderPassToken = bindingToken("render-pass-v2");
    invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBindRenderPass(invocation, 0, &changedRenderPassToken, &renderPass,
                                                           &controlLease, 1),
              VERNON_STATUS_OK);
    EXPECT_EQ(controlLeaseCount, 2);
    vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_EQ(controlLeaseCount, 1);

    VernonProgramBindingTelemetry telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    ASSERT_EQ(vernonRuntimeProgramInstanceGetTelemetry(instance, &telemetry), VERNON_STATUS_OK);
    EXPECT_EQ(telemetry.prepare_count, 2u);
    EXPECT_EQ(telemetry.reuse_count, 2u);
    EXPECT_EQ(telemetry.upload_bytes, sizeof(source));
    EXPECT_EQ(telemetry.upload_ranges, 1u);
    EXPECT_EQ(controlLeaseCount, 1);

    vernonRuntimeProgramInstanceDestroy(instance);
    EXPECT_EQ(controlLeaseCount, 0);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, RetainsExecutableForPersistentInstance) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        auto executable = vernon::runtime::ProgramExecutable::load(context, manifest.data(), manifest.size());
        const VernonProgramParameterView sourceParameter = parameter(executable.get(), "source");
        const VernonProgramParameterView outputParameter = parameter(executable.get(), "output");
        vernon::runtime::ProgramInstance instance(executable);
        executable = {};

        float source = 4.0f;
        float output = 0.0f;
        const VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
        const VernonProgramArgument outputArgument = tensorArgument(outputParameter, output);
        const VernonProgramBindingToken sourceToken = bindingToken("cpp-source-v1");
        const VernonProgramBindingToken outputToken = bindingToken("cpp-output-v1");
        auto invocation = instance.begin();
        invocation.bind(sourceToken, sourceArgument, nullptr, sizeof(source), 1).bind(outputToken, outputArgument);
        EXPECT_FALSE(invocation.forward(false));
        EXPECT_FLOAT_EQ(output, 16.0f);
        const VernonProgramBindingTelemetry telemetry = instance.telemetry();
        EXPECT_EQ(telemetry.prepare_count, 2u);
        EXPECT_EQ(telemetry.upload_bytes, sizeof(source));
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
