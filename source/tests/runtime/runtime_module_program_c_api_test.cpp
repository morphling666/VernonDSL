#include "VernonRuntime.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

extern "C" VernonStatus vernonRegisterModuleProgramFixture(void);
extern "C" VernonStatus vernonRegisterTypedSpecializationFixture(void);

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

VernonProgramArgument tensorArrayArgument(const VernonProgramParameterView &parameter, float *values,
                                          const uint64_t *shape) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = values;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = 1;
    result.tensor.shape = shape;
    static const int64_t stride = sizeof(float);
    result.tensor.byte_strides = &stride;
    result.tensor.byte_size = *shape * sizeof(float);
    return result;
}

VernonProgramArgument scalarArgument(const VernonProgramParameterView &parameter, uint32_t &value) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = &value;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.byte_size = sizeof(value);
    return result;
}

VernonProgramBindingToken bindingToken(const char *value) {
    return {sizeof(VernonProgramBindingToken), value, std::strlen(value)};
}

} // namespace

TEST(RuntimeModuleProgramCApi, ComputeModuleForward9AndVjpGradient6ThroughPublicLifecycle) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    const std::filesystem::path manifestPath = VERNON_MODULE_PROGRAM_MANIFEST;
    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetParameterCount(pipeline), 2u);
    ASSERT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline), 1u);

    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    const VernonProgramParameterView outputParameter = parameter(pipeline, "output");
    ASSERT_EQ(sourceParameter.kind, VERNON_PROGRAM_TENSOR);
    ASSERT_EQ(outputParameter.kind, VERNON_PROGRAM_TENSOR);

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
    VernonProgramParameterView cotangent{};
    VernonProgramParameterView gradient{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0, &cotangent),
        VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(pipeline, VERNON_PROGRAM_BOUNDARY_GRADIENT, 0, &gradient),
        VERNON_STATUS_OK);
    VernonProgramArgument derivativeArguments[]{
        tensorArgument(cotangent, seedValue),
        tensorArgument(gradient, gradientValue),
    };
    ASSERT_EQ(vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments)),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);
    vernonProgramPullbackDestroy(pullback);

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
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, ProgramGraphRetainsNodeLocalPullbackWithoutCompositeAutodiff) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *standalone = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(standalone, nullptr) << lastError(context);

    VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(context);
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId firstNode = UINT32_MAX;
    VernonProgramNodeId secondNode = UINT32_MAX;
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, &firstNode), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, &secondNode), VERNON_STATUS_OK);
    VernonProgramNodeBindingToken firstSource{};
    VernonProgramNodeBindingToken firstOutput{};
    VernonProgramNodeBindingToken secondSource{};
    VernonProgramNodeBindingToken secondOutput{};
    firstSource.struct_size = sizeof(firstSource);
    firstOutput.struct_size = sizeof(firstOutput);
    secondSource.struct_size = sizeof(secondSource);
    secondOutput.struct_size = sizeof(secondOutput);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, firstNode, VERNON_PROGRAM_BOUNDARY_INPUT,
                                                    {"source", std::strlen("source")}, &firstSource),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, firstNode, VERNON_PROGRAM_BOUNDARY_OUTPUT,
                                                    {"output", std::strlen("output")}, &firstOutput),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, secondNode, VERNON_PROGRAM_BOUNDARY_INPUT,
                                                    {"source", std::strlen("source")}, &secondSource),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, secondNode, VERNON_PROGRAM_BOUNDARY_OUTPUT,
                                                    {"output", std::strlen("output")}, &secondOutput),
              VERNON_STATUS_OK);
    VernonProgramGraphValue intermediate{sizeof(VernonProgramGraphValue)};
    ASSERT_EQ(vernonRuntimeProgramGraphCreateValue(graph, &firstOutput, &intermediate), VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphConnectValue(graph, &intermediate, &secondSource), VERNON_STATUS_OK);
    VernonProgramExecutable *composite = vernonRuntimeResolveProgramGraph(graph);
    ASSERT_NE(composite, nullptr) << lastError(context);
    EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(composite), 0u);
    EXPECT_EQ(vernonRuntimeProgramExecutableGetBoundaryCount(composite, VERNON_PROGRAM_BOUNDARY_COTANGENT), 0u);
    EXPECT_EQ(vernonRuntimeProgramExecutableGetBoundaryCount(composite, VERNON_PROGRAM_BOUNDARY_GRADIENT), 0u);

    float source = 3.0f;
    float output = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(parameter(standalone, "source"), source);
    VernonProgramArgument outputArgument = tensorArgument(parameter(standalone, "output"), output);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(composite);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBindNode(invocation, &firstSource, &sourceArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBindNode(invocation, &secondOutput, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    VernonPullback *compositePullback = reinterpret_cast<VernonPullback *>(uintptr_t{1});
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, &compositePullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(compositePullback, nullptr);
    EXPECT_FLOAT_EQ(output, 81.0f);

    VernonPullback *firstPullback = nullptr;
    VernonPullback *secondPullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation, firstNode, &firstPullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation, secondNode, &secondPullback), VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_NE(firstPullback, nullptr);
    ASSERT_NE(secondPullback, nullptr);
    vernonRuntimeProgramInvocationDestroy(invocation);

    VernonProgramParameterView cotangent{};
    VernonProgramParameterView gradient{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0, &cotangent),
        VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone, VERNON_PROGRAM_BOUNDARY_GRADIENT, 0, &gradient),
        VERNON_STATUS_OK);
    float seed = 1.0f;
    float intermediateCotangent = 0.0f;
    VernonProgramArgument secondDerivatives[]{
        tensorArgument(cotangent, seed),
        tensorArgument(gradient, intermediateCotangent),
    };
    ASSERT_EQ(vernonProgramPullbackApply(secondPullback, secondDerivatives, std::size(secondDerivatives)),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(intermediateCotangent, 18.0f);
    float result = 0.0f;
    VernonProgramArgument firstDerivatives[]{
        tensorArgument(cotangent, intermediateCotangent),
        tensorArgument(gradient, result),
    };
    ASSERT_EQ(vernonProgramPullbackApply(firstPullback, firstDerivatives, std::size(firstDerivatives)),
              VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_FLOAT_EQ(result, 108.0f);

    vernonProgramPullbackDestroy(secondPullback);
    vernonProgramPullbackDestroy(firstPullback);
    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(composite);
    vernonRuntimeProgramGraphDestroy(graph);
    vernonRuntimeProgramExecutableDestroy(standalone);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, LoadsCanonicalBundleThroughBundleThenResolve) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    const VernonStringView id = vernonRuntimeProgramBundleGetId(bundle);
    EXPECT_GT(id.size, 0u);

    const VernonProgramVariantSelector invalidSelector{};
    EXPECT_EQ(vernonRuntimeResolveProgram(bundle, &invalidSelector), nullptr);
    const std::string missingName = "NO_SUCH_SPECIALIZATION";
    VernonProgramSpecialization missing{};
    missing.struct_size = sizeof(missing);
    missing.name = {missingName.data(), missingName.size()};
    missing.kind = VERNON_PROGRAM_SPECIALIZATION_BOOL;
    missing.value.boolean_value = 1;
    const VernonProgramVariantSelector missingSelector{sizeof(VernonProgramVariantSelector), &missing, 1, {}};
    EXPECT_EQ(vernonRuntimeResolveProgram(bundle, &missingSelector), nullptr);
    EXPECT_NE(lastError(context).find("no matching variant"), std::string::npos);

    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    VernonProgramExecutable *second = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(second, nullptr) << lastError(context);
    EXPECT_NE(second, pipeline);
    vernonRuntimeProgramExecutableDestroy(second);
    EXPECT_EQ(vernonRuntimeProgramExecutableGetParameterCount(pipeline), 2u);

    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    const VernonProgramParameterView outputParameter = parameter(pipeline, "output");
    float source = 5.0f;
    float output = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(sourceParameter, source);
    VernonProgramArgument outputArgument = tensorArgument(outputParameter, output);
    const VernonProgramBindingToken sourceToken = bindingToken("resolve-source-v1");
    const VernonProgramBindingToken outputToken = bindingToken("resolve-output-v1");

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    EXPECT_FLOAT_EQ(output, 25.0f);

    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCApi, RejectsLegacyEnvelopeBeforeArtifactIo) {
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const size_t typeField = manifest.find("\"type\"");
    ASSERT_NE(typeField, std::string::npos);
    const size_t type = manifest.find("\"program\"", typeField);
    ASSERT_NE(type, std::string::npos);
    manifest.replace(type, std::strlen("\"program\""), "\"pipeline\"");
    manifest.insert(manifest.rfind('}'), R"(,"stage_artifacts":{"legacy":{"artifact":{"path":"../does-not-exist"}}})");

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr), nullptr);
    EXPECT_EQ(lastError(context), "unsupported or invalid Program bundle");
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

class RuntimeProgramEnvelopeRejection : public testing::TestWithParam<const char *> {};

TEST_P(RuntimeProgramEnvelopeRejection, RejectsEveryNonProgramTypeBeforeArtifactIo) {
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const size_t typeField = manifest.find("\"type\"");
    ASSERT_NE(typeField, std::string::npos);
    const size_t type = manifest.find("\"program\"", typeField);
    ASSERT_NE(type, std::string::npos);
    manifest.replace(type, std::strlen("\"program\""), std::string{"\""} + GetParam() + "\"");
    manifest.insert(manifest.rfind('}'), R"(,"stage_artifacts":{"legacy":{"artifact":{"path":"../does-not-exist"}}})");

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr), nullptr);
    EXPECT_EQ(lastError(context), "unsupported or invalid Program bundle");
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(RetiredAndUnknownTypes, RuntimeProgramEnvelopeRejection,
                         testing::Values("pipeline", "program_bundle", "", "Program", "program-v1", "program_v1",
                                         "bundle", "asset", "shader", "kernel", "module", "graph", "executable",
                                         "deployment", "stage", "compute", "graphics", "autodiff", "vjp",
                                         "pipeline_bundle", "cooked_pipeline", "cooked_program", "program_asset",
                                         "program_manifest", "vernon_program", "native_program", "runtime_program",
                                         "programs", "PROGRAM", " program", "program ", "null", "true", "unknown"));

TEST(RuntimeModuleProgramCppApi, RetainsExecutableForPersistentInstance) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const auto asset = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        auto executable = asset.resolve();
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

TEST(RuntimeModuleProgramCppApi, ProgramGraphProvidesNodeScopedFrameBindings) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const auto asset = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        auto standalone = asset.resolve();
        const VernonProgramParameterView sourceParameter = parameter(standalone.get(), "source");
        const VernonProgramParameterView outputParameter = parameter(standalone.get(), "output");
        VernonProgramParameterView cotangent{};
        VernonProgramParameterView gradient{};
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone.get(), VERNON_PROGRAM_BOUNDARY_COTANGENT,
                                                                   0, &cotangent),
                  VERNON_STATUS_OK);
        ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone.get(), VERNON_PROGRAM_BOUNDARY_GRADIENT,
                                                                   0, &gradient),
                  VERNON_STATUS_OK);
        vernon::runtime::ProgramGraph graph(context);
        const auto first = graph.add(asset);
        const auto second = graph.add(asset);
        const auto firstSource = first.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        const auto firstOutput = first.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        const auto secondSource = second.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        const auto secondOutput = second.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        auto executable = graph.compile();
        vernon::runtime::ProgramInstance instance(executable);
        float firstSourceValue = 2.0f;
        float firstOutputValue = 0.0f;
        float secondSourceValue = 3.0f;
        float secondOutputValue = 0.0f;
        auto invocation = instance.begin();
        invocation.node(first)
            .bind(firstSource, tensorArgument(sourceParameter, firstSourceValue))
            .bind(firstOutput, tensorArgument(outputParameter, firstOutputValue));
        invocation.node(second)
            .bind(secondSource, tensorArgument(sourceParameter, secondSourceValue))
            .bind(secondOutput, tensorArgument(outputParameter, secondOutputValue));
        EXPECT_FALSE(invocation.forward());
        EXPECT_FLOAT_EQ(firstOutputValue, 4.0f);
        EXPECT_FLOAT_EQ(secondOutputValue, 9.0f);
        vernon::runtime::ProgramGraph otherGraph(context);
        const auto foreignNode = otherGraph.add(asset);
        EXPECT_THROW(invocation.pullback(foreignNode), std::invalid_argument);
        auto firstPullback = invocation.pullback(first);
        auto secondPullback = invocation.pullback(second);
        float seed = 1.0f;
        float firstGradient = 0.0f;
        float secondGradient = 0.0f;
        VernonProgramArgument firstDerivatives[]{
            tensorArgument(cotangent, seed),
            tensorArgument(gradient, firstGradient),
        };
        VernonProgramArgument secondDerivatives[]{
            tensorArgument(cotangent, seed),
            tensorArgument(gradient, secondGradient),
        };
        firstPullback.apply(firstDerivatives, std::size(firstDerivatives));
        EXPECT_FLOAT_EQ(firstGradient, 4.0f);
        EXPECT_FLOAT_EQ(secondGradient, 0.0f);
        secondPullback.apply(secondDerivatives, std::size(secondDerivatives));
        EXPECT_FLOAT_EQ(firstGradient, 4.0f);
        EXPECT_FLOAT_EQ(secondGradient, 6.0f);
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, TypedVariantsSelectIndependentProgramGraphNodes) {
    ASSERT_EQ(vernonRegisterTypedSpecializationFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_TYPED_SPECIALIZATION_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const auto asset = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        vernon::runtime::ProgramVariant one;
        one.set("extent", uint32_t{1});
        vernon::runtime::ProgramVariant four;
        four.set("extent", uint32_t{4});
        const auto oneExecutable = asset.resolve(one);
        const auto fourExecutable = asset.resolve(four);
        const VernonProgramParameterView oneOutput = parameter(oneExecutable.get(), "output");
        const VernonProgramParameterView fourOutput = parameter(fourExecutable.get(), "output");
        ASSERT_EQ(oneOutput.rank, 1u);
        ASSERT_EQ(fourOutput.rank, 1u);
        EXPECT_EQ(oneOutput.static_shape[0], 1);
        EXPECT_EQ(fourOutput.static_shape[0], 4);

        vernon::runtime::ProgramGraph graph(context);
        const auto oneNode = graph.add(asset, one);
        const auto fourNode = graph.add(asset, four);
        const auto oneBoundary = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        const auto fourBoundary = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        const auto oneGridX = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_x");
        const auto oneGridY = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_y");
        const auto oneGridZ = oneNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_z");
        const auto fourGridX = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_x");
        const auto fourGridY = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_y");
        const auto fourGridZ = fourNode.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "__grid_z");
        auto graphExecutable = graph.compile();
        vernon::runtime::ProgramInstance instance(graphExecutable);
        float oneValues[1]{};
        float fourValues[4]{};
        const uint64_t oneShape[]{1};
        const uint64_t fourShape[]{4};
        uint32_t gridExtent = 1;
        auto invocation = instance.begin();
        invocation.node(oneNode).bind(oneBoundary, tensorArrayArgument(oneOutput, oneValues, oneShape));
        invocation.node(fourNode).bind(fourBoundary, tensorArrayArgument(fourOutput, fourValues, fourShape));
        invocation.node(oneNode)
            .bind(oneGridX, scalarArgument(parameter(oneExecutable.get(), "__grid_x"), gridExtent))
            .bind(oneGridY, scalarArgument(parameter(oneExecutable.get(), "__grid_y"), gridExtent))
            .bind(oneGridZ, scalarArgument(parameter(oneExecutable.get(), "__grid_z"), gridExtent));
        invocation.node(fourNode)
            .bind(fourGridX, scalarArgument(parameter(fourExecutable.get(), "__grid_x"), gridExtent))
            .bind(fourGridY, scalarArgument(parameter(fourExecutable.get(), "__grid_y"), gridExtent))
            .bind(fourGridZ, scalarArgument(parameter(fourExecutable.get(), "__grid_z"), gridExtent));
        EXPECT_FALSE(invocation.forward(false));
        EXPECT_FLOAT_EQ(oneValues[0], 1.0f);
        EXPECT_FLOAT_EQ(fourValues[0], 4.0f);
        VernonPullback *pullback = nullptr;
        EXPECT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation.get(), oneNode.id(), &pullback),
                  VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(pullback, nullptr);
        EXPECT_EQ(lastError(context), "ProgramGraph node is not differentiable");
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeModuleProgramCppApi, ProgramGraphValueConnectsProducerToConsumer) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    std::ifstream input(VERNON_MODULE_PROGRAM_MANIFEST, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    {
        const auto asset = vernon::runtime::ProgramAsset::load(context, manifest.data(), manifest.size());
        auto standalone = asset.resolve();
        const VernonProgramParameterView sourceParameter = parameter(standalone.get(), "source");
        const VernonProgramParameterView outputParameter = parameter(standalone.get(), "output");
        vernon::runtime::ProgramGraph graph(context);
        const auto producer = graph.add(asset);
        const auto consumer = graph.add(asset);
        const auto producerSource = producer.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        const auto producerOutput = producer.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        const auto consumerSource = consumer.boundary(VERNON_PROGRAM_BOUNDARY_INPUT, "source");
        const auto consumerOutput = consumer.boundary(VERNON_PROGRAM_BOUNDARY_OUTPUT, "output");
        graph.connect(graph.createValue(producerOutput), consumerSource);
        auto executable = graph.compile();
        vernon::runtime::ProgramInstance instance(executable);
        float source = 2.0f;
        float output = 0.0f;
        auto invocation = instance.begin();
        invocation.node(producer).bind(producerSource, tensorArgument(sourceParameter, source));
        invocation.node(consumer).bind(consumerOutput, tensorArgument(outputParameter, output));
        EXPECT_FALSE(invocation.forward(false));
        EXPECT_FLOAT_EQ(output, 16.0f);
        VernonPullback *pullback = nullptr;
        EXPECT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation.get(), producer.id(), &pullback),
                  VERNON_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(pullback, nullptr);
        EXPECT_EQ(lastError(context), "ProgramGraph node pullback is unavailable");
    }
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}
