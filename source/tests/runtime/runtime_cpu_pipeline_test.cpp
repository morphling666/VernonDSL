#include "program_fixture_manifest_table.h"
#include "runtime/runtime_state.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
#ifndef VERNON_CPU_CANONICAL_PROGRAM_MANIFEST
#define VERNON_CPU_CANONICAL_PROGRAM_MANIFEST ""
#endif

#endif

namespace {

std::string readFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

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

VernonProgramBindingToken token(const char *value) {
    return {sizeof(VernonProgramBindingToken), value, std::strlen(value)};
}

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
struct CanonicalCpuProgram {
    VernonRuntimeContext *context{};
    VernonProgramBundle *bundle{};
    VernonProgramExecutable *pipeline{};
};

CanonicalCpuProgram loadCanonicalCpuProgram() {
    const auto *fixture = vernon::tests::findProgramFixtureManifest("module_program", VERNON_RUNTIME_CPU);
    EXPECT_NE(fixture, nullptr);
    EXPECT_EQ(fixture ? fixture->prepare() : VERNON_STATUS_INVALID_ARGUMENT, VERNON_STATUS_OK);
    const std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    EXPECT_FALSE(manifest.empty());
    CanonicalCpuProgram result;
    result.context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    EXPECT_NE(result.context, nullptr);
    result.bundle =
        vernonRuntimeLoadProgramBundleWithOptions(result.context, manifest.data(), manifest.size(), nullptr);
    EXPECT_NE(result.bundle, nullptr) << lastError(result.context);
    if (result.bundle)
        result.pipeline = vernonRuntimeResolveProgram(result.bundle, nullptr);
    EXPECT_NE(result.pipeline, nullptr) << lastError(result.context);
    return result;
}

void destroy(CanonicalCpuProgram &program) {
    vernonRuntimeProgramExecutableDestroy(program.pipeline);
    vernonRuntimeProgramBundleDestroy(program.bundle);
    EXPECT_EQ(vernonRuntimeDestroy(program.context), VERNON_STATUS_OK);
}
#endif

} // namespace

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
TEST(RuntimeCpuPipeline, LoadsValidatesAndInvokesBundles) {
    const auto *fixture = vernon::tests::findProgramFixtureManifest("module_program", VERNON_RUNTIME_CPU);
    ASSERT_NE(fixture, nullptr);
    ASSERT_EQ(fixture->prepare(), VERNON_STATUS_OK);
    const std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    VernonRuntimeBackend target = VERNON_RUNTIME_CUDA;
    ASSERT_EQ(vernonRuntimeProgramBundleInspectTarget(manifest.data(), manifest.size(), &target), VERNON_STATUS_OK);
    EXPECT_EQ(target, VERNON_RUNTIME_CPU);

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    EXPECT_EQ(context->livePipelines, 0u);
    VernonProgramExecutable *first = vernonRuntimeResolveProgram(bundle, nullptr);
    VernonProgramExecutable *second = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(first, nullptr) << lastError(context);
    ASSERT_NE(second, nullptr) << lastError(context);
    EXPECT_NE(first, second);
    EXPECT_EQ(context->livePipelines, 2u);

    const VernonProgramParameterView source = parameter(first, "source");
    const VernonProgramParameterView output = parameter(first, "output");
    float sourceValue = 3.0f;
    float outputValue = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(source, sourceValue);
    VernonProgramArgument outputArgument = tensorArgument(output, outputValue);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(first);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    const VernonProgramBindingToken sourceToken = token("source");
    const VernonProgramBindingToken outputToken = token("output");
    ASSERT_EQ(
        vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(sourceValue), 1),
        VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK) << lastError(context);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);

    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(second);
    vernonRuntimeProgramExecutableDestroy(first);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeCpuPipeline, RejectsLegacyExecutableTopology) {
    std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    ASSERT_FALSE(manifest.empty());
    manifest.insert(manifest.rfind('}'), R"(,"stage_artifacts":{"legacy":{"artifact":{"path":"../does-not-exist"}}})");
    const size_t typeField = manifest.find("\"type\"");
    const size_t type = manifest.find("\"program\"", typeField);
    ASSERT_NE(type, std::string::npos);
    manifest.replace(type, std::strlen("\"program\""), "\"pipeline\"");

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), nullptr), nullptr);
    EXPECT_EQ(lastError(context), "unsupported or invalid Program bundle");
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);
}

TEST(RuntimeCpuPipeline, ResolvesAndExecutesNativeBackwardProgramGraph) {
    CanonicalCpuProgram program = loadCanonicalCpuProgram();
    ASSERT_NE(program.pipeline, nullptr);
    EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(program.pipeline), 1u);
    const VernonProgramParameterView source = parameter(program.pipeline, "source");
    const VernonProgramParameterView output = parameter(program.pipeline, "output");
    float sourceValue = 3.0f;
    float outputValue = 0.0f;
    VernonProgramArgument sourceArgument = tensorArgument(source, sourceValue);
    VernonProgramArgument outputArgument = tensorArgument(output, outputValue);
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.pipeline);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    const VernonProgramBindingToken sourceToken = token("source-vjp");
    const VernonProgramBindingToken outputToken = token("output-vjp");
    ASSERT_EQ(
        vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(sourceValue), 1),
        VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    VernonPullback *pullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, &pullback), VERNON_STATUS_OK)
        << lastError(program.context);
    ASSERT_NE(pullback, nullptr);
    EXPECT_FLOAT_EQ(outputValue, 9.0f);

    float seedValue = 1.0f;
    float gradientValue = 0.0f;
    VernonProgramParameterView cotangent{};
    VernonProgramParameterView gradient{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(program.pipeline, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0,
                                                               &cotangent),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(program.pipeline, VERNON_PROGRAM_BOUNDARY_GRADIENT, 0,
                                                               &gradient),
              VERNON_STATUS_OK);
    VernonProgramArgument derivativeArguments[]{
        tensorArgument(cotangent, seedValue),
        tensorArgument(gradient, gradientValue),
    };
    ASSERT_EQ(vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments)),
              VERNON_STATUS_OK)
        << lastError(program.context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonProgramPullbackDestroy(pullback);
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    destroy(program);
}
#endif
