#include "VernonCpuWorkgroupABI.h"
#include "runtime/backend_stage_pipeline.h"
#include "runtime/runtime_state.h"

#include <nlohmann/json.hpp>

#include <atomic>
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

extern "C" VernonStatus vernonRegisterModuleProgramFixture(void);
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
    EXPECT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
    const std::string manifest = readFile(VERNON_CPU_CANONICAL_PROGRAM_MANIFEST);
    EXPECT_FALSE(manifest.empty());
    CanonicalCpuProgram result;
    result.context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    EXPECT_NE(result.context, nullptr);
    result.bundle =
        vernonRuntimeLoadProgramBundleWithOptions(result.context, manifest.data(), manifest.size(), nullptr);
    EXPECT_NE(result.bundle, nullptr) << lastError(result.context);
    if (result.bundle)
        result.pipeline = vernonRuntimeResolveProgram(result.bundle, {nullptr, 0});
    EXPECT_NE(result.pipeline, nullptr) << lastError(result.context);
    return result;
}

void destroy(CanonicalCpuProgram &program) {
    vernonRuntimeProgramExecutableDestroy(program.pipeline);
    vernonRuntimeProgramBundleDestroy(program.bundle);
    EXPECT_EQ(vernonRuntimeDestroy(program.context), VERNON_STATUS_OK);
}
#endif

std::atomic<uint32_t> staticInvocationCount{};

VernonStatus staticallyLinkedFill(const VernonCpuInvocation *invocation) {
    if (!invocation || invocation->arguments_size != VERNON_CPU_RANGE_ARGUMENTS_SIZE_V1)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto *range = reinterpret_cast<VernonCpuRangeV1 *>(const_cast<void *>(invocation->arguments));
    if (!range || range->struct_size != sizeof(*range))
        return VERNON_STATUS_INVALID_ARGUMENT;
    ++staticInvocationCount;
    return VERNON_STATUS_OK;
}

constexpr char kDirectCpuReflection[] =
    "{" VERNON_JSON_VERSION_FIELDS ",\"entries\":[{\"name\":\"fill\","
    "\"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\",\"packed_arguments_size\":12}},"
    "\"workgroup_size\":[1,1,1],"
    "\"dispatch_contract\":{\"unit_grid_axes\":[],\"requires_unit_workgroup\":false},"
    "\"arguments\":[{\"kind\":\"builtin\",\"builtin\":\"global_invocation_id\","
    "\"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\",\"kind\":\"cpu_call\","
    "\"frame_offset\":0,\"root\":{\"kind\":\"array\",\"offset\":0,\"size\":12,\"alignment\":4,"
    "\"shape\":[3],\"byte_strides\":[4],\"children\":[{\"kind\":\"scalar\","
    "\"representation\":\"i32\",\"offset\":0,\"size\":4,\"alignment\":4}]}}},\"index\":0}]}]}";

} // namespace

TEST(RuntimeCpuPipeline, LoadsAndInvokesDirectCpuEntry) {
    staticInvocationCount = 0;
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    VernonStageExecutable *pipeline = vernonRuntimeLoadCpuEntry(runtime, staticallyLinkedFill, kDirectCpuReflection,
                                                                sizeof(kDirectCpuReflection) - 1, "fill", 4);
    ASSERT_NE(pipeline, nullptr) << lastError(runtime);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.compute_grid = {2, 3, 4};
    VernonSubmission *submission = nullptr;
    ASSERT_EQ(vernonRuntimeStageSubmit(pipeline, &invocation, &submission), VERNON_STATUS_OK) << lastError(runtime);
    ASSERT_NE(submission, nullptr) << lastError(runtime);
    EXPECT_EQ(vernonSubmissionWait(submission), VERNON_STATUS_OK);
    vernonSubmissionDestroy(submission);
    EXPECT_EQ(staticInvocationCount.load(), 24u);
    vernonRuntimeStageExecutableDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}

#if !defined(VERNON_RUNTIME_PROFILE_WEB)
TEST(RuntimeCpuPipeline, ReflectsImageConstraintsAndRejectsLegacyMetadata) {
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    VernonStageExecutable pipeline;
    pipeline.context = runtime;
    vernon::runtime::Parameter tensor;
    tensor.slot = 0;
    tensor.name = "values";
    tensor.kind = "tensor";
    pipeline.bindingProjection.parameters.push_back(tensor);
    vernon::runtime::Parameter image;
    image.slot = 1;
    image.name = "output";
    image.kind = "image";
    image.dimension = "3d";
    image.bindingRole = "sampled";
    image.sampleResultClass = "float";
    image.access = "read";
    pipeline.bindingProjection.parameters.push_back(image);

    VernonProgramImageConstraintView constraint{};
    constraint.struct_size = sizeof(constraint);
    EXPECT_EQ(vernonRuntimeStageExecutableGetImageConstraintByParameterIndex(&pipeline, 0, &constraint),
              VERNON_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRuntimeStageExecutableGetImageConstraintByParameterIndex(&pipeline, 1, &constraint),
              VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);
    EXPECT_EQ(constraint.binding_role, VERNON_IMAGE_BINDING_SAMPLED);
    ASSERT_EQ(
        vernonRuntimeStageExecutableFindImageConstraint(&pipeline, {"output", std::strlen("output")}, &constraint),
        VERNON_STATUS_OK);
    EXPECT_EQ(constraint.dimension, VERNON_TEXTURE_3D);

    constexpr char legacyReflection[] = "{" VERNON_JSON_VERSION_FIELDS ",\"legacy\":true,\"entries\":[]}";
    EXPECT_EQ(vernonRuntimeLoadCpuEntry(runtime, staticallyLinkedFill, legacyReflection, sizeof(legacyReflection) - 1,
                                        "fill", 4),
              nullptr);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}

TEST(RuntimeCpuPipeline, LoadsValidatesAndInvokesBundles) {
    ASSERT_EQ(vernonRegisterModuleProgramFixture(), VERNON_STATUS_OK);
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
    VernonProgramExecutable *first = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    VernonProgramExecutable *second = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
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

    const uint64_t shape[]{1};
    float seedValue = 1.0f;
    float gradientValue = 0.0f;
    VernonAdValue seed{sizeof(VernonAdValue),
                       {"output", std::strlen("output")},
                       VERNON_DATA_F32,
                       &seedValue,
                       sizeof(seedValue),
                       1,
                       shape};
    VernonAdValue gradient{sizeof(VernonAdValue),
                           {"source", std::strlen("source")},
                           VERNON_DATA_F32,
                           &gradientValue,
                           sizeof(gradientValue),
                           1,
                           shape};
    VernonAdValueSet seeds{sizeof(VernonAdValueSet), &seed, 1, {}};
    VernonAdValueSet gradients{sizeof(VernonAdValueSet), &gradient, 1, {}};
    ASSERT_EQ(vernonPullbackApply(pullback, &seeds, &gradients), VERNON_STATUS_OK) << lastError(program.context);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonPullbackDestroy(pullback);
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    destroy(program);
}
#endif

#if defined(VERNON_RUNTIME_PROFILE_WEB)
TEST(RuntimeCpuPipeline, WebProfileLoadsMultipleStaticPipelinesWithoutFilesystem) {
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    VernonStageExecutable *first = vernonRuntimeLoadCpuEntry(runtime, staticallyLinkedFill, kDirectCpuReflection,
                                                             sizeof(kDirectCpuReflection) - 1, "fill", 4);
    VernonStageExecutable *second = vernonRuntimeLoadCpuEntry(runtime, staticallyLinkedFill, kDirectCpuReflection,
                                                              sizeof(kDirectCpuReflection) - 1, "fill", 4);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first, second);
    EXPECT_EQ(runtime->livePipelines, 2u);
    vernonRuntimeStageExecutableDestroy(second);
    vernonRuntimeStageExecutableDestroy(first);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
}
#endif
