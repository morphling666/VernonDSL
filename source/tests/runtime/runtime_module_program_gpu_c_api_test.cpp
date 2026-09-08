#include "../support/runtime_rhi_test_utils.h"
#include "VernonRuntime.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

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

VernonProgramArgument tensorArgument(VernonRuntimeContext *context, VernonRhiBuffer buffer,
                                     const VernonProgramParameterView &parameter) {
    static const uint64_t shape[]{1};
    static const int64_t strides[]{sizeof(float)};
    VernonRuntimeProviderResourceReference resource{};
    EXPECT_EQ(vernonRuntimeReferenceRhiBuffer(context, buffer, 0, sizeof(float), &resource), VERNON_STATUS_OK);
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(VernonTensorView);
    result.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    result.tensor.resource = resource;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = 1;
    result.tensor.shape = shape;
    result.tensor.byte_strides = strides;
    result.tensor.byte_size = sizeof(float);
    return result;
}

VernonProgramBindingToken bindingToken(const char *value) {
    return {sizeof(VernonProgramBindingToken), value, std::strlen(value)};
}

void runModuleProgram(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    vernon::tests::OwnedRhiRuntime owned(backend);
    VernonRuntimeContext *context = owned.runtime();
    if (!context)
        GTEST_SKIP() << "GPU backend is unavailable";

    std::ifstream input(manifestPath, std::ios::binary);
    ASSERT_TRUE(input);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    VernonProgramBundle *bundle =
        vernonRuntimeLoadProgramBundleWithOptions(context, manifest.data(), manifest.size(), &options);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, {nullptr, 0});
    ASSERT_NE(pipeline, nullptr) << lastError(context);
    ASSERT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(pipeline), 1u);
    const auto &boundarySlots = pipeline->executionPlan->resolvedProgram->program.abi.boundarySlots;
    const auto outputBoundary = std::find_if(boundarySlots.begin(), boundarySlots.end(), [](const auto &slot) {
        return slot.path == "output" && slot.role == vernon::runtime::program::BoundaryRole::Output;
    });
    ASSERT_NE(outputBoundary, boundarySlots.end());
    EXPECT_EQ(outputBoundary->publication, vernon::runtime::program::BoundaryPublication::CommitAfterSuccess);

    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = sizeof(float);
    descriptor.alignment = alignof(float);
    descriptor.usage =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer sourceBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiBuffer outputBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiBuffer seedBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiBuffer gradientBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &sourceBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &outputBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &seedBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &gradientBuffer), VERNON_RHI_STATUS_OK);
    const float source = 3.0f;
    const float zero = 0.0f;
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), sourceBuffer, 0, &source, sizeof(source)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), outputBuffer, 0, &zero, sizeof(zero)), VERNON_RHI_STATUS_OK);
    const float seedValue = 1.0f;
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), seedBuffer, 0, &seedValue, sizeof(seedValue)),
              VERNON_RHI_STATUS_OK);

    const VernonProgramParameterView sourceParameter = parameter(pipeline, "source");
    const VernonProgramParameterView outputParameter = parameter(pipeline, "output");
    VernonProgramArgument sourceArgument = tensorArgument(context, sourceBuffer, sourceParameter);
    VernonProgramArgument outputArgument = tensorArgument(context, outputBuffer, outputParameter);
    const VernonProgramBindingToken sourceToken = bindingToken("gpu-source-v1");
    const VernonProgramBindingToken outputToken = bindingToken("gpu-output-v1");

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    VernonPullback *pullback = nullptr;
    vernon::runtime::program_execution::setFailureInjectionForTesting(
        vernon::runtime::program_execution::FailureBoundary::Submission);
    EXPECT_NE(vernonRuntimeProgramInvocationForward(invocation, &pullback), VERNON_STATUS_OK);
    vernon::runtime::program_execution::clearFailureInjectionForTesting();
    vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    float unpublished = -1.0f;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, &unpublished, sizeof(unpublished)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(unpublished, zero);

    invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    vernon::runtime::program_execution::setFailureInjectionForTesting(
        vernon::runtime::program_execution::FailureBoundary::Commit);
    EXPECT_NE(vernonRuntimeProgramInvocationForward(invocation, &pullback), VERNON_STATUS_OK);
    vernon::runtime::program_execution::clearFailureInjectionForTesting();
    vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, &unpublished, sizeof(unpublished)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(unpublished, zero);

    invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, &pullback), VERNON_STATUS_OK) << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    ASSERT_NE(pullback, nullptr);

    float output = 0.0f;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, &output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(output, 9.0f);

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
        tensorArgument(context, seedBuffer, cotangent),
        tensorArgument(context, gradientBuffer, gradient),
    };
    constexpr std::array failureBoundaries{
        vernon::runtime::program_execution::FailureBoundary::Planning,
        vernon::runtime::program_execution::FailureBoundary::Allocation,
        vernon::runtime::program_execution::FailureBoundary::Submission,
        vernon::runtime::program_execution::FailureBoundary::TapeValidation,
        vernon::runtime::program_execution::FailureBoundary::Readback,
        vernon::runtime::program_execution::FailureBoundary::Commit,
    };
    for (const auto boundary : failureBoundaries) {
        gradientValue = -31.0f;
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
                  VERNON_RHI_STATUS_OK);
        vernon::runtime::program_execution::setFailureInjectionForTesting(boundary);
        EXPECT_NE(vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments)),
                  VERNON_STATUS_OK)
            << static_cast<int>(boundary);
        vernon::runtime::program_execution::clearFailureInjectionForTesting();
        ASSERT_EQ(
            vernonRhiDeviceDownloadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
            VERNON_RHI_STATUS_OK);
        EXPECT_FLOAT_EQ(gradientValue, -31.0f);
    }
    gradientValue = 0.0f;
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments)),
              VERNON_STATUS_OK)
        << lastError(context);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonProgramPullbackDestroy(pullback);
    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(pipeline);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), gradientBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), seedBuffer), VERNON_RHI_STATUS_OK);
}

struct LoadedProgram {
    VernonProgramBundle *bundle{};
    VernonProgramExecutable *executable{};
};

LoadedProgram loadProgram(vernon::tests::OwnedRhiRuntime &owned, const std::filesystem::path &manifestPath) {
    std::ifstream input(manifestPath, std::ios::binary);
    if (!input)
        return {};
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::string bundleDirectory = manifestPath.parent_path().string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = bundleDirectory.c_str();
    LoadedProgram result;
    result.bundle =
        vernonRuntimeLoadProgramBundleWithOptions(owned.runtime(), manifest.data(), manifest.size(), &options);
    if (result.bundle)
        result.executable = vernonRuntimeResolveProgram(result.bundle, {nullptr, 0});
    return result;
}

VernonProgramArgument bufferArgument(VernonRuntimeContext *context, VernonRhiBuffer buffer,
                                     const VernonProgramParameterView &parameter, const uint64_t *shape,
                                     const int64_t *strides, uint32_t rank, size_t bytes) {
    VernonRuntimeProviderResourceReference resource{};
    EXPECT_EQ(vernonRuntimeReferenceRhiBuffer(context, buffer, 0, bytes, &resource), VERNON_STATUS_OK);
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(result.tensor);
    result.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    result.tensor.resource = resource;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.rank = rank;
    result.tensor.shape = shape;
    result.tensor.byte_strides = strides;
    result.tensor.byte_size = bytes;
    return result;
}

template <typename T>
VernonProgramArgument hostScalarArgument(const VernonProgramParameterView &parameter, const T *value) {
    VernonProgramArgument result{};
    result.slot = parameter.slot;
    result.kind = VERNON_PROGRAM_TENSOR;
    result.tensor.struct_size = sizeof(result.tensor);
    result.tensor.storage = VERNON_TENSOR_HOST;
    result.tensor.host_data = value;
    result.tensor.element_layout = parameter.element_layout;
    result.tensor.access = parameter.access;
    result.tensor.byte_size = sizeof(T);
    return result;
}

void destroyProgram(LoadedProgram &program) {
    if (program.executable)
        vernonRuntimeProgramExecutableDestroy(program.executable);
    if (program.bundle)
        vernonRuntimeProgramBundleDestroy(program.bundle);
    program = {};
}

void runReusedStageModule(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    vernon::tests::OwnedRhiRuntime owned(backend);
    if (!owned.runtime())
        GTEST_SKIP() << "GPU backend is unavailable";
    LoadedProgram program = loadProgram(owned, manifestPath);
    ASSERT_NE(program.bundle, nullptr) << lastError(owned.runtime());
    ASSERT_NE(program.executable, nullptr) << lastError(owned.runtime());

    using namespace vernon::runtime::program;
    const ResolvedNodePlan *first = program.executable->executionPlan->node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *second = program.executable->executionPlan->node(GraphDirection::Forward, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->stage.get(), second->stage.get());
    ASSERT_EQ(first->projections.size(), second->projections.size());
    ASSERT_GE(first->projections.size(), 2u);
    EXPECT_NE(first->projections[0].value, second->projections[0].value);
    EXPECT_NE(first->projections[1].value, second->projections[1].value);

    constexpr std::array<float, 4> source{1, 2, 3, 4};
    constexpr std::array<float, 4> zero{};
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = sizeof(source);
    descriptor.alignment = alignof(float);
    descriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer sourceBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiBuffer outputBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &sourceBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &outputBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), sourceBuffer, 0, source.data(), sizeof(source)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), outputBuffer, 0, zero.data(), sizeof(zero)),
              VERNON_RHI_STATUS_OK);

    constexpr uint64_t shape[]{4};
    constexpr int64_t strides[]{sizeof(float)};
    constexpr uint32_t grid[]{4, 1, 1};
    const VernonProgramParameterView sourceParameter = parameter(program.executable, "source");
    const VernonProgramParameterView outputParameter = parameter(program.executable, "output");
    std::array<VernonProgramArgument, 5> arguments{
        bufferArgument(owned.runtime(), sourceBuffer, sourceParameter, shape, strides, 1, sizeof(source)),
        bufferArgument(owned.runtime(), outputBuffer, outputParameter, shape, strides, 1, sizeof(source)),
        hostScalarArgument(parameter(program.executable, "grid_x"), &grid[0]),
        hostScalarArgument(parameter(program.executable, "grid_y"), &grid[1]),
        hostScalarArgument(parameter(program.executable, "grid_z"), &grid[2]),
    };
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.executable);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    for (size_t index = 0; index < arguments.size(); ++index) {
        const std::string text = "reused-stage-" + std::to_string(index);
        const VernonProgramBindingToken token{sizeof(token), text.data(), text.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &arguments[index], nullptr, 0, 0),
                  VERNON_STATUS_OK);
    }
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    std::array<float, 4> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, output.data(), sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, (std::array<float, 4>{3, 4, 5, 6}));

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
    destroyProgram(program);
}

void runTensorViewChainModule(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    vernon::tests::OwnedRhiRuntime owned(backend);
    if (!owned.runtime())
        GTEST_SKIP() << "GPU backend is unavailable";
    LoadedProgram program = loadProgram(owned, manifestPath);
    ASSERT_NE(program.bundle, nullptr) << lastError(owned.runtime());
    ASSERT_NE(program.executable, nullptr) << lastError(owned.runtime());

    using namespace vernon::runtime::program;
    const ResolvedNodePlan *producer = program.executable->executionPlan->node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *consumer = program.executable->executionPlan->node(GraphDirection::Forward, 1);
    ASSERT_NE(producer, nullptr);
    ASSERT_NE(consumer, nullptr);
    EXPECT_NE(producer->stage.get(), consumer->stage.get());
    const Graph *forward = findGraph(program.executable->executionPlan->resolvedProgram->program, "forward");
    ASSERT_NE(forward, nullptr);
    ASSERT_EQ(forward->nodes.size(), 2u);
    const auto intermediate = std::find_first_of(forward->nodes[0].results.begin(), forward->nodes[0].results.end(),
                                                 forward->nodes[1].operands.begin(), forward->nodes[1].operands.end());
    ASSERT_NE(intermediate, forward->nodes[0].results.end());
    ASSERT_LT(*intermediate, program.executable->executionPlan->resolvedProgram->program.values.size());
    EXPECT_TRUE(program.executable->executionPlan->resolvedProgram->program.values[*intermediate].storage.has_value());
    const std::vector<uint32_t> &predecessors =
        program.executable->executionPlan->predecessors(GraphDirection::Forward, 1);
    EXPECT_NE(std::find(predecessors.begin(), predecessors.end(), 0), predecessors.end());

    constexpr float source = 3.0f;
    constexpr float zero = 0.0f;
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = sizeof(float);
    descriptor.alignment = alignof(float);
    descriptor.usage =
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiBuffer sourceBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    VernonRhiBuffer outputBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &sourceBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &outputBuffer), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), sourceBuffer, 0, &source, sizeof(source)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), outputBuffer, 0, &zero, sizeof(zero)), VERNON_RHI_STATUS_OK);

    constexpr uint64_t shape[]{1};
    constexpr int64_t strides[]{sizeof(float)};
    std::array<VernonProgramArgument, 2> arguments{
        bufferArgument(owned.runtime(), sourceBuffer, parameter(program.executable, "source"), shape, strides, 1,
                       sizeof(source)),
        bufferArgument(owned.runtime(), outputBuffer, parameter(program.executable, "output"), shape, strides, 1,
                       sizeof(source)),
    };
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.executable);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    for (size_t index = 0; index < arguments.size(); ++index) {
        const std::string text = "tensor-view-chain-" + std::to_string(index);
        const VernonProgramBindingToken token{sizeof(token), text.data(), text.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &arguments[index], nullptr, 0, 0),
                  VERNON_STATUS_OK);
    }
    ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);

    float output = 0.0f;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, &output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(output, 8.0f);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
    destroyProgram(program);
}

void runDynamicShapeGridReuse(VernonRuntimeBackend backend, const std::filesystem::path &manifestPath) {
    vernon::tests::OwnedRhiRuntime owned(backend);
    if (!owned.runtime())
        GTEST_SKIP() << "GPU backend is unavailable";
    LoadedProgram program = loadProgram(owned, manifestPath);
    ASSERT_NE(program.bundle, nullptr) << lastError(owned.runtime());
    ASSERT_NE(program.executable, nullptr) << lastError(owned.runtime());
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.executable);
    ASSERT_NE(instance, nullptr);

    const VernonProgramParameterView sourceParameter = parameter(program.executable, "source");
    const VernonProgramParameterView outputParameter = parameter(program.executable, "output");
    VernonProgramParameterView publishedOutputParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(program.executable, VERNON_PROGRAM_BOUNDARY_OUTPUT, 0,
                                                               &publishedOutputParameter),
              VERNON_STATUS_OK);
    const VernonProgramParameterView factorParameter = parameter(program.executable, "factor");
    std::array<VernonProgramParameterView, 3> gridParameters{
        parameter(program.executable, "__grid_x"),
        parameter(program.executable, "__grid_y"),
        parameter(program.executable, "__grid_z"),
    };
    constexpr std::array<size_t, 2> counts{3, 5};
    constexpr std::array<float, 2> factors{2, 3};
    constexpr uint32_t one = 1;
    for (size_t iteration = 0; iteration < counts.size(); ++iteration) {
        const size_t count = counts[iteration];
        std::vector<float> source(count);
        for (size_t index = 0; index < count; ++index)
            source[index] = static_cast<float>(index + 1);
        std::vector<float> zero(count);
        const size_t bytes = count * sizeof(float);
        VernonRhiBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = bytes;
        descriptor.alignment = alignof(float);
        descriptor.usage =
            VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
        descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
        VernonRhiBuffer sourceBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        VernonRhiBuffer outputBuffer{VERNON_RHI_INVALID_HANDLE_INDEX, 0};
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &sourceBuffer), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceCreateBuffer(owned.device(), &descriptor, &outputBuffer), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), sourceBuffer, 0, source.data(), bytes),
                  VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), outputBuffer, 0, zero.data(), bytes),
                  VERNON_RHI_STATUS_OK);

        const uint64_t shape[]{count};
        const int64_t strides[]{sizeof(float)};
        const uint32_t gridX = static_cast<uint32_t>(count);
        std::array<VernonProgramArgument, 7> arguments{
            bufferArgument(owned.runtime(), sourceBuffer, sourceParameter, shape, strides, 1, bytes),
            bufferArgument(owned.runtime(), outputBuffer, outputParameter, shape, strides, 1, bytes),
            bufferArgument(owned.runtime(), outputBuffer, publishedOutputParameter, shape, strides, 1, bytes),
            hostScalarArgument(factorParameter, &factors[iteration]),
            hostScalarArgument(gridParameters[0], &gridX),
            hostScalarArgument(gridParameters[1], &one),
            hostScalarArgument(gridParameters[2], &one),
        };
        VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
        ASSERT_NE(invocation, nullptr);
        for (size_t index = 0; index < arguments.size(); ++index) {
            const std::string text = index >= 5 ? "dynamic-stable-" + std::to_string(index)
                                                : "dynamic-" + std::to_string(iteration) + "-" + std::to_string(index);
            const VernonProgramBindingToken token{sizeof(token), text.data(), text.size()};
            ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &arguments[index], nullptr, 0, 0),
                      VERNON_STATUS_OK);
        }
        ASSERT_EQ(vernonRuntimeProgramInvocationForward(invocation, nullptr), VERNON_STATUS_OK)
            << lastError(owned.runtime());
        vernonRuntimeProgramInvocationDestroy(invocation);
        std::vector<float> output(count);
        ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, output.data(), bytes),
                  VERNON_RHI_STATUS_OK);
        for (size_t index = 0; index < count; ++index)
            EXPECT_FLOAT_EQ(output[index], source[index] * factors[iteration]);
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
    }
    VernonProgramBindingTelemetry telemetry{};
    telemetry.struct_size = sizeof(telemetry);
    ASSERT_EQ(vernonRuntimeProgramInstanceGetTelemetry(instance, &telemetry), VERNON_STATUS_OK);
    EXPECT_EQ(telemetry.prepare_count, 12u);
    EXPECT_EQ(telemetry.reuse_count, 2u);
    vernonRuntimeProgramInstanceDestroy(instance);
    destroyProgram(program);
}

} // namespace

#if defined(VERNON_MODULE_PROGRAM_VULKAN_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, VulkanComputeModuleForward9AndVjpGradient6) {
    runModuleProgram(VERNON_RUNTIME_VULKAN, VERNON_MODULE_PROGRAM_VULKAN_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, VulkanReusesOneStageAcrossDifferentNodeProjections) {
    runReusedStageModule(VERNON_RUNTIME_VULKAN, VERNON_REUSED_STAGE_VULKAN_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, VulkanChainsDistinctComputeKernelsThroughTensorViews) {
    runTensorViewChainModule(VERNON_RUNTIME_VULKAN, VERNON_TENSOR_VIEW_CHAIN_VULKAN_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, VulkanReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    runDynamicShapeGridReuse(VERNON_RUNTIME_VULKAN, VERNON_DYNAMIC_SHAPE_GRID_VULKAN_MANIFEST);
}
#endif

#if defined(VERNON_MODULE_PROGRAM_CUDA_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, CudaComputeModuleForward9AndVjpGradient6) {
    runModuleProgram(VERNON_RUNTIME_CUDA, VERNON_MODULE_PROGRAM_CUDA_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, CudaReusesOneStageAcrossDifferentNodeProjections) {
    runReusedStageModule(VERNON_RUNTIME_CUDA, VERNON_REUSED_STAGE_CUDA_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, CudaChainsDistinctComputeKernelsThroughTensorViews) {
    runTensorViewChainModule(VERNON_RUNTIME_CUDA, VERNON_TENSOR_VIEW_CHAIN_CUDA_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, CudaReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    runDynamicShapeGridReuse(VERNON_RUNTIME_CUDA, VERNON_DYNAMIC_SHAPE_GRID_CUDA_MANIFEST);
}
#endif

#if defined(VERNON_MODULE_PROGRAM_DIRECTX_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, DirectX12ComputeModuleForward9AndVjpGradient6) {
    runModuleProgram(VERNON_RUNTIME_DIRECTX12, VERNON_MODULE_PROGRAM_DIRECTX_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, DirectX12ReusesOneStageAcrossDifferentNodeProjections) {
    runReusedStageModule(VERNON_RUNTIME_DIRECTX12, VERNON_REUSED_STAGE_DIRECTX_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, DirectX12ChainsDistinctComputeKernelsThroughTensorViews) {
    runTensorViewChainModule(VERNON_RUNTIME_DIRECTX12, VERNON_TENSOR_VIEW_CHAIN_DIRECTX_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, DirectX12ReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    runDynamicShapeGridReuse(VERNON_RUNTIME_DIRECTX12, VERNON_DYNAMIC_SHAPE_GRID_DIRECTX_MANIFEST);
}
#endif

#if defined(VERNON_MODULE_PROGRAM_METAL_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, MetalComputeModuleForward9AndVjpGradient6) {
    runModuleProgram(VERNON_RUNTIME_METAL, VERNON_MODULE_PROGRAM_METAL_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, MetalReusesOneStageAcrossDifferentNodeProjections) {
    runReusedStageModule(VERNON_RUNTIME_METAL, VERNON_REUSED_STAGE_METAL_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, MetalChainsDistinctComputeKernelsThroughTensorViews) {
    runTensorViewChainModule(VERNON_RUNTIME_METAL, VERNON_TENSOR_VIEW_CHAIN_METAL_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, MetalReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    runDynamicShapeGridReuse(VERNON_RUNTIME_METAL, VERNON_DYNAMIC_SHAPE_GRID_METAL_MANIFEST);
}
#endif

TEST(RuntimeModuleProgramGpuCApi, OpenGLComputeModuleForward9AndVjpGradient6) {
    runModuleProgram(VERNON_RUNTIME_OPENGL, VERNON_MODULE_PROGRAM_OPENGL_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, OpenGLReusesOneStageAcrossDifferentNodeProjections) {
    runReusedStageModule(VERNON_RUNTIME_OPENGL, VERNON_REUSED_STAGE_OPENGL_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, OpenGLChainsDistinctComputeKernelsThroughTensorViews) {
    runTensorViewChainModule(VERNON_RUNTIME_OPENGL, VERNON_TENSOR_VIEW_CHAIN_OPENGL_MANIFEST);
}
TEST(RuntimeModuleProgramGpuCApi, OpenGLReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    runDynamicShapeGridReuse(VERNON_RUNTIME_OPENGL, VERNON_DYNAMIC_SHAPE_GRID_OPENGL_MANIFEST);
}
