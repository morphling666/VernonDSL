#include "VernonRuntime.h"
#include "program_fixture_runtime_test.h"
#include "rhi/rhi_internal.h"
#include "runtime/program_execution/failure_injection.h"
#include "runtime/resolved_execution_plan.h"
#include "runtime/runtime_state.h"
#include "runtime_rhi_test_utils.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

void runModuleProgram(vernon::tests::OwnedRhiRuntime &owned, const std::filesystem::path &manifestPath) {
    VernonRuntimeContext *context = owned.runtime();

    VernonProgramBundle *bundle = vernon::tests::loadProgramBundle(context, manifestPath);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *pipeline = vernonRuntimeResolveProgram(bundle, nullptr);
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
    const VernonProgramBindingToken sourceToken = bindingToken("gpu-source-main");
    const VernonProgramBindingToken outputToken = bindingToken("gpu-output-main");
    std::array<VernonBoundaryMutation, 8> mutationRecords{};
    VernonInvocationMutationOutcome mutationOutcome{
        sizeof(VernonInvocationMutationOutcome),
        VERNON_INVOCATION_NOT_SUBMITTED,
        mutationRecords.data(),
        mutationRecords.size(),
        0,
        {},
    };
    const auto resetOutcome = [&] {
        mutationOutcome.submission = VERNON_INVOCATION_NOT_SUBMITTED;
        mutationOutcome.mutation_count = 0;
    };
    const auto mutationState = [&](uint32_t slot) {
        const auto found = std::find_if(
            mutationRecords.begin(), mutationRecords.begin() + static_cast<ptrdiff_t>(mutationOutcome.mutation_count),
            [&](const VernonBoundaryMutation &mutation) {
                return mutation.kind == VERNON_MUTATION_PROGRAM_BOUNDARY && mutation.slot == slot;
            });
        return found == mutationRecords.begin() + static_cast<ptrdiff_t>(mutationOutcome.mutation_count)
                   ? VERNON_BOUNDARY_MUTATION_UNCHANGED
                   : found->state;
    };

    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *asynchronous = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(asynchronous, nullptr);
    ASSERT_EQ(
        vernonRuntimeProgramInvocationBind(asynchronous, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
        VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(asynchronous, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    const VernonProgramInvocationOptions asynchronousOptions{sizeof(VernonProgramInvocationOptions),
                                                             VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION,
                                                             VERNON_COMPLETION_ASYNCHRONOUS,
                                                             0,
                                                             {}};
    ASSERT_EQ(vernonRuntimeProgramInvocationSetOptions(asynchronous, &asynchronousOptions), VERNON_STATUS_OK);
    resetOutcome();
    EXPECT_EQ(vernonRuntimeProgramInvocationExecute(asynchronous, 1, &mutationOutcome), VERNON_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(lastError(context), "PROGRAM_SUBMISSION_REQUIRES_WAIT at /abi/boundary_slots/" +
                                      std::to_string(outputParameter.slot) +
                                      "/publication: a commit-after-success boundary is published after a host wait");
    EXPECT_EQ(mutationOutcome.submission, VERNON_INVOCATION_NOT_SUBMITTED);
    EXPECT_EQ(mutationState(outputParameter.slot), VERNON_BOUNDARY_MUTATION_UNCHANGED);
    VernonSubmission *submission = nullptr;
    EXPECT_NE(vernonRuntimeProgramInvocationTakeSubmission(asynchronous, &submission), VERNON_STATUS_OK);
    EXPECT_EQ(submission, nullptr);
    vernonRuntimeProgramInvocationRollback(asynchronous);
    vernonRuntimeProgramInvocationDestroy(asynchronous);

    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceToken, &sourceArgument, nullptr, sizeof(source), 1),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputToken, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    VernonPullback *pullback = nullptr;
    vernon::runtime::program_execution::setFailureInjectionForTesting(
        vernon::runtime::program_execution::FailureBoundary::Submission);
    resetOutcome();
    EXPECT_NE(vernonRuntimeProgramInvocationExecute(invocation, 1, &mutationOutcome), VERNON_STATUS_OK);
    EXPECT_EQ(mutationOutcome.submission, VERNON_INVOCATION_NOT_SUBMITTED);
    EXPECT_EQ(mutationState(outputParameter.slot), VERNON_BOUNDARY_MUTATION_UNCHANGED);
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
        vernon::runtime::program_execution::FailureBoundary::Completion);
    resetOutcome();
    EXPECT_NE(vernonRuntimeProgramInvocationExecute(invocation, 1, &mutationOutcome), VERNON_STATUS_OK);
    EXPECT_EQ(mutationOutcome.submission, VERNON_INVOCATION_INDETERMINATE);
    EXPECT_EQ(mutationState(outputParameter.slot), VERNON_BOUNDARY_MUTATION_UNCHANGED);
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
    vernon::runtime::program_execution::setFailureInjectionForTesting(
        vernon::runtime::program_execution::FailureBoundary::Commit);
    resetOutcome();
    EXPECT_NE(vernonRuntimeProgramInvocationExecute(invocation, 1, &mutationOutcome), VERNON_STATUS_OK);
    EXPECT_EQ(mutationOutcome.submission, VERNON_INVOCATION_COMPLETED);
    EXPECT_EQ(mutationState(outputParameter.slot), VERNON_BOUNDARY_MUTATION_UNCHANGED);
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
    resetOutcome();
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 1, &mutationOutcome), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(pullback, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, &pullback), VERNON_STATUS_OK) << lastError(context);
    EXPECT_EQ(mutationOutcome.submission, VERNON_INVOCATION_COMPLETED);
    EXPECT_EQ(mutationState(outputParameter.slot), VERNON_BOUNDARY_MUTATION_COMMITTED);
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
        vernon::runtime::program_execution::FailureBoundary::Completion,
        vernon::runtime::program_execution::FailureBoundary::TapeValidation,
        vernon::runtime::program_execution::FailureBoundary::Readback,
        vernon::runtime::program_execution::FailureBoundary::Commit,
    };
    for (const auto boundary : failureBoundaries) {
        gradientValue = -31.0f;
        ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
                  VERNON_RHI_STATUS_OK);
        vernon::runtime::program_execution::setFailureInjectionForTesting(boundary);
        resetOutcome();
        EXPECT_NE(
            vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments), &mutationOutcome),
            VERNON_STATUS_OK)
            << static_cast<int>(boundary);
        EXPECT_EQ(mutationState(gradient.slot), VERNON_BOUNDARY_MUTATION_UNCHANGED);
        vernon::runtime::program_execution::clearFailureInjectionForTesting();
        ASSERT_EQ(
            vernonRhiDeviceDownloadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
            VERNON_RHI_STATUS_OK);
        EXPECT_FLOAT_EQ(gradientValue, -31.0f);
    }
    gradientValue = 0.0f;
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
              VERNON_RHI_STATUS_OK);
    resetOutcome();
    ASSERT_EQ(
        vernonProgramPullbackApply(pullback, derivativeArguments, std::size(derivativeArguments), &mutationOutcome),
        VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(mutationState(gradient.slot), VERNON_BOUNDARY_MUTATION_COMMITTED);
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

void runProgramGraphNodePullback(vernon::tests::OwnedRhiRuntime &owned, const std::filesystem::path &manifestPath) {
    VernonRuntimeContext *context = owned.runtime();

    VernonProgramBundle *bundle = vernon::tests::loadProgramBundle(context, manifestPath);
    ASSERT_NE(bundle, nullptr) << lastError(context);
    VernonProgramExecutable *standalone = vernonRuntimeResolveProgram(bundle, nullptr);
    ASSERT_NE(standalone, nullptr) << lastError(context);

    VernonProgramGraph *graph = vernonRuntimeProgramGraphCreate(context);
    ASSERT_NE(graph, nullptr);
    VernonProgramNodeId node = UINT32_MAX;
    ASSERT_EQ(vernonRuntimeProgramGraphAddProgram(graph, bundle, nullptr, {}, &node), VERNON_STATUS_OK);
    VernonProgramNodeBindingToken sourceToken{sizeof(VernonProgramNodeBindingToken)};
    VernonProgramNodeBindingToken outputToken{sizeof(VernonProgramNodeBindingToken)};
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, node, VERNON_PROGRAM_BOUNDARY_INPUT,
                                                    {"source", std::strlen("source")}, &sourceToken),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphFindBoundary(graph, node, VERNON_PROGRAM_BOUNDARY_OUTPUT,
                                                    {"output", std::strlen("output")}, &outputToken),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph, &sourceToken, {"source", std::strlen("source")}),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramGraphExportBoundary(graph, &outputToken, {"output", std::strlen("output")}),
              VERNON_STATUS_OK);
    VernonProgramExecutable *composite = vernonRuntimeResolveProgramGraph(graph);
    ASSERT_NE(composite, nullptr) << lastError(context);
    EXPECT_EQ(vernonRuntimeProgramExecutableHasProgramAutodiff(composite), 0u);

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
    const float seed = 1.0f;
    const float zero = 0.0f;
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), sourceBuffer, 0, &source, sizeof(source)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), outputBuffer, 0, &zero, sizeof(zero)), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), seedBuffer, 0, &seed, sizeof(seed)), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceUploadBuffer(owned.device(), gradientBuffer, 0, &zero, sizeof(zero)),
              VERNON_RHI_STATUS_OK);

    VernonProgramArgument sourceArgument = tensorArgument(context, sourceBuffer, parameter(composite, "source"));
    VernonProgramArgument outputArgument = tensorArgument(context, outputBuffer, parameter(composite, "output"));
    const VernonProgramBindingToken sourceBinding = bindingToken("graph-source");
    const VernonProgramBindingToken outputBinding = bindingToken("graph-output");
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(composite);
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &sourceBinding, &sourceArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &outputBinding, &outputArgument, nullptr, 0, 0),
              VERNON_STATUS_OK);
    VernonPullback *compositePullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 1, nullptr), VERNON_STATUS_OK) << lastError(context);
    EXPECT_NE(vernonRuntimeProgramInvocationGetNodePullback(invocation, node, &compositePullback), VERNON_STATUS_OK);
    compositePullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, &compositePullback), VERNON_STATUS_OK)
        << lastError(context);
    EXPECT_EQ(compositePullback, nullptr);
    VernonPullback *nodePullback = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationGetNodePullback(invocation, node, &nodePullback), VERNON_STATUS_OK)
        << lastError(context);
    vernonRuntimeProgramInvocationDestroy(invocation);
    ASSERT_NE(nodePullback, nullptr);

    VernonProgramParameterView cotangent{};
    VernonProgramParameterView gradient{};
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone, VERNON_PROGRAM_BOUNDARY_COTANGENT, 0, &cotangent),
        VERNON_STATUS_OK);
    ASSERT_EQ(
        vernonRuntimeProgramExecutableGetBoundaryByIndex(standalone, VERNON_PROGRAM_BOUNDARY_GRADIENT, 0, &gradient),
        VERNON_STATUS_OK);
    VernonProgramArgument derivatives[]{
        tensorArgument(context, seedBuffer, cotangent),
        tensorArgument(context, gradientBuffer, gradient),
    };
    ASSERT_EQ(vernonProgramPullbackApply(nodePullback, derivatives, std::size(derivatives), nullptr), VERNON_STATUS_OK)
        << lastError(context);
    float output = 0.0f;
    float gradientValue = 0.0f;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, &output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), gradientBuffer, 0, &gradientValue, sizeof(gradientValue)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(output, 9.0f);
    EXPECT_FLOAT_EQ(gradientValue, 6.0f);

    vernonProgramPullbackDestroy(nodePullback);
    vernonRuntimeProgramInstanceDestroy(instance);
    vernonRuntimeProgramExecutableDestroy(composite);
    vernonRuntimeProgramGraphDestroy(graph);
    vernonRuntimeProgramExecutableDestroy(standalone);
    vernonRuntimeProgramBundleDestroy(bundle);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), gradientBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), seedBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
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

void runReusedStageModule(vernon::tests::OwnedRhiRuntime &owned, const std::filesystem::path &manifestPath) {
    const vernon::tests::OwnedProgramExecutable program(owned.runtime(), manifestPath);
    ASSERT_TRUE(program) << lastError(owned.runtime());

    using namespace vernon::runtime::program;
    const ResolvedNodePlan *first = program.get()->executionPlan->node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *second = program.get()->executionPlan->node(GraphDirection::Forward, 1);
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
    const VernonProgramParameterView sourceParameter = parameter(program.get(), "source");
    const VernonProgramParameterView outputParameter = parameter(program.get(), "output");
    std::array<VernonProgramArgument, 5> arguments{
        bufferArgument(owned.runtime(), sourceBuffer, sourceParameter, shape, strides, 1, sizeof(source)),
        bufferArgument(owned.runtime(), outputBuffer, outputParameter, shape, strides, 1, sizeof(source)),
        hostScalarArgument(parameter(program.get(), "grid_x"), &grid[0]),
        hostScalarArgument(parameter(program.get(), "grid_y"), &grid[1]),
        hostScalarArgument(parameter(program.get(), "grid_z"), &grid[2]),
    };
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.get());
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    for (size_t index = 0; index < arguments.size(); ++index) {
        const std::string text = "reused-stage-" + std::to_string(index);
        const VernonProgramBindingToken token{sizeof(token), text.data(), text.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &arguments[index], nullptr, 0, 0),
                  VERNON_STATUS_OK);
    }
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    std::array<float, 4> output{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, output.data(), sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_EQ(output, (std::array<float, 4>{3, 4, 5, 6}));

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
}

void expectScopeTimings(vernon::tests::OwnedRhiRuntime &owned, const VernonSubmission *submission, bool timings) {
    std::array<VernonScopeTiming, 4> scopes{};
    size_t count = scopes.size();
    const VernonStatus status = vernonSubmissionGetScopeTimings(submission, scopes.data(), scopes.size(), &count);
    if (!timings) {
        ASSERT_EQ(status, VERNON_STATUS_OK) << lastError(owned.runtime());
        EXPECT_EQ(count, 0u);
        return;
    }
    if (!(vernon::rhi::deviceCommandCapabilities(owned.device()) & vernon::rhi::BackendCommandTimers)) {
        EXPECT_EQ(status, VERNON_STATUS_UNSUPPORTED_TARGET);
        return;
    }
    ASSERT_EQ(status, VERNON_STATUS_OK) << lastError(owned.runtime());
    ASSERT_EQ(count, 2u);
    EXPECT_EQ(scopes[0].begin_nanoseconds, 0u);
    for (size_t index = 0; index < count; ++index) {
        EXPECT_EQ(scopes[index].first_node, index);
        EXPECT_EQ(scopes[index].node_count, 1u);
        EXPECT_EQ(scopes[index].device_submission, 0u);
        EXPECT_LE(scopes[index].begin_nanoseconds, scopes[index].end_nanoseconds);
    }
}

void runTensorViewChainModule(vernon::tests::OwnedRhiRuntime &owned, const std::filesystem::path &manifestPath,
                              bool timings) {
    const vernon::tests::OwnedProgramExecutable program(owned.runtime(), manifestPath);
    ASSERT_TRUE(program) << lastError(owned.runtime());

    using namespace vernon::runtime::program;
    const ResolvedNodePlan *producer = program.get()->executionPlan->node(GraphDirection::Forward, 0);
    const ResolvedNodePlan *consumer = program.get()->executionPlan->node(GraphDirection::Forward, 1);
    ASSERT_NE(producer, nullptr);
    ASSERT_NE(consumer, nullptr);
    EXPECT_NE(producer->stage.get(), consumer->stage.get());
    const Graph *forward = findGraph(program.get()->executionPlan->resolvedProgram->program, "forward");
    ASSERT_NE(forward, nullptr);
    ASSERT_EQ(forward->nodes.size(), 2u);
    const auto intermediate = std::find_first_of(forward->nodes[0].results.begin(), forward->nodes[0].results.end(),
                                                 forward->nodes[1].operands.begin(), forward->nodes[1].operands.end());
    ASSERT_NE(intermediate, forward->nodes[0].results.end());
    ASSERT_LT(*intermediate, program.get()->executionPlan->resolvedProgram->program.values.size());
    EXPECT_TRUE(program.get()->executionPlan->resolvedProgram->program.values[*intermediate].storage.has_value());
    const std::vector<uint32_t> &predecessors = program.get()->executionPlan->predecessors(GraphDirection::Forward, 1);
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
        bufferArgument(owned.runtime(), sourceBuffer, parameter(program.get(), "source"), shape, strides, 1,
                       sizeof(source)),
        bufferArgument(owned.runtime(), outputBuffer, parameter(program.get(), "output"), shape, strides, 1,
                       sizeof(source)),
    };
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.get());
    ASSERT_NE(instance, nullptr);
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    ASSERT_NE(invocation, nullptr);
    for (size_t index = 0; index < arguments.size(); ++index) {
        const std::string text = "tensor-view-chain-" + std::to_string(index);
        const VernonProgramBindingToken token{sizeof(token), text.data(), text.size()};
        ASSERT_EQ(vernonRuntimeProgramInvocationBind(invocation, &token, &arguments[index], nullptr, 0, 0),
                  VERNON_STATUS_OK);
    }
    const VernonProgramInvocationOptions options{sizeof(VernonProgramInvocationOptions),
                                                 VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION,
                                                 VERNON_COMPLETION_WAIT,
                                                 timings ? 1u : 0u,
                                                 {}};
    ASSERT_EQ(vernonRuntimeProgramInvocationSetOptions(invocation, &options), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    VernonSubmission *submission = nullptr;
    ASSERT_EQ(vernonRuntimeProgramInvocationTakeSubmission(invocation, &submission), VERNON_STATUS_OK)
        << lastError(owned.runtime());
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    expectScopeTimings(owned, submission, timings);
    vernonSubmissionDestroy(submission);

    float output = 0.0f;
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(owned.device(), outputBuffer, 0, &output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    EXPECT_FLOAT_EQ(output, 8.0f);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), outputBuffer), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(owned.device(), sourceBuffer), VERNON_RHI_STATUS_OK);
}

void runDynamicShapeGridReuse(vernon::tests::OwnedRhiRuntime &owned, const std::filesystem::path &manifestPath) {
    const vernon::tests::OwnedProgramExecutable program(owned.runtime(), manifestPath);
    ASSERT_TRUE(program) << lastError(owned.runtime());
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(program.get());
    ASSERT_NE(instance, nullptr);

    const VernonProgramParameterView sourceParameter = parameter(program.get(), "source");
    const VernonProgramParameterView outputParameter = parameter(program.get(), "output");
    VernonProgramParameterView publishedOutputParameter{};
    ASSERT_EQ(vernonRuntimeProgramExecutableGetBoundaryByIndex(program.get(), VERNON_PROGRAM_BOUNDARY_OUTPUT, 0,
                                                               &publishedOutputParameter),
              VERNON_STATUS_OK);
    const VernonProgramParameterView factorParameter = parameter(program.get(), "factor");
    std::array<VernonProgramParameterView, 3> gridParameters{
        parameter(program.get(), "__grid_x"),
        parameter(program.get(), "__grid_y"),
        parameter(program.get(), "__grid_z"),
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
        ASSERT_EQ(vernonRuntimeProgramInvocationExecute(invocation, 0, nullptr), VERNON_STATUS_OK)
            << lastError(owned.runtime());
        ASSERT_EQ(vernonRuntimeProgramInvocationCommit(invocation, nullptr), VERNON_STATUS_OK)
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
}

const std::vector<vernon::tests::ProgramFixtureManifest> &moduleProgramVjpBackendCases() {
    static const auto cases = vernon::tests::programFixtureCases("module_program");
    return cases;
}

const std::vector<vernon::tests::ProgramFixtureManifest> &moduleProgramComputeBackendCases() {
    static const auto cases = vernon::tests::programFixtureCases("reused_stage");
    return cases;
}

class RuntimeModuleProgramVjpGpuCApi : public vernon::tests::ProgramFixtureRuntimeTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::programVjpFixtureRequirements(GetParam().runtime);
    }
};

TEST_P(RuntimeModuleProgramVjpGpuCApi, ComputeModuleForward9AndVjpGradient6) {
    runModuleProgram(owned(), GetParam().manifestPath);
}

TEST_P(RuntimeModuleProgramVjpGpuCApi, ProgramGraphRetainsNodeLocalPullback) {
    runProgramGraphNodePullback(owned(), GetParam().manifestPath);
}

class RuntimeModuleProgramComputeGpuCApi : public vernon::tests::ProgramFixtureRuntimeTest {
protected:
    vernon::tests::BackendTestRequirements requirements() const override {
        return vernon::tests::computeFixtureRequirements(GetParam().runtime);
    }
};

TEST_P(RuntimeModuleProgramComputeGpuCApi, ReusesOneStageAcrossDifferentNodeProjections) {
    runReusedStageModule(owned(), GetParam().manifestPath);
}

TEST_P(RuntimeModuleProgramComputeGpuCApi, ChainsDistinctComputeKernelsThroughTensorViews) {
    runTensorViewChainModule(owned(), fixture("tensor_view_chain").manifestPath, false);
}

TEST_P(RuntimeModuleProgramComputeGpuCApi, TimesEachComputeNodeOnTheDevice) {
    runTensorViewChainModule(owned(), fixture("tensor_view_chain").manifestPath, true);
}

TEST_P(RuntimeModuleProgramComputeGpuCApi, ReusesLoadedProgramAcrossDynamicShapesAndGrids) {
    runDynamicShapeGridReuse(owned(), fixture("dynamic_shape_grid").manifestPath);
}

INSTANTIATE_TEST_SUITE_P(EnabledTargets, RuntimeModuleProgramVjpGpuCApi,
                         testing::ValuesIn(moduleProgramVjpBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });

INSTANTIATE_TEST_SUITE_P(EnabledTargets, RuntimeModuleProgramComputeGpuCApi,
                         testing::ValuesIn(moduleProgramComputeBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::ProgramFixtureManifest> &info) {
                             return std::string(info.param.target);
                         });

} // namespace
