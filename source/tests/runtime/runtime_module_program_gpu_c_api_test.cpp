#include "VernonRuntime.h"
#include "runtime/program_execution/failure_injection.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

VernonRhiBackend rhiBackend(VernonRuntimeBackend backend) {
    switch (backend) {
    case VERNON_RUNTIME_CUDA:
        return VERNON_RHI_BACKEND_CUDA;
    case VERNON_RUNTIME_VULKAN:
        return VERNON_RHI_BACKEND_VULKAN;
    case VERNON_RUNTIME_DIRECTX12:
        return VERNON_RHI_BACKEND_DIRECTX12;
    case VERNON_RUNTIME_METAL:
        return VERNON_RHI_BACKEND_METAL;
    default:
        return VERNON_RHI_BACKEND_OPENGL;
    }
}

class OwnedGpuRuntime {
public:
    explicit OwnedGpuRuntime(VernonRuntimeBackend backend) {
        if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES) {
            device_ = vernonRhiCreateOpenGLDevice(nullptr, backend == VERNON_RUNTIME_OPENGL_ES);
        } else {
            VernonRhiOwnedDeviceDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.backend = rhiBackend(backend);
            device_ = vernonRhiCreateDevice(&descriptor);
        }
        if (device_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            runtime_ = vernonRuntimeCreateForRhiDevice(backend, device_);
    }

    ~OwnedGpuRuntime() {
        if (runtime_)
            (void)vernonRuntimeDestroy(runtime_);
        if (device_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            (void)vernonRhiDestroyDevice(device_);
    }

    VernonRuntimeContext *runtime() const { return runtime_; }
    VernonRhiDevice device() const { return device_; }

private:
    VernonRhiDevice device_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeContext *runtime_{};
};

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
    OwnedGpuRuntime owned(backend);
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

} // namespace

#if defined(VERNON_MODULE_PROGRAM_VULKAN_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, Vulkan) {
    runModuleProgram(VERNON_RUNTIME_VULKAN, VERNON_MODULE_PROGRAM_VULKAN_MANIFEST);
}
#endif

#if defined(VERNON_MODULE_PROGRAM_CUDA_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, Cuda) { runModuleProgram(VERNON_RUNTIME_CUDA, VERNON_MODULE_PROGRAM_CUDA_MANIFEST); }
#endif

#if defined(VERNON_MODULE_PROGRAM_DIRECTX_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, DirectX12) {
    runModuleProgram(VERNON_RUNTIME_DIRECTX12, VERNON_MODULE_PROGRAM_DIRECTX_MANIFEST);
}
#endif

#if defined(VERNON_MODULE_PROGRAM_METAL_MANIFEST)
TEST(RuntimeModuleProgramGpuCApi, Metal) {
    runModuleProgram(VERNON_RUNTIME_METAL, VERNON_MODULE_PROGRAM_METAL_MANIFEST);
}
#endif

TEST(RuntimeModuleProgramGpuCApi, OpenGL) {
    runModuleProgram(VERNON_RUNTIME_OPENGL, VERNON_MODULE_PROGRAM_OPENGL_MANIFEST);
}
