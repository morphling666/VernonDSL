#include "backend_cuda.h"

#include "runtime_state.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vernon::runtime {
namespace {

#if defined(VERNON_HAS_CUDA_RUNTIME)
VernonStatus cudaFail(VernonRuntimeContext &context, CudaResult result, const char *operation) {
    if (result == kCudaSuccess)
        return VERNON_STATUS_OK;
    const char *name = nullptr;
    const char *description = nullptr;
    CudaDriver &driver = cudaDriver();
    driver.errorName(result, &name);
    driver.errorString(result, &description);
    context.error = std::string(operation) + " failed: " + (name ? name : "CUDA_ERROR") + " (" +
                    (description ? description : "unknown") + ")";
    return VERNON_STATUS_INTERNAL_ERROR;
}
#endif

} // namespace

bool probeCuda(std::string &diagnostic) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaDriver &driver = cudaDriver();
    if (!driver.load()) {
        diagnostic = driver.error;
        return false;
    }
    CudaDevice device{};
    CudaResult status = driver.init(0);
    if (status == kCudaSuccess)
        status = driver.deviceGet(&device, 0);
    if (status == kCudaSuccess)
        return true;
    diagnostic = "CUDA Driver loaded but no usable device was found";
    return false;
#else
    diagnostic = "VernonRuntime was built without CUDA support";
    return false;
#endif
}

bool initializeCudaContext(VernonRuntimeContext &context, uint32_t deviceIndex) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaDriver &driver = cudaDriver();
    auto state = std::make_unique<CudaContextState>();
    if (!driver.load() || driver.init(0) != kCudaSuccess ||
        driver.deviceGet(&state->device, static_cast<int>(deviceIndex)) != kCudaSuccess ||
        driver.primaryContextRetain(&state->context, state->device) != kCudaSuccess)
        return false;
    if (driver.contextSetCurrent(state->context) == kCudaSuccess) {
        installRuntimeBackendState(context, state.release());
        return true;
    }
    driver.primaryContextRelease(state->device);
    return false;
#else
    (void)context;
    (void)deviceIndex;
    return false;
#endif
}

void destroyCudaContext(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaDriver &driver = cudaDriver();
    CudaContextState &state = cudaState(context);
    driver.contextSynchronize();
    driver.primaryContextRelease(state.device);
    state.context = nullptr;
#else
    (void)context;
#endif
}

VernonStatus synchronizeCuda(VernonRuntimeContext &context) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    return cudaFail(context, cudaDriver().contextSynchronize(), "cuCtxSynchronize");
#else
    (void)context;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

bool createCudaBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    auto *state = new CudaBufferState();
    if (cudaFail(*buffer.context, cudaDriver().memoryAllocate(&state->devicePointer, buffer.size), "cuMemAlloc") !=
        VERNON_STATUS_OK) {
        delete state;
        return false;
    }
    installRuntimeBackendState(buffer, state);
    return true;
#else
    (void)buffer;
    return false;
#endif
}

VernonStatus destroyCudaBuffer(VernonDeviceBuffer &buffer) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaBufferState &state = cudaBufferState(buffer);
    if (!state.devicePointer)
        return VERNON_STATUS_OK;
    const VernonStatus status = cudaFail(*buffer.context, cudaDriver().memoryFree(state.devicePointer), "cuMemFree");
    if (status == VERNON_STATUS_OK)
        state.devicePointer = 0;
    return status;
#else
    (void)buffer;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus copyToCudaBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (!source || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer upload";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    const CudaBufferState &state = cudaBufferState(buffer);
    return cudaFail(*buffer.context, cudaDriver().copyHostToDevice(state.devicePointer + offset, source, size),
                    "cuMemcpyHtoD");
#else
    (void)buffer;
    (void)offset;
    (void)source;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus copyFromCudaBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (!destination || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer readback";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    const CudaBufferState &state = cudaBufferState(buffer);
    return cudaFail(*buffer.context, cudaDriver().copyDeviceToHost(destination, state.devicePointer + offset, size),
                    "cuMemcpyDtoH");
#else
    (void)buffer;
    (void)offset;
    (void)destination;
    (void)size;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

bool loadCudaKernel(VernonRuntimeContext &context, const void *artifact, size_t artifactSize, const char *reflection,
                    size_t reflectionSize, const char *entry, size_t entrySize, CudaKernelState &state,
                    ReflectedEntry &metadata) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    try {
        const nlohmann::json parsed = nlohmann::json::parse(reflection, reflection + reflectionSize, nullptr, false);
        const std::string entryName(entry, entrySize);
        if (parsed.is_discarded() || !parseReflection(parsed, entryName, metadata, context.error))
            return false;
        std::string image(static_cast<const char *>(artifact), artifactSize);
        image.push_back('\0');
        CudaDriver &driver = cudaDriver();
        if (cudaFail(context, driver.moduleLoadData(&state.module, image.data(), 0, nullptr, nullptr),
                     "cuModuleLoadDataEx") != VERNON_STATUS_OK)
            return false;
        if (cudaFail(context, driver.moduleGetFunction(&state.function, state.module, entryName.c_str()),
                     "cuModuleGetFunction") == VERNON_STATUS_OK)
            return true;
        driver.moduleUnload(state.module);
        state.module = nullptr;
        return false;
    } catch (const std::exception &exception) {
        context.error = std::string("failed to load CUDA artifact: ") + exception.what();
        return false;
    }
#else
    (void)context;
    (void)artifact;
    (void)artifactSize;
    (void)reflection;
    (void)reflectionSize;
    (void)entry;
    (void)entrySize;
    (void)state;
    (void)metadata;
    return false;
#endif
}

VernonStatus destroyCudaKernel(VernonRuntimeContext &context, CudaKernelState &state) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (!state.module)
        return VERNON_STATUS_OK;
    const VernonStatus status = cudaFail(context, cudaDriver().moduleUnload(state.module), "cuModuleUnload");
    if (status == VERNON_STATUS_OK) {
        state.module = nullptr;
        state.function = nullptr;
    }
    return status;
#else
    (void)context;
    (void)state;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus launchCudaKernel(VernonRuntimeContext &context, const CudaKernelState &state,
                              const ReflectedEntry &metadata, VernonLaunchSize globalSize,
                              const VernonLaunchArgument *arguments, size_t argumentCount) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    struct MemRefDescriptor {
        CudaDevicePointer allocated;
        CudaDevicePointer aligned;
        uint64_t offset;
        uint64_t size;
        uint64_t stride;
    };
    std::vector<MemRefDescriptor> descriptors;
    std::vector<void *> parameters;
    descriptors.reserve(argumentCount);
    parameters.reserve(argumentCount * 5);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : metadata.arguments) {
        if (reflected.kind == "builtin")
            continue;
        const VernonLaunchArgument &argument = arguments[supplied++];
        if (argument.kind == VERNON_LAUNCH_TENSOR) {
            const CudaDevicePointer devicePointer = cudaBufferState(*argument.buffer).devicePointer;
            descriptors.push_back({devicePointer, devicePointer, 0,
                                   reflected.tensorElements
                                       ? reflected.tensorElements
                                       : argument.buffer->size / std::max(reflected.tensorElementSize, size_t{1}),
                                   1});
            MemRefDescriptor &descriptor = descriptors.back();
            parameters.push_back(&descriptor.allocated);
            parameters.push_back(&descriptor.aligned);
            parameters.push_back(&descriptor.offset);
            parameters.push_back(&descriptor.size);
            parameters.push_back(&descriptor.stride);
        } else {
            parameters.push_back(const_cast<void *>(argument.scalar_data));
        }
    }
    const uint32_t *workgroup = metadata.workgroup;
    if (!workgroup[0] || !workgroup[1] || !workgroup[2]) {
        context.error = "CUDA workgroup dimensions must be positive";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return cudaFail(context,
                    cudaDriver().launchKernel(state.function, (globalSize.x - 1) / workgroup[0] + 1,
                                              (globalSize.y - 1) / workgroup[1] + 1,
                                              (globalSize.z - 1) / workgroup[2] + 1, workgroup[0], workgroup[1],
                                              workgroup[2], 0, nullptr, parameters.data(), nullptr),
                    "cuLaunchKernel");
#else
    (void)context;
    (void)state;
    (void)metadata;
    (void)globalSize;
    (void)arguments;
    (void)argumentCount;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
