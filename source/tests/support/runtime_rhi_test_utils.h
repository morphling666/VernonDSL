#ifndef VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H
#define VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H

#include "VernonRuntime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>

namespace vernon::tests {

struct RhiRuntime {
    VernonRhiDevice device{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeContext *runtime{};
};

enum class RhiTestDevicePreference { Hardware, Software };

inline RhiTestDevicePreference configuredDirectXTestDevicePreference() {
    const char *value = std::getenv("VERNON_TEST_DIRECTX_ADAPTER");
    if (!value || std::strcmp(value, "hardware") == 0)
        return RhiTestDevicePreference::Hardware;
    if (std::strcmp(value, "warp") == 0)
        return RhiTestDevicePreference::Software;
    throw std::runtime_error("VERNON_TEST_DIRECTX_ADAPTER must be 'hardware' or 'warp'");
}

class OwnedRhiRuntime {
public:
    explicit OwnedRhiRuntime(VernonRuntimeBackend backend,
                             const VernonOpenGLContextCallbacks *openglCallbacks = nullptr,
                             RhiTestDevicePreference devicePreference = RhiTestDevicePreference::Hardware);
    ~OwnedRhiRuntime();

    OwnedRhiRuntime(const OwnedRhiRuntime &) = delete;
    OwnedRhiRuntime &operator=(const OwnedRhiRuntime &) = delete;
    OwnedRhiRuntime(OwnedRhiRuntime &&) = delete;
    OwnedRhiRuntime &operator=(OwnedRhiRuntime &&) = delete;

    VernonRuntimeContext *get() const { return context_.runtime; }
    VernonRuntimeContext *runtime() const { return context_.runtime; }
    VernonRhiDevice device() const { return context_.device; }
    RhiRuntime &context() { return context_; }
    const RhiRuntime &context() const { return context_; }

private:
    RhiRuntime context_;
};

// Loads the cooked bundle whose blobs sit next to `manifestPath`; null when the manifest is unreadable or rejected.
inline VernonProgramBundle *loadProgramBundle(VernonRuntimeContext *runtime,
                                              const std::filesystem::path &manifestPath) {
    std::ifstream input(manifestPath, std::ios::binary);
    const std::string manifest{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (!runtime || manifest.empty())
        return nullptr;
    const std::string directory = manifestPath.parent_path().u8string();
    VernonProgramBundleLoadOptions options{};
    options.struct_size = sizeof(options);
    options.bundle_directory = directory.c_str();
    return vernonRuntimeLoadProgramBundleWithOptions(runtime, manifest.data(), manifest.size(), &options);
}

class OwnedProgramExecutable {
public:
    OwnedProgramExecutable(VernonRuntimeContext *runtime, const std::filesystem::path &manifestPath,
                           const VernonProgramVariantSelector *selector = nullptr)
        : bundle_(loadProgramBundle(runtime, manifestPath)) {
        if (bundle_)
            executable_ = vernonRuntimeResolveProgram(bundle_, selector);
    }

    ~OwnedProgramExecutable() {
        vernonRuntimeProgramExecutableDestroy(executable_);
        vernonRuntimeProgramBundleDestroy(bundle_);
    }

    OwnedProgramExecutable(const OwnedProgramExecutable &) = delete;
    OwnedProgramExecutable &operator=(const OwnedProgramExecutable &) = delete;
    OwnedProgramExecutable(OwnedProgramExecutable &&) = delete;
    OwnedProgramExecutable &operator=(OwnedProgramExecutable &&) = delete;

    VernonProgramExecutable *get() const { return executable_; }
    explicit operator bool() const { return executable_ != nullptr; }

private:
    VernonProgramBundle *bundle_{};
    VernonProgramExecutable *executable_{};
};

struct RhiBuffer {
    VernonRhiBuffer handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct RhiImage {
    VernonRhiImage handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
};

struct RhiImageView {
    VernonRhiImageView handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct RhiSampler {
    VernonRhiSampler handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

inline VernonRhiStatus completeSubmission(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                          VernonRhiCommandEncoderStats *stats = nullptr) {
    VernonRhiCompletion completion{};
    VernonRhiStatus status = vernonRhiDeviceSubmit(device, encoder, &completion);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    status = vernonRhiCompletionWait(device, completion);
    if (status == VERNON_RHI_STATUS_OK && stats)
        status = vernonRhiCompletionGetCommandStats(device, completion, stats);
    const VernonRhiStatus destroyStatus = vernonRhiDeviceDestroyCompletion(device, completion);
    return status == VERNON_RHI_STATUS_OK ? destroyStatus : status;
}

// The statuses of downloading a buffer and a 4x4 RGBA8 image that the device creates for it; an empty result means the
// device could not create them.
inline std::optional<std::array<VernonRhiStatus, 2>> hostReadStatuses(VernonRhiDevice device) {
    constexpr uint32_t extent = 4;
    std::array<uint8_t, extent * extent * 4> destination{};
    VernonRhiBufferDescriptor bufferDescriptor{};
    bufferDescriptor.struct_size = sizeof(bufferDescriptor);
    bufferDescriptor.size = destination.size();
    bufferDescriptor.usage = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    bufferDescriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    VernonRhiImageDescriptor imageDescriptor{};
    imageDescriptor.struct_size = sizeof(imageDescriptor);
    imageDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    imageDescriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
    imageDescriptor.width = extent;
    imageDescriptor.height = extent;
    imageDescriptor.depth = 1;
    imageDescriptor.mip_levels = 1;
    imageDescriptor.array_layers = 1;
    imageDescriptor.sample_count = 1;
    imageDescriptor.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    VernonRhiBuffer buffer{};
    VernonRhiImage image{};
    if (vernonRhiDeviceCreateBuffer(device, &bufferDescriptor, &buffer) != VERNON_RHI_STATUS_OK)
        return std::nullopt;
    if (vernonRhiDeviceCreateImage(device, &imageDescriptor, &image) != VERNON_RHI_STATUS_OK) {
        vernonRhiDeviceDestroyBuffer(device, buffer);
        return std::nullopt;
    }
    VernonRhiImageDownloadDescriptor download{};
    download.struct_size = sizeof(download);
    download.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
    download.width = extent;
    download.height = extent;
    download.depth = 1;
    download.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
    download.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
    const std::array<VernonRhiStatus, 2> statuses{
        vernonRhiDeviceDownloadBuffer(device, buffer, 0, destination.data(), destination.size()),
        vernonRhiDeviceDownloadImage(device, image, &download, destination.data(), destination.size())};
    vernonRhiDeviceDestroyImage(device, image);
    vernonRhiDeviceDestroyBuffer(device, buffer);
    return statuses;
}

inline VernonRhiBackend rhiBackend(VernonRuntimeBackend backend) {
    switch (backend) {
    case VERNON_RUNTIME_CUDA:
        return VERNON_RHI_BACKEND_CUDA;
    case VERNON_RUNTIME_VULKAN:
        return VERNON_RHI_BACKEND_VULKAN;
    case VERNON_RUNTIME_DIRECTX12:
        return VERNON_RHI_BACKEND_DIRECTX12;
    case VERNON_RUNTIME_OPENGL:
        return VERNON_RHI_BACKEND_OPENGL;
    case VERNON_RUNTIME_OPENGL_ES:
        return VERNON_RHI_BACKEND_OPENGL_ES;
    case VERNON_RUNTIME_METAL:
        return VERNON_RHI_BACKEND_METAL;
    default:
        return VERNON_RHI_BACKEND_CUDA;
    }
}

inline RhiRuntime createRhiRuntime(VernonRuntimeBackend backend,
                                   const VernonOpenGLContextCallbacks *openglCallbacks = nullptr,
                                   RhiTestDevicePreference devicePreference = RhiTestDevicePreference::Hardware,
                                   uint32_t deviceFlags = 0) {
    RhiRuntime result;
    if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES)
        result.device = vernonRhiCreateOpenGLDevice(openglCallbacks, backend == VERNON_RUNTIME_OPENGL_ES, deviceFlags);
    else {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = rhiBackend(backend);
        descriptor.flags = deviceFlags | (devicePreference == RhiTestDevicePreference::Software
                                              ? static_cast<uint32_t>(VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE)
                                              : 0);
        result.device = vernonRhiCreateDevice(&descriptor);
    }
    if (result.device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        result.runtime = vernonRuntimeCreateForRhiDevice(backend, result.device, nullptr);
    return result;
}

inline void destroyRhiRuntime(RhiRuntime &context) {
    if (context.runtime)
        vernonRuntimeDestroy(context.runtime);
    if (context.device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        vernonRhiDestroyDevice(context.device);
    context = {};
}

inline OwnedRhiRuntime::OwnedRhiRuntime(VernonRuntimeBackend backend,
                                        const VernonOpenGLContextCallbacks *openglCallbacks,
                                        RhiTestDevicePreference devicePreference)
    : context_(createRhiRuntime(backend, openglCallbacks, devicePreference)) {}

inline OwnedRhiRuntime::~OwnedRhiRuntime() { destroyRhiRuntime(context_); }

inline RhiBuffer createBuffer(RhiRuntime &context, uint64_t size, uint64_t alignment, uint32_t usage,
                              const void *initialData = nullptr) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.alignment = alignment;
    descriptor.usage = usage | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    RhiBuffer result;
    if (vernonRhiDeviceCreateBuffer(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (initialData &&
        vernonRhiDeviceUploadBuffer(context.device, result.handle, 0, initialData, size) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiBuffer(context.runtime, result.handle, 0, size, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

inline RhiImage createImage(RhiRuntime &context, VernonRhiImageDimension dimension, VernonRhiFormat format,
                            uint32_t width, uint32_t height, uint32_t depth, uint32_t usage, uint32_t arrayLayers = 1) {
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.mip_levels = 1;
    descriptor.array_layers = arrayLayers;
    descriptor.sample_count = 1;
    descriptor.usage = usage | VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    RhiImage result;
    if (vernonRhiDeviceCreateImage(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    return result;
}

inline RhiImageView createImageView(RhiRuntime &context, const RhiImage &image, VernonRhiImageDimension dimension,
                                    VernonRhiFormat format, uint32_t mipLevels = 1, uint32_t arrayLayers = 1,
                                    uint32_t aspects = VERNON_RHI_IMAGE_ASPECT_COLOR) {
    VernonRhiImageViewDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.image = image.handle;
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.mip_level_count = mipLevels;
    descriptor.array_layer_count = arrayLayers;
    descriptor.aspects = aspects;
    RhiImageView result;
    if (vernonRhiDeviceCreateImageView(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiImageView(context.runtime, result.handle, &result.reference) != VERNON_STATUS_OK)
        return {};
    return result;
}

inline RhiSampler createSampler(RhiRuntime &context, uint32_t filter = VERNON_RHI_FILTER_NEAREST,
                                uint32_t address = VERNON_RHI_ADDRESS_CLAMP_TO_EDGE) {
    VernonRhiSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.mip_filter = filter;
    descriptor.address_u = address;
    descriptor.address_v = address;
    descriptor.address_w = address;
    RhiSampler result;
    if (vernonRhiDeviceCreateSampler(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiSampler(context.runtime, result.handle, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

} // namespace vernon::tests

#endif
