#include "VernonFormats.hpp"
#include "VernonHal.h"
#include "backend_test_matrix.h"

#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
#include "glfw_context_owner.h"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using vernon::tests::BackendTestRow;

constexpr uint32_t kWidth = 4;
constexpr uint32_t kHeight = 4;

std::string lastError() {
    const VernonStringView error = vernonHalGetLastError();
    return error.data ? std::string(error.data, error.size) : std::string();
}

bool opengl(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

// Every backend row's device, created through VernonHal.h; an OpenGL device adopts a hidden window's context, which
// is also the window its surface presents. A GPU device the machine lacks skips the test, the CPU device never does.
class HalResources : public testing::TestWithParam<BackendTestRow> {
protected:
    void SetUp() override {
        const VernonRuntimeBackend backend = GetParam().runtime;
        VernonHalDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        if (opengl(backend)) {
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
            const bool desktop = backend == VERNON_RUNTIME_OPENGL;
            try {
                context_ = std::make_unique<vernon::host::GlfwContextOwner>(
                    desktop ? vernon::host::GlfwContextApi::OpenGL : vernon::host::GlfwContextApi::OpenGLES,
                    desktop ? 4 : 3, desktop ? 1 : 0, kWidth, kHeight);
            } catch (const std::runtime_error &error) {
                GTEST_SKIP() << GetParam().name << " context is unavailable: " << error.what();
            }
            callbacks_ = context_->callbacks();
            descriptor.opengl = &callbacks_;
#else
            GTEST_SKIP() << "the GLFW context owner is not built";
#endif
        }
        const VernonHalStatus status = vernonHalCreateDevice(&descriptor, &device_);
        if (backend == VERNON_RUNTIME_CPU)
            ASSERT_EQ(status, VERNON_HAL_OK) << lastError();
        else if (status != VERNON_HAL_OK)
            GTEST_SKIP() << GetParam().name << " device is unavailable: " << lastError();
        ASSERT_NE(vernonHalDeviceGetRuntime(device_), nullptr);
        EXPECT_EQ(vernonHalDeviceGetBackend(device_), backend);
    }

    void TearDown() override {
        if (device_)
            EXPECT_EQ(vernonHalDestroyDevice(device_), VERNON_HAL_OK) << lastError();
    }

    VernonHalBuffer *createBuffer(uint64_t size) {
        VernonHalBufferDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.size = size;
        descriptor.usage = VERNON_HAL_BUFFER_STORAGE;
        descriptor.memory = VERNON_HAL_MEMORY_DEVICE;
        VernonHalBuffer *buffer = nullptr;
        EXPECT_EQ(vernonHalCreateBuffer(device_, &descriptor, &buffer), VERNON_HAL_OK) << lastError();
        return buffer;
    }

    // An RGBA8 2D image of `mips` levels; null, after checking the device reports so, on a device without images.
    VernonHalImage *createImage(uint32_t mips) {
        VernonImageDescriptor descriptor{};
        descriptor.dimension = VERNON_TEXTURE_2D;
        descriptor.extent = {kWidth, kHeight, 1};
        descriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
        descriptor.mip_level_count = mips;
        descriptor.array_layer_count = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_IMAGE_SAMPLED | VERNON_IMAGE_TRANSFER_SOURCE | VERNON_IMAGE_TRANSFER_DESTINATION;
        VernonHalImage *image = nullptr;
        const VernonHalStatus status = vernonHalCreateImage(device_, &descriptor, &image);
        if (GetParam().runtime == VERNON_RUNTIME_CUDA) {
            EXPECT_NE(status, VERNON_HAL_OK);
            return nullptr;
        }
        EXPECT_EQ(status, VERNON_HAL_OK) << lastError();
        return image;
    }

    // An image of `format` and one mip level with only transfer usage; null when the device cannot copy the format.
    VernonHalImage *createCopyableImage(VernonTextureFormat format) {
        uint32_t capabilities = 0;
        EXPECT_EQ(vernonHalDeviceGetFormatCapabilities(device_, format, &capabilities), VERNON_HAL_OK) << lastError();
        if (!(capabilities & VERNON_HAL_FORMAT_COPYABLE))
            return nullptr;
        VernonImageDescriptor descriptor{};
        descriptor.dimension = VERNON_TEXTURE_2D;
        descriptor.extent = {kWidth, kHeight, 1};
        descriptor.format = format;
        descriptor.mip_level_count = 1;
        descriptor.array_layer_count = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_IMAGE_TRANSFER_SOURCE | VERNON_IMAGE_TRANSFER_DESTINATION;
        VernonHalImage *image = nullptr;
        EXPECT_EQ(vernonHalCreateImage(device_, &descriptor, &image), VERNON_HAL_OK) << lastError();
        return image;
    }

    // A surface of the device's window, whose drawable size `resizeWindow` sets; null on a device that presents no
    // window, which must then reject surfaces.
    VernonHalSurface *createSurface() {
        VernonHalSurfaceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        bool presents = false;
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
        if (context_) {
            window_ = context_->surfaceCallbacks();
            window_.drawable_size(window_.user_data, &drawableWidth_, &drawableHeight_);
            surfaceCallbacks_ = window_;
            surfaceCallbacks_.user_data = this;
            surfaceCallbacks_.make_current = [](void *self) { window(self).make_current(window(self).user_data); };
            surfaceCallbacks_.swap_interval = [](void *self, int32_t interval) {
                window(self).swap_interval(window(self).user_data, interval);
            };
            surfaceCallbacks_.swap_buffers = [](void *self) { window(self).swap_buffers(window(self).user_data); };
            surfaceCallbacks_.drawable_size = [](void *self, uint32_t *width, uint32_t *height) {
                *width = static_cast<HalResources *>(self)->drawableWidth_;
                *height = static_cast<HalResources *>(self)->drawableHeight_;
            };
            descriptor.native.opengl = &surfaceCallbacks_;
            presents = true;
        }
#if defined(_WIN32)
        if (GetParam().runtime == VERNON_RUNTIME_VULKAN || GetParam().runtime == VERNON_RUNTIME_DIRECTX12) {
            native_ = std::make_unique<vernon::host::GlfwWindowOwner>(kWidth, kHeight);
            descriptor.native.win32_window = native_->win32Window();
            presents = true;
        }
#endif
#endif
        VernonHalSurface *surface = nullptr;
        const VernonHalStatus created = vernonHalCreateSurface(device_, &descriptor, &surface);
        if (!presents) {
            EXPECT_EQ(created, VERNON_HAL_UNSUPPORTED);
            return nullptr;
        }
        if (created == VERNON_HAL_UNSUPPORTED && GetParam().runtime == VERNON_RUNTIME_DIRECTX12) {
            [] { GTEST_SKIP() << "DirectX 12 cannot present here: " << lastError(); }();
            return nullptr;
        }
        EXPECT_EQ(created, VERNON_HAL_OK) << lastError();
        return surface;
    }

    // An OpenGL window reports the size the test sets, so a resize needs no window system.
    void resizeWindow(uint32_t width, uint32_t height) {
        drawableWidth_ = width;
        drawableHeight_ = height;
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
        if (native_)
            native_->resize(width, height);
#endif
    }

    // A configuration of `surface` for its current extent, or for kWidth x kHeight where the surface takes any.
    static VernonHalSurfaceConfiguration supported(VernonHalSurface *surface) {
        VernonHalSurfaceCapabilities capabilities{};
        EXPECT_EQ(vernonHalSurfaceGetCapabilities(surface, &capabilities), VERNON_HAL_OK) << lastError();
        VernonHalSurfaceConfiguration configuration{};
        configuration.struct_size = sizeof(configuration);
        configuration.format = capabilities.formats[0];
        configuration.usage = VERNON_IMAGE_COLOR_ATTACHMENT;
        configuration.width = capabilities.current_width == UINT32_MAX ? kWidth : capabilities.current_width;
        configuration.height = capabilities.current_height == UINT32_MAX ? kHeight : capabilities.current_height;
        configuration.present_mode = VERNON_HAL_PRESENT_FIFO;
        configuration.image_count = std::max(capabilities.min_image_count, 2u);
        if (capabilities.max_image_count)
            configuration.image_count = std::min(configuration.image_count, capabilities.max_image_count);
        return configuration;
    }

    static VernonHalSurfaceConfiguration configure(VernonHalSurface *surface) {
        const VernonHalSurfaceConfiguration configuration = supported(surface);
        EXPECT_EQ(vernonHalSurfaceConfigure(surface, &configuration), VERNON_HAL_OK) << lastError();
        return configuration;
    }

    // The color region of a whole surface image, in RGBA8.
    static VernonHalImageRegion surfaceRegion(const VernonHalSurfaceConfiguration &configuration) {
        VernonHalImageRegion result = region(0);
        result.width = configuration.width;
        result.height = configuration.height;
        return result;
    }

    static VernonHalImageRegion region(uint32_t mip) {
        VernonHalImageRegion result{};
        result.mip_level = mip;
        result.aspect = VERNON_IMAGE_ASPECT_COLOR;
        result.width = std::max(1u, kWidth >> mip);
        result.height = std::max(1u, kHeight >> mip);
        result.depth = 1;
        result.data_format = VERNON_HAL_IMAGE_DATA_RGBA;
        result.data_type = VERNON_HAL_IMAGE_DATA_UINT8;
        return result;
    }

    VernonHalDevice *device_{};
    uint32_t drawableWidth_{kWidth};
    uint32_t drawableHeight_{kHeight};
#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE)
    std::unique_ptr<vernon::host::GlfwContextOwner> context_;
    VernonOpenGLContextCallbacks callbacks_{};
    VernonOpenGLSurfaceCallbacks window_{};
    VernonOpenGLSurfaceCallbacks surfaceCallbacks_{};
    std::unique_ptr<vernon::host::GlfwWindowOwner> native_;

private:
    static const VernonOpenGLSurfaceCallbacks &window(void *self) { return static_cast<HalResources *>(self)->window_; }
#endif
};

TEST_P(HalResources, BufferWritesReadBack) {
    VernonHalBuffer *buffer = createBuffer(1024);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(vernonHalBufferGetSize(buffer), 1024u);
    std::vector<uint8_t> first(256), second(256);
    for (size_t index = 0; index < first.size(); ++index) {
        first[index] = static_cast<uint8_t>(index);
        second[index] = static_cast<uint8_t>(255 - index);
    }
    const std::array<VernonHalBufferWrite, 2> writes{{{0, first.data(), first.size()}, {512, second.data(), 256}}};
    ASSERT_EQ(vernonHalBufferWrite(buffer, writes.data(), writes.size()), VERNON_HAL_OK) << lastError();
    std::vector<uint8_t> readFirst(256), readSecond(256);
    const std::array<VernonHalBufferRead, 2> reads{{{0, readFirst.data(), 256}, {512, readSecond.data(), 256}}};
    ASSERT_EQ(vernonHalBufferRead(buffer, reads.data(), reads.size()), VERNON_HAL_OK) << lastError();
    EXPECT_EQ(readFirst, first);
    EXPECT_EQ(readSecond, second);
    vernonHalDestroyBuffer(buffer);
}

TEST_P(HalResources, BuffersHaveAnAddressOnlyWhereKernelsTakeOne) {
    VernonHalBuffer *buffer = createBuffer(256);
    ASSERT_NE(buffer, nullptr);
    void *address = reinterpret_cast<void *>(uintptr_t{1});
    const VernonHalStatus status = vernonHalBufferGetAddress(buffer, &address);
    const VernonRuntimeBackend backend = GetParam().runtime;
    if (backend == VERNON_RUNTIME_CPU || backend == VERNON_RUNTIME_CUDA) {
        ASSERT_EQ(status, VERNON_HAL_OK) << lastError();
        EXPECT_NE(address, nullptr);
        void *again = nullptr;
        ASSERT_EQ(vernonHalBufferGetAddress(buffer, &again), VERNON_HAL_OK) << lastError();
        EXPECT_EQ(again, address);
    } else {
        EXPECT_EQ(status, VERNON_HAL_UNSUPPORTED);
        EXPECT_EQ(address, nullptr);
    }
    if (backend == VERNON_RUNTIME_CPU) {
        const std::array<uint8_t, 4> bytes{1, 2, 3, 4};
        const VernonHalBufferWrite write{8, bytes.data(), bytes.size()};
        ASSERT_EQ(vernonHalBufferWrite(buffer, &write, 1), VERNON_HAL_OK) << lastError();
        EXPECT_EQ(std::memcmp(static_cast<uint8_t *>(address) + 8, bytes.data(), bytes.size()), 0);
    }
    EXPECT_EQ(vernonHalBufferGetAddress(nullptr, &address), VERNON_HAL_INVALID_ARGUMENT);
    vernonHalDestroyBuffer(buffer);
}

TEST_P(HalResources, RejectsTransfersOutsideTheBuffer) {
    VernonHalBuffer *buffer = createBuffer(256);
    ASSERT_NE(buffer, nullptr);
    std::vector<uint8_t> bytes(64);
    const VernonHalBufferWrite past{224, bytes.data(), 64};
    EXPECT_EQ(vernonHalBufferWrite(buffer, &past, 1), VERNON_HAL_INVALID_ARGUMENT);
    EXPECT_FALSE(lastError().empty());
    const VernonHalBufferRead wrapping{UINT64_MAX - 8, bytes.data(), 64};
    EXPECT_EQ(vernonHalBufferRead(buffer, &wrapping, 1), VERNON_HAL_INVALID_ARGUMENT);
    const VernonHalBufferRead nowhere{0, nullptr, 64};
    EXPECT_EQ(vernonHalBufferRead(buffer, &nowhere, 1), VERNON_HAL_INVALID_ARGUMENT);
    VernonHalBufferDescriptor empty{};
    empty.struct_size = sizeof(empty);
    VernonHalBuffer *none = nullptr;
    EXPECT_EQ(vernonHalCreateBuffer(device_, &empty, &none), VERNON_HAL_INVALID_ARGUMENT);
    EXPECT_EQ(none, nullptr);
    vernonHalDestroyBuffer(buffer);
}

TEST_P(HalResources, ImageLevelsWriteAndReadBack) {
    VernonHalImage *image = createImage(2);
    if (!image)
        return;
    std::vector<uint8_t> base(kWidth * kHeight * 4), level(4 * 4);
    for (size_t index = 0; index < base.size(); ++index)
        base[index] = static_cast<uint8_t>(index * 3);
    for (size_t index = 0; index < level.size(); ++index)
        level[index] = static_cast<uint8_t>(200 - index);
    const std::array<VernonHalImageWrite, 2> writes{{{region(0), base.data()}, {region(1), level.data()}}};
    ASSERT_EQ(vernonHalImageWrite(image, writes.data(), writes.size()), VERNON_HAL_OK) << lastError();
    std::vector<uint8_t> readBase(base.size()), readLevel(level.size());
    const std::array<VernonHalImageRead, 2> reads{
        {{region(0), readBase.data(), readBase.size()}, {region(1), readLevel.data(), readLevel.size()}}};
    ASSERT_EQ(vernonHalImageRead(image, reads.data(), reads.size()), VERNON_HAL_OK) << lastError();
    EXPECT_EQ(readBase, base);
    EXPECT_EQ(readLevel, level);

    VernonHalImageRegion outside = region(0);
    outside.offset_x = 1;
    const VernonHalImageWrite past{outside, base.data()};
    EXPECT_EQ(vernonHalImageWrite(image, &past, 1), VERNON_HAL_INVALID_ARGUMENT);
    EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
}

TEST_P(HalResources, MipmapsFillTheLowerLevels) {
    VernonHalImage *image = createImage(3);
    if (!image)
        return;
    std::vector<uint8_t> base(kWidth * kHeight * 4);
    for (size_t index = 0; index < base.size(); index += 4) {
        base[index] = 64;
        base[index + 1] = 128;
        base[index + 2] = 192;
        base[index + 3] = 255;
    }
    const VernonHalImageWrite write{region(0), base.data()};
    ASSERT_EQ(vernonHalImageWrite(image, &write, 1), VERNON_HAL_OK) << lastError();
    const VernonHalStatus generated = vernonHalImageGenerateMipmaps(image);
    if (GetParam().runtime == VERNON_RUNTIME_CPU) {
        EXPECT_EQ(generated, VERNON_HAL_UNSUPPORTED);
    } else {
        ASSERT_EQ(generated, VERNON_HAL_OK) << lastError();
        std::array<uint8_t, 4> last{};
        const VernonHalImageRead read{region(2), last.data(), last.size()};
        ASSERT_EQ(vernonHalImageRead(image, &read, 1), VERNON_HAL_OK) << lastError();
        for (size_t channel = 0; channel < last.size(); ++channel)
            EXPECT_NEAR(last[channel], base[channel], 1) << "channel " << channel;
    }
    EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
}

TEST_P(HalResources, ViewsKeepTheirImage) {
    VernonHalImage *image = createImage(2);
    if (!image)
        return;
    VernonImageViewDescriptor descriptor{};
    descriptor.dimension = VERNON_TEXTURE_2D;
    descriptor.format = VERNON_TEXTURE_RGBA8_SRGB;
    descriptor.subresources = {1, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR};
    // OpenGL ES 3.0 has no texture views, so an ES view covers its whole image in the image's format.
    if (GetParam().runtime == VERNON_RUNTIME_OPENGL_ES) {
        descriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
        descriptor.subresources = {0, 2, 0, 1, VERNON_IMAGE_ASPECT_COLOR};
    }
    VernonHalImageView *view = nullptr;
    ASSERT_EQ(vernonHalCreateImageView(image, &descriptor, &view), VERNON_HAL_OK) << lastError();
    EXPECT_EQ(vernonHalImageViewGetImage(view), image);
    EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_INVALID_ARGUMENT);

    VernonImageViewDescriptor outside = descriptor;
    outside.subresources.base_mip_level = 2;
    VernonHalImageView *none = nullptr;
    EXPECT_EQ(vernonHalCreateImageView(image, &outside, &none), VERNON_HAL_INVALID_ARGUMENT);
    EXPECT_EQ(none, nullptr);

    EXPECT_EQ(vernonHalDestroyImageView(view), VERNON_HAL_OK);
    EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
}

TEST_P(HalResources, ADeviceWithObjectsIsNotDestroyed) {
    VernonHalBuffer *buffer = createBuffer(64);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(vernonHalDestroyDevice(device_), VERNON_HAL_INVALID_ARGUMENT);
    vernonHalDestroyBuffer(buffer);
}

TEST_P(HalResources, SamplersCheckTheirDescriptor) {
    VernonHalSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.min_filter = VERNON_HAL_FILTER_LINEAR;
    descriptor.mag_filter = VERNON_HAL_FILTER_LINEAR;
    descriptor.mip_filter = VERNON_HAL_FILTER_LINEAR;
    descriptor.address_u = VERNON_HAL_ADDRESS_CLAMP_TO_EDGE;
    descriptor.address_v = VERNON_HAL_ADDRESS_REPEAT;
    descriptor.address_w = VERNON_HAL_ADDRESS_MIRRORED_REPEAT;
    descriptor.max_anisotropy = 1.0f;
    VernonHalSampler *sampler = nullptr;
    const VernonHalStatus status = vernonHalCreateSampler(device_, &descriptor, &sampler);
    if (GetParam().runtime == VERNON_RUNTIME_CUDA) {
        EXPECT_NE(status, VERNON_HAL_OK);
        return;
    }
    ASSERT_EQ(status, VERNON_HAL_OK) << lastError();
    vernonHalDestroySampler(sampler);
    descriptor.max_anisotropy = 20.0f;
    VernonHalSampler *none = nullptr;
    EXPECT_EQ(vernonHalCreateSampler(device_, &descriptor, &none), VERNON_HAL_INVALID_ARGUMENT) << lastError();
    EXPECT_EQ(none, nullptr);
}

TEST_P(HalResources, ResourcesAreReferencedForScheduleArguments) {
    VernonHalBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = 256;
    descriptor.usage = VERNON_HAL_BUFFER_STORAGE;
    VernonHalBuffer *buffer = nullptr;
    ASSERT_EQ(vernonHalCreateBuffer(device_, &descriptor, &buffer), VERNON_HAL_OK) << lastError();
    VernonRuntimeProviderResourceReference reference{};
    const VernonHalStatus referenced = vernonHalReferenceBuffer(buffer, 64, 128, &reference);
    if (GetParam().runtime == VERNON_RUNTIME_CPU) {
        EXPECT_EQ(referenced, VERNON_HAL_UNSUPPORTED);
    } else {
        ASSERT_EQ(referenced, VERNON_HAL_OK) << lastError();
        EXPECT_EQ(reference.offset, 64u);
        EXPECT_EQ(reference.size, 128u);
    }
    EXPECT_EQ(vernonHalReferenceBuffer(buffer, 192, 128, &reference), VERNON_HAL_INVALID_ARGUMENT);
    vernonHalDestroyBuffer(buffer);

    VernonHalImage *image = createImage(1);
    if (!image)
        return;
    VernonImageViewDescriptor view_descriptor{};
    view_descriptor.dimension = VERNON_TEXTURE_2D;
    view_descriptor.format = VERNON_TEXTURE_RGBA8_UNORM;
    view_descriptor.subresources = {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR};
    VernonHalImageView *view = nullptr;
    ASSERT_EQ(vernonHalCreateImageView(image, &view_descriptor, &view), VERNON_HAL_OK) << lastError();
    const VernonHalStatus viewed = vernonHalReferenceImageView(view, &reference);
    EXPECT_EQ(viewed, GetParam().runtime == VERNON_RUNTIME_CPU ? VERNON_HAL_UNSUPPORTED : VERNON_HAL_OK) << lastError();
    EXPECT_EQ(vernonHalDestroyImageView(view), VERNON_HAL_OK);
    EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
}

TEST_P(HalResources, FormatCapabilitiesNarrowTheBaseline) {
    uint32_t capabilities = 0;
    ASSERT_EQ(vernonHalDeviceGetFormatCapabilities(device_, VERNON_TEXTURE_RGBA8_UNORM, &capabilities), VERNON_HAL_OK)
        << lastError();
    if (GetParam().runtime == VERNON_RUNTIME_CUDA)
        EXPECT_EQ(capabilities, 0u);
    else
        EXPECT_NE(capabilities & VERNON_HAL_FORMAT_COPYABLE, 0u);
    if (GetParam().runtime == VERNON_RUNTIME_CPU)
        EXPECT_EQ(capabilities & VERNON_HAL_FORMAT_RENDERABLE, 0u);
}

size_t elementBytes(VernonHalImageDataType type) {
    switch (type) {
    case VERNON_HAL_IMAGE_DATA_UINT8:
    case VERNON_HAL_IMAGE_DATA_INT8:
        return 1;
    case VERNON_HAL_IMAGE_DATA_UINT16:
    case VERNON_HAL_IMAGE_DATA_INT16:
    case VERNON_HAL_IMAGE_DATA_FLOAT16:
        return 2;
    case VERNON_HAL_IMAGE_DATA_UINT32:
    case VERNON_HAL_IMAGE_DATA_INT32:
    case VERNON_HAL_IMAGE_DATA_FLOAT32:
        return 4;
    }
    return 0;
}

// Pseudo-random texels of `format` that every transfer keeps bit for bit: floats are finite and normal, depth lies in
// [0, 1], and snorm components avoid the second encoding of -1, which a normalized read folds into the first.
std::vector<uint8_t> exactTexels(const vernon::formats::Format &format, size_t texels) {
    std::vector<uint8_t> bytes(texels * format.pixelBytes);
    uint32_t state = 0x9E3779B9u ^ static_cast<uint32_t>(format.texture);
    for (uint8_t &byte : bytes) {
        state = state * 1664525u + 1013904223u;
        byte = static_cast<uint8_t>(state >> 24);
    }
    const size_t width = elementBytes(format.dataType);
    const bool snorm = format.componentClass == vernon::formats::ComponentClass::Snorm;
    for (size_t offset = 0; offset < bytes.size(); offset += width) {
        uint8_t *element = bytes.data() + offset;
        if (format.dataType == VERNON_HAL_IMAGE_DATA_FLOAT32) {
            float value = static_cast<float>(offset / width % 61) / 60.0f;
            if (format.aspects() == VERNON_IMAGE_ASPECT_COLOR) {
                uint32_t bits{};
                std::memcpy(&bits, element, sizeof(bits));
                bits = (bits & ~(1u << 30)) | (1u << 29);
                std::memcpy(&value, &bits, sizeof(value));
            }
            std::memcpy(element, &value, sizeof(value));
        } else if (format.dataType == VERNON_HAL_IMAGE_DATA_FLOAT16) {
            element[1] = static_cast<uint8_t>((element[1] & ~0x40u) | 0x20u);
        } else if (snorm && element[width - 1] == 0x80 && (width == 1 || element[0] == 0)) {
            element[0] = 1;
        } else if (format.texture == VERNON_TEXTURE_R11G11B10_FLOAT) {
            uint32_t bits{};
            std::memcpy(&bits, element, sizeof(bits));
            bits &= ~((1u << 10) | (1u << 21) | (1u << 31));
            std::memcpy(element, &bits, sizeof(bits));
        }
    }
    return bytes;
}

// Every format a device copies keeps its texels exactly through a write and whole and partial reads. Depth-stencil
// images transfer in a packed layout of their own, which DepthStencilImagesReadByAspectAndRegion covers.
TEST_P(HalResources, EveryCopyableFormatRoundTripsItsTexels) {
    for (const vernon::formats::Format &format : vernon::formats::table) {
        if (vernon::formats::hasStencil(format.texture))
            continue;
        SCOPED_TRACE(std::string(format.name));
        VernonHalImage *image = createCopyableImage(format.texture);
        if (!image)
            continue;
        const std::vector<uint8_t> texels = exactTexels(format, size_t{kWidth} * kHeight);
        VernonHalImageRegion whole = region(0);
        whole.aspect = format.aspects();
        whole.data_format = format.dataFormat;
        whole.data_type = format.dataType;
        const VernonHalImageWrite write{whole, texels.data()};
        ASSERT_EQ(vernonHalImageWrite(image, &write, 1), VERNON_HAL_OK) << lastError();
        std::vector<uint8_t> read(texels.size());
        const VernonHalImageRead full{whole, read.data(), read.size()};
        const VernonHalStatus readFull = vernonHalImageRead(image, &full, 1);
        // OpenGL ES 3.0 has no compute pass to read depth with.
        if (GetParam().runtime == VERNON_RUNTIME_OPENGL_ES && vernon::formats::hasDepth(format.texture)) {
            EXPECT_EQ(readFull, VERNON_HAL_UNSUPPORTED) << lastError();
            EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
            continue;
        }
        ASSERT_EQ(readFull, VERNON_HAL_OK) << lastError();
        EXPECT_EQ(read, texels);

        // A region reads its own texels, the image's rows below and right of its offset.
        VernonHalImageRegion corner = whole;
        corner.offset_x = 1;
        corner.offset_y = 2;
        corner.width = 2;
        corner.height = 2;
        std::vector<uint8_t> part(size_t{2} * 2 * format.pixelBytes);
        const VernonHalImageRead partial{corner, part.data(), part.size()};
        ASSERT_EQ(vernonHalImageRead(image, &partial, 1), VERNON_HAL_OK) << lastError();
        for (uint32_t y = 0; y < 2; ++y)
            EXPECT_EQ(std::memcmp(part.data() + size_t{y} * 2 * format.pixelBytes,
                                  texels.data() + (size_t{y + 2} * kWidth + 1) * format.pixelBytes,
                                  size_t{2} * format.pixelBytes),
                      0)
                << "texel row " << y;
        EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
    }
}

// A packed depth-stencil texel is a float depth and a stencil byte in eight bytes.
TEST_P(HalResources, DepthStencilImagesReadByAspectAndRegion) {
    VernonHalImage *image = createCopyableImage(VERNON_TEXTURE_D32_FLOAT_S8_UINT);
    if (!image)
        return;
    constexpr size_t kTexelBytes = 8;
    std::vector<uint8_t> packed(size_t{kWidth} * kHeight * kTexelBytes);
    for (size_t index = 0; index < size_t{kWidth} * kHeight; ++index) {
        const float depth = static_cast<float>(index) / 16.0f;
        std::memcpy(packed.data() + index * kTexelBytes, &depth, sizeof(depth));
        packed[index * kTexelBytes + sizeof(depth)] = static_cast<uint8_t>(3 * index + 1);
    }
    VernonHalImageRegion whole = region(0);
    whole.aspect = VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL;
    whole.data_format = VERNON_HAL_IMAGE_DATA_DEPTH_STENCIL;
    whole.data_type = VERNON_HAL_IMAGE_DATA_FLOAT32;
    const VernonHalImageWrite write{whole, packed.data()};
    ASSERT_EQ(vernonHalImageWrite(image, &write, 1), VERNON_HAL_OK) << lastError();

    // OpenGL ES 3.0 has no compute pass to read depth with.
    const bool reads = GetParam().runtime != VERNON_RUNTIME_OPENGL_ES;
    std::vector<uint8_t> both(packed.size());
    const VernonHalImageRead full{whole, both.data(), both.size()};
    const VernonHalStatus readBoth = vernonHalImageRead(image, &full, 1);
    if (!reads) {
        EXPECT_EQ(readBoth, VERNON_HAL_UNSUPPORTED) << lastError();
        EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
        return;
    }
    ASSERT_EQ(readBoth, VERNON_HAL_OK) << lastError();
    for (size_t index = 0; index < size_t{kWidth} * kHeight; ++index) {
        EXPECT_EQ(std::memcmp(both.data() + index * kTexelBytes, packed.data() + index * kTexelBytes, 5), 0)
            << "texel " << index;
    }

    VernonHalImageRegion corner = whole;
    corner.offset_x = 2;
    corner.offset_y = 1;
    corner.width = 2;
    corner.height = 3;
    corner.aspect = VERNON_IMAGE_ASPECT_DEPTH;
    corner.data_format = VERNON_HAL_IMAGE_DATA_DEPTH;
    std::vector<float> depth(size_t{2} * 3);
    const VernonHalImageRead depthRead{corner, depth.data(), depth.size() * sizeof(float)};
    // The CPU device transfers only whole packed texels.
    if (GetParam().runtime == VERNON_RUNTIME_CPU) {
        EXPECT_EQ(vernonHalImageRead(image, &depthRead, 1), VERNON_HAL_UNSUPPORTED);
        EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
        return;
    }
    ASSERT_EQ(vernonHalImageRead(image, &depthRead, 1), VERNON_HAL_OK) << lastError();
    corner.aspect = VERNON_IMAGE_ASPECT_STENCIL;
    corner.data_format = VERNON_HAL_IMAGE_DATA_STENCIL;
    corner.data_type = VERNON_HAL_IMAGE_DATA_UINT8;
    std::vector<uint8_t> stencil(size_t{2} * 3);
    const VernonHalImageRead stencilRead{corner, stencil.data(), stencil.size()};
    ASSERT_EQ(vernonHalImageRead(image, &stencilRead, 1), VERNON_HAL_OK) << lastError();
    for (uint32_t y = 0; y < 3; ++y)
        for (uint32_t x = 0; x < 2; ++x) {
            const size_t texel = size_t{y + 1} * kWidth + x + 2;
            EXPECT_FLOAT_EQ(depth[y * 2 + x], static_cast<float>(texel) / 16.0f) << "texel " << texel;
            EXPECT_EQ(stencil[y * 2 + x], static_cast<uint8_t>(3 * texel + 1)) << "texel " << texel;
        }
    EXPECT_EQ(vernonHalDestroyImage(image), VERNON_HAL_OK);
}

TEST_P(HalResources, ReportsOutOfDateAfterAResizeUntilConfiguredAgain) {
    VernonHalSurface *surface = createSurface();
    if (!surface)
        return;
    const VernonHalSurfaceConfiguration configured = configure(surface);
    VernonHalSurfaceImage image{};
    ASSERT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_OK) << lastError();
    ASSERT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_OK) << lastError();

    resizeWindow(2 * configured.width, 3 * configured.height);
    EXPECT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_SURFACE_OUT_OF_DATE);
    EXPECT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_INVALID_ARGUMENT);
    const VernonHalSurfaceConfiguration resized = configure(surface);
    EXPECT_EQ(resized.width, 2 * configured.width);
    EXPECT_EQ(resized.height, 3 * configured.height);
    ASSERT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_OK) << lastError();
    EXPECT_EQ(vernonHalImageGetDescriptor(image.image)->extent.width, resized.width);
    ASSERT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_OK) << lastError();
    EXPECT_EQ(vernonHalDestroySurface(surface), VERNON_HAL_OK);
}

TEST_P(HalResources, ASurfacePresentsOneAcquiredImageAtATime) {
    VernonHalSurface *surface = createSurface();
    if (!surface)
        return;
    VernonHalSurfaceCapabilities capabilities{};
    ASSERT_EQ(vernonHalSurfaceGetCapabilities(surface, &capabilities), VERNON_HAL_OK) << lastError();
    ASSERT_GE(capabilities.format_count, 1u);
    EXPECT_NE(capabilities.usage & VERNON_IMAGE_COLOR_ATTACHMENT, 0u);

    VernonHalSurfaceImage image{};
    EXPECT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_INVALID_ARGUMENT);
    const VernonHalSurfaceConfiguration configuration = configure(surface);

    for (int frame = 0; frame < 2; ++frame) {
        ASSERT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_OK) << lastError();
        ASSERT_NE(image.image, nullptr);
        ASSERT_NE(image.view, nullptr);
        EXPECT_EQ(vernonHalImageViewGetImage(image.view), image.image);
        const VernonImageDescriptor *acquired = vernonHalImageGetDescriptor(image.image);
        EXPECT_EQ(acquired->format, configuration.format);
        EXPECT_EQ(acquired->extent.width, configuration.width);
        VernonHalSurfaceImage second{};
        EXPECT_EQ(vernonHalSurfaceAcquire(surface, &second), VERNON_HAL_INVALID_ARGUMENT);
        EXPECT_EQ(vernonHalSurfaceConfigure(surface, &configuration), VERNON_HAL_INVALID_ARGUMENT);
        EXPECT_EQ(vernonHalDestroySurface(surface), VERNON_HAL_INVALID_ARGUMENT);
        EXPECT_EQ(vernonHalDestroyImageView(image.view), VERNON_HAL_INVALID_ARGUMENT);
        EXPECT_EQ(vernonHalDestroyImage(image.image), VERNON_HAL_INVALID_ARGUMENT);
        ASSERT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_OK) << lastError();
    }
    EXPECT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_INVALID_ARGUMENT);
    EXPECT_EQ(vernonHalDestroySurface(surface), VERNON_HAL_OK);
}

TEST_P(HalResources, RejectsConfigurationsOutsideTheCapabilities) {
    VernonHalSurface *surface = createSurface();
    if (!surface)
        return;
    const VernonHalSurfaceConfiguration valid = supported(surface);
    VernonHalSurfaceConfiguration zeroExtent = valid;
    zeroExtent.width = 0;
    EXPECT_EQ(vernonHalSurfaceConfigure(surface, &zeroExtent), VERNON_HAL_INVALID_ARGUMENT);
    VernonHalSurfaceConfiguration noColorAttachment = valid;
    noColorAttachment.usage = VERNON_IMAGE_TRANSFER_SOURCE;
    EXPECT_EQ(vernonHalSurfaceConfigure(surface, &noColorAttachment), VERNON_HAL_INVALID_ARGUMENT);
    VernonHalSurfaceConfiguration unknownMode = valid;
    unknownMode.present_mode = static_cast<VernonHalPresentMode>(3);
    EXPECT_EQ(vernonHalSurfaceConfigure(surface, &unknownMode), VERNON_HAL_INVALID_ARGUMENT);

    VernonHalSurfaceConfiguration srgb = valid;
    srgb.format = VERNON_TEXTURE_RGBA8_SRGB;
    EXPECT_EQ(vernonHalSurfaceConfigure(surface, &srgb), VERNON_HAL_UNSUPPORTED);
    VernonHalSurfaceConfiguration sampled = valid;
    sampled.usage |= VERNON_IMAGE_SAMPLED;
    EXPECT_EQ(vernonHalSurfaceConfigure(surface, &sampled), VERNON_HAL_UNSUPPORTED);
    VernonHalSurfaceConfiguration noImages = valid;
    noImages.image_count = 0;
    EXPECT_EQ(vernonHalSurfaceConfigure(surface, &noImages), VERNON_HAL_UNSUPPORTED);

    VernonHalSurfaceImage image{};
    EXPECT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_INVALID_ARGUMENT);
    EXPECT_EQ(vernonHalDestroySurface(surface), VERNON_HAL_OK);
}

// An acquired image's contents are undefined until written, and again once it is acquired after a present.
TEST_P(HalResources, ASurfaceImageIsReadOnlyOnceWritten) {
    VernonHalSurface *surface = createSurface();
    if (!surface)
        return;
    const VernonHalSurfaceConfiguration configuration = configure(surface);
    const VernonHalImageRegion whole = surfaceRegion(configuration);
    const size_t bytes = size_t{configuration.width} * configuration.height * 4;
    const std::array<std::array<uint8_t, 4>, 3> colors{{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}}};
    for (const auto &color : colors) {
        VernonHalSurfaceImage image{};
        ASSERT_EQ(vernonHalSurfaceAcquire(surface, &image), VERNON_HAL_OK) << lastError();
        std::vector<uint8_t> texels(bytes);
        const VernonHalImageRead read{whole, texels.data(), texels.size()};
        EXPECT_EQ(vernonHalImageRead(image.image, &read, 1), VERNON_HAL_INVALID_ARGUMENT);
        std::vector<uint8_t> filled;
        for (size_t texel = 0; texel < bytes / 4; ++texel)
            filled.insert(filled.end(), color.begin(), color.end());
        const VernonHalImageWrite write{whole, filled.data()};
        ASSERT_EQ(vernonHalImageWrite(image.image, &write, 1), VERNON_HAL_OK) << lastError();
        ASSERT_EQ(vernonHalImageRead(image.image, &read, 1), VERNON_HAL_OK) << lastError();
        EXPECT_EQ(texels, filled);
        ASSERT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_OK) << lastError();
    }
    EXPECT_EQ(vernonHalDestroySurface(surface), VERNON_HAL_OK);
}

#if defined(VERNON_GLFW_CONTEXT_OWNER_AVAILABLE) && defined(_WIN32)
// Each surface presents its own window and keeps its own images, so one device presents both in turn.
TEST_P(HalResources, TwoSurfacesPresentTheirOwnImages) {
    if (GetParam().runtime != VERNON_RUNTIME_VULKAN && GetParam().runtime != VERNON_RUNTIME_DIRECTX12)
        GTEST_SKIP() << "only Vulkan and DirectX 12 surfaces present windows the test opens";
    VernonHalSurface *first = createSurface();
    if (!first)
        return;
    const vernon::host::GlfwWindowOwner secondWindow(kWidth, kHeight);
    VernonHalSurfaceDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.native.win32_window = secondWindow.win32Window();
    VernonHalSurface *second = nullptr;
    ASSERT_EQ(vernonHalCreateSurface(device_, &descriptor, &second), VERNON_HAL_OK) << lastError();

    const std::array<VernonHalSurface *, 2> surfaces{first, second};
    const std::array<std::array<uint8_t, 4>, 2> colors{{{255, 255, 0, 255}, {0, 255, 255, 255}}};
    std::array<VernonHalSurfaceImage, 2> images{};
    std::array<VernonHalImageRegion, 2> regions{};
    for (size_t index = 0; index < surfaces.size(); ++index) {
        regions[index] = surfaceRegion(configure(surfaces[index]));
        ASSERT_EQ(vernonHalSurfaceAcquire(surfaces[index], &images[index]), VERNON_HAL_OK) << lastError();
    }
    EXPECT_NE(images[0].image, images[1].image);
    std::array<std::vector<uint8_t>, 2> filled;
    for (size_t index = 0; index < surfaces.size(); ++index) {
        for (size_t texel = 0; texel < size_t{regions[index].width} * regions[index].height; ++texel)
            filled[index].insert(filled[index].end(), colors[index].begin(), colors[index].end());
        const VernonHalImageWrite write{regions[index], filled[index].data()};
        ASSERT_EQ(vernonHalImageWrite(images[index].image, &write, 1), VERNON_HAL_OK) << lastError();
    }
    for (size_t index = 0; index < surfaces.size(); ++index) {
        std::vector<uint8_t> texels(filled[index].size());
        const VernonHalImageRead read{regions[index], texels.data(), texels.size()};
        ASSERT_EQ(vernonHalImageRead(images[index].image, &read, 1), VERNON_HAL_OK) << lastError();
        EXPECT_EQ(texels, filled[index]);
    }
    for (VernonHalSurface *surface : surfaces) {
        EXPECT_EQ(vernonHalSurfacePresent(surface), VERNON_HAL_OK) << lastError();
        EXPECT_EQ(vernonHalDestroySurface(surface), VERNON_HAL_OK);
    }
}
#endif

INSTANTIATE_TEST_SUITE_P(Backends, HalResources, testing::ValuesIn(vernon::tests::backendTestMatrix),
                         [](const testing::TestParamInfo<BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
