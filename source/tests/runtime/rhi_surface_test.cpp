#include "backend_runtime_owner.h"
#include "backend_test_matrix.h"
#include "vernon_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace {

#if defined(_WIN32)
#define VERNON_TEST_GL_CALL __stdcall
#else
#define VERNON_TEST_GL_CALL
#endif

constexpr uint32_t kWidth = 4;
constexpr uint32_t kHeight = 2;

// The GL entry points that read the window's back buffer.
struct WindowReader {
    unsigned(VERNON_TEST_GL_CALL *getError)(){};
    void(VERNON_TEST_GL_CALL *getIntegerv)(unsigned, int *){};
    void(VERNON_TEST_GL_CALL *bindFramebuffer)(unsigned, unsigned){};
    void(VERNON_TEST_GL_CALL *bindBuffer)(unsigned, unsigned){};
    void(VERNON_TEST_GL_CALL *readBuffer)(unsigned){};
    void(VERNON_TEST_GL_CALL *readPixels)(int, int, int, int, unsigned, unsigned, void *){};
};

// The surface presents the device context's window, and reports a drawable size the test sets, so a resize needs no
// window system. Every swap first reads the window's back buffer, which holds what present blitted.
class RhiSurface : public testing::TestWithParam<vernon::tests::BackendTestRow> {
protected:
    void SetUp() override {
        ASSERT_TRUE(GetParam().rhi.has_value());
        vernon::tests::BackendTestRequirements requirements;
        requirements.graphics = true;
        requirements.windowWidth = kWidth;
        requirements.windowHeight = kHeight;
        const vernon::tests::BackendProbeResult probe = runtime_.initialize(GetParam(), requirements);
        if (!probe.available()) {
            if (probe.skippable())
                GTEST_SKIP() << probe.reason;
            FAIL() << probe.reason;
        }
        window_ = runtime_.openglSurface();
        if (window_) {
            callbacks_ = *window_;
            callbacks_.user_data = this;
            callbacks_.make_current = [](void *self) { window(self).make_current(window(self).user_data); };
            callbacks_.swap_interval = [](void *self, int32_t interval) {
                window(self).swap_interval(window(self).user_data, interval);
            };
            callbacks_.swap_buffers = [](void *self) {
                static_cast<RhiSurface *>(self)->readWindow();
                window(self).swap_buffers(window(self).user_data);
            };
            callbacks_.drawable_size = [](void *self, uint32_t *width, uint32_t *height) {
                *width = static_cast<RhiSurface *>(self)->drawableWidth_;
                *height = static_cast<RhiSurface *>(self)->drawableHeight_;
            };
            ASSERT_NO_FATAL_FAILURE(loadWindowReader(*runtime_.openglContext()));
        }
    }

    VernonRhiDevice device() { return runtime_.context().device; }

    VernonRhiStatus createSurface(VernonRhiSurface &surface, VernonRhiBackend backend) {
        VernonRhiSurfaceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = backend;
        descriptor.native.opengl = &callbacks_;
        return vernonRhiDeviceCreateSurface(device(), &descriptor, &surface);
    }

    // Surfaces exist only on OpenGL so far; every other backend must reject them.
    bool createSurfaceOrExpectUnsupported(VernonRhiSurface &surface) {
        const VernonRhiStatus status = createSurface(surface, *GetParam().rhi);
        if (!window_) {
            EXPECT_EQ(status, VERNON_RHI_STATUS_UNSUPPORTED);
            return false;
        }
        EXPECT_EQ(status, VERNON_RHI_STATUS_OK) << vernon::test::text(vernonRhiDeviceGetLastError(device()));
        return status == VERNON_RHI_STATUS_OK;
    }

    VernonRhiStatus acquire(VernonRhiSurface surface, VernonRhiSurfaceImage &image) {
        image = {};
        image.struct_size = sizeof(image);
        return vernonRhiSurfaceAcquireImage(device(), surface, &image);
    }

    static VernonRhiSurfaceConfiguration configuration(uint32_t imageCount, uint32_t width = kWidth,
                                                       uint32_t height = kHeight) {
        VernonRhiSurfaceConfiguration result{};
        result.struct_size = sizeof(result);
        result.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
        result.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
        result.width = width;
        result.height = height;
        result.present_mode = VERNON_RHI_PRESENT_FIFO;
        result.image_count = imageCount;
        return result;
    }

    VernonRhiCommandEncoder beginEncoder() {
        VernonRhiCommandEncoderDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.required_capabilities = VERNON_RHI_QUEUE_GRAPHICS;
        VernonRhiCommandEncoder encoder{};
        EXPECT_EQ(vernonRhiDeviceCreateCommandEncoder(device(), &descriptor, &encoder), VERNON_RHI_STATUS_OK);
        return encoder;
    }

    VernonRhiStatus beginRendering(VernonRhiCommandEncoder encoder, VernonRhiImageView view,
                                   VernonRhiLoadOperation load, const std::array<float, 4> &clear = {},
                                   uint32_t firstRow = 0, uint32_t rowCount = kHeight) {
        VernonRhiColorAttachment color{};
        color.view = view;
        color.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
        color.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
        color.load_operation = load;
        color.store_operation = VERNON_RHI_STORE_PRESERVE;
        for (size_t channel = 0; channel < clear.size(); ++channel)
            color.clear_color.f32[channel] = clear[channel];
        VernonRhiRenderingDescriptor rendering{};
        rendering.struct_size = sizeof(rendering);
        rendering.color_attachments = &color;
        rendering.color_attachment_count = 1;
        rendering.offset_y = firstRow;
        rendering.width = kWidth;
        rendering.height = rowCount;
        rendering.layers = 1;
        return vernonRhiCommandEncoderBeginRendering(device(), encoder, &rendering);
    }

    void submit(VernonRhiCommandEncoder encoder) {
        ASSERT_EQ(vernonRhiCommandEncoderFinish(device(), encoder), VERNON_RHI_STATUS_OK);
        VernonRhiCompletion completion{};
        ASSERT_EQ(vernonRhiDeviceSubmit(device(), encoder, &completion), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCompletionWait(device(), completion), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device()));
        EXPECT_EQ(vernonRhiDeviceDestroyCompletion(device(), completion), VERNON_RHI_STATUS_OK);
    }

    void clear(const std::vector<VernonRhiImageView> &views, const std::array<float, 4> &color) {
        const VernonRhiCommandEncoder encoder = beginEncoder();
        for (const VernonRhiImageView view : views) {
            ASSERT_EQ(beginRendering(encoder, view, VERNON_RHI_LOAD_CLEAR, color), VERNON_RHI_STATUS_OK);
            ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), encoder), VERNON_RHI_STATUS_OK);
        }
        submit(encoder);
    }

    VernonRhiStatus download(VernonRhiImage image, std::vector<uint8_t> &texels) {
        VernonRhiImageDownloadDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.aspect = VERNON_RHI_IMAGE_ASPECT_COLOR;
        descriptor.width = kWidth;
        descriptor.height = kHeight;
        descriptor.depth = 1;
        descriptor.destination_format = VERNON_RHI_IMAGE_DATA_RGBA;
        descriptor.destination_type = VERNON_RHI_IMAGE_DATA_UINT8;
        texels.assign(size_t{kWidth} * kHeight * 4, 0);
        return vernonRhiDeviceDownloadImage(device(), image, &descriptor, texels.data(), texels.size());
    }

    VernonRhiImage createCopyDestination() {
        VernonRhiImageDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.dimension = VERNON_RHI_IMAGE_2D;
        descriptor.format = VERNON_RHI_FORMAT_RGBA8_UNORM;
        descriptor.width = kWidth;
        descriptor.height = kHeight;
        descriptor.depth = 1;
        descriptor.mip_levels = 1;
        descriptor.array_layers = 1;
        descriptor.sample_count = 1;
        descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
        VernonRhiImage image{};
        EXPECT_EQ(vernonRhiDeviceCreateImage(device(), &descriptor, &image), VERNON_RHI_STATUS_OK);
        return image;
    }

    VernonRhiStatus copy(VernonRhiCommandEncoder encoder, VernonRhiImage source, VernonRhiImage destination) {
        VernonRhiImageCopyRegion region{};
        region.struct_size = sizeof(region);
        region.width = kWidth;
        region.height = kHeight;
        region.depth = 1;
        region.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
        return vernonRhiCommandEncoderCopyImage(device(), encoder, source, destination, &region, 1);
    }

    static std::vector<uint8_t> filled(const std::array<uint8_t, 4> &texel) {
        std::vector<uint8_t> result;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index)
            result.insert(result.end(), texel.begin(), texel.end());
        return result;
    }

    uint32_t drawableWidth_{kWidth};
    uint32_t drawableHeight_{kHeight};
    // The window's bottom-left kWidth x kHeight texels at the last swap, rows from the bottom of the window.
    std::vector<uint8_t> windowTexels_;
    unsigned windowReadError_{};

private:
    static const VernonOpenGLSurfaceCallbacks &window(void *self) { return *static_cast<RhiSurface *>(self)->window_; }

    void loadWindowReader(const VernonOpenGLContextCallbacks &context) {
        const auto load = [&context](auto &function, const char *name) {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                context.get_proc_address(context.user_data, name));
            ASSERT_NE(function, nullptr) << name;
        };
        load(reader_.getError, "glGetError");
        load(reader_.getIntegerv, "glGetIntegerv");
        load(reader_.bindFramebuffer, "glBindFramebuffer");
        load(reader_.bindBuffer, "glBindBuffer");
        load(reader_.readBuffer, "glReadBuffer");
        load(reader_.readPixels, "glReadPixels");
    }

    // Runs inside present with the device's context current on the window, so it restores the bindings it changes.
    void readWindow() {
        constexpr unsigned kReadFramebuffer = 0x8CA8;
        constexpr unsigned kReadFramebufferBinding = 0x8CAA;
        constexpr unsigned kPixelPackBuffer = 0x88EB;
        constexpr unsigned kPixelPackBufferBinding = 0x88ED;
        constexpr unsigned kBack = 0x0405;
        constexpr unsigned kRgba = 0x1908;
        constexpr unsigned kUnsignedByte = 0x1401;
        int framebuffer = 0;
        int packBuffer = 0;
        reader_.getIntegerv(kReadFramebufferBinding, &framebuffer);
        reader_.getIntegerv(kPixelPackBufferBinding, &packBuffer);
        reader_.bindFramebuffer(kReadFramebuffer, 0);
        reader_.bindBuffer(kPixelPackBuffer, 0);
        reader_.readBuffer(kBack);
        windowTexels_.assign(size_t{kWidth} * kHeight * 4, 0);
        reader_.readPixels(0, 0, kWidth, kHeight, kRgba, kUnsignedByte, windowTexels_.data());
        reader_.bindBuffer(kPixelPackBuffer, static_cast<unsigned>(packBuffer));
        reader_.bindFramebuffer(kReadFramebuffer, static_cast<unsigned>(framebuffer));
        windowReadError_ = reader_.getError();
    }

    WindowReader reader_;

    vernon::tests::BackendRuntimeOwner runtime_;
    std::optional<VernonOpenGLSurfaceCallbacks> window_;
    VernonOpenGLSurfaceCallbacks callbacks_{};
};

TEST_P(RhiSurface, PresentsClearedImagesThatReadBackExactly) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    VernonRhiSurfaceCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    ASSERT_EQ(vernonRhiSurfaceGetCapabilities(device(), surface, &capabilities), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(capabilities.format_count, 1u);
    EXPECT_EQ(capabilities.formats[0], VERNON_RHI_FORMAT_RGBA8_UNORM);
    EXPECT_EQ(capabilities.usage, VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    EXPECT_EQ(capabilities.present_modes, 0b111u);
    EXPECT_EQ(capabilities.current_width, kWidth);
    EXPECT_EQ(capabilities.current_height, kHeight);
    const VernonRhiSurfaceConfiguration configured = configuration(2);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK)
        << vernon::test::text(vernonRhiDeviceGetLastError(device()));

    const std::array<std::array<uint8_t, 4>, 3> colors{{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}}};
    std::vector<VernonRhiImage> acquired;
    for (const auto &color : colors) {
        VernonRhiSurfaceImage image{};
        ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(image.suboptimal, 0u);
        acquired.push_back(image.image);
        clear({image.view}, {color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f, color[3] / 255.0f});
        std::vector<uint8_t> texels;
        ASSERT_EQ(download(image.image, texels), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device()));
        EXPECT_EQ(texels, filled(color));
        ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK)
            << vernon::test::text(vernonRhiDeviceGetLastError(device()));
    }
    EXPECT_NE(acquired[0].index, acquired[1].index);
    EXPECT_EQ(acquired[0].index, acquired[2].index);
    EXPECT_EQ(acquired[0].generation, acquired[2].generation);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceIsImageValid(device(), acquired[0]), 0u);
}

TEST_P(RhiSurface, PresentShowsTheImageTopRowAtTheTopOfTheWindow) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    const VernonRhiSurfaceConfiguration configured = configuration(1);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK);
    VernonRhiSurfaceImage image{};
    ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
    const VernonRhiCommandEncoder encoder = beginEncoder();
    ASSERT_EQ(beginRendering(encoder, image.view, VERNON_RHI_LOAD_CLEAR, {1.0f, 0.0f, 0.0f, 1.0f}),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), encoder), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(beginRendering(encoder, image.view, VERNON_RHI_LOAD_CLEAR, {0.0f, 1.0f, 0.0f, 1.0f}, 0, 1),
              VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), encoder), VERNON_RHI_STATUS_OK);
    submit(encoder);
    const std::array<uint8_t, 4> green{0, 255, 0, 255};
    const std::array<uint8_t, 4> red{255, 0, 0, 255};
    std::vector<uint8_t> topRow;
    std::vector<uint8_t> bottomRow;
    for (uint32_t column = 0; column < kWidth; ++column) {
        topRow.insert(topRow.end(), green.begin(), green.end());
        bottomRow.insert(bottomRow.end(), red.begin(), red.end());
    }
    std::vector<uint8_t> texels;
    ASSERT_EQ(download(image.image, texels), VERNON_RHI_STATUS_OK);
    std::vector<uint8_t> imageRows = topRow;
    imageRows.insert(imageRows.end(), bottomRow.begin(), bottomRow.end());
    ASSERT_EQ(texels, imageRows);

    ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK)
        << vernon::test::text(vernonRhiDeviceGetLastError(device()));
    EXPECT_EQ(windowReadError_, 0u);
    std::vector<uint8_t> windowRows = bottomRow;
    windowRows.insert(windowRows.end(), topRow.begin(), topRow.end());
    EXPECT_EQ(windowTexels_, windowRows);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiSurface, TwoSurfacesPresentImagesWrittenByOneSubmission) {
    std::array<VernonRhiSurface, 2> surfaces{};
    for (VernonRhiSurface &surface : surfaces)
        if (!createSurfaceOrExpectUnsupported(surface))
            return;
    const VernonRhiSurfaceConfiguration configured = configuration(1);
    std::vector<VernonRhiImage> images;
    std::vector<VernonRhiImageView> views;
    for (const VernonRhiSurface surface : surfaces) {
        ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK);
        VernonRhiSurfaceImage image{};
        ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
        images.push_back(image.image);
        views.push_back(image.view);
    }
    clear(views, {1.0f, 1.0f, 0.0f, 1.0f});
    for (const VernonRhiImage image : images) {
        std::vector<uint8_t> texels;
        ASSERT_EQ(download(image, texels), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(texels, filled({255, 255, 0, 255}));
    }
    for (const VernonRhiSurface surface : surfaces) {
        EXPECT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
    }
}

TEST_P(RhiSurface, RejectsReadsOfAnImageUntilItIsWritten) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    const VernonRhiSurfaceConfiguration configured = configuration(1);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK);
    const VernonRhiImage destination = createCopyDestination();

    // The only image is acquired again in the second frame, and its contents are undefined again.
    for (int frame = 0; frame < 2; ++frame) {
        VernonRhiSurfaceImage image{};
        ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
        std::vector<uint8_t> texels;
        EXPECT_EQ(download(image.image, texels), VERNON_RHI_STATUS_INVALID_ARGUMENT);
        const VernonRhiCommandEncoder encoder = beginEncoder();
        EXPECT_EQ(beginRendering(encoder, image.view, VERNON_RHI_LOAD_PRESERVE), VERNON_RHI_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(copy(encoder, image.image, destination), VERNON_RHI_STATUS_INVALID_ARGUMENT);
        ASSERT_EQ(beginRendering(encoder, image.view, VERNON_RHI_LOAD_CLEAR), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), encoder), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(beginRendering(encoder, image.view, VERNON_RHI_LOAD_PRESERVE), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), encoder), VERNON_RHI_STATUS_OK);
        EXPECT_EQ(copy(encoder, image.image, destination), VERNON_RHI_STATUS_OK);
        submit(encoder);
        EXPECT_EQ(download(image.image, texels), VERNON_RHI_STATUS_OK);
        ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK);
    }
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device(), destination), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiSurface, IgnoresWritesOfAnEncoderThatIsNeverSubmitted) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    const VernonRhiSurfaceConfiguration configured = configuration(1);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK);
    const VernonRhiImage destination = createCopyDestination();
    VernonRhiSurfaceImage image{};
    ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);

    const VernonRhiCommandEncoder abandoned = beginEncoder();
    ASSERT_EQ(beginRendering(abandoned, image.view, VERNON_RHI_LOAD_CLEAR), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), abandoned), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderFinish(device(), abandoned), VERNON_RHI_STATUS_OK);
    std::vector<uint8_t> texels;
    EXPECT_EQ(download(image.image, texels), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(vernonRhiDeviceDestroyCommandEncoder(device(), abandoned), VERNON_RHI_STATUS_OK);

    const VernonRhiCommandEncoder reader = beginEncoder();
    EXPECT_EQ(beginRendering(reader, image.view, VERNON_RHI_LOAD_PRESERVE), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(copy(reader, image.image, destination), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(beginRendering(reader, image.view, VERNON_RHI_LOAD_CLEAR), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiCommandEncoderEndRendering(device(), reader), VERNON_RHI_STATUS_OK);
    submit(reader);
    EXPECT_EQ(download(image.image, texels), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device(), destination), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiSurface, RejectsOperationsOutOfOrder) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    VernonRhiSurfaceImage image{};
    EXPECT_EQ(acquire(surface, image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    const VernonRhiSurfaceConfiguration configured = configuration(2);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK);
    VernonRhiSurfaceImage unsized{};
    EXPECT_EQ(vernonRhiSurfaceAcquireImage(device(), surface, &unsized), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
    VernonRhiSurfaceImage second{};
    EXPECT_EQ(acquire(surface, second), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyImageView(device(), image.view), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroyImage(device(), image.image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceIsImageValid(device(), image.image), 1u);
    ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(acquire(surface, image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
}

TEST_P(RhiSurface, RejectsDescriptorsAndConfigurationsOutsideTheCapabilities) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    VernonRhiSurface mismatched{};
    EXPECT_EQ(createSurface(mismatched, VERNON_RHI_BACKEND_VULKAN), VERNON_RHI_STATUS_INVALID_ARGUMENT);

    VernonRhiSurfaceConfiguration zeroExtent = configuration(1, 0, kHeight);
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &zeroExtent), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    VernonRhiSurfaceConfiguration noColorAttachment = configuration(1);
    noColorAttachment.usage = VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &noColorAttachment), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    VernonRhiSurfaceConfiguration unknownMode = configuration(1);
    unknownMode.present_mode = static_cast<VernonRhiPresentMode>(3);
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &unknownMode), VERNON_RHI_STATUS_INVALID_ARGUMENT);

    VernonRhiSurfaceConfiguration srgb = configuration(1);
    srgb.format = VERNON_RHI_FORMAT_RGBA8_SRGB;
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &srgb), VERNON_RHI_STATUS_UNSUPPORTED);
    VernonRhiSurfaceConfiguration sampled = configuration(1);
    sampled.usage |= VERNON_RHI_IMAGE_SAMPLED;
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &sampled), VERNON_RHI_STATUS_UNSUPPORTED);
    VernonRhiSurfaceConfiguration noImages = configuration(0);
    EXPECT_EQ(vernonRhiSurfaceConfigure(device(), surface, &noImages), VERNON_RHI_STATUS_UNSUPPORTED);

    VernonRhiSurfaceImage image{};
    EXPECT_EQ(acquire(surface, image), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
}

TEST_P(RhiSurface, ReportsOutOfDateAfterAResizeUntilConfiguredAgain) {
    VernonRhiSurface surface{};
    if (!createSurfaceOrExpectUnsupported(surface))
        return;
    const VernonRhiSurfaceConfiguration configured = configuration(2);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &configured), VERNON_RHI_STATUS_OK);
    VernonRhiSurfaceImage image{};
    ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK);

    drawableWidth_ = 2 * kWidth;
    drawableHeight_ = 3 * kHeight;
    EXPECT_EQ(acquire(surface, image), VERNON_RHI_STATUS_SURFACE_OUT_OF_DATE);
    EXPECT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_INVALID_ARGUMENT);
    VernonRhiSurfaceCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    ASSERT_EQ(vernonRhiSurfaceGetCapabilities(device(), surface, &capabilities), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(capabilities.current_width, drawableWidth_);
    EXPECT_EQ(capabilities.current_height, drawableHeight_);
    const VernonRhiSurfaceConfiguration resized = configuration(2, drawableWidth_, drawableHeight_);
    ASSERT_EQ(vernonRhiSurfaceConfigure(device(), surface, &resized), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(acquire(surface, image), VERNON_RHI_STATUS_OK);
    ASSERT_EQ(vernonRhiSurfacePresent(device(), surface), VERNON_RHI_STATUS_OK);
    EXPECT_EQ(vernonRhiDeviceDestroySurface(device(), surface), VERNON_RHI_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(Backends, RhiSurface, testing::ValuesIn(vernon::tests::rhiBackendCases()),
                         [](const testing::TestParamInfo<vernon::tests::BackendTestRow> &info) {
                             return std::string(info.param.name);
                         });

} // namespace
