#include "graphics_host.h"

#include "embedded_bundles.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <string>

namespace {

constexpr std::array<float, 6> kFullscreenTriangle{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
constexpr uint64_t kTriangleShape[] = {3, 2};
constexpr int64_t kTriangleStrides[] = {2 * sizeof(float), sizeof(float)};

GLFWwindow *window(void *userData) { return static_cast<GLFWwindow *>(userData); }

void makeCurrent(void *userData) { glfwMakeContextCurrent(window(userData)); }

void *getProcAddress(void *, const char *name) { return reinterpret_cast<void *>(glfwGetProcAddress(name)); }

void swapInterval(void *, int32_t interval) {
#if defined(__EMSCRIPTEN__)
    // The browser paces frames through the main loop instead.
    (void)interval;
#else
    glfwSwapInterval(interval);
#endif
}

void swapBuffers(void *userData) { glfwSwapBuffers(window(userData)); }

void drawableSize(void *userData, uint32_t *width, uint32_t *height) {
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window(userData), &framebufferWidth, &framebufferHeight);
    *width = static_cast<uint32_t>(std::max(framebufferWidth, 0));
    *height = static_cast<uint32_t>(std::max(framebufferHeight, 0));
}

std::string halError() {
    const VernonStringView error = vernonHalGetLastError();
    return error.data && error.size ? std::string(error.data, error.size) : std::string("unknown HAL error");
}

} // namespace

GraphicsHost::~GraphicsHost() { shutdown(); }

bool GraphicsHost::initialize(const char *title, uint32_t width, uint32_t height) {
    if (glfwInit() != GLFW_TRUE) {
        std::cerr << "failed to initialize GLFW\n";
        return false;
    }
    glfwInitialized_ = true;
#if defined(__EMSCRIPTEN__)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#else
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
#endif
    window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title, nullptr, nullptr);
    if (!window_) {
        std::cerr << "failed to create a GLFW graphics context\n";
        shutdown();
        return false;
    }
    glfwMakeContextCurrent(window_);

    VernonOpenGLContextCallbacks context{};
    context.struct_size = sizeof(context);
    context.user_data = window_;
    context.make_current = &makeCurrent;
    context.get_proc_address = &getProcAddress;
    context.api_version_major = 3;
    VernonHalDeviceDescriptor deviceDescriptor{};
    deviceDescriptor.struct_size = sizeof(deviceDescriptor);
#if defined(__EMSCRIPTEN__)
    context.api_version_minor = 0;
    deviceDescriptor.backend = VERNON_RUNTIME_OPENGL_ES;
#else
    context.api_version_minor = 3;
    deviceDescriptor.backend = VERNON_RUNTIME_OPENGL;
#endif
    deviceDescriptor.opengl = &context;
    if (vernonHalCreateDevice(&deviceDescriptor, &device_) != VERNON_HAL_OK) {
        std::cerr << "failed to create the Vernon OpenGL device: " << halError() << '\n';
        shutdown();
        return false;
    }

    VernonOpenGLSurfaceCallbacks presentation{};
    presentation.struct_size = sizeof(presentation);
    presentation.user_data = window_;
    presentation.make_current = &makeCurrent;
    presentation.swap_interval = &swapInterval;
    presentation.swap_buffers = &swapBuffers;
    presentation.drawable_size = &drawableSize;
    VernonHalSurfaceDescriptor surfaceDescriptor{};
    surfaceDescriptor.struct_size = sizeof(surfaceDescriptor);
    surfaceDescriptor.native.opengl = &presentation;
    if (vernonHalCreateSurface(device_, &surfaceDescriptor, &surface_) != VERNON_HAL_OK) {
        std::cerr << "failed to create the window surface: " << halError() << '\n';
        shutdown();
        return false;
    }

    VernonHalBufferDescriptor vertexDescriptor{};
    vertexDescriptor.struct_size = sizeof(vertexDescriptor);
    vertexDescriptor.size = sizeof(kFullscreenTriangle);
    vertexDescriptor.usage = VERNON_HAL_BUFFER_VERTEX;
    const VernonHalBufferWrite vertexWrite{0, kFullscreenTriangle.data(), sizeof(kFullscreenTriangle)};
    VernonHalSamplerDescriptor samplerDescriptor{};
    samplerDescriptor.struct_size = sizeof(samplerDescriptor);
    samplerDescriptor.min_filter = VERNON_HAL_FILTER_LINEAR;
    samplerDescriptor.mag_filter = VERNON_HAL_FILTER_LINEAR;
    samplerDescriptor.address_u = VERNON_HAL_ADDRESS_CLAMP_TO_EDGE;
    samplerDescriptor.address_v = VERNON_HAL_ADDRESS_CLAMP_TO_EDGE;
    samplerDescriptor.address_w = VERNON_HAL_ADDRESS_CLAMP_TO_EDGE;
    if (vernonHalCreateBuffer(device_, &vertexDescriptor, &vertices_) != VERNON_HAL_OK ||
        vernonHalBufferWrite(vertices_, &vertexWrite, 1) != VERNON_HAL_OK ||
        vernonHalReferenceBuffer(vertices_, 0, sizeof(kFullscreenTriangle), &verticesReference_) != VERNON_HAL_OK ||
        vernonHalCreateSampler(device_, &samplerDescriptor, &sampler_) != VERNON_HAL_OK ||
        vernonHalReferenceSampler(sampler_, &samplerReference_) != VERNON_HAL_OK) {
        std::cerr << "failed to create the compositor resources: " << halError() << '\n';
        shutdown();
        return false;
    }

    std::string error;
    if (!compositor_.link(runtime(), vernon_external_engine::panel_bundle::kManifest,
                          vernon_external_engine::panel_bundle::kManifestSize,
                          vernon_external_engine::panel_bundle::kCookedDirectory, error) ||
        !compositor_.bindDeviceTensor("position", verticesReference_, sizeof(kFullscreenTriangle), 2, kTriangleShape,
                                      kTriangleStrides) ||
        !compositor_.bindSampler("panel_sampler", samplerReference_)) {
        std::cerr << "failed to link the panel compositor: " << error << '\n';
        shutdown();
        return false;
    }
    passSlot_ = compositor_.slotOfKind(VERNON_SCHEDULE_SLOT_PASS);
    drawSlot_ = compositor_.slotOfKind(VERNON_SCHEDULE_SLOT_DRAW);
    dynamicSlot_ = compositor_.slotOfKind(VERNON_SCHEDULE_SLOT_DYNAMIC);
    if (passSlot_ == SIZE_MAX || drawSlot_ == SIZE_MAX || dynamicSlot_ == SIZE_MAX) {
        std::cerr << "the panel compositor is not a graphics schedule\n";
        shutdown();
        return false;
    }
    attachment_.location = 0;
    attachment_.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
    attachment_.clear_color.f32[3] = 1.0F;
    renderPass_.struct_size = sizeof(renderPass_);
    renderPass_.color_attachments = &attachment_;
    renderPass_.color_attachment_count = 1;
    compositor_.write(passSlot_, static_cast<const VernonRenderPass *>(&renderPass_));
    VernonDrawCommand draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = 3;
    draw.instance_count = 1;
    compositor_.write(drawSlot_, draw);
    return true;
}

bool GraphicsHost::configureSurface(uint32_t width, uint32_t height) {
    VernonHalSurfaceCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    if (vernonHalSurfaceGetCapabilities(surface_, &capabilities) != VERNON_HAL_OK)
        return false;
    VernonHalSurfaceConfiguration configuration{};
    configuration.struct_size = sizeof(configuration);
    configuration.format = VERNON_TEXTURE_RGBA8_UNORM;
    configuration.usage = VERNON_IMAGE_COLOR_ATTACHMENT;
    configuration.width = width;
    configuration.height = height;
    configuration.present_mode = VERNON_HAL_PRESENT_FIFO;
    configuration.image_count = std::max(capabilities.min_image_count, 2u);
    if (capabilities.max_image_count)
        configuration.image_count = std::min(configuration.image_count, capabilities.max_image_count);
    if (vernonHalSurfaceConfigure(surface_, &configuration) != VERNON_HAL_OK)
        return false;
    surfaceWidth_ = width;
    surfaceHeight_ = height;
    return true;
}

bool GraphicsHost::drawPanel(VernonRuntimeProviderResourceReference target, const VernonHalImageView *panel, uint32_t x,
                             uint32_t width, uint32_t height, bool clear) {
    const VernonImageDescriptor *source = vernonHalImageGetDescriptor(vernonHalImageViewGetImage(panel));
    VernonRuntimeProviderResourceReference panelReference{};
    if (!source || !source->extent.width || !source->extent.height ||
        vernonHalReferenceImageView(panel, &panelReference) != VERNON_HAL_OK)
        return false;
    const double sourceAspect = static_cast<double>(source->extent.width) / source->extent.height;
    uint32_t fitWidth = width;
    uint32_t fitHeight = height;
    if (sourceAspect > static_cast<double>(width) / height)
        fitHeight = std::max(1u, static_cast<uint32_t>(width / sourceAspect));
    else
        fitWidth = std::max(1u, static_cast<uint32_t>(height * sourceAspect));

    attachment_.view = target;
    attachment_.load_operation = clear ? VERNON_RUNTIME_PROVIDER_LOAD_CLEAR : VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE;
    renderPass_.render_area[0] = 0;
    renderPass_.render_area[1] = 0;
    renderPass_.render_area[2] = surfaceWidth_;
    renderPass_.render_area[3] = surfaceHeight_;
    VernonDynamicState dynamic{};
    dynamic.struct_size = sizeof(dynamic);
    dynamic.viewport[0] = x + (width - fitWidth) / 2;
    dynamic.viewport[1] = (height - fitHeight) / 2;
    dynamic.viewport[2] = fitWidth;
    dynamic.viewport[3] = fitHeight;
    std::copy(std::begin(dynamic.viewport), std::end(dynamic.viewport), std::begin(dynamic.scissor));
    compositor_.write(dynamicSlot_, dynamic);
    if (!compositor_.bindImage("panel", panelReference))
        return false;
    std::string error;
    if (!compositor_.call(error)) {
        std::cerr << "failed to draw a panel: " << error << '\n';
        return false;
    }
    return true;
}

bool GraphicsHost::presentSplit(const VernonHalImageView *leftView, const VernonHalImageView *rightView) {
    uint32_t width = 0;
    uint32_t height = 0;
    if (!surface_ || !leftView || !rightView || !framebufferSize(width, height))
        return false;
    if ((width != surfaceWidth_ || height != surfaceHeight_) && !configureSurface(width, height))
        return false;
    VernonHalSurfaceImage image{};
    VernonHalStatus acquired = vernonHalSurfaceAcquire(surface_, &image);
    if (acquired == VERNON_HAL_SURFACE_OUT_OF_DATE && configureSurface(width, height))
        acquired = vernonHalSurfaceAcquire(surface_, &image);
    VernonRuntimeProviderResourceReference target{};
    if (acquired != VERNON_HAL_OK || vernonHalReferenceImageView(image.view, &target) != VERNON_HAL_OK) {
        std::cerr << "failed to acquire the window surface: " << halError() << '\n';
        return false;
    }
    const uint32_t leftWidth = width / 2;
    const bool drawn = drawPanel(target, leftView, 0, leftWidth, height, true) &&
                       drawPanel(target, rightView, leftWidth, width - leftWidth, height, false);
    const bool presented = vernonHalSurfacePresent(surface_) == VERNON_HAL_OK;
    if (!presented)
        std::cerr << "failed to present the window surface: " << halError() << '\n';
    return drawn && presented;
}

bool GraphicsHost::createPanelImage(uint32_t width, uint32_t height, uint32_t usage, VernonHalImage *&image,
                                    VernonHalImageView *&view) const {
    destroyPanelImage(image, view);
    const VernonImageDescriptor imageDescriptor{
        VERNON_TEXTURE_2D, {width, height, 1}, VERNON_TEXTURE_RGBA8_UNORM, 1, 1, 1, usage | VERNON_IMAGE_SAMPLED};
    const VernonImageViewDescriptor viewDescriptor{
        VERNON_TEXTURE_2D, VERNON_TEXTURE_RGBA8_UNORM, {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR}};
    if (device_ && vernonHalCreateImage(device_, &imageDescriptor, &image) == VERNON_HAL_OK &&
        vernonHalCreateImageView(image, &viewDescriptor, &view) == VERNON_HAL_OK)
        return true;
    std::cerr << "failed to create a panel image: " << halError() << '\n';
    destroyPanelImage(image, view);
    return false;
}

void GraphicsHost::destroyPanelImage(VernonHalImage *&image, VernonHalImageView *&view) {
    if (view)
        (void)vernonHalDestroyImageView(view);
    view = nullptr;
    if (image)
        (void)vernonHalDestroyImage(image);
    image = nullptr;
}

bool GraphicsHost::framebufferSize(uint32_t &width, uint32_t &height) const {
    if (!window_)
        return false;
    drawableSize(window_, &width, &height);
    return width && height;
}

void GraphicsHost::pollEvents() { glfwPollEvents(); }

bool GraphicsHost::shouldClose() const { return !window_ || glfwWindowShouldClose(window_) != 0; }

void GraphicsHost::shutdown() {
    compositor_.reset();
    if (sampler_)
        vernonHalDestroySampler(sampler_);
    sampler_ = nullptr;
    if (vertices_)
        vernonHalDestroyBuffer(vertices_);
    vertices_ = nullptr;
    if (surface_)
        (void)vernonHalDestroySurface(surface_);
    surface_ = nullptr;
    surfaceWidth_ = 0;
    surfaceHeight_ = 0;
    if (device_ && vernonHalDestroyDevice(device_) != VERNON_HAL_OK)
        std::cerr << "failed to destroy the Vernon OpenGL device: " << halError() << '\n';
    device_ = nullptr;
    if (window_)
        glfwDestroyWindow(window_);
    window_ = nullptr;
    if (glfwInitialized_)
        glfwTerminate();
    glfwInitialized_ = false;
}
