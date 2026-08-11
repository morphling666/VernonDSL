#include "graphics_host.h"

#include <cstddef>
#include <iostream>

namespace {

using GlEnum = unsigned int;
using GlInt = int;
using GlSizei = int;
using GlUint = unsigned int;

constexpr GlEnum kReadFramebuffer = 0x8CA8;
constexpr GlEnum kDrawFramebuffer = 0x8CA9;
constexpr GlEnum kFramebufferComplete = 0x8CD5;
constexpr GlEnum kColorAttachment0 = 0x8CE0;
constexpr GlEnum kTexture2d = 0x0DE1;
constexpr GlEnum kColorBufferBit = 0x00004000;
constexpr GlEnum kScissorTest = 0x0C11;
constexpr GlEnum kLinear = 0x2601;
constexpr GlEnum kNoError = 0;

void makeCurrent(void *userData) { glfwMakeContextCurrent(static_cast<GLFWwindow *>(userData)); }

void *getProcAddress(void *, const char *name) { return reinterpret_cast<void *>(glfwGetProcAddress(name)); }

} // namespace

struct GraphicsHost::PresentationState {
    using GenFramebuffers = void (*)(GlSizei, GlUint *);
    using BindFramebuffer = void (*)(GlEnum, GlUint);
    using FramebufferTexture2D = void (*)(GlEnum, GlEnum, GlEnum, GlUint, GlInt);
    using CheckFramebufferStatus = GlEnum (*)(GlEnum);
    using BlitFramebuffer = void (*)(GlInt, GlInt, GlInt, GlInt, GlInt, GlInt, GlInt, GlInt, GlEnum, GlEnum);
    using DeleteFramebuffers = void (*)(GlSizei, const GlUint *);
    using ReadBuffer = void (*)(GlEnum);
    using Disable = void (*)(GlEnum);
    using ClearColor = void (*)(float, float, float, float);
    using Clear = void (*)(GlEnum);
    using GetError = GlEnum (*)();

    GenFramebuffers genFramebuffers{};
    BindFramebuffer bindFramebuffer{};
    FramebufferTexture2D framebufferTexture2D{};
    CheckFramebufferStatus checkFramebufferStatus{};
    BlitFramebuffer blitFramebuffer{};
    DeleteFramebuffers deleteFramebuffers{};
    ReadBuffer readBuffer{};
    Disable disable{};
    ClearColor clearColor{};
    Clear clear{};
    GetError getError{};
    GlUint framebuffers[2]{};

    bool initialize() {
        genFramebuffers = reinterpret_cast<GenFramebuffers>(glfwGetProcAddress("glGenFramebuffers"));
        bindFramebuffer = reinterpret_cast<BindFramebuffer>(glfwGetProcAddress("glBindFramebuffer"));
        framebufferTexture2D = reinterpret_cast<FramebufferTexture2D>(glfwGetProcAddress("glFramebufferTexture2D"));
        checkFramebufferStatus =
            reinterpret_cast<CheckFramebufferStatus>(glfwGetProcAddress("glCheckFramebufferStatus"));
        blitFramebuffer = reinterpret_cast<BlitFramebuffer>(glfwGetProcAddress("glBlitFramebuffer"));
        deleteFramebuffers = reinterpret_cast<DeleteFramebuffers>(glfwGetProcAddress("glDeleteFramebuffers"));
        readBuffer = reinterpret_cast<ReadBuffer>(glfwGetProcAddress("glReadBuffer"));
        disable = reinterpret_cast<Disable>(glfwGetProcAddress("glDisable"));
        clearColor = reinterpret_cast<ClearColor>(glfwGetProcAddress("glClearColor"));
        clear = reinterpret_cast<Clear>(glfwGetProcAddress("glClear"));
        getError = reinterpret_cast<GetError>(glfwGetProcAddress("glGetError"));
        if (!genFramebuffers || !bindFramebuffer || !framebufferTexture2D || !checkFramebufferStatus ||
            !blitFramebuffer || !deleteFramebuffers || !readBuffer || !disable || !clearColor || !clear || !getError)
            return false;
        genFramebuffers(2, framebuffers);
        return framebuffers[0] && framebuffers[1] && getError() == kNoError;
    }

    void shutdown() {
        if (deleteFramebuffers && (framebuffers[0] || framebuffers[1]))
            deleteFramebuffers(2, framebuffers);
        framebuffers[0] = 0;
        framebuffers[1] = 0;
    }
};

GraphicsHost::GraphicsHost() = default;

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
        glfwTerminate();
        glfwInitialized_ = false;
        return false;
    }
    glfwMakeContextCurrent(window_);
#if !defined(__EMSCRIPTEN__)
    glfwSwapInterval(1);
#endif

    VernonOpenGLContextCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.user_data = window_;
    callbacks.make_current = &makeCurrent;
    callbacks.get_proc_address = &getProcAddress;
#if defined(__EMSCRIPTEN__)
    callbacks.api_version_major = 3;
    callbacks.api_version_minor = 0;
    device_ = vernonRhiCreateOpenGLDevice(&callbacks, 1);
#else
    callbacks.api_version_major = 3;
    callbacks.api_version_minor = 3;
    device_ = vernonRhiCreateOpenGLDevice(&callbacks, 0);
#endif
    if (device_.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        std::cerr << "failed to create the Vernon OpenGL device\n";
        shutdown();
        return false;
    }
    presentation_ = std::make_unique<PresentationState>();
    if (!presentation_->initialize()) {
        std::cerr << "failed to initialize OpenGL presentation resources\n";
        shutdown();
        return false;
    }
    return true;
}

bool GraphicsHost::presentSplit(VernonRhiImage leftImage, uint32_t leftWidth, uint32_t leftHeight,
                                VernonRhiImage rightImage, uint32_t rightWidth, uint32_t rightHeight) {
    if (!presentation_)
        return false;
    PresentationState &gl = *presentation_;

    uint32_t framebufferWidth = 0;
    uint32_t framebufferHeight = 0;
    if (!framebufferSize(framebufferWidth, framebufferHeight))
        return false;
    uint64_t nativeTextures[2]{};
    if (vernonRhiDeviceGetImageNativeHandle(device_, leftImage, &nativeTextures[0]) != VERNON_RHI_STATUS_OK ||
        vernonRhiDeviceGetImageNativeHandle(device_, rightImage, &nativeTextures[1]) != VERNON_RHI_STATUS_OK)
        return false;

    struct Rect {
        GlInt x0;
        GlInt y0;
        GlInt x1;
        GlInt y1;
    };
    const auto fit = [](uint32_t panelX, uint32_t panelWidth, uint32_t panelHeight, uint32_t sourceWidth,
                        uint32_t sourceHeight) {
        const double sourceAspect = static_cast<double>(sourceWidth) / sourceHeight;
        const double panelAspect = static_cast<double>(panelWidth) / panelHeight;
        uint32_t width = panelWidth;
        uint32_t height = panelHeight;
        if (sourceAspect > panelAspect)
            height = static_cast<uint32_t>(panelWidth / sourceAspect);
        else
            width = static_cast<uint32_t>(panelHeight * sourceAspect);
        const uint32_t x = panelX + (panelWidth - width) / 2;
        const uint32_t y = (panelHeight - height) / 2;
        return Rect{static_cast<GlInt>(x), static_cast<GlInt>(y), static_cast<GlInt>(x + width),
                    static_cast<GlInt>(y + height)};
    };
    const uint32_t leftPanelWidth = framebufferWidth / 2;
    const Rect destinations[] = {
        fit(0, leftPanelWidth, framebufferHeight, leftWidth, leftHeight),
        fit(leftPanelWidth, framebufferWidth - leftPanelWidth, framebufferHeight, rightWidth, rightHeight)};
    const uint32_t sourceWidths[] = {leftWidth, rightWidth};
    const uint32_t sourceHeights[] = {leftHeight, rightHeight};

    gl.bindFramebuffer(kDrawFramebuffer, 0);
    gl.disable(kScissorTest);
    gl.clearColor(0.0F, 0.0F, 0.0F, 1.0F);
    gl.clear(kColorBufferBit);
    for (size_t index = 0; index < 2; ++index) {
        gl.bindFramebuffer(kReadFramebuffer, gl.framebuffers[index]);
        gl.framebufferTexture2D(kReadFramebuffer, kColorAttachment0, kTexture2d,
                                static_cast<GlUint>(nativeTextures[index]), 0);
        gl.readBuffer(kColorAttachment0);
        if (gl.checkFramebufferStatus(kReadFramebuffer) != kFramebufferComplete) {
            gl.bindFramebuffer(kReadFramebuffer, 0);
            return false;
        }
        gl.bindFramebuffer(kDrawFramebuffer, 0);
        const Rect &destination = destinations[index];
        gl.blitFramebuffer(0, 0, static_cast<GlInt>(sourceWidths[index]), static_cast<GlInt>(sourceHeights[index]),
                           destination.x0, destination.y0, destination.x1, destination.y1, kColorBufferBit, kLinear);
    }
    gl.bindFramebuffer(kReadFramebuffer, 0);
    glfwSwapBuffers(window_);
    return gl.getError() == kNoError;
}

bool GraphicsHost::framebufferSize(uint32_t &width, uint32_t &height) const {
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    if (window_)
        glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0)
        return false;
    width = static_cast<uint32_t>(framebufferWidth);
    height = static_cast<uint32_t>(framebufferHeight);
    return true;
}

void GraphicsHost::pollEvents() { glfwPollEvents(); }

bool GraphicsHost::shouldClose() const { return !window_ || glfwWindowShouldClose(window_) != 0; }

void GraphicsHost::shutdown() {
    if (presentation_) {
        if (window_)
            glfwMakeContextCurrent(window_);
        presentation_->shutdown();
        presentation_.reset();
    }
    if (device_.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
        vernonRhiDestroyDevice(device_);
        device_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    }
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    if (glfwInitialized_) {
        glfwTerminate();
        glfwInitialized_ = false;
    }
}
