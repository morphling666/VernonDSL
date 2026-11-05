#ifndef VERNON_EXTERNAL_ENGINE_GRAPHICS_HOST_H
#define VERNON_EXTERNAL_ENGINE_GRAPHICS_HOST_H

#include "VernonHal.h"
#include "linked_program.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdint>

class GraphicsHost {
public:
    GraphicsHost() = default;
    GraphicsHost(const GraphicsHost &) = delete;
    GraphicsHost &operator=(const GraphicsHost &) = delete;
    ~GraphicsHost();

    bool initialize(const char *title, uint32_t width, uint32_t height);
    // Draws each panel's sampled image view aspect-fit into its half of the window, then presents.
    bool presentSplit(const VernonHalImageView *leftView, const VernonHalImageView *rightView);
    bool framebufferSize(uint32_t &width, uint32_t &height) const;
    void pollEvents();
    bool shouldClose() const;
    void shutdown();

    // An RGBA8 2D image on the device, sampled by the compositor, and a view of all of it.
    bool createPanelImage(uint32_t width, uint32_t height, uint32_t usage, VernonHalImage *&image,
                          VernonHalImageView *&view) const;
    static void destroyPanelImage(VernonHalImage *&image, VernonHalImageView *&view);

    VernonHalDevice *device() const { return device_; }
    VernonRuntimeContext *runtime() const { return device_ ? vernonHalDeviceGetRuntime(device_) : nullptr; }

private:
    bool configureSurface(uint32_t width, uint32_t height);
    bool drawPanel(VernonRuntimeProviderResourceReference target, const VernonHalImageView *panel, uint32_t x,
                   uint32_t width, uint32_t height, bool clear);

    bool glfwInitialized_{};
    GLFWwindow *window_{};
    VernonHalDevice *device_{};
    VernonHalSurface *surface_{};
    VernonHalBuffer *vertices_{};
    VernonHalSampler *sampler_{};
    VernonRuntimeProviderResourceReference verticesReference_{};
    VernonRuntimeProviderResourceReference samplerReference_{};
    vernon_external_engine::LinkedProgram compositor_;
    size_t passSlot_{SIZE_MAX};
    size_t drawSlot_{SIZE_MAX};
    size_t dynamicSlot_{SIZE_MAX};
    VernonColorAttachment attachment_{};
    VernonRenderPass renderPass_{};
    uint32_t surfaceWidth_{};
    uint32_t surfaceHeight_{};
};

#endif
