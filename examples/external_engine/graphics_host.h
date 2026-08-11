#ifndef VERNON_EXTERNAL_ENGINE_GRAPHICS_HOST_H
#define VERNON_EXTERNAL_ENGINE_GRAPHICS_HOST_H

#include "VernonRHI.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdint>
#include <memory>

class GraphicsHost {
public:
    GraphicsHost();
    ~GraphicsHost();

    bool initialize(const char *title, uint32_t width, uint32_t height);
    bool presentSplit(VernonRhiImage leftImage, uint32_t leftWidth, uint32_t leftHeight, VernonRhiImage rightImage,
                      uint32_t rightWidth, uint32_t rightHeight);
    bool framebufferSize(uint32_t &width, uint32_t &height) const;
    void pollEvents();
    bool shouldClose() const;
    void shutdown();

    VernonRhiDevice device() const { return device_; }

private:
    struct PresentationState;

    bool glfwInitialized_{};
    GLFWwindow *window_{};
    VernonRhiDevice device_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::unique_ptr<PresentationState> presentation_;
};

#endif
