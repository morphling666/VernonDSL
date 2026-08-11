#include "engine_demo.h"
#include "graphics_host.h"

#include <iostream>
#include <memory>

namespace {

constexpr uint32_t kWindowWidth = 1440;
constexpr uint32_t kWindowHeight = 450;

class SplitScreenDemo final : public ExternalEngineDemo {
public:
    ~SplitScreenDemo() override { shutdown(); }

    bool initialize(bool headless) override {
        headless_ = headless;
        cpu_ = createCpuFractalPanel();
        if (!cpu_)
            return false;
        if (headless_)
            return cpu_->initialize(nullptr, true);

        if (!graphics_.initialize("Vernon CPU + GPU Execution Graphs", kWindowWidth, kWindowHeight))
            return false;
        mandelbulb_ = createMandelbulbPanel();
        if (!mandelbulb_ || !cpu_->initialize(&graphics_, false) || !mandelbulb_->initialize(&graphics_, false))
            return false;
        return true;
    }

    bool renderFrame(double elapsedSeconds) override {
        if (headless_)
            return cpu_->renderFrame(elapsedSeconds, 0, 0);

        uint32_t framebufferWidth = 0;
        uint32_t framebufferHeight = 0;
        if (!graphics_.framebufferSize(framebufferWidth, framebufferHeight)) {
            graphics_.pollEvents();
            return true;
        }
        const uint32_t rightPanelWidth = framebufferWidth - framebufferWidth / 2;
        if (!cpu_->renderFrame(elapsedSeconds, framebufferWidth / 2, framebufferHeight) ||
            !mandelbulb_->renderFrame(elapsedSeconds, rightPanelWidth, framebufferHeight) ||
            !graphics_.presentSplit(cpu_->image(), cpu_->imageWidth(), cpu_->imageHeight(), mandelbulb_->image(),
                                    mandelbulb_->imageWidth(), mandelbulb_->imageHeight()))
            return false;
        graphics_.pollEvents();
        if (frame_++ == 0)
            std::cout << "Vernon split-screen CPU and GPU animation started\n";
        return true;
    }

    bool shouldClose() const override { return !headless_ && graphics_.shouldClose(); }

    void shutdown() override {
        if (mandelbulb_)
            mandelbulb_->shutdown();
        if (cpu_)
            cpu_->shutdown();
        mandelbulb_.reset();
        cpu_.reset();
        graphics_.shutdown();
    }

private:
    bool headless_{};
    GraphicsHost graphics_;
    std::unique_ptr<ExternalEnginePanel> cpu_;
    std::unique_ptr<ExternalEnginePanel> mandelbulb_;
    uint64_t frame_{};
};

} // namespace

std::unique_ptr<ExternalEngineDemo> createExternalEngineDemo() { return std::make_unique<SplitScreenDemo>(); }
