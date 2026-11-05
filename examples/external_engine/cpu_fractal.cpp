#include "engine_demo.h"
#include "graphics_host.h"
#include "linked_program.h"

#include "embedded_bundles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 320;
constexpr uint64_t kPixelShape[] = {kHeight, kWidth};
constexpr int64_t kPixelStrides[] = {kWidth * sizeof(float), sizeof(float)};
constexpr std::array<uint32_t, 3> kGrid{kWidth / 16, kHeight / 16, 1};
constexpr std::array<const char *, 3> kGridNames{"__grid_x", "__grid_y", "__grid_z"};

class CpuFractalPanel final : public ExternalEnginePanel {
public:
    ~CpuFractalPanel() override { shutdown(); }

    bool initialize(GraphicsHost *graphics, bool headless) override {
        graphics_ = graphics;
        headless_ = headless;
        VernonHalDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = VERNON_RUNTIME_CPU;
        if (vernonHalCreateDevice(&descriptor, &device_) != VERNON_HAL_OK) {
            std::cerr << "failed to create the CPU device\n";
            return false;
        }
        std::string error;
        if (!program_.link(vernonHalDeviceGetRuntime(device_), vernon_external_engine::cpu_bundle::kManifest,
                           vernon_external_engine::cpu_bundle::kManifestSize,
                           vernon_external_engine::cpu_bundle::kCookedDirectory, error)) {
            std::cerr << "failed to load the fractal Program: " << error << '\n';
            return false;
        }
        pixels_.resize(static_cast<size_t>(kWidth) * kHeight);
        bool bound = program_.bindHostTensor("pixels", pixels_.data(), pixels_.size() * sizeof(float), 2, kPixelShape,
                                             kPixelStrides) &&
                     program_.bindHostTensor("time", &time_, sizeof(time_));
        for (size_t axis = 0; axis < kGrid.size(); ++axis)
            bound = bound && program_.bindHostTensor(kGridNames[axis], &kGrid[axis], sizeof(kGrid[axis]));
        if (!bound) {
            std::cerr << "the fractal Program does not take its pixels, time and grid\n";
            return false;
        }
        if (headless_)
            return true;
        rgba_.resize(pixels_.size() * 4);
        return graphics_ &&
               graphics_->createPanelImage(kWidth, kHeight, VERNON_IMAGE_TRANSFER_DESTINATION, image_, view_);
    }

    bool renderFrame(double elapsedSeconds, uint32_t, uint32_t) override {
        time_ = static_cast<float>(elapsedSeconds);
        std::string error;
        if (!program_.call(error)) {
            std::cerr << "fractal Program call failed: " << error << '\n';
            return false;
        }
        if (headless_) {
            const double checksum = std::accumulate(pixels_.begin(), pixels_.end(), 0.0);
            if (!std::isfinite(checksum) || checksum <= 0.0)
                return false;
            std::cout << "Vernon CPU fractal headless checksum: " << checksum << '\n';
            return true;
        }
        for (size_t index = 0; index < pixels_.size(); ++index) {
            const auto gray = static_cast<uint8_t>(std::clamp(pixels_[index], 0.0F, 1.0F) * 255.0F);
            rgba_[index * 4] = gray;
            rgba_[index * 4 + 1] = gray;
            rgba_[index * 4 + 2] = gray;
            rgba_[index * 4 + 3] = 255;
        }
        VernonHalImageWrite write{};
        write.region.aspect = VERNON_IMAGE_ASPECT_COLOR;
        write.region.width = kWidth;
        write.region.height = kHeight;
        write.region.depth = 1;
        write.region.data_format = VERNON_HAL_IMAGE_DATA_RGBA;
        write.region.data_type = VERNON_HAL_IMAGE_DATA_UINT8;
        write.source = rgba_.data();
        if (vernonHalImageWrite(image_, &write, 1) != VERNON_HAL_OK) {
            const VernonStringView message = vernonHalGetLastError();
            std::cerr << "failed to upload the CPU fractal image: " << std::string(message.data, message.size) << '\n';
            return false;
        }
        return true;
    }

    const VernonHalImageView *view() const override { return view_; }

    void shutdown() override {
        GraphicsHost::destroyPanelImage(image_, view_);
        program_.reset();
        if (device_)
            (void)vernonHalDestroyDevice(device_);
        device_ = nullptr;
    }

private:
    bool headless_{};
    GraphicsHost *graphics_{};
    VernonHalDevice *device_{};
    vernon_external_engine::LinkedProgram program_;
    VernonHalImage *image_{};
    VernonHalImageView *view_{};
    std::vector<float> pixels_;
    std::vector<uint8_t> rgba_;
    float time_{};
};

} // namespace

std::unique_ptr<ExternalEnginePanel> createCpuFractalPanel() { return std::make_unique<CpuFractalPanel>(); }
