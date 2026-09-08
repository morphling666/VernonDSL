#include "canonical_program_call.h"
#include "engine_demo.h"
#include "graphics_host.h"

#include "embedded_bundles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#ifndef VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION
#error VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION must name the generated registration function
#endif

extern "C" VernonStatus VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION(void);

namespace {

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 320;
constexpr uint64_t kPixelShape[] = {kHeight, kWidth};
constexpr int64_t kPixelStrides[] = {kWidth * sizeof(float), sizeof(float)};
constexpr std::array<uint32_t, 3> kGrid{kWidth / 16, kHeight / 16, 1};

class CpuFractalPanel final : public ExternalEnginePanel {
public:
    ~CpuFractalPanel() override { shutdown(); }

    bool initialize(GraphicsHost *graphics, bool headless) override {
        graphics_ = graphics;
        headless_ = headless;
        if (VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION() != VERNON_STATUS_OK) {
            std::cerr << "failed to register statically linked Vernon CPU entries\n";
            return false;
        }
        runtime_ = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
        if (!runtime_)
            return false;
        VernonProgramBundleLoadOptions bundleOptions{};
        bundleOptions.struct_size = sizeof(bundleOptions);
        bundleOptions.bundle_directory = vernon_external_engine::cpu_bundle::kCookedDirectory;
        const VernonProgramBundleLoadOptions *loadOptions = &bundleOptions;
        try {
            program_.emplace(vernon::runtime::ProgramExecutable::load(
                runtime_, vernon_external_engine::cpu_bundle::kManifest,
                vernon_external_engine::cpu_bundle::kManifestSize, {nullptr, 0}, loadOptions));
            instance_ = std::make_unique<vernon::runtime::ProgramInstance>(*program_);
        } catch (const std::exception &exception) {
            const std::string diagnostic = vernon_external_engine::programRuntimeError(runtime_);
            std::cerr << "failed to load the fractal Program: " << (diagnostic.empty() ? exception.what() : diagnostic)
                      << '\n';
            return false;
        }
        VernonProgramExecutable *pipeline = program_->get();

        constexpr std::array<const char *, 5> names{"pixels", "time", "__grid_x", "__grid_y", "__grid_z"};
        std::array<VernonProgramParameterView, 5> parameters{};
        for (size_t index = 0; index < parameters.size(); ++index)
            if (vernonRuntimeProgramExecutableFindParameter(
                    pipeline, {names[index], std::char_traits<char>::length(names[index])}, &parameters[index]) !=
                VERNON_STATUS_OK)
                return false;
        pixels_.resize(static_cast<size_t>(kWidth) * kHeight);
        rgba_.resize(pixels_.size() * 4);

        arguments_[0].slot = parameters[0].slot;
        arguments_[0].kind = VERNON_PROGRAM_TENSOR;
        arguments_[0].tensor.struct_size = sizeof(VernonTensorView);
        arguments_[0].tensor.storage = VERNON_TENSOR_HOST;
        arguments_[0].tensor.host_data = pixels_.data();
        arguments_[0].tensor.element_layout = parameters[0].element_layout;
        arguments_[0].tensor.access = parameters[0].access;
        arguments_[0].tensor.rank = 2;
        arguments_[0].tensor.shape = kPixelShape;
        arguments_[0].tensor.byte_strides = kPixelStrides;
        arguments_[0].tensor.byte_size = pixels_.size() * sizeof(float);
        arguments_[1].slot = parameters[1].slot;
        arguments_[1].kind = VERNON_PROGRAM_TENSOR;
        arguments_[1].tensor.struct_size = sizeof(VernonTensorView);
        arguments_[1].tensor.storage = VERNON_TENSOR_HOST;
        arguments_[1].tensor.host_data = &time_;
        arguments_[1].tensor.element_layout = parameters[1].element_layout;
        arguments_[1].tensor.access = parameters[1].access;
        arguments_[1].tensor.byte_size = sizeof(time_);
        for (size_t axis = 0; axis < kGrid.size(); ++axis) {
            VernonProgramArgument &argument = arguments_[axis + 2];
            argument.slot = parameters[axis + 2].slot;
            argument.kind = VERNON_PROGRAM_TENSOR;
            argument.tensor.struct_size = sizeof(VernonTensorView);
            argument.tensor.storage = VERNON_TENSOR_HOST;
            argument.tensor.host_data = &kGrid[axis];
            argument.tensor.element_layout = parameters[axis + 2].element_layout;
            argument.tensor.access = parameters[axis + 2].access;
            argument.tensor.byte_size = sizeof(kGrid[axis]);
        }
        if (!headless_) {
            if (!graphics_)
                return false;
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
            descriptor.usage = VERNON_RHI_IMAGE_TRANSFER_DESTINATION | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
            if (vernonRhiDeviceCreateImage(graphics_->device(), &descriptor, &presentImage_) != VERNON_RHI_STATUS_OK)
                return false;
        }

        return true;
    }

    bool renderFrame(double elapsedSeconds, uint32_t, uint32_t) override {
        time_ = static_cast<float>(elapsedSeconds);
        std::string error;
        if (!vernon_external_engine::invokeProgram(runtime_, *program_, *instance_, arguments_.data(),
                                                   arguments_.size(), nullptr, error)) {
            std::cerr << "fractal Program invocation failed: " << error << '\n';
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
        VernonRhiImageUploadDescriptor upload{};
        upload.struct_size = sizeof(upload);
        upload.width = kWidth;
        upload.height = kHeight;
        upload.depth = 1;
        upload.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
        upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
        upload.data = rgba_.data();
        return vernonRhiDeviceUploadImage(graphics_->device(), presentImage_, &upload, 1) == VERNON_RHI_STATUS_OK;
    }

    VernonRhiImage image() const override { return presentImage_; }
    uint32_t imageWidth() const override { return kWidth; }
    uint32_t imageHeight() const override { return kHeight; }

    void shutdown() override {
        instance_.reset();
        program_.reset();
        if (!headless_ && graphics_ && presentImage_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImage(graphics_->device(), presentImage_);
        presentImage_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        if (runtime_)
            vernonRuntimeDestroy(runtime_);
        runtime_ = nullptr;
    }

private:
    bool headless_{};
    GraphicsHost *graphics_{};
    VernonRuntimeContext *runtime_{};
    std::optional<vernon::runtime::ProgramExecutable> program_;
    std::unique_ptr<vernon::runtime::ProgramInstance> instance_;
    VernonRhiImage presentImage_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::vector<float> pixels_;
    std::vector<uint8_t> rgba_;
    float time_{};
    std::array<VernonProgramArgument, 5> arguments_{};
};

} // namespace

std::unique_ptr<ExternalEnginePanel> createCpuFractalPanel() { return std::make_unique<CpuFractalPanel>(); }
