#include "engine_demo.h"
#include "graphics_host.h"

#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"
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

#ifndef VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION
#error VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION must name the generated registration function
#endif

extern "C" VernonStatus VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION(void);

namespace {

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 320;
constexpr uint64_t kPixelShape[] = {kHeight, kWidth};
constexpr int64_t kPixelStrides[] = {kWidth * sizeof(float), sizeof(float)};

std::string runtimeError(VernonRuntimeContext *runtime) {
    const VernonStringView error = vernonRuntimeGetLastError(runtime);
    return error.data ? std::string(error.data, error.size) : "unknown Runtime error";
}

class FractalComputePass final : public vernon::execution::ComputePass {
public:
    FractalComputePass(vernon::execution::GraphBuffer pixels, VernonRuntimeContext *runtime,
                       VernonLoadedPipeline *pipeline, VernonPipelineInvocation *invocation)
        : ComputePass("fractal-compute"), pixels_(pixels), runtime_(runtime), pipeline_(pipeline),
          invocation_(invocation) {}

    void declare() override { write(pixels_, VERNON_RHI_STATE_COMMON); }

    VernonRhiStatus execute(vernon::execution::ComputeEncoder &,
                            const vernon::execution::ExecutionResources &) override {
        if (vernonRuntimePipelineInvoke(pipeline_, invocation_) == VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_OK;
        std::cerr << "fractal invocation failed: " << runtimeError(runtime_) << '\n';
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    vernon::execution::GraphBuffer pixels_;
    VernonRuntimeContext *runtime_{};
    VernonLoadedPipeline *pipeline_{};
    VernonPipelineInvocation *invocation_{};
};

class FractalPresentPass final : public vernon::execution::ComputePass {
public:
    FractalPresentPass(vernon::execution::GraphBuffer pixels, const std::vector<float> *values,
                       std::vector<uint8_t> *rgba, GraphicsHost *graphics, VernonRhiImage image, bool headless)
        : ComputePass("fractal-present"), pixels_(pixels), values_(values), rgba_(rgba), graphics_(graphics),
          image_(image), headless_(headless) {
        setFlags(vernon::execution::PassSideEffect | vernon::execution::PassNeverCull);
    }

    void declare() override { read(pixels_, VERNON_RHI_STATE_COMMON); }

    VernonRhiStatus execute(vernon::execution::ComputeEncoder &,
                            const vernon::execution::ExecutionResources &) override {
        if (headless_) {
            const double checksum = std::accumulate(values_->begin(), values_->end(), 0.0);
            if (!std::isfinite(checksum) || checksum <= 0.0)
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            std::cout << "Vernon CPU fractal headless checksum: " << checksum << '\n';
            return VERNON_RHI_STATUS_OK;
        }
        for (size_t index = 0; index < values_->size(); ++index) {
            const auto gray = static_cast<uint8_t>(std::clamp((*values_)[index], 0.0F, 1.0F) * 255.0F);
            (*rgba_)[index * 4] = gray;
            (*rgba_)[index * 4 + 1] = gray;
            (*rgba_)[index * 4 + 2] = gray;
            (*rgba_)[index * 4 + 3] = 255;
        }
        VernonRhiImageUploadDescriptor upload{};
        upload.struct_size = sizeof(upload);
        upload.width = kWidth;
        upload.height = kHeight;
        upload.depth = 1;
        upload.source_format = VERNON_RHI_IMAGE_DATA_RGBA;
        upload.source_type = VERNON_RHI_IMAGE_DATA_UINT8;
        upload.data = rgba_->data();
        return vernonRhiDeviceUploadImage(graphics_->device(), image_, &upload, 1);
    }

private:
    vernon::execution::GraphBuffer pixels_;
    const std::vector<float> *values_{};
    std::vector<uint8_t> *rgba_{};
    GraphicsHost *graphics_{};
    VernonRhiImage image_{};
    bool headless_{};
};

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
#if defined(__EMSCRIPTEN__)
        const VernonPipelineBundleLoadOptions *loadOptions = nullptr;
#else
        VernonPipelineBundleLoadOptions desktopOptions{};
        desktopOptions.struct_size = sizeof(desktopOptions);
        desktopOptions.bundle_directory = vernon_external_engine::cpu_bundle::kCookedDirectory;
        const VernonPipelineBundleLoadOptions *loadOptions = &desktopOptions;
#endif
        bundle_ =
            vernonRuntimeLoadPipelineBundleWithOptions(runtime_, vernon_external_engine::cpu_bundle::kManifest,
                                                       vernon_external_engine::cpu_bundle::kManifestSize, loadOptions);
        pipeline_ = bundle_ ? vernonRuntimeResolvePipeline(bundle_, {nullptr, 0}) : nullptr;
        if (!pipeline_) {
            std::cerr << "failed to load the fractal pipeline: " << runtimeError(runtime_) << '\n';
            return false;
        }

        VernonPipelineParameterView pixelsParameter{};
        VernonPipelineParameterView timeParameter{};
        if (vernonRuntimeLoadedPipelineFindParameter(pipeline_, {"pixels", 6}, &pixelsParameter) != VERNON_STATUS_OK ||
            vernonRuntimeLoadedPipelineFindParameter(pipeline_, {"time", 4}, &timeParameter) != VERNON_STATUS_OK)
            return false;
        pixels_.resize(static_cast<size_t>(kWidth) * kHeight);
        rgba_.resize(pixels_.size() * 4);

        arguments_[0].slot = pixelsParameter.slot;
        arguments_[0].kind = VERNON_PIPELINE_TENSOR;
        arguments_[0].tensor.struct_size = sizeof(VernonTensorView);
        arguments_[0].tensor.storage = VERNON_TENSOR_HOST;
        arguments_[0].tensor.host_data = pixels_.data();
        arguments_[0].tensor.element_layout = pixelsParameter.element_layout;
        arguments_[0].tensor.access = pixelsParameter.access;
        arguments_[0].tensor.rank = 2;
        arguments_[0].tensor.shape = kPixelShape;
        arguments_[0].tensor.byte_strides = kPixelStrides;
        arguments_[0].tensor.byte_size = pixels_.size() * sizeof(float);
        arguments_[1].slot = timeParameter.slot;
        arguments_[1].kind = VERNON_PIPELINE_TENSOR;
        arguments_[1].tensor.struct_size = sizeof(VernonTensorView);
        arguments_[1].tensor.storage = VERNON_TENSOR_HOST;
        arguments_[1].tensor.host_data = &time_;
        arguments_[1].tensor.element_layout = timeParameter.element_layout;
        arguments_[1].tensor.access = timeParameter.access;
        arguments_[1].tensor.byte_size = sizeof(time_);
        invocation_.struct_size = sizeof(invocation_);
        invocation_.abi_version = VERNON_PIPELINE_VERSION;
        invocation_.arguments = arguments_.data();
        invocation_.argument_count = arguments_.size();
        invocation_.compute_grid = {kWidth / 16, kHeight / 16, 1};

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

        graph_ = std::make_unique<vernon::execution::ExecutionGraph>();
        const auto graphPixels = graph_->importHostBuffer(reinterpret_cast<uint64_t>(pixels_.data()), true);
        graph_->emplacePass<FractalComputePass>(graphPixels, runtime_, pipeline_, &invocation_);
        graph_->emplacePass<FractalPresentPass>(graphPixels, &pixels_, &rgba_, graphics_, presentImage_, headless_);
        std::string error;
        if (!graph_->compile(error)) {
            std::cerr << "failed to compile CPU fractal graph: " << error << '\n';
            return false;
        }
        return true;
    }

    bool renderFrame(double elapsedSeconds, uint32_t, uint32_t) override {
        time_ = static_cast<float>(elapsedSeconds);
        return graph_ && graph_->execute() == VERNON_RHI_STATUS_OK;
    }

    VernonRhiImage image() const override { return presentImage_; }
    uint32_t imageWidth() const override { return kWidth; }
    uint32_t imageHeight() const override { return kHeight; }

    void shutdown() override {
        graph_.reset();
        if (!headless_ && graphics_ && presentImage_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            vernonRhiDeviceDestroyImage(graphics_->device(), presentImage_);
        presentImage_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
        if (pipeline_)
            vernonRuntimeLoadedPipelineDestroy(pipeline_);
        if (bundle_)
            vernonRuntimePipelineBundleDestroy(bundle_);
        if (runtime_)
            vernonRuntimeDestroy(runtime_);
        pipeline_ = nullptr;
        bundle_ = nullptr;
        runtime_ = nullptr;
    }

private:
    bool headless_{};
    GraphicsHost *graphics_{};
    VernonRuntimeContext *runtime_{};
    VernonPipelineBundle *bundle_{};
    VernonLoadedPipeline *pipeline_{};
    VernonRhiImage presentImage_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::vector<float> pixels_;
    std::vector<uint8_t> rgba_;
    float time_{};
    std::array<VernonPipelineArgument, 2> arguments_{};
    VernonPipelineInvocation invocation_{};
    std::unique_ptr<vernon::execution::ExecutionGraph> graph_;
};

} // namespace

std::unique_ptr<ExternalEnginePanel> createCpuFractalPanel() { return std::make_unique<CpuFractalPanel>(); }
