#include "engine_demo.h"
#include "graphics_host.h"
#include "linked_program.h"

#include "embedded_bundles.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

namespace {

constexpr uint64_t kVectorShape[] = {3};
constexpr int64_t kVectorStride[] = {sizeof(float)};
constexpr std::array<float, 6> kFullscreenTriangle{-1.0F, -1.0F, 3.0F, -1.0F, -1.0F, 3.0F};
constexpr uint64_t kPositionShape[] = {3, 2};
constexpr int64_t kPositionStrides[] = {2 * sizeof(float), sizeof(float)};

class MandelbulbPanel final : public ExternalEnginePanel {
public:
    ~MandelbulbPanel() override { shutdown(); }

    bool initialize(GraphicsHost *graphics, bool headless) override {
        if (headless) {
            std::cerr << "the Mandelbulb graphics demo requires a browser or desktop window\n";
            return false;
        }
        graphics_ = graphics;
        if (!graphics_)
            return false;
        VernonHalBufferDescriptor vertexDescriptor{};
        vertexDescriptor.struct_size = sizeof(vertexDescriptor);
        vertexDescriptor.size = sizeof(kFullscreenTriangle);
        vertexDescriptor.usage = VERNON_HAL_BUFFER_VERTEX;
        const VernonHalBufferWrite vertexWrite{0, kFullscreenTriangle.data(), sizeof(kFullscreenTriangle)};
        VernonRuntimeProviderResourceReference vertexReference{};
        if (vernonHalCreateBuffer(graphics_->device(), &vertexDescriptor, &vertices_) != VERNON_HAL_OK ||
            vernonHalBufferWrite(vertices_, &vertexWrite, 1) != VERNON_HAL_OK ||
            vernonHalReferenceBuffer(vertices_, 0, sizeof(kFullscreenTriangle), &vertexReference) != VERNON_HAL_OK)
            return false;

        std::string error;
        if (!program_.link(graphics_->runtime(), vernon_external_engine::graphics_bundle::kManifest,
                           vernon_external_engine::graphics_bundle::kManifestSize,
                           vernon_external_engine::graphics_bundle::kCookedDirectory, error)) {
            std::cerr << "failed to load the Mandelbulb Program: " << error << '\n';
            return false;
        }
        const bool bound = program_.bindDeviceTensor("position", vertexReference, sizeof(kFullscreenTriangle), 2,
                                                     kPositionShape, kPositionStrides) &&
                           program_.bindHostTensor("camera_position", cameraPosition_.data(), sizeof(cameraPosition_),
                                                   1, kVectorShape, kVectorStride) &&
                           program_.bindHostTensor("camera_target", cameraTarget_.data(), sizeof(cameraTarget_), 1,
                                                   kVectorShape, kVectorStride) &&
                           program_.bindHostTensor("time", &time_, sizeof(time_)) &&
                           program_.bindHostTensor("power", &power_, sizeof(power_)) &&
                           program_.bindHostTensor("max_iterations", &maxIterations_, sizeof(maxIterations_)) &&
                           program_.bindHostTensor("max_steps", &maxSteps_, sizeof(maxSteps_)) &&
                           program_.bindHostTensor("shadow_steps", &shadowSteps_, sizeof(shadowSteps_));
        const size_t passSlot = program_.slotOfKind(VERNON_SCHEDULE_SLOT_PASS);
        const size_t drawSlot = program_.slotOfKind(VERNON_SCHEDULE_SLOT_DRAW);
        if (!bound || passSlot == SIZE_MAX || drawSlot == SIZE_MAX) {
            std::cerr << "the Mandelbulb Program does not take its vertices, uniforms and pass\n";
            return false;
        }
        attachment_.location = 0;
        attachment_.load_operation = VERNON_RUNTIME_PROVIDER_LOAD_CLEAR;
        attachment_.store_operation = VERNON_RUNTIME_PROVIDER_STORE_PRESERVE;
        attachment_.clear_color.f32[3] = 1.0F;
        renderPass_.struct_size = sizeof(renderPass_);
        renderPass_.color_attachments = &attachment_;
        renderPass_.color_attachment_count = 1;
        program_.write(passSlot, static_cast<const VernonRenderPass *>(&renderPass_));
        VernonDrawCommand draw{};
        draw.struct_size = sizeof(draw);
        draw.vertex_count = 3;
        draw.instance_count = 1;
        program_.write(drawSlot, draw);
        return true;
    }

    bool renderFrame(double elapsedSeconds, uint32_t panelWidth, uint32_t panelHeight) override {
        if (!panelWidth || !panelHeight)
            return true;
        if ((panelWidth != renderWidth_ || panelHeight != renderHeight_) &&
            !recreateRenderTarget(panelWidth, panelHeight))
            return false;
        const float phase = static_cast<float>(elapsedSeconds);
        const float angle = phase * 0.22F + 0.55F;
        cameraPosition_ = {3.15F * std::cos(angle), 0.48F + std::sin(phase * 0.17F) * 0.12F, 3.15F * std::sin(angle)};
        time_ = phase;
        power_ = 8.0F + std::sin(phase * 0.21F) * 0.18F;
        std::string error;
        if (!program_.call(error)) {
            std::cerr << "Mandelbulb Program call failed: " << error << '\n';
            return false;
        }
        if (frame_++ == 0)
            std::cout << "Vernon Mandelbulb Program animation started\n";
        return true;
    }

    const VernonHalImageView *view() const override { return view_; }

    void shutdown() override {
        program_.reset();
        GraphicsHost::destroyPanelImage(image_, view_);
        if (vertices_)
            vernonHalDestroyBuffer(vertices_);
        vertices_ = nullptr;
    }

private:
    bool recreateRenderTarget(uint32_t width, uint32_t height) {
        VernonRuntimeProviderResourceReference viewReference{};
        if (!graphics_->createPanelImage(width, height, VERNON_IMAGE_COLOR_ATTACHMENT, image_, view_) ||
            vernonHalReferenceImageView(view_, &viewReference) != VERNON_HAL_OK)
            return false;
        renderWidth_ = width;
        renderHeight_ = height;
        attachment_.view = viewReference;
        renderPass_.render_area[2] = width;
        renderPass_.render_area[3] = height;
        return true;
    }

    GraphicsHost *graphics_{};
    vernon_external_engine::LinkedProgram program_;
    VernonHalBuffer *vertices_{};
    VernonHalImage *image_{};
    VernonHalImageView *view_{};
    std::array<float, 3> cameraPosition_{};
    std::array<float, 3> cameraTarget_{};
    float time_{};
    float power_{8.0F};
    int32_t maxIterations_{18};
    int32_t maxSteps_{112};
    int32_t shadowSteps_{32};
    VernonColorAttachment attachment_{};
    VernonRenderPass renderPass_{};
    uint32_t renderWidth_{};
    uint32_t renderHeight_{};
    uint64_t frame_{};
};

} // namespace

std::unique_ptr<ExternalEnginePanel> createMandelbulbPanel() { return std::make_unique<MandelbulbPanel>(); }
